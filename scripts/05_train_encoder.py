from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import to_absolute_path
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
from torchinfo import summary
import wandb

from nemulate.data.transforms import AddLatLong
from nemulate.data.transforms import AddNanMask
from nemulate.data.transforms import AddXYZ
from nemulate.data.transforms import AddYearMonth
from nemulate.data.transforms import FieldScaler
from nemulate.data.transforms import Lambda
from nemulate.data.transforms import RandomRotatedRegrid
from nemulate.data.transforms import SubstractForcedResponse
from nemulate.data.transforms import UnwrapFields
from nemulate.data.sources import get_cesm2_members_ids
from nemulate.datasets import ClimateDataset

# # from nemulate.models.coders import EarthAE
# from nemulate.models.inverted_earth import EarthAE_Wide
from nemulate.models import get_model


@hydra.main(
    version_base="1.3", config_name="train_encoder", config_path="../conf"
)
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg, resolve=True))

    wandb.init(
        project="ssh",
        name="auto-encoder-run",
        config=OmegaConf.to_container(cfg, resolve=True),
    )

    cesm_path = Path(to_absolute_path(cfg.cesm_path))
    checkpoint_dir = Path(to_absolute_path(cfg.checkpoint_dir))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    model_name = cfg["model_name"]
    model_arch = cfg["model_arch"]

    merged_path = cesm_path / "merged"
    var_names = list(sorted([p.stem for p in merged_path.glob("*")]))
    print(f"Found {len(var_names)} variables: {var_names}")

    # order is important, scaler and forced response are computed in the native grid

    bhalf = torch.bfloat16
    device = torch.device(f"cuda:{cfg.device_id}")

    tfm = nn.Sequential(
        SubstractForcedResponse(
            cesm_path.joinpath("moments"),
            "{var}_forced_response_1.nc",
            var_names=var_names,
        ),
        FieldScaler(
            cesm_path / "stats" / "all_var_time_stats.nc",
            field_format="{var}_var_globalstd",
            var_names=var_names,
        ),
        AddLatLong(grid_path=cesm_path / "grid_info.nc"),
        RandomRotatedRegrid(grid_path=cesm_path / "grid_info.nc", p=1.0),
        AddNanMask(var_names),
        AddXYZ(),
        AddYearMonth(),
        UnwrapFields(
            {
                "vars": var_names,
                "land_mask": "land_mask",
                "year": "year",
                "month": "month",
                "xyz": ["cart_x", "cart_y", "cart_z"],
            }
        ),
    )

    members = get_cesm2_members_ids(merged_path, var_name=var_names)

    # Optimizer and LR scheduler: linearly decay from base_lr to final_lr
    base_lr = cfg.base_lr
    final_lr = cfg.final_lr

    climate_ds = ClimateDataset(
        merged_path,
        members=members,
        variables=var_names,
        interval=24,
        loading_time_chunck_size=24,
        transform=tfm,
        compute=True,
    )

    climate_dl = DataLoader(
        climate_ds,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        prefetch_factor=cfg.prefetch_factor,
        shuffle=True,
        persistent_workers=True,
        pin_memory=cfg.pin_memory,
    )

    # model = EarthAE(width_base=64, num_layers=4, include_land_mask=True).
    model = get_model(model_arch)(
        include_land_mask=True, land_mask_channels=len(var_names)
    ).to(device, dtype=bhalf)
    #   memory_format=torch.channels_last)

    with torch.autocast(device_type="cuda:0", dtype=bhalf):
        summary(model, input_data=torch.randn(16, 5, 180, 360).to(device))

    criterion = nn.MSELoss()

    # Handle resume configuration (step for scheduler + logging)
    resume_ckpt = cfg.get("resume_checkpoint")
    resume_step = int(cfg.get("resume_step", 0)) if resume_ckpt else 0

    grad_accumulation_steps = cfg.get("grad_accumulation_steps", 1)
    epochs = cfg.epochs
    total_steps = (epochs * len(climate_dl)) // grad_accumulation_steps

    initial_lr = final_lr + (base_lr - final_lr) * (
        1 - resume_step / total_steps
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=initial_lr, weight_decay=1e-3
    )

    step = resume_step

    warmup_steps = cfg.warmup_steps // grad_accumulation_steps

    # for param_group in optimizer.param_groups:
    # param_group["lr"] = initial_lr
    # for group in optimizer.param_groups:
    # group.setdefault("initial_lr", initial_lr)
    from nemulate.utils.train import WarmupScheduler

    warmup_scheduler = WarmupScheduler(
        optimizer,
        warmup_steps=warmup_steps,
    )

    decay_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1.0,
        end_factor=final_lr / base_lr,
        total_iters=total_steps - warmup_steps,
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[
            warmup_scheduler,
            decay_scheduler,
        ],
        milestones=[warmup_steps],
    )

    if resume_ckpt:
        resume_path = Path(to_absolute_path(resume_ckpt))
        print(f"Resuming from checkpoint {resume_path} at step {resume_step}")
        state_dict = torch.load(resume_path, map_location=device)
        model.load_state_dict(state_dict)
    else:
        print("Saving initial ckpt")
        torch.save(
            model.state_dict(),
            checkpoint_dir / f"{model_name}_initial.ckpt",
        )

    optimizer.zero_grad()
    checkpoint_interval = cfg.checkpoint_interval

    for e in range(epochs):
        pbar = tqdm(climate_dl, desc=f"EarchAE {e + 1}/{epochs}", initial=step)
        total_loss = 0.0
        num_batches = 0
        for batch_idx, batch in enumerate(pbar):
            with torch.autocast(device_type="cuda", dtype=bhalf):
                # (batch_size, num_vars, interval, lat, lon)
                fields = batch["vars"].to(device)
                mask = batch["land_mask"].to(device)

                # (batch_size, interval, num_vars, lat, lon)
                fields = fields.transpose(1, 2).contiguous()
                mask = mask.transpose(1, 2).contiguous()

                # (batch_size*interval, num_vars, lat, lon)
                fields = fields.view(-1, *fields.shape[2:])
                # .to( memory_format=torch.channels_last)
                mask = mask.view(-1, *mask.shape[2:])
                # .to( memory_format=torch.channels_last)

                reconstruction, _ = model(fields, land_mask=mask.float())
                reconstruction = reconstruction.masked_fill(mask, 0.0)

                loss = criterion(fields, reconstruction)
                loss = loss / grad_accumulation_steps

            loss.backward()

            num_batches += 1
            step += 1

            if (batch_idx + 1) % grad_accumulation_steps == 0:
                clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if step % checkpoint_interval == 0:
                torch.save(
                    model.state_dict(),
                    str(checkpoint_dir / f"{model_name}_step_{step}.ckpt"),
                )

            loss_val = loss.item() * grad_accumulation_steps
            total_loss += loss_val

            wandb.log(
                {
                    "epoch": e,
                    "step": step,
                    "mse_loss": loss_val,
                    "lr": optimizer.param_groups[0]["lr"],
                }
            )
            pbar.set_postfix(
                loss=total_loss / num_batches,
                lr=optimizer.param_groups[0]["lr"],
            )
        torch.save(
            model.state_dict(),
            str(checkpoint_dir / f"earthae_{e}.ckpt"),
        )


if __name__ == "__main__":
    main()
