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
from nemulate.models.coders import EarthAE


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

    merged_path = cesm_path / "merged"
    var_names = list(sorted([p.stem for p in merged_path.glob("*")]))
    print(f"Found {len(var_names)} variables: {var_names}")

    # order is important, scaler and forced response are computed in the native grid

    bhalf = torch.bfloat16
    device = torch.device(f"cuda:{cfg.device_id}")

    tfm = nn.Sequential(
        Lambda(lambda ins: ins.compute()),
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

    model = EarthAE(width_base=64, num_layers=4, include_land_mask=True).to(
        device, dtype=bhalf
    )

    with torch.autocast(device_type="cuda:0", dtype=bhalf):
        print(
            summary(model, input_data=torch.randn(16, 5, 180, 360).to(device))
        )

    criterion = nn.MSELoss()

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

    # Handle resume configuration (step for scheduler + logging)
    resume_ckpt = cfg.get("resume_checkpoint")
    resume_step = int(cfg.get("resume_step", 0)) if resume_ckpt else 0

    epochs = cfg.epochs
    total_steps = epochs * len(climate_dl)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=base_lr, weight_decay=1e-3
    )
    if resume_step > 0:
        for group in optimizer.param_groups:
            group.setdefault("initial_lr", group["lr"])

    # LR scheduler: when resuming, align internal epoch so LR picks up
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1.0,
        end_factor=final_lr / base_lr,
        total_iters=total_steps,
        last_epoch=resume_step - 1 if resume_step > 0 else -1,
    )

    if resume_ckpt:
        resume_path = Path(to_absolute_path(resume_ckpt))
        print(f"Resuming from checkpoint {resume_path} at step {resume_step}")
        state_dict = torch.load(resume_path, map_location=device)
        model.load_state_dict(state_dict)
        step = resume_step
    else:
        step = 0
        print("Saving initial ckpt")
        torch.save(
            model.state_dict(),
            str(checkpoint_dir / "earthae_initial.ckpt"),
        )

    for e in range(epochs):
        pbar = tqdm(climate_dl, desc=f"EarchAE {e + 1}/{epochs}")
        total_loss = 0.0
        num_batches = 0
        for batch in pbar:
            num_batches += 1
            with torch.autocast(device_type="cuda", dtype=bhalf):
                # (batch_size, num_vars, interval, lat, lon)
                fields = batch["vars"].to(device)
                mask = batch["land_mask"].to(device)

                # (batch_size, interval, num_vars, lat, lon)
                fields = fields.transpose(1, 2).contiguous()
                mask = mask.transpose(1, 2).contiguous()

                # (batch_size*interval, num_vars, lat, lon)
                fields = fields.view(-1, *fields.shape[2:])
                mask = mask.view(-1, *mask.shape[2:])

                _, reconstruction = model(fields, land_mask=mask.float())
                reconstruction = reconstruction.masked_fill(mask, 0.0)

                loss = criterion(fields, reconstruction)

            optimizer.zero_grad()

            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()

            loss_val = loss.item()
            total_loss += loss_val
            pbar.set_postfix(loss=total_loss / num_batches)

            if (step + 1) % 1000 == 0:
                torch.save(
                    model.state_dict(),
                    str(checkpoint_dir / f"earthae_step_{step}.ckpt"),
                )
            wandb.log(
                {
                    "epoch": e,
                    "step": step,
                    "mse_loss": loss_val,
                    "lr": optimizer.param_groups[0]["lr"],
                }
            )

            step += 1
        torch.save(
            model.state_dict(),
            str(checkpoint_dir / f"earthae_{e}.ckpt"),
        )


if __name__ == "__main__":
    main()
