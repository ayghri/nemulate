from dataclasses import dataclass
from pathlib import Path

import hydra
from hydra.core.config_store import ConfigStore
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

wandb.init(project="ssh", name="auto-encoder-run")


@dataclass
class TrainConfig:
    cesm_path: str
    checkpoint_dir: str = "/buckets/checkpoints"
    batch_size: int = 4
    num_workers: int = 8
    prefetch_factor: int = 4
    pin_memory: bool = True
    epochs: int = 10
    base_lr: float = 1e-3
    final_lr: float = 1e-5

    device_id = 0


cs = ConfigStore.instance()
cs.store(name="train_encoder", node=TrainConfig)


@hydra.main(version_base="1.3", config_name="train_encoder")
def main(cfg: TrainConfig) -> None:
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

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=base_lr, weight_decay=1e-3
    )

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

    epochs = cfg.epochs
    total_steps = epochs * len(climate_dl)

    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1.0,
        end_factor=final_lr / base_lr,
        total_iters=total_steps,
    )
    step = 0

    print("Saving initial ckpt")
    torch.save(model.state_dict(), str(checkpoint_dir / "earthae_initial.ckpt"))

    for e in range(epochs):
        pbar = tqdm(climate_dl, desc=f"EarchAE {e + 1}/{epochs}")
        total_loss = 0.0
        num_batches = 0
        for batch in pbar:
            with torch.autocast(device_type="cuda", dtype=bhalf):
                # (batch_size, num_vars, interval, lat, lon)
                fields = batch["vars"].to(device)
                # (batch_size, num_vars, lat, lon), True on land
                mask = batch["land_mask"].to(device)
                # (batch_size, num_vars, lat, lon)
                mask_encoded = model.encode_land_mask(mask)
                # (batch_size, interval, num_vars lat, lon)
                fields = fields.transpose(1, 2).contiguous()
                fields = fields + mask_encoded.unsqueeze(1)
                fields = fields.view(-1, *fields.shape[2:])

                _, reconstruction = model(fields)
                reconstruction = reconstruction.view(
                    batch["vars"].shape[0],
                    batch["vars"].shape[2],
                    fields.shape[1:],
                )
                reconstruction = reconstruction.masked_fill(
                    mask.unsqueeze(1), 0.0
                )

                loss = criterion(fields, reconstruction)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            num_batches += 1
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
