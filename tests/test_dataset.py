from pathlib import Path
from time import perf_counter
from torch import nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import click
import torch

from nemulate import data as ndata
from nemulate.data.transforms import AddLatLong
from nemulate.data.transforms import AddNanMask
from nemulate.data.transforms import AddXYZ
from nemulate.data.transforms import AddYearMonth
from nemulate.data.transforms import FieldScaler
from nemulate.data.transforms import Lambda
from nemulate.data.transforms import RandomRotatedRegrid
from nemulate.data.transforms import SubstractForcedResponse
from nemulate.data.transforms import UnwrapFields

from nemulate.datasets import ClimateDataset
from nemulate.data.sources import get_cesm2_members_ids

from tqdm import tqdm


from nemulate.models.coders import EarthAE
from torchinfo import summary
import wandb


bhalf = torch.bfloat16


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--cesm-path",
    type=click.Path(path_type=Path, exists=True, file_okay=False),
    required=True,
    help="Root folder containing CESM data",
)
@click.option(
    "--batch-size",
    "-b",
    type=int,
    default=4,
    show_default=True,
    help="Batch size for the DataLoader",
)
@click.option(
    "--num-workers",
    type=int,
    default=8,
    show_default=True,
    help="Number of worker processes for the DataLoader",
)
@click.option(
    "--prefetch-factor",
    type=int,
    default=4,
    show_default=True,
    help="Prefetch factor for the DataLoader",
)
@click.option(
    "--persistent-workers/--no-persistent-workers",
    default=True,
    help="Use persistent workers for the DataLoader",
)
@click.option(
    "--pin-memory/--no-pin-memory",
    default=True,
    help="Pin memory for the DataLoader",
)
def main(
    cesm_path: Path,
    batch_size: int = 4,
    num_workers: int = 8,
    prefetch_factor: int = 4,
    persistent_workers: bool = True,
    pin_memory: bool = True,
):
    merged_path = cesm_path / "merged"
    var_names = list(sorted([p.stem for p in merged_path.glob("*")]))
    print(f"Found {len(var_names)} variables: {var_names}")

    # wandb.init(project="ssh", name="auto-encoder-run")
    device = torch.device("cuda:0")

    tfm = nn.Sequential(
        # Lambda(lambda ins: ins.compute()),
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
        AddLatLong(
            grid_path=cesm_path / "grid_info.nc",
        ),
        RandomRotatedRegrid(
            grid_path=cesm_path / "grid_info.nc",
            p=0.5,
        ),
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

    climate_ds = ClimateDataset(
        merged_path,
        members=members,
        variables=var_names,
        interval=12,
        loading_time_chunck_size=24,
        transform=tfm,
    )
    climate_dl = DataLoader(
        climate_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        shuffle=True,
        persistent_workers=persistent_workers,
        pin_memory=pin_memory,
    )

    pbar = tqdm(climate_ds, desc="Browsing dataloader")
    i = 0
    for batch in pbar:
        # (batch_size, interval, num_vars, lat, lon)
        fields = batch["vars"].to(device)
        masks = batch["land_mask"].to(device)
        print(f"Fields shape: {fields.shape}, Masks shape: {masks.shape}")
        if i >= 5:
            break

    pbar = tqdm(climate_dl, desc="Browsing dataloader")
    i = 0
    for batch in pbar:
        # (batch_size, interval, num_vars, lat, lon)
        fields = batch["vars"].to(device)
        masks = batch["land_mask"].to(device)
        print(f"Fields shape: {fields.shape}, Masks shape: {masks.shape}")
        if i >= 5:
            break



    # for e in range(epochs):
    #     pbar = tqdm(climate_dl, desc=f"EarchAE {e + 1}/{epochs}")
    #     total_loss = 0.0
    #     num_batches = 0
    #     for batch in pbar:
    #         with torch.autocast(device_type="cuda", dtype=bhalf):
    #             # (batch_size, interval, num_vars, lat, lon)
    #             fields = batch["vars"].to(device)
    #             # (batch_size, num_vars, lat, lon)
    #             mask = batch["land_mask"].to(device)
    #             mask_encoded = model.encode_land_mask(mask)
    #             fields = fields + mask_encoded.unsqueeze(1)
    #             fields = fields.transpose(1, 2).contiguous()
    #             fields = fields.view(-1, *fields.shape[2:])

    #             _, reconstruction = model(fields)
    #             reconstruction = reconstruction.masked_fill(
    #                 mask.unsqueeze(1) == 0, 0.0
    #             )

    #             loss = criterion(fields, reconstruction)

    #         # optimizer.zero_grad()
    #         # loss.backward()
    #         # optimizer.step()
    #         # clip_grad_norm_(model.parameters(), max_norm=1.0)
    #         num_batches += 1
    #         loss_val = loss.item()
    #         total_loss += loss_val
    #         pbar.set_postfix(loss=total_loss / num_batches)

    #         if (step + 1) % 1000 == 0:
    #             torch.save(
    #                 model.state_dict(),
    #                 f"/buckets/checkpoints/earthae_step_{step}.ckpt",
    #             )
    #         # wandb.log({"epoch": e, "step": step, "mse_loss": loss_val})

    #         step += 1

    #     torch.save(model.state_dict(), f"/buckets/checkpoints/earthae_{e}.ckpt")


if __name__ == "__main__":
    main()
