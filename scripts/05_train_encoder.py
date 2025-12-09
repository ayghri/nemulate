from nemulate import data as ndata
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

from nemulate.data.transforms import RandomRotatedRegrid
from nemulate.data.transforms import FieldScaler
from nemulate.data.transforms import SubstractForcedResponse
from nemulate.data.transforms import UnwrapFields
from nemulate.data.transforms import AddYearMonth
from nemulate.data.transforms import AddNanMask
from nemulate.data.transforms import AddXYZ
from nemulate.data.transforms import Lambda
from torch.utils.data import DataLoader
from tqdm import tqdm


from nemulate.datasets import ClimateDataset
from nemulate.data.sources import get_cesm2_members_ids

from torch import nn
import torch

from nemulate.models.coders import EarthAE
from torchinfo import summary
import wandb

wandb.init(project="ssh", name="auto-encoder-run")

bhalf = torch.bfloat16
device = torch.device("cuda:0")


sim_path = Path("/buckets/datasets/ssh/simulations/cesm2/monthly")
# merged_path = sim_path / "merged"
merged_path = Path("/misc/datasets/merged/")


var_names = list(sorted([p.stem for p in merged_path.glob("*")]))
print(f"Found {len(var_names)} variables: {var_names}")


# order is important, scaler and forced response are computed in the native grid

tfm = nn.Sequential(
    Lambda(lambda ins: ins.compute()),
    SubstractForcedResponse(
        sim_path.joinpath("moments"),
        "{var}_forced_response_1.nc",
        var_names=var_names,
    ),
    FieldScaler(
        sim_path / "stats" / "all_var_time_stats.nc",
        field_format="{var}_var_globalstd",
        var_names=var_names,
    ),
    RandomRotatedRegrid(
        grid_path=sim_path / "grid_info.nc",
        target_degree=1.0,
        rotation_lows_deg=(-30, -60),
        rotation_highs_deg=(30, 60),
        rotation_axis="xy",
        queue_length=10,
        refresh_period=10,
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


model = EarthAE(width_base=64, num_layers=4).to(device, dtype=bhalf)

with torch.autocast(device_type="cuda:0", dtype=bhalf):
    print(summary(model, input_data=torch.randn(16, 5, 180, 360).to(device)))


criterion = nn.MSELoss()

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)


climate_ds = ClimateDataset(
    merged_path,
    members=members,
    variables=var_names,
    interval=6,
    loading_time_chunck_size=24,
    transform=tfm,
)

climate_dl = DataLoader(
    climate_ds,
    batch_size=4,
    shuffle=True,
    num_workers=8,
    prefetch_factor=4,
    persistent_workers=True,
    pin_memory=True,
)

epochs = 10
step = 0

print("Saving initial ckpt")
torch.save(model.state_dict(), f"/buckets/checkpoints/earthae_initial.ckpt")

for e in range(epochs):
    pbar = tqdm(climate_dl, desc=f"EarchAE {e + 1}/{epochs}")
    total_loss = 0.0
    num_batches = 0
    for batch in pbar:
        with torch.autocast(device_type="cuda", dtype=bhalf):
            fields = batch["vars"].to(device)
            fields = fields.transpose(1, 2).contiguous()
            fields = fields.view(-1, *fields.shape[2:])

            _, reconstruction = model(fields)

            loss = criterion(fields, reconstruction)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        num_batches += 1
        loss_val = loss.item()
        total_loss += loss_val
        pbar.set_postfix(loss=total_loss / num_batches)

        if (step + 1) % 1000 == 0:
            torch.save(
                model.state_dict(),
                f"/buckets/checkpoints/earthae_step_{step}.ckpt",
            )
        wandb.log(
            {
                "epoch": e,
                "step": step,
                "mse_loss": loss_val,
            }
        )

        step += 1

    torch.save(model.state_dict(), f"/buckets/checkpoints/earthae_{e}.ckpt")
