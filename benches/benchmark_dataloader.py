from pathlib import Path
from time import perf_counter
from torch import nn
from torch import nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import click

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

from tqdm import tqdm


class MarkerTransform(nn.Module):
    def __init__(self, marker, name):
        super().__init__()
        self.marker = marker
        self.name = name

    def forward(self, ins):
        self.marker.mark(self.name)
        return ins


class TransformTimer:
    def __init__(self):
        super().__init__()
        self.start = 0
        self.markers = {}

    def mark(self, name):
        now = perf_counter()
        self.markers[name] = now - self.start + self.markers.get(name, 0)
        self.start = now

    def __enter__(self):
        self.start = perf_counter()
        self.markers = {}
        return self.markers

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def checkpoint(self, name):
        return MarkerTransform(self, name)


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--cesm-path",
    type=click.Path(path_type=Path, exists=True, file_okay=False),
    required=True,
    help="Root folder containing CESM data",
)
def main(cesm_path: Path):
    merged_path = cesm_path / "merged"
    var_names = list(sorted([p.stem for p in merged_path.glob("*")]))
    print(f"Found {len(var_names)} variables: {var_names}")

    timer = TransformTimer()
    tfm = nn.Sequential(
        timer.checkpoint("start"),
        Lambda(lambda ins: ins.compute()),
        timer.checkpoint("compute"),
        SubstractForcedResponse(
            cesm_path.joinpath("moments"),
            "{var}_forced_response_1.nc",
            var_names=var_names,
        ),
        timer.checkpoint("forced_response"),
        FieldScaler(
            cesm_path / "stats" / "all_var_time_stats.nc",
            field_format="{var}_var_globalstd",
            var_names=var_names,
        ),
        timer.checkpoint("scaler"),
        AddLatLong(
            grid_path=cesm_path / "grid_info.nc",
        ),
        timer.checkpoint("add_lat_long"),
        RandomRotatedRegrid(
            grid_path=cesm_path / "grid_info.nc",
            target_degree=1.0,
            rotation_lows_deg=(-30, -60),
            rotation_highs_deg=(30, 60),
            rotation_axis="xy",
            queue_length=10,
            refresh_period=5,
        ),
        timer.checkpoint("random_rotate"),
        AddNanMask(var_names),
        timer.checkpoint("add_nan"),
        AddXYZ(),
        timer.checkpoint("add_xyz"),
        AddYearMonth(),
        timer.checkpoint("add_time"),
        UnwrapFields(
            {
                "vars": var_names,
                "land_mask": "land_mask",
                "year": "year",
                "month": "month",
                "xyz": ["cart_x", "cart_y", "cart_z"],
            }
        ),
        timer.checkpoint("unwrap"),
    )

    from nemulate.datasets import ClimateDataset
    from nemulate.data.sources import get_cesm2_members_ids

    members = get_cesm2_members_ids(merged_path, var_name=var_names)

    ds = ClimateDataset(
        merged_path,
        members=members,
        variables=var_names,
        interval=12,
        loading_time_chunck_size=24,
        transform=tfm,
    )

    num_iters = 256
    print("Dataset has:", len(ds), "elements")
    samples_indices = np.random.choice(len(ds), num_iters, replace=False)
    with timer as results:
        for i in tqdm(range(num_iters)):
            _ = ds[samples_indices[i]]

    for name, t in results.items():
        print(name, "took on average", t / num_iters)

    # dl = DataLoader(
    #     ds, batch_size=2, shuffle=True, num_workers=2, prefetch_factor=4
    # )


if __name__ == "__main__":
    main()
