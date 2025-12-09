import nemulate.data as ndata
from pathlib import Path
from tqdm import tqdm
import xarray as xr
from argparse import ArgumentParser

monthly_path = Path("/buckets/datasets/ssh/simulations/cesm2/monthly")
cesm2_path = monthly_path / "merged"
momments_path = monthly_path / "moments"
stats_path = monthly_path / "stats"
stats_path.mkdir(exist_ok=True)


# var_names = [p.stem for p in cesm2_path.glob("*")]
# print(f"Found {len(var_names)} variables: {var_names}")
# for v_name in var_names:
def compute_stats_for_variable(v_name: str) -> None:
    forced_response = ndata.read_netcdf(
        momments_path / f"{v_name}_forced_response_1.nc", time_chunk_size=None
    )

    print(f"Computing variability stats {v_name}")
    members = ndata.get_cesm2_members_ids(cesm2_path, v_name)

    def generate_members():
        for m in tqdm(members):
            yield ndata.get_cesm2_member_months(
                cesm2_path, v_name, m, time_chunk_size=None
            )

    sq_mean = None
    all_min = None
    all_max = None
    for member_data in generate_members():
        member_data = member_data - forced_response
        if sq_mean is None:
            sq_mean = (member_data) ** 2 / len(members)
            all_min = member_data
            all_max = member_data
        else:
            sq_mean = sq_mean + (member_data) ** 2 / len(members)
            all_min = xr.ufuncs.minimum(all_min, member_data)
            all_max = xr.ufuncs.maximum(all_max, member_data)
    out_path = f"{v_name}_variability_%s.nc"

    print(f"Saving variability stats for '{v_name}' to {stats_path}")
    sq_mean.to_netcdf(stats_path / (out_path % "sqmean"))
    all_min.to_netcdf(stats_path / (out_path % "min"))
    all_max.to_netcdf(stats_path / (out_path % "min"))


arg_parser = ArgumentParser(
    description="Compute variability statistics for CESM2 variables."
)
arg_parser.add_argument(
    "--variables",
    nargs="+",
    help="Subset of variable names to process. Defaults to all.",
)

if __name__ == "__main__":
    args = arg_parser.parse_args()
    var_names = args.variables

    var_names = sorted(
        [
            p.name
            for p in cesm2_path.iterdir()
            if p.is_dir() and p.name in var_names
        ]
    )
    print(f"Found {len(var_names)} variables: {var_names}")

    for v_name in var_names:
        compute_stats_for_variable(v_name)
