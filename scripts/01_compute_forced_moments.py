from typing import Dict, Tuple, Iterable
from pathlib import Path

import click
from tqdm import tqdm
import xarray as xr

import nemulate.data as ndata


def compute_members_average(
    members: Iterable[xr.Dataset],
    num_members: int,
    degrees: Tuple[int, ...] = (1,),
) -> Dict[int, xr.Dataset]:
    """Mean of member**d across members for each degree in `degrees`."""

    averages: Dict[int, xr.Dataset] = {}

    iterator = members
    processed = 0

    for ds in iterator:
        for d in degrees:
            if d in averages:
                averages[d] = averages[d] + (ds**d) / num_members
            else:
                averages[d] = (ds**d) / num_members

        processed += 1

    if processed != num_members:
        raise RuntimeError(
            f"Warning: expected {num_members} members, got {processed}"
        )

    return averages


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--monthly-root",
    type=click.Path(path_type=Path, exists=True, file_okay=False),
    default=Path("/buckets/datasets/ssh/simulations/cesm2/merged/monthly"),
    show_default=True,
    help="Root containing one subdir per variable (merged monthly).",
)
@click.option(
    "--out-root",
    type=click.Path(path_type=Path, file_okay=False),
    default=Path("/buckets/datasets/ssh/simulations/cesm2/"),
    show_default=True,
    help="Directory to write output NetCDF files.",
)
@click.option(
    "--vars",
    "vars_",
    multiple=True,
    help="Subset of variable names to process (repeat option). Defaults to all.",
)
@click.option(
    "--degrees",
    multiple=True,
    type=int,
    default=(1, 2),
    show_default=True,
    help="Degrees to compute (d in member**d). Repeat option.",
)
@click.option("--verbose", is_flag=True, help="Show progress bars.")
def main(
    monthly_root: Path,
    out_root: Path,
    vars_: Tuple[str, ...],
    degrees: Tuple[int, ...],
    verbose: bool,
) -> None:
    """Compute forced moments (member-mean of member**d) for CESM2 variables."""
    if not monthly_root.exists():
        raise click.UsageError(f"Monthly root not found: {monthly_root}")
    out_root.mkdir(parents=True, exist_ok=True)

    var_names = list(vars_) or sorted(
        [p.name for p in monthly_root.iterdir() if p.is_dir()]
    )
    click.echo(f"Found {len(var_names)} variable(s): {var_names}")

    for v_name in var_names:
        click.echo(f"\nComputing forced moments for '{v_name}': {degrees}")
        members_ids = ndata.get_cesm2_members_ids(monthly_root, v_name)
        click.echo(f"Members: {len(members_ids)}")

        members_iter: Iterable[xr.Dataset] = (
            ndata.get_cesm2_member_months(
                monthly_root, v_name, m, time_chunk_size=None
            )
            for m in tqdm(
                members_ids, disable=not verbose, desc=f"Load {v_name}"
            )
        )

        deg_to_forced = compute_members_average(
            members_iter,
            num_members=len(members_ids),
            degrees=tuple(degrees),
        )

        click.echo(f"Saving forced moments for '{v_name}' to {out_root}")
        for degree, forced_ds in deg_to_forced.items():
            out_path = out_root / f"{v_name}_forced_response_{degree}.nc"
            forced_ds.to_netcdf(out_path)
            click.echo(f"  - saved: {out_path}")


if __name__ == "__main__":
    main()
