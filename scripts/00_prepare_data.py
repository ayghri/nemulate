from pathlib import Path
from typing import Tuple
import click

from nemulate.data.organize import merge_and_save_datasets


# DEFAULT_DATA_PATH = Path(
#     "/buckets/datasets/ssh/simulations/cesm2/monthly/download"
# )
# DEFAULT_STORAGE_PATH = Path(
#     "/buckets/datasets/ssh/simulations/cesm2/monthly/merged"
# )

CHUNK_SIZE = (24, 384, 320)
KEEP_DIMS = ("time", "nlat", "nlon")


def _discover_variables(data_path: Path, only: Tuple[str, ...]) -> list[Path]:
    vars_dirs = [p for p in data_path.glob("*") if p.is_dir()]
    if only:
        names = set(only)
        vars_dirs = [p for p in vars_dirs if p.name in names]
    return vars_dirs


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--data-path",
    type=click.Path(path_type=Path, exists=True, file_okay=False),
    # default=DEFAULT_DATA_PATH,
    # show_default=True,
    required=True,
    help="Root folder containing per-variable subdirectories.",
)
@click.option(
    "--storage-path",
    type=click.Path(path_type=Path, file_okay=False),
    # default=DEFAULT_STORAGE_PATH,
    # show_default=True,
    required=True,
    help="Destination for merged outputs (one folder per variable).",
)
@click.option(
    "--variable",
    "variables",
    multiple=True,
    required=True,
    help="Limit processing to these variable names (repeatable).",
)
@click.option(
    "--dry-run", is_flag=True, help="Print actions without modifying files."
)
@click.option("-v", "--verbose", is_flag=True, help="Verbose output.")
def main(
    data_path: Path,
    storage_path: Path,
    variables: Tuple[str, ...],
    dry_run: bool,
    verbose: bool,
):
    print("Arguments:")
    print(f"  data_path: {data_path}")
    print(f"  storage_path: {storage_path}")
    print(f"  variables: {variables}")
    print(f"  dry_run: {dry_run}")
    print(f"  verbose: {verbose}")
    """Prepare CESM2 monthly datasets: merge per-member files."""
    variable_dirs = _discover_variables(data_path, variables)
    if not variable_dirs:
        raise click.UsageError("No variable directories found.")
    if verbose:
        click.echo(
            f"Found {len(variable_dirs)} variables: {[p.name for p in variable_dirs]}"
        )

    # Merge each variable's members
    if not dry_run:
        storage_path.mkdir(parents=True, exist_ok=True)
    for var_path in variable_dirs:
        out_dir = storage_path / var_path.name
        if dry_run:
            click.echo(
                f"Would merge members for '{var_path.name}' -> {out_dir}"
            )
            continue
        merge_and_save_datasets(
            variable_path=var_path,
            variable=var_path.name,
            target_path=out_dir,
            chunk_size=CHUNK_SIZE,
            keep_dims=KEEP_DIMS,
        )


if __name__ == "__main__":
    main()
