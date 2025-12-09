#!/usr/bin/env python3
"""
Batch regrid all CESM2 variables/members to 1° and save ocean-only values.

- For each variable under --input, iterate all members (*.nc)
- Regrid to a uniform 1x1 degree grid (xesmf bilinear, periodic)
- Compute a per-variable land mask from the first available member's first timestep
- Save the mask once as metadata: lat, lon, ocean_mask, and ocean indices
- For each member, save compressed NPZ with values at ocean points only, and time (years, months)

Example:
  python scripts/04_batch_regrid_ocean_only.py \
      --input /buckets/datasets/ssh/simulations/cesm2/merged/monthly \
      --output /buckets/datasets/ssh/processed/cesm2_1deg_ocean_only \
      --grid-nc /buckets/datasets/ssh/simulations/cesm2/grid_info.nc \
      --start-year 1900
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import xarray as xr

from nemulate import data as ndata
from nemulate.data.regrid import CESM2Regrid


def infer_grid_path(input_dir: Path) -> Path:
    # Heuristic based on notebooks: input/../.. / grid_info.nc
    candidate = (input_dir / "../.." / "grid_info.nc").resolve()
    if candidate.exists():
        return candidate
    raise FileNotFoundError(
        f"Could not infer grid_info.nc; please pass --grid-nc. Tried: {candidate}"
    )


def list_variables(input_dir: Path) -> list[str]:
    return sorted([p.name for p in input_dir.iterdir() if p.is_dir()])


def list_members(var_dir: Path) -> list[str]:
    return sorted([p.stem for p in var_dir.glob("*.nc")])


def load_member_dataset(input_dir: Path, var_name: str, member: str, start_year: int) -> xr.Dataset:
    return ndata.get_cesm2_member_months(
        input_dir, var_name, member, start_year=start_year
    )


def compute_mask_and_coords(da: xr.DataArray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Expect da dims (time, lat, lon) with 2D lat/lon coords
    sample = da.isel(time=0)
    lat = sample["lat"].values
    lon = sample["lon"].values
    values = sample.values
    ocean_mask = ~np.isnan(values)
    assert ocean_mask.ndim == 2, "Expected 2D mask after regridding"
    return ocean_mask, lat, lon


def times_to_year_month(ds: xr.Dataset) -> Tuple[np.ndarray, np.ndarray]:
    tvals = ds["time"].values
    # Handle cftime or numpy datetimes
    years = np.array([getattr(t, "year", int(str(t)[:4])) for t in tvals], dtype=np.int16)
    months = np.array([getattr(t, "month", int(str(t)[5:7])) for t in tvals], dtype=np.int8)
    return years, months


def save_mask(mask_path: Path, ocean_mask: np.ndarray, lat: np.ndarray, lon: np.ndarray) -> None:
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    # Indices for convenience
    ii, jj = np.where(ocean_mask)
    np.savez_compressed(
        mask_path,
        ocean_mask=ocean_mask,
        lat=lat,
        lon=lon,
        i=ii.astype(np.int32),
        j=jj.astype(np.int32),
        shape=np.array(ocean_mask.shape, dtype=np.int16),
    )


def save_member_ocean_values(out_path: Path, values_2d_time: np.ndarray, ocean_mask: np.ndarray, years: np.ndarray, months: np.ndarray, dtype: np.dtype | None = None) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if dtype is not None:
        values_2d_time = values_2d_time.astype(dtype)
    # Flatten ocean-only values per time step into (T, N)
    T = values_2d_time.shape[0]
    N = ocean_mask.sum().item()
    ocean_vals = values_2d_time[:, ocean_mask].reshape(T, N)
    np.savez_compressed(
        out_path,
        values=ocean_vals,
        years=years,
        months=months,
        dtype=str(values_2d_time.dtype),
        shape_full=np.array(values_2d_time.shape[1:], dtype=np.int16),
    )


def process_variable(
    input_dir: Path,
    output_dir: Path,
    var_name: str,
    regridder: CESM2Regrid,
    degree: int,
    start_year: int,
    dtype: str | None,
) -> None:
    var_dir = input_dir / var_name
    members = list_members(var_dir)
    if not members:
        print(f"[WARN] No members found for {var_name}")
        return

    out_var_dir = output_dir / var_name
    mask_file = out_var_dir / f"{var_name}_land_mask_{degree}deg.npz"

    ocean_mask: np.ndarray
    lat: np.ndarray
    lon: np.ndarray

    # Compute or load mask
    if mask_file.exists():
        m = np.load(mask_file)
        ocean_mask = m["ocean_mask"]
        lat = m["lat"]
        lon = m["lon"]
        print(f"[INFO] Loaded mask for {var_name} from {mask_file}")
    else:
        # Build mask from first available member's first timestep after regridding
        first_member = members[0]
        ds = load_member_dataset(input_dir, var_name, first_member, start_year)
        da = ds[var_name]
        da_re = regridder.regrid(da, degree)
        ocean_mask, lat, lon = compute_mask_and_coords(da_re)
        save_mask(mask_file, ocean_mask, lat, lon)
        print(f"[INFO] Saved mask for {var_name} -> {mask_file}")

    # Process each member
    for memb in members:
        out_member = out_var_dir / f"{memb}_ocean_{degree}deg.npz"
        if out_member.exists():
            print(f"[SKIP] {var_name}/{memb} (exists)")
            continue
        print(f"[RUN ] {var_name}/{memb}")
        ds = load_member_dataset(input_dir, var_name, memb, start_year)
        da = ds[var_name]
        da_re = regridder.regrid(da, degree)
        # Align mask shape
        vals = da_re.values  # (time, lat, lon)
        assert vals.ndim == 3 and vals.shape[1:] == ocean_mask.shape, (
            f"Shape mismatch for {var_name}/{memb}: {vals.shape} vs mask {ocean_mask.shape}"
        )
        years, months = times_to_year_month(ds)
        save_member_ocean_values(
            out_member,
            vals,
            ocean_mask,
            years,
            months,
            dtype=np.dtype(dtype) if dtype else None,
        )


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--input", type=Path, required=True, help="Input CESM2 merged monthly directory (variables as subfolders)")
    ap.add_argument("--output", type=Path, required=True, help="Output directory for ocean-only NPZs")
    ap.add_argument("--grid-nc", type=Path, default=None, help="Path to grid_info.nc (if omitted, inferred)")
    ap.add_argument("--degree", type=int, default=1, help="Target grid resolution in degrees")
    ap.add_argument("--start-year", type=int, default=1850, help="Start year to load from")
    ap.add_argument("--vars", nargs="*", default=None, help="Optional list of variables to process; defaults to all found under input")
    ap.add_argument("--dtype", type=str, default=None, help="Optional dtype to cast saved values (e.g., float32)")

    args = ap.parse_args()

    input_dir: Path = args.input
    output_dir: Path = args.output
    grid_nc: Path = args.grid_nc or infer_grid_path(input_dir)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    vars_to_process = args.vars or list_variables(input_dir)
    if not vars_to_process:
        raise RuntimeError(f"No variable folders found under {input_dir}")

    print(f"[INFO] Using grid file: {grid_nc}")
    regridder = CESM2Regrid(grid_nc)

    for var_name in vars_to_process:
        try:
            process_variable(
                input_dir,
                output_dir,
                var_name,
                regridder,
                degree=int(args.degree),
                start_year=int(args.start_year),
                dtype=args.dtype,
            )
        except Exception as e:
            print(f"[ERROR] Failed {var_name}: {e}")


if __name__ == "__main__":
    main()
