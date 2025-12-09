#!/usr/bin/env python3
import argparse
import sys
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except Exception:
    HAS_CARTOPY = False


def parse_args():
    p = argparse.ArgumentParser(description="Compute and plot SSH linear trend map from NetCDF with TLAT/TLONG.")
    p.add_argument("--nc", required=True, help="Path to NetCDF file")
    p.add_argument("--var", default="SSH", help="Variable name for sea surface height (default: SSH)")
    p.add_argument("--time-dim", default=None, help="Time dimension name (auto-detected if not provided)")
    p.add_argument("--lat-var", default=None, help="Latitude var name (default: TLAT or lat)")
    p.add_argument("--lon-var", default=None, help="Longitude var name (default: TLONG or lon)")
    p.add_argument("--to-mm", action="store_true", help="Convert trend to mm/yr (assumes input is meters)")
    p.add_argument("--out", default="ssh_trend.png", help="Output figure filename")
    p.add_argument("--percentile", type=float, default=98.0, help="Symmetric color limit percentile (default: 98)")
    return p.parse_args()


def time_to_years(time_da: xr.DataArray) -> xr.DataArray:
    """
    Convert time coordinate to float years elapsed since the first sample.
    Works with numpy datetime64 and cftime.
    """
    vals = time_da.values
    # numpy datetime64
    if np.issubdtype(np.array(vals).dtype, np.datetime64):
        years = (vals - vals[0]) / np.timedelta64(1, "D") / 365.25
        return xr.DataArray(years, dims=time_da.dims, coords=time_da.coords, name="time_years")
    # cftime or object datetimes
    try:
        import cftime  # type: ignore
        cal = time_da.attrs.get("calendar", "standard")
        units = "days since 1970-01-01 00:00:00"
        # date2num supports list of cftime datetimes
        days = cftime.date2num(vals.tolist(), units=units, calendar=cal)
        years = (days - days[0]) / 365.25
        return xr.DataArray(years, dims=time_da.dims, coords=time_da.coords, name="time_years")
    except Exception as e:
        raise RuntimeError(f"Unsupported time coordinate type: {type(vals[0])}. Error: {e}")


def linear_trend_per_year(da: xr.DataArray, time_dim: str = "time") -> xr.DataArray:
    """
    OLS slope per year along time_dim.
    Returns DataArray with same non-time dims. Units: (da.units)/year.
    """
    if time_dim not in da.dims:
        raise ValueError(f"time dimension '{time_dim}' not found in data variable dims {da.dims}")

    t_years = time_to_years(da[time_dim])
    # center time and data
    x = t_years - t_years.mean(time_dim, skipna=True)
    y = da
    y_mean = y.mean(time_dim, skipna=True)
    x2_sum = (x ** 2).sum(time_dim, skipna=True)
    # slope = sum( (x) * (y - y_mean) ) / sum( x^2 )
    slope = ((x * (y - y_mean)).sum(time_dim, skipna=True) / x2_sum)
    slope.attrs = dict(da.attrs)  # copy attrs if any
    # Annotate units as per year if present
    units = da.attrs.get("units", None)
    if units:
        slope.attrs["units"] = f"{units}/yr"
    else:
        slope.attrs["units"] = "per year"
    slope.name = f"{getattr(da, 'name', 'var')}_trend"
    return slope


def find_coord_var(ds: xr.Dataset, candidates):
    for name in candidates:
        if name in ds:
            return name
        if name in ds.coords:
            return name
    # Also search case-insensitively
    lower = {k.lower(): k for k in list(ds.data_vars) + list(ds.coords)}
    for name in candidates:
        if name.lower() in lower:
            return lower[name.lower()]
    return None


def main():
    args = parse_args()
    try:
        ds = xr.open_dataset(args.nc, decode_times=True)
    except Exception as e:
        print(f"Failed to open dataset: {e}", file=sys.stderr)
        sys.exit(1)

    if args.var not in ds:
        print(f"Variable '{args.var}' not found. Available: {list(ds.data_vars)}", file=sys.stderr)
        sys.exit(1)

    var = ds[args.var]

    # Detect time dimension
    time_dim = args.time_dim
    if time_dim is None:
        # pick first dim that looks like time or has datetime dtype
        cand = [d for d in var.dims if "time" in d.lower()]
        if cand:
            time_dim = cand[0]
        else:
            # fall back to 'time' if present in ds
            time_dim = "time" if "time" in ds.dims else None
    if time_dim is None or time_dim not in var.dims:
        print(f"Could not determine time dimension for '{args.var}'. Provide --time-dim.", file=sys.stderr)
        sys.exit(1)

    # Locate lat/lon
    lat_name = args.lat_var or find_coord_var(ds, ["TLAT", "lat", "latitude"])
    lon_name = args.lon_var or find_coord_var(ds, ["TLONG", "lon", "longitude"])
    if lat_name is None or lon_name is None:
        print("Could not find TLAT/TLONG or lat/lon variables. Use --lat-var/--lon-var.", file=sys.stderr)
        sys.exit(1)

    lat = ds[lat_name]
    lon = ds[lon_name]

    # Compute trend per year
    trend = linear_trend_per_year(var, time_dim=time_dim)

    # Optional unit conversion to mm/yr (common for SSH in meters)
    scale_label = ""
    if args.to_mm:
        trend = trend * 1000.0
        trend.attrs["units"] = "mm/yr"
        scale_label = " (mm/yr)"

    # Get time range for title
    t = ds[time_dim]
    try:
        t0 = np.datetime_as_string(np.array(t.values[0]).astype("datetime64[D]"))
        t1 = np.datetime_as_string(np.array(t.values[-1]).astype("datetime64[D]"))
    except Exception:
        # fallback to string repr
        t0 = str(t.values[0])
        t1 = str(t.values[-1])

    # Prepare lon/lat for plotting: wrap lon to [-180, 180], fill masked, and mask trend where invalid
    lon_values = np.asanyarray(lon.values)
    lat_values = np.asanyarray(lat.values)
    # If masked arrays, convert to ndarrays with NaNs first
    if np.ma.isMaskedArray(lon_values):
        lon_values = lon_values.filled(np.nan)
    if np.ma.isMaskedArray(lat_values):
        lat_values = lat_values.filled(np.nan)
    # Wrap longitudes
    lon_values = np.where(lon_values > 180, lon_values - 360, lon_values)
    # Invalid where lat/lon are not finite
    invalid = ~np.isfinite(lon_values) | ~np.isfinite(lat_values)
    # pcolormesh requires finite X/Y; fill invalid with finite dummies
    lon_clean = lon_values.copy()
    lat_clean = lat_values.copy()
    lon_clean[invalid] = 0.0
    lat_clean[invalid] = 0.0
    # Mask trend at invalid grid points so they are not rendered
    invalid_da = xr.DataArray(invalid, dims=lat.dims, coords=lat.coords)
    trend_plot = trend.where(~invalid_da)

    # Determine color limits from percentile
    abs_vals = np.abs(trend.values)
    finite = abs_vals[np.isfinite(abs_vals)]
    if finite.size == 0:
        print("Trend contains no finite values to plot.", file=sys.stderr)
        sys.exit(1)
    vmax = np.percentile(finite, args.percentile)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    # Plot
    if not HAS_CARTOPY:
        print("cartopy is required for curvilinear TLAT/TLONG plotting. Install: pip install cartopy", file=sys.stderr)
        sys.exit(1)

    proj = ccrs.Robinson()
    pc = ccrs.PlateCarree()

    fig = plt.figure(figsize=(11, 5.5), constrained_layout=True)
    ax = plt.axes(projection=proj)
    ax.set_global()
    ax.coastlines(linewidth=0.6)
    ax.add_feature(cfeature.LAND, facecolor="0.9")
    ax.gridlines(draw_labels=False, linewidth=0.3, color="0.5", alpha=0.5, linestyle="--")

    h = ax.pcolormesh(
        lon_clean, lat_clean, trend_plot.values,
        transform=pc, cmap="RdBu_r", norm=norm, shading="auto"
    )
    cbar = plt.colorbar(h, ax=ax, orientation="horizontal", pad=0.05, fraction=0.06)
    cbar.set_label(f"SSH linear trend{scale_label} ({trend.attrs.get('units', '')})")

    title_units = trend.attrs.get("units", "")
    ax.set_title(f"SSH linear trend {t0} to {t1} [{title_units}]")

    plt.savefig(args.out, dpi=200)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()