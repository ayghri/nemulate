"""Generic, clean plotting utilities for scalar fields on the globe.

This module provides a small, consistent API for plotting 2D fields
on latitude/longitude grids using Cartopy + Matplotlib. It supports
1D or 2D lat/lon coordinates, optional cyclic wrapping, diverging
norms centered on a value (e.g., 0 for trends/anomalies), and simple
frame export for time series.
"""

from typing import Iterable, Mapping, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.util import add_cyclic_point
from cartopy.mpl.geoaxes import GeoAxes
from matplotlib.colors import Colormap, TwoSlopeNorm, Normalize
from tqdm import tqdm

__all__ = [
    "plot_field",
    "plot_trend",
    "plot_frames",
]


# ----------------------------
# helpers
# ----------------------------


def _get_projection(
    projection: str | ccrs.Projection,
    center_longitude: float,
) -> ccrs.Projection:
    if isinstance(projection, ccrs.Projection):
        return projection
    name = (projection or "PlateCarree").lower()
    if name in {"platecarree", "pc"}:
        return ccrs.PlateCarree(central_longitude=center_longitude)
    if name in {"robinson", "rb"}:
        # cartopy type stubs may expect int; cast to int to satisfy
        return ccrs.Robinson(central_longitude=int(center_longitude))
    if name in {"mollweide", "mw"}:
        return ccrs.Mollweide(central_longitude=int(center_longitude))
    # default fallback
    return ccrs.PlateCarree(central_longitude=center_longitude)


def _maybe_add_cyclic(
    field: np.ndarray,
    lons: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Add a cyclic point along longitude when lons is 1D and matches field."""
    if field.ndim != 2:
        return field, lons
    if lons.ndim == 1 and field.shape[1] == lons.shape[0]:
        f, lons_cyc = add_cyclic_point(field, coord=lons)
        return f, lons_cyc
    return field, lons


def _coerce_lat_lon(
    lats: np.ndarray,
    lons: np.ndarray,
    field: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return 2D meshgrid lat/lon matching field for contour/pcolormesh."""
    if lats.ndim == 2 and lons.ndim == 2:
        return lats, lons
    if lats.ndim == 1 and lons.ndim == 1:
        # Assume field shape is (lat, lon). meshgrid returns (X, Y) == (lon2d, lat2d)
        lon2d, lat2d = np.meshgrid(lons, lats)
        return lat2d, lon2d
    raise ValueError("lats/lons must be both 1D or both 2D arrays.")


def _finite_min_max(arr: np.ndarray) -> Tuple[float, float]:
    arr = np.asarray(arr)
    if np.isnan(arr).all():
        return 0.0, 1.0
    return np.nanmin(arr), np.nanmax(arr)


def plot_field(
    field: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    *,
    title: str | None = None,
    unit: str | None = None,
    projection: str | ccrs.Projection = "Robinson",
    center_longitude: float = 0.0,
    cmap: str | Colormap = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    norm: Normalize | None = None,
    levels: int | Sequence[float] | None = 60,
    add_land: bool = True,
    coastline_resolution: str = "110m",
    gridlines: bool = True,
    colorbar: bool = True,
    cbar_orientation: str = "vertical",
    cbar_kwargs: dict | None = None,
    figsize: Tuple[int, int] = (12, 6),
    add_cyclic: bool = True,
    mask: np.ndarray | None = None,
    show: bool = False,
    save: str | None = None,
):
    """Plot a scalar 2D field on a lat/lon grid.

    - Accepts 1D or 2D lat/lon arrays. If 1D, assumes field shape (lat, lon).
    - Optionally adds a cyclic point along longitude.
    - Returns (fig, ax, mappable, cbar) for further customization.
    """

    data = np.array(field, dtype=float)
    if mask is not None:
        data = np.ma.array(data, mask=mask)

    # Add cyclic if applicable
    lon_vec = lons
    if add_cyclic and lons.ndim == 1:
        data, lon_vec = _maybe_add_cyclic(data, lons)

    # Prepare grid for plotting
    lat2d, lon2d = _coerce_lat_lon(lats, lon_vec, data)

    # value range
    lo, hi = _finite_min_max(data)
    vmin = lo if vmin is None else vmin
    vmax = hi if vmax is None else vmax

    # projection and axes
    proj = _get_projection(projection, center_longitude)
    fig = plt.figure(figsize=figsize)
    ax: GeoAxes = plt.axes(projection=proj)  # type: ignore[assignment]
    ax.set_global()
    ax.coastlines(resolution=coastline_resolution, linewidth=0.8)
    if add_land:
        ax.add_feature(cfeature.LAND, zorder=10, facecolor="white")
    if gridlines:
        ax.gridlines(linestyle="-", color="grey", alpha=0.4)

    # contourf levels
    used_levels: Iterable[float] | int | None = levels
    if isinstance(levels, int) and levels > 0:
        used_levels = np.linspace(vmin, vmax, levels)

    mappable = ax.contourf(
        lon2d,
        lat2d,
        data,
        levels=used_levels,
        transform=ccrs.PlateCarree(),
        corner_mask=True,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        norm=norm,
        extend="both",
        antialiased=False,
        # edge_colors="none"
        # linewidth=0,
    )

    cbar = None
    if colorbar:
        cbar_kwargs = dict(pad=0.02, aspect=20, shrink=0.8) | (
            cbar_kwargs or {}
        )
        cbar = plt.colorbar(
            mappable, ax=ax, orientation=cbar_orientation, **cbar_kwargs
        )
        if unit:
            cbar.set_label(unit, size=11, rotation=0, labelpad=12)

    if title:
        ax.set_title(title)

    if save:
        fig.savefig(save, bbox_inches="tight", dpi=150)
    if show:
        plt.show()

    return fig, ax, mappable, cbar


def plot_trend(
    field: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    *,
    title: str | None = None,
    unit: str | None = None,
    center: float = 0.0,
    cmap: str | Colormap = "coolwarm",
    vmin: float | None = None,
    vmax: float | None = None,
    levels: int | Sequence[float] | None = 60,
    projection: str | ccrs.Projection = "Robinson",
    center_longitude: float = 0.0,
    **kwargs,
):
    """Plot a diverging trend/anomaly map centered on a value (default 0)."""
    data = np.array(field, dtype=float)
    lo, hi = _finite_min_max(data)
    vmin = lo if vmin is None else vmin
    vmax = hi if vmax is None else vmax
    norm = TwoSlopeNorm(vmin=vmin, vcenter=center, vmax=vmax)
    return plot_field(
        data,
        lats,
        lons,
        title=title,
        unit=unit,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        levels=levels,
        projection=projection,
        center_longitude=center_longitude,
        cbar_orientation=kwargs.pop("cbar_orientation", "horizontal"),
        **{**kwargs, "norm": norm},
    )


def plot_frames(
    time_to_field: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    *,
    name_format: str = "frame_{:04d}.png",
    title: str | None = None,
    cmap: str | Colormap = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    levels: int | Sequence[float] | None = 60,
    projection: str | ccrs.Projection = "Robinson",
    center_longitude: float = 0.0,
    unit: str | None = None,
    add_cyclic: bool = True,
    figsize: Tuple[int, int] = (12, 6),
) -> None:
    """Export a sequence of PNG frames for a time-indexed mapping of fields."""
    for i in tqdm(range(time_to_field.shape[0])):
        ts = i
        fld = time_to_field[i]
        fig, ax, *_ = plot_field(
            fld,
            lats,
            lons,
            title=(f"{title} - {ts}" if title else str(ts)),
            unit=unit,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            levels=levels,
            projection=projection,
            center_longitude=center_longitude,
            add_cyclic=add_cyclic,
            figsize=figsize,
            show=False,
        )
        fig.savefig(name_format.format(i), bbox_inches="tight", dpi=150)
        plt.close(fig)
