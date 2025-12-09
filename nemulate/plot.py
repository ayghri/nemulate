from cartopy import feature, util
from cartopy.util import add_cyclic_point
from matplotlib import colormaps
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import Normalize, TwoSlopeNorm
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np


def plot_field(
    field: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    title: str,
    unit: str = "",
    figsize=(18, 9),
    cover_land: bool = True,
    cmap=None,
    center_long: int = 210,
    resolution: int = 110,
    limits: tuple | None = None,
    step: float | None = None,
    tick_step: float | None = None,
    gap_around_zero: float | None = None,
    projection: str = "Robinson",
    scatter_nodes: np.ndarray | None = None,
    s: float = 0.2,
):
    """
    Generic geospatial plotter for 2D climate fields on a uniform lat-lon grid.

    Behavior
    - vmin/vmax are computed dynamically from the input data (after cyclic add).
    - Colorbar/tesselation bounds are controlled via `limits` and `step` with defaults.

    Args
    - field: 2D array with shape (n_lat, n_lon)
    - lats: 1D array of latitudes (n_lat)
    - lons: 1D array of longitudes (n_lon)
    - limits: (low, high) bounds for colorbar. Default=(-2.2, 2.2)
    - step: bin size for colorbar bounds. Default=0.2
    - tick_step: spacing for colorbar ticks. Default = step*2
    - gap_around_zero: if provided and limits span negative/positive, exclude
      (-gap, +gap) from the bounds to create a white/neutral gap around zero.
    """
    assert field.ndim == 2, "field must be 2D (lat x lon)"
    assert field.shape == (lats.shape[0], lons.shape[0]), "shape mismatch"

    # Add cyclic point across longitude
    cyclic_map, cyclic_lons = util.add_cyclic_point(field, lons)

    # Dynamic vmin/vmax from data
    data_vmin = float(np.nanmin(cyclic_map))
    data_vmax = float(np.nanmax(cyclic_map))

    # Defaults for bounds
    if limits is None:
        limits = (-2.2, 2.2)
    low, high = limits
    if step is None:
        step = 0.2
    if tick_step is None:
        tick_step = step * 2

    # Build boundaries; optionally leave a gap around zero for divergent fields
    if gap_around_zero is not None and low < 0 < high:
        neg = np.arange(low, 0 + step, step)
        pos_start = max(gap_around_zero, step)
        pos = np.arange(pos_start, high + step, step)
        boundaries = np.concatenate([neg, pos])
    else:
        boundaries = np.arange(low, high + step, step)

    # Colormap defaults
    if cmap is None:
        # Use a sensible diverging map centered near zero by default
        cmap = plt.get_cmap("coolwarm")

    norm = BoundaryNorm(boundaries=boundaries, ncolors=cmap.N)

    plt.figure(figsize=figsize)
    if projection == "Robinson":
        ax = plt.axes(projection=ccrs.Robinson(central_longitude=center_long))
    else:
        ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=center_long))

    ax.set_global()  # type: ignore[attr-defined]
    ax.coastlines(resolution=f"{resolution}m", linewidth=1, color="grey")  # type: ignore[attr-defined]
    if cover_land:
        ax.add_feature(cfeature.LAND, facecolor="white", zorder=10)  # type: ignore[attr-defined]

    plt.title(title)

    # Use boundary-based normalization; vmin/vmax from data are not passed,
    # but computed above for potential future use (e.g., automated limits).
    cf = plt.contourf(
        cyclic_lons,
        lats,
        cyclic_map,
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        norm=norm,
        levels=boundaries,
        vmin=data_vmin,
        vmax=data_vmax,
        extend="both",
        antialiased=False,
    )

    if scatter_nodes is not None:
        ax.scatter(
            scatter_nodes[:, 1],
            scatter_nodes[:, 0],
            s=s,
            c="black",
            alpha=0.5,
            transform=ccrs.PlateCarree(),
        )

    cb = plt.colorbar(
        cf,
        ax=ax,
        orientation="horizontal",
        boundaries=boundaries,
        spacing="uniform",
        shrink=0.6,
        pad=0.05,
    )
    if unit:
        cb.set_label(unit, size=11, rotation=0, labelpad=6)
    cb.ax.tick_params(labelsize=9)
    cb.set_ticks(list(np.arange(low, high + tick_step / 2, tick_step).astype(float)))

    ax.gridlines(linestyle="-", color="grey", alpha=0.4)  # type: ignore[attr-defined]
    plt.tight_layout()
    plt.show()

def cartplot_uniform_grid(
    uniform_values,
    lat_linspace,
    long_linspace,
    title,
    figsize=(18, 9),
    cover_land=True,
    # mask=None,
):
    assert uniform_values.shape[0] == lat_linspace.shape[0]
    assert uniform_values.shape[1] == long_linspace.shape[0]
    plt.figure(figsize=figsize)
    cyclic_map, cyclic_long = util.add_cyclic_point(
        uniform_values, long_linspace
    )
    ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=210))
    v_min = cyclic_map.min()
    v_max = cyclic_map.max()

    levels = np.linspace(v_min, v_max, 60)
    resolution = 110
    plt.title(title)
    plt.contourf(
        cyclic_long,
        lat_linspace,
        cyclic_map,
        levels=levels,
        transform=ccrs.PlateCarree(),
        corner_mask=True,
        cmap="coolwarm",
        antialiased=False,
        # mask=mask,
    )
    if cover_land:
        ax.add_feature(feature.LAND, facecolor="white", zorder=10)
    ax.coastlines(resolution=f"{resolution}m", linewidth=1)
    # ax.gridlines(draw_labels=True, linestyle="--", color="black")
    # cbar = plt.colorbar(shrink=0.3)

    plt.show()


def cartplot(
    values,
    lats_mesh,
    longs_mesh,
    title,
    unit,
    mask=None,
    center_long=210,
    resolution=100,
):
    # fig = plt.figure(figsize=(16, 8))
    if mask is not None:
        unmask = np.logical_not(mask)
        values = values[unmask]
        lats_mesh = lats_mesh[unmask]
        longs_mesh = longs_mesh[unmask]
    _ = plt.figure(figsize=(16, 8))
    ax = plt.axes(projection=ccrs.Robinson(central_longitude=center_long))
    ax.set_global()
    ax.coastlines(resolution=f"{resolution}m", linewidth=1)
    ax.gridlines(linestyle="--", color="black")
    plt.contourf(
        longs_mesh,
        lats_mesh,
        values,
        transform=ccrs.PlateCarree(central_longitude=center_long),
        cmap=plt.cm.jet,  # pyright: ignore
    )

    plt.title(title)
    cb = plt.colorbar(
        ax=ax, orientation="vertical", pad=0.02, aspect=16, shrink=0.8
    )
    cb.set_label(unit, size=12, rotation=0, labelpad=15)
    cb.ax.tick_params(labelsize=12)
    plt.show()


def timelapse_frames(
    time_to_data: dict, longs, lats, title, name_format: str, levels=None
):
    for i, timestamp in enumerate(time_to_data.keys()):
        # fig = plt.figure(figsize=(16, 8))
        _ = plt.figure(figsize=(16, 8))
        ax = plt.axes(projection=ccrs.Robinson(central_longitude=210))
        ax.set_global()
        ax.coastlines(resolution="110m", linewidth=1)
        # ax.gridlines(linestyle="--", color="black")
        # values = twosat[m].variables['sla'][:][0]
        values = time_to_data[timestamp]
        # current = datetime(m[0], m[1], 1)
        plt.contourf(
            longs,
            lats,
            values,
            levels=levels,
            transform=ccrs.PlateCarree(central_longitude=0),
            cmap=colormaps["jet"],
        )
        plt.title(f"{title} - {timestamp}", size=10)
        cb = plt.colorbar(
            ax=ax, orientation="vertical", pad=0.02, aspect=16, shrink=0.8
        )
        cb.set_label("m", size=12, rotation=0, labelpad=15)
        cb.ax.tick_params(labelsize=12)
        # plt.savefig(f"plots/sla_twosat_{i:04d}.png", bbox_inches='tight')
        plt.savefig(f"{name_format}_{i:04d}.png", bbox_inches="tight")


def saumya_plot(
    map_to_plot,
    mask,
    lats,
    longs,
    v_min=None,
    v_max=None,
    cmap="coolwarm",
):
    """
    Function that plots a contour plot of the 2D array (xr)
    """

    # map_to_plot = np.transpose(xr)
    # print(np.min(map), np.max(map), map.shape)

    # map_to_plot = np.ma.masked_where(np.isnan(map_to_plot), map_to_plot)
    # print(np.min(map_to_plot), np.max(map_to_plot), map_to_plot.shape)
    map_to_plot = np.ma.masked_where(mask, map_to_plot)

    # lats = dataset.variables["lat"][:]
    # lons = dataset.variables["lon"][:]

    map_to_plot, longs = add_cyclic_point(map_to_plot, coord=longs)

    ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=210))
    v_min = v_min if v_min is not None else map_to_plot.min()
    v_max = v_max if v_max is not None else map_to_plot.max()

    levels = np.linspace(v_min, v_max, 60)
    norm = None
    if v_min < 0:
        ##for plotting the trends we use a diverging color palette that is centered at 0
        ## (I use this check to see if this function is called to plot a trend, but there can be better ways)
        norm = TwoSlopeNorm(vmin=v_min, vcenter=0, vmax=v_max)

    plt.contourf(
        longs,
        lats,
        map_to_plot,
        cmap=cmap,
        vmin=v_min,
        vmax=v_max,
        levels=levels,
        transform=ccrs.PlateCarree(),
        extend="both",
        norm=norm,
    )

    cbar = plt.colorbar()

    if v_min < 0:
        cbar.set_ticks(range(v_min, v_max + 1))
    else:  ## when we plot standard deviation, we end up in this "else" condition
        cbar.set_ticks(np.arange(v_min, v_max + 0.1, 0.1))

    ax.coastlines()
    ax.set_global()
    # plt.savefig(folder_saving + "/" + save_file)
    # plt.close()


# os.makedirs(folder_saving, exist_ok=True)
# # take a sample "nc" file to get the lat/long for plotting
# nc = path_sample_nc_file
#
# dataset = netCDF4.Dataset(nc)
#
# map_to_plot = np.transpose(xr)
# # print(np.min(map), np.max(map), map.shape)
# map_to_plot = np.ma.masked_where(np.isnan(map_to_plot), map_to_plot)
# print(np.min(map_to_plot), np.max(map_to_plot), map_to_plot.shape)
#
# lats = dataset.variables["lat"][:]
# lons = dataset.variables["lon"][:]
#
# if lowresolution:
#     lats = block_reduce(lats, (2,), np.mean)
#     lons = block_reduce(lons, (2,), np.mean)
#
# print(lats.min(), lats.max(), lats.shape)
# print(lons.min(), lons.max(), lons.shape)
#
# map_to_plot, lons = add_cyclic_point(map_to_plot, coord=lons)
#
# ax = plt.axes(
#     projection=ccrs.PlateCarree(central_longitude=210)
# )
# v_min = v_min if v_min is not None else map_to_plot.min()
# v_max = v_max if v_max is not None else map_to_plot.max()
#
# levels = np.linspace(v_min, v_max, 60)
# norm=None
# if v_min<0:
# ##for plotting the trends we use a diverging color palette that is centered at 0
# ## (I use this check to see if this function is called to plot a trend, but there can be better ways)
#     norm = TwoSlopeNorm(vmin=v_min, vcenter=0, vmax=v_max)
#
# plt.contourf(
#     lons,
#     lats,
#     map_to_plot,
#     cmap= cmap,
#     vmin=v_min,
#     vmax=v_max,
#     levels=levels,
#     transform=ccrs.PlateCarree(),
#     extend="both",
#     norm=norm,
# )
#
# cbar = plt.colorbar()
#
# if v_min < 0:
#     cbar.set_ticks(range(v_min, v_max + 1))
# else: ## when we plot standard deviation, we end up in this "else" condition
#     cbar.set_ticks(np.arange(v_min, v_max + 0.1, 0.1))
#
# ax.coastlines()
# ax.set_global()
# plt.savefig(folder_saving + "/" + save_file)
# plt.close()


def plot_cesm2_grid(
    values, tlongs, tlats, title, depth=48, v_min=None, v_max=None
):
    # pop1d = read_nc("/tmp/ssh/pop/map_1x1d_to_gx1v6_bilin_da_100716.nc")
    # tlongs = pop1d.variables["xc_b"][:].reshape(384, 320)
    # tlats = pop1d.variables["yc_b"][:].reshape(384, 320)
    coords = np.concatenate([tlongs[:, :, None], tlats[:, :, None]], axis=-1)
    mesh = Icosphere(depth=depth)
    print("Starting transferring to mesh")
    mesh_values = transfer_to_mesh(
        mesh, coords.reshape(-1, 2), values.reshape(-1)
    )

    uniform_long = np.arange(-360, 360, 2) / 2
    uniform_lat = np.arange(-180, 180, 2) / 2
    uniform_longs, uniform_lats = np.meshgrid(uniform_long, uniform_lat)

    uniform_coords = np.concatenate(
        [
            uniform_longs[:, :, None],
            uniform_lats[:, :, None],
        ],
        axis=-1,
    ).reshape(-1, 2)

    print("Starting transferring from mesh")
    uniform_values = transfer_from_mesh(
        mesh,
        uniform_coords,
        mesh_values.reshape(-1),
    )
    uniform_map = uniform_values.reshape(180, 360)
    cyclic_map, cyclic_long = util.add_cyclic_point(uniform_map, uniform_long)
    ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=210))
    v_min = v_min or cyclic_map.min()
    v_max = v_max or cyclic_map.max()

    levels = np.linspace(v_min, v_max, 60)
    resolution = 110
    plt.title(title)
    plt.contourf(
        cyclic_long,
        uniform_lat,
        cyclic_map,
        levels=levels,
        transform=ccrs.PlateCarree(),
        corner_mask=True,
        cmap="coolwarm",
        antialiased=False,
    )
    ax.add_feature(feature.LAND, facecolor="white", zorder=10)
    ax.coastlines(resolution=f"{resolution}m", linewidth=1)
    ax.gridlines(draw_labels=True, linestyle="--", color="black")
    cbar = plt.colorbar(shrink=0.3)

    plt.show()


def ncl_colors():
    # Define the color hex codes
    colors = [
        "#000000",  # Black
        "#191970",  # MidnightBlue
        "#104E8B",  # DodgerBlue4
        "#1874CD",  # DodgerBlue3
        "#5CACEE",  # SteelBlue2
        "#79CDCD",  # DarkSlateGray3
        "#ADD8E6",  # LightBlue
        "#E0FFFF",  # LightCyan1
        "#FFFFFF",  # White
        "#FFFFFF",  # White (duplicated intentionally)
        "#FFE4B5",  # Moccasin
        "#F4A460",  # SandyBrown
        "#EE7600",  # DarkOrange2
        "#D2691E",  # Chocolate
        "#FF4040",  # Brown1
        "#FF0000",  # Red1
        "#CD0000",  # Red3
        "#8B0000",  # Red4
    ]

    # Create a custom colormap
    custom_cmap = LinearSegmentedColormap.from_list("custom_cmap", colors)
    return custom_cmap


def plot_trends(
    uniform_values,
    lat_linspace,
    long_linspace,
    title,
    figsize=(18, 9),
    cover_land=True,
    unit="cm",
    scatter_nodes=None,
    s=0.2,
):
    # Delegate to the new generic plotter with trend-friendly defaults
    plot_field(
        field=uniform_values,
        lats=lat_linspace,
        lons=long_linspace,
        title=title,
        unit=unit,
        figsize=figsize,
        cover_land=cover_land,
        cmap=ListedColormap(
            [
                "#050CFB",
                "#162AF4",
                "#2D4BEB",
                "#3B6BE4",
                "#2B7CF2",
                "#1A93FF",
                "#0CA8FF",
                "#07BFFD",
                "#4BC3F6",
                "#8CCDEF",
                "#FFFFFF",
                "#FFFFFF",
                "#FFF88E",
                "#FFE646",
                "#FFCF21",
                "#FFB508",
                "#FF9700",
                "#FF7C00",
                "#FF5200",
                "#FF1F00",
                "#F40000",
                "#DA0000",
            ]
        ),
        limits=(-2.2, 2.2),
        step=0.2,
        gap_around_zero=0.2,
        scatter_nodes=scatter_nodes,
        s=s,
        tick_step=0.4,
    )

def plotting(
    ML_predicted_trend_for_past,
    ML_predicted_trend_for_future,
    ML_predicted_trend_for_future_std,
    observations,
    weight_map,
    folder_saving,
    path_sample_nc_file,
    lowresolution=True,
):
    """
    Function that makes the plots for visualization and also estimates other metrics:
        RMS and correlation scores (both weighted)
    """
    plt.clf()

    observations_copy = np.copy(observations)
    observations_copy[np.isnan(ML_predicted_trend_for_past)] = (
        np.nan
    )  ##we are only concerned with the pixels where predictions are made
    print(np.isnan(observations).sum(), np.isnan(observations_copy).sum())

    eval.plot(
        observations_copy
        * 10,  ##convert trend from cm/year to mm/year when plotting
        folder_saving,
        "true_trend_for_1993_2022.png",
        path_sample_nc_file,
        lowresolution,
        v_min=-2,
        v_max=3,
    )

    observations_rms, _ = eval.evaluation_metrics(
        None,
        observations_copy,
        mask=~np.isnan(observations_copy),
        weight_map=weight_map,
        single_map=True,
    )

    print(
        "rms of true trend in mm/yr for 1993-2022: ",
        observations_rms * 10,
    )

    eval.plot(
        ML_predicted_trend_for_past * 10,
        folder_saving,
        "mlp_predicted_trend_for_1993_2022.png",
        path_sample_nc_file,
        lowresolution,
        v_min=-2,
        v_max=3,
    )

    ml_predicted_trend_for_past_rms, _ = eval.evaluation_metrics(
        None,
        ML_predicted_trend_for_past,
        mask=~np.isnan(ML_predicted_trend_for_past),
        weight_map=weight_map,
        single_map=True,
    )

    print(
        "rms of mlp predicted trend in mm/yr for 1993-2022: ",
        ml_predicted_trend_for_past_rms * 10,
    )

    eval.plot(
        ML_predicted_trend_for_future * 10,
        folder_saving,
        "mlp_predicted_trend_2023_2052.png",
        path_sample_nc_file,
        lowresolution,
        v_min=-2,
        v_max=3,
    )

    ml_predicted_trend_for_future_rms, _ = eval.evaluation_metrics(
        None,
        ML_predicted_trend_for_future,
        mask=~np.isnan(ML_predicted_trend_for_future),
        weight_map=weight_map,
        single_map=True,
    )

    print(
        "rms of mlp predicted trend in mm/yr for 2023-2052: ",
        ml_predicted_trend_for_future_rms * 10,
    )

    rmse_between_obs_and_ml_pred, _ = eval.evaluation_metrics(
        None,
        (observations - ML_predicted_trend_for_past),
        mask=~np.isnan(ML_predicted_trend_for_past),
        weight_map=weight_map,
        single_map=True,
    )

    print(
        "rmse of mlp predicted trend in mm/yr for 1993-2022: ",
        rmse_between_obs_and_ml_pred * 10,
    )

    print("correlation on training period")
    eval.get_correlation(observations, ML_predicted_trend_for_past, weight_map)

    eval.plot(
        (observations - ML_predicted_trend_for_past) * 10,
        folder_saving,
        "difference_plot_with_mlp_predicted_trend_1993-2022.png",
        path_sample_nc_file,
        lowresolution,
        v_min=-1,
        v_max=1,
    )

    print("correlation on future period of ml prediction with the persistence")
    eval.get_correlation(
        observations, ML_predicted_trend_for_future, weight_map
    )

    eval.plot(
        (observations - ML_predicted_trend_for_future) * 10,
        folder_saving,
        "difference_plot_wrt_persistence_of_mlp_predicted_trend_2023-2052.png",
        path_sample_nc_file,
        lowresolution,
        v_min=-2,
        v_max=3,
    )

    eval.plot(
        ML_predicted_trend_for_future_std * 10,
        folder_saving,
        "std_of_mlp_predicted_trend_2023_2052.png",
        path_sample_nc_file,
        lowresolution,
        v_min=0,
        v_max=1,
        cmap="YlOrRd",
    )

    ml_pred_trend_for_future_std_rms, _ = eval.evaluation_metrics(
        None,
        ML_predicted_trend_for_future_std,
        mask=~np.isnan(ML_predicted_trend_for_future_std),
        weight_map=weight_map,
        single_map=True,
    )

    print(
        "ml predicted trend for 2023-2052 std RMS in mm/year: ",
        ml_pred_trend_for_future_std_rms * 10,
    )


def plotting_for_leave_one_out(
    observations,
    leave_one_out_future_trend,
    ML_predicted_trend_for_future,
    folder_saving,
    path_sample_nc_file,
    weight_map,
    lowresolution=True,
):
    """
    For leave one out experiment results:
    Function that makes additional plots for visualization and also estimates other
    metrics:
    RMS and correlation scores - both weighted
    (using the future trend for the climate model used as the label in the experiment)
    """

    leave_one_out_future_trend_copy = np.copy(leave_one_out_future_trend)
    leave_one_out_future_trend_copy[np.isnan(ML_predicted_trend_for_future)] = (
        np.nan
    )
    print(
        np.isnan(leave_one_out_future_trend).sum(),
        np.isnan(leave_one_out_future_trend_copy).sum(),
    )

    eval.plot(
        leave_one_out_future_trend_copy * 10,
        folder_saving,
        "true_trend_2023_2052.png",
        path_sample_nc_file,
        lowresolution,
        v_min=-2,
        v_max=3,
    )

    true_future_trend_rms, _ = eval.evaluation_metrics(
        None,
        leave_one_out_future_trend_copy,
        mask=~np.isnan(leave_one_out_future_trend_copy),
        weight_map=weight_map,
        single_map=True,
    )

    print(
        "rms of true trend in mm/yr for 2023-2052: ",
        true_future_trend_rms * 10,
    )

    rmse_between_future_trend_and_ml_pred, _ = eval.evaluation_metrics(
        None,
        (leave_one_out_future_trend - ML_predicted_trend_for_future),
        mask=~np.isnan(ML_predicted_trend_for_future),
        weight_map=weight_map,
        single_map=True,
    )

    print(
        "rmse of mlp predicted trend in mm/yr for 2023-2052: ",
        rmse_between_future_trend_and_ml_pred * 10,
    )
    eval.get_correlation(
        leave_one_out_future_trend, ML_predicted_trend_for_future, weight_map
    )

    eval.plot(
        (leave_one_out_future_trend - ML_predicted_trend_for_future) * 10,
        folder_saving,
        "difference_plot_with_mlp_predicted_trend_2023-2052.png",
        path_sample_nc_file,
        lowresolution,
        v_min=-2,
        v_max=3,
    )

    rmse_between_future_trend_and_persistence, _ = eval.evaluation_metrics(
        None,
        (leave_one_out_future_trend - observations),
        # observations here are the past trend for the climate model choosen as  label
        mask=~np.isnan(ML_predicted_trend_for_future),
        weight_map=weight_map,
        single_map=True,
    )

    print(
        "rmse between the trend for 2023-2052 and persistence (trend for 1993-2022)"
        "in mm/year: ",
        rmse_between_future_trend_and_persistence * 10,
    )
