from typing import Tuple, MutableSequence, List, Dict
from pathlib import Path
from collections import deque
import numpy as np
import torch

# import xesmf as xe
import xarray as xr

from sphedron import transform as sph_transform

from .regrid import CESM2Regrid
from .bases import read_netcdf


class Lambda(torch.nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, ins):
        return self.func(ins)


class SubstractForcedResponse(torch.nn.Module):
    def __init__(
        self,
        forced_response_path: Path,
        name_format: str,
        var_names: List[str],
    ):
        super().__init__()
        responses = []
        for v_name in var_names:
            responses.append(
                read_netcdf(
                    forced_response_path / name_format.format(var=v_name),
            ).astype("float32").load()
            )

        forced_response = xr.merge(responses, compat="no_conflicts")

        if "z_t" in forced_response.coords:
            forced_response = forced_response.isel(z_t=0)

        self.forced_response = forced_response.drop_vars(
            ["z_t", "ULONG", "ULAT"]
        )

    def forward(self, ins):
        return ins - self.forced_response.sel(time=ins.time)


class RegridTransform(torch.nn.Module):
    def __init__(self, model, degree, grid_nc_path):
        if model.lower() == "cesm2":
            self.regridder = CESM2Regrid.from_netcdf(grid_nc_path)
        else:
            raise NotImplementedError
        self.degree = degree

    def forward(self, inputs: xr.DataArray):
        return self.regridder.regrid(inputs, degree=self.degree)


class FieldScaler(torch.nn.Module):
    def __init__(self, stats_path: Path, field_format, var_names):
        super().__init__()

        stats = read_netcdf(stats_path)

        renames = {
            field_format.format(var=v_name): v_name for v_name in var_names
        }

        if "time" in stats.coords:
            stats = stats.mean("time")

        self.scalers = stats.rename_vars(renames)
        for v_name in var_names:
            if v_name not in self.scalers.variables:
                raise ValueError(
                    f"{v_name} not part of the the FieldScaler stats"
                )

    def forward(self, ins):
        return ins / self.scalers

        # for var_name in self.scalers.variables:
        #     if var_name in ins:
        #         ins[var_name] /= self.scalers[var_name]


class AddNanMask(torch.nn.Module):
    def __init__(self, var_names):
        super().__init__()
        self.var_names = var_names

    def forward(self, ins):
        mask = xr.concat([ins[v].isnull() for v in self.var_names], dim="var")
        ins["land_mask"] = mask
        return ins


class UnwrapFields(torch.nn.Module):
    def __init__(self, vars_combos: Dict[str, str | List[str]]):
        super().__init__()
        self.combos = vars_combos

    def forward(self, ins):
        fields = {}
        for name, combo in self.combos.items():
            if isinstance(combo, str):
                values = torch.tensor(ins[combo].values)
            else:
                assert isinstance(combo, list)
                values = torch.from_numpy(
                    ins[combo].to_array().fillna(0.0).values
                )
            if values.dtype == torch.float64:
                values = values.float()
            fields[name] = values
        return fields


class AddXYZ(torch.nn.Module):
    @staticmethod
    def latlong_to_xyz_ufunc(lat, lon):
        shape = lat.shape
        latlon = np.c_[lat.reshape(-1), lon.reshape(-1)]
        xyz = sph_transform.latlong_to_xyz(latlon)
        xyz = xyz.reshape(shape + (3,))

        return xyz[..., 0], xyz[..., 1], xyz[..., 2]

    def forward(self, ins):
        lat = ins["latitude"]
        lon = ins["longitude"]
        x, y, z = xr.apply_ufunc(
            self.latlong_to_xyz_ufunc,
            lat,
            lon,
            input_core_dims=[[], []],
            output_core_dims=[[], [], []],
            output_dtypes=[lat.dtype, lat.dtype, lat.dtype],
        )

        ins["cart_x"] = x
        ins["cart_y"] = y
        ins["cart_z"] = z
        return ins


class AddYearMonth(torch.nn.Module):
    @staticmethod
    def get_month_year(time):
        return time.dt.year.values, time.dt.month.values

    def forward(self, ins):
        year, month = self.get_month_year(ins.time)
        ins = ins.assign(year=(("time",), year))
        ins = ins.assign(month=(("time",), month))
        return ins


class AddLatLong(torch.nn.Module):
    def __init__(
        self,
        # native_grid: xr.Dataset,
        grid_path: Path,
        lat_name: str = "TLAT",
        lon_name: str = "TLONG",
    ):
        super().__init__()

        self.lat_name = lat_name
        self.lon_name = lon_name
        self.native_grid = read_netcdf(grid_path)

    def forward(self, inputs):
        inputs["latitude"] = self.native_grid[self.lat_name].copy()
        inputs["longitude"] = self.native_grid[self.lon_name].copy()
        return inputs


class RandomRotatedRegrid(torch.nn.Module):
    def __init__(
        self,
        # native_grid: xr.Dataset,
        grid_path: Path,
        target_degree: float = 1.0,
        queue_length: int = 10,
        refresh_period: int = 5,
        rotation_lows_deg: Tuple[float, ...] = (-30, -60),
        rotation_highs_deg: Tuple[float, ...] = (30, 60),
        p=1.0,
        rotation_axis: str = "xy",
        random_regrid=True,
        lat_name: str = "TLAT",
        lon_name: str = "TLONG",
    ):
        super().__init__()
        assert len(rotation_lows_deg) == len(rotation_highs_deg)
        assert len(rotation_lows_deg) == len(rotation_axis)

        # self.native_grid = native_grid
        grid = read_netcdf(grid_path)
        self.p = p

        grid["lat"] = grid[lat_name]
        grid["lon"] = grid[lon_name]
        self.native_grid = grid
        self.degree = target_degree
        self.rot_lows = np.deg2rad(rotation_lows_deg)
        self.rot_highs = np.deg2rad(rotation_highs_deg)
        self.rot_axis = rotation_axis

        self.regridder_queue: MutableSequence[CESM2Regrid] = deque(
            [], maxlen=queue_length
        )
        self._regrider_angles: MutableSequence[List] = deque(
            [], maxlen=queue_length
        )

        self.refresh_period = refresh_period
        self.step = 0
        self.random_regrid = random_regrid
        self.lat_name = lat_name
        self.lon_name = lon_name

        native_latlon = (
            np.stack(
                [
                    self.native_grid[self.lat_name],
                    self.native_grid[self.lon_name],
                ]
            )
            .transpose(1, 2, 0)
            .reshape(-1, 2)
        )
        self.xyz = sph_transform.latlong_to_xyz(native_latlon)

        self.normal_grid = CESM2Regrid(self.native_grid.copy())

    def sample_regridder(self):
        sampled_angles = list(np.random.uniform(self.rot_lows, self.rot_highs))
        rotated_latlong = sph_transform.xyz_to_latlong(
            sph_transform.rotate_nodes(
                self.xyz, axis=self.rot_axis, angles=sampled_angles
            )
        )

        grid = self.native_grid.copy()

        rotated_latlong = rotated_latlong.reshape(
            self.native_grid[self.lat_name].shape + (2,)
        )

        grid["lat"].values = rotated_latlong[:, :, 0].copy()
        grid["lon"].values = rotated_latlong[:, :, 1].copy()

        self.regridder_queue.append(CESM2Regrid(grid))
        self._regrider_angles.append(sampled_angles)

    def forward(self, inputs: xr.Dataset):
        # inputs["latitude"] = self.native_grid[self.lat_name].copy()
        # inputs["longitude"] = self.native_grid[self.lon_name].copy()

        if self.step % self.refresh_period == 0:
            self.sample_regridder()
        self.step += 1
        if np.random.uniform(0, 1) <= self.p:
            regridder = self.regridder_queue[
                self.step % len(self.regridder_queue)
            ]
        else:
            regridder = self.normal_grid

        return regridder.regrid(inputs, degree=self.degree)
