from typing import Dict
import xesmf as xe
import xarray as xr
from .bases import read_netcdf


def regrid_cesm2_data(xarr, var_name):
    if "TLONG" in xarr.coords:
        xarr = xarr.rename({"TLONG": "lon", "TLAT": "lat"})
    ds_out = xe.util.grid_global(1, 1)
    regridder = xe.Regridder(xarr, ds_out, "bilinear", periodic=True)
    data_regridded = regridder(xarr[var_name])
    return data_regridded


class CESM2Regrid:
    def __init__(self, grid: xr.Dataset):
        self.grid = grid
        # regridding weights are cached here for each degree
        self.regridder: Dict[float, xe.Regridder] = {}

    @classmethod
    def from_netcdf(cls, grid_nc_path):
        native_grid = read_netcdf(grid_nc_path)
        assert "TLAT" in native_grid.variables
        assert "TLONG" in native_grid.variables
        native_grid["lat"] = native_grid["TLAT"]
        native_grid["lon"] = native_grid["TLONG"]
        return cls(native_grid)

    def regrid(self, xarr, degree: float = 1.0):
        # create regridder if not already cached
        if degree not in self.regridder:
            self.regridder[degree] = xe.Regridder(
                self.grid,
                xe.util.grid_global(degree, degree),
                "bilinear",
                periodic=True,
            )

        xarr["lat"] = self.grid["lat"]
        xarr["lon"] = self.grid["lon"]

        return self.regridder[degree](xarr, keep_attrs=True)


# The regridded data is now in an xarray.DataArray with a uniform 1x1 degree grid.


# 1. Load your CESM2 dataset
# Replace 'your_cesm2_data.nc' with the path to your file.
# You might need to identify the correct variable names for data, lat, and lon.


# 2. Create a target 1x1 degree uniform grid


# 3. Create the regridder object
# The input dataset needs to have 'lat' and 'lon' as 2D coordinates.
# If your dataset has 1D coordinates, this will also work.


# 4. Apply the regridding to your data variable

# print("Regridding complete. Here is the regridded data:")
# print(data_regridded)

# # You can now work with your data_regridded object, for example, plot it.
# import matplotlib.pyplot as plt

# plt.figure(figsize=(12, 6))
# data_regridded.plot()
# plt.title('Regridded CESM2 Data (1x1 degree)')
# plt.show()
