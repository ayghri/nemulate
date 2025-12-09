# from netCDF3 import Dataset  # pyright: ignore
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from seasurfaceh.datasets.reader import MonthStamp
from seasurfaceh.datasets.reader import load_cm_generator
import numpy as np

# from pathlib import Path
import glob

HADI_ICE_MASK = -1000
HADI_LAND_MASK = -32768


def plot_nc(nc, var):
    plt.imshow(nc.variables[var][0])
    plt.show()


# %%


def mask_hadi(raw_dataset):
    pass


def change_resolution(hdf_path, target=1.0):
    lat = None
    long = None


# if __name__ == "__main__":
# combine_cesm2("./data/simulations/cesm2/")


class CMLoader:
    def __init__(self, path, field):
        self.path = path
        self.field = field
        self.nc_files = glob.glob(path + f"/{field}/*.nc")
        assert len(self.nc_files) > 0


def get_yearly(source_dir, variable):
    current_year = {}
    year = {}
    yearly_data = {}
    for month, simulation, data in load_cm_generator(source_dir, variable):
        m = MonthStamp.from_str(month)
        if m.year != current_year:
            current_year = m.year
            if current_year != -1:
                if len(year) != 12:
                    print(str(m), simulation)
                    break
            # yearly_data.append((m.year, simulation, np.stack(year)))
        # year.append(data)


# variable = "SHF"
# source_dir = f"/datasets/ssh/simulations/cesm2/{variable}/"
# target_path = f"/datasets/ssh/simulations/cesm2/{variable}.h5"
# get_yearly(source_dir, variable)
