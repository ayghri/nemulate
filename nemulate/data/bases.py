from functools import total_ordering
from pathlib import Path
import xarray as xr
import cftime
import re
import shutil


def delete_dir(path):
    """Remove a directory and all its contents."""
    shutil.rmtree(path)


def read_netcdf(netcdf_path: Path | str, time_chunk_size=None) -> xr.Dataset:
    """
    Opens a NetCDF file and optionally chunks the dataset by time.

    Parameters:
    netcdf_path (Path): The file path to the NetCDF dataset.
    time_chunk_size (int, optional): The size of the chunks for the 'time' dimension.
                                 If None, the dataset is not chunked.

    Returns:
    xr.Dataset: The opened (and possibly chunked) NetCDF dataset.
    """

    if time_chunk_size is not None:
        dataset = xr.open_dataset(netcdf_path, chunks={"time": time_chunk_size})
    else:
        dataset = xr.open_dataset(netcdf_path)

    return dataset


def offset_netcdf_time(
    ds: xr.Dataset, offset_value: int = -1, offset_type="MS"
):
    time_idx = ds.get_index("time").shift(offset_value, offset_type).values  # type: ignore
    ds.coords["time"] = time_idx
    return ds


def compute_yearly_mean(dataset: xr.Dataset):
    year_groups = dataset.groupby("time.year").mean(skipna=False)
    # twelves = xr.where(year_groups.isnull(), 12.0, 12.0)
    # year_mean = year_groups / twelves
    return year_groups


@total_ordering
class MonthStamp:
    """
    Represents a specific month and year.

    Parameters:
    - year (int): The year of the month stamp.
    - month (int): The month of the month stamp.

    Methods:
    - increment(times=1): Increments the month stamp by the specified number of times.
    - cftime: Returns a cftime.DatetimeNoLeap object representing the month stamp.
    - from_str(month_str): Creates a MonthStamp object from a string in the format 'YYYYMM'.
    - __eq__(other): Compares if two MonthStamp objects are equal.
    - __lt__(other): Compares if one MonthStamp object is less than another.

    Returns:
    - str: A string representation of the MonthStamp object.
    """

    def __init__(self, year: int, month: int):
        self.year = year
        self.month = month

    def increment(self, times=1):
        for _ in range(times):
            self.year = self.year + self.month // 12
            self.month = (self.month) % 12 + 1
        return self

    @property
    def cftime(self):
        return cftime.DatetimeNoLeap(year=self.year, month=self.month, day=1)

    @property
    def pdtime(self):
        return f"{self.year}-{self.month}-01"

    def __str__(self):
        return f"{self.year:04d}{self.month:02d}"

    def __repr__(self) -> str:
        return str(self)

    @classmethod
    def from_str(cls, month_str):
        return cls(year=int(month_str[:4]), month=int(month_str[4:]))

    def __eq__(self, other):
        return str(self) == str(other)

    def __lt__(self, other):
        return str(self) < str(other)


@total_ordering
class NCName:
    def __init__(
        self, sim: str, variable: str, start_month: int, end_month: int
    ) -> None:
        """
        NCName representation with simulation name, variable name, start month, and end month.

        Parameters:
        sim (str): The simulation name.
        variable (str): The variable name.
        start_month (int): The start month.
        end_month (int): The end month.
        """
        self.sim = sim
        self.var = variable
        self.start = start_month
        self.end = end_month

    def __eq__(self, other):
        return self.start == other.start

    def __lt__(self, other):
        return self.start < other.start

    def __repr__(self) -> str:
        return f"{self.sim}:{self.var}:{self.start}-{self.end}"


def strip_cesm2_nc_fname(nc_fname) -> NCName:
    result = re.match(
        r".*_g17.LE2.(\d{4}.\d{3})\..*\.([A-Z]*)\.(\d{6})-(\d{6})\.nc", nc_fname
    )
    assert result is not None
    assert result.lastindex == 4
    return NCName(
        sim=result.group(1),
        variable=result.group(2),
        start_month=int(result.group(3)),
        end_month=int(result.group(4)),
    )


def strip_organized_nc_fname(nc_fname) -> NCName:
    result = re.match(r"(\d{6})-(\d{6}).nc", nc_fname)
    assert result is not None
    assert result.lastindex == 2
    return NCName(
        sim="",
        variable="",
        start_month=int(result.group(1)),
        end_month=int(result.group(2)),
    )
