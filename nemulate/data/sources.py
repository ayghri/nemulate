from numpy.typing import NDArray
from typing import Dict, List, Tuple
from pathlib import Path
import re
from tqdm import tqdm
import numpy as np
import xarray as xr

from .bases import MonthStamp
from .bases import read_netcdf
from .bases import offset_netcdf_time


def get_common_members(variable_paths: List[str] | List[Path]):
    """
    Identify simulation members that are common across all specified variable directories.
    Given multiple variable directories with structure var_i/sim_j/startmonth_endmonth.nc,
    identifies simulation members ("sim_j") that are present in ALL variable directories.

    Parameters:
    - variable_dirs: List of paths to variable directories containing simulation member subdirectories.

    Returns:
    - common_members: List of simulation members that exist in all variable directories.
    """

    # Get all subdirectories (simulation members) for each variable
    per_var_members_path = [
        [g for g in Path(path).glob("*")] for path in variable_paths
    ]

    per_var_members = [[p.name for p in ds] for ds in per_var_members_path]

    # Find members that exist in all variables (intersection)
    all_members = set([p.name for p in per_var_members_path[0]])
    # Compute intersection across all variable directories
    common_members = set([p.name for p in per_var_members_path[0]])
    for i in range(1, len(variable_paths)):
        member_names = [p.name for p in per_var_members_path[i]]
        common_members = common_members.intersection(member_names)
        all_members = all_members.union(member_names)

    return list(all_members), list(common_members), list(per_var_members)


def get_cesm2_members_ids(
    cesm2_path: str | Path, var_name: str | List[str]
) -> List:
    """
    Retrieve sorted list of CESM2 member names for a specified variable.

    Parameters:
    cesm2_path (str): Path to the CESM2 data: cesm2_path/var_name/member[i].nc
    var_name (str): Name of the variable(s) to filter the files.

    Returns:
    list: A sorted member file names (without extensions) for the specified variable.
    """
    cesm2_path = Path(cesm2_path)
    if isinstance(var_name, str):
        var_name = [var_name]
    per_var_members = []
    for v in var_name:
        per_var_members.append([m.stem for m in cesm2_path.glob(f"{v}/*")])

    common_members = set(per_var_members[0])
    for i in range(1, len(per_var_members)):
        common_members = common_members.intersection(set(per_var_members[i]))

    return list(sorted(common_members))


def get_cesm2_member_months(
    cesm2_path,
    var_name,
    member_name,
    start_year: int = 1850,
    start_month: int = 1,
    end_year: int = 2100,
    end_month: int = 12,
    time_chunk_size: int | None = 240,
) -> xr.Dataset:
    """
    Load a CESM2 member dataset from a NetCDF file.

    Parameters:
    cesm2_path (str or Path): The path to the CESM2 data directory.
    var_name (str): The name of the variable to load.
    member_name (str): The name of the member to load.
    start_year (int, optional): The starting year for the data (default is 1850).
    start_month (int, optional): The starting month for the data (default is 1).
    chunk_size (int or None, optional): The size of the chunks for loading data.

    Returns:
    xarray.Dataset: The loaded dataset filtered by the specified start time.
    """
    start_time = MonthStamp(year=start_year, month=start_month)
    end_time = MonthStamp(year=end_year, month=end_month)
    dataset = read_netcdf(
        Path(cesm2_path).joinpath(f"{var_name}/{member_name}.nc"),
        time_chunk_size=time_chunk_size,
    )
    # Adjust time coordinate if it starts in February
    # This seems  to be a CESM2 bug
    if dataset.coords["time"].values[0].month == 2:
        dataset = offset_netcdf_time(dataset)

    dataset = dataset.where(
        (dataset.time >= start_time.cftime) & (dataset.time <= end_time.cftime),
        drop=True,
    )
    return dataset


def get_cesm2_monthly(
    cesm2_monthly_path,
    var_name,
    start_year: int = 1850,
    start_month: int = 1,
    chunk_size=None,
) -> Dict[str, xr.Dataset]:
    members = get_cesm2_members_ids(cesm2_monthly_path, var_name)
    dataset = {}
    if chunk_size is not None:
        print(f"Loading chunks of {chunk_size} months for CESM2:{var_name}")
    else:
        print(f"All data for CESM2:{var_name}")
    for memb_name in tqdm(members):
        dataset[memb_name] = get_cesm2_member_months(
            cesm2_monthly_path,
            var_name=var_name,
            member_name=memb_name,
            start_year=start_year,
            start_month=start_month,
            time_chunk_size=chunk_size,
        )
    return dataset


def generate_monthly_cesm2_members(
    cesm2_path, var_name, start_year=1850, start_month=1
):
    members = get_cesm2_members_ids(cesm2_path, var_name)
    for memb in tqdm(members):
        yield (
            memb,
            get_cesm2_member_months(
                cesm2_path,
                var_name,
                member_name=memb,
                start_year=start_year,
                start_month=start_month,
            ),
        )


def read_hadi_line(line):
    """
    This function reads a line of values and extracts integers from it.

    Args:
    line (str): A string containing values.

    Returns:
    list: A list of integers extracted from the input line.
    """
    line_values = re.findall(r"-?\d+", line)
    assert len(line_values) == 360
    line_values_int = [int(v) for v in line_values]
    return line_values_int


def read_year_lines(lines, year=None):
    """
    Read lines of data for a specific year.

    Parameters:
    lines (list): List of lines containing data.
    year (int): The year to filter the data for.

    Returns:
    tuple: A tuple containing the formatted year and an array of values.
    """
    text_year = lines[0].split()[2]
    text_month = lines[0].split()[1]
    if year is not None:
        assert int(text_year) == year
    values = []
    for line in range(1, len(lines)):
        # print(lines[l])
        # values.append([int(i) for i in lines[l].split()])
        values.append(read_hadi_line(lines[line]))
    return "{}{:02d}".format(text_year, int(text_month)), np.array(values)


def read_hadi_month(lines: List[str]) -> Tuple[int, int, NDArray]:
    """
    Reads a raw month HADI data from a list of strings.

    Parameters:
    lines (List[str]):
    A list of strings representing the lines of the raw month HADI data.
    """

    # assert len(lines) == 181
    header = lines[0].lower()
    assert "rows" in header and "columns" in header.lower()
    month, year = [int(e) for e in re.findall(r"\d+", header)[1:3]]
    values = []
    for line_idx in range(1, len(lines)):
        values.append(read_hadi_line(lines[line_idx]))
    assert len(values) == 180
    return year, month, np.array(values, dtype=np.int16).T


def get_hadi(path) -> Dict[Tuple[int], NDArray]:
    """
    Read Hadi data from text files in the specified path and return a date-sorted dictionary.
    Args:
        path (str): The path to the directory containing the text files.
    Returns:
        dict: dictionary, keys are tuples of year and month, and values are data.
    """
    dataset = {}
    for data_path in tqdm(list(Path(path).glob("*.txt"))):
        with open(data_path, "r") as f:
            lines = f.readlines()
            for i in range(0, len(lines), 181):
                year, month, data = read_hadi_month(lines[i : i + 181])
                dataset[(year, month)] = data
    return dict(sorted(dataset.items(), key=lambda x: x[0]))


def load_altimetry(altimetry_path, var_name):
    file_name = {
        "wind": "tauxy.nc",
        "twosat": "twosat_sla.nc",
        "hadi": "hadi_sst.nc",
        "aviso": "aviso.msla_dated.nc",
    }[var_name]
    return xr.open_dataset(Path(altimetry_path).joinpath(file_name))


# def delete_uncommon_members(
#     variable_dirs: List[str] | List[Path],
#     dry_run: bool = True,
#     verbose: bool = False,
# ) -> Dict[str, List[str]]:
#     """
#     Find and optionally delete simulation members that don't exist across all variable directories.

#     Given multiple variable directories with structure var_i/sim_j/startmonth_endmonth.nc,
#     identifies simulation members ("sim_j") that are not present in ALL variable directories.

#     Parameters:
#     - variable_dirs: List of paths to variable directories containing simulation member subdirectories.
#     - dry_run: If True, only identifies uncommon members without deleting them (default is True).

#     Returns:
#     - uncommon_per_variable: Dictionary mapping variable path strings to lists of their uncommon members.
#     """

#     all_members, common_members, per_var_members = get_common_members(
#         variable_dirs
#     )

#     variable_paths = [Path(d) for d in variable_dirs]

#     # Uncommon members are those that exist in some but not all variables
#     uncommon_members = list(all_members.difference(common_members))
#     if verbose:
#         print("Uncommon:", uncommon_members, "total: ", len(uncommon_members))

#     # Build dictionary for uncommon members from each variable
#     uncommon_per_var = {"all": uncommon_members}
#     for i, var_path in enumerate(variable_paths):
#         if verbose:
#             print(var_path.name, end=": ")
#         uncommon_per_var[str(var_path)] = []
#         for um in uncommon_members:
#             # If this uncommon member is NOT in this variable's members, it's missing
#             if um not in per_var_members[i]:
#                 if verbose:
#                     print(um, end=", ")
#                 uncommon_per_var[var_path.__str__()].append(um)
#         if verbose:
#             print()

#     # Actually delete the uncommon members if not in dry run mode
#     if not dry_run:
#         for var_path in uncommon_per_var:
#             for uncommon in uncommon_per_var[var_path]:
#                 delete_dir(Path(var_path).joinpath(uncommon))

#     return uncommon_per_var
