from pathlib import Path
from typing import Dict, List, Tuple
import xarray as xr
from .bases import strip_cesm2_nc_fname
from .bases import strip_organized_nc_fname


def sort_members(dataset_path, dry_run=False):
    """
    Sort CESM2 NetCDF files into organized directory structure by simulation member.

    Given files named like "cesm2_simulation_var_startmonth_endmonth.nc",
    sorts them into folders named by simulation member: sim/startmonth_endmonth.nc

    Parameters:
    - dataset_path (str): Path to the directory containing the files to be sorted.
    - dry_run (bool): If True, only prints actions without actually moving files (default is False).

    Returns:
    - None
    """
    dataset_path = Path(dataset_path)
    nc_files = list(dataset_path.glob("*.nc"))

    # Check if any NetCDF files exist
    if len(nc_files) == 0:
        raise FileNotFoundError(f"No *.nc files found in {dataset_path}")

    for nc_f in nc_files:
        # Parse filename to extract simulation info
        nc_name = strip_cesm2_nc_fname(nc_f.name)

        # Create simulation directory if it doesn't exist
        sim_path = dataset_path.joinpath(nc_name.sim)
        sim_path.mkdir(parents=True, exist_ok=True)

        # Create new filename with time range format
        new_nc_f = sim_path.joinpath(f"{nc_name.start}-{nc_name.end}.nc")

        if dry_run:
            print(f"Moving {nc_f.name} to {new_nc_f}")
        else:
            nc_f.rename(new_nc_f)


def merge_monthly_member(
    member_path: Path | str,
    variable: str,
    save=False,
    target_path=None,
    chunk_size: Tuple[int, ...] | None = None,
    keep_dims: Tuple[str, ...] | None = None,
):
    """
    Merge multiple monthly NetCDF files into a single file concatenated along the time dimension.

    This function takes a directory containing multiple monthly NetCDF files (e.g., "2000-01.nc", "2000-02.nc")
    and merges them into a single NetCDF file with the same name as the directory.

    Parameters:
    - member_path (Path | str): Path to the directory containing monthly NetCDF files to merge.
    - variable (str): Variable name to extract from each NetCDF file.
    - remove_folder (bool): If True, removes the input directory after successful merging (default is False).

    Returns:
    - None: Saves the merged dataset to a .nc file in the parent directory.
    """
    member_path = Path(member_path)

    # Collect all NetCDF files and extract their time information
    nc_names = []
    for nc_f in list(member_path.glob("*.nc")):
        # Extract time period info from filename and pair with file path
        time_info = strip_organized_nc_fname(nc_f.name)
        nc_names.append((time_info, nc_f))

    # Sort files by time period to ensure correct temporal ordering
    nc_names = list(sorted(nc_names, key=lambda x: x[0]))

    # Load and extract the specified variable from each file
    datasets = []
    for time_info, nc_file in nc_names:
        datasets.append(
            xr.open_dataset(nc_file, decode_timedelta=True)[variable]
        )

    ds = xr.concat(datasets, dim="time")
    if keep_dims is not None:
        drop_dims = set(ds.dims) - set(keep_dims)
        ds = ds.squeeze(drop_dims, drop=True)

    if save:
        # Create output path: same name as input directory but with .nc extension
        if target_path is None:
            target_path = member_path.with_suffix(".nc")
        else:
            target_path = Path(target_path)

        print(f"Saving concatenated data to {target_path}", end=", ")
        if chunk_size is not None:
            print("with chunking", end=", ")
            ds.to_netcdf(
                target_path,
                engine="netcdf4",
                encoding={
                    variable: {
                        "dtype": "float32",
                        "complevel": 1,
                        "zlib": True,
                        "chunksizes": chunk_size,
                    }
                },
            )
        else:
            ds.to_netcdf(
                target_path,
            )
        print("Done!")

    return ds


def merge_and_save_datasets(
    variable_path, variable, target_path, chunk_size, keep_dims
):
    """
    Merge datasets for a specific variable from multiple simulation member sub-folders.

    This function iterates through all subdirectories in the given variable path
    and merges monthly data for each simulation member into a single file.

    Parameters:
    - variable_path (str): The path to the directory containing simulation member sub-folders with datasets.
    - variable (str): The specific variable name to merge datasets for.
    - remove_folder (bool): Flag to indicate whether to remove the sub-folders after merging datasets (default is False).

    Returns:
    - None
    """
    target_path = Path(target_path)
    target_path.mkdir(parents=True, exist_ok=True)
    # Find all subdirectories (simulation members) in the variable path
    member_paths = list(
        filter(lambda x: x.is_dir(), Path(variable_path).glob("*"))
    )

    # Check if any subdirectories exist
    if len(member_paths) == 0:
        raise FileNotFoundError(f"{variable_path} contains no sub-folders")

    # Process each simulation member directory
    for member in member_paths:
        merge_monthly_member(
            member,
            variable,
            save=True,
            target_path=target_path.joinpath(f"{member.name}.nc"),
            chunk_size=chunk_size,
            keep_dims=keep_dims,
        )


# import h5py
# import pickle
# import os
# from .reader import load_cm_generator
#
#
# def add_to_hdf(data, hdf, name):
#     if isinstance(data, dict):
#         for k in data:
#             add_to_hdf(data[k], hdf, f"{k}/{name}")
#     else:
#         hdf.create_dataset(
#             name,
#             data=data,
#         )
#
#
# def save_to_hdf(source_dir, target_path, variable="SSH", data_only=False):
#     dataset = h5py.File(target_path, "a")
#     # if variable not in dataset:
#     # dataset.create_group(variable)
#     latlong_added = False
#     # nc_file = nc_files[0]
#     for month, simulation, data in load_cm_generator(
#         source_dir=source_dir, variable=variable
#     ):
#         # print(data.data, data.data.dtype, data.data.shape)
#         # break
#         if data_only:
#             data = data.data
#         # for i in range(nc.variables["time"].shape[0]):
#         # if (year, month) not in dataset:
#         # dataset[(year, month)] = {}
#         # dataset[(year, month)][simulation] = nc.variables[variable][i]
#         data_path = f"{variable}/{month}/{simulation}"
#         # if data_path not in dataset:
#         dataset.create_dataset(data_path, data=data, compression="gzip")
#     dataset.close()
#     return target_path
#
#
# def save_dictionary(data, target_path, variable):
#     dataset = h5py.File(target_path, "a")
#     add_to_hdf(data, dataset, variable)
#
#
# def picklize(obj, target_path, replace=False):
#     if os.path.exists(target_path) and not replace:
#         raise AssertionError(f"{target_path} already exists")
#     with open(target_path, "wb") as handle:
#         pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)
#     return True
