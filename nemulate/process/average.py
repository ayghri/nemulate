from typing import Dict, Tuple, Generator
import xarray as xr
# import numpy as np
# from tqdm import tqdm

# from nemulate.data.organize import
# from nemulate.data.bases import MonthStamp
# from nemulate.data.sources import generate_monthly_cesm2_members


# def get_yearly(file, variable):
#     year_data = []
#     for month, simulation, data in generate_monthly_cesm2_members(
#         file, variable
#     ):
#         # print(str(month))
#         m = MonthStamp.from_str(month)
#         year_data.append(data[:].data.reshape((1,) + data.data.shape))
#         if len(year_data) == 12:
#             yield (m.year, simulation, np.concatenate(year_data).mean(0))
#             year_data = []


# def month_to_year_generator(source_dir, variable):
#     files = organize_files(source_dir)
#     for simulation, sim_files in tqdm(files.items()):
#         for f, month_s, month_e in sim_files:
#             for year, simulation, data in get_yearly(f, variable):
#                 # print(data.shape)
#                 yield {simulation: {year: data}}


# def month_to_year(source_dir, variable):
#     files = organize_files(source_dir)
#     dataset = {}
#     for simulation, sim_files in tqdm(files.items()):
#         if simulation not in dataset:
#             dataset[simulation] = {}
#         for f, month_s, month_e in sim_files:
#             for year, simulation, data in get_yearly(f, variable):
#                 # print(data.shape)
#                 dataset[simulation][year] = data
#     return dataset


def compute_members_average(
    members_generator: Generator,
    num_members: int,
    degrees: Tuple[int, ...] = (1,),
    verbose=False,
) -> Dict[int, xr.Dataset]:
    averages: Dict[int, xr.Dataset] = {d: None for d in degrees}  # type: ignore
    # average = None
    count = 1
    if verbose:
        print()

    for member_sim in members_generator:
        for d in degrees:
            if averages[d] is None:
                averages[d] = xr.ufuncs.power(member_sim, d)
            else:
                averages[d] += xr.ufuncs.power(member_sim, d)
        if verbose:
            print(f"Processed 1/{num_members}", end="\r")

        count = count + 1
    for d in degrees:
        averages[d] = averages[d] / num_members
    return averages
