from nemulate import data as ndata
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


sim_path = Path("/buckets/datasets/ssh/simulations/cesm2/")
merged_path = sim_path / "merged/monthly"

var_names = [p.stem for p in merged_path.glob("*")]
print(f"Found {len(var_names)} variables: {var_names}")

ssh = ndata.get_cesm2_member_months(merged_path, "SSH", "1281.014")

from nemulate.data.regrid import CESM2Regrid

regridder = CESM2Regrid(sim_path / "grid_info.nc")

regrided_field = regridder.regrid(ssh["SSH"], 1)
