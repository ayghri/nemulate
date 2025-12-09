from nemulate import data as ndata
from pathlib import Path
from nemulate.visualize.plots import plot_frames
from nemulate.data.regrid import CESM2Regrid

sim_path = Path("/buckets/datasets/ssh/simulations/cesm2/")
merged_path = sim_path / "merged/monthly"

ssh = ndata.get_cesm2_member_months(merged_path, "SSH", "1281.014")

regridder = CESM2Regrid(sim_path / "grid_info.nc")
regrided_field = regridder.regrid(ssh["SSH"], 1)
plot_frames(
    regrided_field.values,
    regrided_field.lat.values[:, 0],
    regrided_field.lon.values[0, :],
    name_format="dist/frame_{:04d}.png",
    vmin=-250.5,
    vmax=164.5
)
