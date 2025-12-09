from typing import Any, List, Mapping, Optional, Sequence, Union
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import xarray as xr


from torch.utils.data import Dataset

from .data.sources import get_cesm2_member_months


ArrayLike = Union[xr.DataArray, np.ndarray]


def _ensure_time_lat_lon(
    da: xr.DataArray,
    time_name: str = "time",
    lat_name: str = "lat",
    lon_name: str = "lon",
) -> xr.DataArray:
    """
    Ensure DataArray has dims ordered as (time, lat, lon).
    If lat/lon dims are not present, but the array is 3D with one time dim,
    infer the two spatial dims and rename them to (lat_name, lon_name).
    """
    dims = list(da.dims)
    if time_name not in dims:
        raise ValueError(
            f"DataArray must have time dim '{time_name}', got {da.dims}"
        )

    # Identify spatial dims; reduce extras if present
    spatial_dims = [d for d in dims if d != time_name]
    if len(spatial_dims) != 2:
        # Try to reduce to two spatial dims by squeezing or selecting known extra dims
        known_extra = ("level", "lev", "z", "z_t", "depth", "height")
        da2 = da
        # loop with a safety cap
        for _ in range(8):
            dims2 = list(da2.dims)
            spatial_dims2 = [d for d in dims2 if d != time_name]
            if len(spatial_dims2) <= 2:
                break
            # Prefer squeezing singleton dims first
            squeezed = False
            for d in list(spatial_dims2):
                if da2.sizes.get(d, None) == 1:
                    da2 = da2.isel({d: 0}, drop=True)
                    squeezed = True
            if squeezed:
                continue
            # Then select first index along known extra dims
            picked = False
            for d in spatial_dims2:
                if d in known_extra:
                    da2 = da2.isel({d: 0}, drop=True)
                    picked = True
                    break
            if picked:
                continue
            # If still more than two spatial dims, cannot infer safely
            break
        # Recompute dims after attempts
        dims = list(da2.dims)
        spatial_dims = [d for d in dims if d != time_name]
        if len(spatial_dims) != 2:
            raise ValueError(
                f"Expected exactly two spatial dims besides '{time_name}', got {spatial_dims} from {da2.dims}. "
                "Provide 2D data or pre-select/squeeze extra dims (e.g., isel)."
            )
        da = da2

    lat_dim, lon_dim = spatial_dims
    # If dims are already named as requested, just transpose
    if {lat_dim, lon_dim} == {lat_name, lon_name}:
        return da.transpose(time_name, lat_name, lon_name)

    # Otherwise, rename spatial dims to (lat_name, lon_name) in a stable order
    da2 = da.rename({lat_dim: lat_name, lon_dim: lon_name})
    return da2.transpose(time_name, lat_name, lon_name)


def _stack_variables(
    data: Union[xr.Dataset, Mapping[str, xr.DataArray], Sequence[xr.DataArray]],
    var_names: Optional[Sequence[str]] = None,
    time_name: str = "time",
    lat_name: str = "lat",
    lon_name: str = "lon",
    var_dim: str = "var",
) -> xr.DataArray:
    """
    Build an (time, var, lat, lon) DataArray from various inputs.
    - If data is a Dataset, select var_names (or all variables) and stack.
    - If data is a mapping, use keys as var names.
    - If data is a sequence, require var_names.
    """
    if isinstance(data, xr.Dataset):
        vars_to_use = (
            list(var_names) if var_names is not None else list(data.data_vars)
        )
        das = []
        for v in vars_to_use:
            if v not in data:
                raise KeyError(f"Variable '{v}' not found in Dataset")
            das.append(
                _ensure_time_lat_lon(data[v], time_name, lat_name, lon_name)
            )
        stacked = xr.concat(das, dim=var_dim)
        stacked = stacked.assign_coords({var_dim: vars_to_use})
        return stacked

    if isinstance(data, Mapping):
        vars_to_use = (
            list(data.keys()) if var_names is None else list(var_names)
        )
        das = []
        for v in vars_to_use:
            if v not in data:
                raise KeyError(f"Variable '{v}' not provided in mapping")
            das.append(
                _ensure_time_lat_lon(data[v], time_name, lat_name, lon_name)
            )
        stacked = xr.concat(das, dim=var_dim)
        stacked = stacked.assign_coords({var_dim: vars_to_use})
        return stacked

    # Sequence of DataArrays
    if var_names is None:
        raise ValueError(
            "var_names must be provided when stacking a sequence of DataArrays"
        )
    das = [
        _ensure_time_lat_lon(da, time_name, lat_name, lon_name)
        for da in data  # type: ignore[arg-type]
    ]
    stacked = xr.concat(das, dim=var_dim)
    stacked = stacked.assign_coords({var_dim: list(var_names)})
    return stacked


@dataclass
class WindowIndex:
    start: int
    stop: int  # exclusive


class CESMDataset(Dataset):
    def __init__(
        self,
        data: Union[
            xr.Dataset, Mapping[str, xr.DataArray], Sequence[xr.DataArray]
        ],
        window: int,
        var_names: Optional[Sequence[str]] = None,
        stride: int = 1,
        mask: Optional[xr.DataArray] = None,
        time_name: str = "time",
        lat_name: str = "lat",
        lon_name: str = "lon",
        var_dim: str = "var",
        drop_nan_time: bool = True,
    ) -> None:
        super().__init__()
        if window <= 0:
            raise ValueError("window must be > 0")
        if stride <= 0:
            raise ValueError("stride must be > 0")

        self.time_name = time_name
        self.lat_name = lat_name
        self.lon_name = lon_name
        self.var_dim = var_dim
        self.window = int(window)
        self.stride = int(stride)

        arr = _stack_variables(
            data,
            var_names=var_names,
            time_name=time_name,
            lat_name=lat_name,
            lon_name=lon_name,
            var_dim=var_dim,
        )  # (time, var, lat, lon)

        # Optionally drop all-NaN time steps
        if drop_nan_time:
            # time step valid if any value is finite
            valid_t = np.any(np.isfinite(arr.values), axis=(1, 2, 3))
            if valid_t.ndim != 1 or valid_t.shape[0] != arr.sizes[time_name]:
                raise RuntimeError(
                    "Unexpected valid_t shape when filtering time"
                )
            arr = arr.isel({time_name: valid_t})

        # Compute mask if not provided: valid where all variables have finite values at least once across time
        if mask is None:
            finite_any_time = np.any(
                np.isfinite(arr.values), axis=0
            )  # (var, lat, lon)
            mask_np = np.all(finite_any_time, axis=0)  # (lat, lon)
            # Build coords if available; otherwise fall back to index coords
            lat_coords = arr.coords.get(lat_name, None)
            lon_coords = arr.coords.get(lon_name, None)
            if lat_coords is None:
                lat_coords = np.arange(arr.sizes[lat_name])
            if lon_coords is None:
                lon_coords = np.arange(arr.sizes[lon_name])
            mask = xr.DataArray(
                mask_np,
                coords={lat_name: lat_coords, lon_name: lon_coords},
                dims=(lat_name, lon_name),
            )
        else:
            # ensure dims order and coords match
            mask = mask.transpose(lat_name, lon_name)
            # broadcast/align coords if needed
            mask = mask.reindex_like(
                arr.isel({time_name: 0, var_dim: 0}), method=None
            )
            if mask.dtype != bool:
                mask = mask.astype(bool)

        self._arr = arr
        self._mask = mask

        T = arr.sizes[time_name]
        if T < window:
            raise ValueError(f"Not enough time steps ({T}) for window={window}")

        # Precompute window indices
        self._windows: List[WindowIndex] = []
        for start in range(0, T - window + 1, self.stride):
            self._windows.append(WindowIndex(start=start, stop=start + window))

    @property
    def mask(self) -> xr.DataArray:
        return self._mask

    @property
    def data(self) -> xr.DataArray:
        return self._arr

    def __len__(self) -> int:
        return len(self._windows)

    def __getitem__(self, idx: int):
        win = self._windows[idx]
        slice_x = self._arr.isel({self.time_name: slice(win.start, win.stop)})
        return {
            "x": slice_x,  # (time, var, lat, lon)
            "mask": self._mask,  # (lat, lon)
            "t0": win.start,
            "time": slice_x[self.time_name],
        }


def collate_xarray_time_var_lat_lon(
    samples: Sequence[Mapping[str, xr.DataArray]],
) -> Mapping[str, xr.DataArray]:
    """
    Collate a list of samples from ClimateWindowDataset into a batched xarray structure.
    Returns a dict with:
      - x: (batch, time, var, lat, lon)
      - mask: (batch, lat, lon) if masks differ, else (lat, lon)
      - t0: (batch,) int64
      - time: (batch, time) coordinate values per sample
    """
    if len(samples) == 0:
        raise ValueError("No samples to collate")

    # Stack x along new batch dim
    xs = [s["x"] for s in samples]
    x_b = xr.concat(xs, dim="batch")
    x_b = x_b.assign_coords(batch=np.arange(len(xs)))

    # Collate masks: if all identical, keep a single mask, else stack
    masks = [s["mask"] for s in samples]
    all_equal = all(
        bool((masks[0].values == m.values).all()) for m in masks[1:]
    )
    if all_equal:
        mask_b = masks[0]
    else:
        mask_b = xr.concat(masks, dim="batch").assign_coords(
            batch=np.arange(len(masks))
        )

    # t0 and time per sample
    t0_np = np.array([s["t0"] for s in samples], dtype=np.int64)
    t0 = xr.DataArray(
        t0_np, dims=("batch",), coords={"batch": np.arange(len(samples))}
    )
    times = xr.concat([s["time"] for s in samples], dim="batch").assign_coords(
        batch=np.arange(len(samples))
    )

    return {"x": x_b, "mask": mask_b, "t0": t0, "time": times}


def collate_to_torch(
    samples: Sequence[Mapping[str, xr.DataArray]],
    *,
    dtype: Optional[Any] = None,
    device: Optional[Any] = None,
):
    """
    Collate ClimateWindowDataset samples into PyTorch tensors.

    Returns a dict with:
      - x: torch.Tensor (batch, time, var, lat, lon)
      - mask: torch.BoolTensor (batch, lat, lon) if masks differ, else (lat, lon)
      - t0: torch.LongTensor (batch,)
      - time: list of xarray time coords (length=batch)
    """
    import torch  # local import to avoid hard dependency for non-torch users

    xarr = collate_xarray_time_var_lat_lon(samples)
    x = torch.as_tensor(xarr["x"].values, dtype=dtype, device=device)

    mask_da = xarr["mask"]
    if "batch" in mask_da.dims:
        mask_t = torch.as_tensor(
            mask_da.values, dtype=torch.bool, device=device
        )
    else:
        mask_t = torch.as_tensor(
            mask_da.values, dtype=torch.bool, device=device
        )

    t0_t = torch.as_tensor(xarr["t0"].values, dtype=torch.long, device=device)
    # keep time coords as list of DataArray (can be converted downstream if needed)
    # split along batch
    times_b = []
    if "batch" in xarr["time"].dims:
        for b in range(xarr["time"].sizes["batch"]):
            times_b.append(xarr["time"].isel(batch=b))
    else:
        times_b = [xarr["time"]]

    return {"x": x, "mask": mask_t, "t0": t0_t, "time": times_b}


class ClimateDataset(Dataset):
    """
    A PyTorch Dataset for loading random intervals of climate data.
    """

    def __init__(
        self,
        data_path: str | Path,
        variables: List[str],
        members: List[str],
        interval: int,
        start_year: int = 1850,
        end_year: int = 2100,
        transform=None,
        loading_time_chunck_size=120,
    ):
        self.data_path = Path(data_path)
        self.variables = variables
        self.members = members
        self.interval = interval
        self.start_year = start_year
        self.end_year = end_year

        if not self.members:
            raise ValueError("The 'members' list cannot be empty.")
        if not self.variables:
            raise ValueError("The 'variables' list cannot be empty.")

        # Determine the length of the time series from a sample file
        self._time_len = self._get_time_len()
        self._samples_per_member = self._time_len - self.interval + 1
        self._length = len(self.members) * self._samples_per_member
        self.transform = transform
        self.loading_time_chunk_size = loading_time_chunck_size

    def _get_time_len(self) -> int:
        """
        Gets the time series length from a sample data file.
        """
        sample_member = self.members[0]
        sample_var = self.variables[0]
        time_range = -1
        for v in self.variables:
            try:
                dataset = get_cesm2_member_months(
                    self.data_path,
                    sample_var,
                    sample_member,
                )
                if time_range == -1:
                    time_range = len(dataset.time)
                else:
                    if len(dataset.time) != time_range:
                        raise ValueError(
                            f"Inconsistent time lengths for variable '{v}' and member '{sample_member}'"
                        )

            except FileNotFoundError:
                raise FileNotFoundError(
                    f"Could not find data for variable '{sample_var}' and member '{sample_member}' at path '{self.data_path}'"
                )

        return time_range

    def __len__(self) -> int:
        """
        Returns the total number of samples in the dataset.
        """
        return self._length

    def __getitem__(self, idx: int) -> xr.Dataset:
        member_idx = idx // self._samples_per_member
        start_time = idx % self._samples_per_member
        end_time = start_time + self.interval

        member = self.members[member_idx]

        member_data = []
        for v in self.variables:
            ds = get_cesm2_member_months(
                self.data_path,
                v,
                member,
                start_year=self.start_year,
                end_year=self.end_year,
                time_chunk_size=self.loading_time_chunk_size,
            ).isel(time=slice(start_time, end_time))
            member_data.append(ds)

        stacked_data = xr.merge(member_data, compat="no_conflicts")
        del member_data
        if "z_t" in stacked_data.coords:
            stacked_data = stacked_data.isel(z_t=0)

        stacked_data = stacked_data.drop_vars(
            set(stacked_data.coords).intersection({"ULAT", "ULONG", "z_t"})
        )

        if self.transform is not None:
            stacked_data = self.transform(stacked_data)

        return stacked_data
