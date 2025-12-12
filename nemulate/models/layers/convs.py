from typing import Optional, Tuple
import torch
from torch import nn
from torch import Tensor
from torch.nn.common_types import _size_2_t
from torch.nn.modules.utils import _pair
import torch.nn.functional as F


def polar_reflection(
    x,
    north_padding,
    south_padding,
    lat_dim=-2,
    lon_dim=-1,
):
    r"""
    Map in the format
          . North
     ____/ \_____
    |1 2 3 4 5 6 |
    |7 8 9 0 1 2 |  = Map
    |3 4 5 6 7 8 |
    |____\ /_____
          . South

    with north_padding = 2, south_padding=1
    first reflect the north and south pads
    7 8 9 0 1 2
    1 2 3 4 5 6
    ... Map
    3 4 5 6 7 8

    Then shift them by num_long // 2, so that the final result is:

    0 1 2 7 8 9
    4 5 6 1 2 3
    ... Map
    6 7 8 3 4 5

    """
    assert north_padding > 0 or south_padding > 0, (
        "At least one of north_padding or south_padding must be greater than 0."
    )

    H = x.size(lat_dim)
    W = x.size(lon_dim)

    assert north_padding < H or south_padding < H, (
        "north_padding or south_padding can not exceed H-1"
    )

    shift = W // 2
    parts = []

    if north_padding > 0:
        # start from 0, since on the sphere at the poles, crossing north
        # yields to other rolled side of the 2D grid
        n_pad = x.narrow(lat_dim, 0, north_padding)
        # flip it vertically to create the reflection
        n_pad = torch.flip(n_pad, dims=[lat_dim])
        # roll the reflected slice horizontally
        parts.append(torch.roll(n_pad, shifts=shift, dims=[lon_dim]))

    parts.append(x)

    if south_padding > 0:
        s_pad = x.narrow(lat_dim, -south_padding, south_padding)
        s_pad = torch.flip(s_pad, dims=[lat_dim])
        s_pad = torch.roll(s_pad, shifts=shift, dims=[lon_dim])
        parts.append(s_pad)

    return torch.cat(parts, dim=lat_dim)


def build_reduce_block(in_channels, out_channels, kernel_n):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=(2 * kernel_n + 1, 4 * (kernel_n + 1) + 1),
        stride=2,
        padding=(kernel_n, 2 * (kernel_n + 1)),
    )


def build_conv_block(
    in_channels,
    hidden_channels,
    out_channels,
    kernel_n,
    lat_offset,
):
    conv_block = nn.Sequential(
        nn.Conv2d(
            in_channels,
            hidden_channels,
            kernel_size=(2 * kernel_n + 1, 4 * kernel_n + 1),
            stride=1,
            padding=(kernel_n, 2 * kernel_n),
        ),
        nn.Tanh(),
        nn.Conv2d(
            hidden_channels,
            out_channels,
            kernel_size=(kernel_n + lat_offset, 2 * kernel_n),
            stride=1,
        ),
    )
    return conv_block


class MapStack(nn.Module):
    # def __init__(self, horizontal_n, vertical_n) -> None:
    def __init__(self) -> None:
        super().__init__()
        # self.hor_n = horizontal_n
        # self.ver_n = vertical_n

    def forward(self, inputs):
        mid_map = inputs
        top_map = torch.flip(
            torch.concat(
                [
                    mid_map[:, :, :, 180:],
                    mid_map[:, :, :, :180],
                ],
                dim=3,
            ),
            dims=[2],
        )
        center_column = torch.concat([top_map, mid_map, top_map], dim=2)
        stack = torch.concat(
            [center_column, center_column, center_column], dim=3
        )
        return stack


class EarthConv2d(nn.Conv2d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding: _size_2_t = -1,
        device=None,
        dtype=None,
    ) -> None:
        kernel_size_ = _pair(kernel_size)

        super().__init__(
            in_channels,
            out_channels,
            kernel_size_,
            stride,
            padding=0,
            dilation=dilation,
            groups=groups,
            bias=bias,
            device=device,
            dtype=dtype,
        )

        kernel_size_ = _pair(kernel_size)
        padding = _pair(padding)
        pad_latitude, pad_longitude = padding
        if pad_latitude < 0:
            pad_latitude = (kernel_size_[0] - 1) // 2
        if pad_longitude < 0:
            pad_longitude = (kernel_size_[1] - 1) // 2

        self.pad_lat = pad_latitude
        self.pad_lon = pad_longitude

    def _pad_on_sphere(self, x: Tensor):
        if self.pad_lat:
            x = polar_reflection(x, self.pad_lat, self.pad_lat)
        if self.pad_lon:
            x = F.pad(
                x,
                (self.pad_lon, self.pad_lon, 0, 0),
                mode="circular",
            )
        return x

    def _conv_forward(
        self, input: Tensor, weight: Tensor, bias: Optional[Tensor]
    ):
        return F.conv2d(
            self._pad_on_sphere(input),
            weight,
            bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


if __name__ == "__main__":
    import time

    def benchmark(fn, name=""):
        print(f"--- Benchmarking: {name} ---")
        # warmap
        for _ in range(10):
            _ = fn()

        torch.cuda.synchronize()
        start_time = time.perf_counter()

        for _ in range(100):
            _ = fn()

        torch.cuda.synchronize()
        end_time = time.perf_counter()

        print(f"Time per call: {(end_time - start_time) / 100 * 1e6:.2f} µs\n")

    # Use a large, realistic tensor size
    batch_size = 32
    channels = 256
    height, width = 384, 320
    padding = (4, 4)
    shift = 320 // 2

    # Ensure we are on a capable GPU

    input_gpu = torch.randn(
        batch_size,
        channels,
        height,
        width,
        device="cuda",
        dtype=torch.float16,
    )
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(
        f"Input Tensor: {input_gpu.shape}, {input_gpu.element_size() * input_gpu.numel() / 1e9:.2f} GB\n"
    )

    # --- Benchmark Native Reflection Padding (v_shift=0) ---
    benchmark(
        lambda: polar_reflection(input_gpu, padding[0], padding[1]),
        name="Polar Reflection Pad",
    )

    # --- Benchmark Custom Padding with v_shift=0 ---
    # This should have nearly identical performance to the native version
    benchmark(
        torch.compile(
            lambda: polar_reflection(input_gpu, padding[0], padding[1]),
            fullgraph=True,
        ),
        name="Polar Reflection Pad (Compiled)",
    )
