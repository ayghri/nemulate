from torch import nn
from typing import Optional, List, Tuple, Union
from torch import Tensor
from torch.nn.modules.utils import _single, _pair, _reverse_repeat_tuple
from torch.nn.parameter import Parameter
from torch.nn.common_types import _size_2_t
import torch.nn.functional as F
import math
import torch


class MaskConvNd(nn.Module):

    __constants__ = [
        "stride",
        "padding",
        "dilation",
        "padding_mode",
        "output_padding",
        "in_channels",
        "out_channels",
        "kernel_size",
    ]

    def _conv_forward(self, input: Tensor, weight: Tensor) -> Tensor:  # type: ignore[empty-body]
        ...

    _reversed_padding_repeated_twice: List[int]
    kernel_size: Tuple[int, ...]
    stride: Tuple[int, ...]
    padding: Union[str, Tuple[int, ...]]
    dilation: Tuple[int, ...]
    output_padding: Tuple[int, ...]
    padding_mode: str

    def __init__(
        self,
        kernel_size: Tuple[int, ...],
        stride: Tuple[int, ...],
        padding: Tuple[int, ...],
        dilation: Tuple[int, ...],
        output_padding: Tuple[int, ...],
        padding_mode: str,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        valid_padding_strings = {"same", "valid"}
        if isinstance(padding, str):
            if padding not in valid_padding_strings:
                raise ValueError(
                    f"Invalid padding string {padding!r}, should be one of {valid_padding_strings}"
                )
            if padding == "same" and any(s != 1 for s in stride):
                raise ValueError(
                    "padding='same' is not supported for strided convolutions"
                )

        valid_padding_modes = {"zeros", "reflect", "replicate", "circular"}
        if padding_mode not in valid_padding_modes:
            raise ValueError(
                f"padding_mode must be one of {valid_padding_modes}, but got padding_mode='{padding_mode}'"
            )
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.output_padding = output_padding
        self.padding_mode = padding_mode
        # `_reversed_padding_repeated_twice` is the padding to be passed to
        # `F.pad` if needed (e.g., for non-zero padding types that are
        # implemented as two ops: padding + conv). `F.pad` accepts paddings in
        # reverse order than the dimension.
        if isinstance(self.padding, str):
            self._reversed_padding_repeated_twice = [0, 0] * len(kernel_size)
            if padding == "same":
                for d, k, i in zip(
                    dilation, kernel_size, range(len(kernel_size) - 1, -1, -1)
                ):
                    total_padding = d * (k - 1)
                    left_pad = total_padding // 2
                    self._reversed_padding_repeated_twice[2 * i] = left_pad
                    self._reversed_padding_repeated_twice[2 * i + 1] = (
                        total_padding - left_pad
                    )
        else:
            self._reversed_padding_repeated_twice = _reverse_repeat_tuple(
                self.padding, 2
            )

        self.weight = Parameter(
            torch.ones(
                (1, 1, *kernel_size),
                **factory_kwargs,
            )
            / math.prod(kernel_size),
            requires_grad=False,
        )

    def extra_repr(self):
        s = (
            "{in_channels}, {out_channels}, kernel_size={kernel_size}"
            ", stride={stride}"
        )
        if self.padding != (0,) * len(self.padding):
            s += ", padding={padding}"
        if self.dilation != (1,) * len(self.dilation):
            s += ", dilation={dilation}"
        if self.output_padding != (0,) * len(self.output_padding):
            s += ", output_padding={output_padding}"
        if self.padding_mode != "zeros":
            s += ", padding_mode={padding_mode}"
        return s.format(**self.__dict__)

    def __setstate__(self, state):
        super().__setstate__(state)
        if not hasattr(self, "padding_mode"):
            self.padding_mode = "zeros"


class MaskConv2d(MaskConvNd):
    def __init__(
        self,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: Union[str, _size_2_t] = 0,
        dilation: _size_2_t = 1,
        padding_mode: str = "zeros",  # TODO: refine this type
        mask_threshold: float = 1.0,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        kernel_size_ = _pair(kernel_size)
        stride_ = _pair(stride)
        padding_ = padding if isinstance(padding, str) else _pair(padding)
        dilation_ = _pair(dilation)
        super().__init__(
            kernel_size_,
            stride_,
            padding_,
            dilation_,
            _pair(0),
            padding_mode,
            **factory_kwargs,
        )
        self.mask_threshold = mask_threshold

    def _conv_forward(self, input: Tensor, weight: Tensor):
        if self.padding_mode != "zeros":
            return F.conv2d(
                F.pad(
                    input, self._reversed_padding_repeated_twice, mode=self.padding_mode
                ),
                weight,
                None,  # bias
                self.stride,
                _pair(0),
                self.dilation,
                1,
            )
        return F.conv2d(
            input,
            weight,
            None,  # bias
            self.stride,
            self.padding,
            self.dilation,
            1,
        )

    def forward(self, input: Tensor) -> Tensor:
        return torch.ge(self.forward_values(input), self.mask_threshold).bool()

    def forward_values(self, input: Tensor) -> Tensor:
        with torch.no_grad():
            mask_vals = self._conv_forward(input.type_as(self.weight), self.weight)
        return mask_vals


class MaskConvTransposeNd(MaskConvNd):
    def __init__(
        self,
        kernel_size,
        stride,
        padding,
        dilation,
        output_padding,
        padding_mode,
        device=None,
        dtype=None,
    ) -> None:

        if padding_mode != "zeros":
            raise ValueError(
                f'Only "zeros" padding mode is supported for {self.__class__.__name__}'
            )

        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(
            kernel_size,
            stride,
            padding,
            dilation,
            output_padding,
            padding_mode,
            **factory_kwargs,
        )

    # dilation being an optional parameter is for backwards
    # compatibility
    def _output_padding(
        self,
        input: Tensor,
        output_size: Optional[List[int]],
        stride: List[int],
        padding: List[int],
        kernel_size: List[int],
        num_spatial_dims: int,
        dilation: Optional[List[int]] = None,
    ) -> List[int]:
        if output_size is None:
            ret = _single(self.output_padding)  # converting to list if was not already
        else:
            has_batch_dim = input.dim() == num_spatial_dims + 2
            num_non_spatial_dims = 2 if has_batch_dim else 1
            if len(output_size) == num_non_spatial_dims + num_spatial_dims:
                output_size = output_size[num_non_spatial_dims:]
            if len(output_size) != num_spatial_dims:
                raise ValueError(
                    "ConvTranspose{}D: for {}D input, output_size must have {} or {} elements (got {})".format(
                        num_spatial_dims,
                        input.dim(),
                        num_spatial_dims,
                        num_non_spatial_dims + num_spatial_dims,
                        len(output_size),
                    )
                )

            min_sizes = torch.jit.annotate(List[int], [])
            max_sizes = torch.jit.annotate(List[int], [])
            for d in range(num_spatial_dims):
                dim_size = (
                    (input.size(d + num_non_spatial_dims) - 1) * stride[d]
                    - 2 * padding[d]
                    + (dilation[d] if dilation is not None else 1)
                    * (kernel_size[d] - 1)
                    + 1
                )
                min_sizes.append(dim_size)
                max_sizes.append(min_sizes[d] + stride[d] - 1)

            for i in range(len(output_size)):
                size = output_size[i]
                min_size = min_sizes[i]
                max_size = max_sizes[i]
                if size < min_size or size > max_size:
                    raise ValueError(
                        f"requested an output size of {output_size}, but valid sizes range "
                        f"from {min_sizes} to {max_sizes} (for an input of {input.size()[2:]})"
                    )

            res = torch.jit.annotate(List[int], [])
            for d in range(num_spatial_dims):
                res.append(output_size[d] - min_sizes[d])

            ret = res
        return ret


class MaskConvTranspose2d(MaskConvTransposeNd):
    def __init__(
        self,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: _size_2_t = 0,
        output_padding: _size_2_t = 0,
        dilation: _size_2_t = 1,
        padding_mode: str = "zeros",
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        output_padding = _pair(output_padding)
        super().__init__(
            kernel_size,
            stride,
            padding,
            dilation,
            output_padding,
            padding_mode,
            **factory_kwargs,
        )

    def forward(self, input: Tensor, output_size: Optional[List[int]] = None) -> Tensor:
        if self.padding_mode != "zeros":
            raise ValueError(
                "Only `zeros` padding mode is supported for ConvTranspose2d"
            )

        assert isinstance(self.padding, tuple)
        # One cannot replace List by Tuple or Sequence in "_output_padding" because
        # TorchScript does not support `Sequence[T]` or `Tuple[T, ...]`.
        num_spatial_dims = 2
        output_padding = self._output_padding(
            input,
            output_size,
            self.stride,
            self.padding,
            self.kernel_size,  # type: ignore[arg-type]
            num_spatial_dims,
            self.dilation,
        )  # type: ignore[arg-type]

        return F.conv_transpose2d(
            input,
            self.weight,
            None,  # bias
            self.stride,
            self.padding,
            output_padding,
            1,  # groups
            self.dilation,
        )


class MaskedConv2d(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: Union[str, _size_2_t] = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",  # TODO: refine this type
        mask_threshold: float = 1.0,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.conv2d = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
            device=device,
            dtype=device,
        )
        self.mask_conv2d = MaskConv2d(
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            padding_mode=padding_mode,
            mask_threshold=mask_threshold,
            device=device,
            dtype=dtype,
        )

    def forward(self, inputs, mask=None):
        outputs = self.conv2d(inputs)
        if mask is not None:
            mask_multiplier = self.mask_conv2d.forward_values(mask)
            mask = torch.ge(mask_multiplier, self.mask_conv2d.mask_threshold)
            outputs = torch.masked_fill(outputs, mask, 0.0)
            outputs = outputs / (mask_multiplier + 1e-10)
            return outputs, mask
        return outputs
