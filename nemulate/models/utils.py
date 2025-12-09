from typing import Dict, List, Tuple, Optional
from torch import nn
import torch
from torch import autograd
from torch import cuda


NAME2LAYER = {
    "linear": nn.Linear,
    "dropout": nn.Dropout,
    "relu": nn.ReLU,
    "silu": nn.SiLU,
}


def get_layer(layer_name: str, **kwargs) -> nn.Module:
    if layer_name not in NAME2LAYER.keys():
        raise ValueError(
            f"Unknown {layer_name}, known: {list(NAME2LAYER.keys())}"
        )
    return NAME2LAYER[layer_name](**kwargs)


def build_network(layers_list: List[Tuple[str, Dict]]) -> nn.Module:
    layers = []
    for layer_name in layers_list:
        layers.append(get_layer(layer_name[0], layer_name[1]))
    return nn.Sequential(*layers)


class PrefetchFunction(autograd.Function):
    copy_stream: Optional[torch.cuda.Stream] = None

    @staticmethod
    def forward(
        ctx,
        x,
        prefetch_cache: Dict[str, torch.Tensor],
        prefetch_hint: str,
        target_device: torch.device,
    ):
        ctx.device = target_device
        ctx.prefetch_cache = prefetch_cache
        ctx.prefetch_hint = prefetch_hint
        if PrefetchFunction.copy_stream is None:
            PrefetchFunction.copy_stream = torch.cuda.Stream(
                device=target_device
            )
        return x

    @staticmethod
    def backward(ctx, grad_cpu):
        device = ctx.device
        prefetch_cache = ctx.prefetch_cache
        prefetch_hint = ctx.prefetch_hint
        end_event = torch.cuda.Event(blocking=False)
        with torch.cuda.stream(PrefetchFunction.copy_stream):
            prefetched = grad_cpu.to(device, non_blocking=True)
            end_event.record(PrefetchFunction.copy_stream)
        prefetch_cache[prefetch_hint] = (prefetched, end_event)
        return None, None, None, None


class OffloadToCPU(autograd.Function):
    @staticmethod
    def forward(
        ctx, x, prefetch_cache: Dict[str, torch.Tensor], prefetch_hint: str
    ):
        ctx.device = x.device
        ctx.prefetch_cache = prefetch_cache
        ctx.prefetch_hint = prefetch_hint
        old_x = x
        x = x.to("cpu", non_blocking=True).pin_memory()
        old_x.storage().resize_(0)  # HACK
        return x

    @staticmethod
    def backward(ctx, grad_x):
        device = ctx.device
        prefetch_cache = ctx.prefetch_cache
        prefetch_hint = ctx.prefetch_hint
        assert grad_x.device == "cpu", (
            f"Expected grad_x to be on CPU, but got {grad_x.device}"
        )
        if prefetch_hint in prefetch_cache:
            grad_x, end_event = prefetch_cache[prefetch_hint]
            torch.cuda.current_stream(device).wait_event(end_event)
            del prefetch_cache[prefetch_hint]
            assert grad_x.device == device, (
                f"Expected prefetch cache to have device {device}, but got {grad_x.device}"
            )
            return grad_x, None, None, None
        else:
            return grad_x.to(device, non_blocking=True), None, None, None
