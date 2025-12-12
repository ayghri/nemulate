from typing import Type
import torch_geometric.nn as gnn
from torch import nn as tnn


def get_model(name) -> Type[tnn.Module]:
    if name == "InvertedEarthAE":
        from nemulate.models.inverted_earth import InvertedEarthAE

        return InvertedEarthAE
    if name == "EarthAE":
        from nemulate.models.coders import EarthAE

        return EarthAE

    raise ValueError(f"Unknown model: {name}")
