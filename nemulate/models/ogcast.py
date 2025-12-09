import torch
from torch import Tensor
from torch.nn import LayerNorm

import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear


def _addindent(s_, numSpaces):
    s = s_.split("\n")
    # don't do anything for single-line stuff
    if len(s) == 1:
        return s_
    first = s.pop(0)
    s = [(numSpaces * " ") + line for line in s]
    s = "\n".join(s)
    s = first + "\n" + s
    return s


def _check_dims(*dims):
    m = min(dims)
    M = max(dims)
    if m == -1:
        if M > 0:
            raise ValueError(f"Either all dimensions must be provided or none: {dims}")
        return -1
    return sum(dims)


class GCastMLP(Linear):

    def __init__(
        self,
        out_channels: int,
        in_channels: int = -1,
        bias: bool = True,
        layer_norm: bool = True,
        activate=True,
    ):
        super().__init__(in_channels, out_channels, bias)
        if layer_norm:
            self.norm_layer = LayerNorm(out_channels)
        else:
            self.norm_layer = None
        self.activate = activate

    def forward(self, x: Tensor) -> Tensor:
        z = F.linear(x, self.weight, self.bias)
        if self.activate:
            z = F.silu(z)
        if self.norm_layer is not None:
            z = self.norm_layer(F.silu(z))
        return z

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}({self.in_channels}, "
            f"{self.out_channels}, bias={self.bias is not None}, "
            f"layer_norm={self.norm_layer is not None}, "
            f"swich={self.activate})"
        )


class GCastEdgeEncoder(torch.nn.Module):
    def __init__(
        self,
        out_channels: int,
        sender_dim: int = -1,
        receiver_dim: int = -1,
        edge_dim: int = -1,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        in_channels = _check_dims(sender_dim, receiver_dim, edge_dim)
        self.edge_proj = GCastMLP(in_channels=in_channels, out_channels=out_channels)

    def forward(
        self,
        sender_x,
        receiver_x,
        edge_index,
        edge_attr,
    ) -> Tensor:
        return self.edge_proj(
            torch.concat(
                [sender_x[edge_index[0]], receiver_x[edge_index[1]], edge_attr],
                dim=-1,
            )
        )


class GCastNodeEncoder(MessagePassing):

    def __init__(
        self,
        out_channels: int,
        receiver_dim: int = -1,
        edge_dim: int = -1,
        *args,
        **kwargs,
    ):
        super().__init__(aggr="sum", *args, **kwargs)

        self.in_channels = _check_dims(receiver_dim, edge_dim)
        self.out_channels = out_channels
        self.receiver_dim = receiver_dim
        self.edge_dim = edge_dim
        self.node_proj = GCastMLP(
            in_channels=self.in_channels, out_channels=out_channels
        )

    def forward(
        self,
        sender_x,
        receiver_x,
        edge_index,
        edge_attr,
    ) -> Tensor:
        edge_aggr = self.propagate(
            edge_index,
            x=sender_x,
            edge_attr=edge_attr,
            size=(sender_x.shape[0], receiver_x.shape[0]),
        )
        # print("GCastNodeEncoder", edge_aggr.shape, receiver_x.shape)
        return self.node_proj(torch.concat([receiver_x, edge_aggr], dim=-1))

    def message(self, x_j, edge_attr: Tensor) -> Tensor: # type: ignore
        return edge_attr

    def __repr__(self):
        # We treat the extra repr like the sub-module, one item per line
        extra_lines = []
        extra_repr = self.extra_repr()
        # empty string will be split into list ['']
        if extra_repr:
            extra_lines = extra_repr.split("\n")
        child_lines = []
        for key, module in self._modules.items():
            mod_str = repr(module)
            mod_str = _addindent(mod_str, 2)
            child_lines.append("(" + key + "): " + mod_str)
        lines = extra_lines + child_lines

        main_str = (
            f"{self._get_name()}("
            f"receiver_dim={self.receiver_dim}, edge_dim={self.edge_dim})("
        )
        if lines:
            # simple one-liner info, which most builtin Modules will use
            if len(extra_lines) == 1 and not child_lines:
                main_str += extra_lines[0]
            else:
                main_str += "\n  " + "\n  ".join(lines) + "\n"

        main_str += ")"
        return main_str


class GCastHeterocoder(torch.nn.Module):

    def __init__(
        self,
        sender_dim: int,
        receiver_dim: int,
        edge_dim: int,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.edges_encoder = GCastEdgeEncoder(
            out_channels=edge_dim,
            sender_dim=sender_dim,
            receiver_dim=receiver_dim,
            edge_dim=edge_dim,
        )
        self.receivers_encoder = GCastNodeEncoder(
            out_channels=receiver_dim,
            receiver_dim=receiver_dim,
            edge_dim=edge_dim,
        )

        self.senders_encoder = GCastMLP(out_channels=sender_dim, in_channels=sender_dim)

    def forward(self, sender_x, receiver_x, edge_index, edge_attr):
        # Equation (7)
        edge_attr_p = self.edges_encoder(
            sender_x=sender_x,
            receiver_x=receiver_x,
            edge_index=edge_index,
            edge_attr=edge_attr,
        )
        # Equation (8)
        mesh_x_p = self.receivers_encoder(
            sender_x=sender_x,
            receiver_x=receiver_x,
            edge_index=edge_index,
            edge_attr=edge_attr_p,
        )
        # Equation (9)
        sender_x = sender_x + self.senders_encoder(sender_x)
        receiver_x = receiver_x + mesh_x_p
        edge_attr = edge_attr + edge_attr_p
        return sender_x, receiver_x, edge_attr


class GCastAutocoder(torch.nn.Module):

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.edge_encoder = GCastEdgeEncoder(
            out_channels=edge_dim,
            sender_dim=node_dim,
            receiver_dim=node_dim,
            edge_dim=edge_dim,
        )
        self.mesh_node_encoder = GCastNodeEncoder(
            node_dim,
            receiver_dim=node_dim,
            edge_dim=edge_dim,
        )

    def forward(self, x, edge_index, edge_attr):
        # Equation (11)
        edge_attr_p = self.edge_encoder(x, x, edge_attr, edge_index)
        # Equation (12)
        x_p = self.mesh_node_encoder(
            sender_x=x,
            receiver_x=x,
            edge_index=edge_index,
            edge_attr=edge_attr_p,
        )
        # Equations (13)
        x = x + x_p
        edge_attr = edge_attr + edge_attr_p
        return x, edge_attr

