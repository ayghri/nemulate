import torch
from torch import nn
from torch import Tensor

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.norm import LayerNorm


def _addindent(s_, numSpaces):
    s = s_.split("\n")
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
            raise ValueError(
                f"Either all dimensions must be provided or none: {dims}"
            )
        return -1
    return sum(dims)


class MLP(torch.nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        out_channels: int,
        in_channels: int = -1,
        layernorm: bool = True,
    ):
        super().__init__()
        layers = [
            Linear(in_channels, hidden_dim),
            nn.SiLU(),
            Linear(hidden_dim, out_channels),
        ]
        if layernorm:
            layers.append(LayerNorm(in_channels=out_channels, mode="graph"))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.mlp(x)


class EdgeEncoder(torch.nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        out_channels: int,
        edge_index: torch.Tensor,
        sender_dim: int = -1,
        receiver_dim: int = -1,
        edge_dim: int = -1,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        in_channels = _check_dims(sender_dim, receiver_dim, edge_dim)
        self.edge_index = edge_index
        self.edge_proj = MLP(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            out_channels=out_channels,
        )

    def forward(
        self,
        sender_x,
        receiver_x,
        edge_attr,
    ) -> Tensor:
        return self.edge_proj(
            torch.concat(
                [
                    sender_x[self.edge_index[0]],
                    receiver_x[self.edge_index[1]],
                    edge_attr,
                ],
                dim=-1,
            )
        )


class NodeEncoder(MessagePassing):
    def __init__(
        self,
        edge_index: torch.Tensor,
        hidden_dim: int,
        out_channels: int,
        receiver_dim: int = -1,
        edge_dim: int = -1,
        layernorm_node: bool = True,
        *args,
        **kwargs,
    ):
        super().__init__(aggr="mean", *args, **kwargs)

        self.edge_index = edge_index
        self.in_channels = _check_dims(receiver_dim, edge_dim)
        self.out_channels = out_channels
        self.receiver_dim = receiver_dim
        self.edge_dim = edge_dim
        self.node_proj = MLP(
            in_channels=self.in_channels,
            hidden_dim=hidden_dim,
            out_channels=out_channels,
            layernorm=layernorm_node,
        )

    def forward(
        self,
        sender_x,
        receiver_x,
        edge_attr,
    ) -> Tensor:
        edge_aggr = self.propagate(
            self.edge_index,
            x=sender_x,
            edge_attr=edge_attr,
            size=(sender_x.shape[0], receiver_x.shape[0]),
        )
        return self.node_proj(torch.concat([receiver_x, edge_aggr], dim=-1))

    def message(self, x_j, edge_attr: Tensor) -> Tensor:  # type: ignore
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


class Heterocoder(torch.nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        sender_dim: int,
        receiver_dim: int,
        edge_dim: int,
        edge_index: torch.Tensor,
        encoder: bool = True,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.edges_encoder = EdgeEncoder(
            edge_index=edge_index,
            hidden_dim=hidden_dim,
            out_channels=edge_dim,
        )
        self.receivers_encoder = NodeEncoder(
            edge_index=edge_index,
            hidden_dim=hidden_dim,
            out_channels=receiver_dim,
        )
        if encoder:
            self.senders_encoder = MLP(
                in_channels=sender_dim,
                hidden_dim=hidden_dim,
                out_channels=sender_dim,
            )

    def forward(self, sender_x, receiver_x, edge_attr):
        # Equation (7)
        edge_attr_p = self.edges_encoder(
            sender_x=sender_x,
            receiver_x=receiver_x,
            edge_attr=edge_attr,
        )
        # Equation (8)
        receiver_x_p = self.receivers_encoder(
            sender_x=sender_x,
            receiver_x=receiver_x,
            edge_attr=edge_attr_p,
        )
        # Equation (9)
        if self.encoder:
            sender_x = sender_x + self.senders_encoder(sender_x)
            edge_attr = edge_attr + edge_attr_p
        receiver_x = receiver_x + receiver_x_p
        return sender_x, receiver_x, edge_attr


class Autocoder(torch.nn.Module):
    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int,
        edge_index: torch.Tensor,
    ) -> None:
        super().__init__()
        self.edge_encoder = EdgeEncoder(
            edge_index=edge_index,
            hidden_dim=hidden_dim,
            out_channels=edge_dim,
        )
        self.node_encoder = NodeEncoder(
            edge_index=edge_index,
            hidden_dim=hidden_dim,
            out_channels=node_dim,
        )

    def forward(self, x, edge_attr):
        # Equation (11)
        edge_attr_p = self.edge_encoder(
            x,
            x,
            edge_attr=edge_attr,
        )
        # Equation (12)
        x_p = self.node_encoder(
            sender_x=x,
            receiver_x=x,
            edge_attr=edge_attr_p,
        )
        # Equations (13)
        x = x + x_p
        edge_attr = edge_attr + edge_attr_p
        return x, edge_attr


class EdgeScaler(MessagePassing):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(aggr="max", *args, **kwargs)

    def forward(
        self,
        node_x,
        edge_index,
        edge_scale,
        edge_attr,
    ) -> Tensor:
        edge_scaler = self.propagate(
            edge_index,
            x=node_x,
            edge_attr=edge_scale,
        )
        return edge_attr / edge_scaler[edge_index[1]]

    def message(self, x_j, edge_attr: Tensor) -> Tensor:  # type: ignore
        return edge_attr


class GraphcastEmbedding(torch.nn.Module):
    def __init__(self, hidden_dim, node_dim, edge_dim, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.grid_v_embed = MLP(hidden_dim=hidden_dim, out_channels=node_dim)
        self.mesh_v_embed = MLP(hidden_dim=hidden_dim, out_channels=node_dim)
        self.g2m_e_embed = MLP(hidden_dim=hidden_dim, out_channels=edge_dim)
        self.m2m_e_embed = MLP(hidden_dim=hidden_dim, out_channels=edge_dim)
        self.m2g_e_embed = MLP(hidden_dim=hidden_dim, out_channels=edge_dim)

    def forward(self, grid_v, mesh_v, g2m_e_attr, m2m_e_attr, m2g_e_attr):
        grid_v = self.grid_v_embed(grid_v)
        mesh_v = self.mesh_v_embed(mesh_v)
        g2m_e_attr = self.g2m_e_embed(g2m_e_attr)
        m2m_e_attr = self.m2m_e_embed(m2m_e_attr)
        m2g_e_attr = self.m2g_e_embed(m2g_e_attr)
        return grid_v, mesh_v, g2m_e_attr, m2m_e_attr, m2g_e_attr


class GraphcastProcessor(torch.nn.Module):
    def __init__(
        self,
        m2m_edge_index,
        num_layers,
        node_dim,
        edge_dim,
        hidden_dim,
    ) -> None:
        super().__init__()

        self.layers = []
        for i in range(num_layers):
            layer = Autocoder(
                edge_index=m2m_edge_index,
                node_dim=node_dim,
                edge_dim=edge_dim,
                hidden_dim=hidden_dim,
            )
            self.register_module(f"processor_layer_{i}", layer)
            self.layers.append(layer)

    def forward(self, mesh_v, edge_attr):
        for layer in self.layers:
            mesh_v, edge_attr = layer(mesh_v, edge_attr)
        return mesh_v, edge_attr


class Graphcast(torch.nn.Module):
    def __init__(
        self,
        g2m_edge_index,
        m2m_edge_index,
        m2g_edge_index,
        hidden_dim,
        node_embed_dim,
        edge_embed_dim,
        out_channels,
        num_processor_layers,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.embedder = GraphcastEmbedding(
            hidden_dim=hidden_dim,
            node_dim=node_embed_dim,
            edge_dim=edge_embed_dim,
        )
        self.encoder = Heterocoder(
            hidden_dim=hidden_dim,
            sender_dim=node_embed_dim,
            receiver_dim=node_embed_dim,
            edge_dim=edge_embed_dim,
            edge_index=g2m_edge_index,
        )
        self.processor = GraphcastProcessor(
            m2m_edge_index,
            num_layers=num_processor_layers,
            node_dim=node_embed_dim,
            edge_dim=edge_embed_dim,
            hidden_dim=hidden_dim,
        )
        self.decoder = Heterocoder(
            hidden_dim=hidden_dim,
            sender_dim=node_embed_dim,
            receiver_dim=node_embed_dim,
            edge_dim=edge_embed_dim,
            edge_index=m2g_edge_index,
            encoder=False,
        )
        self.predictor = MLP(
            hidden_dim=hidden_dim, out_channels=out_channels, layernorm=False
        )

    def forward(
        self,
        grid_v,
        mesh_v,
        g2m_edge_attr,
        m2m_edge_attr,
        m2g_edge_attr,
    ):
        grid_v, mesh_v, g2m_edge_attr, m2m_edge_attr, m2g_edge_attr = (
            self.embedder(
                grid_v, mesh_v, g2m_edge_attr, m2m_edge_attr, m2g_edge_attr
            )
        )
        grid_v, mesh_v, _ = self.encoder(grid_v, mesh_v, g2m_edge_attr)
        mesh_v, _ = self.processor(mesh_v, m2m_edge_attr)
        _, grid_v, _ = self.decoder(mesh_v, grid_v, m2g_edge_attr)
        pred = self.predictor(grid_v)
        return pred
