import torch
from torch import Tensor

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import (
    Adj,
    OptTensor,
    Size,
)


class GINEConv(MessagePassing):
    r"""The modified :class:`GINConv` operator from the `"Strategies for
    Pre-training Graph Neural Networks" <https://arxiv.org/abs/1905.12265>`_
    paper.

    .. math::
        \mathbf{x}^{\prime}_i = h_{\mathbf{\Theta}} \left( (1 + \epsilon) \cdot
        \mathbf{x}_i + \sum_{j \in \mathcal{N}(i)} \mathrm{ReLU}
        ( \mathbf{x}_j + \mathbf{e}_{j,i} ) \right)

    that is able to incorporate edge features :math:`\mathbf{e}_{j,i}` into
    the aggregation procedure.

    Args:
        nn (torch.nn.Module): A neural network :math:`h_{\mathbf{\Theta}}` that
            maps node features :obj:`x` of shape :obj:`[-1, in_channels]` to
            shape :obj:`[-1, out_channels]`, *e.g.*, defined by
            :class:`torch.nn.Sequential`.
        eps (float, optional): (Initial) :math:`\epsilon`-value.
            (default: :obj:`0.`)
        train_eps (bool, optional): If set to :obj:`True`, :math:`\epsilon`
            will be a trainable parameter. (default: :obj:`False`)
        edge_dim (int, optional): Edge feature dimensionality. If set to
            :obj:`None`, node and edge feature dimensionality is expected to
            match. Other-wise, edge features are linearly transformed to match
            node feature dimensionality. (default: :obj:`None`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.

    Shapes:
        - **input:**
          node features :math:`(|\mathcal{V}|, F_{in})` or
          :math:`((|\mathcal{V_s}|, F_{s}), (|\mathcal{V_t}|, F_{t}))`
          if bipartite,
          edge indices :math:`(2, |\mathcal{E}|)`,
          edge features :math:`(|\mathcal{E}|, D)` *(optional)*
        - **output:** node features :math:`(|\mathcal{V}|, F_{out})` or
          :math:`(|\mathcal{V}_t|, F_{out})` if bipartite
    """

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        out_channels: int,
        **kwargs,
    ):
        # kwargs.setdefault("aggr", "add")
        super().__init__(aggr="max", **kwargs)
        self.node_lin = Linear(node_dim, out_channels)
        self.edge_lin = Linear(edge_dim, out_channels)

        self.reset_parameters()

    def reset_parameters(self):
        self.node_lin.reset_parameters()
        self.edge_lin.reset_parameters()

    def forward(
        self,
        x: Tensor,
        edge_index: Adj,
        edge_attr: OptTensor = None,
        size: Size = None,
    ) -> Tensor:
        # propagate_type: (x: OptPairTensor, edge_attr: OptTensor)
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=size)
        out = torch.concat([x, out], dim=-1)
        return out
        # return self.node_lin(out)

    def message(self, x_j, edge_attr: Tensor) -> Tensor: # type: ignore
        print(x_j, edge_attr)
        # return self.edge_lin(edge_attr)
        return edge_attr

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(node_lin={self.node_lin}, edge_lin={self.edge_lin})"


class GCastConv(MessagePassing):

    def __init__(
        self,
        source_dim: int,
        target_dim: int,
        edge_dim: int,
        out_channels: int,
        **kwargs,
    ):
        # kwargs.setdefault("aggr", "add")
        super().__init__(aggr="max", **kwargs)
        self.source_lin = Linear(source_dim, out_channels)
        self.target_lin = Linear(target_dim, out_channels)
        self.edge_lin = Linear(edge_dim, out_channels)

        self.reset_parameters()

    def reset_parameters(self):
        self.source_lin.reset_parameters()
        self.target_lin.reset_parameters()
        self.edge_lin.reset_parameters()

    def forward(
        self,
        source_x: Tensor,
        target_x: Tensor,
        edge_index: Adj,
        edge_attr: Tensor,
    ) -> Tensor:
        sourcex = self.source_lin(source_x)
        targetx = self.target_lin(target_x)
        edgex = self.edge_lin(edge_attr)
        out = self.propagate(
            edge_index, sourcex=sourcex, targetx=targetx, edge_attr=edgex
        )
        out = torch.concat([targetx, out], dim=-1)
        return out
        # return self.node_lin(out)

    def message(self, x_j, x_i, edge_attr: Tensor) -> Tensor: # type: ignore
        # print(x_j, edge_attr)
        print(x_j, x_i, edge_attr)
        # return self.edge_lin(edge_attr)
        return edge_attr + x_j

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(node_lin={self.source_lin}, edge_lin={self.edge_lin})"
