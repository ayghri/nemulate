import torch.nn as nn
import torch
from torch.utils.checkpoint import checkpoint
from torch.utils.checkpoint import checkpoint as ckpt


class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.

        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(
            in_channels=self.input_dim + self.hidden_dim,
            out_channels=4 * self.hidden_dim,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias,
        )

    def forward(self, input_tensor, h_state, c_state, checkpointing=False):
        # h_cur, c_cur = cur_state
        h_cur = h_state
        c_cur = c_state

        # concatenate along channel axis
        combined = torch.cat([input_tensor, h_cur], dim=1)

        if checkpointing:
            combined_conv = checkpoint(self.conv, combined, use_reentrant=True)
        else:
            combined_conv = self.conv(combined)

        cc_i, cc_f, cc_o, cc_g = torch.split(
            combined_conv, self.hidden_dim, dim=1
        )
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (
            torch.zeros(
                batch_size,
                self.hidden_dim,
                height,
                width,
            )
            .to(self.conv.weight)
            .requires_grad_(),
            torch.zeros(
                batch_size,
                self.hidden_dim,
                height,
                width,
            )
            .to(self.conv.weight)
            .requires_grad_(),
        )


class ConvLSTM(nn.Module):
    """

    Parameters:
        input_dim: Number of channels in input
        hidden_dim: Number of hidden channels
        kernel_size: Size of kernel in convolutions
        num_layers: Number of LSTM layers stacked on each other
        batch_first: Whether or not dimension 0 is the batch or not
        bias: Bias or no bias in Convolution
        return_all_layers: Return the list of computations for all layers
        Note: Will do same padding.

    Input:
        A tensor of size B, T, C, H, W or T, B, C, H, W
    Output:
        A tuple of two lists of length num_layers (or length 1 if return_all_layers is False).
            0 - layer_output_list is the list of lists of length T of each output
            1 - last_state_list is the list of last states
                    each element of the list is a tuple (h, c) for hidden state and memory
    Example:
        >> x = torch.rand((32, 10, 64, 128, 128))
        >> convlstm = ConvLSTM(64, 16, 3, 1, True, True, False)
        >> _, last_states = convlstm(x)
        >> h = last_states[0][0]  # 0 for layer index, 0 for h index
    """

    def __init__(
        self,
        input_dim,
        hidden_dim,
        kernel_size,
        num_layers,
        bias=True,
        return_all_layers=False,
    ):
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError("Inconsistent list length.")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(
                ConvLSTMCell(
                    input_dim=cur_input_dim,
                    hidden_dim=self.hidden_dim[i],
                    kernel_size=self.kernel_size[i],
                    bias=self.bias,
                )
            )

        self.cell_list = nn.ModuleList(cell_list)

    def forward(
        self,
        input_tensor,
        hidden_state=None,
        checkpointing=False,
        checkpoint_conv=False,
    ):
        """

        Parameters
        ----------
        input_tensor: todo
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful

        Returns
        -------
        last_state_list, layer_output
        """

        b, _, _, h, w = input_tensor.size()

        # Implement stateful ConvLSTM

        hidden_state = self._init_hidden(batch_size=b, image_size=(h, w))

        # layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor
        layer_output = None

        for layer_idx in range(self.num_layers):
            h_state, c_state = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                if not checkpointing:
                    h_state, c_state = self.cell_list[layer_idx](
                        cur_layer_input[:, t, :, :, :],
                        h_state,
                        c_state,
                        checkpointing=checkpoint_conv,
                    )
                else:
                    h_state, c_state = checkpoint(
                        self.cell_list[layer_idx],
                        cur_layer_input[:, t, :, :, :],
                        h_state,
                        c_state,
                        use_reentrant=True,
                    )
                output_inner.append(h_state)

            layer_output = torch.stack(output_inner, dim=1)

            # layer_output_list.append(layer_output)
            # if self.return_all_layers:
            #     last_state_list.append([h_state, c_state])
            # else:
            if layer_idx == self.num_layers - 1:
                last_state_list.append([h_state, c_state])

            cur_layer_input = layer_output

        # return layer_output_list, last_state_list
        return layer_output

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(
                self.cell_list[i].init_hidden(batch_size, image_size)  # type: ignore
            )
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (
            isinstance(kernel_size, tuple)
            or (
                isinstance(kernel_size, list)
                and all([isinstance(elem, tuple) for elem in kernel_size])
            )
        ):
            raise ValueError("`kernel_size` must be tuple or list of tuples")

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param


class StackedConvLSTM(torch.nn.Module):
    def __init__(
        self,
        cells,
        refeed_mode="teacher",
        p_sched=lambda s: 0.0,
        use_checkpoint=False,
        gamma=1.0,
        rollout_K=None,
    ):
        """
        cells: list of LSTM-like cells (x, (h,c)) -> (y, (h',c'))
        refeed_mode: "teacher" | "hard" | "soft" | "mixed"
        p_sched(step): prob of using model output instead of teacher (scheduled sampling)
        use_checkpoint: checkpoint each (layer,time) step
        gamma: discount for multi-step rollout loss
        rollout_K: if not None and targets extend beyond inputs, roll out K extra steps
        """
        super().__init__()
        self.cells = torch.nn.ModuleList(cells)
        self.refeed_mode = refeed_mode
        self.p_sched = p_sched
        self.use_checkpoint = use_checkpoint
        self.gamma = gamma
        self.rollout_K = rollout_K

    def forward(self, x_seq, y_seq=None, step_idx=0, state=None, embed=None):
        """
        x_seq: [B, T_in, C, H, W] inputs observed up to time T_in
        y_seq: optional targets [B, T_out, ...] for teacher forcing / loss
        state: optional list of (h_l, c_l) for each layer
        embed: optional callable to map discrete tokens -> embeddings (if needed)
        Returns:
            preds_top: [B, T_in + K, ...] predictions from top layer
            final_state: list[(h_L, c_L)]
            loss (if y_seq provided)
        """
        B, T_in = x_seq.shape[:2]
        L = len(self.cells)
        # define per cell
        if state is None:
            state = [
                cell.init_state(B, x_seq.device, x_seq.dtype)
                for cell in self.cells
            ]

        preds = []
        loss = x_seq.new_tensor(0.0) if y_seq is not None else None

        def cell_step(cell, x, hc):
            if self.use_checkpoint:
                # checkpoint wrapper must be a pure function of tensors
                def fn(_x, _h, _c):
                    y, (h2, c2) = cell(_x, (_h, _c))
                    return y, h2, c2

                y, h2, c2 = ckpt(fn, x, hc[0], hc[1], use_reentrant=True)
                return y, (h2, c2)
            else:
                return cell(x, hc)

        # ---- Warm phase: consume observed inputs time-major ----
        for t in range(T_in):
            x_t = x_seq[:, t]  # input to layer 0 at time t
            for l in range(L):
                y_t, state[l] = cell_step(self.cells[l], x_t, state[l])
                x_t = y_t  # feed up to the next layer
            preds.append(x_t)

            # Optionally compute per-step loss on observed window
            if y_seq is not None and t < y_seq.shape[1]:
                loss = loss + torch.nn.functional.mse_loss(
                    x_t, y_seq[:, t]
                )  # or your criterion

            # ---- Scheduled sampling / self-conditioning for next *time* step input ----
            # If your model expects the previous *output* as the next *input*, replace the next x_seq[:, t+1]
            # Below is a generic hook; adapt for discrete vs continuous outputs.
            if self.refeed_mode != "teacher" and t + 1 < T_in:
                p = self.p_sched(step_idx)
                if self.refeed_mode == "hard":
                    use_model = (
                        torch.rand(B, device=x_seq.device) < p
                    ).float()[:, None, None, None]
                    # For discrete tokens, argmax/sample + embed; for continuous, use x_t directly
                    x_model = x_t.detach()
                    x_gt = x_seq[:, t + 1]
                    x_seq[:, t + 1] = (
                        use_model * x_model + (1 - use_model) * x_gt
                    )
                elif self.refeed_mode == "soft":
                    # Mix prediction and ground truth (keeps grad)
                    alpha = p
                    x_seq[:, t + 1] = (
                        alpha * x_t + (1 - alpha) * x_seq[:, t + 1]
                    )
                elif self.refeed_mode == "mixed":
                    # e.g., 50% soft, 50% hard — customize
                    pass

        # ---- Closed-loop rollout beyond observed inputs (predict future) ----
        K = int(self.rollout_K or 0)
        for k in range(1, K + 1):
            x_t = preds[-1]  # start from last top-layer output
            for l in range(L):
                y_t, state[l] = cell_step(self.cells[l], x_t, state[l])
                x_t = y_t
            preds.append(x_t)

            if y_seq is not None and T_in - 1 + k < y_seq.shape[1]:
                w = self.gamma ** (k - 1)
                loss = loss + w * torch.nn.functional.mse_loss(
                    x_t, y_seq[:, T_in - 1 + k]
                )

        preds_top = torch.stack(preds, dim=1)  # [B, T_in + K, ...]
        return (
            preds_top,
            state,
            (loss / max(1, (T_in + K))) if loss is not None else None,
        )
