from typing import Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .utils import opt_rnn_state_t, rnn_state_t


##############################################################################
# minimal RNN
##############################################################################


def parallel_scan_log(log_coeffs, log_values):
    """
    A log-space implementation of parallel assosiative scan based on Heinsen.

    F. A. Heinsen, "Efficient Parallelization of a Ubiquitous Sequential
    Computation," arXiv preprint arXiv:2311.06281, 2023.
    """
    # log_coeffs: (batch_size, seq_len, input_size)
    # log_values: (batch_size, seq_len + 1, input_size)

    a_star = F.pad(torch.cumsum(log_coeffs, dim=1), (0, 0, 1, 0))

    log_h0_plus_b_star = torch.logcumsumexp(log_values - a_star, dim=1)
    log_h = a_star + log_h0_plus_b_star
    return torch.exp(log_h)


def g(x: Tensor) -> Tensor:
    """
    continuous activation function defined in B.2.1 and B.3.
    """
    return torch.where(x >= 0, x + 0.5, torch.sigmoid(x))


def log_g(x: Tensor) -> Tensor:
    """
    logarithm of activation 'g'
    """
    return torch.where(x >= 0, (F.relu(x) + 0.5).log(), -F.softplus(-x))


class MinRNNBase(nn.Module):
    def parallel_forward(
        self, input: Tensor, hx: Optional[Tensor] = None
    ) -> Tensor:
        """Forward computation in parallel mode of minimal RNNs. The parallel
        mode should be used during the training.

        Args:
            - input: torch.Tensor. Shape is (N, L, H_in).
            - hx: (optional) torch.Tensor. The initial hidden state with
                shape (N, H_hid) or (N, 1, H_hid).
        """
        raise NotImplementedError

    def sequential_forward(
        self, input: Tensor, hx: Optional[Tensor] = None
    ) -> Tensor:
        """Forward computation in sequential mode of minimal RNNs. The
        sequential mode should be used during the inference.

        Args:
            - input: torch.Tensor. Shape is (N, H_in).
            - hx: (optional) torch.Tensor. The initial hidden state with
                shape (N, H_hid) or (N, 1, H_hid).
        """
        raise NotImplementedError

    def forward(self, input: Tensor, hx: Optional[Tensor] = None) -> Tensor:
        """Collection of the forward computation. The minimal RNNs switch two
        modes between training and inference. The parallel mode is used during
        the training and the sequential mode is used during the inference.
        During the training, the series of input are given. During the
        inference, the inputs are given step by step.

        Args:
            - input: torch.Tensor. During the training, shape is (N, L, H_in).
                During the inference, shape is (N, H_in).
            - hx: (optional) torch.Tensor: The initial hidden state with
        """
        if self.training:
            return self.parallel_forward(input, hx)
        else:
            return self.sequential_forward(input, hx)


class MinGRU(MinRNNBase):
    __constants__ = ["input_size", "hidden_size", "bias"]

    input_size: int
    hidden_size: int
    bias: bool

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
    ) -> None:
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias

        self.linear_ih = nn.Linear(input_size, 2 * hidden_size, bias=True)

    def parallel_forward(
        self, input: Tensor, hx: Optional[Tensor] = None
    ) -> Tensor:
        if hx is None:
            hx = torch.zeros(
                input.size(0),
                self.hidden_size,
                dtype=input.dtype,
                device=input.device,
            )
            hx = hx + 1e-6  # avoid log(0)

        if hx.dim() != 2:
            raise ValueError(
                f"MinGRU: Expected hidden to be 2D, got {hx.dim()}D instead"
            )

        # for sequential computation, expand shape to (N, 1, H_hid)
        hx = hx.unsqueeze(1)

        if input.dim() != 3:
            raise ValueError(
                "MinGRU: Expected input to be 3D in training mode, "
                f"got {input.dim()}D instead"
            )

        z, h_inter = self.linear_ih(input).chunk(2, dim=-1)

        log_z = -F.softplus(-z)
        log_coeffs = -F.softplus(z)
        log_h_0 = hx.log()
        log_tilde_h = log_g(h_inter)
        h = parallel_scan_log(
            log_coeffs, torch.cat([log_h_0, log_z + log_tilde_h], dim=1)
        )
        return h[:, 1:]

    def sequential_forward(
        self, input: Tensor, hx: Optional[Tensor] = None
    ) -> Tensor:
        if hx is None:
            hx = torch.zeros(
                input.size(0),
                self.hidden_size,
                dtype=input.dtype,
                device=input.device,
            )

        if hx.dim() != 2:
            raise ValueError(
                f"MinGRU: Expected hidden to be 2D, got {hx.dim()}D instead"
            )

        if input.dim() != 2:
            raise ValueError(
                "MinGRU: Expected input to be 2D in evaluation mode, "
                f"got {input.dim()}D instead"
            )

        z, h_inter = self.linear_ih(input).chunk(2, dim=-1)

        z = torch.sigmoid(z)
        h_tilde = g(h_inter)
        h_t = (1 - z) * hx + z * h_tilde
        return h_t


class MinLSTM(MinRNNBase):
    __constants__ = ["input_size", "hidden_size", "bias"]

    input_size: int
    hidden_size: int
    bias: bool

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
    ) -> None:
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias

        self.linear_ih = nn.Linear(input_size, 3 * hidden_size, bias=True)

    def parallel_forward(
        self, input: Tensor, hx: Optional[Tensor] = None
    ) -> Tensor:
        if hx is None:
            hx = torch.zeros(
                input.size(0),
                self.hidden_size,
                dtype=input.dtype,
                device=input.device,
            )
            hx = hx + 1e-6  # avoid log(0)

        if hx.dim() != 2:
            raise ValueError(
                f"MinLSTM: Expected hidden to be 2D, got {hx.dim()}D instead"
            )

        # for sequential computation, expand shape to (N, 1, H_hid)
        hx = hx.unsqueeze(1)

        if input.dim() != 3:
            raise ValueError(
                "MinLSTM: Expected input to be 3D in training mode, "
                f"got {input.dim()}D instead"
            )

        f_inter, i_inter, h_inter = self.linear_ih(input).chunk(3, dim=-1)

        diff = F.softplus(-f_inter) - F.softplus(-i_inter)
        log_f = -F.softplus(diff)
        log_i = -F.softplus(-diff)
        log_h_0 = hx.log()
        log_tilde_h = log_g(h_inter)
        h = parallel_scan_log(
            log_f, torch.cat([log_h_0, log_i + log_tilde_h], dim=1)
        )
        return h[:, 1:]

    def sequential_forward(
        self, input: Tensor, hx: Optional[Tensor] = None
    ) -> Tensor:
        if hx is None:
            hx = torch.zeros(
                input.size(0),
                self.hidden_size,
                dtype=input.dtype,
                device=input.device,
            )

        if hx.dim() != 2:
            raise ValueError(
                f"MinLSTM: Expected hidden to be 2D, got {hx.dim()}D instead"
            )

        if input.dim() != 2:
            raise ValueError(
                "MinLSTM: Expected input to be 2D in evaluation mode, "
                f"got {input.dim()}D instead"
            )

        f_inter, i_inter, h_inter = self.linear_ih(input).chunk(3, dim=-1)

        f_t = torch.sigmoid(f_inter)
        i_t = torch.sigmoid(i_inter)
        tilde_h_t = g(h_inter)
        f_prime_t = f_t / (f_t + i_t)
        i_prime_t = i_t / (f_t + i_t)
        h_t = f_prime_t * hx + i_prime_t * tilde_h_t
        return h_t


##############################################################################
# trainable initial hidden state
##############################################################################


class InitialStateWrapper(nn.Module):
    def __init__(
        self,
        rnn_cell: nn.RNNCellBase | MinRNNBase,
        trainable: bool = False,
    ) -> None:
        super().__init__()
        self.rnn_cell = rnn_cell
        self.trainable = trainable

        self.rnn_type = type(rnn_cell).__name__
        param = next(rnn_cell.parameters())
        device = param.device
        dtype = param.dtype
        hidden_size = rnn_cell.hidden_size

        if self.rnn_type == "LSTMCell":
            self.initial_state = nn.ParameterList(
                [
                    nn.Parameter(
                        torch.zeros(hidden_size, device=device, dtype=dtype),
                        requires_grad=trainable,
                    )
                    for _ in range(2)  # hidden, cell
                ]
            )
        else:
            self.initial_state = nn.Parameter(
                torch.zeros(hidden_size, device=device, dtype=dtype),
                requires_grad=trainable,
            )

    def get_initial_state(self, batch_size: int) -> rnn_state_t:
        return torch.tile(self.initial_state[None], (batch_size, 1))

    def forward(self, input: Tensor, state: opt_rnn_state_t = None) -> Tensor:
        if state is None:
            batch_size = input.size(0)
            state = self.get_initial_state(batch_size)
        return self.rnn_cell.forward(input, state)
