from typing import Optional

import torch
from torch import nn, Tensor
from torch.nn.utils import spectral_norm

from .utils import (
    rnn_state_t,
    rnn_state_tuple_t,
    opt_rnn_state_t,
    opt_rnn_state_tuple_t,
)
from .torch_modules import MinRNNBase, MinGRU, MinLSTM, InitialStateWrapper
from agents.models.common.utils import return_activiation_fcn


def get_network_type(rnn_type: str) -> str:
    rnn_type = rnn_type.lower()
    if rnn_type in ("mingru", "minlstm"):
        network_type = "minrnn"
    else:
        network_type = "rnn"
    return network_type


def load_rnn_cell(
    rnn_type: str, input_dim: int, hidden_dim: int, bias: bool = True
) -> nn.RNNCellBase | MinRNNBase:
    rnn_type = rnn_type.lower()
    if rnn_type == "rnn":
        cell = nn.RNNCell(input_dim, hidden_dim, bias=bias)
    elif rnn_type == "gru":
        cell = nn.GRUCell(input_dim, hidden_dim, bias=bias)
    elif rnn_type == "lstm":
        cell = nn.LSTMCell(input_dim, hidden_dim, bias=bias)
    elif rnn_type == "mingru":
        cell = MinGRU(input_dim, hidden_dim, bias=bias)
    elif rnn_type == "minlstm":
        cell = MinLSTM(input_dim, hidden_dim, bias=bias)
    else:
        ValueError("Module is not implemented! Please check spelling.")
    return cell


class TwoLayerResNetRNN(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 100,
        rnn_type: str = "lstm",
        # activation: str = 'relu',
        dropout_rate: float = 0.25,
        spectral_norm: bool = False,
        use_norm: bool = False,
        norm_style: int = "BatchNorm",
        train_initial_state: bool = False,
    ) -> None:
        super().__init__()

        self.l1 = load_rnn_cell(rnn_type, hidden_dim, hidden_dim)
        self.l2 = load_rnn_cell(rnn_type, hidden_dim, hidden_dim)
        if train_initial_state:
            self.l1 = InitialStateWrapper(self.l1, trainable=True)
            self.l2 = InitialStateWrapper(self.l2, trainable=True)
        if spectral_norm:
            self.l1 = spectral_norm(self.l1)
            self.l2 = spectral_norm(self.l2)

        self.dropout = nn.Dropout(dropout_rate)
        self.use_norm = use_norm

        if use_norm:
            if norm_style == "BatchNorm":
                self.normalizer = nn.BatchNorm1d(hidden_dim)
            elif norm_style == "LayerNorm":
                self.normalizer = torch.nn.LayerNorm(hidden_dim, eps=1e-06)
            else:
                raise ValueError("not a defined norm type")

    def forward(
        self, x: Tensor, l1_state: opt_rnn_state_t, l2_state: opt_rnn_state_t
    ) -> tuple[Tensor, rnn_state_t, rnn_state_t]:
        x_input = x
        if self.use_norm:
            x = self.normalizer(x)
        l1_state = self.l1(self.dropout(x), l1_state)
        x = l1_state[0] if isinstance(l1_state, tuple) else l1_state
        if self.use_norm:
            x = self.normalizer(x)
        l2_state = self.l2(self.dropout(x), l2_state)
        x = l2_state[0] if isinstance(l2_state, tuple) else l2_state
        return x + x_input, l1_state, l2_state


class RecurrentNetwork(nn.Module):
    """
    Simple multi layer perceptron network which can be generated with different
    activation functions with and without spectral normalization of the weights
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 100,
        num_hidden_layers: int = 1,
        output_dim=1,
        rnn_type: str = "lstm",
        activation: str = "ReLU",
        train_initial_state: bool = False,
        device: str = "cuda",
    ):
        super(RecurrentNetwork, self).__init__()
        self.network_type = get_network_type(rnn_type)
        # define number of variables in an input sequence
        self.input_dim = input_dim
        # the dimension of neurons in the hidden layer
        self.hidden_dim = hidden_dim
        self.num_hidden_layers = num_hidden_layers
        self.num_rnn_layers = num_hidden_layers - 1
        # number of samples per batch
        self.output_dim = output_dim
        # set up the network
        self.layers = nn.ModuleList(
            [nn.Linear(self.input_dim, self.hidden_dim)]
        )
        for _ in range(self.num_rnn_layers):
            cell = load_rnn_cell(rnn_type, self.hidden_dim, self.hidden_dim)
            if train_initial_state:
                cell = InitialStateWrapper(cell, trainable=True)
            self.layers.append(cell)
        self.layers.append(nn.Linear(self.hidden_dim, self.output_dim))

        # build the activation layer
        self.act = return_activiation_fcn(activation)
        self._device = device
        self.layers.to(self._device)

    def forward(
        self, x: Tensor, rnn_states: opt_rnn_state_tuple_t = None
    ) -> tuple[Tensor, rnn_state_tuple_t]:
        assert rnn_states is None or len(rnn_states) == self.num_rnn_layers

        if rnn_states is None:
            rnn_states = (None,) * self.num_rnn_layers

        new_rnn_states = [None] * self.num_rnn_layers
        for idx, layer in enumerate(self.layers):
            if idx == 0:
                out = self.act(layer(x))
            else:
                if idx < len(self.layers) - 2:
                    out, new_rnn_state = layer(out, rnn_states[idx - 1])
                    new_rnn_states[idx - 1] = new_rnn_state
                else:
                    out = layer(out)

        return out, tuple(new_rnn_states)

    def get_device(self, device: torch.device) -> None:
        self._device = device
        self.layers.to(device)

    def get_params(self):
        return self.layers.parameters()


class ResidualRNN(nn.Module):
    """
    Simple multi layer perceptron network with residual connections for
    benchmarking the performance of different networks. The resiudal layers
    are based on the IBC paper implementation, which uses 2 residual lalyers
    with pre-actication with or without dropout and normalization.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 100,
        num_hidden_layers: int = 2,
        output_dim: int = 1,
        rnn_type: str = "lstm",
        dropout: int = 0,
        use_spectral_norm: bool = False,
        use_norm: bool = False,
        norm_style: str = "BatchNorm",
        train_initial_state: bool = False,
        device: str = "cuda",
    ):
        super(ResidualRNN, self).__init__()
        self.network_type = get_network_type(rnn_type)
        self._device = device
        # set up the network

        assert num_hidden_layers % 2 == 0
        self.num_rnn_layers = num_hidden_layers // 2

        if use_spectral_norm:
            self.layers = nn.ModuleList(
                [spectral_norm(nn.Linear(input_dim, hidden_dim))]
            )
        else:
            self.layers = nn.ModuleList([nn.Linear(input_dim, hidden_dim)])
        self.layers.extend(
            [
                TwoLayerResNetRNN(
                    hidden_dim=hidden_dim,
                    rnn_type=rnn_type,
                    dropout_rate=dropout,
                    spectral_norm=use_spectral_norm,
                    use_norm=use_norm,
                    norm_style=norm_style,
                    train_initial_state=train_initial_state,
                )
                for i in range(1, num_hidden_layers, 2)
            ]
        )
        self.layers.append(nn.Linear(hidden_dim, output_dim))
        self.layers.to(self._device)

    def forward(
        self,
        x: Tensor,
        rnn_states: Optional[
            tuple[opt_rnn_state_tuple_t, opt_rnn_state_tuple_t]
        ] = None,
    ):
        if rnn_states is None:
            rnn_states = ((None, None),) * self.num_rnn_layers

        new_rnn_states = [None] * self.num_rnn_layers
        for idx, layer in enumerate(self.layers):
            if idx == 0 or idx == len(self.layers) - 1:
                x = layer(x.to(torch.float32))
            else:
                l1_state, l2_state = rnn_states[idx - 1]
                x, new_l1_state, new_l2_state = layer(
                    x.to(torch.float32), l1_state, l2_state
                )
                new_rnn_states[idx - 1] = (new_l1_state, new_l2_state)

        return x, tuple(new_rnn_states)

    def get_device(self, device: torch.device):
        self._device = device
        self.layers.to(device)

    def get_params(self):
        return self.layers.parameters()
