from typing import Optional

from torch import Tensor


rnn_state_t = tuple[Tensor, Tensor] | Tensor
opt_rnn_state_t = Optional[rnn_state_t]

rnn_state_tuple_t = tuple[rnn_state_t, ...]
opt_rnn_state_tuple_t = Optional[rnn_state_tuple_t]
