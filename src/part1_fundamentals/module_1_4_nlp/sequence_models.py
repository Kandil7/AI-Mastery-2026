"""
NLP Sequence Models Module.

This module provides from-scratch implementations of recurrent neural networks,
including RNN, LSTM, and GRU cells.

Features:
- Vanilla RNN cell
- LSTM (Long Short-Term Memory) cell
- GRU (Gated Recurrent Unit) cell
- Bidirectional wrappers
- Multi-layer stacking
- Sequence-to-sequence models

Example Usage:
    >>> from sequence_models import RNN, LSTM, GRU
    >>> 
    >>> # LSTM
    >>> lstm = LSTM(input_size=100, hidden_size=256)
    >>> x = np.random.randn(32, 10, 100)  # (batch, seq_len, input_size)
    >>> output, hidden = lstm.forward(x)
    >>> 
    >>> # Bidirectional LSTM
    >>> bi_lstm = LSTM(input_size=100, hidden_size=256, bidirectional=True)
    >>> output, hidden = bi_lstm.forward(x)
"""

from typing import Union, List, Dict, Tuple, Optional, Callable
import numpy as np
from numpy.typing import ArrayLike
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

ArrayLike3D = Union[np.ndarray, List]


@dataclass
class RNNState:
    """RNN hidden state container."""
    h: np.ndarray  # Hidden state
    
    def copy(self) -> 'RNNState':
        """Create a copy of the state."""
        return RNNState(h=self.h.copy())


@dataclass
class LSTMState:
    """LSTM hidden state container."""
    h: np.ndarray  # Hidden state
    c: np.ndarray  # Cell state
    
    def copy(self) -> 'LSTMState':
        """Create a copy of the state."""
        return LSTMState(h=self.h.copy(), c=self.c.copy())


@dataclass
class GRUState:
    """GRU hidden state container."""
    h: np.ndarray  # Hidden state
    
    def copy(self) -> 'GRUState':
        """Create a copy of the state."""
        return GRUState(h=self.h.copy())


class RNNCell:
    """
    Vanilla RNN cell.
    
    h_t = tanh(W_ih @ x_t + b_ih + W_hh @ h_{t-1} + b_hh)
    
    Example:
        >>> cell = RNNCell(input_size=100, hidden_size=256)
        >>> x = np.random.randn(32, 100)
        >>> h_prev = np.zeros((32, 256))
        >>> h_next = cell.forward(x, h_prev)
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        nonlinearity: str = 'tanh'
    ):
        """
        Initialize RNN cell.
        
        Args:
            input_size: Input feature dimension.
            hidden_size: Hidden state dimension.
            nonlinearity: Activation function ('tanh' or 'relu').
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.nonlinearity = nonlinearity
        
        # Initialize weights (Xavier initialization)
        scale_ih = np.sqrt(2.0 / (input_size + hidden_size))
        scale_hh = np.sqrt(2.0 / (hidden_size + hidden_size))
        
        self.W_ih = np.random.randn(hidden_size, input_size) * scale_ih
        self.W_hh = np.random.randn(hidden_size, hidden_size) * scale_hh
        self.b_ih = np.zeros(hidden_size)
        self.b_hh = np.zeros(hidden_size)
        
        # Gradients
        self.grad_W_ih = np.zeros_like(self.W_ih)
        self.grad_W_hh = np.zeros_like(self.W_hh)
        self.grad_b_ih = np.zeros_like(self.b_ih)
        self.grad_b_hh = np.zeros_like(self.b_hh)
        
        # Cache for backward
        self._cache: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None
        
        logger.debug(f"RNNCell: input={input_size}, hidden={hidden_size}, "
                    f"nonlinearity={nonlinearity}")
    
    def _activation(self, x: np.ndarray) -> np.ndarray:
        """Apply activation function."""
        if self.nonlinearity == 'tanh':
            return np.tanh(x)
        elif self.nonlinearity == 'relu':
            return np.maximum(0, x)
        else:
            raise ValueError(f"Unknown nonlinearity: {self.nonlinearity}")
    
    def _activation_derivative(self, x: np.ndarray, output: np.ndarray) -> np.ndarray:
        """Compute activation derivative."""
        if self.nonlinearity == 'tanh':
            return 1 - output ** 2
        elif self.nonlinearity == 'relu':
            return (x > 0).astype(np.float64)
        else:
            raise ValueError(f"Unknown nonlinearity: {self.nonlinearity}")
    
    def forward(
        self,
        x: np.ndarray,
        h_prev: np.ndarray
    ) -> np.ndarray:
        """
        Forward pass for one time step.
        
        Args:
            x: Input (batch_size, input_size).
            h_prev: Previous hidden state (batch_size, hidden_size).
        
        Returns:
            np.ndarray: New hidden state.
        """
        x = np.asarray(x, dtype=np.float64)
        h_prev = np.asarray(h_prev, dtype=np.float64)
        
        # Compute pre-activation
        pre_act = x @ self.W_ih.T + h_prev @ self.W_hh.T + self.b_ih + self.b_hh
        
        # Apply activation
        h_next = self._activation(pre_act)
        
        # Cache for backward
        self._cache = (x, h_prev, pre_act)
        
        logger.debug(f"RNNCell forward: input {x.shape}, hidden {h_next.shape}")
        return h_next
    
    def backward(self, grad_output: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Backward pass for one time step.
        
        Args:
            grad_output: Gradient w.r.t. output (batch_size, hidden_size).
        
        Returns:
            Tuple: (grad_x, grad_h_prev)
        """
        if self._cache is None:
            raise ValueError("Must call forward() before backward()")
        
        x, h_prev, pre_act = self._cache
        
        # Gradient through activation
        act_deriv = self._activation_derivative(pre_act, grad_output / 
                       (x @ self.W_ih.T + h_prev @ self.W_hh.T + self.b_ih + self.b_hh))
        # Recompute output for derivative
        output = self._activation(pre_act)
        act_deriv = 1 - output ** 2 if self.nonlinearity == 'tanh' else (pre_act > 0).astype(np.float64)
        
        grad_pre_act = grad_output * act_deriv
        
        # Gradients w.r.t. parameters
        self.grad_W_ih += grad_pre_act.T @ x
        self.grad_W_hh += grad_pre_act.T @ h_prev
        self.grad_b_ih += np.sum(grad_pre_act, axis=0)
        self.grad_b_hh += np.sum(grad_pre_act, axis=0)
        
        # Gradients w.r.t. inputs
        grad_x = grad_pre_act @ self.W_ih
        grad_h_prev = grad_pre_act @ self.W_hh
        
        return grad_x, grad_h_prev
    
    def get_parameters(self) -> Dict[str, np.ndarray]:
        """Get trainable parameters."""
        return {
            'W_ih': self.W_ih,
            'W_hh': self.W_hh,
            'b_ih': self.b_ih,
            'b_hh': self.b_hh,
        }
    
    def get_gradients(self) -> Dict[str, np.ndarray]:
        """Get parameter gradients."""
        return {
            'W_ih': self.grad_W_ih,
            'W_hh': self.grad_W_hh,
            'b_ih': self.grad_b_ih,
            'b_hh': self.grad_b_hh,
        }
    
    def zero_grad(self) -> None:
        """Zero out gradients."""
        self.grad_W_ih = np.zeros_like(self.W_ih)
        self.grad_W_hh = np.zeros_like(self.W_hh)
        self.grad_b_ih = np.zeros_like(self.b_ih)
        self.grad_b_hh = np.zeros_like(self.b_hh)


class LSTMCell:
    """
    LSTM (Long Short-Term Memory) cell.
    
    Gates:
    - Input gate: i_t = σ(W_i @ [h_{t-1}, x_t] + b_i)
    - Forget gate: f_t = σ(W_f @ [h_{t-1}, x_t] + b_f)
    - Output gate: o_t = σ(W_o @ [h_{t-1}, x_t] + b_o)
    - Cell candidate: g_t = tanh(W_g @ [h_{t-1}, x_t] + b_g)
    
    Updates:
    - c_t = f_t * c_{t-1} + i_t * g_t
    - h_t = o_t * tanh(c_t)
    
    Example:
        >>> cell = LSTMCell(input_size=100, hidden_size=256)
        >>> x = np.random.randn(32, 100)
        >>> h_prev = np.zeros((32, 256))
        >>> c_prev = np.zeros((32, 256))
        >>> h_next, c_next = cell.forward(x, h_prev, c_prev)
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int
    ):
        """
        Initialize LSTM cell.
        
        Args:
            input_size: Input feature dimension.
            hidden_size: Hidden state dimension.
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Combined input and hidden size
        combined_size = input_size + hidden_size
        
        # Initialize weights (Xavier initialization)
        scale = np.sqrt(2.0 / (combined_size + hidden_size))
        
        # Gate weights (concatenated: i, f, g, o)
        self.W = np.random.randn(4 * hidden_size, combined_size) * scale
        self.b = np.zeros(4 * hidden_size)
        
        # Set forget gate bias to 1.0 (common practice for better gradient flow)
        self.b[hidden_size:2*hidden_size] = 1.0
        
        # Gradients
        self.grad_W = np.zeros_like(self.W)
        self.grad_b = np.zeros_like(self.b)
        
        # Cache for backward
        self._cache: Optional[Tuple] = None
        
        logger.debug(f"LSTMCell: input={input_size}, hidden={hidden_size}")
    
    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Numerically stable sigmoid."""
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))
    
    def forward(
        self,
        x: np.ndarray,
        h_prev: np.ndarray,
        c_prev: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward pass for one time step.
        
        Args:
            x: Input (batch_size, input_size).
            h_prev: Previous hidden state (batch_size, hidden_size).
            c_prev: Previous cell state (batch_size, hidden_size).
        
        Returns:
            Tuple: (h_next, c_next)
        """
        x = np.asarray(x, dtype=np.float64)
        h_prev = np.asarray(h_prev, dtype=np.float64)
        c_prev = np.asarray(c_prev, dtype=np.float64)
        
        batch_size = x.shape[0]
        
        # Concatenate input and hidden state
        combined = np.concatenate([h_prev, x], axis=1)
        
        # Compute gates
        gates = combined @ self.W.T + self.b  # (batch, 4*hidden)
        
        i_gate = self._sigmoid(gates[:, :self.hidden_size])  # Input gate
        f_gate = self._sigmoid(gates[:, self.hidden_size:2*self.hidden_size])  # Forget gate
        g_gate = np.tanh(gates[:, 2*self.hidden_size:3*self.hidden_size])  # Cell candidate
        o_gate = self._sigmoid(gates[:, 3*self.hidden_size:])  # Output gate
        
        # Update cell state
        c_next = f_gate * c_prev + i_gate * g_gate
        
        # Update hidden state
        h_next = o_gate * np.tanh(c_next)
        
        # Cache for backward
        self._cache = (x, h_prev, c_prev, combined, gates, i_gate, f_gate, g_gate, o_gate, c_next)
        
        logger.debug(f"LSTMCell forward: input {x.shape}, hidden {h_next.shape}")
        return h_next, c_next
    
    def backward(
        self,
        grad_h: np.ndarray,
        grad_c: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Backward pass for one time step.
        
        Args:
            grad_h: Gradient w.r.t. hidden state.
            grad_c: Gradient w.r.t. cell state from next timestep.
        
        Returns:
            Tuple: (grad_x, grad_h_prev, grad_c_prev)
        """
        if self._cache is None:
            raise ValueError("Must call forward() before backward()")
        
        x, h_prev, c_prev, combined, gates, i_gate, f_gate, g_gate, o_gate, c_next = self._cache
        
        batch_size = x.shape[0]
        
        # Gradient through output
        tanh_c_next = np.tanh(c_next)
        grad_o = grad_h * tanh_c_next
        
        grad_c_next = grad_h * o_gate * (1 - tanh_c_next ** 2) + grad_c
        
        # Gradient through cell state
        grad_f = grad_c_next * c_prev
        grad_i = grad_c_next * g_gate
        grad_g = grad_c_next * i_gate
        grad_c_prev = grad_c_next * f_gate
        
        # Gradient through gates (sigmoid/tanh derivatives)
        grad_i_pre = grad_i * i_gate * (1 - i_gate)
        grad_f_pre = grad_f * f_gate * (1 - f_gate)
        grad_g_pre = grad_g * (1 - g_gate ** 2)
        grad_o_pre = grad_o * o_gate * (1 - o_gate)
        
        # Combine gate gradients
        grad_gates = np.concatenate([grad_i_pre, grad_f_pre, grad_g_pre, grad_o_pre], axis=1)
        
        # Gradients w.r.t. parameters
        self.grad_W += grad_gates.T @ combined
        self.grad_b += np.sum(grad_gates, axis=0)
        
        # Gradient w.r.t. combined input
        grad_combined = grad_gates @ self.W
        
        # Split gradient
        grad_h_prev = grad_combined[:, :self.hidden_size]
        grad_x = grad_combined[:, self.hidden_size:]
        
        return grad_x, grad_h_prev, grad_c_prev
    
    def get_parameters(self) -> Dict[str, np.ndarray]:
        """Get trainable parameters."""
        return {'W': self.W, 'b': self.b}
    
    def get_gradients(self) -> Dict[str, np.ndarray]:
        """Get parameter gradients."""
        return {'W': self.grad_W, 'b': self.grad_b}
    
    def zero_grad(self) -> None:
        """Zero out gradients."""
        self.grad_W = np.zeros_like(self.W)
        self.grad_b = np.zeros_like(self.b)


class GRUCell:
    """
    GRU (Gated Recurrent Unit) cell.
    
    Gates:
    - Reset gate: r_t = σ(W_r @ [h_{t-1}, x_t] + b_r)
    - Update gate: z_t = σ(W_z @ [h_{t-1}, x_t] + b_z)
    - New gate: n_t = tanh(W_n @ [r_t * h_{t-1}, x_t] + b_n)
    
    Update:
    - h_t = (1 - z_t) * n_t + z_t * h_{t-1}
    
    Example:
        >>> cell = GRUCell(input_size=100, hidden_size=256)
        >>> x = np.random.randn(32, 100)
        >>> h_prev = np.zeros((32, 256))
        >>> h_next = cell.forward(x, h_prev)
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int
    ):
        """
        Initialize GRU cell.
        
        Args:
            input_size: Input feature dimension.
            hidden_size: Hidden state dimension.
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        combined_size = input_size + hidden_size
        scale = np.sqrt(2.0 / (combined_size + hidden_size))
        
        # Gate weights (concatenated: r, z, n)
        self.W = np.random.randn(3 * hidden_size, combined_size) * scale
        self.b = np.zeros(3 * hidden_size)
        
        # Gradients
        self.grad_W = np.zeros_like(self.W)
        self.grad_b = np.zeros_like(self.b)
        
        # Cache for backward
        self._cache: Optional[Tuple] = None
        
        logger.debug(f"GRUCell: input={input_size}, hidden={hidden_size}")
    
    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Numerically stable sigmoid."""
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))
    
    def forward(
        self,
        x: np.ndarray,
        h_prev: np.ndarray
    ) -> np.ndarray:
        """
        Forward pass for one time step.
        
        Args:
            x: Input (batch_size, input_size).
            h_prev: Previous hidden state (batch_size, hidden_size).
        
        Returns:
            np.ndarray: New hidden state.
        """
        x = np.asarray(x, dtype=np.float64)
        h_prev = np.asarray(h_prev, dtype=np.float64)
        
        batch_size = x.shape[0]
        
        # Concatenate input and hidden state
        combined = np.concatenate([h_prev, x], axis=1)
        
        # Compute gates
        gates = combined @ self.W.T + self.b
        
        r_gate = self._sigmoid(gates[:, :self.hidden_size])  # Reset gate
        z_gate = self._sigmoid(gates[:, self.hidden_size:2*self.hidden_size])  # Update gate
        
        # Compute new gate with reset
        reset_hidden = r_gate * h_prev
        combined_reset = np.concatenate([reset_hidden, x], axis=1)
        
        # New gate weights are in the last third of W
        W_n = self.W[2*self.hidden_size:]
        b_n = self.b[2*self.hidden_size:]
        n_gate = np.tanh(combined_reset @ W_n.T + b_n)
        
        # Update hidden state
        h_next = (1 - z_gate) * n_gate + z_gate * h_prev
        
        # Cache for backward
        self._cache = (x, h_prev, combined, gates, r_gate, z_gate, n_gate, reset_hidden, combined_reset)
        
        logger.debug(f"GRUCell forward: input {x.shape}, hidden {h_next.shape}")
        return h_next
    
    def backward(self, grad_output: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Backward pass for one time step.
        
        Args:
            grad_output: Gradient w.r.t. output.
        
        Returns:
            Tuple: (grad_x, grad_h_prev)
        """
        if self._cache is None:
            raise ValueError("Must call forward() before backward()")
        
        x, h_prev, combined, gates, r_gate, z_gate, n_gate, reset_hidden, combined_reset = self._cache
        
        batch_size = x.shape[0]
        
        # Gradient through output
        grad_z = grad_output * (h_prev - n_gate)
        grad_n = grad_output * (1 - z_gate)
        grad_h_from_z = grad_output * z_gate
        
        # Gradient through new gate
        grad_n_pre = grad_n * (1 - n_gate ** 2)
        
        # Gradient for new gate weights
        W_n = self.W[2*self.hidden_size:]
        b_n = self.b[2*self.hidden_size:]
        
        grad_combined_reset = grad_n_pre @ W_n
        self.grad_W[2*self.hidden_size:] += grad_n_pre.T @ combined_reset
        self.grad_b[2*self.hidden_size:] += np.sum(grad_n_pre, axis=0)
        
        # Split gradient from combined_reset
        grad_reset_hidden = grad_combined_reset[:, :self.hidden_size]
        grad_x_from_n = grad_combined_reset[:, self.hidden_size:]
        
        # Gradient through reset gate
        grad_r = grad_reset_hidden * h_prev
        grad_h_from_r = grad_reset_hidden * r_gate
        
        # Gradient through reset gate pre-activation
        grad_r_pre = grad_r * r_gate * (1 - r_gate)
        
        # Gradient through update gate pre-activation
        grad_z_pre = grad_z * z_gate * (1 - z_gate)
        
        # Combine gate gradients
        grad_gates = np.concatenate([grad_r_pre, grad_z_pre, grad_n_pre * 0], axis=1)
        grad_gates[:, 2*self.hidden_size:] = 0  # Zero out n_gate gradient (already handled)
        
        # Gradients for r and z gate weights
        self.grad_W[:2*self.hidden_size] += grad_gates[:, :2*self.hidden_size].T @ combined
        self.grad_b[:2*self.hidden_size] += np.sum(grad_gates[:, :2*self.hidden_size], axis=0)
        
        # Gradient w.r.t. combined input
        grad_combined = grad_gates[:, :2*self.hidden_size] @ self.W[:2*self.hidden_size]
        
        # Split gradient
        grad_h_prev = grad_combined[:, :self.hidden_size] + grad_h_from_z + grad_h_from_r
        grad_x = grad_combined[:, self.hidden_size:] + grad_x_from_n
        
        return grad_x, grad_h_prev
    
    def get_parameters(self) -> Dict[str, np.ndarray]:
        """Get trainable parameters."""
        return {'W': self.W, 'b': self.b}
    
    def get_gradients(self) -> Dict[str, np.ndarray]:
        """Get parameter gradients."""
        return {'W': self.grad_W, 'b': self.grad_b}
    
    def zero_grad(self) -> None:
        """Zero out gradients."""
        self.grad_W = np.zeros_like(self.W)
        self.grad_b = np.zeros_like(self.b)


class RNN:
    """
    Multi-layer RNN.
    
    Example:
        >>> rnn = RNN(input_size=100, hidden_size=256, num_layers=2)
        >>> x = np.random.randn(32, 10, 100)  # (batch, seq_len, input_size)
        >>> output, hidden = rnn.forward(x)
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        nonlinearity: str = 'tanh',
        bidirectional: bool = False,
        dropout: float = 0.0
    ):
        """
        Initialize RNN.
        
        Args:
            input_size: Input feature dimension.
            hidden_size: Hidden state dimension.
            num_layers: Number of layers.
            nonlinearity: Activation function.
            bidirectional: Use bidirectional RNN.
            dropout: Dropout probability between layers.
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.nonlinearity = nonlinearity
        self.bidirectional = bidirectional
        self.dropout = dropout
        
        # Create cells
        self.cells = self._create_cells()
        
        # Dropout mask cache
        self._dropout_masks: List[np.ndarray] = []
        
        logger.info(f"RNN: input={input_size}, hidden={hidden_size}, "
                   f"layers={num_layers}, bidirectional={bidirectional}")
    
    def _create_cells(self) -> List[RNNCell]:
        """Create RNN cells for all layers."""
        cells = []
        
        num_directions = 2 if self.bidirectional else 1
        effective_hidden = self.hidden_size
        
        for layer in range(self.num_layers):
            layer_input_size = self.input_size if layer == 0 else effective_hidden * num_directions
            
            if self.bidirectional:
                # Forward cell
                cells.append(RNNCell(layer_input_size, self.hidden_size, self.nonlinearity))
                # Backward cell
                cells.append(RNNCell(layer_input_size, self.hidden_size, self.nonlinearity))
            else:
                cells.append(RNNCell(layer_input_size, self.hidden_size, self.nonlinearity))
        
        return cells
    
    def forward(
        self,
        x: np.ndarray,
        h0: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward pass through RNN.
        
        Args:
            x: Input (batch_size, seq_len, input_size).
            h0: Initial hidden state (num_layers * num_directions, batch_size, hidden_size).
        
        Returns:
            Tuple: (output, hidden)
                - output: (batch_size, seq_len, num_directions * hidden_size)
                - hidden: (num_layers * num_directions, batch_size, hidden_size)
        """
        x = np.asarray(x, dtype=np.float64)
        batch_size, seq_len, _ = x.shape
        
        num_directions = 2 if self.bidirectional else 1
        
        # Initialize hidden state
        if h0 is None:
            h0 = np.zeros((self.num_layers * num_directions, batch_size, self.hidden_size))
        
        # Process sequence
        outputs = []
        hidden_states = []
        
        cell_idx = 0
        for layer in range(self.num_layers):
            layer_outputs = []
            h = h0[cell_idx]
            
            if self.bidirectional:
                h_backward = h0[cell_idx + 1]
                
                # Forward pass
                fwd_outputs = []
                for t in range(seq_len):
                    h = self.cells[cell_idx].forward(x[:, t, :], h)
                    
                    # Apply dropout
                    if self.dropout > 0 and layer < self.num_layers - 1:
                        mask = (np.random.rand(*h.shape) > self.dropout).astype(np.float64)
                        h = h * mask / (1 - self.dropout)
                        if t == 0:
                            self._dropout_masks.append(mask)
                    
                    fwd_outputs.append(h)
                
                # Backward pass
                bwd_outputs = []
                for t in reversed(range(seq_len)):
                    h_backward = self.cells[cell_idx + 1].forward(x[:, t, :], h_backward)
                    bwd_outputs.append(h_backward)
                bwd_outputs.reverse()
                
                # Concatenate forward and backward
                for t in range(seq_len):
                    combined = np.concatenate([fwd_outputs[t], bwd_outputs[t]], axis=1)
                    layer_outputs.append(combined)
                
                hidden_states.append(h)
                hidden_states.append(h_backward)
                cell_idx += 2
            else:
                for t in range(seq_len):
                    h = self.cells[cell_idx].forward(x[:, t, :], h)
                    
                    # Apply dropout
                    if self.dropout > 0 and layer < self.num_layers - 1:
                        mask = (np.random.rand(*h.shape) > self.dropout).astype(np.float64)
                        h = h * mask / (1 - self.dropout)
                        if t == 0:
                            self._dropout_masks.append(mask)
                    
                    layer_outputs.append(h)
                
                hidden_states.append(h)
                cell_idx += 1
            
            # Update input for next layer
            if layer < self.num_layers - 1:
                x = np.stack(layer_outputs, axis=1)
        
        output = np.stack(layer_outputs, axis=1)
        hidden = np.stack(hidden_states, axis=0)
        
        logger.debug(f"RNN forward: input {x.shape}, output {output.shape}")
        return output, hidden
    
    def get_parameters(self) -> List[Dict[str, np.ndarray]]:
        """Get all parameters."""
        return [cell.get_parameters() for cell in self.cells]
    
    def get_gradients(self) -> List[Dict[str, np.ndarray]]:
        """Get all gradients."""
        return [cell.get_gradients() for cell in self.cells]
    
    def zero_grad(self) -> None:
        """Zero out all gradients."""
        for cell in self.cells:
            cell.zero_grad()


class LSTM(RNN):
    """
    Multi-layer LSTM.
    
    Example:
        >>> lstm = LSTM(input_size=100, hidden_size=256, num_layers=2)
        >>> x = np.random.randn(32, 10, 100)
        >>> output, (h_n, c_n) = lstm.forward(x)
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bidirectional: bool = False,
        dropout: float = 0.0
    ):
        """
        Initialize LSTM.
        
        Args:
            input_size: Input feature dimension.
            hidden_size: Hidden state dimension.
            num_layers: Number of layers.
            bidirectional: Use bidirectional LSTM.
            dropout: Dropout probability between layers.
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.dropout = dropout
        
        self.cells = self._create_lstm_cells()
        self._dropout_masks: List[np.ndarray] = []
        
        logger.info(f"LSTM: input={input_size}, hidden={hidden_size}, "
                   f"layers={num_layers}, bidirectional={bidirectional}")
    
    def _create_lstm_cells(self) -> List[LSTMCell]:
        """Create LSTM cells for all layers."""
        cells = []
        num_directions = 2 if self.bidirectional else 1
        
        for layer in range(self.num_layers):
            layer_input_size = self.input_size if layer == 0 else self.hidden_size * num_directions
            
            if self.bidirectional:
                cells.append(LSTMCell(layer_input_size, self.hidden_size))
                cells.append(LSTMCell(layer_input_size, self.hidden_size))
            else:
                cells.append(LSTMCell(layer_input_size, self.hidden_size))
        
        return cells
    
    def forward(
        self,
        x: np.ndarray,
        h0: Optional[np.ndarray] = None,
        c0: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Forward pass through LSTM.
        
        Args:
            x: Input (batch_size, seq_len, input_size).
            h0: Initial hidden state.
            c0: Initial cell state.
        
        Returns:
            Tuple: (output, (h_n, c_n))
        """
        x = np.asarray(x, dtype=np.float64)
        batch_size, seq_len, _ = x.shape
        
        num_directions = 2 if self.bidirectional else 1
        state_size = self.num_layers * num_directions
        
        # Initialize states
        if h0 is None:
            h0 = np.zeros((state_size, batch_size, self.hidden_size))
        if c0 is None:
            c0 = np.zeros((state_size, batch_size, self.hidden_size))
        
        outputs = []
        h_states = []
        c_states = []
        
        cell_idx = 0
        for layer in range(self.num_layers):
            layer_outputs = []
            h = h0[cell_idx].copy()
            c = c0[cell_idx].copy()
            
            if self.bidirectional:
                h_bwd = h0[cell_idx + 1].copy()
                c_bwd = c0[cell_idx + 1].copy()
                
                fwd_outputs = []
                for t in range(seq_len):
                    h, c = self.cells[cell_idx].forward(x[:, t, :], h, c)
                    if self.dropout > 0 and layer < self.num_layers - 1:
                        mask = (np.random.rand(*h.shape) > self.dropout).astype(np.float64)
                        h = h * mask / (1 - self.dropout)
                    fwd_outputs.append(h)
                
                bwd_outputs = []
                for t in reversed(range(seq_len)):
                    h_bwd, c_bwd = self.cells[cell_idx + 1].forward(x[:, t, :], h_bwd, c_bwd)
                    bwd_outputs.append(h_bwd)
                bwd_outputs.reverse()
                
                for t in range(seq_len):
                    combined = np.concatenate([fwd_outputs[t], bwd_outputs[t]], axis=1)
                    layer_outputs.append(combined)
                
                h_states.append(h)
                h_states.append(h_bwd)
                c_states.append(c)
                c_states.append(c_bwd)
                cell_idx += 2
            else:
                for t in range(seq_len):
                    h, c = self.cells[cell_idx].forward(x[:, t, :], h, c)
                    if self.dropout > 0 and layer < self.num_layers - 1:
                        mask = (np.random.rand(*h.shape) > self.dropout).astype(np.float64)
                        h = h * mask / (1 - self.dropout)
                    layer_outputs.append(h)
                
                h_states.append(h)
                c_states.append(c)
                cell_idx += 1
            
            if layer < self.num_layers - 1:
                x = np.stack(layer_outputs, axis=1)
        
        output = np.stack(layer_outputs, axis=1)
        h_n = np.stack(h_states, axis=0)
        c_n = np.stack(c_states, axis=0)
        
        return output, (h_n, c_n)
    
    def get_parameters(self) -> List[Dict[str, np.ndarray]]:
        """Get all parameters."""
        return [cell.get_parameters() for cell in self.cells]
    
    def get_gradients(self) -> List[Dict[str, np.ndarray]]:
        """Get all gradients."""
        return [cell.get_gradients() for cell in self.cells]
    
    def zero_grad(self) -> None:
        """Zero out all gradients."""
        for cell in self.cells:
            cell.zero_grad()


class GRU(RNN):
    """
    Multi-layer GRU.
    
    Example:
        >>> gru = GRU(input_size=100, hidden_size=256, num_layers=2)
        >>> x = np.random.randn(32, 10, 100)
        >>> output, h_n = gru.forward(x)
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bidirectional: bool = False,
        dropout: float = 0.0
    ):
        """
        Initialize GRU.
        
        Args:
            input_size: Input feature dimension.
            hidden_size: Hidden state dimension.
            num_layers: Number of layers.
            bidirectional: Use bidirectional GRU.
            dropout: Dropout probability between layers.
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.dropout = dropout
        
        self.cells = self._create_gru_cells()
        self._dropout_masks: List[np.ndarray] = []
        
        logger.info(f"GRU: input={input_size}, hidden={hidden_size}, "
                   f"layers={num_layers}, bidirectional={bidirectional}")
    
    def _create_gru_cells(self) -> List[GRUCell]:
        """Create GRU cells for all layers."""
        cells = []
        num_directions = 2 if self.bidirectional else 1
        
        for layer in range(self.num_layers):
            layer_input_size = self.input_size if layer == 0 else self.hidden_size * num_directions
            
            if self.bidirectional:
                cells.append(GRUCell(layer_input_size, self.hidden_size))
                cells.append(GRUCell(layer_input_size, self.hidden_size))
            else:
                cells.append(GRUCell(layer_input_size, self.hidden_size))
        
        return cells
    
    def forward(
        self,
        x: np.ndarray,
        h0: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward pass through GRU.
        
        Args:
            x: Input (batch_size, seq_len, input_size).
            h0: Initial hidden state.
        
        Returns:
            Tuple: (output, h_n)
        """
        x = np.asarray(x, dtype=np.float64)
        batch_size, seq_len, _ = x.shape
        
        num_directions = 2 if self.bidirectional else 1
        state_size = self.num_layers * num_directions
        
        if h0 is None:
            h0 = np.zeros((state_size, batch_size, self.hidden_size))
        
        outputs = []
        h_states = []
        
        cell_idx = 0
        for layer in range(self.num_layers):
            layer_outputs = []
            h = h0[cell_idx].copy()
            
            if self.bidirectional:
                h_bwd = h0[cell_idx + 1].copy()
                
                fwd_outputs = []
                for t in range(seq_len):
                    h = self.cells[cell_idx].forward(x[:, t, :], h)
                    if self.dropout > 0 and layer < self.num_layers - 1:
                        mask = (np.random.rand(*h.shape) > self.dropout).astype(np.float64)
                        h = h * mask / (1 - self.dropout)
                    fwd_outputs.append(h)
                
                bwd_outputs = []
                for t in reversed(range(seq_len)):
                    h_bwd = self.cells[cell_idx + 1].forward(x[:, t, :], h_bwd)
                    bwd_outputs.append(h_bwd)
                bwd_outputs.reverse()
                
                for t in range(seq_len):
                    combined = np.concatenate([fwd_outputs[t], bwd_outputs[t]], axis=1)
                    layer_outputs.append(combined)
                
                h_states.append(h)
                h_states.append(h_bwd)
                cell_idx += 2
            else:
                for t in range(seq_len):
                    h = self.cells[cell_idx].forward(x[:, t, :], h)
                    if self.dropout > 0 and layer < self.num_layers - 1:
                        mask = (np.random.rand(*h.shape) > self.dropout).astype(np.float64)
                        h = h * mask / (1 - self.dropout)
                    layer_outputs.append(h)
                
                h_states.append(h)
                cell_idx += 1
            
            if layer < self.num_layers - 1:
                x = np.stack(layer_outputs, axis=1)
        
        output = np.stack(layer_outputs, axis=1)
        h_n = np.stack(h_states, axis=0)
        
        return output, h_n
    
    def get_parameters(self) -> List[Dict[str, np.ndarray]]:
        """Get all parameters."""
        return [cell.get_parameters() for cell in self.cells]
    
    def get_gradients(self) -> List[Dict[str, np.ndarray]]:
        """Get all gradients."""
        return [cell.get_gradients() for cell in self.cells]
    
    def zero_grad(self) -> None:
        """Zero out all gradients."""
        for cell in self.cells:
            cell.zero_grad()


if __name__ == "__main__":
    # Example usage and demonstrations
    print("=" * 60)
    print("Sequence Models Module - Demonstration")
    print("=" * 60)
    
    np.random.seed(42)
    
    batch_size = 4
    seq_len = 10
    input_size = 32
    hidden_size = 64
    
    x = np.random.randn(batch_size, seq_len, input_size)
    
    # RNN Cell
    print("\n1. RNN Cell:")
    rnn_cell = RNNCell(input_size=input_size, hidden_size=hidden_size)
    h_prev = np.zeros((batch_size, hidden_size))
    h_next = rnn_cell.forward(x[:, 0, :], h_prev)
    print(f"   Input: {x[:, 0, :].shape}")
    print(f"   Output: {h_next.shape}")
    
    # LSTM Cell
    print("\n2. LSTM Cell:")
    lstm_cell = LSTMCell(input_size=input_size, hidden_size=hidden_size)
    h_prev = np.zeros((batch_size, hidden_size))
    c_prev = np.zeros((batch_size, hidden_size))
    h_next, c_next = lstm_cell.forward(x[:, 0, :], h_prev, c_prev)
    print(f"   Input: {x[:, 0, :].shape}")
    print(f"   Hidden: {h_next.shape}, Cell: {c_next.shape}")
    
    # GRU Cell
    print("\n3. GRU Cell:")
    gru_cell = GRUCell(input_size=input_size, hidden_size=hidden_size)
    h_prev = np.zeros((batch_size, hidden_size))
    h_next = gru_cell.forward(x[:, 0, :], h_prev)
    print(f"   Input: {x[:, 0, :].shape}")
    print(f"   Output: {h_next.shape}")
    
    # Full RNN
    print("\n4. Multi-layer RNN:")
    rnn = RNN(input_size=input_size, hidden_size=hidden_size, num_layers=2)
    output, hidden = rnn.forward(x)
    print(f"   Input: {x.shape}")
    print(f"   Output: {output.shape}")
    print(f"   Hidden: {hidden.shape}")
    
    # Full LSTM
    print("\n5. Multi-layer LSTM:")
    lstm = LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=2)
    output, (h_n, c_n) = lstm.forward(x)
    print(f"   Input: {x.shape}")
    print(f"   Output: {output.shape}")
    print(f"   Hidden: {h_n.shape}, Cell: {c_n.shape}")
    
    # Bidirectional LSTM
    print("\n6. Bidirectional LSTM:")
    bi_lstm = LSTM(input_size=input_size, hidden_size=hidden_size, 
                   num_layers=1, bidirectional=True)
    output, (h_n, c_n) = bi_lstm.forward(x)
    print(f"   Input: {x.shape}")
    print(f"   Output: {output.shape}")
    print(f"   Hidden: {h_n.shape}")
    
    # Full GRU
    print("\n7. Multi-layer GRU:")
    gru = GRU(input_size=input_size, hidden_size=hidden_size, num_layers=2)
    output, h_n = gru.forward(x)
    print(f"   Input: {x.shape}")
    print(f"   Output: {output.shape}")
    print(f"   Hidden: {h_n.shape}")
    
    print("\n" + "=" * 60)
