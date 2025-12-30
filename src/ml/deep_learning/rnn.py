"""
Recurrent Neural Networks Implementation

This module implements RNNs from scratch using NumPy,
including basic RNN, LSTM, and GRU architectures.
"""

import numpy as np
from typing import Tuple, List, Optional, Union
import matplotlib.pyplot as plt


class RNNCell:
    """
    Basic RNN Cell implementation.
    """
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        """
        Initialize RNN Cell.
        
        Args:
            input_size: Size of input vectors
            hidden_size: Size of hidden state
            output_size: Size of output vectors
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Initialize weights
        # Xavier initialization
        self.W_ih = np.random.randn(input_size, hidden_size) * np.sqrt(1.0 / input_size)
        self.W_hh = np.random.randn(hidden_size, hidden_size) * np.sqrt(1.0 / hidden_size)
        self.W_ho = np.random.randn(hidden_size, output_size) * np.sqrt(1.0 / hidden_size)
        
        # Initialize biases
        self.b_h = np.zeros((1, hidden_size))
        self.b_o = np.zeros((1, output_size))
    
    def forward(self, input_t: np.ndarray, hidden_prev: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward pass for a single time step.
        
        Args:
            input_t: Input at time t of shape (batch_size, input_size)
            hidden_prev: Hidden state at time t-1 of shape (batch_size, hidden_size)
            
        Returns:
            Tuple of (output_t, hidden_t)
        """
        # Compute new hidden state
        hidden_t = np.tanh(np.dot(input_t, self.W_ih) + np.dot(hidden_prev, self.W_hh) + self.b_h)
        
        # Compute output
        output_t = np.dot(hidden_t, self.W_ho) + self.b_o
        
        return output_t, hidden_t


class LSTMCell:
    """
    LSTM Cell implementation.
    """
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        """
        Initialize LSTM Cell.
        
        Args:
            input_size: Size of input vectors
            hidden_size: Size of hidden state
            output_size: Size of output vectors
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Initialize weights for input gate
        self.W_ii = np.random.randn(input_size, hidden_size) * np.sqrt(1.0 / input_size)
        self.W_hi = np.random.randn(hidden_size, hidden_size) * np.sqrt(1.0 / hidden_size)
        self.W_ci = np.random.randn(hidden_size, hidden_size) * np.sqrt(1.0 / hidden_size)
        self.b_i = np.zeros((1, hidden_size))
        
        # Initialize weights for forget gate
        self.W_if = np.random.randn(input_size, hidden_size) * np.sqrt(1.0 / input_size)
        self.W_hf = np.random.randn(hidden_size, hidden_size) * np.sqrt(1.0 / hidden_size)
        self.W_cf = np.random.randn(hidden_size, hidden_size) * np.sqrt(1.0 / hidden_size)
        self.b_f = np.zeros((1, hidden_size))
        
        # Initialize weights for output gate
        self.W_io = np.random.randn(input_size, hidden_size) * np.sqrt(1.0 / input_size)
        self.W_ho = np.random.randn(hidden_size, hidden_size) * np.sqrt(1.0 / hidden_size)
        self.W_co = np.random.randn(hidden_size, hidden_size) * np.sqrt(1.0 / hidden_size)
        self.b_o = np.zeros((1, hidden_size))
        
        # Initialize weights for cell state
        self.W_ig = np.random.randn(input_size, hidden_size) * np.sqrt(1.0 / input_size)
        self.W_hg = np.random.randn(hidden_size, hidden_size) * np.sqrt(1.0 / hidden_size)
        self.b_g = np.zeros((1, hidden_size))
        
        # Initialize weights for output
        self.W_out = np.random.randn(hidden_size, output_size) * np.sqrt(1.0 / hidden_size)
        self.b_out = np.zeros((1, output_size))
    
    def forward(self, input_t: np.ndarray, hidden_prev: np.ndarray, cell_prev: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Forward pass for a single time step.
        
        Args:
            input_t: Input at time t of shape (batch_size, input_size)
            hidden_prev: Hidden state at time t-1 of shape (batch_size, hidden_size)
            cell_prev: Cell state at time t-1 of shape (batch_size, hidden_size)
            
        Returns:
            Tuple of (output_t, hidden_t, cell_t)
        """
        # Input gate
        i_t = self._sigmoid(
            np.dot(input_t, self.W_ii) + 
            np.dot(hidden_prev, self.W_hi) + 
            np.dot(cell_prev, self.W_ci) + 
            self.b_i
        )
        
        # Forget gate
        f_t = self._sigmoid(
            np.dot(input_t, self.W_if) + 
            np.dot(hidden_prev, self.W_hf) + 
            np.dot(cell_prev, self.W_cf) + 
            self.b_f
        )
        
        # Output gate
        o_t = self._sigmoid(
            np.dot(input_t, self.W_io) + 
            np.dot(hidden_prev, self.W_ho) + 
            np.dot(cell_prev, self.W_co) + 
            self.b_o
        )
        
        # Gate input
        g_t = np.tanh(
            np.dot(input_t, self.W_ig) + 
            np.dot(hidden_prev, self.W_hg) + 
            self.b_g
        )
        
        # New cell state
        cell_t = f_t * cell_prev + i_t * g_t
        
        # New hidden state
        hidden_t = o_t * np.tanh(cell_t)
        
        # Output
        output_t = np.dot(hidden_t, self.W_out) + self.b_out
        
        return output_t, hidden_t, cell_t
    
    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Sigmoid activation function."""
        # Clip x to prevent overflow
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))


class GRUCell:
    """
    GRU Cell implementation.
    """
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        """
        Initialize GRU Cell.
        
        Args:
            input_size: Size of input vectors
            hidden_size: Size of hidden state
            output_size: Size of output vectors
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Update gate weights
        self.W_iz = np.random.randn(input_size, hidden_size) * np.sqrt(1.0 / input_size)
        self.W_hz = np.random.randn(hidden_size, hidden_size) * np.sqrt(1.0 / hidden_size)
        self.b_z = np.zeros((1, hidden_size))
        
        # Reset gate weights
        self.W_ir = np.random.randn(input_size, hidden_size) * np.sqrt(1.0 / input_size)
        self.W_hr = np.random.randn(hidden_size, hidden_size) * np.sqrt(1.0 / hidden_size)
        self.b_r = np.zeros((1, hidden_size))
        
        # New memory weights
        self.W_in = np.random.randn(input_size, hidden_size) * np.sqrt(1.0 / input_size)
        self.W_hn = np.random.randn(hidden_size, hidden_size) * np.sqrt(1.0 / hidden_size)
        self.b_n = np.zeros((1, hidden_size))
        
        # Output weights
        self.W_out = np.random.randn(hidden_size, output_size) * np.sqrt(1.0 / hidden_size)
        self.b_out = np.zeros((1, output_size))
    
    def forward(self, input_t: np.ndarray, hidden_prev: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward pass for a single time step.
        
        Args:
            input_t: Input at time t of shape (batch_size, input_size)
            hidden_prev: Hidden state at time t-1 of shape (batch_size, hidden_size)
            
        Returns:
            Tuple of (output_t, hidden_t)
        """
        # Update gate
        z_t = self._sigmoid(
            np.dot(input_t, self.W_iz) + 
            np.dot(hidden_prev, self.W_hz) + 
            self.b_z
        )
        
        # Reset gate
        r_t = self._sigmoid(
            np.dot(input_t, self.W_ir) + 
            np.dot(hidden_prev, self.W_hr) + 
            self.b_r
        )
        
        # New memory content
        n_t = np.tanh(
            np.dot(input_t, self.W_in) + 
            r_t * np.dot(hidden_prev, self.W_hn) + 
            self.b_n
        )
        
        # New hidden state
        hidden_t = (1 - z_t) * hidden_prev + z_t * n_t
        
        # Output
        output_t = np.dot(hidden_t, self.W_out) + self.b_out
        
        return output_t, hidden_t
    
    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Sigmoid activation function."""
        # Clip x to prevent overflow
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))


class RNN:
    """
    Recurrent Neural Network implementation.
    """
    
    def __init__(self, cell_type: str, input_size: int, hidden_size: int, output_size: int, num_layers: int = 1):
        """
        Initialize RNN.
        
        Args:
            cell_type: Type of RNN cell ('rnn', 'lstm', 'gru')
            input_size: Size of input vectors
            hidden_size: Size of hidden state
            output_size: Size of output vectors
            num_layers: Number of RNN layers
        """
        self.cell_type = cell_type
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        
        # Create RNN cells
        self.cells = []
        for i in range(num_layers):
            input_sz = input_size if i == 0 else hidden_size
            if cell_type == 'rnn':
                cell = RNNCell(input_sz, hidden_size, output_size if i == num_layers - 1 else hidden_size)
            elif cell_type == 'lstm':
                cell = LSTMCell(input_sz, hidden_size, output_size if i == num_layers - 1 else hidden_size)
            elif cell_type == 'gru':
                cell = GRUCell(input_sz, hidden_size, output_size if i == num_layers - 1 else hidden_size)
            else:
                raise ValueError(f"Unknown cell type: {cell_type}")
            self.cells.append(cell)
    
    def forward(self, inputs: np.ndarray) -> Tuple[np.ndarray, List]:
        """
        Forward pass through the RNN.
        
        Args:
            inputs: Input sequence of shape (batch_size, sequence_length, input_size)
            
        Returns:
            Tuple of (outputs, hidden_states)
        """
        batch_size, seq_len, _ = inputs.shape
        
        # Initialize hidden states
        if self.cell_type == 'lstm':
            hidden_states = [(np.zeros((batch_size, self.hidden_size)), 
                             np.zeros((batch_size, self.hidden_size))) for _ in range(self.num_layers)]
        else:
            hidden_states = [np.zeros((batch_size, self.hidden_size)) for _ in range(self.num_layers)]
        
        outputs = []
        
        # Process each time step
        for t in range(seq_len):
            input_t = inputs[:, t, :]  # Shape: (batch_size, input_size)
            
            # Process through each layer
            for layer_idx, cell in enumerate(self.cells):
                if self.cell_type == 'lstm':
                    if layer_idx == 0:
                        # First layer
                        output_t, hidden_t, cell_t = cell.forward(input_t, hidden_states[layer_idx][0], hidden_states[layer_idx][1])
                        hidden_states[layer_idx] = (hidden_t, cell_t)
                    else:
                        # Subsequent layers
                        output_t, hidden_t, cell_t = cell.forward(
                            hidden_states[layer_idx-1][0],  # Use hidden state from previous layer
                            hidden_states[layer_idx][0], 
                            hidden_states[layer_idx][1]
                        )
                        hidden_states[layer_idx] = (hidden_t, cell_t)
                else:  # RNN or GRU
                    if layer_idx == 0:
                        # First layer
                        output_t, hidden_t = cell.forward(input_t, hidden_states[layer_idx])
                        hidden_states[layer_idx] = hidden_t
                    else:
                        # Subsequent layers
                        output_t, hidden_t = cell.forward(
                            hidden_states[layer_idx-1],  # Use hidden state from previous layer
                            hidden_states[layer_idx]
                        )
                        hidden_states[layer_idx] = hidden_t
                
                # For all layers except the last, use the hidden state as input to the next layer
                if layer_idx < self.num_layers - 1:
                    input_t = hidden_t
        
            outputs.append(output_t)
        
        # Stack outputs along the sequence dimension
        outputs = np.stack(outputs, axis=1)  # Shape: (batch_size, seq_len, output_size)
        
        return outputs, hidden_states
    
    def predict(self, inputs: np.ndarray) -> np.ndarray:
        """
        Make predictions on input sequence.
        
        Args:
            inputs: Input sequence of shape (batch_size, sequence_length, input_size)
            
        Returns:
            Predictions of shape (batch_size, sequence_length, output_size)
        """
        outputs, _ = self.forward(inputs)
        return outputs


def sequence_loss(outputs: np.ndarray, targets: np.ndarray) -> float:
    """
    Calculate sequence loss (mean squared error across sequence).
    
    Args:
        outputs: Model outputs of shape (batch_size, sequence_length, output_size)
        targets: True targets of shape (batch_size, sequence_length, output_size)
        
    Returns:
        Mean squared error
    """
    return np.mean((outputs - targets) ** 2)


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate accuracy score."""
    # For sequence data, we might want to compare element-wise
    return np.mean(y_true == y_pred)


def cross_entropy_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate cross-entropy loss.
    
    Args:
        y_true: True labels (one-hot encoded)
        y_pred: Predicted probabilities
        
    Returns:
        Cross-entropy loss
    """
    # Clip predictions to prevent log(0)
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=-1))


def create_lstm_classifier(
    input_size: int,
    hidden_size: int,
    num_classes: int,
    num_layers: int = 1
) -> RNN:
    """
    Create an LSTM-based classifier.
    
    Args:
        input_size: Size of input vectors
        hidden_size: Size of hidden state
        num_classes: Number of output classes
        num_layers: Number of LSTM layers
        
    Returns:
        RNN object with LSTM cells
    """
    return RNN(
        cell_type='lstm',
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=num_classes,
        num_layers=num_layers
    )


def create_gru_regressor(
    input_size: int,
    hidden_size: int,
    output_size: int,
    num_layers: int = 1
) -> RNN:
    """
    Create a GRU-based regressor.
    
    Args:
        input_size: Size of input vectors
        hidden_size: Size of hidden state
        output_size: Size of output vectors
        num_layers: Number of GRU layers
        
    Returns:
        RNN object with GRU cells
    """
    return RNN(
        cell_type='gru',
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        num_layers=num_layers
    )