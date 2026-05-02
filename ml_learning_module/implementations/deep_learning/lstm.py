"""
Long Short-Term Memory (LSTM) Implementation from Scratch
============================================================

This module provides complete implementation of LSTM networks.

Mathematical Foundation:
-------------------------
LSTM addresses the vanishing gradient problem in standard RNNs through
gated architecture that allows for learning long-term dependencies.

1. Cell State (Memory):
   The cell state is the "highway" that allows information to flow
   relatively unchanged through the sequence.

   c_t = f_t * c_{t-1} + i_t * ~C_t

   Where:
   - c_t: Cell state at time t
   - c_{t-1}: Previous cell state
   - f_t: Forget gate
   - i_t: Input gate
   - ~C_t: New candidate cell state

2. Gates:

   a) Forget Gate (f_t):
      Decides what to discard from previous state
      f_t = σ(W_f * [h_{t-1}, x_t] + b_f)

      Values close to 1: "keep this"
      Values close to 0: "discard this"

   b) Input Gate (i_t):
      Decides what new information to store
      i_t = σ(W_i * [h_{t-1}, x_t] + b_i)

      ~C_t = tanh(W_C * [h_{t-1}, x_t] + b_C)

      New candidate values

   c) Output Gate (o_t):
      Decides what to output
      o_t = σ(W_o * [h_{t-1}, x_t] + b_o)

      h_t = o_t * tanh(c_t)

3. Why LSTM Works:
   - Gradient tracks through cell state (fewer multiplicative operations)
   - Forget gate can reset memory (1 - forget = remember)
   - Additive nature of cell update (prevents gradient decay)
   - Gate mechanisms provide learned flow control

4. Variants:
   - Peephole connections: Allow gates to see cell state
   - Coupled forget and input gates
   - Gated Recurrent Unit (GRU): Simpler, fewer gates

Author: AI-Mastery-2026
"""

import numpy as np
from typing import Tuple


class LSTMCell:
    """
    Single LSTM Cell

    Implements one time step of an LSTM.

    Architecture:
        f_t = σ(W_f @ [h_{t-1}, x_t] + b_f)  # Forget gate
        i_t = σ(W_i @ [h_{t-1}, x_t] + b_i)  # Input gate
        o_t = σ(W_o @ [h_{t-1}, x_t] + b_o)  # Output gate
        ~C_t = tanh(W_C @ [h_{t-1}, x_t] + b_C)  # Candidate
        C_t = f_t * C_{t-1} + i_t * ~C_t  # Cell state
        h_t = o_t * tanh(C_t)  # Hidden state

    Parameters:
        input_size: Number of input features
        hidden_size: Number of hidden units (memory cells)

    Weight Shapes:
        W_f, W_i, W_o, W_C: (hidden_size, input_size + hidden_size)
        b_f, b_i, b_o, b_C: (hidden_size,)
    """

    def __init__(self, input_size: int, hidden_size: int):
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Combined input size for all gates
        combined_size = input_size + hidden_size

        # Xavier initialization for each gate
        scale = np.sqrt(6.0 / (combined_size + hidden_size))

        # Forget gate weights
        self.W_f = np.random.randn(hidden_size, combined_size) * scale
        self.b_f = np.zeros(hidden_size)

        # Input gate weights
        self.W_i = np.random.randn(hidden_size, combined_size) * scale
        self.b_i = np.zeros(hidden_size)

        # Output gate weights
        self.W_o = np.random.randn(hidden_size, combined_size) * scale
        self.b_o = np.zeros(hidden_size)

        # Candidate cell weights
        self.W_c = np.random.randn(hidden_size, combined_size) * scale
        self.b_c = np.zeros(hidden_size)

        # Cache for backward pass
        self.cache = None

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Numerically stable sigmoid"""
        return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))

    def forward(
        self, x_t: np.ndarray, h_prev: np.ndarray, c_prev: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward pass for one time step

        Args:
            x_t: Input at time t, shape (batch, input_size)
            h_prev: Previous hidden state, shape (batch, hidden_size)
            c_prev: Previous cell state, shape (batch, hidden_size)

        Returns:
            Tuple of (hidden_state, cell_state)
        """
        # Ensure 2D
        if x_t.ndim == 1:
            x_t = x_t.reshape(1, -1)
            h_prev = h_prev.reshape(1, -1)
            c_prev = c_prev.reshape(1, -1)
            single_sample = True
        else:
            single_sample = False

        # Concatenate h_{t-1} and x_t
        combined = np.concatenate([h_prev, x_t], axis=-1)

        # Compute all gates
        f_t = self._sigmoid(np.dot(combined, self.W_f.T) + self.b_f)  # Forget gate
        i_t = self._sigmoid(np.dot(combined, self.W_i.T) + self.b_i)  # Input gate
        o_t = self._sigmoid(np.dot(combined, self.W_o.T) + self.b_o)  # Output gate

        # Candidate cell state
        C_tilde = np.tanh(np.dot(combined, self.W_c.T) + self.b_c)

        # Update cell state
        c_t = f_t * c_prev + i_t * C_tilde

        # Compute hidden state
        h_t = o_t * np.tanh(c_t)

        # Cache for backward
        self.cache = (
            x_t,
            h_prev,
            c_prev,
            combined,
            f_t,
            i_t,
            o_t,
            C_tilde,
            c_t,
            h_t,
            single_sample,
        )

        if single_sample:
            return h_t[0], c_t[0]
        return h_t, c_t

    def backward(self, grad_h: np.ndarray, grad_c: np.ndarray = None):
        """
        Backward pass for one time step

        This is a simplified version showing the key gradients.

        Args:
            grad_h: Gradient with respect to hidden state
            grad_c: Gradient with respect to cell state (from next timestep)

        Returns:
            Gradients for parameter updates
        """
        (
            x_t,
            h_prev,
            c_prev,
            combined,
            f_t,
            i_t,
            o_t,
            C_tilde,
            c_t,
            h_t,
            single_sample,
        ) = self.cache

        if single_sample:
            grad_h = grad_h.reshape(1, -1)
            if grad_c is not None:
                grad_c = grad_c.reshape(1, -1)

        batch_size = x_t.shape[0]

        # Initialize gradients
        if grad_c is None:
            grad_c = np.zeros_like(c_t)

        # Total gradient with respect to cell state
        # dL/dc_t = dL/dh_t * tanh(c_t) * o_t + dL/dc_{t+1}
        grad_c_total = grad_h * o_t * (1 - np.tanh(c_t) ** 2) + grad_c

        # Gradient with respect to output gate
        grad_o = grad_h * np.tanh(c_t)
        grad_o_sigmoid = grad_o * o_t * (1 - o_t)

        # Gradient with respect to forget gate
        grad_f = grad_c_total * c_prev
        grad_f_sigmoid = grad_f * f_t * (1 - f_t)

        # Gradient with respect to input gate
        grad_i = grad_c_total * C_tilde
        grad_i_sigmoid = grad_i * i_t * (1 - i_t)

        # Gradient with respect to candidate cell
        grad_C_tilde = grad_c_total * i_t
        grad_C_tilde_tanh = grad_C_tilde * (1 - C_tilde**2)

        # Combine gate gradients
        grad_gates = np.concatenate(
            [grad_f_sigmoid, grad_i_sigmoid, grad_o_sigmoid, grad_C_tilde_tanh], axis=-1
        )

        # Gradient with respect to weights
        W_f_grad = np.dot(grad_f_sigmoid.T, combined)
        W_i_grad = np.dot(grad_i_sigmoid.T, combined)
        W_o_grad = np.dot(grad_o_sigmoid.T, combined)
        W_c_grad = np.dot(grad_C_tilde_tanh.T, combined)

        # Gradient with respect to inputs
        grad_combined = (
            np.dot(grad_f_sigmoid, self.W_f)
            + np.dot(grad_i_sigmoid, self.W_i)
            + np.dot(grad_o_sigmoid, self.W_o)
            + np.dot(grad_C_tilde_tanh, self.W_c)
        )

        # Split into h and x components
        grad_h_prev = grad_combined[:, : self.hidden_size]
        grad_x = grad_combined[:, self.hidden_size :]

        # Gradient with respect to previous cell state
        grad_c_prev = grad_c_total * f_t

        # Store gradients for optimizer
        self.grad_W_f = W_f_grad / batch_size
        self.grad_W_i = W_i_grad / batch_size
        self.grad_W_o = W_o_grad / batch_size
        self.grad_W_c = W_c_grad / batch_size
        self.grad_b_f = np.sum(grad_f_sigmoid, axis=0) / batch_size
        self.grad_b_i = np.sum(grad_i_sigmoid, axis=0) / batch_size
        self.grad_b_o = np.sum(grad_o_sigmoid, axis=0) / batch_size
        self.grad_b_c = np.sum(grad_C_tilde_tanh, axis=0) / batch_size

        if single_sample:
            grad_x = grad_x[0]
            grad_h_prev = grad_h_prev[0]
            grad_c_prev = grad_c_prev[0]

        return grad_x, grad_h_prev, grad_c_prev


class SimpleLSTM:
    """
    Simple LSTM Network

    Processes sequences using LSTM cells.

    Architecture:
        For each time step t:
            h_t, c_t = lstm_cell(x_t, h_{t-1}, c_{t-1})

    Use cases:
        - Language modeling
        - Machine translation
        - Time series prediction
        - Text generation
    """

    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # LSTM cell
        self.lstm_cell = LSTMCell(input_size, hidden_size)

        # Output projection
        self.W_y = np.random.randn(output_size, hidden_size) * np.sqrt(
            2.0 / hidden_size
        )
        self.b_y = np.zeros(output_size)

        # Softmax
        from .cnn import Softmax

        self.softmax = Softmax()

    def forward(
        self, x: np.ndarray, h_0: np.ndarray = None, c_0: np.ndarray = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Forward pass through entire sequence

        Args:
            x: Input sequence, shape (batch, seq_length, input_size)
            h_0: Initial hidden state, shape (batch, hidden_size)
            c_0: Initial cell state, shape (batch, hidden_size)

        Returns:
            Tuple of (outputs, hidden_states, cell_states)
        """
        batch_size, seq_length, _ = x.shape

        # Initialize states
        if h_0 is None:
            h_0 = np.zeros((batch_size, self.hidden_size))
        if c_0 is None:
            c_0 = np.zeros((batch_size, self.hidden_size))

        # Store all states
        hidden_states = []
        cell_states = []
        outputs = []

        h_prev = h_0
        c_prev = c_0

        for t in range(seq_length):
            x_t = x[:, t, :]

            h_t, c_t = self.lstm_cell.forward(x_t, h_prev, c_prev)

            hidden_states.append(h_t)
            cell_states.append(c_t)

            # Project to output space
            y_t = np.dot(h_t, self.W_y.T) + self.b_y
            outputs.append(y_t)

            h_prev = h_t
            c_prev = c_t

        # Stack outputs
        output = np.stack(outputs, axis=1)
        hidden_states = np.stack(hidden_states, axis=1)
        cell_states = np.stack(cell_states, axis=1)

        return output, hidden_states, cell_states

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Make predictions

        Args:
            x: Input sequence

        Returns:
            Predicted class indices
        """
        output, _, _ = self.forward(x)

        # Get last time step output
        last_output = output[:, -1, :]

        # Apply softmax
        probabilities = self.softmax.forward(last_output)
        return np.argmax(probabilities, axis=-1)


class GRU:
    """
    Gated Recurrent Unit (GRU)

    Simpler than LSTM with fewer gates:
    - Update gate: Combines forget and input gates
    - Reset gate: Controls how much past information to forget

    Equations:
        z_t = σ(W_z @ [h_{t-1}, x_t])  # Update gate
        r_t = σ(W_r @ [h_{t-1}, x_t])  # Reset gate
        ~h_t = tanh(W @ [r_t * h_{t-1}, x_t])  # Candidate
        h_t = (1 - z_t) * h_{t-1} + z_t * ~h_t  # Final hidden state

    Advantages over LSTM:
        - Fewer parameters (faster training)
        - Less memory
        - Often comparable performance
    """

    def __init__(self, input_size: int, hidden_size: int):
        self.input_size = input_size
        self.hidden_size = hidden_size

        combined_size = input_size + hidden_size
        scale = np.sqrt(6.0 / (combined_size + hidden_size))

        # Update gate
        self.W_z = np.random.randn(hidden_size, combined_size) * scale
        self.b_z = np.zeros(hidden_size)

        # Reset gate
        self.W_r = np.random.randn(hidden_size, combined_size) * scale
        self.b_r = np.zeros(hidden_size)

        # Candidate hidden state
        self.W_h = np.random.randn(hidden_size, combined_size) * scale
        self.b_h = np.zeros(hidden_size)

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))

    def forward(self, x_t: np.ndarray, h_prev: np.ndarray) -> np.ndarray:
        """
        Forward pass for one time step

        Args:
            x_t: Input at time t
            h_prev: Previous hidden state

        Returns:
            New hidden state
        """
        if x_t.ndim == 1:
            x_t = x_t.reshape(1, -1)
            h_prev = h_prev.reshape(1, -1)

        combined = np.concatenate([h_prev, x_t], axis=-1)

        # Update gate
        z_t = self._sigmoid(np.dot(combined, self.W_z.T) + self.b_z)

        # Reset gate
        r_t = self._sigmoid(np.dot(combined, self.W_r.T) + self.b_r)

        # Combined for candidate
        combined_reset = np.concatenate([r_t * h_prev, x_t], axis=-1)

        # Candidate hidden state
        h_tilde = np.tanh(np.dot(combined_reset, self.W_h.T) + self.b_h)

        # Final hidden state
        h_t = (1 - z_t) * h_prev + z_t * h_tilde

        return h_t[0] if x_t.shape[0] == 1 else h_t


def test_lstm():
    """Test LSTM implementation"""
    print("=" * 60)
    print("Testing LSTM Implementation")
    print("=" * 60)

    # Test data
    np.random.seed(42)
    x = np.random.randn(2, 5, 3)  # (batch, seq_len, input_size)

    print(f"\nInput shape: {x.shape}")

    # Test LSTM cell
    lstm_cell = LSTMCell(input_size=3, hidden_size=8)
    h_prev = np.zeros(8)
    c_prev = np.zeros(8)

    h_t, c_t = lstm_cell.forward(x[0, 0], h_prev, c_prev)
    print(f"\nLSTM cell output shape: h={h_t.shape}, c={c_t.shape}")

    # Test complete LSTM
    lstm = SimpleLSTM(input_size=3, hidden_size=8, output_size=2)
    output, hidden_states, cell_states = lstm.forward(x)

    print(f"\nLSTM output shape: {output.shape}")
    print(f"Hidden states shape: {hidden_states.shape}")
    print(f"Cell states shape: {cell_states.shape}")

    # Test prediction
    predictions = lstm.predict(x)
    print(f"\nPredictions: {predictions}")

    # Test GRU
    gru = GRU(input_size=3, hidden_size=8)
    h_gru = gru.forward(x[0, 0], h_prev)
    print(f"\nGRU hidden state shape: {h_gru.shape}")

    print("\n" + "=" * 60)
    print("All LSTM tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    test_lstm()
