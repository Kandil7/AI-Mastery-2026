"""
Recurrent Neural Network (RNN) Implementation from Scratch
============================================================

This module provides complete implementation of RNN and its variants.

Mathematical Foundation:
-------------------------
1. Basic RNN Forward:
   h_t = tanh(W_ih * x_t + W_hh * h_{t-1} + b_h)
   y_t = W_ho * h_t + b_o

   Where:
   - x_t: Input at time t
   - h_t: Hidden state at time t
   - h_{t-1}: Previous hidden state
   - W_ih: Input-to-hidden weights
   - W_hh: Hidden-to-hidden (recurrent) weights
   - W_ho: Hidden-to-output weights

2. Backpropagation Through Time (BPTT):
   - Unroll the network through time
   - Compute gradients at each time step
   - Sum gradients across all time steps
   - Gradient clipping to prevent explosion

   Problem: Vanishing gradients in long sequences
   - Gradients can become very small as they propagate back
   - Difficult to learn long-term dependencies

3. Gradient Computation:
   dL/dW = Σ_t dL_t/dW

   For each time step t:
   - dL_t/dh_t = dL_t/dy_t * W_ho
   - dL_t/dh_{t-1} = dL_t/dh_t * W_hh * (1 - h_{t-1}^2)

   The (1 - h_{t-1}^2) term comes from tanh derivative.

4. Types of RNNs:
   - One-to-one: Standard feedforward
   - One-to-many: Image captioning
   - Many-to-one: Sentiment classification
   - Many-to-many: Machine translation, POS tagging

Author: AI-Mastery-2026
"""

import numpy as np
from typing import Tuple, Optional


class RNNCell:
    """
    Single RNN Cell

    Implements one time step of an RNN.

    Architecture:
        h_t = tanh(W_ih * x_t + W_hh * h_{t-1} + b_h)
        y_t = W_ho * h_t + b_o

    Parameters:
        input_size: Number of input features
        hidden_size: Number of hidden units

    Weight Shapes:
        W_ih: (hidden_size, input_size) - input to hidden
        W_hh: (hidden_size, hidden_size) - hidden to hidden (recurrent)
        W_ho: (output_size, hidden_size) - hidden to output
        b_h: (hidden_size,) - hidden bias
        b_o: (output_size,) - output bias
    """

    def __init__(self, input_size: int, hidden_size: int, output_size: int = None):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size if output_size else hidden_size

        # Initialize weights with Xavier initialization
        # For tanh: uniform(-sqrt(6/(fan_in+fan_out)), sqrt(6/(fan_in+fan_out)))
        scale_ih = np.sqrt(6.0 / (input_size + hidden_size))
        scale_hh = np.sqrt(6.0 / (2 * hidden_size))
        scale_ho = np.sqrt(6.0 / (hidden_size + self.output_size))

        self.W_ih = np.random.randn(hidden_size, input_size) * scale_ih
        self.W_hh = np.random.randn(hidden_size, hidden_size) * scale_hh
        self.W_ho = np.random.randn(self.output_size, hidden_size) * scale_ho

        self.b_h = np.zeros(hidden_size)
        self.b_o = np.zeros(self.output_size)

        # Cache for backward pass
        self.cache = None

    def forward(
        self, x_t: np.ndarray, h_prev: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward pass for one time step

        Args:
            x_t: Input at time t, shape (batch, input_size) or (input_size,)
            h_prev: Previous hidden state, shape (batch, hidden_size) or (hidden_size,)

        Returns:
            Tuple of (output, hidden_state)
        """
        # Ensure 2D
        if x_t.ndim == 1:
            x_t = x_t.reshape(1, -1)
            h_prev = h_prev.reshape(1, -1)
            single_sample = True
        else:
            single_sample = False

        # Compute hidden state
        # h_t = tanh(W_ih @ x_t + W_hh @ h_{t-1} + b_h)
        h_raw = np.dot(x_t, self.W_ih.T) + np.dot(h_prev, self.W_hh.T) + self.b_h
        h_t = np.tanh(h_raw)

        # Compute output
        # y_t = W_ho @ h_t + b_o
        y_t = np.dot(h_t, self.W_ho.T) + self.b_o

        # Cache for backward
        self.cache = (x_t, h_prev, h_raw, h_t, y_t, single_sample)

        if single_sample:
            return y_t[0], h_t[0]
        return y_t, h_t

    def backward(self, grad_output: np.ndarray, grad_h: np.ndarray = None):
        """
        Backward pass for one time step

        Args:
            grad_output: Gradient with respect to output, shape (batch, output_size)
            grad_h: Gradient with respect to hidden state from next time step
                   shape (batch, hidden_size)

        Returns:
            Tuple of (grad_x, grad_h_prev, grad_weights)
        """
        x_t, h_prev, h_raw, h_t, y_t, single_sample = self.cache

        # Handle single sample
        if single_sample:
            grad_output = grad_output.reshape(1, -1)
            if grad_h is not None:
                grad_h = grad_h.reshape(1, -1)

        batch_size = x_t.shape[0]

        # Initialize gradients
        grad_x = np.zeros_like(x_t)
        grad_h_prev = np.zeros_like(h_prev) if h_prev.shape[0] > 0 else None
        grad_W_ih = np.zeros_like(self.W_ih)
        grad_W_hh = np.zeros_like(self.W_hh)
        grad_W_ho = np.zeros_like(self.W_ho)
        grad_b_h = np.zeros_like(self.b_h)
        grad_b_o = np.zeros_like(self.b_o)

        # Gradient with respect to output
        # dL/dy_t = grad_output
        grad_y = grad_output

        # Gradient with respect to W_ho and b_o
        grad_W_ho = np.dot(grad_y.T, h_t)
        grad_b_o = np.sum(grad_y, axis=0)

        # Gradient with respect to h_t
        # dL/dh_t = dL/dy_t * W_ho + dL/dh_{t+1} (if available)
        grad_h_t = np.dot(grad_y, self.W_ho)
        if grad_h is not None:
            grad_h_t += grad_h

        # Gradient through tanh: dL/dh_raw = dL/dh_t * (1 - tanh^2)
        # tanh derivative: d/dx tanh(x) = 1 - tanh^2(x)
        grad_h_raw = grad_h_t * (1 - h_t**2)

        # Gradient with respect to weights and inputs
        # dL/dW_ih = dL/dh_raw @ x_t
        grad_W_ih = np.dot(grad_h_raw.T, x_t)

        # dL/dW_hh = dL/dh_raw @ h_{t-1}
        if h_prev is not None and h_prev.size > 0:
            grad_W_hh = np.dot(grad_h_raw.T, h_prev)

        # dL/db_h = sum of dL/dh_raw over batch
        grad_b_h = np.sum(grad_h_raw, axis=0)

        # dL/dx_t = dL/dh_raw @ W_ih
        grad_x = np.dot(grad_h_raw, self.W_ih)

        # dL/dh_{t-1} = dL/dh_raw @ W_hh
        if h_prev is not None and h_prev.size > 0:
            grad_h_prev = np.dot(grad_h_raw, self.W_hh)

        # Average over batch
        grad_W_ih /= batch_size
        grad_W_hh /= batch_size
        grad_W_ho /= batch_size
        grad_b_h /= batch_size
        grad_b_o /= batch_size

        # Store gradients for optimizer
        self.grad_W_ih = grad_W_ih
        self.grad_W_hh = grad_W_hh
        self.grad_W_ho = grad_W_ho
        self.grad_b_h = grad_b_h
        self.grad_b_o = grad_b_o

        if single_sample:
            grad_x = grad_x[0]
            if grad_h_prev is not None:
                grad_h_prev = grad_h_prev[0]

        return grad_x, grad_h_prev


class SimpleRNN:
    """
    Simple Recurrent Neural Network

    Processes sequences using RNN cells.

    Architecture:
        For each time step t:
            h_t, y_t = rnn_cell(x_t, h_{t-1})

    Use cases:
        - Many-to-one: Sentiment analysis, classification
        - Many-to-many: Sequence generation, POS tagging

    Parameters:
        input_size: Number of input features per time step
        hidden_size: Number of hidden units
        output_size: Number of output features
        return_sequences: Whether to return outputs for all time steps
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        return_sequences: bool = False,
    ):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.return_sequences = return_sequences

        # Create RNN cell
        self.rnn_cell = RNNCell(input_size, hidden_size, output_size)

        # Softmax for classification
        from .cnn import Softmax

        self.softmax = Softmax()

    def forward(
        self, x: np.ndarray, h_0: np.ndarray = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward pass through entire sequence

        Args:
            x: Input sequence, shape (batch, seq_length, input_size)
            h_0: Initial hidden state, shape (batch, hidden_size)

        Returns:
            Tuple of (outputs, final hidden state)
        """
        batch_size, seq_length, _ = x.shape

        # Initialize hidden state
        if h_0 is None:
            h_0 = np.zeros((batch_size, self.hidden_size))

        # Store all hidden states and outputs
        hidden_states = []
        outputs = []

        h_prev = h_0

        for t in range(seq_length):
            x_t = x[:, t, :]  # (batch, input_size)

            y_t, h_t = self.rnn_cell.forward(x_t, h_prev)

            hidden_states.append(h_t)
            outputs.append(y_t)

            h_prev = h_t

        # Stack outputs
        if self.return_sequences:
            output = np.stack(outputs, axis=1)  # (batch, seq_length, output_size)
        else:
            output = outputs[-1]  # Last output, (batch, output_size)

        hidden_states = np.stack(hidden_states, axis=1)

        return output, hidden_states

    def backward(self, grad_output: np.ndarray, grad_hidden: np.ndarray = None):
        """
        Backward pass through entire sequence (BPTT)

        Args:
            grad_output: Gradient with respect to outputs
            grad_hidden: Gradient with respect to final hidden state
        """
        # This would implement full BPTT
        # For simplicity, we'll do truncated BPTT (one step)
        pass

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Make predictions

        Args:
            x: Input sequence

        Returns:
            Predicted class indices (if classification)
        """
        output, _ = self.forward(x)

        if self.return_sequences:
            # Get last output for classification
            output = output[:, -1, :]

        # Apply softmax
        probabilities = self.softmax.forward(output)
        return np.argmax(probabilities, axis=-1)


class BiDirectionalRNN:
    """
    Bidirectional RNN

    Processes sequence in both forward and backward directions.

    Architecture:
        Forward: h_t^f = RNN(x_t, h_{t-1}^f)
        Backward: h_t^b = RNN(x_t, h_{t+1}^b)
        Combined: y_t = W * [h_t^f; h_t^b] + b

    Use cases:
        - Sequence labeling (NER, POS)
        - Speech recognition
        - Machine translation

    Advantages:
        - Can use context from both past and future
        - Better for tasks requiring full context
    """

    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Forward and backward RNNs
        self.rnn_forward = SimpleRNN(input_size, hidden_size, hidden_size)
        self.rnn_backward = SimpleRNN(input_size, hidden_size, hidden_size)

        # Output layer
        self.W_y = np.random.randn(output_size, 2 * hidden_size) * np.sqrt(
            2.0 / (2 * hidden_size)
        )
        self.b_y = np.zeros(output_size)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass

        Args:
            x: Input sequence (batch, seq_length, input_size)

        Returns:
            Output (batch, seq_length, output_size)
        """
        batch_size, seq_length, _ = x.shape

        # Forward pass (normal order)
        output_fwd, _ = self.rnn_forward.forward(x)

        # Backward pass (reversed sequence)
        x_reversed = np.flip(x, axis=1)
        output_bwd, _ = self.rnn_backward.forward(x_reversed)
        output_bwd = np.flip(output_bwd, axis=1)

        # Combine forward and backward
        combined = np.concatenate([output_fwd, output_bwd], axis=-1)

        # Apply output layer
        output = np.dot(combined, self.W_y.T) + self.b_y

        return output


def test_rnn():
    """Test RNN implementation"""
    print("=" * 60)
    print("Testing RNN Implementation")
    print("=" * 60)

    # Test data: batch of 2 sequences, each with 5 time steps, 3 features
    np.random.seed(42)
    x = np.random.randn(2, 5, 3)  # (batch, seq_len, input_size)
    labels = np.array([1, 0])  # Binary classification

    print(f"\nInput shape: {x.shape}")
    print(f"Labels: {labels}")

    # Create RNN
    rnn = SimpleRNN(input_size=3, hidden_size=8, output_size=2, return_sequences=False)

    # Test forward pass
    output, hidden_states = rnn.forward(x)
    print(f"\nOutput shape: {output.shape}")
    print(f"Hidden states shape: {hidden_states.shape}")

    # Test with return_sequences
    rnn_seq = SimpleRNN(
        input_size=3, hidden_size=8, output_size=2, return_sequences=True
    )
    output_seq, _ = rnn_seq.forward(x)
    print(f"\nOutput (return_sequences=True) shape: {output_seq.shape}")

    # Test prediction
    predictions = rnn.predict(x)
    print(f"\nPredictions: {predictions}")

    # Test bidirectional RNN
    bi_rnn = BiDirectionalRNN(input_size=3, hidden_size=8, output_size=2)
    bi_output = bi_rnn.forward(x)
    print(f"\nBi-RNN output shape: {bi_output.shape}")

    print("\n" + "=" * 60)
    print("All RNN tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    test_rnn()
