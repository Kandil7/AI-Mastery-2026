"""
Multilayer Perceptron (MLP) Implementation from Scratch.

This module provides a complete MLP implementation using the layers,
activations, losses, and optimizers from this package.

Features:
- Flexible architecture definition
- Multiple activation functions
- Multiple loss functions
- Multiple optimizers
- Training with batching
- Validation and early stopping
- Model saving/loading

Example Usage:
    >>> import numpy as np
    >>> from mlp import MLP, Trainer
    >>> 
    >>> # Create MLP
    >>> model = MLP(
    ...     input_size=784,
    ...     hidden_sizes=[256, 128],
    ...     output_size=10,
    ...     activation='relu',
    ...     dropout=0.5
    ... )
    >>> 
    >>> # Train model
    >>> trainer = Trainer(model, loss='cross_entropy', optimizer='adam')
    >>> history = trainer.fit(X_train, y_train, X_val, y_val, epochs=10)
    >>> 
    >>> # Make predictions
    >>> predictions = model.predict(X_test)
"""

from typing import Union, Optional, List, Tuple, Dict, Any, Callable
import numpy as np
from numpy.typing import ArrayLike
import logging
import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime

from .layers import (
    Layer, Dense, Dropout, BatchNormalization, Sequential, Flatten,
    ReLU, Sigmoid, Tanh, LeakyReLU, ELU, Swish, GELU, Identity
)
from .losses import (
    LossFunction, MSELoss, CrossEntropyLoss, BinaryCrossEntropyLoss,
    get_loss
)
from .optimizers import (
    Optimizer, SGD, Adam, RMSprop, Adagrad, AdamW,
    get_optimizer, LearningRateScheduler, GradientClipper
)

logger = logging.getLogger(__name__)

ArrayLike2D = Union[np.ndarray, List]


@dataclass
class TrainingConfig:
    """Configuration for training."""
    batch_size: int = 32
    epochs: int = 100
    learning_rate: float = 0.001
    optimizer: str = 'adam'
    loss: str = 'cross_entropy'
    weight_decay: float = 0.0
    gradient_clip: Optional[float] = None
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 1e-4
    validation_split: float = 0.0
    shuffle: bool = True
    verbose: bool = True


@dataclass
class TrainingHistory:
    """Training history container."""
    train_loss: List[float]
    val_loss: List[float]
    train_acc: List[float]
    val_acc: List[float]
    learning_rates: List[float]
    best_epoch: int
    best_val_loss: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class MLP:
    """
    Multilayer Perceptron (Feedforward Neural Network).
    
    Example:
        >>> mlp = MLP(
        ...     input_size=784,
        ...     hidden_sizes=[256, 128, 64],
        ...     output_size=10,
        ...     activation='relu',
        ...     dropout=0.5,
        ...     use_batch_norm=True
        ... )
        >>> x = np.random.randn(32, 784)
        >>> output = mlp.forward(x)
        >>> output.shape
        (32, 10)
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_sizes: List[int],
        output_size: int,
        activation: str = 'relu',
        dropout: float = 0.0,
        use_batch_norm: bool = False,
        weight_init: str = 'he',
        name: str = 'mlp'
    ):
        """
        Initialize MLP.
        
        Args:
            input_size: Number of input features.
            hidden_sizes: List of hidden layer sizes.
            output_size: Number of output classes/features.
            activation: Activation function name.
            dropout: Dropout probability. 0 for no dropout.
            use_batch_norm: Whether to use batch normalization.
            weight_init: Weight initialization method.
            name: Model name.
        """
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.activation_name = activation
        self.dropout = dropout
        self.use_batch_norm = use_batch_norm
        self.weight_init = weight_init
        self.name = name
        
        # Build network
        self.layers = self._build_network()
        
        logger.info(f"MLP created: {input_size} -> {hidden_sizes} -> {output_size}")
    
    def _get_activation(self, name: str) -> Layer:
        """Get activation layer by name."""
        activations = {
            'relu': ReLU,
            'sigmoid': Sigmoid,
            'tanh': Tanh,
            'leaky_relu': LeakyReLU,
            'elu': ELU,
            'swish': Swish,
            'gelu': GELU,
            'identity': Identity,
            'linear': Identity,
        }
        
        if name.lower() not in activations:
            raise ValueError(f"Unknown activation: {name}")
        
        return activations[name.lower()]()
    
    def _build_network(self) -> List[Layer]:
        """Build the network layers."""
        layers = []
        
        # Input to first hidden
        layer_sizes = [self.input_size] + self.hidden_sizes
        
        for i in range(len(layer_sizes) - 1):
            # Dense layer
            layers.append(Dense(
                input_size=layer_sizes[i],
                output_size=layer_sizes[i + 1],
                weight_init=self.weight_init,
                name=f'fc{i + 1}'
            ))
            
            # Batch normalization
            if self.use_batch_norm:
                layers.append(BatchNormalization(
                    num_features=layer_sizes[i + 1],
                    name=f'bn{i + 1}'
                ))
            
            # Activation
            layers.append(self._get_activation(self.activation_name))
            
            # Dropout
            if self.dropout > 0:
                layers.append(Dropout(p=self.dropout, name=f'dropout{i + 1}'))
        
        # Output layer (no activation, no dropout)
        layers.append(Dense(
            input_size=self.hidden_sizes[-1] if self.hidden_sizes else self.input_size,
            output_size=self.output_size,
            weight_init=self.weight_init,
            name='output'
        ))
        
        return layers
    
    def forward(
        self,
        x: np.ndarray,
        training: bool = True
    ) -> np.ndarray:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor (batch_size, input_size).
            training: Whether in training mode.
        
        Returns:
            np.ndarray: Output tensor.
        """
        for layer in self.layers:
            if isinstance(layer, (Dropout, BatchNormalization)):
                x = layer.forward(x, training=training)
            else:
                x = layer.forward(x)
        return x
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Backward pass through the network.
        
        Args:
            grad_output: Gradient from loss function.
        
        Returns:
            np.ndarray: Gradient w.r.t. input.
        """
        grad = grad_output
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad
    
    def predict(
        self,
        x: np.ndarray,
        return_probs: bool = False
    ) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            x: Input data.
            return_probs: If True, return probabilities instead of class indices.
        
        Returns:
            np.ndarray: Predictions (class indices or probabilities).
        """
        x = np.asarray(x, dtype=np.float64)
        probs = self.forward(x, training=False)
        
        if return_probs:
            # Apply softmax for probabilities
            probs_shifted = probs - np.max(probs, axis=1, keepdims=True)
            exp_probs = np.exp(probs_shifted)
            probs = exp_probs / np.sum(exp_probs, axis=1, keepdims=True)
            return probs
        
        return np.argmax(probs, axis=1)
    
    def get_parameters(self) -> List[Dict[str, np.ndarray]]:
        """Get all trainable parameters."""
        params = []
        for layer in self.layers:
            layer_params = layer.get_parameters()
            if layer_params:
                params.append(layer_params)
        return params
    
    def get_gradients(self) -> List[Dict[str, np.ndarray]]:
        """Get all parameter gradients."""
        grads = []
        for layer in self.layers:
            layer_grads = layer.get_gradients()
            if layer_grads:
                grads.append(layer_grads)
        return grads
    
    def zero_grad(self) -> None:
        """Zero out all gradients."""
        for layer in self.layers:
            layer.zero_grad()
    
    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        total = 0
        for layer in self.layers:
            for param in layer.get_parameters().values():
                total += param.size
        return total
    
    def save(self, filepath: str) -> None:
        """
        Save model weights to file.
        
        Args:
            filepath: Path to save file.
        """
        weights = []
        for layer in self.layers:
            layer_weights = layer.get_parameters()
            if layer_weights:
                weights.append({
                    'name': layer.name,
                    'weights': {k: v.tolist() for k, v in layer_weights.items()}
                })
        
        model_data = {
            'config': {
                'input_size': self.input_size,
                'hidden_sizes': self.hidden_sizes,
                'output_size': self.output_size,
                'activation': self.activation_name,
                'dropout': self.dropout,
                'use_batch_norm': self.use_batch_norm,
                'weight_init': self.weight_init,
            },
            'weights': weights
        }
        
        with open(filepath, 'w') as f:
            json.dump(model_data, f)
        
        logger.info(f"Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'MLP':
        """
        Load model from file.
        
        Args:
            filepath: Path to model file.
        
        Returns:
            MLP: Loaded model.
        """
        with open(filepath, 'r') as f:
            model_data = json.load(f)
        
        config = model_data['config']
        model = cls(
            input_size=config['input_size'],
            hidden_sizes=config['hidden_sizes'],
            output_size=config['output_size'],
            activation=config['activation'],
            dropout=config['dropout'],
            use_batch_norm=config['use_batch_norm'],
            weight_init=config['weight_init']
        )
        
        # Load weights
        weight_idx = 0
        for layer in model.layers:
            layer_params = layer.get_parameters()
            if layer_params:
                saved_weights = model_data['weights'][weight_idx]['weights']
                loaded_params = {k: np.array(v) for k, v in saved_weights.items()}
                layer.set_parameters(loaded_params)
                weight_idx += 1
        
        logger.info(f"Model loaded from {filepath}")
        return model


class Trainer:
    """
    Trainer for MLP models.
    
    Example:
        >>> model = MLP(input_size=784, hidden_sizes=[256], output_size=10)
        >>> trainer = Trainer(model, loss='cross_entropy', optimizer='adam')
        >>> history = trainer.fit(X_train, y_train, X_val, y_val, epochs=10)
    """
    
    def __init__(
        self,
        model: MLP,
        loss: Union[str, LossFunction] = 'cross_entropy',
        optimizer: Union[str, Optimizer] = 'adam',
        learning_rate: float = 0.001,
        weight_decay: float = 0.0,
        gradient_clip: Optional[float] = None,
        lr_scheduler: Optional[LearningRateScheduler] = None
    ):
        """
        Initialize Trainer.
        
        Args:
            model: MLP model to train.
            loss: Loss function name or instance.
            optimizer: Optimizer name or instance.
            learning_rate: Learning rate.
            weight_decay: Weight decay for regularization.
            gradient_clip: Gradient clipping value.
            lr_scheduler: Learning rate scheduler.
        """
        self.model = model
        
        # Setup loss
        if isinstance(loss, str):
            self.loss_fn = get_loss(loss)
        else:
            self.loss_fn = loss
        
        # Setup optimizer
        if isinstance(optimizer, str):
            self.optimizer = get_optimizer(
                optimizer,
                learning_rate=learning_rate,
                weight_decay=weight_decay
            )
        else:
            self.optimizer = optimizer
        
        # Gradient clipping
        self.gradient_clipper = None
        if gradient_clip is not None:
            self.gradient_clipper = GradientClipper(max_norm=gradient_clip)
        
        # Learning rate scheduler
        self.lr_scheduler = lr_scheduler
        
        logger.info(f"Trainer initialized: loss={type(self.loss_fn).__name__}, "
                   f"optimizer={type(self.optimizer).__name__}")
    
    def _compute_accuracy(
        self,
        y_pred: np.ndarray,
        y_true: np.ndarray
    ) -> float:
        """Compute classification accuracy."""
        if y_pred.ndim > 1:
            y_pred = np.argmax(y_pred, axis=1)
        return float(np.mean(y_pred == y_true))
    
    def _create_batches(
        self,
        X: np.ndarray,
        y: np.ndarray,
        batch_size: int,
        shuffle: bool = True
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Create mini-batches."""
        n_samples = X.shape[0]
        indices = np.arange(n_samples)
        
        if shuffle:
            np.random.shuffle(indices)
        
        batches = []
        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            batch_indices = indices[start_idx:end_idx]
            batches.append((X[batch_indices], y[batch_indices]))
        
        return batches
    
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        batch_size: int = 32,
        epochs: int = 100,
        shuffle: bool = True,
        verbose: bool = True,
        early_stopping_patience: int = 10,
        early_stopping_min_delta: float = 1e-4
    ) -> TrainingHistory:
        """
        Train the model.
        
        Args:
            X_train: Training features.
            y_train: Training labels.
            X_val: Validation features.
            y_val: Validation labels.
            batch_size: Mini-batch size.
            epochs: Number of training epochs.
            shuffle: Whether to shuffle training data.
            verbose: Print training progress.
            early_stopping_patience: Patience for early stopping.
            early_stopping_min_delta: Minimum improvement to count.
        
        Returns:
            TrainingHistory: Training history.
        """
        X_train = np.asarray(X_train, dtype=np.float64)
        y_train = np.asarray(y_train, dtype=np.int32).flatten()
        
        if X_val is not None:
            X_val = np.asarray(X_val, dtype=np.float64)
            y_val = np.asarray(y_val, dtype=np.int32).flatten()
        
        # History tracking
        train_losses = []
        val_losses = []
        train_accs = []
        val_accs = []
        learning_rates = []
        
        best_val_loss = float('inf')
        patience_counter = 0
        best_epoch = 0
        
        for epoch in range(epochs):
            # Update learning rate
            if self.lr_scheduler is not None:
                lr = self.lr_scheduler.get_lr(epoch)
                self.optimizer.set_learning_rate(lr)
            learning_rates.append(self.optimizer.get_learning_rate())
            
            # Training
            self.model.zero_grad()
            epoch_loss = 0.0
            epoch_correct = 0
            n_batches = 0
            
            batches = self._create_batches(X_train, y_train, batch_size, shuffle)
            
            for X_batch, y_batch in batches:
                # Forward pass
                logits = self.model.forward(X_batch, training=True)
                
                # Compute loss
                loss = self.loss_fn.forward(logits, y_batch)
                epoch_loss += loss
                
                # Compute accuracy
                preds = np.argmax(logits, axis=1)
                epoch_correct += np.sum(preds == y_batch)
                
                # Backward pass
                grad = self.loss_fn.backward()
                self.model.backward(grad)
                
                n_batches += 1
            
            # Update parameters
            all_params = {}
            all_grads = {}
            param_idx = 0
            
            for layer in self.model.layers:
                layer_params = layer.get_parameters()
                layer_grads = layer.get_gradients()
                
                for name, param in layer_params.items():
                    all_params[f'layer{param_idx}_{name}'] = param
                    all_grads[f'layer{param_idx}_{name}'] = layer_grads.get(name, np.zeros_like(param))
                param_idx += 1
            
            # Gradient clipping
            if self.gradient_clipper is not None:
                all_grads = self.gradient_clipper.clip(all_grads)
            
            # Optimizer step
            updated_params = self.optimizer.step(all_params, all_grads)
            
            # Update model parameters
            param_idx = 0
            for layer in self.model.layers:
                layer_params = layer.get_parameters()
                for name in layer_params.keys():
                    key = f'layer{param_idx}_{name}'
                    if key in updated_params:
                        layer_params[name] = updated_params[key]
                layer.set_parameters(layer_params)
                param_idx += 1
            
            # Compute epoch metrics
            train_loss = epoch_loss / n_batches
            train_acc = epoch_correct / len(y_train)
            train_losses.append(train_loss)
            train_accs.append(train_acc)
            
            # Validation
            if X_val is not None:
                val_logits = self.model.forward(X_val, training=False)
                val_loss = self.loss_fn.forward(val_logits, y_val)
                val_acc = self._compute_accuracy(val_logits, y_val)
                val_losses.append(val_loss)
                val_accs.append(val_acc)
                
                # Early stopping check
                if val_loss < best_val_loss - early_stopping_min_delta:
                    best_val_loss = val_loss
                    best_epoch = epoch
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if verbose:
                    print(f"Epoch {epoch + 1}/{epochs} - "
                          f"train_loss: {train_loss:.4f}, train_acc: {train_acc:.4f}, "
                          f"val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}, "
                          f"lr: {learning_rates[-1]:.6f}")
                
                if patience_counter >= early_stopping_patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch + 1}")
                    break
            else:
                if verbose:
                    print(f"Epoch {epoch + 1}/{epochs} - "
                          f"train_loss: {train_loss:.4f}, train_acc: {train_acc:.4f}, "
                          f"lr: {learning_rates[-1]:.6f}")
        
        history = TrainingHistory(
            train_loss=train_losses,
            val_loss=val_losses if val_losses else train_losses,
            train_acc=train_accs,
            val_acc=val_accs if val_accs else train_accs,
            learning_rates=learning_rates,
            best_epoch=best_epoch,
            best_val_loss=best_val_loss if best_val_loss < float('inf') else train_losses[-1]
        )
        
        logger.info(f"Training completed: best_val_loss={history.best_val_loss:.4f} at epoch {history.best_epoch + 1}")
        return history
    
    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        batch_size: int = 32
    ) -> Dict[str, float]:
        """
        Evaluate model on data.
        
        Args:
            X: Features.
            y: Labels.
            batch_size: Batch size.
        
        Returns:
            Dict: Evaluation metrics.
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int32).flatten()
        
        # Full batch prediction
        logits = self.model.forward(X, training=False)
        loss = self.loss_fn.forward(logits, y)
        acc = self._compute_accuracy(logits, y)
        
        return {'loss': loss, 'accuracy': acc}


def create_mlp_classifier(
    input_size: int,
    n_classes: int,
    hidden_sizes: List[int] = [256, 128],
    activation: str = 'relu',
    dropout: float = 0.5,
    use_batch_norm: bool = True
) -> MLP:
    """
    Create an MLP classifier with recommended defaults.
    
    Args:
        input_size: Number of input features.
        n_classes: Number of output classes.
        hidden_sizes: Hidden layer sizes.
        activation: Activation function.
        dropout: Dropout probability.
        use_batch_norm: Use batch normalization.
    
    Returns:
        MLP: Classifier model.
    
    Example:
        >>> model = create_mlp_classifier(784, 10)
    """
    return MLP(
        input_size=input_size,
        hidden_sizes=hidden_sizes,
        output_size=n_classes,
        activation=activation,
        dropout=dropout,
        use_batch_norm=use_batch_norm
    )


def create_mlp_regressor(
    input_size: int,
    output_size: int = 1,
    hidden_sizes: List[int] = [128, 64],
    activation: str = 'relu',
    dropout: float = 0.0,
    use_batch_norm: bool = False
) -> MLP:
    """
    Create an MLP regressor with recommended defaults.
    
    Args:
        input_size: Number of input features.
        output_size: Number of output values.
        hidden_sizes: Hidden layer sizes.
        activation: Activation function.
        dropout: Dropout probability.
        use_batch_norm: Use batch normalization.
    
    Returns:
        MLP: Regressor model.
    
    Example:
        >>> model = create_mlp_regressor(10, 1)
    """
    return MLP(
        input_size=input_size,
        hidden_sizes=hidden_sizes,
        output_size=output_size,
        activation=activation,
        dropout=dropout,
        use_batch_norm=use_batch_norm
    )


if __name__ == "__main__":
    # Example usage and demonstration
    print("=" * 60)
    print("MLP Module - Demonstration")
    print("=" * 60)
    
    np.random.seed(42)
    
    # Create synthetic classification data
    print("\n1. Creating synthetic classification data...")
    n_samples = 1000
    n_features = 20
    n_classes = 3
    
    X = np.random.randn(n_samples, n_features)
    y = np.random.randint(0, n_classes, n_samples)
    
    # Split data
    split = int(0.8 * n_samples)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    print(f"   Training: {len(X_train)} samples")
    print(f"   Testing: {len(X_test)} samples")
    
    # Create model
    print("\n2. Creating MLP model...")
    model = MLP(
        input_size=n_features,
        hidden_sizes=[64, 32],
        output_size=n_classes,
        activation='relu',
        dropout=0.3,
        use_batch_norm=True
    )
    
    print(f"   Architecture: {n_features} -> 64 -> 32 -> {n_classes}")
    print(f"   Total parameters: {model.count_parameters()}")
    
    # Create trainer
    print("\n3. Training model...")
    trainer = Trainer(
        model,
        loss='cross_entropy',
        optimizer='adam',
        learning_rate=0.001,
        gradient_clip=1.0
    )
    
    # Train
    history = trainer.fit(
        X_train, y_train,
        X_test, y_test,
        batch_size=32,
        epochs=20,
        early_stopping_patience=5,
        verbose=True
    )
    
    # Evaluate
    print("\n4. Evaluating model...")
    metrics = trainer.evaluate(X_test, y_test)
    print(f"   Test Loss: {metrics['loss']:.4f}")
    print(f"   Test Accuracy: {metrics['accuracy']:.4f}")
    
    # Predictions
    print("\n5. Making predictions...")
    predictions = model.predict(X_test[:5])
    probs = model.predict(X_test[:5], return_probs=True)
    print(f"   True labels: {y_test[:5]}")
    print(f"   Predictions: {predictions}")
    print(f"   Probabilities:\n{probs}")
    
    # Save and load
    print("\n6. Saving and loading model...")
    model.save('test_mlp.json')
    loaded_model = MLP.load('test_mlp.json')
    
    # Verify loaded model
    loaded_preds = loaded_model.predict(X_test[:5])
    print(f"   Loaded model predictions: {loaded_preds}")
    print(f"   Predictions match: {np.array_equal(predictions, loaded_preds)}")
    
    # Clean up
    os.remove('test_mlp.json')
    
    print("\n" + "=" * 60)
