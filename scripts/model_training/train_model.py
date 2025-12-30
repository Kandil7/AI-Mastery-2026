"""
Model Training Script
=====================
Unified training script for various ML models.

Supports:
- Classical ML (sklearn-style)
- Neural networks (from scratch)
- Configurable via command line or config files

Author: AI-Mastery-2026
"""

import numpy as np
import json
import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import argparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass
class TrainingConfig:
    """Training configuration."""
    # Data
    data_path: str = "data/synthetic/classification.npz"
    train_split: float = 0.8
    
    # Model
    model_type: str = "neural_network"  # neural_network, logistic, random_forest
    
    # Neural network specific
    hidden_layers: Tuple[int, ...] = (64, 32)
    activation: str = "relu"
    output_activation: str = "sigmoid"
    
    # Training
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 0.01
    
    # Output
    output_dir: str = "models"
    model_name: str = "model"
    save_format: str = "joblib"
    
    # Logging
    log_interval: int = 10
    
    @classmethod
    def from_file(cls, path: str) -> 'TrainingConfig':
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(**data)
    
    def save(self, path: str):
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)


# ============================================================
# DATA LOADING
# ============================================================

def load_data(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load data from npz file."""
    data = np.load(path)
    X = data['X']
    y = data['y']
    return X, y


def train_test_split(
    X: np.ndarray, 
    y: np.ndarray, 
    train_ratio: float = 0.8,
    shuffle: bool = True,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split data into train and test sets."""
    np.random.seed(random_state)
    n = len(X)
    
    if shuffle:
        indices = np.random.permutation(n)
    else:
        indices = np.arange(n)
    
    split_idx = int(n * train_ratio)
    train_idx = indices[:split_idx]
    test_idx = indices[split_idx:]
    
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


# ============================================================
# MODEL CREATION
# ============================================================

def create_model(config: TrainingConfig) -> Any:
    """Create model based on configuration."""
    if config.model_type == "neural_network":
        # Import from our library
        from src.ml.deep_learning import NeuralNetwork
        
        return NeuralNetwork(
            hidden_layers=list(config.hidden_layers),
            activation=config.activation,
            output_activation=config.output_activation,
            learning_rate=config.learning_rate
        )
    
    elif config.model_type == "logistic":
        from src.ml.classical import LogisticRegressionScratch
        return LogisticRegressionScratch(
            learning_rate=config.learning_rate,
            max_iter=config.epochs
        )
    
    elif config.model_type == "random_forest":
        from src.ml.classical import RandomForestScratch
        return RandomForestScratch(n_estimators=100, max_depth=10)
    
    else:
        raise ValueError(f"Unknown model type: {config.model_type}")


# ============================================================
# TRAINING LOOP
# ============================================================

@dataclass
class TrainingMetrics:
    """Training metrics."""
    epoch: int
    train_loss: float
    train_accuracy: float
    val_loss: Optional[float] = None
    val_accuracy: Optional[float] = None
    epoch_time: float = 0.0


def compute_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute classification accuracy."""
    if y_pred.ndim > 1:
        y_pred = np.argmax(y_pred, axis=1)
    if y_true.ndim > 1:
        y_true = np.argmax(y_true, axis=1)
    return np.mean(y_true == y_pred)


def compute_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute cross-entropy loss."""
    eps = 1e-10
    y_pred = np.clip(y_pred, eps, 1 - eps)
    
    if y_true.ndim == 1:
        # Binary classification
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    else:
        # Multi-class
        return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))


def train_epoch(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int
) -> Tuple[float, float]:
    """Train for one epoch."""
    n_samples = len(X)
    indices = np.random.permutation(n_samples)
    
    total_loss = 0.0
    n_batches = 0
    
    for start_idx in range(0, n_samples, batch_size):
        batch_idx = indices[start_idx:start_idx + batch_size]
        X_batch = X[batch_idx]
        y_batch = y[batch_idx]
        
        # Forward pass
        if hasattr(model, 'forward'):
            y_pred = model.forward(X_batch)
        else:
            y_pred = model.predict_proba(X_batch)
        
        # Compute loss
        batch_loss = compute_loss(y_batch, y_pred)
        total_loss += batch_loss
        n_batches += 1
        
        # Backward pass
        if hasattr(model, 'backward'):
            model.backward(X_batch, y_batch)
    
    avg_loss = total_loss / n_batches
    
    # Compute accuracy
    if hasattr(model, 'forward'):
        y_pred = model.forward(X)
    else:
        y_pred = model.predict_proba(X)
    accuracy = compute_accuracy(y, (y_pred > 0.5).astype(int))
    
    return avg_loss, accuracy


def evaluate(model: Any, X: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """Evaluate model on data."""
    if hasattr(model, 'forward'):
        y_pred = model.forward(X)
    elif hasattr(model, 'predict_proba'):
        y_pred = model.predict_proba(X)
    else:
        y_pred = model.predict(X)
    
    loss = compute_loss(y, y_pred)
    accuracy = compute_accuracy(y, (y_pred > 0.5).astype(int))
    
    return loss, accuracy


def train(
    config: TrainingConfig,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None
) -> Tuple[Any, list]:
    """
    Main training function.
    
    Returns:
        Tuple of (trained_model, metrics_history)
    """
    logger.info(f"Training {config.model_type} model...")
    logger.info(f"Train samples: {len(X_train)}, Features: {X_train.shape[1]}")
    
    model = create_model(config)
    
    # Initialize model if needed
    if hasattr(model, 'initialize'):
        n_features = X_train.shape[1]
        n_outputs = 1 if y_train.ndim == 1 else y_train.shape[1]
        model.initialize(n_features, n_outputs)
    
    metrics_history = []
    best_val_acc = 0.0
    
    for epoch in range(config.epochs):
        start_time = time.time()
        
        # Train
        train_loss, train_acc = train_epoch(
            model, X_train, y_train, config.batch_size
        )
        
        # Validate
        if X_val is not None:
            val_loss, val_acc = evaluate(model, X_val, y_val)
        else:
            val_loss, val_acc = None, None
        
        epoch_time = time.time() - start_time
        
        metrics = TrainingMetrics(
            epoch=epoch + 1,
            train_loss=train_loss,
            train_accuracy=train_acc,
            val_loss=val_loss,
            val_accuracy=val_acc,
            epoch_time=epoch_time
        )
        metrics_history.append(asdict(metrics))
        
        # Log progress
        if (epoch + 1) % config.log_interval == 0 or epoch == 0:
            log_msg = f"Epoch {epoch + 1}/{config.epochs} - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}"
            if val_acc is not None:
                log_msg += f" - Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
            logger.info(log_msg)
        
        # Track best
        if val_acc is not None and val_acc > best_val_acc:
            best_val_acc = val_acc
    
    logger.info(f"Training complete. Best val accuracy: {best_val_acc:.4f}")
    return model, metrics_history


# ============================================================
# MODEL SAVING
# ============================================================

def save_model(model: Any, config: TrainingConfig, metrics: list):
    """Save trained model and metadata."""
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = output_dir / f"{config.model_name}.{config.save_format}"
    
    if config.save_format == "joblib":
        import joblib
        joblib.dump(model, model_path)
    else:
        import pickle
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
    
    # Save config
    config.save(output_dir / f"{config.model_name}_config.json")
    
    # Save metrics
    with open(output_dir / f"{config.model_name}_metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)
    
    logger.info(f"Model saved to {model_path}")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Train ML models")
    parser.add_argument("--config", help="Path to config JSON file")
    parser.add_argument("--data", default="data/synthetic/classification.npz")
    parser.add_argument("--model-type", default="neural_network",
                       choices=["neural_network", "logistic", "random_forest"])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output-dir", default="models")
    parser.add_argument("--name", default="model")
    args = parser.parse_args()
    
    # Load or create config
    if args.config:
        config = TrainingConfig.from_file(args.config)
    else:
        config = TrainingConfig(
            data_path=args.data,
            model_type=args.model_type,
            epochs=args.epochs,
            learning_rate=args.lr,
            batch_size=args.batch_size,
            output_dir=args.output_dir,
            model_name=args.name
        )
    
    # Load data
    logger.info(f"Loading data from {config.data_path}")
    X, y = load_data(config.data_path)
    
    # Split
    X_train, X_val, y_train, y_val = train_test_split(X, y, config.train_split)
    
    # Train
    model, metrics = train(config, X_train, y_train, X_val, y_val)
    
    # Save
    save_model(model, config, metrics)
    
    print("\n✓ Training complete!")
    print(f"  Model saved to: {config.output_dir}/{config.model_name}")


if __name__ == "__main__":
    main()
