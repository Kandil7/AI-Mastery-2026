"""
GitHub Issue Classifier Training Script
========================================

This script implements the complete training pipeline for the GitHub Issue Classifier capstone project.

Components:
1. Synthetic dataset generation (2000+ samples)
2. TF-IDF vectorization
3. Neural network training using src/ml/deep_learning.py
4. Model evaluation and metrics
5. Model persistence for deployment

Target: >85% test accuracy
"""

import numpy as np
import pandas as pd
import pickle
import json
import os
from typing import List, Tuple, Dict
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


def generate_issue_dataset(n_samples: int = 2000) -> Tuple[List[str], np.ndarray]:
    """
    Generate synthetic GitHub issues dataset with balanced classes.
    
    Args:
        n_samples: Total number of samples to generate
        
    Returns:
        Tuple of (issue_texts, labels)
    """
    print(f"Generating {n_samples} synthetic GitHub issues...")
    
    templates = {
        'bug': [
            "Error when {action}: {error_msg}",
            "Application crashes during {action}",
            "{feature} is broken after update",
            "Bug: {feature} not working correctly",
            "Critical: {error_msg} occurs when {action}",
            "Unexpected behavior in {feature}",
            "Cannot {action} - getting {error_msg}",
        ],
        'feature': [
            "Request: Add {feature} functionality",
            "Feature suggestion: {action}",
            "Would be great to have {feature}",
            "Enhancement: Support for {feature}",
            "Proposal: Add ability to {action}",
            "New feature: Integrate {feature}",
            "Improvement: Better {feature} handling",
        ],
        'question': [
            "How to {action}?",
            "Question about {feature}",
            "What is the best way to {action}?",
            "Documentation unclear about {feature}",
            "Need help with {action}",
            "Confused about {feature} implementation",
            "Can someone explain how to {action}?",
        ],
        'documentation': [
            "Docs: Update section on {feature}",
            "Missing documentation for {feature}",
            "Typo in {feature} documentation",
            "Add examples for {action}",
            "Documentation: Fix {feature} section",
            "README needs update for {action}",
            "Improve {feature} documentation clarity",
        ]
    }
    
    actions = [
        'login', 'signup', 'export data', 'import files', 'run tests',
        'deploy app', 'configure settings', 'install package', 'build project',
        'authenticate user', 'process payment', 'send email', 'upload file'
    ]
    
    features = [
        'authentication', 'database', 'API', 'caching', 'logging',
        'payment gateway', 'notification system', 'search functionality',
        'user permissions', 'file storage', 'email service', 'analytics'
    ]
    
    errors = [
        'TypeError', 'ValueError', 'ConnectionError', 'TimeoutError',
        'AuthenticationError', 'PermissionDenied', 'NotFoundError',
        'ValidationError', 'DatabaseError', 'NetworkError'
    ]
    
    issues = []
    labels = []
    
    label_to_idx = {'bug': 0, 'feature': 1, 'question': 2, 'documentation': 3}
    
    # Generate balanced dataset
    samples_per_class = n_samples // len(label_to_idx)
    
    for label_name, label_idx in label_to_idx.items():
        for _ in range(samples_per_class):
            template = np.random.choice(templates[label_name])
            
            issue = template.format(
                action=np.random.choice(actions),
                feature=np.random.choice(features),
                error_msg=np.random.choice(errors)
            )
            
            issues.append(issue)
            labels.append(label_idx)
    
    # Shuffle the dataset
    indices = np.random.permutation(len(issues))
    issues = [issues[i] for i in indices]
    labels = np.array([labels[i] for i in indices])
    
    print(f"✓ Generated {len(issues)} issues across {len(label_to_idx)} classes")
    print(f"  Class distribution: {dict(zip(*np.unique(labels, return_counts=True)))}")
    
    return issues, labels


def preprocess_text(texts: List[str], max_features: int = 500) -> Tuple[np.ndarray, TfidfVectorizer]:
    """
    Vectorize text using TF-IDF.
    
    Args:
        texts: List of text strings
        max_features: Maximum number of features to extract
        
    Returns:
        Tuple of (vectorized_data, fitted_vectorizer)
    """
    print(f"\nPreprocessing text with TF-IDF (max_features={max_features})...")
    
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        stop_words='english',
        ngram_range=(1, 2),  # Include bigrams for better context
        min_df=2,  # Ignore very rare terms
    )
    
    X = vectorizer.fit_transform(texts).toarray()
    
    print(f"✓ Text vectorized to shape {X.shape}")
    print(f"  Vocabulary size: {len(vectorizer.vocabulary_)}")
    
    return X, vectorizer


def build_model(input_dim: int, num_classes: int):
    """
    Build neural network classifier using src/ml/deep_learning.py
    
    Architecture:
    - Input: input_dim features
    - Dense(128) + ReLU + Dropout(0.3)
    - Dense(64) + ReLU + Dropout(0.2)
    - Dense(num_classes) + Softmax
    """
    from src.ml.deep_learning import NeuralNetwork, Dense, Activation, Dropout
    
    print(f"\nBuilding neural network...")
    print(f"  Architecture: {input_dim} -> 128 -> 64 -> {num_classes}")
    
    model = NeuralNetwork()
    
    # Layer 1: Dense + ReLU + Dropout
    model.add(Dense(input_dim, 128, weight_init='he'))
    model.add(Activation('relu'))
    model.add(Dropout(0.3))
    
    # Layer 2: Dense + ReLU + Dropout
    model.add(Dense(128, 64, weight_init='he'))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    
    # Output layer: Dense + Softmax
    model.add(Dense(64, num_classes, weight_init='xavier'))
    model.add(Activation('softmax'))
    
    print("✓ Model built successfully")
    model.summary()
    
    return model


def train_model(model, X_train, y_train, X_val, y_val, epochs=100, batch_size=32):
    """Train the model with validation monitoring."""
    from src.ml.deep_learning import CrossEntropyLoss
    
    print(f"\nTraining model...")
    print(f"  Epochs: {epochs}, Batch size: {batch_size}")
    print(f"  Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
    
    # Compile model
    model.compile(loss=CrossEntropyLoss(), learning_rate=0.01)
    
    # Train with validation
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        verbose=True
    )
    
    print("✓ Training complete")
    
    return history


def evaluate_model(model, X_test, y_test, label_names):
    """Comprehensive model evaluation."""
    print(f"\nEvaluating model on test set ({len(X_test)} samples)...")
    
    # Get predictions
    y_pred = model.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"✓ Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Detailed classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=label_names, digits=4))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    return accuracy, y_pred, cm


def plot_training_curves(history, save_path='outputs/training_curves.png'):
    """Plot training and validation curves."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss curve
    axes[0].plot(history['loss'], label='Train Loss', linewidth=2)
    axes[0].plot(history['val_loss'], label='Validation Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training & Validation Loss', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy curve
    axes[1].plot(history['accuracy'], label='Train Accuracy', linewidth=2)
    axes[1].plot(history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].set_title('Training & Validation Accuracy', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Training curves saved to {save_path}")
    plt.close()


def plot_confusion_matrix(cm, label_names, save_path='outputs/confusion_matrix.png'):
    """Plot confusion matrix heatmap."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=label_names,
        yticklabels=label_names,
        cbar_kws={'label': 'Count'},
        square=True,
        linewidths=0.5
    )
    plt.xlabel('Predicted Label', fontsize=13, fontweight='bold')
    plt.ylabel('True Label', fontsize=13, fontweight='bold')
    plt.title('Confusion Matrix - GitHub Issue Classifier', fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Confusion matrix saved to {save_path}")
    plt.close()


def save_model(model, vectorizer, label_names, metadata, save_path='models/issue_classifier.pkl'):
    """Save complete model package for deployment."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Extract model parameters
    model_data = {
        'layer_params': [layer.get_params() for layer in model.layers],
        'architecture': {
            'input_dim': model.layers[0].input_dim if hasattr(model.layers[0], 'input_dim') else None,
            'layers': [str(layer) for layer in model.layers]
        },
        'vectorizer_vocab': vectorizer.vocabulary_,
        'vectorizer_idf': vectorizer.idf_.tolist(),
        'vectorizer_params': {
            'max_features': vectorizer.max_features,
            'ngram_range': vectorizer.ngram_range,
        },
        'labels': label_names,
        'metadata': metadata
    }
    
    with open(save_path, 'wb') as f:
        pickle.dump(model_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    file_size_mb = os.path.getsize(save_path) / (1024 * 1024)
    print(f"✓ Model saved to {save_path} ({file_size_mb:.2f} MB)")
    
    # Also save metadata as JSON for easy inspection
    metadata_path = save_path.replace('.pkl', '_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump({
            'labels': label_names,
            'metadata': metadata,
            'architecture': model_data['architecture'],
            'vectorizer_params': model_data['vectorizer_params']
        }, f, indent=2)
    print(f"✓ Metadata saved to {metadata_path}")


def main():
    """Main training pipeline."""
    print("="*70)
    print("GitHub Issue Classifier - Training Pipeline")
    print("="*70)
    
    # Configuration
    N_SAMPLES = 2000
    MAX_FEATURES = 500
    TEST_SIZE = 0.2
    VAL_SIZE = 0.1  # 10% of training data
    EPOCHS = 100
    BATCH_SIZE = 32
    
    # Label mapping
    label_names = ['bug', 'feature', 'question', 'documentation']
    
    # Step 1: Generate dataset
    texts, labels = generate_issue_dataset(n_samples=N_SAMPLES)
    
    # Step 2: Preprocess text
    X, vectorizer = preprocess_text(texts, max_features=MAX_FEATURES)
    
    # Step 3: Split data
    print(f"\nSplitting dataset (test={TEST_SIZE}, val={VAL_SIZE})...")
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, labels, test_size=TEST_SIZE, random_state=42, stratify=labels
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=VAL_SIZE, random_state=42, stratify=y_train_full
    )
    
    print(f"✓ Dataset split complete:")
    print(f"  Train: {X_train.shape}")
    print(f"  Validation: {X_val.shape}")
    print(f"  Test: {X_test.shape}")
    
    # Step 4: Build model
    model = build_model(input_dim=MAX_FEATURES, num_classes=len(label_names))
    
    # Step 5: Train model
    history = train_model(model, X_train, y_train, X_val, y_val, epochs=EPOCHS, batch_size=BATCH_SIZE)
    
    # Step 6: Evaluate model
    test_accuracy, y_pred, cm = evaluate_model(model, X_test, y_test, label_names)
    
    # Step 7: Generate visualizations
    print("\nGenerating visualizations...")
    plot_training_curves(history)
    plot_confusion_matrix(cm, label_names)
    
    # Step 8: Save model
    metadata = {
        'test_accuracy': float(test_accuracy),
        'n_samples': N_SAMPLES,
        'n_features': MAX_FEATURES,
        'n_classes': len(label_names),
        'epochs': EPOCHS,
        'batch_size': BATCH_SIZE,
        'final_train_loss': float(history['loss'][-1]),
        'final_val_loss': float(history['val_loss'][-1]),
        'final_train_acc': float(history['accuracy'][-1]),
        'final_val_acc': float(history['val_accuracy'][-1]),
    }
    
    save_model(model, vectorizer, label_names, metadata)
    
    print("\n" + "="*70)
    print("✅ TRAINING COMPLETE!")
    print("="*70)
    print(f"Final Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    print(f"Target Achieved: {'YES ✓' if test_accuracy >= 0.85 else 'NO ✗ (target: 85%)'}")
    print(f"\nOutputs:")
    print(f"  - Model: models/issue_classifier.pkl")
    print(f"  - Metadata: models/issue_classifier_metadata.json")
    print(f"  - Training curves: outputs/training_curves.png")
    print(f"  - Confusion matrix: outputs/confusion_matrix.png")
    print("="*70)


if __name__ == "__main__":
    main()
