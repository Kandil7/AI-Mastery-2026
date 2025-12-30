"""
Train and save a pre-trained scikit-learn model for the API.

This script creates sample models that can be loaded by the API
for real inference instead of dummy logic.
"""

import numpy as np
import os
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_models_directory():
    """Create models directory if it doesn't exist."""
    models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
    os.makedirs(models_dir, exist_ok=True)
    return models_dir


def train_classification_model():
    """Train and return a classification model."""
    logger.info("Training classification model...")
    
    # Generate synthetic classification data
    X, y = make_classification(
        n_samples=1000,
        n_features=10,
        n_informative=8,
        n_redundant=2,
        n_classes=2,
        random_state=42
    )
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Create pipeline with scaling and classifier
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        ))
    ])
    
    # Train
    pipeline.fit(X_train, y_train)
    
    # Evaluate
    train_score = pipeline.score(X_train, y_train)
    test_score = pipeline.score(X_test, y_test)
    
    logger.info(f"Classification Model - Train Accuracy: {train_score:.4f}, Test Accuracy: {test_score:.4f}")
    
    return pipeline, {
        'model_type': 'RandomForestClassifier',
        'n_features': 10,
        'n_classes': 2,
        'train_accuracy': train_score,
        'test_accuracy': test_score,
        'feature_names': [f'feature_{i}' for i in range(10)]
    }


def train_regression_model():
    """Train and return a regression model."""
    logger.info("Training regression model...")
    
    # Generate synthetic regression data
    X, y = make_regression(
        n_samples=1000,
        n_features=10,
        n_informative=8,
        noise=10,
        random_state=42
    )
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Create pipeline with scaling and regressor
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', GradientBoostingRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        ))
    ])
    
    # Train
    pipeline.fit(X_train, y_train)
    
    # Evaluate
    train_score = pipeline.score(X_train, y_train)
    test_score = pipeline.score(X_test, y_test)
    
    logger.info(f"Regression Model - Train R²: {train_score:.4f}, Test R²: {test_score:.4f}")
    
    return pipeline, {
        'model_type': 'GradientBoostingRegressor',
        'n_features': 10,
        'train_r2': train_score,
        'test_r2': test_score,
        'feature_names': [f'feature_{i}' for i in range(10)]
    }


def train_logistic_model():
    """Train and return a logistic regression model for binary classification."""
    logger.info("Training logistic regression model...")
    
    # Generate synthetic classification data
    X, y = make_classification(
        n_samples=1000,
        n_features=5,
        n_informative=4,
        n_redundant=1,
        n_classes=2,
        random_state=42
    )
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Create pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(max_iter=1000, random_state=42))
    ])
    
    # Train
    pipeline.fit(X_train, y_train)
    
    # Evaluate
    train_score = pipeline.score(X_train, y_train)
    test_score = pipeline.score(X_test, y_test)
    
    logger.info(f"Logistic Model - Train Accuracy: {train_score:.4f}, Test Accuracy: {test_score:.4f}")
    
    return pipeline, {
        'model_type': 'LogisticRegression',
        'n_features': 5,
        'n_classes': 2,
        'train_accuracy': train_score,
        'test_accuracy': test_score,
        'feature_names': [f'feature_{i}' for i in range(5)]
    }


def save_models():
    """Train and save all models to the models directory."""
    models_dir = create_models_directory()
    
    # Train and save classification model
    clf_model, clf_metadata = train_classification_model()
    clf_path = os.path.join(models_dir, 'classification_model.joblib')
    joblib.dump({'model': clf_model, 'metadata': clf_metadata}, clf_path)
    logger.info(f"Saved classification model to: {clf_path}")
    
    # Train and save regression model
    reg_model, reg_metadata = train_regression_model()
    reg_path = os.path.join(models_dir, 'regression_model.joblib')
    joblib.dump({'model': reg_model, 'metadata': reg_metadata}, reg_path)
    logger.info(f"Saved regression model to: {reg_path}")
    
    # Train and save logistic model
    log_model, log_metadata = train_logistic_model()
    log_path = os.path.join(models_dir, 'logistic_model.joblib')
    joblib.dump({'model': log_model, 'metadata': log_metadata}, log_path)
    logger.info(f"Saved logistic model to: {log_path}")
    
    # Save metadata index
    metadata_path = os.path.join(models_dir, 'models_metadata.json')
    import json
    all_metadata = {
        'classification_model': clf_metadata,
        'regression_model': reg_metadata,
        'logistic_model': log_metadata,
        'default_model': 'classification_model'
    }
    with open(metadata_path, 'w') as f:
        json.dump(all_metadata, f, indent=2)
    logger.info(f"Saved metadata to: {metadata_path}")
    
    return models_dir


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train and save pre-trained models')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory for models (default: ./models)')
    
    args = parser.parse_args()
    
    logger.info("=" * 50)
    logger.info("Training and saving pre-trained models...")
    logger.info("=" * 50)
    
    models_dir = save_models()
    
    logger.info("=" * 50)
    logger.info(f"All models saved to: {models_dir}")
    logger.info("=" * 50)
