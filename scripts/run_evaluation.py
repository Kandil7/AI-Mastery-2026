"""
Script to run evaluation metrics on AI models.
Loads predictions and ground truth, then computes accuracy, precision, recall, F1, etc.

Usage:
    python scripts/run_evaluation.py --truth data/test_y.csv --preds data/preds.csv --task classification
"""

import argparse
import pandas as pd
import numpy as np
import logging
from pathlib import Path
import sys
import os

# Add project root to path to import src modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.foundation_utils import mse_loss, cross_entropy_loss

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def evaluate_classification(y_true, y_pred, threshold=0.5):
    """Compute classification metrics."""
    # Convert probabilities to binary predictions
    y_pred_binary = (y_pred >= threshold).astype(int)
    
    tp = np.sum((y_pred_binary == 1) & (y_true == 1))
    fp = np.sum((y_pred_binary == 1) & (y_true == 0))
    tn = np.sum((y_pred_binary == 0) & (y_true == 0))
    fn = np.sum((y_pred_binary == 0) & (y_true == 1))
    
    accuracy = (tp + tn) / len(y_true)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    metrics = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "Cross Entropy Loss": cross_entropy_loss(y_true, y_pred)
    }
    return metrics

def evaluate_regression(y_true, y_pred):
    """Compute regression metrics."""
    mse = mse_loss(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_true - y_pred))
    
    # R-squared
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    
    metrics = {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "R2 Score": r2
    }
    return metrics

def main():
    parser = argparse.ArgumentParser(description="Run evaluation on predictions")
    parser.add_argument("--truth", type=str, required=True, help="Path to ground truth CSV (column 'target')")
    parser.add_argument("--preds", type=str, required=True, help="Path to predictions CSV (column 'prediction')")
    parser.add_argument("--task", type=str, choices=['classification', 'regression'], required=True, help="Task type")
    
    args = parser.parse_args()

    # Load data
    try:
        df_true = pd.read_csv(args.truth)
        df_pred = pd.read_csv(args.preds)
        
        if 'target' not in df_true.columns or 'prediction' not in df_pred.columns:
            logger.error("Input CSVs must have 'target' and 'prediction' columns respectively")
            return
            
        y_true = df_true['target'].values
        y_pred = df_pred['prediction'].values
        
        if len(y_true) != len(y_pred):
            logger.error(f"Length mismatch: Truth={len(y_true)}, Preds={len(y_pred)}")
            return
            
        logger.info(f"Evaluating {args.task} on {len(y_true)} samples...")
        
        if args.task == 'classification':
            metrics = evaluate_classification(y_true, y_pred)
        else:
            metrics = evaluate_regression(y_true, y_pred)
            
        print("\n=== Evaluation Results ===")
        for k, v in metrics.items():
            print(f"{k:<20}: {v:.4f}")
            
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")

if __name__ == "__main__":
    main()
