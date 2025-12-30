"""
Dataset Downloader Script
=========================
Downloads sample datasets for the AI Engineer Toolkit projects.
Datasets:
1. Iris (Classification)
2. Housing Prices (Regression)
3. Tiny Shakespeare (LLM Training)
4. Credit Card Fraud (Anomaly Detection - Sample)

Usage:
    python download_sample_datasets.py
"""

import os
import logging
import urllib.request
import pandas as pd
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths
DATA_DIR = Path("data/raw")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Dataset URLs
DATASETS = {
    "iris.csv": "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",
    "housing.csv": "https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/housing.csv",
    "tiny_shakespeare.txt": "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
}

def download_file(url: str, dest_path: Path):
    """Download a file from URL to destination path."""
    if dest_path.exists():
        logger.info(f"File already exists: {dest_path}")
        return

    logger.info(f"Downloading {url} to {dest_path}...")
    try:
        urllib.request.urlretrieve(url, dest_path)
        logger.info(f"Successfully downloaded {dest_path}")
    except Exception as e:
        logger.error(f"Failed to download {url}: {e}")

def process_iris():
    """Add headers to Iris dataset."""
    iris_path = DATA_DIR / "iris.csv"
    if not iris_path.exists():
        return
    
    try:
        df = pd.read_csv(iris_path, header=None)
        # Check if headers already exist (simple heuristic)
        if isinstance(df.iloc[0, 0], str) and "sepal" in str(df.iloc[0, 0]).lower():
            logger.info("Iris dataset already has headers.")
            return

        df.columns = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]
        df.to_csv(iris_path, index=False)
        logger.info("Added headers to Iris dataset.")
    except Exception as e:
        logger.error(f"Error processing Iris dataset: {e}")

def generate_mock_fraud_data():
    """Generate a mock credit card fraud dataset since real one is huge/private."""
    fraud_path = DATA_DIR / "credit_card_fraud_sample.csv"
    if fraud_path.exists():
        logger.info(f"File already exists: {fraud_path}")
        return

    logger.info("Generating mock credit card fraud dataset...")
    try:
        import numpy as np
        np.random.seed(42)
        n_samples = 10000
        
        # Features V1-V28 (PCA transformed features usually)
        data = np.random.randn(n_samples, 28)
        
        # Time and Amount
        time = np.arange(n_samples)
        amount = np.random.exponential(scale=100, size=n_samples)
        
        # Class (0: Normal, 1: Fraud) - heavily imbalanced
        # 0.5% fraud rate
        is_fraud = np.random.choice([0, 1], size=n_samples, p=[0.995, 0.005])
        
        columns = [f"V{i+1}" for i in range(28)] + ["Time", "Amount", "Class"]
        df = pd.DataFrame(data, columns=[f"V{i+1}" for i in range(28)])
        df["Time"] = time
        df["Amount"] = amount
        df["Class"] = is_fraud
        
        df.to_csv(fraud_path, index=False)
        logger.info(f"Successfully generated {fraud_path}")
    except Exception as e:
        logger.error(f"Failed to generate fraud data: {e}")

def main():
    logger.info("Starting dataset download process...")
    
    # 1. Download standard datasets
    for filename, url in DATASETS.items():
        download_file(url, DATA_DIR / filename)
    
    # 2. Process specific datasets
    process_iris()
    
    # 3. Generate synthetic/mock datasets where real ones are tricky
    generate_mock_fraud_data()
    
    logger.info("Dataset download process complete.")
    logger.info(f"Data directory content: {list(DATA_DIR.glob('*'))}")

if __name__ == "__main__":
    main()
