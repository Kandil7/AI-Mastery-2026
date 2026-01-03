#!/usr/bin/env python3
"""
Comprehensive Week 02 Notebook Generator
Creates complete Jupyter notebooks with mathematical derivations and from-scratch implementations
"""

import json
import os
from pathlib import Path
from typing import List, Dict

BASE_DIR = Path("k:/learning/technical/ai-ml/AI-Mastery-2026/notebooks/week_02")

def create_notebook(cells):
    """Create Jupyter notebook structure."""
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10.0"}
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

def md(content):
    """Markdown cell."""
    return {"cell_type": "markdown", "metadata": {}, "source": content if isinstance(content, list) else [content]}

def code(content):
    """Code cell."""
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], 
            "source": content if isinstance(content, list) else [content]}

# ============================================================================
# NOTEBOOK 1: LOGISTIC REGRESSION COMPLETE
# ============================================================================
def create_logistic_regression_notebook():
    """Complete Logistic Regression with full derivations."""
    return create_notebook([
        md(["# 🎯 Logistic Regression: Complete Implementation\n", "\n",
            "## What You'll Learn\n", "1. Derive cost function from maximum likelihood\n",
            "2. Implement binary & multiclass from scratch\n", "3. Add L1/L2 regularization\n",
            "4. Visualize decision boundaries & ROC curves\n", "\n---\n"]),
        
        code(["import numpy as np\nimport matplotlib.pyplot as plt\nimport seaborn as sns\n",
              "from sklearn.datasets import make_classification, load_breast_cancer\n",
              "from sklearn.model_selection import train_test_split\n",
              "from sklearn.preprocessing import StandardScaler\n",
              "from sklearn.metrics import accuracy_score, roc_curve, auc, confusion_matrix\n\n",
              "np.random.seed(42)\nplt.style.use('seaborn-v0_8')\n",
              "print('✅ Setup complete!')\n"]),
        
        md(["---\n# Chapter 1: Mathematical Foundation\n\n",
            "## Sigmoid Function\n\n$$\\sigma(z) = \\frac{1}{1 + e^{-z}}$$\n\n",
            "**Properties:**\n- Range: (0, 1)\n- Derivative: $\\sigma'(z) = \\sigma(z)(1-\\sigma(z))$\n",
            "- Decision boundary at $z=0$\n"]),
        
        code(["def sigmoid(z):\n    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))\n\n",
              "def sigmoid_derivative(z):\n    s = sigmoid(z)\n    return s * (1 - s)\n\n",
              "# Visualize\nz = np.linspace(-10, 10, 200)\n",
              "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))\n",
              "ax1.plot(z, sigmoid(z), 'b-', lw=2)\n",
              "ax1.axhline(0.5, color='r', linestyle='--', alpha=0.5)\n",
              "ax1.set_title('Sigmoid Function')\nax1.grid(True, alpha=0.3)\n",
              "ax2.plot(z, sigmoid_derivative(z), 'g-', lw=2)\n",
              "ax2.set_title('Sigmoid Derivative')\nax2.grid(True, alpha=0.3)\n",
              "plt.tight_layout()\nplt.show()\n"]),
        
        md(["## Binary Cross-Entropy Loss\n\n",
            "$$J(\\mathbf{w}) = -\\frac{1}{m} \\sum_{i=1}^{m} \\left[ y^{(i)} \\log(h(\\mathbf{x}^{(i)})) + (1-y^{(i)}) \\log(1-h(\\mathbf{x}^{(i)})) \\right]$$\n\n",
            "**Gradient:**\n\n$$\\frac{\\partial J}{\\partial \\mathbf{w}} = \\frac{1}{m} \\sum_{i=1}^{m} (h(\\mathbf{x}^{(i)}) - y^{(i)}) \\mathbf{x}^{(i)}$$\n"]),
        
        code(["class LogisticRegression:\n",
              "    def __init__(self, learning_rate=0.01, n_iterations=1000, regularization=None, lambda_=0.01):\n",
              "        self.lr = learning_rate\n        self.n_iters = n_iterations\n",
              "        self.reg = regularization\n        self.lambda_ = lambda_\n",
              "        self.weights = None\n        self.bias = None\n        self.cost_history = []\n    \n",
              "    def fit(self, X, y):\n",
              "        n_samples, n_features = X.shape\n",
              "        self.weights = np.zeros(n_features)\n        self.bias = 0\n        \n",
              "        for i in range(self.n_iters):\n",
              "            # Forward pass\n            z = X @ self.weights + self.bias\n",
              "            y_pred = sigmoid(z)\n            \n",
              "            # Compute cost\n            cost = -np.mean(y * np.log(y_pred + 1e-10) + (1-y) * np.log(1-y_pred + 1e-10))\n",
              "            \n            # Add regularization\n",
              "            if self.reg == 'l2':\n",
              "                cost += (self.lambda_ / (2 * n_samples)) * np.sum(self.weights ** 2)\n",
              "            elif self.reg == 'l1':\n",
              "                cost += (self.lambda_ / n_samples) * np.sum(np.abs(self.weights))\n",
              "            \n            self.cost_history.append(cost)\n            \n",
              "            # Compute gradients\n",
              "            dw = (1/n_samples) * (X.T @ (y_pred - y))\n",
              "            db = (1/n_samples) * np.sum(y_pred - y)\n            \n",
              "            # Add regularization gradient\n",
              "            if self.reg == 'l2':\n",
              "                dw += (self.lambda_ / n_samples) * self.weights\n",
              "            elif self.reg == 'l1':\n",
              "                dw += (self.lambda_ / n_samples) * np.sign(self.weights)\n            \n",
              "            # Update parameters\n",
              "            self.weights -= self.lr * dw\n",
              "            self.bias -= self.lr * db\n        \n",
              "        return self\n    \n",
              "    def predict_proba(self, X):\n",
              "        z = X @ self.weights + self.bias\n",
              "        return sigmoid(z)\n    \n",
              "    def predict(self, X, threshold=0.5):\n",
              "        return (self.predict_proba(X) >= threshold).astype(int)\n    \n",
              "    def score(self, X, y):\n",
              "        return accuracy_score(y, self.predict(X))\n"]),
        
        md(["---\n# Chapter 2: Application to Real Data\n"]),
        
        code(["# Load breast cancer dataset\n",
              "data = load_breast_cancer()\nX, y = data.data, data.target\n",
              "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n\n",
              "# Standardize\nscaler = StandardScaler()\n",
              "X_train_scaled = scaler.fit_transform(X_train)\n",
              "X_test_scaled = scaler.transform(X_test)\n\n",
              "# Train models\nmodels = {\n",
              "    'No Reg': LogisticRegression(learning_rate=0.1, n_iterations=1000),\n",
              "    'L2 Reg': LogisticRegression(learning_rate=0.1, n_iterations=1000, regularization='l2', lambda_=0.1),\n",
              "    'L1 Reg': LogisticRegression(learning_rate=0.1, n_iterations=1000, regularization='l1', lambda_=0.1)\n",
              "}\n\nresults = {}\n",
              "for name, model in models.items():\n",
              "    model.fit(X_train_scaled, y_train)\n",
              "    train_acc = model.score(X_train_scaled, y_train)\n",
              "    test_acc = model.score(X_test_scaled, y_test)\n",
              "    results[name] = {'train': train_acc, 'test': test_acc}\n",
              "    print(f'{name:10} | Train: {train_acc:.4f} | Test: {test_acc:.4f}')\n"]),
        
        code(["# ROC Curves\nfig, ax = plt.subplots(figsize=(10, 7))\n",
              "for name, model in models.items():\n",
              "    y_pred_proba = model.predict_proba(X_test_scaled)\n",
              "    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)\n",
              "    roc_auc = auc(fpr, tpr)\n",
              "    ax.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.3f})')\n\n",
              "ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random')\n",
              "ax.set_xlabel('False Positive Rate')\nax.set_ylabel('True Positive Rate')\n",
              "ax.set_title('ROC Curves: Logistic Regression Variants')\n",
              "ax.legend()\nax.grid(True, alpha=0.3)\nplt.show()\n"]),
        
        md(["---\n# Summary\n\n",
            "| Concept | Formula | Purpose |\n",
            "|---------|---------|----------|\n",
            "| Sigmoid | $\\sigma(z) = 1/(1+e^{-z})$ | Squeeze to [0,1] |\n",
            "| BCE Loss | $-y\\log(h) - (1-y)\\log(1-h)$ | Measure error |\n",
            "| Gradient | $(h-y)\\mathbf{x}$ | Update direction |\n",
            "| L2 Reg | $\\lambda \\|\\mathbf{w}\\|^2$ | Prevent overfitting |\n"]),
    ])

# Continue with more notebooks...
def create_knn_notebook():
    """K-Nearest Neighbors complete implementation."""
    return create_notebook([
        md(["# 🎯 K-Nearest Neighbors (KNN): Complete Implementation\n\n",
            "## Learning Objectives\n",
            "1. Understand distance metrics mathematically\n",
            "2. Implement KNN for classification & regression\n",
            "3. Optimize with k-d trees\n",
            "4. Visualize decision boundaries\n"]),
        code(["import numpy as np\nimport matplotlib.pyplot as plt\n",
              "from collections import Counter\n",
              "from sklearn.datasets import make_classification, make_moons\n",
              "from sklearn.model_selection import train_test_split\n",
              "print('✅ Setup complete!')\n"]),
        md(["---\n# Chapter 1: Distance Metrics\n\n",
            "## Euclidean Distance\n$$d(\\mathbf{x}, \\mathbf{y}) = \\sqrt{\\sum_{i=1}^{n} (x_i - y_i)^2}$$\n\n",
            "## Manhattan Distance\n$$d(\\mathbf{x}, \\mathbf{y}) = \\sum_{i=1}^{n} |x_i - y_i|$$\n\n",
            "## Minkowski Distance (generalization)\n$$d(\\mathbf{x}, \\mathbf{y}) = \\left(\\sum_{i=1}^{n} |x_i - y_i|^p\\right)^{1/p}$$\n"]),
        code(["class KNN:\n",
              "    def __init__(self, k=3, metric='euclidean', weighted=False):\n",
              "        self.k = k\n        self.metric = metric\n        self.weighted = weighted\n    \n",
              "    def _distance(self, x1, x2):\n",
              "        if self.metric == 'euclidean':\n",
              "            return np.sqrt(np.sum((x1 - x2) ** 2))\n",
              "        elif self.metric == 'manhattan':\n",
              "            return np.sum(np.abs(x1 - x2))\n",
              "        elif self.metric == 'minkowski':\n",
              "            p = 3\n            return np.sum(np.abs(x1 - x2) ** p) ** (1/p)\n    \n",
              "    def fit(self, X, y):\n",
              "        self.X_train = X\n        self.y_train = y\n        return self\n    \n",
              "    def predict(self, X):\n",
              "        return np.array([self._predict_single(x) for x in X])\n    \n",
              "    def _predict_single(self, x):\n",
              "        # Compute distances to all training samples\n",
              "        distances = [self._distance(x, x_train) for x_train in self.X_train]\n        \n",
              "        # Get k nearest neighbors\n",
              "        k_indices = np.argsort(distances)[:self.k]\n",
              "        k_nearest_labels = self.y_train[k_indices]\n        \n",
              "        if self.weighted:\n",
              "            k_distances = np.array(distances)[k_indices]\n",
              "            weights = 1 / (k_distances + 1e-10)\n",
              "            labels, counts = np.unique(k_nearest_labels, return_counts=True)\n",
              "            weighted_counts = {}\n",
              "            for i, label in enumerate(k_nearest_labels):\n",
              "                weighted_counts[label] = weighted_counts.get(label, 0) + weights[i]\n",
              "            return max(weighted_counts, key=weighted_counts.get)\n",
              "        else:\n",
              "            return Counter(k_nearest_labels).most_common(1)[0][0]\n"]),
        code(["# Test on moons dataset\nX, y = make_moons(n_samples=200, noise=0.1, random_state=42)\n",
              "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)\n\n",
              "knn = KNN(k=5)\nknn.fit(X_train, y_train)\n",
              "y_pred = knn.predict(X_test)\n",
              "acc = np.mean(y_pred == y_test)\nprint(f'Test Accuracy: {acc:.4f}')\n",
              "\n# Visualize decision boundary\nh = 0.02\n",
              "x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1\n",
              "y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1\n",
              "xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))\n",
              "Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])\n",
              "Z = Z.reshape(xx.shape)\n\n",
              "plt.figure(figsize=(10, 7))\n",
              "plt.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')\n",
              "plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolors='k')\n",
              "plt.title(f'KNN Decision Boundary (k={knn.k})')\nplt.show()\n"]),
    ])

# Generate all notebooks
def main():
    print("🚀 Generating Week 02 Comprehensive Notebooks...\n")
    
    notebooks = {
        "03_logistic_regression_complete.ipynb": create_logistic_regression_notebook(),
        "04_knn_complete.ipynb": create_knn_notebook(),
    }
    
    for filename, notebook in notebooks.items():
        output_path = BASE_DIR / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(notebook, f, indent=2)
        
        print(f"✅ Created: {filename}")
    
    print(f"\n🎉 Generated {len(notebooks)} notebooks successfully!")
    print(f"📂 Location: {BASE_DIR}")

if __name__ == "__main__":
    main()

