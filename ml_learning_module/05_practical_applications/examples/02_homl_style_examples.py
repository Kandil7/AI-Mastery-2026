"""
Hands-On Machine Learning Style Examples
========================================

Complete examples matching the style of "Hands-On Machine Learning" by Aurélien Géron.
This file demonstrates end-to-end ML workflows with pipelines, model selection, and ensembles.

Author: AI-Mastery-2026
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple


def example_end_to_end_classification():
    """
    End-to-End Classification Pipeline (HOML Chapter 3 style)

    Complete workflow:
    1. Data loading and exploration
    2. Prepare train/test sets
    3. Pipeline with preprocessing
    4. Model selection (GridSearch)
    5. Evaluate on test set
    """
    print("=" * 70)
    print("End-to-End Classification Pipeline")
    print("(Based on Hands-On Machine Learning)")
    print("=" * 70)

    # ========================================
    # Step 1: Generate and explore data
    # ========================================
    print("\n1. Data Generation and Exploration")
    print("-" * 40)

    np.random.seed(42)
    n_samples = 1000

    # Create synthetic dataset (similar to make_moons or make_classification)
    from sklearn.datasets import make_moons

    X, y = make_moons(n_samples=n_samples, noise=0.15, random_state=42)

    print(f"Dataset size: {X.shape}")
    print(f"Classes: {np.unique(y)}")
    print(f"Class distribution: {np.bincount(y)}")

    # ========================================
    # Step 2: Split data
    # ========================================
    print("\n2. Train/Test Split")
    print("-" * 40)

    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")

    # ========================================
    # Step 3: Build Pipeline
    # ========================================
    print("\n3. Build ML Pipeline")
    print("-" * 40)

    # Our Pipeline implementation
    from implementations.ml.pipeline import StandardScaler, Pipeline
    from implementations.ml.svm import SVM
    from implementations.ml.evaluation import Metrics

    # Create pipeline: Scale -> SVM
    pipeline = Pipeline(
        [("scaler", StandardScaler()), ("svm", SVM(kernel="rbf", C=1.0, gamma=1.0))]
    )

    print("Pipeline: StandardScaler -> SVM(RBF)")

    # ========================================
    # Step 4: Simple Grid Search
    # ========================================
    print("\n4. Model Selection (Grid Search)")
    print("-" * 40)

    # Quick grid search over a few parameters
    best_score = 0
    best_params = {}

    for C in [0.1, 1.0, 10.0]:
        for gamma in [0.5, 1.0, 2.0]:
            svm = SVM(kernel="rbf", C=C, gamma=gamma)

            # Quick validation
            split = int(0.8 * len(y_train))
            svm.fit(X_train[:split], y_train[:split])
            score = svm.score(X_train[split:], y_train[split:])

            print(f"  C={C}, gamma={gamma}: {score:.4f}")

            if score > best_score:
                best_score = score
                best_params = {"C": C, "gamma": gamma}

    print(f"\nBest params: {best_params} with score: {best_score:.4f}")

    # ========================================
    # Step 5: Train on full training set
    # ========================================
    print("\n5. Final Training")
    print("-" * 40)

    final_svm = SVM(kernel="rbf", **best_params)
    final_svm.fit(X_train, y_train)

    train_score = final_svm.score(X_train, y_train)
    test_score = final_svm.score(X_test, y_test)

    print(f"Training accuracy: {train_score:.4f}")
    print(f"Test accuracy: {test_score:.4f}")

    # ========================================
    # Step 6: Visualize results
    # ========================================
    print("\n6. Visualization")
    print("-" * 40)

    try:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Plot decision boundary
        h = 0.02
        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

        Z = final_svm.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        axes[0].contourf(xx, yy, Z, alpha=0.4, cmap="RdYlBu")
        axes[0].scatter(
            X_test[y_test == 0, 0],
            X_test[y_test == 0, 1],
            c="blue",
            marker="o",
            label="Class 0",
            alpha=0.6,
        )
        axes[0].scatter(
            X_test[y_test == 1, 0],
            X_test[y_test == 1, 1],
            c="red",
            marker="s",
            label="Class 1",
            alpha=0.6,
        )
        axes[0].set_title(f"SVM Decision Boundary (Test Acc: {test_score:.2%})")
        axes[0].set_xlabel("Feature 1")
        axes[0].set_ylabel("Feature 2")
        axes[0].legend()

        # Confusion matrix visualization
        y_pred = final_svm.predict(X_test)

        cm = Metrics.confusion_matrix(y_test, y_pred)

        im = axes[1].imshow(cm, interpolation="nearest", cmap="Blues")
        axes[1].set_title("Confusion Matrix")
        axes[1].set_xlabel("Predicted")
        axes[1].set_ylabel("Actual")

        # Add text annotations
        for i in range(2):
            for j in range(2):
                axes[1].text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=14)

        plt.tight_layout()
        plt.show()

    except ImportError:
        print("Matplotlib not available for visualization")


def example_ensemble_comparison():
    """
    Compare Different Ensemble Methods (HOML Chapter 7)

    Compares:
    - Voting (Hard/Soft)
    - Random Forest
    - Gradient Boosting
    """
    print("\n" + "=" * 70)
    print("Ensemble Methods Comparison")
    print("=" * 70)

    # Generate data
    np.random.seed(42)
    from sklearn.datasets import make_blobs

    X, y = make_blobs(n_samples=500, centers=3, random_state=42)

    print(f"Dataset: {X.shape}, {len(np.unique(y))} classes")

    # Note: In full implementation, would train multiple models
    # Here we demonstrate the concept

    print("\nEnsemble Methods:")
    print("-" * 40)
    print("""
    1. Voting Classifier:
       - Hard voting: Majority vote
       - Soft voting: Average probabilities
    
    2. Random Forest:
       - Bagging + random feature selection
       - Good for high-dimensional data
    
    3. Gradient Boosting:
       - Sequential error correction
       - Often best accuracy
    
    4. Stacking:
       - Meta-learner on base predictions
       - Most flexible
    """)


def example_model_selection():
    """
    Model Selection with Cross-Validation (HOML Chapter 2)
    """
    print("\n" + "=" * 70)
    print("Model Selection with Cross-Validation")
    print("=" * 70)

    # Generate data
    np.random.seed(42)
    X = np.random.randn(300, 5)
    y = (X[:, 0] + 0.5 * X[:, 1] - X[:, 2] > 0).astype(int)

    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")

    # Cross-validation manually
    from implementations.ml.evaluation import CrossValidator

    print("\n5-Fold Cross-Validation:")
    print("-" * 40)

    splits = CrossValidator.k_fold_split(
        len(y), n_splits=5, shuffle=True, random_state=42
    )

    for fold, (train_idx, val_idx) in enumerate(splits):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        # Train simple model
        from implementations.ml.linear_regression import LogisticRegression

        model = LogisticRegression(learning_rate=0.1, n_epochs=500)
        model.fit(X_tr, y_tr)

        score = model.score(X_val, y_val)
        print(f"  Fold {fold + 1}: {score:.4f}")


def example_preprocessing():
    """
    Data Preprocessing Pipelines (HOML Chapter 2)
    """
    print("\n" + "=" * 70)
    print("Data Preprocessing Pipelines")
    print("=" * 70)

    # Generate data with different scales
    np.random.seed(42)
    X = np.random.randn(100, 3)

    # Feature 1: Normal (mean=0, std=1)
    # Feature 2: Uniform [0, 10]
    X[:, 1] = np.random.uniform(0, 10, 100)
    # Feature 3: Larger values
    X[:, 2] = X[:, 2] * 100

    print("Before Scaling:")
    print(f"  Feature means: {X.mean(axis=0).round(2)}")
    print(f"  Feature stds: {X.std(axis=0).round(2)}")

    # StandardScaler
    from implementations.ml.pipeline import StandardScaler, MinMaxScaler

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("\nAfter StandardScaler (z-score):")
    print(f"  Feature means: {X_scaled.mean(axis=0).round(4)}")
    print(f"  Feature stds: {X_scaled.std(axis=0).round(4)}")

    # MinMaxScaler
    minmax = MinMaxScaler(feature_range=(0, 1))
    X_minmax = minmax.fit_transform(X)

    print("\nAfter MinMaxScaler (0-1):")
    print(
        f"  Feature range: [{X_minmax.min(axis=0).round(2)}, {X_minmax.max(axis=0).round(2)}]"
    )


def run_all_examples():
    """Run all HOML-style examples"""
    example_end_to_end_classification()
    example_ensemble_comparison()
    example_model_selection()
    example_preprocessing()


if __name__ == "__main__":
    run_all_examples()

    print("\n" + "=" * 70)
    print("All Hands-On ML examples completed!")
    print("=" * 70)
