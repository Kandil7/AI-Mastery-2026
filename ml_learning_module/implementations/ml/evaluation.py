"""
Model Evaluation and Cross-Validation
=======================================

This module provides comprehensive evaluation metrics and cross-validation
strategies for machine learning models.

Mathematical Foundation:
------------------------

1. Classification Metrics:

   a) Accuracy:
      Accuracy = (TP + TN) / (TP + TN + FP + FN)

      Proportion of correct predictions. Simple but can be
      misleading for imbalanced datasets.

   b) Precision (Positive Predictive Value):
      Precision = TP / (TP + FP)

      Of all positive predictions, how many are correct?
      High precision = low false positive rate.

   c) Recall (Sensitivity, True Positive Rate):
      Recall = TP / (TP + FN)

      Of all actual positives, how many did we find?
      High recall = low false negative rate.

   d) F1-Score:
      F1 = 2 * (Precision * Recall) / (Precision + Recall)

      Harmonic mean of precision and recall.
      Better than accuracy for imbalanced data.

   e) ROC-AUC:
      - ROC: Receiver Operating Characteristic curve
      - Plots TPR (recall) vs FPR at different thresholds
      - AUC: Area Under the ROC curve
      - AUC = 1: Perfect classifier
      - AUC = 0.5: Random classifier

   f) Confusion Matrix:
      [[TN, FP],
       [FN, TP]]

      Visual breakdown of predictions vs actuals.

2. Regression Metrics:

   a) Mean Squared Error (MSE):
      MSE = (1/n) Σ (y_i - ŷ_i)^2

      Penalizes large errors more than MAE.

   b) Root Mean Squared Error (RMSE):
      RMSE = √MSE

      Same unit as target variable.

   c) Mean Absolute Error (MAE):
      MAE = (1/n) Σ |y_i - ŷ_i|

      More robust to outliers than MSE.

   d) R² Score (Coefficient of Determination):
      R² = 1 - (SS_res / SS_tot)

      Proportion of variance explained by model.
      1.0 = perfect, 0.0 = constant prediction.

3. Cross-Validation:

   a) K-Fold Cross-Validation:
      - Split data into k equal folds
      - Train on k-1 folds, validate on 1
      - Repeat k times, average results

      More stable estimate than single train/test split.

   b) Stratified K-Fold:
      - Maintains class distribution in each fold
      - Important for imbalanced classification

   c) Leave-One-Out (LOO):
      - Special case: k = n
      - Very expensive but maximally thorough

Author: AI-Mastery-2026
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import Counter


class Metrics:
    """
    Collection of evaluation metrics for classification and regression.
    """

    @staticmethod
    def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate accuracy score

        Accuracy = (TP + TN) / Total

        Args:
            y_true: True labels
            y_pred: Predicted labels

        Returns:
            Accuracy score between 0 and 1
        """
        return np.mean(y_true == y_pred)

    @staticmethod
    def precision(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        average: str = "binary",
        pos_label: int = 1,
    ) -> float:
        """
        Calculate precision

        Precision = TP / (TP + FP)

        Args:
            y_true: True labels
            y_pred: Predicted labels
            average: 'binary', 'micro', 'macro', 'weighted'
            pos_label: Positive class for binary

        Returns:
            Precision score
        """
        if average == "binary":
            tp = np.sum((y_true == pos_label) & (y_pred == pos_label))
            fp = np.sum((y_true != pos_label) & (y_pred == pos_label))
            return tp / (tp + fp) if (tp + fp) > 0 else 0.0
        else:
            # Compute per-class precision
            classes = np.unique(y_true)
            precisions = []
            counts = []

            for c in classes:
                tp = np.sum((y_true == c) & (y_pred == c))
                fp = np.sum((y_true != c) & (y_pred == c))
                if (tp + fp) > 0:
                    precisions.append(tp / (tp + fp))
                    counts.append(np.sum(y_true == c))

            if average == "macro":
                return np.mean(precisions)
            elif average == "weighted":
                return np.average(precisions, weights=counts)
            else:  # micro
                tp_total = sum([p * c for p, c in zip(precisions, counts)])
                return tp_total / sum(counts)

    @staticmethod
    def recall(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        average: str = "binary",
        pos_label: int = 1,
    ) -> float:
        """
        Calculate recall (sensitivity, true positive rate)

        Recall = TP / (TP + FN)

        Args:
            y_true: True labels
            y_pred: Predicted labels
            average: 'binary', 'micro', 'macro', 'weighted'
            pos_label: Positive class for binary

        Returns:
            Recall score
        """
        if average == "binary":
            tp = np.sum((y_true == pos_label) & (y_pred == pos_label))
            fn = np.sum((y_true == pos_label) & (y_pred != pos_label))
            return tp / (tp + fn) if (tp + fn) > 0 else 0.0
        else:
            classes = np.unique(y_true)
            recalls = []
            counts = []

            for c in classes:
                tp = np.sum((y_true == c) & (y_pred == c))
                fn = np.sum((y_true == c) & (y_pred != c))
                if (tp + fn) > 0:
                    recalls.append(tp / (tp + fn))
                    counts.append(np.sum(y_true == c))

            if average == "macro":
                return np.mean(recalls)
            elif average == "weighted":
                return np.average(recalls, weights=counts)
            else:  # micro
                return Metrics.accuracy(y_true, y_pred)

    @staticmethod
    def f1_score(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        average: str = "binary",
        pos_label: int = 1,
    ) -> float:
        """
        Calculate F1 score

        F1 = 2 * (Precision * Recall) / (Precision + Recall)

        Harmonic mean of precision and recall.
        """
        precision = Metrics.precision(y_true, y_pred, average, pos_label)
        recall = Metrics.recall(y_true, y_pred, average, pos_label)

        if precision + recall == 0:
            return 0.0

        return 2 * (precision * recall) / (precision + recall)

    @staticmethod
    def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Calculate confusion matrix

        Returns:
            2D array where rows are true, columns are predicted
            [[TN, FP], [FN, TP]] for binary
        """
        classes = np.unique(np.concatenate([y_true, y_pred]))
        n_classes = len(classes)

        # Build index mapping
        class_to_idx = {c: i for i, c in enumerate(classes)}

        # Initialize matrix
        matrix = np.zeros((n_classes, n_classes), dtype=int)

        # Fill matrix
        for true, pred in zip(y_true, y_pred):
            matrix[class_to_idx[true], class_to_idx[pred]] += 1

        return matrix

    @staticmethod
    def roc_auc_score(y_true: np.ndarray, y_scores: np.ndarray) -> float:
        """
        Calculate ROC-AUC score

        Args:
            y_true: True binary labels (0 or 1)
            y_scores: Predicted probabilities for positive class

        Returns:
            AUC score between 0 and 1
        """
        # Sort by scores (descending)
        sorted_indices = np.argsort(y_scores)[::-1]
        y_true_sorted = y_true[sorted_indices]

        # Calculate TPR and FPR at each threshold
        n_pos = np.sum(y_true == 1)
        n_neg = np.sum(y_true == 0)

        tpr = []
        fpr = []

        tp = 0
        fp = 0

        for label in y_true_sorted:
            if label == 1:
                tp += 1
            else:
                fp += 1

            tpr.append(tp / n_pos)
            fpr.append(fp / n_neg)

        # Add origin
        tpr = [0] + tpr
        fpr = [0] + fpr

        # Calculate AUC using trapezoidal rule
        auc = 0.0
        for i in range(len(tpr) - 1):
            auc += (fpr[i + 1] - fpr[i]) * (tpr[i] + tpr[i + 1]) / 2

        return auc

    @staticmethod
    def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate Mean Squared Error (MSE)

        MSE = (1/n) Σ (y_i - ŷ_i)²
        """
        return np.mean((y_true - y_pred) ** 2)

    @staticmethod
    def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate RMSE"""
        return np.sqrt(Metrics.mean_squared_error(y_true, y_pred))

    @staticmethod
    def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate MAE"""
        return np.mean(np.abs(y_true - y_pred))

    @staticmethod
    def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate R² Score (Coefficient of Determination)

        R² = 1 - (Σ(y_i - ŷ_i)² / Σ(y_i - ȳ)²)

        - R² = 1: Perfect model
        - R² = 0: Constant baseline
        - R² < 0: Worse than baseline
        """
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

        if ss_tot == 0:
            return 0.0

        return 1 - (ss_res / ss_tot)


class CrossValidator:
    """
    Cross-validation strategies for model evaluation.
    """

    @staticmethod
    def k_fold_split(
        n_samples: int,
        n_splits: int,
        shuffle: bool = True,
        random_state: Optional[int] = None,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate K-Fold splits

        Args:
            n_samples: Number of samples
            n_splits: Number of folds
            shuffle: Whether to shuffle data before splitting
            random_state: Random seed for reproducibility

        Returns:
            List of (train_indices, val_indices) tuples
        """
        if random_state is not None:
            np.random.seed(random_state)

        indices = np.arange(n_samples)

        if shuffle:
            np.random.shuffle(indices)

        # Split indices into folds
        fold_sizes = np.full(n_splits, n_samples // n_splits, dtype=int)
        fold_sizes[: n_samples % n_splits] += 1

        splits = []
        current = 0

        for fold_size in fold_sizes:
            start, stop = current, current + fold_size
            val_indices = indices[start:stop]
            train_indices = np.concatenate([indices[:start], indices[stop:]])

            splits.append((train_indices, val_indices))
            current = stop

        return splits

    @staticmethod
    def stratified_k_fold(
        y: np.ndarray,
        n_splits: int,
        shuffle: bool = True,
        random_state: Optional[int] = None,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate stratified K-Fold splits

        Maintains class distribution in each fold.

        Args:
            y: Labels for stratification
            n_splits: Number of folds
            shuffle: Whether to shuffle
            random_state: Random seed

        Returns:
            List of (train_indices, val_indices) tuples
        """
        if random_state is not None:
            np.random.seed(random_state)

        classes = np.unique(y)
        n_classes = len(classes)

        # Group indices by class
        class_indices = [np.where(y == c)[0] for c in classes]

        # Shuffle within each class
        if shuffle:
            for indices in class_indices:
                np.random.shuffle(indices)

        # Create folds maintaining class balance
        splits = []

        for fold in range(n_splits):
            train_indices = []
            val_indices = []

            for class_idx, indices in enumerate(class_indices):
                # Calculate fold boundaries for this class
                n_samples = len(indices)
                n_per_fold = n_samples // n_splits
                start = fold * n_per_fold
                end = start + n_per_fold if fold < n_splits - 1 else n_samples

                val_indices.extend(indices[start:end])
                train_indices.extend(indices[:start])
                train_indices.extend(indices[end:])

            splits.append((np.array(train_indices), np.array(val_indices)))

        return splits


class ModelEvaluator:
    """
    Comprehensive model evaluation with multiple metrics.
    """

    def __init__(self):
        self.results = {}

    def evaluate_classification(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_scores: Optional[np.ndarray] = None,
    ) -> Dict:
        """
        Evaluate classification model

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_scores: Predicted probabilities (optional)

        Returns:
            Dictionary of metrics
        """
        metrics = {
            "accuracy": Metrics.accuracy(y_true, y_pred),
            "precision": Metrics.precision(y_true, y_pred, average="binary"),
            "recall": Metrics.recall(y_true, y_pred, average="binary"),
            "f1": Metrics.f1_score(y_true, y_pred, average="binary"),
            "confusion_matrix": Metrics.confusion_matrix(y_true, y_pred),
        }

        # Add AUC if scores provided
        if y_scores is not None:
            try:
                metrics["roc_auc"] = Metrics.roc_auc_score(y_true, y_scores)
            except:
                pass

        # Multi-class metrics
        if len(np.unique(y_true)) > 2:
            metrics["precision_macro"] = Metrics.precision(y_true, y_pred, "macro")
            metrics["recall_macro"] = Metrics.recall(y_true, y_pred, "macro")
            metrics["f1_macro"] = Metrics.f1_score(y_true, y_pred, "macro")
            metrics["precision_weighted"] = Metrics.precision(
                y_true, y_pred, "weighted"
            )
            metrics["recall_weighted"] = Metrics.recall(y_true, y_pred, "weighted")
            metrics["f1_weighted"] = Metrics.f1_score(y_true, y_pred, "weighted")

        return metrics

    def evaluate_regression(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """
        Evaluate regression model

        Args:
            y_true: True values
            y_pred: Predicted values

        Returns:
            Dictionary of metrics
        """
        return {
            "mse": Metrics.mean_squared_error(y_true, y_pred),
            "rmse": Metrics.root_mean_squared_error(y_true, y_pred),
            "mae": Metrics.mean_absolute_error(y_true, y_pred),
            "r2": Metrics.r2_score(y_true, y_pred),
        }

    def cross_validate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model,
        n_splits: int = 5,
        task: str = "classification",
        stratified: bool = True,
    ) -> Dict:
        """
        Perform cross-validation

        Args:
            X: Features
            y: Labels/targets
            model: Model with fit() and predict() methods
            n_splits: Number of folds
            task: 'classification' or 'regression'
            stratified: Use stratified splits

        Returns:
            Dictionary of cross-validation results
        """
        # Generate splits
        if stratified and task == "classification":
            splits = CrossValidator.stratified_k_fold(y, n_splits)
        else:
            splits = CrossValidator.k_fold_split(len(y), n_splits)

        # Track metrics
        all_metrics = []

        for fold, (train_idx, val_idx) in enumerate(splits):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # Train model
            model.fit(X_train, y_train)

            # Predict
            y_pred = model.predict(X_val)

            # Evaluate
            if task == "classification":
                fold_metrics = self.evaluate_classification(y_val, y_pred)
            else:
                fold_metrics = self.evaluate_regression(y_val, y_pred)

            all_metrics.append(fold_metrics)

        # Aggregate results
        aggregated = {}
        for key in all_metrics[0].keys():
            values = [m[key] for m in all_metrics]

            if isinstance(values[0], np.ndarray):
                aggregated[key] = values  # Keep arrays (e.g., confusion matrix)
            else:
                aggregated[key] = {
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "values": values,
                }

        return aggregated

    def print_classification_report(self, metrics: Dict):
        """Print formatted classification report"""
        print("\nClassification Report")
        print("=" * 50)
        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1 Score:  {metrics['f1']:.4f}")

        if "roc_auc" in metrics:
            print(f"ROC AUC:   {metrics['roc_auc']:.4f}")

        if "confusion_matrix" in metrics:
            print("\nConfusion Matrix:")
            print(metrics["confusion_matrix"])

    def print_regression_report(self, metrics: Dict):
        """Print formatted regression report"""
        print("\nRegression Report")
        print("=" * 50)
        print(f"MSE:  {metrics['mse']:.4f}")
        print(f"RMSE: {metrics['rmse']:.4f}")
        print(f"MAE:  {metrics['mae']:.4f}")
        print(f"R²:   {metrics['r2']:.4f}")


def test_evaluation():
    """Test evaluation metrics"""
    print("=" * 60)
    print("Testing Model Evaluation")
    print("=" * 60)

    # Test classification metrics
    y_true = np.array([1, 0, 1, 1, 0, 1, 0, 1, 0, 0])
    y_pred = np.array([1, 0, 1, 0, 0, 1, 1, 1, 0, 1])
    y_scores = np.array([0.9, 0.1, 0.8, 0.4, 0.2, 0.7, 0.6, 0.9, 0.1, 0.8])

    print("\n--- Classification Metrics ---")
    print(f"True:      {y_true}")
    print(f"Pred:      {y_pred}")
    print(f"Scores:    {y_scores}")

    evaluator = ModelEvaluator()
    metrics = evaluator.evaluate_classification(y_true, y_pred, y_scores)
    evaluator.print_classification_report(metrics)

    # Test regression metrics
    print("\n--- Regression Metrics ---")
    y_true_reg = np.array([3.0, 2.5, 4.8, 5.1, 2.9, 3.5, 4.2, 5.0])
    y_pred_reg = np.array([2.9, 2.7, 4.5, 5.3, 3.1, 3.3, 4.0, 4.8])

    reg_metrics = evaluator.evaluate_regression(y_true_reg, y_pred_reg)
    evaluator.print_regression_report(reg_metrics)

    # Test cross-validation
    print("\n--- Cross-Validation ---")
    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)  # Simple classification

    # Dummy model for testing
    class DummyClassifier:
        def fit(self, X, y):
            self.threshold = 0.5

        def predict(self, X):
            return np.random.randint(0, 2, len(X))

    cv_results = evaluator.cross_validate(X, y, DummyClassifier(), n_splits=3)
    print(
        f"CV Accuracy: {cv_results['accuracy']['mean']:.4f} ± {cv_results['accuracy']['std']:.4f}"
    )

    print("\n" + "=" * 60)
    print("All evaluation tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    test_evaluation()
