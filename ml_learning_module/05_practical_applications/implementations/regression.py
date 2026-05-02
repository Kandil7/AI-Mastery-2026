"""
Regression Pipeline Implementation
==================================

Complete regression pipeline with multiple algorithms
and evaluation metrics.

Author: AI-Mastery-2026
"""

import numpy as np
from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass


@dataclass
class RegressionResult:
    """Store regression results"""

    predictions: np.ndarray
    mse: float
    rmse: float
    mae: float
    r2_score: float
    metrics: Dict


class LinearRegressionModel:
    """
    Linear Regression using Normal Equation

    Solves: w = (X^T X)^(-1) X^T y

    For multiple features with polynomial terms.
    """

    def __init__(self, include_bias: bool = True):
        self.include_bias = include_bias
        self.weights = None
        self.bias = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearRegressionModel":
        """Fit using normal equation"""
        if self.include_bias:
            X = np.column_stack([np.ones(len(X)), X])

        # Normal equation: w = (X^T X)^(-1) X^T y
        XTX = X.T @ X
        XTX_inv = np.linalg.inv(XTX + 1e-10 * np.eye(XTX.shape[0]))
        self.weights = XTX_inv @ X.T @ y

        if self.include_bias:
            self.bias = self.weights[0]
            self.weights = self.weights[1:]

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        return np.dot(X, self.weights) + self.bias


class RidgeRegression:
    """
    Ridge Regression (L2 Regularized Linear Regression)

    Solves: w = (X^T X + λI)^(-1) X^T y

    Where λ is the regularization parameter.
    """

    def __init__(self, alpha: float = 1.0, include_bias: bool = True):
        self.alpha = alpha
        self.include_bias = include_bias
        self.weights = None
        self.bias = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RidgeRegression":
        """Fit using closed-form solution"""
        if self.include_bias:
            X = np.column_stack([np.ones(len(X)), X])
            n_features = X.shape[1]
        else:
            n_features = X.shape[1]

        # Ridge: w = (X^T X + λI)^(-1) X^T y
        XTX = X.T @ X
        regularization = self.alpha * np.eye(n_features)

        if self.include_bias:
            regularization[0, 0] = 0  # Don't regularize bias

        XTX_reg = XTX + regularization
        XTX_inv = np.linalg.inv(XTX_reg + 1e-10 * np.eye(n_features))
        self.weights = XTX_inv @ X.T @ y

        if self.include_bias:
            self.bias = self.weights[0]
            self.weights = self.weights[1:]

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        return np.dot(X, self.weights) + self.bias


class PolynomialRegression:
    """
    Polynomial Regression

    Extends linear regression by adding polynomial features.
    """

    def __init__(self, degree: int = 2, include_bias: bool = True):
        self.degree = degree
        self.include_bias = include_bias
        self.linear_model = LinearRegressionModel(include_bias)

    def _create_polynomial_features(self, X: np.ndarray) -> np.ndarray:
        """Create polynomial features"""
        n_samples = X.shape[0]
        n_features = X.shape[1]

        # Start with original features
        poly_features = [X]

        # Add higher degree features
        for d in range(2, self.degree + 1):
            for i in range(n_features):
                poly_features.append(X[:, i : i + 1] ** d)

        return np.hstack(poly_features)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PolynomialRegression":
        """Fit polynomial regression"""
        X_poly = self._create_polynomial_features(X)
        self.linear_model.fit(X_poly, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        X_poly = self._create_polynomial_features(X)
        return self.linear_model.predict(X_poly)


class GradientBoostingRegressor:
    """
    Gradient Boosting Regressor

    Ensemble method that builds trees sequentially,
    each tree correcting the errors of the previous ones.

    Algorithm:
        1. Initialize with mean prediction
        2. For m = 1 to n_estimators:
           a. Compute pseudo-residuals
           b. Fit tree to residuals
           c. Update model
    """

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 3,
        min_samples_split: int = 2,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.trees = []
        self.initial_prediction = None

    def _build_tree(self, X: np.ndarray, residuals: np.ndarray) -> dict:
        """Build a simple decision tree"""
        n_samples = len(residuals)

        # If too few samples, return leaf
        if n_samples < self.min_samples_split:
            return {"value": np.mean(residuals)}

        # Find best split (simplified - use variance reduction)
        best_gain = -float("inf")
        best_feature = 0
        best_threshold = 0

        n_features = X.shape[1]

        for feature in range(min(n_features, 10)):  # Limit features
            thresholds = np.percentile(X[:, feature], [25, 50, 75])

            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask

                if np.sum(left_mask) < 1 or np.sum(right_mask) < 1:
                    continue

                # Variance reduction
                var_total = np.var(residuals)
                var_left = np.var(residuals[left_mask])
                var_right = np.var(residuals[right_mask])

                gain = (
                    var_total
                    - (np.sum(left_mask) * var_left + np.sum(right_mask) * var_right)
                    / n_samples
                )

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold

        if best_gain <= 0:
            return {"value": np.mean(residuals)}

        # Split
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask

        left_tree = self._build_tree(X[left_mask], residuals[left_mask])
        right_tree = self._build_tree(X[right_mask], residuals[right_mask])

        return {
            "feature": best_feature,
            "threshold": best_threshold,
            "left": left_tree,
            "right": right_tree,
        }

    def _predict_tree(self, x: np.ndarray, node: dict) -> float:
        """Predict using tree"""
        if "value" in node:
            return node["value"]

        if x[node["feature"]] <= node["threshold"]:
            return self._predict_tree(x, node["left"])
        else:
            return self._predict_tree(x, node["right"])

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GradientBoostingRegressor":
        """Fit gradient boosting"""
        n_samples = X.shape[0]

        # Initialize with mean
        self.initial_prediction = np.mean(y)

        # Current predictions
        predictions = np.full(n_samples, self.initial_prediction)

        # Build trees
        for i in range(self.n_estimators):
            # Compute residuals
            residuals = y - predictions

            # Build tree
            tree = self._build_tree(X, residuals)
            self.trees.append(tree)

            # Update predictions
            tree_preds = np.array([self._predict_tree(x, tree) for x in X])
            predictions += self.learning_rate * tree_preds

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        n_samples = X.shape[0]
        predictions = np.full(n_samples, self.initial_prediction)

        for tree in self.trees:
            tree_preds = np.array([self._predict_tree(x, tree) for x in X])
            predictions += self.learning_rate * tree_preds

        return predictions


def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> RegressionResult:
    """Evaluate regression results"""

    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_true - y_pred))

    # R² Score
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    return RegressionResult(
        predictions=y_pred,
        mse=mse,
        rmse=rmse,
        mae=mae,
        r2_score=r2,
        metrics={"mse": mse, "rmse": rmse, "mae": mae, "r2": r2},
    )


def demo_regression():
    """Demonstrate regression algorithms"""
    print("=" * 60)
    print("Regression Demo")
    print("=" * 60)

    # Create sample data
    np.random.seed(42)
    n_samples = 200

    # Generate data with noise
    X = np.sort(np.random.rand(n_samples) * 10)
    y = 3 * X**2 - 5 * X + 2 + np.random.randn(n_samples) * 10

    X = X.reshape(-1, 1)

    # Split
    split = int(0.8 * len(y))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    print(f"Training samples: {len(y_train)}")
    print(f"Test samples: {len(y_test)}")

    # ========================================
    # Demo 1: Linear Regression
    # ========================================
    print("\n--- Linear Regression ---")

    lr = LinearRegressionModel()
    lr.fit(X_train, y_train)

    y_pred_lr = lr.predict(X_test)
    result_lr = evaluate_regression(y_test, y_pred_lr)

    print(f"MSE:  {result_lr.mse:.4f}")
    print(f"RMSE: {result_lr.rmse:.4f}")
    print(f"MAE:  {result_lr.mae:.4f}")
    print(f"R²:   {result_lr.r2_score:.4f}")

    # ========================================
    # Demo 2: Polynomial Regression
    # ========================================
    print("\n--- Polynomial Regression (degree=2) ---")

    poly = PolynomialRegression(degree=2)
    poly.fit(X_train, y_train)

    y_pred_poly = poly.predict(X_test)
    result_poly = evaluate_regression(y_test, y_pred_poly)

    print(f"MSE:  {result_poly.mse:.4f}")
    print(f"RMSE: {result_poly.rmse:.4f}")
    print(f"R²:   {result_poly.r2_score:.4f}")

    # ========================================
    # Demo 3: Ridge Regression
    # ========================================
    print("\n--- Ridge Regression ---")

    ridge = RidgeRegression(alpha=1.0)
    ridge.fit(X_train, y_train)

    y_pred_ridge = ridge.predict(X_test)
    result_ridge = evaluate_regression(y_test, y_pred_ridge)

    print(f"MSE:  {result_ridge.mse:.4f}")
    print(f"R²:   {result_ridge.r2_score:.4f}")

    # ========================================
    # Demo 4: Gradient Boosting
    # ========================================
    print("\n--- Gradient Boosting ---")

    gb = GradientBoostingRegressor(n_estimators=50, max_depth=3)
    gb.fit(X_train, y_train)

    y_pred_gb = gb.predict(X_test)
    result_gb = evaluate_regression(y_test, y_pred_gb)

    print(f"MSE:  {result_gb.mse:.4f}")
    print(f"R²:   {result_gb.r2_score:.4f}")

    # ========================================
    # Summary
    # ========================================
    print("\n--- Summary ---")
    print(f"Linear Regression:   R² = {result_lr.r2_score:.4f}")
    print(f"Polynomial (deg=2):   R² = {result_poly.r2_score:.4f}")
    print(f"Ridge:               R² = {result_ridge.r2_score:.4f}")
    print(f"Gradient Boosting:    R² = {result_gb.r2_score:.4f}")


if __name__ == "__main__":
    demo_regression()

    print("\n" + "=" * 60)
    print("Regression complete!")
    print("=" * 60)
