"""
ML Pipeline and Model Selection
================================

Complete pipeline utilities and model selection tools
based on "Hands-On Machine Learning" concepts.

Author: AI-Mastery-2026
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Callable, Any
from dataclasses import dataclass
import time


class StandardScaler:
    """
    StandardScaler: Standardize features by removing mean and scaling to unit variance.

    z = (x - mean) / std

    Similar to sklearn.preprocessing.StandardScaler
    """

    def __init__(self):
        self.mean_ = None
        self.scale_ = None
        self.n_features_in_ = None

    def fit(self, X: np.ndarray) -> "StandardScaler":
        """Compute mean and std for scaling"""
        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0)
        self.scale_[self.scale_ == 0] = 1  # Prevent division by zero
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Scale features"""
        return (X - self.mean_) / self.scale_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform in one step"""
        return self.fit(X).transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Reverse scaling"""
        return X * self.scale_ + self.mean_


class MinMaxScaler:
    """
    MinMaxScaler: Scale features to a given range (default [0, 1])

    z = (x - min) / (max - min)
    """

    def __init__(self, feature_range: Tuple[float, float] = (0, 1)):
        self.feature_range = feature_range
        self.min_ = None
        self.max_ = None
        self.scale_ = None

    def fit(self, X: np.ndarray) -> "MinMaxScaler":
        """Compute min and max for scaling"""
        self.min_ = np.min(X, axis=0)
        self.max_ = np.max(X, axis=0)
        self.scale_ = (self.feature_range[1] - self.feature_range[0]) / (
            self.max_ - self.min_ + 1e-10
        )
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Scale features"""
        return self.feature_range[0] + (X - self.min_) * self.scale_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


class Pipeline:
    """
    Pipeline: Chain multiple transformations and a final estimator.

    Similar to sklearn.pipeline.Pipeline

    Example:
        >>> pipeline = Pipeline([
        ...     ('scaler', StandardScaler()),
        ...     ('classifier', LogisticRegression())
        ... ])
        >>> pipeline.fit(X_train, y_train)
        >>> predictions = pipeline.predict(X_test)
    """

    def __init__(self, steps: List[Tuple[str, Any]]):
        """
        Initialize Pipeline.

        Args:
            steps: List of (name, transformer/estimator) tuples
        """
        self.steps = steps
        self.named_steps = {name: step for name, step in steps}

        # Fit each step in order
        for i, (name, step) in enumerate(steps):
            if not hasattr(step, "fit"):
                raise ValueError(f"Step {name} has no fit method")

    def fit(self, X: np.ndarray, y: np.ndarray = None) -> "Pipeline":
        """Fit the pipeline"""
        X_transformed = X

        for name, step in self.steps:
            if hasattr(step, "fit_transform"):
                X_transformed = step.fit_transform(X_transformed, y)
            elif hasattr(step, "fit"):
                if y is not None and hasattr(step, "predict"):
                    step.fit(X_transformed, y)
                else:
                    step.fit(X_transformed, y)

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data through pipeline"""
        for name, step in self.steps[:-1]:
            if hasattr(step, "transform"):
                X = step.transform(X)
        return X

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using the pipeline"""
        X_transformed = self.transform(X)
        final_step = self.steps[-1][1]

        if hasattr(final_step, "predict"):
            return final_step.predict(X_transformed)
        return X_transformed

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute score"""
        predictions = self.predict(X)
        return np.mean(predictions == y)


@dataclass
class GridSearchResult:
    """Store grid search results"""

    best_params: Dict
    best_score: float
    cv_results: Dict
    best_estimator: Any


class GridSearchCV:
    """
    GridSearchCV: Exhaustive search over specified parameter values.

    Similar to sklearn.model_selection.GridSearchCV

    Example:
        >>> param_grid = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
        >>> grid = GridSearchCV(SVM(), param_grid, cv=3)
        >>> grid.fit(X, y)
        >>> print(grid.best_params_)
    """

    def __init__(
        self,
        estimator: Any,
        param_grid: Dict,
        cv: int = 3,
        scoring: str = "accuracy",
        n_jobs: int = 1,
    ):
        self.estimator = estimator
        self.param_grid = param_grid
        self.cv = cv
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.best_params_ = None
        self.best_score_ = None
        self.best_estimator_ = None
        self.cv_results_ = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GridSearchCV":
        """Run grid search"""

        # Generate all parameter combinations
        param_combinations = self._generate_params()

        print(f"Testing {len(param_combinations)} parameter combinations...")

        best_score = 0
        best_params = None
        best_estimator = None
        all_results = []

        for params in param_combinations:
            # Create new estimator with these params
            estimator = self._clone_estimator(params)

            # Cross-validation
            fold_scores = []

            # Simple K-fold split
            indices = np.random.permutation(len(y))
            fold_size = len(y) // self.cv

            for fold in range(self.cv):
                # Split data
                val_start = fold * fold_size
                val_end = val_start + fold_size if fold < self.cv - 1 else len(y)

                val_idx = indices[val_start:val_end]
                train_idx = np.concatenate([indices[:val_start], indices[val_end:]])

                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]

                # Fit and evaluate
                estimator.fit(X_train, y_train)

                if hasattr(estimator, "predict"):
                    y_pred = estimator.predict(X_val)
                    score = np.mean(y_pred == y_val)
                elif hasattr(estimator, "score"):
                    score = estimator.score(X_val, y_val)
                else:
                    score = 0

                fold_scores.append(score)

            mean_score = np.mean(fold_scores)
            all_results.append(
                {
                    "params": params,
                    "mean_score": mean_score,
                    "std_score": np.std(fold_scores),
                }
            )

            if mean_score > best_score:
                best_score = mean_score
                best_params = params
                best_estimator = self._clone_estimator(params)
                best_estimator.fit(X, y)

        # Store results
        self.best_params_ = best_params
        self.best_score_ = best_score
        self.best_estimator_ = best_estimator
        self.cv_results_ = all_results

        print(f"Best score: {best_score:.4f}")
        print(f"Best params: {best_params}")

        return self

    def _generate_params(self) -> List[Dict]:
        """Generate all parameter combinations"""
        import itertools

        keys = list(self.param_grid.keys())
        values = list(self.param_grid.values())

        combinations = list(itertools.product(*values))
        return [dict(zip(keys, combo)) for combo in combinations]

    def _clone_estimator(self, params: Dict) -> Any:
        """Clone estimator with given parameters"""
        # Simple cloning - in practice use sklearn's clone
        from copy import deepcopy

        estimator = deepcopy(self.estimator)

        for key, value in params.items():
            if hasattr(estimator, key):
                setattr(estimator, key, value)

        return estimator

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using best estimator"""
        return self.best_estimator_.predict(X)


class RandomizedSearchCV:
    """
    RandomizedSearchCV: Random search over parameter distributions.

    More efficient than GridSearch when parameter space is large.
    """

    def __init__(
        self,
        estimator: Any,
        param_distributions: Dict,
        n_iter: int = 10,
        cv: int = 3,
        scoring: str = "accuracy",
        random_state: int = 42,
    ):
        self.estimator = estimator
        self.param_distributions = param_distributions
        self.n_iter = n_iter
        self.cv = cv
        self.scoring = scoring
        self.random_state = random_state
        self.best_params_ = None
        self.best_score_ = None
        self.best_estimator_ = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomizedSearchCV":
        """Run random search"""
        np.random.seed(self.random_state)

        print(f"Testing {self.n_iter} random parameter combinations...")

        best_score = 0
        best_params = None
        best_estimator = None

        for i in range(self.n_iter):
            # Sample random parameters
            params = {}
            for key, dist in self.param_distributions.items():
                if isinstance(dist, list):
                    params[key] = np.random.choice(dist)
                elif isinstance(dist, tuple) and len(dist) == 2:
                    # Assume (min, max) for uniform distribution
                    params[key] = np.random.uniform(dist[0], dist[1])
                else:
                    params[key] = dist

            # Create and evaluate estimator
            estimator = self._create_estimator(params)

            # Simple cross-validation
            fold_scores = []
            indices = np.random.permutation(len(y))
            fold_size = len(y) // self.cv

            for fold in range(self.cv):
                val_start = fold * fold_size
                val_end = val_start + fold_size if fold < self.cv - 1 else len(y)

                val_idx = indices[val_start:val_end]
                train_idx = np.concatenate([indices[:val_start], indices[val_end:]])

                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]

                estimator.fit(X_train, y_train)

                if hasattr(estimator, "predict"):
                    y_pred = estimator.predict(X_val)
                    score = np.mean(y_pred == y_val)
                else:
                    score = 0

                fold_scores.append(score)

            mean_score = np.mean(fold_scores)

            if mean_score > best_score:
                best_score = mean_score
                best_params = params
                best_estimator = self._create_estimator(params)
                best_estimator.fit(X, y)

        self.best_params_ = best_params
        self.best_score_ = best_score
        self.best_estimator_ = best_estimator

        print(f"Best score: {best_score:.4f}")
        print(f"Best params: {best_params}")

        return self

    def _create_estimator(self, params: Dict) -> Any:
        """Create estimator with params"""
        from copy import deepcopy

        estimator = deepcopy(self.estimator)

        for key, value in params.items():
            if hasattr(estimator, key):
                setattr(estimator, key, value)

        return estimator

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.best_estimator_.predict(X)


def demo_pipeline():
    """Demonstrate Pipeline and GridSearchCV"""
    print("=" * 60)
    print("Pipeline and Model Selection Demo")
    print("=" * 60)

    # Generate sample data
    np.random.seed(42)
    X = np.random.randn(200, 4)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    # Add some noise
    X += np.random.randn(200, 4) * 0.1

    print(f"Dataset: {X.shape}")

    # Demo 1: Pipeline with Scaling
    print("\n--- Pipeline Demo ---")

    scaler = StandardScaler()
    print(f"Before scaling - Mean: {X.mean(axis=0)[:2]}, Std: {X.std(axis=0)[:2]}")

    X_scaled = scaler.fit_transform(X)
    print(
        f"After scaling - Mean: {X_scaled.mean(axis=0)[:2]}, Std: {X_scaled.std(axis=0)[:2]}"
    )

    # Demo 2: Grid Search
    print("\n--- Grid Search Demo ---")

    # Simulate grid search (actual SVM might be slow)
    from implementations.ml.linear_regression import LogisticRegression

    param_grid = {"learning_rate": [0.01, 0.1, 0.5], "n_iterations": [100, 500]}

    # Quick demo with LogisticRegression instead
    lr = LogisticRegression()

    print("Testing parameter combinations...")

    best_score = 0
    best_params = {}

    for lr_param in [0.01, 0.1]:
        for n_iter in [100, 200]:
            lr_test = LogisticRegression(learning_rate=lr_param, n_epochs=n_iter)

            # Simple train-test split
            split = int(0.8 * len(y))
            lr_test.fit(X[:split], y[:split])
            score = lr_test.score(X[split:], y[split:])

            print(f"  lr={lr_param}, n_iter={n_iter}: {score:.4f}")

            if score > best_score:
                best_score = score
                best_params = {"learning_rate": lr_param, "n_iterations": n_iter}

    print(f"\nBest: {best_params} with score {best_score:.4f}")


if __name__ == "__main__":
    demo_pipeline()
    print("\nPipeline and Model Selection complete!")
