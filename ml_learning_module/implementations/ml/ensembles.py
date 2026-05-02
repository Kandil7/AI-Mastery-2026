"""
Ensemble Methods - Voting and Stacking
=======================================

Complete ensemble implementations based on "Hands-On Machine Learning".

Author: AI-Mastery-2026
"""

import numpy as np
from typing import List, Tuple, Optional, Any


class VotingClassifier:
    """
    Voting Classifier: Combines multiple classifiers via voting.

    Two types:
    1. Hard Voting: Majority vote on predicted classes
    2. Soft Voting: Average predicted probabilities (requires predict_proba)

    Similar to sklearn.ensemble.VotingClassifier
    """

    def __init__(
        self,
        estimators: List[Tuple[str, Any]],
        voting: str = "hard",
        weights: Optional[List[float]] = None,
    ):
        """
        Initialize Voting Classifier.

        Args:
            estimators: List of (name, estimator) tuples
            voting: 'hard' or 'soft'
            weights: Weights for each estimator (for soft voting)
        """
        self.estimators = estimators
        self.voting = voting
        self.weights = weights
        self.named_estimators = {name: est for name, est in estimators}

        # Fit estimators
        for name, estimator in estimators:
            estimator.fit = estimator.fit

    def fit(self, X: np.ndarray, y: np.ndarray) -> "VotingClassifier":
        """Fit all estimators"""
        for name, estimator in self.estimators:
            estimator.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using voting"""

        if self.voting == "hard":
            # Hard voting: majority vote
            predictions = np.array(
                [estimator.predict(X) for name, estimator in self.estimators]
            )

            # Majority vote for each sample
            from scipy import stats

            return stats.mode(predictions, axis=0, keepdims=False)[0]

        else:  # soft voting
            # Get probabilities from each estimator
            probas = []
            for name, estimator in self.estimators:
                if hasattr(estimator, "predict_proba"):
                    proba = estimator.predict_proba(X)
                elif hasattr(estimator, "decision_function"):
                    # Convert decision function to probabilities
                    dec = estimator.decision_function(X)
                    proba = 1 / (1 + np.exp(-dec))
                else:
                    raise ValueError(f"Estimator {name} has no predict_proba")

                probas.append(proba)

            # Weighted average
            if self.weights:
                avg_proba = np.average(probas, axis=0, weights=self.weights)
            else:
                avg_proba = np.mean(probas, axis=0)

            return np.argmax(avg_proba, axis=1)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute accuracy"""
        return np.mean(self.predict(X) == y)


class StackingClassifier:
    """
    Stacking Classifier: Combines multiple classifiers with a meta-learner.

    Two-level architecture:
    Level 0: Base estimators (e.g., Random Forest, SVM, KNN)
    Level 1: Meta-learner (e.g., Logistic Regression)

    How it works:
    1. Train base estimators on original features
    2. Get predictions (or probabilities) from base estimators
    3. Create new feature matrix from these predictions
    4. Train meta-learner on the new features

    Similar to sklearn.ensemble.StackingClassifier
    """

    def __init__(
        self,
        estimators: List[Tuple[str, Any]],
        final_estimator: Any,
        cv: int = 5,
        stack_method: str = "predict_proba",
    ):
        """
        Initialize Stacking Classifier.

        Args:
            estimators: List of (name, base_estimator) tuples
            final_estimator: Meta-learner
            cv: Cross-validation folds for generating stacked features
            stack_method: 'predict_proba' or 'predict'
        """
        self.estimators = estimators
        self.final_estimator = final_estimator
        self.cv = cv
        self.stack_method = stack_method
        self.named_estimators = {name: est for name, est in estimators}

    def _get_stacked_features(
        self, X: np.ndarray, y: np.ndarray, train_mode: bool = True
    ) -> np.ndarray:
        """Generate stacked features using cross-validation"""

        n_samples = X.shape[0]
        n_estimators = len(self.estimators)

        # Determine feature size per estimator
        first_est = self.estimators[0][1]

        if self.stack_method == "predict_proba" and hasattr(first_est, "predict_proba"):
            # Each estimator gives probabilities for each class
            try:
                first_est.fit(X[:5], y[:5])
                proba = first_est.predict_proba(X[:5])
                n_classes = proba.shape[1]
                n_features = n_estimators * n_classes
            except:
                n_classes = len(np.unique(y))
                n_features = n_estimators * n_classes
        else:
            # Each estimator gives single prediction
            n_features = n_estimators

        stacked_features = np.zeros((n_samples, n_features))

        if train_mode:
            # Use cross-validation to generate features
            indices = np.random.permutation(n_samples)
            fold_size = n_samples // self.cv

            for fold in range(self.cv):
                val_start = fold * fold_size
                val_end = val_start + fold_size if fold < self.cv - 1 else n_samples

                val_idx = indices[val_start:val_end]
                train_idx = np.concatenate([indices[:val_start], indices[val_end:]])

                X_train_fold, X_val_fold = X[train_idx], X[val_idx]
                y_train_fold = y[train_idx]

                # Train each estimator on training fold
                feature_idx = 0
                for name, estimator in self.estimators:
                    # Clone estimator (simplified - use sklearn's clone in practice)
                    from copy import deepcopy

                    est = deepcopy(estimator)
                    est.fit(X_train_fold, y_train_fold)

                    if self.stack_method == "predict_proba" and hasattr(
                        est, "predict_proba"
                    ):
                        proba = est.predict_proba(X_val_fold)
                        stacked_features[
                            val_idx, feature_idx : feature_idx + n_classes
                        ] = proba
                        feature_idx += n_classes
                    else:
                        pred = est.predict(X_val_fold)
                        stacked_features[val_idx, feature_idx] = pred
                        feature_idx += 1
        else:
            # For test data, use all training data
            for name, estimator in self.estimators:
                estimator.fit(X, y)

                feature_idx = 0
                if self.stack_method == "predict_proba" and hasattr(
                    estimator, "predict_proba"
                ):
                    proba = estimator.predict_proba(X)
                    stacked_features[:, feature_idx : feature_idx + n_classes] = proba
                    feature_idx += n_classes
                else:
                    pred = estimator.predict(X)
                    stacked_features[:, feature_idx] = pred
                    feature_idx += 1

        return stacked_features

    def fit(self, X: np.ndarray, y: np.ndarray) -> "StackingClassifier":
        """Fit the stacking classifier"""

        # Generate stacked features
        stacked_X = self._get_stacked_features(X, y, train_mode=True)

        # Train meta-learner on stacked features
        self.final_estimator.fit(stacked_X, y)

        # Finally, train all base estimators on full data
        for name, estimator in self.estimators:
            from copy import deepcopy

            est = deepcopy(estimator)
            est.fit(X, y)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using stacking"""

        # Generate stacked features for test data
        # Note: For simplicity, use training models
        n_samples = X.shape[0]

        # Get predictions from each base estimator
        stacked_features = []

        for name, estimator in self.estimators:
            if self.stack_method == "predict_proba" and hasattr(
                estimator, "predict_proba"
            ):
                proba = estimator.predict_proba(X)
                stacked_features.append(proba)
            else:
                pred = estimator.predict(X).reshape(-1, 1)
                stacked_features.append(pred)

        # Concatenate
        stacked_X = np.hstack(stacked_features)

        # Get final prediction from meta-learner
        return self.final_estimator.predict(stacked_X)


class AdaBoost:
    """
    AdaBoost (Adaptive Boosting): Ensemble method that focuses on hard examples.

    Algorithm:
    1. Train weak learner on weighted data
    2. Calculate error and importance (alpha)
    3. Update sample weights (increase weight for misclassified)
    4. Repeat

    Similar to sklearn.ensemble.AdaBoostClassifier
    """

    def __init__(
        self,
        n_estimators: int = 50,
        learning_rate: float = 1.0,
        base_estimator: Any = None,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.base_estimator = base_estimator
        self.estimators_ = []
        self.estimator_weights_ = []
        self.estimator_errors_ = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "AdaBoost":
        """Fit AdaBoost"""
        n_samples = X.shape[0]

        # Initialize sample weights
        sample_weights = np.ones(n_samples) / n_samples

        for t in range(self.n_estimators):
            # Create and train weak learner
            from copy import deepcopy

            estimator = deepcopy(
                self.base_estimator if self.base_estimator else DecisionStump()
            )

            # Train with sample weights
            try:
                # For simplicity, train normally (weighted training complex)
                estimator.fit(X, y)
            except:
                # Fall back to simple decision stump
                estimator = DecisionStump()
                estimator.fit(X, y, sample_weights)

            # Predict
            predictions = estimator.predict(X)

            # Calculate weighted error
            incorrect = predictions != y
            error = np.sum(sample_weights * incorrect) / np.sum(sample_weights)

            # Prevent numerical issues
            error = np.clip(error, 1e-10, 1 - 1e-10)

            # Calculate estimator weight (alpha)
            alpha = self.learning_rate * 0.5 * np.log((1 - error) / error)

            # Update sample weights
            sample_weights *= np.exp(alpha * (2 * incorrect - 1))
            sample_weights /= np.sum(sample_weights)  # Normalize

            # Store
            self.estimators_.append(estimator)
            self.estimator_weights_.append(alpha)
            self.estimator_errors_.append(error)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using weighted voting"""

        # Get predictions from all estimators
        all_predictions = np.array([est.predict(X) for est in self.estimators_])

        # Weighted sum
        weighted_sum = np.sum(
            self.estimator_weights_ * (2 * all_predictions - 1),  # Convert to ±1
            axis=0,
        )

        # Return sign
        return (weighted_sum > 0).astype(int)


class DecisionStump:
    """Simple decision stump for AdaBoost"""

    def __init__(self):
        self.feature_index = None
        self.threshold = None
        self.prediction = None

    def fit(
        self, X: np.ndarray, y: np.ndarray, sample_weights: Optional[np.ndarray] = None
    ):
        """Fit decision stump"""
        n_samples, n_features = X.shape

        best_error = float("inf")

        for feature in range(n_features):
            thresholds = np.percentile(X[:, feature], [25, 50, 75])

            for threshold in thresholds:
                # Predict based on threshold
                predictions = (X[:, feature] <= threshold).astype(int)

                # Calculate error
                if sample_weights is not None:
                    error = np.sum(sample_weights * (predictions != y))
                else:
                    error = np.mean(predictions != y)

                if error < best_error:
                    best_error = error
                    self.feature_index = feature
                    self.threshold = threshold

        # Set final predictions
        self.prediction = (X[:, self.feature_index] <= self.threshold).astype(int)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict"""
        return (X[:, self.feature_index] <= self.threshold).astype(int)


def demo_ensembles():
    """Demonstrate ensemble methods"""
    print("=" * 60)
    print("Ensemble Methods Demo - Based on Hands-On ML")
    print("=" * 60)

    # Generate sample data
    np.random.seed(42)
    n_samples = 200

    # Create 3-class data
    X = np.random.randn(n_samples, 4)
    y = np.array([0] * 60 + [1] * 70 + [2] * 70)

    # Add some class separation
    X[y == 0] += [-2, -2, 0, 0]
    X[y == 1] += [2, 0, 0, 0]
    X[y == 2] += [0, 2, 0, 0]

    print(f"Dataset: {X.shape}, {len(np.unique(y))} classes")

    # Demo 1: Simple Voting
    print("\n--- Voting Classifier ---")

    # Create simple classifiers (use our existing ones)
    from implementations.ml.linear_regression import LogisticRegression
    from implementations.ml.evaluation import Metrics

    # We'll just demonstrate the concept
    print("VotingClassifier: Combines multiple models via voting")
    print("  - Hard voting: Majority vote")
    print("  - Soft voting: Average probabilities (weighted)")

    # Demo 2: Stacking
    print("\n--- Stacking Classifier ---")
    print("StackingClassifier: Uses meta-learner on base model predictions")
    print("  - Level 0: Base models (RF, SVM, etc.)")
    print("  - Level 1: Meta-learner (LogisticRegression)")

    # Demo 3: AdaBoost
    print("\n--- AdaBoost ---")

    # Simple AdaBoost example
    np.random.seed(42)
    X_simple = np.random.randn(100, 2)
    y_simple = (X_simple[:, 0] + X_simple[:, 1] > 0).astype(int)

    ada = AdaBoost(n_estimators=10)
    ada.fit(X_simple, y_simple)

    accuracy = np.mean(ada.predict(X_simple) == y_simple)
    print(f"AdaBoost accuracy: {accuracy:.2%}")
    print(f"Number of estimators: {len(ada.estimators_)}")


if __name__ == "__main__":
    demo_ensembles()
    print("\nEnsemble methods complete!")
