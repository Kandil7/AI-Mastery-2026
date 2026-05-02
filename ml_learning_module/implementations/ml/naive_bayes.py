"""
Naive Bayes Classifiers
=======================

Implementation of Naive Bayes classifiers from scratch.
Based on Bayes' theorem with strong independence assumptions.

Author: ML Learning Module
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import defaultdict


class GaussianNB:
    """
    Gaussian Naive Bayes Classifier

    Assumes features follow normal distribution within each class.
    P(x|y) = N(μ_y, σ²_y)

    Mathematical Foundation:
    -------------------------
    Using Bayes' theorem:
    P(y|x) = P(x|y) * P(y) / P(x)

    Taking log and using independence assumption:
    log P(y|x) ∝ log P(y) + Σ log P(x_i|y)

    For Gaussian:
    P(x_i|y) = (1 / √(2πσ²)) * exp(-(x_i - μ)² / 2σ²)
    """

    def __init__(self, var_smoothing: float = 1e-9):
        """
        Parameters
        ----------
        var_smoothing : float, default=1e-9
            Additive smoothing for variance (to avoid division by zero)
        """
        self.var_smoothing = var_smoothing
        self.classes_ = None
        self.class_prior_ = None
        self.theta_ = None  # Mean of each feature per class
        self.var_ = None  # Variance of each feature per class
        self.epsilon_ = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GaussianNB":
        """
        Fit the Gaussian Naive Bayes classifier.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training features
        y : array-like of shape (n_samples,)
            Target labels

        Returns
        -------
        self : object
            Fitted classifier
        """
        X = np.asarray(X)
        y = np.asarray(y)

        n_samples, n_features = X.shape

        # Get unique classes
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)

        # Compute class prior P(y)
        self.class_prior_ = np.zeros(n_classes)
        for i, c in enumerate(self.classes_):
            self.class_prior_[i] = np.sum(y == c) / n_samples

        # Compute mean and variance for each class
        self.theta_ = np.zeros((n_classes, n_features))
        self.var_ = np.zeros((n_classes, n_features))

        for i, c in enumerate(self.classes_):
            X_c = X[y == c]
            self.theta_[i] = X_c.mean(axis=0)
            self.var_[i] = X_c.var(axis=0) + self.var_smoothing

        return self

    def _compute_log_likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Compute log P(x|y) for each class.

        Uses Gaussian PDF: log P(x|y) = Σ [log(1/√(2πσ²)) - (x-μ)²/(2σ²)]
        """
        n_samples, n_features = X.shape
        n_classes = len(self.classes_)

        log_likelihood = np.zeros((n_samples, n_classes))

        for i in range(n_classes):
            # Log of normalization constant: -0.5 * log(2πσ²)
            log_norm = -0.5 * (
                n_features * np.log(2 * np.pi) + np.log(self.var_[i]).sum()
            )

            # Log of exponential part: -(x-μ)²/(2σ²)
            diff = X - self.theta_[i]
            log_exp = -0.5 * (diff**2 / self.var_[i]).sum(axis=1)

            log_likelihood[:, i] = log_norm + log_exp

        return log_likelihood

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.

        P(y|x) ∝ P(y) * P(x|y)
        log P(y|x) = log P(y) + log P(x|y)
        """
        X = np.asarray(X)

        log_prior = np.log(self.class_prior_)
        log_likelihood = self._compute_log_likelihood(X)

        # Log posterior (unnormalized)
        log_posterior = log_prior + log_likelihood

        # Convert to probabilities using softmax
        # log sum exp trick for numerical stability
        log_posterior_max = np.max(log_posterior, axis=1, keepdims=True)
        probs = np.exp(log_posterior - log_posterior_max)
        probs = probs / probs.sum(axis=1, keepdims=True)

        return probs

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        probs = self.predict_proba(X)
        return self.classes_[np.argmax(probs, axis=1)]

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return accuracy score."""
        return np.mean(self.predict(X) == y)


class MultinomialNB:
    """
    Multinomial Naive Bayes Classifier

    Suitable for discrete features (e.g., word counts, TF-IDF).

    Typically used for document classification where features are
    word frequencies or similar count-based representations.
    """

    def __init__(self, alpha: float = 1.0):
        """
        Parameters
        ----------
        alpha : float, default=1.0
            Additive (Laplace/Lidstone) smoothing parameter
        """
        self.alpha = alpha
        self.classes_ = None
        self.class_log_prior_ = None
        self.feature_log_prob_ = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MultinomialNB":
        """Fit the Multinomial Naive Bayes classifier."""
        X = np.asarray(X)
        y = np.asarray(y)

        n_samples, n_features = X.shape
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)

        # Compute class prior with smoothing
        class_counts = np.zeros(n_classes)
        for i, c in enumerate(self.classes_):
            class_counts[i] = np.sum(y == c)

        self.class_log_prior_ = np.log(
            (class_counts + self.alpha) / (n_samples + n_classes * self.alpha)
        )

        # Compute feature probabilities (with smoothing)
        self.feature_log_prob_ = np.zeros((n_classes, n_features))

        for i, c in enumerate(self.classes_):
            X_c = X[y == c]
            # Total feature count for this class
            feature_counts = X_c.sum(axis=0) + self.alpha
            # Normalize to get probabilities
            self.feature_log_prob_[i] = np.log(feature_counts / feature_counts.sum())

        return self

    def predict_log_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict log probabilities for each class."""
        X = np.asarray(X)

        # Log likelihood: sum of log feature probabilities
        # For multinomial: log P(x|y) = Σ x_i * log P(feature_i | class)
        # But we have count data, so use the features directly
        log_likelihood = X @ self.feature_log_prob_.T

        # Log posterior = log prior + log likelihood
        return self.class_log_prior_ + log_likelihood

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        log_proba = self.predict_log_proba(X)

        # Convert to probabilities
        log_proba_max = np.max(log_proba, axis=1, keepdims=True)
        probs = np.exp(log_proba - log_proba_max)
        probs = probs / probs.sum(axis=1, keepdims=True)

        return probs

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        log_proba = self.predict_log_proba(X)
        return self.classes_[np.argmax(log_proba, axis=1)]


class BernoulliNB:
    """
    Bernoulli Naive Bayes Classifier

    Designed for binary/boolean features.
    Uses binary occurrence indicators (0 or 1) rather than counts.
    """

    def __init__(self, alpha: float = 1.0, binarize: Optional[float] = 0.0):
        """
        Parameters
        ----------
        alpha : float, default=1.0
            Smoothing parameter
        binarize : float, default=0.0
            Threshold for binarizing continuous features
            If None, features are expected to be binary
        """
        self.alpha = alpha
        self.binarize = binarize
        self.classes_ = None
        self.class_log_prior_ = None
        self.feature_log_prob_ = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BernoulliNB":
        """Fit the Bernoulli Naive Bayes classifier."""
        X = np.asarray(X)
        y = np.asarray(y)

        # Binarize if needed
        if self.binarize is not None:
            X = (X > self.binarize).astype(float)

        n_samples, n_features = X.shape
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)

        # Class prior
        class_counts = np.zeros(n_classes)
        for i, c in enumerate(self.classes_):
            class_counts[i] = np.sum(y == c)

        self.class_log_prior_ = np.log(
            (class_counts + self.alpha) / (n_samples + n_classes * self.alpha)
        )

        # Feature probabilities for both present and absent
        # P(x_i=1|y) and P(x_i=0|y) = 1 - P(x_i=1|y)
        self.feature_log_prob_ = np.zeros((n_classes, n_features))

        for i, c in enumerate(self.classes_):
            X_c = X[y == c]

            # Count of feature present in class
            feature_present = X_c.sum(axis=0) + self.alpha
            # Count of feature absent in class
            feature_absent = (1 - X_c).sum(axis=0) + self.alpha

            # Total counts (present + absent + smoothing)
            total = feature_present + feature_absent

            # Log probability of feature being present
            self.feature_log_prob_[i] = np.log(feature_present / total)

        return self

    def predict_log_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict log probabilities."""
        X = np.asarray(X)

        if self.binarize is not None:
            X = (X > self.binarize).astype(float)

        # For Bernoulli: log P(x|y) = Σ [x_i * log P(x_i=1|y) + (1-x_i) * log P(x_i=0|y)]
        # But we store log P(x_i=1|y), so we need to compute:
        # log P(x_i=0|y) = log(1 - exp(log P(x_i=1|y)))

        log_prob_present = X * self.feature_log_prob_
        log_prob_absent = (1 - X) * np.log(1 - np.exp(self.feature_log_prob_))

        log_likelihood = (log_prob_present + log_prob_absent).sum(axis=1, keepdims=True)

        return self.class_log_prior_ + log_likelihood.flatten()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        log_proba = self.predict_log_proba(X)

        # Need to reshape for multi-class
        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        log_proba = (
            log_proba.reshape(n_samples, n_classes)
            if log_proba.ndim > 1
            else log_proba.reshape(-1, 1)
        )

        # Convert to probabilities
        log_proba_max = np.max(log_proba, axis=1, keepdims=True)
        probs = np.exp(log_proba - log_proba_max)
        probs = probs / probs.sum(axis=1, keepdims=True)

        return probs

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        log_proba = self.predict_log_proba(X)

        if log_proba.ndim > 1:
            return self.classes_[np.argmax(log_proba, axis=1)]
        else:
            return self.classes_[
                np.argmax(log_proba.reshape(-1, len(self.classes_)), axis=1)
            ]


class CategoricalNB:
    """
    Categorical Naive Bayes Classifier

    Suitable for categorical features with discrete values.
    Each feature can have different numbers of categories.
    """

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.classes_ = None
        self.class_log_prior_ = None
        self.feature_log_prob_ = None
        self.n_categories_ = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "CategoricalNB":
        """Fit Categorical Naive Bayes."""
        X = np.asarray(X, dtype=int)
        y = np.asarray(y)

        n_samples, n_features = X.shape
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)

        # Number of categories for each feature
        self.n_categories_ = X.max(axis=0) + 1

        # Class prior
        class_counts = np.zeros(n_classes)
        for i, c in enumerate(self.classes_):
            class_counts[i] = np.sum(y == c)

        self.class_log_prior_ = np.log(
            (class_counts + self.alpha) / (n_samples + n_classes * self.alpha)
        )

        # Feature probabilities for each category value
        self.feature_log_prob_ = []

        for feature_idx in range(n_features):
            n_cats = self.n_categories_[feature_idx]
            feature_probs = np.zeros((n_classes, n_cats))

            for i, c in enumerate(self.classes_):
                X_c = X[y == c, feature_idx]

                # Count occurrences of each category
                category_counts = np.zeros(n_cats)
                for cat in X_c:
                    category_counts[cat] += 1

                # Add smoothing and normalize
                category_probs = (category_counts + self.alpha) / (
                    len(X_c) + n_cats * self.alpha
                )
                feature_probs[i] = np.log(category_probs)

            self.feature_log_prob_.append(feature_probs)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        X = np.asarray(X, dtype=int)

        log_proba = self.class_log_prior_.copy()

        for feature_idx in range(X.shape[1]):
            for i, c in enumerate(self.classes_):
                cat = X[0, feature_idx] if X.ndim == 1 else X[:, feature_idx]
                log_proba[i] += (
                    self.feature_log_prob_[feature_idx][i, cat].sum()
                    if X.ndim > 1
                    else self.feature_log_prob_[feature_idx][i, cat]
                )

        return self.classes_[np.argmax(log_proba)]


# ============================================================================
# Demonstration
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Naive Bayes Classifiers - Demonstration")
    print("=" * 70)

    # Generate sample data
    np.random.seed(42)
    n_samples = 200
    n_features = 4

    # Create 3 classes with Gaussian distributions
    X = np.vstack(
        [
            np.random.randn(60, n_features) + np.array([0, 0, 0, 0]),
            np.random.randn(70, n_features) + np.array([3, 3, 3, 3]),
            np.random.randn(70, n_features) + np.array([-3, 1, -2, 2]),
        ]
    )
    y = np.array([0] * 60 + [1] * 70 + [2] * 70)

    # Shuffle
    indices = np.random.permutation(n_samples)
    X, y = X[indices], y[indices]

    # Split
    X_train, X_test = X[:150], X[150:]
    y_train, y_test = y[:150], y[150:]

    print("\n1. Gaussian Naive Bayes:")
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    y_pred = gnb.predict(X_test)
    accuracy = np.mean(y_pred == y_test)
    print(f"   Accuracy: {accuracy:.4f}")
    print(f"   Class priors: {gnb.class_prior_}")

    print("\n2. Multinomial Naive Bayes (with count-like features):")
    # Create count-like features
    X_count = np.abs(X * 10).astype(int)
    X_count = np.clip(X_count, 0, 20)  # Limit counts
    X_train_cnt, X_test_cnt = X_count[:150], X_count[150:]

    mnb = MultinomialNB(alpha=1.0)
    mnb.fit(X_train_cnt, y_train)
    y_pred = mnb.predict(X_test_cnt)
    accuracy = np.mean(y_pred == y_test)
    print(f"   Accuracy: {accuracy:.4f}")

    print("\n3. Bernoulli Naive Bayes (with binary features):")
    # Create binary features
    X_binary = (X > 0).astype(float)
    X_train_bin, X_test_bin = X_binary[:150], X_binary[150:]

    bnb = BernoulliNB(alpha=1.0, binarize=None)
    bnb.fit(X_train_bin, y_train)
    y_pred = bnb.predict(X_test_bin)
    accuracy = np.mean(y_pred == y_test)
    print(f"   Accuracy: {accuracy:.4f}")

    print("\n" + "=" * 70)
    print("All Naive Bayes classifiers working!")
    print("=" * 70)
