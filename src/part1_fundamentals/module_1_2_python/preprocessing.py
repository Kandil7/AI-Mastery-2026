"""
Data Preprocessing Module for Machine Learning.

This module provides comprehensive preprocessing utilities including:
- Feature scaling (StandardScaler, MinMaxScaler, RobustScaler)
- Encoding (OneHotEncoder, LabelEncoder, OrdinalEncoder)
- Missing value handling (Imputer)
- Train/test splitting
- Feature selection

Example Usage:
    >>> import numpy as np
    >>> from preprocessing import StandardScaler, OneHotEncoder, train_test_split
    >>> 
    >>> # Feature scaling
    >>> X = np.array([[1, 2], [3, 4], [5, 6]])
    >>> scaler = StandardScaler()
    >>> X_scaled = scaler.fit_transform(X)
    >>> 
    >>> # Encoding
    >>> categories = ['cat', 'dog', 'cat', 'bird']
    >>> encoder = OneHotEncoder()
    >>> encoded = encoder.fit_transform(categories)
    >>> 
    >>> # Train/test split
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
"""

from typing import Union, List, Dict, Any, Optional, Tuple, Callable
import numpy as np
from numpy.typing import ArrayLike
import logging
from collections import Counter

logger = logging.getLogger(__name__)

ArrayLike2D = Union[np.ndarray, List[List[float]]]
ArrayLike1D = Union[np.ndarray, List]


class StandardScaler:
    """
    Standardize features by removing mean and scaling to unit variance.
    
    Transforms features to have mean = 0 and variance = 1.
    
    Formula: z = (x - μ) / σ
    
    Example:
        >>> scaler = StandardScaler()
        >>> X = np.array([[1, 2], [3, 4], [5, 6]])
        >>> X_scaled = scaler.fit_transform(X)
        >>> np.allclose(X_scaled.mean(axis=0), 0, atol=1e-10)
        True
        >>> np.allclose(X_scaled.std(axis=0), 1, atol=1e-10)
        True
    """
    
    def __init__(self, with_mean: bool = True, with_std: bool = True, epsilon: float = 1e-8):
        """
        Initialize StandardScaler.
        
        Args:
            with_mean: Whether to center data. Default: True.
            with_std: Whether to scale data. Default: True.
            epsilon: Small value to avoid division by zero.
        """
        self.with_mean = with_mean
        self.with_std = with_std
        self.epsilon = epsilon
        
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None
        self.n_features_in_: int = 0
        
        logger.debug(f"StandardScaler initialized: with_mean={with_mean}, with_std={with_std}")
    
    def fit(
        self,
        X: ArrayLike2D
    ) -> 'StandardScaler':
        """
        Compute mean and std for scaling.
        
        Args:
            X: Training data (n_samples, n_features).
        
        Returns:
            self: Fitted scaler.
        """
        X = np.asarray(X, dtype=np.float64)
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        self.n_features_in_ = X.shape[1]
        
        if self.with_mean:
            self.mean_ = np.mean(X, axis=0)
        else:
            self.mean_ = np.zeros(X.shape[1])
        
        if self.with_std:
            self.std_ = np.std(X, axis=0, ddof=0)
            # Prevent division by zero
            self.std_ = np.where(self.std_ < self.epsilon, 1.0, self.std_)
        else:
            self.std_ = np.ones(X.shape[1])
        
        logger.info(f"StandardScaler fitted: {self.n_features_in_} features")
        return self
    
    def transform(self, X: ArrayLike2D) -> np.ndarray:
        """
        Scale data using computed statistics.
        
        Args:
            X: Data to scale.
        
        Returns:
            np.ndarray: Scaled data.
        """
        if self.mean_ is None or self.std_ is None:
            raise ValueError("Scaler not fitted. Call fit() first.")
        
        X = np.asarray(X, dtype=np.float64)
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"Expected {self.n_features_in_} features, got {X.shape[1]}")
        
        return (X - self.mean_) / self.std_
    
    def fit_transform(self, X: ArrayLike2D) -> np.ndarray:
        """
        Fit and transform in one step.
        
        Args:
            X: Training data.
        
        Returns:
            np.ndarray: Scaled data.
        """
        self.fit(X)
        return self.transform(X)
    
    def inverse_transform(self, X: ArrayLike2D) -> np.ndarray:
        """
        Reverse the scaling transformation.
        
        Args:
            X: Scaled data.
        
        Returns:
            np.ndarray: Original scale data.
        """
        if self.mean_ is None or self.std_ is None:
            raise ValueError("Scaler not fitted.")
        
        X = np.asarray(X, dtype=np.float64)
        return X * self.std_ + self.mean_


class MinMaxScaler:
    """
    Scale features to a specified range.
    
    Transforms features to be within [feature_min, feature_max].
    
    Formula: X_scaled = (X - X_min) / (X_max - X_min) * (max - min) + min
    
    Example:
        >>> scaler = MinMaxScaler(feature_range=(0, 1))
        >>> X = np.array([[1, 2], [3, 4], [5, 6]])
        >>> X_scaled = scaler.fit_transform(X)
        >>> np.allclose(X_scaled.min(), 0) and np.allclose(X_scaled.max(), 1)
        True
    """
    
    def __init__(
        self,
        feature_range: Tuple[float, float] = (0, 1),
        clip: bool = False,
        epsilon: float = 1e-8
    ):
        """
        Initialize MinMaxScaler.
        
        Args:
            feature_range: Target range (min, max). Default: (0, 1).
            clip: Clip transformed values to range. Default: False.
            epsilon: Small value to avoid division by zero.
        """
        self.feature_range = feature_range
        self.clip = clip
        self.epsilon = epsilon
        
        self.min_: Optional[np.ndarray] = None
        self.scale_: Optional[np.ndarray] = None
        self.n_features_in_: int = 0
        
        logger.debug(f"MinMaxScaler initialized: range={feature_range}")
    
    def fit(self, X: ArrayLike2D) -> 'MinMaxScaler':
        """
        Compute min and max for scaling.
        
        Args:
            X: Training data.
        
        Returns:
            self: Fitted scaler.
        """
        X = np.asarray(X, dtype=np.float64)
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        self.n_features_in_ = X.shape[1]
        
        feature_min, feature_max = self.feature_range
        
        data_min = np.min(X, axis=0)
        data_max = np.max(X, axis=0)
        
        # Compute scale
        data_range = data_max - data_min
        data_range = np.where(data_range < self.epsilon, 1.0, data_range)
        
        self.min_ = data_min
        self.scale_ = (feature_max - feature_min) / data_range
        
        logger.info(f"MinMaxScaler fitted: {self.n_features_in_} features")
        return self
    
    def transform(self, X: ArrayLike2D) -> np.ndarray:
        """
        Scale data using computed statistics.
        
        Args:
            X: Data to scale.
        
        Returns:
            np.ndarray: Scaled data.
        """
        if self.min_ is None or self.scale_ is None:
            raise ValueError("Scaler not fitted. Call fit() first.")
        
        X = np.asarray(X, dtype=np.float64)
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        feature_min, feature_max = self.feature_range
        
        X_scaled = (X - self.min_) * self.scale_
        
        if self.clip:
            X_scaled = np.clip(X_scaled, feature_min, feature_max)
        
        return X_scaled
    
    def fit_transform(self, X: ArrayLike2D) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(X)
        return self.transform(X)
    
    def inverse_transform(self, X: ArrayLike2D) -> np.ndarray:
        """
        Reverse the scaling transformation.
        
        Args:
            X: Scaled data.
        
        Returns:
            np.ndarray: Original scale data.
        """
        if self.min_ is None or self.scale_ is None:
            raise ValueError("Scaler not fitted.")
        
        X = np.asarray(X, dtype=np.float64)
        return X / self.scale_ + self.min_


class RobustScaler:
    """
    Scale features using statistics that are robust to outliers.
    
    Uses median and IQR (Interquartile Range) instead of mean and std.
    
    Formula: X_scaled = (X - median) / IQR
    
    Example:
        >>> scaler = RobustScaler()
        >>> X = np.array([[1, 2], [3, 4], [100, 6]])  # Outlier
        >>> X_scaled = scaler.fit_transform(X)
    """
    
    def __init__(
        self,
        quantile_range: Tuple[float, float] = (25.0, 75.0),
        with_centering: bool = True,
        with_scaling: bool = True,
        epsilon: float = 1e-8
    ):
        """
        Initialize RobustScaler.
        
        Args:
            quantile_range: Quantile range for IQR. Default: (25.0, 75.0).
            with_centering: Center using median. Default: True.
            with_scaling: Scale using IQR. Default: True.
            epsilon: Small value to avoid division by zero.
        """
        self.quantile_range = quantile_range
        self.with_centering = with_centering
        self.with_scaling = with_scaling
        self.epsilon = epsilon
        
        self.center_: Optional[np.ndarray] = None
        self.scale_: Optional[np.ndarray] = None
        self.n_features_in_: int = 0
        
        logger.debug(f"RobustScaler initialized: quantile_range={quantile_range}")
    
    def fit(self, X: ArrayLike2D) -> 'RobustScaler':
        """
        Compute median and IQR for scaling.
        
        Args:
            X: Training data.
        
        Returns:
            self: Fitted scaler.
        """
        X = np.asarray(X, dtype=np.float64)
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        self.n_features_in_ = X.shape[1]
        
        if self.with_centering:
            self.center_ = np.median(X, axis=0)
        else:
            self.center_ = np.zeros(X.shape[1])
        
        if self.with_scaling:
            q1 = np.percentile(X, self.quantile_range[0], axis=0)
            q3 = np.percentile(X, self.quantile_range[1], axis=0)
            self.scale_ = q3 - q1
            self.scale_ = np.where(self.scale_ < self.epsilon, 1.0, self.scale_)
        else:
            self.scale_ = np.ones(X.shape[1])
        
        logger.info(f"RobustScaler fitted: {self.n_features_in_} features")
        return self
    
    def transform(self, X: ArrayLike2D) -> np.ndarray:
        """
        Scale data using computed statistics.
        
        Args:
            X: Data to scale.
        
        Returns:
            np.ndarray: Scaled data.
        """
        if self.center_ is None or self.scale_ is None:
            raise ValueError("Scaler not fitted.")
        
        X = np.asarray(X, dtype=np.float64)
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        return (X - self.center_) / self.scale_
    
    def fit_transform(self, X: ArrayLike2D) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(X)
        return self.transform(X)


class LabelEncoder:
    """
    Encode categorical labels as integers.
    
    Example:
        >>> encoder = LabelEncoder()
        >>> labels = ['cat', 'dog', 'cat', 'bird', 'dog']
        >>> encoded = encoder.fit_transform(labels)
        >>> encoded
        array([1, 2, 1, 0, 2])
        >>> encoder.classes_
        array(['bird', 'cat', 'dog'])
    """
    
    def __init__(self):
        """Initialize LabelEncoder."""
        self.classes_: Optional[np.ndarray] = None
        self.label_to_int_: Dict[Any, int] = {}
        self.int_to_label_: Dict[int, Any] = {}
        
        logger.debug("LabelEncoder initialized")
    
    def fit(self, y: ArrayLike1D) -> 'LabelEncoder':
        """
        Fit encoder to labels.
        
        Args:
            y: Labels to encode.
        
        Returns:
            self: Fitted encoder.
        """
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        
        self.label_to_int_ = {label: i for i, label in enumerate(self.classes_)}
        self.int_to_label_ = {i: label for label, i in self.label_to_int_.items()}
        
        logger.info(f"LabelEncoder fitted: {len(self.classes_)} classes")
        return self
    
    def transform(self, y: ArrayLike1D) -> np.ndarray:
        """
        Transform labels to integers.
        
        Args:
            y: Labels to transform.
        
        Returns:
            np.ndarray: Encoded labels.
        """
        if self.classes_ is None:
            raise ValueError("Encoder not fitted.")
        
        y = np.asarray(y)
        return np.array([self.label_to_int_.get(label, -1) for label in y])
    
    def fit_transform(self, y: ArrayLike1D) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(y)
        return self.transform(y)
    
    def inverse_transform(self, y: ArrayLike1D) -> np.ndarray:
        """
        Transform integers back to labels.
        
        Args:
            y: Encoded labels.
        
        Returns:
            np.ndarray: Original labels.
        """
        if self.int_to_label_ is None:
            raise ValueError("Encoder not fitted.")
        
        y = np.asarray(y, dtype=np.int32)
        return np.array([self.int_to_label_.get(i, None) for i in y])


class OneHotEncoder:
    """
    One-hot encode categorical features.
    
    Example:
        >>> encoder = OneHotEncoder()
        >>> categories = [['cat'], ['dog'], ['cat'], ['bird']]
        >>> encoded = encoder.fit_transform(categories)
        >>> encoded.shape
        (4, 3)
    """
    
    def __init__(
        self,
        handle_unknown: str = 'ignore',
        drop: Optional[str] = None,
        sparse: bool = False
    ):
        """
        Initialize OneHotEncoder.
        
        Args:
            handle_unknown: How to handle unknown categories ('ignore' or 'error').
            drop: Whether to drop a category ('first' or None).
            sparse: Return sparse matrix. Default: False.
        """
        self.handle_unknown = handle_unknown
        self.drop = drop
        self.sparse = sparse
        
        self.categories_: List[np.ndarray] = []
        self.n_features_in_: int = 0
        self.n_categories_out_: int = 0
        
        logger.debug(f"OneHotEncoder initialized: handle_unknown={handle_unknown}")
    
    def fit(self, X: ArrayLike2D) -> 'OneHotEncoder':
        """
        Fit encoder to data.
        
        Args:
            X: Categorical data (n_samples, n_features).
        
        Returns:
            self: Fitted encoder.
        """
        X = np.asarray(X)
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        self.n_features_in_ = X.shape[1]
        self.categories_ = []
        self.n_categories_out_ = 0
        
        for j in range(self.n_features_in_):
            categories = np.unique(X[:, j])
            
            if self.drop == 'first' and len(categories) > 1:
                categories = categories[1:]
            
            self.categories_.append(categories)
            self.n_categories_out_ += len(categories)
        
        logger.info(f"OneHotEncoder fitted: {self.n_features_in_} features, "
                   f"{self.n_categories_out_} output categories")
        return self
    
    def transform(self, X: ArrayLike2D) -> np.ndarray:
        """
        Transform data to one-hot encoding.
        
        Args:
            X: Categorical data.
        
        Returns:
            np.ndarray: One-hot encoded data.
        """
        if not self.categories_:
            raise ValueError("Encoder not fitted.")
        
        X = np.asarray(X)
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        n_samples = X.shape[0]
        encoded = np.zeros((n_samples, self.n_categories_out_))
        
        col_idx = 0
        for j in range(self.n_features_in_):
            categories = self.categories_[j]
            cat_to_idx = {cat: i for i, cat in enumerate(categories)}
            
            for i, val in enumerate(X[:, j]):
                if val in cat_to_idx:
                    encoded[i, col_idx + cat_to_idx[val]] = 1
                elif self.handle_unknown == 'error':
                    raise ValueError(f"Unknown category: {val}")
                # else: ignore (leave as 0)
            
            col_idx += len(categories)
        
        return encoded
    
    def fit_transform(self, X: ArrayLike2D) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(X)
        return self.transform(X)
    
    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform one-hot encoding back to categories.
        
        Args:
            X: One-hot encoded data.
        
        Returns:
            np.ndarray: Categorical data.
        """
        if not self.categories_:
            raise ValueError("Encoder not fitted.")
        
        n_samples = X.shape[0]
        result = np.empty((n_samples, self.n_features_in_), dtype=object)
        
        col_idx = 0
        for j in range(self.n_features_in_):
            categories = self.categories_[j]
            n_cats = len(categories)
            
            for i in range(n_samples):
                cat_idx = np.argmax(X[i, col_idx:col_idx + n_cats])
                result[i, j] = categories[cat_idx]
            
            col_idx += n_cats
        
        return result


class OrdinalEncoder:
    """
    Encode categorical features as integers with ordinal relationship.
    
    Example:
        >>> encoder = OrdinalEncoder()
        >>> categories = [['low'], ['medium'], ['high'], ['low']]
        >>> encoded = encoder.fit_transform(categories)
    """
    
    def __init__(self, categories: Optional[List[List[Any]]] = None):
        """
        Initialize OrdinalEncoder.
        
        Args:
            categories: List of categories per feature. If None, inferred from data.
        """
        self.categories = categories
        self.categories_: List[np.ndarray] = []
        self.n_features_in_: int = 0
        
        logger.debug("OrdinalEncoder initialized")
    
    def fit(self, X: ArrayLike2D) -> 'OrdinalEncoder':
        """
        Fit encoder to data.
        
        Args:
            X: Categorical data.
        
        Returns:
            self: Fitted encoder.
        """
        X = np.asarray(X)
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        self.n_features_in_ = X.shape[1]
        self.categories_ = []
        
        if self.categories is not None:
            self.categories_ = [np.array(cats) for cats in self.categories]
        else:
            for j in range(self.n_features_in_):
                self.categories_.append(np.unique(X[:, j]))
        
        logger.info(f"OrdinalEncoder fitted: {self.n_features_in_} features")
        return self
    
    def transform(self, X: ArrayLike2D) -> np.ndarray:
        """
        Transform data to ordinal encoding.
        
        Args:
            X: Categorical data.
        
        Returns:
            np.ndarray: Encoded data.
        """
        if not self.categories_:
            raise ValueError("Encoder not fitted.")
        
        X = np.asarray(X)
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        n_samples, n_features = X.shape
        encoded = np.zeros((n_samples, n_features), dtype=np.int32)
        
        for j in range(n_features):
            cat_to_idx = {cat: i for i, cat in enumerate(self.categories_[j])}
            for i in range(n_samples):
                encoded[i, j] = cat_to_idx.get(X[i, j], -1)
        
        return encoded
    
    def fit_transform(self, X: ArrayLike2D) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(X)
        return self.transform(X)


class SimpleImputer:
    """
    Impute missing values in data.
    
    Strategies:
    - 'mean': Replace with column mean (numeric only)
    - 'median': Replace with column median (numeric only)
    - 'most_frequent': Replace with most frequent value
    - 'constant': Replace with specified fill value
    
    Example:
        >>> imputer = SimpleImputer(strategy='mean')
        >>> X = np.array([[1, 2], [np.nan, 3], [7, 6]])
        >>> X_imputed = imputer.fit_transform(X)
    """
    
    def __init__(
        self,
        strategy: str = 'mean',
        fill_value: Optional[Any] = None,
        missing_values: Any = np.nan
    ):
        """
        Initialize SimpleImputer.
        
        Args:
            strategy: Imputation strategy.
            fill_value: Value for 'constant' strategy.
            missing_values: Value to consider as missing.
        """
        self.strategy = strategy
        self.fill_value = fill_value
        self.missing_values = missing_values
        
        self.statistics_: Optional[np.ndarray] = None
        self.n_features_in_: int = 0
        
        logger.debug(f"SimpleImputer initialized: strategy={strategy}")
    
    def fit(self, X: ArrayLike2D) -> 'SimpleImputer':
        """
        Compute imputation values.
        
        Args:
            X: Training data.
        
        Returns:
            self: Fitted imputer.
        """
        X = np.asarray(X, dtype=np.float64)
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        self.n_features_in_ = X.shape[1]
        self.statistics_ = np.zeros(X.shape[1])
        
        for j in range(self.n_features_in_):
            col = X[:, j]
            mask = ~np.isnan(col) if np.isnan(self.missing_values) else col != self.missing_values
            
            if self.strategy == 'mean':
                self.statistics_[j] = np.mean(col[mask])
            elif self.strategy == 'median':
                self.statistics_[j] = np.median(col[mask])
            elif self.strategy == 'most_frequent':
                values, counts = np.unique(col[mask], return_counts=True)
                self.statistics_[j] = values[np.argmax(counts)]
            elif self.strategy == 'constant':
                self.statistics_[j] = self.fill_value if self.fill_value is not None else 0
            else:
                raise ValueError(f"Unknown strategy: {self.strategy}")
        
        logger.info(f"SimpleImputer fitted: {self.n_features_in_} features")
        return self
    
    def transform(self, X: ArrayLike2D) -> np.ndarray:
        """
        Impute missing values.
        
        Args:
            X: Data to impute.
        
        Returns:
            np.ndarray: Imputed data.
        """
        if self.statistics_ is None:
            raise ValueError("Imputer not fitted.")
        
        X = np.asarray(X, dtype=np.float64).copy()
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        for j in range(self.n_features_in_):
            mask = np.isnan(X[:, j]) if np.isnan(self.missing_values) else X[:, j] == self.missing_values
            X[mask, j] = self.statistics_[j]
        
        return X
    
    def fit_transform(self, X: ArrayLike2D) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(X)
        return self.transform(X)


def train_test_split(
    *arrays: ArrayLike,
    test_size: Union[float, int] = 0.25,
    train_size: Optional[Union[float, int]] = None,
    random_state: Optional[int] = None,
    shuffle: bool = True,
    stratify: Optional[ArrayLike] = None
) -> List[np.ndarray]:
    """
    Split data into training and testing sets.
    
    Args:
        *arrays: One or more arrays to split (X, y, etc.).
        test_size: Test set size (fraction or absolute). Default: 0.25.
        train_size: Train set size (fraction or absolute). If None, 1 - test_size.
        random_state: Random seed for reproducibility.
        shuffle: Whether to shuffle before splitting.
        stratify: Array for stratified splitting.
    
    Returns:
        List[np.ndarray]: Split arrays (X_train, X_test, y_train, y_test, ...).
    
    Example:
        >>> X = np.random.randn(100, 5)
        >>> y = np.random.randint(0, 2, 100)
        >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        >>> len(X_train) + len(X_test)
        100
    """
    n_samples = len(arrays[0])
    
    # Validate all arrays have same length
    for i, arr in enumerate(arrays[1:], 1):
        if len(arr) != n_samples:
            raise ValueError(f"All arrays must have same length. Array 0 has {n_samples}, array {i} has {len(arr)}")
    
    # Determine split sizes
    if isinstance(test_size, float):
        test_size = int(n_samples * test_size)
    
    if train_size is None:
        train_size = n_samples - test_size
    elif isinstance(train_size, float):
        train_size = int(n_samples * train_size)
    
    if train_size + test_size > n_samples:
        raise ValueError(f"train_size + test_size ({train_size + test_size}) > n_samples ({n_samples})")
    
    # Set random seed
    if random_state is not None:
        np.random.seed(random_state)
    
    # Create indices
    indices = np.arange(n_samples)
    
    if stratify is not None:
        # Stratified split
        stratify = np.asarray(stratify)
        train_indices = []
        test_indices = []
        
        for label in np.unique(stratify):
            label_indices = np.where(stratify == label)[0]
            np.random.shuffle(label_indices)
            
            n_test = int(len(label_indices) * test_size / n_samples * len(stratify))
            n_test = max(1, min(n_test, len(label_indices) - 1))
            
            test_indices.extend(label_indices[:n_test])
            train_indices.extend(label_indices[n_test:])
        
        train_indices = np.array(train_indices)
        test_indices = np.array(test_indices)
        
        if shuffle:
            np.random.shuffle(train_indices)
            np.random.shuffle(test_indices)
    else:
        # Random split
        if shuffle:
            np.random.shuffle(indices)
        
        train_indices = indices[:train_size]
        test_indices = indices[train_size:train_size + test_size]
    
    # Split arrays
    result = []
    for arr in arrays:
        arr = np.asarray(arr)
        result.append(arr[train_indices])
        result.append(arr[test_indices])
    
    logger.info(f"train_test_split: train={len(train_indices)}, test={len(test_indices)}")
    return result


def cross_validation_split(
    X: ArrayLike2D,
    n_splits: int = 5,
    shuffle: bool = True,
    random_state: Optional[int] = None
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Create k-fold cross-validation splits.
    
    Args:
        X: Data to split.
        n_splits: Number of folds. Default: 5.
        shuffle: Whether to shuffle before splitting.
        random_state: Random seed.
    
    Returns:
        List[Tuple]: List of (train_indices, test_indices) tuples.
    
    Example:
        >>> X = np.random.randn(100, 5)
        >>> splits = cross_validation_split(X, n_splits=5)
        >>> len(splits)
        5
    """
    X = np.asarray(X)
    n_samples = X.shape[0]
    
    if random_state is not None:
        np.random.seed(random_state)
    
    indices = np.arange(n_samples)
    if shuffle:
        np.random.shuffle(indices)
    
    fold_size = n_samples // n_splits
    splits = []
    
    for i in range(n_splits):
        start = i * fold_size
        end = start + fold_size if i < n_splits - 1 else n_samples
        
        test_indices = indices[start:end]
        train_indices = np.concatenate([indices[:start], indices[end:]])
        
        splits.append((train_indices, test_indices))
    
    logger.info(f"cross_validation_split: {n_splits} folds")
    return splits


class FeatureSelector:
    """
    Feature selection utilities.
    
    Methods:
    - Variance threshold: Remove low-variance features
    - Correlation threshold: Remove highly correlated features
    - SelectKBest: Select top k features by score
    """
    
    def __init__(self):
        """Initialize FeatureSelector."""
        self.selected_features_: Optional[np.ndarray] = None
        self.n_features_in_: int = 0
        
        logger.debug("FeatureSelector initialized")
    
    def variance_threshold(
        self,
        X: ArrayLike2D,
        threshold: float = 0.0
    ) -> np.ndarray:
        """
        Remove features with variance below threshold.
        
        Args:
            X: Feature matrix.
            threshold: Minimum variance threshold.
        
        Returns:
            np.ndarray: Boolean mask of selected features.
        """
        X = np.asarray(X, dtype=np.float64)
        variances = np.var(X, axis=0)
        
        mask = variances > threshold
        self.selected_features_ = mask
        self.n_features_in_ = X.shape[1]
        
        logger.info(f"variance_threshold: {np.sum(mask)}/{len(mask)} features selected")
        return mask
    
    def correlation_threshold(
        self,
        X: ArrayLike2D,
        threshold: float = 0.9
    ) -> np.ndarray:
        """
        Remove highly correlated features.
        
        Args:
            X: Feature matrix.
            threshold: Maximum correlation threshold.
        
        Returns:
            np.ndarray: Boolean mask of selected features.
        """
        X = np.asarray(X, dtype=np.float64)
        n_features = X.shape[1]
        
        # Compute correlation matrix
        corr_matrix = np.corrcoef(X, rowvar=False)
        
        # Find highly correlated pairs
        mask = np.ones(n_features, dtype=bool)
        
        for i in range(n_features):
            if mask[i]:
                for j in range(i + 1, n_features):
                    if mask[j] and abs(corr_matrix[i, j]) > threshold:
                        mask[j] = False  # Remove the later feature
        
        self.selected_features_ = mask
        self.n_features_in_ = n_features
        
        logger.info(f"correlation_threshold: {np.sum(mask)}/{n_features} features selected")
        return mask
    
    def select_k_best(
        self,
        X: ArrayLike2D,
        y: ArrayLike1D,
        k: int,
        score_func: str = 'f_classif'
    ) -> np.ndarray:
        """
        Select top k features based on scoring function.
        
        Args:
            X: Feature matrix.
            y: Target variable.
            k: Number of features to select.
            score_func: Scoring function ('f_classif', 'chi2', 'mutual_info').
        
        Returns:
            np.ndarray: Boolean mask of selected features.
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        n_features = X.shape[1]
        
        k = min(k, n_features)
        scores = np.zeros(n_features)
        
        if score_func == 'f_classif':
            # ANOVA F-value
            for j in range(n_features):
                feature = X[:, j]
                classes = np.unique(y)
                
                # Between-class variance
                grand_mean = np.mean(feature)
                ss_between = sum(
                    np.sum(y == c) * (np.mean(feature[y == c]) - grand_mean) ** 2
                    for c in classes
                )
                
                # Within-class variance
                ss_within = sum(
                    np.sum((feature[y == c] - np.mean(feature[y == c])) ** 2)
                    for c in classes
                )
                
                if ss_within > 0:
                    scores[j] = (ss_between / (len(classes) - 1)) / (ss_within / (len(y) - len(classes)))
                else:
                    scores[j] = 0
        
        elif score_func == 'chi2':
            # Chi-squared statistic (for non-negative features)
            X_nonneg = X - X.min(axis=0)
            for j in range(n_features):
                observed = np.bincount(y, weights=X_nonneg[:, j])
                expected = np.sum(X_nonneg[:, j]) * np.bincount(y) / len(y)
                scores[j] = np.sum((observed - expected) ** 2 / (expected + 1e-10))
        
        elif score_func == 'mutual_info':
            # Approximate mutual information using correlation
            for j in range(n_features):
                scores[j] = abs(np.corrcoef(X[:, j], y)[0, 1])
        
        else:
            raise ValueError(f"Unknown score_func: {score_func}")
        
        # Select top k features
        top_k_indices = np.argsort(scores)[-k:]
        mask = np.zeros(n_features, dtype=bool)
        mask[top_k_indices] = True
        
        self.selected_features_ = mask
        self.n_features_in_ = n_features
        
        logger.info(f"select_k_best: {k} features selected")
        return mask
    
    def transform(self, X: ArrayLike2D) -> np.ndarray:
        """
        Transform data to selected features.
        
        Args:
            X: Feature matrix.
        
        Returns:
            np.ndarray: Transformed data.
        """
        if self.selected_features_ is None:
            raise ValueError("FeatureSelector not fitted.")
        
        X = np.asarray(X)
        return X[:, self.selected_features_]
    
    def fit_transform(
        self,
        X: ArrayLike2D,
        y: Optional[ArrayLike1D] = None,
        method: str = 'variance',
        **kwargs
    ) -> np.ndarray:
        """
        Fit and transform in one step.
        
        Args:
            X: Feature matrix.
            y: Target (for select_k_best).
            method: Selection method.
            **kwargs: Method-specific arguments.
        
        Returns:
            np.ndarray: Transformed data.
        """
        if method == 'variance':
            self.variance_threshold(X, **kwargs)
        elif method == 'correlation':
            self.correlation_threshold(X, **kwargs)
        elif method == 'select_k_best':
            if y is None:
                raise ValueError("y required for select_k_best")
            self.select_k_best(X, y, **kwargs)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return self.transform(X)


if __name__ == "__main__":
    # Example usage and demonstrations
    print("=" * 60)
    print("Preprocessing Module - Demonstration")
    print("=" * 60)
    
    np.random.seed(42)
    
    # StandardScaler
    print("\n1. StandardScaler:")
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print(f"   Original mean: {X.mean(axis=0)}")
    print(f"   Scaled mean: {X_scaled.mean(axis=0)}")
    print(f"   Scaled std: {X_scaled.std(axis=0)}")
    
    # MinMaxScaler
    print("\n2. MinMaxScaler:")
    scaler_mm = MinMaxScaler(feature_range=(0, 1))
    X_mm = scaler_mm.fit_transform(X)
    print(f"   Min: {X_mm.min(axis=0)}")
    print(f"   Max: {X_mm.max(axis=0)}")
    
    # LabelEncoder
    print("\n3. LabelEncoder:")
    labels = ['cat', 'dog', 'cat', 'bird', 'dog', 'cat']
    le = LabelEncoder()
    encoded = le.fit_transform(labels)
    print(f"   Labels: {labels}")
    print(f"   Encoded: {encoded}")
    print(f"   Classes: {le.classes_}")
    
    # OneHotEncoder
    print("\n4. OneHotEncoder:")
    categories = [['cat'], ['dog'], ['cat'], ['bird']]
    ohe = OneHotEncoder()
    encoded_ohe = ohe.fit_transform(categories)
    print(f"   Categories: {categories}")
    print(f"   One-hot encoded:\n{encoded_ohe}")
    
    # SimpleImputer
    print("\n5. SimpleImputer:")
    X_missing = np.array([[1, 2], [np.nan, 3], [7, np.nan], [4, 5]])
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X_missing)
    print(f"   With missing:\n{X_missing}")
    print(f"   Imputed:\n{X_imputed}")
    
    # Train/test split
    print("\n6. Train/Test Split:")
    X_data = np.random.randn(100, 5)
    y_data = np.random.randint(0, 2, 100)
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=42)
    print(f"   Total: {len(X_data)}")
    print(f"   Train: {len(X_train)}, Test: {len(X_test)}")
    
    # Feature selection
    print("\n7. Feature Selection:")
    X_feat = np.random.randn(100, 10)
    selector = FeatureSelector()
    mask = selector.variance_threshold(X_feat, threshold=0.5)
    print(f"   Original features: {X_feat.shape[1]}")
    print(f"   Selected features: {np.sum(mask)}")
    
    print("\n" + "=" * 60)
