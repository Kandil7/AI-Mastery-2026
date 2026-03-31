"""
Module 1.2: Python for Machine Learning.

This module provides Python implementations for ML data processing and algorithms:
- Data processing with NumPy and Pandas
- ML algorithms from scratch (Linear/Logistic Regression, Decision Trees, Random Forest, K-Means, PCA)
- Preprocessing utilities (scaling, encoding, imputation, train/test split)

Example Usage:
    >>> from module_1_2_python import DataProcessor, ArrayOperations
    >>> from module_1_2_python import LinearRegression, LogisticRegression
    >>> from module_1_2_python import DecisionTreeClassifier, RandomForestClassifier
    >>> from module_1_2_python import KMeans, PCA
    >>> from module_1_2_python import StandardScaler, OneHotEncoder, train_test_split
    >>> 
    >>> # Data processing
    >>> import pandas as pd
    >>> import numpy as np
    >>> processor = DataProcessor()
    >>> df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    >>> stats = processor.describe(df)
    >>> 
    >>> # ML algorithms
    >>> X = np.random.randn(100, 5)
    >>> y = np.random.randint(0, 2, 100)
    >>> model = LogisticRegression()
    >>> model.fit(X, y)
    >>> predictions = model.predict(X)
    >>> 
    >>> # Preprocessing
    >>> scaler = StandardScaler()
    >>> X_scaled = scaler.fit_transform(X)
    >>> X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)
"""

from .data_processing import (
    ArrayOperations,
    DataProcessor,
)

from .ml_algorithms import (
    LinearRegression,
    LogisticRegression,
    DecisionTreeClassifier,
    DecisionTreeNode,
    RandomForestClassifier,
    KMeans,
    PCA,
    ModelMetrics,
)

from .preprocessing import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    LabelEncoder,
    OneHotEncoder,
    OrdinalEncoder,
    SimpleImputer,
    FeatureSelector,
    train_test_split,
    cross_validation_split,
)

__all__ = [
    # Data processing
    'ArrayOperations',
    'DataProcessor',
    
    # ML algorithms
    'LinearRegression',
    'LogisticRegression',
    'DecisionTreeClassifier',
    'DecisionTreeNode',
    'RandomForestClassifier',
    'KMeans',
    'PCA',
    'ModelMetrics',
    
    # Preprocessing
    'StandardScaler',
    'MinMaxScaler',
    'RobustScaler',
    'LabelEncoder',
    'OneHotEncoder',
    'OrdinalEncoder',
    'SimpleImputer',
    'FeatureSelector',
    'train_test_split',
    'cross_validation_split',
]

__version__ = '1.0.0'
__author__ = 'AI Mastery 2026'
