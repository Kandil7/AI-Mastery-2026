"""
Data Processing Module for Machine Learning.

This module provides comprehensive data manipulation capabilities using NumPy and Pandas,
including array operations, DataFrame manipulation, aggregation, and merging.

Example Usage:
    >>> import numpy as np
    >>> import pandas as pd
    >>> from data_processing import DataProcessor, ArrayOperations
    >>> 
    >>> # Array operations
    >>> arr_ops = ArrayOperations()
    >>> arr = np.array([[1, 2, 3], [4, 5, 6]])
    >>> normalized = arr_ops.normalize(arr)
    >>> 
    >>> # DataFrame operations
    >>> df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    >>> processor = DataProcessor()
    >>> stats = processor.describe(df)
"""

from typing import Union, List, Dict, Any, Optional, Tuple, Callable
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

ArrayLike = Union[np.ndarray, List, Series]
DataFrameLike = Union[DataFrame, Dict]


class ArrayOperations:
    """
    NumPy array operations for machine learning data processing.
    
    This class provides methods for:
    - Array creation and manipulation
    - Normalization and standardization
    - Broadcasting operations
    - Reshaping and indexing
    - Statistical operations
    
    Example:
        >>> ops = ArrayOperations()
        >>> arr = np.array([[1, 2, 3], [4, 5, 6]])
        >>> normalized = ops.normalize(arr)
        >>> standardized = ops.standardize(arr)
    """
    
    def __init__(self, epsilon: float = 1e-8):
        """
        Initialize ArrayOperations.
        
        Args:
            epsilon: Small value for numerical stability. Default: 1e-8.
        """
        self.epsilon = epsilon
        logger.debug(f"ArrayOperations initialized with epsilon={epsilon}")
    
    def normalize(
        self,
        arr: np.ndarray,
        axis: Optional[int] = None,
        norm_type: str = 'l2'
    ) -> np.ndarray:
        """
        Normalize array to unit norm.
        
        Args:
            arr: Input array.
            axis: Axis along which to normalize. None for entire array.
            norm_type: Norm type ('l1', 'l2', 'max'). Default: 'l2'.
        
        Returns:
            np.ndarray: Normalized array.
        
        Example:
            >>> ops = ArrayOperations()
            >>> arr = np.array([[3, 4], [6, 8]])
            >>> normalized = ops.normalize(arr, axis=1)
            >>> np.allclose(np.linalg.norm(normalized, axis=1), 1.0)
            True
        """
        arr = np.asarray(arr, dtype=np.float64)
        
        if axis is None:
            if norm_type == 'l2':
                norm = np.linalg.norm(arr)
            elif norm_type == 'l1':
                norm = np.sum(np.abs(arr))
            elif norm_type == 'max':
                norm = np.max(np.abs(arr))
            else:
                raise ValueError(f"Unknown norm_type: {norm_type}")
            
            return arr / (norm + self.epsilon)
        
        else:
            if norm_type == 'l2':
                norm = np.linalg.norm(arr, axis=axis, keepdims=True)
            elif norm_type == 'l1':
                norm = np.sum(np.abs(arr), axis=axis, keepdims=True)
            elif norm_type == 'max':
                norm = np.max(np.abs(arr), axis=axis, keepdims=True)
            else:
                raise ValueError(f"Unknown norm_type: {norm_type}")
            
            return arr / (norm + self.epsilon)
    
    def standardize(
        self,
        arr: np.ndarray,
        axis: Optional[int] = 0,
        return_params: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Standardize array to zero mean and unit variance (z-score normalization).
        
        Formula: z = (x - μ) / σ
        
        Args:
            arr: Input array.
            axis: Axis along which to compute statistics. Default: 0 (column-wise).
            return_params: If True, also return mean and std. Default: False.
        
        Returns:
            np.ndarray or Tuple: Standardized array, optionally with (mean, std).
        
        Example:
            >>> ops = ArrayOperations()
            >>> arr = np.array([[1, 2], [3, 4], [5, 6]])
            >>> standardized = ops.standardize(arr)
            >>> np.allclose(np.mean(standardized, axis=0), 0, atol=1e-10)
            True
            >>> np.allclose(np.std(standardized, axis=0), 1, atol=1e-10)
            True
        """
        arr = np.asarray(arr, dtype=np.float64)
        
        mean = np.mean(arr, axis=axis, keepdims=True)
        std = np.std(arr, axis=axis, keepdims=True, ddof=0)
        
        standardized = (arr - mean) / (std + self.epsilon)
        
        if return_params:
            return standardized, mean, std
        return standardized
    
    def min_max_scale(
        self,
        arr: np.ndarray,
        feature_range: Tuple[float, float] = (0, 1),
        axis: Optional[int] = 0,
        return_params: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Scale array values to a specified range.
        
        Formula: X_scaled = (X - X_min) / (X_max - X_min) * (max - min) + min
        
        Args:
            arr: Input array.
            feature_range: Target range (min, max). Default: (0, 1).
            axis: Axis along which to scale. Default: 0.
            return_params: If True, also return min and max. Default: False.
        
        Returns:
            np.ndarray or Tuple: Scaled array, optionally with (min, max).
        
        Example:
            >>> ops = ArrayOperations()
            >>> arr = np.array([[1, 2], [3, 4], [5, 6]])
            >>> scaled = ops.min_max_scale(arr, feature_range=(0, 1))
            >>> np.allclose(scaled.min(), 0) and np.allclose(scaled.max(), 1)
            True
        """
        arr = np.asarray(arr, dtype=np.float64)
        min_val, max_val = feature_range
        
        if axis is None:
            arr_min = np.min(arr)
            arr_max = np.max(arr)
        else:
            arr_min = np.min(arr, axis=axis, keepdims=True)
            arr_max = np.max(arr, axis=axis, keepdims=True)
        
        scaled = (arr - arr_min) / (arr_max - arr_min + self.epsilon)
        scaled = scaled * (max_val - min_val) + min_val
        
        if return_params:
            return scaled, arr_min, arr_max
        return scaled
    
    def robust_scale(
        self,
        arr: np.ndarray,
        axis: Optional[int] = 0
    ) -> np.ndarray:
        """
        Scale using median and IQR (robust to outliers).
        
        Formula: X_scaled = (X - median) / IQR
        
        Args:
            arr: Input array.
            axis: Axis along which to scale. Default: 0.
        
        Returns:
            np.ndarray: Robustly scaled array.
        
        Example:
            >>> ops = ArrayOperations()
            >>> arr = np.array([[1, 2], [3, 4], [100, 6]])  # Outlier
            >>> scaled = ops.robust_scale(arr)
        """
        arr = np.asarray(arr, dtype=np.float64)
        
        median = np.median(arr, axis=axis, keepdims=True)
        q1 = np.percentile(arr, 25, axis=axis, keepdims=True)
        q3 = np.percentile(arr, 75, axis=axis, keepdims=True)
        iqr = q3 - q1
        
        return (arr - median) / (iqr + self.epsilon)
    
    def one_hot_encode(
        self,
        arr: np.ndarray,
        num_classes: Optional[int] = None
    ) -> np.ndarray:
        """
        One-hot encode integer labels.
        
        Args:
            arr: Integer labels array.
            num_classes: Number of classes. If None, inferred from data.
        
        Returns:
            np.ndarray: One-hot encoded array.
        
        Example:
            >>> ops = ArrayOperations()
            >>> labels = np.array([0, 1, 2, 1, 0])
            >>> one_hot = ops.one_hot_encode(labels)
            >>> one_hot.shape
            (5, 3)
        """
        arr = np.asarray(arr, dtype=np.int32).flatten()
        
        if num_classes is None:
            num_classes = int(np.max(arr)) + 1
        
        n_samples = len(arr)
        one_hot = np.zeros((n_samples, num_classes))
        one_hot[np.arange(n_samples), arr] = 1
        
        return one_hot
    
    def label_encode(
        self,
        arr: np.ndarray,
        return_mapping: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, Dict]]:
        """
        Encode categorical labels as integers.
        
        Args:
            arr: Categorical labels.
            return_mapping: If True, also return label-to-int mapping.
        
        Returns:
            np.ndarray or Tuple: Encoded labels, optionally with mapping.
        
        Example:
            >>> ops = ArrayOperations()
            >>> labels = np.array(['cat', 'dog', 'bird', 'cat'])
            >>> encoded, mapping = ops.label_encode(labels, return_mapping=True)
            >>> mapping
            {'bird': 0, 'cat': 1, 'dog': 2}
        """
        unique_labels = np.unique(arr)
        label_to_int = {label: i for i, label in enumerate(unique_labels)}
        
        encoded = np.array([label_to_int[label] for label in arr])
        
        if return_mapping:
            return encoded, label_to_int
        return encoded
    
    def reshape(
        self,
        arr: np.ndarray,
        new_shape: Tuple[int, ...],
        order: str = 'C'
    ) -> np.ndarray:
        """
        Reshape array without changing data.
        
        Args:
            arr: Input array.
            new_shape: Target shape.
            order: Read order ('C' for row-major, 'F' for column-major).
        
        Returns:
            np.ndarray: Reshaped array.
        
        Example:
            >>> ops = ArrayOperations()
            >>> arr = np.array([[1, 2, 3], [4, 5, 6]])
            >>> reshaped = ops.reshape(arr, (3, 2))
            >>> reshaped.shape
            (3, 2)
        """
        arr = np.asarray(arr)
        return arr.reshape(new_shape, order=order)
    
    def flatten(
        self,
        arr: np.ndarray,
        order: str = 'C'
    ) -> np.ndarray:
        """
        Flatten array to 1D.
        
        Args:
            arr: Input array.
            order: Read order ('C', 'F', or 'A').
        
        Returns:
            np.ndarray: Flattened array.
        """
        return np.asarray(arr).flatten(order=order)
    
    def expand_dims(
        self,
        arr: np.ndarray,
        axis: int
    ) -> np.ndarray:
        """
        Add a new axis to array.
        
        Args:
            arr: Input array.
            axis: Position of new axis.
        
        Returns:
            np.ndarray: Array with expanded dimensions.
        
        Example:
            >>> ops = ArrayOperations()
            >>> arr = np.array([1, 2, 3])
            >>> expanded = ops.expand_dims(arr, axis=0)
            >>> expanded.shape
            (1, 3)
        """
        return np.expand_dims(np.asarray(arr), axis=axis)
    
    def squeeze(
        self,
        arr: np.ndarray,
        axis: Optional[Union[int, Tuple[int, ...]]] = None
    ) -> np.ndarray:
        """
        Remove single-dimensional entries.
        
        Args:
            arr: Input array.
            axis: Specific axes to squeeze. None removes all size-1 dims.
        
        Returns:
            np.ndarray: Squeezed array.
        """
        return np.squeeze(np.asarray(arr), axis=axis)
    
    def concatenate(
        self,
        arrays: List[np.ndarray],
        axis: int = 0
    ) -> np.ndarray:
        """
        Concatenate arrays along an axis.
        
        Args:
            arrays: List of arrays to concatenate.
            axis: Axis along which to concatenate.
        
        Returns:
            np.ndarray: Concatenated array.
        """
        return np.concatenate([np.asarray(a) for a in arrays], axis=axis)
    
    def stack(
        self,
        arrays: List[np.ndarray],
        axis: int = 0
    ) -> np.ndarray:
        """
        Stack arrays along a new axis.
        
        Args:
            arrays: List of arrays to stack.
            axis: Position of new axis.
        
        Returns:
            np.ndarray: Stacked array.
        """
        return np.stack([np.asarray(a) for a in arrays], axis=axis)
    
    def split(
        self,
        arr: np.ndarray,
        indices_or_sections: Union[int, List[int]],
        axis: int = 0
    ) -> List[np.ndarray]:
        """
        Split array into multiple sub-arrays.
        
        Args:
            arr: Input array.
            indices_or_sections: Number of splits or split indices.
            axis: Axis along which to split.
        
        Returns:
            List[np.ndarray]: List of sub-arrays.
        """
        return np.split(np.asarray(arr), indices_or_sections, axis=axis)
    
    def pad(
        self,
        arr: np.ndarray,
        pad_width: Union[int, Tuple[int, int], List[Tuple[int, int]]],
        mode: str = 'constant',
        constant_values: Union[float, Tuple[float, float]] = 0
    ) -> np.ndarray:
        """
        Pad array with specified values.
        
        Args:
            arr: Input array.
            pad_width: Number of values to pad.
            mode: Padding mode ('constant', 'edge', 'reflect', etc.).
            constant_values: Value for 'constant' mode.
        
        Returns:
            np.ndarray: Padded array.
        """
        return np.pad(np.asarray(arr), pad_width, mode=mode, 
                     constant_values=constant_values)
    
    def rolling_window(
        self,
        arr: np.ndarray,
        window_size: int,
        step: int = 1,
        axis: int = 0
    ) -> np.ndarray:
        """
        Create rolling windows over array.
        
        Args:
            arr: Input array.
            window_size: Size of rolling window.
            step: Step size between windows.
            axis: Axis along which to create windows.
        
        Returns:
            np.ndarray: Array of rolling windows.
        
        Example:
            >>> ops = ArrayOperations()
            >>> arr = np.array([1, 2, 3, 4, 5])
            >>> windows = ops.rolling_window(arr, window_size=3)
            >>> windows
            array([[1, 2, 3],
                   [2, 3, 4],
                   [3, 4, 5]])
        """
        arr = np.asarray(arr)
        
        if axis != 0:
            arr = np.moveaxis(arr, axis, 0)
        
        n = arr.shape[0]
        n_windows = (n - window_size) // step + 1
        
        indices = np.arange(window_size)[None, :] + step * np.arange(n_windows)[:, None]
        windows = arr[indices]
        
        if axis != 0:
            windows = np.moveaxis(windows, 0, axis + 1)
        
        return windows
    
    def moving_average(
        self,
        arr: np.ndarray,
        window_size: int,
        axis: int = 0
    ) -> np.ndarray:
        """
        Compute moving average.
        
        Args:
            arr: Input array.
            window_size: Size of moving window.
            axis: Axis along which to compute.
        
        Returns:
            np.ndarray: Moving average array.
        """
        arr = np.asarray(arr, dtype=np.float64)
        
        if axis != 0:
            arr = np.moveaxis(arr, axis, 0)
        
        kernel = np.ones(window_size) / window_size
        result = np.convolve(arr.flatten(), kernel, mode='valid')
        
        if axis != 0:
            result = np.moveaxis(result, 0, axis)
        
        return result


class DataProcessor:
    """
    Pandas DataFrame processing for machine learning.
    
    This class provides methods for:
    - Data loading and inspection
    - Data cleaning and transformation
    - Aggregation and grouping
    - Merging and joining
    - Statistical summaries
    
    Example:
        >>> import pandas as pd
        >>> processor = DataProcessor()
        >>> df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        >>> stats = processor.describe(df)
        >>> cleaned = processor.handle_missing(df, strategy='mean')
    """
    
    def __init__(self):
        """Initialize DataProcessor."""
        logger.debug("DataProcessor initialized")
    
    def load_csv(
        self,
        filepath: str,
        **kwargs
    ) -> DataFrame:
        """
        Load data from CSV file.
        
        Args:
            filepath: Path to CSV file.
            **kwargs: Additional arguments for pd.read_csv.
        
        Returns:
            DataFrame: Loaded data.
        """
        df = pd.read_csv(filepath, **kwargs)
        logger.info(f"Loaded CSV from {filepath}: {df.shape}")
        return df
    
    def load_json(
        self,
        filepath: str,
        **kwargs
    ) -> DataFrame:
        """
        Load data from JSON file.
        
        Args:
            filepath: Path to JSON file.
            **kwargs: Additional arguments for pd.read_json.
        
        Returns:
            DataFrame: Loaded data.
        """
        df = pd.read_json(filepath, **kwargs)
        logger.info(f"Loaded JSON from {filepath}: {df.shape}")
        return df
    
    def load_excel(
        self,
        filepath: str,
        sheet_name: Optional[Union[str, int]] = 0,
        **kwargs
    ) -> DataFrame:
        """
        Load data from Excel file.
        
        Args:
            filepath: Path to Excel file.
            sheet_name: Sheet name or index.
            **kwargs: Additional arguments for pd.read_excel.
        
        Returns:
            DataFrame: Loaded data.
        """
        df = pd.read_excel(filepath, sheet_name=sheet_name, **kwargs)
        logger.info(f"Loaded Excel from {filepath}: {df.shape}")
        return df
    
    def info(self, df: DataFrame) -> Dict[str, Any]:
        """
        Get comprehensive information about DataFrame.
        
        Args:
            df: Input DataFrame.
        
        Returns:
            Dict: DataFrame information.
        
        Example:
            >>> processor = DataProcessor()
            >>> df = pd.DataFrame({'A': [1, 2, None], 'B': ['a', 'b', 'c']})
            >>> info = processor.info(df)
            >>> 'n_rows' in info and 'n_cols' in info
            True
        """
        return {
            'n_rows': len(df),
            'n_cols': len(df.columns),
            'columns': list(df.columns),
            'dtypes': df.dtypes.to_dict(),
            'non_null': df.count().to_dict(),
            'null_counts': df.isnull().sum().to_dict(),
            'memory_usage': df.memory_usage(deep=True).sum(),
        }
    
    def describe(
        self,
        df: DataFrame,
        include: str = 'all',
        percentiles: Optional[List[float]] = None
    ) -> DataFrame:
        """
        Generate descriptive statistics.
        
        Args:
            df: Input DataFrame.
            include: Columns to include ('all', 'numeric', 'object').
            percentiles: Percentiles to include. Default: [0.25, 0.5, 0.75].
        
        Returns:
            DataFrame: Descriptive statistics.
        """
        if percentiles is None:
            percentiles = [0.25, 0.5, 0.75]
        
        return df.describe(include=include, percentiles=percentiles)
    
    def handle_missing(
        self,
        df: DataFrame,
        strategy: str = 'drop',
        columns: Optional[List[str]] = None,
        fill_value: Optional[Any] = None
    ) -> DataFrame:
        """
        Handle missing values in DataFrame.
        
        Strategies:
        - 'drop': Remove rows with missing values
        - 'mean': Fill with column mean (numeric only)
        - 'median': Fill with column median (numeric only)
        - 'mode': Fill with column mode
        - 'ffill': Forward fill
        - 'bfill': Backward fill
        - 'constant': Fill with specified value
        
        Args:
            df: Input DataFrame.
            strategy: Handling strategy. Default: 'drop'.
            columns: Specific columns to process. None for all.
            fill_value: Value for 'constant' strategy.
        
        Returns:
            DataFrame: DataFrame with missing values handled.
        
        Example:
            >>> processor = DataProcessor()
            >>> df = pd.DataFrame({'A': [1, 2, None], 'B': [4, None, 6]})
            >>> cleaned = processor.handle_missing(df, strategy='mean')
        """
        df = df.copy()
        
        if columns is None:
            columns = df.columns.tolist()
        
        if strategy == 'drop':
            df = df.dropna(subset=columns)
        
        elif strategy == 'mean':
            for col in columns:
                if pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = df[col].fillna(df[col].mean())
        
        elif strategy == 'median':
            for col in columns:
                if pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = df[col].fillna(df[col].median())
        
        elif strategy == 'mode':
            for col in columns:
                df[col] = df[col].fillna(df[col].mode().iloc[0] if len(df[col].mode()) > 0 else df[col])
        
        elif strategy == 'ffill':
            df[columns] = df[columns].ffill()
        
        elif strategy == 'bfill':
            df[columns] = df[columns].bfill()
        
        elif strategy == 'constant':
            df[columns] = df[columns].fillna(fill_value)
        
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        logger.info(f"Handled missing values with strategy '{strategy}'")
        return df
    
    def remove_duplicates(
        self,
        df: DataFrame,
        subset: Optional[List[str]] = None,
        keep: str = 'first'
    ) -> DataFrame:
        """
        Remove duplicate rows.
        
        Args:
            df: Input DataFrame.
            subset: Columns to consider for duplicates. None for all.
            keep: Which duplicate to keep ('first', 'last', False).
        
        Returns:
            DataFrame: DataFrame with duplicates removed.
        """
        result = df.drop_duplicates(subset=subset, keep=keep)
        removed = len(df) - len(result)
        logger.info(f"Removed {removed} duplicate rows")
        return result
    
    def rename_columns(
        self,
        df: DataFrame,
        mapping: Dict[str, str]
    ) -> DataFrame:
        """
        Rename columns.
        
        Args:
            df: Input DataFrame.
            mapping: Old name to new name mapping.
        
        Returns:
            DataFrame: DataFrame with renamed columns.
        """
        return df.rename(columns=mapping)
    
    def select_columns(
        self,
        df: DataFrame,
        columns: Union[List[str], Callable[[str], bool]]
    ) -> DataFrame:
        """
        Select specific columns.
        
        Args:
            df: Input DataFrame.
            columns: Column names or predicate function.
        
        Returns:
            DataFrame: DataFrame with selected columns.
        
        Example:
            >>> processor = DataProcessor()
            >>> df = pd.DataFrame({'A': [1], 'B': [2], 'C': [3]})
            >>> selected = processor.select_columns(df, ['A', 'C'])
            >>> list(selected.columns)
            ['A', 'C']
        """
        if callable(columns):
            columns = [col for col in df.columns if columns(col)]
        
        return df[columns]
    
    def filter_rows(
        self,
        df: DataFrame,
        condition: Union[str, Callable[[DataFrame], Series]]
    ) -> DataFrame:
        """
        Filter rows based on condition.
        
        Args:
            df: Input DataFrame.
            condition: Query string or predicate function.
        
        Returns:
            DataFrame: Filtered DataFrame.
        
        Example:
            >>> processor = DataProcessor()
            >>> df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
            >>> filtered = processor.filter_rows(df, 'A > 1')
            >>> len(filtered)
            2
        """
        if callable(condition):
            mask = condition(df)
        else:
            mask = df.eval(condition)
        
        return df[mask]
    
    def sort(
        self,
        df: DataFrame,
        by: Union[str, List[str]],
        ascending: Union[bool, List[bool]] = True
    ) -> DataFrame:
        """
        Sort DataFrame by columns.
        
        Args:
            df: Input DataFrame.
            by: Column(s) to sort by.
            ascending: Sort order.
        
        Returns:
            DataFrame: Sorted DataFrame.
        """
        return df.sort_values(by=by, ascending=ascending)
    
    def group_by(
        self,
        df: DataFrame,
        by: Union[str, List[str]],
        aggregations: Optional[Dict[str, Union[str, List[str]]]] = None
    ) -> Union[DataFrameGroupBy, DataFrame]:
        """
        Group DataFrame and apply aggregations.
        
        Args:
            df: Input DataFrame.
            by: Column(s) to group by.
            aggregations: Column to aggregation function mapping.
        
        Returns:
            DataFrame: Aggregated DataFrame or GroupBy object.
        
        Example:
            >>> processor = DataProcessor()
            >>> df = pd.DataFrame({'A': ['x', 'x', 'y'], 'B': [1, 2, 3]})
            >>> grouped = processor.group_by(df, 'A', {'B': 'mean'})
            >>> grouped['B'].iloc[0]
            1.5
        """
        grouped = df.groupby(by)
        
        if aggregations is not None:
            return grouped.agg(aggregations)
        
        return grouped
    
    def pivot(
        self,
        df: DataFrame,
        index: Union[str, List[str]],
        columns: Union[str, List[str]],
        values: Optional[Union[str, List[str]]] = None,
        aggfunc: str = 'mean'
    ) -> DataFrame:
        """
        Create pivot table.
        
        Args:
            df: Input DataFrame.
            index: Column(s) for index.
            columns: Column(s) for columns.
            values: Column(s) for values.
            aggfunc: Aggregation function.
        
        Returns:
            DataFrame: Pivot table.
        """
        return df.pivot_table(
            index=index,
            columns=columns,
            values=values,
            aggfunc=aggfunc
        )
    
    def melt(
        self,
        df: DataFrame,
        id_vars: Optional[List[str]] = None,
        value_vars: Optional[List[str]] = None,
        var_name: str = 'variable',
        value_name: str = 'value'
    ) -> DataFrame:
        """
        Unpivot DataFrame from wide to long format.
        
        Args:
            df: Input DataFrame.
            id_vars: Columns to keep as identifiers.
            value_vars: Columns to unpivot.
            var_name: Name for variable column.
            value_name: Name for value column.
        
        Returns:
            DataFrame: Melted DataFrame.
        """
        return df.melt(
            id_vars=id_vars,
            value_vars=value_vars,
            var_name=var_name,
            value_name=value_name
        )
    
    def merge(
        self,
        left: DataFrame,
        right: DataFrame,
        on: Optional[Union[str, List[str]]] = None,
        left_on: Optional[Union[str, List[str]]] = None,
        right_on: Optional[Union[str, List[str]]] = None,
        how: str = 'inner'
    ) -> DataFrame:
        """
        Merge two DataFrames.
        
        Args:
            left: Left DataFrame.
            right: Right DataFrame.
            on: Column(s) to join on (must be in both).
            left_on: Column(s) in left DataFrame.
            right_on: Column(s) in right DataFrame.
            how: Join type ('inner', 'outer', 'left', 'right').
        
        Returns:
            DataFrame: Merged DataFrame.
        """
        return pd.merge(
            left, right,
            on=on,
            left_on=left_on,
            right_on=right_on,
            how=how
        )
    
    def join(
        self,
        left: DataFrame,
        right: DataFrame,
        on: Optional[str] = None,
        how: str = 'left'
    ) -> DataFrame:
        """
        Join two DataFrames on index.
        
        Args:
            left: Left DataFrame.
            right: Right DataFrame.
            on: Column in left to join on.
            how: Join type.
        
        Returns:
            DataFrame: Joined DataFrame.
        """
        return left.join(right, on=on, how=how)
    
    def concat(
        self,
        dataframes: List[DataFrame],
        axis: int = 0,
        ignore_index: bool = False
    ) -> DataFrame:
        """
        Concatenate DataFrames.
        
        Args:
            dataframes: List of DataFrames to concatenate.
            axis: Axis to concatenate along.
            ignore_index: If True, reset index.
        
        Returns:
            DataFrame: Concatenated DataFrame.
        """
        return pd.concat(dataframes, axis=axis, ignore_index=ignore_index)
    
    def apply(
        self,
        df: DataFrame,
        func: Callable,
        axis: int = 0,
        columns: Optional[List[str]] = None
    ) -> DataFrame:
        """
        Apply function to DataFrame.
        
        Args:
            df: Input DataFrame.
            func: Function to apply.
            axis: Axis to apply along (0=columns, 1=rows).
            columns: Specific columns to apply to.
        
        Returns:
            DataFrame: Transformed DataFrame.
        """
        if columns is not None:
            df = df.copy()
            df[columns] = df[columns].apply(func, axis=axis)
            return df
        return df.apply(func, axis=axis)
    
    def map_column(
        self,
        df: DataFrame,
        column: str,
        mapping: Union[Dict, Callable]
    ) -> DataFrame:
        """
        Map values in a column.
        
        Args:
            df: Input DataFrame.
            column: Column to map.
            mapping: Dict or function for mapping.
        
        Returns:
            DataFrame: DataFrame with mapped column.
        """
        df = df.copy()
        df[column] = df[column].map(mapping)
        return df
    
    def bin_column(
        self,
        df: DataFrame,
        column: str,
        bins: Union[int, List[float]],
        labels: Optional[List[str]] = None,
        right: bool = True
    ) -> DataFrame:
        """
        Bin continuous values into discrete intervals.
        
        Args:
            df: Input DataFrame.
            column: Column to bin.
            bins: Number of bins or bin edges.
            labels: Labels for bins.
            right: If True, bins are right-closed.
        
        Returns:
            DataFrame: DataFrame with binned column.
        """
        df = df.copy()
        df[f'{column}_binned'] = pd.cut(
            df[column],
            bins=bins,
            labels=labels,
            right=right
        )
        return df
    
    def get_dummies(
        self,
        df: DataFrame,
        columns: Optional[List[str]] = None,
        drop_first: bool = False
    ) -> DataFrame:
        """
        Convert categorical variables to dummy/indicator variables.
        
        Args:
            df: Input DataFrame.
            columns: Columns to encode. None for all object columns.
            drop_first: If True, drop first category (avoid multicollinearity).
        
        Returns:
            DataFrame: DataFrame with dummy variables.
        """
        if columns is None:
            columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        return pd.get_dummies(df, columns=columns, drop_first=drop_first)
    
    def correlation_matrix(
        self,
        df: DataFrame,
        method: str = 'pearson'
    ) -> DataFrame:
        """
        Compute correlation matrix.
        
        Args:
            df: Input DataFrame.
            method: Correlation method ('pearson', 'spearman', 'kendall').
        
        Returns:
            DataFrame: Correlation matrix.
        """
        return df.corr(method=method)
    
    def covariance_matrix(self, df: DataFrame) -> DataFrame:
        """
        Compute covariance matrix.
        
        Args:
            df: Input DataFrame.
        
        Returns:
            DataFrame: Covariance matrix.
        """
        return df.cov()
    
    def sample(
        self,
        df: DataFrame,
        n: Optional[int] = None,
        frac: Optional[float] = None,
        replace: bool = False,
        random_state: Optional[int] = None
    ) -> DataFrame:
        """
        Random sample of DataFrame.
        
        Args:
            df: Input DataFrame.
            n: Number of rows to sample.
            frac: Fraction of rows to sample.
            replace: Sample with replacement.
            random_state: Random seed.
        
        Returns:
            DataFrame: Sampled DataFrame.
        """
        return df.sample(
            n=n,
            frac=frac,
            replace=replace,
            random_state=random_state
        )
    
    def stratified_sample(
        self,
        df: DataFrame,
        column: str,
        n: Optional[int] = None,
        frac: Optional[float] = None,
        random_state: Optional[int] = None
    ) -> DataFrame:
        """
        Stratified random sample preserving class distribution.
        
        Args:
            df: Input DataFrame.
            column: Column to stratify by.
            n: Number of rows per stratum.
            frac: Fraction of rows per stratum.
            random_state: Random seed.
        
        Returns:
            DataFrame: Stratified sample.
        """
        return df.groupby(column, group_keys=False).apply(
            lambda x: x.sample(n=n, frac=frac, random_state=random_state)
        )
    
    def cross_tabulate(
        self,
        df: DataFrame,
        index: str,
        columns: str,
        values: Optional[str] = None,
        aggfunc: Optional[str] = None,
        normalize: bool = False
    ) -> DataFrame:
        """
        Create cross-tabulation table.
        
        Args:
            df: Input DataFrame.
            index: Row variable.
            columns: Column variable.
            values: Values to aggregate.
            aggfunc: Aggregation function.
            normalize: If True, return proportions.
        
        Returns:
            DataFrame: Cross-tabulation table.
        """
        return pd.crosstab(
            df[index],
            df[columns],
            values=df[values] if values else None,
            aggfunc=aggfunc,
            normalize=normalize
        )
    
    def detect_outliers_iqr(
        self,
        df: DataFrame,
        columns: Optional[List[str]] = None,
        multiplier: float = 1.5
    ) -> DataFrame:
        """
        Detect outliers using IQR method.
        
        Args:
            df: Input DataFrame.
            columns: Columns to check. None for all numeric.
            multiplier: IQR multiplier for bounds. Default: 1.5.
        
        Returns:
            DataFrame: Boolean mask of outliers.
        """
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        outlier_mask = pd.DataFrame(False, index=df.index, columns=columns)
        
        for col in columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower = Q1 - multiplier * IQR
            upper = Q3 + multiplier * IQR
            
            outlier_mask[col] = (df[col] < lower) | (df[col] > upper)
        
        return outlier_mask
    
    def remove_outliers_iqr(
        self,
        df: DataFrame,
        columns: Optional[List[str]] = None,
        multiplier: float = 1.5
    ) -> DataFrame:
        """
        Remove outliers using IQR method.
        
        Args:
            df: Input DataFrame.
            columns: Columns to check.
            multiplier: IQR multiplier.
        
        Returns:
            DataFrame: DataFrame with outliers removed.
        """
        outlier_mask = self.detect_outliers_iqr(df, columns, multiplier)
        
        # Remove rows where any column is an outlier
        clean_mask = ~outlier_mask.any(axis=1)
        
        logger.info(f"Removed {outlier_mask.sum().sum()} outlier values")
        return df[clean_mask]


if __name__ == "__main__":
    # Example usage and demonstrations
    print("=" * 60)
    print("Data Processing Module - Demonstration")
    print("=" * 60)
    
    # Array operations
    print("\n1. Array Operations:")
    arr_ops = ArrayOperations()
    
    arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    print(f"   Original:\n{arr}")
    
    normalized = arr_ops.normalize(arr, axis=1)
    print(f"   L2 Normalized (row-wise):\n{normalized}")
    
    standardized = arr_ops.standardize(arr)
    print(f"   Standardized:\n{standardized}")
    
    scaled = arr_ops.min_max_scale(arr, feature_range=(0, 1))
    print(f"   Min-Max Scaled:\n{scaled}")
    
    # One-hot encoding
    labels = np.array([0, 1, 2, 1, 0])
    one_hot = arr_ops.one_hot_encode(labels)
    print(f"\n   Labels: {labels}")
    print(f"   One-hot encoded:\n{one_hot}")
    
    # DataFrame operations
    print("\n2. DataFrame Operations:")
    processor = DataProcessor()
    
    df = pd.DataFrame({
        'name': ['Alice', 'Bob', 'Charlie', 'Diana'],
        'age': [25, 30, 35, 28],
        'salary': [50000, 60000, 75000, 55000],
        'department': ['Engineering', 'Sales', 'Engineering', 'Marketing']
    })
    
    print(f"   Original DataFrame:\n{df}")
    
    info = processor.info(df)
    print(f"\n   Info: {info['n_rows']} rows, {info['n_cols']} columns")
    
    print(f"\n   Describe:\n{processor.describe(df, include='numeric')}")
    
    # Group by
    grouped = processor.group_by(df, 'department', {'salary': 'mean', 'age': 'mean'})
    print(f"\n   Grouped by department:\n{grouped}")
    
    # One-hot encode
    encoded = processor.get_dummies(df, columns=['department'])
    print(f"\n   One-hot encoded:\n{encoded}")
    
    # Missing values
    df_missing = df.copy()
    df_missing.loc[0, 'salary'] = np.nan
    df_missing.loc[2, 'age'] = np.nan
    print(f"\n   With missing values:\n{df_missing}")
    
    cleaned = processor.handle_missing(df_missing, strategy='mean')
    print(f"\n   After handling missing:\n{cleaned}")
    
    print("\n" + "=" * 60)
