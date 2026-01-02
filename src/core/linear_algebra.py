"""
Pure Python Implementation of Linear Algebra Components.
This module strictly AVOIDS NumPy to build intuition for the underlying math.
"""
import math
from typing import List, Union, Tuple

class Vector:
    def __init__(self, data: List[float]):
        self.data = [float(x) for x in data]
        self.size = len(data)

    def __repr__(self):
        return f"Vector({self.data})"

    def __add__(self, other: 'Vector') -> 'Vector':
        if self.size != other.size:
            raise ValueError("Vectors must be same size")
        return Vector([a + b for a, b in zip(self.data, other.data)])

    def __sub__(self, other: 'Vector') -> 'Vector':
        if self.size != other.size:
            raise ValueError("Vectors must be same size")
        return Vector([a - b for a, b in zip(self.data, other.data)])

    def __mul__(self, other: Union[float, int]) -> 'Vector':
        """Scalar multiplication"""
        return Vector([x * other for x in self.data])

    def dot(self, other: 'Vector') -> float:
        """Dot product"""
        if self.size != other.size:
            raise ValueError("Vectors must be same size")
        return sum(a * b for a, b in zip(self.data, other.data))

    def norm(self) -> float:
        """L2 Norm (Euclidean distance)"""
        return math.sqrt(sum(x**2 for x in self.data))


class Matrix:
    def __init__(self, data: List[List[float]]):
        """
        Args:
            data: List of lists (rows). Assumes valid rectangular shape.
        """
        self.data = [[float(x) for x in row] for row in data]
        self.rows = len(data)
        self.cols = len(data[0]) if self.rows > 0 else 0

    def __repr__(self):
        return f"Matrix({self.rows}x{self.cols})"

    def shape(self) -> Tuple[int, int]:
        return (self.rows, self.cols)

    def row(self, idx: int) -> Vector:
        return Vector(self.data[idx])

    def col(self, idx: int) -> Vector:
        return Vector([self.data[r][idx] for r in range(self.rows)])

    def transpose(self) -> 'Matrix':
        """Swap rows and cols"""
        new_data = [[self.data[r][c] for r in range(self.rows)] for c in range(self.cols)]
        return Matrix(new_data)

    def matmul(self, other: 'Matrix') -> 'Matrix':
        """Matrix Multiplication (Naive O(n^3))"""
        if self.cols != other.rows:
            raise ValueError(f"Shape mismatch: ({self.rows},{self.cols}) x ({other.rows},{other.cols})")
        
        result = [[0.0 for _ in range(other.cols)] for _ in range(self.rows)]
        
        # Optimization: Pre-fetch other columns to avoid cache misses (in C), 
        # here likely just list access improvements
        other_t = other.transpose() 
        
        for i in range(self.rows):
            row_vec = self.data[i]
            for j in range(other.cols):
                # dot product of row i and col j
                col_vec = other_t.data[j]
                val = sum(a * b for a, b in zip(row_vec, col_vec))
                result[i][j] = val
                
        return Matrix(result)

    def __matmul__(self, other: 'Matrix') -> 'Matrix':
        return self.matmul(other)

    def copy(self) -> 'Matrix':
        """Deep copy of the matrix"""
        return Matrix([row[:] for row in self.data])

    def __getitem__(self, idx: Tuple[int, int]) -> float:
        return self.data[idx[0]][idx[1]]

    def __setitem__(self, idx: Tuple[int, int], val: float):
        self.data[idx[0]][idx[1]] = float(val)

    @staticmethod
    def zeros(rows: int, cols: int) -> 'Matrix':
        return Matrix([[0.0] * cols for _ in range(rows)])

    @staticmethod
    def identity(n: int) -> 'Matrix':
        data = [[0.0] * n for _ in range(n)]
        for i in range(n):
            data[i][i] = 1.0
        return Matrix(data)

    def inverse(self) -> 'Matrix':
        """Gauss-Jordan Elimination for Inverse (Pure Python)"""
        if self.rows != self.cols:
            raise ValueError("Matrix must be square")
        
        n = self.rows
        # Augmented matrix [A | I]
        aug = [self.data[i][:] + [0.0]*n for i in range(n)]
        for i in range(n):
            aug[i][n+i] = 1.0
            
        # Gauss-Jordan
        for i in range(n):
            # Pivot
            pivot = aug[i][i]
            if abs(pivot) < 1e-10:
                raise ValueError("Matrix is singular")
                
            # Scale row i
            for j in range(2*n):
                aug[i][j] /= pivot
                
            # Eliminate other rows
            for k in range(n):
                if k != i:
                    factor = aug[k][i]
                    for j in range(2*n):
                        aug[k][j] -= factor * aug[i][j]
                        
        # Extract inverse (right half)
        inv_data = [row[n:] for row in aug]
        return Matrix(inv_data)

