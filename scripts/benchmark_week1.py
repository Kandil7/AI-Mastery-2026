"""
Benchmark Script: Week 1 Foundation
Compares Pure Python 'Matrix' implementation vs NumPy.
"""
import time
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.core.linear_algebra import Matrix

def benchmark_matmul(n: int):
    print(f"\n--- Benchmarking Matrix Multiplication ({n}x{n}) ---")
    
    # Generate random data
    data_a = np.random.rand(n, n).tolist()
    data_b = np.random.rand(n, n).tolist()
    
    # 1. Custom Implementation
    mat_a = Matrix(data_a)
    mat_b = Matrix(data_b)
    
    start_time = time.time()
    _ = mat_a @ mat_b
    custom_time = time.time() - start_time
    print(f"Custom Implementation: {custom_time:.6f} seconds")
    
    # 2. NumPy Implementation
    np_a = np.array(data_a)
    np_b = np.array(data_b)
    
    start_time = time.time()
    _ = np_a @ np_b
    np_time = time.time() - start_time
    print(f"NumPy Implementation:  {np_time:.6f} seconds")
    
    speedup = custom_time / np_time if np_time > 0 else 0
    print(f"NumPy is {speedup:.2f}x faster")

def benchmark_inverse(n: int):
    print(f"\n--- Benchmarking Matrix Inverse ({n}x{n}) ---")
    
    # Generate random data (ensure non-singular by adding identity)
    data = (np.random.rand(n, n) + np.eye(n) * n).tolist()
    
    # 1. Custom Implementation
    mat = Matrix(data)
    start_time = time.time()
    _ = mat.inverse()
    custom_time = time.time() - start_time
    print(f"Custom Implementation: {custom_time:.6f} seconds")
    
    # 2. NumPy Implementation
    np_mat = np.array(data)
    start_time = time.time()
    _ = np.linalg.inv(np_mat)
    np_time = time.time() - start_time
    print(f"NumPy Implementation:  {np_time:.6f} seconds")
    
    speedup = custom_time / np_time if np_time > 0 else 0
    print(f"NumPy is {speedup:.2f}x faster")


def benchmark_eigenvalues(n: int):
    print(f"\n--- Benchmarking Eigenvalues (Power Iteration) ({n}x{n}) ---")
    
    # Generate symmetric matrix for real eigenvalues
    data = np.random.rand(n, n)
    data = (data + data.T) / 2
    data_list = data.tolist()
    
    # 1. Custom Implementation
    mat = Matrix(data_list)
    start_time = time.time()
    eig_custom = mat.eigenvalues(iterations=50)
    custom_time = time.time() - start_time
    print(f"Custom Implementation: {custom_time:.6f} seconds (Max Eig: {eig_custom:.4f})")
    
    # 2. NumPy Implementation
    start_time = time.time()
    eig_np = np.max(np.abs(np.linalg.eigvals(data)))
    np_time = time.time() - start_time
    print(f"NumPy Implementation:  {np_time:.6f} seconds (Max Eig: {eig_np:.4f})")
    
    diff = abs(eig_custom - eig_np)
    print(f"Accuracy Diff: {diff:.6f}")
    
    speedup = custom_time / np_time if np_time > 0 else 0
    print(f"NumPy is {speedup:.2f}x faster")

if __name__ == "__main__":
    print("🚀 Week 1 Benchmark: The Cost of Abstraction")
    benchmark_matmul(50)
    benchmark_inverse(30)
    benchmark_eigenvalues(50)


    # benchmark_matmul(500) # Large (Warning: O(n^3) in Python is slow!)
