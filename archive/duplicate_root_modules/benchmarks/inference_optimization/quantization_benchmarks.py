"""
Quantization Benchmark Script
=============================
Compares performance (latency) and memory footprint of different precision formats.
Focuses on Matrix Multiplication (Linear Layers), the bottleneck in LLMs.

Precisions tested:
1. FP32 (Single Precision)
2. FP16 / BF16 (Half Precision)
3. INT8 (8-bit Integer - Simulated/Real)
"""

import time
import torch
import torch.nn as nn
import numpy as np
import psutil
import os
import gc

# Configure
MATRIX_SIZE = (4096, 4096)  # Simulates a large layer in a 7B model
BATCH_SIZE = 128
ITERATIONS = 50
WARMUP = 10

def get_memory_usage_mb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def benchmark_op(name, op_func, input_data):
    # Warmup
    for _ in range(WARMUP):
        _ = op_func(input_data)
    
    # Timing
    start_time = time.time()
    for _ in range(ITERATIONS):
        _ = op_func(input_data)
    end_time = time.time()
    
    avg_latency = (end_time - start_time) / ITERATIONS * 1000  # ms
    return avg_latency

def run_benchmarks():
    print(f"Benchmarking Matrix Multiplication {MATRIX_SIZE} x {MATRIX_SIZE}")
    print(f"System: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. FP32
    print("-" * 30)
    print("Benchmarking FP32...")
    gc.collect()
    mem_before = get_memory_usage_mb()
    
    weight_fp32 = torch.randn(MATRIX_SIZE, device=device, dtype=torch.float32)
    input_fp32 = torch.randn((BATCH_SIZE, MATRIX_SIZE[0]), device=device, dtype=torch.float32)
    
    def run_fp32(x):
        return torch.matmul(x, weight_fp32)
    
    lat_fp32 = benchmark_op("FP32", run_fp32, input_fp32)
    mem_fp32 = get_memory_usage_mb() - mem_before
    # Note: RSS is rough estimate, but weight size is known: 4096*4096*4 bytes = 64MB
    
    print(f"FP32 Latency: {lat_fp32:.2f} ms")
    print(f"FP32 Est. Weight Memory: {weight_fp32.element_size() * weight_fp32.numel() / 1024**2:.2f} MB")
    
    # Clean up
    del weight_fp32, input_fp32
    gc.collect()
    
    # 2. FP16 (if supported)
    print("-" * 30)
    print("Benchmarking FP16/BF16...")
    try:
        dtype = torch.float16 if device.type == 'cuda' else torch.bfloat16
        weight_fp16 = torch.randn(MATRIX_SIZE, device=device, dtype=dtype)
        input_fp16 = torch.randn((BATCH_SIZE, MATRIX_SIZE[0]), device=device, dtype=dtype)
        
        def run_fp16(x):
            return torch.matmul(x, weight_fp16)
        
        lat_fp16 = benchmark_op("FP16", run_fp16, input_fp16)
        
        print(f"FP16 Latency: {lat_fp16:.2f} ms")
        print(f"FP16 Speedup: {lat_fp32 / lat_fp16:.2f}x")
        print(f"FP16 Est. Weight Memory: {weight_fp16.element_size() * weight_fp16.numel() / 1024**2:.2f} MB")
        
        del weight_fp16, input_fp16
        gc.collect()
        
    except Exception as e:
        print(f"FP16 Benchmark failed (likely hardware support): {e}")

    # 3. INT8 (Quantized)
    print("-" * 30)
    print("Benchmarking INT8 (PyTorch Quantization)...")
    try:
        # Prepare for quantization
        weight_fp32_for_q = torch.randn(MATRIX_SIZE, dtype=torch.float32)
        # Dynamic quantization for Linear layer
        linear = nn.Linear(MATRIX_SIZE[0], MATRIX_SIZE[1])
        linear.weight.data = weight_fp32_for_q
        
        quantized_linear = torch.quantization.quantize_dynamic(
            linear, {nn.Linear}, dtype=torch.qint8
        )
        
        # Benchmarking quantized model is tricky on some CPUs without AVX512 VNNI, but let's try
        input_q = torch.randn((BATCH_SIZE, MATRIX_SIZE[0]), dtype=torch.float32)
        
        def run_int8(x):
            return quantized_linear(x)
        
        lat_int8 = benchmark_op("INT8", run_int8, input_q)
        
        # Memory of quantized weights (rough)
        # QInt8 is 1 byte per element + scales
        q_size_mb = (MATRIX_SIZE[0] * MATRIX_SIZE[1]) * 1 / 1024**2
        
        print(f"INT8 Latency: {lat_int8:.2f} ms")
        print(f"INT8 Speedup vs FP32: {lat_fp32 / lat_int8:.2f}x")
        print(f"INT8 Est. Weight Memory: ~{q_size_mb:.2f} MB")
        
    except Exception as e:
        print(f"INT8 Benchmark failed: {e}")

if __name__ == "__main__":
    run_benchmarks()
