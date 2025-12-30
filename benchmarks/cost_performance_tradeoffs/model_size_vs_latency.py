"""
Cost-Performance Tradeoff Benchmarks
=====================================
Analyze model size vs latency vs cost tradeoffs.

Compare different model configurations:
- Model sizes (7B, 13B, 70B)
- Quantization levels (FP16, INT8, INT4)
- Hardware options (A10G, A100-40G, A100-80G)

Author: AI-Mastery-2026
"""

import numpy as np
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass, field
import json


# ============================================================
# HARDWARE SPECIFICATIONS
# ============================================================

@dataclass
class GPUSpec:
    """GPU hardware specifications."""
    name: str
    memory_gb: float
    tflops_fp16: float
    cost_per_hour: float
    memory_bandwidth_gbps: float


GPU_SPECS = {
    "A10G": GPUSpec("A10G", 24, 31.2, 1.00, 600),
    "A100-40G": GPUSpec("A100-40G", 40, 312, 3.50, 1555),
    "A100-80G": GPUSpec("A100-80G", 80, 312, 4.50, 2039),
    "H100": GPUSpec("H100", 80, 989, 8.00, 3350),
}


# ============================================================
# MODEL SPECIFICATIONS
# ============================================================

@dataclass
class ModelSpec:
    """LLM model specifications."""
    name: str
    params_billions: float
    hidden_dim: int
    num_layers: int
    num_heads: int
    
    def memory_fp16_gb(self) -> float:
        """Estimate FP16 model memory in GB."""
        return self.params_billions * 2  # 2 bytes per param
    
    def memory_int8_gb(self) -> float:
        """Estimate INT8 model memory in GB."""
        return self.params_billions * 1  # 1 byte per param
    
    def memory_int4_gb(self) -> float:
        """Estimate INT4 model memory in GB."""
        return self.params_billions * 0.5  # 0.5 bytes per param


MODEL_SPECS = {
    "llama-7b": ModelSpec("llama-7b", 7, 4096, 32, 32),
    "llama-13b": ModelSpec("llama-13b", 13, 5120, 40, 40),
    "llama-70b": ModelSpec("llama-70b", 70, 8192, 80, 64),
}


# ============================================================
# PERFORMANCE ESTIMATION
# ============================================================

@dataclass
class PerformanceEstimate:
    """Estimated performance for a configuration."""
    model: str
    gpu: str
    quantization: str
    num_gpus: int
    
    # Memory
    model_memory_gb: float
    kv_cache_memory_gb: float
    total_memory_gb: float
    memory_utilization: float
    
    # Latency
    prefill_latency_ms: float  # Time to first token
    decode_latency_ms: float   # Per token
    
    # Throughput
    tokens_per_second: float
    requests_per_second: float
    
    # Cost
    cost_per_hour: float
    cost_per_million_tokens: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "gpu": self.gpu,
            "quantization": self.quantization,
            "num_gpus": self.num_gpus,
            "memory_utilization": f"{self.memory_utilization:.1%}",
            "prefill_latency_ms": round(self.prefill_latency_ms, 1),
            "decode_latency_ms": round(self.decode_latency_ms, 2),
            "tokens_per_second": round(self.tokens_per_second, 1),
            "cost_per_million_tokens": f"${self.cost_per_million_tokens:.4f}",
        }


class PerformanceEstimator:
    """
    Estimate LLM inference performance.
    
    Based on roofline model analysis:
    - Compute-bound: Prefill/large batch
    - Memory-bound: Decode/small batch
    """
    
    def __init__(self, context_length: int = 4096, output_length: int = 256):
        self.context_length = context_length
        self.output_length = output_length
    
    def estimate_kv_cache_size(
        self, 
        model: ModelSpec, 
        batch_size: int = 1
    ) -> float:
        """
        Estimate KV cache size in GB.
        
        KV cache = 2 * num_layers * hidden_dim * seq_len * batch * dtype_size
        """
        bytes_per_element = 2  # FP16
        kv_per_layer = 2 * model.hidden_dim * self.context_length * batch_size
        total_bytes = model.num_layers * kv_per_layer * bytes_per_element
        return total_bytes / (1024 ** 3)
    
    def estimate_prefill_latency(
        self,
        model: ModelSpec,
        gpu: GPUSpec,
        num_gpus: int,
        quantization: str
    ) -> float:
        """
        Estimate prefill latency (time to first token).
        
        Prefill is compute-bound: dominated by matrix multiplications.
        """
        # FLOPs for prefill ≈ 2 * params * context_length
        flops = 2 * model.params_billions * 1e9 * self.context_length
        
        # Effective compute with quantization
        compute_factor = {
            "fp16": 1.0,
            "int8": 1.3,  # INT8 tensor cores
            "int4": 1.5,  # Further speedup
        }.get(quantization, 1.0)
        
        # Total compute capacity
        total_tflops = gpu.tflops_fp16 * num_gpus * compute_factor
        
        # Time in ms
        latency_sec = flops / (total_tflops * 1e12)
        return latency_sec * 1000
    
    def estimate_decode_latency(
        self,
        model: ModelSpec,
        gpu: GPUSpec,
        num_gpus: int,
        quantization: str
    ) -> float:
        """
        Estimate per-token decode latency.
        
        Decode is memory-bound: need to load entire model weights per token.
        """
        # Model size based on quantization
        if quantization == "fp16":
            model_size_gb = model.memory_fp16_gb()
        elif quantization == "int8":
            model_size_gb = model.memory_int8_gb()
        else:
            model_size_gb = model.memory_int4_gb()
        
        # Time to load model from memory
        bandwidth_gbps = gpu.memory_bandwidth_gbps * num_gpus
        load_time_sec = model_size_gb / bandwidth_gbps
        
        return load_time_sec * 1000
    
    def estimate(
        self,
        model_name: str,
        gpu_name: str,
        quantization: str = "fp16",
        batch_size: int = 1
    ) -> PerformanceEstimate:
        """Estimate full performance for a configuration."""
        
        model = MODEL_SPECS[model_name]
        gpu = GPU_SPECS[gpu_name]
        
        # Determine model memory
        if quantization == "fp16":
            model_memory = model.memory_fp16_gb()
        elif quantization == "int8":
            model_memory = model.memory_int8_gb()
        else:
            model_memory = model.memory_int4_gb()
        
        # Calculate minimum GPUs needed
        kv_cache_size = self.estimate_kv_cache_size(model, batch_size)
        total_memory_needed = model_memory + kv_cache_size
        
        num_gpus = max(1, int(np.ceil(total_memory_needed / (gpu.memory_gb * 0.9))))
        
        # Estimate latencies
        prefill_latency = self.estimate_prefill_latency(
            model, gpu, num_gpus, quantization
        )
        decode_latency = self.estimate_decode_latency(
            model, gpu, num_gpus, quantization
        )
        
        # Throughput
        total_latency_sec = (prefill_latency + self.output_length * decode_latency) / 1000
        tokens_per_second = self.output_length / total_latency_sec
        requests_per_second = 1 / total_latency_sec * batch_size
        
        # Cost
        cost_per_hour = gpu.cost_per_hour * num_gpus
        tokens_per_hour = tokens_per_second * 3600
        cost_per_million = (cost_per_hour / tokens_per_hour) * 1e6
        
        return PerformanceEstimate(
            model=model_name,
            gpu=gpu_name,
            quantization=quantization,
            num_gpus=num_gpus,
            model_memory_gb=model_memory,
            kv_cache_memory_gb=kv_cache_size,
            total_memory_gb=total_memory_needed,
            memory_utilization=total_memory_needed / (gpu.memory_gb * num_gpus),
            prefill_latency_ms=prefill_latency,
            decode_latency_ms=decode_latency,
            tokens_per_second=tokens_per_second,
            requests_per_second=requests_per_second,
            cost_per_hour=cost_per_hour,
            cost_per_million_tokens=cost_per_million
        )


# ============================================================
# TRADEOFF ANALYSIS
# ============================================================

def analyze_tradeoffs() -> List[PerformanceEstimate]:
    """Run tradeoff analysis across configurations."""
    
    estimator = PerformanceEstimator()
    results = []
    
    configurations = [
        # Small model - various GPUs
        ("llama-7b", "A10G", "fp16"),
        ("llama-7b", "A10G", "int8"),
        ("llama-7b", "A10G", "int4"),
        ("llama-7b", "A100-40G", "fp16"),
        
        # Medium model
        ("llama-13b", "A10G", "int8"),
        ("llama-13b", "A100-40G", "fp16"),
        ("llama-13b", "A100-40G", "int8"),
        ("llama-13b", "A100-80G", "fp16"),
        
        # Large model
        ("llama-70b", "A100-40G", "int4"),
        ("llama-70b", "A100-80G", "fp16"),
        ("llama-70b", "A100-80G", "int8"),
        ("llama-70b", "H100", "fp16"),
        ("llama-70b", "H100", "int8"),
    ]
    
    for model, gpu, quant in configurations:
        try:
            estimate = estimator.estimate(model, gpu, quant)
            results.append(estimate)
        except Exception as e:
            print(f"Error estimating {model}/{gpu}/{quant}: {e}")
    
    return results


def print_tradeoff_table(results: List[PerformanceEstimate]):
    """Print formatted tradeoff table."""
    
    print("\n" + "=" * 100)
    print("MODEL SIZE vs LATENCY vs COST TRADEOFFS")
    print("=" * 100)
    print(f"\n{'Model':<12} {'GPU':<12} {'Quant':<6} {'GPUs':<5} {'Mem%':<6} "
          f"{'TTFT(ms)':<10} {'TPS':<8} {'$/MTok':<10}")
    print("-" * 100)
    
    for r in results:
        print(f"{r.model:<12} {r.gpu:<12} {r.quantization:<6} {r.num_gpus:<5} "
              f"{r.memory_utilization:>5.0%} {r.prefill_latency_ms:>9.1f} "
              f"{r.tokens_per_second:>7.1f} ${r.cost_per_million_tokens:>8.4f}")
    
    print("=" * 100)
    
    # Find best configurations
    print("\n📊 KEY INSIGHTS:")
    
    # Best cost per token
    cheapest = min(results, key=lambda x: x.cost_per_million_tokens)
    print(f"  💰 Cheapest: {cheapest.model}/{cheapest.gpu}/{cheapest.quantization} "
          f"- ${cheapest.cost_per_million_tokens:.4f}/MTok")
    
    # Fastest
    fastest = min(results, key=lambda x: x.prefill_latency_ms)
    print(f"  ⚡ Lowest Latency: {fastest.model}/{fastest.gpu}/{fastest.quantization} "
          f"- {fastest.prefill_latency_ms:.1f}ms TTFT")
    
    # Best throughput
    highest_tps = max(results, key=lambda x: x.tokens_per_second)
    print(f"  🚀 Highest Throughput: {highest_tps.model}/{highest_tps.gpu}/{highest_tps.quantization} "
          f"- {highest_tps.tokens_per_second:.1f} TPS")


def save_results(results: List[PerformanceEstimate], path: str):
    """Save results to JSON."""
    data = [r.to_dict() for r in results]
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"\nResults saved to {path}")


# ============================================================
# MAIN
# ============================================================

def main():
    print("Running Model Size vs Latency Analysis...")
    
    results = analyze_tradeoffs()
    print_tradeoff_table(results)
    save_results(results, "tradeoff_results.json")
    
    print("\n✅ Analysis complete!")


if __name__ == "__main__":
    main()
