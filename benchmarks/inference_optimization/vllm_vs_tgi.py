"""
Inference Optimization Benchmarks - vLLM vs TGI
================================================
Compare vLLM and Text Generation Inference performance.

Metrics:
- Time to First Token (TTFT)
- Tokens per Second (TPS)
- Throughput (requests/sec)
- GPU Memory Usage

Author: AI-Mastery-2026
"""

import time
import json
import statistics
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

logger = logging.getLogger(__name__)


# ============================================================
# BENCHMARK DATA STRUCTURES
# ============================================================

@dataclass
class BenchmarkConfig:
    """Configuration for benchmark run."""
    model_name: str = "llama-2-7b"
    num_requests: int = 100
    concurrency: int = 10
    prompt_lengths: List[int] = field(default_factory=lambda: [128, 512, 2048])
    max_tokens: int = 256
    warmup_requests: int = 5


@dataclass
class RequestMetrics:
    """Metrics for a single request."""
    request_id: int
    prompt_length: int
    output_length: int
    time_to_first_token_ms: float
    total_time_ms: float
    tokens_per_second: float
    success: bool
    error: Optional[str] = None


@dataclass
class BenchmarkResult:
    """Aggregated benchmark results."""
    framework: str
    model: str
    config: BenchmarkConfig
    
    # Latency
    ttft_p50_ms: float = 0.0
    ttft_p90_ms: float = 0.0
    ttft_p99_ms: float = 0.0
    
    # Throughput
    avg_tokens_per_second: float = 0.0
    total_requests_per_second: float = 0.0
    
    # Success
    success_rate: float = 0.0
    total_requests: int = 0
    
    # Resource
    peak_gpu_memory_gb: float = 0.0
    
    raw_metrics: List[RequestMetrics] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d['config'] = asdict(self.config)
        d['raw_metrics'] = [asdict(m) for m in self.raw_metrics]
        return d


# ============================================================
# MOCK CLIENTS (Replace with real implementations)
# ============================================================

class VLLMClient:
    """
    vLLM client wrapper.
    
    Replace with actual vLLM API calls in production:
    ```python
    from openai import OpenAI
    client = OpenAI(base_url="http://localhost:8000/v1")
    ```
    """
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.model = "llama-2-7b"
    
    def generate(
        self, 
        prompt: str, 
        max_tokens: int = 256,
        stream: bool = True
    ) -> Dict[str, Any]:
        """
        Generate text with vLLM.
        
        Simulated response for demo. Replace with:
        ```python
        response = client.completions.create(
            model=self.model,
            prompt=prompt,
            max_tokens=max_tokens,
            stream=stream
        )
        ```
        """
        # Simulate generation time based on prompt/output length
        prompt_tokens = len(prompt.split())
        
        # Simulated performance characteristics of vLLM
        prefill_time_ms = prompt_tokens * 0.5  # Fast prefill
        decode_time_ms = max_tokens * 10  # ~100 TPS
        
        time.sleep((prefill_time_ms + decode_time_ms) / 1000)
        
        return {
            "choices": [{"text": "Generated response " * (max_tokens // 4)}],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": max_tokens,
                "total_tokens": prompt_tokens + max_tokens
            },
            "metrics": {
                "time_to_first_token_ms": prefill_time_ms,
                "total_time_ms": prefill_time_ms + decode_time_ms
            }
        }


class TGIClient:
    """
    Text Generation Inference client wrapper.
    
    Replace with actual TGI API calls in production:
    ```python
    from huggingface_hub import InferenceClient
    client = InferenceClient(model="http://localhost:8080")
    ```
    """
    
    def __init__(self, base_url: str = "http://localhost:8080"):
        self.base_url = base_url
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        stream: bool = True
    ) -> Dict[str, Any]:
        """
        Generate text with TGI.
        
        Simulated response for demo.
        """
        prompt_tokens = len(prompt.split())
        
        # TGI characteristics (slightly different performance profile)
        prefill_time_ms = prompt_tokens * 0.6
        decode_time_ms = max_tokens * 12  # ~85 TPS
        
        time.sleep((prefill_time_ms + decode_time_ms) / 1000)
        
        return {
            "generated_text": "Generated response " * (max_tokens // 4),
            "details": {
                "prefill_time_ms": prefill_time_ms,
                "decode_time_ms": decode_time_ms,
                "tokens_generated": max_tokens
            }
        }


# ============================================================
# BENCHMARK RUNNER
# ============================================================

class InferenceBenchmark:
    """
    Run inference benchmarks comparing different frameworks.
    
    Example:
        >>> benchmark = InferenceBenchmark()
        >>> results = benchmark.run_comparison()
        >>> benchmark.print_report(results)
    """
    
    def __init__(self, config: Optional[BenchmarkConfig] = None):
        self.config = config or BenchmarkConfig()
        self.vllm_client = VLLMClient()
        self.tgi_client = TGIClient()
    
    def _generate_prompt(self, length: int) -> str:
        """Generate a prompt of approximately the given length."""
        base = "This is a test prompt. Please generate a helpful response. "
        repetitions = max(1, length // len(base.split()))
        return (base * repetitions)[:length * 5]  # Approximate word to char
    
    def _run_single_request(
        self,
        client: Any,
        request_id: int,
        prompt_length: int
    ) -> RequestMetrics:
        """Run a single inference request and collect metrics."""
        prompt = self._generate_prompt(prompt_length)
        
        start_time = time.time()
        
        try:
            response = client.generate(
                prompt=prompt,
                max_tokens=self.config.max_tokens
            )
            
            end_time = time.time()
            total_time_ms = (end_time - start_time) * 1000
            
            # Extract metrics (format differs by framework)
            if hasattr(client, 'vllm'):
                ttft = response.get('metrics', {}).get('time_to_first_token_ms', 0)
            else:
                ttft = response.get('details', {}).get('prefill_time_ms', 0)
            
            output_tokens = self.config.max_tokens
            tps = output_tokens / (total_time_ms / 1000) if total_time_ms > 0 else 0
            
            return RequestMetrics(
                request_id=request_id,
                prompt_length=prompt_length,
                output_length=output_tokens,
                time_to_first_token_ms=ttft,
                total_time_ms=total_time_ms,
                tokens_per_second=tps,
                success=True
            )
            
        except Exception as e:
            end_time = time.time()
            return RequestMetrics(
                request_id=request_id,
                prompt_length=prompt_length,
                output_length=0,
                time_to_first_token_ms=0,
                total_time_ms=(end_time - start_time) * 1000,
                tokens_per_second=0,
                success=False,
                error=str(e)
            )
    
    def run_benchmark(
        self,
        client: Any,
        framework_name: str
    ) -> BenchmarkResult:
        """Run full benchmark suite for a framework."""
        
        print(f"Running {framework_name} benchmark...")
        
        # Warmup
        print("  Warming up...")
        for i in range(self.config.warmup_requests):
            self._run_single_request(client, -1, 128)
        
        # Run benchmarks
        all_metrics: List[RequestMetrics] = []
        
        for prompt_length in self.config.prompt_lengths:
            print(f"  Testing prompt length: {prompt_length}")
            
            with ThreadPoolExecutor(max_workers=self.config.concurrency) as executor:
                futures = [
                    executor.submit(
                        self._run_single_request,
                        client,
                        i,
                        prompt_length
                    )
                    for i in range(self.config.num_requests // len(self.config.prompt_lengths))
                ]
                
                for future in as_completed(futures):
                    all_metrics.append(future.result())
        
        # Aggregate results
        successful = [m for m in all_metrics if m.success]
        
        ttft_values = [m.time_to_first_token_ms for m in successful]
        tps_values = [m.tokens_per_second for m in successful]
        
        total_time = sum(m.total_time_ms for m in successful) / 1000
        
        result = BenchmarkResult(
            framework=framework_name,
            model=self.config.model_name,
            config=self.config,
            ttft_p50_ms=statistics.median(ttft_values) if ttft_values else 0,
            ttft_p90_ms=sorted(ttft_values)[int(len(ttft_values) * 0.9)] if ttft_values else 0,
            ttft_p99_ms=sorted(ttft_values)[int(len(ttft_values) * 0.99)] if ttft_values else 0,
            avg_tokens_per_second=statistics.mean(tps_values) if tps_values else 0,
            total_requests_per_second=len(successful) / total_time if total_time > 0 else 0,
            success_rate=len(successful) / len(all_metrics) if all_metrics else 0,
            total_requests=len(all_metrics),
            raw_metrics=all_metrics
        )
        
        return result
    
    def run_comparison(self) -> Dict[str, BenchmarkResult]:
        """Run benchmarks for all frameworks."""
        results = {}
        
        # vLLM
        results['vllm'] = self.run_benchmark(self.vllm_client, "vLLM")
        
        # TGI
        results['tgi'] = self.run_benchmark(self.tgi_client, "TGI")
        
        return results
    
    def print_report(self, results: Dict[str, BenchmarkResult]):
        """Print benchmark comparison report."""
        print("\n" + "=" * 60)
        print("INFERENCE BENCHMARK RESULTS")
        print("=" * 60)
        
        print(f"\nConfig: {self.config.num_requests} requests, "
              f"concurrency={self.config.concurrency}, "
              f"max_tokens={self.config.max_tokens}")
        
        print("\n" + "-" * 60)
        print(f"{'Metric':<30} {'vLLM':>12} {'TGI':>12}")
        print("-" * 60)
        
        vllm = results.get('vllm')
        tgi = results.get('tgi')
        
        if vllm and tgi:
            metrics = [
                ("TTFT P50 (ms)", vllm.ttft_p50_ms, tgi.ttft_p50_ms),
                ("TTFT P90 (ms)", vllm.ttft_p90_ms, tgi.ttft_p90_ms),
                ("TTFT P99 (ms)", vllm.ttft_p99_ms, tgi.ttft_p99_ms),
                ("Avg Tokens/sec", vllm.avg_tokens_per_second, tgi.avg_tokens_per_second),
                ("Requests/sec", vllm.total_requests_per_second, tgi.total_requests_per_second),
                ("Success Rate", vllm.success_rate * 100, tgi.success_rate * 100),
            ]
            
            for name, v_val, t_val in metrics:
                winner = "←" if v_val < t_val else "→" if t_val < v_val else "="
                print(f"{name:<30} {v_val:>10.2f} {winner} {t_val:>10.2f}")
        
        print("=" * 60)
    
    def save_results(self, results: Dict[str, BenchmarkResult], path: str):
        """Save results to JSON file."""
        data = {k: v.to_dict() for k, v in results.items()}
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Results saved to {path}")


# ============================================================
# MAIN
# ============================================================

def main():
    """Run inference benchmarks."""
    import argparse
    
    parser = argparse.ArgumentParser(description="LLM Inference Benchmarks")
    parser.add_argument("--requests", type=int, default=100, help="Number of requests")
    parser.add_argument("--concurrency", type=int, default=10, help="Concurrent requests")
    parser.add_argument("--max-tokens", type=int, default=256, help="Max tokens per request")
    parser.add_argument("--output", default="benchmark_results.json", help="Output file")
    args = parser.parse_args()
    
    config = BenchmarkConfig(
        num_requests=args.requests,
        concurrency=args.concurrency,
        max_tokens=args.max_tokens
    )
    
    benchmark = InferenceBenchmark(config)
    results = benchmark.run_comparison()
    benchmark.print_report(results)
    benchmark.save_results(results, args.output)


if __name__ == "__main__":
    main()
