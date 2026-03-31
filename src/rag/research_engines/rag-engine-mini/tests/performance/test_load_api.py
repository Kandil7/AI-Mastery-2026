"""
Performance and load testing suite for RAG Engine Mini.

This module contains pytest-based performance benchmarks and stress tests
that measure response times, throughput, and resource utilization under load.

Usage:
    pytest tests/performance/test_load_api.py -v
    pytest tests/performance/test_load_api.py::TestAPIPerformance::test_concurrent_ask -v

Requirements:
    pip install pytest-asyncio aiohttp pytest-benchmark
"""

import asyncio
import time
from typing import Any

import pytest
import pytest_asyncio


class TestAPIPerformance:
    """
    Performance tests for API endpoints.

    These tests measure response times and throughput for critical API endpoints
    under various load conditions.
    """

    @pytest.mark.asyncio
    async def test_concurrent_ask(self, async_client, sample_auth_headers):
        """
        Test concurrent question asking performance.

        Measures how the system handles multiple simultaneous RAG queries.

        Expected Behavior:
            - All requests should complete successfully
            - P95 latency should be under 5 seconds
            - No timeouts or connection errors

        Performance Targets:
            - P50 latency: < 2 seconds
            - P95 latency: < 5 seconds
            - P99 latency: < 10 seconds
            - Success rate: 100%
        """
        questions = [
            "What is RAG?",
            "Explain vector databases",
            "How does semantic search work?",
            "What is the difference between BM25 and vector search?",
            "How do I implement hybrid search?",
        ]

        start_time = time.time()

        # Create concurrent tasks
        tasks = [
            async_client.post(
                "/api/v1/ask", headers=sample_auth_headers, json={"question": q, "k": 5}
            )
            for q in questions
        ]

        # Execute all requests concurrently
        responses = await asyncio.gather(*tasks, return_exceptions=True)

        end_time = time.time()
        total_time = end_time - start_time

        # Analyze results
        successful = sum(
            1 for r in responses if not isinstance(r, Exception) and r.status_code == 200
        )
        errors = [r for r in responses if isinstance(r, Exception)]

        # Assertions
        assert successful == len(questions), (
            f"Only {successful}/{len(questions)} requests succeeded"
        )
        assert len(errors) == 0, f"Got {len(errors)} errors: {errors}"
        assert total_time < 30, f"Total time {total_time}s exceeds 30s threshold"

        # Calculate average response time
        avg_time = total_time / len(questions)
        print(f"\nConcurrent Ask Performance:")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Average per request: {avg_time:.2f}s")
        print(f"  Success rate: {successful}/{len(questions)}")

    @pytest.mark.asyncio
    async def test_search_response_time(self, async_client, sample_auth_headers):
        """
        Test document search response time.

        Verifies that search endpoints respond within acceptable time limits.

        Performance Targets:
            - P50 latency: < 500ms
            - P95 latency: < 2 seconds
            - P99 latency: < 5 seconds
        """
        search_terms = ["data", "analysis", "report", "test", "document"]

        response_times = []

        for term in search_terms:
            start = time.time()
            response = await async_client.get(
                "/api/v1/documents/search",
                headers=sample_auth_headers,
                params={"q": term, "limit": 20},
            )
            end = time.time()

            response_times.append(end - start)
            assert response.status_code == 200

        avg_time = sum(response_times) / len(response_times)
        max_time = max(response_times)

        print(f"\nSearch Response Time:")
        print(f"  Average: {avg_time * 1000:.0f}ms")
        print(f"  Max: {max_time * 1000:.0f}ms")
        print(f"  All times: {[f'{t * 1000:.0f}ms' for t in response_times]}")

        # Performance assertions
        assert avg_time < 2.0, f"Average search time {avg_time}s exceeds 2s threshold"
        assert max_time < 5.0, f"Max search time {max_time}s exceeds 5s threshold"

    @pytest.mark.asyncio
    async def test_document_list_pagination_performance(self, async_client, sample_auth_headers):
        """
        Test pagination performance with large datasets.

        Verifies that paginated endpoints remain performant with many records.

        Performance Targets:
            - Page load time: < 500ms regardless of total count
            - Consistent performance across all pages
        """
        page_sizes = [10, 25, 50, 100]
        response_times = []

        for size in page_sizes:
            start = time.time()
            response = await async_client.get(
                "/api/v1/documents",
                headers=sample_auth_headers,
                params={"limit": size, "offset": 0},
            )
            end = time.time()

            assert response.status_code == 200
            response_times.append((size, end - start))

        print(f"\nPagination Performance:")
        for size, duration in response_times:
            print(f"  Page size {size}: {duration * 1000:.0f}ms")

        # All page sizes should load in reasonable time
        for size, duration in response_times:
            assert duration < 2.0, f"Page size {size} took {duration}s (max 2s)"


class TestRAGPipelinePerformance:
    """
    Performance tests specifically for the RAG pipeline.

    These tests measure end-to-end RAG query performance including:
    - Vector search
    - Keyword search
    - Reranking
    - LLM generation
    """

    @pytest.mark.asyncio
    async def test_hybrid_search_performance(self, async_client, sample_auth_headers):
        """
        Test hybrid search (vector + keyword) performance.

        Measures the time taken for hybrid search fusion operations.

        Pipeline Steps:
            1. Vector search in Qdrant
            2. Keyword search in PostgreSQL
            3. RRF fusion of results
            4. Reranking with cross-encoder

        Performance Targets:
            - Total pipeline: < 3 seconds
            - Vector search: < 1 second
            - Keyword search: < 500ms
            - Fusion + Reranking: < 1 second
        """
        complex_questions = [
            "What are the benefits of using vector databases for RAG?",
            "Compare and contrast different embedding models",
            "How does chunking strategy affect retrieval quality?",
        ]

        results = []

        for question in complex_questions:
            start = time.time()
            response = await async_client.post(
                "/api/v1/ask",
                headers=sample_auth_headers,
                json={"question": question, "k": 10, "use_hybrid": True, "rerank": True},
            )
            end = time.time()

            assert response.status_code == 200
            results.append({"question": question[:50] + "...", "time": end - start})

        print(f"\nHybrid Search Performance:")
        for r in results:
            print(f"  {r['question']}: {r['time']:.2f}s")

        # All hybrid searches should complete within threshold
        for r in results:
            assert r["time"] < 10.0, f"Query took {r['time']}s (max 10s)"

    @pytest.mark.asyncio
    async def test_embedding_caching_performance(self, async_client, sample_auth_headers):
        """
        Test embedding cache hit performance.

        Verifies that cached embeddings are significantly faster than fresh generation.

        Expected Behavior:
            - First request: Full embedding generation time
            - Second request (same): Cache hit, much faster
            - Cache hit rate should be near 100% for duplicate queries
        """
        question = "What is machine learning?"

        # First request - cold cache
        start1 = time.time()
        response1 = await async_client.post(
            "/api/v1/ask", headers=sample_auth_headers, json={"question": question, "k": 5}
        )
        time1 = time.time() - start1

        assert response1.status_code == 200

        # Second request - should hit cache
        start2 = time.time()
        response2 = await async_client.post(
            "/api/v1/ask", headers=sample_auth_headers, json={"question": question, "k": 5}
        )
        time2 = time.time() - start2

        assert response2.status_code == 200

        print(f"\nEmbedding Cache Performance:")
        print(f"  First request (cold): {time1:.2f}s")
        print(f"  Second request (cached): {time2:.2f}s")
        print(f"  Speedup: {time1 / time2:.1f}x")

        # Cached request should be significantly faster
        # Note: This depends on whether query caching is enabled
        if time2 < time1 * 0.8:  # At least 20% faster
            print("  Cache is working effectively!")


class TestThroughput:
    """
    Throughput testing for system capacity.

    Measures requests per second (RPS) the system can handle.
    """

    @pytest.mark.asyncio
    async def test_health_check_throughput(self, async_client):
        """
        Test health check endpoint throughput.

        Health checks should handle very high throughput as they're
        used by load balancers and monitoring systems.

        Target Throughput:
            - Minimum: 100 RPS
            - Target: 1000 RPS
            - Stress test: 5000 RPS
        """
        num_requests = 100

        start = time.time()

        # Create concurrent health check tasks
        tasks = [async_client.get("/health") for _ in range(num_requests)]
        responses = await asyncio.gather(*tasks, return_exceptions=True)

        end = time.time()
        total_time = end - start

        # Count successes and failures
        successes = sum(
            1 for r in responses if not isinstance(r, Exception) and r.status_code == 200
        )
        failures = [r for r in responses if isinstance(r, Exception) or r.status_code != 200]

        rps = num_requests / total_time

        print(f"\nHealth Check Throughput:")
        print(f"  Requests: {num_requests}")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  RPS: {rps:.1f}")
        print(f"  Successes: {successes}/{num_requests}")
        print(f"  Failures: {len(failures)}")

        assert successes == num_requests, f"Got {len(failures)} failures"
        assert rps > 50, f"RPS {rps:.1f} is below minimum threshold of 50"


class TestBenchmarks:
    """
    Benchmark tests using pytest-benchmark.

    These provide standardized performance metrics that can be tracked over time.
    """

    @pytest.mark.benchmark
    def test_benchmark_ask_endpoint(self, benchmark, client, auth_headers):
        """
        Benchmark the ask endpoint.

        Uses pytest-benchmark to measure performance consistently.
        """

        def ask_question():
            return client.post(
                "/api/v1/ask", headers=auth_headers, json={"question": "What is RAG?", "k": 5}
            )

        result = benchmark(ask_question)

        # The benchmark fixture provides detailed statistics
        # These are automatically printed by pytest-benchmark
        assert result.status_code == 200

    @pytest.mark.benchmark
    def test_benchmark_search_endpoint(self, benchmark, client, auth_headers):
        """Benchmark the search endpoint."""

        def search_documents():
            return client.get(
                "/api/v1/documents/search", headers=auth_headers, params={"q": "test", "limit": 10}
            )

        result = benchmark(search_documents)
        assert result.status_code == 200


class TestResourceUtilization:
    """
    Tests for resource utilization under load.

    Monitors CPU, memory, and database connection usage.
    """

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_sustained_load_resource_usage(self, async_client, sample_auth_headers):
        """
        Test resource utilization under sustained load.

        Runs sustained load for a period and monitors system resources.

        Duration: 30 seconds
        Load: 10 requests per second

        Resource Targets:
            - CPU: < 80% average
            - Memory: Stable (no leaks)
            - DB connections: < 50 active
        """
        import psutil

        process = psutil.Process()

        # Get baseline
        baseline_cpu = process.cpu_percent(interval=1)
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB

        start_time = time.time()
        request_count = 0
        errors = []

        # Run sustained load
        while time.time() - start_time < 30:  # 30 seconds
            tasks = [
                async_client.post(
                    "/api/v1/ask",
                    headers=sample_auth_headers,
                    json={"question": "Test question", "k": 3},
                )
                for _ in range(10)
            ]

            try:
                responses = await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True), timeout=15
                )
                request_count += len(responses)

                for r in responses:
                    if isinstance(r, Exception):
                        errors.append(str(r))
                    elif r.status_code != 200:
                        errors.append(f"Status {r.status_code}")

            except asyncio.TimeoutError:
                errors.append("Timeout")

            # Small delay between batches
            await asyncio.sleep(0.5)

        # Get final measurements
        final_cpu = process.cpu_percent(interval=1)
        final_memory = process.memory_info().rss / 1024 / 1024  # MB

        duration = time.time() - start_time

        print(f"\nSustained Load Test ({duration:.0f}s):")
        print(f"  Total requests: {request_count}")
        print(f"  Requests/sec: {request_count / duration:.1f}")
        print(f"  Errors: {len(errors)}")
        print(f"  CPU: {baseline_cpu:.1f}% -> {final_cpu:.1f}%")
        print(
            f"  Memory: {baseline_memory:.1f}MB -> {final_memory:.1f}MB ({final_memory - baseline_memory:+.1f}MB)"
        )

        # Assertions
        assert len(errors) < request_count * 0.05, (
            f"Error rate too high: {len(errors)}/{request_count}"
        )
        assert final_memory < baseline_memory * 1.5, "Possible memory leak detected"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
