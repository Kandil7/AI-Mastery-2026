"""
Smoke Tests for RAG Engine Mini

End-to-end verification that the entire system is operational.
These tests verify all major components work together correctly.

Usage:
    # Run all smoke tests
    python scripts/smoke_test.py

    # Run specific tests
    python scripts/smoke_test.py --test api,auth

    # Verbose output
    python scripts/smoke_test.py -v

    # With custom base URL
    python scripts/smoke_test.py --url https://api.example.com

Exit Codes:
    0 - All tests passed
    1 - One or more tests failed
"""

import argparse
import os
import sys
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import requests


@dataclass
class TestResult:
    """Result of a smoke test."""

    name: str
    passed: bool
    duration_ms: float
    error_message: Optional[str] = None
    details: Optional[Dict] = None


class SmokeTestRunner:
    """
    Runner for smoke tests.

    Verifies that all critical system components are operational:
    - API availability
    - Authentication
    - Database connectivity
    - Vector store
    - LLM providers
    - File storage
    - Background workers
    """

    def __init__(self, base_url: str, api_key: Optional[str] = None, verbose: bool = False):
        """
        Initialize smoke test runner.

        Args:
            base_url: Base URL of the API
            api_key: Optional API key for authenticated tests
            verbose: Whether to print detailed output
        """
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.verbose = verbose
        self.results: List[TestResult] = []

    def log(self, message: str, level: str = "info") -> None:
        """Print message if verbose mode is enabled."""
        if self.verbose:
            prefix = {
                "info": "[INFO]",
                "success": "[PASS]",
                "error": "[FAIL]",
                "warn": "[WARN]",
            }.get(level, "[INFO]")
            print(f"{prefix} {message}")

    def run_test(self, test_func, name: str) -> TestResult:
        """
        Run a single test and record result.

        Args:
            test_func: Function to execute
            name: Name of the test

        Returns:
            TestResult with pass/fail status
        """
        self.log(f"Running: {name}")
        start_time = time.time()

        try:
            test_func()
            duration = (time.time() - start_time) * 1000
            result = TestResult(name=name, passed=True, duration_ms=duration)
            self.log(f"{name}: PASSED ({duration:.0f}ms)", "success")
        except Exception as e:
            duration = (time.time() - start_time) * 1000
            result = TestResult(name=name, passed=False, duration_ms=duration, error_message=str(e))
            self.log(f"{name}: FAILED - {e}", "error")

        self.results.append(result)
        return result

    def test_health_endpoint(self) -> None:
        """Test basic health check endpoint."""
        response = requests.get(f"{self.base_url}/health", timeout=10)

        if response.status_code != 200:
            raise Exception(f"Health check failed: HTTP {response.status_code}")

        data = response.json()
        if data.get("status") != "healthy":
            raise Exception(f"Unhealthy status: {data.get('status')}")

    def test_ready_endpoint(self) -> None:
        """Test readiness probe."""
        response = requests.get(f"{self.base_url}/health/ready", timeout=10)

        if response.status_code != 200:
            raise Exception(f"Readiness check failed: HTTP {response.status_code}")

    def test_api_documentation(self) -> None:
        """Test that API documentation is accessible."""
        response = requests.get(f"{self.base_url}/docs", timeout=10)

        if response.status_code != 200:
            raise Exception(f"API docs not accessible: HTTP {response.status_code}")

    def test_authentication_required(self) -> None:
        """Test that protected endpoints require authentication."""
        response = requests.get(f"{self.base_url}/api/v1/documents", timeout=10)

        if response.status_code not in [401, 403]:
            raise Exception(f"Expected 401/403, got {response.status_code}")

    def test_authenticated_request(self) -> None:
        """Test authenticated request with valid token."""
        if not self.api_key:
            raise Exception("No API key provided")

        headers = {"Authorization": f"Bearer {self.api_key}"}
        response = requests.get(f"{self.base_url}/api/v1/documents", headers=headers, timeout=10)

        if response.status_code not in [200, 404]:  # 404 is OK if no documents
            raise Exception(f"Authenticated request failed: HTTP {response.status_code}")

    def test_document_list(self) -> None:
        """Test document listing endpoint."""
        if not self.api_key:
            raise Exception("No API key provided")

        headers = {"Authorization": f"Bearer {self.api_key}"}
        response = requests.get(f"{self.base_url}/api/v1/documents", headers=headers, timeout=10)

        if response.status_code != 200:
            raise Exception(f"Document list failed: HTTP {response.status_code}")

        data = response.json()
        if not isinstance(data, (list, dict)):
            raise Exception("Invalid response format")

    def test_search_endpoint(self) -> None:
        """Test document search endpoint."""
        if not self.api_key:
            raise Exception("No API key provided")

        headers = {"Authorization": f"Bearer {self.api_key}"}
        response = requests.get(
            f"{self.base_url}/api/v1/documents/search",
            headers=headers,
            params={"q": "test"},
            timeout=10,
        )

        if response.status_code != 200:
            raise Exception(f"Search failed: HTTP {response.status_code}")

    def test_ask_endpoint(self) -> None:
        """Test RAG question endpoint."""
        if not self.api_key:
            raise Exception("No API key provided")

        headers = {"Authorization": f"Bearer {self.api_key}"}
        response = requests.post(
            f"{self.base_url}/api/v1/ask",
            headers=headers,
            json={"question": "What is this?", "k": 3},
            timeout=30,
        )

        if response.status_code != 200:
            raise Exception(f"Ask endpoint failed: HTTP {response.status_code}")

        data = response.json()
        if "answer" not in data:
            raise Exception("Response missing 'answer' field")

    def test_query_history(self) -> None:
        """Test query history endpoint."""
        if not self.api_key:
            raise Exception("No API key provided")

        headers = {"Authorization": f"Bearer {self.api_key}"}
        response = requests.get(
            f"{self.base_url}/api/v1/queries/history", headers=headers, timeout=10
        )

        if response.status_code != 200:
            raise Exception(f"Query history failed: HTTP {response.status_code}")

    def test_rate_limiting(self) -> None:
        """Test that rate limiting is working."""
        # Make rapid requests
        responses = []
        for _ in range(10):
            response = requests.get(f"{self.base_url}/health", timeout=5)
            responses.append(response.status_code)

        # All should succeed or some should be rate limited
        if 429 in responses:
            self.log("Rate limiting is active", "success")

    def run_all_tests(self, test_filter: Optional[List[str]] = None) -> Tuple[int, int]:
        """
        Run all smoke tests.

        Args:
            test_filter: Optional list of test names to run

        Returns:
            Tuple of (passed_count, total_count)
        """
        tests = [
            ("health", self.test_health_endpoint),
            ("ready", self.test_ready_endpoint),
            ("api_docs", self.test_api_documentation),
            ("auth_required", self.test_authentication_required),
            ("auth_valid", self.test_authenticated_request),
            ("documents", self.test_document_list),
            ("search", self.test_search_endpoint),
            ("ask", self.test_ask_endpoint),
            ("history", self.test_query_history),
            ("rate_limit", self.test_rate_limiting),
        ]

        print(f"\n{'=' * 60}")
        print(f"RAG Engine Smoke Tests")
        print(f"Base URL: {self.base_url}")
        print(f"{'=' * 60}\n")

        for name, test_func in tests:
            if test_filter and name not in test_filter:
                continue

            self.run_test(test_func, name)

        return self.print_summary()

    def print_summary(self) -> Tuple[int, int]:
        """
        Print test summary.

        Returns:
            Tuple of (passed_count, total_count)
        """
        print(f"\n{'=' * 60}")
        print("Test Summary")
        print(f"{'=' * 60}\n")

        passed = sum(1 for r in self.results if r.passed)
        failed = sum(1 for r in self.results if not r.passed)
        total = len(self.results)

        # Print passed tests
        for result in self.results:
            if result.passed:
                print(f"✓ {result.name:20} ({result.duration_ms:.0f}ms)")

        # Print failed tests
        if failed > 0:
            print()
            for result in self.results:
                if not result.passed:
                    print(f"✗ {result.name:20} - {result.error_message}")

        print(f"\n{'=' * 60}")
        print(f"Total: {total} | Passed: {passed} | Failed: {failed}")
        print(f"{'=' * 60}\n")

        return passed, total


def main():
    """Main entry point for smoke tests."""
    parser = argparse.ArgumentParser(
        description="Run smoke tests on RAG Engine deployment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all tests
  python scripts/smoke_test.py
  
  # Run with API key
  python scripts/smoke_test.py --api-key sk_xxx
  
  # Run specific tests
  python scripts/smoke_test.py --test health,auth,documents
  
  # Custom deployment
  python scripts/smoke_test.py --url https://api.example.com --api-key sk_xxx
        """,
    )

    parser.add_argument(
        "--url",
        default=os.getenv("RAG_API_URL", "http://localhost:8000"),
        help="Base URL of the API (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--api-key", default=os.getenv("RAG_API_KEY"), help="API key for authenticated tests"
    )
    parser.add_argument("--test", help="Comma-separated list of tests to run (default: all)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")

    args = parser.parse_args()

    # Parse test filter
    test_filter = None
    if args.test:
        test_filter = [t.strip() for t in args.test.split(",")]

    # Run tests
    runner = SmokeTestRunner(base_url=args.url, api_key=args.api_key, verbose=args.verbose)

    passed, total = runner.run_all_tests(test_filter)

    # Exit with appropriate code
    if passed == total:
        print("✓ All smoke tests passed!")
        sys.exit(0)
    else:
        print(f"✗ {total - passed} of {total} tests failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
