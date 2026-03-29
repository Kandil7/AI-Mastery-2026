#!/usr/bin/env python3
"""
AI-Mastery-2026 Architecture Verification Script
=================================================

Comprehensive verification of the AI-Mastery-2026 architecture.

Usage:
------
    python scripts/verify_architecture.py

Or via make:
    make verify-architecture

Verification Categories:
------------------------
1. Module Imports - Verify all modules import correctly
2. Code Quality - Check formatting, linting, type hints
3. Documentation - Verify README files exist
4. Tests - Check test coverage and fixtures
5. Production Readiness - Verify API, auth, monitoring
6. Docker - Check Docker configurations
"""

import sys
import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import json

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ANSI color codes
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    RESET = '\033[0m'
    BOLD = '\033[1m'


@dataclass
class CheckResult:
    """Result of a single verification check."""
    name: str
    passed: bool
    message: str
    severity: str = "info"  # critical, high, medium, low, info


class ArchitectureVerifier:
    """Comprehensive architecture verification."""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.results: List[CheckResult] = []
        self.start_time = datetime.now()

    def log(self, message: str, color: str = Colors.RESET):
        """Print colored message."""
        print(f"{color}{message}{Colors.RESET}")

    def add_result(self, result: CheckResult):
        """Add a verification result."""
        self.results.append(result)

        status = f"{Colors.GREEN}✓ PASS{Colors.RESET}" if result.passed else f"{Colors.RED}✗ FAIL{Colors.RESET}"
        severity_color = {
            "critical": Colors.RED,
            "high": Colors.RED,
            "medium": Colors.YELLOW,
            "low": Colors.BLUE,
            "info": Colors.CYAN,
        }.get(result.severity, Colors.RESET)

        if not result.passed:
            self.log(f"  {status} [{result.severity.upper()}] {result.name}: {result.message}", severity_color)

    # ============================================================
    # Module Import Verification
    # ============================================================

    def verify_module_imports(self):
        """Verify all core modules import correctly."""
        self.log("\n" + "=" * 60, Colors.BOLD)
        self.log("1. MODULE IMPORT VERIFICATION", Colors.BOLD)
        print("=" * 60)

        # Core modules
        core_modules = [
            "src.core",
            "src.ml",
            "src.llm",
            "src.rag",
            "src.embeddings",
            "src.vector_stores",
            "src.agents",
            "src.evaluation",
            "src.production",
            "src.orchestration",
            "src.safety",
            "src.utils",
        ]

        for module_name in core_modules:
            try:
                __import__(module_name)
                self.add_result(CheckResult(
                    name=f"Import: {module_name}",
                    passed=True,
                    message="Module imports successfully",
                    severity="info"
                ))
            except ImportError as e:
                # Check if it's an optional dependency issue (torch, etc.)
                error_str = str(e).lower()
                is_optional = any(dep in error_str for dep in ['torch', 'tensorflow', 'pytorch', 'dll'])

                self.add_result(CheckResult(
                    name=f"Import: {module_name}",
                    passed=False,
                    message=f"Optional dependency: {e}" if is_optional else str(e),
                    severity="low" if is_optional else "high"
                ))
            except OSError as e:
                # DLL loading issues are environment-specific, not code issues
                error_str = str(e).lower()
                is_env_issue = any(issue in error_str for issue in ['dll', 'shared object', 'library'])

                self.add_result(CheckResult(
                    name=f"Import: {module_name}",
                    passed=False,
                    message=f"Environment issue (not code): {e}" if is_env_issue else str(e),
                    severity="low" if is_env_issue else "medium"
                ))
            except Exception as e:
                # Other errors
                self.add_result(CheckResult(
                    name=f"Import: {module_name}",
                    passed=False,
                    message=f"Import error: {type(e).__name__}: {e}",
                    severity="medium"
                ))

    # ============================================================
    # Code Quality Verification
    # ============================================================

    def verify_code_quality(self):
        """Verify code quality standards."""
        self.log("\n" + "=" * 60, Colors.BOLD)
        self.log("2. CODE QUALITY VERIFICATION", Colors.BOLD)
        print("=" * 60)

        # Check for __init__.py files
        src_dirs = [
            "src",
            "src/core",
            "src/ml",
            "src/llm",
            "src/rag",
            "src/production",
            "src/utils",
            "src/agents",
            "src/embeddings",
            "src/retrieval",
            "src/reranking",
        ]

        for dir_path in src_dirs:
            init_file = self.project_root / dir_path / "__init__.py"
            exists = init_file.exists()
            self.add_result(CheckResult(
                name=f"__init__.py: {dir_path}",
                passed=exists,
                message="Exists" if exists else "Missing",
                severity="high" if not exists else "info"
            ))

        # Check for docstrings in key modules
        key_files = [
            "src/utils/errors.py",
            "src/utils/logging.py",
            "src/utils/config.py",
            "src/utils/types.py",
            "src/production/api.py",
            "src/production/auth.py",
            "src/production/caching.py",
        ]

        for file_path in key_files:
            full_path = self.project_root / file_path
            if full_path.exists():
                content = full_path.read_text(encoding='utf-8')
                has_module_docstring = '"""' in content[:500]
                self.add_result(CheckResult(
                    name=f"Docstring: {file_path}",
                    passed=has_module_docstring,
                    message="Has module docstring" if has_module_docstring else "Missing module docstring",
                    severity="medium" if not has_module_docstring else "info"
                ))

    # ============================================================
    # Documentation Verification
    # ============================================================

    def verify_documentation(self):
        """Verify documentation completeness."""
        self.log("\n" + "=" * 60, Colors.BOLD)
        self.log("3. DOCUMENTATION VERIFICATION", Colors.BOLD)
        print("=" * 60)

        # Required documentation files
        required_docs = [
            "README.md",
            "docs/README.md",
            "docs/INDEX.md",
            "docs/01_learning_roadmap/README.md",
            "docs/04_production/README.md",
        ]

        for doc_path in required_docs:
            full_path = self.project_root / doc_path
            exists = full_path.exists()
            self.add_result(CheckResult(
                name=f"Documentation: {doc_path}",
                passed=exists,
                message="Exists" if exists else "Missing",
                severity="high" if not exists else "info"
            ))

        # Check for module README files
        module_dirs = [
            "src/core",
            "src/ml",
            "src/llm",
            "src/rag",
            "src/production",
        ]

        for dir_path in module_dirs:
            readme_path = self.project_root / dir_path / "README.md"
            exists = readme_path.exists()
            self.add_result(CheckResult(
                name=f"Module README: {dir_path}",
                passed=exists,
                message="Exists" if exists else "Missing",
                severity="low" if not exists else "info"
            ))

    # ============================================================
    # Test Verification
    # ============================================================

    def verify_tests(self):
        """Verify test infrastructure."""
        self.log("\n" + "=" * 60, Colors.BOLD)
        self.log("4. TEST INFRASTRUCTURE VERIFICATION", Colors.BOLD)
        print("=" * 60)

        # Check for conftest.py
        conftest_path = self.project_root / "tests" / "conftest.py"
        exists = conftest_path.exists()
        self.add_result(CheckResult(
            name="Test fixtures: conftest.py",
            passed=exists,
            message="Exists" if exists else "Missing",
            severity="high" if not exists else "info"
        ))

        # Count test files
        test_dir = self.project_root / "tests"
        test_files = list(test_dir.glob("test_*.py"))
        self.add_result(CheckResult(
            name="Test files count",
            passed=len(test_files) >= 10,
            message=f"Found {len(test_files)} test files",
            severity="info"
        ))

        # Check for fixture categories
        if exists:
            content = conftest_path.read_text(encoding='utf-8')
            fixture_categories = [
                "Document Fixtures",
                "Chunking Fixtures",
                "Embedding Fixtures",
                "Vector Store Fixtures",
                "RAG Pipeline Fixtures",
                "Cache Fixtures",
            ]

            for category in fixture_categories:
                has_category = category in content
                self.add_result(CheckResult(
                    name=f"Fixture category: {category}",
                    passed=has_category,
                    message="Present" if has_category else "Missing",
                    severity="low" if not has_category else "info"
                ))

    # ============================================================
    # Production Readiness Verification
    # ============================================================

    def verify_production_readiness(self):
        """Verify production readiness components."""
        self.log("\n" + "=" * 60, Colors.BOLD)
        self.log("5. PRODUCTION READINESS VERIFICATION", Colors.BOLD)
        print("=" * 60)

        # Check for production components
        production_components = [
            ("src/production/api.py", "FastAPI application"),
            ("src/production/auth.py", "Authentication module"),
            ("src/production/caching.py", "Semantic caching"),
            ("src/production/monitoring.py", "Monitoring system"),
            ("src/production/observability.py", "Observability"),
            ("src/production/query_enhancement.py", "Query enhancement"),
        ]

        for file_path, description in production_components:
            full_path = self.project_root / file_path
            exists = full_path.exists()
            self.add_result(CheckResult(
                name=f"Production: {description}",
                passed=exists,
                message="Implemented" if exists else "Missing",
                severity="high" if not exists else "info"
            ))

        # Check for health check endpoint
        api_path = self.project_root / "src" / "production" / "api.py"
        if api_path.exists():
            content = api_path.read_text(encoding='utf-8')
            has_health = "/health" in content
            self.add_result(CheckResult(
                name="Health check endpoint",
                passed=has_health,
                message="Implemented" if has_health else "Missing",
                severity="high" if not has_health else "info"
            ))

        # Check for rate limiting
        auth_path = self.project_root / "src" / "production" / "auth.py"
        if auth_path.exists():
            content = auth_path.read_text(encoding='utf-8')
            has_rate_limit = "RateLimiter" in content
            self.add_result(CheckResult(
                name="Rate limiting",
                passed=has_rate_limit,
                message="Implemented" if has_rate_limit else "Missing",
                severity="medium" if not has_rate_limit else "info"
            ))

    # ============================================================
    # Docker & Infrastructure Verification
    # ============================================================

    def verify_docker_infrastructure(self):
        """Verify Docker and infrastructure configurations."""
        self.log("\n" + "=" * 60, Colors.BOLD)
        self.log("6. DOCKER & INFRASTRUCTURE VERIFICATION", Colors.BOLD)
        print("=" * 60)

        # Check for Docker files
        docker_files = [
            ("Dockerfile", "Main Dockerfile"),
            ("docker-compose.yml", "Docker Compose"),
            ("Dockerfile.streamlit", "Streamlit Dockerfile"),
        ]

        for file_path, description in docker_files:
            full_path = self.project_root / file_path
            exists = full_path.exists()
            self.add_result(CheckResult(
                name=f"Docker: {description}",
                passed=exists,
                message="Exists" if exists else "Missing",
                severity="medium" if not exists else "info"
            ))

        # Check for monitoring configuration
        monitoring_files = [
            ("config/prometheus.yml", "Prometheus config"),
            ("config/grafana", "Grafana dashboards"),
        ]

        for file_path, description in monitoring_files:
            full_path = self.project_root / file_path
            exists = full_path.exists()
            self.add_result(CheckResult(
                name=f"Monitoring: {description}",
                passed=exists,
                message="Exists" if exists else "Missing",
                severity="low" if not exists else "info"
            ))

    # ============================================================
    # Configuration Verification
    # ============================================================

    def verify_configuration(self):
        """Verify configuration files."""
        self.log("\n" + "=" * 60, Colors.BOLD)
        self.log("7. CONFIGURATION VERIFICATION", Colors.BOLD)
        print("=" * 60)

        # Check for configuration files
        config_files = [
            (".pre-commit-config.yaml", "Pre-commit hooks"),
            ("Makefile", "Makefile commands"),
            ("requirements.txt", "Requirements"),
            ("requirements-dev.txt", "Dev requirements"),
            ("setup.py", "Setup script"),
        ]

        for file_path, description in config_files:
            full_path = self.project_root / file_path
            exists = full_path.exists()
            self.add_result(CheckResult(
                name=f"Config: {description}",
                passed=exists,
                message="Exists" if exists else "Missing",
                severity="high" if not exists else "info"
            ))

        # Verify Makefile has required commands
        makefile_path = self.project_root / "Makefile"
        if makefile_path.exists():
            content = makefile_path.read_text(encoding='utf-8')
            required_commands = [
                "install",
                "test",
                "lint",
                "format",
                "type-check",
                "docker-build",
                "docs",
            ]

            for cmd in required_commands:
                has_cmd = f"{cmd}:" in content
                self.add_result(CheckResult(
                    name=f"Makefile command: {cmd}",
                    passed=has_cmd,
                    message="Present" if has_cmd else "Missing",
                    severity="low" if not has_cmd else "info"
                ))

    # ============================================================
    # Summary Report
    # ============================================================

    def generate_summary(self) -> Dict:
        """Generate verification summary."""
        total = len(self.results)
        passed = sum(1 for r in self.results if r.passed)
        failed = total - passed

        by_severity = {
            "critical": {"total": 0, "failed": 0},
            "high": {"total": 0, "failed": 0},
            "medium": {"total": 0, "failed": 0},
            "low": {"total": 0, "failed": 0},
            "info": {"total": 0, "failed": 0},
        }

        for result in self.results:
            by_severity[result.severity]["total"] += 1
            if not result.passed:
                by_severity[result.severity]["failed"] += 1

        duration = (datetime.now() - self.start_time).total_seconds()

        return {
            "timestamp": datetime.now().isoformat(),
            "duration_seconds": duration,
            "total_checks": total,
            "passed": passed,
            "failed": failed,
            "pass_rate": passed / total * 100 if total > 0 else 0,
            "by_severity": by_severity,
        }

    def print_summary(self):
        """Print verification summary."""
        summary = self.generate_summary()

        self.log("\n" + "=" * 60, Colors.BOLD)
        self.log("VERIFICATION SUMMARY", Colors.BOLD)
        print("=" * 60)

        self.log(f"\nTotal Checks: {summary['total_checks']}", Colors.BOLD)
        self.log(f"Passed: {summary['passed']}", Colors.GREEN)
        self.log(f"Failed: {summary['failed']}", Colors.RED if summary['failed'] > 0 else Colors.GREEN)
        self.log(f"Pass Rate: {summary['pass_rate']:.1f}%", Colors.BOLD)
        self.log(f"Duration: {summary['duration_seconds']:.2f}s", Colors.CYAN)

        self.log("\nFailures by Severity:", Colors.BOLD)
        for severity, counts in summary['by_severity'].items():
            if counts['total'] > 0:
                color = {
                    "critical": Colors.RED,
                    "high": Colors.RED,
                    "medium": Colors.YELLOW,
                    "low": Colors.BLUE,
                    "info": Colors.CYAN,
                }.get(severity, Colors.RESET)

                self.log(f"  {severity.upper()}: {counts['failed']}/{counts['total']}", color)

        # Production readiness assessment
        critical_failures = summary['by_severity']['critical']['failed']
        high_failures = summary['by_severity']['high']['failed']

        self.log("\n" + "=" * 60, Colors.BOLD)
        if critical_failures > 0 or high_failures > 0:
            self.log("⚠️  PRODUCTION READINESS: NOT READY", Colors.RED)
            self.log(f"   Critical issues: {critical_failures}", Colors.RED)
            self.log(f"   High priority issues: {high_failures}", Colors.RED)
        elif summary['pass_rate'] >= 95:
            self.log("✅ PRODUCTION READINESS: READY", Colors.GREEN)
            self.log("   All critical and high-priority checks passed!", Colors.GREEN)
        else:
            self.log("⚠️  PRODUCTION READINESS: NEEDS REVIEW", Colors.YELLOW)
            self.log(f"   Pass rate: {summary['pass_rate']:.1f}%", Colors.YELLOW)

        print("=" * 60)

    def run_all_checks(self):
        """Run all verification checks."""
        self.log("\n" + "=" * 70, Colors.BOLD + Colors.CYAN)
        self.log("  AI-MASTERY-2026 ARCHITECTURE VERIFICATION", Colors.BOLD + Colors.CYAN)
        self.log(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", Colors.CYAN)
        self.log("=" * 70, Colors.BOLD + Colors.CYAN)

        self.verify_module_imports()
        self.verify_code_quality()
        self.verify_documentation()
        self.verify_tests()
        self.verify_production_readiness()
        self.verify_docker_infrastructure()
        self.verify_configuration()

        self.print_summary()

        return self.generate_summary()


def main():
    """Main entry point."""
    verifier = ArchitectureVerifier(PROJECT_ROOT)
    summary = verifier.run_all_checks()

    # Save results to file
    results_file = PROJECT_ROOT / "verification_results.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to: {results_file}")

    # Exit with error code if critical or high priority failures
    critical_failures = summary['by_severity']['critical']['failed']
    high_failures = summary['by_severity']['high']['failed']

    if critical_failures > 0 or high_failures > 0:
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
