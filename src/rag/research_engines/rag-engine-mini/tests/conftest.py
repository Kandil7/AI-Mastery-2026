"""
Pytest Configuration and Fixtures
===================================
Shared fixtures for unit and integration tests.
"""

import pytest
from fastapi.testclient import TestClient

from src.main import app
from src.core.bootstrap import reset_container


@pytest.fixture(scope="function")
def client() -> TestClient:
    """Create a test client for the FastAPI app."""
    reset_container()  # Clear cached container
    return TestClient(app)


@pytest.fixture
def api_key() -> str:
    """Demo API key for testing."""
    return "test_api_key_12345678"


@pytest.fixture
def auth_headers(api_key: str) -> dict[str, str]:
    """Headers with API key for authenticated requests."""
    return {"X-API-KEY": api_key}
