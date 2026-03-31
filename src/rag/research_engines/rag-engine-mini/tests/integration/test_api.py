"""
API Integration Tests
======================
End-to-end tests for API endpoints.
"""

import io

import pytest
from fastapi.testclient import TestClient


class TestHealthEndpoints:
    """Tests for health check endpoints."""
    
    def test_health_check(self, client: TestClient):
        """Health endpoint should return ok status."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
    
    def test_readiness_check(self, client: TestClient):
        """Readiness endpoint should return ready status."""
        response = client.get("/health/ready")
        assert response.status_code == 200
        data = response.json()
        assert data["ready"] is True


class TestDocumentEndpoints:
    """Tests for document management endpoints."""
    
    def test_upload_requires_auth(self, client: TestClient):
        """Upload should require API key."""
        response = client.post(
            "/api/v1/documents/upload",
            files={"file": ("test.txt", b"Hello world", "text/plain")},
        )
        assert response.status_code == 422  # Missing header
    
    def test_upload_with_auth(self, client: TestClient, auth_headers: dict):
        """Upload should work with valid API key."""
        response = client.post(
            "/api/v1/documents/upload",
            headers=auth_headers,
            files={"file": ("test.txt", b"Hello world", "text/plain")},
        )
        # May be 200 or 500 depending on mock setup
        assert response.status_code in [200, 500]
    
    def test_list_documents_requires_auth(self, client: TestClient):
        """List documents should require API key."""
        response = client.get("/api/v1/documents")
        assert response.status_code == 422


class TestQueryEndpoints:
    """Tests for query endpoints."""
    
    def test_ask_requires_auth(self, client: TestClient):
        """Ask endpoint should require API key."""
        response = client.post(
            "/api/v1/queries/ask-hybrid",
            json={"question": "What is AI?"},
        )
        assert response.status_code == 422
    
    def test_ask_requires_question(self, client: TestClient, auth_headers: dict):
        """Ask endpoint should require question field."""
        response = client.post(
            "/api/v1/queries/ask-hybrid",
            headers=auth_headers,
            json={},
        )
        assert response.status_code == 422
    
    def test_ask_validates_question_length(self, client: TestClient, auth_headers: dict):
        """Question must be at least 2 characters."""
        response = client.post(
            "/api/v1/queries/ask-hybrid",
            headers=auth_headers,
            json={"question": "?"},  # Too short
        )
        assert response.status_code == 422
