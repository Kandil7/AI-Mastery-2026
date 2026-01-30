"""
Integration Tests for RAG Pipeline
===================================
End-to-end tests for key user workflows.

اختبارات التكامل لخط أنبوب RAG
"""

import pytest
from httpx import AsyncClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.adapters.persistence.postgres.models import User, Document
from src.adapters.persistence.postgres.bootstrap import create_session


class TestAuthFlow:
    """Test authentication and authorization flow."""

    @pytest.mark.asyncio
    async def test_user_registration_and_login(self, async_client):
        """Test user registration followed by login."""
        # Register new user
        register_response = await async_client.post(
            "/api/v1/auth/register",
            json={
                "email": "test@example.com",
                "password": "SecurePass123!",
            },
        )
        assert register_response.status_code == 201

        # Login with credentials
        login_response = await async_client.post(
            "/api/v1/auth/login",
            json={
                "email": "test@example.com",
                "password": "SecurePass123!",
            },
        )
        assert login_response.status_code == 200

        data = login_response.json()
        assert "access_token" in data
        assert "refresh_token" in data

    @pytest.mark.asyncio
    async def test_invalid_login(self, async_client):
        """Test login with invalid credentials."""
        response = await async_client.post(
            "/api/v1/auth/login",
            json={
                "email": "test@example.com",
                "password": "WrongPassword123!",
            },
        )
        assert response.status_code == 401


class TestDocumentWorkflow:
    """Test document upload, search, and deletion."""

    @pytest.mark.asyncio
    async def test_upload_search_delete_document(self, async_client, auth_headers):
        """Test complete document workflow."""
        # Upload document
        upload_response = await async_client.post(
            "/api/v1/documents",
            headers=auth_headers,
            files={"file": ("test.pdf", b"Test content", "application/pdf")},
        )
        assert upload_response.status_code == 201

        doc_data = upload_response.json()
        doc_id = doc_data["id"]

        # Search documents
        search_response = await async_client.get(
            "/api/v1/documents/search?query=test",
            headers=auth_headers,
        )
        assert search_response.status_code == 200

        results = search_response.json()["results"]
        assert len(results) > 0
        assert any(r["id"] == doc_id for r in results)

        # Delete document
        delete_response = await async_client.delete(
            f"/api/v1/documents/{doc_id}",
            headers=auth_headers,
        )
        assert delete_response.status_code == 204

    @pytest.mark.asyncio
    async def test_document_reindexing(
        self, async_client, auth_headers, db_session, sample_document
    ):
        """Test document re-indexing."""
        # Create document with old status
        db_session.add(sample_document)
        db_session.commit()

        # Request re-index
        response = await async_client.post(
            f"/api/v1/documents/{sample_document.id}/reindex",
            headers=auth_headers,
        )
        assert response.status_code == 200

        # Verify status updated
        db_session.refresh(sample_document)
        assert sample_document.status in ["queued", "indexed"]


class TestRAGPipeline:
    """Test end-to-end RAG workflow."""

    @pytest.mark.asyncio
    async def test_ask_question_workflow(
        self,
        async_client,
        auth_headers,
        db_session,
        sample_user,
        sample_document,
    ):
        """Test complete ask question workflow."""
        # Setup: Create user and document
        db_session.add(sample_user)
        db_session.add(sample_document)
        db_session.commit()

        # Ask question
        question_response = await async_client.post(
            "/api/v1/ask",
            headers=auth_headers,
            json={
                "question": "What is RAG?",
                "k": 5,
            },
        )
        assert question_response.status_code == 200

        data = question_response.json()
        assert "answer" in data
        assert "sources" in data
        assert len(data["sources"]) <= 5

        # Verify chat session created
        sessions = db_session.query(ChatSession).filter_by(user_id=sample_user.id).all()
        assert len(sessions) == 1

    @pytest.mark.asyncio
    async def test_chat_history_workflow(
        self,
        async_client,
        auth_headers,
        db_session,
        sample_user,
    ):
        """Test chat history retrieval and management."""
        # Setup: Create user
        db_session.add(sample_user)
        db_session.commit()

        # Ask multiple questions
        for i in range(3):
            await async_client.post(
                "/api/v1/ask",
                headers=auth_headers,
                json={
                    "question": f"Question {i}",
                    "k": 5,
                },
            )

        # Retrieve query history
        history_response = await async_client.get(
            "/api/v1/queries/history",
            headers=auth_headers,
        )
        assert history_response.status_code == 200

        history = history_response.json()
        assert "questions" in history
        assert len(history["questions"]) == 3


class TestBulkOperations:
    """Test bulk upload and delete operations."""

    @pytest.mark.asyncio
    async def test_bulk_upload(self, async_client, auth_headers):
        """Test uploading multiple documents."""
        # Create test files
        files = [("files", (f"file{i}.pdf", b"Content", "application/pdf")) for i in range(5)]

        response = await async_client.post(
            "/api/v1/documents/bulk",
            headers=auth_headers,
            files=files,
        )
        assert response.status_code == 201

        data = response.json()
        assert "uploaded" in data
        assert data["uploaded"] == 5

    @pytest.mark.asyncio
    async def test_bulk_delete(
        self,
        async_client,
        auth_headers,
        db_session,
        sample_documents,
    ):
        """Test deleting multiple documents."""
        # Create test documents
        doc_ids = [doc.id for doc in sample_documents]

        # Bulk delete
        response = await async_client.post(
            "/api/v1/documents/bulk/delete",
            headers=auth_headers,
            json={"document_ids": doc_ids},
        )
        assert response.status_code == 204

        # Verify deletion
        for doc_id in doc_ids:
            doc = db_session.query(Document).filter_by(id=doc_id).first()
            assert doc is None or doc.deleted_at is not None


class TestAdminEndpoints:
    """Test admin and monitoring endpoints."""

    @pytest.mark.asyncio
    async def test_health_endpoints(self, async_client):
        """Test health check endpoints."""
        # Basic health
        response = await async_client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

        # Readiness
        response = await async_client.get("/health/ready")
        assert response.status_code in [200, 503]

        # Deep health
        response = await async_client.get("/health/deep")
        assert response.status_code == 200
        data = response.json()
        assert "database" in data
        assert "redis" in data
        assert "qdrant" in data

    @pytest.mark.asyncio
    async def test_metrics_endpoint(self, async_client):
        """Test Prometheus metrics endpoint."""
        response = await async_client.get("/metrics")
        assert response.status_code == 200

        # Verify Prometheus format
        content = response.text
        assert "# HELP" in content or "# TYPE" in content


class TestTenantIsolation:
    """Test multi-tenant isolation."""

    @pytest.mark.asyncio
    async def test_tenant_document_isolation(
        self,
        async_client,
        auth_headers_1,
        auth_headers_2,
        db_session,
        user_tenant_1,
        user_tenant_2,
        doc_tenant_1,
        doc_tenant_2,
    ):
        """Test that users can't access other tenants' documents."""
        # Setup: Create users and documents for different tenants
        db_session.add(user_tenant_1)
        db_session.add(user_tenant_2)
        db_session.add(doc_tenant_1)
        db_session.add(doc_tenant_2)
        db_session.commit()

        # Tenant 1 searches for their document
        search_1 = await async_client.get(
            "/api/v1/documents/search?query=test",
            headers=auth_headers_1,
        )
        assert search_1.status_code == 200
        results_1 = search_1.json()["results"]
        assert any(r["id"] == doc_tenant_1.id for r in results_1)
        assert not any(r["id"] == doc_tenant_2.id for r in results_1)

        # Tenant 2 searches for their document
        search_2 = await async_client.get(
            "/api/v1/documents/search?query=test",
            headers=auth_headers_2,
        )
        assert search_2.status_code == 200
        results_2 = search_2.json()["results"]
        assert any(r["id"] == doc_tenant_2.id for r in results_2)
        assert not any(r["id"] == doc_tenant_1.id for r in results_2)


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def async_client():
    """Provide async HTTP client for testing."""
    return AsyncClient(base_url="http://localhost:8000")


@pytest.fixture
def db_session():
    """Provide database session for integration tests."""
    engine = create_engine(
        "postgresql://postgres:postgres@localhost:5432/rag_test",
        pool_pre_ping=True,
    )
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


@pytest.fixture
def sample_user():
    """Provide sample user for testing."""
    return User(
        id="user-123",
        email="test@example.com",
        api_key="sk_test1234567890123456789",
    )


@pytest.fixture
def sample_document():
    """Provide sample document for testing."""
    return Document(
        id="doc-123",
        user_id="user-123",
        filename="test.pdf",
        content_type="application/pdf",
        file_path="/uploads/test.pdf",
        size_bytes=1024,
        file_sha256="abc123",
        status="indexed",
    )


@pytest.fixture
def sample_documents():
    """Provide multiple sample documents."""
    return [
        Document(
            id=f"doc-{i}",
            user_id="user-123",
            filename=f"file{i}.pdf",
            content_type="application/pdf",
            file_path=f"/uploads/file{i}.pdf",
            size_bytes=1024 * (i + 1),
            status="indexed",
        )
        for i in range(5)
    ]


@pytest.fixture
def auth_headers(sample_user):
    """Provide authentication headers."""
    return {
        "Authorization": f"Bearer {generate_jwt_token(sample_user)}",
        "Content-Type": "application/json",
    }


@pytest.fixture
def auth_headers_1(user_tenant_1):
    """Provide authentication headers for tenant 1."""
    return {
        "Authorization": f"Bearer {generate_jwt_token(user_tenant_1)}",
        "Content-Type": "application/json",
    }


@pytest.fixture
def auth_headers_2(user_tenant_2):
    """Provide authentication headers for tenant 2."""
    return {
        "Authorization": f"Bearer {generate_jwt_token(user_tenant_2)}",
        "Content-Type": "application/json",
    }


@pytest.fixture
def user_tenant_1():
    """Provide user for tenant 1."""
    return User(
        id="user-tenant1",
        email="tenant1@example.com",
        api_key="sk_tenant1_123456789",
    )


@pytest.fixture
def user_tenant_2():
    """Provide user for tenant 2."""
    return User(
        id="user-tenant2",
        email="tenant2@example.com",
        api_key="sk_tenant2_123456789",
    )


@pytest.fixture
def doc_tenant_1():
    """Provide document for tenant 1."""
    return Document(
        id="doc-tenant1",
        user_id="user-tenant1",
        filename="tenant1.pdf",
        content_type="application/pdf",
        file_path="/uploads/tenant1.pdf",
        size_bytes=1024,
        status="indexed",
    )


@pytest.fixture
def doc_tenant_2():
    """Provide document for tenant 2."""
    return Document(
        id="doc-tenant2",
        user_id="user-tenant2",
        filename="tenant2.pdf",
        content_type="application/pdf",
        file_path="/uploads/tenant2.pdf",
        size_bytes=1024,
        status="indexed",
    )


def generate_jwt_token(user):
    """Generate JWT token for testing."""
    # Simplified - in production use proper JWT library
    return f"fake_jwt_token_for_{user.id}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "integration"])
