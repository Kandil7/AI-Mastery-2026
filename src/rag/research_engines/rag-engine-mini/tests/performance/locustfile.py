"""
Locust load testing configuration for RAG Engine Mini.

This module provides load testing scenarios for the RAG API endpoints using Locust.
It simulates realistic user behavior including authentication, document uploads,
question asking, and chat interactions.

Usage:
    locust -f tests/performance/locustfile.py --host=http://localhost:8000

Environment Variables:
    TEST_USER_EMAIL: Email for test user (default: test@example.com)
    TEST_USER_PASSWORD: Password for test user (default: TestPass123!)
    API_KEY: Optional API key for authenticated requests
"""

import os
import random
import string
from typing import Optional

from locust import HttpUser, between, task
from locust.exception import StopUser


class RAGAPIUser(HttpUser):
    """
    Simulated user interacting with the RAG Engine API.

    This user class represents a typical API consumer who:
    1. Authenticates with the system
    2. Uploads documents
    3. Asks questions about documents
    4. Manages chat sessions
    5. Searches through document collections

    Attributes:
        wait_time: Random wait between 1-5 seconds between tasks
        token: JWT access token for authenticated requests
        api_key: Optional API key for key-based auth
        documents: List of document IDs created by this user
    """

    wait_time = between(1, 5)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.token: Optional[str] = None
        self.api_key: Optional[str] = os.getenv("API_KEY")
        self.documents: list[str] = []
        self.user_email: str = os.getenv("TEST_USER_EMAIL", "test@example.com")
        self.user_password: str = os.getenv("TEST_USER_PASSWORD", "TestPass123!")

    def on_start(self):
        """
        Called when a user starts.

        Attempts to authenticate and obtain a JWT token.
        If API_KEY is set, uses API key authentication instead.
        """
        if self.api_key:
            # Using API key authentication
            return

        # Attempt to login
        response = self.client.post(
            "/api/v1/auth/login", json={"email": self.user_email, "password": self.user_password}
        )

        if response.status_code == 200:
            data = response.json()
            self.token = data.get("access_token")
        else:
            # Try to register if login fails
            self._register_user()

    def _register_user(self):
        """Register a new user if login fails."""
        # Generate unique email
        random_suffix = "".join(random.choices(string.ascii_lowercase + string.digits, k=8))
        email = f"loadtest_{random_suffix}@example.com"

        response = self.client.post(
            "/api/v1/auth/register", json={"email": email, "password": self.user_password}
        )

        if response.status_code == 201:
            # Login with new credentials
            login_response = self.client.post(
                "/api/v1/auth/login", json={"email": email, "password": self.user_password}
            )

            if login_response.status_code == 200:
                self.token = login_response.json().get("access_token")
                self.user_email = email

    def get_headers(self) -> dict:
        """Get request headers with authentication."""
        if self.api_key:
            return {"X-API-Key": self.api_key}
        elif self.token:
            return {"Authorization": f"Bearer {self.token}"}
        return {}

    @task(3)
    def ask_question(self):
        """
        Simulate asking a question to the RAG system.

        Weight: 3 (high frequency)
        """
        questions = [
            "What is the main topic of this document?",
            "Summarize the key points",
            "What are the important dates mentioned?",
            "Who are the main people discussed?",
            "What conclusions are drawn?",
            "Explain the methodology used",
            "What are the limitations?",
            "How does this relate to previous work?",
        ]

        self.client.post(
            "/api/v1/ask",
            headers=self.get_headers(),
            json={"question": random.choice(questions), "k": random.randint(3, 10)},
            name="/api/v1/ask",
        )

    @task(2)
    def search_documents(self):
        """
        Simulate searching for documents.

        Weight: 2 (medium frequency)
        """
        search_terms = [
            "report",
            "analysis",
            "data",
            "results",
            "methodology",
            "conclusion",
            "introduction",
            "summary",
        ]

        self.client.get(
            "/api/v1/documents/search",
            headers=self.get_headers(),
            params={"q": random.choice(search_terms), "limit": random.randint(5, 20)},
            name="/api/v1/documents/search",
        )

    @task(2)
    def list_documents(self):
        """
        Simulate listing all documents.

        Weight: 2 (medium frequency)
        """
        self.client.get(
            "/api/v1/documents",
            headers=self.get_headers(),
            params={"limit": random.randint(10, 50), "offset": random.randint(0, 100)},
            name="/api/v1/documents",
        )

    @task(1)
    def get_document(self):
        """
        Simulate retrieving a specific document.

        Weight: 1 (low frequency)
        Only works if documents exist.
        """
        if self.documents:
            doc_id = random.choice(self.documents)
            self.client.get(
                f"/api/v1/documents/{doc_id}",
                headers=self.get_headers(),
                name="/api/v1/documents/{id}",
            )

    @task(1)
    def create_chat_session(self):
        """
        Simulate creating a new chat session.

        Weight: 1 (low frequency)
        """
        response = self.client.post(
            "/api/v1/chat/sessions",
            headers=self.get_headers(),
            json={"title": f"Load Test Session {random.randint(1, 1000)}"},
            name="/api/v1/chat/sessions",
        )

        if response.status_code == 201:
            # Could store session ID for future chat messages
            pass

    @task(2)
    def health_check(self):
        """
        Simulate health check requests.

        Weight: 2 (medium frequency)
        Used by monitoring systems.
        """
        self.client.get("/health", name="/health")

    @task(1)
    def get_query_history(self):
        """
        Simulate retrieving query history.

        Weight: 1 (low frequency)
        """
        self.client.get(
            "/api/v1/queries/history",
            headers=self.get_headers(),
            params={"limit": random.randint(5, 20)},
            name="/api/v1/queries/history",
        )


class RAGReadOnlyUser(HttpUser):
    """
    Simulated read-only user (no authentication required).

    This user only performs read operations on public endpoints.
    Useful for testing public API access and health checks.
    """

    wait_time = between(2, 10)

    @task(5)
    def health_check(self):
        """Check system health."""
        self.client.get("/health", name="/health (readonly)")

    @task(3)
    def readiness_check(self):
        """Check system readiness."""
        self.client.get("/health/ready", name="/health/ready")

    @task(2)
    def liveness_check(self):
        """Check system liveness."""
        self.client.get("/health/live", name="/health/live")


class RAGHeavyUser(HttpUser):
    """
    Simulated heavy user performing intensive operations.

    This user performs more resource-intensive tasks like:
    - Bulk operations
    - Complex queries
    - Document uploads

    Use sparingly to simulate power users.
    """

    wait_time = between(5, 15)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.token: Optional[str] = None
        self.api_key: Optional[str] = os.getenv("API_KEY")

    def on_start(self):
        """Authenticate on start."""
        if self.api_key:
            return

        # Try to get a token
        response = self.client.post(
            "/api/v1/auth/login",
            json={
                "email": os.getenv("TEST_USER_EMAIL", "test@example.com"),
                "password": os.getenv("TEST_USER_PASSWORD", "TestPass123!"),
            },
        )

        if response.status_code == 200:
            self.token = response.json().get("access_token")

    def get_headers(self) -> dict:
        """Get authentication headers."""
        if self.api_key:
            return {"X-API-Key": self.api_key}
        elif self.token:
            return {"Authorization": f"Bearer {self.token}"}
        return {}

    @task(3)
    def complex_query(self):
        """Perform complex multi-step RAG queries."""
        # Step 1: Search for documents
        search_response = self.client.get(
            "/api/v1/documents/search",
            headers=self.get_headers(),
            params={"q": "complex analysis", "limit": 5},
            name="heavy/search",
        )

        if search_response.status_code == 200:
            # Step 2: Ask detailed question
            self.client.post(
                "/api/v1/ask",
                headers=self.get_headers(),
                json={
                    "question": "Provide a comprehensive analysis of the search results",
                    "k": 10,
                },
                name="heavy/ask",
            )

    @task(1)
    def bulk_operation_simulation(self):
        """Simulate bulk operations (lighter version for load testing)."""
        # This would trigger actual bulk operations in production
        # For load testing, we just hit the endpoint
        self.client.post(
            "/api/v1/documents/bulk/upload",
            headers=self.get_headers(),
            json={
                "documents": [{"title": f"Doc {i}", "content": "Sample content"} for i in range(5)]
            },
            name="heavy/bulk-upload",
        )


# Configuration for different load patterns
class SteadyLoadShape:
    """
    Steady load pattern for baseline performance testing.

    Maintains a constant number of users over time.
    """

    def tick(self):
        """Return user count and spawn rate."""
        # 50 users, 10 users per second spawn rate
        return 50, 10


class SpikeLoadShape:
    """
    Spike load pattern for stress testing.

    Simulates sudden traffic spikes to test system resilience.
    """

    def tick(self):
        """Return user count based on time."""
        import time

        run_time = time.time()

        # Spike every 60 seconds
        if run_time % 60 < 10:
            # Spike: 200 users
            return 200, 50
        else:
            # Normal: 50 users
            return 50, 10


class RampUpLoadShape:
    """
    Ramp-up load pattern for capacity testing.

    Gradually increases load to find system breaking point.
    """

    def tick(self):
        """Return increasing user count."""
        import time

        run_time = time.time()

        # Increase users by 10 every 30 seconds, max 300
        user_count = min(10 + int(run_time / 30) * 10, 300)
        return user_count, 20


if __name__ == "__main__":
    # This allows running the file directly for testing
    import sys

    print("Locustfile loaded successfully!")
    print("Run with: locust -f tests/performance/locustfile.py --host=http://localhost:8000")
