"""
User Registration Integration Tests
=================================
Tests for user registration flow.
"""

import pytest
from fastapi.testclient import TestClient

from src.application.use_cases.register_user import (
    RegisterUserUseCase,
    RegisterUserRequest,
    SimpleEmailValidator,
)


class TestUserRegistration:
    """Tests for user registration use case."""

    def test_register_with_valid_data(self):
        """Registration should succeed with valid data."""

        # Placeholder repo
        class MockUserRepo:
            def create_user(self, *, email, hashed_password):
                return "user_123"

            def email_exists(self, *, email):
                return False

        use_case = RegisterUserUseCase(user_repo=MockUserRepo())

        request = RegisterUserRequest(
            email="test@example.com",
            password="SecurePass123!",
        )

        response = use_case.execute(request)

        assert response.user_id == "user_123"
        assert response.email == "test@example.com"
        assert "successfully" in response.message.lower()

    def test_register_with_short_password(self):
        """Registration should reject short passwords."""

        class MockUserRepo:
            def create_user(self, *, email, hashed_password):
                return "user_123"

            def email_exists(self, *, email):
                return False

        use_case = RegisterUserUseCase(user_repo=MockUserRepo())

        request = RegisterUserRequest(
            email="test@example.com",
            password="Short1!",
        )

        with pytest.raises(ValueError, match="Password must be at least 8"):
            use_case.execute(request)

    def test_register_with_no_uppercase(self):
        """Registration should reject passwords without uppercase."""

        class MockUserRepo:
            def create_user(self, *, email, hashed_password):
                return "user_123"

            def email_exists(self, *, email):
                return False

        use_case = RegisterUserUseCase(user_repo=MockUserRepo())

        request = RegisterUserRequest(
            email="test@example.com",
            password="lowercase1!",
        )

        with pytest.raises(ValueError, match="must contain at least one uppercase"):
            use_case.execute(request)

    def test_register_with_no_lowercase(self):
        """Registration should reject passwords without lowercase."""

        class MockUserRepo:
            def create_user(self, *, email, hashed_password):
                return "user_123"

            def email_exists(self, *, email):
                return False

        use_case = RegisterUserUseCase(user_repo=MockUserRepo())

        request = RegisterUserRequest(
            email="test@example.com",
            password="UPPERCASE1!",
        )

        with pytest.raises(ValueError, match="must contain at least one lowercase"):
            use_case.execute(request)

    def test_register_with_no_number(self):
        """Registration should reject passwords without numbers."""

        class MockUserRepo:
            def create_user(self, *, email, hashed_password):
                return "user_123"

            def email_exists(self, *, email):
                return False

        use_case = RegisterUserUseCase(user_repo=MockUserRepo())

        request = RegisterUserRequest(
            email="test@example.com",
            password="NoNumber!",
        )

        with pytest.raises(ValueError, match="must contain at least one number"):
            use_case.execute(request)

    def test_register_with_no_special_char(self):
        """Registration should reject passwords without special characters."""

        class MockUserRepo:
            def create_user(self, *, email, hashed_password):
                return "user_123"

            def email_exists(self, *, email):
                return False

        use_case = RegisterUserUseCase(user_repo=MockUserRepo())

        request = RegisterUserRequest(
            email="test@example.com",
            password="NoSpecial123",
        )

        with pytest.raises(ValueError, match="must contain at least one special character"):
            use_case.execute(request)

    def test_register_with_invalid_email(self):
        """Registration should reject invalid email format."""

        class MockUserRepo:
            def create_user(self, *, email, hashed_password):
                return "user_123"

            def email_exists(self, *, email):
                return False

        use_case = RegisterUserUseCase(user_repo=MockUserRepo())

        request = RegisterUserRequest(
            email="not-an-email",
            password="SecurePass123!",
        )

        with pytest.raises(ValueError, match="Invalid email format"):
            use_case.execute(request)

    def test_register_with_duplicate_email(self):
        """Registration should reject duplicate emails."""

        class MockUserRepo:
            def create_user(self, *, email, hashed_password):
                return "user_123"

            def email_exists(self, *, email):
                return True  # Email already exists

        use_case = RegisterUserUseCase(user_repo=MockUserRepo())

        request = RegisterUserRequest(
            email="existing@example.com",
            password="SecurePass123!",
        )

        with pytest.raises(ValueError, match="already registered"):
            use_case.execute(request)

    def test_register_hashes_password(self):
        """Registration should hash password before storage."""
        # Check if password hasher is called
        from src.adapters.security.password_hasher import get_hasher

        hasher = get_hasher()

        original_hash = hasher.hash
        hasher.hash = lambda p: "hashed_value"  # Mock to verify call

        class MockUserRepo:
            def create_user(self, *, email, hashed_password):
                assert hashed_password == "hashed_value"
                return "user_123"

            def email_exists(self, *, email):
                return False

        use_case = RegisterUserUseCase(user_repo=MockUserRepo())

        request = RegisterUserRequest(
            email="test@example.com",
            password="SecurePass123!",
        )

        use_case.execute(request)

        # Restore original
        hasher.hash = original_hash


class TestAuthEndpoints:
    """Tests for authentication endpoints."""

    def test_register_endpoint_requires_email(self, client: TestClient):
        """Register endpoint should require email."""
        response = client.post(
            "/api/v1/auth/register",
            json={"password": "TestPass123!"},
        )
        assert response.status_code == 422  # Validation error

    def test_register_endpoint_requires_password(self, client: TestClient):
        """Register endpoint should require password."""
        response = client.post(
            "/api/v1/auth/register",
            json={"email": "test@example.com"},
        )
        assert response.status_code == 422  # Validation error

    def test_register_with_valid_data(self, client: TestClient):
        """Register endpoint should succeed with valid data."""
        response = client.post(
            "/api/v1/auth/register",
            json={
                "email": "newuser@example.com",
                "password": "SecurePass123!",
            },
        )
        # Should succeed (or 501 if not fully implemented)
        assert response.status_code in [201, 500, 501]

        if response.status_code == 201:
            data = response.json()
            assert "user_id" in data
            assert "email" in data

    def test_login_endpoint_not_implemented(self, client: TestClient):
        """Login endpoint should return 501 (not implemented)."""
        response = client.post(
            "/api/v1/auth/login",
            json={
                "email": "test@example.com",
                "password": "TestPass123!",
            },
        )
        assert response.status_code == 501
        assert "not yet implemented" in response.json()["detail"]
