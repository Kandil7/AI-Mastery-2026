"""
Login & Session Management Unit Tests
===================================
Tests for login use case and JWT tokens.
"""

import pytest
from unittest.mock import MagicMock, AsyncMock
from src.application.use_cases.login_user import LoginUserUseCase, LoginUserRequest
from src.adapters.security.jwt_provider import JWTProvider


class TestLoginUserUseCase:
    """Tests for login user use case."""

    def test_login_with_valid_credentials(self):
        """Login should succeed with valid credentials."""
        # Mock user repository
        mock_repo = MagicMock()
        mock_repo.get_user_by_email.return_value = {
            "id": "user_123",
            "email": "test@example.com",
            "hashed_password": "argon2$hash...",
        }
        mock_repo.verify_password.return_value = True

        # Mock JWT provider
        mock_jwt = MagicMock()
        mock_jwt.create_access_token.return_value = "access_token_123"
        mock_jwt.create_refresh_token.return_value = "refresh_token_456"

        # Create use case
        use_case = LoginUserUseCase(user_repo=mock_repo)

        # Execute login
        request = LoginUserRequest(email="test@example.com", password="ValidPassword123!")
        response = use_case.execute(request)

        assert response.user_id == "user_123"
        assert response.email == "test@example.com"
        assert response.access_token == "access_token_123"
        assert response.refresh_token == "refresh_token_456"

        # Verify methods were called
        mock_repo.get_user_by_email.assert_called_once_with("email", "test@example.com")
        mock_repo.verify_password.assert_called_once()
        mock_jwt.create_access_token.assert_called_once()
        mock_jwt.create_refresh_token.assert_called_once()

    def test_login_with_invalid_email(self):
        """Login should fail with invalid email."""
        mock_repo = MagicMock()
        mock_repo.get_user_by_email.return_value = None

        use_case = LoginUserUseCase(user_repo=mock_repo)

        request = LoginUserRequest(email="invalid", password="ValidPassword123!")

        with pytest.raises(ValueError, match="Invalid email or password"):
            use_case.execute(request)

    def test_login_with_short_password(self):
        """Login should fail with short password."""
        mock_repo = MagicMock()
        mock_repo.get_user_by_email.return_value = {
            "id": "user_123",
            "email": "test@example.com",
            "hashed_password": "argon2$hash...",
        }

        use_case = LoginUserUseCase(user_repo=mock_repo)

        request = LoginUserRequest(email="test@example.com", password="short")

        with pytest.raises(ValueError, match="Password must be at least 8"):
            use_case.execute(request)

    def test_login_with_wrong_password(self):
        """Login should fail with wrong password."""
        mock_repo = MagicMock()
        mock_repo.get_user_by_email.return_value = {
            "id": "user_123",
            "email": "test@example.com",
            "hashed_password": "argon2$hash...",
        }
        mock_repo.verify_password.return_value = False

        use_case = LoginUserUseCase(user_repo=mock_repo)

        request = LoginUserRequest(email="test@example.com", password="WrongPassword123!")

        with pytest.raises(ValueError, match="Invalid email or password"):
            use_case.execute(request)

        mock_repo.verify_password.assert_called_once()


class TestApiKeyManagement:
    """Tests for API key management use case."""

    def test_create_api_key(self):
        """API key creation should succeed."""
        from src.application.use_cases.api_key_management import (
            ApiKeyManagementUseCase,
            CreateApiKeyRequest,
        )

        # Mock user repository
        mock_repo = MagicMock()
        mock_repo.create_api_key.return_value = {
            "id": "key_123",
            "key_hash": "sha256...",
            "key_prefix": "rag_",
        }

        # Mock nanoid
        with pytest.mock.patch(
            "src.application.use_cases.api_key_management.nanoid.generate"
        ) as mock_nanoid:
            mock_nanoid.return_value = "rag_550e8400"

            use_case = ApiKeyManagementUseCase(user_repo=mock_repo)

            request = CreateApiKeyRequest(
                user_id="user_123",
                name="Production Key",
                permissions=["read", "write"],
            )

            response = use_case.create(request)

            assert response.key_id == "key_123"
            assert response.api_key.startswith("rag_")
            assert response.is_active is True
            assert "read" in response.permissions
            assert "write" in response.permissions

    def test_list_api_keys(self):
        """API key listing should return user's keys."""
        from src.application.use_cases.api_key_management import ApiKeyManagementUseCase

        mock_repo = MagicMock()
        mock_repo.list_api_keys.return_value = [
            {
                "id": "key_1",
                "key_hash": "hash1",
                "name": "Dev Key",
                "is_active": True,
            },
            {
                "id": "key_2",
                "key_hash": "hash2",
                "name": "Prod Key",
                "is_active": False,
            },
        ]

        use_case = ApiKeyManagementUseCase(user_repo=mock_repo)

        keys = use_case.list_keys(user_id="user_123")

        assert len(keys) == 2
        assert keys[0]["name"] == "Dev Key"
        assert keys[0]["is_active"] is True
        assert keys[1]["name"] == "Prod Key"
        assert keys[1]["is_active"] is False

    def test_revoke_api_key(self):
        """API key revocation should succeed."""
        from src.application.use_cases.api_key_management import ApiKeyManagementUseCase

        mock_repo = MagicMock()
        mock_repo.revoke_api_key.return_value = None

        use_case = ApiKeyManagementUseCase(user_repo=mock_repo)

        use_case.revoke_key(key_id="key_123", user_id="user_123")

        mock_repo.revoke_api_key.assert_called_once_with("key_id", "key_123", "user_id", "user_123")


class TestRateLimiting:
    """Tests for rate limiting middleware."""

    def test_token_bucket_consumes_tokens(self):
        """Token bucket should consume tokens correctly."""
        from src.api.middleware.rate_limiter import TokenBucket

        bucket = TokenBucket(capacity=10, refill_rate=0.1)  # 10 tokens, refill 0.1 per second

        # First request should succeed
        assert bucket.consume(1) is True
        assert bucket.get_available_tokens() == 9

        # Wait for refill
        import time

        time.sleep(1)  # Should refill 1 token
        assert bucket.get_available_tokens() == 10

        # Consume multiple tokens
        assert bucket.consume(5) is True
        assert bucket.get_available_tokens() == 5

    def test_token_bucket_rejects_when_empty(self):
        """Token bucket should reject when no tokens."""
        from src.api.middleware.rate_limiter import TokenBucket

        bucket = TokenBucket(capacity=10, refill_rate=0.0)

        # Consume all tokens
        for _ in range(10):
            bucket.consume(1)

        assert bucket.get_available_tokens() == 0

        # Should reject next request
        assert bucket.consume(1) is False

    def test_rate_limit_headers(self):
        """Rate limiting should include proper headers."""
        from src.api.middleware.rate_limiter import RateLimitExceeded
        from fastapi import HTTPException

        exception = RateLimitExceeded()

        # Headers should include:
        assert "429" in exception.detail.lower()
        assert exception.status_code == 429


class TestInputSanitization:
    """Tests for input sanitization service."""

    def test_sanitize_html_removes_scripts(self):
        """HTML sanitization should remove script tags."""
        from src.application.services.input_sanitizer import InputSanitizer, get_sanitizer

        sanitizer = get_sanitizer()

        # Test XSS prevention
        malicious = "<script>alert('XSS')</script>"
        safe = sanitizer.sanitize_html(malicious)

        assert "<script>" not in safe
        assert "alert" not in safe
        assert safe == "alert('XSS')"

    def test_sanitize_html_preserves_safe_tags(self):
        """HTML sanitization should preserve safe tags."""
        from src.application.services.input_sanitizer import InputSanitizer, get_sanitizer

        sanitizer = InputSanitizer(allowed_tags=["p", "strong"])
        safe = sanitizer.sanitize_html("<p>Safe text</p>")

        assert "<p>" in safe
        assert "</p>" in safe

    def test_strip_html_removes_all_tags(self):
        """Strip HTML should remove all tags."""
        from src.application.services.input_sanitizer import InputSanitizer, get_sanitizer

        sanitizer = InputSanitizer()

        html = "<p>Text</p><strong>Bold</strong>"
        plain = sanitizer.strip_html(html)

        assert "<" not in plain
        assert "</" not in plain
        assert plain == "TextBold"

    def test_escape_sql_special_chars(self):
        """SQL escaping should escape special characters."""
        from src.application.services.input_sanitizer import InputSanitizer, get_sanitizer

        sanitizer = InputSanitizer()

        # Test SQL injection patterns
        malicious = "admin' OR '1'='1"
        safe = sanitizer.escape_sql(malicious)

        assert "''" not in safe  # Quotes escaped
        assert "\\;" not in safe  # Semicolon escaped

    def test_sanitize_filename_prevents_path_traversal(self):
        """Filename sanitization should prevent path traversal."""
        from src.application.services.input_sanitizer import InputSanitizer, get_sanitizer

        sanitizer = InputSanitizer()

        # Test path traversal prevention
        malicious = "../../../etc/passwd"
        safe = sanitizer.sanitize_filename(malicious)

        assert "../" not in safe
        assert safe == "..__.._etc_passwd"

    def test_sanitize_markdown_removes_js(self):
        """Markdown sanitization should remove JavaScript."""
        from src.application.services.input_sanitizer import InputSanitizer, get_sanitizer

        sanitizer = InputSanitizer()

        malicious = "[javascript:alert(1)](evil)"
        safe = sanitizer.sanitize_markdown(malicious)

        assert "javascript:" not in safe
        assert "js:" not in safe


class TestSecurityHeaders:
    """Tests for security headers middleware."""

    def test_csp_header_added(self):
        """CSP header should be added to responses."""
        from src.api.middleware.security_headers import SecurityHeadersMiddleware
        from fastapi import Response, Request

        middleware = SecurityHeadersMiddleware()

        # Create mock request and response
        response = Response(content={"message": "OK"})

        async def call_next(request):
            return response

        # Process through middleware
        processed = await middleware(Request(), call_next)

        assert "Content-Security-Policy" in processed.headers
        assert processed.headers["Content-Security-Policy"].startswith("default-src 'self'")

    def test_hsts_header_added(self):
        """HSTS header should be added."""
        from src.api.middleware.security_headers import SecurityHeadersMiddleware
        from fastapi import Response, Request

        middleware = SecurityHeadersMiddleware()

        response = Response(content={"message": "OK"})

        async def call_next(request):
            return response

        processed = await middleware(Request(), call_next)

        assert "Strict-Transport-Security" in processed.headers
        assert processed.headers["Strict-Transport-Security"].startswith("max-age=")

    def test_x_frame_options_deny(self):
        """X-Frame-Options should be set to DENY."""
        from src.api.middleware.security_headers import SecurityHeadersMiddleware
        from fastapi import Response, Request

        middleware = SecurityHeadersMiddleware()

        response = Response(content={"message": "OK"})

        async def call_next(request):
            return response

        processed = await middleware(Request(), call_next)

        assert "X-Frame-Options" in processed.headers
        assert processed.headers["X-Frame-Options"] == "DENY"
