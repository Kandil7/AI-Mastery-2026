"""
JWT Provider Unit Tests
========================
Tests for JWT token generation and validation.
"""

import pytest
from jose import jwt, JWTError
from src.adapters.security.jwt_provider import JWTProvider, get_jwt_provider


class TestJWTProviderInitialization:
    """Tests for JWT provider initialization."""

    def test_init_with_defaults(self):
        """Provider should initialize with default parameters."""
        provider = JWTProvider(secret_key="test-secret")
        assert provider is not None
        assert provider._access_expire_minutes == 15
        assert provider._refresh_expire_days == 7

    def test_init_with_custom_params(self):
        """Provider should accept custom expiration times."""
        provider = JWTProvider(
            secret_key="test-secret",
            access_expire_minutes=30,
            refresh_expire_days=14,
        )
        assert provider._access_expire_minutes == 30
        assert provider._refresh_expire_days == 14

    def test_init_with_algorithm(self):
        """Provider should accept custom algorithm."""
        provider = JWTProvider(
            secret_key="test-secret",
            algorithm="RS256",
        )
        assert provider._algorithm == "RS256"


class TestAccessTokenGeneration:
    """Tests for access token creation."""

    def test_create_access_token_returns_string(self):
        """Access token should be a JWT string."""
        provider = JWTProvider(secret_key="test-secret")
        token = provider.create_access_token(
            user_id="user_123",
            tenant_id="tenant_456",
        )
        assert isinstance(token, str)
        assert len(token) > 0

    def test_access_token_structure(self):
        """Access token should have correct structure (3 parts)."""
        provider = JWTProvider(secret_key="test-secret")
        token = provider.create_access_token(
            user_id="user_123",
            tenant_id="tenant_456",
        )
        parts = token.split(".")
        assert len(parts) == 3  # header.payload.signature

    def test_access_token_with_additional_claims(self):
        """Access token should include additional claims."""
        provider = JWTProvider(secret_key="test-secret")
        token = provider.create_access_token(
            user_id="user_123",
            tenant_id="tenant_456",
            additional_claims={"role": "admin", "permissions": ["read", "write"]},
        )

        # Decode to check claims
        payload = jwt.decode(token, "test-secret", algorithms=["HS256"])
        assert payload["role"] == "admin"
        assert "read" in payload["permissions"]

    def test_access_token_expiration(self):
        """Access token should expire in configured time."""
        provider = JWTProvider(secret_key="test-secret", access_expire_minutes=15)
        token = provider.create_access_token(
            user_id="user_123",
            tenant_id="tenant_456",
        )

        payload = jwt.decode(token, "test-secret", algorithms=["HS256"])
        # Check exp is roughly 15 minutes in future
        import time

        now = int(time.time())
        assert payload["exp"] > now
        assert payload["exp"] - now < (15 * 60 + 60)  # Within 16 minutes

    def test_access_token_includes_jti(self):
        """Access token should include unique JWT ID."""
        provider = JWTProvider(secret_key="test-secret")
        token1 = provider.create_access_token(
            user_id="user_123",
            tenant_id="tenant_456",
        )
        token2 = provider.create_access_token(
            user_id="user_123",
            tenant_id="tenant_456",
        )

        payload1 = jwt.decode(token1, "test-secret", algorithms=["HS256"])
        payload2 = jwt.decode(token2, "test-secret", algorithms=["HS256"])
        assert payload1["jti"] != payload2["jti"]


class TestRefreshTokenGeneration:
    """Tests for refresh token creation."""

    def test_create_refresh_token_returns_string(self):
        """Refresh token should be a JWT string."""
        provider = JWTProvider(secret_key="test-secret")
        token = provider.create_refresh_token(user_id="user_123")
        assert isinstance(token, str)
        assert len(token) > 0

    def test_refresh_token_expiration(self):
        """Refresh token should expire in configured days."""
        provider = JWTProvider(secret_key="test-secret", refresh_expire_days=7)
        token = provider.create_refresh_token(user_id="user_123")

        payload = jwt.decode(token, "test-secret", algorithms=["HS256"])
        import time

        now = int(time.time())
        # Check exp is roughly 7 days in future
        assert payload["exp"] > now
        assert payload["exp"] - now < (7 * 24 * 3600 + 3600)  # Within 8 days

    def test_refresh_token_includes_jti(self):
        """Refresh token should include unique JWT ID."""
        provider = JWTProvider(secret_key="test-secret")
        token1 = provider.create_refresh_token(user_id="user_123")
        token2 = provider.create_refresh_token(user_id="user_123")

        payload1 = jwt.decode(token1, "test-secret", algorithms=["HS256"])
        payload2 = jwt.decode(token2, "test-secret", algorithms=["HS256"])
        assert payload1["jti"] != payload2["jti"]

    def test_refresh_token_longer_than_access(self):
        """Refresh token should have longer expiration than access token."""
        provider = JWTProvider(secret_key="test-secret")
        access = provider.create_access_token(
            user_id="user_123",
            tenant_id="tenant_456",
        )
        refresh = provider.create_refresh_token(user_id="user_123")

        access_payload = jwt.decode(access, "test-secret", algorithms=["HS256"])
        refresh_payload = jwt.decode(refresh, "test-secret", algorithms=["HS256"])

        assert refresh_payload["exp"] > access_payload["exp"]


class TestTokenVerification:
    """Tests for token validation."""

    def test_verify_valid_access_token(self):
        """Valid access token should decode successfully."""
        provider = JWTProvider(secret_key="test-secret")
        token = provider.create_access_token(
            user_id="user_123",
            tenant_id="tenant_456",
        )

        payload = provider.verify_access_token(token)
        assert payload["sub"] == "user_123"
        assert payload["type"] == "access"

    def test_verify_valid_refresh_token(self):
        """Valid refresh token should decode successfully."""
        provider = JWTProvider(secret_key="test-secret")
        token = provider.create_refresh_token(user_id="user_123")

        payload = provider.verify_refresh_token(token)
        assert payload["sub"] == "user_123"
        assert payload["type"] == "refresh"

    def test_verify_wrong_type_fails(self):
        """Verifying access token as refresh should fail."""
        provider = JWTProvider(secret_key="test-secret")
        access_token = provider.create_access_token(
            user_id="user_123",
            tenant_id="tenant_456",
        )

        with pytest.raises(JWTError):
            provider.verify_refresh_token(access_token)

    def test_verify_invalid_token_fails(self):
        """Invalid token should raise JWTError."""
        provider = JWTProvider(secret_key="test-secret")
        invalid_token = "invalid.jwt.token.string"

        with pytest.raises(JWTError):
            provider.verify_access_token(invalid_token)

    def test_verify_wrong_secret_fails(self):
        """Token signed with different secret should fail."""
        provider1 = JWTProvider(secret_key="secret1")
        provider2 = JWTProvider(secret_key="secret2")

        token = provider1.create_access_token(
            user_id="user_123",
            tenant_id="tenant_456",
        )

        with pytest.raises(JWTError):
            provider2.verify_access_token(token)


class TestTokenExpiration:
    """Tests for token expiration."""

    def test_expired_token_raises_error(self):
        """Expired token should raise JWTError."""
        provider = JWTProvider(
            secret_key="test-secret",
            access_expire_minutes=-1,  # Already expired
        )
        token = provider.create_access_token(
            user_id="user_123",
            tenant_id="tenant_456",
        )

        with pytest.raises(JWTError, match="expired"):
            provider.verify_access_token(token)

    def test_future_token_valid(self):
        """Token with future expiration should be valid."""
        provider = JWTProvider(
            secret_key="test-secret",
            access_expire_minutes=60,  # 1 hour in future
        )
        token = provider.create_access_token(
            user_id="user_123",
            tenant_id="tenant_456",
        )

        # Should not raise
        payload = provider.verify_access_token(token)
        assert payload["sub"] == "user_123"


class TestTokenRotation:
    """Tests for refresh token rotation."""

    def test_rotate_generates_new_tokens(self):
        """Token rotation should generate new access and refresh tokens."""
        provider = JWTProvider(secret_key="test-secret")
        old_refresh = provider.create_refresh_token(user_id="user_123")

        new_access, new_refresh = provider.rotate_refresh_token(old_refresh)

        # New tokens should be different
        assert new_access != ""
        assert new_refresh != ""
        assert new_refresh != old_refresh

        # Verify new access token
        access_payload = jwt.decode(new_access, "test-secret", algorithms=["HS256"])
        assert access_payload["sub"] == "user_123"
        assert access_payload["type"] == "access"

        # Verify new refresh token
        refresh_payload = jwt.decode(new_refresh, "test-secret", algorithms=["HS256"])
        assert refresh_payload["sub"] == "user_123"
        assert refresh_payload["type"] == "refresh"

    def test_rotate_invalid_token_raises_error(self):
        """Rotating invalid refresh token should raise error."""
        provider = JWTProvider(secret_key="test-secret")

        with pytest.raises(JWTError):
            provider.rotate_refresh_token("invalid.token")


class TestSingleton:
    """Tests for singleton pattern."""

    def test_get_provider_returns_same_instance(self):
        """get_jwt_provider should return singleton instance."""
        provider1 = get_jwt_provider()
        provider2 = get_jwt_provider()
        assert provider1 is provider2
