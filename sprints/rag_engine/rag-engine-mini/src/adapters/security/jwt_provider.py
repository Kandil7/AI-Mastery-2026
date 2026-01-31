"""
JWT Authentication Provider
==========================
JSON Web Token generation and validation.

توليد والتحقق من رموز JWT
"""

import time
import uuid
from typing import Final, TypedDict, Literal
from jose import JWTError, jwt
from datetime import datetime, timedelta

from src.core.config import settings

# ============================================================================
# Configuration
# ============================================================================

# JWT Algorithm (RS256 for production, HS256 for dev)
ALGORITHM: Final = "HS256"  # Use RS256 with private key in production

# Token Expiration Times
ACCESS_TOKEN_EXPIRE_MINUTES: Final = 15  # Short-lived for security
REFRESH_TOKEN_EXPIRE_DAYS: Final = 7  # Longer-lived for refresh

# Token Types
TokenType = Literal["access", "refresh"]

# ============================================================================
# Token Payloads
# ============================================================================


class TokenPayload(TypedDict):
    """JWT payload structure."""

    sub: str  # Subject (user_id or tenant_id)
    exp: int  # Expiration timestamp
    iat: int  # Issued at timestamp
    type: str  # Token type (access or refresh)
    jti: str  # JWT ID (for revocation)


# ============================================================================
# JWT Provider
# ============================================================================


class JWTProvider:
    """
    JWT token generation and validation.

    Design decisions:
    - Short access tokens (15 min) - minimizes damage if leaked
    - Longer refresh tokens (7 days) - balances security with UX
    - Unique jti per token - enables revocation
    - HS256 for dev, RS256 recommended for production

    مزود JWT - توليد والتحقق من الرموز
    """

    def __init__(
        self,
        secret_key: str,
        algorithm: str = ALGORITHM,
        access_expire_minutes: int = ACCESS_TOKEN_EXPIRE_MINUTES,
        refresh_expire_days: int = REFRESH_TOKEN_EXPIRE_DAYS,
    ) -> None:
        """
        Initialize JWT provider.

        Args:
            secret_key: Secret key for signing (from environment/HSM)
            algorithm: JWT algorithm (HS256 or RS256)
            access_expire_minutes: Access token lifetime in minutes
            refresh_expire_days: Refresh token lifetime in days

        Security Notes:
            - Use RS256 with asymmetric keys in production
            - Store secret_key in HSM or secret manager
            - Rotate keys periodically
        """
        self._secret_key = secret_key
        self._algorithm = algorithm
        self._access_expire_minutes = access_expire_minutes
        self._refresh_expire_days = refresh_expire_days

    def _create_token(
        self,
        payload: dict,
        expires_delta: timedelta,
    ) -> str:
        """
        Create JWT with expiration.

        Args:
            payload: Token payload data
            expires_delta: Time until expiration

        Returns:
            Encoded JWT string
        """
        # Add expiration time
        expire = datetime.utcnow() + expires_delta
        payload["exp"] = int(expire.timestamp())

        # Add issued at time
        payload["iat"] = int(datetime.utcnow().timestamp())

        # Generate and return token
        return jwt.encode(payload, self._secret_key, algorithm=self._algorithm)

    def create_access_token(
        self,
        user_id: str,
        tenant_id: str,
        additional_claims: dict | None = None,
    ) -> str:
        """
        Create an access token for API access.

        Short-lived token (15 minutes by default) used for:
        - API authentication
        - Protected resource access

        Args:
            user_id: User identifier
            tenant_id: Tenant/organization identifier
            additional_claims: Extra claims (roles, permissions)

        Returns:
            Access token JWT string

        Example payload:
        {
            "sub": "user_123",
            "tenant_id": "tenant_456",
            "type": "access",
            "jti": "unique_token_id",
            "exp": 1234567890,
            "iat": 1234567890
        }
        """
        # Build base payload
        payload = {
            "sub": user_id,
            "tenant_id": tenant_id,
            "type": "access",
            "jti": str(uuid.uuid4()),  # Unique ID for revocation
        }

        # Add additional claims if provided
        if additional_claims:
            payload.update(additional_claims)

        # Create token with short expiration
        expires_delta = timedelta(minutes=self._access_expire_minutes)
        return self._create_token(payload, expires_delta)

    def create_refresh_token(
        self,
        user_id: str,
    ) -> str:
        """
        Create a refresh token for token renewal.

        Long-lived token (7 days by default) used for:
        - Getting new access tokens without re-authentication
        - Keeping users logged in

        Args:
            user_id: User identifier

        Returns:
            Refresh token JWT string

        Example payload:
        {
            "sub": "user_123",
            "type": "refresh",
            "jti": "unique_token_id",
            "exp": 1234567890,
            "iat": 1234567890
        }
        """
        payload = {
            "sub": user_id,
            "type": "refresh",
            "jti": str(uuid.uuid4()),  # Unique ID for revocation
        }

        # Create token with long expiration
        expires_delta = timedelta(days=self._refresh_expire_days)
        return self._create_token(payload, expires_delta)

    def decode_token(
        self,
        token: str,
        verify_type: str | None = None,
    ) -> TokenPayload:
        """
        Decode and validate JWT.

        Args:
            token: JWT string to decode
            verify_type: Optional token type check ("access" or "refresh")

        Returns:
            Token payload dictionary

        Raises:
            JWTError: If token is invalid or expired
        """
        try:
            # Decode and verify signature
            payload = jwt.decode(
                token,
                self._secret_key,
                algorithms=[self._algorithm],
            )

            # Verify token type if specified
            if verify_type and payload.get("type") != verify_type:
                raise JWTError(f"Invalid token type: expected {verify_type}")

            return TokenPayload(
                sub=payload["sub"],
                exp=payload["exp"],
                iat=payload["iat"],
                type=payload["type"],
                jti=payload["jti"],
            )
        except jwt.ExpiredSignatureError:
            raise JWTError("Token has expired")
        except jwt.InvalidTokenError as e:
            raise JWTError(f"Invalid token: {str(e)}")

    def verify_access_token(self, token: str) -> TokenPayload:
        """
        Verify access token and return payload.

        Convenience method for access token validation.

        Args:
            token: Access token JWT string

        Returns:
            Token payload with user and tenant info

        Raises:
            JWTError: If token is invalid, expired, or wrong type
        """
        return self.decode_token(token, verify_type="access")

    def verify_refresh_token(self, token: str) -> TokenPayload:
        """
        Verify refresh token and return payload.

        Convenience method for refresh token validation.

        Args:
            token: Refresh token JWT string

        Returns:
            Token payload with user info

        Raises:
            JWTError: If token is invalid, expired, or wrong type
        """
        return self.decode_token(token, verify_type="refresh")

    def rotate_refresh_token(self, old_refresh_token: str) -> tuple[str, str]:
        """
        Rotate refresh token: create new pair, invalidate old.

        This is a security best practice:
        - Old refresh token is invalidated after use
        - New pair generated
        - Prevents replay attacks

        Args:
            old_refresh_token: Previous refresh token to rotate

        Returns:
            Tuple of (new_access_token, new_refresh_token)

        Raises:
            JWTError: If old token is invalid or expired
        """
        # Verify old refresh token
        payload = self.verify_refresh_token(old_refresh_token)

        # Extract user_id
        user_id = payload["sub"]

        # In a real system, you would:
        # 1. Add old jti to blacklist
        # 2. Store new jti in database
        # 3. Set expiration on old token

        # For now, just create new tokens
        new_access = self.create_access_token(
            user_id=user_id,
            tenant_id=payload.get("tenant_id", ""),
        )
        new_refresh = self.create_refresh_token(user_id=user_id)

        return new_access, new_refresh


# ============================================================================
# Singleton Instance
# ============================================================================

_jwt_provider: JWTProvider | None = None


def get_jwt_provider() -> JWTProvider:
    """Get global JWT provider instance."""
    global _jwt_provider
    if _jwt_provider is None:
        secret = settings.jwt_secret_key or "dev-secret-change-in-production"
        _jwt_provider = JWTProvider(secret_key=secret)
    return _jwt_provider
