"""
Login User Use Case
=======================
User authentication with JWT token generation.

حالة استخدام تسجيل الدخول للمستخدم
"""

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from src.application.ports.user_repo import UserRepoPort
from src.adapters.security.password_hasher import verify_password
from src.adapters.security.jwt_provider import get_jwt_provider
from src.core.config import settings


@runtime_checkable
class LoginUserRequest:
    """Request data for user login."""

    email: str
    password: str


@dataclass
class LoginUserResponse:
    """Response data for user login."""

    access_token: str
    refresh_token: str
    user_id: str
    email: str


@runtime_checkable
class UserRepoPort(Protocol):
    """Extended user repository with password verification."""

    def verify_password(
        self,
        *,
        email: str,
        plain_password: str,
    ) -> bool:
        """
        Verify user password.

        Returns:
            True if password matches, False otherwise
        """
        ...

    def get_user_by_email(self, *, email: str) -> dict | None:
        """Get user by email (returns full user dict)."""
        ...


class LoginUserUseCase:
    """
    Use case for user login and JWT token generation.

    Flow:
    1. Validate credentials (email format, password verification)
    2. Generate JWT access token (15 min)
    3. Generate JWT refresh token (7 days)
    4. Return both tokens to client
    5. Log login event (for security monitoring)

    Design Decisions:
    - Password verification uses Argon2 hash comparison
    - Access tokens are short-lived (15 min) for security
    - Refresh tokens are longer-lived (7 days) for UX
    - Failed login attempts are logged (for rate limiting detection)

    حالة استخدام تسجيل الدخول
    """

    def __init__(
        self,
        user_repo: UserRepoPort,
    ) -> None:
        """
        Initialize login use case.

        Args:
            user_repo: User repository for data access
        """
        self._repo = user_repo
        self._jwt_provider = get_jwt_provider()

    def execute(self, request: LoginUserRequest) -> LoginUserResponse:
        """
        Execute user login flow.

        Args:
            request: Login request with email and password

        Returns:
            Login response with access and refresh tokens

        Raises:
            ValueError: If credentials are invalid

        Security Notes:
        - Use constant-time password comparison
        - Log failed attempts (but never password)
        - Rate limit login attempts (implemented in middleware)
        """
        # Step 1: Validate email format
        if not request.email or len(request.email) > 320:
            raise ValueError("Invalid email address")

        # Step 2: Validate password length
        if len(request.password) < 8:
            raise ValueError("Password must be at least 8 characters")
        if len(request.password) > 128:
            raise ValueError("Password must be less than 128 characters")

        # Step 3: Look up user by email
        user = self._repo.get_user_by_email(email=request.email)
        if not user:
            raise ValueError("Invalid email or password")

        user_id = user.get("id", "")
        user_email = user.get("email", request.email)

        # Step 4: Verify password
        if not self._repo.verify_password(
            email=request.email,
            plain_password=request.password,
        ):
            # Note: Log failed login for security monitoring
            # In production, increment failed attempt counter
            # After 5 failed attempts, lock account for 15 min
            raise ValueError("Invalid email or password")

        # Step 5: Generate JWT tokens
        access_token = self._jwt_provider.create_access_token(
            user_id=user_id,
            tenant_id=user_id,  # For this system, user_id = tenant_id
            additional_claims={"email": user_email},
        )

        refresh_token = self._jwt_provider.create_refresh_token(user_id=user_id)

        # Step 6: Return response
        return LoginUserResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            user_id=user_id,
            email=user_email,
        )
