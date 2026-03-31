"""
Application Ports - User Repository
=================================
Port for user data access.
"""

from typing import Sequence, Protocol, runtime_checkable
from src.domain.entities import TenantId


@runtime_checkable
class UserRepoPort(Protocol):
    """Repository for user data operations."""

    def create_user(
        self,
        *,
        email: str,
        hashed_password: str,
    ) -> str:
        """
        Create a new user.

        Args:
            email: User email address
            hashed_password: Argon2 hash of password

        Returns:
            User ID
        """
        ...

    def get_user_by_email(
        self,
        *,
        email: str,
    ) -> dict | None:
        """
        Get user by email.

        Args:
            email: User email address

        Returns:
            User dict with id, email, hashed_password, created_at
            or None if not found
        """
        ...

    def get_user_by_id(
        self,
        *,
        user_id: str,
    ) -> dict | None:
        """
        Get user by ID.

        Args:
            user_id: User identifier

        Returns:
            User dict or None if not found
        """
        ...

    def email_exists(
        self,
        *,
        email: str,
    ) -> bool:
        """
        Check if email already exists.

        Args:
            email: Email address to check

        Returns:
            True if email exists, False otherwise
        """
        ...

    def update_password(
        self,
        *,
        user_id: str,
        hashed_password: str,
    ) -> None:
        """
        Update user's password.

        Args:
            user_id: User identifier
            hashed_password: New password hash
        """
        ...
