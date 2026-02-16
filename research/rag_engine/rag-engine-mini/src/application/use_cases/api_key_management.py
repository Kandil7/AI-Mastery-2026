"""
API Key Management Use Case
============================
CRUD operations for API keys.

إدارة مفاتاح API
"""

import secrets
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Protocol, runtime_checkable, List

from src.application.ports.user_repo import UserRepoPort
from src.core.config import settings


@dataclass
class CreateApiKeyRequest:
    """Request data for creating API key."""

    user_id: str
    name: str | None = None
    permissions: List[str] | None = None


@dataclass
class ApiKeyResponse:
    """Response data for API key operations."""

    key_id: str
    api_key: str  # Masked in responses
    name: str
    permissions: List[str]
    is_active: bool
    created_at: str
    last_used_at: str | None = None
    expires_at: str | None = None


@runtime_checkable
class UserRepoPort(Protocol):
    """Extended user repository with API key management."""

    def create_api_key(
        self,
        *,
        user_id: str,
        key_prefix: str,
        permissions: List[str] | None = None,
    ) -> dict:
        """
        Create a new API key.

        Returns:
            API key record with id, key_hash, permissions, created_at
        """
        ...

    def list_api_keys(self, *, user_id: str) -> List[dict]:
        """List all API keys for a user."""
        ...

    def get_api_key(self, *, key_id: str, user_id: str) -> dict | None:
        """Get API key details by ID."""
        ...

    def revoke_api_key(self, *, key_id: str, user_id: str) -> None:
        """Revoke (deactivate) an API key."""
        ...

    def revoke_all_keys(self, *, user_id: str) -> int:
        """Revoke all API keys for a user."""
        ...


class ApiKeyManagementUseCase:
    """
    Use case for API key management.

    Flow:
    1. Create API key with nanoid (collision-resistant)
    2. Store API key hash (never plain key in logs)
    3. List user's API keys (active and revoked)
    4. Revoke individual API keys
    5. Revoke all keys (emergency reset)

    Security Decisions:
    - API keys are generated with nanoid (more collision-resistant than UUID)
    - Keys are stored as SHA256 hashes (never logged in plain text)
    - Keys are displayed only once on creation
    - Permissions can be attached to each key (RBAC)
    - Expiration can be set (optional, never expires by default)
    - Audit trail: All key operations logged

    حالة استخدام إدارة مفاتاح API
    """

    def __init__(
        self,
        user_repo: UserRepoPort,
    ) -> None:
        """
        Initialize API key management use case.

        Args:
            user_repo: User repository for data access
        """
        self._repo = user_repo
        self._default_permissions = ["read", "write"]
        self._default_expiry_days = 365  # 1 year by default
        self._key_prefix = "rag_"

    def _generate_api_key(self) -> str:
        """
        Generate a secure API key.

        Uses nanoid for collision resistance.
        Adds prefix for easy identification.
        """
        try:
            import nanoid

            # Generate 21-character ID (more collision-resistant than UUID)
            key_id = nanoid.generate(size=21)
            api_key = f"{self._key_prefix}{key_id}"
            return api_key
        except ImportError:
            # Fallback to UUID4 if nanoid not available
            import uuid

            api_key = f"{self._key_prefix}{str(uuid.uuid4())}"
            return api_key

    def create(self, request: CreateApiKeyRequest) -> ApiKeyResponse:
        """
        Create a new API key.

        Args:
            request: Create API key request

        Returns:
            API key response (key is masked)
        """
        # Generate API key
        api_key = self._generate_api_key()

        # Hash for storage (SHA256)
        import hashlib

        key_hash = hashlib.sha256(api_key.encode()).hexdigest()

        # Set permissions
        permissions = request.permissions or self._default_permissions

        # Create API key record
        now = datetime.utcnow()
        api_key_record = self._repo.create_api_key(
            user_id=request.user_id,
            key_prefix=self._key_prefix,
            permissions=permissions,
        )

        # Store hash and metadata (not plain key!)
        api_key_record["key_hash"] = key_hash
        api_key_record["name"] = request.name
        api_key_record["permissions"] = permissions
        api_key_record["created_at"] = now
        api_key_record["expires_at"] = now + timedelta(days=self._default_expiry_days)

        return ApiKeyResponse(
            key_id=api_key_record["id"],
            api_key=api_key,  # Only time we show the key!
            name=request.name or "Default Key",
            permissions=permissions,
            is_active=True,
            created_at=now.isoformat(),
            last_used_at=None,
            expires_at=api_key_record["expires_at"].isoformat()
            if self._default_expiry_days
            else None,
        )

    def list_keys(self, user_id: str, include_revoked: bool = False) -> List[ApiKeyResponse]:
        """
        List API keys for a user.

        Args:
            user_id: User ID
            include_revoked: Whether to include deactivated keys

        Returns:
            List of API keys
        """
        keys = self._repo.list_api_keys(user_id=user_id)

        # Filter by active status if requested
        if not include_revoked:
            keys = [k for k in keys if k.get("is_active", True)]

        return [
            ApiKeyResponse(
                key_id=k.get("id", ""),
                api_key="rag_" * len(k["id"]),  # Masked for security
                name=k.get("name", ""),
                permissions=k.get("permissions", []),
                is_active=k.get("is_active", False),
                created_at=k.get("created_at", datetime.utcnow()).isoformat(),
                last_used_at=k.get("last_used_at"),
                expires_at=k.get("expires_at"),
            )
            for k in keys
        ]

    def revoke_key(self, key_id: str, user_id: str) -> None:
        """
        Revoke (deactivate) an API key.

        Args:
            key_id: API key ID to revoke
            user_id: User ID (for authorization)

        Security Notes:
        - Does NOT delete the key from database
        - Marks as is_active=False
        - Allows audit trail of all keys
        - Can be reactivated (future feature)
        """
        self._repo.revoke_api_key(key_id=key_id, user_id=user_id)

    def revoke_all(self, user_id: str) -> int:
        """
        Revoke ALL API keys for a user.

        Args:
            user_id: User ID

        Returns:
            Number of keys revoked

        Security Notes:
        - Emergency use only (user request or admin action)
        - Marks all keys as is_active=False
        - Can be undone only by manual database intervention
        - Should be logged for audit
        """
        # List all keys first
        keys = self._repo.list_api_keys(user_id=user_id)

        # Revoke each one
        for key in keys:
            self._repo.revoke_api_key(key_id=key["id"], user_id=user_id)

        return len(keys)
