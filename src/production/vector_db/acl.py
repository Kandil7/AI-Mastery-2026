"""
ACL-Based Filtering Module
==========================

Object-level access control for vector search (Weaviate pattern).
Prevents semantic data leakage by attaching ACLs as metadata
and filtering search results by user permissions.

Classes:
    ACLPermission: Access control permission levels
    ACLEntry: Access control list entry
    ACLFilter: Object-level access control for vector search

Author: AI-Mastery-2026
"""

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


class ACLPermission:
    """Access control permission levels."""

    READ = "read"
    WRITE = "write"
    ADMIN = "admin"
    NONE = "none"


@dataclass
class ACLEntry:
    """Access control list entry."""

    user_id: str
    permission: str
    granted_at: float = None
    expires_at: Optional[float] = None

    def __post_init__(self):
        if self.granted_at is None:
            self.granted_at = time.time()

    def is_valid(self) -> bool:
        """Check if ACL entry is still valid."""
        if self.expires_at is None:
            return True
        return time.time() < self.expires_at


class ACLFilter:
    """
    Object-level access control for vector search (Weaviate pattern).

    Prevents semantic data leakage by attaching ACLs as metadata
    and filtering search results by user permissions.

    Key Features:
    - Attach ACLs during indexing
    - Filter results before returning to user
    - Support for read/write/admin permissions
    - Time-based permission expiry

    Reference: Weaviate RBAC, Pinecone Enterprise
    """

    ACL_METADATA_KEY = "_acl"

    def __init__(self, default_permission: str = ACLPermission.NONE):
        """
        Initialize ACL filter.

        Args:
            default_permission: Permission for users not in ACL
        """
        self.default_permission = default_permission
        self._access_log: List[Dict[str, Any]] = []

    @staticmethod
    def create_acl_metadata(
        owner_id: str,
        read_users: Optional[List[str]] = None,
        write_users: Optional[List[str]] = None,
        public: bool = False,
    ) -> Dict[str, Any]:
        """
        Create ACL metadata to attach to a vector.

        Args:
            owner_id: ID of the document owner
            read_users: Users with read access
            write_users: Users with write access
            public: If True, everyone can read

        Returns:
            ACL metadata dict
        """
        acl = {
            "owner": owner_id,
            "public": public,
            "read": read_users or [],
            "write": write_users or [],
            "created_at": time.time(),
        }
        return {ACLFilter.ACL_METADATA_KEY: acl}

    def check_permission(
        self,
        user_id: str,
        metadata: Dict[str, Any],
        required_permission: str = ACLPermission.READ,
    ) -> bool:
        """
        Check if user has required permission.

        Args:
            user_id: User requesting access
            metadata: Vector metadata containing ACL
            required_permission: Permission level needed

        Returns:
            True if user has permission
        """
        acl = metadata.get(self.ACL_METADATA_KEY)

        if not acl:
            # No ACL defined - use default
            return self.default_permission != ACLPermission.NONE

        # Owner has all permissions
        if acl.get("owner") == user_id:
            return True

        # Public documents allow read
        if acl.get("public") and required_permission == ACLPermission.READ:
            return True

        # Check explicit permissions
        if required_permission == ACLPermission.READ:
            return user_id in acl.get("read", []) or user_id in acl.get("write", [])
        elif required_permission == ACLPermission.WRITE:
            return user_id in acl.get("write", [])
        elif required_permission == ACLPermission.ADMIN:
            return acl.get("owner") == user_id

        return False

    def filter_search_results(
        self,
        user_id: str,
        results: List[Tuple[str, float, Dict[str, Any]]],
        permission: str = ACLPermission.READ,
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Filter search results by user permissions.

        This is the key security function that prevents users
        from seeing vectors they don't have access to.

        Args:
            user_id: User requesting results
            results: Raw search results
            permission: Required permission level

        Returns:
            Filtered results the user can access
        """
        filtered = []

        for vector_id, score, metadata in results:
            if self.check_permission(user_id, metadata, permission):
                filtered.append((vector_id, score, metadata))
                self._log_access(user_id, vector_id, "granted", permission)
            else:
                self._log_access(user_id, vector_id, "denied", permission)

        return filtered

    def _log_access(
        self, user_id: str, vector_id: str, decision: str, permission: str
    ) -> None:
        """Log access attempt for audit."""
        self._access_log.append(
            {
                "timestamp": time.time(),
                "user_id": user_id,
                "vector_id": vector_id,
                "decision": decision,
                "permission_checked": permission,
            }
        )

        # Keep only last 10000 entries
        if len(self._access_log) > 10000:
            self._access_log = self._access_log[-10000:]

    def get_access_log(
        self,
        user_id: Optional[str] = None,
        decision: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Get access log with optional filters.

        Args:
            user_id: Filter by user
            decision: Filter by decision (granted/denied)
            limit: Max entries to return

        Returns:
            Matching log entries
        """
        entries = self._access_log

        if user_id:
            entries = [e for e in entries if e["user_id"] == user_id]
        if decision:
            entries = [e for e in entries if e["decision"] == decision]

        return entries[-limit:]

    def get_security_stats(self) -> Dict[str, Any]:
        """Get security statistics."""
        if not self._access_log:
            return {"total_checks": 0}

        total = len(self._access_log)
        granted = sum(1 for e in self._access_log if e["decision"] == "granted")
        denied = total - granted

        return {
            "total_checks": total,
            "granted": granted,
            "denied": denied,
            "denial_rate": denied / total if total > 0 else 0,
            "unique_users": len(set(e["user_id"] for e in self._access_log)),
            "unique_vectors": len(set(e["vector_id"] for e in self._access_log)),
        }
