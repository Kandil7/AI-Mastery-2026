"""
Password Hashing Module
======================
Industry-standard password hashing with Argon2.

نمج تجزئة كلمات المرور باستخدام Argon2
"""

from typing import Final
import argon2
from argon2 import PasswordHasher

# ============================================================================
# Configuration
# ============================================================================

# Argon2id is recommended for password hashing
# See: https://password-hashing.net/argon2
DEFAULT_TIME_COST: Final = 3  # Number of iterations
DEFAULT_MEMORY_COST: Final = 65536  # Memory in KiB (64MB)
DEFAULT_PARALLELISM: Final = 4  # Number of threads
DEFAULT_HASH_LEN: Final = 32  # Output hash length in bytes
DEFAULT_SALT_LEN: Final = 16  # Salt length in bytes

# ============================================================================
# Password Hasher
# ============================================================================


class Argon2PasswordHasher:
    """
    Argon2id password hasher for secure password storage.

    Argon2id is the current recommendation for password hashing because:
    - Resistant to GPU/ASIC attacks
    - Tunable memory cost (prevents parallel attacks)
    - Side-channel resistant

    Uses constant-time comparison to prevent timing attacks.

    مزود تجزئة كلمات المرور Argon2id
    """

    def __init__(
        self,
        time_cost: int = DEFAULT_TIME_COST,
        memory_cost: int = DEFAULT_MEMORY_COST,
        parallelism: int = DEFAULT_PARALLELISM,
        hash_len: int = DEFAULT_HASH_LEN,
        salt_len: int = DEFAULT_SALT_LEN,
    ) -> None:
        """
        Initialize hasher with security parameters.

        Args:
            time_cost: Number of iterations (higher = slower but more secure)
            memory_cost: Memory in KiB (higher = more GPU resistance)
            parallelism: Number of parallel threads (typically 1-8)
            hash_len: Output hash length in bytes
            salt_len: Random salt length in bytes

        Security Guidelines:
            - time_cost >= 2 (recommended: 3-4)
            - memory_cost >= 16384 KiB (16MB minimum)
            - parallelism <= 8 (to prevent DoS)
        """
        self._hasher = PasswordHasher(
            time_cost=time_cost,
            memory_cost=memory_cost,
            parallelism=parallelism,
            hash_len=hash_len,
            salt_len=salt_len,
            type=argon2.Type.ID,  # Argon2id for password hashing
        )

    def hash(self, password: str) -> str:
        """
        Hash a password with Argon2id.

        Returns encoded hash string that includes:
        - Algorithm identifier
        - Parameters (time_cost, memory_cost, etc.)
        - Salt
        - Hash output

        Format: $argon2id$v=19$m=65536,t=3,p=4$salt$hash

        Args:
            password: Plain text password to hash

        Returns:
            Encoded hash string suitable for storage

        Raises:
            ValueError: If password is empty or too short
        """
        if not password:
            raise ValueError("Password cannot be empty")

        if len(password) < 8:
            raise ValueError("Password must be at least 8 characters")

        # Argon2 automatically generates random salt
        return self._hasher.hash(password)

    def verify(self, password: str, hashed: str) -> bool:
        """
        Verify a password against a stored hash.

        Uses constant-time comparison to prevent timing attacks.
        Timing attacks try to infer password correctness
        by measuring response time variations.

        Args:
            password: Plain text password to verify
            hashed: Stored hash string to compare against

        Returns:
            True if password matches hash, False otherwise
        """
        try:
            return self._hasher.verify(hashed, password)
        except argon2.exceptions.VerifyMismatchError:
            return False

    def needs_rehash(self, hashed: str) -> bool:
        """
        Check if stored hash needs re-hashing.

        This allows upgrading security parameters when:
        - Default parameters change
        - New Argon2 recommendations emerge

        Args:
            hashed: Stored hash string

        Returns:
            True if hash parameters differ from current defaults
        """
        try:
            # Argon2 can parse the hash and check parameters
            return self._hasher.check_needs_rehash(hashed)
        except Exception:
            # If we can't parse it, better to rehash
            return True


# ============================================================================
# Singleton Instance
# ============================================================================

# Global hasher instance with production-safe defaults
_hasher: Argon2PasswordHasher | None = None


def get_hasher() -> Argon2PasswordHasher:
    """Get global password hasher instance."""
    global _hasher
    if _hasher is None:
        _hasher = Argon2PasswordHasher()
    return _hasher


def hash_password(password: str) -> str:
    """
    Convenience function to hash a password.

    Hash a password with production-safe defaults.

    Example:
        >>> hash_password("MySecurePassword123!")
        '$argon2id$v=19$m=65536,t=3,p=4$...'
    """
    return get_hasher().hash(password)


def verify_password(password: str, hashed: str) -> bool:
    """
    Convenience function to verify a password.

    Example:
        >>> verify_password("wrong", hash)
        False
        >>> verify_password("correct", hash)
        True
    """
    return get_hasher().verify(password, hashed)
