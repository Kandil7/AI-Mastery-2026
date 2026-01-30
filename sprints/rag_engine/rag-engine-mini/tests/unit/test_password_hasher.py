"""
Password Hashing Unit Tests
===========================
Tests for Argon2 password hasher.
"""

import pytest
from src.adapters.security.password_hasher import (
    Argon2PasswordHasher,
    hash_password,
    verify_password,
)


class TestArgon2PasswordHasher:
    """Tests for Argon2 password hasher."""

    def test_initialization(self):
        """Hasher should initialize with defaults."""
        hasher = Argon2PasswordHasher()
        assert hasher is not None

    def test_hash_returns_string(self):
        """Hash should return string."""
        hasher = Argon2PasswordHasher()
        result = hasher.hash("TestPassword123!")
        assert isinstance(result, str)
        assert result.startswith("$argon2id$")

    def test_hash_requires_min_length(self):
        """Hash should require minimum password length."""
        hasher = Argon2PasswordHasher()
        with pytest.raises(ValueError):
            hasher.hash("short")

    def test_hash_rejects_empty_password(self):
        """Hash should reject empty password."""
        hasher = Argon2PasswordHasher()
        with pytest.raises(ValueError):
            hasher.hash("")

    def test_verify_correct_password(self):
        """Verify should return True for correct password."""
        hasher = Argon2PasswordHasher()
        hashed = hasher.hash("CorrectPassword123!")
        assert hasher.verify("CorrectPassword123!", hashed) is True

    def test_verify_incorrect_password(self):
        """Verify should return False for incorrect password."""
        hasher = Argon2PasswordHasher()
        hashed = hasher.hash("CorrectPassword123!")
        assert hasher.verify("WrongPassword123!", hashed) is False

    def test_verify_with_wrong_hash_format(self):
        """Verify should handle invalid hash format."""
        hasher = Argon2PasswordHasher()
        assert hasher.verify("password", "invalid_hash") is False

    def test_different_passwords_different_hashes(self):
        """Different passwords should produce different hashes."""
        hasher = Argon2PasswordHasher()
        hash1 = hasher.hash("Password1")
        hash2 = hasher.hash("Password2")
        assert hash1 != hash2

    def test_same_password_different_hashes(self):
        """Same password should produce different hashes (salt)."""
        hasher = Argon2PasswordHasher()
        hash1 = hasher.hash("SamePassword")
        hash2 = hasher.hash("SamePassword")
        assert hash1 != hash2

    def test_needs_rehash_with_different_params(self):
        """Should detect when parameters change."""
        hasher = Argon2PasswordHasher(time_cost=3)
        hashed = hasher.hash("TestPassword")

        # Create hasher with different params
        new_hasher = Argon2PasswordHasher(time_cost=4)
        assert new_hasher.needs_rehash(hashed) is True

    def test_needs_rehash_with_same_params(self):
        """Should not need rehash with same params."""
        hasher = Argon2PasswordHasher()
        hashed = hasher.hash("TestPassword")
        assert hasher.needs_rehash(hashed) is False


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_hash_password_function(self):
        """hash_password should work as convenience."""
        hashed = hash_password("TestPass123!")
        assert isinstance(hashed, str)
        assert hashed.startswith("$argon2id$")

    def test_verify_password_function(self):
        """verify_password should work as convenience."""
        hashed = hash_password("TestPass123!")
        assert verify_password("TestPass123!", hashed) is True
        assert verify_password("WrongPass123!", hashed) is False
