"""
User Lookup Repository
=======================
PostgreSQL implementation for API key → user_id lookup.

مستودع بحث المستخدم
"""

from sqlalchemy import select

from src.adapters.persistence.postgres.db import SessionLocal
from src.adapters.persistence.postgres.models import User


class UserLookupRepo:
    """
    Repository for looking up user_id by API key.
    
    Used in authentication to convert API key to tenant_id.
    
    مستودع للبحث عن معرف المستخدم بمفتاح API
    """
    
    def get_user_id_by_api_key(self, api_key: str) -> str | None:
        """
        Look up user_id by API key.
        
        Args:
            api_key: The API key from request header
            
        Returns:
            user_id if found, None otherwise
        """
        with SessionLocal() as db:
            stmt = select(User.id).where(User.api_key == api_key)
            return db.execute(stmt).scalar_one_or_none()
    
    def get_user_by_email(self, email: str) -> User | None:
        """
        Get user by email.
        
        Args:
            email: User email
            
        Returns:
            User object if found
        """
        with SessionLocal() as db:
            stmt = select(User).where(User.email == email)
            return db.execute(stmt).scalar_one_or_none()
