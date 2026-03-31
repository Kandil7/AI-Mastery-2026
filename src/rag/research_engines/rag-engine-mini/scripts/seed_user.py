"""
Seed Demo User Script
======================
Creates a demo user with API key for testing.

سكربت إنشاء مستخدم تجريبي
"""

import hashlib
import secrets
import uuid
from datetime import datetime

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.adapters.persistence.postgres.db import Base
from src.adapters.persistence.postgres.models import User
from src.core.config import settings


def create_demo_user():
    """Create a demo user in the database."""

    # Create database engine
    engine = create_engine(settings.database_url)

    # Create tables if they don't exist
    Base.metadata.create_all(engine)

    # Create session
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = SessionLocal()

    try:
        # Check if demo user already exists
        existing_user = db.query(User).filter(User.email == "demo@example.com").first()

        if existing_user:
            print(f"Demo user already exists with API key: {existing_user.api_key}")
            return existing_user

        # Generate secure API key
        api_key = "demo_" + secrets.token_urlsafe(32)

        # Create new user
        user_id = str(uuid.uuid4())
        user = User(
            id=user_id,
            email="demo@example.com",
            api_key=api_key,
            created_at=datetime.utcnow()
        )

        # Add to database
        db.add(user)
        db.commit()
        db.refresh(user)

        print("=" * 60)
        print("🌱 Demo User Created Successfully")
        print("=" * 60)
        print(f"  User ID:  {user.id}")
        print(f"  Email:    {user.email}")
        print(f"  API Key:  {user.api_key}")
        print("=" * 60)
        print()
        print("Use the API key in X-API-KEY header:")
        print(f'  curl -H "X-API-KEY: {user.api_key}" http://localhost:8000/health')
        print()

        return user

    finally:
        db.close()


def main() -> None:
    """Create demo user with API key."""
    create_demo_user()


if __name__ == "__main__":
    main()
