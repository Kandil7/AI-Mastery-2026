"""
SQLAlchemy Database Setup
==========================
Database engine, session factory, and base model.

إعداد قاعدة البيانات SQLAlchemy
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase

from src.core.config import settings


class Base(DeclarativeBase):
    """
    Base class for all ORM models.
    
    الفئة الأساسية لجميع نماذج ORM
    """
    pass


# Create database engine
engine = create_engine(
    settings.database_url,
    pool_pre_ping=True,
    pool_size=settings.db_pool_size,
    max_overflow=settings.db_max_overflow,
)

# Session factory
SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
)


def get_db():
    """
    Dependency for getting database sessions.
    
    تبعية للحصول على جلسات قاعدة البيانات
    
    Usage:
        with SessionLocal() as db:
            db.execute(...)
            db.commit()
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_engine():
    """Return the configured SQLAlchemy engine."""
    return engine
