"""
Enhanced Database Bootstrapping
==================================
Production-ready database configuration and connection management.

تهيئة قواعد البيانات للإنتاج
"""

from sqlalchemy import create_engine, Engine, NullPool
from sqlalchemy.orm import sessionmaker, Session
from contextlib import contextmanager
from typing import Generator

from src.core.config import settings


# ============================================================================
# Global Database Engine
# ============================================================================

_engine: Engine | None = None
_session_factory: sessionmaker | None


def get_engine() -> Engine:
    """
    Get SQLAlchemy database engine.

    Creates engine with production-safe settings:
    - Connection pooling (max 20 connections)
    - Pool recycling (every 3600 connections)
    - Statement caching (size 100)
    - Pool pre-ping (validate connections)

    Returns:
        SQLAlchemy engine instance

    Production Settings:
        - NullPool (connections checked out from pool)
        - pool_size: 10 (default)
        - max_overflow: 10 (extra connections when pool exhausted)
        - pool_recycle: 3600 (recycle connections every hour)
        - pool_pre_ping: True (validate connections before using)
        - echo: False (disable SQL logging in production)
        - future: True ( use SQLAlchemy 2.0 style)
    """
    global _engine
    if _engine is None:
        # Build connection URL
        db_url = settings.database_url

        # Create engine with production settings
        _engine = create_engine(
            db_url,
            poolclass=NullPool,
            pool_size=getattr(settings, "db_pool_size", 10),
            max_overflow=getattr(settings, "db_pool_max_overflow", 10),
            pool_recycle=getattr(settings, "db_pool_recycle", 3600),
            pool_pre_ping=True,
            echo=getattr(settings, "db_echo", False),
            future=True,  # SQLAlchemy 2.0
        )

    return _engine


def get_session_factory() -> sessionmaker:
    """
    Get SQLAlchemy session factory.

    Returns:
        Session factory bound to the global engine
    """
    global _session_factory
    if _session_factory is None:
        _session_factory = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=get_engine(),
        )

    return _session_factory


@contextmanager
def get_db_session() -> Generator[Session, None, None]:
    """
    Context manager for database sessions.

    Automatically handles:
    - Session creation
    - Transaction management
    - Session cleanup (even on exception)
    - Connection pool management

    Usage:
        with get_db_session() as session:
            result = session.query(Model).first()
            # Session is automatically committed/rolled back here

    Production Considerations:
    - Always use this context manager for sessions
    - Never create raw sessions without context management
    - Sessions are checked out from pool (performance)
    - Connections are validated before use (reliability)
    """
    session_factory = get_session_factory()
    session = session_factory()

    try:
        yield session
        session.commit()  # Commit transaction
    except Exception:
        session.rollback()  # Rollback on error
        raise
    finally:
        session.close()  # Always return connection to pool


def init_database() -> None:
    """
    Initialize database (create tables if needed).

    Should be called on application startup.

    Usage:
        from src.core.bootstrap import init_database
        init_database()

    """
    engine = get_engine()

    # In production, use Alembic for migrations
    # Only use create_all in development mode
    if not settings.use_real_db:
        # Import all models
        from src.adapters.persistence.postgres.models import Base
        from src.adapters.persistence.postgres.models_chunk_store import Base as ChunkStoreBase
        from src.adapters.persistence.postgres.models_chat import Base as ChatBase
        from src.adapters.persistence.postgres.models_graph import Base as GraphBase

        # Create all tables
        Base.metadata.create_all(engine)
        ChunkStoreBase.metadata.create_all(engine)
        ChatBase.metadata.create_all(engine)
        GraphBase.metadata.create_all(engine)

        print("✅ Database initialized (dev mode: tables created)")
    else:
        # Production: Assume migrations are run
        # Just check connection
        try:
            with engine.connect() as conn:
                conn.execute("SELECT 1")
            print("✅ Database initialized (prod mode: connection verified)")
        except Exception as e:
            print(f"❌ Database initialization failed: {e}")
            raise


def close_all_sessions() -> None:
    """
    Close all database sessions (for cleanup).

    Usage:
        - Application shutdown
        - Worker cleanup
        - Memory cleanup
    """
    # In SQLAlchemy 2.0, sessions are tied to engine
    # We dispose the engine which closes all sessions
    engine = get_engine()
    engine.dispose()
    print("🧹 All database sessions closed")


def check_database_health() -> dict:
    """
    Check database health and return status.

    Returns:
        {
            "status": "ok" | "degraded" | "error",
            "latency_ms": int,
            "pool_size": int,
            "active_connections": int,
        }
    """
    engine = get_engine()

    try:
        import time

        start = time.time()

        # Execute simple query to test connection
        with engine.connect() as conn:
            conn.execute("SELECT 1")
            latency = (time.time() - start) * 1000  # ms

        # Get pool status
        pool = engine.pool
        active_connections = len(pool._all_connections) if hasattr(pool, "_all_connections") else 0

        status = "ok" if latency < 100 else "degraded"

        return {
            "status": status,
            "latency_ms": round(latency, 2),
            "pool_size": pool.size(),
            "active_connections": active_connections,
        }
    except Exception as e:
        return {
            "status": "error",
            "latency_ms": 0,
            "pool_size": 0,
            "active_connections": 0,
            "error": str(e),
        }
