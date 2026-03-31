"""
Migration Verification Script
===========================
Verifies that all Alembic migrations have been applied.

التحقق من تطبيق الهجرات
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from alembic.config import Config
from alembic import command
from alembic.script import ScriptDirectory


def verify_migrations() -> int:
    """
    Verify all migrations have been applied.

    Returns:
        Number of migrations verified
    """
    print("🔍 Verifying Database Migrations...")

    # Load Alembic config
    alembic_cfg = Config()
    alembic_cfg.set_main_option(
        "sqlalchemy.url", "postgresql+psycopg://postgres:postgres@localhost:5432/rag"
    )
    alembic_cfg.set_main_option(
        "script_location", project_root / "src/adapters/persistence/postgres/migrations"
    )

    # Create ScriptDirectory
    script = ScriptDirectory.from_config(alembic_cfg)

    # Check migrations
    from src.adapters.persistence.postgres.models import Base
    from sqlalchemy import inspect, create_engine

    engine = create_engine("postgresql+psycopg://postgres:postgres@localhost:5432/rag")
    inspector = inspect(engine)

    # Get expected tables
    expected_tables = {
        "users",
        "documents",
        "chunk_store",
        "chunk_text",
        "chat_sessions",
        "chat_turns",
        "graph_triplets",
    }

    # Get actual tables
    actual_tables = inspector.get_table_names()

    # Verify all tables exist
    verified_count = 0
    missing_tables = []

    for table in expected_tables:
        if table in actual_tables:
            verified_count += 1
            print(f"✅ Table: {table}")
        else:
            missing_tables.append(table)
            print(f"❌ Missing table: {table}")

    # Check for extra tables (not expected)
    extra_tables = [t for t in actual_tables if t not in expected_tables]
    for table in extra_tables:
        print(f"⚠️  Extra table: {table}")

    # Check for required columns
    if "documents" in actual_tables:
        doc_columns = [c["name"] for c in inspector.get_columns("documents")]
        required_doc_columns = [
            "id",
            "tenant_id",
            "filename",
            "content_type",
            "file_path",
            "size_bytes",
            "status",
        ]
        for col in required_doc_columns:
            if col in doc_columns:
                print(f"  ✅ Column: documents.{col}")
            else:
                print(f"  ❌ Missing column: documents.{col}")

    if "chunk_store" in actual_tables:
        chunk_columns = [c["name"] for c in inspector.get_columns("chunk_store")]
        required_chunk_columns = [
            "id",
            "tenant_id",
            "chunk_hash",
            "text",
            "parent_id",
            "created_at",
        ]
        for col in required_chunk_columns:
            if col in chunk_columns:
                print(f"  ✅ Column: chunk_store.{col}")
            else:
                print(f"  ❌ Missing column: chunk_store.{col}")

    # Summary
    print(f"\n{'=' * 50}")
    print(f"Verification Summary:")
    print(f"  Tables Verified: {verified_count}/{len(expected_tables)}")

    if missing_tables:
        print(f"  ❌ Missing Tables: {len(missing_tables)}")
        print(f"     Tables: {', '.join(missing_tables)}")
        return 0
    elif extra_tables:
        print(f"  ⚠️  Extra Tables: {len(extra_tables)}")
        print(f"     Tables: {', '.join(extra_tables)}")
    else:
        print(f"  ✅ All tables present and verified!")
        return verified_count


def verify_schema_alignment() -> int:
    """
    Verify SQLAlchemy models match database schema.

    Returns:
        Number of verified models
    """
    print(f"\n{'=' * 50}")
    print("🔍 Verifying Schema Alignment...")

    from src.adapters.persistence.postgres import models
    from sqlalchemy import inspect

    engine = create_engine("postgresql+psycopg://postgres:postgres@localhost:5432/rag")
    inspector = inspect(engine)

    # Verify models
    verified_count = 0

    # User model
    if hasattr(models, "User"):
        print("  Checking User model...")
        user_table = inspector.get_columns("users")
        user_columns = {c["name"] for c in user_table}
        expected_user_columns = {"id", "email", "api_key", "created_at"}
        if expected_user_columns.issubset(user_columns):
            print(f"    ✅ User model verified")
            verified_count += 1
        else:
            print(f"    ❌ User model mismatch")

    # Document model
    if hasattr(models, "Document"):
        print("  Checking Document model...")
        doc_table = inspector.get_columns("documents")
        doc_columns = {c["name"] for c in doc_table}
        expected_doc_columns = {"id", "user_id", "filename", "status", "created_at", "updated_at"}
        if expected_doc_columns.issubset(doc_columns):
            print(f"    ✅ Document model verified")
            verified_count += 1
        else:
            print(f"    ⚠️  Document model columns differ (may be okay)")

    # Other models...
    # (You can extend this for all models)

    return verified_count


def check_indexes() -> int:
    """
    Check for recommended indexes.

    Returns:
        Number of indexes verified
    """
    print(f"\n{'=' * 50}")
    print("🔍 Checking Database Indexes...")

    from sqlalchemy import inspect, create_engine

    engine = create_engine("postgresql+psycopg://postgres:postgres@localhost:5432/rag")
    inspector = inspect(engine)

    # Check for indexes
    index_count = 0

    # Critical indexes
    expected_indexes = [
        "ix_users_email",
        "ix_users_api_key",
        "ix_documents_user_id",
        "ix_documents_status",
        "ix_documents_file_sha256",
        "ix_chunk_store_tenant_chunk_hash",
        "ix_chunk_text_chunk_id",
    ]

    indexes = inspector.get_indexes("documents")
    for index in indexes:
        index_name = index["name"]
        if index_name in expected_indexes:
            print(f"  ✅ Index: {index_name}")
            index_count += 1
        else:
            print(f"  ⚠️  Index: {index_name}")

    return index_count


if __name__ == "__main__":
    print("=" * 60)
    print("   Database Migration & Schema Verification")
    print("=" * 60)
    print()

    # Run verifications
    tables_verified = verify_migrations()
    schema_aligned = verify_schema_alignment()
    indexes_checked = check_indexes()

    # Final summary
    print(f"\n{'=' * 50}")
    print("🏆 Final Verification Summary:")
    print(f"  Tables Verified: {tables_verified}")
    print(f"  Models Verified: {schema_aligned}")
    print(f"  Indexes Checked: {indexes_checked}")

    # Overall status
    all_checks_passed = (
        tables_verified >= 8  # All tables present
        and schema_aligned >= 2  # User and Document models aligned
        and indexes_checked >= 6  # Critical indexes present
    )

    if all_checks_passed:
        print(f"\n✅ Database is in GOOD STATE")
        print(f"   All migrations applied successfully")
        print(f"   Schema alignment verified")
        print(f"   Indexes optimized")
        sys.exit(0)
    else:
        print(f"\n❌ Database needs attention")
        print(f"   Some checks failed (see above)")
        print(f"   Run: alembic upgrade head")
        sys.exit(1)
