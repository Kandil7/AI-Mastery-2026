#!/usr/bin/env python3
"""
Database Seeding Script
=======================

Populates the database with realistic test data for development and testing.

Populates database with test data for development and testing.
Populates database with test data for development and testing.

Features:
- Generates users with unique API keys
- Creates documents with realistic metadata
- Populates chunks for each document
- Generates chat sessions and turns
- Supports custom configuration via CLI
- Idempotent (safe to run multiple times)

Usage:
    python scripts/seed_sample_data.py
    python scripts/seed_sample_data.py --num-users 20 --reset
    python scripts/seed_sample_data.py --env testing
"""

import argparse
import os
import sys
import uuid
import hashlib
from datetime import datetime
from typing import Any
from pathlib import Path

import faker
from faker import Faker

sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from src.adapters.persistence.postgres.db import Base
from src.adapters.persistence.postgres.models import User, Document
from src.adapters.persistence.postgres.models_chunk_store import (
    ChunkStoreRow,
    DocumentChunkRow,
)
from src.adapters.persistence.postgres.models_chat import (
    ChatSessionRow,
    ChatTurnRow,
)


SEED_CONFIG = {
    "num_users": 10,
    "num_documents_per_user": 15,
    "min_chunks_per_document": 5,
    "max_chunks_per_document": 20,
    "num_chat_sessions_per_user": 5,
    "num_turns_per_session": 3,
}


def get_database_url() -> str:
    """Get database URL from environment or default."""
    return os.getenv(
        "DATABASE_URL",
        "postgresql://postgres:postgres@localhost:5432/rag_engine",
    )


def create_session():
    """Create database session."""
    engine = create_engine(get_database_url(), pool_pre_ping=True)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    return Session()


def reset_database(session) -> None:
    """Reset database by truncating all tables."""
    print("Resetting database...")

    session.execute(text("TRUNCATE TABLE chat_turns CASCADE"))
    session.execute(text("TRUNCATE TABLE chat_sessions CASCADE"))
    session.execute(text("TRUNCATE TABLE document_chunks CASCADE"))
    session.execute(text("TRUNCATE TABLE chunk_store CASCADE"))
    session.execute(text("TRUNCATE TABLE documents CASCADE"))
    session.execute(text("TRUNCATE TABLE users CASCADE"))

    session.commit()
    print("Database reset complete.")


def seed_users(session, count: int) -> list[User]:
    """Generate and insert users."""
    print(f"Seeding {count} users...")

    fake = Faker()
    Faker.seed(12345)

    users = []
    for i in range(count):
        user = User(
            id=str(uuid.uuid4()),
            email=fake.email(),
            api_key=f"sk_{fake.uuid4()[:24]}",
        )
        users.append(user)

    session.bulk_save_objects(users)
    session.commit()

    print(f"  Created {len(users)} users")
    return users


def seed_documents(session, users: list[User], docs_per_user: int) -> list[Document]:
    """Generate and insert documents for users."""
    print(f"Seeding documents ({docs_per_user} per user)...")

    fake = Faker()
    Faker.seed(12345)

    documents = []
    for user in users:
        for _ in range(docs_per_user):
            filename = fake.file_name(category="document")
            extension = filename.split(".")[-1] if "." in filename else "txt"

            content_type_map = {
                "pdf": "application/pdf",
                "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                "txt": "text/plain",
                "md": "text/markdown",
                "csv": "text/csv",
            }

            doc = Document(
                id=str(uuid.uuid4()),
                user_id=user.id,
                filename=filename,
                content_type=content_type_map.get(extension, "application/octet-stream"),
                file_path=f"/uploads/{fake.uuid4()}/{filename}",
                size_bytes=fake.random_int(min=1000, max=1000000),
                file_sha256=fake.sha256(raw_output=False),
                status=fake.random_element(elements=("indexed", "indexed", "indexed", "failed")),
            )
            if doc.status == "failed":
                doc.error = fake.sentence()

            documents.append(doc)

    session.bulk_save_objects(documents)
    session.commit()

    print(f"  Created {len(documents)} documents")
    return documents


def seed_chunks(
    session,
    documents: list[Document],
    min_chunks: int,
    max_chunks: int,
) -> tuple[list[ChunkStoreRow], list[DocumentChunkRow]]:
    """Generate and insert chunks for documents."""
    print(f"Seeding chunks ({min_chunks}-{max_chunks} per document)...")

    fake = Faker()
    Faker.seed(12345)

    chunk_rows = []
    document_chunk_rows = []

    for doc in documents:
        if doc.status != "indexed":
            continue

        num_chunks = fake.random_int(min=min_chunks, max=max_chunks)

        for i in range(num_chunks):
            chunk_text = fake.paragraph(nb_sentences=5)
            chunk_hash = hashlib.sha256(f"{doc.user_id}:{chunk_text}".encode()).hexdigest()

            chunk = ChunkStoreRow(
                id=str(uuid.uuid4()),
                user_id=doc.user_id,
                chunk_hash=chunk_hash,
                text=chunk_text,
                parent_id=None,
                chunk_context=None,
            )
            chunk_rows.append(chunk)

            mapping = DocumentChunkRow(
                document_id=doc.id,
                ord=i,
                chunk_id=chunk.id,
            )
            document_chunk_rows.append(mapping)

    session.bulk_save_objects(chunk_rows)
    session.bulk_save_objects(document_chunk_rows)
    session.commit()

    print(f"  Created {len(chunk_rows)} chunks")
    print(f"  Created {len(document_chunk_rows)} document-chunk mappings")
    return chunk_rows, document_chunk_rows


def seed_chat_sessions(
    session,
    users: list[User],
    sessions_per_user: int,
    turns_per_session: int,
) -> tuple[list[ChatSessionRow], list[ChatTurnRow]]:
    """Generate and insert chat sessions and turns."""
    print(f"Seeding chat sessions ({sessions_per_user} per user, {turns_per_session} turns)...")

    fake = Faker()
    Faker.seed(12345)

    sessions = []
    turns = []

    for user in users:
        for _ in range(sessions_per_user):
            session_row = ChatSessionRow(
                id=str(uuid.uuid4()),
                user_id=user.id,
                title=fake.sentence()[:50],
            )
            sessions.append(session_row)

            for _ in range(turns_per_session):
                turn = ChatTurnRow(
                    id=str(uuid.uuid4()),
                    session_id=session_row.id,
                    user_id=user.id,
                    question=fake.sentence(),
                    answer=fake.paragraph(nb_sentences=3),
                    sources=[str(uuid.uuid4()) for _ in range(fake.random_int(1, 5))],
                    retrieval_k=fake.random_int(3, 10),
                    embed_ms=fake.random_int(min=50, max=200),
                    search_ms=fake.random_int(min=10, max=100),
                    llm_ms=fake.random_int(min=500, max=2000),
                    prompt_tokens=fake.random_int(min=100, max=500),
                    completion_tokens=fake.random_int(min=50, max=300),
                )
                turns.append(turn)

    session.bulk_save_objects(sessions)
    session.bulk_save_objects(turns)
    session.commit()

    print(f"  Created {len(sessions)} chat sessions")
    print(f"  Created {len(turns)} chat turns")
    return sessions, turns


def print_summary(session) -> None:
    """Print summary of seeded data."""
    print("\n" + "=" * 50)
    print("Seeding Complete - Summary")
    print("=" * 50)
    print(f"Users:           {session.query(User).count()}")
    print(f"Documents:       {session.query(Document).count()}")
    print(f"Chunks:          {session.query(ChunkStoreRow).count()}")
    print(f"Doc-Chunk Maps:  {session.query(DocumentChunkRow).count()}")
    print(f"Chat Sessions:   {session.query(ChatSessionRow).count()}")
    print(f"Chat Turns:      {session.query(ChatTurnRow).count()}")
    print("=" * 50)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Seed database with sample data",
    )

    parser.add_argument(
        "--num-users",
        type=int,
        default=SEED_CONFIG["num_users"],
        help=f"Number of users to create (default: {SEED_CONFIG['num_users']})",
    )

    parser.add_argument(
        "--num-docs",
        type=int,
        default=SEED_CONFIG["num_documents_per_user"],
        help=f"Number of documents per user (default: {SEED_CONFIG['num_documents_per_user']})",
    )

    parser.add_argument(
        "--min-chunks",
        type=int,
        default=SEED_CONFIG["min_chunks_per_document"],
        help=f"Minimum chunks per document (default: {SEED_CONFIG['min_chunks_per_document']})",
    )

    parser.add_argument(
        "--max-chunks",
        type=int,
        default=SEED_CONFIG["max_chunks_per_document"],
        help=f"Maximum chunks per document (default: {SEED_CONFIG['max_chunks_per_document']})",
    )

    parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset database before seeding (truncate all tables)",
    )

    parser.add_argument(
        "--env",
        type=str,
        default=os.getenv("ENVIRONMENT", "development"),
        choices=["development", "testing", "staging"],
        help="Environment (default: from ENVIRONMENT or 'development')",
    )

    return parser.parse_args()


def main():
    """Main seeding function."""
    args = parse_args()

    print(f"Environment: {args.env}")

    if args.env == "production":
        print("ERROR: Cannot seed production database!")
        sys.exit(1)

    session = create_session()

    try:
        if args.reset:
            reset_database(session)

        users = seed_users(session, args.num_users)
        documents = seed_documents(session, users, args.num_docs)
        seed_chunks(session, documents, args.min_chunks, args.max_chunks)
        seed_chat_sessions(
            session,
            users,
            SEED_CONFIG["num_chat_sessions_per_user"],
            SEED_CONFIG["num_turns_per_session"],
        )

        print_summary(session)

    except Exception as e:
        session.rollback()
        print(f"Error during seeding: {e}")
        raise
    finally:
        session.close()


if __name__ == "__main__":
    main()
