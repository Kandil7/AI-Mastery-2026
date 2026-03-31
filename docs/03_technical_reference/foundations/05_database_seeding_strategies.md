# Database Seeding Strategies

## Table of Contents
1. [What is Database Seeding?](#what-is-database-seeding)
2. [Why Seeding Matters](#why-seeding-matters)
3. [Seeding Strategies](#seeding-strategies)
4. [Data Generation Approaches](#data-generation-approaches)
5. [Best Practices](#best-practices)
6. [Implementation in RAG Engine](#implementation-in-rag-engine)

---

## What is Database Seeding?

Database seeding is the process of **populating a database with initial data** for development, testing, or demonstration purposes. Unlike migrations that modify schema, seeding adds data to tables.

### Key Characteristics:
- **Deterministic**: Same seed data produces consistent results
- **Isolated**: Test data doesn't mix with production data
- **Realistic**: Data should mirror production patterns
- **Idempotent**: Can be run multiple times safely

### Arabic
عملية زراعة قاعدة البيانات هي ملء قاعدة البيانات بالبيانات الأولية لأغراض التطوير والاختبار والعرض. على عكس الترحيلات التي تعدل المخطط, تضيف الزراعة بيانات إلى الجداول.

---

## Why Seeding Matters

### 1. Development Productivity
```python
# Without seeding: Need to manually create users
user = User(email="test@example.com")
session.add(user)
session.commit()

# With seeding: Pre-populated test data available instantly
users = session.query(User).all()  # Returns 10+ test users
```

### 2. Testing Realism
- **Empty databases** hide edge cases
- **Real data patterns** reveal performance issues
- **Multi-tenant scenarios** need diverse data

### 3. API Testing
```bash
# Test document upload with pre-existing documents
curl -X POST http://api.example.com/documents \
  -H "Authorization: Bearer $API_KEY" \
  -F "file=@document.pdf"
```

### 4. UI/UX Validation
- Verify list pagination
- Test search with realistic content
- Validate filtering with varied data

### Arabic
1. **إنتاجية التطوير**: توفير وقت إنشاء بيانات الاختبار
2. **واقعية الاختبار**: الكشف عن مشاكل الأداء
3. **اختبار API**: التحقق من النقاط النهائية
4. **التحقق من UI/UX**: اختبار التصفح والبحث

---

## Seeding Strategies

### Strategy 1: Static Data (Hardcoded)
**Best for**: Reference data, lookups

```python
seed_data = {
    "users": [
        {"email": "admin@example.com", "api_key": "sk_admin_123"},
        {"email": "user1@example.com", "api_key": "sk_user1_456"},
    ]
}
```

**Pros**: Simple, predictable
**Cons**: Inflexible, limited variety

---

### Strategy 2: Programmatic Generation
**Best for**: Dynamic, realistic data

```python
from faker import Faker

fake = Faker()

for _ in range(10):
    user = User(
        email=fake.email(),
        api_key=f"sk_{fake.uuid4()[:16]}"
    )
    session.add(user)
```

**Pros**: Realistic, scalable
**Cons**: Non-deterministic (unless seeded)

---

### Strategy 3: Factory Pattern (Factory Boy)
**Best for**: Complex relationships

```python
import factory

class UserFactory(factory.Factory):
    class Meta:
        model = User

    email = factory.Faker('email')
    api_key = factory.Faker('uuid4')

# Create related entities
user = UserFactory()
documents = [DocumentFactory(user=user) for _ in range(5)]
```

**Pros**: Handles relationships, clean API
**Cons**: Additional dependency

---

### Strategy 4: Hybrid (Static + Programmatic)
**Best for**: Mixed data types

```python
# Static reference users
STATIC_USERS = [
    {"email": "admin@example.com", "role": "admin"},
]

# Programmatic test users
for _ in range(20):
    user = User(email=fake.email(), role="user")
```

**Used in**: RAG Engine Mini

---

## Data Generation Approaches

### 1. Faker Library
Python's `Faker` generates realistic fake data:

```python
from faker import Faker

fake = Faker()
Faker.seed(12345)  # For reproducibility

# Examples
print(fake.email())           # 'james.smith@example.com'
print(fake.name())            # 'Jennifer Davis'
print(fake.sentence())        # 'Perferendis voluptas maiores.'
print(fake.paragraph(nb_sentences=5))
print(fake.file_name())       # 'report.csv'
print(fake.mime_type())       # 'text/plain'
```

### 2. Content Generation for RAG
For RAG systems, need realistic document content:

```python
def generate_document_content(topic: str) -> str:
    """Generate realistic document text for testing"""
    content = f"# {topic}\n\n"
    content += fake.paragraph(nb_sentences=3) + "\n\n"
    content += "## Overview\n\n"
    content += fake.paragraph(nb_sentences=5) + "\n\n"
    content += "## Technical Details\n\n"
    content += fake.paragraph(nb_sentences=8) + "\n\n"
    content += "## Conclusion\n\n"
    content += fake.paragraph(nb_sentences=2)
    return content
```

### 3. Realistic Chunk Distribution
Documents should have realistic chunk counts:

```python
import numpy as np

chunk_counts = np.random.poisson(lam=8, size=100)
# Most documents: 5-12 chunks
# Some: 15-20 chunks
# Few: 1-3 chunks
```

### Arabic
1. **مكتبة Faker**: توليد بيانات واقعية
2. **توليد المحتوى لـ RAG**: نصوص واقعية للمستندات
3. **توزيع القطع**: أعداد واقعية للقطع لكل مستند

---

## Best Practices

### 1. Idempotency
```python
def seed_users():
    existing = session.query(User).count()
    if existing == 0:
        create_users()
    else:
        print(f"Skipping seeding: {existing} users exist")
```

### 2. Environment-Aware
```python
import os

def seed():
    env = os.getenv("ENVIRONMENT", "development")

    if env == "production":
        raise Exception("Never seed production!")

    if env == "testing":
        seed_minimal()
    else:
        seed_comprehensive()
```

### 3. Transaction Safety
```python
from sqlalchemy.exc import SQLAlchemyError

def seed():
    try:
        session.begin()
        create_users()
        create_documents()
        create_chunks()
        session.commit()
    except SQLAlchemyError:
        session.rollback()
        raise
```

### 4. Fast Reset
```python
def reset_database():
    """Quick reset for development"""
    session.execute(text("TRUNCATE users CASCADE"))
    session.execute(text("TRUNCATE documents CASCADE"))
    session.commit()
    seed()
```

### 5. Performance Optimization
```python
# Bulk insert instead of individual inserts
session.bulk_save_objects(users)
# Or use copy_from for very large datasets
```

### Arabic
1. **المعاودة (Idempotency)**: يمكن تشغيلها عدة مرات
2. **حساب البيئة**: عدم زراعة الإنتاج
3. **سلامة المعاملات**: التراجع عند الفشل
4. **إعادة التعيين السريعة**: لتطوير سريع
5. **تحسين الأداء**: إدراجات مجمعة

---

## Implementation in RAG Engine

### File Structure
```
scripts/
├── seed_sample_data.py          # Main seeding script
├── seed_factories.py             # Factory definitions
└── seed_data/                    # Static reference data
    ├── users.json
    └── documents.json
```

### Seeding Script Design

#### 1. Configuration
```python
SEED_CONFIG = {
    "num_users": 10,
    "num_documents_per_user": 15,
    "min_chunks_per_document": 5,
    "max_chunks_per_document": 20,
    "num_chat_sessions_per_user": 5,
    "num_turns_per_session": 3,
}
```

#### 2. User Seeding
```python
def seed_users(session, count: int):
    users = []
    for _ in range(count):
        user = User(
            email=fake.email(),
            api_key=f"sk_{fake.uuid4()[:24]}",
        )
        users.append(user)

    session.bulk_save_objects(users)
    return users
```

#### 3. Document Seeding
```python
def seed_documents(session, users):
    documents = []
    for user in users:
        for _ in range(SEED_CONFIG["num_documents_per_user"]):
            doc = Document(
                user_id=user.id,
                filename=fake.file_name(category="document"),
                content_type=fake.mime_type(),
                file_path=f"/uploads/{fake.uuid4()}",
                size_bytes=fake.random_int(min=1000, max=1000000),
                status="indexed",
            )
            documents.append(doc)

    session.bulk_save_objects(documents)
    return documents
```

#### 4. Chunk Seeding
```python
def seed_chunks(session, documents):
    chunk_rows = []
    document_chunk_rows = []

    for doc in documents:
        num_chunks = fake.random_int(
            min=SEED_CONFIG["min_chunks_per_document"],
            max=SEED_CONFIG["max_chunks_per_document"]
        )

        for i in range(num_chunks):
            chunk_hash = sha256(
                f"{doc.id}:{i}:{fake.sentence()}".encode()
            ).hexdigest()

            chunk = ChunkStoreRow(
                id=str(uuid4()),
                user_id=doc.user_id,
                chunk_hash=chunk_hash,
                text=fake.paragraph(nb_sentences=5),
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
```

#### 5. Chat Seeding
```python
def seed_chat_sessions(session, users):
    sessions = []
    turns = []

    for user in users:
        for _ in range(SEED_CONFIG["num_chat_sessions_per_user"]):
            session_row = ChatSessionRow(
                id=str(uuid4()),
                user_id=user.id,
                title=fake.sentence()[:50],
            )
            sessions.append(session_row)

            for i in range(SEED_CONFIG["num_turns_per_session"]):
                turn = ChatTurnRow(
                    id=str(uuid4()),
                    session_id=session_row.id,
                    user_id=user.id,
                    question=fake.sentence(),
                    answer=fake.paragraph(nb_sentences=3),
                    sources=[str(uuid4()) for _ in range(3)],
                    retrieval_k=3,
                    embed_ms=fake.random_int(min=50, max=200),
                    search_ms=fake.random_int(min=10, max=100),
                    llm_ms=fake.random_int(min=500, max=2000),
                    prompt_tokens=fake.random_int(min=100, max=500),
                    completion_tokens=fake.random_int(min=50, max=300),
                )
                turns.append(turn)

    session.bulk_save_objects(sessions)
    session.bulk_save_objects(turns)
```

---

## Running the Seed Script

### Command Line
```bash
# Seed development database
python scripts/seed_sample_data.py --env development

# Seed with custom config
python scripts/seed_sample_data.py --num-users 50 --num-docs 200

# Reset and reseed
python scripts/seed_sample_data.py --reset
```

### Environment Variables
```bash
export DATABASE_URL="postgresql://user:pass@localhost/rag_dev"
export ENVIRONMENT="development"

python scripts/seed_sample_data.py
```

---

## Common Pitfalls

### 1. Foreign Key Violations
**Problem**: Inserting chunks before documents
```python
# ❌ Wrong order
session.add_all(chunks)  # Fails: document_id references non-existent doc
session.add_all(documents)

# ✅ Correct order
session.add_all(documents)
session.add_all(chunks)
```

### 2. Duplicate API Keys
**Problem**: Random UUIDs may collide (unlikely but possible)
```python
# Check uniqueness before insert
existing_keys = {u.api_key for u in session.query(User.api_key).all()}
```

### 3. Large Seeds Slow Development
**Solution**: Use config for different environments
```python
if env == "testing":
    SEED_CONFIG["num_users"] = 2
else:
    SEED_CONFIG["num_users"] = 50
```

### 4. Orphaned Records
**Problem**: Deleting user doesn't cascade to documents
```sql
-- Ensure CASCADE is set in migrations
FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
```

---

## Testing Seeded Data

### 1. Verify Counts
```python
def test_seeding():
    assert session.query(User).count() == SEED_CONFIG["num_users"]
    assert session.query(Document).count() > 0
    assert session.query(ChunkStoreRow).count() > 0
```

### 2. Verify Relationships
```python
def test_relationships():
    user = session.query(User).first()
    assert len(user.documents) > 0
    assert user.documents[0].user == user
```

### 3. Verify Realism
```python
def test_realistic_distribution():
    chunks_per_doc = [
        session.query(DocumentChunkRow)
            .filter_by(document_id=doc.id)
            .count()
        for doc in session.query(Document).limit(20)
    ]

    avg = sum(chunks_per_doc) / len(chunks_per_doc)
    assert 5 < avg < 20  # Realistic range
```

---

## Related Concepts

### 1. Fixtures
Persistent data for unit tests (pytest fixtures)

### 2. Migrations
Schema changes (Alembic) vs Data seeding

### 3. Factories
Reusable object creation (Factory Boy)

### 4. Mock Data
Runtime substitution for external services

---

## Summary

| Aspect | Recommendation |
|--------|----------------|
| **Tool** | Faker for variety |
| **Strategy** | Hybrid (static + programmatic) |
| **Idempotency** | Check before insert |
| **Performance** | Bulk insert |
| **Safety** | Environment checks |
| **Testing** | Verify counts & relationships |

### Key Takeaways

1. **Seeding accelerates development** by providing instant test data
2. **Use Faker for realistic data** that mirrors production patterns
3. **Keep seeds idempotent** to support multiple runs
4. **Bulk inserts improve performance** for large datasets
5. **Environment-aware seeding** prevents production accidents

---

## Further Reading

- [Faker Documentation](https://faker.readthedocs.io/)
- [Factory Boy](https://factoryboy.readthedocs.io/)
- [SQLAlchemy Bulk Operations](https://docs.sqlalchemy.org/en/14/orm/persistence_techniques.html#bulk-operations)
- [Database Seeding Best Practices](https://martinfowler.com/articles/evodb.html)