# Storage Factory Pattern

## Introduction

The storage factory pattern enables selecting storage backends (S3, GCS, Azure, Local) based on configuration, providing flexibility and multi-cloud support.

## Learning Objectives

By end of this guide, you will understand:
- **Factory design pattern and when to use it**
- **Configuration-driven architecture**
- **Multi-cloud storage strategies**
- **Storage abstraction layers**
- **Error handling and fallbacks**
- **Singleton pattern for resource management**

---

## Factory Pattern Overview

### What is Factory Pattern?

The factory pattern provides an interface for creating objects without specifying their exact classes. It allows:

1. **Runtime selection**: Choose implementation at startup
2. **Configuration-driven**: Backends selected via config
3. **Loose coupling**: Client code doesn't depend on concrete classes
4. **Easy testing**: Mock implementations for tests

### Factory Structure

```
Client Code
    ↓
    calls factory.create_storage()
    ↓
    StorageFactory (factory)
    ↓
    returns FileStore (interface)
    ↓
    Concrete Implementation (S3/GCS/Azure/Local)
```

---

## Storage Types

### S3 (Amazon Simple Storage Service)

**Best for:** Production with AWS infrastructure

**Features:**
- High durability (99.999999999%)
- Multi-region replication
- Lifecycle policies (auto-delete)
- Event notifications

**Configuration:**
```python
{
    "storage_type": "s3",
    "s3_bucket": "my-rag-documents",
    "s3_region": "us-east-1",
    "s3_access_key": "AKIAIOSFODNN7EXAMPLE",
    "s3_secret_key": "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
}
```

### GCS (Google Cloud Storage)

**Best for:** Production with Google Cloud infrastructure

**Features:**
- Strong consistency
- Multi-regional buckets
- Object versioning
- Nearline storage class

**Configuration:**
```python
{
    "storage_type": "gcs",
    "gcs_bucket": "my-rag-documents",
    "gcs_credentials_path": "/path/to/credentials.json",
}
```

### Azure Blob Storage

**Best for:** Production with Azure infrastructure

**Features:**
- Hot, Cool, and Archive tiers
- Geo-redundant storage
- Azure CDN integration
- Immutable blob support

**Configuration:**
```python
{
    "storage_type": "azure",
    "azure_container": "rag-documents",
    "azure_account_name": "myaccount",
    "azure_account_key": "base64_encoded_key",
}
```

### Local File Storage

**Best for:** Development, testing, small deployments

**Features:**
- No external dependencies
- Fast for local development
- Simple debugging

**Configuration:**
```python
{
    "storage_type": "local",
    "storage_path": "./data/uploads",
}
```

---

## Implementation: Storage Factory

### Basic Structure

```python
class StorageFactory:
    """Factory for creating file storage instances."""

    @staticmethod
    def create_storage(config: dict) -> FileStore:
        """
        Create storage instance based on configuration.

        Priority:
        1. S3 (if credentials provided)
        2. GCS (if credentials provided)
        3. Azure Blob (if credentials provided)
        4. Local (fallback)
        """
        storage_type = config.get("storage_type", "local").lower()

        if storage_type == "s3":
            return StorageFactory._create_s3(config)
        elif storage_type == "gcs":
            return StorageFactory._create_gcs(config)
        elif storage_type == "azure":
            return StorageFactory._create_azure(config)
        else:
            return StorageFactory._create_local(config)
```

### S3 Implementation

```python
@staticmethod
def _create_s3(config: dict) -> S3FileStore:
    """
    Create S3 storage.

    Required config:
    - s3_bucket: Bucket name
    - s3_region: AWS region
    - s3_access_key: AWS access key
    - s3_secret_key: AWS secret key
    """
    if not all([
        config.get("s3_bucket"),
        config.get("s3_region"),
        config.get("s3_access_key"),
        config.get("s3_secret_key"),
    ]):
        raise ValueError(
            "S3 storage requires: s3_bucket, s3_region, "
            "s3_access_key, s3_secret_key"
        )

    return S3FileStore(
        bucket_name=config["s3_bucket"],
        region=config["s3_region"],
        access_key=config["s3_access_key"],
        secret_key=config["s3_secret_key"],
        prefix=config.get("storage_prefix", "uploads/"),
    )
```

### Fallback Strategy

Always provide a fallback to prevent system failures:

```python
@staticmethod
def create_storage(config: dict) -> FileStore:
    storage_type = config.get("storage_type", "local").lower()

    try:
        if storage_type == "s3":
            return StorageFactory._create_s3(config)
        elif storage_type == "gcs":
            return StorageFactory._create_gcs(config)
        elif storage_type == "azure":
            return StorageFactory._create_azure(config)

    except Exception as e:
        log.error("storage_creation_failed", storage_type=storage_type, error=str(e))

        # Fallback to local storage
        log.warning("falling_back_to_local_storage")
        return StorageFactory._create_local(config)
```

---

## Singleton Pattern

### Why Singleton?

Storage connections (especially S3/GCS/Azure) are expensive to create. Singleton pattern ensures:

1. **One instance per process**: Reduces resource usage
2. **Connection pooling**: Reuse connections
3. **Consistent state**: Same configuration across all calls

### Singleton Implementation

```python
_storage_instance: Optional[FileStore] = None

def get_storage() -> FileStore:
    """
    Get singleton storage instance.

    Returns:
        FileStore instance (created on first call)
    """
    global _storage_instance

    if _storage_instance is None:
        try:
            _storage_instance = StorageFactory.create_storage()
            log.info("storage_singleton_created", type=type(_storage_instance).__name__)
        except Exception as e:
            log.error("storage_initialization_failed", error=str(e))
            raise RuntimeError(f"Failed to initialize storage: {str(e)}") from e

    return _storage_instance

def reset_storage() -> None:
    """Reset storage instance (for testing)."""
    global _storage_instance
    _storage_instance = None
```

---

## Integration with Document Management

### Using Factory in Services

```python
from src.adapters.filestore.factory import get_storage

class DocumentManagementService:
    """Document management with pluggable storage."""

    def __init__(self):
        # Get storage instance from factory
        self._storage = get_storage()

    def upload_document(self, file_data, filename: str) -> str:
        """Upload document using configured storage."""
        return self._storage.store_file(
            tenant_id="tenant-123",
            filename=filename,
            content=file_data,
        )

    def delete_document(self, storage_path: str) -> bool:
        """Delete document from configured storage."""
        return self._storage.delete_file(storage_path)
```

### Benefits

1. **Multi-cloud support**: Deploy to any cloud provider
2. **Easy switching**: Change config to switch backends
3. **Development flexibility**: Use local for dev, cloud for prod
4. **Cost optimization**: Choose provider based on pricing
5. **Compliance**: Select provider based on data residency requirements

---

## Error Handling

### Configuration Validation

```python
def validate_storage_config(config: dict) -> List[str]:
    """
    Validate storage configuration.

    Returns:
        List of validation errors (empty if valid)
    """
    errors = []

    storage_type = config.get("storage_type", "local")

    if storage_type == "s3":
        required = ["s3_bucket", "s3_region", "s3_access_key", "s3_secret_key"]
        for field in required:
            if not config.get(field):
                errors.append(f"S3 missing required field: {field}")

    elif storage_type == "gcs":
        required = ["gcs_bucket", "gcs_credentials_path"]
        for field in required:
            if not config.get(field):
                errors.append(f"GCS missing required field: {field}")

    elif storage_type == "azure":
        required = ["azure_container", "azure_account_name", "azure_account_key"]
        for field in required:
            if not config.get(field):
                errors.append(f"Azure missing required field: {field}")

    return errors
```

### Graceful Degradation

```python
def create_storage_with_fallback(config: dict) -> FileStore:
    """Create storage with fallback to local on error."""
    try:
        return StorageFactory.create_storage(config)
    except Exception as e:
        log.error("primary_storage_failed", error=str(e))

        # Attempt fallback to local storage
        try:
            local_config = {"storage_type": "local"}
            return StorageFactory._create_local(local_config)
        except Exception as fallback_error:
            log.critical("all_storage_backends_failed", primary=str(e), fallback=str(fallback_error))
            raise RuntimeError(
                "All storage backends failed to initialize. "
                "Cannot proceed without storage."
            )
```

---

## Best Practices

### ✅ DO

1. **Provide fallback storage**
```python
# GOOD: Always have local storage fallback
try:
    storage = create_cloud_storage()
except Exception:
    storage = create_local_storage()
```

2. **Validate configuration early**
```python
# GOOD: Validate at startup
errors = validate_storage_config(config)
if errors:
    raise ValueError(f"Invalid storage config: {'; '.join(errors)}")
```

3. **Use singleton pattern**
```python
# GOOD: One storage instance per process
storage = get_storage()  # Returns singleton
```

4. **Log storage selection**
```python
# GOOD: Log which backend is used
log.info(
    "storage_backend_selected",
    backend=storage_type,
    bucket=bucket_name,
)
```

### ❌ DON'T

1. **Don't create new instances unnecessarily**
```python
# BAD: New storage object every call
def upload(file):
    storage = create_storage()  # Wasteful!
    return storage.store_file(file)
```

2. **Don't hardcode storage type**
```python
# BAD: Hardcoded S3
storage = S3FileStore(bucket="fixed-bucket")

# GOOD: Use factory
storage = create_storage(config)  # Configurable
```

3. **Don't ignore validation errors**
```python
# BAD: Proceed with invalid config
storage = create_storage(config)  # May fail silently

# GOOD: Validate first
validate_storage_config(config)
storage = create_storage(config)
```

---

## Testing

### Unit Tests

```python
import pytest
from unittest.mock import Mock, patch

def test_storage_factory_creates_s3():
    """Test S3 storage creation."""
    config = {
        "storage_type": "s3",
        "s3_bucket": "test-bucket",
        "s3_region": "us-east-1",
        "s3_access_key": "test-key",
        "s3_secret_key": "test-secret",
    }

    storage = StorageFactory.create_storage(config)

    assert isinstance(storage, S3FileStore)
    assert storage.bucket_name == "test-bucket"

def test_storage_factory_fallback_to_local():
    """Test fallback to local on error."""
    config = {"storage_type": "s3"}  # Missing required fields

    with patch('src.adapters.filestore.S3FileStore') as mock_s3:
        mock_s3.side_effect = Exception("S3 not available")

        storage = StorageFactory.create_storage(config)

        # Should fallback to local
        assert isinstance(storage, LocalFileStore)

def test_storage_singleton():
    """Test singleton pattern."""
    reset_storage()

    storage1 = get_storage()
    storage2 = get_storage()

    assert storage1 is storage2  # Same instance
```

---

## Migration Between Backends

### Strategy

When switching storage backends:

1. **New backend**: Configure new provider
2. **Run migration**: Copy files from old to new
3. **Cutover**: Switch app to use new backend
4. **Retain old**: Keep old backend for rollback

### Migration Script

```python
import asyncio
from src.adapters.filestore.s3_store import S3FileStore
from src.adapters.filestore.gcs_store import GCSFileStore

async def migrate_storage(
    source: S3FileStore,
    destination: GCSFileStore,
    max_files: int = 1000,
):
    """
    Migrate files from S3 to GCS.

    Args:
        source: S3 storage to migrate from
        destination: GCS storage to migrate to
        max_files: Maximum files to migrate (safety limit)
    """
    migrated = 0
    failed = 0

    # List all files in S3
    for file_info in source.list_files(prefix="uploads/"):
        if migrated >= max_files:
            log.warning("migration_limit_reached")
            break

        try:
            # Download from S3
            content = source.read_file(file_info.path)

            # Upload to GCS
            new_path = destination.store_file(
                tenant_id=file_info.tenant_id,
                filename=file_info.filename,
                content=content,
            )

            log.info("file_migrated", path=file_info.path, new_path=new_path)
            migrated += 1

        except Exception as e:
            log.error("file_migration_failed", path=file_info.path, error=str(e))
            failed += 1

    log.info("migration_complete", migrated=migrated, failed=failed)
    return {"migrated": migrated, "failed": failed}
```

---

## Summary

### Key Takeaways:

1. **Factory pattern** enables runtime backend selection
2. **Configuration-driven** allows flexible deployment
3. **Fallback strategy** prevents total system failure
4. **Singleton pattern** reduces resource usage
5. **Multi-cloud support** provides flexibility
6. **Validation early** prevents runtime failures
7. **Testing** ensures all backends work correctly

### Best Practices:

- ✅ Provide fallback storage (local always works)
- ✅ Validate configuration at startup
- ✅ Use singleton pattern for expensive resources
- ✅ Log storage backend selection
- ✅ Handle errors gracefully
- ✅ Test all storage backends

### Anti-Patterns:

- ❌ Don't create new storage instances unnecessarily
- ❌ Don't hardcode storage type in code
- ❌ Don't proceed with invalid configuration
- ❌ Don't ignore validation errors
- ❌ Don't use multiple storage types simultaneously

---

## Additional Resources

- **Factory Pattern**: Refactoring Guru articles
- **AWS S3 SDK**: https://boto3.amazonaws.com/
- **GCS SDK**: https://googleapis.dev/python/storage/latest
- **Azure SDK**: https://docs.microsoft.com/en-us/azure/storage/
- **Storage Best Practices**: OWASP storage security guide
