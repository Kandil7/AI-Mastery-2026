"""File store adapters package."""

from src.adapters.filestore.local_store import LocalFileStore
try:
    from src.adapters.filestore.s3_store import S3FileStore
except (ModuleNotFoundError, ImportError):
    S3FileStore = None
try:
    from src.adapters.filestore.gcs_store import GCSFileStore
except (ModuleNotFoundError, ImportError):
    GCSFileStore = None
try:
    from src.adapters.filestore.azure_blob_store import AzureBlobFileStore
except (ModuleNotFoundError, ImportError):
    AzureBlobFileStore = None

__all__ = [
    "LocalFileStore",
    "S3FileStore",
    "GCSFileStore",
    "AzureBlobFileStore",
]
