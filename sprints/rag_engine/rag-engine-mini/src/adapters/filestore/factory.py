"""
File Store Factory
==================
Creates file store adapters based on configuration.

Priority:
1. S3 (if credentials provided)
2. GCS (if credentials provided)
3. Azure Blob (if credentials provided)
4. Local (fallback)
"""

from src.core.config import Settings
from src.adapters.filestore import (
    LocalFileStore,
    S3FileStore,
    GCSFileStore,
    AzureBlobFileStore,
)


def create_file_store(settings: Settings):
    """
    Create a file store adapter based on settings.

    Raises:
        ValueError: If backend is unsupported or missing dependencies.
    """
    backend = settings.filestore_backend.lower()

    if backend == "local":
        return LocalFileStore(
            upload_dir=settings.upload_dir,
            max_mb=settings.max_upload_mb,
        )

    if backend == "s3":
        if S3FileStore is None:
            raise ValueError("S3FileStore unavailable. Install boto3 to enable S3 backend.")
        return S3FileStore(
            bucket_name=settings.s3_bucket,
            region=settings.s3_region,
            aws_access_key_id=settings.s3_access_key_id,
            aws_secret_access_key=settings.s3_secret_access_key,
            prefix=settings.s3_prefix,
            max_mb=settings.max_upload_mb,
        )

    if backend == "gcs":
        if GCSFileStore is None:
            raise ValueError(
                "GCSFileStore unavailable. Install google-cloud-storage to enable GCS backend."
            )
        return GCSFileStore(
            bucket_name=settings.gcs_bucket,
            project_id=settings.gcs_project_id,
            credentials_path=settings.gcs_credentials_path,
            prefix=settings.gcs_prefix,
            max_mb=settings.max_upload_mb,
        )

    if backend == "azure":
        if AzureBlobFileStore is None:
            raise ValueError(
                "AzureBlobFileStore unavailable. Install azure-storage-blob to enable Azure backend."
            )
        return AzureBlobFileStore(
            container_name=settings.azure_container,
            connection_string=settings.azure_connection_string,
            account_url=settings.azure_account_url,
            account_name=settings.azure_account_name,
            account_key=settings.azure_account_key,
            prefix=settings.azure_prefix,
            max_mb=settings.max_upload_mb,
        )

    raise ValueError(f"Unsupported filestore backend: {backend}")
