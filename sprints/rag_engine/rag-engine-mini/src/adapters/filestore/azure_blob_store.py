"""
Azure Blob Storage Adapter
=========================
Implementation of file storage using Azure Blob Storage.

محول تخزين الملفات باستخدام Azure Blob Storage
"""

import hashlib
import logging
from typing import Optional

from azure.storage.blob import BlobServiceClient, ContainerClient
from azure.core.exceptions import AzureError

from src.domain.entities import StoredFile
from src.domain.errors import FileTooLargeError

log = logging.getLogger(__name__)


class AzureBlobFileStore:
    """
    Azure Blob Storage adapter for file storage.

    Uses Azure Blob Storage for scalable file storage with automatic retry logic.
    Supports container-level access control and custom metadata.

    محول Azure Blob Storage لتخزين الملفات
    """

    def __init__(
        self,
        container_name: str,
        connection_string: Optional[str] = None,
        account_url: Optional[str] = None,
        account_name: Optional[str] = None,
        account_key: Optional[str] = None,
        prefix: str = "uploads/",
        max_mb: int = 20,
    ) -> None:
        """
        Initialize Azure Blob Storage file store.

        Args:
            container_name: Azure container name
            connection_string: Azure storage connection string (optional)
            account_url: Azure storage account URL (optional)
            account_name: Azure storage account name (optional)
            account_key: Azure storage account key (optional)
            prefix: Blob prefix for all files (default: "uploads/")
            max_mb: Maximum file size in MB

        تهيئة تخزين Azure Blob
        """
        self._container_name = container_name
        self._prefix = prefix.rstrip("/") + "/"
        self._max_bytes = max_mb * 1024 * 1024

        # Initialize blob service client
        if connection_string:
            self._blob_service = BlobServiceClient.from_connection_string(connection_string)
        elif account_url and account_key:
            from azure.identity import DefaultAzureCredential

            credential = DefaultAzureCredential()
            self._blob_service = BlobServiceClient(account_url=account_url, credential=credential)
        else:
            # Use environment variables or default credentials
            self._blob_service = BlobServiceClient.from_connection_string(
                os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
            )

        # Get container reference
        try:
            self._container_client: ContainerClient = self._blob_service.get_container_client(
                container_name
            )
            # Verify container exists
            self._container_client.get_container_properties()
            log.info(f"Azure container verified: {container_name}")
        except AzureError as e:
            log.error(f"Azure container not found: {container_name}")
            raise

    async def save_upload(
        self,
        *,
        tenant_id: str,
        upload_filename: str,
        content_type: str,
        data: bytes,
    ) -> StoredFile:
        """
        Save uploaded file to Azure Blob Storage.

        Args:
            tenant_id: Tenant/user ID for namespacing
            upload_filename: Original filename
            content_type: MIME type
            data: File content

        Returns:
            StoredFile with Azure blob name as path

        حفظ الملف المرفوع في Azure Blob Storage
        """
        import os

        # Check size limit
        if len(data) > self._max_bytes:
            raise FileTooLargeError(len(data), self._max_bytes)

        # Generate unique blob name
        file_hash = hashlib.md5(data).hexdigest()[:10]
        safe_name = upload_filename.replace("/", "_").replace("\\", "_")

        # Use tenant-based prefix for isolation
        import time

        timestamp = int(time.time())

        blob_name = f"{self._prefix}{tenant_id}/{timestamp}_{file_hash}_{safe_name}"

        # Upload to Azure Blob Storage
        try:
            blob_client = self._container_client.get_blob_client(blob_name)
            blob_client.upload_blob(
                data,
                overwrite=True,
                content_settings={
                    "content_type": content_type,
                },
                metadata={
                    "tenant-id": tenant_id,
                    "original-filename": upload_filename,
                },
            )
            log.info(f"File uploaded to Azure Blob: {blob_name}")
        except AzureError as e:
            log.error(f"Azure Blob upload failed: {e}")
            raise RuntimeError(f"Failed to upload to Azure Blob: {str(e)}")

        # Return blob name as path (format: azure://container/blob)
        azure_uri = f"azure://{self._container_name}/{blob_name}"

        return StoredFile(
            path=azure_uri,
            filename=upload_filename,
            content_type=content_type,
            size_bytes=len(data),
        )

    async def delete(self, path: str) -> bool:
        """
        Delete a file from Azure Blob Storage.

        Args:
            path: Azure URI (azure://container/blob) or blob name

        Returns:
            True if deleted, False if not found

        حذف ملف من Azure Blob Storage
        """
        import os

        # Parse Azure URI or use path as blob name
        if path.startswith("azure://"):
            # azure://container/blob
            parts = path[8:].split("/", 1)
            container_name = parts[0]
            blob_name = parts[1] if len(parts) > 1 else ""
            container_client = self._blob_service.get_container_client(container_name)
        else:
            container_client = self._container_client
            blob_name = path

        try:
            blob_client = container_client.get_blob_client(blob_name)
            blob_client.delete_blob()
            log.info(f"File deleted from Azure Blob: {blob_name}")
            return True
        except AzureError as e:
            if e.status_code == 404:
                return False
            log.error(f"Azure Blob delete failed: {e}")
            return False

    async def exists(self, path: str) -> bool:
        """
        Check if file exists in Azure Blob Storage.

        Args:
            path: Azure URI or blob name

        Returns:
            True if exists, False otherwise

        التحقق من وجود الملف في Azure Blob Storage
        """
        import os

        # Parse Azure URI or use path as blob name
        if path.startswith("azure://"):
            parts = path[8:].split("/", 1)
            container_name = parts[0]
            blob_name = parts[1] if len(parts) > 1 else ""
            container_client = self._blob_service.get_container_client(container_name)
        else:
            container_client = self._container_client
            blob_name = path

        try:
            blob_client = container_client.get_blob_client(blob_name)
            blob_client.get_blob_properties()
            return True
        except AzureError as e:
            if e.status_code == 404:
                return False
            log.error(f"Azure Blob exists check failed: {e}")
            return False

    async def read_file(self, path: str) -> bytes:
        """
        Read file content from Azure Blob Storage.

        Args:
            path: Azure URI or blob name

        Returns:
            File content as bytes

        قراءة محتوى الملف من Azure Blob Storage
        """
        import os

        # Parse Azure URI or use path as blob name
        if path.startswith("azure://"):
            parts = path[8:].split("/", 1)
            container_name = parts[0]
            blob_name = parts[1] if len(parts) > 1 else ""
            container_client = self._blob_service.get_container_client(container_name)
        else:
            container_client = self._container_client
            blob_name = path

        try:
            blob_client = container_client.get_blob_client(blob_name)
            blob = blob_client.download_blob()
            return blob.readall()
        except AzureError as e:
            if e.status_code == 404:
                raise FileNotFoundError(f"File not found in Azure Blob: {blob_name}")
            log.error(f"Azure Blob read failed: {e}")
            raise

    async def store_file(
        self,
        *,
        tenant_id: str,
        filename: str,
        content: bytes,
    ) -> str:
        """
        Store file and return path (helper method).

        Args:
            tenant_id: Tenant ID
            filename: Original filename
            content: File content

        Returns:
            Storage path/URI

        تخزين الملف وإرجاع المسار
        """
        stored = await self.save_upload(
            tenant_id=tenant_id,
            upload_filename=filename,
            content_type="application/octet-stream",
            data=content,
        )
        return stored.path

    def generate_sas_url(
        self,
        path: str,
        expires_in: int = 3600,
    ) -> str:
        """
        Generate a SAS URL for file access.

        Args:
            path: Azure URI or blob name
            expires_in: URL expiration time in seconds (default: 1 hour)

        Returns:
            SAS URL

        إنشاء URL SAS للوصول للملف
        """
        import os
        from datetime import datetime, timedelta
        from azure.storage.blob import generate_blob_sas, BlobSasPermissions

        # Parse Azure URI or use path as blob name
        if path.startswith("azure://"):
            parts = path[8:].split("/", 1)
            container_name = parts[0]
            blob_name = parts[1] if len(parts) > 1 else ""
            container_client = self._blob_service.get_container_client(container_name)
        else:
            container_client = self._container_client
            blob_name = path

        try:
            blob_client = container_client.get_blob_client(blob_name)

            # Generate SAS token
            sas_token = generate_blob_sas(
                account_name=self._blob_service.account_name,
                container_name=container_client.container_name,
                blob_name=blob_name,
                account_key=self._blob_service.credential.account_key,
                permission=BlobSasPermissions(read=True),
                expiry=datetime.utcnow() + timedelta(seconds=expires_in),
            )

            # Build SAS URL
            blob_url = blob_client.url
            return f"{blob_url}?{sas_token}"
        except AzureError as e:
            log.error(f"Failed to generate SAS URL: {e}")
            raise


if __name__ == "__main__":
    import asyncio

    async def test():
        # Test Azure Blob adapter
        azure_store = AzureBlobFileStore(
            container_name="rag-engine-uploads",
            connection_string=os.environ.get("AZURE_STORAGE_CONNECTION_STRING"),
        )

        # Test save
        stored = await azure_store.save_upload(
            tenant_id="test-tenant",
            upload_filename="test.pdf",
            content_type="application/pdf",
            data=b"test content",
        )
        print(f"Stored: {stored}")

        # Test exists
        exists = await azure_store.exists(stored.path)
        print(f"Exists: {exists}")

        # Test read
        content = await azure_store.read_file(stored.path)
        print(f"Content: {content}")

    # asyncio.run(test())
    print("AzureBlobFileStore ready. Uncomment test() to run.")
