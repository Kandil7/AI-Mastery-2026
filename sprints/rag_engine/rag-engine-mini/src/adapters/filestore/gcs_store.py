"""
Google Cloud Storage Adapter
=============================
Implementation of file storage using Google Cloud Storage.

محول تخزين الملفات باستخدام Google Cloud Storage
"""

import hashlib
import logging
import tempfile
from typing import Optional, AsyncIterator, Tuple

from google.cloud import storage
from google.cloud.exceptions import GoogleCloudError

from src.domain.entities import StoredFile
from src.domain.errors import FileTooLargeError

log = logging.getLogger(__name__)


class GCSFileStore:
    """
    Google Cloud Storage adapter for file storage.

    Uses GCS for scalable file storage with automatic retry logic.
    Supports bucket-level access control and custom metadata.

    محول Google Cloud Storage لتخزين الملفات
    """

    def __init__(
        self,
        bucket_name: str,
        project_id: Optional[str] = None,
        credentials_path: Optional[str] = None,
        prefix: str = "uploads/",
        max_mb: int = 20,
    ) -> None:
        """
        Initialize GCS file store.

        Args:
            bucket_name: GCS bucket name
            project_id: GCP project ID (optional, uses credentials default)
            credentials_path: Path to service account JSON key (optional)
            prefix: Object prefix for all files (default: "uploads/")
            max_mb: Maximum file size in MB

        تهيئة تخزين GCS
        """
        self._bucket_name = bucket_name
        self._prefix = prefix.rstrip("/") + "/"
        self._max_bytes = max_mb * 1024 * 1024

        # Initialize GCS client
        if credentials_path:
            from google.oauth2 import service_account

            credentials = service_account.Credentials.from_service_account_file(credentials_path)
            self._client = storage.Client(project=project_id, credentials=credentials)
        else:
            self._client = storage.Client(project=project_id)

        # Get bucket reference
        try:
            self._bucket = self._client.bucket(bucket_name)
            # Verify bucket exists
            self._bucket.reload()
            log.info(f"GCS bucket verified: {bucket_name}")
        except GoogleCloudError as e:
            log.error(f"GCS bucket not found: {bucket_name}")
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
        Save uploaded file to GCS.

        Args:
            tenant_id: Tenant/user ID for namespacing
            upload_filename: Original filename
            content_type: MIME type
            data: File content

        Returns:
            StoredFile with GCS object name as path

        حفظ الملف المرفوع في GCS
        """
        # Check size limit
        if len(data) > self._max_bytes:
            raise FileTooLargeError(len(data), self._max_bytes)

        # Generate unique GCS object name
        file_hash = hashlib.md5(data).hexdigest()[:10]
        safe_name = upload_filename.replace("/", "_").replace("\\", "_")

        # Use tenant-based prefix for isolation
        import time

        timestamp = int(time.time())

        gcs_key = f"{self._prefix}{tenant_id}/{timestamp}_{file_hash}_{safe_name}"

        # Upload to GCS
        try:
            blob = self._bucket.blob(gcs_key)
            blob.upload_from_string(
                data,
                content_type=content_type,
            )
            blob.metadata = {
                "tenant-id": tenant_id,
                "original-filename": upload_filename,
            }
            blob.patch()
            log.info(f"File uploaded to GCS: {gcs_key}")
        except GoogleCloudError as e:
            log.error(f"GCS upload failed: {e}")
            raise RuntimeError(f"Failed to upload to GCS: {str(e)}")

        # Return GCS key as path (format: gs://bucket/key)
        gcs_uri = f"gs://{self._bucket_name}/{gcs_key}"

        return StoredFile(
            path=gcs_uri,
            filename=upload_filename,
            content_type=content_type,
            size_bytes=len(data),
        )

    async def delete(self, path: str) -> bool:
        """
        Delete a file from GCS.

        Args:
            path: GCS URI (gs://bucket/key) or GCS object name

        Returns:
            True if deleted, False if not found

        حذف ملف من GCS
        """
        # Parse GCS URI or use path as key
        if path.startswith("gs://"):
            # gs://bucket/key
            parts = path[5:].split("/", 1)
            bucket_name = parts[0]
            key = parts[1] if len(parts) > 1 else ""
            bucket = self._client.bucket(bucket_name)
        else:
            bucket = self._bucket
            key = path

        try:
            blob = bucket.blob(key)
            blob.delete()
            log.info(f"File deleted from GCS: {key}")
            return True
        except GoogleCloudError as e:
            if e.code == 404:
                return False
            log.error(f"GCS delete failed: {e}")
            return False

    async def exists(self, path: str) -> bool:
        """
        Check if file exists in GCS.

        Args:
            path: GCS URI or GCS object name

        Returns:
            True if exists, False otherwise

        التحقق من وجود الملف في GCS
        """
        # Parse GCS URI or use path as key
        if path.startswith("gs://"):
            parts = path[5:].split("/", 1)
            bucket_name = parts[0]
            key = parts[1] if len(parts) > 1 else ""
            bucket = self._client.bucket(bucket_name)
        else:
            bucket = self._bucket
            key = path

        try:
            blob = bucket.blob(key)
            return blob.exists()
        except GoogleCloudError as e:
            log.error(f"GCS exists check failed: {e}")
            return False

    async def read_file(self, path: str) -> bytes:
        """
        Read file content from GCS.

        Args:
            path: GCS URI or GCS object name

        Returns:
            File content as bytes

        قراءة محتوى الملف من GCS
        """
        # Parse GCS URI or use path as key
        if path.startswith("gs://"):
            parts = path[5:].split("/", 1)
            bucket_name = parts[0]
            key = parts[1] if len(parts) > 1 else ""
            bucket = self._client.bucket(bucket_name)
        else:
            bucket = self._bucket
            key = path

        try:
            blob = bucket.blob(key)
            return blob.download_as_bytes()
        except GoogleCloudError as e:
            if e.code == 404:
                raise FileNotFoundError(f"File not found in GCS: {key}")
            log.error(f"GCS read failed: {e}")
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

    def generate_signed_url(
        self,
        path: str,
        expires_in: int = 3600,
    ) -> str:
        """
        Generate a signed URL for file access.

        Args:
            path: GCS URI or GCS object name
            expires_in: URL expiration time in seconds (default: 1 hour)

        Returns:
            Signed URL

        إنشاء URL موقعة للوصول للملف
        """
        # Parse GCS URI or use path as key
        if path.startswith("gs://"):
            parts = path[5:].split("/", 1)
            bucket_name = parts[0]
            key = parts[1] if len(parts) > 1 else ""
            bucket = self._client.bucket(bucket_name)
        else:
            bucket = self._bucket
            key = path

        try:
            blob = bucket.blob(key)
            url = blob.generate_signed_url(
                version="v4",
                expiration=expires_in,
                method="GET",
            )
            return url
        except GoogleCloudError as e:
            log.error(f"Failed to generate signed URL: {e}")
            raise

    async def save_upload_stream(
        self,
        *,
        tenant_id: str,
        upload_filename: str,
        content_type: str,
        data_stream: AsyncIterator[bytes],
    ) -> Tuple[StoredFile, str]:
        """
        Save uploaded file to GCS using a stream.

        Returns stored file and sha256 hash.
        """
        import aiofiles

        hasher = hashlib.sha256()
        total_bytes = 0

        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".tmp")
        tmp_path = tmp_file.name
        tmp_file.close()

        try:
            async with aiofiles.open(tmp_path, "wb") as f:
                async for chunk in data_stream:
                    if not chunk:
                        continue
                    total_bytes += len(chunk)
                    if total_bytes > self._max_bytes:
                        raise FileTooLargeError(total_bytes, self._max_bytes)
                    hasher.update(chunk)
                    await f.write(chunk)

            file_hash = hasher.hexdigest()[:10]
            safe_name = upload_filename.replace("/", "_").replace("\\", "_")
            import time

            timestamp = int(time.time())
            gcs_key = f"{self._prefix}{tenant_id}/{timestamp}_{file_hash}_{safe_name}"

            try:
                blob = self._bucket.blob(gcs_key)
                blob.upload_from_filename(tmp_path, content_type=content_type)
                blob.metadata = {
                    "tenant-id": tenant_id,
                    "original-filename": upload_filename,
                }
                blob.patch()
                log.info(f"File uploaded to GCS: {gcs_key}")
            except GoogleCloudError as e:
                log.error(f"GCS upload failed: {e}")
                raise RuntimeError(f"Failed to upload to GCS: {str(e)}")

            gcs_uri = f"gs://{self._bucket_name}/{gcs_key}"
            return (
                StoredFile(
                    path=gcs_uri,
                    filename=upload_filename,
                    content_type=content_type,
                    size_bytes=total_bytes,
                ),
                hasher.hexdigest(),
            )
        finally:
            try:
                import os

                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except OSError:
                pass


if __name__ == "__main__":
    import asyncio

    async def test():
        # Test GCS adapter
        gcs_store = GCSFileStore(
            bucket_name="rag-engine-uploads",
            project_id="my-project",
        )

        # Test save
        stored = await gcs_store.save_upload(
            tenant_id="test-tenant",
            upload_filename="test.pdf",
            content_type="application/pdf",
            data=b"test content",
        )
        print(f"Stored: {stored}")

        # Test exists
        exists = await gcs_store.exists(stored.path)
        print(f"Exists: {exists}")

        # Test read
        content = await gcs_store.read_file(stored.path)
        print(f"Content: {content}")

    # asyncio.run(test())
    print("GCSFileStore ready. Uncomment test() to run.")
