"""
AWS S3 File Store Adapter
==========================
Implementation of file storage using AWS S3.

محول تخزين الملفات باستخدام AWS S3
"""

import hashlib
import logging
import tempfile
from typing import Optional, AsyncIterator, Tuple
import boto3
from botocore.exceptions import BotoCoreError, ClientError

from src.domain.entities import StoredFile
from src.domain.errors import FileTooLargeError

log = logging.getLogger(__name__)


class S3FileStore:
    """
    AWS S3 adapter for file storage.

    Uses multipart upload for large files and handles S3-specific operations.
    Includes retry logic and error handling.

    محول AWS S3 لتخزين الملفات
    """

    def __init__(
        self,
        bucket_name: str,
        region: str = "us-east-1",
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        prefix: str = "uploads/",
        max_mb: int = 20,
    ) -> None:
        """
        Initialize S3 file store.

        Args:
            bucket_name: S3 bucket name
            region: AWS region
            aws_access_key_id: AWS access key (optional, uses env vars if not provided)
            aws_secret_access_key: AWS secret key (optional, uses env vars if not provided)
            prefix: Key prefix for all files (default: "uploads/")
            max_mb: Maximum file size in MB

        تهيئة تخزين S3
        """
        self._bucket_name = bucket_name
        self._prefix = prefix.rstrip("/") + "/"
        self._max_bytes = max_mb * 1024 * 1024

        # Initialize S3 client
        session = boto3.Session(
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=region,
        )
        self._s3 = session.client("s3")
        self._region = region

        # Verify bucket exists
        try:
            self._s3.head_bucket(Bucket=bucket_name)
            log.info(f"S3 bucket verified: {bucket_name} in {region}")
        except ClientError as e:
            if e.response["Error"]["Code"] == "404":
                log.error(f"S3 bucket not found: {bucket_name}")
                raise
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
        Save uploaded file to S3.

        Args:
            tenant_id: Tenant/user ID for namespacing
            upload_filename: Original filename
            content_type: MIME type
            data: File content

        Returns:
            StoredFile with S3 key as path

        حفظ الملف المرفوع في S3
        """
        # Check size limit
        if len(data) > self._max_bytes:
            raise FileTooLargeError(len(data), self._max_bytes)

        # Generate unique S3 key
        file_hash = hashlib.md5(data).hexdigest()[:10]
        safe_name = upload_filename.replace("/", "_").replace("\\", "_")

        # Use tenant-based prefix for isolation
        import time

        timestamp = int(time.time())

        s3_key = f"{self._prefix}{tenant_id}/{timestamp}_{file_hash}_{safe_name}"

        # Upload to S3 (using simple put_object for small files)
        try:
            self._s3.put_object(
                Bucket=self._bucket_name,
                Key=s3_key,
                Body=data,
                ContentType=content_type,
                Metadata={
                    "tenant-id": tenant_id,
                    "original-filename": upload_filename,
                },
            )
            log.info(f"File uploaded to S3: {s3_key}")
        except (BotoCoreError, ClientError) as e:
            log.error(f"S3 upload failed: {e}")
            raise RuntimeError(f"Failed to upload to S3: {str(e)}")

        # Return S3 key as path (format: s3://bucket/key)
        s3_uri = f"s3://{self._bucket_name}/{s3_key}"

        return StoredFile(
            path=s3_uri,
            filename=upload_filename,
            content_type=content_type,
            size_bytes=len(data),
        )

    async def delete(self, path: str) -> bool:
        """
        Delete a file from S3.

        Args:
            path: S3 URI (s3://bucket/key) or S3 key

        Returns:
            True if deleted, False if not found

        حذف ملف من S3
        """
        # Parse S3 URI or use path as key
        if path.startswith("s3://"):
            # s3://bucket/key
            parts = path[5:].split("/", 1)
            bucket = parts[0]
            key = parts[1] if len(parts) > 1 else ""
        else:
            # Assume it's just the key
            bucket = self._bucket_name
            key = path

        try:
            self._s3.delete_object(Bucket=bucket, Key=key)
            log.info(f"File deleted from S3: {key}")
            return True
        except ClientError as e:
            if e.response["Error"]["Code"] == "NoSuchKey":
                return False
            log.error(f"S3 delete failed: {e}")
            return False

    async def exists(self, path: str) -> bool:
        """
        Check if file exists in S3.

        Args:
            path: S3 URI or S3 key

        Returns:
            True if exists, False otherwise

        التحقق من وجود الملف في S3
        """
        # Parse S3 URI or use path as key
        if path.startswith("s3://"):
            parts = path[5:].split("/", 1)
            bucket = parts[0]
            key = parts[1] if len(parts) > 1 else ""
        else:
            bucket = self._bucket_name
            key = path

        try:
            self._s3.head_object(Bucket=bucket, Key=key)
            return True
        except ClientError as e:
            if e.response["Error"]["Code"] == "404":
                return False
            raise

    async def read_file(self, path: str) -> bytes:
        """
        Read file content from S3.

        Args:
            path: S3 URI or S3 key

        Returns:
            File content as bytes

        قراءة محتوى الملف من S3
        """
        # Parse S3 URI or use path as key
        if path.startswith("s3://"):
            parts = path[5:].split("/", 1)
            bucket = parts[0]
            key = parts[1] if len(parts) > 1 else ""
        else:
            bucket = self._bucket_name
            key = path

        try:
            response = self._s3.get_object(Bucket=bucket, Key=key)
            return response["Body"].read()
        except ClientError as e:
            if e.response["Error"]["Code"] == "NoSuchKey":
                raise FileNotFoundError(f"File not found in S3: {key}")
            log.error(f"S3 read failed: {e}")
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

    async def save_upload_stream(
        self,
        *,
        tenant_id: str,
        upload_filename: str,
        content_type: str,
        data_stream: AsyncIterator[bytes],
    ) -> Tuple[StoredFile, str]:
        """
        Save uploaded file to S3 using a stream.

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
            s3_key = f"{self._prefix}{tenant_id}/{timestamp}_{file_hash}_{safe_name}"

            try:
                self._s3.upload_file(
                    Filename=tmp_path,
                    Bucket=self._bucket_name,
                    Key=s3_key,
                    ExtraArgs={
                        "ContentType": content_type,
                        "Metadata": {
                            "tenant-id": tenant_id,
                            "original-filename": upload_filename,
                        },
                    },
                )
                log.info(f"File uploaded to S3: {s3_key}")
            except (BotoCoreError, ClientError) as e:
                log.error(f"S3 upload failed: {e}")
                raise RuntimeError(f"Failed to upload to S3: {str(e)}")

            s3_uri = f"s3://{self._bucket_name}/{s3_key}"
            return (
                StoredFile(
                    path=s3_uri,
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

    def generate_presigned_url(
        self,
        path: str,
        expires_in: int = 3600,
    ) -> str:
        """
        Generate a presigned URL for file access.

        Args:
            path: S3 URI or S3 key
            expires_in: URL expiration time in seconds (default: 1 hour)

        Returns:
            Presigned URL

        إنشاء URL موقعة للوصول للملف
        """
        # Parse S3 URI or use path as key
        if path.startswith("s3://"):
            parts = path[5:].split("/", 1)
            bucket = parts[0]
            key = parts[1] if len(parts) > 1 else ""
        else:
            bucket = self._bucket_name
            key = path

        try:
            url = self._s3.generate_presigned_url(
                "get_object",
                Params={"Bucket": bucket, "Key": key},
                ExpiresIn=expires_in,
            )
            return url
        except ClientError as e:
            log.error(f"Failed to generate presigned URL: {e}")
            raise


if __name__ == "__main__":
    import asyncio

    async def test():
        # Test S3 adapter
        s3_store = S3FileStore(
            bucket_name="rag-engine-uploads",
            region="us-east-1",
        )

        # Test save
        stored = await s3_store.save_upload(
            tenant_id="test-tenant",
            upload_filename="test.pdf",
            content_type="application/pdf",
            data=b"test content",
        )
        print(f"Stored: {stored}")

        # Test exists
        exists = await s3_store.exists(stored.path)
        print(f"Exists: {exists}")

        # Test read
        content = await s3_store.read_file(stored.path)
        print(f"Content: {content}")

    # asyncio.run(test())
    print("S3FileStore ready. Uncomment test() to run.")
