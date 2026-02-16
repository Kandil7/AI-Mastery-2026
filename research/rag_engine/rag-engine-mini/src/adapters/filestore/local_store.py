"""
Local File Store Adapter
=========================
Implementation of FileStorePort for local filesystem.

محول تخزين الملفات المحلية
"""

import hashlib
import os
import time
import tempfile
from typing import AsyncIterator, Tuple
from pathlib import Path

import aiofiles

from src.domain.entities import StoredFile
from src.domain.errors import FileTooLargeError


class LocalFileStore:
    """
    Local filesystem adapter implementing FileStorePort.
    
    محول نظام الملفات المحلي
    """
    
    def __init__(
        self,
        upload_dir: str = "./uploads",
        max_mb: int = 20,
    ) -> None:
        """
        Initialize local file store.
        
        Args:
            upload_dir: Directory for uploaded files
            max_mb: Maximum file size in MB
        """
        self._dir = upload_dir
        self._max_bytes = max_mb * 1024 * 1024
        
        # Ensure directory exists
        Path(self._dir).mkdir(parents=True, exist_ok=True)
    
    async def save_upload(
        self,
        *,
        tenant_id: str,
        upload_filename: str,
        content_type: str,
        data: bytes,
    ) -> StoredFile:
        """
        Save uploaded file to local filesystem.
        
        File naming: {timestamp}_{tenant}_{hash}_{filename}
        This ensures uniqueness while remaining human-readable.
        """
        # Check size limit
        if len(data) > self._max_bytes:
            raise FileTooLargeError(len(data), self._max_bytes)
        
        # Generate unique filename
        file_hash = hashlib.md5(data).hexdigest()[:10]
        safe_name = upload_filename.replace("/", "_").replace("\\", "_")
        timestamp = int(time.time())
        
        filename = f"{timestamp}_{tenant_id[:8]}_{file_hash}_{safe_name}"
        path = os.path.join(self._dir, filename)
        
        # Write file asynchronously
        async with aiofiles.open(path, "wb") as f:
            await f.write(data)
        
        return StoredFile(
            path=path,
            filename=upload_filename,
            content_type=content_type,
            size_bytes=len(data),
        )
    
    async def delete(self, path: str) -> bool:
        """Delete a stored file."""
        try:
            if os.path.exists(path):
                os.remove(path)
                return True
            return False
        except OSError:
            return False
    
    async def exists(self, path: str) -> bool:
        """Check if file exists."""
        return os.path.exists(path)

    async def save_upload_stream(
        self,
        *,
        tenant_id: str,
        upload_filename: str,
        content_type: str,
        data_stream: AsyncIterator[bytes],
    ) -> Tuple[StoredFile, str]:
        """
        Save uploaded file to local filesystem from a stream.

        Returns stored file and sha256 hash.
        """
        hasher = hashlib.sha256()
        total_bytes = 0

        tmp_dir = Path(self._dir)
        tmp_dir.mkdir(parents=True, exist_ok=True)

        tmp_file = tempfile.NamedTemporaryFile(delete=False, dir=self._dir, suffix=".tmp")
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
            timestamp = int(time.time())
            filename = f"{timestamp}_{tenant_id[:8]}_{file_hash}_{safe_name}"
            final_path = os.path.join(self._dir, filename)

            os.replace(tmp_path, final_path)

            return (
                StoredFile(
                    path=final_path,
                    filename=upload_filename,
                    content_type=content_type,
                    size_bytes=total_bytes,
                ),
                hasher.hexdigest(),
            )
        except Exception:
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except OSError:
                pass
            raise
