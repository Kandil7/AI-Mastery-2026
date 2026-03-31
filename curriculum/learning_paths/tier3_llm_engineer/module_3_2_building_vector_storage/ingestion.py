"""
Document Ingestion Module

Production-ready document ingestion with support for:
- PDF, HTML, Markdown, JSON parsing
- API connectors (GitHub, Google Drive)
- Metadata extraction and preservation
- Batch processing with progress tracking
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional, Union

import httpx

logger = logging.getLogger(__name__)


@dataclass
class Document:
    """
    Represents a document with content and metadata.

    Attributes:
        content: The text content of the document
        metadata: Document metadata (source, timestamps, etc.)
        id: Unique document identifier
        embedding: Optional pre-computed embedding
    """

    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    id: Optional[str] = None
    embedding: Optional[List[float]] = None

    def __post_init__(self) -> None:
        if not self.id:
            # Generate ID from content hash
            self.id = hashlib.sha256(self.content.encode()).hexdigest()[:16]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "content": self.content,
            "metadata": self.metadata,
            "embedding": self.embedding,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Document":
        """Create from dictionary."""
        return cls(
            content=data["content"],
            metadata=data.get("metadata", {}),
            id=data.get("id"),
            embedding=data.get("embedding"),
        )

    def __len__(self) -> int:
        return len(self.content)


class BaseParser(ABC):
    """Abstract base class for document parsers."""

    @abstractmethod
    async def parse(self, source: Union[str, bytes, Path]) -> List[Document]:
        """
        Parse source into documents.

        Args:
            source: File path, URL, or raw content

        Returns:
            List of parsed documents
        """
        pass

    @abstractmethod
    def supports(self, source: Union[str, Path]) -> bool:
        """Check if parser supports this source type."""
        pass


class PDFParser(BaseParser):
    """
    PDF document parser.

    Supports:
    - Text extraction
    - Metadata extraction
    - Page-level chunking
    - OCR fallback (with pytesseract)
    """

    def __init__(
        self,
        extract_images: bool = False,
        use_ocr: bool = False,
        page_separator: str = "\n\n---PAGE---\n\n",
    ) -> None:
        self.extract_images = extract_images
        self.use_ocr = use_ocr
        self.page_separator = page_separator

        # Try to import PDF libraries
        self._pypdf = None
        self._pdfplumber = None
        self._pytesseract = None

        try:
            import pypdf
            self._pypdf = pypdf
        except ImportError:
            logger.warning("pypdf not installed. PDF parsing unavailable.")

        try:
            import pdfplumber
            self._pdfplumber = pdfplumber
        except ImportError:
            pass

        if use_ocr:
            try:
                import pytesseract
                self._pytesseract = pytesseract
            except ImportError:
                logger.warning("pytesseract not installed. OCR unavailable.")

    def supports(self, source: Union[str, Path]) -> bool:
        if isinstance(source, str):
            return source.lower().endswith(".pdf")
        return isinstance(source, Path) and source.suffix.lower() == ".pdf"

    async def parse(self, source: Union[str, bytes, Path]) -> List[Document]:
        """Parse PDF into documents."""
        if not self._pypdf and not self._pdfplumber:
            raise RuntimeError("No PDF library available. Install pypdf or pdfplumber.")

        # Load PDF
        if isinstance(source, bytes):
            pdf_bytes = source
        elif isinstance(source, (str, Path)):
            with open(source, "rb") as f:
                pdf_bytes = f.read()
        else:
            raise ValueError(f"Unsupported source type: {type(source)}")

        documents = []

        # Try pdfplumber first (better text extraction)
        if self._pdfplumber:
            documents = await self._parse_with_pdfplumber(pdf_bytes)
        elif self._pypdf:
            documents = await self._parse_with_pypdf(pdf_bytes)

        logger.info(f"Parsed PDF: {len(documents)} pages/documents")
        return documents

    async def _parse_with_pdfplumber(self, pdf_bytes: bytes) -> List[Document]:
        """Parse using pdfplumber."""
        import io

        documents = []

        with io.BytesIO(pdf_bytes) as bio:
            with self._pdfplumber.open(bio) as pdf:
                metadata = {
                    "source_type": "pdf",
                    "total_pages": len(pdf.pages),
                    "parsed_at": datetime.utcnow().isoformat(),
                }

                # Extract metadata
                if pdf.metadata:
                    metadata.update({
                        "title": pdf.metadata.get("Title", ""),
                        "author": pdf.metadata.get("Author", ""),
                        "subject": pdf.metadata.get("Subject", ""),
                    })

                for i, page in enumerate(pdf.pages):
                    text = page.extract_text() or ""

                    # OCR fallback if enabled and no text
                    if not text.strip() and self.use_ocr and self._pytesseract:
                        try:
                            image = page.to_image()
                            text = self._pytesseract.image_to_string(image.original)
                        except Exception as e:
                            logger.warning(f"OCR failed for page {i}: {e}")

                    if text.strip():
                        doc = Document(
                            content=text,
                            metadata={
                                **metadata,
                                "page": i + 1,
                                "source": f"page_{i + 1}",
                            },
                        )
                        documents.append(doc)

        return documents

    async def _parse_with_pypdf(self, pdf_bytes: bytes) -> List[Document]:
        """Parse using pypdf."""
        import io

        documents = []

        with io.BytesIO(pdf_bytes) as bio:
            reader = self._pypdf.PdfReader(bio)

            metadata = {
                "source_type": "pdf",
                "total_pages": len(reader.pages),
                "parsed_at": datetime.utcnow().isoformat(),
            }

            # Extract document metadata
            if reader.metadata:
                metadata.update({
                    "title": reader.metadata.get("/Title", ""),
                    "author": reader.metadata.get("/Author", ""),
                    "subject": reader.metadata.get("/Subject", ""),
                })

            for i, page in enumerate(reader.pages):
                text = page.extract_text() or ""

                if text.strip():
                    doc = Document(
                        content=text,
                        metadata={
                            **metadata,
                            "page": i + 1,
                            "source": f"page_{i + 1}",
                        },
                    )
                    documents.append(doc)

        return documents


class HTMLParser(BaseParser):
    """
    HTML document parser.

    Features:
    - Text extraction with structure preservation
    - Link extraction
    - Metadata from meta tags
    - Script/style removal
    """

    def __init__(
        self,
        extract_links: bool = True,
        preserve_headings: bool = True,
        remove_scripts: bool = True,
        remove_styles: bool = True,
    ) -> None:
        self.extract_links = extract_links
        self.preserve_headings = preserve_headings
        self.remove_scripts = remove_scripts
        self.remove_styles = remove_styles

        self._beautifulsoup = None
        try:
            from bs4 import BeautifulSoup
            self._beautifulsoup = BeautifulSoup
        except ImportError:
            logger.warning("BeautifulSoup not installed. HTML parsing unavailable.")

    def supports(self, source: Union[str, Path]) -> bool:
        if isinstance(source, str):
            return source.lower().endswith((".html", ".htm")) or source.startswith("http")
        return isinstance(source, Path) and source.suffix.lower() in (".html", ".htm")

    async def parse(self, source: Union[str, bytes, Path]) -> List[Document]:
        """Parse HTML into documents."""
        if not self._beautifulsoup:
            raise RuntimeError("BeautifulSoup not installed.")

        # Load HTML content
        if isinstance(source, bytes):
            html_content = source.decode("utf-8")
        elif isinstance(source, Path):
            with open(source, "r", encoding="utf-8") as f:
                html_content = f.read()
        elif source.startswith("http"):
            # Fetch URL
            async with httpx.AsyncClient() as client:
                response = await client.get(source)
                response.raise_for_status()
                html_content = response.text
        else:
            html_content = source

        soup = self._beautifulsoup(html_content, "html.parser")

        # Remove unwanted elements
        if self.remove_scripts:
            for script in soup(["script", "noscript"]):
                script.decompose()

        if self.remove_styles:
            for style in soup(["style", "link"]):
                style.decompose()

        # Extract metadata
        metadata = {
            "source_type": "html",
            "parsed_at": datetime.utcnow().isoformat(),
        }

        title = soup.find("title")
        if title:
            metadata["title"] = title.get_text(strip=True)

        for meta in soup.find_all("meta"):
            name = meta.get("name", "")
            content = meta.get("content", "")
            if name and content:
                metadata[f"meta_{name}"] = content

        # Extract links if enabled
        links = []
        if self.extract_links:
            for a in soup.find_all("a", href=True):
                links.append({
                    "text": a.get_text(strip=True),
                    "href": a["href"],
                })
            metadata["links"] = links

        # Extract text with heading preservation
        text_parts = []
        for element in soup.body.find_all(["p", "h1", "h2", "h3", "h4", "h5", "h6", "li", "pre"]):
            text = element.get_text(" ", strip=True)
            if text:
                tag = element.name
                if self.preserve_headings and tag.startswith("h"):
                    level = tag[1]
                    text = f"\n{'#' * int(level)} {text}\n"
                text_parts.append(text)

        content = "\n\n".join(text_parts)

        if not content.strip():
            return []

        doc = Document(
            content=content,
            metadata=metadata,
        )

        logger.info(f"Parsed HTML: {len(content)} characters, {len(links)} links")
        return [doc]


class MarkdownParser(BaseParser):
    """
    Markdown document parser.

    Features:
    - Frontmatter extraction
    - Structure preservation
    - Code block handling
    - Link/image reference extraction
    """

    def __init__(
        self,
        extract_frontmatter: bool = True,
        preserve_code: bool = True,
        max_code_length: int = 10000,
    ) -> None:
        self.extract_frontmatter = extract_frontmatter
        self.preserve_code = preserve_code
        self.max_code_length = max_code_length

    def supports(self, source: Union[str, Path]) -> bool:
        if isinstance(source, str):
            return source.lower().endswith((".md", ".markdown", ".mdx"))
        return isinstance(source, Path) and source.suffix.lower() in (".md", ".markdown", ".mdx")

    async def parse(self, source: Union[str, bytes, Path]) -> List[Document]:
        """Parse Markdown into documents."""
        # Load content
        if isinstance(source, bytes):
            content = source.decode("utf-8")
        elif isinstance(source, Path):
            with open(source, "r", encoding="utf-8") as f:
                content = f.read()
        else:
            content = source

        metadata = {
            "source_type": "markdown",
            "parsed_at": datetime.utcnow().isoformat(),
        }

        # Extract frontmatter
        frontmatter = {}
        if self.extract_frontmatter and content.startswith("---"):
            parts = content.split("---", 2)
            if len(parts) >= 3:
                frontmatter_text = parts[1].strip()
                content = parts[2].strip()

                # Parse YAML frontmatter
                try:
                    import yaml
                    frontmatter = yaml.safe_load(frontmatter_text) or {}
                except ImportError:
                    # Simple key-value parsing
                    for line in frontmatter_text.split("\n"):
                        if ":" in line:
                            key, value = line.split(":", 1)
                            frontmatter[key.strip()] = value.strip()

        metadata.update(frontmatter)

        # Extract code blocks if preserving
        code_blocks = []
        if self.preserve_code:
            code_pattern = r"```(\w+)?\n([\s\S]*?)```"
            for match in re.finditer(code_pattern, content):
                language = match.group(1) or "text"
                code = match.group(2)
                if len(code) <= self.max_code_length:
                    code_blocks.append({
                        "language": language,
                        "code": code,
                    })
            metadata["code_blocks"] = code_blocks

        # Extract links and images
        links = re.findall(r"\[([^\]]+)\]\(([^)]+)\)", content)
        images = re.findall(r"!\[([^\]]*)\]\(([^)]+)\)", content)

        metadata["links"] = [{"text": t, "url": u} for t, u in links]
        metadata["images"] = [{"alt": a, "url": u} for a, u in images]

        doc = Document(
            content=content,
            metadata=metadata,
        )

        logger.info(f"Parsed Markdown: {len(content)} characters")
        return [doc]


class JSONParser(BaseParser):
    """
    JSON document parser.

    Features:
    - Nested structure flattening
    - Array handling
    - Schema detection
    """

    def __init__(
        self,
        flatten: bool = True,
        text_fields: Optional[List[str]] = None,
        separator: str = " > ",
    ) -> None:
        self.flatten = flatten
        self.text_fields = text_fields or ["text", "content", "body", "description", "title"]
        self.separator = separator

    def supports(self, source: Union[str, Path]) -> bool:
        if isinstance(source, str):
            return source.lower().endswith(".json")
        return isinstance(source, Path) and source.suffix.lower() == ".json"

    async def parse(self, source: Union[str, bytes, Path]) -> List[Document]:
        """Parse JSON into documents."""
        # Load JSON
        if isinstance(source, bytes):
            data = json.loads(source.decode("utf-8"))
        elif isinstance(source, Path):
            with open(source, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            data = json.loads(source)

        documents = []

        if isinstance(data, list):
            # Handle array of items
            for i, item in enumerate(data):
                docs = self._process_item(item, {"index": i, "source_type": "json_array"})
                documents.extend(docs)
        else:
            # Handle single object
            documents = self._process_item(data, {"source_type": "json_object"})

        logger.info(f"Parsed JSON: {len(documents)} documents")
        return documents

    def _process_item(
        self,
        item: Any,
        base_metadata: Dict[str, Any],
        path: str = "",
    ) -> List[Document]:
        """Process a JSON item recursively."""
        documents = []

        if isinstance(item, dict):
            # Look for text fields
            text_content = []
            remaining = {}

            for key, value in item.items():
                current_path = f"{path}{self.separator}{key}" if path else key

                if key in self.text_fields and isinstance(value, str):
                    text_content.append(f"{key}: {value}")
                elif isinstance(value, (dict, list)):
                    # Recurse into nested structures
                    nested_docs = self._process_item(value, base_metadata.copy(), current_path)
                    documents.extend(nested_docs)
                else:
                    remaining[key] = value

            if text_content:
                content = "\n".join(text_content)
                doc = Document(
                    content=content,
                    metadata={
                        **base_metadata,
                        "path": path or "root",
                        "fields": list(remaining.keys()),
                    },
                )
                documents.append(doc)

        elif isinstance(item, list):
            for i, elem in enumerate(item):
                nested_docs = self._process_item(elem, base_metadata.copy(), f"{path}[{i}]")
                documents.extend(nested_docs)

        elif isinstance(item, str) and path:
            doc = Document(
                content=item,
                metadata={
                    **base_metadata,
                    "path": path,
                },
            )
            documents.append(doc)

        return documents


class GitHubConnector:
    """
    GitHub API connector for repository ingestion.

    Features:
    - Repository file listing
    - File content fetching
    - Issue/PR extraction
    - Rate limit handling
    """

    def __init__(
        self,
        token: Optional[str] = None,
        base_url: str = "https://api.github.com",
        rate_limit_delay: float = 0.1,
    ) -> None:
        self.token = token or os.getenv("GITHUB_TOKEN")
        self.base_url = base_url
        self.rate_limit_delay = rate_limit_delay
        self._last_request_time = 0.0

        self._headers = {
            "Accept": "application/vnd.github.v3+json",
        }
        if self.token:
            self._headers["Authorization"] = f"token {self.token}"

    async def _rate_limit_wait(self) -> None:
        """Enforce rate limiting."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.rate_limit_delay:
            await asyncio.sleep(self.rate_limit_delay - elapsed)
        self._last_request_time = time.time()

    async def list_files(
        self,
        owner: str,
        repo: str,
        path: str = "",
        ref: str = "main",
    ) -> List[Dict[str, Any]]:
        """List files in a repository path."""
        await self._rate_limit_wait()

        url = f"{self.base_url}/repos/{owner}/{repo}/contents/{path}"
        params = {"ref": ref}

        async with httpx.AsyncClient() as client:
            response = await client.get(url, headers=self._headers, params=params)
            response.raise_for_status()
            data = response.json()

        if isinstance(data, list):
            return data
        return [data]

    async def get_file_content(
        self,
        owner: str,
        repo: str,
        path: str,
        ref: str = "main",
    ) -> str:
        """Get file content from repository."""
        await self._rate_limit_wait()

        url = f"{self.base_url}/repos/{owner}/{repo}/contents/{path}"
        params = {"ref": ref}

        async with httpx.AsyncClient() as client:
            response = await client.get(url, headers=self._headers, params=params)
            response.raise_for_status()
            data = response.json()

        # Decode base64 content
        import base64
        content = base64.b64decode(data["content"]).decode("utf-8")
        return content

    async def ingest_repository(
        self,
        owner: str,
        repo: str,
        extensions: Optional[List[str]] = None,
        max_files: int = 100,
        ref: str = "main",
    ) -> AsyncGenerator[Document, None]:
        """
        Ingest files from a GitHub repository.

        Args:
            owner: Repository owner
            repo: Repository name
            extensions: File extensions to include
            max_files: Maximum files to ingest
            ref: Branch or tag

        Yields:
            Document objects
        """
        extensions = extensions or [".md", ".txt", ".py", ".js", ".ts", ".json", ".yaml", ".yml"]
        files_processed = 0

        async def traverse(path: str = "") -> None:
            nonlocal files_processed

            if files_processed >= max_files:
                return

            items = await self.list_files(owner, repo, path, ref)

            for item in items:
                if files_processed >= max_files:
                    break

                if item["type"] == "dir":
                    await traverse(item["path"])
                elif any(item["name"].endswith(ext) for ext in extensions):
                    try:
                        content = await self.get_file_content(owner, repo, item["path"], ref)

                        doc = Document(
                            content=content,
                            metadata={
                                "source_type": "github",
                                "owner": owner,
                                "repo": repo,
                                "path": item["path"],
                                "url": item.get("html_url", ""),
                                "sha": item.get("sha", ""),
                            },
                        )
                        files_processed += 1
                        yield doc
                    except Exception as e:
                        logger.warning(f"Failed to ingest {item['path']}: {e}")

        async for doc in traverse():
            yield doc


class GoogleDriveConnector:
    """
    Google Drive API connector for file ingestion.

    Requires Google Cloud credentials setup.
    """

    def __init__(
        self,
        credentials_path: Optional[str] = None,
        scopes: Optional[List[str]] = None,
    ) -> None:
        self.credentials_path = credentials_path or os.getenv("GOOGLE_CREDENTIALS_PATH")
        self.scopes = scopes or ["https://www.googleapis.com/auth/drive.readonly"]

        self._service = None
        self._creds = None

    async def authenticate(self) -> None:
        """Authenticate with Google Drive API."""
        try:
            from google.oauth2 import service_account
            from googleapiclient.discovery import build

            if not self.credentials_path:
                raise ValueError("Google credentials path not provided")

            self._creds = service_account.Credentials.from_service_account_file(
                self.credentials_path,
                scopes=self.scopes,
            )
            self._service = build("drive", "v3", credentials=self._creds)

            logger.info("Authenticated with Google Drive API")
        except ImportError:
            raise RuntimeError(
                "Google API libraries not installed. "
                "Run: pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib"
            )

    async def list_files(
        self,
        folder_id: str = "root",
        mime_types: Optional[List[str]] = None,
        query: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """List files in a folder."""
        if not self._service:
            await self.authenticate()

        q = f"'{folder_id}' in parents and trashed = false"
        if mime_types:
            mime_query = " or ".join(f"mimeType = '{m}'" for m in mime_types)
            q += f" and ({mime_query})"
        if query:
            q += f" and {query}"

        results = []
        page_token = None

        while True:
            response = self._service.files().list(
                q=q,
                pageSize=100,
                fields="nextPageToken, files(id, name, mimeType, size, modifiedTime)",
                pageToken=page_token,
            ).execute()

            results.extend(response.get("files", []))
            page_token = response.get("nextPageToken")

            if not page_token:
                break

        return results

    async def download_file(self, file_id: str) -> bytes:
        """Download file content."""
        if not self._service:
            await self.authenticate()

        request = self._service.files().get_media(fileId=file_id)

        import io
        from googleapiclient.http import MediaIoBaseDownload

        buffer = io.BytesIO()
        downloader = MediaIoBaseDownload(buffer, request)

        done = False
        while not done:
            status, done = downloader.next_chunk()

        return buffer.getvalue()

    async def ingest_folder(
        self,
        folder_id: str = "root",
        mime_types: Optional[List[str]] = None,
    ) -> AsyncGenerator[Document, None]:
        """
        Ingest files from a Google Drive folder.

        Yields:
            Document objects
        """
        mime_types = mime_types or [
            "text/plain",
            "application/pdf",
            "text/markdown",
            "application/vnd.google-apps.document",
        ]

        files = await self.list_files(folder_id, mime_types)

        for file_info in files:
            try:
                content_bytes = await self.download_file(file_info["id"])

                # Try to decode as text
                try:
                    content = content_bytes.decode("utf-8")
                except UnicodeDecodeError:
                    # Skip binary files
                    continue

                doc = Document(
                    content=content,
                    metadata={
                        "source_type": "google_drive",
                        "file_id": file_info["id"],
                        "file_name": file_info["name"],
                        "mime_type": file_info["mimeType"],
                        "modified": file_info.get("modifiedTime", ""),
                    },
                )
                yield doc
            except Exception as e:
                logger.warning(f"Failed to ingest {file_info['name']}: {e}")


class DocumentIngestor:
    """
    Main document ingestion orchestrator.

    Features:
    - Multi-format support
    - Parallel processing
    - Progress tracking
    - Deduplication
    """

    def __init__(
        self,
        parsers: Optional[List[BaseParser]] = None,
        max_concurrent: int = 5,
        deduplicate: bool = True,
    ) -> None:
        self.parsers = parsers or self._default_parsers()
        self.max_concurrent = max_concurrent
        self.deduplicate = deduplicate
        self._seen_hashes: set = set()

    def _default_parsers(self) -> List[BaseParser]:
        """Create default parser set."""
        return [
            PDFParser(),
            HTMLParser(),
            MarkdownParser(),
            JSONParser(),
        ]

    def get_parser(self, source: Union[str, Path]) -> Optional[BaseParser]:
        """Get appropriate parser for source."""
        for parser in self.parsers:
            if parser.supports(source):
                return parser
        return None

    async def ingest(
        self,
        sources: List[Union[str, Path]],
        progress_callback: Optional[callable] = None,
    ) -> List[Document]:
        """
        Ingest multiple sources.

        Args:
            sources: List of file paths or URLs
            progress_callback: Optional callback for progress updates

        Returns:
            List of ingested documents
        """
        documents = []
        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def ingest_source(source: Union[str, Path]) -> List[Document]:
            async with semaphore:
                parser = self.get_parser(source)
                if not parser:
                    logger.warning(f"No parser for: {source}")
                    return []

                try:
                    docs = await parser.parse(source)

                    # Deduplicate
                    if self.deduplicate:
                        unique_docs = []
                        for doc in docs:
                            if doc.id not in self._seen_hashes:
                                self._seen_hashes.add(doc.id)
                                unique_docs.append(doc)
                        docs = unique_docs

                    if progress_callback:
                        progress_callback(len(documents), len(sources))

                    return docs
                except Exception as e:
                    logger.error(f"Failed to ingest {source}: {e}")
                    return []

        tasks = [ingest_source(source) for source in sources]
        results = await asyncio.gather(*tasks)

        for docs in results:
            documents.extend(docs)

        logger.info(f"Ingested {len(documents)} documents from {len(sources)} sources")
        return documents

    async def ingest_directory(
        self,
        directory: Union[str, Path],
        recursive: bool = True,
        pattern: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """
        Ingest all supported files in a directory.

        Args:
            directory: Directory path
            recursive: Whether to search recursively
            pattern: Optional glob pattern
            **kwargs: Passed to ingest()
        """
        directory = Path(directory)
        sources = []

        if recursive:
            glob_pattern = "**/*"
        else:
            glob_pattern = "*"

        if pattern:
            glob_pattern = f"{glob_pattern}{pattern}"

        for file_path in directory.glob(glob_pattern):
            if file_path.is_file():
                parser = self.get_parser(str(file_path))
                if parser:
                    sources.append(file_path)

        return await self.ingest(sources, **kwargs)

    def clear_cache(self) -> None:
        """Clear deduplication cache."""
        self._seen_hashes.clear()


# Import time for rate limiting
import time
