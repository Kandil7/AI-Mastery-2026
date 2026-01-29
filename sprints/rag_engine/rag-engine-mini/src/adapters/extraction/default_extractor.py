"""
Default Text Extractor Adapter
===============================
Text extraction from PDF, DOCX, and TXT files.

محول استخراج النص من PDF و DOCX و TXT
"""

import docx
import fitz  # PyMuPDF

from src.domain.entities import ExtractedText
from src.domain.errors import TextExtractionError, UnsupportedFileTypeError


class DefaultTextExtractor:
    """
    Default text extractor supporting PDF, DOCX, and TXT.
    Uses PyMuPDF for high-quality PDF extraction.
    """
    
    def extract(self, file_path: str, content_type: str) -> ExtractedText:
        # ... (same logic as before)
        lower_path = file_path.lower()
        if content_type == "application/pdf" or lower_path.endswith(".pdf"):
            return self._extract_pdf(file_path)
        
        if lower_path.endswith(".docx") or content_type in (
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ):
            return self._extract_docx(file_path)
        
        if lower_path.endswith(".txt") or content_type.startswith("text/"):
            return self._extract_txt(file_path)
        
        # Unsupported
        extension = file_path.rsplit(".", 1)[-1] if "." in file_path else "unknown"
        raise UnsupportedFileTypeError(extension, ["pdf", "docx", "txt"])
    
    def _extract_pdf(self, file_path: str) -> ExtractedText:
        """Extract text from PDF using PyMuPDF."""
        try:
            doc = fitz.open(file_path)
            pages = []
            
            for i, page in enumerate(doc, start=1):
                text = page.get_text("text") or ""
                if text.strip():
                    pages.append(f"\n--- Page {i} ---\n{text}")
            
            full_text = "\n".join(pages).strip()
            
            return ExtractedText(
                text=full_text,
                metadata={
                    "type": "pdf",
                    "pages": len(doc),
                    "engine": "pymupdf",
                },
            )
        except Exception as e:
            raise TextExtractionError(f"PyMuPDF extraction failed: {e}") from e
    
    def _extract_docx(self, file_path: str) -> ExtractedText:
        """Extract text from DOCX."""
        try:
            doc = docx.Document(file_path)
            paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
            full_text = "\n".join(paragraphs).strip()
            
            return ExtractedText(
                text=full_text,
                metadata={
                    "type": "docx",
                    "paragraphs": len(paragraphs),
                },
            )
        except Exception as e:
            raise TextExtractionError(f"DOCX extraction failed: {e}") from e
    
    def _extract_txt(self, file_path: str) -> ExtractedText:
        """Extract text from TXT file."""
        # Try multiple encodings
        for encoding in ("utf-8", "utf-8-sig", "latin-1", "cp1256"):
            try:
                with open(file_path, "r", encoding=encoding) as f:
                    text = f.read().strip()
                
                return ExtractedText(
                    text=text,
                    metadata={
                        "type": "txt",
                        "encoding": encoding,
                    },
                )
            except UnicodeDecodeError:
                continue
        
        raise TextExtractionError("Unable to decode text file with any encoding")
