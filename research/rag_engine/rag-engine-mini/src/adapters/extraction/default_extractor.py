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
        """Extract text and tables from PDF using PyMuPDF."""
        try:
            doc = fitz.open(file_path)
            pages = []
            tables_count = 0
            
            for i, page in enumerate(doc, start=1):
                # 1. Extract plain text
                text = page.get_text("text") or ""
                
                # 2. Extract Tables (Stage 4)
                # We find tables and convert them to Markdown to preserve structure
                tabs = page.find_tables()
                table_mds = []
                for tab in tabs:
                    df = tab.to_pandas() # Requires pandas
                    if not df.empty:
                        table_mds.append(f"\n[Table found on Page {i}]:\n" + df.to_markdown(index=False))
                        tables_count += 1
                
                page_content = f"\n--- Page {i} ---\n{text}"
                if table_mds:
                    page_content += "\n" + "\n".join(table_mds)
                    
                if page_content.strip():
                    pages.append(page_content)
            
            full_text = "\n".join(pages).strip()
            
            return ExtractedText(
                text=full_text,
                metadata={
                    "type": "pdf",
                    "pages": len(doc),
                    "engine": "pymupdf",
                    "tables_detected": tables_count,
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
