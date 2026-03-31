"""
Text Extraction Interface
==========================
Protocol for text extraction from documents.

واجهة استخراج النص من المستندات
"""

from typing import Protocol

from src.domain.entities import ExtractedText


class TextExtractor(Protocol):
    """
    Protocol for extracting text from various document formats.
    
    Implementations: PDF, DOCX, TXT extractors
    
    Design Decision: Protocol-based for flexibility.
    Different extractors can be composed or chained.
    
    قرار التصميم: قائم على البروتوكول للمرونة
    """
    
    def extract(self, file_path: str, content_type: str) -> ExtractedText:
        """
        Extract text from a file.
        
        Args:
            file_path: Path to the file
            content_type: MIME type of the file
        
        Returns:
            ExtractedText with text and metadata
        
        Raises:
            TextExtractionError: If extraction fails
            UnsupportedFileTypeError: If file type is not supported
        """
        ...
