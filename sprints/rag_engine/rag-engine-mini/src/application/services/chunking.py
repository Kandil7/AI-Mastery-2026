"""
Token-Aware Chunking Service
=============================
Pure service for splitting text into chunks.

خدمة تقطيع النص المدركة للرموز
"""

from typing import List

import tiktoken

from src.domain.entities import ChunkSpec


def chunk_text_token_aware(
    text: str,
    spec: ChunkSpec | None = None,
) -> List[str]:
    """
    Split text into token-aware chunks with overlap.
    
    Args:
        text: Full text to chunk
        spec: Chunking specification (defaults to 512 tokens, 50 overlap)
    
    Returns:
        List of text chunks
    
    Design Decision: Token-aware chunking instead of character-based:
    - Maps directly to model limits
    - More consistent chunk sizes
    - Proper handling of multi-byte characters
    
    قرار التصميم: تقطيع مدرك للرموز بدلاً من الأحرف
    
    Example:
        >>> chunks = chunk_text_token_aware(long_text, ChunkSpec(max_tokens=512, overlap_tokens=50))
        >>> len(chunks)
        5
    """
    if spec is None:
        spec = ChunkSpec()
    
    if not text or not text.strip():
        return []
    
    # Get tiktoken encoding
    try:
        enc = tiktoken.get_encoding(spec.encoding_name)
    except Exception:
        # Fallback to default encoding
        enc = tiktoken.get_encoding("cl100k_base")
    
    # Tokenize
    tokens = enc.encode(text)
    
    if len(tokens) == 0:
        return []
    
    # Chunking parameters
    max_tokens = max(50, spec.max_tokens)  # Minimum 50 tokens per chunk
    overlap = min(spec.overlap_tokens, max_tokens - 1)  # Can't overlap more than chunk size
    
    chunks: List[str] = []
    start = 0
    
    while start < len(tokens):
        # Calculate end position
        end = min(start + max_tokens, len(tokens))
        
        # Extract chunk tokens and decode
        chunk_tokens = tokens[start:end]
        chunk_text = enc.decode(chunk_tokens).strip()
        
        if chunk_text:
            chunks.append(chunk_text)
        
        # If we've reached the end, stop
        if end == len(tokens):
            break
        
        # Move start with overlap
        start = end - overlap
    
    return chunks


def count_tokens(text: str, encoding_name: str = "cl100k_base") -> int:
    """
    Count tokens in text.
    
    Args:
        text: Text to count
        encoding_name: tiktoken encoding name
    
    Returns:
        Token count
    
    عد الرموز في النص
    """
    try:
        enc = tiktoken.get_encoding(encoding_name)
        return len(enc.encode(text))
    except Exception:
        # Rough estimate fallback
        return len(text) // 4


def truncate_to_tokens(
    text: str,
    max_tokens: int,
    encoding_name: str = "cl100k_base",
) -> str:
    """
    Truncate text to fit within token limit.
    
    Args:
        text: Text to truncate
        max_tokens: Maximum tokens
        encoding_name: tiktoken encoding name
    
    Returns:
        Truncated text
    
    قص النص ليناسب حد الرموز
    """
    try:
        enc = tiktoken.get_encoding(encoding_name)
        tokens = enc.encode(text)
        
        if len(tokens) <= max_tokens:
            return text
        
        return enc.decode(tokens[:max_tokens])
    except Exception:
        # Fallback to character-based
        return text[:max_tokens * 4]
