"""
Code-Aware Chunking Strategy

Implements language-aware code splitting that preserves structure.

Best for:
- Code documentation and repositories
- Technical manuals with code examples
- When preserving function/class boundaries is important

How it works:
1. Detect programming language (or use specified)
2. Split at structural boundaries (functions, classes, etc.)
3. Validate syntax (balanced braces, parens)
4. Preserve language-specific constructs

Supported Languages:
- Python
- JavaScript/TypeScript
- Java
- C/C++

Example:
    >>> from src.rag.chunking import CodeChunker, ChunkingConfig
    >>> config = ChunkingConfig(
    ...     chunk_size=1000,
    ...     language="python",
    ... )
    >>> chunker = CodeChunker(config)
    >>> chunks = chunker.chunk({"id": "code1", "content": "def foo():..."})
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from .base import BaseChunker, Chunk, ChunkingConfig
from .recursive import RecursiveChunker

logger = logging.getLogger(__name__)


class CodeChunker(BaseChunker):
    """
    Code-aware chunking with language-specific handling.

    This strategy understands code structure and splits at
    appropriate boundaries to preserve:
    - Function definitions
    - Class definitions
    - Import statements
    - Logical code blocks

    Attributes:
        config: Chunking configuration
        language: Programming language for chunking

    Example:
        >>> chunker = CodeChunker(
        ...     ChunkingConfig(language="python", chunk_size=1000)
        ... )
        >>> chunks = chunker.chunk({
        ...     "id": "main.py",
        ...     "content": "def main():\\n    pass"
        ... })
    """

    # Language-specific separators
    LANGUAGE_SEPARATORS = {
        "python": [
            "\nclass ",
            "\ndef ",
            "\nasync def ",
            "\n@decorator",
            "\n@staticmethod",
            "\n@classmethod",
            "\n\n",  # Blank lines
            "\n",
            " ",
            "",
        ],
        "javascript": [
            "\nfunction ",
            "\nasync function ",
            "\nconst ",
            "\nlet ",
            "\nvar ",
            "\nclass ",
            "\nexport ",
            "\nimport ",
            "\n\n",
            "\n",
            " ",
            "",
        ],
        "typescript": [
            "\nfunction ",
            "\nasync function ",
            "\nconst ",
            "\nlet ",
            "\nvar ",
            "\nclass ",
            "\ninterface ",
            "\ntype ",
            "\nenum ",
            "\nexport ",
            "\nimport ",
            "\n\n",
            "\n",
            " ",
            "",
        ],
        "java": [
            "\npublic class ",
            "\nprivate class ",
            "\nprotected class ",
            "\npublic ",
            "\nprivate ",
            "\nprotected ",
            "\nstatic ",
            "\n\n",
            "\n",
            " ",
            "",
        ],
        "cpp": [
            "\nclass ",
            "\nstruct ",
            "\nvoid ",
            "\nint ",
            "\nfloat ",
            "\ndouble ",
            "\nchar ",
            "\ntemplate ",
            "\nnamespace ",
            "\n\n",
            "\n",
            " ",
            "",
        ],
        "go": [
            "\nfunc ",
            "\ntype ",
            "\nconst ",
            "\nvar ",
            "\nimport ",
            "\npackage ",
            "\n\n",
            "\n",
            " ",
            "",
        ],
        "rust": [
            "\nfn ",
            "\npub fn ",
            "\nstruct ",
            "\npub struct ",
            "\nenum ",
            "\npub enum ",
            "\nimpl ",
            "\ntrait ",
            "\nmod ",
            "\n\n",
            "\n",
            " ",
            "",
        ],
    }

    # Language detection patterns
    LANGUAGE_PATTERNS = {
        "python": [
            r"\bdef\s+\w+\s*\(",
            r"\bclass\s+\w+\s*[:\(]",
            r"\bimport\s+\w+",
            r"\bfrom\s+\w+\s+import",
            r"^\s*#",  # Comments
        ],
        "javascript": [
            r"\bfunction\s+\w*\s*\(",
            r"\bconst\s+\w+\s*=",
            r"\blet\s+\w+\s*=",
            r"\bvar\s+\w+\s*=",
            r"=>",
            r"\bimport\s+",
            r"\bexport\s+",
        ],
        "typescript": [
            r"\binterface\s+\w+",
            r"\btype\s+\w+\s*=",
            r":\s*(string|number|boolean|any)\b",
            r"<\w+>",
        ],
        "java": [
            r"\bpublic\s+class\s+",
            r"\bprivate\s+class\s+",
            r"\bSystem\.out\.println",
            r"\bString\[\]\s+args",
            r"\b@Override",
        ],
        "cpp": [
            r"#include\s*<",
            r"\bstd::",
            r"\bcout\s*<<",
            r"\bcin\s*>>",
            r"\btemplate\s*<",
        ],
        "go": [
            r"\bfunc\s+\w*\s*\(",
            r"\bpackage\s+\w+",
            r"\bimport\s+\(",
            r":=\s*",
        ],
        "rust": [
            r"\bfn\s+\w+\s*\(",
            r"\blet\s+mut\s+",
            r"\bimpl\s+",
            r"\btrait\s+",
            r"->\s*\w+",
        ],
    }

    def __init__(
        self,
        config: Optional[ChunkingConfig] = None,
        language: Optional[str] = None,
    ) -> None:
        """
        Initialize the code chunker.

        Args:
            config: Chunking configuration
            language: Programming language (auto-detected if not provided)
        """
        super().__init__(config)

        if language:
            self.language = language.lower()
        else:
            self.language = self.config.language.lower()

        self._logger = logging.getLogger(self.__class__.__name__)

        # Get separators for language
        self._separators = self.LANGUAGE_SEPARATORS.get(
            self.language,
            self.LANGUAGE_SEPARATORS["python"],
        )

    def chunk(self, document: Dict[str, Any]) -> List[Chunk]:
        """
        Split code document at structural boundaries.

        Args:
            document: Document dictionary with 'id' and 'content' fields

        Returns:
            List of Chunk objects preserving code structure

        Raises:
            ValueError: If document is invalid
        """
        self._validate_document(document)

        content = document.get("content", "")
        doc_id = document.get("id", "unknown")

        # Auto-detect language if needed
        if self.language == "auto":
            self.language = self._detect_language(content)
            self._separators = self.LANGUAGE_SEPARATORS.get(
                self.language,
                self.LANGUAGE_SEPARATORS["python"],
            )
            self._logger.debug(
                f"Auto-detected language: {self.language} for {doc_id}"
            )

        self._logger.debug(
            f"Starting code chunking for {doc_id} "
            f"(language: {self.language})"
        )

        # Split using recursive with code-aware separators
        chunk_texts = self._split_code(content)

        # Validate and create chunks
        chunks = []
        current_offset = 0

        for chunk_text in chunk_texts:
            # Validate code chunk
            if not self._is_valid_code_chunk(chunk_text):
                self._logger.debug(
                    f"Skipping invalid code chunk in {doc_id}: "
                    f"{chunk_text[:50]}..."
                )
                continue

            chunk = self._create_chunk(
                content=chunk_text,
                document=document,
                start_index=current_offset,
                end_index=current_offset + len(chunk_text),
                extra_metadata={
                    "chunk_method_detail": "code",
                    "language": self.language,
                    "has_functions": self._has_functions(chunk_text),
                    "has_classes": self._has_classes(chunk_text),
                },
            )
            chunks.append(chunk)
            current_offset += len(chunk_text)

        self._logger.info(
            f"Created {len(chunks)} code chunks from document {doc_id} "
            f"({len(content)} characters)"
        )

        return chunks

    def _split_code(self, code: str) -> List[str]:
        """
        Split code at structural boundaries.

        Args:
            code: Code to split

        Returns:
            List of code chunks
        """
        # Use recursive splitter with code-aware separators
        splitter = RecursiveChunker(
            ChunkingConfig(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap,
            ),
            separators=self._separators,
        )

        chunks = splitter.split_text(code)

        # Post-process to ensure valid code blocks
        validated = self._validate_code_chunks(chunks)

        return validated

    def _detect_language(self, code: str) -> str:
        """
        Detect programming language from code content.

        Args:
            code: Code to analyze

        Returns:
            Detected language name
        """
        scores: Dict[str, int] = {}

        for language, patterns in self.LANGUAGE_PATTERNS.items():
            score = 0
            for pattern in patterns:
                matches = re.findall(pattern, code, re.MULTILINE)
                score += len(matches)
            scores[language] = score

        if not scores or max(scores.values()) == 0:
            return "python"  # Default

        detected = max(scores, key=scores.get)
        self._logger.debug(f"Language detection scores: {scores}")

        return detected

    def _validate_code_chunks(self, chunks: List[str]) -> List[str]:
        """
        Validate and fix code chunks.

        Checks for:
        - Balanced braces
        - Balanced parentheses
        - Balanced quotes

        Args:
            chunks: List of code chunks

        Returns:
            List of validated chunks
        """
        validated = []

        for chunk in chunks:
            if self._is_valid_code_chunk(chunk):
                validated.append(chunk)
            else:
                # Try to fix
                fixed = self._try_fix_chunk(chunk)
                if fixed:
                    validated.append(fixed)
                else:
                    self._logger.debug(
                        f"Skipping unfixable chunk: {chunk[:50]}..."
                    )

        return validated

    def _is_valid_code_chunk(self, chunk: str) -> bool:
        """
        Check if a code chunk is syntactically valid.

        Args:
            chunk: Code chunk to validate

        Returns:
            True if valid, False otherwise
        """
        if not chunk.strip():
            return False

        # Skip very small chunks
        if len(chunk.strip()) < 10:
            return False

        # Check balanced braces
        if not self._has_balanced_braces(chunk):
            return False

        # Check balanced parentheses
        if not self._has_balanced_parens(chunk):
            return False

        # Check balanced brackets
        if not self._has_balanced_brackets(chunk):
            return False

        return True

    def _has_balanced_braces(self, text: str) -> bool:
        """Check if curly braces are balanced."""
        count = 0
        in_string = False
        string_char = None
        escape_next = False

        for char in text:
            if escape_next:
                escape_next = False
                continue

            if char == "\\":
                escape_next = True
                continue

            if char in '"\'':
                if not in_string:
                    in_string = True
                    string_char = char
                elif char == string_char:
                    in_string = False
            elif not in_string:
                if char == "{":
                    count += 1
                elif char == "}":
                    count -= 1
                    if count < 0:
                        return False

        return count == 0

    def _has_balanced_parens(self, text: str) -> bool:
        """Check if parentheses are balanced."""
        count = 0
        in_string = False
        escape_next = False

        for char in text:
            if escape_next:
                escape_next = False
                continue

            if char == "\\":
                escape_next = True
                continue

            if char in '"\'':
                if not in_string:
                    in_string = True
                elif not escape_next:
                    in_string = False
            elif not in_string:
                if char == "(":
                    count += 1
                elif char == ")":
                    count -= 1
                    if count < 0:
                        return False

        return count == 0

    def _has_balanced_brackets(self, text: str) -> bool:
        """Check if square brackets are balanced."""
        count = 0
        in_string = False

        for char in text:
            if char in '"\'':
                in_string = not in_string
            elif not in_string:
                if char == "[":
                    count += 1
                elif char == "]":
                    count -= 1
                    if count < 0:
                        return False

        return count == 0

    def _try_fix_chunk(self, chunk: str) -> Optional[str]:
        """
        Try to fix an invalid code chunk.

        Strategies:
        - Remove incomplete function/class definitions
        - Add missing closing braces
        - Remove orphaned code

        Args:
            chunk: Code chunk to fix

        Returns:
            Fixed chunk or None if unfixable
        """
        if not chunk.strip():
            return None

        lines = chunk.split("\n")
        fixed_lines = []
        skip_until_dedent = False
        expected_indent = 0

        for line in lines:
            stripped = line.strip()

            # Skip incomplete definitions
            if re.match(r"^(def|class|function|struct|interface)\s+\w+", stripped):
                if not stripped.endswith((":","{","}")):
                    # Incomplete definition, skip
                    skip_until_dedent = True
                    expected_indent = len(line) - len(line.lstrip())
                    continue

            if skip_until_dedent:
                current_indent = len(line) - len(line.lstrip())
                if current_indent <= expected_indent and stripped:
                    skip_until_dedent = False
                else:
                    continue

            fixed_lines.append(line)

        fixed = "\n".join(fixed_lines)

        if fixed.strip() and len(fixed.strip()) > 10:
            return fixed

        return None

    def _has_functions(self, code: str) -> bool:
        """Check if code contains function definitions."""
        patterns = [
            r"\bdef\s+\w+\s*\(",  # Python
            r"\bfunction\s+\w*\s*\(",  # JavaScript
            r"\b(func|fn)\s+\w+\s*\(",  # Go/Rust
        ]
        return any(re.search(p, code) for p in patterns)

    def _has_classes(self, code: str) -> bool:
        """Check if code contains class definitions."""
        patterns = [
            r"\bclass\s+\w+",  # Python, JavaScript, Java
            r"\bstruct\s+\w+",  # C++, Rust
            r"\binterface\s+\w+",  # TypeScript, Java
        ]
        return any(re.search(p, code) for p in patterns)


def create_code_chunker(
    chunk_size: int = 1000,
    chunk_overlap: int = 100,
    language: str = "auto",
    min_chunk_size: int = 50,
) -> CodeChunker:
    """
    Factory function to create a CodeChunker.

    Args:
        chunk_size: Target chunk size in tokens
        chunk_overlap: Overlap between chunks in tokens
        language: Programming language or 'auto' for detection
        min_chunk_size: Minimum acceptable chunk size

    Returns:
        Configured CodeChunker instance

    Example:
        >>> chunker = create_code_chunker(
        ...     language="python",
        ...     chunk_size=512,
        ... )
    """
    config = ChunkingConfig(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        min_chunk_size=min_chunk_size,
        language=language,
    )

    return CodeChunker(config, language=language)
