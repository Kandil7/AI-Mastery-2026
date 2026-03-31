"""
Module 3.2: Building Vector Storage

Production-ready implementations for building vector storage systems:
- Ingestion: Document parsers and API connectors
- Splitting: Text chunking strategies
- Embeddings: Embedding generation and caching
- Vector DB: Vector database integrations
"""

from .ingestion import (
    DocumentIngestor,
    PDFParser,
    HTMLParser,
    MarkdownParser,
    JSONParser,
    GitHubConnector,
    GoogleDriveConnector,
    Document,
)
from .splitting import (
    TextSplitter,
    RecursiveSplitter,
    SemanticSplitter,
    TokenSplitter,
    CodeSplitter,
)
from .embeddings import (
    EmbeddingGenerator,
    SentenceTransformerEmbeddings,
    OpenAIEmbeddings,
    EmbeddingCache,
)
from .vector_db import (
    VectorDatabase,
    QdrantClient,
    FAISSClient,
    ChromaClient,
    PineconeClient,
    HybridSearchConfig,
)

__all__ = [
    # Ingestion
    "DocumentIngestor",
    "PDFParser",
    "HTMLParser",
    "MarkdownParser",
    "JSONParser",
    "GitHubConnector",
    "GoogleDriveConnector",
    "Document",
    # Splitting
    "TextSplitter",
    "RecursiveSplitter",
    "SemanticSplitter",
    "TokenSplitter",
    "CodeSplitter",
    # Embeddings
    "EmbeddingGenerator",
    "SentenceTransformerEmbeddings",
    "OpenAIEmbeddings",
    "EmbeddingCache",
    # Vector DB
    "VectorDatabase",
    "QdrantClient",
    "FAISSClient",
    "ChromaClient",
    "PineconeClient",
    "HybridSearchConfig",
]

__version__ = "1.0.0"
