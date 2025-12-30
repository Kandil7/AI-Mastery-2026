"""
Data Ingestion Pipeline

This script reads documents from a directory, chunks them, generates embeddings,
and builds an HNSW vector index for the RAG system.
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Generator
from dataclasses import dataclass, asdict
import hashlib
import pickle
from datetime import datetime

import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Document:
    """Represents a document in the RAG system."""
    id: str
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[np.ndarray] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        d = asdict(self)
        if self.embedding is not None:
            d['embedding'] = self.embedding.tolist()
        return d


@dataclass
class Chunk:
    """Represents a text chunk."""
    id: str
    content: str
    doc_id: str
    chunk_index: int
    metadata: Dict[str, Any]


class TextChunker:
    """
    Splits text into overlapping chunks for better retrieval.
    
    Uses sentence-aware splitting to avoid breaking mid-sentence.
    """
    
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        separators: List[str] = None
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", " ", ""]
    
    def split_text(self, text: str) -> List[str]:
        """Split text into chunks with overlap."""
        if len(text) <= self.chunk_size:
            return [text] if text.strip() else []
        
        chunks = []
        
        # Try to split by separators in order
        for separator in self.separators:
            if separator:
                splits = text.split(separator)
            else:
                splits = list(text)
            
            if len(splits) > 1:
                current_chunk = ""
                for split in splits:
                    # Add separator back (except for empty separator)
                    piece = split + (separator if separator else "")
                    
                    if len(current_chunk) + len(piece) <= self.chunk_size:
                        current_chunk += piece
                    else:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                        # Start new chunk with overlap
                        if self.chunk_overlap > 0 and current_chunk:
                            overlap = current_chunk[-self.chunk_overlap:]
                            current_chunk = overlap + piece
                        else:
                            current_chunk = piece
                
                if current_chunk:
                    chunks.append(current_chunk.strip())
                break
        
        return [c for c in chunks if c.strip()]
    
    def chunk_document(self, doc: Document) -> List[Chunk]:
        """Chunk a document into multiple chunks."""
        text_chunks = self.split_text(doc.content)
        chunks = []
        
        for i, text in enumerate(text_chunks):
            chunk_id = f"{doc.id}_chunk_{i}"
            chunks.append(Chunk(
                id=chunk_id,
                content=text,
                doc_id=doc.id,
                chunk_index=i,
                metadata={
                    **doc.metadata,
                    'chunk_index': i,
                    'total_chunks': len(text_chunks)
                }
            ))
        
        return chunks


class EmbeddingModel:
    """
    Wrapper for embedding models.
    
    Supports sentence-transformers if available, falls back to TF-IDF.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self.fallback_mode = False
        
        self._load_model()
    
    def _load_model(self):
        """Load the embedding model."""
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)
            logger.info(f"Loaded sentence-transformer model: {self.model_name}")
        except ImportError:
            logger.warning("sentence-transformers not available, using TF-IDF fallback")
            self.fallback_mode = True
            from sklearn.feature_extraction.text import TfidfVectorizer
            self.model = TfidfVectorizer(max_features=384)
    
    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts to embeddings."""
        if self.fallback_mode:
            # TF-IDF fallback
            if not hasattr(self.model, 'vocabulary_'):
                embeddings = self.model.fit_transform(texts).toarray()
            else:
                embeddings = self.model.transform(texts).toarray()
            return embeddings.astype(np.float32)
        else:
            return self.model.encode(texts, show_progress_bar=True)
    
    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        if self.fallback_mode:
            return self.model.max_features or 384
        return self.model.get_sentence_embedding_dimension()


class HNSWIndex:
    """
    HNSW (Hierarchical Navigable Small World) vector index.
    
    Uses hnswlib if available, falls back to brute-force search.
    """
    
    def __init__(self, dimension: int, max_elements: int = 10000, ef_construction: int = 200, M: int = 16):
        self.dimension = dimension
        self.max_elements = max_elements
        self.ef_construction = ef_construction
        self.M = M
        self.index = None
        self.id_to_doc: Dict[int, str] = {}
        self.doc_to_id: Dict[str, int] = {}
        self.current_id = 0
        self.fallback_mode = False
        self.embeddings: List[np.ndarray] = []
        
        self._initialize_index()
    
    def _initialize_index(self):
        """Initialize the HNSW index."""
        try:
            import hnswlib
            self.index = hnswlib.Index(space='cosine', dim=self.dimension)
            self.index.init_index(
                max_elements=self.max_elements,
                ef_construction=self.ef_construction,
                M=self.M
            )
            self.index.set_ef(50)
            logger.info(f"Initialized HNSW index with dimension={self.dimension}")
        except ImportError:
            logger.warning("hnswlib not available, using numpy fallback (slower)")
            self.fallback_mode = True
    
    def add_items(self, doc_ids: List[str], embeddings: np.ndarray):
        """Add items to the index."""
        if self.fallback_mode:
            for doc_id, embedding in zip(doc_ids, embeddings):
                self.embeddings.append(embedding)
                self.id_to_doc[self.current_id] = doc_id
                self.doc_to_id[doc_id] = self.current_id
                self.current_id += 1
        else:
            internal_ids = []
            for doc_id in doc_ids:
                self.id_to_doc[self.current_id] = doc_id
                self.doc_to_id[doc_id] = self.current_id
                internal_ids.append(self.current_id)
                self.current_id += 1
            
            self.index.add_items(embeddings, internal_ids)
    
    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[tuple]:
        """Search for nearest neighbors."""
        if self.fallback_mode:
            if not self.embeddings:
                return []
            
            # Brute force cosine similarity
            embeddings = np.array(self.embeddings)
            query = query_embedding.reshape(1, -1)
            
            # Cosine similarity
            norms = np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query)
            similarities = np.dot(embeddings, query.T).flatten() / (norms + 1e-10)
            
            # Get top k
            top_indices = np.argsort(similarities)[::-1][:k]
            
            return [(self.id_to_doc[idx], float(similarities[idx])) for idx in top_indices]
        else:
            labels, distances = self.index.knn_query(query_embedding.reshape(1, -1), k=k)
            return [(self.id_to_doc[int(idx)], float(1 - dist)) for idx, dist in zip(labels[0], distances[0])]
    
    def save(self, path: str):
        """Save index to disk."""
        data = {
            'id_to_doc': self.id_to_doc,
            'doc_to_id': self.doc_to_id,
            'current_id': self.current_id,
            'dimension': self.dimension,
            'fallback_mode': self.fallback_mode
        }
        
        if self.fallback_mode:
            data['embeddings'] = self.embeddings
        else:
            index_path = path + '.hnsw'
            self.index.save_index(index_path)
            data['index_path'] = index_path
        
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        
        logger.info(f"Saved index to: {path}")
    
    def load(self, path: str):
        """Load index from disk."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        self.id_to_doc = data['id_to_doc']
        self.doc_to_id = data['doc_to_id']
        self.current_id = data['current_id']
        self.dimension = data['dimension']
        self.fallback_mode = data['fallback_mode']
        
        if self.fallback_mode:
            self.embeddings = data['embeddings']
        else:
            self._initialize_index()
            self.index.load_index(data['index_path'])
        
        logger.info(f"Loaded index from: {path}")


class DataIngestionPipeline:
    """
    Complete data ingestion pipeline for RAG.
    
    Reads documents, chunks them, generates embeddings, and builds index.
    """
    
    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        chunk_size: int = 512,
        chunk_overlap: int = 50
    ):
        self.chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.embedding_model = EmbeddingModel(embedding_model)
        self.index = None
        self.documents: Dict[str, Document] = {}
        self.chunks: Dict[str, Chunk] = {}
    
    def _read_file(self, file_path: Path) -> Optional[str]:
        """Read a file and return its content."""
        try:
            suffix = file_path.suffix.lower()
            
            if suffix in ['.txt', '.md', '.py', '.json', '.yaml', '.yml']:
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()
            elif suffix == '.pdf':
                # Try to read PDF
                try:
                    import PyPDF2
                    with open(file_path, 'rb') as f:
                        reader = PyPDF2.PdfReader(f)
                        return ' '.join(page.extract_text() for page in reader.pages)
                except ImportError:
                    logger.warning(f"PyPDF2 not installed, skipping PDF: {file_path}")
                    return None
            else:
                # Try to read as text
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        return f.read()
                except Exception:
                    return None
        except Exception as e:
            logger.error(f"Error reading {file_path}: {e}")
            return None
    
    def _generate_doc_id(self, content: str, file_path: str) -> str:
        """Generate a unique document ID."""
        hash_input = f"{file_path}:{len(content)}"
        return hashlib.md5(hash_input.encode()).hexdigest()[:12]
    
    def ingest_directory(
        self,
        directory: str,
        extensions: List[str] = None,
        recursive: bool = True
    ) -> int:
        """
        Ingest all documents from a directory.
        
        Args:
            directory: Path to the directory
            extensions: List of file extensions to include (e.g., ['.txt', '.md'])
            recursive: Whether to recursively search subdirectories
        
        Returns:
            Number of documents ingested
        """
        directory = Path(directory)
        extensions = extensions or ['.txt', '.md', '.py', '.json', '.pdf']
        
        if not directory.exists():
            raise ValueError(f"Directory not found: {directory}")
        
        # Find all files
        if recursive:
            files = [f for f in directory.rglob('*') if f.is_file()]
        else:
            files = [f for f in directory.iterdir() if f.is_file()]
        
        # Filter by extension
        files = [f for f in files if f.suffix.lower() in extensions]
        
        logger.info(f"Found {len(files)} files to ingest")
        
        # Read and process files
        documents = []
        for file_path in files:
            content = self._read_file(file_path)
            if content:
                doc_id = self._generate_doc_id(content, str(file_path))
                doc = Document(
                    id=doc_id,
                    content=content,
                    metadata={
                        'source': str(file_path),
                        'filename': file_path.name,
                        'extension': file_path.suffix,
                        'size': len(content),
                        'ingested_at': datetime.now().isoformat()
                    }
                )
                documents.append(doc)
                self.documents[doc_id] = doc
        
        logger.info(f"Successfully read {len(documents)} documents")
        
        return len(documents)
    
    def process_documents(self) -> int:
        """
        Process all ingested documents: chunk and generate embeddings.
        
        Returns:
            Number of chunks created
        """
        if not self.documents:
            raise ValueError("No documents to process. Run ingest_directory first.")
        
        # Chunk all documents
        all_chunks = []
        for doc in self.documents.values():
            chunks = self.chunker.chunk_document(doc)
            all_chunks.extend(chunks)
            for chunk in chunks:
                self.chunks[chunk.id] = chunk
        
        logger.info(f"Created {len(all_chunks)} chunks from {len(self.documents)} documents")
        
        # Generate embeddings
        chunk_texts = [chunk.content for chunk in all_chunks]
        logger.info("Generating embeddings...")
        embeddings = self.embedding_model.encode(chunk_texts)
        
        # Build index
        self.index = HNSWIndex(dimension=embeddings.shape[1], max_elements=len(all_chunks) + 1000)
        chunk_ids = [chunk.id for chunk in all_chunks]
        self.index.add_items(chunk_ids, embeddings)
        
        logger.info(f"Built HNSW index with {len(all_chunks)} vectors")
        
        return len(all_chunks)
    
    def save(self, output_dir: str):
        """Save all data to disk."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save documents
        docs_path = output_dir / 'documents.json'
        with open(docs_path, 'w') as f:
            json.dump({k: v.to_dict() for k, v in self.documents.items()}, f)
        
        # Save chunks
        chunks_path = output_dir / 'chunks.json'
        with open(chunks_path, 'w') as f:
            json.dump({k: asdict(v) for k, v in self.chunks.items()}, f)
        
        # Save index
        if self.index:
            index_path = output_dir / 'index.pkl'
            self.index.save(str(index_path))
        
        logger.info(f"Saved pipeline data to: {output_dir}")
    
    def load(self, input_dir: str):
        """Load all data from disk."""
        input_dir = Path(input_dir)
        
        # Load documents
        docs_path = input_dir / 'documents.json'
        if docs_path.exists():
            with open(docs_path, 'r') as f:
                docs_data = json.load(f)
                for k, v in docs_data.items():
                    embedding = v.pop('embedding', None)
                    if embedding:
                        embedding = np.array(embedding)
                    self.documents[k] = Document(**v, embedding=embedding)
        
        # Load chunks
        chunks_path = input_dir / 'chunks.json'
        if chunks_path.exists():
            with open(chunks_path, 'r') as f:
                chunks_data = json.load(f)
                for k, v in chunks_data.items():
                    self.chunks[k] = Chunk(**v)
        
        # Load index
        index_path = input_dir / 'index.pkl'
        if index_path.exists():
            self.index = HNSWIndex(dimension=384)  # Will be overwritten by load
            self.index.load(str(index_path))
        
        logger.info(f"Loaded pipeline data from: {input_dir}")
    
    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search for relevant chunks."""
        if not self.index:
            raise ValueError("Index not built. Run process_documents first.")
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query])[0]
        
        # Search
        results = self.index.search(query_embedding, k=k)
        
        # Get chunk details
        search_results = []
        for chunk_id, score in results:
            if chunk_id in self.chunks:
                chunk = self.chunks[chunk_id]
                search_results.append({
                    'chunk_id': chunk_id,
                    'content': chunk.content,
                    'score': score,
                    'metadata': chunk.metadata
                })
        
        return search_results


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Ingest documents for RAG')
    parser.add_argument('--input-dir', type=str, required=True,
                       help='Directory containing documents to ingest')
    parser.add_argument('--output-dir', type=str, default='./data/rag_index',
                       help='Directory to save the index')
    parser.add_argument('--chunk-size', type=int, default=512,
                       help='Size of text chunks')
    parser.add_argument('--chunk-overlap', type=int, default=50,
                       help='Overlap between chunks')
    parser.add_argument('--extensions', type=str, nargs='+',
                       default=['.txt', '.md', '.py'],
                       help='File extensions to include')
    parser.add_argument('--model', type=str, default='all-MiniLM-L6-v2',
                       help='Embedding model name')
    
    args = parser.parse_args()
    
    logger.info("=" * 50)
    logger.info("RAG Data Ingestion Pipeline")
    logger.info("=" * 50)
    
    # Initialize pipeline
    pipeline = DataIngestionPipeline(
        embedding_model=args.model,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap
    )
    
    # Ingest documents
    num_docs = pipeline.ingest_directory(
        args.input_dir,
        extensions=args.extensions
    )
    
    if num_docs == 0:
        logger.warning("No documents found. Exiting.")
        sys.exit(1)
    
    # Process documents
    num_chunks = pipeline.process_documents()
    
    # Save
    pipeline.save(args.output_dir)
    
    logger.info("=" * 50)
    logger.info(f"Ingestion complete!")
    logger.info(f"Documents: {num_docs}")
    logger.info(f"Chunks: {num_chunks}")
    logger.info(f"Output: {args.output_dir}")
    logger.info("=" * 50)


if __name__ == '__main__':
    main()
