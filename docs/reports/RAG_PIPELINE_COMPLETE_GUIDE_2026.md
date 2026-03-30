# Complete RAG Pipeline Guide 2026: From Scratch to Production

> **Retrieval-Augmented Generation (RAG)** - The definitive guide to building production-grade RAG systems

---

## Table of Contents

1. [What is RAG and Why 2026 is Different](#1-what-is-rag-and-why-2026-is-different)
2. [Complete Architecture Deep-Dive](#2-complete-architecture-deep-dive)
3. [Phase 1: Data Ingestion Pipeline](#3-phase-1-data-ingestion-pipeline)
4. [Phase 2: Chunking Strategies](#4-phase-2-chunking-strategies)
5. [Phase 3: Embeddings & Vector Databases](#5-phase-3-embeddings--vector-databases)
6. [Phase 4: Query Transformation](#6-phase-4-query-transformation)
7. [Phase 5: Retrieval & Reranking](#7-phase-5-retrieval--reranking)
8. [Phase 6: Generation & Response](#8-phase-6-generation--response)
9. [Phase 7: Evaluation & Monitoring](#9-phase-7-evaluation--monitoring)
10. [Production Deployment](#10-production-deployment)
11. [Common Failure Modes & Solutions](#11-common-failure-modes--solutions)
12. [Complete Implementation Example](#12-complete-implementation-example)

---

## 1. What is RAG and Why 2026 is Different

### The RAG Problem

**Traditional LLM Limitation:**
```
User asks about your company's 2026 pricing → LLM has no knowledge (trained on old data)
Result: Hallucination or "I don't know"
```

**RAG Solution:**
```
User asks about 2026 pricing → Retrieve from your database → Augment LLM prompt → Accurate answer
```

### Why 2026 is Different

| 2023-2024 RAG | 2026 Production RAG |
|--------------|---------------------|
| Simple document upload | Multi-source data ingestion (APIs, DBs, docs, webhooks) |
| Basic chunking | Semantic, late, and adaptive chunking |
| Pure vector search | Hybrid retrieval (vector + BM25 + metadata filtering) |
| Top-k results | Reranking with cross-encoders |
| One-shot generation | Query transformation, decomposition, multi-turn |
| No evaluation | Continuous evaluation pipelines |
| Static index | Incremental updates + real-time sync |

### RAG vs Fine-Tuning: When to Use What

```python
# Decision Framework

USE RAG WHEN:
- Data changes frequently (pricing, policies, docs)
- Need to cite sources
- Proprietary/internal data
- Cost-sensitive (no retraining)
- Need explainability

USE FINE-TUNING WHEN:
- Stable domain knowledge
- Need specific style/tone
- Task-specific behavior (classification, extraction)
- Have 1000+ high-quality examples
- Latency critical (no retrieval step)

BEST OF BOTH: Fine-tune for style + RAG for knowledge
```

---

## 2. Complete Architecture Deep-Dive

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         PRODUCTION RAG PIPELINE 2026                     │
└─────────────────────────────────────────────────────────────────────────┘

┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│ DATA SOURCES │     │  INGESTION   │     │   PROCESSING │
│              │     │   PIPELINE   │     │   PIPELINE   │
│ • Documents  │────▶│              │────▶│              │
│ • Databases  │     │ • Connectors │     │ • Chunking   │
│ • APIs       │     │ • Scheduling │     │ • Embedding  │
│ • Webhooks   │     │ • Validation │     │ • Indexing   │
└──────────────┘     └──────────────┘     └──────────────┘
                            │                    │
                            ▼                    ▼
                     ┌──────────────────────────────┐
                     │      VECTOR DATABASE         │
                     │                              │
                     │  ┌────────┐  ┌────────┐     │
                     │  │ Chunks │  │Metadata│     │
                     │  │ +      │  │ +      │     │
                     │  │Vectors │  │Filters │     │
                     │  └────────┘  └────────┘     │
                     └──────────────────────────────┘
                                    │
                                    │ ◀──── Query Flow ────▶
                                    ▼
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   RESPONSE   │     │  GENERATION  │     │   RETRIEVAL  │
│   & STREAM   │◀────│   ENHANCEMENT│◀────│   & RERANK   │
│              │     │              │     │              │
│ • Streaming  │     │ • Prompt     │     │ • Hybrid     │
│ • Citations  │     │ • Context    │     │ • Reranking  │
│ • Feedback   │     │ • Guardrails │     │ • Filtering  │
└──────────────┘     └──────────────┘     └──────────────┘
       │
       ▼
┌──────────────────────────────────────────┐
│      EVALUATION & MONITORING LOOP        │
│                                          │
│  • Retrieval Quality (Precision@K)      │
│  • Answer Quality (Faithfulness)        │
│  • User Feedback (Thumbs up/down)       │
│  • Cost & Latency Tracking              │
└──────────────────────────────────────────┘
```

### Component Interaction Flow

```python
# Pseudocode for complete flow

class RAGPipeline:
    def __init__(self):
        self.ingestion = DataIngestionPipeline()
        self.chunking = ChunkingStrategy()
        self.embeddings = EmbeddingModel()
        self.vector_db = VectorDatabase()
        self.query_transformer = QueryTransformer()
        self.retriever = HybridRetriever()
        self.reranker = CrossEncoderReranker()
        self.generator = LLMGenerator()
        self.evaluator = EvaluationPipeline()
    
    def index_documents(self, documents):
        # Phase 1-3: Ingest → Chunk → Embed → Store
        raw_data = self.ingestion.extract(documents)
        chunks = self.chunking.split(raw_data)
        embeddings = self.embeddings.encode(chunks)
        self.vector_db.insert(chunks, embeddings, metadata)
    
    def query(self, user_question):
        # Phase 4-7: Transform → Retrieve → Rerank → Generate
        transformed_query = self.query_transformer.rewrite(user_question)
        candidates = self.retriever.search(transformed_query, top_k=50)
        reranked = self.reranker.rerank(candidates, user_question, top_k=5)
        response = self.generator.generate(user_question, reranked)
        self.evaluator.log(user_question, response, reranked)
        return response
```

---

## 3. Phase 1: Data Ingestion Pipeline

### The Critical First Step

**80% of RAG failures happen at ingestion.** Garbage in = garbage out.

### Data Source Categories

```python
from enum import Enum
from typing import List, Dict, Any
from dataclasses import dataclass

class DataSourceType(Enum):
    PRIMARY = "primary"      # 10-20% of docs, 80% of questions (START HERE)
    SECONDARY = "secondary"  # Add after MVP works
    ARCHIVAL = "archival"    # Historical, rarely accessed

@dataclass
class DataSource:
    name: str
    source_type: DataSourceType
    connector_type: str  # file, api, database, webhook
    update_frequency: str  # real-time, hourly, daily, weekly
    priority: int  # 1-10, higher = more important
```

### Ingestion Pipeline Architecture

```python
import asyncio
from abc import ABC, abstractmethod
from datetime import datetime
import hashlib

class DataIngestionPipeline:
    """
    Production ingestion with:
    - Incremental updates (delta sync)
    - Checksum validation
    - Error handling & retry
    - Metadata extraction
    """
    
    def __init__(self):
        self.connectors = {}
        self.update_strategy = "incremental"  # or "full"
    
    async def ingest(self, source: DataSource) -> List[Dict]:
        """Main ingestion entry point"""
        
        # Step 1: Connect and extract
        raw_data = await self._extract(source)
        
        # Step 2: Validate and clean
        validated = self._validate(raw_data)
        
        # Step 3: Check for changes (incremental)
        if self.update_strategy == "incremental":
            new_data = await self._detect_changes(source, validated)
        else:
            new_data = validated
        
        # Step 4: Extract metadata
        enriched = self._extract_metadata(new_data, source)
        
        # Step 5: Store in staging
        await self._store_staging(enriched)
        
        return enriched
    
    async def _extract(self, source: DataSource) -> Any:
        """Extract from various sources"""
        
        if source.connector_type == "file":
            return await self._extract_files(source)
        elif source.connector_type == "api":
            return await self._extract_api(source)
        elif source.connector_type == "database":
            return await self._extract_database(source)
        elif source.connector_type == "webhook":
            return await self._extract_webhook(source)
    
    async def _extract_files(self, source) -> List[Dict]:
        """
        Handle multiple file formats with proper parsing
        """
        from pathlib import Path
        
        parsers = {
            '.pdf': self._parse_pdf,
            '.docx': self._parse_docx,
            '.txt': self._parse_txt,
            '.md': self._parse_markdown,
            '.html': self._parse_html,
            '.json': self._parse_json,
        }
        
        documents = []
        for file_path in Path(source.path).glob("**/*"):
            if file_path.suffix in parsers:
                content = await parsers[file_path.suffix](file_path)
                documents.append({
                    'content': content,
                    'source': str(file_path),
                    'source_type': 'file',
                    'file_name': file_path.name,
                    'file_size': file_path.stat().st_size,
                    'modified_date': datetime.fromtimestamp(
                        file_path.stat().st_mtime
                    ).isoformat(),
                    'checksum': self._compute_checksum(file_path),
                })
        
        return documents
    
    async def _parse_pdf(self, file_path) -> str:
        """
        PDF parsing with layout preservation
        2026 Best Practice: Use specialized parsers, not naive text extraction
        """
        # Option 1: LlamaParse (best for complex layouts)
        # from llama_parse import LlamaParse
        # parser = LlamaParse(api_key="...", result_type="markdown")
        # return await parser.aget_result(str(file_path))
        
        # Option 2: Unstructured.io (open source)
        from unstructured.partition.pdf import partition_pdf
        
        elements = partition_pdf(filename=str(file_path))
        return "\n\n".join([str(el) for el in elements])
        
        # Option 3: Docling (IBM, good for tables)
        # from docling.document_converter import DocumentConverter
        # converter = DocumentConverter()
        # result = converter.convert(str(file_path))
        # return result.document.export_to_markdown()
    
    def _compute_checksum(self, file_path) -> str:
        """For incremental update detection"""
        with open(file_path, 'rb') as f:
            return hashlib.sha256(f.read()).hexdigest()
    
    async def _detect_changes(
        self, 
        source: DataSource, 
        current_data: List[Dict]
    ) -> List[Dict]:
        """
        Incremental update logic
        
        Key insight: Don't re-process unchanged documents!
        """
        # Get previously indexed checksums
        previous_checksums = await self._get_stored_checksums(source)
        
        # Filter to only new/changed documents
        new_documents = []
        for doc in current_data:
            if doc['checksum'] not in previous_checksums:
                new_documents.append(doc)
                print(f"New/changed: {doc['source']}")
        
        # Handle deletions (documents that no longer exist)
        deleted_checksums = previous_checksums - {
            doc['checksum'] for doc in current_data
        }
        if deleted_checksums:
            await self._mark_as_deleted(deleted_checksums)
        
        return new_documents
    
    def _extract_metadata(
        self, 
        documents: List[Dict], 
        source: DataSource
    ) -> List[Dict]:
        """
        Enrich with metadata for better filtering
        
        Critical for production: Metadata enables:
        - Source-based filtering
        - Date range queries
        - Permission-based access
        - Better citations
        """
        for doc in documents:
            # Auto-extract metadata from content
            doc['metadata'] = {
                'source_name': source.name,
                'source_type': source.source_type.value,
                'ingested_at': datetime.now().isoformat(),
                'word_count': len(doc['content'].split()),
                
                # Extract from content (use LLM for complex extraction)
                'document_type': self._classify_document_type(doc['content']),
                'topics': self._extract_topics(doc['content']),
                'entities': self._extract_entities(doc['content']),
                
                # Preserve for citations
                'citation_info': {
                    'title': doc.get('file_name', 'Unknown'),
                    'source': doc['source'],
                }
            }
        
        return documents
    
    def _classify_document_type(self, content: str) -> str:
        """Classify document: policy, technical, faq, tutorial, etc."""
        # Use a small LLM or rule-based classifier
        content_lower = content.lower()[:500]
        
        if any(word in content_lower for word in ['policy', 'procedure', 'guideline']):
            return 'policy'
        elif any(word in content_lower for word in ['api', 'endpoint', 'parameter']):
            return 'technical'
        elif any(word in content_lower for word in ['question', 'answer', 'faq']):
            return 'faq'
        elif any(word in content_lower for word in ['tutorial', 'guide', 'how-to']):
            return 'tutorial'
        else:
            return 'general'
    
    def _extract_topics(self, content: str) -> List[str]:
        """Extract key topics (use LLM or keyword extraction)"""
        # Simple keyword-based (replace with LLM for production)
        keywords = {
            'pricing': ['price', 'cost', 'billing', 'subscription'],
            'security': ['authentication', 'authorization', 'encryption'],
            'api': ['endpoint', 'request', 'response', 'parameter'],
        }
        
        content_lower = content.lower()
        return [
            topic for topic, words in keywords.items()
            if any(word in content_lower for word in words)
        ]
    
    def _extract_entities(self, content: str) -> List[str]:
        """Extract named entities (products, features, etc.)"""
        # Use spaCy or LLM for NER
        # import spacy
        # nlp = spacy.load("en_core_web_sm")
        # doc = nlp(content)
        # return [ent.text for ent in doc.ents]
        return []  # Placeholder
```

### Update Strategies: Full vs Incremental

```python
class UpdateStrategy:
    """
    Choose based on your data characteristics
    """
    
    FULL_RE_INDEXING = {
        'when': [
            "Small dataset (< 10k documents)",
            "Schema changes",
            "Embedding model updates",
            "Monthly/weekly scheduled refresh"
        ],
        'pros': ['Simple', 'No stale data', 'Clean state'],
        'cons': ['Expensive', 'Downtime', 'Resource intensive'],
        'implementation': '''
            async def full_reindex(self):
                # 1. Create new index (don't delete old yet)
                new_index = await self.vector_db.create_index("v2")
                
                # 2. Process all documents
                for doc in all_documents:
                    await self.index_document(doc, index=new_index)
                
                # 3. Validate new index
                if await self.validate_index(new_index):
                    # 4. Swap alias (zero downtime)
                    await self.vector_db.swap_alias("main", new_index)
                    # 5. Delete old index
                    await self.vector_db.delete_index("v1")
        '''
    }
    
    INCREMENTAL_UPDATE = {
        'when': [
            "Large dataset",
            "Frequent changes (daily/hourly)",
            "Real-time requirements",
            "Cost-sensitive operations"
        ],
        'pros': ['Efficient', 'Always fresh', 'Low cost'],
        'cons': ['Complex', 'Potential drift', 'Deletion handling'],
        'implementation': '''
            async def incremental_update(self):
                # 1. Get last sync timestamp
                last_sync = await self.get_last_sync_time()
                
                # 2. Fetch only changed documents
                changed_docs = await db.query(
                    "SELECT * FROM documents WHERE updated_at > ?",
                    last_sync
                )
                
                # 3. Update/add changed documents
                for doc in changed_docs:
                    await self.upsert_document(doc)
                
                # 4. Handle deletions
                deleted_ids = await db.query(
                    "SELECT id FROM soft_deletes WHERE deleted_at > ?",
                    last_sync
                )
                for doc_id in deleted_ids:
                    await self.soft_delete(doc_id)
                
                # 5. Update sync timestamp
                await self.update_sync_time(datetime.now())
        '''
    }
    
    HYBRID_APPROACH = {
        'description': 'Best of both worlds',
        'strategy': '''
            - Incremental updates: Every hour (freshness)
            - Full re-index: Every Sunday at 2 AM (clean state)
            - Emergency re-index: On schema changes
        '''
    }
```

### Document Parsing: 2026 Best Practices

```python
class DocumentParser:
    """
    2026 Reality: Not all parsers are equal
    
    Hierarchy of parsing quality:
    1. LlamaParse (best for complex layouts, tables, figures)
    2. Unstructured.io (good open-source alternative)
    3. Docling (IBM, excellent for tables)
    4. LLMWhisperer (good for scanned docs)
    5. PyPDF2/pdfplumber (avoid - loses layout)
    """
    
    def __init__(self, parser_type: str = "llama"):
        self.parser_type = parser_type
    
    async def parse(self, file_path: str) -> Dict:
        """Parse with appropriate tool"""
        
        if self.parser_type == "llama":
            return await self._parse_with_llama(file_path)
        elif self.parser_type == "unstructured":
            return await self._parse_with_unstructured(file_path)
        elif self.parser_type == "docling":
            return await self._parse_with_docling(file_path)
    
    async def _parse_with_llama(self, file_path: str) -> Dict:
        """
        LlamaParse: Best for complex documents
        
        Why it's better:
        - Preserves tables as markdown
        - Handles multi-column layouts
        - Extracts figures with descriptions
        - Returns structured markdown
        """
        from llama_parse import LlamaParse
        
        parser = LlamaParse(
            api_key="your-api-key",
            result_type="markdown",  # or "json" for structured
            verbose=True,
            language="en",
        )
        
        result = await parser.aget_result(file_path)
        
        return {
            'content': result.text,
            'format': 'markdown',
            'tables_extracted': True,
            'figures_described': True,
        }
    
    async def _parse_with_unstructured(self, file_path: str) -> Dict:
        """
        Unstructured.io: Open source, good general purpose
        
        Install: pip install unstructured[pdf]
        """
        from unstructured.partition.auto import partition
        
        elements = partition(filename=file_path)
        
        # Group by type
        content_sections = []
        for el in elements:
            content_sections.append({
                'type': type(el).__name__,
                'content': str(el),
                'metadata': el.metadata.__dict__ if hasattr(el, 'metadata') else {}
            })
        
        return {
            'content': "\n\n".join([str(el) for el in elements]),
            'format': 'text',
            'structured_sections': content_sections,
        }
    
    async def _parse_with_docling(self, file_path: str) -> Dict:
        """
        Docling: IBM's parser, excellent for tables
        
        Install: pip install docling
        """
        from docling.document_converter import DocumentConverter
        
        converter = DocumentConverter()
        result = converter.convert(file_path)
        
        # Export as markdown (preserves tables)
        markdown = result.document.export_to_markdown()
        
        return {
            'content': markdown,
            'format': 'markdown',
            'tables_preserved': True,
        }
```

---

## 4. Phase 2: Chunking Strategies

### Why Chunking Matters

```
Bad chunking:
"Welcome to our company... [500 words of intro] ...pricing is $99"
         ↑                                        ↑
    Query: "What is the price?"              Lost in noise

Good chunking:
"Product Pricing: Basic plan $99/month, Pro plan $199/month"
         ↑
    Query: "What is the price?" → DIRECT MATCH
```

### Chunking Strategy Comparison

| Strategy | Chunk Size | When to Use | Pros | Cons |
|----------|-----------|-------------|------|------|
| **Fixed-size** | 512 tokens | Simple docs, MVP | Fast, predictable | Breaks context |
| **Recursive** | 256-1024 | General purpose | Preserves structure | More complex |
| **Semantic** | Variable | High-value docs | Context-aware | Slow, expensive |
| **Late chunking** | Full doc → chunks | Complex reasoning | Global context | Memory intensive |
| **Agentic** | Dynamic | Multi-hop queries | Query-aware | Very slow |

### Implementation: All Strategies

```python
from typing import List, Generator
from dataclasses import dataclass
import tiktoken

@dataclass
class Chunk:
    content: str
    chunk_id: str
    document_id: str
    start_index: int
    end_index: int
    metadata: dict
    
class ChunkingStrategy:
    """
    2026 Best Practice: Use different strategies for different document types
    """
    
    def __init__(self, strategy: str = "recursive"):
        self.strategy = strategy
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
    
    def chunk(self, document: Dict) -> List[Chunk]:
        """Main entry point"""
        
        if self.strategy == "fixed":
            return self._fixed_size_chunking(document)
        elif self.strategy == "recursive":
            return self._recursive_chunking(document)
        elif self.strategy == "semantic":
            return self._semantic_chunking(document)
        elif self.strategy == "late":
            return self._late_chunking(document)
    
    def _fixed_size_chunking(
        self, 
        document: Dict,
        chunk_size: int = 512,
        overlap: int = 50
    ) -> List[Chunk]:
        """
        Simple fixed-size chunking
        
        Use when: Speed is critical, documents are uniform
        """
        content = document['content']
        tokens = self.tokenizer.encode(content)
        
        chunks = []
        for i in range(0, len(tokens), chunk_size - overlap):
            chunk_tokens = tokens[i:i + chunk_size]
            chunk_text = self.tokenizer.decode(chunk_tokens)
            
            chunks.append(Chunk(
                content=chunk_text,
                chunk_id=f"{document['id']}_chunk_{len(chunks)}",
                document_id=document['id'],
                start_index=i,
                end_index=i + len(chunk_tokens),
                metadata={
                    **document.get('metadata', {}),
                    'chunk_size': len(chunk_tokens),
                    'chunk_method': 'fixed',
                }
            ))
        
        return chunks
    
    def _recursive_chunking(
        self, 
        document: Dict,
        chunk_size: int = 512,
        chunk_overlap: int = 50
    ) -> List[Chunk]:
        """
        Recursive chunking: Split by structure
        
        Hierarchy: document → sections → subsections → paragraphs → sentences
        
        Use when: Documents have clear structure (most cases)
        """
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        
        content = document['content']
        
        # LangChain's recursive splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=[
                "\n\n\n",  # Section breaks
                "\n\n",    # Paragraph breaks
                "\n",      # Line breaks
                ". ",      # Sentences
                " ",       # Words
                ""         # Characters
            ],
        )
        
        texts = text_splitter.split_text(content)
        
        chunks = []
        current_index = 0
        for i, text in enumerate(texts):
            chunks.append(Chunk(
                content=text,
                chunk_id=f"{document['id']}_chunk_{i}",
                document_id=document['id'],
                start_index=current_index,
                end_index=current_index + len(text),
                metadata={
                    **document.get('metadata', {}),
                    'chunk_size': len(text),
                    'chunk_method': 'recursive',
                }
            ))
            current_index += len(text)
        
        return chunks
    
    def _semantic_chunking(
        self, 
        document: Dict,
        embedding_model=None
    ) -> List[Chunk]:
        """
        Semantic chunking: Split by meaning, not characters
        
        Algorithm:
        1. Split into sentences
        2. Embed each sentence
        3. Group sentences with similar embeddings
        4. Break when semantic distance exceeds threshold
        
        Use when: High-value documents, complex topics
        """
        from sentence_transformers import SentenceTransformer
        import numpy as np
        from sklearn.metrics.pairwise import cosine_similarity
        
        if embedding_model is None:
            embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        content = document['content']
        
        # Step 1: Split into sentences
        sentences = self._split_into_sentences(content)
        
        if len(sentences) < 2:
            return [self._create_single_chunk(document, content)]
        
        # Step 2: Embed sentences
        embeddings = embedding_model.encode(sentences)
        
        # Step 3: Compute similarity between consecutive sentences
        similarities = []
        for i in range(len(sentences) - 1):
            sim = cosine_similarity(
                [embeddings[i]], 
                [embeddings[i + 1]]
            )[0][0]
            similarities.append(sim)
        
        # Step 4: Find break points (where similarity drops)
        threshold = 0.5  # Tune based on your content
        break_points = [0]
        for i, sim in enumerate(similarities):
            if sim < threshold:
                break_points.append(i + 1)
        break_points.append(len(sentences))
        
        # Step 5: Create chunks from grouped sentences
        chunks = []
        for i in range(len(break_points) - 1):
            start_idx = break_points[i]
            end_idx = break_points[i + 1]
            chunk_text = " ".join(sentences[start_idx:end_idx])
            
            if chunk_text.strip():  # Skip empty chunks
                chunks.append(Chunk(
                    content=chunk_text,
                    chunk_id=f"{document['id']}_chunk_{len(chunks)}",
                    document_id=document['id'],
                    start_index=start_idx,
                    end_index=end_idx,
                    metadata={
                        **document.get('metadata', {}),
                        'chunk_size': len(chunk_text),
                        'chunk_method': 'semantic',
                        'sentence_count': end_idx - start_idx,
                        'avg_similarity': np.mean(
                            similarities[start_idx:end_idx-1]
                        ) if end_idx > start_idx + 1 else 1.0,
                    }
                ))
        
        return chunks
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Simple sentence splitter (use nltk/spaCy for production)"""
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _late_chunking(
        self, 
        document: Dict,
        llm=None
    ) -> List[Chunk]:
        """
        Late chunking: Keep full document, chunk at query time
        
        Algorithm:
        1. Store full document with summary
        2. At query time, use LLM to extract relevant sections
        3. Dynamic chunking based on query
        
        Use when: Complex reasoning, multi-hop questions
        """
        # Store full document with metadata
        # Actual chunking happens at query time
        
        return [Chunk(
            content=document['content'],
            chunk_id=f"{document['id']}_full",
            document_id=document['id'],
            start_index=0,
            end_index=len(document['content']),
            metadata={
                **document.get('metadata', {}),
                'chunk_method': 'late',
                'requires_query_time_processing': True,
            }
        )]
    
    def _create_single_chunk(
        self, 
        document: Dict, 
        content: str
    ) -> List[Chunk]:
        """Fallback for short documents"""
        return [Chunk(
            content=content,
            chunk_id=f"{document['id']}_chunk_0",
            document_id=document['id'],
            start_index=0,
            end_index=len(content),
            metadata={
                **document.get('metadata', {}),
                'chunk_method': 'single',
            }
        )]
```

### Chunking Best Practices

```python
class ChunkingBestPractices:
    """
    Lessons from production RAG systems
    """
    
    GUIDELINES = {
        'chunk_size': {
            'recommendation': '256-1024 tokens',
            'reasoning': '''
                Too small (< 256): Loses context, fragmented answers
                Too large (> 1024): Noise, retrieval confusion
                Sweet spot: 512 tokens for general purpose
            ''',
            'adjustments': {
                'technical_docs': '512-768 (more context needed)',
                'faq': '256-512 (concise answers)',
                'policies': '768-1024 (full sections)',
            }
        },
        
        'overlap': {
            'recommendation': '10-20% of chunk size',
            'reasoning': '''
                Prevents context loss at boundaries
                50 tokens overlap for 512-token chunks
            '''
        },
        
        'metadata': {
            'always_include': [
                'document_id',
                'chunk_index',
                'total_chunks',
                'source_type',
                'created_date',
            ],
            'optional_but_useful': [
                'section_title',
                'parent_chunk_id',  # For hierarchical retrieval
                'has_table',
                'has_code',
                'topic_tags',
            ]
        },
        
        'special_handling': {
            'tables': 'Keep tables intact, don't split across chunks',
            'code': 'Preserve indentation, include full functions',
            'lists': 'Keep related list items together',
            'headers': 'Include section headers in each chunk',
        }
    }
    
    @staticmethod
    def add_section_headers(chunk: Chunk, document_structure: dict) -> Chunk:
        """
        Add section headers to chunk content
        
        Why: Provides context for retrieval
        Example: "## Pricing Plans\n\nThe basic plan costs $99..."
        """
        section_title = document_structure.get(
            chunk.start_index, 
            "Unknown Section"
        )
        chunk.content = f"## {section_title}\n\n{chunk.content}"
        chunk.metadata['section_title'] = section_title
        return chunk
    
    @staticmethod
    def preserve_tables(chunk: Chunk) -> Chunk:
        """
        Ensure tables remain in markdown format
        
        Why: LLMs understand markdown tables better than plain text
        """
        # Detect if chunk contains table
        if '|' in chunk.content and '-|-' in chunk.content:
            chunk.metadata['has_table'] = True
            # Don't apply text-only transformations
        return chunk
```

---

## 5. Phase 3: Embeddings & Vector Databases

### Understanding Embeddings

```python
"""
What are embeddings?

Text → [0.123, -0.456, 0.789, ...] (1536 numbers)
         ↑
    Dense vector capturing semantic meaning

Similar texts → Similar vectors (close in vector space)

Example:
"cat" → [0.1, -0.5, 0.8, ...]
"dog" → [0.15, -0.48, 0.75, ...]  ← Close to "cat"
"car" → [-0.8, 0.3, -0.2, ...]    ← Far from both
"""
```

### Embedding Model Selection (2026)

| Model | Dimensions | Speed | Quality | Best For |
|-------|-----------|-------|---------|----------|
| **text-embedding-3-large** | 3072 | Medium | Excellent | Production, multi-lingual |
| **text-embedding-3-small** | 1536 | Fast | Very Good | Cost-sensitive, English |
| **BGE-large-en-v1.5** | 1024 | Medium | Excellent | Open source, retrieval |
| **BGE-m3** | 1024 | Medium | Excellent | Multi-lingual, long context |
| **E5-large-v2** | 1024 | Fast | Very Good | General purpose |
| **Voyage-large-2** | 1536 | Fast | Excellent | Production, reranking |

### Embedding Implementation

```python
from typing import List, Union
import numpy as np
from dataclasses import dataclass

@dataclass
class EmbeddingResult:
    vectors: List[List[float]]
    model: str
    dimensions: int
    usage: dict

class EmbeddingModel:
    """
    2026 Best Practice: Support multiple providers for fallback
    """
    
    def __init__(
        self, 
        provider: str = "openai",
        model: str = "text-embedding-3-small",
        fallback_providers: List[str] = None
    ):
        self.provider = provider
        self.model = model
        self.fallback_providers = fallback_providers or ["openai", "local"]
        self._cache = {}  # Simple caching
    
    def encode(
        self, 
        texts: Union[str, List[str]],
        batch_size: int = 32
    ) -> np.ndarray:
        """
        Generate embeddings with automatic fallback
        
        Production considerations:
        - Batch processing for efficiency
        - Caching to avoid re-computation
        - Fallback on API failures
        """
        if isinstance(texts, str):
            texts = [texts]
        
        # Check cache first
        cached_results = []
        texts_to_embed = []
        for text in texts:
            cache_key = self._hash(text)
            if cache_key in self._cache:
                cached_results.append(self._cache[cache_key])
            else:
                cached_results.append(None)
                texts_to_embed.append(text)
        
        # Embed uncached texts
        if texts_to_embed:
            new_embeddings = self._embed_with_fallback(texts_to_embed)
            
            # Update cache
            for text, embedding in zip(texts_to_embed, new_embeddings):
                self._cache[self._hash(text)] = embedding
            
            # Merge results
            result = []
            new_idx = 0
            for i, cached in enumerate(cached_results):
                if cached is not None:
                    result.append(cached)
                else:
                    result.append(new_embeddings[new_idx])
                    new_idx += 1
            
            return np.array(result)
        
        return np.array(cached_results)
    
    def _embed_with_fallback(
        self, 
        texts: List[str]
    ) -> List[List[float]]:
        """Try primary provider, fallback on failure"""
        
        for provider in self.fallback_providers:
            try:
                if provider == "openai":
                    return self._embed_openai(texts)
                elif provider == "local":
                    return self._embed_local(texts)
                elif provider == "cohere":
                    return self._embed_cohere(texts)
            except Exception as e:
                print(f"Provider {provider} failed: {e}")
                continue
        
        raise RuntimeError("All embedding providers failed")
    
    def _embed_openai(self, texts: List[str]) -> List[List[float]]:
        """OpenAI embeddings with batching"""
        from openai import OpenAI
        
        client = OpenAI()
        
        # Batch to avoid rate limits
        all_embeddings = []
        for i in range(0, len(texts), 32):
            batch = texts[i:i+32]
            
            response = client.embeddings.create(
                model=self.model,
                input=batch,
            )
            
            batch_embeddings = [
                item.embedding for item in response.data
            ]
            all_embeddings.extend(batch_embeddings)
        
        return all_embeddings
    
    def _embed_local(self, texts: List[str]) -> List[List[float]]:
        """
        Local embeddings with SentenceTransformers
        
        Install: pip install sentence-transformers
        """
        from sentence_transformers import SentenceTransformer
        
        model = SentenceTransformer('BAAI/bge-large-en-v1.5')
        embeddings = model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            normalize_embeddings=True,  # Important for cosine similarity
        )
        
        return embeddings.tolist()
    
    def _embed_cohere(self, texts: List[str]) -> List[List[float]]:
        """Cohere embeddings (good alternative)"""
        import cohere
        
        client = cohere.Client()
        response = client.embed(
            texts=texts,
            model='embed-english-v3.0',
            input_type='search_document',  # or 'search_query' for queries
        )
        
        return response.embeddings
    
    def _hash(self, text: str) -> str:
        """Simple hash for caching"""
        import hashlib
        return hashlib.md5(text.encode()).hexdigest()
```

### Vector Database Selection

```python
"""
Vector Database Comparison (2026)

┌─────────────────┬──────────────┬─────────────┬──────────────┬─────────────┐
│ Feature         │ Pinecone     │ Weaviate    │ Qdrant       │ pgvector    │
├─────────────────┼──────────────┼─────────────┼──────────────┼─────────────┤
│ Type            │ Managed      │ Self/Managed│ Self/Managed│ PostgreSQL  │
│ Performance     │ ⭐⭐⭐⭐⭐       │ ⭐⭐⭐⭐       │ ⭐⭐⭐⭐⭐      │ ⭐⭐⭐        │
│ Ease of Use     │ ⭐⭐⭐⭐⭐       │ ⭐⭐⭐⭐       │ ⭐⭐⭐⭐       │ ⭐⭐⭐        │
│ Cost            │ $$$$         │ $$          │ $$           │ $           │
│ Metadata Filter │ Excellent    │ Excellent   │ Excellent    │ Good        │
│ Hybrid Search   │ Yes          │ Yes         │ Yes          │ Yes         │
│ Scalability     │ Automatic    │ Manual      │ Manual       │ Manual      │
│ Best For        │ Production   │ Flexible    │ Performance  │ Existing PG │
└─────────────────┴──────────────┴─────────────┴──────────────┴─────────────┘
"""

class VectorDatabase:
    """
    Abstract interface for vector databases
    
    Production tip: Design for portability
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.client = self._connect()
    
    def _connect(self):
        """Connect to configured vector DB"""
        
        if self.config['type'] == 'pinecone':
            return self._connect_pinecone()
        elif self.config['type'] == 'weaviate':
            return self._connect_weaviate()
        elif self.config['type'] == 'qdrant':
            return self._connect_qdrant()
        elif self.config['type'] == 'pgvector':
            return self._connect_pgvector()
    
    def _connect_pinecone(self):
        """
        Pinecone: Managed, easiest for production
        
        Install: pip install pinecone-client
        """
        from pinecone import Pinecone
        
        pc = Pinecone(
            api_key=self.config['api_key'],
            environment=self.config.get('environment', 'us-west1-gcp')
        )
        
        return pc
    
    def _connect_weaviate(self):
        """
        Weaviate: Flexible, good for GraphQL lovers
        
        Install: pip install weaviate-client
        """
        import weaviate
        
        client = weaviate.Client(
            url=self.config['url'],
            auth_client_secret=weaviate.AuthApiKey(
                api_key=self.config['api_key']
            ) if self.config.get('api_key') else None,
        )
        
        return client
    
    def _connect_qdrant(self):
        """
        Qdrant: High performance, Rust-based
        
        Install: pip install qdrant-client
        """
        from qdrant_client import QdrantClient
        
        client = QdrantClient(
            url=self.config.get('url'),
            api_key=self.config.get('api_key'),
        )
        
        return client
    
    def _connect_pgvector(self):
        """
        pgvector: PostgreSQL extension, good if already using Postgres
        
        Install: pip install psycopg2-binary
        """
        import psycopg2
        from psycopg2.extras import execute_values
        
        conn = psycopg2.connect(
            host=self.config['host'],
            database=self.config['database'],
            user=self.config['user'],
            password=self.config['password'],
        )
        
        return conn
    
    async def insert(
        self,
        chunks: List,
        embeddings: np.ndarray,
        metadata: List[dict] = None
    ):
        """Insert chunks with embeddings"""
        
        if self.config['type'] == 'pinecone':
            await self._insert_pinecone(chunks, embeddings, metadata)
        elif self.config['type'] == 'weaviate':
            await self._insert_weaviate(chunks, embeddings, metadata)
        elif self.config['type'] == 'qdrant':
            await self._insert_qdrant(chunks, embeddings, metadata)
        elif self.config['type'] == 'pgvector':
            await self._insert_pgvector(chunks, embeddings, metadata)
    
    async def _insert_pinecone(
        self,
        chunks: List,
        embeddings: np.ndarray,
        metadata: List[dict] = None
    ):
        """Insert into Pinecone"""
        
        index = self.client.Index(self.config['index_name'])
        
        # Prepare upsert data
        upsert_data = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            meta = {
                **(metadata[i] if metadata else {}),
                'content': chunk.content,
            }
            
            upsert_data.append({
                'id': chunk.chunk_id,
                'values': embedding.tolist(),
                'metadata': meta,
            })
        
        # Batch upsert
        batch_size = 100
        for i in range(0, len(upsert_data), batch_size):
            batch = upsert_data[i:i+batch_size]
            index.upsert(vectors=batch)
    
    async def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        filters: dict = None
    ) -> List[dict]:
        """Search with optional metadata filtering"""
        
        if self.config['type'] == 'pinecone':
            return await self._search_pinecone(query_embedding, top_k, filters)
        # ... other implementations
    
    async def _search_pinecone(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        filters: dict = None
    ) -> List[dict]:
        """Search Pinecone with filters"""
        
        index = self.client.Index(self.config['index_name'])
        
        results = index.query(
            vector=query_embedding.tolist(),
            top_k=top_k,
            filter=filters,
            include_metadata=True,
        )
        
        return results.matches
```

### Index Configuration

```python
class IndexConfiguration:
    """
    Vector index configuration for different databases
    
    Key parameters that affect performance:
    """
    
    PINECONE_CONFIG = {
        'metric': 'cosine',  # or 'euclidean', 'dotproduct'
        'dimension': 1536,   # Match embedding model
        
        # Performance tuning
        'pods': 1,           # More pods = more throughput
        'replicas': 1,       # More replicas = more availability
        'shards': 1,         # More shards = better parallelism
        
        # Index type
        'index_type': 'pod',  # or 'serverless' (new)
    }
    
    QDRANT_CONFIG = {
        'vector_params': {
            'size': 1536,
            'distance': 'Cosine',
        },
        
        # HNSW index (approximate nearest neighbor)
        'hnsw_config': {
            'm': 16,              # Connections per node (higher = more accurate)
            'ef_construct': 100,  # Build-time search depth
            'full_scan_threshold': 10000,
        },
        
        # Optimizations
        'optimizers_config': {
            'deleted_threshold': 0.2,
            'vacuum_min_vector_number': 1000,
        },
    }
    
    WEAVIATE_CONFIG = {
        'vectorIndexType': 'hnsw',
        'vectorIndexConfig': {
            'maxConnections': 64,
            'efConstruction': 128,
            'ef': -1,  # Dynamic at query time
        },
        
        # Quantization (reduce memory)
        'quantizer': 'pq',  # Product quantization
        'pq': {
            'segments': 96,
            'centroids': 256,
        },
    }
    
    @staticmethod
    def choose_index_type(data_size: str, query_pattern: str) -> dict:
        """
        Choose index configuration based on requirements
        """
        
        if data_size == "small" and query_pattern == "exact":
            return {
                'type': 'flat',  # Exact search
                'when': '< 10k vectors, need 100% recall',
            }
        elif data_size == "medium" and query_pattern == "balanced":
            return {
                'type': 'hnsw',  # Approximate, balanced
                'when': '10k - 1M vectors, good balance',
                'params': {'m': 16, 'ef_construction': 200},
            }
        elif data_size == "large" and query_pattern == "fast":
            return {
                'type': 'ivf_pq',  # Inverted file + quantization
                'when': '> 1M vectors, need speed over accuracy',
                'params': {'nlist': 4096, 'nprobe': 32},
            }
```

---

## 6. Phase 4: Query Transformation

### Why Transform Queries?

```
User asks: "How much does it cost?"
         ↓ (too vague, no context)
Transformed: "What is the pricing for the product plans?"
         ↓
Better retrieval → Better answer

User asks: "Can I integrate it with my CRM and also track sales?"
         ↓ (compound question)
Decomposed: 
  Q1: "What CRM integrations are available?"
  Q2: "Does it support sales tracking?"
         ↓
Multiple retrievals → Comprehensive answer
```

### Query Transformation Techniques

```python
from typing import List, Optional
from dataclasses import dataclass
import json

@dataclass
class TransformedQuery:
    original: str
    rewritten: str
    decomposition: List[str] = None
    intent: str = None
    entities: List[str] = None

class QueryTransformer:
    """
    Transform user queries for better retrieval
    
    Techniques:
    1. Query Rewriting (clarify, expand)
    2. Query Decomposition (break into sub-questions)
    3. Step-back Prompting (ask broader question)
    4. HyDE (Hypothetical Document Embedding)
    """
    
    def __init__(self, llm_client):
        self.llm = llm_client
    
    def transform(
        self, 
        query: str,
        conversation_history: List[dict] = None,
        transformation_type: str = "rewrite"
    ) -> TransformedQuery:
        """Main transformation entry point"""
        
        if transformation_type == "rewrite":
            return self._rewrite_query(query, conversation_history)
        elif transformation_type == "decompose":
            return self._decompose_query(query)
        elif transformation_type == "step_back":
            return self._step_back_query(query)
        elif transformation_type == "hyde":
            return self._hyde_query(query)
    
    def _rewrite_query(
        self, 
        query: str,
        conversation_history: List[dict] = None
    ) -> TransformedQuery:
        """
        Rewrite query to be more specific and self-contained
        
        Why: Users ask vague questions, need to add context
        """
        
        # Add conversation context if available
        context_prompt = ""
        if conversation_history:
            context_prompt = f"""
            Conversation history:
            {json.dumps(conversation_history[-3:], indent=2)}
            
            """
        
        prompt = f"""
        {context_prompt}
        User query: "{query}"
        
        Rewrite this query to be:
        1. More specific and detailed
        2. Self-contained (resolve pronouns like "it", "they")
        3. Include relevant context from conversation
        4. Keep the same intent
        
        Return ONLY the rewritten query, nothing else.
        """
        
        rewritten = self.llm.generate(prompt, max_tokens=100)
        
        return TransformedQuery(
            original=query,
            rewritten=rewritten.strip(),
            intent=self._classify_intent(query),
        )
    
    def _decompose_query(self, query: str) -> TransformedQuery:
        """
        Break complex queries into simpler sub-questions
        
        Why: Single retrieval often can't answer multi-part questions
        """
        
        prompt = f"""
        Query: "{query}"
        
        Break this into 2-4 simpler sub-questions that,
        when answered together, fully address the original query.
        
        Return as JSON array:
        ["sub-question 1", "sub-question 2", ...]
        """
        
        response = self.llm.generate(prompt, max_tokens=200)
        
        try:
            sub_questions = json.loads(response.strip())
        except:
            sub_questions = [query]  # Fallback
        
        return TransformedQuery(
            original=query,
            rewritten=query,
            decomposition=sub_questions,
            intent="compound",
        )
    
    def _step_back_query(self, query: str) -> TransformedQuery:
        """
        Ask a broader "step-back" question first
        
        Why: Broader context helps retrieve better background info
        
        Example:
        User: "What's the return policy for electronics?"
        Step-back: "What are the general return policies?"
        """
        
        prompt = f"""
        Query: "{query}"
        
        Ask a broader, more general question that would
        provide helpful context for answering the specific query.
        
        Return ONLY the broader question.
        """
        
        broader = self.llm.generate(prompt, max_tokens=100)
        
        return TransformedQuery(
            original=query,
            rewritten=broader.strip(),
            intent="step_back",
        )
    
    def _hyde_query(self, query: str) -> TransformedQuery:
        """
        HyDE: Hypothetical Document Embedding
        
        Algorithm:
        1. Generate a hypothetical answer
        2. Embed the hypothetical answer
        3. Search for similar documents
        
        Why: Hypothetical answers are often closer to 
             actual answers in embedding space
        """
        
        prompt = f"""
        Query: "{query}"
        
        Write a hypothetical answer to this query.
        Make it detailed and specific (3-5 sentences).
        
        This will be used to find similar documents.
        """
        
        hypothetical = self.llm.generate(prompt, max_tokens=200)
        
        return TransformedQuery(
            original=query,
            rewritten=hypothetical.strip(),
            intent="hyde",
        )
    
    def _classify_intent(self, query: str) -> str:
        """
        Classify query intent for routing
        
        Common intents:
        - factual: Looking for specific fact
        - how_to: Step-by-step instructions
        - comparison: Comparing options
        - troubleshooting: Problem solving
        - exploratory: General information
        """
        
        # Simple rule-based (use LLM for production)
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['what', 'when', 'who', 'how much']):
            return 'factual'
        elif any(word in query_lower for word in ['how to', 'how do', 'steps']):
            return 'how_to'
        elif any(word in query_lower for word in ['vs', 'versus', 'compare', 'better']):
            return 'comparison'
        elif any(word in query_lower for word in ['error', 'problem', 'issue', 'not working']):
            return 'troubleshooting'
        else:
            return 'exploratory'
```

### Advanced: Multi-Turn Conversation Handling

```python
class ConversationAwareRetriever:
    """
    Handle follow-up questions in conversations
    
    Challenge: "How much does it cost?" makes sense only in context
    """
    
    def __init__(self, retriever, query_transformer):
        self.retriever = retriever
        self.transformer = query_transformer
        self.history = []
    
    async def query(
        self, 
        user_query: str,
        max_history_turns: int = 3
    ):
        """
        Process query with conversation context
        """
        
        # Step 1: Rewrite query with context
        transformed = self.transformer.transform(
            user_query,
            conversation_history=self.history[-max_history_turns:]
        )
        
        # Step 2: Retrieve with rewritten query
        results = await self.retriever.search(transformed.rewritten)
        
        # Step 3: Update history
        self.history.append({
            'role': 'user',
            'content': user_query,
        })
        # (Will add assistant response after generation)
        
        return {
            'query': transformed,
            'results': results,
        }
    
    def add_response(self, response: str):
        """Add assistant response to history"""
        self.history.append({
            'role': 'assistant',
            'content': response,
        })
    
    def clear_history(self):
        """Reset conversation"""
        self.history = []
```

---

## 7. Phase 5: Retrieval & Reranking

### Hybrid Retrieval

```python
"""
Why Hybrid Retrieval?

Pure Vector Search Problem:
- Great for semantic similarity
- Bad for exact keyword matching
- Misses jargon, acronyms, product names

Example:
Query: "API-2026-X endpoint"
Vector search might return: "How to connect to services" (semantically similar)
BM25 returns: "API-2026-X endpoint documentation" (exact match)

Hybrid = Best of both worlds
"""

class HybridRetriever:
    """
    Combine vector search + keyword search + metadata filtering
    """
    
    def __init__(
        self,
        vector_db,
        embedding_model,
        bm25_index=None,
    ):
        self.vector_db = vector_db
        self.embeddings = embedding_model
        self.bm25_index = bm25_index
        self.weights = {
            'vector': 0.5,
            'bm25': 0.5,
        }
    
    async def search(
        self,
        query: str,
        top_k: int = 50,  # Retrieve more for reranking
        filters: dict = None,
    ) -> List[dict]:
        """
        Hybrid retrieval with reciprocal rank fusion
        """
        
        # Step 1: Vector search
        query_embedding = self.embeddings.encode([query])[0]
        vector_results = await self.vector_db.search(
            query_embedding,
            top_k=top_k,
            filters=filters,
        )
        
        # Step 2: BM25 keyword search
        bm25_results = await self._bm25_search(query, top_k=top_k)
        
        # Step 3: Combine with Reciprocal Rank Fusion (RRF)
        combined = self._reciprocal_rank_fusion(
            vector_results,
            bm25_results,
            weights=self.weights,
            top_k=top_k,
        )
        
        return combined
    
    async def _bm25_search(
        self, 
        query: str, 
        top_k: int
    ) -> List[dict]:
        """
        BM25 keyword search
        
        Install: pip install rank-bm25
        """
        from rank_bm25 import BM25Okapi
        
        if self.bm25_index is None:
            # Build index on the fly (slow, do this once at startup)
            documents = await self._get_all_documents()
            tokenized_docs = [doc.split() for doc in documents]
            self.bm25_index = BM25Okapi(tokenized_docs)
        
        scores = self.bm25_index.get_scores(query.split())
        
        # Return top-k by score
        top_indices = scores.argsort()[-top_k:][::-1]
        return top_indices
    
    def _reciprocal_rank_fusion(
        self,
        vector_results: List[dict],
        bm25_results: List[dict],
        weights: dict,
        top_k: int,
        k: int = 60,  # RRF constant
    ) -> List[dict]:
        """
        Reciprocal Rank Fusion (RRF)
        
        Formula: RRF_score = Σ (weight / (rank + k))
        
        Why RRF: Simple, effective, no tuning needed
        """
        
        from collections import defaultdict
        
        scores = defaultdict(float)
        
        # Score from vector results
        for rank, result in enumerate(vector_results):
            scores[result.id] += weights['vector'] / (rank + k)
        
        # Score from BM25 results
        for rank, result in enumerate(bm25_results):
            scores[result.id] += weights['bm25'] / (rank + k)
        
        # Sort by combined score
        sorted_results = sorted(
            scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return sorted_results[:top_k]
```

### Reranking

```python
"""
Why Reranking?

Retrieval gets 50 candidates fast (approximate)
Reranking picks best 5 carefully (precise)

Analogy:
- Retrieval = Casting net wide (catch many fish)
- Reranking = Carefully selecting the best ones

Performance gain: 20-40% improvement in retrieval quality
"""

class CrossEncoderReranker:
    """
    Rerank retrieved chunks using cross-encoder
    
    Cross-encoder vs Bi-encoder:
    - Bi-encoder (embeddings): Encode separately, compare vectors (fast)
    - Cross-encoder: Encode query+document together (slow but accurate)
    """
    
    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-large",
        top_k: int = 5,
    ):
        self.top_k = top_k
        self.model = self._load_model(model_name)
    
    def _load_model(self, model_name: str):
        """
        Load cross-encoder model
        
        Options:
        - BAAI/bge-reranker-large (open source, excellent)
        - BAAI/bge-reranker-base (faster, good)
        - Cohere Rerank (API, production-ready)
        - Voyage AI rerank-2.5 (API, excellent)
        """
        from sentence_transformers import CrossEncoder
        
        return CrossEncoder(model_name)
    
    def rerank(
        self,
        query: str,
        candidates: List[dict],
        top_k: int = None,
    ) -> List[dict]:
        """
        Rerank candidates by relevance to query
        """
        
        if not candidates:
            return []
        
        top_k = top_k or self.top_k
        
        # Prepare pairs for cross-encoder
        pairs = [
            [query, candidate['content']] 
            for candidate in candidates
        ]
        
        # Get relevance scores
        scores = self.model.predict(pairs)
        
        # Sort by score
        scored_candidates = list(zip(candidates, scores))
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Return top-k with scores
        return [
            {
                **candidate,
                'relevance_score': float(score),
            }
            for candidate, score in scored_candidates[:top_k]
        ]
    
    def rerank_with_cohere(
        self,
        query: str,
        candidates: List[dict],
        top_k: int = None,
    ) -> List[dict]:
        """
        Alternative: Use Cohere's Rerank API
        
        Pros: No model management, production-ready
        Cons: Cost, latency
        """
        import cohere
        
        client = cohere.Client()
        
        documents = [c['content'] for c in candidates]
        
        response = client.rerank(
            model='rerank-english-v3.0',
            query=query,
            documents=documents,
            top_n=top_k or self.top_k,
        )
        
        reranked = []
        for result in response.results:
            candidate = candidates[result.index].copy()
            candidate['relevance_score'] = result.relevance_score
            reranked.append(candidate)
        
        return reranked
```

### Metadata Filtering

```python
class MetadataFiltering:
    """
    Filter results by metadata BEFORE or AFTER retrieval
    
    Use cases:
    - Source filtering (only from specific docs)
    - Date range (only recent info)
    - Permission-based (user can only see X)
    - Document type (only FAQs, not policies)
    """
    
    def build_filter(
        self,
        source_names: List[str] = None,
        date_range: tuple = None,
        document_types: List[str] = None,
        custom_filters: dict = None,
    ) -> dict:
        """
        Build filter for vector database
        
        Pinecone filter example:
        """
        
        filters = {}
        
        if source_names:
            filters['source_name'] = {'$in': source_names}
        
        if date_range:
            filters['ingested_at'] = {
                '$gte': date_range[0].isoformat(),
                '$lte': date_range[1].isoformat(),
            }
        
        if document_types:
            filters['document_type'] = {'$in': document_types}
        
        if custom_filters:
            filters.update(custom_filters)
        
        return filters
    
    def apply_post_filter(
        self,
        results: List[dict],
        filter_fn,
    ) -> List[dict]:
        """
        Apply custom filter after retrieval
        
        Use when: Complex filtering not supported by vector DB
        """
        
        return [r for r in results if filter_fn(r)]
    
    # Example: Permission-based filtering
    def permission_filter(
        self,
        user_permissions: List[str],
    ):
        """Filter results based on user permissions"""
        
        def filter_fn(result):
            required_permission = result.get('metadata', {}).get(
                'required_permission', 
                'public'
            )
            return required_permission in user_permissions
        
        return filter_fn
```

---

## 8. Phase 6: Generation & Response

### Prompt Engineering for RAG

```python
class RAGPromptBuilder:
    """
    Build effective prompts for RAG generation
    
    Key principles:
    1. Clear role and task
    2. Context from retrieval
    3. Answer format specification
    4. Citation requirements
    5. Guardrails
    """
    
    def build_prompt(
        self,
        query: str,
        retrieved_chunks: List[dict],
        conversation_history: List[dict] = None,
        system_prompt: str = None,
    ) -> str:
        """
        Build complete prompt for LLM
        """
        
        # Default system prompt
        if system_prompt is None:
            system_prompt = self._default_system_prompt()
        
        # Build context from chunks
        context = self._build_context(retrieved_chunks)
        
        # Build full prompt
        prompt = f"""{system_prompt}

## Context from Knowledge Base

{context}

## Instructions

1. Answer the user's question using ONLY the context above
2. If the answer is not in the context, say "I don't have information about that"
3. Cite your sources using [1], [2], etc.
4. Be concise but complete
5. If you're unsure, acknowledge the uncertainty

## User Question

{query}

## Answer

"""
        
        return prompt
    
    def _default_system_prompt(self) -> str:
        return """You are a helpful assistant with access to a knowledge base.

Your role:
- Provide accurate, cited answers
- Admit when you don't know
- Never make up information
- Use the provided context as your primary source"""
    
    def _build_context(self, chunks: List[dict]) -> str:
        """
        Format retrieved chunks for the prompt
        
        Include:
        - Chunk content
        - Source information
        - Relevance score (optional)
        """
        
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            source = chunk.get('metadata', {}).get('source_name', 'Unknown')
            context_parts.append(f"""
[Source {i}: {source}]
{chunk['content']}
---
""")
        
        return "\n".join(context_parts)
```

### Response Generation

```python
class LLMGenerator:
    """
    Generate responses with streaming, citations, and guardrails
    """
    
    def __init__(
        self,
        model: str = "gpt-4o",
        temperature: float = 0.1,  # Low for factual accuracy
        max_tokens: int = 1000,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
    
    async def generate(
        self,
        prompt: str,
        stream: bool = True,
    ):
        """
        Generate response with optional streaming
        """
        
        from openai import AsyncOpenAI
        
        client = AsyncOpenAI()
        
        if stream:
            return self._stream_response(client, prompt)
        else:
            return await self._complete_response(client, prompt)
    
    async def _complete_response(self, client, prompt: str) -> str:
        """Get complete response"""
        
        response = await client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        
        return response.choices[0].message.content
    
    async def _stream_response(self, client, prompt: str):
        """Stream response token by token"""
        
        stream = await client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stream=True,
        )
        
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    
    def parse_citations(self, response: str, chunks: List[dict]) -> dict:
        """
        Parse citations from response
        
        Returns:
        {
            'answer': 'Clean answer text',
            'citations': [
                {'reference': '[1]', 'source': 'doc_name', 'content': '...'},
            ]
        }
        """
        
        import re
        
        # Find all citations
        citation_pattern = r'\[(\d+)\]'
        citations = re.findall(citation_pattern, response)
        
        # Build citation metadata
        citation_info = []
        for ref_num in set(citations):
            idx = int(ref_num) - 1
            if idx < len(chunks):
                citation_info.append({
                    'reference': f'[{ref_num}]',
                    'source': chunks[idx].get('metadata', {}).get('source_name'),
                    'content': chunks[idx]['content'][:200] + '...',
                })
        
        # Remove citations from answer
        clean_answer = re.sub(citation_pattern, '', response)
        
        return {
            'answer': clean_answer.strip(),
            'citations': citation_info,
        }
```

### Guardrails

```python
class ResponseGuardrails:
    """
    Safety checks before returning response
    
    Checks:
    1. Hallucination detection
    2. Toxicity/profanity
    3. PII leakage
    4. Prompt injection
    """
    
    def __init__(self, llm_client):
        self.llm = llm_client
    
    async def validate(
        self,
        query: str,
        response: str,
        retrieved_chunks: List[dict],
    ) -> dict:
        """
        Validate response before returning
        
        Returns:
        {
            'is_valid': bool,
            'issues': List[str],
            'confidence': float,
        }
        """
        
        issues = []
        
        # Check 1: Hallucination (is response grounded in context?)
        hallucination_check = await self._check_hallucination(
            query, response, retrieved_chunks
        )
        if not hallucination_check['grounded']:
            issues.append(f"Response may not be grounded in context: {hallucination_check['reason']}")
        
        # Check 2: Toxicity
        toxicity_score = self._check_toxicity(response)
        if toxicity_score > 0.7:
            issues.append("Response contains potentially toxic content")
        
        # Check 3: PII
        pii_detected = self._check_pii(response)
        if pii_detected:
            issues.append("Response may contain PII")
        
        return {
            'is_valid': len(issues) == 0,
            'issues': issues,
            'confidence': 1.0 - (len(issues) * 0.25),
        }
    
    async def _check_hallucination(
        self,
        query: str,
        response: str,
        chunks: List[dict],
    ) -> dict:
        """
        Use LLM to check if response is grounded in retrieved context
        """
        
        context = "\n\n".join([c['content'] for c in chunks[:5]])
        
        prompt = f"""
        Query: {query}
        
        Retrieved Context:
        {context}
        
        Response:
        {response}
        
        Is the response fully supported by the retrieved context?
        Answer YES if all claims are supported, NO if there are unsupported claims.
        Explain briefly.
        
        Format:
        Grounded: YES/NO
        Reason: ...
        """
        
        result = await self.llm.generate(prompt)
        
        return {
            'grounded': 'YES' in result.upper(),
            'reason': result.split('Reason:')[-1].strip() if 'Reason:' in result else '',
        }
    
    def _check_toxicity(self, text: str) -> float:
        """
        Check toxicity using simple heuristics or API
        
        Production: Use Perspective API or similar
        """
        
        # Simple profanity check (use proper library in production)
        profanity_words = ['bad_word1', 'bad_word2']  # Replace with real list
        text_lower = text.lower()
        
        profanity_count = sum(
            1 for word in profanity_words 
            if word in text_lower
        )
        
        return min(profanity_count / 5, 1.0)  # Normalize to 0-1
    
    def _check_pii(self, text: str) -> bool:
        """
        Check for PII patterns
        
        Production: Use Presidio or similar
        """
        
        import re
        
        # Email pattern
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        
        # Phone pattern (US)
        phone_pattern = r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'
        
        # SSN pattern
        ssn_pattern = r'\b\d{3}-\d{2}-\d{4}\b'
        
        return bool(
            re.search(email_pattern, text) or
            re.search(phone_pattern, text) or
            re.search(ssn_pattern, text)
        )
```

---

## 9. Phase 7: Evaluation & Monitoring

### Why Evaluation Matters

```
Without evaluation:
- Don't know if RAG is working
- Can't detect regressions
- No data-driven improvements

With evaluation:
- Track retrieval quality over time
- Identify failure patterns
- Measure user satisfaction
- Justify improvements with metrics
```

### Evaluation Framework

```python
from dataclasses import dataclass
from typing import List
import json

@dataclass
class EvaluationSample:
    query: str
    retrieved_chunks: List[dict]
    generated_answer: str
    ground_truth_answer: str = None
    user_feedback: dict = None

class RAGEvaluator:
    """
    Comprehensive RAG evaluation
    
    Metrics:
    1. Retrieval Quality
       - Precision@K
       - Recall@K
       - MRR (Mean Reciprocal Rank)
       - NDCG (Normalized Discounted Cumulative Gain)
    
    2. Generation Quality
       - Faithfulness (grounded in context)
       - Answer Relevance
       - Context Precision
       - Hallucination rate
    
    3. System Quality
       - Latency
       - Cost per query
       - User satisfaction
    """
    
    def __init__(self, llm_client):
        self.llm = llm_client
        self.samples: List[EvaluationSample] = []
    
    def add_sample(self, sample: EvaluationSample):
        """Add evaluation sample"""
        self.samples.append(sample)
    
    async def evaluate_retrieval(
        self,
        sample: EvaluationSample,
    ) -> dict:
        """
        Evaluate retrieval quality
        
        Requires ground truth for accurate metrics
        """
        
        if not sample.ground_truth_answer:
            return {'error': 'Ground truth required'}
        
        # Embed ground truth and chunks
        gt_embedding = self.embeddings.encode([sample.ground_truth_answer])[0]
        chunk_embeddings = self.embeddings.encode(
            [c['content'] for c in sample.retrieved_chunks]
        )
        
        # Compute similarity
        from sklearn.metrics.pairwise import cosine_similarity
        
        similarities = cosine_similarity(
            [gt_embedding],
            chunk_embeddings
        )[0]
        
        # Precision@K (how many retrieved are relevant?)
        threshold = 0.5
        relevant_count = sum(1 for sim in similarities if sim > threshold)
        precision_at_k = relevant_count / len(sample.retrieved_chunks)
        
        # MRR (was relevant content ranked high?)
        first_relevant_rank = next(
            (i for i, sim in enumerate(similarities) if sim > threshold),
            len(similarities)
        )
        mrr = 1.0 / (first_relevant_rank + 1)
        
        return {
            'precision_at_k': precision_at_k,
            'mrr': mrr,
            'avg_similarity': similarities.mean(),
            'max_similarity': similarities.max(),
        }
    
    async def evaluate_generation(
        self,
        sample: EvaluationSample,
    ) -> dict:
        """
        Evaluate generation quality using LLM-as-judge
        """
        
        # Faithfulness: Is answer grounded in context?
        faithfulness = await self._evaluate_faithfulness(sample)
        
        # Answer Relevance: Does it answer the query?
        relevance = await self._evaluate_relevance(sample)
        
        # Context Precision: Is the right info in context?
        context_precision = await self._evaluate_context_precision(sample)
        
        return {
            'faithfulness': faithfulness,
            'relevance': relevance,
            'context_precision': context_precision,
        }
    
    async def _evaluate_faithfulness(
        self,
        sample: EvaluationSample,
    ) -> dict:
        """
        Check if generated answer is supported by retrieved context
        """
        
        prompt = f"""
        Retrieved Context:
        {json.dumps([c['content'] for c in sample.retrieved_chunks], indent=2)}
        
        Generated Answer:
        {sample.generated_answer}
        
        Rate faithfulness (1-5):
        1 = Completely hallucinated, not in context
        3 = Partially supported, some claims not in context
        5 = Every claim is supported by the context
        
        Provide score and brief explanation.
        
        Format:
        Score: 1-5
        Explanation: ...
        """
        
        result = await self.llm.generate(prompt)
        
        # Parse score
        score_line = [l for l in result.split('\n') if 'Score:' in l][0]
        score = int(score_line.split(':')[1].strip())
        
        return {
            'score': score / 5.0,  # Normalize to 0-1
            'explanation': result.split('Explanation:')[-1].strip(),
        }
    
    async def _evaluate_relevance(
        self,
        sample: EvaluationSample,
    ) -> dict:
        """
        Check if answer is relevant to the query
        """
        
        prompt = f"""
        Query: {sample.query}
        
        Generated Answer:
        {sample.generated_answer}
        
        Rate relevance (1-5):
        1 = Completely irrelevant
        3 = Partially relevant, misses key points
        5 = Directly and completely answers the query
        
        Format:
        Score: 1-5
        Explanation: ...
        """
        
        result = await self.llm.generate(prompt)
        
        score_line = [l for l in result.split('\n') if 'Score:' in l][0]
        score = int(score_line.split(':')[1].strip())
        
        return {
            'score': score / 5.0,
            'explanation': result.split('Explanation:')[-1].strip(),
        }
    
    async def _evaluate_context_precision(
        self,
        sample: EvaluationSample,
    ) -> dict:
        """
        Check if retrieved context contains information needed to answer
        """
        
        if not sample.ground_truth_answer:
            return {'score': 0, 'explanation': 'No ground truth'}
        
        prompt = f"""
        Query: {sample.query}
        
        Retrieved Context:
        {json.dumps([c['content'] for c in sample.retrieved_chunks], indent=2)}
        
        Ground Truth Answer:
        {sample.ground_truth_answer}
        
        Does the retrieved context contain sufficient information 
        to derive the ground truth answer?
        
        Rate (1-5):
        1 = No relevant information
        3 = Some relevant info, but incomplete
        5 = All necessary information present
        
        Format:
        Score: 1-5
        Explanation: ...
        """
        
        result = await self.llm.generate(prompt)
        
        score_line = [l for l in result.split('\n') if 'Score:' in l][0]
        score = int(score_line.split(':')[1].strip())
        
        return {
            'score': score / 5.0,
            'explanation': result.split('Explanation:')[-1].strip(),
        }
    
    def compute_aggregate_metrics(self) -> dict:
        """
        Compute aggregate metrics across all samples
        """
        
        if not self.samples:
            return {'error': 'No samples'}
        
        retrieval_scores = []
        faithfulness_scores = []
        relevance_scores = []
        
        for sample in self.samples:
            # Aggregate retrieval metrics
            # (would need async calls in real implementation)
            pass
        
        return {
            'total_samples': len(self.samples),
            'avg_faithfulness': sum(faithfulness_scores) / len(faithfulness_scores) if faithfulness_scores else 0,
            'avg_relevance': sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0,
        }
```

### Continuous Monitoring

```python
class RAGMonitor:
    """
    Production monitoring for RAG systems
    
    Track:
    - Query patterns
    - Retrieval quality
    - User feedback
    - Cost and latency
    - Failure modes
    """
    
    def __init__(self, database):
        self.db = database
    
    async def log_query(
        self,
        query: str,
        response: str,
        retrieved_chunks: List[dict],
        latency_ms: float,
        cost_usd: float,
        user_feedback: dict = None,
    ):
        """Log query for monitoring"""
        
        await self.db.insert('query_logs', {
            'timestamp': datetime.now().isoformat(),
            'query': query,
            'response': response,
            'num_chunks_retrieved': len(retrieved_chunks),
            'avg_chunk_score': self._compute_avg_score(retrieved_chunks),
            'latency_ms': latency_ms,
            'cost_usd': cost_usd,
            'user_feedback': user_feedback,
        })
    
    def get_dashboard_metrics(self) -> dict:
        """
        Get metrics for monitoring dashboard
        """
        
        return {
            'queries_last_24h': self._count_queries(hours=24),
            'avg_latency_ms': self._avg_latency(hours=24),
            'avg_user_rating': self._avg_user_rating(hours=24),
            'top_queries': self._get_top_queries(hours=24),
            'failure_rate': self._compute_failure_rate(hours=24),
            'cost_last_24h': self._compute_cost(hours=24),
        }
    
    def detect_degradation(self) -> List[dict]:
        """
        Detect performance degradation
        
        Alerts when:
        - Latency increases > 50%
        - User satisfaction drops > 20%
        - Retrieval quality drops
        """
        
        alerts = []
        
        current_latency = self._avg_latency(hours=1)
        baseline_latency = self._avg_latency(hours=24, end_hours_ago=24)
        
        if current_latency > baseline_latency * 1.5:
            alerts.append({
                'type': 'latency_spike',
                'current': current_latency,
                'baseline': baseline_latency,
                'severity': 'warning',
            })
        
        # Similar checks for other metrics
        
        return alerts
```

### Building Evaluation Dataset

```python
class EvaluationDatasetBuilder:
    """
    Build ground truth dataset for evaluation
    
    Sources:
    1. Historical Q&A logs
    2. Manually curated questions
    3. Synthetic generation
    4. User feedback
    """
    
    def generate_synthetic_questions(
        self,
        documents: List[dict],
        num_questions: int = 100,
    ) -> List[dict]:
        """
        Use LLM to generate evaluation questions from documents
        """
        
        questions = []
        
        for doc in documents[:20]:  # Sample documents
            prompt = f"""
            Document:
            {doc['content'][:2000]}
            
            Generate 5 diverse questions that can be answered 
            from this document. Include:
            - 2 factual questions
            - 2 how-to questions
            - 1 comparison question
            
            Format as JSON array:
            [
                {{"question": "...", "answer": "...", "type": "factual"}},
            ]
            """
            
            result = self.llm.generate(prompt)
            doc_questions = json.loads(result)
            
            for q in doc_questions:
                q['source_document'] = doc['id']
                questions.append(q)
        
        return questions[:num_questions]
    
    def collect_user_feedback(
        self,
        query: str,
        response: str,
    ) -> dict:
        """
        Collect explicit user feedback
        
        UI: Thumbs up/down + optional comment
        """
        
        return {
            'query': query,
            'response': response,
            'feedback_type': 'thumbs_up',  # or 'thumbs_down'
            'comment': '...',  # Optional
            'timestamp': datetime.now().isoformat(),
        }
```

---

## 10. Production Deployment

### Deployment Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    PRODUCTION DEPLOYMENT                     │
└─────────────────────────────────────────────────────────────┘

┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Load      │     │   RAG       │     │   Vector    │
│   Balancer  │────▶│   Service   │────▶│   Database  │
│             │     │  (K8s/Docker)│    │  Cluster    │
└─────────────┘     └─────────────┘     └─────────────┘
                           │
                           ▼
                    ┌─────────────┐
                    │   LLM       │
                    │   (API)     │
                    └─────────────┘
                           │
                           ▼
                    ┌─────────────┐
                    │ Monitoring  │
                    │ & Logging   │
                    └─────────────┘
```

### Docker Deployment

```dockerfile
# Dockerfile for RAG service
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```yaml
# docker-compose.yml
version: '3.8'

services:
  rag-service:
    build: .
    ports:
      - "8000:8000"
    environment:
      - VECTOR_DB_URL=qdrant:6333
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    depends_on:
      - qdrant
    deploy:
      replicas: 3
      resources:
        limits:
          cpus: '2'
          memory: 4G

  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
    volumes:
      - qdrant_storage:/qdrant/storage

volumes:
  qdrant_storage:
```

### Kubernetes Deployment

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: rag-service
spec:
  replicas: 3
  selector:
    matchLabels:
      app: rag-service
  template:
    metadata:
      labels:
        app: rag-service
    spec:
      containers:
      - name: rag-service
        image: rag-service:latest
        ports:
        - containerPort: 8000
        env:
        - name: VECTOR_DB_URL
          value: "qdrant.rag-system.svc.cluster.local:6333"
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: api-secrets
              key: openai-key
        resources:
          requests:
            memory: "2Gi"
            cpu: "1"
          limits:
            memory: "4Gi"
            cpu: "2"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
---
apiVersion: v1
kind: Service
metadata:
  name: rag-service
spec:
  selector:
    app: rag-service
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
```

---

## 11. Common Failure Modes & Solutions

### Failure Mode 1: Poor Retrieval Quality

```
Symptoms:
- Retrieved chunks don't answer the query
- High hallucination rate

Root Causes:
1. Bad chunking (context lost)
2. Wrong embedding model
3. No hybrid retrieval

Solutions:
✓ Use semantic chunking for complex docs
✓ Switch to BGE-large or text-embedding-3-large
✓ Implement hybrid retrieval (vector + BM25)
✓ Add reranking step
```

### Failure Mode 2: Stale Information

```
Symptoms:
- Users report outdated answers
- Index doesn't reflect recent changes

Root Causes:
1. No incremental updates
2. Long re-indexing cycles

Solutions:
✓ Implement incremental update pipeline
✓ Schedule full re-index weekly
✓ Add change detection (checksums)
✓ Real-time webhook for critical updates
```

### Failure Mode 3: High Latency

```
Symptoms:
- Response time > 5 seconds
- Users abandon queries

Root Causes:
1. Too many retrieved chunks
2. Slow reranking
3. No caching

Solutions:
✓ Reduce top_k before reranking (50 → 20)
✓ Use faster reranker (bge-reranker-base)
✓ Implement query caching
✓ Use streaming responses
```

### Failure Mode 4: Hallucinations

```
Symptoms:
- LLM makes up information
- Citations don't match content

Root Causes:
1. Weak system prompt
2. Low temperature
3. No guardrails

Solutions:
✓ Strengthen system prompt ("use ONLY context")
✓ Set temperature = 0.1
✓ Add hallucination detection guardrail
✓ Include "I don't know" in training
```

---

## 12. Complete Implementation Example

### Full RAG Pipeline

```python
"""
Complete RAG Pipeline Implementation

This is a production-ready example combining all components
"""

import asyncio
from typing import List, Dict, Optional
from dataclasses import dataclass
from openai import AsyncOpenAI
from sentence_transformers import SentenceTransformer, CrossEncoder
import numpy as np

@dataclass
class RAGConfig:
    # Embeddings
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536
    
    # Vector DB
    vector_db_type: str = "qdrant"
    vector_db_url: str = "http://localhost:6333"
    collection_name: str = "rag_collection"
    
    # Retrieval
    retrieval_top_k: int = 50
    rerank_top_k: int = 5
    
    # Generation
    llm_model: str = "gpt-4o"
    temperature: float = 0.1
    
    # Reranking
    reranker_model: str = "BAAI/bge-reranker-large"

class ProductionRAGPipeline:
    """
    Complete RAG pipeline from ingestion to generation
    """
    
    def __init__(self, config: RAGConfig):
        self.config = config
        
        # Initialize components
        self.openai = AsyncOpenAI()
        self.embedding_model = SentenceTransformer('BAAI/bge-large-en-v1.5')
        self.reranker = CrossEncoder(config.reranker_model)
        
        # Vector DB connection
        self.vector_db = self._connect_vector_db()
        
        # Monitoring
        self.query_log = []
    
    def _connect_vector_db(self):
        """Connect to Qdrant"""
        from qdrant_client import QdrantClient
        
        return QdrantClient(url=self.config.vector_db_url)
    
    async def index_documents(
        self,
        documents: List[Dict],
        chunk_size: int = 512,
        chunk_overlap: int = 50,
    ):
        """
        Complete indexing pipeline
        
        documents: List of {
            'id': str,
            'content': str,
            'metadata': dict
        }
        """
        
        print(f"Indexing {len(documents)} documents...")
        
        # Step 1: Chunk documents
        all_chunks = []
        for doc in documents:
            chunks = self._recursive_chunk(
                doc['content'],
                chunk_size,
                chunk_overlap,
                doc['id'],
                doc.get('metadata', {})
            )
            all_chunks.extend(chunks)
        
        print(f"Created {len(all_chunks)} chunks")
        
        # Step 2: Generate embeddings
        chunk_texts = [chunk['content'] for chunk in all_chunks]
        embeddings = self.embedding_model.encode(
            chunk_texts,
            batch_size=32,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
        
        # Step 3: Store in vector DB
        await self._store_in_vector_db(all_chunks, embeddings)
        
        print(f"Indexed {len(all_chunks)} chunks successfully")
    
    def _recursive_chunk(
        self,
        content: str,
        chunk_size: int,
        overlap: int,
        doc_id: str,
        metadata: dict,
    ) -> List[Dict]:
        """Recursive chunking with structure preservation"""
        
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            separators=["\n\n\n", "\n\n", "\n", ". ", " ", ""],
        )
        
        texts = splitter.split_text(content)
        
        chunks = []
        for i, text in enumerate(texts):
            chunks.append({
                'id': f"{doc_id}_chunk_{i}",
                'content': text,
                'metadata': {
                    **metadata,
                    'chunk_index': i,
                    'total_chunks': len(texts),
                    'doc_id': doc_id,
                }
            })
        
        return chunks
    
    async def _store_in_vector_db(
        self,
        chunks: List[Dict],
        embeddings: np.ndarray,
    ):
        """Store chunks with embeddings in Qdrant"""
        
        from qdrant_client.models import PointStruct, VectorParams, Distance
        
        # Create collection if not exists
        if not await self._collection_exists():
            await self.vector_db.create_collection(
                collection_name=self.config.collection_name,
                vectors_config=VectorParams(
                    size=embeddings.shape[1],
                    distance=Distance.COSINE,
                ),
            )
        
        # Prepare points
        points = []
        for chunk, embedding in zip(chunks, embeddings):
            point = PointStruct(
                id=hash(chunk['id']) % (2**63),  # Qdrant needs int IDs
                vector=embedding.tolist(),
                payload={
                    'content': chunk['content'],
                    **chunk['metadata'],
                },
            )
            points.append(point)
        
        # Upsert in batches
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i:i+batch_size]
            await self.vector_db.upsert(
                collection_name=self.config.collection_name,
                points=batch,
            )
    
    async def _collection_exists(self) -> bool:
        """Check if collection exists"""
        try:
            await self.vector_db.get_collection(self.config.collection_name)
            return True
        except:
            return False
    
    async def query(
        self,
        question: str,
        filters: Optional[Dict] = None,
        stream: bool = False,
    ) -> Dict:
        """
        Complete query pipeline
        
        Returns: {
            'answer': str,
            'sources': List[Dict],
            'latency_ms': float,
        }
        """
        
        import time
        start_time = time.time()
        
        # Step 1: Embed query
        query_embedding = self.embedding_model.encode([question])[0]
        
        # Step 2: Retrieve from vector DB
        search_results = await self.vector_db.search(
            collection_name=self.config.collection_name,
            query_vector=query_embedding.tolist(),
            limit=self.config.retrieval_top_k,
            filter=filters,
        )
        
        # Step 3: Rerank
        reranked = self._rerank_results(question, search_results)
        
        # Step 4: Generate answer
        answer = await self._generate_answer(question, reranked)
        
        # Step 5: Log query
        latency_ms = (time.time() - start_time) * 1000
        await self._log_query(question, answer, reranked, latency_ms)
        
        return {
            'answer': answer,
            'sources': [
                {
                    'content': r['content'],
                    'source': r.get('metadata', {}).get('source_name'),
                    'score': r.get('score', 0),
                }
                for r in reranked[:self.config.rerank_top_k]
            ],
            'latency_ms': latency_ms,
        }
    
    def _rerank_results(
        self,
        query: str,
        results: List[Dict],
    ) -> List[Dict]:
        """Rerank retrieved results"""
        
        # Prepare pairs
        pairs = [[query, result['content']] for result in results]
        
        # Get scores
        scores = self.reranker.predict(pairs)
        
        # Sort by score
        scored_results = []
        for result, score in zip(results, scores):
            result['score'] = float(score)
            scored_results.append(result)
        
        scored_results.sort(key=lambda x: x['score'], reverse=True)
        
        return scored_results[:self.config.rerank_top_k]
    
    async def _generate_answer(
        self,
        question: str,
        chunks: List[Dict],
    ) -> str:
        """Generate answer using LLM"""
        
        # Build prompt
        context = "\n\n".join([
            f"[Source {i+1}]: {chunk['content']}"
            for i, chunk in enumerate(chunks[:5])
        ])
        
        prompt = f"""You are a helpful assistant with access to a knowledge base.

## Context

{context}

## Instructions

1. Answer using ONLY the context above
2. If the answer is not in the context, say so
3. Cite sources as [1], [2], etc.
4. Be concise but complete

## Question

{question}

## Answer

"""
        
        # Generate
        response = await self.openai.chat.completions.create(
            model=self.config.llm_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.config.temperature,
            max_tokens=1000,
        )
        
        return response.choices[0].message.content
    
    async def _log_query(
        self,
        question: str,
        answer: str,
        chunks: List[Dict],
        latency_ms: float,
    ):
        """Log query for monitoring"""
        
        self.query_log.append({
            'timestamp': asyncio.get_event_loop().time(),
            'question': question,
            'answer': answer,
            'num_chunks': len(chunks),
            'latency_ms': latency_ms,
        })
        
        # Keep only last 1000 queries
        if len(self.query_log) > 1000:
            self.query_log = self.query_log[-1000:]
    
    def get_metrics(self) -> Dict:
        """Get performance metrics"""
        
        if not self.query_log:
            return {'error': 'No queries yet'}
        
        latencies = [q['latency_ms'] for q in self.query_log]
        
        return {
            'total_queries': len(self.query_log),
            'avg_latency_ms': np.mean(latencies),
            'p95_latency_ms': np.percentile(latencies, 95),
            'p99_latency_ms': np.percentile(latencies, 99),
        }


# Usage Example
async def main():
    # Initialize pipeline
    config = RAGConfig()
    rag = ProductionRAGPipeline(config)
    
    # Index documents
    documents = [
        {
            'id': 'doc_1',
            'content': 'Your document content here...',
            'metadata': {'source_name': 'Documentation'},
        },
        # ... more documents
    ]
    
    await rag.index_documents(documents)
    
    # Query
    result = await rag.query("What is the pricing?")
    
    print(f"Answer: {result['answer']}")
    print(f"Sources: {result['sources']}")
    print(f"Latency: {result['latency_ms']:.2f}ms")
    
    # Metrics
    metrics = rag.get_metrics()
    print(f"Metrics: {metrics}")

if __name__ == "__main__":
    asyncio.run(main())
```

---

## Summary: RAG Pipeline Checklist

### ✅ Pre-Production

- [ ] Data sources identified and prioritized (start with 10-20% primary sources)
- [ ] Ingestion pipeline with incremental updates
- [ ] Document parsing with proper tools (LlamaParse/Unstructured)
- [ ] Chunking strategy selected (recursive for general, semantic for high-value)
- [ ] Embedding model chosen (text-embedding-3-small or BGE-large)
- [ ] Vector database deployed (Qdrant/Pinecone/Weaviate)
- [ ] Hybrid retrieval implemented (vector + BM25)
- [ ] Reranker configured (BGE-reranker-large or Cohere)
- [ ] Prompt templates with guardrails
- [ ] Evaluation dataset created (100+ Q&A pairs)

### ✅ Production

- [ ] Streaming responses enabled
- [ ] Citation tracking implemented
- [ ] Query logging active
- [ ] Latency monitoring (< 3s p95)
- [ ] Cost tracking per query
- [ ] User feedback collection
- [ ] Alert system for degradation
- [ ] Incremental update pipeline running
- [ ] Weekly full re-index scheduled

### ✅ Post-Launch

- [ ] Review user feedback weekly
- [ ] Analyze failure cases
- [ ] Update evaluation dataset
- [ ] Tune chunk size based on performance
- [ ] Optimize retrieval parameters
- [ ] Add new data sources gradually

---

**Last Updated:** March 25, 2026  
**Based on:** Kapa.ai RAG Pipeline Guide + Production Best Practices