# 🔍 Tutorial 4: Introduction to RAG

**Build your first Retrieval-Augmented Generation system in 60 minutes**

---

## 🎯 What You'll Build

By the end of this tutorial, you will have:

- ✅ Understood what RAG is and why it matters
- ✅ Built a simple RAG system from scratch
- ✅ Created document chunking pipeline
- ✅ Implemented vector search with embeddings
- ✅ Generated answers with context
- ✅ Tested with real questions

**Time Required:** 60 minutes  
**Difficulty:** ⭐⭐⭐☆☆ (Intermediate)  
**Prerequisites:** Tutorials 1-3, basic Python knowledge

---

## 📋 What You'll Learn

- What is Retrieval-Augmented Generation (RAG)?
- Why RAG reduces hallucinations
- Document chunking strategies
- Embeddings and vector search
- Context-augmented generation
- Building a complete RAG pipeline

---

## 🧠 Step 1: Understand RAG (10 minutes)

### What is RAG?

**RAG = Retrieval + Generation**

Instead of asking an LLM to answer from memory (which causes hallucinations), RAG:

1. **Retrieves** relevant documents from a knowledge base
2. **Augments** the prompt with those documents
3. **Generates** an answer grounded in real data

### Why RAG Matters

**Without RAG:**
```
User: "What is AI-Mastery-2026's certification policy?"
LLM: *Makes up an answer* (hallucination)
```

**With RAG:**
```
User: "What is AI-Mastery-2026's certification policy?"
System: *Finds relevant documents* → *Augments prompt* → *Answers with citations*
LLM: "According to the documentation, certificates are valid for 3 years..."
```

### RAG Architecture

```
User Query
    ↓
[Embedding Model] → Query Vector
    ↓
[Vector Database] → Retrieve Top-K Documents
    ↓
[Context Builder] → Combine Query + Documents
    ↓
[LLM] → Generate Grounded Answer
    ↓
User receives answer with citations
```

### When to Use RAG

✅ **Use RAG when:**
- You have proprietary/private data
- Answers need to be factual and cited
- Knowledge changes frequently
- Hallucinations are unacceptable

❌ **Don't use RAG when:**
- General knowledge questions
- Creative writing tasks
- Simple classification
- Real-time data needed

---

## 🛠️ Step 2: Setup (5 minutes)

### Install Dependencies

```bash
# Install required libraries
pip install numpy scikit-learn
pip install sentence-transformers  # For embeddings

# Verify installation
python -c "from sentence_transformers import SentenceTransformer; print('✅ Embeddings ready!')"
```

### Import Libraries

```python
import numpy as np
from typing import List, Dict, Tuple
from sklearn.metrics.pairwise import cosine_similarity

# For embeddings (optional - will provide fallback if not installed)
try:
    from sentence_transformers import SentenceTransformer
    HAS_EMBEDDINGS = True
    print("✅ Using SentenceTransformer for embeddings")
except ImportError:
    HAS_EMBEDDINGS = False
    print("💡 Tip: Install sentence-transformers for better embeddings: pip install sentence-transformers")

print("✅ RAG libraries loaded!")
```

---

## 📄 Step 3: Create Sample Knowledge Base (10 minutes)

### What We're Building

We'll create a RAG system that answers questions about **AI-Mastery-2026** using its own documentation.

### Sample Documents

```python
# Sample knowledge base (in real app, these come from files/databases)
KNOWLEDGE_BASE = [
    {
        "id": "doc_1",
        "source": "courses/README.md",
        "content": """AI-Mastery-2026 offers a comprehensive 6-tier educational program. 
        Tier 0 is for absolute beginners with no programming experience. 
        Tier 1 covers mathematical foundations including linear algebra, calculus, and probability. 
        Tier 2 focuses on classical machine learning and deep learning fundamentals. 
        Tier 3 teaches LLM engineering including transformers, RAG systems, and fine-tuning. 
        Tier 4 covers production deployment, MLOps, and scaling. 
        Tier 5 is the capstone project where students build real-world AI systems."""
    },
    {
        "id": "doc_2",
        "source": "courses/CERTIFICATION.md",
        "content": """AI-Mastery-2026 offers 6 certification levels. 
        Level 1 is AI Foundations Certificate for completing Tier 0. 
        Level 2 is ML Fundamentals Certificate for Tier 1. 
        Level 3 is ML Practitioner Certificate for Tier 2. 
        Level 4 is LLM Engineer Certificate for Tier 3, the most popular certification. 
        Level 5 is Production AI Expert Certificate for Tier 4. 
        Level 6 is AI Mastery Capstone Certificate, the highest honor. 
        Certificates for Levels 4-5 are valid for 3 years and require renewal. 
        Levels 1-3 and 6 have lifetime validity."""
    },
    {
        "id": "doc_3",
        "source": "courses/COURSE_CATALOG.md",
        "content": """The program includes 50+ course modules totaling 621 hours of content. 
        There are 67 tutorials across 5 tutorial series. 
        Series 1 covers Getting Started with 10 beginner tutorials. 
        Series 2 provides 15 Deep Dive tutorials for intermediate learners. 
        Series 3 has 12 Production Patterns tutorials for advanced users. 
        Series 4 includes 20 Real-World Project tutorials. 
        Series 5 offers 10 Advanced Topics tutorials for experts. 
        Assessment includes 1000+ quiz questions and 15+ coding challenges."""
    },
    {
        "id": "doc_4",
        "source": "IMPLEMENTATION_ROADMAP.md",
        "content": """The implementation roadmap spans 20 weeks to public launch. 
        Phase 1 (Weeks 1-8) focuses on content completion for all tiers. 
        Phase 2 (Weeks 9-12) builds the LMS platform with FastAPI backend and React frontend. 
        Phase 3 (Weeks 13-16) runs beta testing with 100 users. 
        Phase 4 (Weeks 17-20) executes the public launch. 
        Year 1 target is 15,000 learners with $1.6M revenue projection. 
        Year 3 goal is 100,000+ learners globally with multi-language support."""
    },
    {
        "id": "doc_5",
        "source": "courses/README.md",
        "content": """AI-Mastery-2026 uses a White-Box learning approach. 
        Students first learn the mathematics behind algorithms. 
        Then they implement algorithms from scratch in pure Python. 
        Only after understanding fundamentals do they use production libraries. 
        Every module includes hands-on labs with real-world projects. 
        The program emphasizes production-ready skills including deployment, monitoring, and scaling. 
        Students graduate with a portfolio of 6+ completed projects."""
    },
    {
        "id": "doc_6",
        "source": "docs/02_instructor_guide/INSTRUCTOR_GUIDE.md",
        "content": """Instructors follow a structured teaching methodology. 
        Each 3-hour module is divided into: Introduction (15 min), Theory (45 min), 
        Live Demo (30 min), Hands-On Lab (60 min), Q&A (15 min), and Wrap-Up (15 min). 
        Instructors use the Think-Pair-Share active learning strategy. 
        Live coding is preferred over slide presentations. 
        Student progress is tracked through quizzes, lab submissions, and projects. 
        Early intervention is provided to students showing warning signs of falling behind."""
    }
]

print(f"✅ Knowledge base loaded with {len(KNOWLEDGE_BASE)} documents")
```

---

## ✂️ Step 4: Implement Document Chunking (10 minutes)

### Why Chunk Documents?

LLMs have context limits (typically 4K-128K tokens). We split documents into smaller chunks for:
- Better retrieval accuracy
- Efficient context usage
- More precise answers

### Chunking Implementation

```python
class DocumentChunker:
    """Split documents into overlapping chunks."""
    
    def __init__(self, chunk_size: int = 200, chunk_overlap: int = 50):
        """
        Initialize chunker.
        
        Args:
            chunk_size: Maximum characters per chunk
            chunk_overlap: Overlapping characters between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks.
        
        Args:
            text: Text to chunk
        
        Returns:
            List of text chunks
        """
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence boundary
                last_period = text.rfind('.', start, end)
                last_newline = text.rfind('\n', start, end)
                break_point = max(last_period, last_newline)
                
                if break_point > start:
                    end = break_point + 1
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - self.chunk_overlap
        
        return chunks
    
    def chunk_documents(self, documents: List[Dict]) -> List[Dict]:
        """
        Chunk multiple documents.
        
        Args:
            documents: List of document dicts
        
        Returns:
            List of chunk dicts with metadata
        """
        all_chunks = []
        
        for doc in documents:
            text_chunks = self.chunk_text(doc['content'])
            
            for i, chunk_text in enumerate(text_chunks):
                chunk = {
                    'id': f"{doc['id']}_chunk_{i}",
                    'source': doc['source'],
                    'content': chunk_text,
                    'chunk_index': i,
                    'total_chunks': len(text_chunks)
                }
                all_chunks.append(chunk)
        
        return all_chunks


# Test chunking
chunker = DocumentChunker(chunk_size=150, chunk_overlap=30)
chunks = chunker.chunk_documents(KNOWLEDGE_BASE)

print(f"✅ Created {len(chunks)} chunks from {len(KNOWLEDGE_BASE)} documents")
print(f"\n📄 Sample chunk:")
print(f"ID: {chunks[0]['id']}")
print(f"Source: {chunks[0]['source']}")
print(f"Content: {chunks[0]['content'][:100]}...")
```

---

## 🔢 Step 5: Implement Embeddings (10 minutes)

### What are Embeddings?

Embeddings convert text into **dense vectors** where similar texts have similar vectors.

```
"What is RAG?" → [0.12, -0.45, 0.78, ..., 0.34]  (384 dimensions)
```

### Embedding Implementation

```python
class EmbeddingModel:
    """Generate embeddings for text."""
    
    def __init__(self):
        """Initialize embedding model."""
        if HAS_EMBEDDINGS:
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            self.dimension = 384
            print("✅ Using SentenceTransformer (384 dimensions)")
        else:
            self.model = None
            self.dimension = 100  # Simple TF-IDF-like fallback
            print("💡 Using simple embedding fallback")
    
    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for text.
        
        Args:
            text: Text to embed
        
        Returns:
            Embedding vector
        """
        if self.model:
            embedding = self.model.encode(text)
            return embedding
        else:
            # Simple fallback: word frequency-based embedding
            return self._simple_embedding(text)
    
    def _simple_embedding(self, text: str) -> np.ndarray:
        """Simple embedding based on word frequencies."""
        words = text.lower().split()
        embedding = np.zeros(self.dimension)
        
        for i, word in enumerate(words[:self.dimension]):
            # Simple hash-based mapping
            embedding[i % self.dimension] += 1
        
        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding
    
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of texts
        
        Returns:
            Matrix of embeddings
        """
        embeddings = [self.embed_text(text) for text in texts]
        return np.array(embeddings)


# Test embeddings
embedder = EmbeddingModel()

# Embed chunks
chunk_texts = [chunk['content'] for chunk in chunks]
chunk_embeddings = embedder.embed_texts(chunk_texts)

print(f"\n✅ Generated embeddings for {len(chunks)} chunks")
print(f"   Embedding shape: {chunk_embeddings.shape}")
print(f"   Dimension: {chunk_embeddings.shape[1]}")
```

---

## 🔍 Step 6: Implement Vector Search (10 minutes)

### How Vector Search Works

1. Embed the query
2. Calculate similarity to all chunks
3. Return top-K most similar chunks

### Search Implementation

```python
class VectorSearch:
    """Search for relevant chunks using embeddings."""
    
    def __init__(self, chunks: List[Dict], embeddings: np.ndarray):
        """
        Initialize search index.
        
        Args:
            chunks: List of chunk dicts
            embeddings: Embedding matrix for chunks
        """
        self.chunks = chunks
        self.embeddings = embeddings
    
    def search(self, query: str, embedder: EmbeddingModel, top_k: int = 3) -> List[Tuple[Dict, float]]:
        """
        Search for most relevant chunks.
        
        Args:
            query: User query
            embedder: Embedding model
            top_k: Number of results to return
        
        Returns:
            List of (chunk, similarity_score) tuples
        """
        # Embed query
        query_embedding = embedder.embed_text(query)
        
        # Calculate similarities
        similarities = cosine_similarity([query_embedding], self.embeddings)[0]
        
        # Get top-K indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Return results
        results = []
        for idx in top_indices:
            chunk = self.chunks[idx]
            score = float(similarities[idx])
            results.append((chunk, score))
        
        return results


# Test search
search_engine = VectorSearch(chunks, chunk_embeddings)

# Example query
query = "What certifications does AI-Mastery-2026 offer?"
results = search_engine.search(query, embedder, top_k=3)

print(f"\n🔍 Query: {query}")
print(f"\n📊 Top {len(results)} Results:")
for i, (chunk, score) in enumerate(results, 1):
    print(f"\n{i}. Score: {score:.4f}")
    print(f"   Source: {chunk['source']}")
    print(f"   Content: {chunk['content'][:100]}...")
```

---

## 🤖 Step 7: Build RAG Pipeline (10 minutes)

### Complete RAG System

```python
class SimpleRAG:
    """Complete Retrieval-Augmented Generation system."""
    
    def __init__(self, documents: List[Dict]):
        """
        Initialize RAG system.
        
        Args:
            documents: List of document dicts
        """
        print("🔧 Initializing RAG system...")
        
        # Initialize components
        self.chunker = DocumentChunker(chunk_size=200, chunk_overlap=50)
        self.embedder = EmbeddingModel()
        
        # Create chunks
        self.chunks = self.chunker.chunk_documents(documents)
        print(f"   ✅ Created {len(self.chunks)} chunks")
        
        # Generate embeddings
        chunk_texts = [chunk['content'] for chunk in self.chunks]
        self.chunk_embeddings = self.embedder.embed_texts(chunk_texts)
        print(f"   ✅ Generated embeddings")
        
        # Initialize search
        self.search_engine = VectorSearch(self.chunks, self.chunk_embeddings)
        print(f"   ✅ Search engine ready")
        
        print("✅ RAG system initialized!\n")
    
    def retrieve(self, query: str, top_k: int = 3) -> List[Tuple[Dict, float]]:
        """
        Retrieve relevant documents.
        
        Args:
            query: User query
            top_k: Number of documents to retrieve
        
        Returns:
            List of (chunk, score) tuples
        """
        return self.search_engine.search(query, self.embedder, top_k)
    
    def build_context(self, results: List[Tuple[Dict, float]]) -> str:
        """
        Build context from retrieved chunks.
        
        Args:
            results: Search results
        
        Returns:
            Formatted context string
        """
        context_parts = []
        
        for chunk, score in results:
            context_parts.append(
                f"[Source: {chunk['source']}]\n{chunk['content']}"
            )
        
        return "\n\n".join(context_parts)
    
    def generate_prompt(self, query: str, context: str) -> str:
        """
        Build prompt with context.
        
        Args:
            query: User query
            context: Retrieved context
        
        Returns:
            Formatted prompt
        """
        prompt = f"""You are a helpful assistant for AI-Mastery-2026. 
Answer the user's question based ONLY on the provided context. 
If the answer cannot be found in the context, say "I don't have enough information to answer that question."

Context:
{context}

Question: {query}

Answer:"""
        
        return prompt
    
    def answer(self, query: str, top_k: int = 3) -> Dict:
        """
        Answer a question using RAG.
        
        Args:
            query: User question
            top_k: Number of documents to retrieve
        
        Returns:
            Answer dict with query, context, prompt, and sources
        """
        print(f"🔍 Processing query: {query}")
        
        # Step 1: Retrieve
        results = self.retrieve(query, top_k)
        
        # Step 2: Build context
        context = self.build_context(results)
        
        # Step 3: Generate prompt
        prompt = self.generate_prompt(query, context)
        
        # Step 4: Extract sources
        sources = list(set([chunk['source'] for chunk, _ in results]))
        
        return {
            'query': query,
            'context': context,
            'prompt': prompt,
            'sources': sources,
            'retrieved_chunks': len(results)
        }


# Initialize RAG system
rag = SimpleRAG(KNOWLEDGE_BASE)
```

---

## 💬 Step 8: Test RAG System (5 minutes)

### Ask Questions

```python
# Test queries
queries = [
    "What certifications are available?",
    "How long is the program?",
    "What is the white-box approach?",
    "How many tutorials are there?",
    "What is the implementation timeline?",
]

print("=" * 80)
print("🤖 RAG System - Question Answering")
print("=" * 80)

for query in queries:
    result = rag.answer(query, top_k=2)
    
    print(f"\n❓ Question: {query}")
    print(f"📚 Sources: {', '.join(result['sources'])}")
    print(f"📄 Retrieved: {result['retrieved_chunks']} chunks")
    print(f"\n📝 Context Preview:")
    print(result['context'][:200] + "...")
    print("-" * 80)
```

### Interactive Mode

```python
print("\n💬 Interactive RAG Q&A")
print("=" * 80)
print("Ask questions about AI-Mastery-2026 (type 'quit' to exit)\n")

while True:
    query = input("Your question: ").strip()
    
    if not query or query.lower() in ['quit', 'exit', 'q']:
        print("\nThanks for using RAG Q&A! 👋")
        break
    
    result = rag.answer(query, top_k=3)
    
    print(f"\n📚 Sources: {', '.join(result['sources'])}")
    print(f"\n📝 Retrieved Context:")
    print(result['context'])
    print("\n" + "=" * 80 + "\n")
```

---

## 📊 Step 9: Evaluate RAG Quality (5 minutes)

### Simple Evaluation

```python
class RAGEvaluator:
    """Evaluate RAG system quality."""
    
    def __init__(self, rag: SimpleRAG):
        self.rag = rag
    
    def test_retrieval(self, test_cases: List[Tuple[str, List[str]]]):
        """
        Test retrieval accuracy.
        
        Args:
            test_cases: List of (query, expected_sources) tuples
        """
        print("📊 Retrieval Evaluation")
        print("=" * 60)
        
        correct = 0
        total = len(test_cases)
        
        for query, expected_sources in test_cases:
            result = self.rag.answer(query, top_k=3)
            retrieved_sources = result['sources']
            
            # Check if any expected source was retrieved
            found = any(src in retrieved_sources for src in expected_sources)
            
            if found:
                correct += 1
                print(f"✅ Query: {query[:50]}...")
            else:
                print(f"❌ Query: {query[:50]}...")
                print(f"   Expected: {expected_sources}")
                print(f"   Got: {retrieved_sources}")
        
        accuracy = correct / total * 100
        print(f"\n📈 Retrieval Accuracy: {accuracy:.1f}% ({correct}/{total})")


# Test cases
test_cases = [
    ("What certifications are offered?", ["courses/CERTIFICATION.md"]),
    ("How many hours of content?", ["courses/COURSE_CATALOG.md"]),
    ("What is the teaching methodology?", ["docs/02_instructor_guide/INSTRUCTOR_GUIDE.md"]),
    ("What is the launch timeline?", ["IMPLEMENTATION_ROADMAP.md"]),
    ("What is the white-box approach?", ["courses/README.md"]),
]

# Run evaluation
evaluator = RAGEvaluator(rag)
evaluator.test_retrieval(test_cases)
```

---

## ✅ Tutorial Checklist

- [ ] Understood RAG architecture
- [ ] Created knowledge base
- [ ] Implemented document chunking
- [ ] Generated embeddings
- [ ] Built vector search
- [ ] Created complete RAG pipeline
- [ ] Tested with real questions
- [ ] Evaluated retrieval accuracy

---

## 🎓 Key Takeaways

1. **RAG = Retrieval + Generation** - Ground answers in real data
2. **Chunking** - Split documents for better retrieval
3. **Embeddings** - Convert text to vectors for similarity search
4. **Vector Search** - Find relevant chunks using cosine similarity
5. **Context Augmentation** - Combine query with retrieved context
6. **Evaluation** - Test retrieval accuracy with known queries

---

## 🚀 Next Steps

1. **Enhance Your RAG:**
   - Use better embeddings (SentenceTransformers)
   - Add reranking (Cross-Encoder)
   - Implement hybrid search (BM25 + Vector)
   - Connect to LLM API for generation

2. **Continue Learning:**
   - Tier 3, Module 3.4: RAG Fundamentals
   - Tier 3, Module 3.5: Advanced RAG Patterns
   - Tutorial Series 2: Deep Dives

3. **Build Projects:**
   - Document Q&A system
   - Customer support chatbot
   - Research paper assistant

---

## 💡 Challenge (Optional)

**Build a production RAG system!**

1. Load real PDF/Markdown files
2. Use OpenAI embeddings
3. Connect to GPT-4 for generation
4. Add citation tracking
5. Deploy as web API

**Share your RAG system in Discord!** 🔍

---

**Tutorial Created:** April 2, 2026  
**Last Updated:** April 2, 2026  
**Estimated Time:** 60 minutes  
**Difficulty:** Intermediate

---

[← Back to Tutorials](../README.md) | [Previous: Chatbot](03-simple-chatbot.md) | [Next: Deploy API](05-deploy-api.md)
