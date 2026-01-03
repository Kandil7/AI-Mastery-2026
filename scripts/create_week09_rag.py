#!/usr/bin/env python3
"""
Week 09: LLM Engineering & RAG - High-Value Sprint Phase 2
Create comprehensive notebooks on modern LLM applications
"""

import json
from pathlib import Path

BASE_DIR = Path("k:/learning/technical/ai-ml/AI-Mastery-2026/notebooks/week_09")

def nb(cells):
    return {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.10.0"}}, "nbformat": 4, "nbformat_minor": 4}

def md(c): 
    return {"cell_type": "markdown", "metadata": {}, "source": c if isinstance(c, list) else [c]}

def code(c): 
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": c if isinstance(c, list) else [c]}

# TRANSFORMERS & ATTENTION
transformers_cells = [
    md(["# 🎯 Transformers & Attention: Complete Guide\n\n## The Architecture That Changed Everything\n\n**\"Attention is All You Need\"** - Vaswani et al., 2017\n\nComprehensive breakdown of the Transformer architecture and self-attention mechanism.\n\n---\n"]),
    
    code(["import numpy as np\nimport matplotlib.pyplot as plt\nnp.random.seed(42)\nprint('✅ Transformers ready!')\n"]),
    
    md(["## Self-Attention Mathematics\n\n### The Core Innovation\n\n**Input**: Sequence of vectors $X = [x_1, x_2, ..., x_n]$\n\n**Three learned projections**:\n- **Query**: $Q = XW_Q$\n- **Key**: $K = XW_K$  \n- **Value**: $V = XW_V$\n\n**Attention formula**:\n\n$$\\text{Attention}(Q, K, V) = \\text{softmax}\\left(\\frac{QK^T}{\\sqrt{d_k}}\\right)V$$\n\n**Why $\\sqrt{d_k}$?** Prevents dot products from growing too large.\n\n### Multi-Head Attention\n\nRun h parallel attention \"heads\":\n\n$$\\text{MultiHead}(Q,K,V) = \\text{Concat}(\\text{head}_1, ..., \\text{head}_h)W^O$$\n\nwhere each $\\text{head}_i = \\text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$\n\n**Benefit**: Learn different representation subspaces\n"]),
    
    code(["# Simplified self-attention\ndef self_attention(Q, K, V):\n    \"\"\"Scaled dot-product attention.\"\"\"\n    d_k = Q.shape[-1]\n    scores = np.matmul(Q, K.transpose(-2, -1)) / np.sqrt(d_k)\n    attention_weights = np.exp(scores) / np.sum(np.exp(scores), axis=-1, keepdims=True)\n    output = np.matmul(attention_weights, V)\n    return output, attention_weights\n\n# Example\nseq_len, d_model = 4, 8\nX = np.random.randn(seq_len, d_model)\nQ = K = V = X  # Self-attention\n\noutput, weights = self_attention(Q, K, V)\nprint(f'Input shape: {X.shape}')\nprint(f'Output shape: {output.shape}')\nprint(f'Attention weights shape: {weights.shape}')\nprint('✅ Self-attention computed!')\n"]),
    
    md(["## Transformer Architecture\n\n### Encoder\n1. Input Embedding + Positional Encoding\n2. **N layers** of:\n   - Multi-Head Self-Attention\n   - Add & Norm\n   - Feed-Forward Network\n   - Add & Norm\n\n### Decoder  \n1. Output Embedding + Positional Encoding\n2. **N layers** of:\n   - Masked Multi-Head Self-Attention\n   - Add & Norm\n   - Multi-Head Cross-Attention (with encoder)\n   - Add & Norm\n   - Feed-Forward\n   - Add & Norm\n\n### Why Transformers Won\n\n✅ **Parallelizable** (unlike RNNs)\n✅ **Long-range dependencies** (O(1) vs O(n) for RNNs)\n✅ **Scalable** to billions of parameters\n✅ **Transfer learning** (BERT, GPT)\n"]),
]

# RAG PIPELINE
rag_cells = [
    md(["# 🔍 RAG: Retrieval Augmented Generation\n\n## Giving LLMs Long-Term Memory\n\n**Problem**: LLMs have cutoff knowledge dates\n**Solution**: Retrieve relevant context dynamically\n\n---\n"]),
    
    code(["import numpy as np\nprint('✅ RAG concepts ready!')\n"]),
    
    md(["## RAG Architecture\n\n### The Pipeline\n\n1. **Ingestion**\n   - Chunk documents\n   - Generate embeddings\n   - Store in vector DB\n\n2. **Retrieval**\n   - User query → embedding\n   - Similarity search\n   - Top-k relevant chunks\n\n3. **Generation**\n   - Inject chunks into prompt\n   - LLM generates answer\n   - Cite sources\n\n### Chunking Strategies\n\n**Fixed-size**: 512 tokens, overlap 50\n**Semantic**: Split on paragraphs/sentences\n**Recursive**: Hierarchical chunking\n\n### Embedding Models\n\n| Model | Dimensions | Speed | Quality |\n|-------|-----------|-------|----------|\n| **all-MiniLM-L6-v2** | 384 | Fast | Good |\n| **text-embedding-ada-002** | 1536 | Medium | Excellent |\n| **BGE-large** | 1024 | Slow | Best |\n\n### Vector Databases\n\n**Chroma**: Simple, local, great for prototyping\n**Qdrant**: Production-ready, scalable\n**Pinecone**: Managed, serverless\n**Weaviate**: Hybrid search (dense + sparse)\n"]),
    
    code(["# RAG Pipeline (simplified)\nclass SimpleRAG:\n    def __init__(self, documents):\n        self.documents = documents\n        self.embeddings = self._embed_documents(documents)\n    \n    def _embed_documents(self, docs):\n        # Simplified: random embeddings\n        # In practice: use SentenceTransformer\n        return np.random.randn(len(docs), 384)\n    \n    def retrieve(self, query, k=3):\n        # Embed query\n        query_emb = np.random.randn(384)  # Simplified\n        \n        # Compute similarities (cosine)\n        similarities = np.dot(self.embeddings, query_emb)\n        similarities /= (np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_emb))\n        \n        # Top-k\n        top_k_idx = np.argsort(similarities)[-k:][::-1]\n        return [self.documents[i] for i in top_k_idx]\n    \n    def generate(self, query):\n        # Retrieve context\n        context = self.retrieve(query)\n        \n        # Build prompt\n        prompt = f\"Context: {' '.join(context)}\\n\\nQuestion: {query}\\n\\nAnswer:\"\n        \n        # In practice: call LLM API\n        return \"[LLM would generate answer here]\"\n\nprint('✅ RAG pipeline structure!')\n"]),
    
    md(["## Advanced RAG Techniques\n\n### HyDE (Hypothetical Document Embeddings)\n1. Generate hypothetical answer\n2. Embed hypothetical doc\n3. Retrieve similar docs\n\n### Multi-Query\n1. Generate multiple query variations\n2. Retrieve for each\n3. Combine results\n\n### Reranking\n1. Retrieve top-20 with fast method\n2. Rerank with slower, better model\n3. Return top-5\n\n### Production Considerations\n\n✅ **Caching**: Cache embeddings & retrievals\n✅ **Monitoring**: Track retrieval quality\n✅ **Evaluation**: Use RAGAS metrics\n✅ **Cost**: Embedding API calls add up\n"]),
]

# VECTOR DATABASES
vector_db_cells = [
    md(["# 🗄️ Vector Databases\n\n## Storing and Searching Embeddings at Scale\n\n---\n"]),
    
    code(["import numpy as np\nprint('✅ Vector DB concepts!')\n"]),
    
    md(["## Why Vector Databases?\n\n**Traditional DB**: Exact match on keywords\n**Vector DB**: Semantic similarity search\n\n### Core Operations\n\n1. **Insert**: Add vector + metadata\n2. **Search**: Find k nearest neighbors\n3. **Update**: Modify vectors\n4. **Delete**: Remove vectors\n\n### Similarity Metrics\n\n**Cosine**: $\\frac{a \\cdot b}{\\|a\\|\\|b\\|}$ (most common)\n**Euclidean**: $\\sqrt{\\sum(a_i - b_i)^2}$\n**Dot Product**: $a \\cdot b$\n\n### HNSW (Hierarchical Navigable Small World)\n\n**The secret sauce** of fast vector search\n\n- **Graph-based** search structure\n- **Multiple layers**: Coarse → Fine\n- **Complexity**: O(log n) vs O(n) brute force\n- **Trade-off**: Memory for speed\n"]),
    
    md(["## Chroma Example\n\n```python\nimport chromadb\n\nclient = chromadb.Client()\ncollection = client.create_collection('docs')\n\n# Add documents\ncollection.add(\n    documents=['AI is amazing', 'ML is powerful'],\n    ids=['doc1', 'doc2']\n)\n\n# Query\nresults = collection.query(\n    query_texts=['artificial intelligence'],\n    n_results=2\n)\n```\n\n## Qdrant Example\n\n```python\nfrom qdrant_client import QdrantClient\n\nclient = QdrantClient(':memory:')\n\nclient.create_collection(\n    collection_name='docs',\n    vectors_config={'size': 384, 'distance': 'Cosine'}\n)\n\nclient.upsert(\n    collection_name='docs',\n    points=[{'id': 1, 'vector': [0.1]*384}]\n)\n```\n"]),
]

# WEEK 09 INDEX
week09_index = [
    md(["# 📚 Week 09: LLM Engineering & RAG\n\n## Modern AI Applications\n\n### Learning Path\n\n1. **[Transformers & Attention](01_transformers_attention.ipynb)** ⭐\n   - Self-attention mathematics\n   - Multi-head attention\n   - Transformer architecture\n\n2. **[RAG Pipeline](02_rag_pipeline.ipynb)**\n   - Retrieval augmented generation\n   - Chunking strategies\n   - Advanced techniques (HyDE, reranking)\n\n3. **[Vector Databases](03_vector_databases.ipynb)**\n   - Chroma, Qdrant comparison\n   - HNSW algorithm\n   - Production considerations\n\n---\n\n## Integration with Project\n\nSee `src/llm/` and `src/production/` for production implementations.\n"]),
]

if __name__ == "__main__":
    print("🚀 Creating Week 09: LLM & RAG notebooks...\n")
    
    notebooks = {
        "01_transformers_attention.ipynb": nb(transformers_cells),
        "02_rag_pipeline.ipynb": nb(rag_cells),
        "03_vector_databases.ipynb": nb(vector_db_cells),
        "week_09_index.ipynb": nb(week09_index),
    }
    
    for filename, notebook in notebooks.items():
        output = BASE_DIR / filename
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, 'w', encoding='utf-8') as f:
            json.dump(notebook, f, indent=2)
        print(f"✅ {filename}")
    
    print("\n🎉 Week 09 COMPLETE! LLM & RAG foundation ready.")
    print("📊 Total: 4 notebooks covering modern LLM applications")
    print("\n📈 Sprint Progress: 50% complete (Phases 1-2 done)")
    print("📈 Next: Week 13 Production ML (deployment, monitoring)")
