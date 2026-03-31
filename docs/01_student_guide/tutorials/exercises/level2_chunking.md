# 🧪 Exercise 2: Chunking Strategies and Their Impact

## 🎯 Objective
Understand how different chunking strategies affect retrieval quality and experiment with various approaches.

## 📋 Prerequisites
- Understanding of embeddings (Exercise 1)
- Access to sample documents for testing
- Basic understanding of tokenization

## 🧪 Exercise Tasks

### Task 1: Compare Chunking Methods
1. Locate the chunking service in `src/core/chunking.py`
2. Apply different chunking strategies to a sample document (>1000 words):
   - Recursive character splitting (default)
   - Semantic chunking (if available)
   - Sentence-level chunking
   - Paragraph-level chunking
3. Count the number of chunks produced by each method
4. Assess the semantic coherence of each chunk

### Task 2: Tune Chunk Size and Overlap
1. Experiment with different chunk sizes: 256, 512, 1024 tokens
2. Test different overlap values: 0, 64, 128 tokens
3. Measure the impact on retrieval quality using sample queries
4. Document the trade-offs between chunk size and retrieval performance

### Task 3: Evaluate Chunk Quality
1. Create sample queries that require information spanning multiple sentences
2. Test retrieval performance with different chunking strategies
3. Assess whether important context is preserved or broken
4. Identify the best approach for your specific use case

## 🛠️ Implementation Hints
```python
from src.core.chunking import chunk_text_token_aware

# Sample document for testing
sample_doc = """
Paste your sample document here...
"""

# Test different chunking strategies
strategies = [
    {"max_tokens": 256, "overlap": 32},
    {"max_tokens": 512, "overlap": 64},
    {"max_tokens": 1024, "overlap": 128}
]

for strategy in strategies:
    chunks = chunk_text_token_aware(
        text=sample_doc,
        max_tokens=strategy["max_tokens"],
        overlap_tokens=strategy["overlap"]
    )
    print(f"Strategy {strategy}: {len(chunks)} chunks generated")
    
    # Evaluate chunk quality
    avg_chunk_len = sum(len(chunk) for chunk in chunks) / len(chunks)
    print(f"Average chunk length: {avg_chunk_len}")
```

## 🧠 Reflection Questions
1. How does chunk size affect retrieval quality?
2. What role does overlap play in preserving context?
3. Which chunking strategy works best for your document type?
4. How might document structure influence chunking decisions?

## 📊 Success Criteria
- Successfully applied multiple chunking strategies to sample documents
- Quantified the impact of different parameters on chunk count and quality
- Identified optimal settings for your use case
- Understood the trade-offs between different approaches

## 🚀 Challenge Extension
Implement a custom chunking strategy that respects document sections (e.g., chapters, articles) and evaluate its effectiveness.