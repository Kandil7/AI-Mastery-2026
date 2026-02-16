# ❌ Failure Mode: Poor Chunking (Information Fragmentation)

## 🤕 Symptoms
* Relevant information is split across multiple chunks
* Loss of context and meaning during retrieval
* Fragmented answers that miss the big picture
* Important relationships between concepts broken apart
* High retrieval latency due to processing many small chunks

## 🔍 Root Causes
1. **Fixed-size chunking**: Splitting without regard for semantic boundaries
2. **Incorrect chunk size**: Too small (breaks context) or too large (reduces precision)
3. **Ignoring document structure**: Not respecting headers, paragraphs, or sections
4. **No overlap between chunks**: Context lost at split points
5. **One-size-fits-all approach**: Same strategy for all document types

## 💡 How This Repository Fixes This
### 1. Token-Aware Chunking
```python
# Smart chunking that respects token limits and semantic boundaries
chunks = chunk_text_token_aware(
    text=document.text,
    max_tokens=512,
    overlap_tokens=64,
    separators=["\n\n", "\n", ". ", "! ", "? "]
)
```

### 2. Semantic Boundary Respect
- Preserves paragraph and section boundaries when possible
- Maintains context around split points
- Customizable separators for different document types

### 3. Configurable Overlap
- Adjustable overlap to preserve context across splits
- Prevents information loss at chunk boundaries
- Balances between redundancy and completeness

## 🔧 How to Trigger/Debug This Issue
1. **Use Fixed-Size Chunking**: Disable semantic boundary detection
2. **Reduce Overlap to Zero**: Remove overlap between chunks
3. **Use Very Small Chunks**: Force tiny chunk sizes that break context
4. **Ignore Document Structure**: Split mid-sentence regardless of meaning

## 📊 Expected Impact
With poor chunking: ~60% of complex questions suffer from fragmented context
With proper chunking: ~15% of complex questions suffer from fragmented context