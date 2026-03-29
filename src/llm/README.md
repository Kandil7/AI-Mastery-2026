# LLM Module

**Transformer architectures and Large Language Model implementations.**

## Topics Covered

### Transformer Architecture
- Multi-head attention
- Positional encodings
- Encoder-decoder structure
- BERT architecture
- GPT architecture

### Attention Mechanisms
- Scaled dot-product attention
- Causal masking
- RoPE (Rotary Position Embeddings)
- Flash attention

### Advanced RAG
- Semantic chunking
- Hybrid retrieval
- Reranking
- Query enhancement

### Fine-tuning
- LoRA adapters
- Prompt tuning
- Full fine-tuning

### Agents
- Tool use
- Multi-agent systems
- Planning

## Usage

```python
from src.llm.transformer import Transformer, MultiHeadAttention
from src.llm.attention import scaled_dot_product_attention

# Create transformer
model = Transformer(
    vocab_size=32000,
    d_model=512,
    n_heads=8,
    n_layers=6,
)

# Generate text
output = model.generate(prompt, max_tokens=100)
```

## Related Modules

- [`src/core`](../core/) - Mathematical foundations
- [`src/rag`](../rag/) - RAG pipeline
- [`src/agents`](../agents/) - Agent systems
