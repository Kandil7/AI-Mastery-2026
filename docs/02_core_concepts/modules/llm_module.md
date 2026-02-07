# Guide: Module `src/llm`

The `src/llm` module contains from-scratch implementations of the core components that power modern Large Language Models (LLMs). This includes the Transformer architecture, Retrieval-Augmented Generation (RAG) pipelines, and Parameter-Efficient Fine-Tuning (PEFT) techniques.

## 1. `attention.py`

This file is a deep dive into the architecture that started the modern LLM revolution: the Transformer, introduced in "Attention Is All You Need."

### Key Components

*   **Positional Encodings**:
    *   `sinusoidal_positional_encoding`: The original method used in the first Transformer to give the model information about the order of tokens.
    *   `rotary_positional_embedding` (RoPE): A more advanced technique used in modern LLMs like Llama, which encodes relative position information more effectively.

*   **`scaled_dot_product_attention`**:
    *   The core function of the attention mechanism. It calculates how much "attention" a query token should pay to a set of key tokens and returns a weighted sum of the value tokens.
    *   The implementation includes the scaling factor (`1/√d_k`) and support for masking (essential for preventing the model from "cheating" by looking at future tokens in auto-regressive models).

*   **`MultiHeadAttention`**:
    *   A class that wraps the `scaled_dot_product_attention` function.
    *   It runs the attention mechanism multiple times in parallel with different linear projections of the queries, keys, and values. This allows the model to jointly attend to information from different representation subspaces at different positions.

*   **`FeedForwardNetwork`**:
    *   The other key component of a Transformer block. It's a simple, position-wise, two-layer feed-forward network that provides non-linearity and further processing capacity.

*   **`LayerNorm`**:
    *   A normalization technique that is critical for stabilizing the training of deep Transformers. It normalizes the inputs across the features for each token independently.

*   **`TransformerBlock`**:
    *   This class combines `MultiHeadAttention` and a `FeedForwardNetwork`, along with residual connections ("Add") and layer normalization ("Norm"), into a complete Transformer encoder block. The implementation supports the now-standard "Pre-LayerNorm" architecture for improved training stability.

## 2. `rag.py`

This file provides a complete, modular pipeline for Retrieval-Augmented Generation (RAG), a technique for grounding LLM responses in external knowledge.

### Key Components

*   **`Document`**: A simple data class to hold text content and its associated metadata.

*   **`TextChunker`**:
    *   Splits large documents into smaller, manageable chunks that can be embedded and retrieved.
    *   Supports multiple strategies: `'fixed'` size, `'semantic'` (sentence-based), and `'recursive'` splitting.

*   **`EmbeddingModel`**:
    *   A wrapper for an embedding function. It provides a simple interface to convert text chunks into vector embeddings. The file includes a dummy implementation for demonstration purposes.

*   **`Retriever`**:
    *   The core retrieval component. It takes an embedded query and finds the most relevant document chunks from its index using cosine similarity.

*   **`Reranker`**:
    *   An optional component that takes the initial retrieved results and re-orders them for better relevance. This is often done with a more powerful (but slower) cross-encoder model.

*   **`ContextAssembler`**:
    *   Takes the final list of retrieved documents and formats them into a single string to be "stuffed" into the LLM's context window along with the user's query.

*   **`RAGPipeline`**:
    *   A class that orchestrates the entire process: it takes a user query, retrieves relevant documents, assembles a context, and (optionally) passes it to an LLM to generate a final answer.

## 3. `fine_tuning.py`

This file explores Parameter-Efficient Fine-Tuning (PEFT), a set of techniques for adapting a pre-trained LLM to a new task without having to retrain all of its billions of parameters.

### Key Techniques

*   **`LoRALayer` (Low-Rank Adaptation)**:
    *   The core of LoRA. Instead of updating the original weight matrix `W`, LoRA freezes `W` and trains two small, low-rank matrices `A` and `B`. The update is represented as `B @ A`.
    *   This drastically reduces the number of trainable parameters (e.g., by 99%).
    *   The `LinearWithLoRA` class demonstrates how this can be applied to a standard dense layer.

*   **`AdapterLayer`**:
    *   An alternative PEFT method where small, bottleneck-shaped neural networks ("adapters") are inserted between the layers of a pre-trained model. Only the adapters are trained.

*   **Quantization (`quantize_nf4`)**:
    *   A function to simulate 4-bit NormalFloat (NF4) quantization. This is a key part of **QLoRA**, where fine-tuning is performed on a model whose weights have been quantized to a very low precision (e.g., 4-bit), further reducing memory requirements.

---

The `src/llm` module provides a robust, from-scratch foundation for understanding the architecture and application of modern Large Language Models.
