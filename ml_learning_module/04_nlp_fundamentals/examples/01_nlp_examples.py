"""
NLP Fundamentals - Worked Examples
=================================

Step-by-step examples for text processing and embeddings.

Author: AI-Mastery-2026
"""

import numpy as np
import re
from collections import Counter


def example_text_preprocessing():
    """Example 1: Complete Text Preprocessing Pipeline"""
    print("=" * 60)
    print("Example 1: Text Preprocessing Pipeline")
    print("=" * 60)

    raw_text = """
    The QUICK brown fox jumps OVER the lazy dog! 
    It's not what you think, it's 100% amazing!
    Don't miss this: visit http://example.com for more info.
    """

    print("Original text:")
    print(raw_text)

    # Step 1: Lowercase
    text = raw_text.lower()
    print("\n1. Lowercase:")
    print(text)

    # Step 2: Remove URLs
    text = re.sub(r"http\S+|www\.\S+", "", text)
    print("\n2. Remove URLs:")
    print(text)

    # Step 3: Expand contractions
    contractions = {"don't": "do not", "it's": "it is", "can't": "cannot"}
    for old, new in contractions.items():
        text = text.replace(old, new)
    print("\n3. Expand contractions:")
    print(text)

    # Step 4: Remove special characters
    text = re.sub(r"[^a-z\s]", "", text)
    print("\n4. Remove special characters:")
    print(text)

    # Step 5: Tokenize
    tokens = text.split()
    print("\n5. Tokenize:")
    print(tokens)

    # Step 6: Remove stopwords
    stopwords = {
        "the",
        "a",
        "an",
        "is",
        "are",
        "was",
        "were",
        "over",
        "what",
        "you",
        "for",
    }
    tokens = [t for t in tokens if t not in stopwords]
    print("\n6. Remove stopwords:")
    print(tokens)

    # Step 7: Join back
    cleaned = " ".join(tokens)
    print("\n7. Final cleaned text:")
    print(cleaned)


def example_bag_of_words():
    """Example 2: Bag of Words Implementation"""
    print("\n" + "=" * 60)
    print("Example 2: Bag of Words")
    print("=" * 60)

    # Sample documents
    documents = [
        "the cat sat on the mat",
        "the dog played in the garden",
        "a cat is a small furry animal",
        "dogs are loyal companions",
        "cats and dogs are pets",
    ]

    # Build vocabulary
    vocab = {}
    for doc in documents:
        for word in doc.split():
            if word not in vocab:
                vocab[word] = len(vocab)

    print(f"Vocabulary ({len(vocab)} words):")
    print(vocab)

    # Create feature vectors
    def bow_vector(doc, vocab):
        vector = [0] * len(vocab)
        for word in doc.split():
            if word in vocab:
                vector[vocab[word]] += 1
        return vector

    print("\nDocument vectors:")
    for i, doc in enumerate(documents):
        vec = bow_vector(doc, vocab)
        print(f"  Doc {i + 1}: {vec[:8]}...")  # Show first 8


def example_tfidf():
    """Example 3: TF-IDF Calculation"""
    print("\n" + "=" * 60)
    print("Example 3: TF-IDF")
    print("=" * 60)

    documents = [
        "the cat sat on the mat",
        "the dog played in the garden",
        "cat and dog are pets",
    ]

    # Build vocabulary
    vocab = {}
    for doc in documents:
        for word in doc.split():
            if word not in vocab:
                vocab[word] = len(vocab)

    print(f"Vocabulary: {vocab}")

    # Calculate TF
    def compute_tf(doc, vocab):
        tf = [0] * len(vocab)
        words = doc.split()
        for word in words:
            if word in vocab:
                tf[vocab[word]] += 1
        # Normalize by total words
        return [x / len(words) for x in tf]

    # Calculate IDF
    n_docs = len(documents)
    idf = [0] * len(vocab)
    for word, idx in vocab.items():
        doc_count = sum(1 for doc in documents if word in doc.split())
        idf[idx] = np.log(n_docs / doc_count)

    print(f"\nIDF values: {dict(zip(vocab.keys(), [f'{x:.2f}' for x in idf]))}")

    # Calculate TF-IDF for first document
    tf = compute_tf(documents[0], vocab)
    tfidf = [t * i for t, i in zip(tf, idf)]

    print(f"\nDocument 1: '{documents[0]}'")
    print(f"TF: {[f'{x:.2f}' for x in tf[:5]]}")
    print(f"IDF: {[f'{x:.2f}' for x in idf[:5]]}")
    print(f"TF-IDF: {[f'{x:.3f}' for x in tfidf[:5]]}")


def example_word_embeddings():
    """Example 4: Word Embedding Properties"""
    print("\n" + "=" * 60)
    print("Example 4: Word Embedding Arithmetic")
    print("=" * 60)

    # Simulated word vectors (in reality, these come from training)
    # These are simplified for demonstration
    vectors = {
        "king": np.array([0.8, 0.9, 0.1]),
        "queen": np.array([0.8, 0.85, 0.15]),
        "man": np.array([0.7, 0.2, 0.3]),
        "woman": np.array([0.7, 0.25, 0.35]),
        "prince": np.array([0.75, 0.6, 0.2]),
        "princess": np.array([0.75, 0.55, 0.25]),
    }

    def cosine_similarity(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    print("Word Embedding Properties:")
    print("-" * 40)

    # Similarity
    print("\n1. Similarity between related words:")
    sim_king_queen = cosine_similarity(vectors["king"], vectors["queen"])
    sim_king_man = cosine_similarity(vectors["king"], vectors["man"])
    print(f"   king-queen: {sim_king_queen:.4f}")
    print(f"   king-man: {sim_king_man:.4f}")

    # Analogies
    print("\n2. Vector arithmetic (analogy):")
    # king - man + woman ≈ queen
    result = vectors["king"] - vectors["man"] + vectors["woman"]
    print(f"   king - man + woman = {result}")
    print(f"   Expected queen: {vectors['queen']}")
    print(f"   Similarity to queen: {cosine_similarity(result, vectors['queen']):.4f}")


def example_positional_encoding():
    """Example 5: Positional Encoding"""
    print("\n" + "=" * 60)
    print("Example 5: Positional Encoding")
    print("=" * 60)

    # Sinusoidal positional encoding
    d_model = 16  # Embedding dimension
    max_len = 10  # Max sequence length
    position = np.arange(max_len).reshape(-1, 1)
    div_term = np.exp(np.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))

    # Create encoding matrix
    pe = np.zeros((max_len, d_model))
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)

    print("Positional Encoding Matrix (first 4 positions):")
    print("-" * 50)
    print(pe[:4, :8])  # Show first 4 positions, first 8 dimensions

    print("\nKey properties:")
    print(f"1. Each position has unique encoding")
    print(f"2. Encoding dimension: {pe.shape}")
    print(f"3. Values range: [{pe.min():.2f}, {pe.max():.2f}]")

    # Show uniqueness
    print("\nPosition 0 vs Position 5 (first 4 dims):")
    print(f"  Pos 0: {pe[0, :4]}")
    print(f"  Pos 5: {pe[5, :4]}")
    print(f"  Different: {not np.allclose(pe[0, :4], pe[5, :4])}")


def example_attention():
    """Example 6: Attention Mechanism"""
    print("\n" + "=" * 60)
    print("Example 6: Scaled Dot-Product Attention")
    print("=" * 60)

    # Simple example
    batch_size = 1
    seq_len = 3
    d_k = 4  # Key dimension

    # Sample Q, K, V
    np.random.seed(42)
    Q = np.random.randn(batch_size, seq_len, d_k)
    K = np.random.randn(batch_size, seq_len, d_k)
    V = np.random.randn(batch_size, seq_len, d_k)

    print(f"Q, K, V shapes: ({batch_size}, {seq_len}, {d_k})")

    # Step 1: Compute attention scores
    scores = np.matmul(Q, K.transpose(0, 2, 1)) / np.sqrt(d_k)
    print(f"\nAttention scores (before softmax):")
    print(scores[0])

    # Step 2: Softmax
    # Numerically stable softmax
    exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    attention_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

    print(f"\nAttention weights (after softmax):")
    print(attention_weights[0])
    print(f"Sum per position: {attention_weights[0].sum(axis=1)}")

    # Step 3: Apply to values
    output = np.matmul(attention_weights, V)

    print(f"\nOutput shape: {output.shape}")
    print(f"Output (first position): {output[0, 0]}")


def run_all_examples():
    """Run all NLP examples"""
    example_text_preprocessing()
    example_bag_of_words()
    example_tfidf()
    example_word_embeddings()
    example_positional_encoding()
    example_attention()


if __name__ == "__main__":
    run_all_examples()

    print("\n" + "=" * 60)
    print("All NLP examples completed!")
    print("=" * 60)
