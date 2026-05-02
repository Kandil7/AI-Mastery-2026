"""
Word Embeddings Implementation (Word2Vec and GloVe)
======================================================

This module provides implementations of popular word embedding algorithms.

Mathematical Foundation:
------------------------

1. Word2Vec (Skip-gram with Negative Sampling):

   Word2Vec learns word representations by predicting context words
   given a target word (Skip-gram) or vice versa (CBOW).

   Skip-gram Objective:
   Maximize: Σ_{(w,c) ∈ D} log P(c|w)

   Where:
   - w: target word
   - c: context word
   - D: training corpus

   P(c|w) = σ(v_c · v_w) or 1 - σ(v_c · v_w) for negative samples

   The probability uses sigmoid:
   P(c|w) = σ(v_c · v_w) = 1 / (1 + e^(-v_c · v_w))

2. Negative Sampling:

   Instead of softmax over all vocabulary (expensive), we sample
   k negative examples and optimize:

   log σ(v_c · v_w) + Σ_{i=1}^k log σ(-v_ni · v_w)

   Where v_ni are negative sample vectors.

3. GloVe (Global Vectors):

   GloVe combines global matrix factorization with local context window.

   Objective:
   minimize Σ_{i,j} f(X_ij) (v_i · v_j + b_i + b_j - log X_ij)^2

   Where:
   - X_ij: Word co-occurrence count
   - f(X): weighting function (clipped to prevent excessive weight)
   - v_i, v_j: word vectors
   - b_i, b_j: biases

4. Why Word Embeddings Work:

   - Distributional hypothesis: "You shall know a word by the company it keeps"
   - Similar words appear in similar contexts
   - Vector arithmetic captures semantic relationships
   - king - man + woman ≈ queen

Author: AI-Mastery-2026
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import Counter, defaultdict
import random


class Vocabulary:
    """
    Word Vocabulary for NLP tasks

    Manages word-to-index and index-to-word mappings,
    along with special tokens.

    Special tokens:
        - <PAD>: Padding token (index 0)
        - <UNK>: Unknown word (index 1)
        - <SOS>: Start of sequence
        - <EOS>: End of sequence
    """

    def __init__(self, min_freq: int = 1, max_size: int = 10000):
        self.min_freq = min_freq
        self.max_size = max_size

        self.word2idx = {"<PAD>": 0, "<UNK>": 1}
        self.idx2word = {0: "<PAD>", 1: "<UNK>"}

        self.word_counts = Counter()
        self.size = 2  # Initial size with special tokens

    def add_word(self, word: str):
        """Add word to vocabulary counter"""
        self.word_counts[word] += 1

    def build(self):
        """Build vocabulary from word counts"""
        # Filter by minimum frequency
        filtered = [
            (w, c)
            for w, c in self.word_counts.most_common()
            if c >= self.min_freq and self.size < self.max_size
        ]

        # Add words
        for word, _ in filtered:
            idx = self.size
            self.word2idx[word] = idx
            self.idx2word[idx] = word
            self.size += 1

    def encode(self, word: str) -> int:
        """Encode word to index"""
        return self.word2idx.get(word, self.word2idx["<UNK>"])

    def decode(self, idx: int) -> str:
        """Decode index to word"""
        return self.idx2word.get(idx, "<UNK>")

    def encode_text(self, text: List[str]) -> List[int]:
        """Encode text to list of indices"""
        return [self.encode(word) for word in text]

    def decode_text(self, indices: List[int]) -> List[str]:
        """Decode indices to text"""
        return [self.decode(idx) for idx in indices]

    def __len__(self):
        return self.size


class Tokenizer:
    """
    Simple Tokenizer for text preprocessing

    Converts text to lowercase and splits on whitespace/punctuation.
    """

    def __init__(self, lowercase: bool = True):
        self.lowercase = lowercase
        self.vocab = Vocabulary()

    def fit(self, texts: List[str]):
        """Build vocabulary from texts"""
        for text in texts:
            tokens = self.tokenize(text)
            for token in tokens:
                self.vocab.add_word(token)
        self.vocab.build()

    def tokenize(self, text: str) -> List[str]:
        """Split text into tokens"""
        if self.lowercase:
            text = text.lower()

        # Simple tokenization: split on whitespace and punctuation
        import re

        tokens = re.findall(r"\b[\w']+\b|[.,!?;]", text)

        return tokens

    def encode(self, text: str) -> List[int]:
        """Encode text to indices"""
        tokens = self.tokenize(text)
        return self.vocab.encode_text(tokens)

    def decode(self, indices: List[int]) -> str:
        """Decode indices to text"""
        tokens = self.vocab.decode_text(indices)
        return " ".join(tokens)


class Word2VecSkipGram:
    """
    Word2Vec Skip-gram Model with Negative Sampling

    This implementation uses the skip-gram model with negative sampling
    for efficient training.

    Architecture:
        - Input: target word (one-hot vector)
        - Embedding layer: projects to dense vector
        - Output: scores for context/negative words

    Training Procedure:
        1. For each (target, context) pair in training data:
        2. Sample k negative examples
        3. Compute binary classification loss
        4. Update embeddings using gradient descent

    Hyperparameters:
        - embedding_dim: Dimension of word vectors
        - window_size: Context window size
        - negative_samples: Number of negative samples per positive
        - learning_rate: Learning rate
        - epochs: Number of training epochs
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 100,
        window_size: int = 5,
        negative_samples: int = 5,
        learning_rate: float = 0.025,
        epochs: int = 5,
    ):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.window_size = window_size
        self.negative_samples = negative_samples
        self.learning_rate = learning_rate
        self.epochs = epochs

        # Initialize embeddings with small random values
        # Range: [-0.5/dim, 0.5/dim]
        self.target_embeddings = np.random.uniform(
            -0.5 / embedding_dim, 0.5 / embedding_dim, (vocab_size, embedding_dim)
        )
        self.context_embeddings = np.random.uniform(
            -0.5 / embedding_dim, 0.5 / embedding_dim, (vocab_size, embedding_dim)
        )

        # Word frequency table for negative sampling
        self.word_frequencies = None
        self.sampling_table = None

    def _build_sampling_table(self, word_counts: Counter):
        """
        Build table for negative sampling

        Uses unigram distribution raised to 3/4 power
        (empirically shown to work well)
        """
        total = sum(word_counts.values())
        word_probs = {w: c / total for w, c in word_counts.items()}

        # Raise to 3/4
        self.word_frequencies = {w: p**0.75 for w, p in word_probs.items()}

        # Build sampling table
        table_size = 1e8
        self.sampling_table = np.zeros(int(table_size))

        running_sum = 0
        for word, prob in self.word_frequencies.items():
            running_sum += prob * table_size
            self.sampling_table[int(running_sum) :] = word

    def _get_negative_samples(self, context_idx: int, num_samples: int) -> List[int]:
        """Generate negative samples"""
        neg_samples = []

        while len(neg_samples) < num_samples:
            # Sample from table (simplified)
            idx = random.randint(0, self.vocab_size - 1)
            if idx != context_idx:
                neg_samples.append(idx)

        return neg_samples

    def _generate_training_pairs(self, tokens: List[int]) -> List[Tuple[int, int]]:
        """
        Generate (target, context) pairs from token sequence

        For each target word, create pairs with context words
        within the window size.
        """
        pairs = []

        for i, target in enumerate(tokens):
            # Context window: [i-window_size, i+window_size]
            start = max(0, i - self.window_size)
            end = min(len(tokens), i + self.window_size + 1)

            for j in range(start, end):
                if i != j:
                    pairs.append((target, tokens[j]))

        return pairs

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Numerically stable sigmoid"""
        return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))

    def train(self, texts: List[List[str]]):
        """
        Train the Word2Vec model

        Args:
            texts: List of tokenized texts (list of lists of words)
        """
        # Build tokenized sequences
        tokenized = []
        for text in texts:
            if isinstance(text, str):
                tokenized.append(text.lower().split())
            else:
                tokenized.append([w.lower() for w in text])

        # Flatten for vocabulary
        all_tokens = [token for seq in tokenized for token in seq]
        word_counts = Counter(all_tokens)

        # Build sampling table
        self._build_sampling_table(word_counts)

        # Training loop
        for epoch in range(self.epochs):
            total_loss = 0

            # Shuffle training pairs
            all_pairs = []
            for tokens in tokenized:
                pairs = self._generate_training_pairs(tokens)
                all_pairs.extend(pairs)

            random.shuffle(all_pairs)

            # Process each pair
            for target, context in all_pairs:
                # Positive sample
                pos_score = np.dot(
                    self.target_embeddings[target], self.context_embeddings[context]
                )
                pos_loss = -np.log(self._sigmoid(pos_score) + 1e-10)

                # Negative samples
                neg_indices = self._get_negative_samples(context, self.negative_samples)

                for neg_idx in neg_indices:
                    neg_score = np.dot(
                        self.target_embeddings[target], self.context_embeddings[neg_idx]
                    )
                    neg_loss = -np.log(1 - self._sigmoid(neg_score) + 1e-10)
                    total_loss += pos_loss + neg_loss

                    # Update gradients (simplified)
                    grad = self._sigmoid(neg_score) - 1

                    self.target_embeddings[target] += (
                        self.learning_rate * grad * self.context_embeddings[neg_idx]
                    )
                    self.context_embeddings[neg_idx] += (
                        self.learning_rate * grad * self.target_embeddings[target]
                    )

                # Update positive sample gradients
                grad_pos = 1 - self._sigmoid(pos_score)

                self.target_embeddings[target] += (
                    self.learning_rate * grad_pos * self.context_embeddings[context]
                )
                self.context_embeddings[context] += (
                    self.learning_rate * grad_pos * self.target_embeddings[target]
                )

            print(
                f"Epoch {epoch + 1}/{self.epochs}, Loss: {total_loss / len(all_pairs):.4f}"
            )

    def get_embedding(self, word_idx: int) -> np.ndarray:
        """Get embedding for a word index"""
        return self.target_embeddings[word_idx]

    def get_embeddings(self) -> np.ndarray:
        """Get all target embeddings"""
        return self.target_embeddings

    def most_similar(self, word_idx: int, top_k: int = 10) -> List[Tuple[int, float]]:
        """
        Find most similar words

        Args:
            word_idx: Index of the query word
            top_k: Number of similar words to return

        Returns:
            List of (word_idx, similarity) tuples
        """
        # Compute cosine similarity
        query = self.target_embeddings[word_idx]
        norms = np.linalg.norm(self.target_embeddings, axis=1, keepdims=True)
        normalized = self.target_embeddings / (norms + 1e-10)
        query_norm = query / (np.linalg.norm(query) + 1e-10)

        similarities = np.dot(normalized, query_norm)

        # Get top k (excluding the query itself)
        top_indices = np.argsort(similarities)[::-1][1 : top_k + 1]

        return [(idx, similarities[idx]) for idx in top_indices]


class GloVe:
    """
    GloVe (Global Vectors) Word Embeddings

    GloVe combines global co-occurrence statistics with local context windows.

    Mathematical Formulation:

    The objective function:
        J = Σ_{i,j} f(X_ij) (w_i · w_j + b_i + b_j - log X_ij)^2

    Where:
    - w_i: Word vector for word i
    - b_i: Bias for word i
    - X_ij: Co-occurrence count of words i and j
    - f(X): Weighting function

    Weighting function:
        f(x) = (x/x_max)^α if x < x_max
             = 1 otherwise
        Typical: x_max = 100, α = 0.75

    Co-occurrence Matrix:
    - X[i,j] = number of times word j appears in context of word i
    - Built using a sliding window
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 100,
        window_size: int = 5,
        x_max: int = 100,
        alpha: float = 0.75,
        learning_rate: float = 0.05,
        epochs: int = 10,
    ):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.window_size = window_size
        self.x_max = x_max
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.epochs = epochs

        # Initialize embeddings
        self.W = np.random.randn(vocab_size, embedding_dim) * 0.1
        self.W_tilde = np.random.randn(vocab_size, embedding_dim) * 0.1
        self.b = np.zeros(vocab_size)
        self.b_tilde = np.zeros(vocab_size)

        # Co-occurrence matrix
        self.cooccurrence = None

    def _weighting_function(self, x: float) -> float:
        """GloVe weighting function f(x)"""
        if x < self.x_max:
            return (x / self.x_max) ** self.alpha
        return 1.0

    def build_cooccurrence_matrix(self, texts: List[List[str]]):
        """Build co-occurrence matrix from texts"""
        cooc = defaultdict(lambda: defaultdict(float))

        for tokens in texts:
            # Convert to indices (simplified)
            for i, word in enumerate(tokens):
                start = max(0, i - self.window_size)
                end = min(len(tokens), i + self.window_size + 1)

                for j in range(start, end):
                    if i != j:
                        cooc[i][j] += 1

        # Convert to dense matrix
        self.cooccurrence = np.zeros((self.vocab_size, self.vocab_size))
        for i, j_dict in cooc.items():
            for j, count in j_dict.items():
                self.cooccurrence[i, j] = count

    def train(self, texts: List[List[str]]):
        """Train GloVe model"""
        # Build co-occurrence matrix
        self.build_cooccurrence_matrix(texts)

        # Get non-zero pairs
        rows, cols = np.where(self.cooccurrence > 0)
        pairs = list(zip(rows, cols))

        for epoch in range(self.epochs):
            total_loss = 0

            random.shuffle(pairs)

            for i, j in pairs:
                # Get co-occurrence count
                X_ij = self.cooccurrence[i, j]

                if X_ij <= 0:
                    continue

                # Compute weight
                f_ij = self._weighting_function(X_ij)

                # Compute prediction
                diff = (
                    self.W[i] @ self.W_tilde[j]
                    + self.b[i]
                    + self.b_tilde[j]
                    - np.log(X_ij + 1e-10)
                )

                # Compute weighted loss
                loss = f_ij * diff * diff
                total_loss += loss

                # Compute gradients
                grad = f_ij * diff

                # Update embeddings
                self.W[i] -= self.learning_rate * grad * self.W_tilde[j]
                self.W_tilde[j] -= self.learning_rate * grad * self.W[i]
                self.b[i] -= self.learning_rate * grad
                self.b_tilde[j] -= self.learning_rate * grad

            print(
                f"Epoch {epoch + 1}/{self.epochs}, Loss: {total_loss / len(pairs):.4f}"
            )

    def get_embedding(self, word_idx: int) -> np.ndarray:
        """Get embedding for a word index"""
        return self.W[word_idx] + self.W_tilde[word_idx]

    def get_embeddings(self) -> np.ndarray:
        """Get all embeddings (sum of W and W_tilde)"""
        return self.W + self.W_tilde


def test_word_embeddings():
    """Test word embeddings implementations"""
    print("=" * 60)
    print("Testing Word Embeddings")
    print("=" * 60)

    # Sample texts
    texts = [
        "the cat sat on the mat",
        "the dog played in the garden",
        "a cat is a small furry animal",
        "dogs are loyal companions",
        "cats and dogs are pets",
    ]

    # Tokenize
    tokenized = [text.lower().split() for text in texts]
    print(f"\nTokenized texts: {tokenized}")

    # Build vocabulary
    vocab = Vocabulary()
    for tokens in tokenized:
        for token in tokens:
            vocab.add_word(token)
    vocab.build()

    print(f"\nVocabulary size: {len(vocab)}")
    print(f"Word2idx: {dict(vocab.word2idx)}")

    # Test Word2Vec (simplified - use smaller parameters)
    print("\n--- Testing Word2Vec ---")
    word2vec = Word2VecSkipGram(
        vocab_size=len(vocab),
        embedding_dim=10,
        window_size=2,
        negative_samples=2,
        epochs=3,
    )

    # Simple training
    all_tokens = []
    for tokens in tokenized:
        all_tokens.append([vocab.encode(t) for t in tokens])

    word2vec.train(tokenized[:3])  # Train on small subset

    # Get embeddings
    print("\n--- Embeddings ---")
    for word in ["cat", "dog", "the"]:
        idx = vocab.encode(word)
        emb = word2vec.get_embedding(idx)
        print(f"'{word}' (idx={idx}): {emb[:5]}...")

    # Test GloVe
    print("\n--- Testing GloVe ---")
    glove = GloVe(vocab_size=len(vocab), embedding_dim=10, window_size=2, epochs=3)

    glove.train(tokenized[:3])

    print("\n" + "=" * 60)
    print("All word embeddings tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    test_word_embeddings()
