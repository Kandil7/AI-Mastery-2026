"""
NLP Fundamentals - Exercises
===========================

Practice problems for text processing and embeddings.

Author: AI-Mastery-2026
"""

import numpy as np


# ============================================================================
# EXERCISE 1: Text Cleaning
# ============================================================================


def exercise_text_cleaning():
    """
    Exercise 1: Clean Text Data

    Clean this text:
    "Visit us at http://www.example.com!!! Don't hesitate to CALL now!"

    Steps:
    1. Remove URLs
    2. Convert to lowercase
    3. Expand contractions
    4. Remove special characters (keep spaces)
    5. Remove extra whitespace
    """
    text = "Visit us at http://www.example.com!!! Don't hesitate to CALL now!"

    # Your code here
    pass


# ============================================================================
# EXERCISE 2: Tokenization
# ============================================================================


def exercise_tokenization():
    """
    Exercise 2: Implement Tokenization

    Tokenize: "NLP is amazing! It's truly incredible."

    Consider:
    - Lowercase
    - Remove punctuation
    - Split into words
    """
    text = "NLP is amazing! It's truly incredible."

    # Your code here
    pass


# ============================================================================
# EXERCISE 3: Build Vocabulary
# ============================================================================


def exercise_vocabulary():
    """
    Exercise 3: Build Word Vocabulary

    Given documents:
    ["the cat", "the dog", "a cat"]

    Build vocabulary with indices:
    {"<PAD>": 0, "<UNK>": 1, ...}

    Handle unknown words with <UNK>
    """
    documents = ["the cat", "the dog", "a cat"]

    # Your code here
    pass


# ============================================================================
# EXERCISE 4: One-Hot Encoding
# ============================================================================


def exercise_one_hot():
    """
    Exercise 4: One-Hot Encoding

    Given vocabulary:
    {"the": 0, "cat": 1, "sat": 2}

    Encode: "the cat sat"

    Expected: [[1,0,0], [0,1,0], [0,0,1]]
    """
    vocab = {"the": 0, "cat": 1, "sat": 2}
    text = "the cat sat"

    # Your code here
    pass


# ============================================================================
# EXERCISE 5: TF-IDF Weights
# ============================================================================


def exercise_tfidf():
    """
    Exercise 5: Calculate TF-IDF

    Documents:
    1: "cat sat"
    2: "dog ran"
    3: "cat and dog"

    Calculate TF-IDF for word "cat" in document 1

    TF = count("cat") / total_words = 1/2 = 0.5
    IDF = log(total_docs / docs_containing_cat) = log(3/2)

    TF-IDF = ?
    """
    # Your code here
    pass


# ============================================================================
# EXERCISE 6: Word Embedding Similarity
# ============================================================================


def exercise_embedding_similarity():
    """
    Exercise 6: Cosine Similarity

    Given vectors:
    v1 = [1, 2, 3]
    v2 = [2, 4, 6]
    v3 = [-1, -2, -3]

    Calculate cosine similarity:
    - v1 vs v2 (should be ~1, same direction)
    - v1 vs v3 (should be ~-1, opposite)
    - v1 vs v1 (should be 1)
    """
    v1 = np.array([1, 2, 3])
    v2 = np.array([2, 4, 6])
    v3 = np.array([-1, -2, -3])

    def cosine_sim(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    print(f"v1 vs v2: {cosine_sim(v1, v2):.4f}")
    print(f"v1 vs v3: {cosine_sim(v1, v3):.4f}")
    print(f"v1 vs v1: {cosine_sim(v1, v1):.4f}")


# ============================================================================
# EXERCISE 7: Attention Weights
# ============================================================================


def exercise_attention_weights():
    """
    Exercise 7: Softmax Attention

    Given attention scores: [3.0, 1.0, 2.0]

    Calculate:
    1. Subtract max (numerical stability)
    2. Apply exp
    3. Normalize

    exp([3,1,2]) = [e³, e¹, e²]
    """
    scores = np.array([3.0, 1.0, 2.0])

    # Your code here (numerically stable softmax)

    # Expected: roughly [0.84, 0.04, 0.11]


# ============================================================================
# SOLUTIONS
# ============================================================================


def solutions():
    """Print solutions"""

    print("=" * 60)
    print("SOLUTIONS")
    print("=" * 60)

    # Solution 1
    print("\n--- Exercise 1: Text Cleaning ---")
    text = "Visit us at http://www.example.com!!! Don't hesitate to CALL now!"
    text = text.lower()
    text = text.replace("don't", "do not")
    text = text.replace("!", ".")
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z\s.]", "", text)
    text = " ".join(text.split())
    print(f"Result: {text}")

    # Solution 6
    print("\n--- Exercise 6: Embedding Similarity ---")
    v1 = np.array([1, 2, 3])
    v2 = np.array([2, 4, 6])
    v3 = np.array([-1, -2, -3])

    def cosine_sim(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    print(f"v1 vs v2: {cosine_sim(v1, v2):.4f} (same direction)")
    print(f"v1 vs v3: {cosine_sim(v1, v3):.4f} (opposite)")
    print(f"v1 vs v1: {cosine_sim(v1, v1):.4f} (identical)")

    # Solution 7
    print("\n--- Exercise 7: Attention Weights ---")
    scores = np.array([3.0, 1.0, 2.0])
    scores_stable = scores - np.max(scores)
    exp_scores = np.exp(scores_stable)
    weights = exp_scores / np.sum(exp_scores)
    print(f"Attention weights: {weights}")


if __name__ == "__main__":
    print("Running NLP exercises...")
    exercise_embedding_similarity()
    # solutions()  # Uncomment to see all solutions
