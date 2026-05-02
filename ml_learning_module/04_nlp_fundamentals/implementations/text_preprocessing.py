"""
Text Preprocessing Implementation
==================================

This module provides comprehensive text preprocessing utilities
for NLP tasks.

Author: AI-Mastery-2026
"""

import re
import string
from typing import List, Tuple, Dict, Set, Optional
from collections import Counter


class TextPreprocessor:
    """
    Comprehensive text preprocessing for NLP tasks.

    Operations include:
    - Lowercasing
    - Removing special characters
    - Removing stopwords
    - Stemming and Lemmatization
    - Handling contractions
    - Removing numbers
    - Normalizing whitespace

    Usage:
        >>> preprocessor = TextPreprocessor()
        >>> text = "Hello, World! This is a test."
        >>> cleaned = preprocessor.preprocess(text)
        >>> print(cleaned)
    """

    def __init__(
        self,
        remove_stopwords: bool = True,
        stem: bool = False,
        remove_numbers: bool = True,
        min_word_length: int = 2,
    ):
        """
        Initialize preprocessor

        Args:
            remove_stopwords: Whether to remove common stopwords
            stem: Whether to apply stemming
            remove_numbers: Whether to remove numeric characters
            min_word_length: Minimum word length to keep
        """
        self.remove_stopwords = remove_stopwords
        self.stem = stem
        self.remove_numbers = remove_numbers
        self.min_word_length = min_word_length

        # Common English stopwords
        self.stopwords = self._load_stopwords()

        # Simple stemmer
        self.suffixes = ["ing", "ed", "ness", "ly", "ment", "tion", "sion", "ity", "y"]

    def _load_stopwords(self) -> Set[str]:
        """Load common English stopwords"""
        return {
            "i",
            "me",
            "my",
            "myself",
            "we",
            "our",
            "ours",
            "ourselves",
            "you",
            "you're",
            "you've",
            "you'll",
            "you'd",
            "your",
            "yours",
            "yourself",
            "yourselves",
            "he",
            "him",
            "his",
            "himself",
            "she",
            "she's",
            "her",
            "hers",
            "herself",
            "it",
            "it's",
            "its",
            "itself",
            "they",
            "them",
            "their",
            "theirs",
            "themselves",
            "what",
            "which",
            "who",
            "whom",
            "this",
            "that",
            "that'll",
            "these",
            "those",
            "am",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "having",
            "do",
            "does",
            "did",
            "doing",
            "a",
            "an",
            "the",
            "and",
            "but",
            "if",
            "or",
            "because",
            "as",
            "until",
            "while",
            "of",
            "at",
            "by",
            "for",
            "with",
            "about",
            "against",
            "between",
            "into",
            "through",
            "during",
            "before",
            "after",
            "above",
            "below",
            "to",
            "from",
            "up",
            "down",
            "in",
            "out",
            "on",
            "off",
            "over",
            "under",
            "again",
            "further",
            "then",
            "once",
            "here",
            "there",
            "when",
            "where",
            "why",
            "how",
            "all",
            "any",
            "both",
            "each",
            "few",
            "more",
            "most",
            "other",
            "some",
            "such",
            "no",
            "nor",
            "not",
            "only",
            "own",
            "same",
            "so",
            "than",
            "too",
            "very",
            "s",
            "t",
            "can",
            "will",
            "just",
            "don",
            "don't",
            "should",
            "should've",
            "now",
            "d",
            "ll",
            "m",
            "o",
            "re",
            "ve",
            "y",
            "ain",
            "aren",
            "aren't",
            "couldn",
            "couldn't",
            "didn",
            "didn't",
            "doesn",
            "doesn't",
            "hadn",
            "hadn't",
            "hasn",
            "hasn't",
            "haven",
            "haven't",
            "isn",
            "isn't",
            "ma",
            "mightn",
            "mightn't",
            "mustn",
            "mustn't",
            "needn",
            "needn't",
            "shan",
            "shan't",
            "shouldn",
            "shouldn't",
            "wasn",
            "wasn't",
            "weren",
            "weren't",
            "won",
            "won't",
            "wouldn",
            "wouldn't",
        }

    def to_lowercase(self, text: str) -> str:
        """Convert text to lowercase"""
        return text.lower()

    def remove_punctuation(self, text: str) -> str:
        """Remove punctuation marks"""
        return text.translate(str.maketrans("", "", string.punctuation))

    def expand_contractions(self, text: str) -> str:
        """Expand common English contractions"""
        contractions_map = {
            "won't": "will not",
            "can't": "cannot",
            "n't": " not",
            "'re": " are",
            "'s": " is",
            "'d": " would",
            "'ll": " will",
            "'ve": " have",
            "'m": " am",
        }

        for contraction, expansion in contractions_map.items():
            text = text.lower().replace(contraction, expansion)

        return text

    def remove_numbers(self, text: str) -> str:
        """Remove numeric characters"""
        return re.sub(r"\d+", "", text)

    def remove_whitespace(self, text: str) -> str:
        """Normalize whitespace"""
        return " ".join(text.split())

    def remove_special_characters(self, text: str) -> str:
        """Remove special characters, keeping only alphanumeric and spaces"""
        return re.sub(r"[^a-zA-Z\s]", "", text)

    def simple_stem(self, word: str) -> str:
        """
        Simple suffix-stripping stemmer

        Removes common suffixes to get root form.
        This is a simplified version - for production,
        consider using NLTK's PorterStemmer or SnowballStemmer.
        """
        word = word.lower()

        # Don't stem short words
        if len(word) <= 3:
            return word

        # Try each suffix
        for suffix in self.suffixes:
            if word.endswith(suffix) and len(word) - len(suffix) >= 3:
                return word[: -len(suffix)]

        return word

    def remove_stopwords_from_tokens(self, tokens: List[str]) -> List[str]:
        """Remove stopwords from token list"""
        return [token for token in tokens if token not in self.stopwords]

    def tokenize(self, text: str) -> List[str]:
        """
        Split text into tokens (words)

        Args:
            text: Input text

        Returns:
            List of tokens
        """
        # Convert to lowercase
        text = self.to_lowercase(text)

        # Expand contractions
        text = self.expand_contractions(text)

        # Remove special characters
        text = self.remove_special_characters(text)

        # Remove numbers if enabled
        if self.remove_numbers:
            text = self.remove_numbers(text)

        # Split on whitespace
        tokens = text.split()

        # Remove very short words
        if self.min_word_length > 1:
            tokens = [t for t in tokens if len(t) >= self.min_word_length]

        # Remove stopwords
        if self.remove_stopwords:
            tokens = self.remove_stopwords_from_tokens(tokens)

        # Apply stemming
        if self.stem:
            tokens = [self.simple_stem(t) for t in tokens]

        # Clean up whitespace
        tokens = [t.strip() for t in tokens if t.strip()]

        return tokens

    def preprocess(self, text: str) -> str:
        """
        Complete preprocessing pipeline

        Args:
            text: Raw input text

        Returns:
            Preprocessed text
        """
        tokens = self.tokenize(text)
        return " ".join(tokens)

    def preprocess_batch(self, texts: List[str]) -> List[str]:
        """
        Preprocess multiple texts

        Args:
            texts: List of raw texts

        Returns:
            List of preprocessed texts
        """
        return [self.preprocess(text) for text in texts]


class BagOfWords:
    """
    Bag of Words (BoW) Feature Extraction

    Converts text into numerical feature vectors based on word counts.

    Mathematical Formulation:
        Given vocabulary V = {w₁, w₂, ..., wₙ}
        Document d is represented as vector:
            d = [count(w₁,d), count(w₂,d), ..., count(wn,d)]

    Extensions:
        - TF (Term Frequency): Raw counts
        - TF-IDF: Term Frequency × Inverse Document Frequency
    """

    def __init__(
        self, max_features: Optional[int] = None, min_df: int = 1, max_df: float = 1.0
    ):
        """
        Initialize BoW vectorizer

        Args:
            max_features: Maximum vocabulary size
            min_df: Minimum document frequency
            max_df: Maximum document frequency (as fraction)
        """
        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df
        self.vocabulary_ = {}
        self.idf_ = {}

    def fit(self, documents: List[str]) -> "BagOfWords":
        """
        Learn vocabulary from documents

        Args:
            documents: List of text documents

        Returns:
            Self
        """
        # Count document frequencies
        doc_freq = Counter()
        for doc in documents:
            words = set(doc.split())
            for word in words:
                doc_freq[word] += 1

        n_docs = len(documents)

        # Filter by min_df and max_df
        vocab = {}
        for word, freq in doc_freq.items():
            if freq >= self.min_df and freq / n_docs <= self.max_df:
                vocab[word] = freq

        # Sort by frequency and limit to max_features
        if self.max_features:
            vocab = dict(
                sorted(vocab.items(), key=lambda x: x[1], reverse=True)[
                    : self.max_features
                ]
            )

        # Create vocabulary with indices
        self.vocabulary_ = {word: idx for idx, word in enumerate(vocab.keys())}

        # Compute IDF
        for word, freq in vocab.items():
            self.idf_[word] = np.log(n_docs / freq) + 1

        return self

    def transform(self, documents: List[str]) -> np.ndarray:
        """
        Transform documents to feature vectors

        Args:
            documents: List of text documents

        Returns:
            Feature matrix of shape (n_docs, n_features)
        """
        n_docs = len(documents)
        n_features = len(self.vocabulary_)

        # Initialize matrix with zeros
        features = np.zeros((n_docs, n_features))

        for i, doc in enumerate(documents):
            words = doc.split()
            for word in words:
                if word in self.vocabulary_:
                    features[i, self.vocabulary_[word]] += 1

        return features

    def fit_transform(self, documents: List[str]) -> np.ndarray:
        """Fit and transform in one step"""
        return self.fit(documents).transform(documents)

    def get_feature_names(self) -> List[str]:
        """Get list of feature names (vocabulary words)"""
        return list(self.vocabulary_.keys())


class TFIDFVectorizer:
    """
    TF-IDF (Term Frequency - Inverse Document Frequency) Vectorizer

    Weights terms by their importance:

    TF-IDF(w, d, D) = TF(w, d) × IDF(w, D)

    Where:
        TF(w, d) = count of w in document d
        IDF(w, D) = log(N / df(w)) + 1
            N = total documents
            df(w) = documents containing w
    """

    def __init__(self, max_features: Optional[int] = None):
        """Initialize TF-IDF vectorizer"""
        self.max_features = max_features
        self.vocabulary_ = {}
        self.idf_ = {}

    def fit(self, documents: List[str]) -> "TFIDFVectorizer":
        """Learn vocabulary and IDF from documents"""
        n_docs = len(documents)

        # Count document frequencies
        doc_freq = Counter()
        for doc in documents:
            words = set(doc.split())
            for word in words:
                doc_freq[word] += 1

        # Create vocabulary
        vocab = dict(sorted(doc_freq.items(), key=lambda x: x[1], reverse=True))

        if self.max_features:
            vocab = dict(list(vocab.items())[: self.max_features])

        self.vocabulary_ = {word: idx for idx, word in enumerate(vocab.keys())}

        # Compute IDF
        for word, df in vocab.items():
            self.idf_[word] = np.log(n_docs / df) + 1

        return self

    def transform(self, documents: List[str]) -> np.ndarray:
        """Transform documents to TF-IDF vectors"""
        n_docs = len(documents)
        n_features = len(self.vocabulary_)

        # Get BoW features
        bow = BagOfWords()
        bow.vocabulary_ = self.vocabulary_
        bow_features = bow.transform(documents)

        # Apply IDF weighting
        idf_array = np.array(
            [self.idf_.get(word, 0) for word in bow.get_feature_names()]
        )

        tfidf_features = bow_features * idf_array

        return tfidf_features

    def fit_transform(self, documents: List[str]) -> np.ndarray:
        """Fit and transform"""
        return self.fit(documents).transform(documents)


def demo_text_preprocessing():
    """Demonstrate text preprocessing"""
    print("=" * 60)
    print("Text Preprocessing Demo")
    print("=" * 60)

    # Sample texts
    texts = [
        "The quick brown fox jumps over the lazy dog!",
        "Machine learning is amazing and very interesting.",
        "I can't believe this is happening to me.",
        "Natural Language Processing (NLP) is fun!",
        "The meeting has been postponed to 2024.",
    ]

    print("\nOriginal texts:")
    for i, text in enumerate(texts):
        print(f"  {i + 1}: {text}")

    # Create preprocessor
    preprocessor = TextPreprocessor(
        remove_stopwords=True, remove_numbers=True, min_word_length=2
    )

    # Preprocess
    print("\nPreprocessed texts:")
    for i, text in enumerate(texts):
        cleaned = preprocessor.preprocess(text)
        tokens = preprocessor.tokenize(text)
        print(f"  {i + 1}: {cleaned}")
        print(f"      Tokens: {tokens}")

    # Create BoW features
    print("\n--- Bag of Words ---")
    preprocessed = [preprocessor.preprocess(t) for t in texts]

    bow = BagOfWords(max_features=20)
    bow_features = bow.fit_transform(preprocessed)

    print(f"Feature matrix shape: {bow_features.shape}")
    print(f"Vocabulary: {bow.get_feature_names()}")

    # Create TF-IDF features
    print("\n--- TF-IDF ---")
    tfidf = TFIDFVectorizer(max_features=20)
    tfidf_features = tfidf.fit_transform(preprocessed)

    print(f"TF-IDF matrix shape: {tfidf_features.shape}")

    # Show example
    print("\nExample TF-IDF vector for first document:")
    print(tfidf_features[0])


def test_preprocessing():
    """Run tests"""
    print("\n" + "=" * 60)
    print("Testing Preprocessing Functions")
    print("=" * 60)

    preprocessor = TextPreprocessor()

    # Test lowercase
    result = preprocessor.to_lowercase("HELLO World")
    assert result == "hello world", f"Failed: {result}"
    print("✓ Lowercase test passed")

    # Test remove punctuation
    result = preprocessor.remove_punctuation("Hello, World!")
    assert result == "Hello World", f"Failed: {result}"
    print("✓ Punctuation test passed")

    # Test tokenize
    tokens = preprocessor.tokenize("The quick brown fox")
    assert tokens == ["quick", "brown", "fox"], f"Failed: {tokens}"
    print("✓ Tokenize test passed")

    print("\nAll tests passed!")


if __name__ == "__main__":
    demo_text_preprocessing()
    test_preprocessing()

    print("\n" + "=" * 60)
    print("Text preprocessing complete!")
    print("=" * 60)
