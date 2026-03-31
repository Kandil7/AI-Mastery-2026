"""
Module 1.4: Natural Language Processing.

This module provides comprehensive NLP implementations from scratch:
- Tokenization (Word, BPE, WordPiece, SentencePiece)
- Embeddings (Word2Vec, GloVe, Positional Encodings)
- Sequence Models (RNN, LSTM, GRU)
- Text Preprocessing (Stemming, Lemmatization, TF-IDF)

Example Usage:
    >>> from module_1_4_nlp import WordTokenizer, BPETokenizer
    >>> from module_1_4_nlp import Word2Vec, GloVe, PositionalEncoding
    >>> from module_1_4_nlp import RNN, LSTM, GRU
    >>> from module_1_4_nlp import Stemmer, Lemmatizer, TFIDFVectorizer
    >>> 
    >>> # Tokenization
    >>> tokenizer = WordTokenizer()
    >>> tokens = tokenizer.tokenize("Hello world!")
    >>> 
    >>> # Word embeddings
    >>> w2v = Word2Vec(embedding_dim=100)
    >>> w2v.train(["hello world", "hello there"])
    >>> vector = w2v.get_vector("hello")
    >>> 
    >>> # Sequence models
    >>> lstm = LSTM(input_size=100, hidden_size=256)
    >>> x = np.random.randn(32, 10, 100)
    >>> output, (h_n, c_n) = lstm.forward(x)
    >>> 
    >>> # TF-IDF
    >>> vectorizer = TFIDFVectorizer()
    >>> tfidf = vectorizer.fit_transform(["hello world", "hello there"])
"""

from .tokenization import (
    Tokenizer,
    Token,
    WordTokenizer,
    CharTokenizer,
    BPETokenizer,
    WordPieceTokenizer,
    SentencePieceTokenizer,
    get_tokenizer,
)

from .embeddings import (
    Embedding,
    Word2Vec,
    GloVe,
    PositionalEncoding,
    TokenEmbeddings,
    create_pretrained_embeddings,
)

from .sequence_models import (
    RNNState,
    LSTMState,
    GRUState,
    RNNCell,
    LSTMCell,
    GRUCell,
    RNN,
    LSTM,
    GRU,
)

from .text_preprocessing import (
    Stemmer,
    Lemmatizer,
    StopWordsRemover,
    NGramGenerator,
    TFIDFVectorizer,
    TextNormalizer,
)

__all__ = [
    # Tokenization
    'Tokenizer',
    'Token',
    'WordTokenizer',
    'CharTokenizer',
    'BPETokenizer',
    'WordPieceTokenizer',
    'SentencePieceTokenizer',
    'get_tokenizer',
    
    # Embeddings
    'Embedding',
    'Word2Vec',
    'GloVe',
    'PositionalEncoding',
    'TokenEmbeddings',
    'create_pretrained_embeddings',
    
    # Sequence Models
    'RNNState',
    'LSTMState',
    'GRUState',
    'RNNCell',
    'LSTMCell',
    'GRUCell',
    'RNN',
    'LSTM',
    'GRU',
    
    # Text Preprocessing
    'Stemmer',
    'Lemmatizer',
    'StopWordsRemover',
    'NGramGenerator',
    'TFIDFVectorizer',
    'TextNormalizer',
]

__version__ = '1.0.0'
__author__ = 'AI Mastery 2026'
