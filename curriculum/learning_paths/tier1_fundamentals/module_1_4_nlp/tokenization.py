"""
NLP Tokenization Module.

This module provides comprehensive text tokenization implementations,
including word tokenization, subword tokenization (BPE), and sentence piece.

Features:
- Word-level tokenization
- Character-level tokenization
- Byte Pair Encoding (BPE)
- WordPiece tokenization
- SentencePiece-style tokenization
- Special tokens handling

Example Usage:
    >>> from tokenization import WordTokenizer, BPETokenizer
    >>> 
    >>> # Word tokenization
    >>> tokenizer = WordTokenizer()
    >>> tokens = tokenizer.tokenize("Hello, world!")
    >>> 
    >>> # BPE tokenization
    >>> bpe = BPETokenizer(vocab_size=1000)
    >>> bpe.train(["hello world", "hello there", "world peace"])
    >>> tokens = bpe.encode("hello world")
"""

from typing import Union, List, Dict, Tuple, Optional, Set, Iterator
import numpy as np
import re
import logging
from collections import Counter, defaultdict
from dataclasses import dataclass
import json
import os

logger = logging.getLogger(__name__)


@dataclass
class Token:
    """Represents a token with metadata."""
    text: str
    id: int
    start: Optional[int] = None
    end: Optional[int] = None
    is_special: bool = False


class Tokenizer:
    """Base class for tokenizers."""
    
    def __init__(
        self,
        vocab_size: int = 30000,
        special_tokens: Optional[List[str]] = None
    ):
        """
        Initialize tokenizer.
        
        Args:
            vocab_size: Vocabulary size.
            special_tokens: List of special tokens.
        """
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens or ['<pad>', '<unk>', '<bos>', '<eos>']
        
        self.vocab: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        self.unk_token = '<unk>'
        self.pad_token = '<pad>'
        self.bos_token = '<bos>'
        self.eos_token = '<eos>'
        
        self._setup_special_tokens()
    
    def _setup_special_tokens(self) -> None:
        """Setup special tokens in vocabulary."""
        for i, token in enumerate(self.special_tokens):
            self.vocab[token] = i
            self.id_to_token[i] = token
        
        # Set special token IDs
        self.unk_token_id = self.vocab.get('<unk>', 1)
        self.pad_token_id = self.vocab.get('<pad>', 0)
        self.bos_token_id = self.vocab.get('<bos>', 2)
        self.eos_token_id = self.vocab.get('<eos>', 3)
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into tokens.
        
        Args:
            text: Input text.
        
        Returns:
            List[str]: List of tokens.
        """
        raise NotImplementedError
    
    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
        truncation: bool = True,
        max_length: Optional[int] = None
    ) -> List[int]:
        """
        Encode text to token IDs.
        
        Args:
            text: Input text.
            add_special_tokens: Add BOS/EOS tokens.
            truncation: Truncate to max_length.
            max_length: Maximum sequence length.
        
        Returns:
            List[int]: Token IDs.
        """
        tokens = self.tokenize(text)
        token_ids = self.convert_tokens_to_ids(tokens)
        
        if add_special_tokens:
            token_ids = [self.bos_token_id] + token_ids + [self.eos_token_id]
        
        if truncation and max_length is not None:
            if len(token_ids) > max_length:
                token_ids = token_ids[:max_length]
                if add_special_tokens:
                    token_ids[-1] = self.eos_token_id
        
        return token_ids
    
    def decode(
        self,
        token_ids: List[int],
        skip_special_tokens: bool = True
    ) -> str:
        """
        Decode token IDs to text.
        
        Args:
            token_ids: Token IDs.
            skip_special_tokens: Skip special tokens in output.
        
        Returns:
            str: Decoded text.
        """
        tokens = self.convert_ids_to_tokens(token_ids)
        
        if skip_special_tokens:
            tokens = [t for t in tokens if t not in self.special_tokens]
        
        return self.convert_tokens_to_string(tokens)
    
    def convert_tokens_to_ids(self, tokens: List[str]) -> List[int]:
        """Convert tokens to IDs."""
        return [self.vocab.get(token, self.unk_token_id) for token in tokens]
    
    def convert_ids_to_tokens(self, ids: List[int]) -> List[str]:
        """Convert IDs to tokens."""
        return [self.id_to_token.get(id_, self.unk_token) for id_ in ids]
    
    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        """Convert tokens to string."""
        return ' '.join(tokens)
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size."""
        return len(self.vocab)
    
    def save_vocab(self, filepath: str) -> None:
        """Save vocabulary to file."""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump({
                'vocab': self.vocab,
                'special_tokens': self.special_tokens
            }, f, ensure_ascii=False, indent=2)
        logger.info(f"Vocabulary saved to {filepath}")
    
    def load_vocab(self, filepath: str) -> None:
        """Load vocabulary from file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.vocab = data['vocab']
        self.special_tokens = data.get('special_tokens', self.special_tokens)
        self.id_to_token = {v: k for k, v in self.vocab.items()}
        self._setup_special_tokens()
        
        logger.info(f"Vocabulary loaded from {filepath}")


class WordTokenizer(Tokenizer):
    """
    Word-level tokenizer with various splitting strategies.
    
    Example:
        >>> tokenizer = WordTokenizer()
        >>> tokens = tokenizer.tokenize("Hello, world! How are you?")
        >>> tokens
        ['Hello', ',', 'world', '!', 'How', 'are', 'you', '?']
    """
    
    def __init__(
        self,
        lowercase: bool = True,
        split_pattern: str = r'\w+|[^\w\s]',
        **kwargs
    ):
        """
        Initialize WordTokenizer.
        
        Args:
            lowercase: Convert to lowercase.
            split_pattern: Regex pattern for tokenization.
            **kwargs: Additional arguments for Tokenizer.
        """
        super().__init__(**kwargs)
        self.lowercase = lowercase
        self.split_pattern = split_pattern
        self._compiled_pattern = re.compile(split_pattern)
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into words.
        
        Args:
            text: Input text.
        
        Returns:
            List[str]: List of word tokens.
        """
        if self.lowercase:
            text = text.lower()
        
        tokens = self._compiled_pattern.findall(text)
        return [t for t in tokens if t.strip()]
    
    def build_vocab(
        self,
        texts: List[str],
        min_freq: int = 1,
        max_vocab_size: Optional[int] = None
    ) -> None:
        """
        Build vocabulary from texts.
        
        Args:
            texts: List of texts.
            min_freq: Minimum token frequency.
            max_vocab_size: Maximum vocabulary size.
        """
        counter = Counter()
        
        for text in texts:
            tokens = self.tokenize(text)
            counter.update(tokens)
        
        # Filter by frequency
        filtered = [(token, freq) for token, freq in counter.items() 
                   if freq >= min_freq]
        filtered.sort(key=lambda x: -x[1])
        
        # Build vocabulary
        vocab_size = max_vocab_size or self.vocab_size
        available_slots = vocab_size - len(self.special_tokens)
        
        for token, _ in filtered[:available_slots]:
            if token not in self.vocab:
                idx = len(self.vocab)
                self.vocab[token] = idx
                self.id_to_token[idx] = token
        
        logger.info(f"Vocabulary built: {len(self.vocab)} tokens")


class CharTokenizer(Tokenizer):
    """
    Character-level tokenizer.
    
    Example:
        >>> tokenizer = CharTokenizer()
        >>> tokens = tokenizer.tokenize("Hello")
        >>> tokens
        ['H', 'e', 'l', 'l', 'o']
    """
    
    def __init__(
        self,
        lowercase: bool = True,
        **kwargs
    ):
        """
        Initialize CharTokenizer.
        
        Args:
            lowercase: Convert to lowercase.
            **kwargs: Additional arguments for Tokenizer.
        """
        super().__init__(**kwargs)
        self.lowercase = lowercase
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into characters.
        
        Args:
            text: Input text.
        
        Returns:
            List[str]: List of character tokens.
        """
        if self.lowercase:
            text = text.lower()
        return list(text)
    
    def build_vocab(self, texts: List[str]) -> None:
        """
        Build vocabulary from texts.
        
        Args:
            texts: List of texts.
        """
        chars = set()
        for text in texts:
            chars.update(self.tokenize(text))
        
        for char in sorted(chars):
            if char not in self.vocab:
                idx = len(self.vocab)
                self.vocab[char] = idx
                self.id_to_token[idx] = char
        
        logger.info(f"Character vocabulary built: {len(self.vocab)} characters")


class BPETokenizer(Tokenizer):
    """
    Byte Pair Encoding (BPE) tokenizer.
    
    BPE iteratively merges the most frequent token pairs.
    
    Example:
        >>> bpe = BPETokenizer(vocab_size=1000)
        >>> bpe.train(["hello world", "hello there", "world peace"])
        >>> tokens = bpe.encode("hello world")
    """
    
    def __init__(
        self,
        vocab_size: int = 30000,
        min_frequency: int = 2,
        **kwargs
    ):
        """
        Initialize BPETokenizer.
        
        Args:
            vocab_size: Target vocabulary size.
            min_frequency: Minimum frequency for merges.
            **kwargs: Additional arguments for Tokenizer.
        """
        super().__init__(vocab_size=vocab_size, **kwargs)
        self.min_frequency = min_frequency
        self.merges: List[Tuple[str, str]] = []
        self.merge_ranks: Dict[Tuple[str, str], int] = {}
    
    def _get_word_freqs(
        self,
        texts: List[str]
    ) -> Dict[Tuple[str, ...], int]:
        """Get word frequencies as character tuples."""
        word_freqs = Counter()
        
        for text in texts:
            words = text.lower().split()
            word_freqs.update(words)
        
        # Convert to character tuples
        char_word_freqs = {}
        for word, freq in word_freqs.items():
            char_word_freqs[tuple(word) + ('</w>',)] = freq
        
        return char_word_freqs
    
    def _get_pair_freqs(
        self,
        word_freqs: Dict[Tuple[str, ...], int]
    ) -> Counter:
        """Get frequencies of adjacent token pairs."""
        pair_freqs = Counter()
        
        for word, freq in word_freqs.items():
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                pair_freqs[pair] += freq
        
        return pair_freqs
    
    def _merge_pair(
        self,
        word_freqs: Dict[Tuple[str, ...], int],
        pair: Tuple[str, str]
    ) -> Dict[Tuple[str, ...], int]:
        """Merge all occurrences of a pair."""
        new_word_freqs = {}
        bigram = pair[0] + pair[1]
        
        for word, freq in word_freqs.items():
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == pair[0] and word[i + 1] == pair[1]:
                    new_word.append(bigram)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            
            new_word_freqs[tuple(new_word)] = freq
        
        return new_word_freqs
    
    def train(
        self,
        texts: List[str],
        show_progress: bool = True
    ) -> None:
        """
        Train BPE on texts.
        
        Args:
            texts: Training texts.
            show_progress: Show training progress.
        """
        # Get initial word frequencies
        word_freqs = self._get_word_freqs(texts)
        
        # Get initial vocabulary (all characters)
        char_vocab = set()
        for word in word_freqs.keys():
            char_vocab.update(word)
        
        # Add special tokens and characters to vocabulary
        for char in sorted(char_vocab):
            if char not in self.vocab:
                idx = len(self.vocab)
                self.vocab[char] = idx
                self.id_to_token[idx] = char
        
        # Iteratively merge most frequent pairs
        num_merges = self.vocab_size - len(self.vocab)
        
        for i in range(num_merges):
            pair_freqs = self._get_pair_freqs(word_freqs)
            
            if not pair_freqs:
                break
            
            # Get most frequent pair
            best_pair = pair_freqs.most_common(1)[0]
            
            if best_pair[1] < self.min_frequency:
                break
            
            pair = best_pair[0]
            
            # Record merge
            self.merges.append(pair)
            self.merge_ranks[pair] = i
            
            # Add merged token to vocabulary
            merged_token = pair[0] + pair[1]
            if merged_token not in self.vocab:
                idx = len(self.vocab)
                self.vocab[merged_token] = idx
                self.id_to_token[idx] = merged_token
            
            # Apply merge
            word_freqs = self._merge_pair(word_freqs, pair)
            
            if show_progress and (i + 1) % 100 == 0:
                logger.info(f"BPE training: {i + 1}/{num_merges} merges, "
                           f"vocab size: {len(self.vocab)}")
        
        logger.info(f"BPE training completed: {len(self.vocab)} tokens, "
                   f"{len(self.merges)} merges")
    
    def _tokenize_word(self, word: str) -> List[str]:
        """Tokenize a single word using learned merges."""
        # Start with characters
        tokens = list(word.lower()) + ['</w>']
        
        # Apply merges in order
        for merge in self.merges:
            i = 0
            while i < len(tokens) - 1:
                if tokens[i] == merge[0] and tokens[i + 1] == merge[1]:
                    tokens[i] = merge[0] + merge[1]
                    tokens.pop(i + 1)
                i += 1
        
        return tokens
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text using BPE.
        
        Args:
            text: Input text.
        
        Returns:
            List[str]: BPE tokens.
        """
        tokens = []
        for word in text.split():
            word_tokens = self._tokenize_word(word)
            tokens.extend(word_tokens)
        return tokens
    
    def save(self, filepath: str) -> None:
        """Save BPE tokenizer."""
        self.save_vocab(filepath + '.vocab')
        
        with open(filepath + '.merges', 'w', encoding='utf-8') as f:
            for pair in self.merges:
                f.write(f"{pair[0]} {pair[1]}\n")
        
        logger.info(f"BPE tokenizer saved to {filepath}")
    
    def load(self, filepath: str) -> None:
        """Load BPE tokenizer."""
        self.load_vocab(filepath + '.vocab')
        
        self.merges = []
        with open(filepath + '.merges', 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split(' ', 1)
                if len(parts) == 2:
                    self.merges.append((parts[0], parts[1]))
        
        logger.info(f"BPE tokenizer loaded from {filepath}")


class WordPieceTokenizer(Tokenizer):
    """
    WordPiece tokenizer (used in BERT).
    
    Similar to BPE but uses a different scoring function for merges.
    Uses ## prefix for subword tokens.
    
    Example:
        >>> tokenizer = WordPieceTokenizer(vocab_size=1000)
        >>> tokenizer.train(["hello world", "hello there"])
        >>> tokens = tokenizer.encode("hello world")
    """
    
    def __init__(
        self,
        vocab_size: int = 30000,
        max_input_chars_per_word: int = 100,
        **kwargs
    ):
        """
        Initialize WordPieceTokenizer.
        
        Args:
            vocab_size: Target vocabulary size.
            max_input_chars_per_word: Maximum characters per word.
            **kwargs: Additional arguments for Tokenizer.
        """
        super().__init__(vocab_size=vocab_size, **kwargs)
        self.max_input_chars = max_input_chars_per_word
        self.subword_prefix = '##'
    
    def train(self, texts: List[str], show_progress: bool = True) -> None:
        """
        Train WordPiece on texts.
        
        Args:
            texts: Training texts.
            show_progress: Show training progress.
        """
        # Count word frequencies
        word_freqs = Counter()
        for text in texts:
            for word in text.lower().split():
                if len(word) <= self.max_input_chars:
                    word_freqs[word] += 1
        
        # Initialize vocabulary with characters
        char_vocab = set()
        for word in word_freqs.keys():
            char_vocab.update(word)
        
        for char in sorted(char_vocab):
            if char not in self.vocab:
                idx = len(self.vocab)
                self.vocab[char] = idx
                self.id_to_token[idx] = char
        
        # Iteratively add best subwords
        while len(self.vocab) < self.vocab_size:
            # Count all possible subwords
            subword_freqs = Counter()
            
            for word, freq in word_freqs.items():
                # All possible subwords
                for i in range(len(word)):
                    for j in range(i + 1, min(len(word) + 1, i + 10)):
                        subword = word[i:j]
                        if i > 0:
                            subword = self.subword_prefix + subword
                        subword_freqs[subword] += freq
            
            if not subword_freqs:
                break
            
            # Add most frequent subword
            best_subword = subword_freqs.most_common(1)[0][0]
            
            if best_subword not in self.vocab:
                idx = len(self.vocab)
                self.vocab[best_subword] = idx
                self.id_to_token[idx] = best_subword
            
            if show_progress and len(self.vocab) % 100 == 0:
                logger.info(f"WordPiece training: vocab size {len(self.vocab)}")
        
        logger.info(f"WordPiece training completed: {len(self.vocab)} tokens")
    
    def _tokenize_word(self, word: str) -> List[str]:
        """Tokenize word using greedy longest-match-first."""
        if len(word) > self.max_input_chars:
            return [self.unk_token]
        
        tokens = []
        start = 0
        
        while start < len(word):
            end = len(word)
            cur_token = None
            
            while start < end:
                substr = word[start:end]
                if start > 0:
                    substr = self.subword_prefix + substr
                
                if substr in self.vocab:
                    cur_token = substr
                    break
                
                end -= 1
            
            if cur_token is None:
                tokens.append(self.unk_token)
                break
            
            tokens.append(cur_token)
            start = end
        
        return tokens
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text using WordPiece."""
        tokens = []
        for word in text.lower().split():
            word_tokens = self._tokenize_word(word)
            tokens.extend(word_tokens)
        return tokens


class SentencePieceTokenizer(Tokenizer):
    """
    SentencePiece-style tokenizer.
    
    Treats input as raw sequence of characters including spaces.
    Uses Unicode characters and special handling for whitespace.
    
    Example:
        >>> tokenizer = SentencePieceTokenizer(vocab_size=1000)
        >>> tokenizer.train(["hello world", "hello there"])
        >>> tokens = tokenizer.encode("hello world")
    """
    
    def __init__(
        self,
        vocab_size: int = 30000,
        model_type: str = 'unigram',
        **kwargs
    ):
        """
        Initialize SentencePieceTokenizer.
        
        Args:
            vocab_size: Target vocabulary size.
            model_type: Model type ('unigram' or 'bpe').
            **kwargs: Additional arguments for Tokenizer.
        """
        super().__init__(vocab_size=vocab_size, **kwargs)
        self.model_type = model_type
        self.whitespace_symbol = '▁'  # U+2581
    
    def _preprocess(self, text: str) -> str:
        """Preprocess text for SentencePiece."""
        # Replace spaces with special symbol
        text = text.replace(' ', self.whitespace_symbol)
        return text
    
    def train(self, texts: List[str], show_progress: bool = True) -> None:
        """
        Train SentencePiece on texts.
        
        Args:
            texts: Training texts.
            show_progress: Show training progress.
        """
        # Preprocess and count character frequencies
        char_freqs = Counter()
        
        for text in texts:
            processed = self._preprocess(text.lower())
            char_freqs.update(processed)
        
        # Add whitespace symbol to special tokens if not present
        if self.whitespace_symbol not in self.special_tokens:
            self.special_tokens.append(self.whitespace_symbol)
            self._setup_special_tokens()
        
        # Initialize with characters
        for char, _ in char_freqs.most_common():
            if char not in self.vocab:
                idx = len(self.vocab)
                self.vocab[char] = idx
                self.id_to_token[idx] = char
        
        # For simplicity, use BPE-style training
        if self.model_type == 'bpe':
            bpe_tokenizer = BPETokenizer(
                vocab_size=self.vocab_size,
                special_tokens=self.special_tokens
            )
            
            # Preprocess texts
            processed_texts = [self._preprocess(t.lower()) for t in texts]
            bpe_tokenizer.train(processed_texts, show_progress=False)
            
            # Copy vocabulary
            self.vocab = bpe_tokenizer.vocab.copy()
            self.id_to_token = bpe_tokenizer.id_to_token.copy()
            self.merges = bpe_tokenizer.merges
        
        logger.info(f"SentencePiece training completed: {len(self.vocab)} tokens")
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text using SentencePiece."""
        processed = self._preprocess(text.lower())
        
        if hasattr(self, 'merges'):
            # Use BPE-style tokenization
            tokens = []
            for word in processed.split(self.whitespace_symbol):
                if word:
                    word_tokens = list(word)
                    # Apply merges
                    for merge in self.merges:
                        i = 0
                        while i < len(word_tokens) - 1:
                            if word_tokens[i] == merge[0] and word_tokens[i + 1] == merge[1]:
                                word_tokens[i] = merge[0] + merge[1]
                                word_tokens.pop(i + 1)
                            i += 1
                    tokens.extend(word_tokens)
            return tokens
        else:
            return list(processed)
    
    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        """Convert tokens back to string."""
        result = ''.join(tokens)
        result = result.replace(self.whitespace_symbol, ' ')
        return result


def get_tokenizer(
    tokenizer_type: str = 'word',
    **kwargs
) -> Tokenizer:
    """
    Factory function to get tokenizer by type.
    
    Args:
        tokenizer_type: Type of tokenizer ('word', 'char', 'bpe', 'wordpiece', 'sentencepiece').
        **kwargs: Additional arguments for tokenizer.
    
    Returns:
        Tokenizer: Tokenizer instance.
    
    Raises:
        ValueError: If tokenizer type is not recognized.
    
    Example:
        >>> tokenizer = get_tokenizer('word', lowercase=True)
        >>> tokenizer = get_tokenizer('bpe', vocab_size=10000)
    """
    tokenizers = {
        'word': WordTokenizer,
        'char': CharTokenizer,
        'bpe': BPETokenizer,
        'wordpiece': WordPieceTokenizer,
        'sentencepiece': SentencePieceTokenizer,
    }
    
    tokenizer_type_lower = tokenizer_type.lower()
    if tokenizer_type_lower not in tokenizers:
        raise ValueError(f"Unknown tokenizer type: {tokenizer_type}. "
                        f"Available: {list(tokenizers.keys())}")
    
    return tokenizers[tokenizer_type_lower](**kwargs)


if __name__ == "__main__":
    # Example usage and demonstrations
    print("=" * 60)
    print("Tokenization Module - Demonstration")
    print("=" * 60)
    
    # Word tokenizer
    print("\n1. Word Tokenizer:")
    word_tokenizer = WordTokenizer(lowercase=True)
    text = "Hello, World! How are you doing today?"
    tokens = word_tokenizer.tokenize(text)
    print(f"   Text: {text}")
    print(f"   Tokens: {tokens}")
    
    # Build vocabulary
    texts = [
        "Hello world",
        "Hello there",
        "How are you",
        "World peace",
    ]
    word_tokenizer.build_vocab(texts)
    print(f"   Vocabulary size: {word_tokenizer.get_vocab_size()}")
    
    # Character tokenizer
    print("\n2. Character Tokenizer:")
    char_tokenizer = CharTokenizer()
    char_tokens = char_tokenizer.tokenize("Hello")
    print(f"   Text: Hello")
    print(f"   Tokens: {char_tokens}")
    
    # BPE tokenizer
    print("\n3. BPE Tokenizer:")
    bpe_tokenizer = BPETokenizer(vocab_size=100, min_frequency=1)
    training_texts = [
        "hello world",
        "hello there",
        "world peace",
        "hello hello world",
    ] * 10
    bpe_tokenizer.train(training_texts, show_progress=False)
    print(f"   Vocabulary size: {bpe_tokenizer.get_vocab_size()}")
    print(f"   Number of merges: {len(bpe_tokenizer.merges)}")
    
    bpe_tokens = bpe_tokenizer.tokenize("hello world")
    print(f"   Tokens for 'hello world': {bpe_tokens}")
    
    # Encode/decode
    encoded = bpe_tokenizer.encode("hello world")
    decoded = bpe_tokenizer.decode(encoded)
    print(f"   Encoded: {encoded}")
    print(f"   Decoded: {decoded}")
    
    # WordPiece tokenizer
    print("\n4. WordPiece Tokenizer:")
    wp_tokenizer = WordPieceTokenizer(vocab_size=50)
    wp_tokenizer.train(training_texts, show_progress=False)
    print(f"   Vocabulary size: {wp_tokenizer.get_vocab_size()}")
    
    wp_tokens = wp_tokenizer.tokenize("hello world")
    print(f"   Tokens for 'hello world': {wp_tokens}")
    
    # SentencePiece tokenizer
    print("\n5. SentencePiece Tokenizer:")
    sp_tokenizer = SentencePieceTokenizer(vocab_size=100)
    sp_tokenizer.train(training_texts, show_progress=False)
    print(f"   Vocabulary size: {sp_tokenizer.get_vocab_size()}")
    
    sp_tokens = sp_tokenizer.tokenize("hello world")
    print(f"   Tokens for 'hello world': {sp_tokens}")
    
    print("\n" + "=" * 60)
