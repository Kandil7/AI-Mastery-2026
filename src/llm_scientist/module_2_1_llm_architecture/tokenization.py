"""
Tokenization - Module 2.1.3

Production-ready tokenization implementations:
- BPE (Byte Pair Encoding) tokenizer
- WordPiece tokenizer
- SentencePiece tokenizer
- Token processor with special tokens

References:
- "Neural Machine Translation of Rare Words with Subword Units" (Sennrich et al., 2015)
- "BERT: Pre-training of Deep Bidirectional Transformers" (Devlin et al., 2018)
- "SentencePiece: A simple and language independent subword tokenizer" (Kudo & Richardson, 2018)
"""

import re
import json
import logging
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor

logger = logging.getLogger(__name__)


@dataclass
class TokenizerOutput:
    """Output from tokenization."""
    input_ids: List[int]
    attention_mask: List[int]
    token_type_ids: Optional[List[int]] = None
    tokens: List[str] = field(default_factory=list)
    special_tokens_mask: List[int] = field(default_factory=list)
    
    def to_tensor(self) -> Dict[str, Tensor]:
        """Convert to PyTorch tensors."""
        result = {
            'input_ids': torch.tensor(self.input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(self.attention_mask, dtype=torch.long),
        }
        if self.token_type_ids is not None:
            result['token_type_ids'] = torch.tensor(self.token_type_ids, dtype=torch.long)
        return result
    
    def to_batch_tensor(self) -> Dict[str, Tensor]:
        """Convert to batched PyTorch tensors (adds batch dimension)."""
        result = {
            'input_ids': torch.tensor([self.input_ids], dtype=torch.long),
            'attention_mask': torch.tensor([self.attention_mask], dtype=torch.long),
        }
        if self.token_type_ids is not None:
            result['token_type_ids'] = torch.tensor([self.token_type_ids], dtype=torch.long)
        return result


class BPETokenizer:
    """
    Byte Pair Encoding (BPE) Tokenizer.
    
    BPE iteratively merges the most frequent pair of bytes/characters
    to build a vocabulary of subword units.
    
    Args:
        vocab_size: Target vocabulary size
        min_frequency: Minimum frequency for merges
        special_tokens: List of special tokens
        unk_token: Unknown token
        bos_token: Beginning of sequence token
        eos_token: End of sequence token
        pad_token: Padding token
        
    Example:
        >>> tokenizer = BPETokenizer(vocab_size=10000)
        >>> tokenizer.train(["hello world", "hello there"])
        >>> tokens = tokenizer.encode("hello world")
        >>> text = tokenizer.decode(tokens)
    """
    
    def __init__(
        self,
        vocab_size: int = 30000,
        min_frequency: int = 2,
        special_tokens: Optional[List[str]] = None,
        unk_token: str = "<unk>",
        bos_token: str = "<s>",
        eos_token: str = "</s>",
        pad_token: str = "<pad>",
    ):
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        
        # Special tokens
        self.unk_token = unk_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.pad_token = pad_token
        
        # Build special tokens list
        self.special_tokens = special_tokens or []
        self.special_tokens.extend([unk_token, bos_token, eos_token, pad_token])
        
        # Vocabulary mappings
        self.vocab: Dict[str, int] = {}
        self.inv_vocab: Dict[int, str] = {}
        self.merges: List[Tuple[str, str]] = []
        
        # Character vocabulary (base)
        self.char_vocab: Dict[str, int] = {}
        
        # Compiled patterns for efficiency
        self._word_pattern = re.compile(r'\w+|[^\w\s]')
    
    def _get_stats(
        self,
        tokenized_text: List[List[str]],
    ) -> Counter:
        """Get pair frequencies from tokenized text."""
        pairs = Counter()
        for words in tokenized_text:
            for word in words:
                symbols = tuple(word)
                for i in range(len(symbols) - 1):
                    pairs[(symbols[i], symbols[i + 1])] += 1
        return pairs
    
    def _merge_pair(
        self,
        pair: Tuple[str, str],
        tokenized_text: List[List[str]],
    ) -> List[List[str]]:
        """Merge a pair of symbols in all words."""
        new_text = []
        bigram = ''.join(pair)
        
        for words in tokenized_text:
            new_words = []
            for word in words:
                # Replace all occurrences of the pair
                new_word = word.replace(' '.join(pair), bigram)
                new_words.append(new_word)
            new_text.append(new_words)
        
        return new_text
    
    def _tokenize_word(self, word: str) -> List[str]:
        """Tokenize a single word into characters."""
        return list(word)
    
    def train(
        self,
        texts: List[str],
        show_progress: bool = True,
    ) -> None:
        """
        Train BPE tokenizer on texts.
        
        Args:
            texts: List of training texts
            show_progress: Whether to show training progress
        """
        logger.info(f"Training BPE tokenizer on {len(texts)} texts...")
        
        # Initialize character vocabulary
        char_counter = Counter()
        for text in texts:
            for char in text:
                char_counter[char] += 1
        
        # Build initial character vocabulary
        for char, _ in char_counter.most_common():
            self.char_vocab[char] = len(self.char_vocab)
        
        # Initialize vocabulary with special tokens and characters
        for token in self.special_tokens:
            self.vocab[token] = len(self.vocab)
        
        for char in self.char_vocab:
            self.vocab[char] = len(self.vocab)
        
        # Tokenize texts into words, then characters
        tokenized_texts = []
        word_freq = Counter()
        
        for text in texts:
            words = self._word_pattern.findall(text.lower())
            word_freq.update(words)
            tokenized_words = [self._tokenize_word(w) for w in words]
            tokenized_texts.append(tokenized_words)
        
        # BPE iterations
        current_vocab_size = len(self.vocab)
        target_vocab_size = self.vocab_size - len(self.special_tokens)
        
        iteration = 0
        while current_vocab_size < target_vocab_size:
            # Get pair statistics
            pairs = self._get_stats(tokenized_texts)
            
            # Filter by minimum frequency
            pairs = Counter({p: c for p, c in pairs.items() if c >= self.min_frequency})
            
            if not pairs:
                logger.info(f"No more pairs with frequency >= {self.min_frequency}")
                break
            
            # Get most frequent pair
            best_pair = pairs.most_common(1)[0][0]
            
            # Merge the pair
            tokenized_texts = self._merge_pair(best_pair, tokenized_texts)
            self.merges.append(best_pair)
            
            # Add new token to vocabulary
            new_token = ''.join(best_pair)
            if new_token not in self.vocab:
                self.vocab[new_token] = len(self.vocab)
                current_vocab_size += 1
            
            iteration += 1
            if show_progress and iteration % 100 == 0:
                logger.info(f"Iteration {iteration}: vocab size = {current_vocab_size}")
        
        # Build inverse vocabulary
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        
        logger.info(f"Training complete. Final vocab size: {len(self.vocab)}")
    
    def _word_to_ids(self, word: str) -> List[int]:
        """Convert a word to token IDs."""
        word = word.lower()
        
        # Try to find the word in vocabulary
        if word in self.vocab:
            return [self.vocab[word]]
        
        # Try to segment into known subwords
        tokens = []
        start = 0
        while start < len(word):
            end = len(word)
            found = False
            
            while end > start:
                subword = word[start:end]
                if subword in self.vocab:
                    tokens.append(self.vocab[subword])
                    start = end
                    found = True
                    break
                end -= 1
            
            if not found:
                # Unknown character
                tokens.append(self.vocab.get(self.unk_token, 0))
                start += 1
        
        return tokens
    
    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
        truncation: bool = False,
        max_length: Optional[int] = None,
        padding: bool = False,
    ) -> TokenizerOutput:
        """
        Encode text to token IDs.
        
        Args:
            text: Input text
            add_special_tokens: Whether to add BOS/EOS tokens
            truncation: Whether to truncate
            max_length: Maximum sequence length
            padding: Whether to pad to max_length
        
        Returns:
            TokenizerOutput with input_ids and attention_mask
        """
        # Tokenize into words
        words = self._word_pattern.findall(text.lower())
        
        # Convert words to token IDs
        token_ids = []
        tokens = []
        
        for word in words:
            word_ids = self._word_to_ids(word)
            token_ids.extend(word_ids)
            for tid in word_ids:
                tokens.append(self.inv_vocab.get(tid, self.unk_token))
        
        # Add special tokens
        special_tokens_mask = [0] * len(token_ids)
        
        if add_special_tokens:
            bos_id = self.vocab.get(self.bos_token, 0)
            eos_id = self.vocab.get(self.eos_token, 0)
            token_ids = [bos_id] + token_ids + [eos_id]
            tokens = [self.bos_token] + tokens + [self.eos_token]
            special_tokens_mask = [1] + special_tokens_mask + [1]
        
        # Create attention mask
        attention_mask = [1] * len(token_ids)
        
        # Truncate if needed
        if truncation and max_length is not None:
            if len(token_ids) > max_length:
                token_ids = token_ids[:max_length]
                tokens = tokens[:max_length]
                attention_mask = attention_mask[:max_length]
                special_tokens_mask = special_tokens_mask[:max_length]
        
        # Pad if needed
        if padding and max_length is not None:
            pad_id = self.vocab.get(self.pad_token, 0)
            padding_length = max_length - len(token_ids)
            token_ids.extend([pad_id] * padding_length)
            tokens.extend([self.pad_token] * padding_length)
            attention_mask.extend([0] * padding_length)
            special_tokens_mask.extend([1] * padding_length)
        
        return TokenizerOutput(
            input_ids=token_ids,
            attention_mask=attention_mask,
            tokens=tokens,
            special_tokens_mask=special_tokens_mask,
        )
    
    def decode(
        self,
        token_ids: List[int],
        skip_special_tokens: bool = True,
    ) -> str:
        """
        Decode token IDs to text.
        
        Args:
            token_ids: List of token IDs
            skip_special_tokens: Whether to skip special tokens
        
        Returns:
            Decoded text
        """
        special_ids = {
            self.vocab.get(self.bos_token, 0),
            self.vocab.get(self.eos_token, 0),
            self.vocab.get(self.pad_token, 0),
            self.vocab.get(self.unk_token, 0),
        }
        
        tokens = []
        for tid in token_ids:
            if skip_special_tokens and tid in special_ids:
                continue
            tokens.append(self.inv_vocab.get(tid, ''))
        
        # Join tokens (handle subword merging)
        text = ''
        for token in tokens:
            if token.startswith(' ') or len(text) == 0:
                text += token
            else:
                text += token
        
        return text.strip()
    
    def save(self, path: Union[str, Path]) -> None:
        """Save tokenizer to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'vocab': self.vocab,
            'merges': self.merges,
            'special_tokens': self.special_tokens,
            'unk_token': self.unk_token,
            'bos_token': self.bos_token,
            'eos_token': self.eos_token,
            'pad_token': self.pad_token,
            'vocab_size': self.vocab_size,
            'min_frequency': self.min_frequency,
        }
        
        with open(path / 'tokenizer.json', 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Tokenizer saved to {path}")
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> 'BPETokenizer':
        """Load tokenizer from file."""
        path = Path(path)
        
        with open(path / 'tokenizer.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        tokenizer = cls(
            vocab_size=data.get('vocab_size', 30000),
            min_frequency=data.get('min_frequency', 2),
            special_tokens=data.get('special_tokens', []),
            unk_token=data.get('unk_token', '<unk>'),
            bos_token=data.get('bos_token', '<s>'),
            eos_token=data.get('eos_token', '</s>'),
            pad_token=data.get('pad_token', '<pad>'),
        )
        
        tokenizer.vocab = data['vocab']
        tokenizer.inv_vocab = {v: k for k, v in data['vocab'].items()}
        tokenizer.merges = data.get('merges', [])
        
        return tokenizer


class WordPieceTokenizer:
    """
    WordPiece Tokenizer (used in BERT).
    
    Similar to BPE but uses a different merging criterion based on
    language model likelihood rather than frequency.
    
    Args:
        vocab_size: Target vocabulary size
        max_input_chars_per_word: Maximum characters per word
        special_tokens: List of special tokens
        
    Example:
        >>> tokenizer = WordPieceTokenizer(vocab_size=30000)
        >>> tokenizer.train(["hello world", "hello there"])
        >>> tokens = tokenizer.encode("hello world")
    """
    
    def __init__(
        self,
        vocab_size: int = 30000,
        max_input_chars_per_word: int = 100,
        special_tokens: Optional[List[str]] = None,
        unk_token: str = "[UNK]",
        cls_token: str = "[CLS]",
        sep_token: str = "[SEP]",
        pad_token: str = "[PAD]",
        mask_token: str = "[MASK]",
    ):
        self.vocab_size = vocab_size
        self.max_input_chars_per_word = max_input_chars_per_word
        
        # Special tokens (BERT-style)
        self.unk_token = unk_token
        self.cls_token = cls_token
        self.sep_token = sep_token
        self.pad_token = pad_token
        self.mask_token = mask_token
        
        self.special_tokens = special_tokens or []
        self.special_tokens.extend([
            unk_token, cls_token, sep_token, pad_token, mask_token
        ])
        
        # Vocabulary
        self.vocab: Dict[str, int] = {}
        self.inv_vocab: Dict[int, str] = {}
        
        # Word pattern
        self._word_pattern = re.compile(r'\w+|[^\w\s]')
    
    def _count_subwords(
        self,
        texts: List[str],
    ) -> Counter:
        """Count subword frequencies."""
        counter = Counter()
        
        for text in texts:
            words = self._word_pattern.findall(text.lower())
            for word in words:
                # Add word with ## prefix for continuation
                counter[word] += 1
                for i in range(1, len(word)):
                    counter[f"##{word[i:]}"] += 1
        
        return counter
    
    def train(
        self,
        texts: List[str],
        show_progress: bool = True,
    ) -> None:
        """
        Train WordPiece tokenizer.
        
        Args:
            texts: List of training texts
            show_progress: Whether to show progress
        """
        logger.info(f"Training WordPiece tokenizer on {len(texts)} texts...")
        
        # Initialize vocabulary with special tokens
        for token in self.special_tokens:
            self.vocab[token] = len(self.vocab)
        
        # Count subwords
        subword_counts = self._count_subwords(texts)
        
        # Add most frequent subwords to vocabulary
        for subword, count in subword_counts.most_common(self.vocab_size - len(self.special_tokens)):
            if subword not in self.vocab:
                self.vocab[subword] = len(self.vocab)
        
        # Build inverse vocabulary
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        
        logger.info(f"Training complete. Final vocab size: {len(self.vocab)}")
    
    def _wordpiece_tokenize(self, word: str) -> List[str]:
        """Tokenize a word using WordPiece algorithm."""
        word = word.lower()
        
        if len(word) > self.max_input_chars_per_word:
            return [self.unk_token]
        
        tokens = []
        start = 0
        
        while start < len(word):
            end = len(word)
            cur = None
            
            while start < end:
                substr = word[start:end]
                if start > 0:
                    substr = f"##{substr}"
                
                if substr in self.vocab:
                    cur = substr
                    break
                
                end -= 1
            
            if cur is None:
                tokens.append(self.unk_token)
                break
            
            tokens.append(cur)
            start = end
        
        return tokens
    
    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
        truncation: bool = False,
        max_length: Optional[int] = None,
        padding: bool = False,
    ) -> TokenizerOutput:
        """Encode text to token IDs."""
        words = self._word_pattern.findall(text)
        
        token_ids = []
        tokens = []
        
        for word in words:
            word_tokens = self._wordpiece_tokenize(word)
            for token in word_tokens:
                token_ids.append(self.vocab.get(token, self.vocab[self.unk_token]))
                tokens.append(token)
        
        # Add special tokens for BERT
        if add_special_tokens:
            cls_id = self.vocab[self.cls_token]
            sep_id = self.vocab[self.sep_token]
            token_ids = [cls_id] + token_ids + [sep_id]
            tokens = [self.cls_token] + tokens + [self.sep_token]
        
        # Attention mask
        attention_mask = [1] * len(token_ids)
        
        # Truncate
        if truncation and max_length is not None and len(token_ids) > max_length:
            token_ids = token_ids[:max_length]
            tokens = tokens[:max_length]
            attention_mask = attention_mask[:max_length]
        
        # Pad
        if padding and max_length is not None:
            pad_id = self.vocab[self.pad_token]
            padding_length = max_length - len(token_ids)
            token_ids.extend([pad_id] * padding_length)
            tokens.extend([self.pad_token] * padding_length)
            attention_mask.extend([0] * padding_length)
        
        return TokenizerOutput(
            input_ids=token_ids,
            attention_mask=attention_mask,
            tokens=tokens,
        )
    
    def decode(
        self,
        token_ids: List[int],
        skip_special_tokens: bool = True,
    ) -> str:
        """Decode token IDs to text."""
        special_ids = {
            self.vocab.get(self.cls_token, 0),
            self.vocab.get(self.sep_token, 0),
            self.vocab.get(self.pad_token, 0),
            self.vocab.get(self.mask_token, 0),
        }
        
        tokens = []
        for tid in token_ids:
            if skip_special_tokens and tid in special_ids:
                continue
            tokens.append(self.inv_vocab.get(tid, ''))
        
        # Join tokens (handle ## prefix)
        text = ''
        for token in tokens:
            if token.startswith('##'):
                text += token[2:]
            elif text and not text.endswith(' '):
                text += ' ' + token
            else:
                text += token
        
        return text.strip()
    
    def save(self, path: Union[str, Path]) -> None:
        """Save tokenizer to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'vocab': self.vocab,
            'special_tokens': self.special_tokens,
            'vocab_size': self.vocab_size,
        }
        
        with open(path / 'tokenizer.json', 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        # Also save vocab.txt for compatibility
        with open(path / 'vocab.txt', 'w', encoding='utf-8') as f:
            for token in sorted(self.vocab.keys(), key=lambda x: self.vocab[x]):
                f.write(token + '\n')
        
        logger.info(f"Tokenizer saved to {path}")
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> 'WordPieceTokenizer':
        """Load tokenizer from file."""
        path = Path(path)
        
        with open(path / 'tokenizer.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        tokenizer = cls(
            vocab_size=data.get('vocab_size', 30000),
            special_tokens=data.get('special_tokens', []),
        )
        
        tokenizer.vocab = data['vocab']
        tokenizer.inv_vocab = {v: k for k, v in data['vocab'].items()}
        
        return tokenizer


class SentencePieceTokenizer:
    """
    SentencePiece Tokenizer.
    
    Treats input as raw sequence of characters without pre-tokenization.
    Uses unigram language model for subword segmentation.
    
    This is a simplified implementation. For production, use the
    official sentencepiece package: pip install sentencepiece
    
    Args:
        vocab_size: Target vocabulary size
        model_type: Model type ('unigram' or 'bpe')
        special_tokens: List of special tokens
        
    Example:
        >>> tokenizer = SentencePieceTokenizer(vocab_size=8000)
        >>> tokenizer.train(["hello world", "hello there"])
        >>> tokens = tokenizer.encode("hello world")
    """
    
    def __init__(
        self,
        vocab_size: int = 8000,
        model_type: str = 'unigram',
        special_tokens: Optional[List[str]] = None,
        unk_token: str = "<unk>",
        bos_token: str = "<s>",
        eos_token: str = "</s>",
        pad_token: str = "<pad>",
    ):
        self.vocab_size = vocab_size
        self.model_type = model_type
        
        # Special tokens
        self.unk_token = unk_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.pad_token = pad_token
        
        self.special_tokens = special_tokens or []
        self.special_tokens.extend([unk_token, bos_token, eos_token, pad_token])
        
        # Vocabulary
        self.vocab: Dict[str, int] = {}
        self.inv_vocab: Dict[int, str] = {}
        
        # Unigram probabilities (for unigram model)
        self.unigram_probs: Dict[str, float] = {}
    
    def _build_character_vocab(self, texts: List[str]) -> Dict[str, int]:
        """Build initial character vocabulary."""
        char_counter = Counter()
        for text in texts:
            for char in text:
                char_counter[char] += 1
        
        vocab = {}
        for char, _ in char_counter.most_common():
            vocab[char] = len(vocab)
        
        return vocab
    
    def _get_subwords(self, text: str) -> List[str]:
        """Get all possible subwords from text."""
        subwords = []
        for i in range(len(text)):
            for j in range(i + 1, min(i + 10, len(text) + 1)):
                subwords.append(text[i:j])
        return subwords
    
    def train(
        self,
        texts: List[str],
        show_progress: bool = True,
    ) -> None:
        """
        Train SentencePiece tokenizer.
        
        Args:
            texts: List of training texts
            show_progress: Whether to show progress
        """
        logger.info(f"Training SentencePiece tokenizer ({self.model_type})...")
        
        # Initialize with special tokens
        for token in self.special_tokens:
            self.vocab[token] = len(self.vocab)
        
        # Build character vocabulary
        char_vocab = self._build_character_vocab(texts)
        for char in char_vocab:
            if char not in self.vocab:
                self.vocab[char] = len(self.vocab)
        
        # Count subword frequencies
        subword_counts = Counter()
        for text in texts:
            for subword in self._get_subwords(text):
                subword_counts[subword] += 1
        
        # Add most frequent subwords
        remaining = self.vocab_size - len(self.vocab)
        for subword, _ in subword_counts.most_common(remaining):
            if subword not in self.vocab:
                self.vocab[subword] = len(self.vocab)
        
        # Build inverse vocabulary
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        
        # Compute unigram probabilities (simplified)
        total = sum(subword_counts.values())
        for subword in self.vocab:
            self.unigram_probs[subword] = subword_counts.get(subword, 0) / total
        
        logger.info(f"Training complete. Final vocab size: {len(self.vocab)}")
    
    def _encode_unigram(self, text: str) -> List[str]:
        """Encode using unigram model (greedy)."""
        tokens = []
        i = 0
        
        while i < len(text):
            best_token = text[i]
            best_prob = self.unigram_probs.get(best_token, 0)
            
            # Try longer substrings
            for j in range(i + 1, min(i + 10, len(text) + 1)):
                subword = text[i:j]
                prob = self.unigram_probs.get(subword, 0)
                if prob > best_prob:
                    best_token = subword
                    best_prob = prob
            
            tokens.append(best_token)
            i += len(best_token)
        
        return tokens
    
    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
        truncation: bool = False,
        max_length: Optional[int] = None,
        padding: bool = False,
    ) -> TokenizerOutput:
        """Encode text to token IDs."""
        # Encode using unigram model
        if self.model_type == 'unigram':
            tokens = self._encode_unigram(text)
        else:
            # Fallback to character-level
            tokens = list(text)
        
        # Convert to IDs
        token_ids = [
            self.vocab.get(token, self.vocab.get(self.unk_token, 0))
            for token in tokens
        ]
        
        # Add special tokens
        if add_special_tokens:
            bos_id = self.vocab.get(self.bos_token, 0)
            eos_id = self.vocab.get(self.eos_token, 0)
            token_ids = [bos_id] + token_ids + [eos_id]
            tokens = [self.bos_token] + tokens + [self.eos_token]
        
        # Attention mask
        attention_mask = [1] * len(token_ids)
        
        # Truncate
        if truncation and max_length is not None and len(token_ids) > max_length:
            token_ids = token_ids[:max_length]
            tokens = tokens[:max_length]
            attention_mask = attention_mask[:max_length]
        
        # Pad
        if padding and max_length is not None:
            pad_id = self.vocab.get(self.pad_token, 0)
            padding_length = max_length - len(token_ids)
            token_ids.extend([pad_id] * padding_length)
            tokens.extend([self.pad_token] * padding_length)
            attention_mask.extend([0] * padding_length)
        
        return TokenizerOutput(
            input_ids=token_ids,
            attention_mask=attention_mask,
            tokens=tokens,
        )
    
    def decode(
        self,
        token_ids: List[int],
        skip_special_tokens: bool = True,
    ) -> str:
        """Decode token IDs to text."""
        special_ids = {
            self.vocab.get(self.bos_token, 0),
            self.vocab.get(self.eos_token, 0),
            self.vocab.get(self.pad_token, 0),
            self.vocab.get(self.unk_token, 0),
        }
        
        tokens = []
        for tid in token_ids:
            if skip_special_tokens and tid in special_ids:
                continue
            tokens.append(self.inv_vocab.get(tid, ''))
        
        return ''.join(tokens)
    
    def save(self, path: Union[str, Path]) -> None:
        """Save tokenizer to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'vocab': self.vocab,
            'unigram_probs': self.unigram_probs,
            'model_type': self.model_type,
            'special_tokens': self.special_tokens,
            'vocab_size': self.vocab_size,
        }
        
        with open(path / 'tokenizer.json', 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Tokenizer saved to {path}")
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> 'SentencePieceTokenizer':
        """Load tokenizer from file."""
        path = Path(path)
        
        with open(path / 'tokenizer.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        tokenizer = cls(
            vocab_size=data.get('vocab_size', 8000),
            model_type=data.get('model_type', 'unigram'),
            special_tokens=data.get('special_tokens', []),
        )
        
        tokenizer.vocab = data['vocab']
        tokenizer.inv_vocab = {v: k for k, v in data['vocab'].items()}
        tokenizer.unigram_probs = data.get('unigram_probs', {})
        
        return tokenizer


class TokenProcessor:
    """
    Token Processor for batch processing and advanced features.
    
    Provides utilities for:
    - Batch encoding/decoding
    - Padding and truncation
    - Token type IDs for sequence pairs
    - Special token management
    
    Args:
        tokenizer: Base tokenizer (BPE, WordPiece, or SentencePiece)
        
    Example:
        >>> tokenizer = BPETokenizer(vocab_size=10000)
        >>> processor = TokenProcessor(tokenizer)
        >>> batch = processor.encode_batch(["hello", "world"])
    """
    
    def __init__(self, tokenizer: Union[BPETokenizer, WordPieceTokenizer, SentencePieceTokenizer]):
        self.tokenizer = tokenizer
    
    def encode_batch(
        self,
        texts: List[str],
        add_special_tokens: bool = True,
        truncation: bool = True,
        max_length: Optional[int] = None,
        padding: bool = True,
        return_tensors: str = 'pt',
    ) -> Dict[str, Tensor]:
        """
        Encode a batch of texts.
        
        Args:
            texts: List of texts to encode
            add_special_tokens: Whether to add special tokens
            truncation: Whether to truncate
            max_length: Maximum sequence length
            padding: Whether to pad
            return_tensors: Return format ('pt' for PyTorch)
        
        Returns:
            Dictionary of tensors
        """
        # Determine max length if not provided
        if max_length is None and padding:
            max_length = max(
                len(self.tokenizer.encode(t, add_special_tokens=add_special_tokens).input_ids)
                for t in texts
            )
        
        # Encode all texts
        outputs = [
            self.tokenizer.encode(
                text,
                add_special_tokens=add_special_tokens,
                truncation=truncation,
                max_length=max_length,
                padding=padding and max_length is not None,
            )
            for text in texts
        ]
        
        # Pad to same length if needed
        if padding and max_length is None:
            max_length = max(len(o.input_ids) for o in outputs)
            for output in outputs:
                while len(output.input_ids) < max_length:
                    pad_id = self.tokenizer.vocab.get(
                        getattr(self.tokenizer, 'pad_token', '<pad>'),
                        0
                    )
                    output.input_ids.append(pad_id)
                    output.attention_mask.append(0)
        
        # Convert to tensors
        if return_tensors == 'pt':
            return {
                'input_ids': torch.tensor([o.input_ids for o in outputs], dtype=torch.long),
                'attention_mask': torch.tensor([o.attention_mask for o in outputs], dtype=torch.long),
            }
        
        return {
            'input_ids': [o.input_ids for o in outputs],
            'attention_mask': [o.attention_mask for o in outputs],
        }
    
    def encode_pair(
        self,
        text1: str,
        text2: str,
        add_special_tokens: bool = True,
        truncation: bool = True,
        max_length: Optional[int] = None,
    ) -> TokenizerOutput:
        """
        Encode a pair of texts (for sequence pair tasks).
        
        Args:
            text1: First text
            text2: Second text
            add_special_tokens: Whether to add special tokens
            truncation: Whether to truncate
            max_length: Maximum sequence length
        
        Returns:
            TokenizerOutput with token_type_ids
        """
        # Encode both texts
        output1 = self.tokenizer.encode(text1, add_special_tokens=False)
        output2 = self.tokenizer.encode(text2, add_special_tokens=False)
        
        # Combine
        input_ids = output1.input_ids + output2.input_ids
        tokens = output1.tokens + output2.tokens
        
        # Token type IDs (0 for first sequence, 1 for second)
        token_type_ids = [0] * len(output1.input_ids) + [1] * len(output2.input_ids)
        
        # Add special tokens
        if add_special_tokens:
            if isinstance(self.tokenizer, WordPieceTokenizer):
                # BERT-style: [CLS] text1 [SEP] text2 [SEP]
                cls_id = self.tokenizer.vocab.get(self.tokenizer.cls_token, 0)
                sep_id = self.tokenizer.vocab.get(self.tokenizer.sep_token, 0)
                
                input_ids = [cls_id] + input_ids + [sep_id]
                tokens = [self.tokenizer.cls_token] + tokens + [self.tokenizer.sep_token]
                token_type_ids = [0] + token_type_ids + [1]
            else:
                # Generic: <s> text1 text2 </s>
                bos_id = self.tokenizer.vocab.get(getattr(self.tokenizer, 'bos_token', '<s>'), 0)
                eos_id = self.tokenizer.vocab.get(getattr(self.tokenizer, 'eos_token', '</s>'), 0)
                
                input_ids = [bos_id] + input_ids + [eos_id]
                tokens = [getattr(self.tokenizer, 'bos_token', '<s>')] + tokens + [getattr(self.tokenizer, 'eos_token', '</s>')]
                token_type_ids = [0] + token_type_ids + [0]
        
        # Attention mask
        attention_mask = [1] * len(input_ids)
        
        # Truncate
        if truncation and max_length is not None and len(input_ids) > max_length:
            input_ids = input_ids[:max_length]
            tokens = tokens[:max_length]
            attention_mask = attention_mask[:max_length]
            token_type_ids = token_type_ids[:max_length]
        
        return TokenizerOutput(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            tokens=tokens,
        )
    
    def decode_batch(
        self,
        token_ids: Union[List[List[int]], Tensor],
        skip_special_tokens: bool = True,
    ) -> List[str]:
        """
        Decode a batch of token IDs.
        
        Args:
            token_ids: Batch of token IDs
            skip_special_tokens: Whether to skip special tokens
        
        Returns:
            List of decoded texts
        """
        if isinstance(token_ids, Tensor):
            token_ids = token_ids.tolist()
        
        return [
            self.tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)
            for ids in token_ids
        ]
