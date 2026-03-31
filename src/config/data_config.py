"""
Data Configuration Classes
==========================

Configuration classes for data pipelines and preprocessing.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from pathlib import Path


@dataclass
class DataConfig:
    """
    Data pipeline configuration.

    Attributes:
        data_dir: Directory containing data files
        train_file: Training data file path
        val_file: Validation data file path
        test_file: Test data file path
        batch_size: Batch size for data loading
        num_workers: Number of data loading workers
        shuffle: Shuffle training data
        pin_memory: Pin memory for faster GPU transfer
        persistent_workers: Keep workers alive between epochs
    """
    data_dir: Path = field(default_factory=lambda: Path("data"))
    train_file: Optional[Path] = None
    val_file: Optional[Path] = None
    test_file: Optional[Path] = None
    batch_size: int = 32
    num_workers: int = 4
    shuffle: bool = True
    pin_memory: bool = True
    persistent_workers: bool = True

    def __post_init__(self):
        """Resolve paths relative to data_dir."""
        if self.train_file and not self.train_file.is_absolute():
            self.train_file = self.data_dir / self.train_file
        if self.val_file and not self.val_file.is_absolute():
            self.val_file = self.data_dir / self.val_file
        if self.test_file and not self.test_file.is_absolute():
            self.test_file = self.data_dir / self.test_file

    @property
    def has_train(self) -> bool:
        """Check if training file exists."""
        return self.train_file is not None and self.train_file.exists()

    @property
    def has_val(self) -> bool:
        """Check if validation file exists."""
        return self.val_file is not None and self.val_file.exists()

    @property
    def has_test(self) -> bool:
        """Check if test file exists."""
        return self.test_file is not None and self.test_file.exists()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "data_dir": str(self.data_dir),
            "train_file": str(self.train_file) if self.train_file else None,
            "val_file": str(self.val_file) if self.val_file else None,
            "test_file": str(self.test_file) if self.test_file else None,
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "shuffle": self.shuffle,
            "pin_memory": self.pin_memory,
            "persistent_workers": self.persistent_workers,
        }


@dataclass
class PreprocessingConfig:
    """
    Data preprocessing configuration.

    Attributes:
        max_length: Maximum sequence length
        padding: Padding strategy
        truncation: Truncation strategy
        add_special_tokens: Add special tokens
        return_tensors: Type of tensors to return
        normalize: Normalize text
        lowercase: Convert to lowercase
        remove_special_chars: Remove special characters
        max_vocab_size: Maximum vocabulary size
        min_freq: Minimum token frequency
    """
    max_length: int = 512
    padding: str = "max_length"  # max_length, longest, do_not_pad
    truncation: str = "longest_first"  # longest_first, only_first, only_second
    add_special_tokens: bool = True
    return_tensors: str = "pt"  # pt (pytorch), tf (tensorflow), np (numpy)

    # Text preprocessing
    normalize: bool = True
    lowercase: bool = True
    remove_special_chars: bool = False

    # Vocabulary
    max_vocab_size: int = 30000
    min_freq: int = 2

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "max_length": self.max_length,
            "padding": self.padding,
            "truncation": self.truncation,
            "add_special_tokens": self.add_special_tokens,
            "return_tensors": self.return_tensors,
            "normalize": self.normalize,
            "lowercase": self.lowercase,
            "remove_special_chars": self.remove_special_chars,
            "max_vocab_size": self.max_vocab_size,
            "min_freq": self.min_freq,
        }
