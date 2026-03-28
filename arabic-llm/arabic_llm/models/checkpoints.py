"""
Arabic LLM - Checkpoint Management

Utilities for saving and loading model checkpoints.
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime


def save_checkpoint(
    model,
    tokenizer,
    checkpoint_dir: str,
    metadata: Optional[Dict] = None,
) -> str:
    """
    Save model checkpoint.
    
    Args:
        model: Model to save
        tokenizer: Tokenizer
        checkpoint_dir: Directory to save checkpoint
        metadata: Additional metadata
        
    Returns:
        Path to saved checkpoint
    """
    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    
    # Save model
    model.save_pretrained(checkpoint_path)
    
    # Save tokenizer
    tokenizer.save_pretrained(checkpoint_path)
    
    # Save metadata
    if metadata is None:
        metadata = {}
    
    metadata["saved_at"] = datetime.now().isoformat()
    metadata_path = checkpoint_path / "metadata.json"
    
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    return str(checkpoint_path)


def load_checkpoint(
    checkpoint_dir: str,
    model_class=None,
    tokenizer_class=None,
):
    """
    Load model checkpoint.
    
    Args:
        checkpoint_dir: Directory containing checkpoint
        model_class: Model class to load into
        tokenizer_class: Tokenizer class to load
        
    Returns:
        Model, tokenizer, metadata
    """
    checkpoint_path = Path(checkpoint_dir)
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        # Load model
        if model_class is None:
            model = AutoModelForCausalLM.from_pretrained(
                checkpoint_path,
                trust_remote_code=True,
            )
        else:
            model = model_class.from_pretrained(checkpoint_path)
        
        # Load tokenizer
        if tokenizer_class is None:
            tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
        else:
            tokenizer = tokenizer_class.from_pretrained(checkpoint_path)
        
        # Load metadata
        metadata_path = checkpoint_path / "metadata.json"
        metadata = {}
        if metadata_path.exists():
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        
        return model, tokenizer, metadata
        
    except ImportError as e:
        raise ImportError(
            "Required packages not installed. "
            "Install with: pip install transformers"
        ) from e


def list_checkpoints(base_dir: str) -> List[Dict]:
    """
    List all checkpoints in directory.
    
    Args:
        base_dir: Base directory containing checkpoints
        
    Returns:
        List of checkpoint info dictionaries
    """
    base_path = Path(base_dir)
    
    if not base_path.exists():
        return []
    
    checkpoints = []
    
    for checkpoint_dir in base_path.iterdir():
        if checkpoint_dir.is_dir():
            # Check if it's a valid checkpoint
            model_file = checkpoint_dir / "model.safetensors"
            config_file = checkpoint_dir / "config.json"
            
            if model_file.exists() or config_file.exists():
                # Load metadata if available
                metadata = {}
                metadata_path = checkpoint_dir / "metadata.json"
                if metadata_path.exists():
                    with open(metadata_path, "r", encoding="utf-8") as f:
                        metadata = json.load(f)
                
                checkpoints.append({
                    "path": str(checkpoint_dir),
                    "name": checkpoint_dir.name,
                    "saved_at": metadata.get("saved_at", "Unknown"),
                    "metadata": metadata,
                })
    
    # Sort by saved_at (newest first)
    checkpoints.sort(
        key=lambda x: x.get("saved_at", ""),
        reverse=True
    )
    
    return checkpoints


def get_best_checkpoint(base_dir: str, metric: str = "val_loss") -> Optional[str]:
    """
    Get best checkpoint based on metric.
    
    Args:
        base_dir: Base directory containing checkpoints
        metric: Metric to optimize (lower is better)
        
    Returns:
        Path to best checkpoint
    """
    checkpoints = list_checkpoints(base_dir)
    
    if not checkpoints:
        return None
    
    best_checkpoint: Optional[str] = None
    best_metric = float('inf')
    
    for checkpoint in checkpoints:
        metric_value = checkpoint.get("metadata", {}).get(metric)
        if metric_value is not None and metric_value < best_metric:
            best_metric = metric_value
            best_checkpoint = checkpoint["path"]
    
    return best_checkpoint


def cleanup_old_checkpoints(
    base_dir: str,
    keep_last_n: int = 3,
) -> List[str]:
    """
    Remove old checkpoints, keeping only the last N.
    
    Args:
        base_dir: Base directory containing checkpoints
        keep_last_n: Number of checkpoints to keep
        
    Returns:
        List of removed checkpoint paths
    """
    checkpoints = list_checkpoints(base_dir)
    
    if len(checkpoints) <= keep_last_n:
        return []
    
    removed = []
    
    # Remove all except last N
    for checkpoint in checkpoints[keep_last_n:]:
        checkpoint_path = Path(checkpoint["path"])
        
        # Delete checkpoint directory
        import shutil
        shutil.rmtree(checkpoint_path)
        
        removed.append(checkpoint["path"])
    
    return removed
