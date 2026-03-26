"""
Arabic LLM - Experiment Proposals

Generate experiment proposals for autonomous research.
"""

from dataclasses import dataclass
from typing import List, Dict, Any
import re
from pathlib import Path


@dataclass
class ExperimentProposal:
    """Represents a proposed experiment"""
    
    id: int
    change: str
    modifications: Dict[str, Any]
    category: str = "general"
    
    def apply(self, train_file: Path):
        """Apply modifications to train.py"""
        with open(train_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Apply each modification
        for param, value in self.modifications.items():
            # Find and replace parameter value
            pattern = rf'{param}\s*=\s*[^#\n]+'
            replacement = f'{param} = {value}'
            content = re.sub(pattern, replacement, content)
        
        # Update experiment metadata
        content = re.sub(
            r'EXPERIMENT_ID\s*=\s*\d+',
            f'EXPERIMENT_ID = {self.id}',
            content
        )
        content = re.sub(
            r'EXPERIMENT_CHANGE\s*=.*',
            f'EXPERIMENT_CHANGE = "{self.change}"',
            content
        )
        
        with open(train_file, 'w', encoding='utf-8') as f:
            f.write(content)
    
    def __repr__(self):
        return f"Experiment #{self.id}: {self.change}"


def get_experiment_proposals() -> List[ExperimentProposal]:
    """
    Generate list of experiment proposals.
    
    Returns:
        List of ExperimentProposal objects
    """
    proposals = []
    exp_id = 1
    
    # LoRA rank experiments
    for r in [32, 64, 128, 256]:
        proposals.append(ExperimentProposal(
            id=exp_id,
            change=f"LoRA rank r={r}, alpha={r*2}",
            modifications={
                'LORA_R': r,
                'LORA_ALPHA': r * 2,
            },
            category="lora",
        ))
        exp_id += 1
    
    # Learning rate experiments
    for lr in [1e-4, 2e-4, 5e-4, 1e-3]:
        proposals.append(ExperimentProposal(
            id=exp_id,
            change=f"Learning rate lr={lr}",
            modifications={
                'LEARNING_RATE': lr,
            },
            category="optimizer",
        ))
        exp_id += 1
    
    # Batch size experiments
    for batch in [4, 8, 16, 32]:
        proposals.append(ExperimentProposal(
            id=exp_id,
            change=f"Total batch size={batch}",
            modifications={
                'TOTAL_BATCH_SIZE': batch,
                'GRADIENT_ACCUMULATION_STEPS': max(1, batch // 2),
            },
            category="batch",
        ))
        exp_id += 1
    
    # Depth experiments
    for depth in [6, 8, 12, 16]:
        proposals.append(ExperimentProposal(
            id=exp_id,
            change=f"Model depth={depth}",
            modifications={
                'DEPTH': depth,
            },
            category="architecture",
        ))
        exp_id += 1
    
    # Warmup ratio experiments
    for warmup in [0.01, 0.03, 0.05, 0.10]:
        proposals.append(ExperimentProposal(
            id=exp_id,
            change=f"Warmup ratio={warmup}",
            modifications={
                'WARMUP_RATIO': warmup,
            },
            category="scheduler",
        ))
        exp_id += 1
    
    # Dropout experiments
    for dropout in [0.0, 0.1, 0.2]:
        proposals.append(ExperimentProposal(
            id=exp_id,
            change=f"Dropout={dropout}",
            modifications={
                'DROPOUT': dropout,
                'LORA_DROPOUT': dropout,
            },
            category="regularization",
        ))
        exp_id += 1
    
    return proposals


def get_proposals_by_category(category: str) -> List[ExperimentProposal]:
    """
    Get experiment proposals by category.
    
    Args:
        category: Category name (lora, optimizer, batch, architecture, scheduler, regularization)
        
    Returns:
        List of ExperimentProposal objects
    """
    all_proposals = get_experiment_proposals()
    return [p for p in all_proposals if p.category == category]
