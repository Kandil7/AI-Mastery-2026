"""
Arabic LLM Autonomous Research Agent

This script runs the autonomous research loop:
1. Read program.md for instructions
2. Propose train.py modifications
3. Run training (5-minute budget)
4. Evaluate val_loss
5. Keep/discard changes
6. Repeat

Usage:
    python agent.py --experiments 100 --time-per-exp 300
"""

import os
import sys
import json
import time
import shutil
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Tuple
import re


# =============================================================================
# CONFIGURATION
# =============================================================================

EXPERIMENTS_DIR = Path("experiments")
CHECKPOINTS_DIR = Path("checkpoints")
TRAIN_FILE = Path("train.py")
TRAIN_BACKUP = Path("train.py.backup")
PROGRAM_FILE = Path("program.md")
LOG_FILE = EXPERIMENTS_DIR / "experiment_log.jsonl"
BEST_LOSS_FILE = EXPERIMENTS_DIR / "best_loss.txt"

# Default experiment parameters
DEFAULT_EXPERIMENTS = 100
DEFAULT_TIME_PER_EXP = 300  # 5 minutes


# =============================================================================
# EXPERIMENT PROPOSALS
# =============================================================================

class ExperimentProposal:
    """Represents a proposed experiment"""
    
    def __init__(self, id: int, change: str, modifications: Dict[str, any]):
        self.id = id
        self.change = change
        self.modifications = modifications
    
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
    """Generate list of experiment proposals"""
    
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
            }
        ))
        exp_id += 1
    
    # Learning rate experiments
    for lr in [1e-4, 2e-4, 5e-4, 1e-3]:
        proposals.append(ExperimentProposal(
            id=exp_id,
            change=f"Learning rate lr={lr}",
            modifications={
                'LEARNING_RATE': lr,
            }
        ))
        exp_id += 1
    
    # Batch size experiments
    for batch in [4, 8, 16, 32]:
        proposals.append(ExperimentProposal(
            id=exp_id,
            change=f"Total batch size={batch}",
            modifications={
                'TOTAL_BATCH_SIZE': batch,
                'GRADIENT_ACCUMULATION_STEPS': batch // 2,
            }
        ))
        exp_id += 1
    
    # Depth experiments
    for depth in [6, 8, 12, 16]:
        proposals.append(ExperimentProposal(
            id=exp_id,
            change=f"Model depth={depth}",
            modifications={
                'DEPTH': depth,
            }
        ))
        exp_id += 1
    
    # Warmup ratio experiments
    for warmup in [0.01, 0.03, 0.05, 0.10]:
        proposals.append(ExperimentProposal(
            id=exp_id,
            change=f"Warmup ratio={warmup}",
            modifications={
                'WARMUP_RATIO': warmup,
            }
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
            }
        ))
        exp_id += 1
    
    return proposals


# =============================================================================
# TRAINING EXECUTION
# =============================================================================

def run_training(timeout: int = DEFAULT_TIME_PER_EXP) -> Tuple[Optional[float], Optional[float], bool]:
    """
    Run training script with timeout
    
    Returns:
        (val_loss, train_loss, success)
    """
    print(f"\n🚀 Starting training (timeout: {timeout}s)...")
    
    try:
        # Run training
        result = subprocess.run(
            [sys.executable, str(TRAIN_FILE)],
            capture_output=True,
            text=True,
            timeout=timeout + 60,  # Add 1 minute buffer
            cwd=Path(__file__).parent
        )
        
        # Parse output for metrics
        val_loss = None
        train_loss = None
        
        output = result.stdout + result.stderr
        
        # Look for val_loss in output
        for line in output.split('\n'):
            if 'val_loss:' in line.lower():
                match = re.search(r'val_loss:\s*([\d.]+)', line, re.IGNORECASE)
                if match:
                    val_loss = float(match.group(1))
            
            if 'train_loss:' in line.lower():
                match = re.search(r'train_loss:\s*([\d.]+)', line, re.IGNORECASE)
                if match:
                    train_loss = float(match.group(1))
        
        success = result.returncode == 0
        
        if not success:
            print(f"❌ Training failed with return code {result.returncode}")
            if result.stderr:
                print(f"Error: {result.stderr[:500]}")
        
        return val_loss, train_loss, success
        
    except subprocess.TimeoutExpired:
        print(f"⏰ Training timed out after {timeout}s")
        return None, None, False
    except Exception as e:
        print(f"❌ Error running training: {e}")
        return None, None, False


# =============================================================================
# LOGGING
# =============================================================================

def load_best_loss() -> float:
    """Load best validation loss from file"""
    if BEST_LOSS_FILE.exists():
        with open(BEST_LOSS_FILE, 'r') as f:
            return float(f.read().strip())
    return float('inf')  # No baseline yet


def save_best_loss(loss: float):
    """Save best validation loss to file"""
    EXPERIMENTS_DIR.mkdir(exist_ok=True)
    with open(BEST_LOSS_FILE, 'w') as f:
        f.write(str(loss))


def log_experiment(exp: ExperimentProposal, val_loss: float, train_loss: float, 
                   improved: bool, time_seconds: float):
    """Log experiment result"""
    EXPERIMENTS_DIR.mkdir(exist_ok=True)
    
    log_entry = {
        "experiment": exp.id,
        "change": exp.change,
        "val_loss": val_loss,
        "train_loss": train_loss,
        "improved": improved,
        "time_seconds": time_seconds,
        "timestamp": datetime.now().isoformat(),
    }
    
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')


# =============================================================================
# MAIN AGENT LOOP
# =============================================================================

def run_agent(num_experiments: int = DEFAULT_EXPERIMENTS, 
              time_per_exp: int = DEFAULT_TIME_PER_EXP):
    """Main autonomous agent loop"""
    
    print("=" * 70)
    print("Arabic LLM Autonomous Research Agent")
    print("=" * 70)
    print(f"Experiments: {num_experiments}")
    print(f"Time per experiment: {time_per_exp}s ({time_per_exp/60:.1f}m)")
    print(f"Total time budget: {num_experiments * time_per_exp / 3600:.1f}h")
    print("=" * 70)
    
    # Create directories
    EXPERIMENTS_DIR.mkdir(exist_ok=True)
    CHECKPOINTS_DIR.mkdir(exist_ok=True)
    
    # Load best loss
    best_loss = load_best_loss()
    print(f"\n📊 Current best val_loss: {best_loss:.4f}")
    
    # Get experiment proposals
    proposals = get_experiment_proposals()
    print(f"📋 Loaded {len(proposals)} experiment proposals")
    
    # Backup original train.py
    if TRAIN_FILE.exists():
        shutil.copy(TRAIN_FILE, TRAIN_BACKUP)
        print(f"💾 Backed up {TRAIN_FILE}")
    
    # Run experiments
    start_time = time.time()
    completed = 0
    improved_count = 0
    
    for i, proposal in enumerate(proposals[:num_experiments]):
        exp_start = time.time()
        
        print(f"\n{'='*70}")
        print(f"Experiment {i+1}/{num_experiments}")
        print(f"Proposal: {proposal.change}")
        print(f"{'='*70}")
        
        # Apply modifications
        proposal.apply(TRAIN_FILE)
        print(f"✏️  Applied modifications to {TRAIN_FILE}")
        
        # Run training
        val_loss, train_loss, success = run_training(timeout=time_per_exp)
        
        if success and val_loss is not None:
            # Check if improved
            improved = val_loss < best_loss
            
            if improved:
                print(f"✅ IMPROVED! val_loss: {best_loss:.4f} → {val_loss:.4f}")
                best_loss = val_loss
                save_best_loss(best_loss)
                improved_count += 1
                
                # Keep this configuration (don't revert)
                print(f"💾 Keeping improved configuration")
            else:
                print(f"❌ Not improved. val_loss: {val_loss:.4f} (best: {best_loss:.4f})")
                
                # Revert to backup
                shutil.copy(TRAIN_BACKUP, TRAIN_FILE)
                print(f"↩️  Reverted to baseline configuration")
            
            # Log result
            elapsed = time.time() - exp_start
            log_experiment(proposal, val_loss, train_loss or 0.0, improved, elapsed)
            
        else:
            print(f"❌ Training failed or timed out")
            
            # Revert to backup
            shutil.copy(TRAIN_BACKUP, TRAIN_FILE)
        
        completed += 1
        
        # Print summary
        total_elapsed = time.time() - start_time
        avg_time = total_elapsed / completed
        remaining = (num_experiments - completed) * avg_time
        
        print(f"\n📊 Summary:")
        print(f"  Completed: {completed}/{num_experiments}")
        print(f"  Improved: {improved_count} ({improved_count/completed*100:.1f}%)")
        print(f"  Best val_loss: {best_loss:.4f}")
        print(f"  Avg time/exp: {avg_time/60:.1f}m")
        print(f"  Remaining: {remaining/3600:.1f}h")
    
    # Final summary
    total_time = time.time() - start_time
    
    print(f"\n{'='*70}")
    print("Autonomous Research Complete")
    print(f"{'='*70}")
    print(f"Total experiments: {completed}")
    print(f"Improved: {improved_count} ({improved_count/completed*100:.1f}%)")
    print(f"Final best val_loss: {best_loss:.4f}")
    print(f"Total time: {total_time/3600:.1f}h")
    print(f"{'='*70}")
    
    return 0


# =============================================================================
# CLI
# =============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Arabic LLM Autonomous Research Agent")
    parser.add_argument("--experiments", type=int, default=DEFAULT_EXPERIMENTS,
                       help=f"Number of experiments (default: {DEFAULT_EXPERIMENTS})")
    parser.add_argument("--time-per-exp", type=int, default=DEFAULT_TIME_PER_EXP,
                       help=f"Time per experiment in seconds (default: {DEFAULT_TIME_PER_EXP})")
    parser.add_argument("--resume", action="store_true",
                       help="Resume from last experiment")
    
    args = parser.parse_args()
    
    return run_agent(args.experiments, args.time_per_exp)


if __name__ == "__main__":
    sys.exit(main())
