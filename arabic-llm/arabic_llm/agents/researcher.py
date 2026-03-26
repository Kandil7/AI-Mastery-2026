"""
Arabic LLM - Research Agent

Autonomous research agent for hyperparameter optimization.
Based on Karpathy's autoresearch pattern.
"""

import os
import sys
import json
import time
import shutil
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from datetime import datetime

from .proposals import ExperimentProposal, get_experiment_proposals
from .evaluator import ExperimentEvaluator


class ResearchAgent:
    """
    Autonomous research agent for Arabic LLM hyperparameter optimization.
    
    The agent autonomously:
    1. Proposes experiments (LoRA rank, learning rate, etc.)
    2. Modifies training configuration
    3. Runs training
    4. Evaluates results
    5. Keeps/discard changes based on val_loss
    """
    
    def __init__(
        self,
        train_file: str = "train.py",
        experiments_dir: str = "experiments",
        time_per_exp: int = 300,
    ):
        """
        Initialize research agent.
        
        Args:
            train_file: Path to training script
            experiments_dir: Directory for experiment logs
            time_per_exp: Time budget per experiment (seconds)
        """
        self.train_file = Path(train_file)
        self.experiments_dir = Path(experiments_dir)
        self.time_per_exp = time_per_exp
        
        # Create experiments directory
        self.experiments_dir.mkdir(parents=True, exist_ok=True)
        
        # Backup file
        self.train_backup = self.train_file.with_suffix('.py.backup')
        
        # Evaluator
        self.evaluator = ExperimentEvaluator()
        
        # State
        self.best_loss = self._load_best_loss()
        self.experiment_count = self._load_experiment_count()
    
    def _load_best_loss(self) -> float:
        """Load best validation loss from file"""
        best_loss_file = self.experiments_dir / "best_loss.txt"
        
        if best_loss_file.exists():
            with open(best_loss_file, "r") as f:
                return float(f.read().strip())
        
        return float('inf')
    
    def _save_best_loss(self, loss: float):
        """Save best validation loss to file"""
        best_loss_file = self.experiments_dir / "best_loss.txt"
        
        with open(best_loss_file, "w") as f:
            f.write(str(loss))
    
    def _load_experiment_count(self) -> int:
        """Load experiment count from log"""
        log_file = self.experiments_dir / "experiment_log.jsonl"
        
        if not log_file.exists():
            return 0
        
        count = 0
        with open(log_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    count += 1
        
        return count
    
    def run(self, num_experiments: int = 100):
        """
        Run autonomous research loop.
        
        Args:
            num_experiments: Number of experiments to run
        """
        print("=" * 70)
        print("Arabic LLM Autonomous Research Agent")
        print("=" * 70)
        print(f"Experiments: {num_experiments}")
        print(f"Time per experiment: {self.time_per_exp}s ({self.time_per_exp/60:.1f}m)")
        print(f"Current best val_loss: {self.best_loss:.4f}")
        print("=" * 70)
        
        # Backup train.py
        if self.train_file.exists():
            shutil.copy(self.train_file, self.train_backup)
            print(f"✓ Backed up {self.train_file}")
        
        # Get experiment proposals
        proposals = get_experiment_proposals()
        print(f"✓ Loaded {len(proposals)} experiment proposals")
        
        # Run experiments
        start_time = time.time()
        improved_count = 0
        
        for i, proposal in enumerate(proposals[:num_experiments]):
            exp_start = time.time()
            
            print(f"\n{'='*70}")
            print(f"Experiment {i+1}/{num_experiments}")
            print(f"Proposal: {proposal.change}")
            print(f"{'='*70}")
            
            # Apply modifications
            proposal.apply(self.train_file)
            print(f"✓ Applied modifications")
            
            # Run training
            val_loss, train_loss, success = self._run_training()
            
            if success and val_loss is not None:
                # Evaluate
                improved = val_loss < self.best_loss
                
                if improved:
                    print(f"✅ IMPROVED! val_loss: {self.best_loss:.4f} → {val_loss:.4f}")
                    self.best_loss = val_loss
                    self._save_best_loss(self.best_loss)
                    improved_count += 1
                else:
                    print(f"❌ Not improved. val_loss: {val_loss:.4f} (best: {self.best_loss:.4f})")
                    # Revert
                    shutil.copy(self.train_backup, self.train_file)
                    print(f"✓ Reverted to baseline")
                
                # Log
                elapsed = time.time() - exp_start
                self._log_experiment(proposal, val_loss, train_loss or 0.0, improved, elapsed)
            else:
                print(f"❌ Training failed")
                # Revert
                shutil.copy(self.train_backup, self.train_file)
            
            # Progress
            total_elapsed = time.time() - start_time
            avg_time = total_elapsed / (i + 1)
            remaining = (num_experiments - i - 1) * avg_time
            
            print(f"\n📊 Progress:")
            print(f"  Completed: {i+1}/{num_experiments}")
            print(f"  Improved: {improved_count} ({improved_count/(i+1)*100:.1f}%)")
            print(f"  Best val_loss: {self.best_loss:.4f}")
            print(f"  Remaining: {remaining/3600:.1f}h")
        
        # Final summary
        total_time = time.time() - start_time
        
        print(f"\n{'='*70}")
        print("Research Complete")
        print(f"{'='*70}")
        print(f"Total experiments: {num_experiments}")
        print(f"Improved: {improved_count} ({improved_count/num_experiments*100:.1f}%)")
        print(f"Final best val_loss: {self.best_loss:.4f}")
        print(f"Total time: {total_time/3600:.1f}h")
        print(f"{'='*70}")
    
    def _run_training(self) -> Tuple[Optional[float], Optional[float], bool]:
        """
        Run training script.
        
        Returns:
            (val_loss, train_loss, success)
        """
        import subprocess
        
        try:
            result = subprocess.run(
                [sys.executable, str(self.train_file)],
                capture_output=True,
                text=True,
                timeout=self.time_per_exp + 60,
                cwd=self.train_file.parent
            )
            
            # Parse output
            val_loss = None
            train_loss = None
            
            output = result.stdout + result.stderr
            
            for line in output.split('\n'):
                if 'val_loss:' in line.lower():
                    import re
                    match = re.search(r'val_loss:\s*([\d.]+)', line, re.IGNORECASE)
                    if match:
                        val_loss = float(match.group(1))
                
                if 'train_loss:' in line.lower():
                    import re
                    match = re.search(r'train_loss:\s*([\d.]+)', line, re.IGNORECASE)
                    if match:
                        train_loss = float(match.group(1))
            
            success = result.returncode == 0
            return val_loss, train_loss, success
            
        except subprocess.TimeoutExpired:
            print(f"⏰ Training timed out")
            return None, None, False
        except Exception as e:
            print(f"❌ Error: {e}")
            return None, None, False
    
    def _log_experiment(
        self,
        proposal: ExperimentProposal,
        val_loss: float,
        train_loss: float,
        improved: bool,
        time_seconds: float,
    ):
        """Log experiment result"""
        log_entry = {
            "experiment": self.experiment_count + 1,
            "change": proposal.change,
            "val_loss": val_loss,
            "train_loss": train_loss,
            "improved": improved,
            "time_seconds": time_seconds,
            "timestamp": datetime.now().isoformat(),
        }
        
        log_file = self.experiments_dir / "experiment_log.jsonl"
        
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
        
        self.experiment_count += 1


def run_autonomous_research(
    num_experiments: int = 100,
    time_per_exp: int = 300,
    train_file: str = "train.py",
):
    """
    Run autonomous research.
    
    Args:
        num_experiments: Number of experiments
        time_per_exp: Time per experiment (seconds)
        train_file: Path to training script
    """
    agent = ResearchAgent(
        train_file=train_file,
        experiments_dir="experiments",
        time_per_exp=time_per_exp,
    )
    
    agent.run(num_experiments=num_experiments)
