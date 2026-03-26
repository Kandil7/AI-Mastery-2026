"""
Arabic LLM - Experiment Tracker and Visualization

Tracks experiment progress and generates visualizations.
Based on Karpathy's autoresearch progress.png pattern.

Usage:
    from arabic_llm.agents import ExperimentTracker
    
    tracker = ExperimentTracker()
    tracker.log_experiment(exp_result)
    tracker.update_progress_plot()
"""

import json
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime


class ExperimentTracker:
    """Track and visualize experiment progress"""
    
    def __init__(self, log_dir: str = "experiments"):
        """
        Initialize experiment tracker.
        
        Args:
            log_dir: Directory for experiment logs
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_file = self.log_dir / "experiment_log.jsonl"
        self.progress_png = self.log_dir / "progress.png"
        
        self.experiments = []
        self._load_existing_logs()
    
    def _load_existing_logs(self):
        """Load existing experiment logs"""
        if self.log_file.exists():
            with open(self.log_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        self.experiments.append(json.loads(line))
    
    def log_experiment(self, exp_result: Dict):
        """
        Log experiment result.
        
        Args:
            exp_result: Experiment result dictionary
        """
        self.experiments.append(exp_result)
        self._save_log()
        self.update_progress_plot()
    
    def _save_log(self):
        """Save experiment log to file"""
        with open(self.log_file, 'w', encoding='utf-8') as f:
            for exp in self.experiments:
                f.write(json.dumps(exp, ensure_ascii=False) + '\n')
    
    def update_progress_plot(self):
        """Generate progress.png showing val_bpb over time"""
        try:
            import matplotlib.pyplot as plt
            import numpy as np
        except ImportError:
            print("⚠️  matplotlib not installed. Install with: pip install matplotlib")
            return
        
        if len(self.experiments) < 2:
            return
        
        plt.style.use('seaborn-v0_8-darkgrid')
        fig, axes = plt.subplots(2, 2, figsize=(14, 8))
        
        # Plot 1: val_bpb over experiments
        ax1 = axes[0, 0]
        exp_nums = [e['experiment'] for e in self.experiments]
        val_bpbs = [e['val_bpb'] for e in self.experiments]
        
        ax1.plot(exp_nums, val_bpbs, 'b-o', linewidth=2, markersize=4)
        ax1.axhline(y=min(val_bpbs), color='g', linestyle='--', alpha=0.7, 
                   label=f'Best: {min(val_bpbs):.4f}')
        ax1.set_xlabel('Experiment')
        ax1.set_ylabel('val_bpb (lower is better)')
        ax1.set_title('Progress Over Time')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: val_loss over experiments
        ax2 = axes[0, 1]
        val_losses = [e['val_loss'] for e in self.experiments]
        
        ax2.plot(exp_nums, val_losses, 'r-o', linewidth=2, markersize=4)
        ax2.axhline(y=min(val_losses), color='g', linestyle='--', alpha=0.7,
                   label=f'Best: {min(val_losses):.4f}')
        ax2.set_xlabel('Experiment')
        ax2.set_ylabel('val_loss')
        ax2.set_title('Validation Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Improvement distribution
        ax3 = axes[1, 0]
        improvements = [e.get('improvement', 0) for e in self.experiments if e.get('improved', False)]
        
        if improvements:
            ax3.hist(improvements, bins=20, color='green', alpha=0.7, edgecolor='black')
            ax3.set_xlabel('Improvement (Δ val_bpb)')
            ax3.set_ylabel('Count')
            ax3.set_title('Improvement Distribution')
            ax3.axvline(x=0, color='red', linestyle='-', alpha=0.5)
        else:
            ax3.text(0.5, 0.5, 'No improvements yet', ha='center', va='center', fontsize=14)
            ax3.set_xlim(0, 1)
            ax3.set_ylim(0, 1)
            ax3.set_title('Improvement Distribution')
        
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Success rate over time
        ax4 = axes[1, 1]
        success_rates = []
        for i in range(1, len(self.experiments) + 1):
            subset = self.experiments[:i]
            improved_count = sum(1 for e in subset if e.get('improved', False))
            success_rates.append(improved_count / i * 100)
        
        ax4.plot(exp_nums, success_rates, 'purple', linewidth=2)
        ax4.set_xlabel('Experiment')
        ax4.set_ylabel('Success Rate (%)')
        ax4.set_title('Cumulative Success Rate')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.progress_png, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Progress plot saved to: {self.progress_png}")
    
    def get_statistics(self) -> Dict:
        """
        Get experiment statistics.
        
        Returns:
            Dictionary of statistics
        """
        if not self.experiments:
            return {}
        
        val_bpbs = [e['val_bpb'] for e in self.experiments]
        val_losses = [e['val_loss'] for e in self.experiments]
        improved_count = sum(1 for e in self.experiments if e.get('improved', False))
        
        # Get best experiment
        best_exp = min(self.experiments, key=lambda x: x['val_bpb'])
        
        # Get latest experiment
        latest_exp = self.experiments[-1]
        
        # Calculate trend (last 10 experiments)
        recent_exps = self.experiments[-10:]
        recent_improved = sum(1 for e in recent_exps if e.get('improved', False))
        
        stats = {
            'total_experiments': len(self.experiments),
            'improved_count': improved_count,
            'success_rate': improved_count / len(self.experiments) * 100,
            'best_val_bpb': min(val_bpbs),
            'best_val_loss': min(val_losses),
            'latest_val_bpb': latest_exp['val_bpb'],
            'latest_val_loss': latest_exp['val_loss'],
            'best_experiment': best_exp,
            'recent_success_rate': recent_improved / len(recent_exps) * 100 if recent_exps else 0,
            'avg_time_per_exp': sum(e['time_seconds'] for e in self.experiments) / len(self.experiments),
        }
        
        return stats
    
    def print_summary(self):
        """Print summary of experiment progress"""
        stats = self.get_statistics()
        
        if not stats:
            print("No experiments logged yet.")
            return
        
        print("\n" + "="*70)
        print("EXPERIMENT SUMMARY")
        print("="*70)
        print(f"Total experiments: {stats['total_experiments']}")
        print(f"Improved: {stats['improved_count']} ({stats['success_rate']:.1f}%)")
        print(f"Best val_bpb: {stats['best_val_bpb']:.4f}")
        print(f"Best val_loss: {stats['best_val_loss']:.4f}")
        print(f"Latest val_bpb: {stats['latest_val_bpb']:.4f}")
        print(f"Recent success rate (last 10): {stats['recent_success_rate']:.1f}%")
        print(f"Avg time per experiment: {stats['avg_time_per_exp']/60:.1f}m")
        
        best = stats['best_experiment']
        print(f"\nBest experiment: #{best['experiment']}")
        print(f"  Change: {best['change']}")
        print(f"  val_bpb: {best['val_bpb']:.4f}")
        print(f"  val_loss: {best['val_loss']:.4f}")
        print("="*70)
    
    def get_top_improvements(self, n: int = 10) -> List[Dict]:
        """
        Get top N improvements.
        
        Args:
            n: Number of top improvements to return
            
        Returns:
            List of top improvement experiments
        """
        improved = [e for e in self.experiments if e.get('improved', False)]
        
        # Calculate improvement magnitude
        for i, exp in enumerate(improved):
            if i > 0:
                exp['improvement_magnitude'] = improved[i-1]['val_bpb'] - exp['val_bpb']
            else:
                exp['improvement_magnitude'] = 0
        
        # Sort by improvement magnitude
        improved.sort(key=lambda x: x['val_bpb'])
        
        return improved[:n]


def create_tracker() -> ExperimentTracker:
    """
    Create and return experiment tracker.
    
    Returns:
        ExperimentTracker instance
    """
    return ExperimentTracker()
