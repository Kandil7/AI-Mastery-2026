"""
Arabic LLM - Experiment Evaluator

Evaluate experiment results and determine improvements.
"""

from typing import Dict, List, Optional
from pathlib import Path
import json


class ExperimentEvaluator:
    """
    Evaluate experiment results.
    
    Determines if an experiment improved over baseline
    and provides analysis of results.
    """
    
    def __init__(self, metric: str = "val_loss"):
        """
        Initialize evaluator.
        
        Args:
            metric: Metric to optimize (lower is better)
        """
        self.metric = metric
        self.experiment_log = []
        self._load_log()
    
    def _load_log(self):
        """Load experiment log from file"""
        log_file = Path("experiments/experiment_log.jsonl")
        
        if log_file.exists():
            with open(log_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        self.experiment_log.append(json.loads(line))
    
    def evaluate(
        self,
        val_loss: float,
        train_loss: float,
        baseline_loss: float,
    ) -> Dict:
        """
        Evaluate experiment results.
        
        Args:
            val_loss: Validation loss
            train_loss: Training loss
            baseline_loss: Baseline validation loss
            
        Returns:
            Evaluation results dictionary
        """
        improvement = baseline_loss - val_loss
        improvement_pct = (improvement / baseline_loss) * 100
        
        # Determine if improved
        improved = val_loss < baseline_loss
        
        # Check for overfitting
        overfitting = (train_loss < val_loss * 0.8)
        
        # Check for divergence
        divergence = (val_loss > baseline_loss * 1.5)
        
        return {
            "improved": improved,
            "improvement": improvement,
            "improvement_pct": improvement_pct,
            "overfitting": overfitting,
            "divergence": divergence,
            "val_loss": val_loss,
            "train_loss": train_loss,
            "baseline_loss": baseline_loss,
        }
    
    def get_best_experiment(self) -> Optional[Dict]:
        """
        Get best experiment from log.
        
        Returns:
            Best experiment dictionary or None
        """
        if not self.experiment_log:
            return None
        
        return min(self.experiment_log, key=lambda x: x.get("val_loss", float('inf')))
    
    def get_improvement_history(self) -> List[Dict]:
        """
        Get history of improvements.
        
        Returns:
            List of improvement dictionaries
        """
        improvements = []
        best_loss = float('inf')
        
        for exp in self.experiment_log:
            val_loss = exp.get("val_loss", float('inf'))
            
            if val_loss < best_loss:
                improvements.append({
                    "experiment": exp.get("experiment"),
                    "val_loss": val_loss,
                    "change": exp.get("change"),
                    "improvement": best_loss - val_loss,
                })
                best_loss = val_loss
        
        return improvements
    
    def get_statistics(self) -> Dict:
        """
        Get experiment statistics.
        
        Returns:
            Statistics dictionary
        """
        if not self.experiment_log:
            return {}
        
        val_losses = [exp.get("val_loss", 0) for exp in self.experiment_log]
        improved_count = sum(1 for exp in self.experiment_log if exp.get("improved", False))
        
        return {
            "total_experiments": len(self.experiment_log),
            "improved_count": improved_count,
            "improvement_rate": improved_count / len(self.experiment_log) if self.experiment_log else 0,
            "best_val_loss": min(val_losses) if val_losses else 0,
            "avg_val_loss": sum(val_losses) / len(val_losses) if val_losses else 0,
            "worst_val_loss": max(val_losses) if val_losses else 0,
        }
    
    def generate_report(self) -> str:
        """
        Generate experiment report.
        
        Returns:
            Report string
        """
        stats = self.get_statistics()
        best = self.get_best_experiment()
        improvements = self.get_improvement_history()
        
        report = []
        report.append("=" * 70)
        report.append("EXPERIMENT REPORT")
        report.append("=" * 70)
        report.append("")
        report.append("Statistics:")
        report.append(f"  Total experiments: {stats.get('total_experiments', 0)}")
        report.append(f"  Improved: {stats.get('improved_count', 0)} ({stats.get('improvement_rate', 0)*100:.1f}%)")
        report.append(f"  Best val_loss: {stats.get('best_val_loss', 0):.4f}")
        report.append(f"  Average val_loss: {stats.get('avg_val_loss', 0):.4f}")
        report.append("")
        
        if best:
            report.append("Best Experiment:")
            report.append(f"  Experiment #{best.get('experiment')}")
            report.append(f"  Change: {best.get('change')}")
            report.append(f"  val_loss: {best.get('val_loss', 0):.4f}")
            report.append("")
        
        if improvements:
            report.append("Improvement History:")
            for imp in improvements[:10]:  # Top 10 improvements
                report.append(f"  #{imp['experiment']}: {imp['change']}")
                report.append(f"    val_loss: {imp['val_loss']:.4f} (Δ {imp['improvement']:.4f})")
            report.append("")
        
        return "\n".join(report)


def evaluate_experiment(
    val_loss: float,
    train_loss: float,
    baseline_loss: float,
) -> Dict:
    """
    Evaluate single experiment.
    
    Args:
        val_loss: Validation loss
        train_loss: Training loss
        baseline_loss: Baseline validation loss
        
    Returns:
        Evaluation results dictionary
    """
    evaluator = ExperimentEvaluator()
    return evaluator.evaluate(val_loss, train_loss, baseline_loss)
