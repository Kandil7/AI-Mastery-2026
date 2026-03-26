"""
Arabic LLM - Experiment Analysis

Analyze experiment results and generate insights.
Based on Karpathy's autoresearch analysis.ipynb pattern.

Usage:
    python analysis.py
    
Or convert to Jupyter notebook:
    jupyter nbconvert --to notebook analysis.py
"""

import json
import sys
from pathlib import Path
from typing import Dict, List

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from arabic_llm.agents import ExperimentTracker


def load_experiments(log_file: str = "experiments/experiment_log.jsonl") -> List[Dict]:
    """Load experiment log"""
    log_path = Path(log_file)
    
    if not log_path.exists():
        print(f"❌ Log file not found: {log_file}")
        return []
    
    experiments = []
    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                experiments.append(json.loads(line))
    
    print(f"✅ Loaded {len(experiments)} experiments")
    return experiments


def analyze_top_improvements(experiments: List[Dict], n: int = 10):
    """Analyze top N improvements"""
    print("\n" + "="*70)
    print(f"TOP {n} IMPROVEMENTS")
    print("="*70)
    
    # Filter improved experiments
    improved = [e for e in experiments if e.get('improved', False)]
    
    if not improved:
        print("No improvements found yet.")
        return
    
    # Sort by val_bpb (lower is better)
    improved.sort(key=lambda x: x['val_bpb'])
    
    for i, exp in enumerate(improved[:n], 1):
        print(f"\n{i}. Experiment #{exp['experiment']}")
        print(f"   Change: {exp['change']}")
        print(f"   val_bpb: {exp['val_bpb']:.4f}")
        print(f"   val_loss: {exp['val_loss']:.4f}")
        print(f"   Time: {exp['time_seconds']/60:.1f}m")


def analyze_by_category(experiments: List[Dict]):
    """Analyze performance by category"""
    print("\n" + "="*70)
    print("PERFORMANCE BY CATEGORY")
    print("="*70)
    
    # Extract category from change description
    def extract_category(change: str) -> str:
        change_lower = change.lower()
        
        if 'lora' in change_lower or 'rank' in change_lower:
            return 'LoRA'
        elif 'learning rate' in change_lower or 'lr' in change_lower:
            return 'Optimizer'
        elif 'warmup' in change_lower:
            return 'Schedule'
        elif 'depth' in change_lower or 'hidden' in change_lower or 'head' in change_lower:
            return 'Architecture'
        elif 'batch' in change_lower:
            return 'Batch Size'
        elif 'dropout' in change_lower:
            return 'Regularization'
        else:
            return 'Other'
    
    # Group by category
    categories = {}
    for exp in experiments:
        cat = extract_category(exp['change'])
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(exp)
    
    # Analyze each category
    for cat, cat_exps in sorted(categories.items()):
        improved_count = sum(1 for e in cat_exps if e.get('improved', False))
        success_rate = improved_count / len(cat_exps) * 100 if cat_exps else 0
        best_val_bpb = min(e['val_bpb'] for e in cat_exps)
        
        print(f"\n{cat}:")
        print(f"  Experiments: {len(cat_exps)}")
        print(f"  Success rate: {success_rate:.1f}%")
        print(f"  Best val_bpb: {best_val_bpb:.4f}")


def analyze_trends(experiments: List[Dict]):
    """Analyze trends over time"""
    print("\n" + "="*70)
    print("TRENDS OVER TIME")
    print("="*70)
    
    if len(experiments) < 10:
        print("Not enough experiments for trend analysis (need >= 10)")
        return
    
    # Split into early and recent
    early_exps = experiments[:len(experiments)//2]
    recent_exps = experiments[len(experiments)//2:]
    
    early_best = min(e['val_bpb'] for e in early_exps)
    recent_best = min(e['val_bpb'] for e in recent_exps)
    
    early_success = sum(1 for e in early_exps if e.get('improved', False)) / len(early_exps) * 100
    recent_success = sum(1 for e in recent_exps if e.get('improved', False)) / len(recent_exps) * 100
    
    print(f"\nEarly experiments (first {len(early_exps)}):")
    print(f"  Best val_bpb: {early_best:.4f}")
    print(f"  Success rate: {early_success:.1f}%")
    
    print(f"\nRecent experiments (last {len(recent_exps)}):")
    print(f"  Best val_bpb: {recent_best:.4f}")
    print(f"  Success rate: {recent_success:.1f}%")
    
    improvement = (early_best - recent_best) / early_best * 100
    print(f"\nImprovement: {improvement:.1f}%")


def generate_insights(experiments: List[Dict]):
    """Generate insights from experiments"""
    print("\n" + "="*70)
    print("INSIGHTS & RECOMMENDATIONS")
    print("="*70)
    
    if len(experiments) < 5:
        print("Not enough experiments for insights (need >= 5)")
        return
    
    # Get best experiment
    best_exp = min(experiments, key=lambda x: x['val_bpb'])
    
    # Get recent experiments
    recent_exps = experiments[-20:]
    
    # Analyze what's working
    recent_improved = [e for e in recent_exps if e.get('improved', False)]
    
    if recent_improved:
        print("\n✅ What's working (from recent improvements):")
        for exp in recent_improved[:3]:
            print(f"   - {exp['change']}")
    
    # Analyze what's not working
    recent_failed = [e for e in recent_exps if not e.get('improved', False)]
    
    if recent_failed:
        print("\n⚠️  What's not working (from recent failures):")
        failed_changes = {}
        for exp in recent_failed[:5]:
            # Extract main parameter
            change = exp['change']
            if 'depth' in change.lower():
                failed_changes['depth'] = failed_changes.get('depth', 0) + 1
            elif 'dropout' in change.lower():
                failed_changes['dropout'] = failed_changes.get('dropout', 0) + 1
        
        for param, count in failed_changes.items():
            if count >= 2:
                print(f"   - {param} changes often fail")
    
    # Recommendations
    print("\n💡 Recommendations:")
    print(f"   - Best configuration so far: {best_exp['change']}")
    print(f"   - Best val_bpb: {best_exp['val_bpb']:.4f}")
    print(f"   - Continue exploring variations of successful changes")
    
    if len(recent_improved) > len(recent_failed):
        print("   - Recent trend is positive, continue current approach")
    else:
        print("   - Recent trend is negative, consider new directions")


def create_visualization(experiments: List[Dict]):
    """Create visualization using tracker"""
    from arabic_llm.agents import ExperimentTracker
    
    tracker = ExperimentTracker()
    tracker.experiments = experiments
    tracker.update_progress_plot()
    
    print("\n📊 Visualization saved to: experiments/progress.png")


def main():
    """Main analysis function"""
    print("="*70)
    print("Arabic LLM - Experiment Analysis")
    print("="*70)
    
    # Load experiments
    experiments = load_experiments()
    
    if not experiments:
        return
    
    # Run analyses
    analyze_top_improvements(experiments, n=10)
    analyze_by_category(experiments)
    analyze_trends(experiments)
    generate_insights(experiments)
    create_visualization(experiments)
    
    print("\n" + "="*70)
    print("Analysis complete!")
    print("="*70)


if __name__ == "__main__":
    main()
