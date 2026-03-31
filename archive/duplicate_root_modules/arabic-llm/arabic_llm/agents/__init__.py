"""
Arabic LLM - Autonomous Research Agents

This subpackage contains autonomous research agents:
- Research agent for autonomous experimentation
- Experiment proposal generation
- Experiment evaluation
- Experiment tracking and visualization

Note: Agents require torch and transformers. Import will succeed but
classes will raise ImportError if torch is not properly installed.
"""

# Optional imports - agents require torch
try:
    from .researcher import (
        ResearchAgent,
        run_autonomous_research,
    )
    from .proposals import (
        ExperimentProposal,
        get_experiment_proposals,
    )
    from .evaluator import (
        ExperimentEvaluator,
        evaluate_experiment,
    )
    from .tracker import (
        ExperimentTracker,
        create_tracker,
    )
    AGENTS_AVAILABLE = True
except (ImportError, OSError) as e:
    # torch or dependencies not available - provide stub classes
    AGENTS_AVAILABLE = False
    import warnings
    warnings.warn(f"Agents disabled (requires torch): {e}")
    
    class ResearchAgent:
        def __init__(self, *args, **kwargs):
            raise ImportError("torch required for ResearchAgent. Install with: pip install torch")
    
    def run_autonomous_research(*args, **kwargs):
        raise ImportError("torch required. Install with: pip install torch")
    
    class ExperimentProposal:
        pass
    
    def get_experiment_proposals(*args, **kwargs):
        raise ImportError("torch required. Install with: pip install torch")
    
    class ExperimentEvaluator:
        pass
    
    def evaluate_experiment(*args, **kwargs):
        raise ImportError("torch required. Install with: pip install torch")
    
    class ExperimentTracker:
        pass
    
    def create_tracker(*args, **kwargs):
        raise ImportError("torch required. Install with: pip install torch")

__all__ = [
    # Researcher
    "ResearchAgent",
    "run_autonomous_research",
    # Proposals
    "ExperimentProposal",
    "get_experiment_proposals",
    # Evaluator
    "ExperimentEvaluator",
    "evaluate_experiment",
    # Tracker
    "ExperimentTracker",
    "create_tracker",
    # Status
    "AGENTS_AVAILABLE",
]
