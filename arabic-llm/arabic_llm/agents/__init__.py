"""
Arabic LLM - Autonomous Research Agents

This subpackage contains autonomous research agents:
- Research agent for autonomous experimentation
- Experiment proposal generation
- Experiment evaluation
- Experiment tracking and visualization
"""

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
]
