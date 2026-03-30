"""
Human Evaluation - Module 2.6.2

Human evaluation framework for LLM outputs.
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from enum import Enum

logger = logging.getLogger(__name__)


class RatingScale(str, Enum):
    """Rating scale options."""
    LIKERT_5 = "1-5"
    LIKERT_7 = "1-7"
    BINARY = "yes/no"
    THUMBS = "thumbs_up/thumbs_down"


@dataclass
class EvaluationTask:
    """A human evaluation task."""
    task_id: str
    prompt: str
    response: str
    reference: Optional[str] = None
    criteria: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Annotation:
    """An annotation from a human evaluator."""
    task_id: str
    evaluator_id: str
    ratings: Dict[str, int]
    comments: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    flags: List[str] = field(default_factory=list)


@dataclass
class AnnotationGuidelines:
    """Guidelines for human annotators."""
    task_description: str
    rating_definitions: Dict[str, str]
    examples: List[Dict[str, Any]]
    common_mistakes: List[str]
    
    def to_text(self) -> str:
        """Format guidelines as text."""
        text = f"# Evaluation Guidelines\n\n"
        text += f"## Task Description\n{self.task_description}\n\n"
        text += "## Rating Definitions\n"
        for criterion, definition in self.rating_definitions.items():
            text += f"- **{criterion}**: {definition}\n"
        return text


class QualityControl:
    """Quality control for human evaluations."""
    
    def __init__(
        self,
        min_agreement_threshold: float = 0.7,
        required_annotators: int = 3,
    ):
        self.min_agreement_threshold = min_agreement_threshold
        self.required_annotators = required_annotators
    
    def compute_inter_annotator_agreement(
        self,
        annotations: List[Annotation],
    ) -> Dict[str, float]:
        """Compute inter-annotator agreement."""
        # Group by task
        by_task = {}
        for ann in annotations:
            if ann.task_id not in by_task:
                by_task[ann.task_id] = []
            by_task[ann.task_id].append(ann)
        
        agreements = {}
        
        for task_id, task_anns in by_task.items():
            if len(task_anns) < 2:
                continue
            
            # Compute agreement for each criterion
            criteria = list(task_anns[0].ratings.keys())
            
            for criterion in criteria:
                ratings = [ann.ratings.get(criterion, 0) for ann in task_anns]
                
                # Simple agreement: percentage of pairs that agree within 1 point
                agreements_count = 0
                total_pairs = 0
                
                for i in range(len(ratings)):
                    for j in range(i + 1, len(ratings)):
                        if abs(ratings[i] - ratings[j]) <= 1:
                            agreements_count += 1
                        total_pairs += 1
                
                if total_pairs > 0:
                    key = f"{task_id}_{criterion}"
                    agreements[key] = agreements_count / total_pairs
        
        return agreements
    
    def filter_low_quality(
        self,
        annotations: List[Annotation],
        agreements: Dict[str, float],
    ) -> List[Annotation]:
        """Filter out low-quality annotations."""
        # Find tasks with low agreement
        low_agreement_tasks = {
            k.split('_')[0]
            for k, v in agreements.items()
            if v < self.min_agreement_threshold
        }
        
        # Filter annotations
        return [
            ann for ann in annotations
            if ann.task_id not in low_agreement_tasks
        ]


class HumanEvaluator:
    """
    Human Evaluation Manager.
    
    Manages human evaluation tasks, collects annotations,
    and computes quality metrics.
    
    Example:
        >>> evaluator = HumanEvaluator(output_dir='./eval')
        >>> evaluator.create_tasks(prompts, responses)
        >>> annotations = evaluator.collect_annotations()
    """
    
    def __init__(
        self,
        output_dir: str = './human_eval',
        rating_scale: RatingScale = RatingScale.LIKERT_5,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.rating_scale = rating_scale
        self.tasks: Dict[str, EvaluationTask] = {}
        self.annotations: List[Annotation] = []
        self.guidelines: Optional[AnnotationGuidelines] = None
        
        self.quality_control = QualityControl()
    
    def set_guidelines(self, guidelines: AnnotationGuidelines) -> None:
        """Set annotation guidelines."""
        self.guidelines = guidelines
        
        # Save guidelines
        path = self.output_dir / 'guidelines.txt'
        with open(path, 'w') as f:
            f.write(guidelines.to_text())
    
    def create_tasks(
        self,
        prompts: List[str],
        responses: List[str],
        references: Optional[List[str]] = None,
        criteria: Optional[List[str]] = None,
    ) -> List[str]:
        """Create evaluation tasks."""
        criteria = criteria or ['helpfulness', 'accuracy', 'clarity']
        
        task_ids = []
        
        for i, (prompt, response) in enumerate(zip(prompts, responses)):
            task_id = f"task_{i:04d}"
            
            task = EvaluationTask(
                task_id=task_id,
                prompt=prompt,
                response=response,
                reference=references[i] if references else None,
                criteria=criteria,
            )
            
            self.tasks[task_id] = task
            task_ids.append(task_id)
        
        # Save tasks
        self._save_tasks()
        
        logger.info(f"Created {len(task_ids)} evaluation tasks")
        
        return task_ids
    
    def _save_tasks(self) -> None:
        """Save tasks to file."""
        path = self.output_dir / 'tasks.json'
        
        data = [
            {
                'task_id': task.task_id,
                'prompt': task.prompt,
                'response': task.response,
                'reference': task.reference,
                'criteria': task.criteria,
            }
            for task in self.tasks.values()
        ]
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def add_annotation(self, annotation: Annotation) -> None:
        """Add an annotation."""
        self.annotations.append(annotation)
    
    def add_annotations_batch(
        self,
        annotations_data: List[Dict[str, Any]],
    ) -> None:
        """Add batch of annotations."""
        for data in annotations_data:
            annotation = Annotation(
                task_id=data['task_id'],
                evaluator_id=data['evaluator_id'],
                ratings=data['ratings'],
                comments=data.get('comments', ''),
                flags=data.get('flags', []),
            )
            self.add_annotation(annotation)
    
    def compute_metrics(self) -> Dict[str, Any]:
        """Compute evaluation metrics."""
        # Group annotations by task
        by_task = {}
        for ann in self.annotations:
            if ann.task_id not in by_task:
                by_task[ann.task_id] = []
            by_task[ann.task_id].append(ann)
        
        # Compute average ratings per criterion
        criteria_scores = {}
        
        for task_id, task_anns in by_task.items():
            if not task_anns:
                continue
            
            for criterion in task_anns[0].ratings.keys():
                if criterion not in criteria_scores:
                    criteria_scores[criterion] = []
                
                avg_rating = sum(ann.ratings.get(criterion, 0) for ann in task_anns) / len(task_anns)
                criteria_scores[criterion].append(avg_rating)
        
        # Average across tasks
        avg_scores = {
            criterion: sum(scores) / len(scores) if scores else 0
            for criterion, scores in criteria_scores.items()
        }
        
        # Compute inter-annotator agreement
        agreements = self.quality_control.compute_inter_annotator_agreement(self.annotations)
        avg_agreement = sum(agreements.values()) / len(agreements) if agreements else 0
        
        return {
            'num_tasks': len(self.tasks),
            'num_annotations': len(self.annotations),
            'avg_scores': avg_scores,
            'inter_annotator_agreement': avg_agreement,
        }
    
    def save_results(self) -> None:
        """Save evaluation results."""
        metrics = self.compute_metrics()
        
        path = self.output_dir / 'results.json'
        
        with open(path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Save annotations
        ann_path = self.output_dir / 'annotations.json'
        
        with open(ann_path, 'w') as f:
            json.dump([
                {
                    'task_id': ann.task_id,
                    'evaluator_id': ann.evaluator_id,
                    'ratings': ann.ratings,
                    'comments': ann.comments,
                    'timestamp': ann.timestamp,
                }
                for ann in self.annotations
            ], f, indent=2)
        
        logger.info(f"Results saved to {path}")
