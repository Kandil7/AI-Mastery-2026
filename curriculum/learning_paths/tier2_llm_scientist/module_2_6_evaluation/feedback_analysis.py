"""
Feedback Analysis - Module 2.6.4

Analysis of evaluation feedback for model improvement.
"""

import json
import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


class ErrorCategory(str, Enum):
    """Error categories for feedback."""
    FACTUAL = "factual_error"
    LOGICAL = "logical_error"
    INCOMPLETE = "incomplete"
    IRRELEVANT = "irrelevant"
    UNSAFE = "unsafe_content"
    GRAMMAR = "grammar_error"
    STYLE = "style_issue"
    OTHER = "other"


@dataclass
class ErrorAnalysis:
    """Analysis of errors in model outputs."""
    category: ErrorCategory
    count: int
    percentage: float
    examples: List[Dict[str, Any]] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)


@dataclass
class PatternAnalysis:
    """Pattern analysis results."""
    pattern_type: str
    pattern: str
    frequency: int
    affected_examples: List[str] = field(default_factory=list)


class ErrorCategorizer:
    """
    Categorizes errors in model outputs.
    
    Example:
        >>> categorizer = ErrorCategorizer()
        >>> categories = categorizer.categorize(errors)
    """
    
    def __init__(self):
        self.category_keywords = {
            ErrorCategory.FACTUAL: ['incorrect', 'wrong', 'false', 'not true', 'factually'],
            ErrorCategory.LOGICAL: ['illogical', 'contradiction', 'inconsistent', 'doesn\'t follow'],
            ErrorCategory.INCOMPLETE: ['incomplete', 'missing', 'partial', 'lacks'],
            ErrorCategory.IRRELEVANT: ['irrelevant', 'off-topic', 'tangent', 'unrelated'],
            ErrorCategory.UNSAFE: ['unsafe', 'harmful', 'inappropriate', 'offensive'],
            ErrorCategory.GRAMMAR: ['grammar', 'spelling', 'typo', 'syntax'],
            ErrorCategory.STYLE: ['unclear', 'confusing', 'verbose', 'awkward'],
        }
    
    def categorize(
        self,
        feedback_items: List[Dict[str, Any]],
    ) -> Dict[ErrorCategory, ErrorAnalysis]:
        """Categorize feedback items."""
        category_counts = defaultdict(int)
        category_examples = defaultdict(list)
        
        for item in feedback_items:
            feedback_text = item.get('feedback', '').lower()
            
            categorized = False
            
            for category, keywords in self.category_keywords.items():
                if any(kw in feedback_text for kw in keywords):
                    category_counts[category] += 1
                    category_examples[category].append(item)
                    categorized = True
                    break
            
            if not categorized:
                category_counts[ErrorCategory.OTHER] += 1
                category_examples[ErrorCategory.OTHER].append(item)
        
        total = len(feedback_items)
        
        analyses = {}
        for category, count in category_counts.items():
            analyses[category] = ErrorAnalysis(
                category=category,
                count=count,
                percentage=count / total if total > 0 else 0,
                examples=category_examples[category][:5],
            )
        
        return analyses


class PatternDetector:
    """
    Detects patterns in model errors.
    
    Example:
        >>> detector = PatternDetector()
        >>> patterns = detector.detect(errors)
    """
    
    def __init__(self, min_frequency: int = 3):
        self.min_frequency = min_frequency
    
    def detect(
        self,
        errors: List[Dict[str, Any]],
    ) -> List[PatternAnalysis]:
        """Detect patterns in errors."""
        patterns = []
        
        # Analyze error types by prompt category
        category_errors = defaultdict(list)
        for error in errors:
            category = error.get('category', 'unknown')
            category_errors[category].append(error)
        
        for category, cat_errors in category_errors.items():
            if len(cat_errors) >= self.min_frequency:
                patterns.append(PatternAnalysis(
                    pattern_type='category_concentration',
                    pattern=f"High error rate in {category}",
                    frequency=len(cat_errors),
                    affected_examples=[e.get('prompt', '')[:50] for e in cat_errors[:5]],
                ))
        
        # Analyze error types by error category
        error_type_counts = Counter(
            e.get('error_type', 'unknown') for e in errors
        )
        
        for error_type, count in error_type_counts.most_common():
            if count >= self.min_frequency:
                patterns.append(PatternAnalysis(
                    pattern_type='error_type',
                    pattern=f"Frequent {error_type} errors",
                    frequency=count,
                ))
        
        return patterns


class ImprovementSuggester:
    """
    Suggests improvements based on error analysis.
    
    Example:
        >>> suggester = ImprovementSuggester()
        >>> suggestions = suggester.suggest(error_analysis)
    """
    
    def __init__(self):
        self.suggestions = {
            ErrorCategory.FACTUAL: [
                "Improve fact-checking in training data",
                "Add retrieval-augmented generation",
                "Include knowledge verification step",
            ],
            ErrorCategory.LOGICAL: [
                "Add chain-of-thought training",
                "Include logical reasoning examples",
                "Implement self-consistency checking",
            ],
            ErrorCategory.INCOMPLETE: [
                "Train on more comprehensive examples",
                "Add explicit completeness criteria",
                "Implement multi-step verification",
            ],
            ErrorCategory.IRRELEVANT: [
                "Improve prompt understanding",
                "Add relevance filtering",
                "Train on focused responses",
            ],
            ErrorCategory.UNSAFE: [
                "Strengthen safety fine-tuning",
                "Add content filtering",
                "Implement safety classifiers",
            ],
            ErrorCategory.GRAMMAR: [
                "Improve language modeling",
                "Add grammar correction step",
                "Train on higher quality text",
            ],
            ErrorCategory.STYLE: [
                "Train on clearer examples",
                "Add style guidelines",
                "Implement response refinement",
            ],
        }
    
    def suggest(
        self,
        error_analysis: Dict[ErrorCategory, ErrorAnalysis],
    ) -> List[Dict[str, Any]]:
        """Generate improvement suggestions."""
        suggestions = []
        
        # Sort by percentage
        sorted_categories = sorted(
            error_analysis.items(),
            key=lambda x: x[1].percentage,
            reverse=True,
        )
        
        for category, analysis in sorted_categories:
            if analysis.percentage > 0.1:  # Only for significant issues
                category_suggestions = self.suggestions.get(category, [])
                
                suggestions.append({
                    'category': category.value,
                    'percentage': analysis.percentage,
                    'count': analysis.count,
                    'suggestions': category_suggestions[:3],
                    'priority': 'high' if analysis.percentage > 0.3 else 'medium',
                })
        
        return suggestions


class FeedbackAnalyzer:
    """
    Complete Feedback Analysis System.
    
    Combines error categorization, pattern detection,
    and improvement suggestions.
    
    Example:
        >>> analyzer = FeedbackAnalyzer()
        >>> report = analyzer.analyze(feedback_data)
    """
    
    def __init__(self, output_dir: str = './feedback_analysis'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.categorizer = ErrorCategorizer()
        self.detector = PatternDetector()
        self.suggester = ImprovementSuggester()
    
    def analyze(
        self,
        feedback_data: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Analyze feedback data.
        
        Args:
            feedback_data: List of feedback items
        
        Returns:
            Analysis report
        """
        logger.info(f"Analyzing {len(feedback_data)} feedback items...")
        
        # Categorize errors
        error_analysis = self.categorizer.categorize(feedback_data)
        
        # Detect patterns
        errors_list = [
            {
                'category': item.get('category', 'unknown'),
                'error_type': item.get('error_type', 'unknown'),
                'prompt': item.get('prompt', ''),
            }
            for item in feedback_data
            if item.get('is_error', False)
        ]
        patterns = self.detector.detect(errors_list)
        
        # Generate suggestions
        suggestions = self.suggester.suggest(error_analysis)
        
        # Compile report
        report = {
            'summary': {
                'total_feedback': len(feedback_data),
                'total_errors': len(errors_list),
                'error_rate': len(errors_list) / len(feedback_data) if feedback_data else 0,
            },
            'error_categories': {
                cat.value: {
                    'count': analysis.count,
                    'percentage': analysis.percentage,
                }
                for cat, analysis in error_analysis.items()
            },
            'patterns': [
                {
                    'type': p.pattern_type,
                    'pattern': p.pattern,
                    'frequency': p.frequency,
                }
                for p in patterns
            ],
            'improvement_suggestions': suggestions,
        }
        
        # Save report
        self._save_report(report)
        
        return report
    
    def _save_report(self, report: Dict[str, Any]) -> None:
        """Save analysis report."""
        path = self.output_dir / 'analysis_report.json'
        
        with open(path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Report saved to {path}")
    
    def generate_summary(self, report: Dict[str, Any]) -> str:
        """Generate human-readable summary."""
        summary = "# Feedback Analysis Summary\n\n"
        
        summary += "## Overview\n"
        summary += f"- Total feedback items: {report['summary']['total_feedback']}\n"
        summary += f"- Error rate: {report['summary']['error_rate']:.2%}\n\n"
        
        summary += "## Error Categories\n"
        for cat, data in report['error_categories'].items():
            summary += f"- {cat}: {data['count']} ({data['percentage']:.1%})\n"
        
        summary += "\n## Top Patterns\n"
        for pattern in report['patterns'][:5]:
            summary += f"- {pattern['pattern']} (frequency: {pattern['frequency']})\n"
        
        summary += "\n## Improvement Suggestions\n"
        for suggestion in report['improvement_suggestions'][:5]:
            summary += f"### {suggestion['category']} ({suggestion['priority']} priority)\n"
            for s in suggestion['suggestions']:
                summary += f"- {s}\n"
        
        return summary
