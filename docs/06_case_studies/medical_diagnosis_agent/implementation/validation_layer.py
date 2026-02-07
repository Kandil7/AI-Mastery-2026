"""
Medical Diagnosis Agent - Validation Layer
===========================================
Safety validation and output verification.

Features:
- Confidence calibration
- Contraindication checking
- Output validation
- Escalation logic

Author: AI-Mastery-2026
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


# ============================================================
# VALIDATION RESULTS
# ============================================================

class ValidationStatus(Enum):
    """Validation status codes."""
    PASSED = "passed"
    WARNING = "warning"
    FAILED = "failed"
    ESCALATED = "escalated"


@dataclass
class ValidationResult:
    """Result of validation check."""
    check_name: str
    status: ValidationStatus
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "check": self.check_name,
            "status": self.status.value,
            "message": self.message,
            "details": self.details
        }


@dataclass
class ValidationReport:
    """Complete validation report."""
    passed: bool
    checks: List[ValidationResult]
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "checks": [c.to_dict() for c in self.checks],
            "timestamp": self.timestamp
        }


# ============================================================
# CONFIDENCE CALIBRATION
# ============================================================

class ConfidenceCalibrator:
    """
    Calibrate diagnosis confidence scores.
    
    Applies adjustments based on:
    - Number of symptoms
    - Symptom severity
    - Prior probabilities
    - Model uncertainty
    """
    
    def __init__(
        self,
        min_symptoms_for_confidence: int = 2,
        severity_boost: float = 0.1,
        max_confidence: float = 0.85  # Never 100% confident
    ):
        self.min_symptoms = min_symptoms_for_confidence
        self.severity_boost = severity_boost
        self.max_confidence = max_confidence
    
    def calibrate(
        self,
        raw_probability: float,
        n_symptoms: int,
        has_severe_symptoms: bool = False,
        n_supporting: int = 0,
        n_missing: int = 0
    ) -> Tuple[float, float, float]:
        """
        Calibrate probability to confidence interval.
        
        Returns:
            Tuple of (point_estimate, lower_bound, upper_bound)
        """
        # Adjust for symptom count
        if n_symptoms < self.min_symptoms:
            uncertainty_factor = 1.5
        else:
            uncertainty_factor = 1.0
        
        # Adjust for severity
        if has_severe_symptoms:
            severity_adjustment = self.severity_boost
        else:
            severity_adjustment = 0.0
        
        # Adjust for missing symptoms
        missing_penalty = 0.05 * n_missing
        
        # Calculate adjusted probability
        adjusted = raw_probability + severity_adjustment - missing_penalty
        adjusted = max(0.0, min(self.max_confidence, adjusted))
        
        # Calculate confidence interval
        base_margin = 0.15 * uncertainty_factor
        lower = max(0.0, adjusted - base_margin)
        upper = min(self.max_confidence, adjusted + base_margin / 2)
        
        return adjusted, lower, upper


# ============================================================
# CONTRAINDICATION CHECKER
# ============================================================

@dataclass
class Contraindication:
    """A contraindication warning."""
    condition1: str
    condition2: str
    severity: str
    description: str


class ContraindicationChecker:
    """
    Check for dangerous condition combinations.
    
    Identifies:
    - Conflicting conditions
    - Drug interaction risks
    - Red flag combinations
    """
    
    # Known dangerous combinations
    CONTRAINDICATIONS = [
        Contraindication(
            condition1="heart_attack",
            condition2="gastroenteritis",
            severity="high",
            description="Chest pain with GI symptoms may indicate atypical MI presentation"
        ),
    ]
    
    # Red flag symptom combinations
    RED_FLAGS = {
        frozenset(["chest_pain", "shortness_of_breath"]): "Possible cardiac emergency",
        frozenset(["fever", "headache", "stiff_neck"]): "Possible meningitis",
        frozenset(["sudden_headache", "vision_changes"]): "Possible stroke",
    }
    
    def check(
        self,
        conditions: List[str],
        symptoms: List[str]
    ) -> List[ValidationResult]:
        """Check for contraindications and red flags."""
        results = []
        
        # Check condition combinations
        condition_set = set(conditions)
        for contra in self.CONTRAINDICATIONS:
            if contra.condition1 in condition_set and contra.condition2 in condition_set:
                results.append(ValidationResult(
                    check_name="contraindication",
                    status=ValidationStatus.WARNING,
                    message=contra.description,
                    details={
                        "conditions": [contra.condition1, contra.condition2],
                        "severity": contra.severity
                    }
                ))
        
        # Check red flag symptoms
        symptom_set = set(symptoms)
        for red_flag_combo, message in self.RED_FLAGS.items():
            if red_flag_combo.issubset(symptom_set):
                results.append(ValidationResult(
                    check_name="red_flag",
                    status=ValidationStatus.ESCALATED,
                    message=message,
                    details={"symptoms": list(red_flag_combo)}
                ))
        
        return results


# ============================================================
# OUTPUT VALIDATOR
# ============================================================

class OutputValidator:
    """
    Validate diagnostic output before delivery.
    
    Checks:
    - Disclaimer present
    - Confidence bounds reasonable
    - No unauthorized diagnoses
    - Appropriate escalation
    """
    
    REQUIRED_DISCLAIMER_TERMS = [
        "not constitute medical advice",
        "consult",
        "healthcare professional"
    ]
    
    UNAUTHORIZED_CONDITIONS = [
        "cancer",
        "tumor",
        "terminal"
    ]
    
    def __init__(
        self,
        max_conditions: int = 5,
        min_confidence_for_recommendation: float = 0.2
    ):
        self.max_conditions = max_conditions
        self.min_confidence = min_confidence_for_recommendation
    
    def validate_disclaimer(self, disclaimer: str) -> ValidationResult:
        """Validate disclaimer presence."""
        disclaimer_lower = disclaimer.lower()
        
        missing_terms = [
            term for term in self.REQUIRED_DISCLAIMER_TERMS
            if term.lower() not in disclaimer_lower
        ]
        
        if missing_terms:
            return ValidationResult(
                check_name="disclaimer",
                status=ValidationStatus.FAILED,
                message="Disclaimer missing required terms",
                details={"missing_terms": missing_terms}
            )
        
        return ValidationResult(
            check_name="disclaimer",
            status=ValidationStatus.PASSED,
            message="Disclaimer validated"
        )
    
    def validate_conditions(
        self,
        conditions: List[str]
    ) -> ValidationResult:
        """Validate conditions list."""
        # Check for unauthorized conditions
        unauthorized = [
            c for c in conditions
            if any(u in c.lower() for u in self.UNAUTHORIZED_CONDITIONS)
        ]
        
        if unauthorized:
            return ValidationResult(
                check_name="unauthorized_conditions",
                status=ValidationStatus.FAILED,
                message="Output contains conditions requiring professional diagnosis",
                details={"conditions": unauthorized}
            )
        
        # Check count
        if len(conditions) > self.max_conditions:
            return ValidationResult(
                check_name="condition_count",
                status=ValidationStatus.WARNING,
                message=f"Too many conditions ({len(conditions)} > {self.max_conditions})",
                details={"count": len(conditions)}
            )
        
        return ValidationResult(
            check_name="conditions",
            status=ValidationStatus.PASSED,
            message="Conditions validated"
        )
    
    def validate_confidence(
        self,
        confidence_range: Tuple[float, float]
    ) -> ValidationResult:
        """Validate confidence bounds."""
        low, high = confidence_range
        
        if high > 0.95:
            return ValidationResult(
                check_name="confidence",
                status=ValidationStatus.FAILED,
                message="Confidence too high - must express uncertainty",
                details={"range": confidence_range}
            )
        
        if high - low < 0.05:
            return ValidationResult(
                check_name="confidence",
                status=ValidationStatus.WARNING,
                message="Confidence interval too narrow",
                details={"range": confidence_range}
            )
        
        return ValidationResult(
            check_name="confidence",
            status=ValidationStatus.PASSED,
            message="Confidence bounds validated"
        )


# ============================================================
# ESCALATION HANDLER
# ============================================================

class EscalationHandler:
    """
    Handle escalation decisions.
    
    Determines when to escalate to human review or emergency.
    """
    
    # Automatic escalation triggers
    EMERGENCY_SYMPTOMS = {"chest_pain", "difficulty_breathing", "loss_of_consciousness"}
    ESCALATION_TRIGGERS = {"confusion", "high_fever", "severe_pain"}
    
    def __init__(
        self,
        confidence_threshold: float = 0.5,
        escalation_callback: Optional[callable] = None
    ):
        self.confidence_threshold = confidence_threshold
        self.escalation_callback = escalation_callback
    
    def should_escalate(
        self,
        symptoms: List[str],
        max_confidence: float,
        urgency: str
    ) -> Tuple[bool, str]:
        """
        Determine if escalation is needed.
        
        Returns:
            Tuple of (should_escalate, reason)
        """
        symptom_set = set(symptoms)
        
        # Emergency symptoms
        if symptom_set & self.EMERGENCY_SYMPTOMS:
            return True, "Emergency symptoms detected"
        
        # Escalation triggers
        if symptom_set & self.ESCALATION_TRIGGERS:
            return True, "Symptoms require professional evaluation"
        
        # Low confidence with concerning symptoms
        if max_confidence < self.confidence_threshold and len(symptoms) >= 3:
            return True, "Insufficient confidence for reliable assessment"
        
        # Urgent conditions
        if urgency in ["urgent", "emergency"]:
            return True, f"Urgency level: {urgency}"
        
        return False, ""
    
    def escalate(
        self,
        reason: str,
        context: Dict[str, Any]
    ):
        """Execute escalation."""
        logger.warning(f"ESCALATION: {reason}")
        
        if self.escalation_callback:
            self.escalation_callback(reason, context)


# ============================================================
# COMPLETE VALIDATION PIPELINE
# ============================================================

class ValidationPipeline:
    """
    Complete validation pipeline for diagnostic output.
    
    Runs all validation checks and generates report.
    """
    
    def __init__(self):
        self.calibrator = ConfidenceCalibrator()
        self.contraindication_checker = ContraindicationChecker()
        self.output_validator = OutputValidator()
        self.escalation_handler = EscalationHandler()
    
    def validate(
        self,
        diagnosis_result: Any,  # DiagnosisResult
    ) -> ValidationReport:
        """
        Run complete validation.
        
        Args:
            diagnosis_result: Result from DiagnosticEngine
        
        Returns:
            ValidationReport
        """
        checks = []
        
        # Validate disclaimer
        checks.append(
            self.output_validator.validate_disclaimer(diagnosis_result.disclaimer)
        )
        
        # Validate conditions
        condition_names = [
            c.condition.name for c in diagnosis_result.differential_diagnosis
        ]
        checks.append(
            self.output_validator.validate_conditions(condition_names)
        )
        
        # Validate confidence
        checks.append(
            self.output_validator.validate_confidence(
                diagnosis_result.confidence_range
            )
        )
        
        # Check contraindications
        symptom_names = [s.name for s in diagnosis_result.extracted_symptoms]
        contra_results = self.contraindication_checker.check(
            condition_names, symptom_names
        )
        checks.extend(contra_results)
        
        # Check escalation
        should_escalate, reason = self.escalation_handler.should_escalate(
            symptom_names,
            diagnosis_result.confidence_range[1],
            diagnosis_result.urgency.value
        )
        
        if should_escalate:
            checks.append(ValidationResult(
                check_name="escalation",
                status=ValidationStatus.ESCALATED,
                message=reason
            ))
        
        # Determine overall pass/fail
        failed = any(c.status == ValidationStatus.FAILED for c in checks)
        
        return ValidationReport(
            passed=not failed,
            checks=checks
        )


# ============================================================
# EXPORTS
# ============================================================

__all__ = [
    'ValidationStatus', 'ValidationResult', 'ValidationReport',
    'ConfidenceCalibrator', 'ContraindicationChecker',
    'OutputValidator', 'EscalationHandler', 'ValidationPipeline',
]
