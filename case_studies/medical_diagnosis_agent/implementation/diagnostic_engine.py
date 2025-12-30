"""
Medical Diagnosis Agent - Diagnostic Engine
============================================
Core diagnostic reasoning engine with evidence-based analysis.

Features:
- Symptom extraction and analysis
- Differential diagnosis generation
- Evidence scoring and ranking
- Uncertainty quantification

Author: AI-Mastery-2026
"""

import logging
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import re

logger = logging.getLogger(__name__)


# ============================================================
# DATA STRUCTURES
# ============================================================

class Severity(Enum):
    """Symptom/condition severity levels."""
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"
    CRITICAL = "critical"


class Urgency(Enum):
    """Response urgency levels."""
    ROUTINE = "routine"
    SOON = "soon"
    URGENT = "urgent"
    EMERGENCY = "emergency"


@dataclass
class Symptom:
    """Extracted symptom with attributes."""
    name: str
    severity: Severity = Severity.MILD
    duration: Optional[str] = None
    location: Optional[str] = None
    frequency: Optional[str] = None
    confidence: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "severity": self.severity.value,
            "duration": self.duration,
            "location": self.location,
            "confidence": self.confidence
        }


@dataclass
class Condition:
    """A potential medical condition."""
    name: str
    icd_code: Optional[str] = None
    description: str = ""
    severity: Severity = Severity.MILD
    urgency: Urgency = Urgency.ROUTINE
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "icd_code": self.icd_code,
            "description": self.description,
            "severity": self.severity.value,
            "urgency": self.urgency.value
        }


@dataclass
class DiagnosisCandidate:
    """A candidate diagnosis with supporting evidence."""
    condition: Condition
    probability: float  # 0.0 to 1.0
    supporting_symptoms: List[str] = field(default_factory=list)
    contradicting_symptoms: List[str] = field(default_factory=list)
    missing_symptoms: List[str] = field(default_factory=list)
    evidence_score: float = 0.0
    reasoning: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "condition": self.condition.to_dict(),
            "probability": self.probability,
            "supporting_symptoms": self.supporting_symptoms,
            "contradicting_symptoms": self.contradicting_symptoms,
            "evidence_score": self.evidence_score,
            "reasoning": self.reasoning
        }


@dataclass
class DiagnosisResult:
    """Complete diagnosis result."""
    extracted_symptoms: List[Symptom]
    differential_diagnosis: List[DiagnosisCandidate]
    recommended_actions: List[str]
    urgency: Urgency
    confidence_range: Tuple[float, float]  # (low, high)
    disclaimer: str
    requires_escalation: bool = False
    escalation_reason: Optional[str] = None


# ============================================================
# SYMPTOM EXTRACTION
# ============================================================

class SymptomExtractor:
    """
    Extract symptoms from patient descriptions.
    
    Uses pattern matching and keyword detection.
    In production, use a medical NER model.
    """
    
    # Common symptoms with patterns
    SYMPTOM_PATTERNS = {
        "headache": [r"\bheadache\b", r"\bhead\s+pain\b", r"\bmigraine\b"],
        "fever": [r"\bfever\b", r"\btemperature\b", r"\bhigh\s+temp\b"],
        "cough": [r"\bcough(?:ing)?\b", r"\bdry\s+cough\b", r"\bwet\s+cough\b"],
        "fatigue": [r"\bfatigue\b", r"\btired\b", r"\bexhaust(?:ed|ion)\b"],
        "nausea": [r"\bnausea\b", r"\bnauseous\b", r"\bqueasy\b"],
        "pain": [r"\bpain\b", r"\bache\b", r"\bdiscomfort\b"],
        "shortness_of_breath": [r"\bshortness\s+of\s+breath\b", r"\bdifficulty\s+breathing\b", r"\bbreath(?:less|ing)\b"],
        "dizziness": [r"\bdizzy\b", r"\bdizziness\b", r"\blightheaded\b"],
        "chest_pain": [r"\bchest\s+pain\b", r"\bchest\s+(?:tight|pressure)\b"],
        "abdominal_pain": [r"\bstomach\s+pain\b", r"\babdominal\s+pain\b", r"\bbelly\s+ache\b"],
        "sore_throat": [r"\bsore\s+throat\b", r"\bthroat\s+pain\b"],
        "congestion": [r"\bcongestion\b", r"\bstuffy\s+nose\b", r"\bblocked\s+nose\b"],
        "vomiting": [r"\bvomit(?:ing)?\b", r"\bthrow(?:ing)?\s+up\b"],
        "diarrhea": [r"\bdiarrhea\b", r"\bloose\s+stools?\b"],
        "rash": [r"\brash\b", r"\bskin\s+(?:irritation|redness)\b"],
    }
    
    # Severity indicators
    SEVERITY_PATTERNS = {
        Severity.MILD: [r"\bslight(?:ly)?\b", r"\bmild(?:ly)?\b", r"\ba\s+little\b"],
        Severity.MODERATE: [r"\bmoderate(?:ly)?\b", r"\bsomewhat\b", r"\bfairly\b"],
        Severity.SEVERE: [r"\bsevere(?:ly)?\b", r"\bintense(?:ly)?\b", r"\bbad(?:ly)?\b", r"\bvery\b"],
        Severity.CRITICAL: [r"\bextreme(?:ly)?\b", r"\bunbearable\b", r"\bworst\b"],
    }
    
    # Duration patterns
    DURATION_PATTERN = r"(?:for\s+)?(\d+\s+(?:hours?|days?|weeks?|months?))"
    
    def __init__(self):
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Compile regex patterns."""
        self.compiled_symptoms = {
            symptom: [re.compile(p, re.IGNORECASE) for p in patterns]
            for symptom, patterns in self.SYMPTOM_PATTERNS.items()
        }
        
        self.compiled_severity = {
            severity: [re.compile(p, re.IGNORECASE) for p in patterns]
            for severity, patterns in self.SEVERITY_PATTERNS.items()
        }
    
    def _detect_severity(self, text: str, symptom_pos: int) -> Severity:
        """Detect severity from nearby text."""
        # Look at text around symptom mention
        start = max(0, symptom_pos - 50)
        end = min(len(text), symptom_pos + 50)
        context = text[start:end]
        
        for severity, patterns in self.compiled_severity.items():
            for pattern in patterns:
                if pattern.search(context):
                    return severity
        
        return Severity.MILD
    
    def _detect_duration(self, text: str) -> Optional[str]:
        """Extract duration from text."""
        match = re.search(self.DURATION_PATTERN, text, re.IGNORECASE)
        if match:
            return match.group(1)
        return None
    
    def extract(self, text: str) -> List[Symptom]:
        """
        Extract symptoms from text.
        
        Args:
            text: Patient description
        
        Returns:
            List of extracted symptoms
        """
        symptoms = []
        text_lower = text.lower()
        
        for symptom_name, patterns in self.compiled_symptoms.items():
            for pattern in patterns:
                match = pattern.search(text)
                if match:
                    severity = self._detect_severity(text, match.start())
                    duration = self._detect_duration(text)
                    
                    symptoms.append(Symptom(
                        name=symptom_name,
                        severity=severity,
                        duration=duration,
                        confidence=0.9
                    ))
                    break  # Found this symptom, move to next
        
        return symptoms


# ============================================================
# KNOWLEDGE BASE
# ============================================================

class MedicalKnowledgeBase:
    """
    Simple medical knowledge base for demo.
    
    In production, use a proper medical ontology (SNOMED-CT, ICD-10).
    """
    
    # Condition -> symptom associations
    CONDITION_SYMPTOMS = {
        "common_cold": {
            "symptoms": ["congestion", "sore_throat", "cough", "fatigue"],
            "severity": Severity.MILD,
            "urgency": Urgency.ROUTINE,
            "description": "Viral upper respiratory infection"
        },
        "influenza": {
            "symptoms": ["fever", "fatigue", "cough", "headache", "pain"],
            "severity": Severity.MODERATE,
            "urgency": Urgency.SOON,
            "description": "Seasonal flu virus"
        },
        "migraine": {
            "symptoms": ["headache", "nausea", "dizziness"],
            "severity": Severity.MODERATE,
            "urgency": Urgency.ROUTINE,
            "description": "Severe recurring headache"
        },
        "gastroenteritis": {
            "symptoms": ["nausea", "vomiting", "diarrhea", "abdominal_pain"],
            "severity": Severity.MODERATE,
            "urgency": Urgency.SOON,
            "description": "Stomach flu / GI infection"
        },
        "pneumonia": {
            "symptoms": ["fever", "cough", "shortness_of_breath", "chest_pain", "fatigue"],
            "severity": Severity.SEVERE,
            "urgency": Urgency.URGENT,
            "description": "Lung infection"
        },
        "heart_attack": {
            "symptoms": ["chest_pain", "shortness_of_breath", "dizziness", "pain", "nausea"],
            "severity": Severity.CRITICAL,
            "urgency": Urgency.EMERGENCY,
            "description": "Myocardial infarction - SEEK IMMEDIATE CARE"
        },
    }
    
    def get_conditions_for_symptoms(
        self, 
        symptoms: List[str]
    ) -> List[Tuple[str, float]]:
        """
        Get possible conditions for given symptoms.
        
        Returns list of (condition_name, match_score) tuples.
        """
        symptom_set = set(symptoms)
        scores = []
        
        for condition, info in self.CONDITION_SYMPTOMS.items():
            condition_symptoms = set(info["symptoms"])
            
            # Calculate Jaccard similarity
            intersection = len(symptom_set & condition_symptoms)
            union = len(symptom_set | condition_symptoms)
            
            if intersection > 0:
                score = intersection / union
                scores.append((condition, score))
        
        return sorted(scores, key=lambda x: x[1], reverse=True)
    
    def get_condition_info(self, condition: str) -> Optional[Dict[str, Any]]:
        """Get condition details."""
        return self.CONDITION_SYMPTOMS.get(condition)


# ============================================================
# DIAGNOSTIC ENGINE
# ============================================================

class DiagnosticEngine:
    """
    Main diagnostic reasoning engine.
    
    Pipeline:
        Symptoms -> Knowledge Base -> Scoring -> Ranking -> Validation
    
    Example:
        >>> engine = DiagnosticEngine()
        >>> result = engine.diagnose("I have a headache and feel nauseous")
    """
    
    DISCLAIMER = (
        "This is informational only and does not constitute medical advice. "
        "Please consult a healthcare professional for proper diagnosis and treatment."
    )
    
    def __init__(self):
        self.symptom_extractor = SymptomExtractor()
        self.knowledge_base = MedicalKnowledgeBase()
    
    def _score_candidate(
        self,
        condition: str,
        condition_info: Dict[str, Any],
        patient_symptoms: Set[str]
    ) -> DiagnosisCandidate:
        """Score a diagnosis candidate."""
        expected_symptoms = set(condition_info["symptoms"])
        
        supporting = list(patient_symptoms & expected_symptoms)
        missing = list(expected_symptoms - patient_symptoms)
        contradicting = []  # Would need more complex logic
        
        # Calculate probability (simplified)
        if len(expected_symptoms) > 0:
            match_ratio = len(supporting) / len(expected_symptoms)
        else:
            match_ratio = 0.0
        
        # Adjust for complexity
        evidence_score = match_ratio * (1 - 0.1 * len(missing))
        
        # Generate reasoning
        reasoning = f"Matches {len(supporting)}/{len(expected_symptoms)} expected symptoms."
        if missing:
            reasoning += f" Missing indicators: {', '.join(missing[:3])}."
        
        return DiagnosisCandidate(
            condition=Condition(
                name=condition,
                description=condition_info["description"],
                severity=condition_info["severity"],
                urgency=condition_info["urgency"]
            ),
            probability=min(match_ratio, 0.95),  # Never 100% confident
            supporting_symptoms=supporting,
            contradicting_symptoms=contradicting,
            missing_symptoms=missing,
            evidence_score=evidence_score,
            reasoning=reasoning
        )
    
    def _determine_urgency(
        self, 
        candidates: List[DiagnosisCandidate]
    ) -> Urgency:
        """Determine overall urgency from candidates."""
        if not candidates:
            return Urgency.ROUTINE
        
        # Use highest urgency from probable candidates
        urgencies = [
            c.condition.urgency for c in candidates 
            if c.probability > 0.3
        ]
        
        if Urgency.EMERGENCY in urgencies:
            return Urgency.EMERGENCY
        elif Urgency.URGENT in urgencies:
            return Urgency.URGENT
        elif Urgency.SOON in urgencies:
            return Urgency.SOON
        return Urgency.ROUTINE
    
    def _generate_recommendations(
        self,
        urgency: Urgency,
        candidates: List[DiagnosisCandidate]
    ) -> List[str]:
        """Generate action recommendations."""
        recommendations = []
        
        if urgency == Urgency.EMERGENCY:
            recommendations.append("SEEK IMMEDIATE EMERGENCY CARE")
            recommendations.append("Call emergency services (911) immediately")
        elif urgency == Urgency.URGENT:
            recommendations.append("Schedule urgent medical appointment today")
            recommendations.append("If symptoms worsen, seek emergency care")
        elif urgency == Urgency.SOON:
            recommendations.append("Schedule appointment with healthcare provider within 1-2 days")
            recommendations.append("Monitor symptoms and note any changes")
        else:
            recommendations.append("Monitor symptoms")
            recommendations.append("Schedule routine appointment if symptoms persist beyond 1 week")
        
        recommendations.append("Stay hydrated and rest")
        
        return recommendations
    
    def _check_escalation(
        self,
        symptoms: List[Symptom],
        candidates: List[DiagnosisCandidate]
    ) -> Tuple[bool, Optional[str]]:
        """Check if case requires escalation."""
        # Check for critical symptoms
        critical_symptoms = {"chest_pain", "shortness_of_breath"}
        patient_symptoms = {s.name for s in symptoms}
        
        if critical_symptoms & patient_symptoms:
            return True, "Critical symptoms detected requiring immediate attention"
        
        # Check for critical conditions with high probability
        for candidate in candidates:
            if (candidate.condition.urgency == Urgency.EMERGENCY and 
                candidate.probability > 0.3):
                return True, f"Possible {candidate.condition.name} - requires escalation"
        
        return False, None
    
    def diagnose(self, patient_description: str) -> DiagnosisResult:
        """
        Perform diagnostic analysis.
        
        Args:
            patient_description: Patient's symptom description
        
        Returns:
            DiagnosisResult with differential diagnosis
        """
        # Extract symptoms
        symptoms = self.symptom_extractor.extract(patient_description)
        symptom_names = {s.name for s in symptoms}
        
        if not symptoms:
            return DiagnosisResult(
                extracted_symptoms=[],
                differential_diagnosis=[],
                recommended_actions=["Please describe your symptoms in more detail"],
                urgency=Urgency.ROUTINE,
                confidence_range=(0.0, 0.0),
                disclaimer=self.DISCLAIMER
            )
        
        # Get candidate conditions
        condition_scores = self.knowledge_base.get_conditions_for_symptoms(
            list(symptom_names)
        )
        
        # Score candidates
        candidates = []
        for condition, _ in condition_scores[:5]:  # Top 5
            info = self.knowledge_base.get_condition_info(condition)
            if info:
                candidate = self._score_candidate(condition, info, symptom_names)
                candidates.append(candidate)
        
        # Determine urgency
        urgency = self._determine_urgency(candidates)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(urgency, candidates)
        
        # Check escalation
        requires_escalation, escalation_reason = self._check_escalation(
            symptoms, candidates
        )
        
        # Confidence range
        if candidates:
            probs = [c.probability for c in candidates]
            confidence_range = (min(probs), max(probs))
        else:
            confidence_range = (0.0, 0.0)
        
        return DiagnosisResult(
            extracted_symptoms=symptoms,
            differential_diagnosis=candidates,
            recommended_actions=recommendations,
            urgency=urgency,
            confidence_range=confidence_range,
            disclaimer=self.DISCLAIMER,
            requires_escalation=requires_escalation,
            escalation_reason=escalation_reason
        )


# ============================================================
# EXPORTS
# ============================================================

__all__ = [
    'Severity', 'Urgency',
    'Symptom', 'Condition', 'DiagnosisCandidate', 'DiagnosisResult',
    'SymptomExtractor', 'MedicalKnowledgeBase', 'DiagnosticEngine',
]
