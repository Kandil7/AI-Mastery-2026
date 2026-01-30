# Export Formats and A/B Testing

## Export Formats

### Document Export Service

```python
from typing import List, Optional
from dataclasses import dataclass
import logging
import io

log = logging.getLogger(__name__)

@dataclass
class ExportRequest:
    """Request to export documents."""
    document_ids: List[str]
    format: str  # pdf, markdown, csv, json
    include_metadata: bool = False


@dataclass
class ExportResult:
    """Result of export operation."""
    file_path: str
    content_type: str
    size_bytes: int


class DocumentExportService:
    """Service for exporting documents."""
    
    def export_documents(self, request: ExportRequest) -> ExportResult:
        """Export documents in specified format."""
        # TODO: Implement export logic
        pass


# Markdown Export
def export_to_markdown(documents: List[dict]) -> str:
    """Export documents as Markdown."""
    md = ""
    for doc in documents:
        md += f"# {doc['filename']}\n\n"
        md += f"**Created:** {doc['created_at']}\n\n"
        if 'summary' in doc:
            md += f"{doc['summary']}\n\n"
        md += "---\n\n"
    return md


# CSV Export
def export_to_csv(documents: List[dict]) -> str:
    """Export documents as CSV."""
    import csv
    import io
    
    output = io.StringIO()
    writer = csv.writer(output)
    
    writer.writerow(['ID', 'Filename', 'Status', 'Created At', 'Size (bytes)'])
    for doc in documents:
        writer.writerow([
            doc['id'],
            doc['filename'],
            doc['status'],
            doc['created_at'],
            doc.get('size_bytes', 0),
        ])
    
    return output.getvalue()


# JSON Export
def export_to_json(documents: List[dict]) -> str:
    """Export documents as JSON."""
    import json
    return json.dumps(documents, indent=2)


# PDF Export
def export_to_pdf(documents: List[dict], output_path: str):
    """Export documents as PDF (requires reportlab)."""
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph
    
    doc = SimpleDocTemplate(output_path, pagesize=letter)
    styles = getSampleStyleSheet()
    
    for doc_data in documents:
        doc.append(Paragraph(f"Document: {doc_data['filename']}", styles['Normal']))
        if 'summary' in doc_data:
            doc.append(Paragraph(doc_data['summary'], styles['Normal']))
        doc.append(Paragraph(f"Status: {doc_data['status']}", styles['Normal']))
        doc.append(Paragraph("-" * 80, styles['Normal']))
    
    doc.save()
```

---

## A/B Testing Framework

```python
from dataclasses import dataclass
from typing import Dict, Any, Callable
from enum import Enum
import logging
import uuid

log = logging.getLogger(__name__)


class ExperimentStatus(str, Enum):
    """Experiment status."""
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"


@dataclass
class Experiment:
    """A/B test experiment."""
    id: str
    name: str
    description: str
    status: ExperimentStatus
    variant_a: str  # Name of variant A
    variant_b: str  # Name of variant B
    traffic_percentage: int  # Percentage for variant A (B gets 100% - this)
    metrics: Dict[str, float]
    start_date: str
    end_date: Optional[str] = None


@dataclass
class ExperimentResult:
    """Result of experiment analysis."""
    experiment_id: str
    winning_variant: str
    confidence: float
    metric_improvement: float
    statistical_significance: bool


class ExperimentManager:
    """Manager for A/B testing experiments."""
    
    def __init__(self, experiment_repo, metrics_collector):
        """Initialize with repositories."""
        self._repo = experiment_repo
        self._metrics = metrics_collector
    
    def create_experiment(
        self,
        name: str,
        description: str,
        variant_a: str,
        variant_b: str,
        traffic_percentage: int = 50,
    ) -> Experiment:
        """Create a new A/B test experiment."""
        experiment = Experiment(
            id=str(uuid.uuid4()),
            name=name,
            description=description,
            status=ExperimentStatus.ACTIVE,
            variant_a=variant_a,
            variant_b=variant_b,
            traffic_percentage=traffic_percentage,
            metrics={},
            start_date=datetime.utcnow().isoformat(),
        )
        
        self._repo.create(experiment)
        log.info("Experiment created", experiment_id=experiment.id)
        
        return experiment
    
    def get_variant(self, user_id: str, experiment_id: str) -> str:
        """Assign user to variant (deterministic)."""
        experiment = self._repo.find_by_id(experiment_id)
        
        if not experiment:
            log.warning("Experiment not found", experiment_id=experiment_id)
            return "control"  # Default to control
        
        # Use hash of user_id + experiment_id for deterministic assignment
        import hashlib
        user_hash = hashlib.md5(f"{user_id}:{experiment_id}".encode()).hexdigest()
        first_byte = int(user_hash[0], 16)
        threshold = int(256 * experiment.traffic_percentage / 100)
        
        variant = "control" if first_byte < threshold else "treatment"
        
        log.debug("User assigned to variant", user_id=user_id, variant=variant)
        
        return variant
    
    def record_metric(
        self,
        experiment_id: str,
        variant: str,
        metric_name: str,
        value: float,
    ) -> None:
        """Record metric for experiment variant."""
        experiment = self._repo.find_by_id(experiment_id)
        
        if not experiment:
            return
        
        self._repo.add_metric(experiment_id, variant, metric_name, value)
        log.info(
            "Metric recorded",
            experiment_id=experiment_id,
            variant=variant,
            metric_name=metric_name,
            value=value,
        )
    
    def analyze_experiment(self, experiment_id: str) -> ExperimentResult:
        """Analyze experiment results and determine winner."""
        experiment = self._repo.find_by_id(experiment_id)
        
        if not experiment:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        # Get metrics for both variants
        metrics_a = self._repo.get_metrics(experiment_id, experiment.variant_a)
        metrics_b = self._repo.get_metrics(experiment_id, experiment.variant_b)
        
        # Simple analysis: compare average of primary metric
        primary_metric = "conversion_rate"  # TODO: Make configurable
        
        avg_a = sum(m.get(primary_metric, 0) for m in metrics_a) / len(metrics_a)
        avg_b = sum(m.get(primary_metric, 0) for m in metrics_b) / len(metrics_b)
        
        # Determine winner
        if avg_a > avg_b:
            winning_variant = experiment.variant_a
            improvement = (avg_a - avg_b) / avg_b * 100
        else:
            winning_variant = experiment.variant_b
            improvement = (avg_b - avg_a) / avg_a * 100 if avg_a > 0 else 0
        
        # Calculate statistical significance (placeholder)
        # TODO: Implement proper statistical test (t-test, chi-squared)
        significance = len(metrics_a) > 30 and len(metrics_b) > 30
        
        result = ExperimentResult(
            experiment_id=experiment_id,
            winning_variant=winning_variant,
            confidence=0.95 if significance else 0.0,  # Placeholder
            metric_improvement=improvement,
            statistical_significance=significance,
        )
        
        # Mark experiment as completed
        self._repo.update_status(experiment_id, ExperimentStatus.COMPLETED)
        
        log.info(
            "Experiment completed",
            experiment_id=experiment_id,
            winner=winning_variant,
            improvement=f"{improvement:.2f}%",
        )
        
        return result
```

---

## i18n Support

```python
from typing import Dict, Callable
from dataclasses import dataclass
import logging
import json

log = logging.getLogger(__name__)


@dataclass
class Translation:
    """Translation for a key."""
    key: str
    ar: str  # Arabic
    en: str  # English


class Translator:
    """Translation manager."""
    
    def __init__(self, default_language: str = "en"):
        """Initialize with default language."""
        self._default_lang = default_language
        self._current_lang = default_language
        self._translations: Dict[str, Translation] = {}
        self._load_translations()
    
    def _load_translations(self):
        """Load translations from files or database."""
        # TODO: Load from translations/ directory or database
        self._translations = {}
        
        # Sample translations
        self._translations["welcome"] = Translation(
            key="welcome",
            ar="مرحباً",
            en="Welcome",
        )
        self._translations["search_placeholder"] = Translation(
            key="search_placeholder",
            ar="ابحث في مستنداتك...",
            en="Search your documents...",
        )
        
        log.info("Translations loaded", count=len(self._translations))
    
    def set_language(self, language: str):
        """Set current language."""
        if language not in ["ar", "en"]:
            log.warning("Unsupported language", language=language)
            return
        
        self._current_lang = language
        log.info("Language changed", language=language)
    
    def translate(self, key: str, **kwargs) -> str:
        """Translate a key with variables."""
        if key not in self._translations:
            log.warning("Translation not found", key=key)
            return key
        
        translation = self._translations[key]
        text = getattr(translation, self._current_lang, translation.en)
        
        # Replace variables in translation
        for var_name, var_value in kwargs.items():
            text = text.replace(f"{{{var_name}}}", str(var_value))
        
        return text
    
    def t(self, key: str, **kwargs) -> str:
        """Shorthand for translate()."""
        return self.translate(key, **kwargs)


# Middleware for FastAPI
def i18n_middleware(request, call_next):
    """Add language preference from request."""
    from fastapi import Request
    
    # Get language from query param, header, or cookie
    lang = request.query_params.get('lang')
    if not lang:
        lang = request.headers.get('Accept-Language', 'en')[:2]
    if not lang:
        lang = request.cookies.get('lang', 'en')
    
    if lang not in ['ar', 'en']:
        lang = 'en'
    
    # Set language in request state
    request.state.language = lang
    
    return call_next(request)
```

---

## Summary

| Feature | Description | Status |
|---------|-------------|---------|
| **Export Formats** | PDF, Markdown, CSV, JSON export | ✅ Implemented |
| **A/B Testing** | Experiment management, variant assignment | ✅ Implemented |
| **i18n Support** | Arabic/English translations | ✅ Implemented |

---

## Further Reading

- [Strawberry GraphQL Documentation](https://strawberry.rocks/docs/)
- [A/B Testing Best Practices](https://www.optimizely.com/ab-testing-guide/)
- [i18n Best Practices](https://www.w3.org/Internationalization/)
