"""
Balygh Data Audit & Improvement Script

This script audits your existing datasets and provides recommendations
for improvements based on the complete implementation plan.

Usage:
    python scripts/audit_datasets.py
"""

import os
import sys
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime

# ============================================================================
# Configuration
# ============================================================================

@dataclass
class DataConfig:
    """Dataset configuration and paths"""
    
    # Root directories
    DATASETS_ROOT = Path("K:/learning/technical/ai-ml/AI-Mastery-2026/datasets")
    ARABIC_LLM_ROOT = Path("K:/learning/technical/ai-ml/AI-Mastery-2026/arabic-llm")
    
    # Expected paths
    EXTRACTED_BOOKS = DATASETS_ROOT / "extracted_books"
    METADATA = DATASETS_ROOT / "metadata"
    SANADSET = DATASETS_ROOT / "Sanadset 368K Data on Hadith Narrators"
    SYSTEM_BOOKS = DATASETS_ROOT / "system_book_datasets"
    
    # Output paths
    OUTPUT_DIR = ARABIC_LLM_ROOT / "data"
    JSONL_DIR = OUTPUT_DIR / "jsonl"
    EVAL_DIR = OUTPUT_DIR / "evaluation"
    
    # Target statistics
    TARGET_EXAMPLES = 100000
    TARGET_FIQH_PCT = 0.30
    TARGET_LANG_PCT = 0.35
    TARGET_RAG_PCT = 0.20
    TARGET_OTHER_PCT = 0.15


# ============================================================================
# Data Audit Classes
# ============================================================================

@dataclass
class AuditResult:
    """Result of auditing a dataset"""
    name: str
    status: str  # "found", "missing", "partial"
    count: int = 0
    size_gb: float = 0.0
    quality_score: float = 0.0
    issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "status": self.status,
            "count": self.count,
            "size_gb": round(self.size_gb, 2),
            "quality_score": round(self.quality_score, 2),
            "issues": self.issues,
            "recommendations": self.recommendations,
        }


@dataclass
class CompleteAuditReport:
    """Complete audit report"""
    timestamp: str = ""
    datasets: Dict[str, AuditResult] = field(default_factory=dict)
    overall_quality: float = 0.0
    readiness_score: float = 0.0
    total_issues: int = 0
    total_recommendations: int = 0
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()
    
    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "datasets": {k: v.to_dict() for k, v in self.datasets.items()},
            "overall_quality": round(self.overall_quality, 2),
            "readiness_score": round(self.readiness_score, 2),
            "total_issues": self.total_issues,
            "total_recommendations": self.total_recommendations,
        }


# ============================================================================
# Audit Functions
# ============================================================================

def audit_extracted_books(config: DataConfig) -> AuditResult:
    """Audit extracted books dataset"""
    result = AuditResult(name="extracted_books", status="missing")
    
    if not config.EXTRACTED_BOOKS.exists():
        result.issues.append("Extracted books directory not found")
        result.recommendations.append(
            "Run book extraction from PDFs or import from Shamela"
        )
        return result
    
    # Count files
    txt_files = list(config.EXTRACTED_BOOKS.glob("*.txt"))
    result.count = len(txt_files)
    result.status = "found" if result.count > 0 else "partial"
    
    # Calculate size
    total_size = sum(f.stat().st_size for f in txt_files)
    result.size_gb = total_size / (1024**3)
    
    # Quality checks
    if result.count < 100:
        result.issues.append(f"Only {result.count} books found (target: 8,424)")
        result.recommendations.append(
            "Complete extraction of all 8,424 Shamela books"
        )
    elif result.count < 1000:
        result.quality_score = 0.5
        result.recommendations.append(
            f"Extract more books (current: {result.count}, target: 8,424)"
        )
    elif result.count < 5000:
        result.quality_score = 0.75
        result.recommendations.append(
            "Continue extraction to reach full corpus"
        )
    else:
        result.quality_score = 0.9
    
    if result.size_gb < 1.0:
        result.issues.append(f"Total size {result.size_gb:.2f} GB is low (target: 16.4 GB)")
    
    return result


def audit_metadata(config: DataConfig) -> AuditResult:
    """Audit metadata dataset"""
    result = AuditResult(name="metadata", status="missing")
    
    if not config.METADATA.exists():
        result.issues.append("Metadata directory not found")
        result.recommendations.append(
            "Create metadata directory with books.json, authors.json, categories.json"
        )
        return result
    
    # Check for books.json
    books_json = config.METADATA / "books.json"
    if books_json.exists():
        result.status = "found"
        try:
            with open(books_json, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if isinstance(data, dict):
                result.count = data.get('total', data.get('extracted', 0))
            elif isinstance(data, list):
                result.count = len(data)
            
            result.quality_score = 0.9 if result.count > 5000 else 0.7
            
            if result.count < 8424:
                result.recommendations.append(
                    f"Add metadata for remaining {8424 - result.count} books"
                )
        except json.JSONDecodeError as e:
            result.status = "partial"
            result.issues.append(f"books.json is not valid JSON: {e}")
            result.quality_score = 0.3
    else:
        result.issues.append("books.json not found")
        result.recommendations.append(
            "Create books.json with book metadata (id, title, category, author, etc.)"
        )
    
    return result


def audit_sanadset(config: DataConfig) -> AuditResult:
    """Audit Sanadset Hadith dataset"""
    result = AuditResult(name="sanadset_hadith", status="missing")
    
    if not config.SANADSET.exists():
        result.issues.append("Sanadset directory not found")
        result.recommendations.append(
            "Download Sanadset 368K Hadith narrators dataset from Mendeley Data"
        )
        return result
    
    # Count files
    all_files = list(config.SANADSET.rglob("*"))
    result.count = len(all_files)
    result.status = "found" if result.count > 0 else "partial"
    
    # Calculate size
    total_size = sum(f.stat().st_size for f in all_files if f.is_file())
    result.size_gb = total_size / (1024**3)
    
    # Quality assessment
    if result.count > 10:
        result.quality_score = 0.85
        result.recommendations.append(
            "Integrate Sanadset data into hadith training examples"
        )
    else:
        result.quality_score = 0.5
        result.issues.append("Sanadset data appears incomplete")
    
    return result


def audit_system_books(config: DataConfig) -> AuditResult:
    """Audit system book datasets"""
    result = AuditResult(name="system_books", status="missing")
    
    if not config.SYSTEM_BOOKS.exists():
        result.issues.append("System books directory not found")
        result.recommendations.append(
            "Add structured databases (hadith.db, tafseer.db, etc.)"
        )
        return result
    
    # Count files
    db_files = list(config.SYSTEM_BOOKS.glob("*.db"))
    json_files = list(config.SYSTEM_BOOKS.glob("*.json"))
    result.count = len(db_files) + len(json_files)
    result.status = "found" if result.count > 0 else "partial"
    
    # Calculate size
    all_files = list(config.SYSTEM_BOOKS.rglob("*"))
    total_size = sum(f.stat().st_size for f in all_files if f.is_file())
    result.size_gb = total_size / (1024**3)
    
    # Quality assessment
    if len(db_files) >= 3:
        result.quality_score = 0.8
        result.recommendations.append(
            "Good database coverage. Consider adding more specialized DBs"
        )
    elif len(db_files) > 0:
        result.quality_score = 0.6
        result.recommendations.append(
            "Add more databases (target: hadith.db, tafseer.db, trajim.db)"
        )
    else:
        result.quality_score = 0.4
        result.issues.append("No database files found")
    
    return result


def audit_generated_datasets(config: DataConfig) -> AuditResult:
    """Audit generated SFT datasets"""
    result = AuditResult(name="generated_sft", status="missing")
    
    if not config.JSONL_DIR.exists():
        result.issues.append("JSONL output directory not found")
        result.recommendations.append(
            "Run build_balygh_sft_dataset.py to generate SFT examples"
        )
        return result
    
    # Count JSONL files
    jsonl_files = list(config.JSONL_DIR.glob("*.jsonl"))
    result.count = len(jsonl_files)
    result.status = "found" if result.count > 0 else "partial"
    
    # Count total examples
    total_examples = 0
    for jsonl_file in jsonl_files:
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            total_examples += sum(1 for _ in f)
    
    # Quality assessment
    if total_examples >= config.TARGET_EXAMPLES:
        result.quality_score = 0.95
        result.recommendations.append(
            f"Excellent! {total_examples:,} examples generated. Ready for training."
        )
    elif total_examples >= 50000:
        result.quality_score = 0.8
        result.recommendations.append(
            f"Good progress ({total_examples:,} examples). Generate more to reach {config.TARGET_EXAMPLES:,}"
        )
    elif total_examples >= 10000:
        result.quality_score = 0.6
        result.recommendations.append(
            f"Continue generation (current: {total_examples:,}, target: {config.TARGET_EXAMPLES:,})"
        )
    else:
        result.quality_score = 0.4
        result.issues.append(
            f"Only {total_examples:,} examples generated (target: {config.TARGET_EXAMPLES:,})"
        )
    
    return result


def audit_evaluation_datasets(config: DataConfig) -> AuditResult:
    """Audit evaluation datasets"""
    result = AuditResult(name="evaluation", status="missing")
    
    if not config.EVAL_DIR.exists():
        result.issues.append("Evaluation directory not found")
        result.recommendations.append(
            "Create evaluation datasets (fiqh_eval.jsonl, nahw_eval.jsonl, etc.)"
        )
        return result
    
    # Check for required eval files
    required_files = [
        "fiqh_eval.jsonl",
        "hadith_eval.jsonl",
        "nahw_eval.jsonl",
        "balagha_eval.jsonl",
        "scraping_eval.jsonl",
    ]
    
    found_files = []
    for req_file in required_files:
        if (config.EVAL_DIR / req_file).exists():
            found_files.append(req_file)
    
    result.count = len(found_files)
    result.status = "found" if len(found_files) >= 3 else "partial"
    
    # Quality assessment
    if len(found_files) >= 5:
        result.quality_score = 0.9
        result.recommendations.append(
            "Complete evaluation suite ready. Add more specialized tests if needed."
        )
    elif len(found_files) >= 3:
        result.quality_score = 0.7
        missing = set(required_files) - set(found_files)
        result.issues.append(f"Missing eval files: {missing}")
        result.recommendations.append(
            f"Create missing evaluation files: {missing}"
        )
    else:
        result.quality_score = 0.4
        result.issues.append("Insufficient evaluation datasets")
        result.recommendations.append(
            "Create comprehensive evaluation sets for all roles"
        )
    
    return result


# ============================================================================
# Improvement Recommendations
# ============================================================================

def generate_improvement_plan(audit_report: CompleteAuditReport) -> List[Dict]:
    """Generate prioritized improvement plan based on audit"""
    improvements = []
    
    # Priority 1: Critical missing data
    for name, result in audit_report.datasets.items():
        if result.status == "missing":
            improvements.append({
                "priority": 1,
                "category": "Critical",
                "action": f"Create {name} dataset",
                "details": result.recommendations[0] if result.recommendations else "No details",
                "impact": "Required for training",
            })
    
    # Priority 2: Quality improvements
    for name, result in audit_report.datasets.items():
        if result.quality_score < 0.7 and result.status != "missing":
            improvements.append({
                "priority": 2,
                "category": "High",
                "action": f"Improve {name} quality",
                "details": result.recommendations[0] if result.recommendations else "Quality enhancement needed",
                "impact": f"Current quality: {result.quality_score:.2f}",
            })
    
    # Priority 3: Completeness
    for name, result in audit_report.datasets.items():
        if result.quality_score >= 0.7 and result.status == "found":
            improvements.append({
                "priority": 3,
                "category": "Medium",
                "action": f"Complete {name}",
                "details": result.recommendations[0] if result.recommendations else "Add more data",
                "impact": "Enhance coverage",
            })
    
    return improvements


# ============================================================================
# Main Audit Function
# ============================================================================

def run_complete_audit() -> CompleteAuditReport:
    """Run complete data audit"""
    config = DataConfig()
    report = CompleteAuditReport()
    
    print("=" * 70)
    print("Balygh Data Audit")
    print("=" * 70)
    print()
    
    # Audit each dataset
    audits = [
        ("Extracted Books", audit_extracted_books),
        ("Metadata", audit_metadata),
        ("Sanadset Hadith", audit_sanadset),
        ("System Books", audit_system_books),
        ("Generated SFT", audit_generated_datasets),
        ("Evaluation", audit_evaluation_datasets),
    ]
    
    for name, audit_func in audits:
        print(f"Auditing {name}...")
        result = audit_func(config)
        report.datasets[name] = result
        report.total_issues += len(result.issues)
        report.total_recommendations += len(result.recommendations)
        
        status_icon = "✅" if result.status == "found" else "⚠️" if result.status == "partial" else "❌"
        print(f"  {status_icon} {name}: {result.status}")
        print(f"     Count: {result.count:,}, Size: {result.size_gb:.2f} GB, Quality: {result.quality_score:.2f}")
        
        if result.issues:
            for issue in result.issues[:2]:
                print(f"     ⚠️  {issue}")
        print()
    
    # Calculate overall scores
    quality_scores = [r.quality_score for r in report.datasets.values() if r.status != "missing"]
    if quality_scores:
        report.overall_quality = sum(quality_scores) / len(quality_scores)
    
    found_count = sum(1 for r in report.datasets.values() if r.status == "found")
    report.readiness_score = found_count / len(report.datasets)
    
    # Generate improvement plan
    improvements = generate_improvement_plan(report)
    
    print("=" * 70)
    print(f"Overall Quality: {report.overall_quality:.2f}")
    print(f"Readiness Score: {report.readiness_score:.2f}")
    print(f"Total Issues: {report.total_issues}")
    print(f"Total Recommendations: {report.total_recommendations}")
    print("=" * 70)
    print()
    
    if improvements:
        print("📋 Improvement Plan:")
        print()
        for imp in improvements[:10]:  # Show top 10
            priority_icon = "🔴" if imp["priority"] == 1 else "🟡" if imp["priority"] == 2 else "🟢"
            print(f"  {priority_icon} [{imp['category']}] {imp['action']}")
            print(f"      {imp['details']}")
            print()
    
    # Save report
    output_file = Path("data/audit_report.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(report.to_dict(), f, ensure_ascii=False, indent=2)
    
    print(f"📄 Full audit report saved to: {output_file}")
    
    return report


if __name__ == "__main__":
    run_complete_audit()
