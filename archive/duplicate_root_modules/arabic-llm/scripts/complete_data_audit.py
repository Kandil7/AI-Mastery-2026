"""
Complete Data Audit Script for Balygh

Audits ALL 5 data sources and generates comprehensive report.

Usage:
    python scripts/complete_data_audit.py
"""

import os
import sys
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime
import re

# ============================================================================
# Configuration
# ============================================================================

@dataclass
class AuditConfig:
    """Audit configuration"""
    
    # Root paths
    DATASETS_ROOT = Path("K:/learning/technical/ai-ml/AI-Mastery-2026/datasets")
    ARABIC_LLM_ROOT = Path("K:/learning/technical/ai-ml/AI-Mastery-2026/arabic-llm")
    
    # Data sources
    ARABIC_WEB = DATASETS_ROOT / "arabic_web"
    EXTRACTED_BOOKS = DATASETS_ROOT / "extracted_books"
    METADATA = DATASETS_ROOT / "metadata"
    SANADSET = DATASETS_ROOT / "Sanadset 368K Data on Hadith Narrators"
    SYSTEM_BOOKS = DATASETS_ROOT / "system_book_datasets"
    
    # Output
    OUTPUT_DIR = ARABIC_LLM_ROOT / "data"
    REPORT_FILE = OUTPUT_DIR / "complete_audit_report.json"


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class SourceAudit:
    """Audit result for one data source"""
    name: str
    path: str
    status: str  # "found", "partial", "missing"
    file_count: int = 0
    total_size_gb: float = 0.0
    item_count: int = 0  # books, narrators, etc.
    quality_score: float = 0.0
    issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    details: Dict = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "path": self.path,
            "status": self.status,
            "file_count": self.file_count,
            "total_size_gb": round(self.total_size_gb, 2),
            "item_count": self.item_count,
            "quality_score": round(self.quality_score, 2),
            "issues": self.issues,
            "recommendations": self.recommendations,
            "details": self.details,
        }


@dataclass
class CompleteAuditReport:
    """Complete audit report for all 5 sources"""
    timestamp: str = ""
    sources: Dict[str, SourceAudit] = field(default_factory=dict)
    overall_quality: float = 0.0
    readiness_score: float = 0.0
    total_files: int = 0
    total_size_gb: float = 0.0
    total_items: int = 0
    total_issues: int = 0
    total_recommendations: int = 0
    estimated_examples: int = 0
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()
    
    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "sources": {k: v.to_dict() for k, v in self.sources.items()},
            "overall_quality": round(self.overall_quality, 2),
            "readiness_score": round(self.readiness_score, 2),
            "total_files": self.total_files,
            "total_size_gb": round(self.total_size_gb, 2),
            "total_items": self.total_items,
            "total_issues": self.total_issues,
            "total_recommendations": self.total_recommendations,
            "estimated_examples": self.estimated_examples,
        }


# ============================================================================
# Audit Functions
# ============================================================================

def audit_directory_basic(path: Path) -> Tuple[int, float]:
    """Get basic directory stats"""
    if not path.exists():
        return 0, 0.0
    
    file_count = 0
    total_size = 0
    
    for f in path.rglob("*"):
        if f.is_file():
            file_count += 1
            total_size += f.stat().st_size
    
    return file_count, total_size / (1024**3)


def audit_arabic_web(config: AuditConfig) -> SourceAudit:
    """Audit Arabic Web corpus"""
    audit = SourceAudit(
        name="arabic_web",
        path=str(config.ARABIC_WEB),
        status="missing"
    )
    
    if not config.ARABIC_WEB.exists():
        audit.issues.append("Arabic web directory not found")
        audit.recommendations.append(
            "Add Arabic web corpus (ArabicWeb24, FineWeb-Arabic, or similar)"
        )
        return audit
    
    # Count files and size
    file_count, size_gb = audit_directory_basic(config.ARABIC_WEB)
    audit.file_count = file_count
    audit.total_size_gb = size_gb
    
    # Check file types
    json_files = list(config.ARABIC_WEB.glob("*.json"))
    jsonl_files = list(config.ARABIC_WEB.glob("*.jsonl"))
    txt_files = list(config.ARABIC_WEB.glob("*.txt"))
    
    audit.details = {
        "json_files": len(json_files),
        "jsonl_files": len(jsonl_files),
        "txt_files": len(txt_files),
    }
    
    # Quality assessment
    if file_count > 0:
        audit.status = "found"
        audit.item_count = file_count
        
        if size_gb >= 10.0:
            audit.quality_score = 0.9
            audit.recommendations.append(
                f"Excellent! {size_gb:.2f} GB of Arabic web data. Process for general Arabic."
            )
        elif size_gb >= 1.0:
            audit.quality_score = 0.7
            audit.recommendations.append(
                f"Good start ({size_gb:.2f} GB). Consider adding more web corpus."
            )
        else:
            audit.quality_score = 0.5
            audit.issues.append(f"Small corpus ({size_gb:.2f} GB). Add more data.")
    else:
        audit.status = "partial"
        audit.quality_score = 0.3
        audit.issues.append("Directory exists but no files found")
    
    return audit


def audit_extracted_books(config: AuditConfig) -> SourceAudit:
    """Audit extracted books"""
    audit = SourceAudit(
        name="extracted_books",
        path=str(config.EXTRACTED_BOOKS),
        status="missing"
    )
    
    if not config.EXTRACTED_BOOKS.exists():
        audit.issues.append("Extracted books directory not found")
        audit.recommendations.append(
            "Extract books from PDFs or import from Shamela"
        )
        return audit
    
    # Count files and size
    file_count, size_gb = audit_directory_basic(config.EXTRACTED_BOOKS)
    audit.file_count = file_count
    audit.total_size_gb = size_gb
    
    # Check file types
    txt_files = list(config.EXTRACTED_BOOKS.glob("*.txt"))
    audit.item_count = len(txt_files)
    
    audit.details = {
        "txt_files": len(txt_files),
        "expected_books": 8424,
        "coverage_pct": round(len(txt_files) / 8424 * 100, 1) if txt_files else 0,
    }
    
    # Quality assessment
    if len(txt_files) > 0:
        audit.status = "found"
        
        if len(txt_files) >= 8000:
            audit.quality_score = 0.95
            audit.recommendations.append(
                f"Excellent! {len(txt_files):,} books ({size_gb:.2f} GB). Ready for full processing."
            )
        elif len(txt_files) >= 5000:
            audit.quality_score = 0.8
            audit.recommendations.append(
                f"Good progress ({len(txt_files):,} books). Continue extraction to 8,424."
            )
        elif len(txt_files) >= 1000:
            audit.quality_score = 0.6
            audit.recommendations.append(
                f"Continue extraction (current: {len(txt_files):,}, target: 8,424)"
            )
        else:
            audit.quality_score = 0.4
            audit.issues.append(
                f"Only {len(txt_files):,} books. Need more for comprehensive training."
            )
    else:
        audit.status = "partial"
        audit.quality_score = 0.3
        audit.issues.append("Directory exists but no .txt files found")
    
    return audit


def audit_metadata(config: AuditConfig) -> SourceAudit:
    """Audit metadata"""
    audit = SourceAudit(
        name="metadata",
        path=str(config.METADATA),
        status="missing"
    )
    
    if not config.METADATA.exists():
        audit.issues.append("Metadata directory not found")
        audit.recommendations.append(
            "Create metadata directory with books.json, authors.json, categories.json"
        )
        return audit
    
    # Count files
    json_files = list(config.METADATA.glob("*.json"))
    audit.file_count = len(json_files)
    
    # Check for required files
    required_files = [
        "books.json",
        "authors.json",
        "categories.json",
    ]
    
    found_files = []
    book_count = 0
    
    for req_file in required_files:
        file_path = config.METADATA / req_file
        if file_path.exists():
            found_files.append(req_file)
            
            # Try to count items
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                if isinstance(data, dict):
                    if 'books' in data:
                        book_count = len(data['books'])
                    elif 'total' in data:
                        book_count = data['total']
                elif isinstance(data, list):
                    book_count = len(data)
            except:
                pass
    
    audit.item_count = book_count
    audit.details = {
        "json_files": len(json_files),
        "required_files": len(required_files),
        "found_files": len(found_files),
        "book_count": book_count,
    }
    
    # Quality assessment
    if len(found_files) >= 3:
        audit.status = "found"
        audit.quality_score = 0.9
        
        if book_count >= 8000:
            audit.recommendations.append(
                f"Excellent metadata! {book_count:,} books catalogued."
            )
        else:
            audit.recommendations.append(
                f"Good metadata structure. Add entries for remaining {8424 - book_count} books."
            )
    elif len(found_files) >= 1:
        audit.status = "partial"
        audit.quality_score = 0.6
        missing = set(required_files) - set(found_files)
        audit.issues.append(f"Missing metadata files: {missing}")
        audit.recommendations.append(f"Create missing files: {missing}")
    else:
        audit.status = "partial"
        audit.quality_score = 0.3
        audit.issues.append("Metadata directory exists but no JSON files found")
        audit.recommendations.append("Create books.json, authors.json, categories.json")
    
    return audit


def audit_sanadset(config: AuditConfig) -> SourceAudit:
    """Audit Sanadset hadith narrators"""
    audit = SourceAudit(
        name="sanadset_hadith",
        path=str(config.SANADSET),
        status="missing"
    )
    
    if not config.SANADSET.exists():
        audit.issues.append("Sanadset directory not found")
        audit.recommendations.append(
            "Download Sanadset 368K Hadith narrators dataset"
        )
        return audit
    
    # Count files and size
    file_count, size_gb = audit_directory_basic(config.SANADSET)
    audit.file_count = file_count
    audit.total_size_gb = size_gb
    
    # Check file types
    json_files = list(config.SANADSET.glob("*.json"))
    csv_files = list(config.SANADSET.glob("*.csv"))
    
    # Try to count narrators
    narrator_count = 0
    for jf in json_files:
        try:
            with open(jf, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if isinstance(data, list):
                narrator_count += len(data)
        except:
            pass
    
    audit.item_count = narrator_count
    audit.details = {
        "json_files": len(json_files),
        "csv_files": len(csv_files),
        "narrator_count": narrator_count,
        "expected_narrators": 368000,
        "coverage_pct": round(narrator_count / 368000 * 100, 1) if narrator_count else 0,
    }
    
    # Quality assessment
    if narrator_count > 0:
        audit.status = "found"
        
        if narrator_count >= 300000:
            audit.quality_score = 0.95
            audit.recommendations.append(
                f"Excellent! {narrator_count:,} narrators. Ready for hadith example generation."
            )
        elif narrator_count >= 100000:
            audit.quality_score = 0.8
            audit.recommendations.append(
                f"Good coverage ({narrator_count:,} narrators). Process for hadith training."
            )
        else:
            audit.quality_score = 0.6
            audit.recommendations.append(
                f"Continue adding narrators (current: {narrator_count:,}, target: 368K)"
            )
    elif file_count > 0:
        audit.status = "partial"
        audit.quality_score = 0.5
        audit.issues.append("Files found but couldn't count narrators")
        audit.recommendations.append("Verify data format and structure")
    else:
        audit.status = "partial"
        audit.quality_score = 0.3
        audit.issues.append("Directory exists but no data files found")
    
    return audit


def audit_system_books(config: AuditConfig) -> SourceAudit:
    """Audit system book databases"""
    audit = SourceAudit(
        name="system_books",
        path=str(config.SYSTEM_BOOKS),
        status="missing"
    )
    
    if not config.SYSTEM_BOOKS.exists():
        audit.issues.append("System books directory not found")
        audit.recommendations.append(
            "Add structured databases (hadeeth.db, tafseer.db, etc.)"
        )
        return audit
    
    # Count files and size
    file_count, size_gb = audit_directory_basic(config.SYSTEM_BOOKS)
    audit.file_count = file_count
    audit.total_size_gb = size_gb
    
    # Check file types
    db_files = list(config.SYSTEM_BOOKS.glob("*.db"))
    json_files = list(config.SYSTEM_BOOKS.glob("*.json"))
    sql_files = list(config.SYSTEM_BOOKS.glob("*.sql"))
    
    audit.details = {
        "database_files": len(db_files),
        "json_files": len(json_files),
        "sql_files": len(sql_files),
    }
    
    # Try to count records in databases
    total_records = 0
    if db_files:
        try:
            import sqlite3
            for db_file in db_files:
                conn = sqlite3.connect(str(db_file))
                cursor = conn.cursor()
                # Get table names
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                tables = cursor.fetchall()
                for table in tables:
                    cursor.execute(f"SELECT COUNT(*) FROM {table[0]}")
                    count = cursor.fetchone()[0]
                    total_records += count
                conn.close()
        except Exception as e:
            audit.issues.append(f"Could not read databases: {e}")
    
    audit.item_count = total_records
    
    # Quality assessment
    if len(db_files) >= 3:
        audit.status = "found"
        audit.quality_score = 0.85
        
        if total_records >= 100000:
            audit.recommendations.append(
                f"Excellent! {len(db_files)} databases with {total_records:,} records."
            )
        else:
            audit.recommendations.append(
                f"Good database structure ({len(db_files)} DBs). Add more records."
            )
    elif len(db_files) > 0:
        audit.status = "partial"
        audit.quality_score = 0.6
        audit.issues.append(f"Only {len(db_files)} database(s). Need more for comprehensive coverage.")
        audit.recommendations.append(
            "Create: hadeeth.db, tafseer.db, trajim.db, fiqh.db, language.db"
        )
    else:
        audit.status = "partial"
        audit.quality_score = 0.4
        audit.issues.append("No database files found")
        audit.recommendations.append(
            "Create structured databases from extracted books and metadata"
        )
    
    return audit


# ============================================================================
# Summary & Recommendations
# ============================================================================

def calculate_summary(report: CompleteAuditReport) -> None:
    """Calculate summary statistics"""
    # Total files
    report.total_files = sum(s.file_count for s in report.sources.values())
    
    # Total size
    report.total_size_gb = sum(s.total_size_gb for s in report.sources.values())
    
    # Total items
    report.total_items = sum(s.item_count for s in report.sources.values())
    
    # Total issues
    report.total_issues = sum(len(s.issues) for s in report.sources.values())
    
    # Total recommendations
    report.total_recommendations = sum(len(s.recommendations) for s in report.sources.values())
    
    # Overall quality (average of non-missing sources)
    quality_scores = [
        s.quality_score for s in report.sources.values()
        if s.status != "missing"
    ]
    if quality_scores:
        report.overall_quality = sum(quality_scores) / len(quality_scores)
    
    # Readiness score (fraction of sources found)
    found_count = sum(1 for s in report.sources.values() if s.status == "found")
    report.readiness_score = found_count / len(report.sources)
    
    # Estimated examples
    report.estimated_examples = int(
        report.total_items * 0.8  # Conservative estimate
    )


def generate_priority_actions(report: CompleteAuditReport) -> List[Dict]:
    """Generate prioritized action items"""
    actions = []
    
    # Priority 1: Missing sources
    for name, source in report.sources.items():
        if source.status == "missing":
            actions.append({
                "priority": 1,
                "category": "Critical",
                "action": f"Add {name} dataset",
                "details": source.recommendations[0] if source.recommendations else "No details",
            })
    
    # Priority 2: Low quality sources
    for name, source in report.sources.items():
        if source.quality_score < 0.6 and source.status != "missing":
            actions.append({
                "priority": 2,
                "category": "High",
                "action": f"Improve {name} quality",
                "details": source.recommendations[0] if source.recommendations else "Quality improvement needed",
            })
    
    # Priority 3: Partial sources
    for name, source in report.sources.items():
        if source.status == "partial" and source.quality_score >= 0.6:
            actions.append({
                "priority": 3,
                "category": "Medium",
                "action": f"Complete {name}",
                "details": source.recommendations[0] if source.recommendations else "Add more data",
            })
    
    # Priority 4: Optimization
    for name, source in report.sources.items():
        if source.status == "found" and source.quality_score >= 0.8:
            actions.append({
                "priority": 4,
                "category": "Low",
                "action": f"Optimize {name} processing",
                "details": source.recommendations[0] if source.recommendations else "Ready for processing",
            })
    
    return actions


# ============================================================================
# Main Audit Function
# ============================================================================

def run_complete_audit() -> CompleteAuditReport:
    """Run complete audit of all 5 data sources"""
    config = AuditConfig()
    report = CompleteAuditReport()
    
    print("=" * 70)
    print("Balygh Complete Data Audit")
    print("=" * 70)
    print()
    
    # Audit each source
    audits = [
        ("Arabic Web", audit_arabic_web),
        ("Extracted Books", audit_extracted_books),
        ("Metadata", audit_metadata),
        ("Sanadset Hadith", audit_sanadset),
        ("System Books", audit_system_books),
    ]
    
    for name, audit_func in audits:
        print(f"Auditing {name}...")
        result = audit_func(config)
        report.sources[name] = result

        status_icon = "[OK]" if result.status == "found" else "[!]" if result.status == "partial" else "[X]"
        print(f"  {status_icon} {name}: {result.status}")
        print(f"     Files: {result.file_count:,}, Size: {result.total_size_gb:.2f} GB, Items: {result.item_count:,}")
        print(f"     Quality: {result.quality_score:.2f}")

        if result.issues:
            for issue in result.issues[:2]:
                print(f"     [!] {issue}")
        print()
    
    # Calculate summary
    calculate_summary(report)
    
    # Generate priority actions
    actions = generate_priority_actions(report)
    
    # Print summary
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"Total Files: {report.total_files:,}")
    print(f"Total Size: {report.total_size_gb:.2f} GB")
    print(f"Total Items: {report.total_items:,}")
    print(f"Overall Quality: {report.overall_quality:.2f}")
    print(f"Readiness Score: {report.readiness_score:.2f}")
    print(f"Total Issues: {report.total_issues}")
    print(f"Total Recommendations: {report.total_recommendations}")
    print(f"Estimated Examples: {report.estimated_examples:,}")
    print()
    
    if actions:
        print("- Priority Actions:")
        print()
        for action in actions[:10]:  # Show top 10
            priority_icon = "[!]" if action["priority"] == 1 else "[~]" if action["priority"] == 2 else "[+]"
            print(f"  {priority_icon} [{action['category']}] {action['action']}")
            print(f"      {action['details']}")
            print()

    # Save report
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(config.REPORT_FILE, 'w', encoding='utf-8') as f:
        json.dump(report.to_dict(), f, ensure_ascii=False, indent=2)

    print(f"Full audit report saved to: {config.REPORT_FILE}")
    print()
    print("=" * 70)
    print("Audit Complete!")
    print("=" * 70)
    
    return report


if __name__ == "__main__":
    run_complete_audit()
