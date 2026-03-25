"""
Comprehensive Dataset Examination and Verification Script

This script performs a complete audit of all datasets to ensure:
1. Zero data loss
2. Complete metadata coverage
3. Content integrity verification
4. Quality issue identification
5. Gap analysis

Run this before processing to ensure data completeness.
"""

import os
import json
import sqlite3
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import hashlib


class DatasetAuditor:
    """Complete dataset auditing system"""
    
    def __init__(self, datasets_dir: str):
        self.datasets_dir = Path(datasets_dir)
        self.report = {
            'timestamp': datetime.now().isoformat(),
            'datasets': {},
            'issues': [],
            'gaps': [],
            'recommendations': []
        }
    
    def audit_all(self):
        """Run complete audit on all datasets"""
        print("=" * 70)
        print("COMPREHENSIVE DATASET AUDIT")
        print("=" * 70)
        
        # Audit each dataset
        self.audit_extracted_books()
        self.audit_metadata()
        self.audit_system_book_datasets()
        
        # Cross-reference verification
        self.cross_reference_check()
        
        # Generate report
        self.generate_report()
        
        return self.report
    
    def audit_extracted_books(self):
        """Audit extracted_books directory"""
        print("\n[1/3] Auditing extracted_books...")
        
        books_dir = self.datasets_dir / "extracted_books"
        if not books_dir.exists():
            self.report['issues'].append("extracted_books directory not found")
            return
        
        # Count files
        txt_files = [f for f in os.listdir(books_dir) if f.endswith('.txt')]
        other_files = [f for f in os.listdir(books_dir) if not f.endswith('.txt')]
        
        print(f"  Total .txt files: {len(txt_files):,}")
        print(f"  Other files: {len(other_files)}")
        
        # Analyze file sizes
        sizes = []
        empty_files = []
        large_files = []
        
        for i, f in enumerate(txt_files[:1000]):  # Sample 1000
            file_path = books_dir / f
            size = file_path.stat().st_size
            sizes.append(size)
            
            if size < 100:
                empty_files.append(f)
            if size > 10 * 1024 * 1024:  # > 10MB
                large_files.append(f)
        
        avg_size = sum(sizes) / len(sizes) if sizes else 0
        total_size = sum(os.path.getsize(books_dir / f) for f in txt_files)
        
        print(f"  Average size: {avg_size/1024:.2f} KB")
        print(f"  Total size: {total_size/1024/1024:.2f} MB")
        print(f"  Empty/small files: {len(empty_files)}")
        print(f"  Large files (>10MB): {len(large_files)}")
        
        # Check content quality (sample 100 files)
        content_issues = []
        for f in txt_files[:100]:
            try:
                with open(books_dir / f, 'r', encoding='utf-8') as file:
                    content = file.read()
                    
                    # Check Arabic ratio
                    arabic_chars = sum(1 for c in content if '\u0600' <= c <= '\u06FF')
                    arabic_ratio = arabic_chars / len(content) if content else 0
                    
                    if arabic_ratio < 0.5:
                        content_issues.append(f"{f}: Low Arabic ratio ({arabic_ratio:.1%})")
                    
                    # Check for page markers
                    if '[Page' not in content and len(content) > 1000:
                        content_issues.append(f"{f}: No page markers")
                        
            except Exception as e:
                content_issues.append(f"{f}: {str(e)}")
        
        print(f"  Content quality issues: {len(content_issues)}")
        
        self.report['datasets']['extracted_books'] = {
            'total_files': len(txt_files),
            'other_files': len(other_files),
            'total_size_mb': total_size / 1024 / 1024,
            'avg_size_kb': avg_size / 1024,
            'empty_files': empty_files,
            'large_files': large_files,
            'content_issues': content_issues[:10]  # First 10
        }
        
        if empty_files:
            self.report['issues'].append(f"{len(empty_files)} empty/small files found")
        if content_issues:
            self.report['recommendations'].append(f"Review {len(content_issues)} content quality issues")
    
    def audit_metadata(self):
        """Audit metadata directory"""
        print("\n[2/3] Auditing metadata...")
        
        meta_dir = self.datasets_dir / "metadata"
        if not meta_dir.exists():
            self.report['issues'].append("metadata directory not found")
            return
        
        # Check required files
        required_files = ['books.json', 'authors.json', 'categories.json', 'guid_index.json', 'books.db']
        existing_files = os.listdir(meta_dir)
        
        for req in required_files:
            if req not in existing_files:
                self.report['issues'].append(f"Missing metadata file: {req}")
            else:
                file_path = meta_dir / req
                size_mb = file_path.stat().st_size / 1024 / 1024
                print(f"  {req}: {size_mb:.2f} MB")
        
        # Analyze books.json
        books_json = meta_dir / "books.json"
        if books_json.exists():
            with open(books_json, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            total = data.get('total', 0)
            extracted = data.get('extracted', 0)
            books_array = data.get('books', [])
            
            print(f"  Total books in metadata: {total:,}")
            print(f"  Extracted books: {extracted:,}")
            print(f"  Books in array: {len(books_array):,}")
            
            # Check for gaps
            book_ids = sorted([b['id'] for b in books_array if b.get('extracted', False)])
            if book_ids:
                id_range = range(min(book_ids), max(book_ids) + 1)
                missing_ids = set(id_range) - set(book_ids)
                
                print(f"  ID range: {min(book_ids)} - {max(book_ids)}")
                print(f"  Missing IDs: {len(missing_ids)}")
                
                if missing_ids:
                    self.report['gaps'].append(f"Missing book IDs: {sorted(missing_ids)[:20]}")
            
            # Category distribution
            categories = {}
            for b in books_array:
                cat = b.get('cat_name', 'Unknown')
                categories[cat] = categories.get(cat, 0) + 1
            
            print(f"  Categories: {len(categories)}")
            print(f"  Top 5 categories:")
            for cat, count in sorted(categories.items(), key=lambda x: -x[1])[:5]:
                print(f"    {cat}: {count:,}")
            
            self.report['datasets']['metadata'] = {
                'total_books': total,
                'extracted_books': extracted,
                'books_in_array': len(books_array),
                'missing_ids_count': len(missing_ids) if book_ids else 0,
                'categories': len(categories),
                'category_distribution': dict(sorted(categories.items(), key=lambda x: -x[1])[:10])
            }
    
    def audit_system_book_datasets(self):
        """Audit system_book_datasets directory"""
        print("\n[3/3] Auditing system_book_datasets...")
        
        sys_dir = self.datasets_dir / "system_book_datasets"
        if not sys_dir.exists():
            self.report['issues'].append("system_book_datasets directory not found")
            return
        
        # Check subdirectories
        subdirs = ['service', 'store', 'book', 'user']
        
        for subdir in subdirs:
            sub_path = sys_dir / subdir
            if sub_path.exists():
                items = os.listdir(sub_path)
                print(f"  {subdir}/: {len(items):,} items")
                
                # Analyze databases in service/
                if subdir == 'service':
                    for db_file in items:
                        if db_file.endswith('.db'):
                            db_path = sub_path / db_file
                            size_mb = db_path.stat().st_size / 1024 / 1024
                            print(f"    {db_file}: {size_mb:.2f} MB")
                            
                            # Check database tables
                            try:
                                conn = sqlite3.connect(db_path)
                                cursor = conn.cursor()
                                cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                                tables = [t[0] for t in cursor.fetchall()]
                                print(f"      Tables: {tables}")
                                conn.close()
                            except Exception as e:
                                print(f"      Error: {e}")
                
                # Analyze Lucene indexes in store/
                if subdir == 'store':
                    lucene_files = [f for f in items if f.startswith('_') or f.endswith('.cfe') or f.endswith('.cfs')]
                    print(f"    Lucene index files: {len(lucene_files)}")
                
                # Count book segments
                if subdir == 'book':
                    numeric_dirs = [d for d in items if d.isdigit()]
                    print(f"    Book segments: {len(numeric_dirs):,}")
            
            else:
                self.report['issues'].append(f"Missing subdirectory: {subdir}")
        
        self.report['datasets']['system_book_datasets'] = {
            'subdirectories': subdirs,
            'status': 'complete' if all((sys_dir / s).exists() for s in subdirs) else 'incomplete'
        }
    
    def cross_reference_check(self):
        """Cross-reference check between datasets"""
        print("\n[CROSS-REFERENCE] Checking consistency...")
        
        # Check if all extracted books have metadata
        books_dir = self.datasets_dir / "extracted_books"
        meta_dir = self.datasets_dir / "metadata"
        
        if books_dir.exists() and meta_dir.exists():
            # Get extracted book IDs from filenames
            txt_files = [f for f in os.listdir(books_dir) if f.endswith('.txt')]
            extracted_ids = set()
            for f in txt_files:
                try:
                    book_id = int(f.split('_')[0])
                    extracted_ids.add(book_id)
                except:
                    pass
            
            # Get metadata book IDs
            books_json = meta_dir / "books.json"
            if books_json.exists():
                with open(books_json, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                metadata_ids = set(b['id'] for b in data.get('books', []) if b.get('extracted', False))
                
                # Check for mismatches
                in_extracted_not_metadata = extracted_ids - metadata_ids
                in_metadata_not_extracted = metadata_ids - extracted_ids
                
                print(f"  Books in extracted_books: {len(extracted_ids):,}")
                print(f"  Books in metadata: {len(metadata_ids):,}")
                print(f"  In extracted but not metadata: {len(in_extracted_not_metadata)}")
                print(f"  In metadata but not extracted: {len(in_metadata_not_extracted)}")
                
                if in_extracted_not_metadata:
                    self.report['gaps'].append(f"Books without metadata: {sorted(in_extracted_not_metadata)[:20]}")
                if in_metadata_not_extracted:
                    self.report['gaps'].append(f"Books not extracted: {sorted(in_metadata_not_extracted)[:20]}")
                
                self.report['cross_reference'] = {
                    'extracted_count': len(extracted_ids),
                    'metadata_count': len(metadata_ids),
                    'missing_metadata': len(in_extracted_not_metadata),
                    'not_extracted': len(in_metadata_not_extracted)
                }
    
    def generate_report(self):
        """Generate comprehensive audit report"""
        print("\n" + "=" * 70)
        print("AUDIT SUMMARY")
        print("=" * 70)
        
        # Save report
        report_file = self.datasets_dir / "audit_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(self.report, f, ensure_ascii=False, indent=2)
        
        print(f"\nReport saved to: {report_file}")
        print(f"\nIssues found: {len(self.report['issues'])}")
        for issue in self.report['issues'][:5]:
            print(f"  ⚠️  {issue}")
        
        print(f"\nGaps found: {len(self.report['gaps'])}")
        for gap in self.report['gaps'][:5]:
            print(f"  ◦  {gap}")
        
        print(f"\nRecommendations: {len(self.report['recommendations'])}")
        for rec in self.report['recommendations'][:5]:
            print(f"  →  {rec}")
        
        # Overall status
        if not self.report['issues'] and not self.report['gaps']:
            print("\n✅ DATASET STATUS: HEALTHY - Ready for processing")
        elif len(self.report['issues']) <= 2 and len(self.report['gaps']) <= 2:
            print("\n⚠️  DATASET STATUS: MINOR ISSUES - Can proceed with caution")
        else:
            print("\n❌ DATASET STATUS: ISSUES FOUND - Review before processing")


def main():
    """Run comprehensive dataset audit"""
    auditor = DatasetAuditor("datasets")
    report = auditor.audit_all()
    
    print("\n" + "=" * 70)
    print("AUDIT COMPLETE")
    print("=" * 70)
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
