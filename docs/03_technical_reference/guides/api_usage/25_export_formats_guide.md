# Export Formats - Complete Implementation Guide
# ===========================================

## 📚 Learning Objectives

By the end of this guide, you will understand:
- Export service architecture
- PDF generation with ReportLab
- Markdown export formatting
- CSV export with pandas
- JSON export patterns
- Export API endpoints
- Best practices for large exports

---
## 1. PDF Export

**Implementation:**
```python
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
from typing import List

class PDFExportService:
    """Export documents to PDF."""

    def export_documents(
        self,
        documents: List[Dict],
        title: str = "Document Export",
    ) -> bytes:
        """
        Export documents to PDF.

        Args:
            documents: List of documents with filename, content
            title: Document title

        Returns:
            PDF file content as bytes
        """
        # Create PDF canvas
        pdf = canvas.Canvas("output.pdf", pagesize=letter)

        # Add title
        pdf.setFont("Helvetica-Bold", 18)
        pdf.drawString(1 * inch, 10 * inch, title)

        # Add documents
        y = 9 * inch
        for doc in documents:
            pdf.setFont("Helvetica", 12)
            pdf.drawString(1 * inch, y, f"• {doc['filename']}")
            y -= 0.3 * inch

        pdf.save()
        return b""
```

---
## 2. Markdown Export

**Implementation:**
```python
class MarkdownExportService:
    """Export documents to Markdown."""

    def export_documents(
        self,
        documents: List[Dict],
        title: str = "Document Export",
    ) -> str:
        """Export documents to Markdown."""
        md = f"# {title}\n\n"

        for doc in documents:
            md += f"## {doc['filename']}\n\n"
            md += f"**Content Type:** {doc['content_type']}\n\n"
            md += f"**Size:** {doc['size_bytes']} bytes\n\n"
            md += "---\n\n"

        return md
```

---
## 3. CSV Export

**Implementation:**
```python
import pandas as pd

class CSVExportService:
    """Export documents to CSV."""

    def export_documents(
        self,
        documents: List[Dict],
    ) -> bytes:
        """Export documents to CSV."""
        df = pd.DataFrame(documents)
        return df.to_csv(index=False).encode()
```

---
## 4. JSON Export

**Implementation:**
```python
import json

class JSONExportService:
    """Export documents to JSON."""

    def export_documents(
        self,
        documents: List[Dict],
    ) -> bytes:
        """Export documents to JSON."""
        return json.dumps(documents, indent=2).encode()
```

---
## 📝 Summary

This guide covers PDF, Markdown, CSV, and JSON export implementations.

**Document Version:** 1.0
**Last Updated:** 2026-01-31
**Author:** AI-Mastery-2026
