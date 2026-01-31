"""
Export Services
===============
Services for exporting documents in various formats.

خدمات تصدير المستندات بتنسيقات مختلفة
"""

from typing import List, Dict
import pandas as pd
import json
import io


class PDFExportService:
    """Export documents to PDF using ReportLab."""

    def export_documents(
        self,
        documents: List[Dict],
        title: str = "Document Export",
    ) -> bytes:
        """
        Export documents to PDF.

        Args:
            documents: List of documents with metadata
            title: Document title

        Returns:
            PDF file content as bytes
        """
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas
        from reportlab.lib.units import inch

        # Create PDF canvas to BytesIO
        buffer = io.BytesIO()
        pdf = canvas.Canvas(buffer, pagesize=letter)

        # Add title
        pdf.setFont("Helvetica-Bold", 18)
        pdf.drawString(1 * inch, 10 * inch, title)

        # Add documents
        y = 9 * inch
        for doc in documents:
            pdf.setFont("Helvetica", 12)
            filename = doc.get("filename", doc.get("id", "Unknown"))
            pdf.drawString(1 * inch, y, f"• {filename}")
            y -= 0.3 * inch

            # Add metadata
            pdf.setFont("Helvetica", 9)
            pdf.drawString(1.5 * inch, y, f"  ID: {doc.get('id', 'N/A')}")
            y -= 0.2 * inch
            pdf.drawString(1.5 * inch, y, f"  Status: {doc.get('status', 'N/A')}")
            y -= 0.2 * inch

            # New page if needed
            if y < 1 * inch:
                pdf.showPage()
                y = 10 * inch

        pdf.save()
        return buffer.getvalue()


class MarkdownExportService:
    """Export documents to Markdown."""

    def export_documents(
        self,
        documents: List[Dict],
        title: str = "Document Export",
    ) -> str:
        """
        Export documents to Markdown.

        Args:
            documents: List of documents with metadata
            title: Document title

        Returns:
            Markdown content as string
        """
        md = f"# {title}\n\n"

        for doc in documents:
            md += f"## {doc.get('filename', doc.get('id', 'Unknown'))}\n\n"
            md += f"**Content Type:** {doc.get('content_type', 'N/A')}\n\n"
            md += f"**Size:** {doc.get('size_bytes', 0)} bytes\n\n"
            md += f"**Status:** {doc.get('status', 'unknown')}\n\n"
            md += f"**Created At:** {doc.get('created_at', 'N/A')}\n\n"
            md += "---\n\n"

        return md


class CSVExportService:
    """Export documents to CSV using pandas."""

    def export_documents(
        self,
        documents: List[Dict],
    ) -> bytes:
        """
        Export documents to CSV.

        Args:
            documents: List of documents with metadata

        Returns:
            CSV file content as bytes
        """
        df = pd.DataFrame(documents)
        return df.to_csv(index=False).encode()


class JSONExportService:
    """Export documents to JSON."""

    def export_documents(
        self,
        documents: List[Dict],
    ) -> bytes:
        """
        Export documents to JSON.

        Args:
            documents: List of documents with metadata

        Returns:
            JSON file content as bytes
        """
        return json.dumps(documents, indent=2, default=str).encode()
