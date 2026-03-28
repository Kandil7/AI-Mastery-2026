"""
Data Ingestion Module

Handles ingestion from multiple sources:
- Files (PDF, DOCX, TXT, MD, HTML, JSON)
- APIs (REST, GraphQL)
- Databases (SQLite, PostgreSQL, MySQL)
- Webhooks
"""

from .multi_source_ingestion import (
    MultiSourceIngestionPipeline,
    DataSource,
    DataSourceType,
    ConnectorType,
    Document,
    DocumentParser,
    FileConnector,
    APIConnector,
    DatabaseConnector,
    WebhookConnector,
    create_file_source,
    create_api_source,
    create_database_source,
)

__all__ = [
    # Ingestion Pipelines
    "MultiSourceIngestionPipeline",
    # Models
    "DataSource",
    "DataSourceType",
    "Document",
    "DocumentParser",
    # Connectors
    "FileConnector",
    "APIConnector",
    "DatabaseConnector",
    "WebhookConnector",
    # Factories
    "create_file_source",
    "create_api_source",
    "create_database_source",
]
