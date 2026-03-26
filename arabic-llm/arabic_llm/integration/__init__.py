"""
Arabic LLM - External Integrations

This subpackage contains integrations with external systems:
- System book datasets
- Database connections
- Lucene index support
"""

from .system_books import (
    SystemBookIntegration,
    HadithRecord,
    TafseerRecord,
    BookIndex,
)

from .databases import (
    DatabaseConnection,
    get_database_connection,
)

__all__ = [
    # System books
    "SystemBookIntegration",
    "HadithRecord",
    "TafseerRecord",
    "BookIndex",
    # Databases
    "DatabaseConnection",
    "get_database_connection",
]
