"""
Arabic LLM - External Integrations

This subpackage contains integrations with external systems:
- System book datasets (hadith, tafseer, trajim)
- Database connections
- Lucene index support (TODO)
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
    query_books_db,
    get_book_by_id,
    get_books_by_category,
    get_all_categories,
)

# Lucene support (TODO)
# from .lucene import (
#     LuceneIndex,
#     search_index,
# )

__all__ = [
    # System books
    "SystemBookIntegration",
    "HadithRecord",
    "TafseerRecord",
    "BookIndex",
    # Databases
    "DatabaseConnection",
    "get_database_connection",
    "query_books_db",
    "get_book_by_id",
    "get_books_by_category",
    "get_all_categories",
    # Lucene (TODO)
    # "LuceneIndex",
    # "search_index",
]
