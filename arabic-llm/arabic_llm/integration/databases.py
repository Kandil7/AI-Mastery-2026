"""
Arabic LLM - Database Connection Utilities

Utilities for connecting to and managing database connections.
"""

import sqlite3
from pathlib import Path
from typing import Optional, Dict, Any, List
from contextlib import contextmanager


@contextmanager
def get_database_connection(db_path: str):
    """
    Context manager for database connections.
    
    Args:
        db_path: Path to SQLite database
        
    Yields:
        SQLite connection with row factory
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    
    try:
        yield conn
    finally:
        conn.close()


class DatabaseConnection:
    """Database connection manager"""
    
    def __init__(self, db_path: str):
        """
        Initialize database connection.
        
        Args:
            db_path: Path to SQLite database
        """
        self.db_path = Path(db_path)
        self.conn: Optional[sqlite3.Connection] = None
    
    def connect(self) -> sqlite3.Connection:
        """
        Connect to database.
        
        Returns:
            SQLite connection
        """
        if not self.db_path.exists():
            raise FileNotFoundError(f"Database not found: {self.db_path}")
        
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        return self.conn
    
    def disconnect(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            self.conn = None
    
    def query(self, sql: str, params: tuple = ()) -> List[Dict]:
        """
        Execute SELECT query.
        
        Args:
            sql: SQL query
            params: Query parameters
            
        Returns:
            List of result dictionaries
        """
        if not self.conn:
            self.connect()
        
        cursor = self.conn.cursor()
        cursor.execute(sql, params)
        
        rows = cursor.fetchall()
        return [dict(row) for row in rows]
    
    def execute(self, sql: str, params: tuple = ()) -> int:
        """
        Execute INSERT/UPDATE/DELETE query.
        
        Args:
            sql: SQL query
            params: Query parameters
            
        Returns:
            Number of affected rows
        """
        if not self.conn:
            self.connect()
        
        cursor = self.conn.cursor()
        cursor.execute(sql, params)
        self.conn.commit()
        
        return cursor.rowcount
    
    def get_tables(self) -> List[str]:
        """
        Get list of tables in database.
        
        Returns:
            List of table names
        """
        sql = "SELECT name FROM sqlite_master WHERE type='table'"
        results = self.query(sql)
        return [row['name'] for row in results]
    
    def __enter__(self):
        """Context manager entry"""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.disconnect()


def query_books_db(query: str, params: tuple = ()) -> List[Dict]:
    """
    Query books.db metadata database.
    
    Args:
        query: SQL query
        params: Query parameters
        
    Returns:
        List of result dictionaries
    """
    # Try to find books.db
    db_paths = [
        Path("datasets/metadata/books.db"),
        Path("../datasets/metadata/books.db"),
    ]
    
    for db_path in db_paths:
        if db_path.exists():
            with get_database_connection(str(db_path)) as conn:
                cursor = conn.cursor()
                cursor.execute(query, params)
                rows = cursor.fetchall()
                return [dict(row) for row in rows]
    
    raise FileNotFoundError("books.db not found in expected locations")


def get_book_by_id(book_id: int) -> Optional[Dict]:
    """
    Get book by ID from metadata database.
    
    Args:
        book_id: Book ID
        
    Returns:
        Book metadata dictionary or None
    """
    query = "SELECT * FROM books WHERE id = ?"
    results = query_books_db(query, (book_id,))
    
    if results:
        return results[0]
    return None


def get_books_by_category(category_name: str) -> List[Dict]:
    """
    Get books by category name.
    
    Args:
        category_name: Category name
        
    Returns:
        List of book metadata dictionaries
    """
    query = """
        SELECT b.* FROM books b
        JOIN categories c ON b.cat_id = c.id
        WHERE c.name = ?
    """
    return query_books_db(query, (category_name,))


def get_all_categories() -> List[Dict]:
    """
    Get all categories.
    
    Returns:
        List of category dictionaries
    """
    return query_books_db("SELECT * FROM categories ORDER BY name")
