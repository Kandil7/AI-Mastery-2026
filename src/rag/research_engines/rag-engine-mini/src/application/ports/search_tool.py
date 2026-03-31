"""
Search Tool Port
================
Interface for web search providers (Searxng, Tavily, etc.).

واجهة لأدوات البحث في الويب
"""

from typing import List, Protocol, TypedDict

class SearchResult(TypedDict):
    title: str
    url: str
    content: str

class SearchToolPort(Protocol):
    def search(self, query: str, max_results: int = 5) -> List[SearchResult]:
        """Perform a web search and return results."""
        ...
