"""
Tavily Web Search Adapter
=========================
Implementation of SearchToolPort using Tavily API.

تنفيذ أداة البحث الويب باستخدام Tavily
"""

from typing import List
import requests
import structlog
from src.application.ports.search_tool import SearchToolPort, SearchResult

log = structlog.get_logger()

class TavilySearchAdapter:
    """
    Search adapter for Tavily API.
    
    قرار التصميم: استخدام Tavily لأنه مصمم خصيصاً للـ LLMs (Returns clean context)
    """

    def __init__(self, api_key: str):
        self._api_key = api_key
        self._url = "https://api.tavily.com/search"

    def search(self, query: str, max_results: int = 5) -> List[SearchResult]:
        """
        Perform search via Tavily.
        """
        if not self._api_key:
            log.warning("search_skipped", reason="no_api_key")
            return []

        payload = {
            "api_key": self._api_key,
            "query": query,
            "search_depth": "smart",
            "max_results": max_results,
        }
        
        try:
            response = requests.post(self._url, json=payload, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            results = []
            for r in data.get("results", []):
                results.append(SearchResult(
                    title=r.get("title", ""),
                    url=r.get("url", ""),
                    content=r.get("content", ""),
                ))
            
            log.info("web_search_complete", query=query, results_count=len(results))
            return results
            
        except Exception as e:
            log.error("web_search_failed", error=str(e))
            return []
