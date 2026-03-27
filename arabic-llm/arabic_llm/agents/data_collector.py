"""
Arabic Data Collection Agent - Balygh (بليغ)

This module provides autonomous web scraping and data collection for Arabic text.
It includes:
- Planner Agent: Identifies sources and creates collection strategy
- Scraper Agent: Collects raw text from websites
- Preprocessing Agent: Cleans and normalizes collected data
- Formatter Agent: Converts to training-ready format

Based on implementation plan from llm_arabic_plan.md
"""

import os
import sys
import json
import time
import logging
import hashlib
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any, Generator
from dataclasses import dataclass, field
from datetime import datetime
import re

# HTTP and scraping
try:
    import requests
    from bs4 import BeautifulSoup
    SCRAPING_AVAILABLE = True
except ImportError:
    SCRAPING_AVAILABLE = False
    import warnings
    warnings.warn("requests/beautifulsoup4 not installed. Install with: pip install requests beautifulsoup4")

# Rate limiting
from collections import defaultdict
import threading


# ============================================================================
# LOGGING SETUP
# ============================================================================

logger = logging.getLogger(__name__)


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class SourceConfig:
    """Configuration for a data source"""
    name: str
    base_url: str
    start_url: str
    category: str
    language: str = "ar"
    max_pages: int = 100
    delay_seconds: float = 1.5
    selectors: Dict[str, str] = field(default_factory=dict)
    headers: Dict[str, str] = field(default_factory=lambda: {
        "User-Agent": "Mozilla/5.0 (compatible; BalyghBot/1.0; +https://example.com/bot-info)"
    })
    
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "base_url": self.base_url,
            "start_url": self.start_url,
            "category": self.category,
            "language": self.language,
            "max_pages": self.max_pages,
            "delay_seconds": self.delay_seconds,
            "selectors": self.selectors,
        }


@dataclass
class ScrapedDocument:
    """A document scraped from the web"""
    id: str
    url: str
    title: str
    text: str
    source: str
    category: str
    scraped_at: str = ""
    word_count: int = 0
    char_count: int = 0
    metadata: Dict = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.scraped_at:
            self.scraped_at = datetime.now().isoformat()
        if not self.word_count:
            self.word_count = len(self.text.split())
        if not self.char_count:
            self.char_count = len(self.text)
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "url": self.url,
            "title": self.title,
            "text": self.text,
            "source": self.source,
            "category": self.category,
            "scraped_at": self.scraped_at,
            "word_count": self.word_count,
            "char_count": self.char_count,
            "metadata": self.metadata,
        }
    
    def to_json(self, ensure_ascii: bool = False) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=ensure_ascii)


@dataclass
class CollectionStats:
    """Statistics about data collection"""
    sources_processed: int = 0
    pages_scraped: int = 0
    documents_collected: int = 0
    documents_failed: int = 0
    total_chars: int = 0
    total_words: int = 0
    
    start_time: str = ""
    end_time: str = ""
    
    errors: List[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            "sources_processed": self.sources_processed,
            "pages_scraped": self.pages_scraped,
            "documents_collected": self.documents_collected,
            "documents_failed": self.documents_failed,
            "total_chars": self.total_chars,
            "total_words": self.total_words,
            "avg_doc_length": round(self.total_chars / max(self.documents_collected, 1), 1),
            "start_time": self.start_time,
            "end_time": self.end_time,
        }


# ============================================================================
# RATE LIMITER
# ============================================================================

class RateLimiter:
    """
    Rate limiter for web scraping to avoid overwhelming servers.
    
    Implements per-domain rate limiting with configurable delays.
    """
    
    def __init__(self, min_delay: float = 1.0, max_requests_per_minute: int = 30):
        self.min_delay = min_delay
        self.max_rpm = max_requests_per_minute
        
        self._last_requests: Dict[str, float] = defaultdict(float)
        self._lock = threading.Lock()
    
    def wait(self, domain: str):
        """Wait if necessary to respect rate limits"""
        with self._lock:
            now = time.time()
            last_request = self._last_requests[domain]
            
            # Enforce minimum delay
            elapsed = now - last_request
            if elapsed < self.min_delay:
                sleep_time = self.min_delay - elapsed
                time.sleep(sleep_time)
            
            # Update last request time
            self._last_requests[domain] = time.time()


# ============================================================================
# SCRAPER AGENT
# ============================================================================

class ScraperAgent:
    """
    Autonomous web scraper for Arabic content.
    
    Features:
    - Respectful scraping with rate limiting
    - HTML parsing and text extraction
    - Language detection and filtering
    - Error handling and retry logic
    """
    
    def __init__(
        self,
        rate_limit_delay: float = 1.5,
        timeout_seconds: int = 15,
        max_retries: int = 3,
    ):
        """
        Initialize scraper agent.
        
        Args:
            rate_limit_delay: Minimum delay between requests (seconds)
            timeout_seconds: Request timeout
            max_retries: Maximum retry attempts
        """
        if not SCRAPING_AVAILABLE:
            raise ImportError(
                "Scraping requires requests and beautifulsoup4. "
                "Install with: pip install requests beautifulsoup4"
            )
        
        self.rate_limiter = RateLimiter(min_delay=rate_limit_delay)
        self.timeout = timeout_seconds
        self.max_retries = max_retries
        
        self.headers = {
            "User-Agent": "Mozilla/5.0 (compatible; BalyghBot/1.0; +https://example.com/bot-info)",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "ar-SA,ar;q=0.9,en-US;q=0.8,en;q=0.7",
        }
        
        self.stats = {
            "requests": 0,
            "successes": 0,
            "failures": 0,
            "retries": 0,
        }
    
    def _get_domain(self, url: str) -> str:
        """Extract domain from URL"""
        from urllib.parse import urlparse
        parsed = urlparse(url)
        return parsed.netloc
    
    def _fetch(self, url: str, source_config: SourceConfig) -> Optional[str]:
        """
        Fetch HTML from URL with retry logic.
        
        Args:
            url: URL to fetch
            source_config: Source configuration
            
        Returns:
            HTML content or None if failed
        """
        domain = self._get_domain(url)
        
        for attempt in range(self.max_retries):
            try:
                # Rate limiting
                self.rate_limiter.wait(domain)
                
                # Fetch
                response = requests.get(
                    url,
                    headers=source_config.headers or self.headers,
                    timeout=self.timeout
                )
                response.raise_for_status()
                
                self.stats["requests"] += 1
                self.stats["successes"] += 1
                
                return response.text
                
            except requests.RequestException as e:
                self.stats["failures"] += 1
                if attempt < self.max_retries - 1:
                    self.stats["retries"] += 1
                    wait_time = 2 ** attempt  # Exponential backoff
                    logger.warning(f"Request failed (attempt {attempt+1}): {e}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Request failed after {self.max_retries} attempts: {e}")
                    return None
        
        return None
    
    def _extract_text(self, html: str, source_config: SourceConfig) -> Optional[str]:
        """
        Extract text content from HTML.
        
        Args:
            html: HTML content
            source_config: Source configuration with selectors
            
        Returns:
            Extracted text or None
        """
        soup = BeautifulSoup(html, 'html.parser')
        
        # Remove unwanted elements
        for tag in soup(['script', 'style', 'nav', 'header', 'footer']):
            tag.decompose()
        
        # Try to find main content using selectors
        selectors = source_config.selectors
        
        if selectors.get('content'):
            content_el = soup.select_one(selectors['content'])
            if content_el:
                text = content_el.get_text(' ', strip=True)
                return self._clean_text(text)
        
        # Fallback: find largest text block
        paragraphs = soup.find_all('p')
        if paragraphs:
            text = '\n\n'.join(p.get_text(' ', strip=True) for p in paragraphs)
            return self._clean_text(text)
        
        # Last resort: get all text
        text = soup.get_text(' ', strip=True)
        return self._clean_text(text)
    
    def _extract_title(self, html: str, source_config: SourceConfig) -> str:
        """Extract title from HTML"""
        soup = BeautifulSoup(html, 'html.parser')
        
        # Try selectors first
        if source_config.selectors.get('title'):
            title_el = soup.select_one(source_config.selectors['title'])
            if title_el:
                return title_el.get_text(strip=True)
        
        # Fallback to title tag
        if soup.title:
            return soup.title.get_text(strip=True)
        
        # Fallback to h1
        h1 = soup.find('h1')
        if h1:
            return h1.get_text(strip=True)
        
        return "Untitled"
    
    def _clean_text(self, text: str) -> str:
        """Clean extracted text"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove very short lines
        lines = [line for line in text.split('\n') if len(line.strip()) > 20]
        
        return '\n'.join(lines).strip()
    
    def _is_arabic(self, text: str, threshold: float = 0.5) -> bool:
        """Check if text is primarily Arabic"""
        if not text:
            return False
        
        arabic_chars = sum(1 for c in text if '\u0600' <= c <= '\u06FF')
        ratio = arabic_chars / len(text)
        
        return ratio >= threshold
    
    def scrape_url(self, url: str, source_config: SourceConfig) -> Optional[ScrapedDocument]:
        """
        Scrape a single URL.
        
        Args:
            url: URL to scrape
            source_config: Source configuration
            
        Returns:
            ScrapedDocument or None
        """
        # Fetch HTML
        html = self._fetch(url, source_config)
        if not html:
            return None
        
        # Extract content
        title = self._extract_title(html, source_config)
        text = self._extract_text(html, source_config)
        
        if not text or len(text) < 100:
            logger.debug(f"Text too short from {url}: {len(text)} chars")
            return None
        
        # Language check
        if not self._is_arabic(text):
            logger.debug(f"Non-Arabic content from {url}")
            return None
        
        # Create document
        doc_id = hashlib.md5(f"{url}_{title}".encode()).hexdigest()[:12]
        
        doc = ScrapedDocument(
            id=f"scrape-{doc_id}",
            url=url,
            title=title,
            text=text,
            source=source_config.name,
            category=source_config.category,
            metadata={
                "language": "ar",
                "scraping_method": "beautifulsoup4",
            }
        )
        
        logger.info(f"Scraped: {url} ({doc.char_count} chars)")
        
        return doc
    
    def scrape_site(
        self, 
        source_config: SourceConfig,
        link_extractor: callable = None
    ) -> Generator[ScrapedDocument, None, None]:
        """
        Scrape multiple pages from a site.
        
        Args:
            source_config: Source configuration
            link_extractor: Function to extract links from HTML
            
        Yields:
            ScrapedDocument objects
        """
        # Fetch start page
        html = self._fetch(source_config.start_url, source_config)
        if not html:
            return
        
        # Extract links
        soup = BeautifulSoup(html, 'html.parser')
        
        if link_extractor:
            links = link_extractor(soup)
        else:
            # Default: extract all links from same domain
            domain = self._get_domain(source_config.start_url)
            links = []
            for a in soup.find_all('a', href=True):
                href = a['href']
                if href.startswith('http') and domain in href:
                    links.append(href)
        
        links = list(dict.fromkeys(links))[:source_config.max_pages]
        
        logger.info(f"Found {len(links)} links to scrape from {source_config.name}")
        
        # Scrape each link
        for i, link in enumerate(links, 1):
            if i > source_config.max_pages:
                break
            
            doc = self.scrape_url(link, source_config)
            if doc:
                yield doc
            
            # Delay between requests
            time.sleep(source_config.delay_seconds)


# ============================================================================
# PREPROCESSING AGENT
# ============================================================================

class PreprocessingAgent:
    """
    Preprocess scraped Arabic text.
    
    Applies the 7-stage cleaning pipeline from cleaning.py
    """
    
    def __init__(self):
        from arabic_llm.pipeline.cleaning import ArabicTextCleaner
        self.cleaner = ArabicTextCleaner()
        
        self.stats = {
            "documents_processed": 0,
            "chars_before": 0,
            "chars_after": 0,
        }
    
    def preprocess(self, doc: ScrapedDocument) -> ScrapedDocument:
        """
        Preprocess a scraped document.
        
        Args:
            doc: ScrapedDocument
            
        Returns:
            Preprocessed ScrapedDocument
        """
        self.stats["documents_processed"] += 1
        self.stats["chars_before"] += doc.char_count
        
        # Apply cleaning pipeline
        cleaned_text, operations = self.cleaner.clean(doc.text)
        
        doc.text = cleaned_text
        doc.char_count = len(cleaned_text)
        doc.word_count = len(cleaned_text.split())
        
        self.stats["chars_after"] += doc.char_count
        
        doc.metadata["cleaning_operations"] = operations
        
        logger.debug(f"Preprocessed {doc.id}: {len(operations)} operations")
        
        return doc


# ============================================================================
# FORMATTER AGENT
# ============================================================================

class FormatterAgent:
    """
    Format scraped documents for training data.
    
    Converts scraped text to instruction-answer pairs.
    """
    
    def __init__(self):
        self.stats = {
            "documents_formatted": 0,
            "examples_generated": 0,
        }
    
    def format_for_summarization(self, doc: ScrapedDocument) -> List[Dict]:
        """
        Format document for summarization training.
        
        Args:
            doc: ScrapedDocument
            
        Returns:
            List of training examples
        """
        examples = []
        
        # Create summarization example
        instruction = f"لخّص النص التالي في فقرة واحدة:\n\n{doc.text[:500]}..."
        output = f"العنوان: {doc.title}\n\nالملخص: {doc.text[:200]}..."
        
        example = {
            "instruction": instruction,
            "input": doc.text,
            "output": output,
            "role": "summarizer_ar",
            "skills": ["summarization"],
            "level": "intermediate",
            "domain": doc.category,
            "source": doc.url,
        }
        
        examples.append(example)
        self.stats["examples_generated"] += 1
        
        return examples
    
    def format_for_qa(self, doc: ScrapedDocument) -> List[Dict]:
        """
        Format document for Q&A training.
        
        Args:
            doc: ScrapedDocument
            
        Returns:
            List of training examples
        """
        examples = []
        
        # Generate simple Q&A
        instruction = f"ما الموضوع الرئيسي في النص التالي؟\n\n{doc.text[:300]}..."
        output = f"الموضوع الرئيسي هو: {doc.title}. النص يتحدث عن..."
        
        example = {
            "instruction": instruction,
            "input": doc.text,
            "output": output,
            "role": "rag_assistant",
            "skills": ["rag_grounded_answering"],
            "level": "intermediate",
            "domain": doc.category,
            "source": doc.url,
        }
        
        examples.append(example)
        self.stats["examples_generated"] += 1
        
        return examples


# ============================================================================
# DATA COLLECTION AGENT (MAIN)
# ============================================================================

class DataCollectionAgent:
    """
    Main data collection agent that coordinates all sub-agents.
    
    Usage:
        agent = DataCollectionAgent()
        agent.add_source(source_config)
        documents = agent.collect()
    """
    
    def __init__(self, output_dir: str = "datasets/collected"):
        """
        Initialize data collection agent.
        
        Args:
            output_dir: Directory to save collected data
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.sources: List[SourceConfig] = []
        self.scraper = ScraperAgent()
        self.preprocessor = PreprocessingAgent()
        self.formatter = FormatterAgent()
        
        self.stats = CollectionStats()
    
    def add_source(self, source: SourceConfig):
        """Add a source to collect from"""
        self.sources.append(source)
        logger.info(f"Added source: {source.name} ({source.base_url})")
    
    def collect(
        self,
        save_raw: bool = True,
        save_processed: bool = True,
        save_jsonl: bool = True,
    ) -> CollectionStats:
        """
        Collect data from all sources.
        
        Args:
            save_raw: Save raw scraped documents
            save_processed: Save preprocessed documents
            save_jsonl: Save as training-ready JSONL
            
        Returns:
            CollectionStats
        """
        self.stats.start_time = datetime.now().isoformat()
        
        all_docs: List[ScrapedDocument] = []
        all_examples: List[Dict] = []
        
        for source in self.sources:
            logger.info(f"Collecting from {source.name}...")
            self.stats.sources_processed += 1
            
            # Scrape
            for doc in self.scraper.scrape_site(source):
                self.stats.pages_scraped += 1
                
                # Preprocess
                processed_doc = self.preprocessor.preprocess(doc)
                all_docs.append(processed_doc)
                
                self.stats.documents_collected += 1
                self.stats.total_chars += processed_doc.char_count
                self.stats.total_words += processed_doc.word_count
                
                # Format for training
                examples = self.formatter.format_for_summarization(processed_doc)
                all_examples.extend(examples)
                
                examples = self.formatter.format_for_qa(processed_doc)
                all_examples.extend(examples)
        
        self.stats.end_time = datetime.now().isoformat()
        
        # Save results
        if save_raw:
            self._save_raw(all_docs)
        
        if save_processed:
            self._save_processed(all_docs)
        
        if save_jsonl:
            self._save_jsonl(all_examples)
        
        logger.info(f"Collection complete: {self.stats.documents_collected} documents")
        
        return self.stats
    
    def _save_raw(self, docs: List[ScrapedDocument]):
        """Save raw scraped documents"""
        output_file = self.output_dir / "raw_documents.jsonl"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for doc in docs:
                f.write(doc.to_json() + '\n')
        
        logger.info(f"Saved raw documents: {output_file}")
    
    def _save_processed(self, docs: List[ScrapedDocument]):
        """Save preprocessed documents"""
        output_file = self.output_dir / "processed_documents.jsonl"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for doc in docs:
                f.write(doc.to_json() + '\n')
        
        logger.info(f"Saved processed documents: {output_file}")
    
    def _save_jsonl(self, examples: List[Dict]):
        """Save training-ready JSONL"""
        output_file = self.output_dir / "training_examples.jsonl"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for example in examples:
                f.write(json.dumps(example, ensure_ascii=False) + '\n')
        
        logger.info(f"Saved training examples: {output_file}")


# ============================================================================
# PREDEFINED SOURCES
# ============================================================================

def get_islamic_sources() -> List[SourceConfig]:
    """Get predefined Islamic content sources"""
    return [
        # Note: These are example configurations
        # Replace with actual sources you have permission to scrape
        SourceConfig(
            name="Example Islamic Site",
            base_url="https://example-islamic-site.com",
            start_url="https://example-islamic-site.com/articles",
            category="islamic_studies",
            max_pages=50,
            delay_seconds=2.0,
            selectors={
                "content": "div.article-content",
                "title": "h1.article-title",
            }
        ),
    ]


def get_educational_sources() -> List[SourceConfig]:
    """Get predefined educational content sources"""
    return [
        SourceConfig(
            name="Example Educational Site",
            base_url="https://example-edu-site.com",
            start_url="https://example-edu-site.com/lessons",
            category="education",
            max_pages=100,
            delay_seconds=1.5,
            selectors={
                "content": "div.lesson-content",
                "title": "h1.lesson-title",
            }
        ),
    ]


# ============================================================================
# MAIN - CLI USAGE
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Arabic Data Collection Agent")
    parser.add_argument("--output-dir", default="datasets/collected", help="Output directory")
    parser.add_argument("--sources", nargs='+', default=["islamic"], help="Sources to collect from")
    parser.add_argument("--no-save-raw", action="store_true", help="Don't save raw documents")
    parser.add_argument("--no-save-processed", action="store_true", help="Don't save processed documents")
    parser.add_argument("--no-save-jsonl", action="store_true", help="Don't save JSONL")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create agent
    agent = DataCollectionAgent(output_dir=args.output_dir)
    
    # Add sources
    if "islamic" in args.sources:
        for source in get_islamic_sources():
            agent.add_source(source)
    
    if "educational" in args.sources:
        for source in get_educational_sources():
            agent.add_source(source)
    
    # Collect
    stats = agent.collect(
        save_raw=not args.no_save_raw,
        save_processed=not args.no_save_processed,
        save_jsonl=not args.no_save_jsonl,
    )
    
    # Print stats
    print("\n" + "=" * 60)
    print("Collection Statistics:")
    print("=" * 60)
    for key, value in stats.to_dict().items():
        print(f"  {key}: {value}")
