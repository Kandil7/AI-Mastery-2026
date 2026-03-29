"""
Query Construction Module

Production-ready query construction for:
- SQL generation
- Cypher (Neo4j) generation
- Metadata filters
- Query translation

Features:
- Schema-aware generation
- Validation
- Parameter binding
- Error handling
"""

from __future__ import annotations

import json
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


@dataclass
class ConstructedQuery:
    """A constructed query with metadata."""

    query: str
    query_type: str  # sql, cypher, filter, etc.
    parameters: Dict[str, Any] = field(default_factory=dict)
    tables: List[str] = field(default_factory=list)
    confidence: float = 1.0
    explanation: str = ""
    raw_response: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "query_type": self.query_type,
            "parameters": self.parameters,
            "tables": self.tables,
            "confidence": self.confidence,
            "explanation": self.explanation,
        }


@dataclass
class DatabaseSchema:
    """Database schema information."""

    tables: Dict[str, Dict[str, Any]]  # table_name -> columns
    relationships: List[Dict[str, Any]] = field(default_factory=list)
    description: str = ""

    def to_prompt(self) -> str:
        """Format schema for LLM prompt."""
        lines = ["Database Schema:"]

        for table_name, columns in self.tables.items():
            lines.append(f"\nTable: {table_name}")
            for col_name, col_info in columns.items():
                col_type = col_info.get("type", "TEXT") if isinstance(col_info, dict) else col_info
                lines.append(f"  - {col_name}: {col_type}")

        if self.relationships:
            lines.append("\nRelationships:")
            for rel in self.relationships:
                lines.append(f"  - {rel.get('description', str(rel))}")

        return "\n".join(lines)


class QueryConstructor(ABC):
    """Abstract base class for query constructors."""

    def __init__(
        self,
        llm_client: Any,
        schema: Optional[DatabaseSchema] = None,
        max_retries: int = 3,
    ) -> None:
        self.llm_client = llm_client
        self.schema = schema
        self.max_retries = max_retries

        self._stats = {
            "total_queries": 0,
            "successful": 0,
            "failed": 0,
        }

    @abstractmethod
    async def construct(self, natural_language: str, **kwargs: Any) -> ConstructedQuery:
        """Construct query from natural language."""
        pass

    async def construct_with_validation(
        self,
        natural_language: str,
        **kwargs: Any,
    ) -> ConstructedQuery:
        """Construct query with validation and retry."""
        for attempt in range(self.max_retries):
            try:
                query = await self.construct(natural_language, **kwargs)

                # Validate
                is_valid, error = await self.validate(query)

                if is_valid:
                    self._stats["successful"] += 1
                    return query
                else:
                    logger.warning(f"Query validation failed: {error}")
                    if attempt == self.max_retries - 1:
                        query.explanation = f"Validation error: {error}"
                        query.confidence = 0.5

            except Exception as e:
                logger.error(f"Query construction failed: {e}")
                if attempt == self.max_retries - 1:
                    raise

        self._stats["failed"] += 1
        self._stats["total_queries"] += 1
        return query

    @abstractmethod
    async def validate(self, query: ConstructedQuery) -> Tuple[bool, Optional[str]]:
        """Validate constructed query."""
        pass

    def get_stats(self) -> Dict[str, Any]:
        """Get constructor statistics."""
        total = self._stats["successful"] + self._stats["failed"]
        return {
            **self._stats,
            "success_rate": self._stats["successful"] / total if total > 0 else 0,
        }


class SQLConstructor(QueryConstructor):
    """
    SQL query constructor.

    Generates SQL queries from natural language with
    schema awareness and validation.
    """

    SQL_PROMPT = """You are an expert SQL query generator. Generate a SQL query based on the natural language request.

{schema_info}

Rules:
1. Only use tables and columns from the schema above
2. Use proper SQL syntax for {dialect}
3. Include only necessary columns
4. Use appropriate JOINs for related tables
5. Add comments explaining complex parts

Natural Language: {request}

Generate SQL query:"""

    def __init__(
        self,
        llm_client: Any,
        schema: Optional[DatabaseSchema] = None,
        dialect: str = "postgresql",
        read_only: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(llm_client, schema, **kwargs)
        self.dialect = dialect
        self.read_only = read_only

        self._forbidden_patterns = [
            r"\bDROP\b",
            r"\bDELETE\b",
            r"\bTRUNCATE\b",
            r"\bALTER\b",
            r"\bCREATE\b",
            r"\bINSERT\b",
            r"\bUPDATE\b",
            r"\bREPLACE\b",
        ] if read_only else []

    async def construct(self, natural_language: str, **kwargs: Any) -> ConstructedQuery:
        """Construct SQL query."""
        schema_info = self.schema.to_prompt() if self.schema else "No schema provided"

        prompt = self.SQL_PROMPT.format(
            schema_info=schema_info,
            dialect=self.dialect,
            request=natural_language,
        )

        response = await self.llm_client.generate(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=500,
        )

        sql_text = response.content if hasattr(response, 'content') else str(response)

        # Extract SQL from response
        sql = self._extract_sql(sql_text)

        # Extract tables
        tables = self._extract_tables(sql)

        return ConstructedQuery(
            query=sql,
            query_type="sql",
            tables=tables,
            explanation=self._extract_explanation(sql_text),
            raw_response=sql_text,
        )

    async def validate(self, query: ConstructedQuery) -> Tuple[bool, Optional[str]]:
        """Validate SQL query."""
        sql = query.query.upper()

        # Check for forbidden patterns
        if self.read_only:
            for pattern in self._forbidden_patterns:
                if re.search(pattern, sql):
                    return False, f"Query contains forbidden operation: {pattern}"

        # Check for basic SQL syntax
        if not any(kw in sql for kw in ["SELECT", "WITH", "EXPLAIN"]):
            return False, "Query doesn't appear to be a SELECT query"

        # Validate tables exist in schema
        if self.schema:
            for table in query.tables:
                if table not in self.schema.tables:
                    return False, f"Unknown table: {table}"

        return True, None

    def _extract_sql(self, text: str) -> str:
        """Extract SQL from LLM response."""
        # Look for SQL in code blocks
        import re
        match = re.search(r"```sql\s*([\s\S]*?)```", text, re.IGNORECASE)
        if match:
            return match.group(1).strip()

        # Look for any SQL-like content
        match = re.search(r"(SELECT[\s\S]*?)(?:;|$)", text, re.IGNORECASE)
        if match:
            return match.group(1).strip()

        # Return entire text as fallback
        return text.strip()

    def _extract_tables(self, sql: str) -> List[str]:
        """Extract table names from SQL."""
        tables = []

        # FROM clause
        from_match = re.search(r'\bFROM\s+([a-zA-Z_][a-zA-Z0-9_]*)', sql, re.IGNORECASE)
        if from_match:
            tables.append(from_match.group(1))

        # JOIN clauses
        join_matches = re.findall(r'\bJOIN\s+([a-zA-Z_][a-zA-Z0-9_]*)', sql, re.IGNORECASE)
        tables.extend(join_matches)

        return list(set(tables))

    def _extract_explanation(self, text: str) -> str:
        """Extract explanation from LLM response."""
        # Remove SQL code blocks
        import re
        text = re.sub(r"```sql\s*[\s\S]*?```", "", text, flags=re.IGNORECASE)
        text = re.sub(r"```[\s\S]*?```", "", text)

        # Get remaining text as explanation
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        return "\n".join(lines)[:500]


class CypherConstructor(QueryConstructor):
    """
    Cypher (Neo4j) query constructor.

    Generates Cypher queries for graph database operations.
    """

    CYPHER_PROMPT = """You are an expert Cypher query generator for Neo4j graph databases.

{schema_info}

Node Labels: {node_labels}
Relationship Types: {rel_types}

Rules:
1. Use proper Cypher syntax
2. Match patterns using node labels and relationship types
3. Use parameterized queries where possible
4. Include RETURN clause with relevant properties

Natural Language: {request}

Generate Cypher query:"""

    def __init__(
        self,
        llm_client: Any,
        schema: Optional[DatabaseSchema] = None,
        node_labels: Optional[List[str]] = None,
        rel_types: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(llm_client, schema, **kwargs)
        self.node_labels = node_labels or []
        self.rel_types = rel_types or []

    async def construct(self, natural_language: str, **kwargs: Any) -> ConstructedQuery:
        """Construct Cypher query."""
        schema_info = self.schema.to_prompt() if self.schema else "No schema provided"

        prompt = self.CYPHER_PROMPT.format(
            schema_info=schema_info,
            node_labels=", ".join(self.node_labels),
            rel_types=", ".join(self.rel_types),
            request=natural_language,
        )

        response = await self.llm_client.generate(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=500,
        )

        cypher_text = response.content if hasattr(response, 'content') else str(response)

        # Extract Cypher query
        cypher = self._extract_cypher(cypher_text)

        return ConstructedQuery(
            query=cypher,
            query_type="cypher",
            explanation=self._extract_explanation(cypher_text),
            raw_response=cypher_text,
        )

    async def validate(self, query: ConstructedQuery) -> Tuple[bool, Optional[str]]:
        """Validate Cypher query."""
        cypher = query.query

        # Basic syntax checks
        if not any(kw in cypher.upper() for kw in ["MATCH", "CREATE", "MERGE", "LOAD"]):
            return False, "Query missing required Cypher clause"

        # Check for balanced parentheses
        if cypher.count("(") != cypher.count(")"):
            return False, "Unbalanced parentheses"

        # Check for balanced brackets
        if cypher.count("[") != cypher.count("]"):
            return False, "Unbalanced brackets"

        return True, None

    def _extract_cypher(self, text: str) -> str:
        """Extract Cypher from LLM response."""
        import re

        # Look for code blocks
        match = re.search(r"```cypher\s*([\s\S]*?)```", text, re.IGNORECASE)
        if match:
            return match.group(1).strip()

        match = re.search(r"```[\s\S]*?```", text)
        if match:
            return match.group(0).strip("```").strip()

        return text.strip()

    def _extract_explanation(self, text: str) -> str:
        """Extract explanation from response."""
        import re
        text = re.sub(r"```[\s\S]*?```", "", text)
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        return "\n".join(lines)[:500]


class MetadataFilterConstructor(QueryConstructor):
    """
    Metadata filter constructor.

    Generates filter expressions for vector database queries.
    """

    FILTER_PROMPT = """Generate a metadata filter for a vector database query.

Available metadata fields:
{fields_info}

Operators: $eq, $ne, $in, $nin, $gt, $gte, $lt, $lte, $and, $or

Natural Language: {request}

Generate filter as JSON:"""

    def __init__(
        self,
        llm_client: Any,
        fields: Optional[Dict[str, Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(llm_client, **kwargs)
        self.fields = fields or {}

    async def construct(self, natural_language: str, **kwargs: Any) -> ConstructedQuery:
        """Construct metadata filter."""
        fields_info = "\n".join(
            f"- {name}: {info.get('type', 'any')} ({info.get('description', '')})"
            for name, info in self.fields.items()
        )

        prompt = self.FILTER_PROMPT.format(
            fields_info=fields_info,
            request=natural_language,
        )

        response = await self.llm_client.generate(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=300,
        )

        filter_text = response.content if hasattr(response, 'content') else str(response)

        # Parse JSON filter
        filter_dict = self._parse_filter(filter_text)

        return ConstructedQuery(
            query=json.dumps(filter_dict),
            query_type="metadata_filter",
            parameters=filter_dict,
            raw_response=filter_text,
        )

    async def validate(self, query: ConstructedQuery) -> Tuple[bool, Optional[str]]:
        """Validate filter."""
        try:
            filter_dict = json.loads(query.query)
            return self._validate_filter_structure(filter_dict)
        except json.JSONDecodeError as e:
            return False, f"Invalid JSON: {e}"

    def _parse_filter(self, text: str) -> Dict[str, Any]:
        """Parse filter from LLM response."""
        import re

        # Look for JSON
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

        # Try to parse entire text
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {}

    def _validate_filter_structure(self, filter_dict: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Validate filter structure."""
        if not isinstance(filter_dict, dict):
            return False, "Filter must be a JSON object"

        # Check for valid operators
        valid_operators = {"$eq", "$ne", "$in", "$nin", "$gt", "$gte", "$lt", "$lte", "$and", "$or"}

        def check_value(value: Any) -> Tuple[bool, Optional[str]]:
            if isinstance(value, dict):
                for key, val in value.items():
                    if key.startswith("$"):
                        if key not in valid_operators:
                            return False, f"Invalid operator: {key}"
                    else:
                        valid, err = check_value(val)
                        if not valid:
                            return False, err
            elif isinstance(value, list):
                for item in value:
                    valid, err = check_value(item)
                    if not valid:
                        return False, err
            return True, None

        return check_value(filter_dict)


class QueryTranslator:
    """
    Query translator for multi-language and multi-format support.

    Translates queries between:
    - Natural languages
    - Query formats
    - Domain-specific languages
    """

    TRANSLATE_PROMPT = """Translate the following query from {source_lang} to {target_lang}.
Preserve the meaning and intent exactly.

{source_lang} Query: {query}

{target_lang} Query:"""

    def __init__(self, llm_client: Any) -> None:
        self.llm_client = llm_client

    async def translate(
        self,
        query: str,
        source_lang: str = "en",
        target_lang: str = "es",
    ) -> str:
        """Translate query between languages."""
        prompt = self.TRANSLATE_PROMPT.format(
            source_lang=source_lang,
            target_lang=target_lang,
            query=query,
        )

        response = await self.llm_client.generate(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
        )

        return response.content if hasattr(response, 'content') else str(response)

    async def rewrite(
        self,
        query: str,
        style: str = "formal",  # formal, casual, technical, simple
    ) -> str:
        """Rewrite query in different style."""
        prompt = f"""Rewrite the following query in a {style} style.
Preserve the core meaning but adjust the tone and vocabulary.

Original: {query}

Rewritten:"""

        response = await self.llm_client.generate(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )

        return response.content if hasattr(response, 'content') else str(response)

    async def expand_query(
        self,
        query: str,
        num_variations: int = 3,
    ) -> List[str]:
        """Generate query variations for better retrieval."""
        prompt = f"""Generate {num_variations} different ways to ask the same question.
Each variation should use different wording but have the same meaning.

Original: {query}

Variations (one per line):"""

        response = await self.llm_client.generate(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
        )

        content = response.content if hasattr(response, 'content') else str(response)
        variations = [line.strip() for line in content.split("\n") if line.strip()]

        return variations[:num_variations]

    async def decompose_query(
        self,
        query: str,
    ) -> List[str]:
        """Decompose complex query into sub-queries."""
        prompt = f"""Break down this complex question into simpler sub-questions.
Each sub-question should be answerable independently.

Question: {query}

Sub-questions (one per line):"""

        response = await self.llm_client.generate(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )

        content = response.content if hasattr(response, 'content') else str(response)
        sub_queries = [line.strip() for line in content.split("\n") if line.strip()]

        return sub_queries


class QueryRouter:
    """
    Routes queries to appropriate constructors.

    Determines query type and routes to correct constructor.
    """

    def __init__(
        self,
        sql_constructor: Optional[SQLConstructor] = None,
        cypher_constructor: Optional[CypherConstructor] = None,
        filter_constructor: Optional[MetadataFilterConstructor] = None,
    ) -> None:
        self.sql_constructor = sql_constructor
        self.cypher_constructor = cypher_constructor
        self.filter_constructor = filter_constructor

        self._llm_client = None
        if sql_constructor:
            self._llm_client = sql_constructor.llm_client

    async def route(self, natural_language: str) -> ConstructedQuery:
        """Route and construct query."""
        # First determine query type
        query_type = await self._classify_query(natural_language)

        logger.info(f"Routed query to type: {query_type}")

        if query_type == "sql" and self.sql_constructor:
            return await self.sql_constructor.construct(natural_language)
        elif query_type == "cypher" and self.cypher_constructor:
            return await self.cypher_constructor.construct(natural_language)
        elif query_type == "filter" and self.filter_constructor:
            return await self.filter_constructor.construct(natural_language)
        else:
            raise ValueError(f"No constructor available for query type: {query_type}")

    async def _classify_query(self, query: str) -> str:
        """Classify query type."""
        if not self._llm_client:
            return self._heuristic_classify(query)

        prompt = f"""Classify this query into one of: sql, cypher, filter, or natural.

Query: {query}

Type (just output the type name):"""

        response = await self.llm_client.generate(
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )

        query_type = response.content.strip().lower() if hasattr(response, 'content') else str(response).lower()

        if query_type in ["sql", "cypher", "filter", "natural"]:
            return query_type

        return self._heuristic_classify(query)

    def _heuristic_classify(self, query: str) -> str:
        """Heuristic query classification."""
        query_lower = query.lower()

        # SQL indicators
        sql_keywords = ["select", "from", "where", "join", "group by", "order by"]
        if any(kw in query_lower for kw in sql_keywords):
            return "sql"

        # Cypher indicators
        cypher_patterns = ["match", "return", "create", "merge", "where"]
        if any(kw in query_lower for kw in cypher_patterns):
            return "cypher"

        # Filter indicators
        filter_keywords = ["filter", "where", "category", "type", "status"]
        if any(kw in query_lower for kw in filter_keywords):
            return "filter"

        return "natural"
