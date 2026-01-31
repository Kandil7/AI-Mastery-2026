# GraphQL API for RAG Engine
# ===============================
# GraphQL endpoints for flexible data queries.

# نقاط نهاية GraphQL لـ RAG Engine

import strawberry
import logging
import asyncio
from typing import List, Optional, AsyncGenerator
from datetime import datetime
from enum import Enum

log = logging.getLogger(__name__)

from src.api.subscriptions.document_subscriptions import (
    DocumentSubscription,
    QueryProgressSubscription,
    ChatUpdateSubscription,
)


class DocumentStatus(str, Enum):
    """Document status enum."""

    CREATED = "created"
    INDEXED = "indexed"
    FAILED = "failed"


class QuerySortBy(str, Enum):
    """Query sorting options."""

    CREATED = "created"
    UPDATED = "updated"
    FILENAME = "filename"
    SIZE = "size"


class ExperimentStatus(str, Enum):
    """A/B test experiment status."""

    DRAFT = "draft"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"


@strawberry.type
class DocumentType:
    """Document GraphQL type."""

    id: strawberry.ID
    filename: str
    content_type: str
    size_bytes: int
    status: DocumentStatus
    created_at: datetime
    updated_at: Optional[datetime]


@strawberry.type
class AnswerType:
    """Answer GraphQL type with sources."""

    text: str
    sources: List[str]
    retrieval_k: int
    embed_ms: Optional[int]
    search_ms: Optional[int]
    llm_ms: Optional[int]


@strawberry.type
class ChatSessionType:
    """Chat session GraphQL type."""

    id: strawberry.ID
    title: Optional[str]
    created_at: datetime


@strawberry.type
class QueryHistoryItemType:
    """Query history item GraphQL type."""

    question: str
    answer: str
    sources: List[str]
    timestamp: datetime


@strawberry.type
class FacetType:
    """Search facet GraphQL type."""

    name: str
    count: int


@strawberry.type
class WebhookType:
    """Webhook GraphQL type."""

    id: strawberry.ID
    url: str
    events: List[str]
    active: bool


@strawberry.type
class HealthCheckType:
    """Health check result GraphQL type."""

    service: str
    status: str  # "healthy" | "degraded" | "unhealthy"
    latency_ms: Optional[float]
    message: Optional[str]


@strawberry.type
class HealthCheckResponse:
    """Overall health check response."""

    status: str  # "healthy" | "degraded" | "unhealthy"
    checks: List[HealthCheckType]


@strawberry.type
class SearchResultType:
    """Search result with facets."""

    results: List[DocumentType]
    total: int
    facets: Optional[List[FacetType]]


@strawberry.type
class Query:
    """Root query type for GraphQL API."""

    @strawberry.field
    def documents(
        self,
        info,
        limit: int = 20,
        offset: int = 0,
        status: Optional[DocumentStatus] = None,
    ) -> List[DocumentType]:
        """
        Query documents with pagination and filtering.

        Args:
            info: GraphQL execution context
            limit: Max results (default: 20, max: 100)
            offset: Pagination offset (default: 0)
            status: Filter by status (optional)

        Returns:
            List of documents

        الاستعلام عن المستندات مع الترحيل والتصفية
        """
        # Validate inputs
        if limit < 0:
            raise ValueError("limit must be non-negative")
        if limit > 100:
            limit = 100
        if offset < 0:
            raise ValueError("offset must be non-negative")

        # Get tenant ID from request context
        from src.api.v1.deps import get_tenant_id

        request = info.context.get("request")
        if not request:
            return []

        tenant_id = get_tenant_id(request)

        # Get document repository from context
        doc_repo = info.context.get("doc_repo")
        if not doc_repo:
            return []

        # Query database
        documents = doc_repo.list_documents(
            tenant_id=tenant_id,
            limit=limit,
            offset=offset,
        )

        # Convert domain entities to GraphQL types
        result = []
        for doc in documents:
            # Filter by status if provided
            if status and doc.status.value != status.value:
                continue

            result.append(
                DocumentType(
                    id=doc.document_id,
                    filename=doc.filename,
                    content_type=doc.content_type,
                    size_bytes=doc.size_bytes,
                    status=DocumentStatus(doc.status.value),
                    created_at=doc.created_at,
                    updated_at=doc.updated_at,
                )
            )

        return result

    @strawberry.field
    def document(
        self,
        info,
        document_id: strawberry.ID,
    ) -> Optional[DocumentType]:
        """
        Get a single document by ID.

        Args:
            info: GraphQL execution context
            document_id: Document ID to fetch

        Returns:
            Document or None if not found

        الحصول على مستند واحد بالمعرف
        """
        # Validate input
        if not document_id:
            raise ValueError("document_id is required")

        # Get tenant ID from request context
        from src.api.v1.deps import get_tenant_id

        request = info.context.get("request")
        if not request:
            return None

        tenant_id = get_tenant_id(request)

        # Get document repository from context
        doc_repo = info.context.get("doc_repo")
        if not doc_repo:
            return None

        # Query database
        doc = doc_repo.find_by_id(document_id=str(document_id))

        # Check if document exists and belongs to tenant
        if not doc:
            return None

        if hasattr(doc, "tenant_id") and doc.tenant_id != tenant_id:
            return None

        # Convert to GraphQL type
        return DocumentType(
            id=doc.document_id,
            filename=doc.filename,
            content_type=doc.content_type,
            size_bytes=doc.size_bytes,
            status=DocumentStatus(doc.status.value),
            created_at=doc.created_at,
            updated_at=doc.updated_at,
        )

    @strawberry.field
    def search_documents(
        self,
        info,
        query: str,
        k: int = 10,
        sort_by: QuerySortBy = QuerySortBy.CREATED,
        limit: int = 20,
        offset: int = 0,
    ) -> SearchResultType:
        """
        Search documents with full-text search.

        Args:
            info: GraphQL execution context
            query: Search query (required)
            k: Number of results (default: 10)
            sort_by: Sort order (default: CREATED)
            limit: Max results (default: 20)
            offset: Pagination offset (default: 0)

        Returns:
            Search results with facets

        البحث عن المستندات بالبحث الكامل
        """
        # Validate inputs
        if not query or not query.strip():
            raise ValueError("query is required")

        if k < 1 or k > 100:
            raise ValueError("k must be between 1 and 100")

        if limit < 1 or limit > 100:
            raise ValueError("limit must be between 1 and 100")

        # Get dependencies from context
        from src.api.v1.deps import get_tenant_id

        request = info.context.get("request")
        if not request:
            return SearchResultType(results=[], total=0, facets=None)

        tenant_id = get_tenant_id(request)
        search_service = info.context.get("search_service")
        doc_repo = info.context.get("doc_repo")

        # Perform search
        if search_service:
            results = search_service.search(
                tenant_id=tenant_id,
                query=query,
                k=k,
                sort_by=sort_by.value,
            )

            graphql_results = [
                DocumentType(
                    id=doc["id"],
                    filename=doc["filename"],
                    content_type=doc["content_type"],
                    size_bytes=doc["size_bytes"],
                    status=DocumentStatus(doc["status"]),
                    created_at=doc["created_at"],
                    updated_at=doc.get("updated_at"),
                )
                for doc in results["items"][:limit]
            ]

            facets = []
            for status, count in results.get("facets", {}).get("status", {}).items():
                facets.append(FacetType(name=status, count=count))

            return SearchResultType(
                results=graphql_results,
                total=results["total"],
                facets=facets,
            )
        elif doc_repo:
            # Fallback: simple document list (no search service)
            docs = doc_repo.list_documents(tenant_id=tenant_id, limit=limit, offset=offset)

            return SearchResultType(
                results=[
                    DocumentType(
                        id=doc.document_id,
                        filename=doc.filename,
                        content_type=doc.content_type,
                        size_bytes=doc.size_bytes,
                        status=DocumentStatus(doc.status.value),
                        created_at=doc.created_at,
                        updated_at=doc.updated_at,
                    )
                    for doc in docs
                ],
                total=len(docs),
                facets=None,
            )

        return SearchResultType(results=[], total=0, facets=None)

    @strawberry.field
    def chat_sessions(
        self,
        info,
        limit: int = 20,
        offset: int = 0,
    ) -> List[ChatSessionType]:
        """
        Query chat sessions with pagination.

        Args:
            info: GraphQL execution context
            limit: Max results (default: 20)
            offset: Pagination offset (default: 0)

        Returns:
            List of chat sessions

        الاستعلام عن جلسات المحادثة مع الترحيل
        """
        # Validate inputs
        if limit < 0:
            raise ValueError("limit must be non-negative")
        if limit > 100:
            limit = 100
        if offset < 0:
            raise ValueError("offset must be non-negative")

        # Get tenant ID from request context
        from src.api.v1.deps import get_tenant_id

        request = info.context.get("request")
        if not request:
            return []

        tenant_id = get_tenant_id(request)

        # Get chat repository from context
        chat_repo = info.context.get("chat_repo")
        if not chat_repo:
            return []

        # Query database
        sessions = chat_repo.list_sessions(
            tenant_id=tenant_id,
            limit=limit,
        )

        # Convert to GraphQL types
        return [
            ChatSessionType(
                id=session.session_id,
                title=session.title,
                created_at=session.created_at,
            )
            for session in sessions
        ]

    @strawberry.field
    def chat_session(
        self,
        info,
        session_id: strawberry.ID,
    ) -> Optional[ChatSessionType]:
        """
        Get a single chat session by ID.

        Args:
            info: GraphQL execution context
            session_id: Session ID to fetch

        Returns:
            Chat session or None if not found

        الحصول على جلسة محادثة واحدة بالمعرف
        """
        # Validate input
        if not session_id:
            raise ValueError("session_id is required")

        # Get tenant ID from request context
        from src.api.v1.deps import get_tenant_id

        request = info.context.get("request")
        if not request:
            return None

        tenant_id = get_tenant_id(request)

        # Get chat repository from context
        chat_repo = info.context.get("chat_repo")
        if not chat_repo:
            return None

        # Query database
        session = chat_repo.get_session(
            tenant_id=tenant_id,
            session_id=str(session_id),
        )

        # Check if session exists and belongs to tenant
        if not session:
            return None

        # Convert to GraphQL type
        return ChatSessionType(
            id=session.session_id,
            title=session.title,
            created_at=session.created_at,
        )

    @strawberry.field
    def query_history(
        self,
        info,
        limit: int = 50,
        offset: int = 0,
    ) -> List[QueryHistoryItemType]:
        """
        Get query history with pagination.

        Args:
            info: GraphQL execution context
            limit: Max results (default: 50)
            offset: Pagination offset (default: 0)

        Returns:
            List of query history items

        الحصول على تاريخ الاستعلامات مع الترحيل
        """
        # Validate inputs
        if limit < 0:
            raise ValueError("limit must be non-negative")
        if limit > 100:
            limit = 100
        if offset < 0:
            raise ValueError("offset must be non-negative")

        # Get tenant ID from request context
        from src.api.v1.deps import get_tenant_id

        request = info.context.get("request")
        if not request:
            return []

        tenant_id = get_tenant_id(request)

        # Get query history repository from context
        query_repo = info.context.get("query_history_repo")

        # Query database (if repository available)
        if query_repo:
            history = query_repo.list_queries(
                tenant_id=tenant_id,
                limit=limit,
                offset=offset,
            )

            # Convert to GraphQL types
            return [
                QueryHistoryItemType(
                    question=item.question,
                    answer=item.answer,
                    sources=item.sources,
                    timestamp=item.timestamp,
                )
                for item in history
            ]

        return []

    @strawberry.field
    def health_check(self, info) -> HealthCheckResponse:
        """
        Check health of all system dependencies.

        Args:
            info: GraphQL execution context

        Returns:
            Health check response with status for each service

        التحقق من صحة جميع تبعيات النظام
        """
        import time

        checks = []
        overall_status = "healthy"

        # 1. Database Health Check
        db_check = self._check_database(info)
        checks.append(db_check)
        if db_check.status != "healthy":
            overall_status = "degraded" if db_check.status == "degraded" else "unhealthy"

        # 2. Redis Health Check
        redis_check = self._check_redis(info)
        checks.append(redis_check)
        if redis_check.status != "healthy":
            overall_status = "degraded" if redis_check.status == "degraded" else "unhealthy"

        # 3. Vector Store (Qdrant) Health Check
        vector_check = self._check_vector_store(info)
        checks.append(vector_check)
        if vector_check.status != "healthy":
            overall_status = "degraded" if vector_check.status == "degraded" else "unhealthy"

        # 4. LLM Health Check
        llm_check = self._check_llm(info)
        checks.append(llm_check)
        if llm_check.status != "healthy":
            overall_status = "degraded" if llm_check.status == "degraded" else "unhealthy"

        # 5. File Storage Health Check
        storage_check = self._check_storage(info)
        checks.append(storage_check)
        if storage_check.status != "healthy":
            overall_status = "degraded" if storage_check.status == "degraded" else "unhealthy"

        return HealthCheckResponse(
            status=overall_status,
            checks=checks,
        )

    def _check_database(self, info) -> HealthCheckType:
        """Check database connectivity and performance."""
        import time

        db_repo = info.context.get("db_repo")
        if not db_repo:
            return HealthCheckType(
                service="database",
                status="unhealthy",
                latency_ms=None,
                message="Database repository not available",
            )

        start_time = time.time()
        try:
            # Try a simple query
            result = db_repo.ping()
            latency_ms = (time.time() - start_time) * 1000

            if latency_ms > 500:
                return HealthCheckType(
                    service="database",
                    status="degraded",
                    latency_ms=latency_ms,
                    message="Database slow but responding",
                )
            return HealthCheckType(
                service="database",
                status="healthy",
                latency_ms=latency_ms,
                message=None,
            )
        except Exception as e:
            return HealthCheckType(
                service="database",
                status="unhealthy",
                latency_ms=(time.time() - start_time) * 1000,
                message=str(e),
            )

    def _check_redis(self, info) -> HealthCheckType:
        """Check Redis connectivity and performance."""
        import time

        redis_client = info.context.get("redis_client")
        if not redis_client:
            return HealthCheckType(
                service="redis",
                status="degraded",
                latency_ms=None,
                message="Redis not configured (using in-memory fallback)",
            )

        start_time = time.time()
        try:
            # Try a PING command
            result = redis_client.ping()
            latency_ms = (time.time() - start_time) * 1000

            if not result:
                return HealthCheckType(
                    service="redis",
                    status="unhealthy",
                    latency_ms=latency_ms,
                    message="Redis ping failed",
                )
            if latency_ms > 100:
                return HealthCheckType(
                    service="redis",
                    status="degraded",
                    latency_ms=latency_ms,
                    message="Redis slow but responding",
                )
            return HealthCheckType(
                service="redis",
                status="healthy",
                latency_ms=latency_ms,
                message=None,
            )
        except Exception as e:
            return HealthCheckType(
                service="redis",
                status="unhealthy",
                latency_ms=(time.time() - start_time) * 1000,
                message=str(e),
            )

    def _check_vector_store(self, info) -> HealthCheckType:
        """Check Qdrant vector store connectivity and performance."""
        import time

        vector_store = info.context.get("vector_store")
        if not vector_store:
            return HealthCheckType(
                service="vector_store",
                status="unhealthy",
                latency_ms=None,
                message="Vector store not available",
            )

        start_time = time.time()
        try:
            # Try a simple health check (if available)
            if hasattr(vector_store, "health_check"):
                result = vector_store.health_check()
            else:
                # Fallback: try a simple search
                from src.domain.entities import TenantId
                import numpy as np

                dummy_vector = np.random.randn(768).tolist()
                vector_store.search_scored(
                    query_vector=dummy_vector,
                    tenant_id=TenantId("health-check"),
                    top_k=1,
                )
                result = True

            latency_ms = (time.time() - start_time) * 1000

            if latency_ms > 500:
                return HealthCheckType(
                    service="vector_store",
                    status="degraded",
                    latency_ms=latency_ms,
                    message="Vector store slow but responding",
                )
            return HealthCheckType(
                service="vector_store",
                status="healthy",
                latency_ms=latency_ms,
                message=None,
            )
        except Exception as e:
            return HealthCheckType(
                service="vector_store",
                status="unhealthy",
                latency_ms=(time.time() - start_time) * 1000,
                message=str(e),
            )

    def _check_llm(self, info) -> HealthCheckType:
        """Check LLM service connectivity and performance."""
        import time

        llm = info.context.get("llm")
        if not llm:
            return HealthCheckType(
                service="llm",
                status="unhealthy",
                latency_ms=None,
                message="LLM not available",
            )

        start_time = time.time()
        try:
            # Try a simple generation
            result = llm.generate("Hi", temperature=0.0, max_tokens=5)
            latency_ms = (time.time() - start_time) * 1000

            if not result:
                return HealthCheckType(
                    service="llm",
                    status="unhealthy",
                    latency_ms=latency_ms,
                    message="LLM returned empty response",
                )
            if latency_ms > 5000:
                return HealthCheckType(
                    service="llm",
                    status="degraded",
                    latency_ms=latency_ms,
                    message="LLM slow but responding",
                )
            return HealthCheckType(
                service="llm",
                status="healthy",
                latency_ms=latency_ms,
                message=None,
            )
        except Exception as e:
            return HealthCheckType(
                service="llm",
                status="unhealthy",
                latency_ms=(time.time() - start_time) * 1000,
                message=str(e),
            )

    def _check_storage(self, info) -> HealthCheckType:
        """Check file storage connectivity."""
        import time

        storage = info.context.get("file_storage")
        if not storage:
            return HealthCheckType(
                service="file_storage",
                status="degraded",
                latency_ms=None,
                message="File storage not configured (using local fallback)",
            )

        start_time = time.time()
        try:
            # Try a simple read/write
            test_content = b"health-check"
            test_path = storage.store_file(
                tenant_id="health-check",
                filename=".health-check",
                content=test_content,
            )
            read_content = storage.read_file(test_path)

            latency_ms = (time.time() - start_time) * 1000

            if read_content != test_content:
                return HealthCheckType(
                    service="file_storage",
                    status="unhealthy",
                    latency_ms=latency_ms,
                    message="Storage read/write mismatch",
                )
            if latency_ms > 1000:
                return HealthCheckType(
                    service="file_storage",
                    status="degraded",
                    latency_ms=latency_ms,
                    message="Storage slow but responding",
                )
            return HealthCheckType(
                service="file_storage",
                status="healthy",
                latency_ms=latency_ms,
                message=None,
            )
        except Exception as e:
            return HealthCheckType(
                service="file_storage",
                status="unhealthy",
                latency_ms=(time.time() - start_time) * 1000,
                message=str(e),
            )


@strawberry.type
class ExperimentVariantType:
    """A/B test variant GraphQL type."""

    id: strawberry.ID
    name: str
    allocation: float
    config: List[str]


@strawberry.type
class ExperimentMetricType:
    """A/B test metric GraphQL type."""

    metric_name: str
    metric_value: float
    recorded_at: datetime


@strawberry.type
class ExperimentType:
    """A/B test experiment GraphQL type."""

    id: strawberry.ID
    name: str
    description: str
    status: ExperimentStatus
    created_at: datetime
    started_at: Optional[datetime]
    ended_at: Optional[datetime]
    variants: List[ExperimentVariantType]


@strawberry.type
class ExperimentResultsType:
    """A/B test results GraphQL type."""

    experiment: ExperimentType
    metrics: List[ExperimentMetricType]
    significant: bool
    summary: str

    @strawberry.field
    def experiments(
        self,
        info,
        status: Optional[ExperimentStatus] = None,
        limit: int = 20,
    ) -> List[ExperimentType]:
        """
        Query A/B testing experiments.

        Args:
            info: GraphQL execution context
            status: Filter by experiment status (optional)
            limit: Max results (default: 20)

        Returns:
            List of experiments

        الاستعلام عن تجارب A/B
        """
        # Validate inputs
        if limit < 1 or limit > 100:
            raise ValueError("limit must be between 1 and 100")

        # Get tenant ID from request context
        from src.api.v1.deps import get_tenant_id

        request = info.context.get("request")
        if not request:
            return []

        tenant_id = get_tenant_id(request)

        # Get experiment repository from context
        exp_repo = info.context.get("experiment_repo")
        if not exp_repo:
            return []

        # Query experiments
        experiments = exp_repo.list_experiments(
            tenant_id=tenant_id,
            limit=limit,
        )

        # Convert to GraphQL types
        result = []
        for exp in experiments:
            # Filter by status if provided
            if status and exp.status != status.value:
                continue

            # Get variants
            variants = exp_repo.get_variants(exp.id)

            result.append(
                ExperimentType(
                    id=exp.id,
                    name=exp.name,
                    description=exp.description,
                    status=ExperimentStatus(exp.status),
                    created_at=exp.created_at,
                    started_at=exp.started_at,
                    ended_at=exp.ended_at,
                    variants=[
                        ExperimentVariantType(
                            id=var.id,
                            name=var.name,
                            allocation=var.allocation,
                            config=[f"{k}={v}" for k, v in var.config.items()],
                        )
                        for var in variants
                    ],
                )
            )

        return result

    @strawberry.field
    def experiment(
        self,
        info,
        experiment_id: strawberry.ID,
    ) -> Optional[ExperimentType]:
        """
        Get a single A/B testing experiment.

        Args:
            info: GraphQL execution context
            experiment_id: Experiment ID to fetch

        Returns:
            Experiment or None if not found

        الحصول على تجربة A/B واحدة
        """
        # Validate input
        if not experiment_id:
            raise ValueError("experiment_id is required")

        # Get tenant ID from request context
        from src.api.v1.deps import get_tenant_id

        request = info.context.get("request")
        if not request:
            return None

        tenant_id = get_tenant_id(request)

        # Get experiment repository from context
        exp_repo = info.context.get("experiment_repo")
        if not exp_repo:
            return None

        # Query database
        exp = exp_repo.get_experiment(str(experiment_id))

        # Check if experiment exists and belongs to tenant
        if not exp:
            return None

        # Get variants
        variants = exp_repo.get_variants(exp.id)

        # Convert to GraphQL type
        return ExperimentType(
            id=exp.id,
            name=exp.name,
            description=exp.description,
            status=ExperimentStatus(exp.status),
            created_at=exp.created_at,
            started_at=exp.started_at,
            ended_at=exp.ended_at,
            variants=[
                ExperimentVariantType(
                    id=var.id,
                    name=var.name,
                    allocation=var.allocation,
                    config=[f"{k}={v}" for k, v in var.config.items()],
                )
                for var in variants
            ],
        )

    @strawberry.field
    def experiment_results(
        self,
        info,
        experiment_id: strawberry.ID,
    ) -> Optional[ExperimentResultsType]:
        """
        Get A/B test experiment results with statistical analysis.

        Args:
            info: GraphQL execution context
            experiment_id: Experiment ID to fetch results for

        Returns:
            Experiment results or None if not found

        الحصول على نتائج تجربة A/B
        """
        # Validate input
        if not experiment_id:
            raise ValueError("experiment_id is required")

        # Get tenant ID from request context
        from src.api.v1.deps import get_tenant_id

        request = info.context.get("request")
        if not request:
            return None

        tenant_id = get_tenant_id(request)

        # Get experiment repository from context
        exp_repo = info.context.get("experiment_repo")
        if not exp_repo:
            return None

        # Get experiment
        exp = exp_repo.get_experiment(str(experiment_id))

        # Check if experiment exists
        if not exp:
            return None

        # Get metrics
        from src.application.services.ab_testing import ABTestingService

        ab_service = info.context.get("ab_testing_service")
        if not ab_service:
            return None

        # Analyze results
        analysis = ab_service.analyze_results(str(experiment_id))

        # Get variants
        variants = exp_repo.get_variants(exp.id)

        # Convert to GraphQL type
        experiment_type = ExperimentType(
            id=exp.id,
            name=exp.name,
            description=exp.description,
            status=ExperimentStatus(exp.status),
            created_at=exp.created_at,
            started_at=exp.started_at,
            ended_at=exp.ended_at,
            variants=[
                ExperimentVariantType(
                    id=var.id,
                    name=var.name,
                    allocation=var.allocation,
                    config=[f"{k}={v}" for k, v in var.config.items()],
                )
                for var in variants
            ],
        )

        # Convert metrics
        metrics = [
            ExperimentMetricType(
                metric_name=m.get("metric_name", ""),
                metric_value=m.get("metric_value", 0.0),
                recorded_at=m.get("recorded_at", datetime.utcnow()),
            )
            for m in analysis.get("results", [])
        ]

        return ExperimentResultsType(
            experiment=experiment_type,
            metrics=metrics,
            significant=analysis.get("significant", False),
            summary=f"A/B test analysis: {analysis.get('significant', False)}",
        )


@strawberry.type
class Mutation:
    """Root mutation type for GraphQL API."""

    @strawberry.mutation
    def ask_question(
        self,
        info,
        question: str,
        k: int = 5,
        document_id: Optional[strawberry.ID] = None,
    ) -> AnswerType:
        """
        Ask a question using GraphQL mutation.

        Args:
            info: GraphQL execution context
            question: Question to ask (required)
            k: Number of chunks to retrieve (default: 5)
            document_id: Optional document ID for chat mode

        Returns:
            Answer with text and sources

        طرح سؤال باستخدام GraphQL
        """
        # 1. Validate inputs
        if not question or not question.strip():
            raise ValueError("question is required and cannot be empty")

        if k < 1 or k > 100:
            raise ValueError("k must be between 1 and 100")

        question = question.strip()

        if len(question) > 2000:
            raise ValueError("question too long (max 2000 characters)")

        # 2. Get tenant ID from request context
        from src.api.v1.deps import get_tenant_id

        request = info.context.get("request")
        if not request:
            raise RuntimeError("Request context not available")

        tenant_id = get_tenant_id(request)

        # 3. Get use case from context
        ask_use_case = info.context.get("ask_hybrid_use_case")
        if not ask_use_case:
            raise RuntimeError("Ask use case not available")

        # 4. Execute ask question use case
        from src.application.use_cases.ask_question_hybrid import AskHybridRequest

        request_data = AskHybridRequest(
            tenant_id=tenant_id,
            question=question,
            document_id=str(document_id) if document_id else None,
            rerank_top_n=k,
        )

        result = ask_use_case.execute(request_data)

        # 5. Convert to GraphQL type
        return AnswerType(
            text=result.text,
            sources=result.sources,
            retrieval_k=result.retrieval_k,
            embed_ms=result.embed_ms,
            search_ms=result.search_ms,
            llm_ms=result.llm_ms,
        )

    @strawberry.mutation
    def upload_document(
        self,
        info,
        filename: str,
        content_type: str,
        file_content: bytes,
    ) -> DocumentType:
        """
        Upload and index a new document.

        Process:
        1. Validate file
        2. Store in configured storage (S3/GCS/Azure/Local)
        3. Create database record
        4. Trigger indexing task
        5. Publish update event
        """
        # Validate inputs
        if not filename or not filename.strip():
            raise ValueError("filename is required")
        if not content_type:
            raise ValueError("content_type is required")
        if not file_content:
            raise ValueError("file_content is required")

        filename = filename.strip()

        # Get tenant ID from request context
        from src.api.v1.deps import get_tenant_id

        request = info.context.get("request")
        if not request:
            raise RuntimeError("Request context not available")

        tenant_id = get_tenant_id(request)

        # Get file storage from context
        file_storage = info.context.get("file_storage")
        if not file_storage:
            raise RuntimeError("File storage not available")

        # Get document repository from context
        doc_repo = info.context.get("doc_repo")
        if not doc_repo:
            raise RuntimeError("Document repository not available")

        # Get task queue from context
        task_queue = info.context.get("task_queue")
        if not task_queue:
            raise RuntimeError("Task queue not available")

        # Validate file size (max 50MB)
        file_size = len(file_content)
        if file_size > 50 * 1024 * 1024:
            raise ValueError("File size exceeds maximum of 50MB")

        # Store file using configured storage
        from src.domain.entities import TenantId

        tenant = TenantId(tenant_id)
        storage_path = file_storage.store_file(
            tenant_id=tenant_id,
            filename=filename,
            content=file_content,
        )

        # Create document record
        from src.domain.entities import DocumentId

        document = doc_repo.create(
            tenant_id=tenant_id,
            filename=filename,
            content_type=content_type,
            storage_path=storage_path,
            size_bytes=file_size,
            status="created",
        )

        # Trigger indexing task
        from src.workers.tasks import celery_app

        celery_app.send_task(
            "index_document",
            kwargs={
                "tenant_id": tenant_id,
                "document_id": document.document_id,
            },
        )

        # Publish update event (triggers subscriptions)
        redis_client = info.context.get("redis_client")
        if redis_client:
            await redis_client.publish(
                f"documents:{tenant_id}",
                document.document_id,
            )

        # Return document
        return DocumentType.from_entity(document)

        # Get document repository from context
        doc_repo = info.context.get("doc_repo")
        if not doc_repo:
            raise RuntimeError("Document repository not available")

        try:
            # Retrieve stored file from upload manager
            stored_file = upload_manager.get_file(str(file_content))
            if not stored_file:
                raise ValueError(f"File {file_content} not found")

            # Check for duplicate file by hash (idempotency)
            import hashlib

            file_hash = hashlib.sha256(stored_file.content).hexdigest()

            # Create document record
            document_id = doc_repo.create(
                tenant_id=tenant_id,
                filename=filename,
                file_path=stored_file.path,
                content_type=stored_file.content_type,
                size_bytes=len(stored_file.content),
                file_sha256=file_hash,
                status="queued",
            )

            # Queue for background processing (indexing)
            # In production, this would use Celery
            log.info(f"Document queued for indexing: {document_id}")

            # Trigger webhook for document upload
            webhook_manager = info.context.get("webhook_manager")
            if webhook_manager:
                import asyncio

                asyncio.create_task(
                    webhook_manager.trigger_event(
                        tenant_id=tenant_id,
                        event_type="document.uploaded",
                        payload={
                            "document_id": document_id,
                            "filename": filename,
                            "size_bytes": len(stored_file.content),
                            "content_type": stored_file.content_type,
                        },
                    )
                )

            # Return created document
            return DocumentType(
                id=document_id,
                filename=filename,
                content_type=stored_file.content_type,
                size_bytes=len(stored_file.content),
                status=DocumentStatus.CREATED,
                created_at=datetime.utcnow(),
                updated_at=None,
            )

        except Exception as e:
            log.error("Document upload failed", error=str(e), filename=filename)
            raise RuntimeError(f"Failed to upload document: {str(e)}")

    @strawberry.mutation
    def create_chat_session(
        self,
        info,
        title: Optional[str] = None,
    ) -> ChatSessionType:
        """
        Create a new chat session.

        Args:
            info: GraphQL execution context
            title: Optional session title (auto-generated if not provided)

        Returns:
            Created chat session

        إنشاء جلسة محادثة جديدة
        """
        # Get tenant ID from request context
        from src.api.v1.deps import get_tenant_id

        request = info.context.get("request")
        if not request:
            raise RuntimeError("Request context not available")

        tenant_id = get_tenant_id(request)

        # Get chat repository from context
        chat_repo = info.context.get("chat_repo")
        if not chat_repo:
            raise RuntimeError("Chat repository not available")

        try:
            # Create session via repository
            from src.domain.entities import TenantId

            session_id = chat_repo.create_session(
                tenant_id=TenantId(tenant_id),
                title=title,
            )

            # Get created session details
            session = chat_repo.get_session(
                tenant_id=TenantId(tenant_id),
                session_id=session_id,
            )

            if not session:
                raise RuntimeError("Failed to create chat session")

            return ChatSessionType(
                id=session.session_id,
                title=session.title,
                created_at=session.created_at or datetime.utcnow(),
            )

        except Exception as e:
            log.error("Failed to create chat session", error=str(e))
            raise RuntimeError(f"Failed to create chat session: {str(e)}")

    @strawberry.mutation
    def delete_document(
        self,
        info,
        document_id: strawberry.ID,
    ) -> bool:
        """
        Delete a document and all associated data.

        Cascades:
        - File from storage
        - Chunks from database
        - Vectors from vector store
        - Document record

        Args:
            info: GraphQL execution context
            document_id: Document ID to delete

        Returns:
            True if deleted successfully
        """
        # Get tenant ID from request context
        from src.api.v1.deps import get_tenant_id
        from src.domain.entities import DocumentId

        request = info.context.get("request")
        if not request:
            raise RuntimeError("Request context not available")

        tenant_id = get_tenant_id(request)

        # Get repositories from context
        doc_repo = info.context.get("doc_repo")
        chunk_repo = info.context.get("chunk_repo")
        vector_store = info.context.get("vector_store")
        file_storage = info.context.get("file_storage")

        if not all([doc_repo, file_storage]):
            raise RuntimeError("Required repositories not available")

        tenant = TenantId(tenant_id)
        doc_id = DocumentId(str(document_id))

        try:
            # Get document details
            document = doc_repo.find_by_id(doc_id)

            if not document:
                raise ValueError(f"Document not found: {document_id}")

            # Verify document belongs to tenant
            if hasattr(document, "tenant_id") and document.tenant_id != tenant:
                raise ValueError("Document belongs to different tenant")

            # Delete file from storage
            file_storage.delete_file(document.storage_path)

            # Delete chunks (cascades to vectors)
            chunk_repo.delete_by_document(doc_id)

            # Delete document record
            doc_repo.delete(tenant, doc_id)

            log.info("document_deleted", document_id=doc_id, tenant_id=tenant_id)

            return True

        except ValueError as e:
            log.error("Document validation failed", error=str(e), document_id=doc_id)
            raise
        except Exception as e:
            log.error("Failed to delete document", error=str(e), document_id=doc_id)
            raise RuntimeError(f"Failed to delete document: {str(e)}")

    @strawberry.mutation
    def update_document(
        self,
        info,
        document_id: strawberry.ID,
        filename: Optional[str] = None,
        status: Optional[DocumentStatus] = None,
    ) -> DocumentType:
        """
        Update document metadata.

        Args:
            info: GraphQL execution context
            document_id: Document ID to update
            filename: Optional new filename
            status: Optional new status

        Returns:
            Updated document
        """
        # Get tenant ID from request context
        from src.api.v1.deps import get_tenant_id
        from src.domain.entities import DocumentId

        request = info.context.get("request")
        if not request:
            raise RuntimeError("Request context not available")

        tenant_id = get_tenant_id(request)

        # Get document repository
        doc_repo = info.context.get("doc_repo")
        if not doc_repo:
            raise RuntimeError("Document repository not available")

        tenant = TenantId(tenant_id)
        doc_id = DocumentId(str(document_id))

        try:
            # Get existing document
            document = doc_repo.find_by_id(doc_id)

            if not document:
                raise ValueError(f"Document not found: {document_id}")

            # Verify document belongs to tenant
            if hasattr(document, "tenant_id") and document.tenant_id != tenant:
                raise ValueError("Document belongs to different tenant")

            # Prepare updates
            updates = {}
            if filename and filename.strip():
                updates["filename"] = filename.strip()

            if status:
                updates["status"] = status.value

            if not updates:
                raise ValueError("At least one field to update required")

            # Update document
            updated_doc = doc_repo.update(
                tenant=tenant,
                document_id=doc_id,
                **updates,
            )

            log.info("document_updated", document_id=doc_id, updates=updates)

            return DocumentType.from_entity(updated_doc)

        except ValueError as e:
            log.error("Document validation failed", error=str(e), document_id=doc_id)
            raise
        except Exception as e:
            log.error("Failed to update document", error=str(e), document_id=doc_id)
            raise RuntimeError(f"Failed to update document: {str(e)}")

    @strawberry.mutation
    def create_webhook(
        self,
        info,
        url: str,
        events: List[str],
        secret: str,
    ) -> WebhookType:
        """
        Create a new webhook configuration.

        Args:
            info: GraphQL execution context
            url: Webhook endpoint URL
            events: List of events to subscribe to
            secret: Secret for HMAC signature verification

        Returns:
            Created webhook
        """
        # Get tenant ID from request context
        from src.api.v1.deps import get_tenant_id

        request = info.context.get("request")
        if not request:
            raise RuntimeError("Request context not available")

        tenant_id = get_tenant_id(request)

        # Validate inputs
        if not url or not url.strip():
            raise ValueError("URL is required")

        if not events or len(events) == 0:
            raise ValueError("At least one event is required")

        if not secret or secret.strip():
            raise ValueError("Secret is required")

        # Validate URL format
        from urllib.parse import urlparse
        parsed = urlparse(url.strip())
        if not all([parsed.scheme, parsed.netloc]):
            raise ValueError("Invalid URL format")

        # Validate events
        valid_events = ["document.created", "document.indexed", "query.completed", "chat.message"]
        invalid_events = [e for e in events if e not in valid_events]

        if invalid_events:
            raise ValueError(f"Invalid events: {', '.join(invalid_events)}")

        # Get webhook repository
        webhook_repo = info.context.get("webhook_repo")
        if not webhook_repo:
            raise RuntimeError("Webhook repository not available")

        try:
            # Create webhook
            from src.domain.entities import TenantId

            tenant = TenantId(tenant_id)
            webhook = webhook_repo.create(
                tenant_id=tenant,
                url=url.strip(),
                events=events,
                secret=secret.strip(),
            )

            log.info("webhook_created", webhook_id=webhook.webhook_id, tenant_id=tenant_id)

            return WebhookType(
                id=webhook.webhook_id,
                url=webhook.url,
                events=webhook.events,
                active=webhook.active,
            )

        except Exception as e:
            log.error("Failed to create webhook", error=str(e))
            raise RuntimeError(f"Failed to create webhook: {str(e)}")

    @strawberry.mutation
    def delete_webhook(
        self,
        info,
        webhook_id: strawberry.ID,
    ) -> bool:
        """
        Delete a webhook configuration.

        Args:
            info: GraphQL execution context
            webhook_id: Webhook ID to delete

        Returns:
            True if deleted successfully
        """
        # Get tenant ID from request context
        from src.api.v1.deps import get_tenant_id

        request = info.context.get("request")
        if not request:
            raise RuntimeError("Request context not available")

        tenant_id = get_tenant_id(request)

        # Get webhook repository
        webhook_repo = info.context.get("webhook_repo")
        if not webhook_repo:
            raise RuntimeError("Webhook repository not available")

        tenant = TenantId(tenant_id)

        try:
            # Delete webhook
            webhook_repo.delete(
                tenant_id=tenant,
                webhook_id=webhook_id,
            )

            log.info("webhook_deleted", webhook_id=webhook_id, tenant_id=tenant_id)

            return True

        except ValueError as e:
            log.error("Webhook validation failed", error=str(e), webhook_id=webhook_id)
            raise
        except Exception as e:
            log.error("Failed to delete webhook", error=str(e), webhook_id=webhook_id)
            raise RuntimeError(f"Failed to delete webhook: {str(e)}")

    @strawberry.type
    class Subscription:
    """Root subscription type for GraphQL real-time updates."""

    @strawberry.subscription
    async def document_indexed(
        self, info, document_id: Optional[strawberry.ID] = None
    ) -> AsyncGenerator[DocumentType, None]:
        """
        Subscribe to document indexing events.

        Args:
            info: GraphQL execution context
            document_id: Optional specific document ID to watch. If None, watches all documents.

        Returns:
            Async generator yielding document updates

        الاشتراك في أحداث فهرسة المستندات
        """
        # Get tenant ID from request context
        from src.api.v1.deps import get_tenant_id

        request = info.context.get("request")
        if not request:
            return

        tenant_id = get_tenant_id(request)

        # Get event manager from context
        event_manager = info.context.get("event_manager")
        if not event_manager:
            log.warning("Event manager not available, subscription will yield nothing")
            return

        # Create a queue for this subscription
        queue = asyncio.Queue()

        try:
            # Subscribe to document indexing events
            async def on_document_indexed(event_data: dict):
                # Filter by document_id if specified
                if document_id and str(event_data.get("document_id")) != str(document_id):
                    return

                # Filter by tenant_id for security
                if event_data.get("tenant_id") != tenant_id:
                    return

                # Queue the event
                await queue.put(event_data)

            # Register the callback
            event_manager.subscribe("document.indexed", on_document_indexed)

            # Yield documents as they become available
            while True:
                event_data = await queue.get()

                # Get document repository from context
                doc_repo = info.context.get("doc_repo")
                if not doc_repo:
                    continue

                # Get the document
                doc_id = event_data.get("document_id")
                if doc_id:
                    doc = doc_repo.find_by_id(str(doc_id))
                    if doc and (doc.get("user_id") == tenant_id or not hasattr(doc, "user_id")):
                        yield DocumentType(
                            id=doc.get("document_id", doc_id),
                            filename=doc.get("filename", "unknown"),
                            content_type=doc.get("content_type", "application/octet-stream"),
                            size_bytes=doc.get("size_bytes", 0),
                            status=DocumentStatus(doc.get("status", "created")),
                            created_at=doc.get("created_at", datetime.utcnow()),
                            updated_at=doc.get("updated_at"),
                        )

        except asyncio.CancelledError:
            # Client disconnected
            pass
        except Exception as e:
            log.error("Document indexing subscription error", error=str(e))
        finally:
            # Unsubscribe when done
            try:
                event_manager.unsubscribe("document.indexed", on_document_indexed)
            except:
                pass

    @strawberry.subscription
    async def chat_updates(
        self, info, session_id: Optional[strawberry.ID] = None
    ) -> AsyncGenerator[ChatSessionType, None]:
        """
        Subscribe to chat session updates (new messages, title changes, etc.).

        Args:
            info: GraphQL execution context
            session_id: Optional specific session ID to watch. If None, watches all sessions.

        Returns:
            Async generator yielding chat session updates

        الاشتراك في تحديثات جلسات المحادثة
        """
        # Get tenant ID from request context
        from src.api.v1.deps import get_tenant_id

        request = info.context.get("request")
        if not request:
            return

        tenant_id = get_tenant_id(request)

        # Get event manager from context
        event_manager = info.context.get("event_manager")
        if not event_manager:
            log.warning("Event manager not available, subscription will yield nothing")
            return

        # Create a queue for this subscription
        queue = asyncio.Queue()

        try:
            # Subscribe to chat update events
            async def on_chat_updated(event_data: dict):
                # Filter by session_id if specified
                if session_id and str(event_data.get("session_id")) != str(session_id):
                    return

                # Filter by tenant_id for security
                if event_data.get("tenant_id") != tenant_id:
                    return

                # Queue the event
                await queue.put(event_data)

            # Register the callback
            event_manager.subscribe("chat.updated", on_chat_updated)

            # Yield chat sessions as they update
            while True:
                event_data = await queue.get()

                # Get chat repository from context
                chat_repo = info.context.get("chat_repo")
                if not chat_repo:
                    continue

                # Get the session
                from src.domain.entities import TenantId

                sess_id = event_data.get("session_id")
                if sess_id:
                    session = chat_repo.get_session(
                        tenant_id=TenantId(tenant_id),
                        session_id=str(sess_id),
                    )
                    if session:
                        yield ChatSessionType(
                            id=session.session_id,
                            title=session.title,
                            created_at=session.created_at or datetime.utcnow(),
                        )

        except asyncio.CancelledError:
            # Client disconnected
            pass
        except Exception as e:
            log.error("Chat updates subscription error", error=str(e))
        finally:
            # Unsubscribe when done
            try:
                event_manager.unsubscribe("chat.updated", on_chat_updated)
            except:
                pass


# Combined subscription type
@strawberry.type
class Subscription:
    """All GraphQL subscriptions."""

    document_updates = DocumentSubscription.document_updates
    query_progress = QueryProgressSubscription.query_progress
    chat_updates = ChatUpdateSubscription.chat_updates


# Schema
schema = strawberry.Schema(query=Query, mutation=Mutation, subscription=Subscription)
