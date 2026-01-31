# GraphQL API for RAG Engine
# ===============================
GraphQL endpoints for flexible data queries.

نقاط نهاية GraphQL لـ RAG Engine

import strawberry
from typing import List, Optional
from datetime import datetime
from enum import Enum


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


@strawberry.type
class Mutation:
    """Root mutation type for GraphQL API."""
    
    @strawberry.mutation
    def ask_question(
        self,
        question: str,
        k: int = 5,
        document_id: Optional[strawberry.ID] = None,
    ) -> AnswerType:
        """
        Ask a question using GraphQL mutation.
        
        Args:
            question: Question to ask
            k: Number of chunks to retrieve (default: 5)
            document_id: Optional document ID for chat mode
        
        Returns:
            Answer with text and sources
        
        طرح سؤال باستخدام GraphQL
        """
        # TODO: Integrate with ask_question_hybrid use case
        return AnswerType(
            text="GraphQL ask not yet implemented",
            sources=[],
            retrieval_k=0,
            embed_ms=None,
            search_ms=None,
            llm_ms=None,
        )
    
    @strawberry.mutation
    def upload_document(
        self,
        file_content: strawberry.ID,  # File upload via separate mutation
        filename: str,
    ) -> DocumentType:
        """
        Upload a document.
        
        Args:
            file_content: File ID (uploaded via separate mutation)
            filename: Original filename
        
        Returns:
            Created document
        
        رفع مستند باستخدام GraphQL
        """
        # TODO: Implement document upload
        return DocumentType(
            id="temp-id",
            filename=filename,
            content_type="application/octet-stream",
            size_bytes=0,
            status=DocumentStatus.CREATED,
            created_at=datetime.utcnow(),
            updated_at=None,
        )
    
    @strawberry.mutation
    def create_chat_session(
        self,
        title: Optional[str] = None,
    ) -> ChatSessionType:
        """Create a new chat session."""
        # TODO: Implement session creation
        return ChatSessionType(
            id="temp-id",
            title=title,
            created_at=datetime.utcnow(),
        )


@strawberry.type
class Subscription:
    """Root subscription type for GraphQL real-time updates."""
    
    @strawberry.subscription
    async def document_indexed(self, document_id: strawberry.ID) -> DocumentType:
        """Subscribe to document indexing events."""
        # TODO: Implement real-time updates
        yield DocumentType(
            id=document_id,
            filename="temp.pdf",
            content_type="application/pdf",
            size_bytes=1024,
            status=DocumentStatus.INDEXED,
            created_at=datetime.utcnow(),
            updated_at=None,
        )


# Schema
schema = strawberry.Schema(query=Query, mutation=Mutation, subscription=Subscription)
