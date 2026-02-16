"""
GraphQL Subscriptions
==================
Real-time updates via WebSockets.

تحديثات GraphQL في الوقت الفعلي عبر WebSockets
"""

import asyncio
import json
import logging
from typing import AsyncGenerator, List
from datetime import datetime

import strawberry
from strawberry.subscriptions import Subscription

from src.domain.entities import TenantId, DocumentId
from src.api.v1.deps import get_tenant_id

log = logging.getLogger(__name__)


@strawberry.type
class DocumentUpdateType:
    """Document update notification."""

    document_id: strawberry.ID
    filename: str
    status: str
    updated_at: datetime


@strawberry.type
class QueryProgressType:
    """Query processing progress."""

    query_id: str
    stage: str
    progress: float
    message: str


@strawberry.type
class ChatMessageUpdateType:
    """Chat message update."""

    message_id: strawberry.ID
    session_id: strawberry.ID
    role: str
    content: str
    timestamp: datetime


class DocumentSubscription:
    """Document updates subscription."""

    @strawberry.subscription
    async def document_updates(
        self,
        info,
        tenant_id: str,
    ) -> AsyncGenerator[DocumentUpdateType, None]:
        """
        Stream real-time document status updates.

        Uses Redis pub/sub for scalable message distribution.

        Args:
            info: GraphQL execution info
            tenant_id: Tenant ID for filtering

        Yields:
            DocumentUpdateType for each status change

        Events published to Redis:
            - documents:{tenant_id}
            Format: JSON string with document_id

        Usage:
            subscription {
              documentUpdates(tenantId: "tenant-123") {
                documentId
                filename
                status
                updatedAt
              }
            }
        """
        # Get tenant ID from request
        request = info.context.get("request")
        if not request:
            raise ValueError("Request not available in context")

        # Get services from context
        doc_repo = info.context.get("doc_repo")
        redis_client = info.context.get("redis_client")

        if not redis_client:
            # Fallback: yield no updates if Redis not available
            log.warning("Redis not available, document updates disabled")
            return

        tenant = TenantId(tenant_id)

        # Create subscription channel
        channel = f"documents:{tenant_id}"

        log.info("subscribing_document_updates", tenant_id=tenant_id, channel=channel)

        try:
            # Subscribe to Redis pub/sub
            async with redis_client.pubsub() as pubsub:
                await pubsub.subscribe(channel)

                # Listen for messages
                async for message in pubsub.listen():
                    if message["type"] == "message":
                        try:
                            data = json.loads(message["data"])

                            # Handle delete events
                            if isinstance(data, str) and data.startswith("deleted:"):
                                continue

                            # Get document ID
                            doc_id = DocumentId(data)
                            document = doc_repo.find_by_id(doc_id)

                            if not document:
                                log.warning("document_not_found", document_id=doc_id)
                                continue

                            # Check document belongs to tenant
                            if hasattr(document, "tenant_id") and document.tenant_id != tenant:
                                log.warning("document_wrong_tenant", document_id=doc_id)
                                continue

                            # Yield update
                            yield DocumentUpdateType(
                                document_id=document.document_id,
                                filename=document.filename,
                                status=document.status.value,
                                updated_at=document.updated_at or datetime.utcnow(),
                            )

                        except (json.JSONDecodeError, KeyError) as e:
                            log.error("failed_to_parse_document_update", error=str(e))
                            continue

        except asyncio.CancelledError:
            log.info("document_updates_subscription_cancelled", tenant_id=tenant_id)
            raise
        except Exception as e:
            log.error("document_updates_subscription_error", tenant_id=tenant_id, error=str(e))
            raise


class QueryProgressSubscription:
    """Query progress subscription."""

    @strawberry.subscription
    async def query_progress(
        self,
        info,
        query_id: str,
    ) -> AsyncGenerator[QueryProgressType, None]:
        """
        Stream query processing progress updates.

        Progress stages:
        - received: Query accepted
        - retrieving: Fetching documents
        - reranking: Reordering results
        - generating: LLM generating answer
        - completed: Answer ready

        Args:
            info: GraphQL execution info
            query_id: Query ID to track

        Yields:
            QueryProgressType for each progress update

        Usage:
            subscription {
              queryProgress(queryId: "query-123") {
                queryId
                stage
                progress
                message
              }
            }
        """
        redis_client = info.context.get("redis_client")

        if not redis_client:
            log.warning("Redis not available, query progress disabled")
            return

        progress_key = f"progress:{query_id}"
        log.info("subscribing_query_progress", query_id=query_id, key=progress_key)

        try:
            # Poll for progress updates
            while True:
                progress_data = await redis_client.hgetall(progress_key)

                if not progress_data:
                    await asyncio.sleep(0.1)
                    continue

                stage = progress_data.get(b"stage", b"unknown").decode()
                progress = float(progress_data.get(b"progress", b"0").decode())
                message = progress_data.get(b"message", b"").decode()

                # Yield progress update
                yield QueryProgressType(
                    query_id=query_id,
                    stage=stage,
                    progress=progress,
                    message=message,
                )

                # Exit if completed
                if stage == "completed":
                    log.info("query_progress_completed", query_id=query_id)
                    break

                # Wait before next check
                await asyncio.sleep(0.5)

        except asyncio.CancelledError:
            log.info("query_progress_subscription_cancelled", query_id=query_id)
            raise
        except Exception as e:
            log.error("query_progress_subscription_error", query_id=query_id, error=str(e))
            raise


class ChatUpdateSubscription:
    """Chat message updates subscription."""

    @strawberry.subscription
    async def chat_updates(
        self,
        info,
        session_id: str,
    ) -> AsyncGenerator[ChatMessageUpdateType, None]:
        """
        Stream real-time chat message updates.

        Includes both user messages and AI responses.

        Args:
            info: GraphQL execution info
            session_id: Chat session ID

        Yields:
            ChatMessageUpdateType for each new message

        Usage:
            subscription {
              chatUpdates(sessionId: "session-123") {
                messageId
                sessionId
                role
                content
                timestamp
              }
            }
        """
        redis_client = info.context.get("redis_client")

        if not redis_client:
            log.warning("Redis not available, chat updates disabled")
            return

        session = session_id
        channel = f"chat:{session}"

        log.info("subscribing_chat_updates", session_id=session_id, channel=channel)

        try:
            # Subscribe to Redis pub/sub
            async with redis_client.pubsub() as pubsub:
                await pubsub.subscribe(channel)

                # Listen for messages
                async for message in pubsub.listen():
                    if message["type"] == "message":
                        try:
                            message_data = json.loads(message["data"])

                            # Verify session ID matches
                            if message_data.get("session_id") != session_id:
                                continue

                            # Yield message update
                            yield ChatMessageUpdateType(
                                message_id=message_data.get("message_id"),
                                session_id=message_data.get("session_id"),
                                role=message_data.get("role", "user"),
                                content=message_data.get("content", ""),
                                timestamp=datetime.fromisoformat(
                                    message_data.get("timestamp", datetime.utcnow().isoformat())
                                ),
                            )

                        except (json.JSONDecodeError, KeyError) as e:
                            log.error("failed_to_parse_chat_update", error=str(e))
                            continue

        except asyncio.CancelledError:
            log.info("chat_updates_subscription_cancelled", session_id=session_id)
            raise
        except Exception as e:
            log.error("chat_updates_subscription_error", session_id=session_id, error=str(e))
            raise
