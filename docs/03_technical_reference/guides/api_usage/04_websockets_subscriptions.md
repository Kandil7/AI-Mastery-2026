# GraphQL Subscriptions: WebSockets & Real-Time Updates

## Introduction

GraphQL subscriptions provide real-time data updates over persistent WebSocket connections, unlike queries which fetch data once.

## Learning Objectives

By end of this guide, you will understand:
- **How GraphQL subscriptions differ from queries/mutations**
- **WebSocket protocol and lifecycle**
- **Redis pub/sub for scalable real-time updates**
- **Subscription implementation with Strawberry**
- **Connection handling and cleanup**
- **Subscription patterns for RAG applications**

---

## GraphQL Subscriptions Overview

### Query vs Mutation vs Subscription

| Type | Purpose | Network | Latency | Use Cases |
|--------|----------|----------|-------------|
| **Query** | Fetch data (HTTP) | Low | Reading documents, searching |
| **Mutation** | Modify data (HTTP) | Low | Uploading, deleting documents |
| **Subscription** | Stream updates (WebSocket) | Medium-High | Real-time status, chat messages |

### Subscription Lifecycle

```
1. Client sends subscription request
   ↓
2. Server establishes WebSocket connection
   ↓
3. Server creates async generator
   ↓
4. Server yields updates as they occur
   ↓
5. Client receives each update
   ↓
6. Connection closed (client or server)
   ↓
7. Cleanup runs (unsubscribes from Redis)
```

---

## WebSocket Protocol

### WebSocket vs HTTP

**HTTP (Query/Mutation):**
```
Client → HTTP POST /graphql → Server
Client ← HTTP 200 Response ← Server
(Connection closes)
```

**WebSocket (Subscription):**
```
Client → WebSocket Upgrade → Server
    (Connection established, persistent)

Client ← {"data": {"update1"}} ← Server
Client ← {"data": {"update2"}} ← Server
    (Streaming updates...)

Client → close → Server
(Connection ends)
```

### WebSocket Benefits

1. **Persistent Connection**: Single connection for many updates
2. **Low Latency**: No per-request overhead
3. **Server Push**: Server sends updates proactively
4. **Bandwidth Efficient**: No polling overhead

---

## Redis Pub/Sub for Subscriptions

### Why Redis Pub/Sub?

GraphQL subscriptions need to:
1. **Scale across multiple server instances**
2. **Broadcast updates to all subscribed clients**
3. **Handle connection failures gracefully**
4. **Track active subscriptions**

Redis pub/sub provides:
- **Scalability**: Many publishers/subscribers
- **Reliability**: Message delivery guarantees
- **Performance**: Low latency pub/sub
- **Simplicity**: Easy API

### Pub/Sub Pattern

```
Publisher (Document Service)
    ↓
    publishes: documents:tenant-123
    ↓
Redis Pub/Sub
    ↓
    broadcasts to all subscribers
    ↓
Subscriber 1 (Client A)  ←─ receives update
Subscriber 2 (Client B)  ←─ receives update
Subscriber 3 (Client C)  ←─ receives update
```

### Redis Implementation

```python
import redis

# Publisher (e.g., document status change)
async def publish_document_update(tenant_id: str, document_id: str):
    redis_client = await get_redis()

    await redis_client.publish(
        f"documents:{tenant_id}",
        document_id,
    )
    # All subscribed clients receive document_id

# Subscriber (GraphQL subscription)
async def subscribe_document_updates(tenant_id: str):
    redis_client = await get_redis()

    async with redis_client.pubsub() as pubsub:
        await pubsub.subscribe(f"documents:{tenant_id}")

        async for message in pubsub.listen():
            if message["type"] == "message":
                document_id = message["data"]
                # Yield to GraphQL client
                yield {"document_id": document_id}
```

---

## Strawberry Subscriptions

### Basic Subscription Structure

```python
import strawberry
from strawberry.subscriptions import Subscription
from typing import AsyncGenerator

@strawberry.type
class Subscription:
    @strawberry.subscription
    async def document_updates(
        self,
        tenant_id: str,
    ) -> AsyncGenerator[DocumentUpdateType, None]:
        """
        Stream document updates for tenant.

        Args:
            tenant_id: Tenant ID to filter updates

        Yields:
            DocumentUpdateType for each status change
        """
        # Subscription logic here
        while True:
            update = await get_next_update(tenant_id)
            yield DocumentUpdateType(**update)
```

### AsyncGenerator Pattern

Strawberry subscriptions must return `AsyncGenerator`:

```python
from typing import AsyncGenerator

async def subscription_generator() -> AsyncGenerator[Type, None]:
    """
    Async generator yields updates until closed.

    Returns AsyncGenerator[Type, None]:
        - Type: Type of yielded values
        - None: No return value (generator)
    """
    try:
        while True:
            data = await wait_for_update()
            yield data  # Yield to client
    except asyncio.CancelledError:
        # Client disconnected
        raise
```

---

## Implementation: Document Updates Subscription

### Use Case

Stream real-time document status updates when:
- Document is uploaded
- Document indexing starts
- Document indexing completes
- Document is deleted

### Implementation

```python
@strawberry.type
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

        Channel: documents:{tenant_id}

        Yields DocumentUpdateType for each status change.
        """
        # Get services from context
        redis_client = info.context.get("redis_client")
        doc_repo = info.context.get("doc_repo")

        # Create subscription channel
        channel = f"documents:{tenant_id}"

        # Subscribe to Redis
        async with redis_client.pubsub() as pubsub:
            await pubsub.subscribe(channel)

            # Listen for messages
            async for message in pubsub.listen():
                if message["type"] == "message":
                    document_id = message["data"]

                    # Get document details
                    document = doc_repo.find_by_id(document_id)

                    if not document:
                        continue

                    # Yield update
                    yield DocumentUpdateType(
                        document_id=document_id,
                        filename=document.filename,
                        status=document.status.value,
                        updated_at=document.updated_at,
                    )
```

### Publishing Updates

When document status changes:

```python
# In document service
async def update_document_status(
    tenant_id: str,
    document_id: str,
    status: str,
):
    """Update document status and notify subscribers."""
    doc_repo.update_status(tenant_id, document_id, status)

    # Publish to Redis (triggers subscriptions)
    redis_client.publish(
        f"documents:{tenant_id}",
        document_id,
    )

    # All subscribed clients receive update
```

---

## Implementation: Query Progress Subscription

### Use Case

Show progress for long-running queries:
1. **received**: Query accepted (0%)
2. **retrieving**: Fetching documents (25%)
3. **reranking**: Reordering results (50%)
4. **generating**: LLM generating answer (75%)
5. **completed**: Answer ready (100%)

### Implementation

```python
@strawberry.subscription
async def query_progress(
    self,
    info,
    query_id: str,
) -> AsyncGenerator[QueryProgressType, None]:
    """
    Stream query processing progress.

    Progress stored in Redis hash:
    key: progress:{query_id}
    fields:
        - stage: Current stage
        - progress: 0.0 to 1.0
        - message: Status message

    Yields QueryProgressType for each progress update.
    """
    redis_client = info.context.get("redis_client")

    progress_key = f"progress:{query_id}"

    # Poll for progress
    while True:
        progress_data = await redis_client.hgetall(progress_key)

        if not progress_data:
            await asyncio.sleep(0.1)
            continue

        stage = progress_data.get("stage")
        progress = float(progress_data.get("progress", 0))
        message = progress_data.get("message", "")

        # Yield progress
        yield QueryProgressType(
            query_id=query_id,
            stage=stage,
            progress=progress,
            message=message,
        )

        # Exit when completed
        if stage == "completed":
            break

        await asyncio.sleep(0.5)
```

### Publishing Progress

```python
# In query service
async def execute_query(query_id: str, query: str):
    """Execute query and publish progress."""
    # Update progress
    redis_client.hset(
        f"progress:{query_id}",
        "stage", "received",
        "progress", "0.0",
        "message", "Query accepted",
    )

    # Retrieve documents
    redis_client.hset(
        f"progress:{query_id}",
        "stage", "retrieving",
        "progress", "0.25",
        "message", "Fetching documents",
    )

    documents = await doc_repo.search(query)

    # Rerank
    redis_client.hset(
        f"progress:{query_id}",
        "stage", "reranking",
        "progress", "0.5",
        "message", "Reordering results",
    )

    reranked = reranker.rerank(documents)

    # Generate answer
    redis_client.hset(
        f"progress:{query_id}",
        "stage", "generating",
        "progress", "0.75",
        "message", "Generating answer",
    )

    answer = await llm.generate(query, reranked)

    # Complete
    redis_client.hset(
        f"progress:{query_id}",
        "stage", "completed",
        "progress", "1.0",
        "message", "Answer ready",
    )
```

---

## Implementation: Chat Updates Subscription

### Use Case

Stream real-time chat messages:
- User sends message
- AI responds
- Both parties see updates simultaneously

### Implementation

```python
@strawberry.subscription
async def chat_updates(
    self,
    info,
    session_id: str,
) -> AsyncGenerator[ChatMessageUpdateType, None]:
    """
    Stream real-time chat message updates.

    Channel: chat:{session_id}

    Yields ChatMessageUpdateType for each new message.
    """
    redis_client = info.context.get("redis_client")

    channel = f"chat:{session_id}"

    # Subscribe to Redis
    async with redis_client.pubsub() as pubsub:
        await pubsub.subscribe(channel)

        async for message in pubsub.listen():
            if message["type"] == "message":
                message_data = json.loads(message["data"])

                # Verify session ID
                if message_data.get("session_id") != session_id:
                    continue

                # Yield message
                yield ChatMessageUpdateType(
                    message_id=message_data.get("message_id"),
                    session_id=message_data.get("session_id"),
                    role=message_data.get("role"),
                    content=message_data.get("content"),
                    timestamp=message_data.get("timestamp"),
                )
```

---

## Connection Handling & Cleanup

### Proper Cleanup

Always cleanup subscriptions on disconnect:

```python
@strawberry.subscription
async def my_subscription(
    self,
    tenant_id: str,
) -> AsyncGenerator[Type, None]:
    try:
        # Subscribe to Redis
        async with redis_client.pubsub() as pubsub:
            await pubsub.subscribe(channel)

            # Yield updates
            async for message in pubsub.listen():
                yield parse_message(message)

    except asyncio.CancelledError:
        # Client disconnected
        log.info("subscription_cancelled", tenant_id=tenant_id)
        raise
    except Exception as e:
        log.error("subscription_error", tenant_id=tenant_id, error=str(e))
        raise
    finally:
        # Cleanup always runs
        log.info("subscription_cleanup", tenant_id=tenant_id)
```

### Handling Redis Unavailability

```python
@strawberry.subscription
async def my_subscription(
    self,
    info,
    tenant_id: str,
) -> AsyncGenerator[Type, None]:
    redis_client = info.context.get("redis_client")

    if not redis_client:
        # Redis not available, degrade gracefully
        log.warning("redis_not_available", subscription="my_subscription")
        return  # Yield nothing

    # Normal subscription
    # ...
```

---

## GraphQL Subscription Query

### Client-Side Subscription Query

```graphql
subscription DocumentUpdates($tenantId: String!) {
  documentUpdates(tenantId: $tenantId) {
    documentId
    filename
    status
    updatedAt
  }
}
```

### Client-Side Implementation

```javascript
import { ApolloClient, InMemoryCache } from '@apollo/client';

const client = new ApolloClient({
  uri: 'ws://localhost:8000/graphql',
  cache: new InMemoryCache(),
});

// Subscribe to document updates
const subscription = client.subscribe({
  query: gql`
    subscription DocumentUpdates($tenantId: String!) {
      documentUpdates(tenantId: $tenantId) {
        documentId
        filename
        status
        updatedAt
      }
    }
  `,
  variables: {
    tenantId: 'tenant-123',
  },
}).subscribe({
  next: (data) => {
    console.log('Document updated:', data.documentUpdates);
  },
  error: (err) => {
    console.error('Subscription error:', err);
  },
});
```

---

## Best Practices

### ✅ DO

1. **Use Redis pub/sub for scaling**
```python
# GOOD: Publish to Redis (scales across instances)
redis_client.publish(f"docs:{tenant_id}", doc_id)
```

2. **Handle connection failures gracefully**
```python
# GOOD: Degrade if Redis unavailable
if not redis_client:
    log.warning("redis_unavailable")
    return  # Yield nothing
```

3. **Cleanup on disconnect**
```python
# GOOD: Always cleanup
try:
    # subscription logic
    pass
except asyncio.CancelledError:
    log.info("cancelled")
    raise
finally:
    # cleanup always runs
    pass
```

4. **Filter messages for tenant**
```python
# GOOD: Tenant isolation
if message_data.get("tenant_id") != tenant_id:
    continue  # Skip updates for other tenants
```

### ❌ DON'T

1. **Don't use global state for subscriptions**
```python
# BAD: Shared state across requests
_active_subscriptions = {}

# GOOD: Redis manages subscriptions
redis_client.publish(...)
```

2. **Don't block indefinitely without timeout**
```python
# BAD: Never exits
async def subscription():
    while True:  # No way to stop!
        yield await get_update()

# GOOD: Handle cancellation
try:
    while True:
        yield await get_update()
except asyncio.CancelledError:
    raise
```

3. **Don't forget to unsubscribe**
```python
# BAD: Memory leak
async def subscription():
    await pubsub.subscribe(channel)
    while True:
        yield await pubsub.get_message()

# GOOD: Unsubscribe on exit
async with pubsub:
    await pubsub.subscribe(channel)
    # ...
# async with handles unsubscribe automatically
```

---

## Scaling Considerations

### Multiple Server Instances

With multiple API instances:

```
Instance 1
    ↓
    Subscribers: Client A, Client B
    ↓
    ↓
Instance 2
    ↓
    Subscribers: Client C, Client D
    ↓
    ↓
Redis Pub/Sub
    ↓
    Broadcasts to all
    ↓
    All clients receive updates
```

### Load Balancing

Use load balancer with WebSocket support:
- Sticky sessions: Same instance handles WebSocket lifecycle
- Health checks: Route to healthy instances only

---

## Summary

### Key Takeaways:

1. **Subscriptions** stream real-time updates over WebSocket
2. **Redis pub/sub** enables scalable multi-instance support
3. **AsyncGenerator** is Strawberry's subscription return type
4. **Cleanup** is critical to prevent memory leaks
5. **Tenant filtering** ensures data isolation
6. **Graceful degradation** when Redis unavailable

### Best Practices:

- ✅ Use Redis pub/sub for message distribution
- ✅ Handle asyncio.CancelledError for disconnects
- ✅ Filter messages by tenant/user
- ✅ Degrade gracefully when dependencies unavailable
- ✅ Always cleanup in finally blocks
- ✅ Use tenant-specific channels

### Anti-Patterns:

- ❌ Don't use global state for subscriptions
- ❌ Don't block indefinitely without cancellation
- ❌ Don't forget to unsubscribe on disconnect
- ❌ Don't send updates to wrong tenants

---

## Additional Resources

- **Strawberry Subscriptions**: https://strawberry.rocks/docs/guides/subscriptions
- **WebSocket Protocol**: RFC 6455
- **Redis Pub/Sub**: https://redis.io/docs/manual/pubsub
- **GraphQL Subscriptions**: graphql.org/learn/subscriptions
