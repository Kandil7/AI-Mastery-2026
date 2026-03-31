"""
RAG Memory Module

Production-ready memory implementations:
- Conversation buffer memory
- Summary memory
- Vector store memory
- Entity memory

Features:
- Persistent storage
- Memory compression
- Entity tracking
- Context management
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncGenerator, Deque, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class Message:
    """A single message in conversation."""

    role: str  # user, assistant, system
    content: str
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        return cls(
            role=data["role"],
            content=data["content"],
            timestamp=data.get("timestamp", time.time()),
            metadata=data.get("metadata", {}),
        )


@dataclass
class MemoryEntry:
    """An entry in memory storage."""

    id: str
    content: str
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    access_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "embedding": self.embedding,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "access_count": self.access_count,
        }


class ConversationMemory(ABC):
    """Abstract base class for conversation memory."""

    def __init__(
        self,
        max_tokens: int = 4096,
        persist_path: Optional[Union[str, Path]] = None,
    ) -> None:
        self.max_tokens = max_tokens
        self.persist_path = Path(persist_path) if persist_path else None

        self._messages: Deque[Message] = deque()
        self._stats = {
            "total_messages": 0,
            "total_tokens": 0,
            "compressions": 0,
        }

        if self.persist_path:
            self._load()

    @abstractmethod
    def add_message(self, role: str, content: str, **kwargs: Any) -> None:
        """Add a message to memory."""
        pass

    @abstractmethod
    def get_messages(self, limit: Optional[int] = None) -> List[Message]:
        """Get messages from memory."""
        pass

    @abstractmethod
    def to_prompt_messages(self) -> List[Dict[str, str]]:
        """Convert to LLM prompt format."""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all memory."""
        pass

    def add_user_message(self, content: str, **kwargs: Any) -> None:
        """Add user message."""
        self.add_message("user", content, **kwargs)

    def add_assistant_message(self, content: str, **kwargs: Any) -> None:
        """Add assistant message."""
        self.add_message("assistant", content, **kwargs)

    def get_recent(self, n: int = 5) -> List[Message]:
        """Get n most recent messages."""
        messages = list(self._messages)
        return messages[-n:] if len(messages) > n else messages

    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        return {
            **self._stats,
            "message_count": len(self._messages),
        }

    def _save(self) -> None:
        """Persist memory to disk."""
        if not self.persist_path:
            return

        data = {
            "messages": [m.to_dict() for m in self._messages],
            "stats": self._stats,
        }

        self.persist_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.persist_path, "w") as f:
            json.dump(data, f)

    def _load(self) -> None:
        """Load memory from disk."""
        if not self.persist_path or not self.persist_path.exists():
            return

        try:
            with open(self.persist_path, "r") as f:
                data = json.load(f)

            self._messages = deque(
                Message.from_dict(m) for m in data.get("messages", [])
            )
            self._stats.update(data.get("stats", {}))

            logger.info(f"Loaded {len(self._messages)} messages from {self.persist_path}")
        except Exception as e:
            logger.warning(f"Failed to load memory: {e}")


class ConversationBufferMemory(ConversationMemory):
    """
    Simple buffer memory that stores raw messages.

    Messages are evicted when token limit is exceeded.
    """

    def __init__(
        self,
        max_tokens: int = 4096,
        return_messages: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(max_tokens, **kwargs)
        self.return_messages = return_messages
        self._token_count = 0

    def add_message(self, role: str, content: str, **kwargs: Any) -> None:
        """Add message with token counting."""
        message = Message(role=role, content=content, **kwargs)

        # Estimate tokens (4 chars per token)
        message_tokens = len(content) // 4
        self._token_count += message_tokens

        # Evict old messages if needed
        while self._token_count > self.max_tokens and self._messages:
            old_message = self._messages.popleft()
            self._token_count -= len(old_message.content) // 4

        self._messages.append(message)
        self._stats["total_messages"] += 1
        self._stats["total_tokens"] += message_tokens

        self._save()

    def get_messages(self, limit: Optional[int] = None) -> List[Message]:
        """Get messages, optionally limited."""
        messages = list(self._messages)
        if limit:
            messages = messages[-limit:]
        return messages

    def to_prompt_messages(self) -> List[Dict[str, str]]:
        """Convert to LLM format."""
        return [{"role": m.role, "content": m.content} for m in self._messages]

    def clear(self) -> None:
        """Clear all messages."""
        self._messages.clear()
        self._token_count = 0
        self._save()


class ConversationSummaryMemory(ConversationMemory):
    """
    Summary-based memory that compresses old messages.

    Uses LLM to summarize conversation history when
    token limit is approached.
    """

    SUMMARY_PROMPT = """Summarize the following conversation concisely.
Include key facts, decisions, and context that should be remembered.

Conversation:
{conversation}

Summary:"""

    def __init__(
        self,
        llm_client: Any,
        max_tokens: int = 4096,
        summary_max_tokens: int = 500,
        compression_threshold: float = 0.8,
        **kwargs: Any,
    ) -> None:
        super().__init__(max_tokens, **kwargs)
        self.llm_client = llm_client
        self.summary_max_tokens = summary_max_tokens
        self.compression_threshold = compression_threshold

        self._summary: str = ""
        self._buffer: Deque[Message] = deque()
        self._buffer_tokens = 0

    def add_message(self, role: str, content: str, **kwargs: Any) -> None:
        """Add message and compress if needed."""
        message = Message(role=role, content=content, **kwargs)
        message_tokens = len(content) // 4

        # Add to buffer
        self._buffer.append(message)
        self._buffer_tokens += message_tokens
        self._messages.append(message)

        # Check if compression needed
        current_usage = (self._buffer_tokens + len(self._summary) // 4) / self.max_tokens

        if current_usage > self.compression_threshold:
            asyncio.create_task(self._compress())

        self._stats["total_messages"] += 1
        self._save()

    async def _compress(self) -> None:
        """Compress conversation history."""
        if len(self._buffer) < 2:
            return

        # Get buffer content
        buffer_text = "\n".join(
            f"{m.role}: {m.content}" for m in self._buffer
        )

        # Generate summary
        prompt = self.SUMMARY_PROMPT.format(conversation=buffer_text)

        try:
            response = await self.llm_client.generate(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.summary_max_tokens,
                temperature=0.3,
            )

            new_summary = response.content if hasattr(response, 'content') else str(response)

            # Combine with existing summary
            if self._summary:
                self._summary = f"{self._summary}\n\nPrevious context:\n{new_summary}"
            else:
                self._summary = new_summary

            # Clear buffer
            self._buffer.clear()
            self._buffer_tokens = 0
            self._stats["compressions"] += 1

            logger.info(f"Compressed conversation to summary: {len(self._summary)} chars")
        except Exception as e:
            logger.warning(f"Compression failed: {e}")

    def get_messages(self, limit: Optional[int] = None) -> List[Message]:
        """Get messages including summary."""
        messages = list(self._messages)
        if limit:
            messages = messages[-limit:]
        return messages

    def to_prompt_messages(self) -> List[Dict[str, str]]:
        """Convert to LLM format with summary."""
        messages = []

        # Add summary as system message
        if self._summary:
            messages.append({
                "role": "system",
                "content": f"Conversation Summary:\n{self._summary}",
            })

        # Add buffer messages
        for m in self._buffer:
            messages.append({"role": m.role, "content": m.content})

        return messages

    def get_summary(self) -> str:
        """Get current summary."""
        return self._summary

    def clear(self) -> None:
        """Clear all memory."""
        self._messages.clear()
        self._buffer.clear()
        self._summary = ""
        self._buffer_tokens = 0
        self._save()


class VectorStoreMemory(ConversationMemory):
    """
    Vector store-based memory for semantic retrieval.

    Stores messages as embeddings and retrieves
    relevant context based on query.
    """

    def __init__(
        self,
        vector_store: Any,
        embedding_generator: Any,
        collection: str = "memory",
        top_k: int = 5,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.vector_store = vector_store
        self.embedding_generator = embedding_generator
        self.collection = collection
        self.top_k = top_k

        self._message_embeddings: Dict[str, List[float]] = {}

    def add_message(self, role: str, content: str, **kwargs: Any) -> None:
        """Add message and store embedding."""
        message = Message(role=role, content=content, **kwargs)
        self._messages.append(message)

        # Store embedding asynchronously
        asyncio.create_task(self._store_embedding(message))

        self._stats["total_messages"] += 1
        self._save()

    async def _store_embedding(self, message: Message) -> None:
        """Store message embedding."""
        try:
            embedding_result = await self.embedding_generator.embed_text(message.content)

            # Store in vector store
            await self.vector_store.upsert(
                collection=self.collection,
                records=[{
                    "id": f"msg_{int(message.timestamp)}",
                    "vector": embedding_result.embedding,
                    "metadata": {
                        "role": message.role,
                        "content": message.content,
                        "timestamp": message.timestamp,
                    },
                }],
            )

            self._message_embeddings[str(message.timestamp)] = embedding_result.embedding
        except Exception as e:
            logger.warning(f"Failed to store embedding: {e}")

    async def retrieve_relevant(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> List[Message]:
        """Retrieve relevant messages for query."""
        top_k = top_k or self.top_k

        try:
            # Generate query embedding
            embedding_result = await self.embedding_generator.embed_text(query)

            # Search vector store
            search_result = await self.vector_store.search(
                collection=self.collection,
                query_vector=embedding_result.embedding,
                top_k=top_k,
            )

            # Convert to messages
            messages = []
            for record in search_result.records:
                messages.append(Message(
                    role=record.metadata.get("role", "unknown"),
                    content=record.metadata.get("content", ""),
                    timestamp=record.metadata.get("timestamp", time.time()),
                ))

            return messages
        except Exception as e:
            logger.warning(f"Failed to retrieve relevant messages: {e}")
            return []

    def get_messages(self, limit: Optional[int] = None) -> List[Message]:
        """Get all messages."""
        messages = list(self._messages)
        if limit:
            messages = messages[-limit:]
        return messages

    def to_prompt_messages(self) -> List[Dict[str, str]]:
        """Convert to LLM format."""
        return [{"role": m.role, "content": m.content} for m in self._messages]

    def clear(self) -> None:
        """Clear all memory."""
        self._messages.clear()
        self._message_embeddings.clear()
        self._save()


@dataclass
class Entity:
    """An entity in entity memory."""

    name: str
    type: str  # person, organization, location, concept, etc.
    description: str
    mentions: int = 1
    last_mentioned: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": self.type,
            "description": self.description,
            "mentions": self.mentions,
            "last_mentioned": self.last_mentioned,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Entity":
        return cls(**data)


class EntityMemory(ConversationMemory):
    """
    Entity-based memory for tracking people, places, concepts.

    Automatically extracts and updates entity information
    from conversations.
    """

    ENTITY_EXTRACTION_PROMPT = """Extract entities from the following text.
For each entity, identify:
- Name
- Type (person, organization, location, concept, event, object)
- Brief description

Text: {text}

Extract entities as JSON array:"""

    def __init__(
        self,
        llm_client: Any,
        max_entities: int = 100,
        auto_extract: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.llm_client = llm_client
        self.max_entities = max_entities
        self.auto_extract = auto_extract

        self._entities: Dict[str, Entity] = {}
        self._entity_history: List[Dict[str, Any]] = []

    def add_message(self, role: str, content: str, **kwargs: Any) -> None:
        """Add message and extract entities."""
        message = Message(role=role, content=content, **kwargs)
        self._messages.append(message)

        if self.auto_extract:
            asyncio.create_task(self._extract_entities(content))

        self._stats["total_messages"] += 1
        self._save()

    async def _extract_entities(self, text: str) -> None:
        """Extract entities from text using LLM."""
        prompt = self.ENTITY_EXTRACTION_PROMPT.format(text=text)

        try:
            response = await self.llm_client.generate(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
            )

            content = response.content if hasattr(response, 'content') else str(response)

            # Parse JSON
            import json
            import re
            json_match = re.search(r'\[[\s\S]*\]', content)
            if json_match:
                entities_data = json.loads(json_match.group())

                for entity_data in entities_data:
                    self._update_entity(entity_data)

                # Prune if needed
                if len(self._entities) > self.max_entities:
                    self._prune_entities()

        except Exception as e:
            logger.warning(f"Entity extraction failed: {e}")

    def _update_entity(self, entity_data: Dict[str, Any]) -> None:
        """Update or create entity."""
        name = entity_data.get("name", "").lower()
        if not name:
            return

        if name in self._entities:
            # Update existing
            entity = self._entities[name]
            entity.mentions += 1
            entity.last_mentioned = time.time()
            if entity_data.get("description"):
                entity.description = entity_data["description"]
        else:
            # Create new
            self._entities[name] = Entity(
                name=entity_data.get("name", ""),
                type=entity_data.get("type", "unknown"),
                description=entity_data.get("description", ""),
            )

    def _prune_entities(self) -> None:
        """Remove least mentioned entities."""
        sorted_entities = sorted(
            self._entities.items(),
            key=lambda x: (x[1].mentions, x[1].last_mentioned),
        )

        # Remove oldest 10%
        to_remove = max(1, len(self._entities) // 10)
        for name, _ in sorted_entities[:to_remove]:
            del self._entities[name]

    def get_entity(self, name: str) -> Optional[Entity]:
        """Get entity by name."""
        return self._entities.get(name.lower())

    def get_entities(self, entity_type: Optional[str] = None) -> List[Entity]:
        """Get all entities, optionally filtered by type."""
        entities = list(self._entities.values())
        if entity_type:
            entities = [e for e in entities if e.type == entity_type]
        return entities

    def get_messages(self, limit: Optional[int] = None) -> List[Message]:
        """Get messages."""
        messages = list(self._messages)
        if limit:
            messages = messages[-limit:]
        return messages

    def to_prompt_messages(self) -> List[Dict[str, str]]:
        """Convert to LLM format with entity context."""
        messages = []

        # Add entity context as system message
        if self._entities:
            entity_summary = "\n".join(
                f"- {e.name} ({e.type}): {e.description}"
                for e in list(self._entities.values())[:20]
            )
            messages.append({
                "role": "system",
                "content": f"Known entities:\n{entity_summary}",
            })

        # Add recent messages
        for m in list(self._messages)[-10:]:
            messages.append({"role": m.role, "content": m.content})

        return messages

    def clear(self) -> None:
        """Clear all memory."""
        self._messages.clear()
        self._entities.clear()
        self._entity_history.clear()
        self._save()


class MemoryManager:
    """
    Manager for multiple memory types.

    Coordinates between different memory implementations
    and provides unified interface.
    """

    def __init__(
        self,
        buffer_memory: Optional[ConversationBufferMemory] = None,
        summary_memory: Optional[ConversationSummaryMemory] = None,
        entity_memory: Optional[EntityMemory] = None,
        vector_memory: Optional[VectorStoreMemory] = None,
    ) -> None:
        self.buffer_memory = buffer_memory
        self.summary_memory = summary_memory
        self.entity_memory = entity_memory
        self.vector_memory = vector_memory

    def add_message(self, role: str, content: str, **kwargs: Any) -> None:
        """Add message to all memory types."""
        if self.buffer_memory:
            self.buffer_memory.add_message(role, content, **kwargs)
        if self.summary_memory:
            self.summary_memory.add_message(role, content, **kwargs)
        if self.entity_memory:
            self.entity_memory.add_message(role, content, **kwargs)
        if self.vector_memory:
            self.vector_memory.add_message(role, content, **kwargs)

    def add_user_message(self, content: str, **kwargs: Any) -> None:
        """Add user message."""
        self.add_message("user", content, **kwargs)

    def add_assistant_message(self, content: str, **kwargs: Any) -> None:
        """Add assistant message."""
        self.add_message("assistant", content, **kwargs)

    def get_context(self, query: Optional[str] = None) -> List[Dict[str, str]]:
        """Get context from all memory types."""
        messages = []

        # Get summary context
        if self.summary_memory:
            messages.extend(self.summary_memory.to_prompt_messages())

        # Get entity context
        if self.entity_memory:
            messages.extend(self.entity_memory.to_prompt_messages())

        # Get recent buffer messages
        if self.buffer_memory:
            messages.extend(self.buffer_memory.to_prompt_messages()[-5:])

        # Get relevant vector memory
        if self.vector_memory and query:
            asyncio.create_task(self._add_vector_context(messages, query))

        return messages

    async def _add_vector_context(
        self,
        messages: List[Dict[str, str]],
        query: str,
    ) -> None:
        """Add relevant vector memory context."""
        if self.vector_memory:
            relevant = await self.vector_memory.retrieve_relevant(query)
            for m in relevant[:3]:
                messages.insert(0, {"role": m.role, "content": m.content})

    def clear(self) -> None:
        """Clear all memories."""
        if self.buffer_memory:
            self.buffer_memory.clear()
        if self.summary_memory:
            self.summary_memory.clear()
        if self.entity_memory:
            self.entity_memory.clear()
        if self.vector_memory:
            self.vector_memory.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get combined statistics."""
        stats = {}

        if self.buffer_memory:
            stats["buffer"] = self.buffer_memory.get_stats()
        if self.summary_memory:
            stats["summary"] = self.summary_memory.get_stats()
        if self.entity_memory:
            stats["entity"] = {
                "entities": len(self.entity_memory._entities),
                **self.entity_memory.get_stats(),
            }
        if self.vector_memory:
            stats["vector"] = self.vector_memory.get_stats()

        return stats
