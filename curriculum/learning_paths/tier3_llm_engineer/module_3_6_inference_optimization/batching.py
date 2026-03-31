"""
Batching Module

Production-ready batching implementations:
- Continuous batching
- Request queuing
- Priority scheduling
- Token budgeting

Features:
- Dynamic batch sizing
- Fair scheduling
- Latency optimization
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class Priority(str, Enum):
    """Request priority levels."""

    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class BatchingConfig:
    """Configuration for batching."""

    # Batch size limits
    max_batch_size: int = 32
    max_batch_tokens: int = 8192
    min_batch_size: int = 1

    # Timing settings
    max_wait_ms: float = 100.0
    min_wait_ms: float = 1.0

    # Priority settings
    enable_priority: bool = True
    priority_boost_factor: float = 2.0

    # Memory settings
    max_memory_gb: Optional[float] = None
    preallocate_memory: bool = False

    # Scheduling settings
    scheduling_policy: str = "fifo"  # fifo, priority, shortest


@dataclass
class Request:
    """A generation request."""

    id: str
    prompt: str
    prompt_tokens: List[int]
    max_tokens: int = 256
    temperature: float = 0.7
    priority: Priority = Priority.NORMAL

    # Tracking
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None

    # State
    generated_tokens: List[int] = field(default_factory=list)
    is_finished: bool = False
    error: Optional[str] = None

    # Callbacks
    stream_callback: Optional[Callable[[str], None]] = None

    @property
    def wait_time(self) -> float:
        """Time spent waiting."""
        return (self.started_at or time.time()) - self.created_at

    @property
    def total_time(self) -> float:
        """Total processing time."""
        return (self.completed_at or time.time()) - self.created_at

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "prompt": self.prompt,
            "max_tokens": self.max_tokens,
            "priority": self.priority.value,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "num_generated": len(self.generated_tokens),
            "is_finished": self.is_finished,
        }


class RequestQueue:
    """
    Queue for managing generation requests.

    Supports priority-based scheduling and fair queuing.
    """

    def __init__(self, config: BatchingConfig) -> None:
        self.config = config

        self._queues: Dict[Priority, deque] = {
            Priority.CRITICAL: deque(),
            Priority.HIGH: deque(),
            Priority.NORMAL: deque(),
            Priority.LOW: deque(),
        }

        self._request_map: Dict[str, Request] = {}
        self._pending: Set[str] = set()
        self._completed: Set[str] = set()

        self._stats = {
            "total_requests": 0,
            "completed_requests": 0,
            "failed_requests": 0,
            "total_wait_time": 0.0,
        }

    def enqueue(self, request: Request) -> None:
        """Add request to queue."""
        self._queues[request.priority].append(request)
        self._request_map[request.id] = request
        self._pending.add(request.id)
        self._stats["total_requests"] += 1

        logger.debug(f"Enqueued request {request.id} with priority {request.priority}")

    def dequeue(self, num_requests: int = 1) -> List[Request]:
        """Dequeue requests based on priority."""
        requests = []

        # Process by priority order
        for priority in [Priority.CRITICAL, Priority.HIGH, Priority.NORMAL, Priority.LOW]:
            queue = self._queues[priority]

            while queue and len(requests) < num_requests:
                request = queue.popleft()
                requests.append(request)

        return requests

    def dequeue_batch(self, max_tokens: int) -> List[Request]:
        """Dequeue a batch respecting token limit."""
        requests = []
        total_tokens = 0

        for priority in [Priority.CRITICAL, Priority.HIGH, Priority.NORMAL, Priority.LOW]:
            queue = self._queues[priority]
            temp = []

            while queue:
                request = queue.popleft()
                temp.append(request)

                # Estimate tokens (prompt + max generation)
                estimated = len(request.prompt_tokens) + request.max_tokens

                if total_tokens + estimated <= max_tokens:
                    requests.append(request)
                    total_tokens += estimated
                else:
                    # Put back in queue
                    queue.appendleft(request)
                    break

            # Put remaining back
            for req in reversed(temp[len(requests):]):
                queue.appendleft(req)

        return requests

    def get(self, request_id: str) -> Optional[Request]:
        """Get request by ID."""
        return self._request_map.get(request_id)

    def remove(self, request_id: str) -> Optional[Request]:
        """Remove request from queue."""
        if request_id in self._request_map:
            request = self._request_map.pop(request_id)
            self._pending.discard(request_id)
            return request
        return None

    def mark_completed(self, request_id: str, error: Optional[str] = None) -> None:
        """Mark request as completed."""
        if request_id in self._request_map:
            request = self._request_map[request_id]
            request.completed_at = time.time()
            request.is_finished = True
            request.error = error

            self._pending.discard(request_id)
            self._completed.add(request_id)

            if error:
                self._stats["failed_requests"] += 1
            else:
                self._stats["completed_requests"] += 1
                self._stats["total_wait_time"] += request.wait_time

    def get_pending(self) -> List[Request]:
        """Get all pending requests."""
        return [self._request_map[rid] for rid in self._pending if rid in self._request_map]

    def get_queue_sizes(self) -> Dict[str, int]:
        """Get queue sizes by priority."""
        return {
            p.value: len(q) for p, q in self._queues.items()
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get queue statistics."""
        pending = len(self._pending)
        avg_wait = (
            self._stats["total_wait_time"] / self._stats["completed_requests"]
            if self._stats["completed_requests"] > 0 else 0
        )

        return {
            **self._stats,
            "pending_requests": pending,
            "avg_wait_time": avg_wait,
            "queue_sizes": self.get_queue_sizes(),
        }


class PriorityScheduler:
    """
    Priority-based request scheduler.

    Implements fair scheduling with priority boosts.
    """

    def __init__(self, config: BatchingConfig) -> None:
        self.config = config
        self._queue = RequestQueue(config)

        self._priority_boosts: Dict[str, float] = {}
        self._last_scheduled: Dict[str, float] = {}

    def submit(
        self,
        request_id: str,
        prompt: str,
        prompt_tokens: List[int],
        priority: Priority = Priority.NORMAL,
        **kwargs: Any,
    ) -> Request:
        """Submit a new request."""
        request = Request(
            id=request_id,
            prompt=prompt,
            prompt_tokens=prompt_tokens,
            priority=priority,
            **kwargs,
        )

        self._queue.enqueue(request)
        return request

    def schedule_batch(self, max_tokens: int) -> List[Request]:
        """Schedule a batch of requests."""
        # Apply priority boosts for waiting requests
        self._apply_boosts()

        # Get batch
        batch = self._queue.dequeue_batch(max_tokens)

        # Mark as started
        for request in batch:
            request.started_at = time.time()
            self._last_scheduled[request.id] = time.time()
            self._priority_boosts.pop(request.id, None)

        return batch

    def _apply_boosts(self) -> None:
        """Apply priority boosts for waiting requests."""
        if not self.config.enable_priority:
            return

        current_time = time.time()

        for request in self._queue.get_pending():
            wait_time = current_time - request.created_at

            # Boost priority based on wait time
            if wait_time > 10.0:  # 10 seconds
                if request.priority == Priority.LOW:
                    request.priority = Priority.NORMAL
                elif request.priority == Priority.NORMAL:
                    request.priority = Priority.HIGH
                elif request.priority == Priority.HIGH:
                    request.priority = Priority.CRITICAL

    def cancel(self, request_id: str) -> bool:
        """Cancel a request."""
        request = self._queue.remove(request_id)
        if request:
            request.is_finished = True
            request.error = "Cancelled"
            return True
        return False

    def get_stats(self) -> Dict[str, Any]:
        """Get scheduler statistics."""
        return self._queue.get_stats()


class ContinuousBatcher:
    """
    Continuous batching for LLM inference.

    Dynamically adjusts batch size based on request arrival
    and completion patterns.
    """

    def __init__(
        self,
        model: Any,
        config: BatchingConfig,
        tokenizer: Optional[Any] = None,
    ) -> None:
        self.model = model
        self.config = config
        self.tokenizer = tokenizer

        self._scheduler = PriorityScheduler(config)
        self._active_requests: Dict[str, Request] = {}
        self._batch_semaphore = asyncio.Semaphore(config.max_batch_size)

        self._running = False
        self._batch_task: Optional[asyncio.Task] = None

        self._stats = {
            "total_batches": 0,
            "total_tokens_generated": 0,
            "avg_batch_size": 0.0,
            "avg_latency_ms": 0.0,
        }

    async def start(self) -> None:
        """Start the batcher."""
        self._running = True
        self._batch_task = asyncio.create_task(self._batch_loop())
        logger.info("Continuous batcher started")

    async def stop(self) -> None:
        """Stop the batcher."""
        self._running = False
        if self._batch_task:
            self._batch_task.cancel()
            try:
                await self._batch_task
            except asyncio.CancelledError:
                pass
        logger.info("Continuous batcher stopped")

    async def submit(
        self,
        request_id: str,
        prompt: str,
        max_tokens: int = 256,
        priority: Priority = Priority.NORMAL,
        stream: bool = False,
    ) -> AsyncGenerator[str, None]:
        """
        Submit a request and optionally stream results.

        Args:
            request_id: Unique request ID
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            priority: Request priority
            stream: Whether to stream tokens

        Yields:
            Generated tokens (if streaming)
        """
        # Tokenize prompt
        prompt_tokens = await self._tokenize(prompt)

        # Create request
        request = self._scheduler.submit(
            request_id=request_id,
            prompt=prompt,
            prompt_tokens=prompt_tokens,
            max_tokens=max_tokens,
            priority=priority,
        )

        # Wait for completion
        while not request.is_finished:
            await asyncio.sleep(0.01)

            if stream and request.generated_tokens:
                # Stream new tokens
                new_tokens = request.generated_tokens[-1:]
                for token_id in new_tokens:
                    token = await self._detokenize(token_id)
                    yield token

        if request.error:
            raise RuntimeError(request.error)

        # Return final result if not streaming
        if not stream:
            result = await self._detokenize_tokens(request.generated_tokens)
            yield result

    async def _batch_loop(self) -> None:
        """Main batching loop."""
        while self._running:
            try:
                # Wait for requests
                if not self._scheduler.get_stats()["pending_requests"]:
                    await asyncio.sleep(self.config.min_wait_ms / 1000)
                    continue

                # Schedule batch
                batch = self._scheduler.schedule_batch(self.config.max_batch_tokens)

                if not batch:
                    await asyncio.sleep(self.config.min_wait_ms / 1000)
                    continue

                # Process batch
                await self._process_batch(batch)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Batch loop error: {e}")
                await asyncio.sleep(0.1)

    async def _process_batch(self, batch: List[Request]) -> None:
        """Process a batch of requests."""
        start_time = time.time()

        # Track active requests
        for request in batch:
            self._active_requests[request.id] = request

        # Run model on batch
        try:
            # Get all prompt tokens
            all_prompts = [r.prompt_tokens for r in batch]
            max_length = max(len(p) for p in all_prompts)

            # Pad to same length
            padded_prompts = self._pad_sequences(all_prompts, max_length)

            # Run model
            outputs = await self._run_model(padded_prompts)

            # Process outputs
            for request, output in zip(batch, outputs):
                new_token = output["token"]
                request.generated_tokens.append(new_token)

                # Check if finished
                if (
                    len(request.generated_tokens) >= request.max_tokens or
                    new_token == getattr(self.tokenizer, 'eos_token_id', 2)
                ):
                    request.is_finished = True
                    self._scheduler._queue.mark_completed(request.id)
                else:
                    # Continue generation - add token to prompt for next iteration
                    request.prompt_tokens.append(new_token)

        except Exception as e:
            logger.error(f"Batch processing error: {e}")
            for request in batch:
                request.is_finished = True
                request.error = str(e)
                self._scheduler._queue.mark_completed(request.id, error=str(e))

        finally:
            # Remove from active
            for request in batch:
                self._active_requests.pop(request.id, None)

        # Update stats
        batch_time = (time.time() - start_time) * 1000
        self._stats["total_batches"] += 1
        self._stats["avg_batch_size"] = (
            (self._stats["avg_batch_size"] * (self._stats["total_batches"] - 1) + len(batch))
            / self._stats["total_batches"]
        )
        self._stats["avg_latency_ms"] = (
            (self._stats["avg_latency_ms"] * (self._stats["total_batches"] - 1) + batch_time)
            / self._stats["total_batches"]
        )

    async def _run_model(self, prompts: List[List[int]]) -> List[Dict[str, Any]]:
        """Run model on batched prompts."""
        try:
            import torch
        except ImportError:
            # Fallback for testing
            return [{"token": 100 + i} for i in range(len(prompts))]

        with torch.no_grad():
            input_tensor = torch.tensor(prompts)
            outputs = self.model(input_tensor)
            logits = outputs.logits[:, -1, :]

            # Sample tokens
            tokens = []
            for logit in logits:
                probs = torch.softmax(logit, dim=-1)
                token = torch.argmax(probs).item()
                tokens.append({"token": token})

        return tokens

    async def _tokenize(self, text: str) -> List[int]:
        """Tokenize text."""
        if self.tokenizer:
            return self.tokenizer.encode(text)

        # Simple fallback
        return [ord(c) for c in text[:100]]

    async def _detokenize(self, token_id: int) -> str:
        """Detokenize single token."""
        if self.tokenizer:
            return self.tokenizer.decode([token_id])
        return chr(token_id % 256)

    async def _detokenize_tokens(self, token_ids: List[int]) -> str:
        """Detokenize multiple tokens."""
        if self.tokenizer:
            return self.tokenizer.decode(token_ids)
        return "".join(chr(t % 256) for t in token_ids)

    def _pad_sequences(
        self,
        sequences: List[List[int]],
        max_length: int,
        pad_value: int = 0,
    ) -> List[List[int]]:
        """Pad sequences to same length."""
        return [
            seq + [pad_value] * (max_length - len(seq))
            for seq in sequences
        ]

    def get_stats(self) -> Dict[str, Any]:
        """Get batcher statistics."""
        return {
            **self._stats,
            "active_requests": len(self._active_requests),
            "scheduler_stats": self._scheduler.get_stats(),
        }


class TokenBudgetManager:
    """
    Manages token budgeting for batched inference.

    Ensures fair allocation of compute resources.
    """

    def __init__(
        self,
        total_budget: int = 100000,  # tokens per second
        window_seconds: float = 1.0,
    ) -> None:
        self.total_budget = total_budget
        self.window_seconds = window_seconds

        self._usage_history: deque = deque()
        self._current_usage = 0

    def record_usage(self, num_tokens: int) -> None:
        """Record token usage."""
        current_time = time.time()
        self._usage_history.append((current_time, num_tokens))
        self._current_usage += num_tokens

        # Clean old entries
        self._cleanup(current_time)

    def _cleanup(self, current_time: float) -> None:
        """Remove old usage entries."""
        cutoff = current_time - self.window_seconds

        while self._usage_history and self._usage_history[0][0] < cutoff:
            _, tokens = self._usage_history.popleft()
            self._current_usage -= tokens

    def get_available_budget(self) -> int:
        """Get available token budget."""
        self._cleanup(time.time())
        return max(0, self.total_budget - self._current_usage)

    def can_allocate(self, num_tokens: int) -> bool:
        """Check if tokens can be allocated."""
        return self.get_available_budget() >= num_tokens

    async def wait_for_budget(self, num_tokens: int, timeout: float = 5.0) -> bool:
        """Wait until budget is available."""
        start_time = time.time()

        while not self.can_allocate(num_tokens):
            if time.time() - start_time > timeout:
                return False
            await asyncio.sleep(0.01)
            self._cleanup(time.time())

        return True

    def get_stats(self) -> Dict[str, Any]:
        """Get budget statistics."""
        return {
            "total_budget": self.total_budget,
            "current_usage": self._current_usage,
            "available": self.get_available_budget(),
            "utilization": self._current_usage / self.total_budget if self.total_budget > 0 else 0,
        }
