# Database Reliability Patterns

## Overview

This document provides comprehensive database reliability patterns for production systems. It covers circuit breakers, bulkheads, retry policies, graceful degradation, health checks, and failover automation to ensure database systems remain resilient under various failure conditions.

---

## 1. Circuit Breaker Patterns for Databases

### 1.1 Understanding Database Circuit Breakers

A circuit breaker prevents cascading failures by stopping requests to a failing database and allowing it time to recover.

```python
import time
import threading
from enum import Enum
from dataclasses import dataclass, field
from typing import Callable, Any, Optional, Dict
import logging
from collections import deque

logger = logging.getLogger(__name__)

class CircuitState(Enum):
    CLOSED = "closed"      # Normal operation, requests flow through
    OPEN = "open"          # Failure threshold exceeded, requests blocked
    HALF_OPEN = "half_open"  # Testing if service recovered

@dataclass
class CircuitBreakerConfig:
    failure_threshold: int = 5          # Failures before opening
    success_threshold: int = 3          # Successes to close from half-open
    timeout_duration: float = 30.0      # Seconds before attempting recovery
    half_open_max_calls: int = 3       # Max calls in half-open state
    excluded_exceptions: tuple = ()     # Exceptions that don't count as failures
    slow_call_threshold: float = 5.0    # Seconds to consider call as slow

@dataclass
class CircuitBreakerMetrics:
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    rejected_calls: int = 0
    slow_calls: int = 0
    state_changes: deque = field(default_factory=lambda: deque(maxlen=100))
    last_failure_time: Optional[float] = None
    
    def rejection_rate(self) -> float:
        if self.total_calls == 0:
            return 0.0
        return self.rejected_calls / self.total_calls

class DatabaseCircuitBreaker:
    """
    Circuit breaker implementation for database connections.
    
    State Transitions:
    - CLOSED -> OPEN: When failure_threshold exceeded
    - OPEN -> HALF_OPEN: After timeout_duration
    - HALF_OPEN -> CLOSED: After success_threshold successes
    - HALF_OPEN -> OPEN: On any failure
    """
    
    def __init__(self, name: str, config: Optional[CircuitBreakerConfig] = None):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_state_change = time.time()
        self._half_open_calls = 0
        self._lock = threading.RLock()
        
        self.metrics = CircuitBreakerMetrics()
        self.fallback_handler: Optional[Callable] = None
    
    @property
    def state(self) -> CircuitState:
        with self._lock:
            if self._state == CircuitState.OPEN:
                # Check if we should transition to half-open
                if time.time() - self._last_state_change >= self.config.timeout_duration:
                    self._transition_to(CircuitState.HALF_OPEN)
            
            return self._state
    
    def _transition_to(self, new_state: CircuitState):
        """Transition to a new state."""
        old_state = self._state
        self._state = new_state
        self._last_state_change = time.time()
        
        self.metrics.state_changes.append({
            "timestamp": time.time(),
            "from": old_state.value,
            "to": new_state.value
        })
        
        logger.info(
            f"Circuit breaker '{self.name}' state changed: "
            f"{old_state.value} -> {new_state.value}"
        )
        
        # Reset counters based on new state
        if new_state == CircuitState.HALF_OPEN:
            self._half_open_calls = 0
            self._success_count = 0
        elif new_state == CircuitState.CLOSED:
            self._failure_count = 0
    
    def _record_success(self):
        """Record a successful call."""
        with self._lock:
            self.metrics.successful_calls += 1
            
            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.config.success_threshold:
                    self._transition_to(CircuitState.CLOSED)
            elif self._state == CircuitState.CLOSED:
                # Reset failure count on success in closed state
                self._failure_count = 0
    
    def _record_failure(self, is_slow: bool = False):
        """Record a failed call."""
        with self._lock:
            self.metrics.failed_calls += 1
            if is_slow:
                self.metrics.slow_calls += 1
            
            self.metrics.last_failure_time = time.time()
            
            if self._state == CircuitState.HALF_OPEN:
                # Any failure in half-open goes back to open
                self._transition_to(CircuitState.OPEN)
            elif self._state == CircuitState.CLOSED:
                self._failure_count += 1
                if self._failure_count >= self.config.failure_threshold:
                    self._transition_to(CircuitState.OPEN)
    
    def can_execute(self) -> bool:
        """Check if a call can be executed."""
        return self.state != CircuitState.OPEN
    
    def execute(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute a function with circuit breaker protection.
        
        Args:
            func: Function to execute
            *args, **kwargs: Arguments to pass to function
            
        Returns:
            Result of function execution
            
        Raises:
            CircuitBreakerOpenError: If circuit is open
            Exception: Any exception from the function
        """
        with self._lock:
            self.metrics.total_calls += 1
            
            if not self.can_execute():
                self.metrics.rejected_calls += 1
                
                if self.fallback_handler:
                    return self.fallback_handler(self.name, *args, **kwargs)
                
                raise CircuitBreakerOpenError(
                    f"Circuit breaker '{self.name}' is OPEN"
                )
            
            # Track calls in half-open state
            if self._state == CircuitState.HALF_OPEN:
                if self._half_open_calls >= self.config.half_open_max_calls:
                    self.metrics.rejected_calls += 1
                    raise CircuitBreakerOpenError(
                        f"Circuit breaker '{self.name}' in half-open, max calls exceeded"
                    )
                self._half_open_calls += 1
        
        # Execute the function with timing
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            
            # Check for slow call
            is_slow = duration > self.config.slow_call_threshold
            
            if is_slow:
                logger.warning(
                    f"Circuit breaker '{self.name}' call was slow: "
                    f"{duration:.2f}s > {self.config.slow_call_threshold}s"
                )
            
            self._record_success()
            return result
            
        except self.config.excluded_exceptions:
            # Don't count these as failures
            raise
            
        except Exception as e:
            self._record_failure()
            raise

class CircuitBreakerOpenError(Exception):
    """Exception raised when circuit breaker is open."""
    pass
```

### 1.2 Integration with Database Connections

```python
import psycopg2
from psycopg2 import pool
from contextlib import contextmanager
from typing import Generator, Optional
import logging

logger = logging.getLogger(__name__)

class DatabaseConnectionManager:
    """
    Manages database connections with circuit breaker protection.
    """
    
    def __init__(
        self,
        dsn: str,
        min_connections: int = 2,
        max_connections: int = 10,
        circuit_breaker_config: Optional[CircuitBreakerConfig] = None
    ):
        self.dsn = dsn
        
        # Connection pool
        self._pool = pool.ThreadedConnectionPool(
            minconn=min_connections,
            maxconn=max_connections,
            dsn=dsn
        )
        
        # Circuit breaker for connection failures
        self._connection_circuit = DatabaseCircuitBreaker(
            name="db_connections",
            config=circuit_breaker_config or CircuitBreakerConfig(
                failure_threshold=3,
                timeout_duration=30
            )
        )
        
        # Circuit breaker for query failures
        self._query_circuit = DatabaseCircuitBreaker(
            name="db_queries",
            config=circuit_breaker_config or CircuitBreakerConfig(
                failure_threshold=10,
                slow_call_threshold=10.0
            )
        )
        
        # Set fallback handlers
        self._connection_circuit.fallback_handler = self._connection_fallback
        self._query_circuit.fallback_handler = self._query_fallback
        
        self._fallback_mode = False
    
    def _connection_fallback(self, circuit_name: str, *args, **kwargs):
        """Fallback when connection circuit is open."""
        logger.warning(f"Connection circuit '{circuit_name}' open, using fallback")
        self._fallback_mode = True
        raise DatabaseUnavailableError("Database connection unavailable")
    
    def _query_fallback(self, circuit_name: str, *args, **kwargs):
        """Fallback when query circuit is open."""
        logger.warning(f"Query circuit '{circuit_name}' open, using fallback")
        # Could return cached data or degraded response
        return CachedQueryResult(cached=True)
    
    @contextmanager
    def get_connection(self) -> Generator:
        """
        Get a database connection with circuit breaker protection.
        """
        conn = None
        
        try:
            # Get connection from pool
            conn = self._pool.getconn()
            
            # Test connection
            if not self._test_connection(conn):
                raise DatabaseConnectionError("Connection test failed")
            
            yield conn
            
        except (psycopg2.OperationalError, psycopg2.DatabaseError) as e:
            logger.error(f"Database error: {e}")
            self._connection_circuit.execute(lambda: None)  # Record failure
            raise
            
        finally:
            if conn:
                self._pool.putconn(conn)
    
    def _test_connection(self, conn) -> bool:
        """Test if connection is still valid."""
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            cursor.close()
            return True
        except Exception:
            return False
    
    def execute_with_circuit_breaker(
        self,
        query: str,
        params: tuple = None,
        use_fallback: bool = True
    ):
        """
        Execute a query with circuit breaker protection.
        """
        def _execute():
            with self.get_connection() as conn:
                cursor = conn.cursor()
                try:
                    if params:
                        cursor.execute(query, params)
                    else:
                        cursor.execute(query)
                    
                    # For SELECT queries, fetch results
                    if cursor.description:
                        return cursor.fetchall()
                    
                    # For INSERT/UPDATE/DELETE, commit
                    conn.commit()
                    return cursor.rowcount
                    
                finally:
                    cursor.close()
        
        if use_fallback:
            return self._query_circuit.execute(_execute)
        else:
            return _execute()
    
    def health_check(self) -> dict:
        """Return circuit breaker health status."""
        return {
            "connection_circuit": {
                "state": self._connection_circuit.state.value,
                "metrics": {
                    "total_calls": self._connection_circuit.metrics.total_calls,
                    "rejected_calls": self._connection_circuit.metrics.rejected_calls,
                    "rejection_rate": self._connection_circuit.metrics.rejection_rate()
                }
            },
            "query_circuit": {
                "state": self._query_circuit.state.value,
                "metrics": {
                    "total_calls": self._query_circuit.metrics.total_calls,
                    "slow_calls": self._query_circuit.metrics.slow_calls,
                    "rejection_rate": self._query_circuit.metrics.rejection_rate()
                }
            }
        }

class DatabaseUnavailableError(Exception):
    """Database is unavailable."""
    pass

class DatabaseConnectionError(Exception):
    """Database connection failed."""
    pass

class CachedQueryResult:
    """Represents a cached query result for fallback."""
    def __init__(self, data=None, cached=False):
        self.data = data or []
        self.cached = cached
```

---

## 2. Bulkhead Patterns

### 2.1 Database Bulkhead Implementation

```python
import threading
import time
from queue import Queue, Full, Empty
from dataclasses import dataclass, field
from typing import Callable, Any, Optional, Dict, List
from concurrent.futures import ThreadPoolExecutor, Future
import logging

logger = logging.getLogger(__name__)

@dataclass
class BulkheadConfig:
    max_concurrent_calls: int = 100      # Max simultaneous calls
    max_queue_size: int = 50              # Max queued calls
    queue_timeout: float = 5.0           # Max time to wait in queue
    execution_timeout: float = 30.0       # Max execution time

@dataclass
class BulkheadMetrics:
    total_requests: int = 0
    accepted_requests: int = 0
    rejected_requests: int = 0
    queued_requests: int = 0
    timed_out_requests: int = 0
    average_queue_wait_time: float = 0.0
    average_execution_time: float = 0.0
    _wait_times: List[float] = field(default_factory=list)
    _execution_times: List[float] = field(default_factory=list)

class DatabaseBulkhead:
    """
    Bulkhead pattern for isolating database resources.
    
    Limits concurrent access to prevent resource exhaustion.
    """
    
    def __init__(self, name: str, config: Optional[BulkheadConfig] = None):
        self.name = name
        self.config = config or BulkheadConfig()
        
        # Semaphore to limit concurrent executions
        self._semaphore = threading.Semaphore(self.config.max_concurrent_calls)
        
        # Queue for requests waiting to execute
        self._queue: Queue = Queue(maxsize=self.config.max_queue_size)
        
        # Thread pool for execution
        self._executor = ThreadPoolExecutor(
            max_workers=self.config.max_concurrent_calls,
            thread_name_prefix=f"bulkhead_{name}"
        )
        
        # Metrics tracking
        self.metrics = BulkheadMetrics()
        self._lock = threading.Lock()
        
        # Active calls counter
        self._active_calls = 0
        self._active_lock = threading.Lock()
    
    def execute(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute a function with bulkhead protection.
        
        Args:
            func: Function to execute
            *args, **kwargs: Arguments to pass
            
        Returns:
            Result of function execution
            
        Raises:
            BulkheadRejectedError: If request is rejected
            BulkheadTimeoutError: If request times out in queue
        """
        with self.metrics._lock:
            self.metrics.total_requests += 1
        
        # Try to acquire semaphore (limit concurrent calls)
        acquired = self._semaphore.acquire(
            timeout=self.config.queue_timeout
        )
        
        if not acquired:
            with self.metrics._lock:
                self.metrics.rejected_requests += 1
                self.metrics.queued_requests = self._queue.qsize()
            
            raise BulkheadRejectedError(
                f"Bulkhead '{self.name}' rejected request: queue full"
            )
        
        # Track active calls
        with self._active_lock:
            self._active_calls += 1
        
        start_time = time.time()
        
        try:
            # Execute with timeout
            result = func(*args, **kwargs)
            
            # Track metrics
            execution_time = time.time() - start_time
            with self.metrics._lock:
                self.metrics.accepted_requests += 1
                self.metrics._execution_times.append(execution_time)
                
                # Keep only last 1000 measurements
                if len(self.metrics._execution_times) > 1000:
                    self.metrics._execution_times = self.metrics._execution_times[-1000:]
                
                # Update average
                self.metrics.average_execution_time = sum(
                    self.metrics._execution_times
                ) / len(self.metrics._execution_times)
            
            return result
            
        except Exception as e:
            logger.error(f"Bulkhead '{self.name}' execution error: {e}")
            raise
            
        finally:
            self._semaphore.release()
            
            with self._active_lock:
                self._active_calls -= 1
    
    def execute_async(self, func: Callable, *args, **kwargs) -> Future:
        """Execute function asynchronously within bulkhead."""
        
        def wrapped_func():
            return self.execute(func, *args, **kwargs)
        
        return self._executor.submit(wrapped_func)
    
    def get_available_capacity(self) -> Dict[str, int]:
        """Get current capacity information."""
        return {
            "max_concurrent": self.config.max_concurrent_calls,
            "available_slots": self._semaphore._value,
            "queue_size": self._queue.qsize(),
            "max_queue_size": self.config.max_queue_size,
            "active_calls": self._active_calls
        }

class BulkheadRejectedError(Exception):
    """Request rejected due to bulkhead limits."""
    pass

class BulkheadTimeoutError(Exception):
    """Request timed out waiting for bulkhead."""
    pass
```

### 2.2 Multi-Database Bulkhead Router

```python
from typing import Dict, List, Optional
from dataclasses import dataclass
import random

@dataclass
class DatabaseEndpoint:
    name: str
    dsn: str
    priority: int = 1  # 1 = primary, 2 = secondary
    weight: int = 1     # For load balancing
    max_connections: int = 100
    
    # Bulkhead configuration per endpoint
    bulkhead_config: Optional[BulkheadConfig] = None

class MultiDatabaseBulkheadRouter:
    """
    Routes database requests across multiple endpoints with bulkhead isolation.
    """
    
    def __init__(self, endpoints: List[DatabaseEndpoint]):
        self.endpoints = {ep.name: ep for ep in endpoints}
        
        # Create bulkhead for each endpoint
        self.bulkheads: Dict[str, DatabaseBulkhead] = {}
        for ep in endpoints:
            config = ep.bulkhead_config or BulkheadConfig(
                max_concurrent_calls=ep.max_connections
            )
            self.bulkheads[ep.name] = DatabaseBulkhead(ep.name, config)
        
        # Group endpoints by priority
        self._prioritized_endpoints: Dict[int, List[DatabaseEndpoint]] = {}
        for ep in endpoints:
            if ep.priority not in self._prioritized_endpoints:
                self._prioritized_endpoints[ep.priority] = []
            self._prioritized_endpoints[ep.priority].append(ep)
    
    def _select_endpoint(self) -> DatabaseEndpoint:
        """Select an endpoint based on priority and weight."""
        # Try highest priority first
        for priority in sorted(self._prioritized_endpoints.keys()):
            endpoints = self._prioritized_endpoints[priority]
            
            # Filter by available capacity
            available = [
                ep for ep in endpoints 
                if self.bulkheads[ep.name].get_available_capacity()["available_slots"] > 0
            ]
            
            if available:
                # Weighted random selection
                total_weight = sum(ep.weight for ep in available)
                r = random.uniform(0, total_weight)
                
                for ep in available:
                    r -= ep.weight
                    if r <= 0:
                        return ep
        
        # All endpoints at max capacity, return primary anyway
        # Let bulkhead handle rejection
        return list(self._prioritized_endpoints[1])[0]
    
    def execute(self, query_func: Callable, *args, **kwargs) -> Any:
        """
        Execute query with automatic endpoint selection and bulkhead protection.
        """
        max_retries = len(self.endpoints) - 1
        last_error = None
        
        # Try endpoints in order of priority
        tried_endpoints = set()
        
        for attempt in range(max_retries + 1):
            endpoint = self._select_endpoint()
            
            if endpoint.name in tried_endpoints:
                continue
            
            tried_endpoints.add(endpoint.name)
            bulkhead = self.bulkheads[endpoint.name]
            
            try:
                # Execute within bulkhead
                return bulkhead.execute(query_func, endpoint, *args, **kwargs)
                
            except BulkheadRejectedError as e:
                logger.warning(
                    f"Bulkhead rejected for {endpoint.name}, "
                    f"trying next endpoint"
                )
                last_error = e
                continue
                
            except Exception as e:
                logger.error(f"Error on {endpoint.name}: {e}")
                last_error = e
                continue
        
        # All endpoints failed
        raise AllEndpointsFailedError(
            f"All database endpoints failed after {len(tried_endpoints)} attempts"
        )
    
    def health_check(self) -> Dict:
        """Get health status of all endpoints."""
        return {
            endpoint_name: {
                "bulkhead": bulkhead.get_available_capacity(),
                "metrics": {
                    "total_requests": bulkhead.metrics.total_requests,
                    "rejected_requests": bulkhead.metrics.rejected_requests,
                    "rejection_rate": (
                        bulkhead.metrics.rejected_requests / 
                        max(bulkhead.metrics.total_requests, 1)
                    )
                }
            }
            for endpoint_name, bulkhead in self.bulkheads.items()
        }

class AllEndpointsFailedError(Exception):
    """All database endpoints are unavailable."""
    pass
```

---

## 3. Retry Policies with Exponential Backoff

### 3.1 Database Retry Policy Implementation

```python
import time
import random
import threading
from dataclasses import dataclass
from typing import Callable, Any, Optional, Type, Tuple, List
from functools import wraps
import logging

logger = logging.getLogger(__name__)

@dataclass
class RetryConfig:
    max_attempts: int = 3
    initial_delay: float = 0.1        # seconds
    max_delay: float = 30.0            # seconds
    exponential_base: float = 2.0
    jitter: bool = True                 # Add randomness to prevent thundering herd
    jitter_range: float = 0.1           # +/- 10% jitter
    
    # Exceptions that should trigger retry
    retryable_exceptions: Tuple[Type[Exception], ...] = (
        ConnectionError,
        TimeoutError,
    )
    
    # Exceptions that should NOT be retried
    non_retryable_exceptions: Tuple[Type[Exception], ...] = (
        ValueError,
        TypeError,
    )

class RetryPolicy:
    """
    Configurable retry policy with exponential backoff and jitter.
    """
    
    def __init__(self, config: Optional[RetryConfig] = None):
        self.config = config or RetryConfig()
        self._lock = threading.Lock()
        self._attempt_counts = {}
    
    def calculate_delay(self, attempt: int) -> float:
        """
        Calculate delay for the given attempt number.
        
        Uses exponential backoff with optional jitter:
        delay = min(max_delay, initial_delay * (exponential_base ^ attempt))
        """
        # Exponential backoff
        delay = self.config.initial_delay * (self.config.exponential_base ** attempt)
        
        # Cap at max delay
        delay = min(delay, self.config.max_delay)
        
        # Add jitter
        if self.config.jitter:
            jitter_amount = delay * self.config.jitter_range
            delay = delay + random.uniform(-jitter_amount, jitter_amount)
        
        return max(0, delay)  # Ensure non-negative
    
    def should_retry(self, exception: Exception, attempt: int) -> bool:
        """Determine if we should retry after this exception."""
        
        # Check if we've exceeded max attempts
        if attempt >= self.config.max_attempts:
            return False
        
        # Check non-retryable exceptions
        for exc_type in self.config.non_retryable_exceptions:
            if isinstance(exception, exc_type):
                return False
        
        # Check retryable exceptions
        for exc_type in self.config.retryable_exceptions:
            if isinstance(exception, exc_type):
                return True
        
        # For other exceptions, default to retry
        return True
    
    def execute(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function with retry logic.
        """
        last_exception = None
        
        for attempt in range(self.config.max_attempts):
            try:
                return func(*args, **kwargs)
                
            except Exception as e:
                last_exception = e
                
                if not self.should_retry(e, attempt):
                    logger.warning(
                        f"Non-retryable exception: {type(e).__name__}: {e}"
                    )
                    raise
                
                if attempt < self.config.max_attempts - 1:
                    delay = self.calculate_delay(attempt)
                    logger.warning(
                        f"Attempt {attempt + 1}/{self.config.max_attempts} failed: {e}. "
                        f"Retrying in {delay:.2f}s..."
                    )
                    time.sleep(delay)
                else:
                    logger.error(
                        f"All {self.config.max_attempts} attempts failed"
                    )
        
        raise last_exception
    
    def get_retry_decorator(self):
        """Get a decorator for use with function definitions."""
        
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                return self.execute(func, *args, **kwargs)
            return wrapper
        
        return decorator

# Specialized retry policies for databases
class DatabaseRetryPolicy(RetryPolicy):
    """Retry policy optimized for database operations."""
    
    def __init__(self):
        super().__init__(RetryConfig(
            max_attempts=5,
            initial_delay=0.1,
            max_delay=10.0,
            exponential_base=2.0,
            jitter=True,
            jitter_range=0.2,
            retryable_exceptions=(
                # Connection errors
                psycopg2.OperationalError,
                psycopg2.DatabaseError,
                ConnectionError,
                TimeoutError,
                # Transient errors
                psycopg2.errors.ConnectionException,
                psycopg2.errors.OperationalError,
            ),
            non_retryable_exceptions=(
                # Data errors - don't retry invalid queries
                psycopg2.ProgrammingError,
                psycopg2.DataError,
                psycopg2.IntegrityError,
                ValueError,
                TypeError,
            )
        ))

# Transaction-specific retry with savepoints
class TransactionRetryPolicy(RetryPolicy):
    """
    Retry policy specifically for database transactions.
    Handles transient transaction conflicts.
    """
    
    def __init__(self):
        super().__init__(RetryConfig(
            max_attempts=3,
            initial_delay=0.05,
            max_delay=1.0,
            exponential_base=2.0,
            jitter=True,
            jitter_range=0.1,
            retryable_exceptions=(
                psycopg2.errors.TransactionRollbackError,
                psycopg2.errors.DeadlockDetected,
                psycopg2.errors.SerializationFailure,
                ConnectionError,
            ),
            non_retryable_exceptions=(
                psycopg2.ProgrammingError,
                psycopg2.IntegrityError,
            )
        ))
    
    def execute_with_savepoint(
        self,
        connection,
        func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """
        Execute function within a transaction with savepoint for retry.
        """
        cursor = connection.cursor()
        savepoint_name = f"sp_{int(time.time() * 1000)}"
        
        for attempt in range(self.config.max_attempts):
            try:
                # Create savepoint
                cursor.execute(f"SAVEPOINT {savepoint_name}")
                
                # Execute function
                result = func(cursor, *args, **kwargs)
                
                # Release savepoint (commits this part)
                cursor.execute(f"RELEASE SAVEPOINT {savepoint_name}")
                
                return result
                
            except self.config.retryable_exceptions as e:
                # Rollback to savepoint
                cursor.execute(f"ROLLBACK TO SAVEPOINT {savepoint_name}")
                
                if not self.should_retry(e, attempt):
                    raise
                
                if attempt < self.config.max_attempts - 1:
                    delay = self.calculate_delay(attempt)
                    logger.warning(
                        f"Transaction retry {attempt + 1}/{self.config.max_attempts}: {e}. "
                        f"Waiting {delay:.2f}s"
                    )
                    time.sleep(delay)
                    
            except Exception:
                # Release savepoint on non-retryable error
                cursor.execute(f"RELEASE SAVEPOINT {savepoint_name}")
                raise
        
        raise MaxRetriesExceededError(
            f"Transaction failed after {self.config.max_attempts} attempts"
        )

class MaxRetriesExceededError(Exception):
    """Maximum retry attempts exceeded."""
    pass
```

### 3.2 Decorator-Based Retry

```python
# Decorator for automatic retry on database operations
def with_retry(
    max_attempts: int = 3,
    initial_delay: float = 0.1,
    max_delay: float = 10.0,
    exponential_base: float = 2.0,
    retryable_exceptions: tuple = (Exception,)
):
    """
    Decorator for retrying database operations.
    
    Example:
        @with_retry(max_attempts=5, retryable_exceptions=(ConnectionError, TimeoutError))
        def fetch_data(query):
            return execute_query(query)
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            policy = RetryPolicy(RetryConfig(
                max_attempts=max_attempts,
                initial_delay=initial_delay,
                max_delay=max_delay,
                exponential_base=exponential_base,
                retryable_exceptions=retryable_exceptions
            ))
            return policy.execute(func, *args, **kwargs)
        return wrapper
    return decorator

# Usage examples
class DatabaseOperations:
    """Example database operations with retry logic."""
    
    @staticmethod
    @with_retry(max_attempts=5, retryable_exceptions=(ConnectionError, psycopg2.OperationalError))
    def execute_query(query: str, params: tuple = None):
        """Execute a database query with automatic retry."""
        conn = get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(query, params)
            if cursor.description:
                return cursor.fetchall()
            conn.commit()
        finally:
            conn.close()
    
    @staticmethod
    @with_retry(max_attempts=3, initial_delay=0.5, max_delay=5.0)
    def execute_transaction(queries: List[tuple]):
        """Execute multiple queries in a transaction with retry."""
        conn = get_connection()
        try:
            cursor = conn.cursor()
            for query, params in queries:
                cursor.execute(query, params)
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
```

---

## 4. Graceful Degradation Strategies

### 4.1 Database Degradation Levels

```python
from enum import Enum
from dataclasses import dataclass
from typing import Any, Optional, Dict, List, Callable
import threading
import time
import logging

logger = logging.getLogger(__name__)

class DegradationLevel(Enum):
    FULL = "full"                    # All features working
    DEGRADED_READS = "degraded_reads"  # Read-only mode
    CACHED_ONLY = "cached_only"        # Only cached data available
    READY_ONLY = "ready_only"          # Accepting no writes
    MAINTENANCE = "maintenance"        # Limited operations

@dataclass
class DegradationConfig:
    level: DegradationLevel
    description: str
    allowed_operations: List[str]
    fallback_functions: Dict[str, Callable]
    auto_recovery: bool = True
    auto_recovery_timeout: float = 300.0  # seconds

class GracefulDegradationManager:
    """
    Manages graceful degradation of database functionality.
    """
    
    def __init__(self):
        self._current_level = DegradationLevel.FULL
        self._level_config: Dict[DegradationLevel, DegradationConfig] = {}
        self._lock = threading.Lock()
        self._degradation_history: List[dict] = []
        self._degradation_start_time: Optional[float] = None
    
    def register_level(self, config: DegradationConfig):
        """Register a degradation level configuration."""
        self._level_config[config.level] = config
    
    def set_level(self, level: DegradationLevel, reason: str = ""):
        """
        Set the current degradation level.
        """
        with self._lock:
            old_level = self._current_level
            self._current_level = level
            self._degradation_start_time = time.time()
            
            logger.warning(
                f"Database degradation: {old_level.value} -> {level.value}. "
                f"Reason: {reason}"
            )
            
            self._degradation_history.append({
                "timestamp": time.time(),
                "old_level": old_level.value,
                "new_level": level.value,
                "reason": reason
            })
    
    def get_current_level(self) -> DegradationLevel:
        """Get current degradation level."""
        return self._current_level
    
    def can_execute(self, operation: str) -> tuple[bool, Any]:
        """
        Check if an operation can be executed at current degradation level.
        
        Returns:
            (can_execute: bool, fallback_result: Any)
        """
        with self._lock:
            config = self._level_config.get(self._current_level)
            
            if not config:
                return True, None
            
            if operation in config.allowed_operations:
                return True, None
            
            # Return fallback if available
            fallback_func = config.fallback_functions.get(operation)
            if fallback_func:
                return False, fallback_func()
            
            return False, None
    
    def degrade_reads(self, reason: str = "High load"):
        """Degrade to read-only mode."""
        config = self._level_config.get(DegradationLevel.DEGRADED_READS)
        if config:
            self.set_level(DegradationLevel.DEGRADED_READS, reason)
    
    def degrade_cached(self, reason: str = "Database unavailable"):
        """Degrade to cached data only."""
        config = self._level_config.get(DegradationLevel.CACHED_ONLY)
        if config:
            self.set_level(DegradationLevel.CACHED_ONLY, reason)
    
    def recover_full(self, reason: str = "System recovered"):
        """Recover to full functionality."""
        self.set_level(DegradationLevel.FULL, reason)
    
    def get_status(self) -> Dict:
        """Get current degradation status."""
        with self._lock:
            config = self._level_config.get(self._current_level)
            
            return {
                "current_level": self._current_level.value,
                "description": config.description if config else "Normal operation",
                "degradation_duration": (
                    time.time() - self._degradation_start_time
                    if self._degradation_start_time else 0
                ),
                "history": self._degradation_history[-10:]  # Last 10 changes
            }

# Configure degradation levels
DEGRADATION_CONFIGS = {
    DegradationLevel.FULL: DegradationConfig(
        level=DegradationLevel.FULL,
        description="All database operations available",
        allowed_operations=["read", "write", "delete", "create"],
        fallback_functions={}
    ),
    
    DegradationLevel.DEGRADED_READS: DegradationConfig(
        level=DegradationLevel.DEGRADED_READS,
        description="Read-only mode - writes temporarily disabled",
        allowed_operations=["read"],
        fallback_functions={
            "write": lambda: {"error": "Database in read-only mode"},
            "delete": lambda: {"error": "Database in read-only mode"},
            "create": lambda: {"error": "Database in read-only mode"}
        },
        auto_recovery=True,
        auto_recovery_timeout=300.0
    ),
    
    DegradationLevel.CACHED_ONLY: DegradationConfig(
        level=DegradationLevel.CACHED_ONLY,
        description="Only cached data available",
        allowed_operations=["read_cached"],
        fallback_functions={
            "read": lambda: get_cached_data(),
            "write": lambda: {"error": "Database unavailable"}
        },
        auto_recovery=True,
        auto_recovery_timeout=600.0
    ),
    
    DegradationLevel.READY_ONLY: DegradationConfig(
        level=DegradationLevel.READY_ONLY,
        description="Maintenance mode - limited operations",
        allowed_operations=["health_check"],
        fallback_functions={
            "read": lambda: {"error": "Maintenance mode"},
            "write": lambda: {"error": "Maintenance mode"}
        },
        auto_recovery=False
    )
}
```

### 4.2 Degraded Query Execution

```python
class DegradedQueryExecutor:
    """
    Execute queries with graceful degradation.
    """
    
    def __init__(
        self,
        db_manager: 'DatabaseConnectionManager',
        degradation_manager: GracefulDegradationManager,
        cache_manager: Optional['CacheManager'] = None
    ):
        self.db = db_manager
        self.degradation = degradation_manager
        self.cache = cache_manager
    
    def execute_read(
        self,
        query: str,
        params: tuple = None,
        use_cache: bool = True
    ):
        """Execute a read query with degradation support."""
        
        can_execute, fallback = self.degradation.can_execute("read")
        
        if not can_execute:
            if fallback is not None:
                return fallback
            raise ServiceUnavailableError("Database operations degraded")
        
        # Try cache first if enabled
        if use_cache and self.cache:
            cache_key = self.cache.generate_key(query, params)
            cached_result = self.cache.get(cache_key)
            
            if cached_result is not None:
                logger.debug(f"Cache hit for query: {query[:50]}...")
                return cached_result
        
        # Execute query
        try:
            result = self.db.execute_with_circuit_breaker(query, params)
            
            # Cache successful result
            if use_cache and self.cache and result:
                self.cache.set(cache_key, result, ttl=60)
            
            return result
            
        except Exception as e:
            # Try degraded mode
            if use_cache and self.cache:
                cached_result = self.cache.get(
                    self.cache.generate_key(query, params)
                )
                if cached_result is not None:
                    logger.warning(
                        f"Query failed, returning cached data: {e}"
                    )
                    return cached_result
            
            raise
    
    def execute_write(self, query: str, params: tuple = None):
        """Execute a write query with degradation support."""
        
        can_execute, fallback = self.degradation.can_execute("write")
        
        if not can_execute:
            if fallback is not None:
                return fallback
            raise ServiceUnavailableError("Database writes currently disabled")
        
        return self.db.execute_with_circuit_breaker(query, params)

class ServiceUnavailableError(Exception):
    """Service is unavailable due to degradation."""
    pass
```

---

## 5. Database Health Checks

### 5.1 Comprehensive Health Check Implementation

```python
import time
import psycopg2
from dataclasses import dataclass
from typing import Dict, List, Optional
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class HealthStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"

@dataclass
class HealthCheckResult:
    status: HealthStatus
    component: str
    message: str
    details: Dict
    timestamp: float
    
    def is_healthy(self) -> bool:
        return self.status in (HealthStatus.HEALTHY, HealthStatus.DEGRADED)

class DatabaseHealthChecker:
    """
    Comprehensive health checker for database systems.
    """
    
    def __init__(self, db_config: dict):
        self.db_config = db_config
        self.checks: Dict[str, callable] = {}
        self._register_default_checks()
    
    def _register_default_checks(self):
        """Register default health check functions."""
        self.register_check("connectivity", self._check_connectivity)
        self.register_check("replication", self._check_replication)
        self.register_check("connections", self._check_connections)
        self.register_check("wal_lag", self._check_wal_lag)
        self.register_check("disk_space", self._check_disk_space)
        self.register_check("locks", self._check_locks)
        self.register_check("slow_queries", self._check_slow_queries)
        self.register_check("cache_hit_ratio", self._check_cache_ratio)
    
    def register_check(self, name: str, check_func: callable):
        """Register a custom health check."""
        self.checks[name] = check_func
    
    def check_all(self) -> Dict:
        """Run all health checks."""
        results = []
        
        for name, check_func in self.checks.items():
            try:
                result = check_func()
                results.append(result)
            except Exception as e:
                results.append(HealthCheckResult(
                    status=HealthStatus.UNHEALTHY,
                    component=name,
                    message=f"Check failed: {str(e)}",
                    details={},
                    timestamp=time.time()
                ))
        
        # Determine overall status
        statuses = [r.status for r in results]
        
        if any(s == HealthStatus.UNHEALTHY for s in statuses):
            overall_status = HealthStatus.UNHEALTHY
        elif any(s == HealthStatus.DEGRADED for s in statuses):
            overall_status = HealthStatus.DEGRADED
        else:
            overall_status = HealthStatus.HEALTHY
        
        return {
            "status": overall_status.value,
            "timestamp": time.time(),
            "checks": [r.__dict__ for r in results],
            "summary": {
                "total": len(results),
                "healthy": len([r for r in results if r.status == HealthStatus.HEALTHY]),
                "degraded": len([r for r in results if r.status == HealthStatus.DEGRADED]),
                "unhealthy": len([r for r in results if r.status == HealthStatus.UNHEALTHY])
            }
        }
    
    def _check_connectivity(self) -> HealthCheckResult:
        """Check basic database connectivity."""
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            cursor.close()
            conn.close()
            
            return HealthCheckResult(
                status=HealthStatus.HEALTHY,
                component="connectivity",
                message="Database is reachable",
                details={},
                timestamp=time.time()
            )
        except Exception as e:
            return HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                component="connectivity",
                message=f"Cannot connect to database: {str(e)}",
                details={"error": str(e)},
                timestamp=time.time()
            )
    
    def _check_replication(self) -> HealthCheckResult:
        """Check replication status and lag."""
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()
            
            # Check for replicas
            cursor.execute("""
                SELECT status, 
                       COALESCE(EXTRACT(EPOCH FROM (now() - pg_last_xact_replay_timestamp())), 0) as lag
                FROM pg_stat_wal_receiver
            """)
            
            result = cursor.fetchone()
            cursor.close()
            conn.close()
            
            if result is None:
                return HealthCheckResult(
                    status=HealthStatus.HEALTHY,
                    component="replication",
                    message="No replicas configured (standalone)",
                    details={},
                    timestamp=time.time()
                )
            
            status, lag = result
            
            # Thresholds
            lag_warning = 5.0  # seconds
            lag_critical = 30.0
            
            if lag > lag_critical:
                status = HealthStatus.UNHEALTHY
                message = f"Replication lag critical: {lag:.1f}s"
            elif lag > lag_warning:
                status = HealthStatus.DEGRADED
                message = f"Replication lag high: {lag:.1f}s"
            else:
                status = HealthStatus.HEALTHY
                message = f"Replication healthy, lag: {lag:.1f}s"
            
            return HealthCheckResult(
                status=status,
                component="replication",
                message=message,
                details={"status": status, "lag_seconds": lag},
                timestamp=time.time()
            )
            
        except Exception as e:
            return HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                component="replication",
                message=f"Cannot check replication: {str(e)}",
                details={"error": str(e)},
                timestamp=time.time()
            )
    
    def _check_connections(self) -> HealthCheckResult:
        """Check connection pool utilization."""
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT 
                    (SELECT setting::int FROM pg_settings WHERE name = 'max_connections') as max_conn,
                    (SELECT count(*) FROM pg_stat_activity) as current_conn
            """)
            
            max_conn, current_conn = cursor.fetchone()
            cursor.close()
            conn.close()
            
            usage_pct = (current_conn / max_conn) * 100
            
            if usage_pct > 90:
                status = HealthStatus.UNHEALTHY
                message = f"Connections critical: {usage_pct:.1f}%"
            elif usage_pct > 75:
                status = HealthStatus.DEGRADED
                message = f"Connections high: {usage_pct:.1f}%"
            else:
                status = HealthStatus.HEALTHY
                message = f"Connections normal: {usage_pct:.1f}%"
            
            return HealthCheckResult(
                status=status,
                component="connections",
                message=message,
                details={
                    "max_connections": max_conn,
                    "current_connections": current_conn,
                    "usage_percent": usage_pct
                },
                timestamp=time.time()
            )
            
        except Exception as e:
            return HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                component="connections",
                message=f"Cannot check connections: {str(e)}",
                details={"error": str(e)},
                timestamp=time.time()
            )
    
    def _check_wal_lag(self) -> HealthCheckResult:
        """Check WAL writing lag."""
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT pg_wal_lsn_diff(pg_current_wal_lsn(), pg_last_wal_replay_lsn())
            """)
            
            lag_bytes = cursor.fetchone()[0]
            lag_mb = lag_bytes / (1024 * 1024)
            
            cursor.close()
            conn.close()
            
            if lag_mb > 1024:  # 1GB
                status = HealthStatus.UNHEALTHY
            elif lag_mb > 256:  # 256MB
                status = HealthStatus.DEGRADED
            else:
                status = HealthStatus.HEALTHY
            
            return HealthCheckResult(
                status=status,
                component="wal_lag",
                message=f"WAL lag: {lag_mb:.2f} MB",
                details={"lag_mb": lag_mb},
                timestamp=time.time()
            )
            
        except Exception as e:
            return HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                component="wal_lag",
                message=f"Cannot check WAL: {str(e)}",
                details={"error": str(e)},
                timestamp=time.time()
            )
    
    def _check_disk_space(self) -> HealthCheckResult:
        """Check disk space availability."""
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT 
                    pg_tablespace_name(spcname) as tablespace,
                    pg_tablespace_location(oid) as location,
                    (SELECT setting FROM pg_settings WHERE name = 'data_directory') as data_dir
                FROM pg_tablespace
                WHERE spcname = 'pg_default'
            """)
            
            # Simplified check - just verify we can write
            cursor.execute("CREATE TABLE IF NOT EXISTS health_check_disk (id int)")
            cursor.execute("DROP TABLE IF EXISTS health_check_disk")
            conn.commit()
            
            cursor.close()
            conn.close()
            
            return HealthCheckResult(
                status=HealthStatus.HEALTHY,
                component="disk_space",
                message="Disk space OK",
                details={},
                timestamp=time.time()
            )
            
        except Exception as e:
            return HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                component="disk_space",
                message=f"Disk space issue: {str(e)}",
                details={"error": str(e)},
                timestamp=time.time()
            )
    
    def _check_locks(self) -> HealthCheckResult:
        """Check for problematic locks."""
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT count(*)
                FROM pg_locks
                WHERE mode IN ('AccessExclusiveLock', 'ExclusiveLock')
                AND granted = true
                AND NOT EXISTS (
                    SELECT 1 FROM pg_stat_activity 
                    WHERE pid = pg_locks.pid 
                    AND state = 'idle in transaction'
                )
            """)
            
            lock_count = cursor.fetchone()[0]
            cursor.close()
            conn.close()
            
            if lock_count > 10:
                status = HealthStatus.DEGRADED
            else:
                status = HealthStatus.HEALTHY
            
            return HealthCheckResult(
                status=status,
                component="locks",
                message=f"Active locks: {lock_count}",
                details={"lock_count": lock_count},
                timestamp=time.time()
            )
            
        except Exception as e:
            return HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                component="locks",
                message=f"Cannot check locks: {str(e)}",
                details={"error": str(e)},
                timestamp=time.time()
            )
    
    def _check_slow_queries(self) -> HealthCheckResult:
        """Check for long-running queries."""
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT count(*)
                FROM pg_stat_activity
                WHERE state = 'active'
                AND query_start < now() - interval '5 minutes'
            """)
            
            slow_count = cursor.fetchone()[0]
            cursor.close()
            conn.close()
            
            if slow_count > 5:
                status = HealthStatus.DEGRADED
            else:
                status = HealthStatus.HEALTHY
            
            return HealthCheckResult(
                status=status,
                component="slow_queries",
                message=f"Slow queries: {slow_count}",
                details={"slow_query_count": slow_count},
                timestamp=time.time()
            )
            
        except Exception as e:
            return HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                component="slow_queries",
                message=f"Cannot check queries: {str(e)}",
                details={"error": str(e)},
                timestamp=time.time()
            )
    
    def _check_cache_ratio(self) -> HealthCheckResult:
        """Check database cache hit ratio."""
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT 
                    sum(heap_blks_read) as heap_read,
                    sum(heap_blks_hit) as heap_hit,
                    sum(idx_blks_read) as idx_read,
                    sum(idx_blks_hit) as idx_hit
                FROM pg_statio_user_tables
            """)
            
            heap_read, heap_hit, idx_read, idx_hit = cursor.fetchone()
            cursor.close()
            conn.close()
            
            total_reads = heap_read + idx_read
            total_hits = heap_hit + idx_hit
            
            if total_reads == 0:
                ratio = 100.0
            else:
                ratio = (total_hits / (total_reads + total_hits)) * 100
            
            if ratio < 80:
                status = HealthStatus.DEGRADED
            else:
                status = HealthStatus.HEALTHY
            
            return HealthCheckResult(
                status=status,
                component="cache_hit_ratio",
                message=f"Cache hit ratio: {ratio:.1f}%",
                details={"ratio": ratio, "hits": total_hits, "reads": total_reads},
                timestamp=time.time()
            )
            
        except Exception as e:
            return HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                component="cache_hit_ratio",
                message=f"Cannot check cache: {str(e)}",
                details={"error": str(e)},
                timestamp=time.time()
            )
```

---

## 6. Failover Automation

### 6.1 Automated Database Failover System

```python
import time
import threading
from dataclasses import dataclass
from typing import Optional, List, Callable, Dict
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class FailoverState(Enum):
    IDLE = "idle"
    DETECTING = "detecting"
    FAILING_OVER = "failing_over"
    VERIFYING = "verifying"
    RECOVERED = "recovered"
    FAILED = "failed"

@dataclass
class FailoverConfig:
    health_check_interval: float = 5.0       # seconds
    failure_threshold: int = 3                # consecutive failures before failover
    success_threshold: int = 2                # consecutive successes before recovery
    failover_timeout: float = 60.0            # max time for failover
    pre_failover_script: Optional[str] = None
    post_failover_script: Optional[str] = None
    verification_queries: List[str] = None

@dataclass
class FailoverEvent:
    timestamp: float
    state: FailoverState
    previous_primary: str
    new_primary: str
    duration: float
    success: bool
    message: str

class AutomatedFailoverController:
    """
    Automated failover controller for database clusters.
    """
    
    def __init__(
        self,
        cluster_name: str,
        config: FailoverConfig,
        health_checker: 'DatabaseHealthChecker'
    ):
        self.cluster_name = cluster_name
        self.config = config
        self.health_checker = health_checker
        
        self._state = FailoverState.IDLE
        self._current_primary = ""
        self._failure_count = 0
        self._success_count = 0
        self._lock = threading.Lock()
        
        self._failover_history: List[FailoverEvent] = []
        self._observers: List[Callable] = []
        
        # Start monitoring thread
        self._running = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop)
        self._monitor_thread.daemon = True
        self._monitor_thread.start()
    
    def add_observer(self, callback: Callable):
        """Add observer for failover events."""
        self._observers.append(callback)
    
    def _notify_observers(self, event: FailoverEvent):
        """Notify observers of failover events."""
        for observer in self._observers:
            try:
                observer(event)
            except Exception as e:
                logger.error(f"Observer notification failed: {e}")
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self._running:
            try:
                self._check_health()
            except Exception as e:
                logger.error(f"Health check failed: {e}")
            
            time.sleep(self.config.health_check_interval)
    
    def _check_health(self):
        """Check primary health and trigger failover if needed."""
        
        health = self.health_checker.check_all()
        is_healthy = health["status"] in ("healthy", "degraded")
        
        with self._lock:
            if is_healthy:
                self._failure_count = 0
                self._success_count += 1
                
                if self._state == FailoverState.DETECTING:
                    # Recovered, return to idle
                    self._state = FailoverState.IDLE
                    logger.info("Primary recovered, monitoring continues")
            else:
                self._failure_count += 1
                self._success_count = 0
                
                logger.warning(
                    f"Primary unhealthy: {health['summary']}"
                )
                
                # Check if we should failover
                if (self._state == FailoverState.IDLE and 
                    self._failure_count >= self.config.failure_threshold):
                    
                    self._initiate_failover()
    
    def _initiate_failover(self):
        """Initiate failover process."""
        
        logger.warning("Initiating failover!")
        self._state = FailoverState.FAILING_OVER
        start_time = time.time()
        
        # Pre-failover hook
        if self.config.pre_failover_script:
            self._run_script(self.config.pre_failover_script)
        
        # Find new primary
        new_primary = self._discover_new_primary()
        
        if not new_primary:
            self._state = FailoverState.FAILED
            self._notify_observers(FailoverEvent(
                timestamp=time.time(),
                state=self._state,
                previous_primary=self._current_primary,
                new_primary="",
                duration=time.time() - start_time,
                success=False,
                message="No suitable replica found"
            ))
            return
        
        # Perform failover
        success = self._execute_failover(new_primary)
        
        duration = time.time() - start_time
        
        if success:
            self._state = FailoverState.RECOVERED
            self._current_primary = new_primary
            
            # Post-failover hook
            if self.config.post_failover_script:
                self._run_script(self.config.post_failover_script)
            
            event = FailoverEvent(
                timestamp=time.time(),
                state=self._state,
                previous_primary=self._current_primary,
                new_primary=new_primary,
                duration=duration,
                success=True,
                message=f"Failover completed in {duration:.2f}s"
            )
            
            logger.info(f"Failover successful: {new_primary}")
        else:
            self._state = FailoverState.FAILED
            
            event = FailoverEvent(
                timestamp=time.time(),
                state=self._state,
                previous_primary=self._current_primary,
                new_primary=new_primary,
                duration=duration,
                success=False,
                message="Failover failed"
            )
        
        self._failover_history.append(event)
        self._notify_observers(event)
    
    def _discover_new_primary(self) -> Optional[str]:
        """Discover the best replica to promote."""
        
        # Implementation depends on cluster type (Patroni, etcd, etc.)
        # Simplified implementation:
        
        logger.info("Discovering new primary...")
        
        # Get list of replicas
        replicas = self._get_replicas()
        
        for replica in replicas:
            # Check if replica is healthy and up-to-date
            if self._is_suitable_primary(replica):
                return replica
        
        return None
    
    def _get_replicas(self) -> List[str]:
        """Get list of replica endpoints."""
        # Implementation-specific
        return ["replica-1.database.svc.cluster.local"]
    
    def _is_suitable_primary(self, replica: str) -> bool:
        """Check if replica is suitable for promotion."""
        # Check replication lag, health, etc.
        return True
    
    def _execute_failover(self, new_primary: str) -> bool:
        """Execute the actual failover."""
        
        logger.info(f"Executing failover to {new_primary}")
        
        # Implementation depends on cluster technology
        # For example, Patroni:
        # subprocess.run([
        #     "patronictl", "-c", "/etc/patroni.yml",
        #     "switchover", "--force", "--candidate", new_primary
        # ])
        
        return True
    
    def _verify_failover(self) -> bool:
        """Verify failover was successful."""
        
        if not self.config.verification_queries:
            return True
        
        for query in self.config.verification_queries:
            try:
                result = self.health_checker.db.execute_with_circuit_breaker(query)
                if not result:
                    return False
            except Exception:
                return False
        
        return True
    
    def _run_script(self, script: str):
        """Run a script (pre/post failover)."""
        logger.info(f"Running script: {script}")
        # subprocess.run([script], check=True)
    
    def get_status(self) -> Dict:
        """Get current failover controller status."""
        with self._lock:
            return {
                "cluster": self.cluster_name,
                "state": self._state.value,
                "current_primary": self._current_primary,
                "failure_count": self._failure_count,
                "success_count": self._success_count,
                "history": [
                    {
                        "timestamp": e.timestamp,
                        "state": e.state.value,
                        "duration": e.duration,
                        "success": e.success
                    }
                    for e in self._failover_history[-10:]
                ]
            }
    
    def stop(self):
        """Stop the failover controller."""
        self._running = False
        self._monitor_thread.join()
```

### 6.2 Kubernetes Integration for Failover

```yaml
# Kubernetes service for database failover
apiVersion: v1
kind: Service
metadata:
  name: postgres-primary
  namespace: database
  annotations:
    service.beta.kubernetes.io/aws-load-balancer-type: "nlb"
spec:
  type: ClusterIP
  ports:
  - port: 5432
    targetPort: 5432
    name: postgres
  selector:
    role: primary
---
apiVersion: v1
kind: Service
metadata:
  name: postgres-replicas
  namespace: database
spec:
  type: ClusterIP
  ports:
  - port: 5432
    targetPort: 5432
    name: postgres
  selector:
    role: replica
---
# Pod disruption budget for controlled failover
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: postgres-pdb
  namespace: database
spec:
  minAvailable: 1
  selector:
    matchLabels:
      app: postgres
```

```python
# Kubernetes-aware failover notifier
class KubernetesFailoverNotifier:
    """Notify Kubernetes of database failover events."""
    
    def __init__(self, namespace: str):
        self.namespace = namespace
        self.kubernetes_client = None  # Initialize kubernetes client
    
    def handle_failover_event(self, event: FailoverEvent):
        """Handle failover event and update Kubernetes resources."""
        
        if event.success:
            # Update service endpoints
            self._update_primary_service(event.new_primary)
            
            # Annotate the new primary pod
            self._annotate_pod_as_primary(event.new_primary)
            
            # Notify monitoring
            self._send_notification(event)
        else:
            # Handle failure - may need to create incident
            self._handle_failover_failure(event)
    
    def _update_primary_service(self, new_primary: str):
        """Update the primary service to point to new primary."""
        # Implementation would use Kubernetes API to update endpoints
        logger.info(f"Updating primary service to: {new_primary}")
    
    def _annotate_pod_as_primary(self, pod_name: str):
        """Annotate pod as primary."""
        # Implementation would patch pod annotations
        logger.info(f"Annotating {pod_name} as primary")
    
    def _send_notification(self, event: FailoverEvent):
        """Send notification about failover."""
        message = f"""
        Database Failover Completed
        
        Cluster: {event.previous_primary} -> {event.new_primary}
        Duration: {event.duration:.2f}s
        Status: {"SUCCESS" if event.success else "FAILED"}
        
        Timestamp: {event.timestamp}
        """
        
        # Send to Slack, PagerDuty, etc.
        logger.info(message)
```

---

## Summary

This document covers essential database reliability patterns:

1. **Circuit Breakers**: Prevents cascading failures by tracking errors and opening circuits when thresholds are exceeded
2. **Bulkheads**: Isolates database resources to prevent total system failure
3. **Retry Policies**: Implements exponential backoff with jitter for transient failures
4. **Graceful Degradation**: Maintains partial functionality during database issues
5. **Health Checks**: Comprehensive monitoring of database health metrics
6. **Failover Automation**: Automatic detection and recovery from primary failures

All implementations include production-ready patterns with proper metrics collection, logging, and integration points for monitoring and alerting systems.
