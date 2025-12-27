#!/usr/bin/env python3
"""
ENTERPRISE RESILIENCE FRAMEWORK
Unified connection management with circuit breakers, retry logic, and graceful degradation
"""

import asyncio
import functools
import time
import threading
import random
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any, Callable, TypeVar, Generic, List, Union
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import structlog

logger = structlog.get_logger(__name__)

# Type variable for generic return types
T = TypeVar('T')


class ConnectionState(str, Enum):
    """Connection state machine states"""
    DISCONNECTED = "DISCONNECTED"
    CONNECTING = "CONNECTING"
    CONNECTED = "CONNECTED"
    RECONNECTING = "RECONNECTING"
    DEGRADED = "DEGRADED"  # Connected but with issues
    FAILED = "FAILED"  # Permanent failure
    CIRCUIT_OPEN = "CIRCUIT_OPEN"  # Circuit breaker triggered


class CircuitState(str, Enum):
    """Circuit breaker states"""
    CLOSED = "CLOSED"  # Normal operation
    OPEN = "OPEN"  # Blocking requests
    HALF_OPEN = "HALF_OPEN"  # Testing recovery


@dataclass
class RetryConfig:
    """Configuration for retry behavior"""
    max_retries: int = 5
    initial_delay_ms: int = 1000
    max_delay_ms: int = 60000
    exponential_base: float = 2.0
    jitter: bool = True
    jitter_factor: float = 0.1
    retryable_exceptions: tuple = (Exception,)


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker"""
    failure_threshold: int = 5  # Failures before opening
    success_threshold: int = 3  # Successes to close from half-open
    timeout_seconds: float = 30.0  # Time before trying half-open
    half_open_max_calls: int = 3  # Max test calls in half-open


@dataclass
class ConnectionHealth:
    """Health metrics for a connection"""
    state: ConnectionState = ConnectionState.DISCONNECTED
    last_successful_operation: Optional[datetime] = None
    last_failed_operation: Optional[datetime] = None
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    total_operations: int = 0
    total_failures: int = 0
    average_latency_ms: float = 0.0
    error_rate_percent: float = 0.0
    last_error: Optional[str] = None
    circuit_state: CircuitState = CircuitState.CLOSED


class CircuitBreaker:
    """
    Circuit breaker implementation to prevent cascade failures.

    States:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Too many failures, requests are rejected immediately
    - HALF_OPEN: Testing if service recovered
    """

    def __init__(self, name: str, config: Optional[CircuitBreakerConfig] = None):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: Optional[float] = None
        self._half_open_calls = 0
        self._lock = threading.RLock()

        logger.info(f"Circuit breaker initialized",
                   name=name,
                   failure_threshold=self.config.failure_threshold)

    @property
    def state(self) -> CircuitState:
        with self._lock:
            # Check if we should transition from OPEN to HALF_OPEN
            if self._state == CircuitState.OPEN:
                if self._last_failure_time:
                    elapsed = time.time() - self._last_failure_time
                    if elapsed >= self.config.timeout_seconds:
                        self._transition_to(CircuitState.HALF_OPEN)
            return self._state

    def _transition_to(self, new_state: CircuitState):
        """Transition to new state with logging"""
        old_state = self._state
        self._state = new_state

        if new_state == CircuitState.HALF_OPEN:
            self._half_open_calls = 0
            self._success_count = 0

        logger.warning(f"Circuit breaker state transition",
                      name=self.name,
                      from_state=old_state,
                      to_state=new_state)

    def record_success(self):
        """Record a successful operation"""
        with self._lock:
            self._failure_count = 0

            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.config.success_threshold:
                    self._transition_to(CircuitState.CLOSED)
                    logger.info(f"Circuit breaker recovered", name=self.name)

            elif self._state == CircuitState.CLOSED:
                self._success_count += 1

    def record_failure(self, error: Optional[str] = None):
        """Record a failed operation"""
        with self._lock:
            self._failure_count += 1
            self._success_count = 0
            self._last_failure_time = time.time()

            if self._state == CircuitState.HALF_OPEN:
                # Any failure in half-open immediately opens circuit
                self._transition_to(CircuitState.OPEN)
                logger.error(f"Circuit breaker tripped in half-open",
                            name=self.name, error=error)

            elif self._state == CircuitState.CLOSED:
                if self._failure_count >= self.config.failure_threshold:
                    self._transition_to(CircuitState.OPEN)
                    logger.error(f"Circuit breaker tripped",
                                name=self.name,
                                failures=self._failure_count,
                                error=error)

    def allow_request(self) -> bool:
        """Check if request should be allowed through"""
        state = self.state  # This also handles OPEN -> HALF_OPEN transition

        with self._lock:
            if state == CircuitState.CLOSED:
                return True

            elif state == CircuitState.OPEN:
                return False

            elif state == CircuitState.HALF_OPEN:
                if self._half_open_calls < self.config.half_open_max_calls:
                    self._half_open_calls += 1
                    return True
                return False

        return False

    def reset(self):
        """Force reset the circuit breaker"""
        with self._lock:
            self._failure_count = 0
            self._success_count = 0
            self._half_open_calls = 0
            self._last_failure_time = None
            self._transition_to(CircuitState.CLOSED)
            logger.info(f"Circuit breaker reset", name=self.name)


class CircuitBreakerError(Exception):
    """Raised when circuit breaker is open"""
    pass


def with_circuit_breaker(circuit_breaker: CircuitBreaker):
    """Decorator to wrap function with circuit breaker protection"""
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            if not circuit_breaker.allow_request():
                raise CircuitBreakerError(
                    f"Circuit breaker '{circuit_breaker.name}' is open"
                )

            try:
                result = func(*args, **kwargs)
                circuit_breaker.record_success()
                return result
            except Exception as e:
                circuit_breaker.record_failure(str(e))
                raise

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs) -> T:
            if not circuit_breaker.allow_request():
                raise CircuitBreakerError(
                    f"Circuit breaker '{circuit_breaker.name}' is open"
                )

            try:
                result = await func(*args, **kwargs)
                circuit_breaker.record_success()
                return result
            except Exception as e:
                circuit_breaker.record_failure(str(e))
                raise

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return wrapper

    return decorator


def calculate_backoff_delay(
    attempt: int,
    config: RetryConfig
) -> float:
    """Calculate delay with exponential backoff and optional jitter"""
    delay_ms = config.initial_delay_ms * (config.exponential_base ** attempt)
    delay_ms = min(delay_ms, config.max_delay_ms)

    if config.jitter:
        jitter_range = delay_ms * config.jitter_factor
        delay_ms += random.uniform(-jitter_range, jitter_range)

    return max(delay_ms, 0) / 1000.0  # Convert to seconds


def with_retry(config: Optional[RetryConfig] = None, on_retry: Optional[Callable] = None):
    """
    Decorator to add retry logic with exponential backoff.

    Usage:
        @with_retry(RetryConfig(max_retries=3))
        def my_function():
            ...
    """
    retry_config = config or RetryConfig()

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            last_exception = None

            for attempt in range(retry_config.max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except retry_config.retryable_exceptions as e:
                    last_exception = e

                    if attempt < retry_config.max_retries:
                        delay = calculate_backoff_delay(attempt, retry_config)

                        logger.warning(f"Retry attempt",
                                      function=func.__name__,
                                      attempt=attempt + 1,
                                      max_retries=retry_config.max_retries,
                                      delay_seconds=delay,
                                      error=str(e))

                        if on_retry:
                            on_retry(attempt, e)

                        time.sleep(delay)

            logger.error(f"All retries exhausted",
                        function=func.__name__,
                        attempts=retry_config.max_retries + 1)
            if last_exception is not None:
                raise last_exception
            else:
                raise RuntimeError(f"All retries exhausted for {func.__name__}")

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs) -> T:
            last_exception = None

            for attempt in range(retry_config.max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except retry_config.retryable_exceptions as e:
                    last_exception = e

                    if attempt < retry_config.max_retries:
                        delay = calculate_backoff_delay(attempt, retry_config)

                        logger.warning(f"Retry attempt (async)",
                                      function=func.__name__,
                                      attempt=attempt + 1,
                                      max_retries=retry_config.max_retries,
                                      delay_seconds=delay,
                                      error=str(e))

                        if on_retry:
                            on_retry(attempt, e)

                        await asyncio.sleep(delay)

            logger.error(f"All retries exhausted (async)",
                        function=func.__name__,
                        attempts=retry_config.max_retries + 1)
            if last_exception is not None:
                raise last_exception
            else:
                raise RuntimeError(f"All retries exhausted for {func.__name__}")

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return wrapper

    return decorator


class GracefulDegradation:
    """
    Handles graceful degradation when services are unavailable.
    Provides fallback values and cached data when primary source fails.
    """

    def __init__(self, name: str, cache_ttl_seconds: float = 300.0):
        self.name = name
        self.cache_ttl_seconds = cache_ttl_seconds
        self._cache: Dict[str, Any] = {}
        self._cache_timestamps: Dict[str, float] = {}
        self._degraded_mode = False
        self._lock = threading.RLock()

        logger.info(f"Graceful degradation handler initialized",
                   name=name, cache_ttl=cache_ttl_seconds)

    def cache_value(self, key: str, value: Any):
        """Cache a value for fallback use"""
        with self._lock:
            self._cache[key] = value
            self._cache_timestamps[key] = time.time()

    def get_cached(self, key: str, default: Any = None) -> Optional[Any]:
        """Get cached value if still valid"""
        with self._lock:
            if key not in self._cache:
                return default

            timestamp = self._cache_timestamps.get(key, 0)
            if time.time() - timestamp > self.cache_ttl_seconds:
                # Cache expired
                del self._cache[key]
                del self._cache_timestamps[key]
                return default

            return self._cache[key]

    def enter_degraded_mode(self, reason: str):
        """Enter degraded mode"""
        with self._lock:
            if not self._degraded_mode:
                self._degraded_mode = True
                logger.warning(f"Entering degraded mode",
                              name=self.name, reason=reason)

    def exit_degraded_mode(self):
        """Exit degraded mode"""
        with self._lock:
            if self._degraded_mode:
                self._degraded_mode = False
                logger.info(f"Exiting degraded mode", name=self.name)

    @property
    def is_degraded(self) -> bool:
        return self._degraded_mode

    def execute_with_fallback(
        self,
        primary: Callable[[], T],
        fallback: Callable[[], T],
        cache_key: Optional[str] = None
    ) -> T:
        """Execute primary function with fallback on failure"""
        try:
            result = primary()

            # Cache successful result
            if cache_key:
                self.cache_value(cache_key, result)

            # Exit degraded mode on success
            if self._degraded_mode:
                self.exit_degraded_mode()

            return result

        except Exception as e:
            logger.warning(f"Primary function failed, using fallback",
                          name=self.name,
                          error=str(e))

            self.enter_degraded_mode(str(e))

            # Try cached value first if available
            if cache_key:
                cached = self.get_cached(cache_key)
                if cached is not None:
                    logger.info(f"Using cached value",
                               name=self.name, cache_key=cache_key)
                    return cached

            # Use fallback
            return fallback()


def with_fallback(
    fallback_value: Any = None,
    fallback_func: Optional[Callable] = None,
    log_error: bool = True
):
    """
    Decorator to add fallback behavior on exception.

    Usage:
        @with_fallback(fallback_value=[])
        def get_data():
            ...
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if log_error:
                    logger.warning(f"Function failed, using fallback",
                                  function=func.__name__,
                                  error=str(e))

                if fallback_func:
                    return fallback_func(*args, **kwargs)
                return fallback_value

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs) -> T:
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                if log_error:
                    logger.warning(f"Async function failed, using fallback",
                                  function=func.__name__,
                                  error=str(e))

                if fallback_func:
                    if asyncio.iscoroutinefunction(fallback_func):
                        return await fallback_func(*args, **kwargs)
                    return fallback_func(*args, **kwargs)
                return fallback_value

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return wrapper

    return decorator


class ConnectionManagerBase(ABC):
    """
    Base class for resilient connection management.
    Provides state machine, health tracking, and reconnection logic.
    """

    def __init__(
        self,
        name: str,
        retry_config: Optional[RetryConfig] = None,
        circuit_config: Optional[CircuitBreakerConfig] = None
    ):
        self.name = name
        self.retry_config = retry_config or RetryConfig()

        # State management
        self._state = ConnectionState.DISCONNECTED
        self._state_lock = threading.RLock()
        self._state_change_callbacks: List[Callable] = []

        # Health tracking
        self._health = ConnectionHealth()
        self._health_lock = threading.RLock()

        # Circuit breaker
        self.circuit_breaker = CircuitBreaker(
            name=f"{name}_circuit",
            config=circuit_config
        )

        # Graceful degradation
        self.degradation = GracefulDegradation(
            name=f"{name}_degradation"
        )

        # Reconnection management
        self._reconnect_task: Optional[asyncio.Task] = None
        self._should_reconnect = True

        logger.info(f"Connection manager initialized", name=name)

    @property
    def state(self) -> ConnectionState:
        with self._state_lock:
            return self._state

    @property
    def health(self) -> ConnectionHealth:
        with self._health_lock:
            return ConnectionHealth(
                state=self._health.state,
                last_successful_operation=self._health.last_successful_operation,
                last_failed_operation=self._health.last_failed_operation,
                consecutive_failures=self._health.consecutive_failures,
                consecutive_successes=self._health.consecutive_successes,
                total_operations=self._health.total_operations,
                total_failures=self._health.total_failures,
                average_latency_ms=self._health.average_latency_ms,
                error_rate_percent=self._health.error_rate_percent,
                last_error=self._health.last_error,
                circuit_state=self.circuit_breaker.state
            )

    def _set_state(self, new_state: ConnectionState):
        """Set connection state with callbacks"""
        with self._state_lock:
            old_state = self._state
            if old_state == new_state:
                return

            self._state = new_state
            self._health.state = new_state

            logger.info(f"Connection state changed",
                       name=self.name,
                       from_state=old_state,
                       to_state=new_state)

            # Notify callbacks
            for callback in self._state_change_callbacks:
                try:
                    callback(old_state, new_state)
                except Exception as e:
                    logger.error(f"State change callback error", error=str(e))

    def on_state_change(self, callback: Callable[[ConnectionState, ConnectionState], None]):
        """Register callback for state changes"""
        self._state_change_callbacks.append(callback)

    def record_success(self, latency_ms: Optional[float] = None):
        """Record successful operation"""
        with self._health_lock:
            self._health.last_successful_operation = datetime.now(timezone.utc)
            self._health.consecutive_successes += 1
            self._health.consecutive_failures = 0
            self._health.total_operations += 1

            if latency_ms is not None:
                # Rolling average
                alpha = 0.1
                self._health.average_latency_ms = (
                    alpha * latency_ms +
                    (1 - alpha) * self._health.average_latency_ms
                )

            # Update error rate
            if self._health.total_operations > 0:
                self._health.error_rate_percent = (
                    self._health.total_failures / self._health.total_operations * 100
                )

        self.circuit_breaker.record_success()

        # If we were degraded and now succeeding, exit degraded
        if self.state == ConnectionState.DEGRADED:
            if self._health.consecutive_successes >= 3:
                self._set_state(ConnectionState.CONNECTED)
                self.degradation.exit_degraded_mode()

    def record_failure(self, error: str):
        """Record failed operation"""
        with self._health_lock:
            self._health.last_failed_operation = datetime.now(timezone.utc)
            self._health.consecutive_failures += 1
            self._health.consecutive_successes = 0
            self._health.total_operations += 1
            self._health.total_failures += 1
            self._health.last_error = error

            # Update error rate
            if self._health.total_operations > 0:
                self._health.error_rate_percent = (
                    self._health.total_failures / self._health.total_operations * 100
                )

        self.circuit_breaker.record_failure(error)

        # Handle state transitions on failure
        if self.state == ConnectionState.CONNECTED:
            if self._health.consecutive_failures >= 3:
                self._set_state(ConnectionState.DEGRADED)
                self.degradation.enter_degraded_mode(error)

        if self.circuit_breaker.state == CircuitState.OPEN:
            self._set_state(ConnectionState.CIRCUIT_OPEN)

    @abstractmethod
    async def _connect(self) -> bool:
        """Implement actual connection logic"""
        pass

    @abstractmethod
    async def _disconnect(self):
        """Implement actual disconnection logic"""
        pass

    @abstractmethod
    async def _health_check(self) -> bool:
        """Implement health check logic"""
        pass

    async def connect(self) -> bool:
        """Connect with retry logic"""
        if self.state == ConnectionState.CONNECTED:
            return True

        self._set_state(ConnectionState.CONNECTING)

        for attempt in range(self.retry_config.max_retries + 1):
            try:
                if await self._connect():
                    self._set_state(ConnectionState.CONNECTED)
                    self.record_success()
                    return True

            except Exception as e:
                self.record_failure(str(e))

                if attempt < self.retry_config.max_retries:
                    delay = calculate_backoff_delay(attempt, self.retry_config)
                    logger.warning(f"Connection attempt failed, retrying",
                                  name=self.name,
                                  attempt=attempt + 1,
                                  delay_seconds=delay,
                                  error=str(e))
                    await asyncio.sleep(delay)

        self._set_state(ConnectionState.FAILED)
        return False

    async def disconnect(self):
        """Disconnect gracefully"""
        self._should_reconnect = False

        if self._reconnect_task:
            self._reconnect_task.cancel()

        try:
            await self._disconnect()
        finally:
            self._set_state(ConnectionState.DISCONNECTED)

    async def reconnect(self):
        """Attempt reconnection"""
        if self.state == ConnectionState.RECONNECTING:
            return

        self._set_state(ConnectionState.RECONNECTING)
        await self.connect()

    async def _auto_reconnect_loop(self):
        """Background task for automatic reconnection"""
        while self._should_reconnect:
            try:
                if self.state in (ConnectionState.DISCONNECTED,
                                  ConnectionState.FAILED,
                                  ConnectionState.CIRCUIT_OPEN):

                    # Wait for circuit breaker if open
                    if self.circuit_breaker.state == CircuitState.OPEN:
                        await asyncio.sleep(self.circuit_breaker.config.timeout_seconds)

                    await self.reconnect()

                # Check health periodically
                if self.state == ConnectionState.CONNECTED:
                    if not await self._health_check():
                        self.record_failure("Health check failed")

                await asyncio.sleep(10)  # Check every 10 seconds

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Auto-reconnect error", name=self.name, error=str(e))
                await asyncio.sleep(30)

    def start_auto_reconnect(self):
        """Start automatic reconnection task"""
        self._should_reconnect = True
        self._reconnect_task = asyncio.create_task(self._auto_reconnect_loop())

    def get_health_report(self) -> Dict[str, Any]:
        """Get detailed health report"""
        health = self.health
        return {
            'name': self.name,
            'state': health.state,
            'circuit_state': health.circuit_state,
            'is_healthy': health.state == ConnectionState.CONNECTED,
            'is_degraded': self.degradation.is_degraded,
            'last_successful': health.last_successful_operation.isoformat() if health.last_successful_operation else None,
            'last_failed': health.last_failed_operation.isoformat() if health.last_failed_operation else None,
            'consecutive_failures': health.consecutive_failures,
            'consecutive_successes': health.consecutive_successes,
            'total_operations': health.total_operations,
            'total_failures': health.total_failures,
            'error_rate_percent': round(health.error_rate_percent, 2),
            'average_latency_ms': round(health.average_latency_ms, 2),
            'last_error': health.last_error
        }


class WriteBuffer:
    """
    Buffers write operations when database is slow or unavailable.
    Flushes automatically when connection recovers.
    """

    def __init__(
        self,
        name: str,
        max_buffer_size: int = 1000,
        flush_interval_seconds: float = 5.0
    ):
        self.name = name
        self.max_buffer_size = max_buffer_size
        self.flush_interval = flush_interval_seconds

        self._buffer: List[Dict[str, Any]] = []
        self._lock = threading.RLock()
        self._flush_callback: Optional[Callable] = None
        self._flush_task: Optional[asyncio.Task] = None

        logger.info(f"Write buffer initialized",
                   name=name,
                   max_size=max_buffer_size)

    def add(self, operation: str, data: Dict[str, Any]) -> bool:
        """Add item to buffer"""
        with self._lock:
            if len(self._buffer) >= self.max_buffer_size:
                logger.warning(f"Buffer full, dropping oldest item",
                              name=self.name)
                self._buffer.pop(0)

            self._buffer.append({
                'operation': operation,
                'data': data,
                'timestamp': datetime.now(timezone.utc).isoformat()
            })

            return True

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._buffer)

    def set_flush_callback(self, callback: Callable[[List[Dict]], bool]):
        """Set callback for flushing buffer"""
        self._flush_callback = callback

    async def flush(self) -> int:
        """Flush buffered items"""
        if not self._flush_callback:
            return 0

        with self._lock:
            if not self._buffer:
                return 0

            items = self._buffer.copy()
            self._buffer.clear()

        try:
            success = self._flush_callback(items)
            if success:
                logger.info(f"Buffer flushed",
                           name=self.name,
                           count=len(items))
                return len(items)
            else:
                # Put items back
                with self._lock:
                    self._buffer = items + self._buffer
                return 0

        except Exception as e:
            logger.error(f"Buffer flush failed",
                        name=self.name,
                        error=str(e))
            # Put items back
            with self._lock:
                self._buffer = items + self._buffer
            return 0

    async def _auto_flush_loop(self):
        """Background task for automatic buffer flushing"""
        while True:
            try:
                await asyncio.sleep(self.flush_interval)
                if self._buffer:
                    await self.flush()
            except asyncio.CancelledError:
                # Final flush before shutdown
                await self.flush()
                break
            except Exception as e:
                logger.error(f"Auto-flush error", name=self.name, error=str(e))

    def start_auto_flush(self):
        """Start automatic flush task"""
        self._flush_task = asyncio.create_task(self._auto_flush_loop())

    def stop(self):
        """Stop buffer and flush remaining items"""
        if self._flush_task:
            self._flush_task.cancel()


# Global registry for connection managers
_connection_managers: Dict[str, ConnectionManagerBase] = {}


def register_connection_manager(manager: ConnectionManagerBase):
    """Register a connection manager globally"""
    _connection_managers[manager.name] = manager
    logger.info(f"Connection manager registered", name=manager.name)


def get_connection_manager(name: str) -> Optional[ConnectionManagerBase]:
    """Get a registered connection manager by name"""
    return _connection_managers.get(name)


def get_all_connection_health() -> Dict[str, Dict[str, Any]]:
    """Get health reports for all registered connection managers"""
    return {
        name: manager.get_health_report()
        for name, manager in _connection_managers.items()
    }


# Convenience function for logging resilience events
def log_resilience_event(
    event_type: str,
    component: str,
    **kwargs
):
    """Log a resilience-related event with structured data"""
    logger.info(
        f"Resilience event: {event_type}",
        event_type=event_type,
        component=component,
        **kwargs
    )
