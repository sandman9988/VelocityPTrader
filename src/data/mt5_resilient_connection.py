#!/usr/bin/env python3
"""
MT5 RESILIENT CONNECTION MANAGER
Enterprise-grade MT5 connection with circuit breakers, retry logic, and graceful degradation
"""

import asyncio
import json
import os
import time
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass
import threading

import structlog

from ..utils.resilience import (
    ConnectionManagerBase, ConnectionState, CircuitBreaker, CircuitBreakerConfig,
    RetryConfig, GracefulDegradation, WriteBuffer, with_retry, with_fallback,
    calculate_backoff_delay, CircuitBreakerError, register_connection_manager,
    log_resilience_event
)

logger = structlog.get_logger(__name__)


@dataclass
class MT5Config:
    """MT5 connection configuration"""
    server: str = "VantageInternational-Demo"
    login: int = 0
    password: str = ""
    timeout_ms: int = 60000
    retry_attempts: int = 5
    retry_delay_ms: int = 5000
    zmq_port: int = 5555
    data_file_path: str = "all_mt5_symbols.json"
    fallback_file_path: str = "mt5_fallback_data.json"
    max_data_age_seconds: float = 30.0
    checksum_validation: bool = True


class MT5ResilientConnection(ConnectionManagerBase):
    """
    Resilient MT5 connection manager with:
    - Circuit breaker for connection failures
    - Automatic reconnection with exponential backoff
    - Graceful degradation to cached data
    - Data integrity validation
    - Health monitoring
    """

    def __init__(self, config: Optional[MT5Config] = None):
        self.config = config or MT5Config()

        # Initialize base connection manager
        super().__init__(
            name="mt5_bridge",
            retry_config=RetryConfig(
                max_retries=self.config.retry_attempts,
                initial_delay_ms=self.config.retry_delay_ms,
                max_delay_ms=60000,
                exponential_base=2.0,
                jitter=True
            ),
            circuit_config=CircuitBreakerConfig(
                failure_threshold=5,
                success_threshold=3,
                timeout_seconds=30.0
            )
        )

        # MT5 specific state
        self._zmq_context = None
        self._zmq_socket = None
        self._is_initialized = False
        self._last_tick_time: Optional[datetime] = None
        self._symbols_count = 0
        self._account_info: Dict[str, Any] = {}

        # Data cache for graceful degradation
        self._market_data_cache: Dict[str, Dict[str, Any]] = {}
        self._cache_lock = threading.RLock()

        # File paths
        self.data_file = Path(self.config.data_file_path)
        self.fallback_file = Path(self.config.fallback_file_path)

        # Register with global connection manager registry
        register_connection_manager(self)

        logger.info("MT5 resilient connection initialized",
                   server=self.config.server,
                   zmq_port=self.config.zmq_port)

    async def _connect(self) -> bool:
        """Implement actual MT5 connection logic"""
        try:
            # Validate server
            if self.config.server != "VantageInternational-Demo":
                raise ValueError("Only VantageInternational-Demo server is allowed")

            # Check if MT5 data file exists and is fresh
            if self.data_file.exists():
                file_age = time.time() - self.data_file.stat().st_mtime
                if file_age < self.config.max_data_age_seconds:
                    # Data is fresh, consider connected
                    self._load_data_from_file()
                    self._is_initialized = True
                    logger.info("MT5 connection established via data file",
                               symbols=self._symbols_count)
                    return True

            # If ZMQ is configured, try ZMQ connection
            if self.config.zmq_port > 0:
                try:
                    import zmq.asyncio
                    self._zmq_context = zmq.asyncio.Context()
                    self._zmq_socket = self._zmq_context.socket(zmq.REQ)
                    self._zmq_socket.connect(f"tcp://localhost:{self.config.zmq_port}")
                    self._zmq_socket.setsockopt(zmq.RCVTIMEO, self.config.timeout_ms)
                    self._zmq_socket.setsockopt(zmq.SNDTIMEO, self.config.timeout_ms)

                    # Send heartbeat
                    await self._zmq_socket.send_json({"command": "ping"})
                    response = await self._zmq_socket.recv_json()

                    if response.get("status") == "ok":
                        self._is_initialized = True
                        logger.info("MT5 ZMQ connection established")
                        return True

                except ImportError:
                    logger.debug("ZMQ not available, using file-based connection")
                except Exception as e:
                    logger.warning("ZMQ connection failed", error=str(e))
                    # Clean up context on failure
                    if self._zmq_context:
                        self._zmq_context.term()
                        self._zmq_context = None

            # Fallback to file-based monitoring
            if self.data_file.exists():
                self._load_data_from_file()
                self._is_initialized = True
                self._set_state(ConnectionState.DEGRADED)
                self.degradation.enter_degraded_mode("Using stale data file")
                return True

            # Load fallback data if available
            if self.fallback_file.exists():
                self._load_fallback_data()
                self._is_initialized = True
                self._set_state(ConnectionState.DEGRADED)
                self.degradation.enter_degraded_mode("Using fallback data")
                return True

            return False

        except Exception as e:
            logger.error("MT5 connection failed", error=str(e))
            raise

    async def _disconnect(self):
        """Implement actual MT5 disconnection logic"""
        try:
            if self._zmq_socket:
                self._zmq_socket.close()
                self._zmq_socket = None

            if self._zmq_context:
                self._zmq_context.term()
                self._zmq_context = None

            self._is_initialized = False
            logger.info("MT5 connection closed")

        except Exception as e:
            logger.error("Error closing MT5 connection", error=str(e))

    async def _health_check(self) -> bool:
        """Check MT5 connection health"""
        try:
            # Check data file freshness
            if self.data_file.exists():
                file_age = time.time() - self.data_file.stat().st_mtime
                if file_age < self.config.max_data_age_seconds:
                    return True

            # Check ZMQ connection
            if self._zmq_socket:
                try:
                    await self._zmq_socket.send_json({"command": "ping"})
                    response = await self._zmq_socket.recv_json()
                    return response.get("status") == "ok"
                except Exception:
                    return False

            return False

        except Exception as e:
            logger.warning("MT5 health check failed", error=str(e))
            return False

    def _load_data_from_file(self):
        """Load market data from MT5 data file"""
        try:
            with open(self.data_file, 'r') as f:
                data = json.load(f)

            # Validate checksum if enabled
            if self.config.checksum_validation and 'checksum' in data:
                if not self._validate_checksum(data):
                    raise ValueError("Data integrity check failed")

            self._account_info = data.get('account', {})
            symbols = data.get('symbols', {})
            self._symbols_count = len(symbols)

            # Cache market data
            with self._cache_lock:
                self._market_data_cache = symbols

            # Update cache for degradation handler
            self.degradation.cache_value('symbols', symbols)
            self.degradation.cache_value('account', self._account_info)

            self._last_tick_time = datetime.now(timezone.utc)

            logger.debug("MT5 data loaded from file",
                        symbols_count=self._symbols_count)

        except Exception as e:
            logger.error("Failed to load MT5 data file", error=str(e))
            raise

    def _load_fallback_data(self):
        """Load fallback data when primary source unavailable"""
        try:
            with open(self.fallback_file, 'r') as f:
                data = json.load(f)

            self._account_info = data.get('account', {})
            symbols = data.get('symbols', {})
            self._symbols_count = len(symbols)

            with self._cache_lock:
                self._market_data_cache = symbols

            logger.warning("Using MT5 fallback data",
                          symbols_count=self._symbols_count)

        except Exception as e:
            logger.error("Failed to load MT5 fallback data", error=str(e))
            raise

    def _validate_checksum(self, data: Dict[str, Any]) -> bool:
        """Validate data integrity using checksum"""
        try:
            stored_checksum = data.get('checksum', '')
            data_copy = {k: v for k, v in data.items() if k != 'checksum'}
            computed_checksum = hashlib.sha256(
                json.dumps(data_copy, sort_keys=True).encode()
            ).hexdigest()

            return stored_checksum == computed_checksum

        except Exception as e:
            logger.warning("Checksum validation failed", error=str(e))
            return False

    def _save_fallback_data(self):
        """Save current data as fallback for future use"""
        try:
            with self._cache_lock:
                if not self._market_data_cache:
                    return

                fallback_data = {
                    'account': self._account_info,
                    'symbols': self._market_data_cache,
                    'saved_at': datetime.now(timezone.utc).isoformat(),
                    'source': 'fallback_save'
                }

            with open(self.fallback_file, 'w') as f:
                json.dump(fallback_data, f, indent=2)

            logger.info("MT5 fallback data saved")

        except Exception as e:
            logger.error("Failed to save fallback data", error=str(e))

    def get_market_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get market data with circuit breaker and fallback.
        Returns cached data if connection is degraded.
        """
        # Check circuit breaker
        if not self.circuit_breaker.allow_request():
            # Return cached data
            cached = self.degradation.get_cached(f'symbol_{symbol}')
            if cached:
                log_resilience_event("circuit_breaker_bypass", "mt5",
                                    symbol=symbol, using="cache")
                return cached
            raise CircuitBreakerError("MT5 circuit breaker is open")

        try:
            # Try to get fresh data
            with self._cache_lock:
                data = self._market_data_cache.get(symbol)

            if data:
                self.record_success()
                # Cache for fallback
                self.degradation.cache_value(f'symbol_{symbol}', data)
                return data

            # Reload from file
            if self.data_file.exists():
                self._load_data_from_file()
                with self._cache_lock:
                    data = self._market_data_cache.get(symbol)
                if data:
                    self.record_success()
                    return data

            return None

        except Exception as e:
            self.record_failure(str(e))
            # Try fallback
            cached = self.degradation.get_cached(f'symbol_{symbol}')
            if cached:
                return cached
            raise

    def get_all_symbols(self) -> Dict[str, Dict[str, Any]]:
        """Get all available symbols with resilience"""
        if not self.circuit_breaker.allow_request():
            cached = self.degradation.get_cached('symbols')
            if cached:
                return cached
            raise CircuitBreakerError("MT5 circuit breaker is open")

        try:
            # Reload from file for freshness
            if self.data_file.exists():
                self._load_data_from_file()

            with self._cache_lock:
                symbols = dict(self._market_data_cache)

            self.record_success()
            return symbols

        except Exception as e:
            self.record_failure(str(e))
            cached = self.degradation.get_cached('symbols')
            if cached:
                return cached
            raise

    def get_account_info(self) -> Dict[str, Any]:
        """Get account information with resilience"""
        if not self.circuit_breaker.allow_request():
            cached = self.degradation.get_cached('account')
            if cached:
                return cached
            raise CircuitBreakerError("MT5 circuit breaker is open")

        try:
            if self.data_file.exists():
                self._load_data_from_file()

            self.record_success()
            return self._account_info

        except Exception as e:
            self.record_failure(str(e))
            cached = self.degradation.get_cached('account')
            if cached:
                return cached
            raise

    def get_connection_status(self) -> Dict[str, Any]:
        """Get detailed connection status"""
        health = self.get_health_report()

        return {
            **health,
            'server': self.config.server,
            'is_initialized': self._is_initialized,
            'symbols_count': self._symbols_count,
            'last_tick_time': (
                self._last_tick_time.isoformat()
                if self._last_tick_time else None
            ),
            'data_file_exists': self.data_file.exists(),
            'data_file_age_seconds': (
                time.time() - self.data_file.stat().st_mtime
                if self.data_file.exists() else None
            ),
            'fallback_available': self.fallback_file.exists(),
            'zmq_connected': self._zmq_socket is not None
        }

    async def refresh_data(self) -> bool:
        """Force refresh market data"""
        try:
            if self.data_file.exists():
                self._load_data_from_file()
                self.record_success()
                return True

            return await self._health_check()

        except Exception as e:
            self.record_failure(str(e))
            return False


# Singleton instance
_mt5_connection: Optional[MT5ResilientConnection] = None


def get_mt5_connection(config: Optional[MT5Config] = None) -> MT5ResilientConnection:
    """Get singleton MT5 resilient connection"""
    global _mt5_connection
    if _mt5_connection is None:
        _mt5_connection = MT5ResilientConnection(config)
    return _mt5_connection


async def initialize_mt5_connection(config: Optional[MT5Config] = None) -> MT5ResilientConnection:
    """Initialize MT5 connection with automatic reconnection"""
    connection = get_mt5_connection(config)
    await connection.connect()
    connection.start_auto_reconnect()
    return connection
