#!/usr/bin/env python3
"""
ENTERPRISE DATABASE CONNECTION MANAGER
PostgreSQL connection pooling with atomic operations, circuit breakers, and resilience
"""

import asyncio
import os
import threading
from contextlib import contextmanager, asynccontextmanager
from typing import Optional, Dict, Any, List, AsyncGenerator
from urllib.parse import quote_plus
from datetime import datetime, timezone
import time

from sqlalchemy import create_engine, text, event
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool
from sqlalchemy.exc import SQLAlchemyError, IntegrityError, OperationalError
from sqlalchemy.engine import Engine
import asyncpg
from asyncpg import Connection, Pool
import structlog

from .models import Base, TradingSession, MarketData, Trade, AgentPerformance, SystemHealth
from ..utils.resilience import (
    ConnectionManagerBase, ConnectionState, CircuitBreaker, CircuitBreakerConfig,
    RetryConfig, with_retry, with_fallback, with_circuit_breaker,
    GracefulDegradation, WriteBuffer, register_connection_manager,
    calculate_backoff_delay, CircuitBreakerError, log_resilience_event
)

logger = structlog.get_logger(__name__)

class DatabaseConfig:
    """Database configuration with enterprise defaults"""
    
    def __init__(self):
        # PostgreSQL connection settings
        self.host = os.getenv('POSTGRES_HOST', 'localhost')
        self.port = int(os.getenv('POSTGRES_PORT', 5432))
        self.database = os.getenv('POSTGRES_DB', 'velocity_trader')
        self.username = os.getenv('POSTGRES_USER', 'velocity_trader')
        self.password = os.getenv('POSTGRES_PASSWORD', '')
        
        # Connection pool settings
        self.pool_size = int(os.getenv('DB_POOL_SIZE', 20))
        self.max_overflow = int(os.getenv('DB_MAX_OVERFLOW', 50))
        self.pool_timeout = int(os.getenv('DB_POOL_TIMEOUT', 30))
        self.pool_recycle = int(os.getenv('DB_POOL_RECYCLE', 3600))  # 1 hour
        
        # Connection retry settings
        self.max_retries = int(os.getenv('DB_MAX_RETRIES', 5))
        self.retry_delay = float(os.getenv('DB_RETRY_DELAY', 1.0))
        
        # Security settings
        self.ssl_mode = os.getenv('DB_SSL_MODE', 'prefer')
        self.application_name = 'VelocityTrader'
        
        # Performance settings
        self.statement_timeout = int(os.getenv('DB_STATEMENT_TIMEOUT', 30000))  # 30 seconds
        self.idle_in_transaction_timeout = int(os.getenv('DB_IDLE_TIMEOUT', 60000))  # 1 minute
    
    @property
    def database_url(self) -> str:
        """Generate PostgreSQL URL with proper escaping"""
        password = quote_plus(self.password) if self.password else ''
        password_part = f":{password}" if password else ""
        
        return (
            f"postgresql://{self.username}{password_part}@{self.host}:{self.port}/{self.database}"
            f"?application_name={self.application_name}&sslmode={self.ssl_mode}"
        )
    
    @property
    def async_database_url(self) -> str:
        """Generate async PostgreSQL URL"""
        password = quote_plus(self.password) if self.password else ''
        password_part = f":{password}" if password else ""
        
        return (
            f"postgresql+asyncpg://{self.username}{password_part}@{self.host}:{self.port}/{self.database}"
            f"?application_name={self.application_name}_async&sslmode={self.ssl_mode}"
        )

class DatabaseManager:
    """Enterprise database manager with connection pooling, circuit breakers, and resilience"""

    def __init__(self, config: Optional[DatabaseConfig] = None):
        self.config = config or DatabaseConfig()
        self.engine: Optional[Engine] = None
        self.session_factory: Optional[sessionmaker] = None
        self.async_pool: Optional[Pool] = None
        self._health_check_interval = 60  # seconds
        self._last_health_check = 0

        # Resilience components
        self._state = ConnectionState.DISCONNECTED
        self._state_lock = threading.RLock()

        # Circuit breaker for database operations
        self.circuit_breaker = CircuitBreaker(
            name="database",
            config=CircuitBreakerConfig(
                failure_threshold=5,
                success_threshold=3,
                timeout_seconds=30.0
            )
        )

        # Retry configuration
        self.retry_config = RetryConfig(
            max_retries=self.config.max_retries,
            initial_delay_ms=int(self.config.retry_delay * 1000),
            max_delay_ms=30000,
            exponential_base=2.0,
            jitter=True,
            retryable_exceptions=(OperationalError, SQLAlchemyError)
        )

        # Graceful degradation handler
        self.degradation = GracefulDegradation(
            name="database",
            cache_ttl_seconds=60.0
        )

        # Write buffer for when database is slow/unavailable
        self.write_buffer = WriteBuffer(
            name="database_writes",
            max_buffer_size=1000,
            flush_interval_seconds=5.0
        )

        # Health tracking
        self._consecutive_failures = 0
        self._consecutive_successes = 0
        self._total_operations = 0
        self._total_failures = 0
        self._last_successful_operation: Optional[datetime] = None
        self._last_failed_operation: Optional[datetime] = None
        self._last_error: Optional[str] = None

        logger.info("Database manager initialized with resilience",
                   host=self.config.host,
                   database=self.config.database,
                   circuit_breaker="enabled",
                   retry_max=self.config.max_retries)
    
    def initialize_sync_engine(self) -> Engine:
        """Initialize synchronous database engine with connection pooling"""
        if self.engine is not None:
            return self.engine
        
        try:
            # Create engine with enterprise connection pool
            self.engine = create_engine(
                self.config.database_url,
                poolclass=QueuePool,
                pool_size=self.config.pool_size,
                max_overflow=self.config.max_overflow,
                pool_timeout=self.config.pool_timeout,
                pool_recycle=self.config.pool_recycle,
                pool_pre_ping=True,  # Validate connections
                echo=False,  # Set to True for SQL debugging
                echo_pool=False,
                connect_args={
                    "connect_timeout": 10,
                    "command_timeout": self.config.statement_timeout // 1000,
                    "server_settings": {
                        "application_name": self.config.application_name,
                        "statement_timeout": str(self.config.statement_timeout),
                        "idle_in_transaction_session_timeout": str(self.config.idle_in_transaction_timeout),
                    }
                }
            )
            
            # Create session factory
            self.session_factory = sessionmaker(
                bind=self.engine,
                autocommit=False,
                autoflush=False,  # Manual flush control for atomic operations
                expire_on_commit=False
            )
            
            # Add connection event listeners
            self._setup_connection_events()
            
            # Verify connection
            self._verify_connection()
            
            logger.info("Synchronous database engine initialized successfully")
            return self.engine
            
        except Exception as e:
            logger.error("Failed to initialize database engine", error=str(e))
            raise DatabaseConnectionError(f"Database initialization failed: {e}") from e
    
    async def initialize_async_pool(self) -> Pool:
        """Initialize async connection pool"""
        if self.async_pool is not None:
            return self.async_pool
        
        try:
            self.async_pool = await asyncpg.create_pool(
                host=self.config.host,
                port=self.config.port,
                database=self.config.database,
                user=self.config.username,
                password=self.config.password,
                min_size=5,
                max_size=self.config.pool_size,
                command_timeout=self.config.statement_timeout / 1000,
                server_settings={
                    "application_name": f"{self.config.application_name}_async",
                    "statement_timeout": str(self.config.statement_timeout),
                }
            )
            
            logger.info("Async database pool initialized successfully")
            return self.async_pool
            
        except Exception as e:
            logger.error("Failed to initialize async database pool", error=str(e))
            raise DatabaseConnectionError(f"Async database initialization failed: {e}") from e
    
    def _setup_connection_events(self):
        """Setup connection event listeners for monitoring"""
        
        @event.listens_for(self.engine, "connect")
        def set_postgresql_settings(dbapi_connection, connection_record):
            """Configure PostgreSQL session settings"""
            with dbapi_connection.cursor() as cursor:
                # Set timezone
                cursor.execute("SET timezone = 'UTC'")
                
                # Enable constraint checking
                cursor.execute("SET check_function_bodies = true")
                
                # Performance settings
                cursor.execute("SET synchronous_commit = 'on'")  # Ensure durability
                cursor.execute("SET wal_sync_method = 'fsync'")
                
                logger.debug("PostgreSQL connection configured")
        
        @event.listens_for(self.engine, "checkout")
        def checkout_handler(dbapi_connection, connection_record, connection_proxy):
            """Handle connection checkout"""
            logger.debug("Database connection checked out")
        
        @event.listens_for(self.engine, "checkin")
        def checkin_handler(dbapi_connection, connection_record):
            """Handle connection checkin"""
            logger.debug("Database connection checked in")
    
    def _verify_connection(self):
        """Verify database connection and schema"""
        try:
            with self.get_session() as session:
                result = session.execute(text("SELECT 1"))
                assert result.fetchone()[0] == 1
                
                # Check if tables exist
                result = session.execute(text("""
                    SELECT COUNT(*) FROM information_schema.tables 
                    WHERE table_schema = 'public' AND table_name IN 
                    ('trading_sessions', 'market_data', 'trades', 'agent_performance', 'system_health')
                """))
                
                table_count = result.fetchone()[0]
                logger.info("Database connection verified", tables_found=table_count)
                
        except Exception as e:
            logger.error("Database connection verification failed", error=str(e))
            raise DatabaseConnectionError(f"Connection verification failed: {e}") from e
    
    def _record_success(self):
        """Record successful operation for health tracking"""
        with self._state_lock:
            self._consecutive_successes += 1
            self._consecutive_failures = 0
            self._total_operations += 1
            self._last_successful_operation = datetime.now(timezone.utc)

            if self._state == ConnectionState.DEGRADED:
                if self._consecutive_successes >= 3:
                    self._state = ConnectionState.CONNECTED
                    self.degradation.exit_degraded_mode()
                    log_resilience_event("recovery", "database",
                                        message="Database connection recovered")

        self.circuit_breaker.record_success()

    def _record_failure(self, error: str):
        """Record failed operation for health tracking"""
        with self._state_lock:
            self._consecutive_failures += 1
            self._consecutive_successes = 0
            self._total_operations += 1
            self._total_failures += 1
            self._last_failed_operation = datetime.now(timezone.utc)
            self._last_error = error

            if self._state == ConnectionState.CONNECTED:
                if self._consecutive_failures >= 3:
                    self._state = ConnectionState.DEGRADED
                    self.degradation.enter_degraded_mode(error)
                    log_resilience_event("degradation", "database",
                                        message="Database entering degraded mode",
                                        error=error)

        self.circuit_breaker.record_failure(error)

    @contextmanager
    def get_session(self) -> Session:
        """Get database session with circuit breaker and automatic cleanup"""
        # Check circuit breaker
        if not self.circuit_breaker.allow_request():
            raise CircuitBreakerError("Database circuit breaker is open")

        if self.engine is None:
            self.initialize_sync_engine()

        session = self.session_factory()
        start_time = time.time()
        try:
            yield session
            session.commit()
            self._record_success()
        except Exception as e:
            session.rollback()
            self._record_failure(str(e))
            logger.error("Database session error", error=str(e))
            raise
        finally:
            session.close()
            latency_ms = (time.time() - start_time) * 1000
            if latency_ms > 1000:
                logger.warning("Slow database operation", latency_ms=latency_ms)

    @contextmanager
    def atomic_transaction(self) -> Session:
        """Atomic transaction with circuit breaker and automatic rollback"""
        # Check circuit breaker
        if not self.circuit_breaker.allow_request():
            raise CircuitBreakerError("Database circuit breaker is open")

        if self.engine is None:
            self.initialize_sync_engine()

        session = self.session_factory()
        start_time = time.time()
        try:
            session.begin()
            yield session
            session.commit()
            self._record_success()
            logger.debug("Atomic transaction committed successfully")
        except Exception as e:
            session.rollback()
            self._record_failure(str(e))
            logger.error("Atomic transaction rolled back", error=str(e))
            raise
        finally:
            session.close()
            latency_ms = (time.time() - start_time) * 1000
            if latency_ms > 1000:
                logger.warning("Slow atomic transaction", latency_ms=latency_ms)
    
    @asynccontextmanager
    async def async_connection(self) -> AsyncGenerator[Connection, None]:
        """Get async database connection"""
        if self.async_pool is None:
            await self.initialize_async_pool()
        
        async with self.async_pool.acquire() as connection:
            try:
                yield connection
            except Exception as e:
                logger.error("Async connection error", error=str(e))
                raise
    
    @asynccontextmanager
    async def async_transaction(self) -> AsyncGenerator[Connection, None]:
        """Async atomic transaction"""
        async with self.async_connection() as connection:
            async with connection.transaction():
                try:
                    yield connection
                    logger.debug("Async atomic transaction committed")
                except Exception as e:
                    logger.error("Async atomic transaction rolled back", error=str(e))
                    raise
    
    def get_resilience_status(self) -> Dict[str, Any]:
        """Get resilience and health tracking status"""
        with self._state_lock:
            error_rate = (
                self._total_failures / self._total_operations * 100
                if self._total_operations > 0 else 0
            )

            return {
                "connection_state": self._state.value,
                "circuit_breaker_state": self.circuit_breaker.state.value,
                "is_degraded": self.degradation.is_degraded,
                "consecutive_failures": self._consecutive_failures,
                "consecutive_successes": self._consecutive_successes,
                "total_operations": self._total_operations,
                "total_failures": self._total_failures,
                "error_rate_percent": round(error_rate, 2),
                "last_successful_operation": (
                    self._last_successful_operation.isoformat()
                    if self._last_successful_operation else None
                ),
                "last_failed_operation": (
                    self._last_failed_operation.isoformat()
                    if self._last_failed_operation else None
                ),
                "last_error": self._last_error,
                "write_buffer_size": self.write_buffer.size
            }

    def health_check(self) -> Dict[str, Any]:
        """Comprehensive database health check with resilience status"""
        current_time = time.time()

        # Rate limit health checks
        if current_time - self._last_health_check < self._health_check_interval:
            return {"status": "cached", "last_check": self._last_health_check}

        health_status = {
            "status": "unknown",
            "connection_pool": {},
            "database": {},
            "performance": {},
            "resilience": self.get_resilience_status(),
            "timestamp": current_time
        }
        
        try:
            # Check connection pool
            if self.engine:
                pool = self.engine.pool
                health_status["connection_pool"] = {
                    "size": pool.size(),
                    "checked_in": pool.checkedin(),
                    "checked_out": pool.checkedout(),
                    "overflow": pool.overflow(),
                    "invalid": pool.invalid()
                }
            
            # Check database connectivity
            start_time = time.time()
            with self.get_session() as session:
                # Test query
                result = session.execute(text("SELECT version(), now(), current_database()"))
                db_info = result.fetchone()
                
                # Check table status
                result = session.execute(text("""
                    SELECT 
                        schemaname,
                        tablename,
                        n_tup_ins,
                        n_tup_upd,
                        n_tup_del
                    FROM pg_stat_user_tables
                    WHERE schemaname = 'public'
                """))
                tables = result.fetchall()
                
                query_time = time.time() - start_time
                
                health_status["database"] = {
                    "version": db_info[0] if db_info else None,
                    "server_time": db_info[1].isoformat() if db_info else None,
                    "database_name": db_info[2] if db_info else None,
                    "table_count": len(tables),
                    "tables": [{"schema": t[0], "name": t[1], "inserts": t[2], "updates": t[3], "deletes": t[4]} for t in tables]
                }
                
                health_status["performance"] = {
                    "query_time_ms": round(query_time * 1000, 2),
                    "status": "fast" if query_time < 0.1 else "slow" if query_time < 1.0 else "critical"
                }
                
                health_status["status"] = "healthy"
            
            self._last_health_check = current_time
            
        except Exception as e:
            health_status["status"] = "unhealthy"
            health_status["error"] = str(e)
            logger.error("Database health check failed", error=str(e))
        
        return health_status
    
    def create_tables(self):
        """Create all database tables"""
        if self.engine is None:
            self.initialize_sync_engine()
        
        try:
            Base.metadata.create_all(self.engine)
            logger.info("Database tables created successfully")
        except Exception as e:
            logger.error("Failed to create database tables", error=str(e))
            raise DatabaseSchemaError(f"Table creation failed: {e}") from e
    
    def drop_tables(self):
        """Drop all database tables (USE WITH CAUTION)"""
        if self.engine is None:
            self.initialize_sync_engine()
        
        try:
            Base.metadata.drop_all(self.engine)
            logger.warning("All database tables dropped")
        except Exception as e:
            logger.error("Failed to drop database tables", error=str(e))
            raise DatabaseSchemaError(f"Table dropping failed: {e}") from e
    
    async def close(self):
        """Close all database connections"""
        try:
            if self.async_pool:
                await self.async_pool.close()
                logger.info("Async database pool closed")
            
            if self.engine:
                self.engine.dispose()
                logger.info("Sync database engine disposed")
                
        except Exception as e:
            logger.error("Error closing database connections", error=str(e))

# Custom exceptions
class DatabaseError(Exception):
    """Base database error"""
    pass

class DatabaseConnectionError(DatabaseError):
    """Database connection error"""
    pass

class DatabaseSchemaError(DatabaseError):
    """Database schema error"""
    pass

class AtomicOperationError(DatabaseError):
    """Atomic operation error"""
    pass

# Singleton database manager
_db_manager: Optional[DatabaseManager] = None

def get_database_manager() -> DatabaseManager:
    """Get singleton database manager"""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
    return _db_manager

def initialize_database(config: Optional[DatabaseConfig] = None) -> DatabaseManager:
    """Initialize database with configuration"""
    global _db_manager
    _db_manager = DatabaseManager(config)
    _db_manager.initialize_sync_engine()
    return _db_manager