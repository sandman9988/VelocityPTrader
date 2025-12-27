#!/usr/bin/env python3
"""
ENTERPRISE LOGGING SYSTEM
PostgreSQL-based comprehensive logging with atomic operations
Replaces SQLite-based logging with enterprise-grade data persistence
"""

import asyncio
import json
import time
from datetime import datetime, timezone
from decimal import Decimal
from typing import Optional, Dict, Any, List, Union
from uuid import UUID
from dataclasses import dataclass, asdict
from enum import Enum

import structlog
from structlog.processors import JSONRenderer

from ..database.connection import get_database_manager, DatabaseManager
from ..database.operations import AtomicDataOperations
from ..database.models import TradingMode, AgentType, ActionType, MarketRegime

logger = structlog.get_logger(__name__)

class LogLevel(str, Enum):
    """Log levels for structured logging"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

@dataclass
class TradeLog:
    """Trade log entry for PostgreSQL storage"""
    timestamp: float
    agent_id: str
    instrument: str
    timeframe: str
    action: str
    entry_price: Optional[float]
    exit_price: Optional[float]
    position_size: float
    pnl: Optional[float]
    mae: Optional[float]  # Maximum Adverse Excursion
    mfe: Optional[float]  # Maximum Favorable Excursion
    trade_duration: Optional[float]
    spread_cost: float
    slippage: float
    market_conditions: Dict[str, Any]
    confidence_score: float
    risk_reward_ratio: Optional[float]
    win_loss: Optional[str]
    trade_id: str
    virtual_trade: bool  # True for shadow trading

@dataclass
class PerformanceLog:
    """Performance metrics log entry"""
    timestamp: float
    agent_id: str
    instrument: str
    timeframe: str
    win_rate: float
    profit_factor: float
    sharpe_ratio: float
    max_drawdown: float
    total_trades: int
    avg_trade_duration: float
    hit_ratio: float
    avg_win: float
    avg_loss: float
    consecutive_wins: int
    consecutive_losses: int
    volatility_adjusted_return: float
    kelly_criterion: float
    calmar_ratio: float

@dataclass
class ErrorLog:
    """Error log entry"""
    timestamp: float
    component: str
    error_message: str
    severity: str
    error_type: str
    stack_trace: Optional[str]
    context: Dict[str, Any]
    resolved: bool

class EnterpriseLogger:
    """Enterprise PostgreSQL-based logging system"""
    
    def __init__(self, session_id: Optional[UUID] = None, db_manager: Optional[DatabaseManager] = None):
        self.session_id = session_id
        self.db_manager = db_manager or get_database_manager()
        self.atomic_ops = AtomicDataOperations(self.db_manager)
        
        # Configure structured logging
        self._configure_structured_logging()
        
        # Performance counters
        self.logs_written = 0
        self.errors_logged = 0
        self.start_time = time.time()
        
        logger.info("Enterprise logging system initialized", 
                   session_id=str(session_id) if session_id else None)
    
    def _configure_structured_logging(self):
        """Configure structlog for enterprise logging"""
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                JSONRenderer()
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )
    
    def set_session_id(self, session_id: UUID):
        """Set the active trading session ID"""
        self.session_id = session_id
        logger.info("Session ID updated", session_id=str(session_id))
    
    def log_trade(self, trade_log: TradeLog):
        """Log trade to PostgreSQL atomically"""
        if not self.session_id:
            raise ValueError("Session ID must be set before logging trades")
        
        try:
            # Convert to database format
            trading_mode = TradingMode.VIRTUAL if trade_log.virtual_trade else TradingMode.LIVE
            agent_type = AgentType(trade_log.agent_id)
            action_type = ActionType(trade_log.action)
            
            # Log trade opening or update
            if trade_log.exit_price is None:
                # Opening trade
                trade = self.atomic_ops.create_trade(
                    session_id=self.session_id,
                    agent_type=agent_type,
                    trading_mode=trading_mode,
                    symbol=trade_log.instrument,
                    timeframe=trade_log.timeframe,
                    action=action_type,
                    entry_price=Decimal(str(trade_log.entry_price)),
                    position_size=Decimal(str(trade_log.position_size)),
                    confidence_score=trade_log.confidence_score,
                    market_conditions=trade_log.market_conditions,
                    direction="BUY" if "BUY" in trade_log.action else "SELL" if "SELL" in trade_log.action else None
                )
                
                logger.info("Trade logged", 
                           trade_id=trade.trade_id,
                           action=trade_log.action,
                           symbol=trade_log.instrument,
                           mode="VIRTUAL" if trade_log.virtual_trade else "LIVE")
            else:
                # Closing trade
                trade = self.atomic_ops.close_trade(
                    trade_id=trade_log.trade_id,
                    exit_price=Decimal(str(trade_log.exit_price)),
                    exit_reason="MANUAL_CLOSE",
                    spread_cost=trade_log.spread_cost,
                    slippage=trade_log.slippage
                )
                
                logger.info("Trade closed", 
                           trade_id=trade_log.trade_id,
                           pnl=float(trade.realized_pnl),
                           duration=trade.trade_duration_seconds)
            
            self.logs_written += 1
            
        except Exception as e:
            logger.error("Failed to log trade", error=str(e), trade_id=trade_log.trade_id)
            self.errors_logged += 1
            raise
    
    def log_performance(self, perf_log: PerformanceLog):
        """Log performance metrics to PostgreSQL atomically"""
        if not self.session_id:
            raise ValueError("Session ID must be set before logging performance")
        
        try:
            agent_type = AgentType(perf_log.agent_id)
            
            performance_data = {
                'total_trades': perf_log.total_trades,
                'winning_trades': int(perf_log.total_trades * perf_log.win_rate),
                'total_pnl': 0.0,  # Would be calculated from trades
                'win_rate': perf_log.win_rate,
                'profit_factor': perf_log.profit_factor,
                'sharpe_ratio': perf_log.sharpe_ratio,
                'max_drawdown': perf_log.max_drawdown,
                'calmar_ratio': perf_log.calmar_ratio
            }
            
            self.atomic_ops.update_agent_performance(
                session_id=self.session_id,
                agent_type=agent_type,
                symbol=perf_log.instrument,
                timeframe=perf_log.timeframe,
                performance_data=performance_data
            )
            
            logger.info("Performance logged",
                       agent=perf_log.agent_id,
                       symbol=perf_log.instrument,
                       win_rate=perf_log.win_rate,
                       total_trades=perf_log.total_trades)
            
            self.logs_written += 1
            
        except Exception as e:
            logger.error("Failed to log performance", error=str(e))
            self.errors_logged += 1
            raise
    
    def log_error(self, component: str, error_message: str, severity: str, 
                  error_type: Optional[str] = None, stack_trace: Optional[str] = None,
                  context: Optional[Dict[str, Any]] = None):
        """Log error with enterprise tracking"""
        try:
            error_entry = ErrorLog(
                timestamp=time.time(),
                component=component,
                error_message=error_message,
                severity=severity,
                error_type=error_type or "UNKNOWN",
                stack_trace=stack_trace,
                context=context or {},
                resolved=False
            )
            
            # Log to structured logger
            log_level = LogLevel(severity)
            if log_level == LogLevel.CRITICAL:
                logger.critical("System error", **asdict(error_entry))
            elif log_level == LogLevel.ERROR:
                logger.error("Application error", **asdict(error_entry))
            elif log_level == LogLevel.WARNING:
                logger.warning("Warning condition", **asdict(error_entry))
            else:
                logger.info("Error logged", **asdict(error_entry))
            
            # Persist error to database for tracking and debugging
            try:
                self.atomic_ops.log_error(
                    session_id=self.session_id,
                    component=component,
                    error_message=error_message,
                    severity=severity,
                    error_type=error_type,
                    stack_trace=stack_trace,
                    context=context
                )
            except Exception as db_error:
                # Don't fail if database logging fails
                logger.warning("Could not persist error to database",
                              db_error=str(db_error))

            self.errors_logged += 1
            
        except Exception as e:
            # Fallback logging to prevent logging system failure
            print(f"❌ CRITICAL: Logging system failure: {e}")
            print(f"   Original error: {error_message}")
    
    def log_system_health(self, health_data: Dict[str, Any]):
        """Log system health metrics"""
        if not self.session_id:
            logger.warning("Cannot log system health without active session")
            return
        
        try:
            self.atomic_ops.log_system_health(
                session_id=self.session_id,
                health_data=health_data
            )
            
            logger.debug("System health logged",
                        cpu=health_data.get('cpu_usage', 0),
                        memory=health_data.get('memory_usage', 0),
                        status=health_data.get('overall_status', 'UNKNOWN'))
            
            self.logs_written += 1
            
        except Exception as e:
            logger.error("Failed to log system health", error=str(e))
            self.errors_logged += 1
    
    def log_market_data_batch(self, market_data_batch: List[Dict[str, Any]]):
        """Log market data batch atomically"""
        if not self.session_id:
            raise ValueError("Session ID must be set before logging market data")
        
        try:
            # Validate all data is real (no fake/mock data)
            for data in market_data_batch:
                if data.get('is_fake', False) or data.get('simulated', False):
                    raise ValueError("❌ FAKE DATA REJECTED - Only real MT5 data allowed!")
            
            self.atomic_ops.insert_market_data_batch(
                session_id=self.session_id,
                market_data_list=market_data_batch
            )
            
            logger.info("Market data batch logged",
                       count=len(market_data_batch),
                       symbols=list(set(d['symbol'] for d in market_data_batch[:5])))
            
            self.logs_written += len(market_data_batch)
            
        except Exception as e:
            logger.error("Failed to log market data batch", error=str(e))
            self.errors_logged += 1
            raise
    
    def get_error_statistics(self, hours: int = 24) -> Dict[str, Any]:
        """Get detailed error statistics from database"""
        try:
            return self.atomic_ops.get_error_statistics(
                session_id=self.session_id,
                hours=hours
            )
        except Exception as e:
            logger.warning("Could not retrieve error statistics", error=str(e))
            return {'error': str(e)}

    def get_unresolved_errors(self, limit: int = 50) -> list:
        """Get unresolved errors for debugging"""
        try:
            return self.atomic_ops.get_unresolved_errors(
                session_id=self.session_id,
                limit=limit
            )
        except Exception as e:
            logger.warning("Could not retrieve unresolved errors", error=str(e))
            return []

    def get_statistics(self) -> Dict[str, Any]:
        """Get logging system statistics"""
        uptime = time.time() - self.start_time

        # Get error statistics from database
        error_stats = {}
        try:
            error_stats = self.atomic_ops.get_error_statistics(
                session_id=self.session_id,
                hours=24
            )
        except Exception:
            pass

        return {
            'logs_written': self.logs_written,
            'errors_logged': self.errors_logged,
            'uptime_seconds': uptime,
            'logs_per_second': self.logs_written / uptime if uptime > 0 else 0,
            'error_rate_percent': (self.errors_logged / max(self.logs_written, 1)) * 100,
            'session_id': str(self.session_id) if self.session_id else None,
            'database_health': self.db_manager.health_check() if self.db_manager else None,
            'error_statistics': error_stats
        }
    
    def close(self):
        """Close logging system and cleanup"""
        try:
            stats = self.get_statistics()
            logger.info("Enterprise logging system closing", **stats)
            
        except Exception as e:
            logger.error("Error during logging system shutdown", error=str(e))

# Singleton logger instance
_enterprise_logger: Optional[EnterpriseLogger] = None

def get_enterprise_logger(session_id: Optional[UUID] = None) -> EnterpriseLogger:
    """Get singleton enterprise logger"""
    global _enterprise_logger
    if _enterprise_logger is None:
        _enterprise_logger = EnterpriseLogger(session_id=session_id)
    elif session_id and _enterprise_logger.session_id != session_id:
        _enterprise_logger.set_session_id(session_id)
    return _enterprise_logger

def initialize_enterprise_logging(session_id: UUID) -> EnterpriseLogger:
    """Initialize enterprise logging system"""
    global _enterprise_logger
    _enterprise_logger = EnterpriseLogger(session_id=session_id)
    return _enterprise_logger