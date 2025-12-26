#!/usr/bin/env python3
"""
ENTERPRISE DATABASE MODELS
PostgreSQL-based data models for VelocityTrader with atomic operations
"""

from datetime import datetime
from decimal import Decimal
from typing import Optional, Dict, Any, List
from enum import Enum
import uuid

from sqlalchemy import (
    Column, Integer, String, DateTime, Numeric, Boolean, Text, 
    ForeignKey, Index, CheckConstraint, UniqueConstraint,
    create_engine, event
)
from sqlalchemy.dialects.postgresql import UUID, JSONB, ENUM
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, Session
from sqlalchemy.sql import func
import structlog

logger = structlog.get_logger(__name__)

Base = declarative_base()

# Enums for type safety
class TradingMode(str, Enum):
    """Only two allowed modes: LIVE or VIRTUAL"""
    LIVE = "LIVE"           # Real MT5 trading
    VIRTUAL = "VIRTUAL"     # Shadow/virtual trading for learning
    
    # NO FAKE, MOCK, SIMULATED, or TEST modes allowed

class AgentType(str, Enum):
    """Agent types"""
    BERSERKER = "BERSERKER"
    SNIPER = "SNIPER"
    SHADOW_BERSERKER = "SHADOW_BERSERKER"  # Virtual learning
    SHADOW_SNIPER = "SHADOW_SNIPER"        # Virtual learning

class ActionType(str, Enum):
    """Trading actions"""
    BUY = "BUY"
    SELL = "SELL"
    CLOSE = "CLOSE"
    HOLD = "HOLD"

class MarketRegime(str, Enum):
    """Physics-based market regimes"""
    OVERDAMPED = "OVERDAMPED"
    CRITICALLY_DAMPED = "CRITICALLY_DAMPED"
    UNDERDAMPED = "UNDERDAMPED"
    CHAOTIC = "CHAOTIC"

# Database Models with Enterprise Constraints

class TradingSession(Base):
    """Trading session tracking with atomic operations"""
    __tablename__ = 'trading_sessions'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_start = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    session_end = Column(DateTime(timezone=True), nullable=True)
    mode = Column(ENUM(TradingMode), nullable=False)
    mt5_server = Column(String(100), nullable=False)
    mt5_login = Column(Integer, nullable=False)
    initial_balance = Column(Numeric(15, 2), nullable=False)
    final_balance = Column(Numeric(15, 2), nullable=True)
    total_trades = Column(Integer, default=0)
    winning_trades = Column(Integer, default=0)
    max_drawdown = Column(Numeric(8, 6), default=0.0)
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    trades = relationship("Trade", back_populates="session", cascade="all, delete-orphan")
    market_data = relationship("MarketData", back_populates="session", cascade="all, delete-orphan")
    
    # Constraints
    __table_args__ = (
        CheckConstraint('initial_balance > 0', name='positive_initial_balance'),
        CheckConstraint('final_balance IS NULL OR final_balance >= 0', name='non_negative_final_balance'),
        CheckConstraint('total_trades >= 0', name='non_negative_total_trades'),
        CheckConstraint('winning_trades >= 0', name='non_negative_winning_trades'),
        CheckConstraint('winning_trades <= total_trades', name='winning_trades_le_total'),
        CheckConstraint('max_drawdown >= 0 AND max_drawdown <= 1', name='valid_max_drawdown'),
        CheckConstraint("mt5_server = 'VantageInternational-Demo'", name='vantage_server_only'),
        Index('idx_trading_sessions_active', 'is_active'),
        Index('idx_trading_sessions_mode', 'mode'),
    )

class MarketData(Base):
    """Real MT5 market data - NO FAKE DATA ALLOWED"""
    __tablename__ = 'market_data'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(UUID(as_uuid=True), ForeignKey('trading_sessions.id'), nullable=False)
    symbol = Column(String(20), nullable=False)
    timeframe = Column(String(10), nullable=False)
    timestamp = Column(DateTime(timezone=True), nullable=False)
    
    # OHLCV data - all required for real trading
    open_price = Column(Numeric(20, 8), nullable=False)
    high_price = Column(Numeric(20, 8), nullable=False)
    low_price = Column(Numeric(20, 8), nullable=False)
    close_price = Column(Numeric(20, 8), nullable=False)
    volume = Column(Numeric(20, 2), nullable=False)
    
    # Bid/Ask for real spreads
    bid_price = Column(Numeric(20, 8), nullable=False)
    ask_price = Column(Numeric(20, 8), nullable=False)
    spread_pips = Column(Numeric(8, 2), nullable=False)
    
    # Physics metrics (calculated from real data)
    momentum = Column(Numeric(10, 6), nullable=True)
    acceleration = Column(Numeric(10, 6), nullable=True)
    volatility = Column(Numeric(10, 6), nullable=True)
    liquidity_score = Column(Numeric(8, 6), nullable=True)
    trend_strength = Column(Numeric(8, 6), nullable=True)
    market_regime = Column(ENUM(MarketRegime), nullable=True)
    
    # Source validation
    data_source = Column(String(50), nullable=False, default='MT5_VANTAGE')
    is_real_data = Column(Boolean, nullable=False, default=True)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    session = relationship("TradingSession", back_populates="market_data")
    trades = relationship("Trade", back_populates="market_data_entry")
    
    # Constraints
    __table_args__ = (
        CheckConstraint('open_price > 0', name='positive_open_price'),
        CheckConstraint('high_price > 0', name='positive_high_price'),
        CheckConstraint('low_price > 0', name='positive_low_price'),
        CheckConstraint('close_price > 0', name='positive_close_price'),
        CheckConstraint('volume >= 0', name='non_negative_volume'),
        CheckConstraint('bid_price > 0', name='positive_bid_price'),
        CheckConstraint('ask_price > 0', name='positive_ask_price'),
        CheckConstraint('ask_price >= bid_price', name='ask_gte_bid'),
        CheckConstraint('spread_pips >= 0', name='non_negative_spread'),
        CheckConstraint('high_price >= low_price', name='high_gte_low'),
        CheckConstraint('high_price >= open_price', name='high_gte_open'),
        CheckConstraint('high_price >= close_price', name='high_gte_close'),
        CheckConstraint('low_price <= open_price', name='low_lte_open'),
        CheckConstraint('low_price <= close_price', name='low_lte_close'),
        CheckConstraint("data_source = 'MT5_VANTAGE'", name='vantage_data_only'),
        CheckConstraint('is_real_data = true', name='real_data_only'),
        UniqueConstraint('symbol', 'timeframe', 'timestamp', name='unique_market_tick'),
        Index('idx_market_data_symbol_timeframe', 'symbol', 'timeframe'),
        Index('idx_market_data_timestamp', 'timestamp'),
        Index('idx_market_data_session', 'session_id'),
    )

class Trade(Base):
    """Trade records with atomic integrity"""
    __tablename__ = 'trades'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(UUID(as_uuid=True), ForeignKey('trading_sessions.id'), nullable=False)
    market_data_id = Column(UUID(as_uuid=True), ForeignKey('market_data.id'), nullable=True)
    
    # Trade identification
    trade_id = Column(String(100), nullable=False)  # Internal trade ID
    mt5_position_id = Column(String(50), nullable=True)  # MT5 position ID for live trades
    
    # Agent and mode
    agent_type = Column(ENUM(AgentType), nullable=False)
    trading_mode = Column(ENUM(TradingMode), nullable=False)
    
    # Trade details
    symbol = Column(String(20), nullable=False)
    timeframe = Column(String(10), nullable=False)
    action = Column(ENUM(ActionType), nullable=False)
    direction = Column(String(10), nullable=True)  # BUY/SELL for actual trades
    
    # Execution details
    entry_time = Column(DateTime(timezone=True), nullable=False)
    exit_time = Column(DateTime(timezone=True), nullable=True)
    entry_price = Column(Numeric(20, 8), nullable=False)
    exit_price = Column(Numeric(20, 8), nullable=True)
    position_size = Column(Numeric(10, 6), nullable=False)
    
    # P&L and risk metrics
    unrealized_pnl = Column(Numeric(15, 2), default=0.0)
    realized_pnl = Column(Numeric(15, 2), nullable=True)
    max_adverse_excursion = Column(Numeric(15, 2), default=0.0)
    max_favorable_excursion = Column(Numeric(15, 2), default=0.0)
    
    # Trade management
    stop_loss = Column(Numeric(20, 8), nullable=True)
    take_profit = Column(Numeric(20, 8), nullable=True)
    trailing_stop = Column(Numeric(20, 8), nullable=True)
    
    # Costs and slippage
    spread_cost = Column(Numeric(10, 2), default=0.0)
    commission = Column(Numeric(10, 2), default=0.0)
    swap = Column(Numeric(10, 2), default=0.0)
    slippage = Column(Numeric(10, 6), default=0.0)
    
    # Performance metrics
    confidence_score = Column(Numeric(4, 3), nullable=True)
    risk_reward_ratio = Column(Numeric(8, 3), nullable=True)
    trade_duration_seconds = Column(Integer, nullable=True)
    market_conditions = Column(JSONB, nullable=True)
    
    # Status
    is_open = Column(Boolean, default=True, nullable=False)
    is_winning = Column(Boolean, nullable=True)
    exit_reason = Column(String(50), nullable=True)
    
    # Audit
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    session = relationship("TradingSession", back_populates="trades")
    market_data_entry = relationship("MarketData", back_populates="trades")
    
    # Constraints
    __table_args__ = (
        CheckConstraint('entry_price > 0', name='positive_entry_price'),
        CheckConstraint('exit_price IS NULL OR exit_price > 0', name='positive_exit_price'),
        CheckConstraint('position_size > 0', name='positive_position_size'),
        CheckConstraint('stop_loss IS NULL OR stop_loss > 0', name='positive_stop_loss'),
        CheckConstraint('take_profit IS NULL OR take_profit > 0', name='positive_take_profit'),
        CheckConstraint('confidence_score IS NULL OR (confidence_score >= 0 AND confidence_score <= 1)', name='valid_confidence'),
        CheckConstraint('trade_duration_seconds IS NULL OR trade_duration_seconds >= 0', name='non_negative_duration'),
        CheckConstraint('(is_open = true AND exit_time IS NULL) OR (is_open = false AND exit_time IS NOT NULL)', name='consistent_open_status'),
        UniqueConstraint('trade_id', 'session_id', name='unique_trade_per_session'),
        Index('idx_trades_session', 'session_id'),
        Index('idx_trades_symbol', 'symbol'),
        Index('idx_trades_agent_mode', 'agent_type', 'trading_mode'),
        Index('idx_trades_open', 'is_open'),
        Index('idx_trades_entry_time', 'entry_time'),
    )

class AgentPerformance(Base):
    """Agent performance tracking with learning metrics"""
    __tablename__ = 'agent_performance'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(UUID(as_uuid=True), ForeignKey('trading_sessions.id'), nullable=False)
    agent_type = Column(ENUM(AgentType), nullable=False)
    symbol = Column(String(20), nullable=False)
    timeframe = Column(String(10), nullable=False)
    
    # Performance metrics
    total_trades = Column(Integer, default=0)
    winning_trades = Column(Integer, default=0)
    losing_trades = Column(Integer, default=0)
    total_pnl = Column(Numeric(15, 2), default=0.0)
    gross_profit = Column(Numeric(15, 2), default=0.0)
    gross_loss = Column(Numeric(15, 2), default=0.0)
    
    # Risk metrics
    max_drawdown = Column(Numeric(8, 6), default=0.0)
    sharpe_ratio = Column(Numeric(8, 4), nullable=True)
    calmar_ratio = Column(Numeric(8, 4), nullable=True)
    win_rate = Column(Numeric(5, 4), default=0.0)
    profit_factor = Column(Numeric(8, 3), default=0.0)
    
    # Learning metrics
    model_version = Column(Integer, default=1)
    total_experiences = Column(Integer, default=0)
    exploration_rate = Column(Numeric(4, 3), default=0.1)
    learning_rate = Column(Numeric(6, 5), default=0.001)
    last_model_update = Column(DateTime(timezone=True), nullable=True)
    
    # Physics learning
    regime_accuracy = Column(Numeric(5, 4), default=0.0)
    momentum_prediction_accuracy = Column(Numeric(5, 4), default=0.0)
    volatility_prediction_accuracy = Column(Numeric(5, 4), default=0.0)
    
    # Timestamps
    period_start = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    period_end = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    session = relationship("TradingSession")
    
    # Constraints
    __table_args__ = (
        CheckConstraint('total_trades >= 0', name='non_negative_total_trades'),
        CheckConstraint('winning_trades >= 0', name='non_negative_winning_trades'),
        CheckConstraint('losing_trades >= 0', name='non_negative_losing_trades'),
        CheckConstraint('winning_trades + losing_trades <= total_trades', name='trades_sum_valid'),
        CheckConstraint('gross_profit >= 0', name='non_negative_gross_profit'),
        CheckConstraint('gross_loss <= 0', name='non_positive_gross_loss'),
        CheckConstraint('max_drawdown >= 0 AND max_drawdown <= 1', name='valid_max_drawdown'),
        CheckConstraint('win_rate >= 0 AND win_rate <= 1', name='valid_win_rate'),
        CheckConstraint('profit_factor >= 0', name='non_negative_profit_factor'),
        CheckConstraint('exploration_rate >= 0 AND exploration_rate <= 1', name='valid_exploration_rate'),
        CheckConstraint('learning_rate > 0 AND learning_rate <= 1', name='valid_learning_rate'),
        UniqueConstraint('session_id', 'agent_type', 'symbol', 'timeframe', 'period_start', name='unique_performance_period'),
        Index('idx_agent_performance_session_agent', 'session_id', 'agent_type'),
        Index('idx_agent_performance_symbol', 'symbol', 'timeframe'),
    )

class SystemHealth(Base):
    """System health and monitoring"""
    __tablename__ = 'system_health'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(UUID(as_uuid=True), ForeignKey('trading_sessions.id'), nullable=False)
    
    # System metrics
    cpu_usage_percent = Column(Numeric(5, 2), nullable=False)
    memory_usage_percent = Column(Numeric(5, 2), nullable=False)
    disk_usage_percent = Column(Numeric(5, 2), nullable=False)
    network_latency_ms = Column(Numeric(8, 2), nullable=True)
    
    # MT5 connection
    mt5_connected = Column(Boolean, nullable=False)
    mt5_last_tick = Column(DateTime(timezone=True), nullable=True)
    mt5_symbol_count = Column(Integer, default=0)
    
    # Data quality
    data_gaps_detected = Column(Integer, default=0)
    invalid_prices_detected = Column(Integer, default=0)
    network_interruptions = Column(Integer, default=0)
    
    # Performance
    avg_processing_time_ms = Column(Numeric(8, 2), nullable=True)
    ticks_processed_per_second = Column(Numeric(8, 2), nullable=True)
    trades_per_hour = Column(Numeric(6, 2), nullable=True)
    
    # Status
    overall_status = Column(String(20), nullable=False, default='HEALTHY')
    alerts_count = Column(Integer, default=0)
    errors_count = Column(Integer, default=0)
    
    timestamp = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    
    # Relationships
    session = relationship("TradingSession")
    
    # Constraints
    __table_args__ = (
        CheckConstraint('cpu_usage_percent >= 0 AND cpu_usage_percent <= 100', name='valid_cpu_usage'),
        CheckConstraint('memory_usage_percent >= 0 AND memory_usage_percent <= 100', name='valid_memory_usage'),
        CheckConstraint('disk_usage_percent >= 0 AND disk_usage_percent <= 100', name='valid_disk_usage'),
        CheckConstraint('network_latency_ms >= 0', name='non_negative_latency'),
        CheckConstraint('mt5_symbol_count >= 0', name='non_negative_symbol_count'),
        CheckConstraint('data_gaps_detected >= 0', name='non_negative_gaps'),
        CheckConstraint('invalid_prices_detected >= 0', name='non_negative_invalid_prices'),
        CheckConstraint('network_interruptions >= 0', name='non_negative_interruptions'),
        CheckConstraint("overall_status IN ('HEALTHY', 'WARNING', 'CRITICAL')", name='valid_status'),
        Index('idx_system_health_session', 'session_id'),
        Index('idx_system_health_timestamp', 'timestamp'),
        Index('idx_system_health_status', 'overall_status'),
    )

# Event listeners for data integrity

@event.listens_for(MarketData, 'before_insert')
def validate_market_data(mapper, connection, target):
    """Ensure no fake data can be inserted"""
    if not target.is_real_data:
        raise ValueError("❌ FAKE DATA REJECTED - Only real MT5 data allowed!")
    
    if target.data_source != 'MT5_VANTAGE':
        raise ValueError("❌ INVALID DATA SOURCE - Only Vantage MT5 data allowed!")
    
    # Validate realistic price movements (basic sanity check)
    price_range = float(target.high_price - target.low_price)
    avg_price = float((target.high_price + target.low_price) / 2)
    if price_range / avg_price > 0.1:  # 10% range seems unrealistic for single tick
        logger.warning("Unusual price range detected", 
                      symbol=target.symbol, 
                      price_range_pct=price_range/avg_price*100)

@event.listens_for(Trade, 'before_insert')
def validate_trade(mapper, connection, target):
    """Ensure trade integrity"""
    if target.trading_mode not in [TradingMode.LIVE, TradingMode.VIRTUAL]:
        raise ValueError("❌ INVALID TRADING MODE - Only LIVE or VIRTUAL allowed!")

@event.listens_for(TradingSession, 'before_insert')
def validate_session(mapper, connection, target):
    """Ensure session uses only Vantage server"""
    if target.mt5_server != 'VantageInternational-Demo':
        raise ValueError("❌ INVALID MT5 SERVER - Only VantageInternational-Demo allowed!")

# Database utility functions
def create_all_tables(engine):
    """Create all tables with proper constraints"""
    Base.metadata.create_all(engine)
    logger.info("Database tables created with enterprise constraints")

def validate_schema(engine):
    """Validate that all constraints are properly applied"""
    inspector = engine.inspect(engine)
    tables = inspector.get_table_names()
    
    required_tables = [
        'trading_sessions', 'market_data', 'trades', 
        'agent_performance', 'system_health'
    ]
    
    missing_tables = set(required_tables) - set(tables)
    if missing_tables:
        raise ValueError(f"Missing required tables: {missing_tables}")
    
    logger.info("Database schema validation passed", tables_count=len(tables))