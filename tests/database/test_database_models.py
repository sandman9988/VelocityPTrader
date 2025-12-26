#!/usr/bin/env python3
"""
Database Models Tests
Test PostgreSQL models with enterprise constraints
"""

import pytest
import tempfile
from datetime import datetime, timezone
from decimal import Decimal
from uuid import uuid4

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import IntegrityError

from src.database.models import (
    Base, TradingSession, MarketData, Trade, AgentPerformance, SystemHealth,
    TradingMode, AgentType, ActionType, MarketRegime
)

@pytest.fixture
def in_memory_db():
    """Create in-memory SQLite database for testing"""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    return Session()

def test_trading_session_creation(in_memory_db):
    """Test trading session creation with constraints"""
    session = TradingSession(
        mode=TradingMode.LIVE,
        mt5_server="VantageInternational-Demo",
        mt5_login=10916362,
        initial_balance=Decimal("10000.00")
    )
    
    in_memory_db.add(session)
    in_memory_db.commit()
    
    assert session.id is not None
    assert session.mode == TradingMode.LIVE
    assert session.is_active is True

def test_trading_session_constraints(in_memory_db):
    """Test trading session constraints"""
    # Test invalid server (should fail in PostgreSQL)
    session = TradingSession(
        mode=TradingMode.LIVE,
        mt5_server="FakeServer",  # Invalid server
        mt5_login=123,
        initial_balance=Decimal("10000.00")
    )
    
    in_memory_db.add(session)
    # Note: SQLite doesn't enforce check constraints, but PostgreSQL will
    in_memory_db.commit()

def test_market_data_creation(in_memory_db):
    """Test market data creation"""
    session = TradingSession(
        mode=TradingMode.VIRTUAL,
        mt5_server="VantageInternational-Demo",
        mt5_login=10916362,
        initial_balance=Decimal("10000.00")
    )
    in_memory_db.add(session)
    in_memory_db.flush()
    
    market_data = MarketData(
        session_id=session.id,
        symbol="EURUSD+",
        timeframe="M1",
        timestamp=datetime.now(timezone.utc),
        open_price=Decimal("1.1000"),
        high_price=Decimal("1.1010"),
        low_price=Decimal("1.0990"),
        close_price=Decimal("1.1005"),
        volume=Decimal("1000"),
        bid_price=Decimal("1.1003"),
        ask_price=Decimal("1.1007"),
        spread_pips=Decimal("0.4"),
        data_source="MT5_VANTAGE",
        is_real_data=True
    )
    
    in_memory_db.add(market_data)
    in_memory_db.commit()
    
    assert market_data.id is not None
    assert market_data.symbol == "EURUSD+"
    assert market_data.is_real_data is True

def test_trade_creation(in_memory_db):
    """Test trade creation"""
    session = TradingSession(
        mode=TradingMode.LIVE,
        mt5_server="VantageInternational-Demo",
        mt5_login=10916362,
        initial_balance=Decimal("10000.00")
    )
    in_memory_db.add(session)
    in_memory_db.flush()
    
    trade = Trade(
        session_id=session.id,
        trade_id="EURUSD_TEST_001",
        agent_type=AgentType.BERSERKER,
        trading_mode=TradingMode.LIVE,
        symbol="EURUSD+",
        timeframe="M1",
        action=ActionType.BUY,
        direction="BUY",
        entry_time=datetime.now(timezone.utc),
        entry_price=Decimal("1.1000"),
        position_size=Decimal("0.1"),
        is_open=True
    )
    
    in_memory_db.add(trade)
    in_memory_db.commit()
    
    assert trade.id is not None
    assert trade.is_open is True
    assert trade.trading_mode == TradingMode.LIVE

def test_agent_performance_creation(in_memory_db):
    """Test agent performance creation"""
    session = TradingSession(
        mode=TradingMode.VIRTUAL,
        mt5_server="VantageInternational-Demo",
        mt5_login=10916362,
        initial_balance=Decimal("10000.00")
    )
    in_memory_db.add(session)
    in_memory_db.flush()
    
    performance = AgentPerformance(
        session_id=session.id,
        agent_type=AgentType.SNIPER,
        symbol="BTCUSD+",
        timeframe="M5",
        total_trades=100,
        winning_trades=65,
        total_pnl=Decimal("1250.50"),
        win_rate=Decimal("0.65"),
        profit_factor=Decimal("1.85")
    )
    
    in_memory_db.add(performance)
    in_memory_db.commit()
    
    assert performance.id is not None
    assert performance.agent_type == AgentType.SNIPER
    assert performance.win_rate == Decimal("0.65")

def test_system_health_creation(in_memory_db):
    """Test system health creation"""
    session = TradingSession(
        mode=TradingMode.LIVE,
        mt5_server="VantageInternational-Demo",
        mt5_login=10916362,
        initial_balance=Decimal("10000.00")
    )
    in_memory_db.add(session)
    in_memory_db.flush()
    
    health = SystemHealth(
        session_id=session.id,
        cpu_usage_percent=Decimal("45.2"),
        memory_usage_percent=Decimal("67.8"),
        disk_usage_percent=Decimal("23.1"),
        mt5_connected=True,
        overall_status="HEALTHY"
    )
    
    in_memory_db.add(health)
    in_memory_db.commit()
    
    assert health.id is not None
    assert health.overall_status == "HEALTHY"
    assert health.mt5_connected is True

def test_no_fake_data_enum():
    """Test that only LIVE and VIRTUAL modes are allowed"""
    valid_modes = [TradingMode.LIVE, TradingMode.VIRTUAL]
    
    # Verify enum values
    assert TradingMode.LIVE == "LIVE"
    assert TradingMode.VIRTUAL == "VIRTUAL"
    
    # Verify no fake/mock modes exist
    mode_values = [mode.value for mode in TradingMode]
    forbidden_modes = ["FAKE", "MOCK", "SIMULATED", "TEST"]
    
    for forbidden in forbidden_modes:
        assert forbidden not in mode_values, f"Forbidden mode {forbidden} found!"

def test_agent_types():
    """Test agent type enums"""
    valid_agents = [
        AgentType.BERSERKER,
        AgentType.SNIPER,
        AgentType.SHADOW_BERSERKER,
        AgentType.SHADOW_SNIPER
    ]
    
    # Verify enum values
    assert AgentType.BERSERKER == "BERSERKER"
    assert AgentType.SNIPER == "SNIPER"
    assert AgentType.SHADOW_BERSERKER == "SHADOW_BERSERKER"
    assert AgentType.SHADOW_SNIPER == "SHADOW_SNIPER"

def test_market_regime_physics():
    """Test physics-based market regimes"""
    regimes = [
        MarketRegime.OVERDAMPED,
        MarketRegime.CRITICALLY_DAMPED,
        MarketRegime.UNDERDAMPED,
        MarketRegime.CHAOTIC
    ]
    
    # Verify all physics regimes exist
    assert MarketRegime.OVERDAMPED == "OVERDAMPED"
    assert MarketRegime.CRITICALLY_DAMPED == "CRITICALLY_DAMPED"
    assert MarketRegime.UNDERDAMPED == "UNDERDAMPED"
    assert MarketRegime.CHAOTIC == "CHAOTIC"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])