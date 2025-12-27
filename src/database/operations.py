#!/usr/bin/env python3
"""
ATOMIC DATABASE OPERATIONS
Enterprise-grade atomic operations for VelocityTrader with data integrity
"""

import asyncio
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from typing import Optional, Dict, Any, List, Union, Tuple
from uuid import UUID, uuid4
import time

from sqlalchemy import func, and_, or_, desc, asc
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session
import structlog

from .models import (
    TradingSession, MarketData, Trade, AgentPerformance, SystemHealth, ErrorLog,
    TradingMode, AgentType, ActionType, MarketRegime
)
from .connection import DatabaseManager, get_database_manager, AtomicOperationError

logger = structlog.get_logger(__name__)

class AtomicDataOperations:
    """Atomic database operations with enterprise-grade safety"""
    
    def __init__(self, db_manager: Optional[DatabaseManager] = None):
        self.db = db_manager or get_database_manager()
        self._operation_count = 0
        self._start_time = time.time()
        
        logger.info("Atomic data operations initialized")
    
    # TRADING SESSION OPERATIONS
    
    def create_trading_session(
        self,
        mode: TradingMode,
        mt5_server: str,
        mt5_login: int,
        initial_balance: Decimal
    ) -> TradingSession:
        """Create new trading session with atomic safety"""
        if mt5_server != "VantageInternational-Demo":
            raise ValueError("❌ Only VantageInternational-Demo server allowed!")
        
        try:
            with self.db.atomic_transaction() as session:
                # Check for existing active session
                existing_session = session.query(TradingSession).filter(
                    and_(
                        TradingSession.is_active == True,
                        TradingSession.mode == mode,
                        TradingSession.mt5_login == mt5_login
                    )
                ).first()
                
                if existing_session:
                    raise AtomicOperationError(
                        f"Active {mode} session already exists for login {mt5_login}"
                    )
                
                # Create new session
                new_session = TradingSession(
                    mode=mode,
                    mt5_server=mt5_server,
                    mt5_login=mt5_login,
                    initial_balance=initial_balance
                )
                
                session.add(new_session)
                session.flush()  # Get ID without committing
                
                logger.info("Trading session created atomically", 
                           session_id=str(new_session.id),
                           mode=mode,
                           initial_balance=float(initial_balance))
                
                self._operation_count += 1
                return new_session
                
        except Exception as e:
            logger.error("Failed to create trading session", error=str(e))
            raise AtomicOperationError(f"Session creation failed: {e}") from e
    
    def close_trading_session(self, session_id: UUID, final_balance: Decimal) -> TradingSession:
        """Close trading session atomically"""
        try:
            with self.db.atomic_transaction() as session:
                trading_session = session.query(TradingSession).filter(
                    TradingSession.id == session_id
                ).first()
                
                if not trading_session:
                    raise AtomicOperationError(f"Session {session_id} not found")
                
                if not trading_session.is_active:
                    raise AtomicOperationError(f"Session {session_id} already closed")
                
                # Update session
                trading_session.session_end = datetime.now(timezone.utc)
                trading_session.final_balance = final_balance
                trading_session.is_active = False
                
                # Calculate final statistics
                total_trades = session.query(func.count(Trade.id)).filter(
                    Trade.session_id == session_id
                ).scalar() or 0
                
                winning_trades = session.query(func.count(Trade.id)).filter(
                    and_(
                        Trade.session_id == session_id,
                        Trade.realized_pnl > 0
                    )
                ).scalar() or 0
                
                trading_session.total_trades = total_trades
                trading_session.winning_trades = winning_trades
                
                session.flush()
                
                logger.info("Trading session closed atomically",
                           session_id=str(session_id),
                           total_trades=total_trades,
                           winning_trades=winning_trades,
                           final_balance=float(final_balance))
                
                self._operation_count += 1
                return trading_session
                
        except Exception as e:
            logger.error("Failed to close trading session", error=str(e))
            raise AtomicOperationError(f"Session closing failed: {e}") from e
    
    # MARKET DATA OPERATIONS
    
    def insert_market_data_batch(
        self,
        session_id: UUID,
        market_data_list: List[Dict[str, Any]]
    ) -> List[MarketData]:
        """Insert market data batch atomically"""
        if not market_data_list:
            return []
        
        try:
            with self.db.atomic_transaction() as session:
                inserted_data = []
                
                for data_dict in market_data_list:
                    # Validate required fields
                    required_fields = [
                        'symbol', 'timeframe', 'timestamp', 'open_price', 
                        'high_price', 'low_price', 'close_price', 'volume',
                        'bid_price', 'ask_price', 'spread_pips'
                    ]
                    
                    missing_fields = [field for field in required_fields if field not in data_dict]
                    if missing_fields:
                        raise ValueError(f"Missing required fields: {missing_fields}")
                    
                    # Validate data integrity
                    self._validate_market_data(data_dict)
                    
                    # Create MarketData object
                    market_data = MarketData(
                        session_id=session_id,
                        symbol=data_dict['symbol'],
                        timeframe=data_dict['timeframe'],
                        timestamp=data_dict['timestamp'],
                        open_price=Decimal(str(data_dict['open_price'])),
                        high_price=Decimal(str(data_dict['high_price'])),
                        low_price=Decimal(str(data_dict['low_price'])),
                        close_price=Decimal(str(data_dict['close_price'])),
                        volume=Decimal(str(data_dict['volume'])),
                        bid_price=Decimal(str(data_dict['bid_price'])),
                        ask_price=Decimal(str(data_dict['ask_price'])),
                        spread_pips=Decimal(str(data_dict['spread_pips'])),
                        momentum=Decimal(str(data_dict.get('momentum', 0))) if data_dict.get('momentum') else None,
                        acceleration=Decimal(str(data_dict.get('acceleration', 0))) if data_dict.get('acceleration') else None,
                        volatility=Decimal(str(data_dict.get('volatility', 0))) if data_dict.get('volatility') else None,
                        liquidity_score=Decimal(str(data_dict.get('liquidity_score', 0))) if data_dict.get('liquidity_score') else None,
                        trend_strength=Decimal(str(data_dict.get('trend_strength', 0))) if data_dict.get('trend_strength') else None,
                        market_regime=MarketRegime(data_dict['market_regime']) if data_dict.get('market_regime') else None,
                        data_source='MT5_VANTAGE',
                        is_real_data=True
                    )
                    
                    session.add(market_data)
                    inserted_data.append(market_data)
                
                session.flush()  # Flush to get IDs
                
                logger.info("Market data batch inserted atomically",
                           session_id=str(session_id),
                           count=len(inserted_data),
                           symbols=[data.symbol for data in inserted_data[:5]])  # Log first 5
                
                self._operation_count += len(inserted_data)
                return inserted_data
                
        except Exception as e:
            logger.error("Failed to insert market data batch", error=str(e))
            raise AtomicOperationError(f"Market data insertion failed: {e}") from e
    
    def _validate_market_data(self, data: Dict[str, Any]):
        """Validate market data integrity"""
        # Price validation
        open_price = float(data['open_price'])
        high_price = float(data['high_price'])
        low_price = float(data['low_price'])
        close_price = float(data['close_price'])
        bid_price = float(data['bid_price'])
        ask_price = float(data['ask_price'])
        
        # Basic price logic validation
        if not (low_price <= open_price <= high_price):
            raise ValueError(f"Invalid OHLC: open {open_price} not within low-high range")
        
        if not (low_price <= close_price <= high_price):
            raise ValueError(f"Invalid OHLC: close {close_price} not within low-high range")
        
        if bid_price <= 0 or ask_price <= 0:
            raise ValueError("Bid and ask prices must be positive")
        
        if ask_price < bid_price:
            raise ValueError(f"Ask price {ask_price} cannot be less than bid price {bid_price}")
        
        # Spread validation
        spread_pips = float(data['spread_pips'])
        if spread_pips < 0:
            raise ValueError("Spread cannot be negative")
        
        # Volume validation
        volume = float(data['volume'])
        if volume < 0:
            raise ValueError("Volume cannot be negative")
        
        # Reject obviously fake data
        symbol = data['symbol']
        if 'EURUSD' in symbol and (open_price < 0.5 or open_price > 2.0):
            raise ValueError(f"Unrealistic EURUSD price: {open_price}")
        
        if 'BTCUSD' in symbol and (open_price < 1000 or open_price > 1000000):
            raise ValueError(f"Unrealistic BTCUSD price: {open_price}")
    
    # TRADE OPERATIONS
    
    def create_trade(
        self,
        session_id: UUID,
        agent_type: AgentType,
        trading_mode: TradingMode,
        symbol: str,
        timeframe: str,
        action: ActionType,
        entry_price: Decimal,
        position_size: Decimal,
        market_data_id: Optional[UUID] = None,
        **kwargs
    ) -> Trade:
        """Create new trade atomically"""
        try:
            with self.db.atomic_transaction() as session:
                # Generate unique trade ID
                trade_id = f"{symbol}_{int(time.time() * 1000)}_{uuid4().hex[:8]}"
                
                # Validate position size
                if position_size <= 0:
                    raise ValueError("Position size must be positive")
                
                # Create trade
                trade = Trade(
                    session_id=session_id,
                    market_data_id=market_data_id,
                    trade_id=trade_id,
                    agent_type=agent_type,
                    trading_mode=trading_mode,
                    symbol=symbol,
                    timeframe=timeframe,
                    action=action,
                    direction=kwargs.get('direction'),
                    entry_time=datetime.now(timezone.utc),
                    entry_price=entry_price,
                    position_size=position_size,
                    stop_loss=Decimal(str(kwargs.get('stop_loss'))) if kwargs.get('stop_loss') else None,
                    take_profit=Decimal(str(kwargs.get('take_profit'))) if kwargs.get('take_profit') else None,
                    confidence_score=Decimal(str(kwargs.get('confidence_score'))) if kwargs.get('confidence_score') else None,
                    market_conditions=kwargs.get('market_conditions'),
                    mt5_position_id=kwargs.get('mt5_position_id')
                )
                
                session.add(trade)
                session.flush()
                
                logger.info("Trade created atomically",
                           trade_id=trade_id,
                           agent=agent_type,
                           mode=trading_mode,
                           symbol=symbol,
                           entry_price=float(entry_price))
                
                self._operation_count += 1
                return trade
                
        except Exception as e:
            logger.error("Failed to create trade", error=str(e))
            raise AtomicOperationError(f"Trade creation failed: {e}") from e
    
    def update_trade_position(
        self,
        trade_id: str,
        unrealized_pnl: Decimal,
        max_adverse_excursion: Optional[Decimal] = None,
        max_favorable_excursion: Optional[Decimal] = None
    ) -> Trade:
        """Update trade position atomically"""
        try:
            with self.db.atomic_transaction() as session:
                trade = session.query(Trade).filter(
                    Trade.trade_id == trade_id
                ).first()
                
                if not trade:
                    raise AtomicOperationError(f"Trade {trade_id} not found")
                
                if not trade.is_open:
                    raise AtomicOperationError(f"Trade {trade_id} is already closed")
                
                # Update P&L and risk metrics
                trade.unrealized_pnl = unrealized_pnl
                
                if max_adverse_excursion is not None:
                    trade.max_adverse_excursion = min(trade.max_adverse_excursion or 0, max_adverse_excursion)
                
                if max_favorable_excursion is not None:
                    trade.max_favorable_excursion = max(trade.max_favorable_excursion or 0, max_favorable_excursion)
                
                session.flush()
                
                logger.debug("Trade position updated atomically",
                            trade_id=trade_id,
                            unrealized_pnl=float(unrealized_pnl))
                
                self._operation_count += 1
                return trade
                
        except Exception as e:
            logger.error("Failed to update trade position", error=str(e))
            raise AtomicOperationError(f"Trade position update failed: {e}") from e
    
    def close_trade(
        self,
        trade_id: str,
        exit_price: Decimal,
        exit_reason: str,
        **kwargs
    ) -> Trade:
        """Close trade atomically"""
        try:
            with self.db.atomic_transaction() as session:
                trade = session.query(Trade).filter(
                    Trade.trade_id == trade_id
                ).first()
                
                if not trade:
                    raise AtomicOperationError(f"Trade {trade_id} not found")
                
                if not trade.is_open:
                    raise AtomicOperationError(f"Trade {trade_id} is already closed")
                
                # Calculate final P&L
                if trade.direction == "BUY":
                    realized_pnl = (exit_price - trade.entry_price) * trade.position_size
                else:  # SELL
                    realized_pnl = (trade.entry_price - exit_price) * trade.position_size
                
                # Subtract costs
                total_costs = (trade.spread_cost or 0) + (trade.commission or 0) + (trade.swap or 0)
                realized_pnl -= Decimal(str(total_costs))
                
                # Update trade
                trade.exit_time = datetime.now(timezone.utc)
                trade.exit_price = exit_price
                trade.realized_pnl = realized_pnl
                trade.is_open = False
                trade.is_winning = realized_pnl > 0
                trade.exit_reason = exit_reason
                trade.trade_duration_seconds = int((trade.exit_time - trade.entry_time).total_seconds())
                
                # Update costs if provided
                if 'spread_cost' in kwargs:
                    trade.spread_cost = Decimal(str(kwargs['spread_cost']))
                if 'commission' in kwargs:
                    trade.commission = Decimal(str(kwargs['commission']))
                if 'swap' in kwargs:
                    trade.swap = Decimal(str(kwargs['swap']))
                if 'slippage' in kwargs:
                    trade.slippage = Decimal(str(kwargs['slippage']))
                
                session.flush()
                
                logger.info("Trade closed atomically",
                           trade_id=trade_id,
                           realized_pnl=float(realized_pnl),
                           duration_seconds=trade.trade_duration_seconds,
                           exit_reason=exit_reason)
                
                self._operation_count += 1
                return trade
                
        except Exception as e:
            logger.error("Failed to close trade", error=str(e))
            raise AtomicOperationError(f"Trade closing failed: {e}") from e
    
    # PERFORMANCE TRACKING
    
    def update_agent_performance(
        self,
        session_id: UUID,
        agent_type: AgentType,
        symbol: str,
        timeframe: str,
        performance_data: Dict[str, Any]
    ) -> AgentPerformance:
        """Update agent performance atomically"""
        try:
            with self.db.atomic_transaction() as session:
                # Find existing performance record
                performance = session.query(AgentPerformance).filter(
                    and_(
                        AgentPerformance.session_id == session_id,
                        AgentPerformance.agent_type == agent_type,
                        AgentPerformance.symbol == symbol,
                        AgentPerformance.timeframe == timeframe,
                        AgentPerformance.period_end == None  # Current period
                    )
                ).first()
                
                if not performance:
                    # Create new performance record
                    performance = AgentPerformance(
                        session_id=session_id,
                        agent_type=agent_type,
                        symbol=symbol,
                        timeframe=timeframe,
                        period_start=datetime.now(timezone.utc)
                    )
                    session.add(performance)
                
                # Update performance metrics
                for field, value in performance_data.items():
                    if hasattr(performance, field):
                        if isinstance(value, (int, float)):
                            setattr(performance, field, Decimal(str(value)))
                        else:
                            setattr(performance, field, value)
                
                performance.last_model_update = datetime.now(timezone.utc)
                
                session.flush()
                
                logger.debug("Agent performance updated atomically",
                            agent=agent_type,
                            symbol=symbol,
                            timeframe=timeframe)
                
                self._operation_count += 1
                return performance
                
        except Exception as e:
            logger.error("Failed to update agent performance", error=str(e))
            raise AtomicOperationError(f"Performance update failed: {e}") from e
    
    # SYSTEM HEALTH
    
    def log_system_health(
        self,
        session_id: UUID,
        health_data: Dict[str, Any]
    ) -> SystemHealth:
        """Log system health metrics atomically"""
        try:
            with self.db.atomic_transaction() as session:
                health = SystemHealth(
                    session_id=session_id,
                    cpu_usage_percent=Decimal(str(health_data.get('cpu_usage', 0))),
                    memory_usage_percent=Decimal(str(health_data.get('memory_usage', 0))),
                    disk_usage_percent=Decimal(str(health_data.get('disk_usage', 0))),
                    network_latency_ms=Decimal(str(health_data.get('network_latency', 0))) if health_data.get('network_latency') else None,
                    mt5_connected=bool(health_data.get('mt5_connected', False)),
                    mt5_last_tick=health_data.get('mt5_last_tick'),
                    mt5_symbol_count=int(health_data.get('mt5_symbol_count', 0)),
                    data_gaps_detected=int(health_data.get('data_gaps', 0)),
                    invalid_prices_detected=int(health_data.get('invalid_prices', 0)),
                    network_interruptions=int(health_data.get('network_interruptions', 0)),
                    avg_processing_time_ms=Decimal(str(health_data.get('avg_processing_time', 0))) if health_data.get('avg_processing_time') else None,
                    ticks_processed_per_second=Decimal(str(health_data.get('ticks_per_second', 0))) if health_data.get('ticks_per_second') else None,
                    trades_per_hour=Decimal(str(health_data.get('trades_per_hour', 0))) if health_data.get('trades_per_hour') else None,
                    overall_status=health_data.get('overall_status', 'HEALTHY'),
                    alerts_count=int(health_data.get('alerts_count', 0)),
                    errors_count=int(health_data.get('errors_count', 0))
                )
                
                session.add(health)
                session.flush()
                
                logger.debug("System health logged atomically",
                            status=health.overall_status,
                            cpu=float(health.cpu_usage_percent),
                            memory=float(health.memory_usage_percent))
                
                self._operation_count += 1
                return health
                
        except Exception as e:
            logger.error("Failed to log system health", error=str(e))
            raise AtomicOperationError(f"Health logging failed: {e}") from e

    # ERROR LOGGING

    def log_error(
        self,
        session_id: Optional[UUID],
        component: str,
        error_message: str,
        severity: str,
        error_type: str = "UNKNOWN",
        stack_trace: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> ErrorLog:
        """Log error to database atomically for tracking and debugging"""
        try:
            with self.db.atomic_transaction() as session:
                # Validate severity
                valid_severities = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
                if severity not in valid_severities:
                    severity = 'ERROR'

                # Truncate error message if too long
                if len(error_message) > 10000:
                    error_message = error_message[:10000] + "... [truncated]"

                error_log = ErrorLog(
                    session_id=session_id,
                    component=component[:100] if component else "UNKNOWN",
                    error_type=error_type[:100] if error_type else "UNKNOWN",
                    error_message=error_message,
                    severity=severity,
                    stack_trace=stack_trace,
                    context=context or {},
                    resolved=False
                )

                session.add(error_log)
                session.flush()

                logger.debug("Error logged to database atomically",
                            error_id=str(error_log.id),
                            component=component,
                            severity=severity)

                self._operation_count += 1
                return error_log

        except Exception as e:
            # Don't raise - logging errors shouldn't crash the system
            logger.warning("Failed to persist error to database",
                          original_error=error_message,
                          db_error=str(e))
            return None

    def get_unresolved_errors(
        self,
        session_id: Optional[UUID] = None,
        severity: Optional[str] = None,
        component: Optional[str] = None,
        limit: int = 100
    ) -> List[ErrorLog]:
        """Get unresolved errors for debugging"""
        try:
            with self.db.get_session() as session:
                query = session.query(ErrorLog).filter(ErrorLog.resolved == False)

                if session_id:
                    query = query.filter(ErrorLog.session_id == session_id)
                if severity:
                    query = query.filter(ErrorLog.severity == severity)
                if component:
                    query = query.filter(ErrorLog.component == component)

                errors = query.order_by(desc(ErrorLog.timestamp)).limit(limit).all()

                logger.debug("Unresolved errors retrieved", count=len(errors))
                return errors

        except Exception as e:
            logger.error("Failed to retrieve errors", error=str(e))
            return []

    def resolve_error(
        self,
        error_id: UUID,
        resolution_notes: Optional[str] = None
    ) -> Optional[ErrorLog]:
        """Mark an error as resolved"""
        try:
            with self.db.atomic_transaction() as session:
                error_log = session.query(ErrorLog).filter(
                    ErrorLog.id == error_id
                ).first()

                if not error_log:
                    logger.warning("Error not found for resolution", error_id=str(error_id))
                    return None

                error_log.resolved = True
                error_log.resolved_at = datetime.now(timezone.utc)
                error_log.resolution_notes = resolution_notes

                session.flush()

                logger.info("Error resolved",
                           error_id=str(error_id),
                           component=error_log.component)

                self._operation_count += 1
                return error_log

        except Exception as e:
            logger.error("Failed to resolve error", error=str(e))
            return None

    def get_error_statistics(
        self,
        session_id: Optional[UUID] = None,
        hours: int = 24
    ) -> Dict[str, Any]:
        """Get error statistics for monitoring"""
        try:
            with self.db.get_session() as session:
                cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)

                query = session.query(ErrorLog).filter(ErrorLog.timestamp >= cutoff)
                if session_id:
                    query = query.filter(ErrorLog.session_id == session_id)

                total = query.count()
                critical = query.filter(ErrorLog.severity == 'CRITICAL').count()
                errors = query.filter(ErrorLog.severity == 'ERROR').count()
                warnings = query.filter(ErrorLog.severity == 'WARNING').count()
                resolved = query.filter(ErrorLog.resolved == True).count()

                # Get top error components
                component_counts = session.query(
                    ErrorLog.component,
                    func.count(ErrorLog.id).label('count')
                ).filter(
                    ErrorLog.timestamp >= cutoff
                ).group_by(ErrorLog.component).order_by(desc('count')).limit(5).all()

                return {
                    'period_hours': hours,
                    'total_errors': total,
                    'critical_count': critical,
                    'error_count': errors,
                    'warning_count': warnings,
                    'resolved_count': resolved,
                    'unresolved_count': total - resolved,
                    'resolution_rate': resolved / total if total > 0 else 1.0,
                    'top_components': [{'component': c, 'count': cnt} for c, cnt in component_counts]
                }

        except Exception as e:
            logger.error("Failed to get error statistics", error=str(e))
            return {'error': str(e)}

    # QUERY OPERATIONS
    
    def get_active_trades(self, session_id: UUID, agent_type: Optional[AgentType] = None) -> List[Trade]:
        """Get active trades for session"""
        try:
            with self.db.get_session() as session:
                query = session.query(Trade).filter(
                    and_(
                        Trade.session_id == session_id,
                        Trade.is_open == True
                    )
                )
                
                if agent_type:
                    query = query.filter(Trade.agent_type == agent_type)
                
                trades = query.order_by(desc(Trade.entry_time)).all()
                
                logger.debug("Active trades retrieved",
                           session_id=str(session_id),
                           count=len(trades),
                           agent=agent_type)
                
                return trades
                
        except Exception as e:
            logger.error("Failed to retrieve active trades", error=str(e))
            raise AtomicOperationError(f"Active trades query failed: {e}") from e
    
    def get_recent_market_data(
        self,
        session_id: UUID,
        symbol: str,
        timeframe: str,
        limit: int = 100
    ) -> List[MarketData]:
        """Get recent market data"""
        try:
            with self.db.get_session() as session:
                data = session.query(MarketData).filter(
                    and_(
                        MarketData.session_id == session_id,
                        MarketData.symbol == symbol,
                        MarketData.timeframe == timeframe
                    )
                ).order_by(desc(MarketData.timestamp)).limit(limit).all()
                
                logger.debug("Recent market data retrieved",
                           symbol=symbol,
                           timeframe=timeframe,
                           count=len(data))
                
                return data
                
        except Exception as e:
            logger.error("Failed to retrieve market data", error=str(e))
            raise AtomicOperationError(f"Market data query failed: {e}") from e
    
    def get_operation_stats(self) -> Dict[str, Any]:
        """Get operation statistics"""
        return {
            "total_operations": self._operation_count,
            "uptime_seconds": time.time() - self._start_time,
            "operations_per_second": self._operation_count / (time.time() - self._start_time) if time.time() > self._start_time else 0
        }