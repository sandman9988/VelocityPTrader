#!/usr/bin/env python3
"""
COMPREHENSIVE LOGGING SYSTEM - DEFENSE IN DEPTH
Substantive performance, error, and financial trade logging for accelerated RL
"""

import json
import time
import threading
import sqlite3
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum
import logging
import structlog

class LogLevel(Enum):
    TRADE = "TRADE"
    PERFORMANCE = "PERFORMANCE" 
    ERROR = "ERROR"
    SYSTEM = "SYSTEM"
    REWARD = "REWARD"
    ANALYSIS = "ANALYSIS"

@dataclass
class TradeLog:
    """Financial trade logging for analysis"""
    timestamp: float
    agent_id: str
    instrument: str
    timeframe: str
    action: str  # BUY/SELL/CLOSE
    entry_price: float
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
    win_loss: Optional[str]  # WIN/LOSS/SCRATCH
    trade_id: str
    virtual_trade: bool

@dataclass
class PerformanceLog:
    """Performance metrics for RL acceleration"""
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
    """Comprehensive error logging for defense in depth"""
    timestamp: float
    error_type: str
    error_message: str
    error_code: Optional[str]
    component: str
    severity: str  # CRITICAL/HIGH/MEDIUM/LOW
    stack_trace: str
    context: Dict[str, Any]
    recovery_action: Optional[str]
    error_count: int
    resolved: bool

@dataclass
class RewardShapingLog:
    """Reward shaping for RL optimization"""
    timestamp: float
    agent_id: str
    instrument: str
    state_vector: List[float]
    action_taken: str
    immediate_reward: float
    shaped_reward: float
    reward_components: Dict[str, float]
    q_values: List[float]
    exploration_bonus: float
    risk_penalty: float
    market_regime: str
    learning_rate: float
    epsilon: float

class ComprehensiveLogger:
    """Comprehensive logging system with SQLite backend"""
    
    def __init__(self, db_path: str = "trading_logs.db"):
        self.db_path = db_path
        self.lock = threading.Lock()
        self.setup_database()
        self.setup_structured_logging()
        
        # Performance metrics cache
        self.performance_cache = {}
        self.error_counts = {}
        
        # Start background analytics
        self.start_background_analytics()
    
    def setup_database(self):
        """Initialize SQLite database with proper schema"""
        with sqlite3.connect(self.db_path) as conn:
            # Trade logs table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS trade_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL,
                    agent_id TEXT,
                    instrument TEXT,
                    timeframe TEXT,
                    action TEXT,
                    entry_price REAL,
                    exit_price REAL,
                    position_size REAL,
                    pnl REAL,
                    mae REAL,
                    mfe REAL,
                    trade_duration REAL,
                    spread_cost REAL,
                    slippage REAL,
                    market_conditions TEXT,
                    confidence_score REAL,
                    risk_reward_ratio REAL,
                    win_loss TEXT,
                    trade_id TEXT UNIQUE,
                    virtual_trade BOOLEAN
                )
            ''')
            
            # Performance logs table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS performance_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL,
                    agent_id TEXT,
                    instrument TEXT,
                    timeframe TEXT,
                    win_rate REAL,
                    profit_factor REAL,
                    sharpe_ratio REAL,
                    max_drawdown REAL,
                    total_trades INTEGER,
                    avg_trade_duration REAL,
                    hit_ratio REAL,
                    avg_win REAL,
                    avg_loss REAL,
                    consecutive_wins INTEGER,
                    consecutive_losses INTEGER,
                    volatility_adjusted_return REAL,
                    kelly_criterion REAL,
                    calmar_ratio REAL
                )
            ''')
            
            # Error logs table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS error_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL,
                    error_type TEXT,
                    error_message TEXT,
                    error_code TEXT,
                    component TEXT,
                    severity TEXT,
                    stack_trace TEXT,
                    context TEXT,
                    recovery_action TEXT,
                    error_count INTEGER,
                    resolved BOOLEAN
                )
            ''')
            
            # Reward shaping logs table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS reward_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL,
                    agent_id TEXT,
                    instrument TEXT,
                    state_vector TEXT,
                    action_taken TEXT,
                    immediate_reward REAL,
                    shaped_reward REAL,
                    reward_components TEXT,
                    q_values TEXT,
                    exploration_bonus REAL,
                    risk_penalty REAL,
                    market_regime TEXT,
                    learning_rate REAL,
                    epsilon REAL
                )
            ''')
            
            # Create indexes for performance
            conn.execute('CREATE INDEX IF NOT EXISTS idx_trade_agent_time ON trade_logs(agent_id, timestamp)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_performance_agent_time ON performance_logs(agent_id, timestamp)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_error_severity_time ON error_logs(severity, timestamp)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_reward_agent_time ON reward_logs(agent_id, timestamp)')
    
    def setup_structured_logging(self):
        """Setup structured logging with JSON output"""
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.processors.TimeStamper(fmt="ISO"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.JSONRenderer()
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )
        
        self.logger = structlog.get_logger()
    
    def log_trade(self, trade_log: TradeLog):
        """Log financial trade with comprehensive analysis"""
        try:
            with self.lock:
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute('''
                        INSERT INTO trade_logs (
                            timestamp, agent_id, instrument, timeframe, action,
                            entry_price, exit_price, position_size, pnl, mae, mfe,
                            trade_duration, spread_cost, slippage, market_conditions,
                            confidence_score, risk_reward_ratio, win_loss, trade_id, virtual_trade
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        trade_log.timestamp, trade_log.agent_id, trade_log.instrument,
                        trade_log.timeframe, trade_log.action, trade_log.entry_price,
                        trade_log.exit_price, trade_log.position_size, trade_log.pnl,
                        trade_log.mae, trade_log.mfe, trade_log.trade_duration,
                        trade_log.spread_cost, trade_log.slippage,
                        json.dumps(trade_log.market_conditions), trade_log.confidence_score,
                        trade_log.risk_reward_ratio, trade_log.win_loss,
                        trade_log.trade_id, trade_log.virtual_trade
                    ))
                
                # Structured logging
                self.logger.info(
                    "trade_executed",
                    agent_id=trade_log.agent_id,
                    instrument=trade_log.instrument,
                    action=trade_log.action,
                    pnl=trade_log.pnl,
                    virtual=trade_log.virtual_trade
                )
                
                # Trigger performance recalculation
                self.update_performance_metrics(trade_log.agent_id, trade_log.instrument)
                
        except Exception as e:
            self.log_error("trade_logging", str(e), "CRITICAL", {"trade_id": trade_log.trade_id})
    
    def log_performance(self, perf_log: PerformanceLog):
        """Log performance metrics for RL optimization"""
        try:
            with self.lock:
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute('''
                        INSERT INTO performance_logs (
                            timestamp, agent_id, instrument, timeframe, win_rate,
                            profit_factor, sharpe_ratio, max_drawdown, total_trades,
                            avg_trade_duration, hit_ratio, avg_win, avg_loss,
                            consecutive_wins, consecutive_losses, volatility_adjusted_return,
                            kelly_criterion, calmar_ratio
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        perf_log.timestamp, perf_log.agent_id, perf_log.instrument,
                        perf_log.timeframe, perf_log.win_rate, perf_log.profit_factor,
                        perf_log.sharpe_ratio, perf_log.max_drawdown, perf_log.total_trades,
                        perf_log.avg_trade_duration, perf_log.hit_ratio, perf_log.avg_win,
                        perf_log.avg_loss, perf_log.consecutive_wins, perf_log.consecutive_losses,
                        perf_log.volatility_adjusted_return, perf_log.kelly_criterion,
                        perf_log.calmar_ratio
                    ))
                
                # Cache for quick access
                cache_key = f"{perf_log.agent_id}_{perf_log.instrument}_{perf_log.timeframe}"
                self.performance_cache[cache_key] = perf_log
                
        except Exception as e:
            self.log_error("performance_logging", str(e), "HIGH", {"agent_id": perf_log.agent_id})
    
    def log_error(self, component: str, error_msg: str, severity: str, context: Dict[str, Any] = None):
        """Comprehensive error logging with defense in depth"""
        try:
            error_log = ErrorLog(
                timestamp=time.time(),
                error_type=type(Exception).__name__,
                error_message=error_msg,
                error_code=None,
                component=component,
                severity=severity,
                stack_trace=traceback.format_exc(),
                context=context or {},
                recovery_action=None,
                error_count=self.error_counts.get(component, 0) + 1,
                resolved=False
            )
            
            self.error_counts[component] = error_log.error_count
            
            with self.lock:
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute('''
                        INSERT INTO error_logs (
                            timestamp, error_type, error_message, error_code, component,
                            severity, stack_trace, context, recovery_action, error_count, resolved
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        error_log.timestamp, error_log.error_type, error_log.error_message,
                        error_log.error_code, error_log.component, error_log.severity,
                        error_log.stack_trace, json.dumps(error_log.context),
                        error_log.recovery_action, error_log.error_count, error_log.resolved
                    ))
            
            # Structured logging
            self.logger.error(
                "system_error",
                component=component,
                severity=severity,
                error_count=error_log.error_count,
                message=error_msg
            )
            
        except Exception as e:
            # Fallback logging if main logging fails
            print(f"CRITICAL: Logging system failure: {e}")
    
    def log_reward_shaping(self, reward_log: RewardShapingLog):
        """Log reward shaping for RL acceleration"""
        try:
            with self.lock:
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute('''
                        INSERT INTO reward_logs (
                            timestamp, agent_id, instrument, state_vector, action_taken,
                            immediate_reward, shaped_reward, reward_components, q_values,
                            exploration_bonus, risk_penalty, market_regime, learning_rate, epsilon
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        reward_log.timestamp, reward_log.agent_id, reward_log.instrument,
                        json.dumps(reward_log.state_vector), reward_log.action_taken,
                        reward_log.immediate_reward, reward_log.shaped_reward,
                        json.dumps(reward_log.reward_components), json.dumps(reward_log.q_values),
                        reward_log.exploration_bonus, reward_log.risk_penalty,
                        reward_log.market_regime, reward_log.learning_rate, reward_log.epsilon
                    ))
                
        except Exception as e:
            self.log_error("reward_logging", str(e), "MEDIUM", {"agent_id": reward_log.agent_id})
    
    def update_performance_metrics(self, agent_id: str, instrument: str):
        """Calculate and update performance metrics from trade history"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Get recent trades for this agent/instrument
                trades = conn.execute('''
                    SELECT * FROM trade_logs 
                    WHERE agent_id = ? AND instrument = ? AND pnl IS NOT NULL
                    ORDER BY timestamp DESC LIMIT 100
                ''', (agent_id, instrument)).fetchall()
                
                if len(trades) < 5:  # Need minimum trades for meaningful metrics
                    return
                
                # Calculate performance metrics
                pnls = [trade[9] for trade in trades if trade[9] is not None]  # pnl column
                wins = [pnl for pnl in pnls if pnl > 0]
                losses = [pnl for pnl in pnls if pnl < 0]
                
                win_rate = len(wins) / len(pnls) if pnls else 0
                profit_factor = sum(wins) / abs(sum(losses)) if losses else float('inf')
                
                # Create performance log
                perf_log = PerformanceLog(
                    timestamp=time.time(),
                    agent_id=agent_id,
                    instrument=instrument,
                    timeframe="ALL",
                    win_rate=win_rate,
                    profit_factor=profit_factor,
                    sharpe_ratio=self.calculate_sharpe_ratio(pnls),
                    max_drawdown=self.calculate_max_drawdown(pnls),
                    total_trades=len(pnls),
                    avg_trade_duration=sum(trade[12] or 0 for trade in trades) / len(trades),
                    hit_ratio=win_rate,
                    avg_win=sum(wins) / len(wins) if wins else 0,
                    avg_loss=sum(losses) / len(losses) if losses else 0,
                    consecutive_wins=self.calculate_consecutive_wins(pnls),
                    consecutive_losses=self.calculate_consecutive_losses(pnls),
                    volatility_adjusted_return=self.calculate_volatility_adjusted_return(pnls),
                    kelly_criterion=self.calculate_kelly_criterion(wins, losses),
                    calmar_ratio=self.calculate_calmar_ratio(pnls)
                )
                
                self.log_performance(perf_log)
                
        except Exception as e:
            self.log_error("performance_calculation", str(e), "MEDIUM", 
                         {"agent_id": agent_id, "instrument": instrument})
    
    def calculate_sharpe_ratio(self, returns: List[float]) -> float:
        """Calculate Sharpe ratio for risk-adjusted returns"""
        if not returns or len(returns) < 2:
            return 0.0
        
        import numpy as np
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return == 0:
            return 0.0
        
        return mean_return / std_return
    
    def calculate_max_drawdown(self, returns: List[float]) -> float:
        """Calculate maximum drawdown"""
        if not returns:
            return 0.0
        
        import numpy as np
        cumulative = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = cumulative - running_max
        
        return float(np.min(drawdown))
    
    def calculate_consecutive_wins(self, returns: List[float]) -> int:
        """Calculate current consecutive wins"""
        consecutive = 0
        for pnl in reversed(returns):
            if pnl > 0:
                consecutive += 1
            else:
                break
        return consecutive
    
    def calculate_consecutive_losses(self, returns: List[float]) -> int:
        """Calculate current consecutive losses"""
        consecutive = 0
        for pnl in reversed(returns):
            if pnl < 0:
                consecutive += 1
            else:
                break
        return consecutive
    
    def calculate_volatility_adjusted_return(self, returns: List[float]) -> float:
        """Calculate volatility-adjusted returns"""
        if not returns or len(returns) < 2:
            return 0.0
        
        import numpy as np
        return float(np.mean(returns) / np.std(returns)) if np.std(returns) > 0 else 0.0
    
    def calculate_kelly_criterion(self, wins: List[float], losses: List[float]) -> float:
        """Calculate Kelly Criterion for optimal position sizing"""
        if not wins or not losses:
            return 0.0
        
        win_rate = len(wins) / (len(wins) + len(losses))
        avg_win = sum(wins) / len(wins)
        avg_loss = abs(sum(losses) / len(losses))
        
        if avg_loss == 0:
            return 0.0
        
        return win_rate - ((1 - win_rate) / (avg_win / avg_loss))
    
    def calculate_calmar_ratio(self, returns: List[float]) -> float:
        """Calculate Calmar ratio (annual return / max drawdown)"""
        if not returns:
            return 0.0
        
        annual_return = sum(returns) * 252  # Assuming daily returns
        max_dd = abs(self.calculate_max_drawdown(returns))
        
        if max_dd == 0:
            return float('inf')
        
        return annual_return / max_dd
    
    def start_background_analytics(self):
        """Start background thread for continuous analytics"""
        def analytics_worker():
            while True:
                try:
                    self.run_analytics_cycle()
                    time.sleep(60)  # Run every minute
                except Exception as e:
                    self.log_error("background_analytics", str(e), "LOW")
                    time.sleep(60)
        
        analytics_thread = threading.Thread(target=analytics_worker, daemon=True)
        analytics_thread.start()
    
    def run_analytics_cycle(self):
        """Run analytics and generate insights"""
        try:
            # Analyze error patterns
            self.analyze_error_patterns()
            
            # Generate performance insights
            self.generate_performance_insights()
            
            # Check for anomalies
            self.detect_anomalies()
            
        except Exception as e:
            self.log_error("analytics_cycle", str(e), "LOW")
    
    def analyze_error_patterns(self):
        """Analyze error patterns for system improvement"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Get error frequency by component
                errors = conn.execute('''
                    SELECT component, COUNT(*) as error_count, severity
                    FROM error_logs 
                    WHERE timestamp > ? 
                    GROUP BY component, severity
                    ORDER BY error_count DESC
                ''', (time.time() - 3600,)).fetchall()  # Last hour
                
                # Log high-frequency errors
                for component, count, severity in errors:
                    if count > 10:  # More than 10 errors in an hour
                        self.logger.warning(
                            "high_error_frequency",
                            component=component,
                            error_count=count,
                            severity=severity,
                            timeframe="1h"
                        )
        except Exception as e:
            self.log_error("error_pattern_analysis", str(e), "LOW")
    
    def generate_performance_insights(self):
        """Generate insights from performance data"""
        try:
            # Get top performing agents/instruments
            with sqlite3.connect(self.db_path) as conn:
                top_performers = conn.execute('''
                    SELECT agent_id, instrument, profit_factor, win_rate, sharpe_ratio
                    FROM performance_logs
                    WHERE timestamp > ?
                    ORDER BY profit_factor DESC
                    LIMIT 5
                ''', (time.time() - 86400,)).fetchall()  # Last 24 hours
                
                for agent, instrument, pf, wr, sr in top_performers:
                    self.logger.info(
                        "top_performer",
                        agent_id=agent,
                        instrument=instrument,
                        profit_factor=pf,
                        win_rate=wr,
                        sharpe_ratio=sr
                    )
        except Exception as e:
            self.log_error("performance_insights", str(e), "LOW")
    
    def detect_anomalies(self):
        """Detect performance anomalies"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Check for sudden performance drops
                recent_performance = conn.execute('''
                    SELECT agent_id, instrument, win_rate, profit_factor
                    FROM performance_logs
                    WHERE timestamp > ?
                    ORDER BY timestamp DESC
                ''', (time.time() - 3600,)).fetchall()
                
                for agent, instrument, wr, pf in recent_performance:
                    if wr < 0.3 or pf < 0.5:  # Poor performance thresholds
                        self.logger.warning(
                            "performance_anomaly",
                            agent_id=agent,
                            instrument=instrument,
                            win_rate=wr,
                            profit_factor=pf,
                            alert="poor_performance"
                        )
        except Exception as e:
            self.log_error("anomaly_detection", str(e), "LOW")
    
    def get_analytics_summary(self) -> Dict[str, Any]:
        """Get comprehensive analytics summary"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Trade statistics
                trade_stats = conn.execute('''
                    SELECT COUNT(*) as total_trades,
                           SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as winning_trades,
                           SUM(pnl) as total_pnl,
                           AVG(pnl) as avg_pnl
                    FROM trade_logs
                    WHERE timestamp > ?
                ''', (time.time() - 86400,)).fetchone()
                
                # Error statistics
                error_stats = conn.execute('''
                    SELECT COUNT(*) as total_errors,
                           COUNT(CASE WHEN severity = 'CRITICAL' THEN 1 END) as critical_errors,
                           COUNT(CASE WHEN resolved = 1 THEN 1 END) as resolved_errors
                    FROM error_logs
                    WHERE timestamp > ?
                ''', (time.time() - 86400,)).fetchone()
                
                return {
                    "trade_statistics": {
                        "total_trades": trade_stats[0],
                        "winning_trades": trade_stats[1],
                        "win_rate": trade_stats[1] / trade_stats[0] if trade_stats[0] > 0 else 0,
                        "total_pnl": trade_stats[2],
                        "avg_pnl": trade_stats[3]
                    },
                    "error_statistics": {
                        "total_errors": error_stats[0],
                        "critical_errors": error_stats[1],
                        "resolved_errors": error_stats[2],
                        "resolution_rate": error_stats[2] / error_stats[0] if error_stats[0] > 0 else 0
                    },
                    "system_health": {
                        "timestamp": datetime.now().isoformat(),
                        "performance_cache_size": len(self.performance_cache),
                        "error_components": len(self.error_counts)
                    }
                }
        except Exception as e:
            self.log_error("analytics_summary", str(e), "MEDIUM")
            return {"status": "error", "message": str(e)}


# Example usage and testing
if __name__ == "__main__":
    logger = ComprehensiveLogger()
    
    # Test trade logging
    trade_log = TradeLog(
        timestamp=time.time(),
        agent_id="BERSERKER_PRIME",
        instrument="BTCUSD+",
        timeframe="M15",
        action="BUY",
        entry_price=95000.0,
        exit_price=95500.0,
        position_size=0.1,
        pnl=50.0,
        mae=-25.0,
        mfe=75.0,
        trade_duration=900.0,
        spread_cost=2.0,
        slippage=1.0,
        market_conditions={"volatility": "high", "trend": "bullish"},
        confidence_score=0.85,
        risk_reward_ratio=2.0,
        win_loss="WIN",
        trade_id="TRADE_001",
        virtual_trade=True
    )
    
    logger.log_trade(trade_log)
    print("✅ Comprehensive logging system initialized and tested")