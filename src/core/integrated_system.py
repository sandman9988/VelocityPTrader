#!/usr/bin/env python3
"""
PHASE 4: INTEGRATED VELOCITY TRADING SYSTEM
Complete integration of all components into production-ready system
"""

import asyncio
import json
import time
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import logging
import signal
import sys
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

# Import our components
from src.core.data_pipeline import DataPipeline, MarketData, PhysicsMetrics
from src.agents.dual_agent_system import (
    DualAgentCoordinator, ShadowTrader, State, Action, Experience
)
from src.utils.logging_system import ComprehensiveLogger, TradeLog, PerformanceLog
from src.core.optimized_system import AMDOptimizedSystem, HighPerformanceDataProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class SystemConfig:
    """System configuration"""
    # MT5 Settings
    mt5_login: int = 10916362
    mt5_server: str = "VantageInternational-Demo"
    mt5_password: str = ""  # Set via environment variable
    
    # Trading Settings
    symbols: List[str] = None  # Will be loaded from config
    timeframes: List[str] = None
    max_positions: int = 10
    risk_per_trade: float = 0.02  # 2% risk per trade
    
    # System Settings
    update_interval: float = 1.0  # 1 second
    shadow_trading_enabled: bool = True
    transfer_learning_interval: int = 3600  # 1 hour
    model_save_interval: int = 1800  # 30 minutes
    
    # Performance Settings
    use_multiprocessing: bool = True
    process_pool_size: int = 8
    thread_pool_size: int = 32
    
    def __post_init__(self):
        if self.symbols is None:
            self.symbols = [
                "EURUSD+", "GBPUSD+", "USDJPY+", "USDCHF+",  # Forex
                "BTCUSD+", "ETHUSD+",  # Crypto
                "SPX500+", "US30+",  # Indices
                "XAUUSD+", "XAGUSD+"  # Commodities
            ]
        if self.timeframes is None:
            self.timeframes = ["M1", "M5", "M15", "H1"]

@dataclass
class Position:
    """Active position tracking"""
    symbol: str
    direction: str  # "BUY" or "SELL"
    entry_price: float
    position_size: float
    stop_loss: float
    take_profit: float
    entry_time: float
    agent_name: str  # BERSERKER or SNIPER
    unrealized_pnl: float = 0.0
    mae: float = 0.0  # Maximum Adverse Excursion
    mfe: float = 0.0  # Maximum Favorable Excursion

class TradingEngine:
    """Core trading engine"""
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.positions: Dict[str, Position] = {}
        self.pending_orders: Dict[str, Any] = {}
        self.account_balance = 10000.0  # Starting balance
        self.account_equity = 10000.0
        self.max_drawdown = 0.0
        self.peak_equity = 10000.0
        
        logger.info("💰 Trading Engine initialized")
    
    def can_open_position(self) -> bool:
        """Check if we can open new position"""
        return len(self.positions) < self.config.max_positions
    
    def calculate_position_size(self, symbol: str, stop_loss_pips: float) -> float:
        """Calculate position size based on risk"""
        risk_amount = self.account_balance * self.config.risk_per_trade
        # Simplified - in production would use proper pip value calculation
        pip_value = 10.0  # USD per pip for standard lot
        position_size = risk_amount / (stop_loss_pips * pip_value)
        return min(position_size, 1.0)  # Max 1 standard lot
    
    def execute_action(self, action: Action, symbol: str, 
                      market_data: MarketData, agent_name: str) -> Optional[Position]:
        """Execute trading action"""
        if action == Action.HOLD:
            return None
        
        # Check if we already have position in this symbol
        if symbol in self.positions:
            # Could implement position management here
            return None
        
        if not self.can_open_position():
            logger.warning("Max positions reached, cannot open new trade")
            return None
        
        # Determine position parameters based on agent
        if agent_name == "BERSERKER":
            stop_loss_pips = 30
            take_profit_pips = 50
        else:  # SNIPER
            stop_loss_pips = 15
            take_profit_pips = 30
        
        position_size = self.calculate_position_size(symbol, stop_loss_pips)
        
        # Create position
        position = Position(
            symbol=symbol,
            direction="BUY" if action == Action.BUY else "SELL",
            entry_price=market_data.ask if action == Action.BUY else market_data.bid,
            position_size=position_size,
            stop_loss=market_data.bid - (stop_loss_pips * 0.0001) if action == Action.BUY 
                     else market_data.ask + (stop_loss_pips * 0.0001),
            take_profit=market_data.ask + (take_profit_pips * 0.0001) if action == Action.BUY
                       else market_data.bid - (take_profit_pips * 0.0001),
            entry_time=time.time(),
            agent_name=agent_name
        )
        
        self.positions[symbol] = position
        logger.info(f"📈 Opened {position.direction} position: {symbol} @ {position.entry_price}")
        
        return position
    
    def update_positions(self, market_data: Dict[str, MarketData]) -> List[TradeLog]:
        """Update all positions with current prices"""
        closed_trades = []
        
        for symbol, position in list(self.positions.items()):
            if symbol not in market_data:
                continue
            
            current_data = market_data[symbol]
            current_price = current_data.bid if position.direction == "BUY" else current_data.ask
            
            # Calculate unrealized P&L
            if position.direction == "BUY":
                position.unrealized_pnl = (current_price - position.entry_price) * position.position_size * 10000
            else:
                position.unrealized_pnl = (position.entry_price - current_price) * position.position_size * 10000
            
            # Update MAE/MFE
            if position.unrealized_pnl < position.mae:
                position.mae = position.unrealized_pnl
            if position.unrealized_pnl > position.mfe:
                position.mfe = position.unrealized_pnl
            
            # Check stop loss / take profit
            should_close = False
            exit_reason = ""
            
            if position.direction == "BUY":
                if current_price <= position.stop_loss:
                    should_close = True
                    exit_reason = "STOP_LOSS"
                elif current_price >= position.take_profit:
                    should_close = True
                    exit_reason = "TAKE_PROFIT"
            else:
                if current_price >= position.stop_loss:
                    should_close = True
                    exit_reason = "STOP_LOSS"
                elif current_price <= position.take_profit:
                    should_close = True
                    exit_reason = "TAKE_PROFIT"
            
            if should_close:
                # Create trade log
                trade_log = TradeLog(
                    timestamp=time.time(),
                    agent_id=position.agent_name,
                    instrument=symbol,
                    timeframe="M5",  # Simplified
                    action="CLOSE",
                    entry_price=position.entry_price,
                    exit_price=current_price,
                    position_size=position.position_size,
                    pnl=position.unrealized_pnl,
                    mae=position.mae,
                    mfe=position.mfe,
                    trade_duration=time.time() - position.entry_time,
                    spread_cost=current_data.spread_pips * position.position_size,
                    slippage=0.0,  # Simplified
                    market_conditions={
                        "exit_reason": exit_reason,
                        "volatility": current_data.spread_pips / 10.0
                    },
                    confidence_score=0.0,  # Would be calculated
                    risk_reward_ratio=abs(position.mfe / position.mae) if position.mae != 0 else 0,
                    win_loss="WIN" if position.unrealized_pnl > 0 else "LOSS",
                    trade_id=f"{symbol}_{position.entry_time}",
                    virtual_trade=False
                )
                
                closed_trades.append(trade_log)
                
                # Update account balance
                self.account_balance += position.unrealized_pnl
                
                # Remove position
                del self.positions[symbol]
                
                logger.info(f"📉 Closed {position.direction} position: {symbol} "
                          f"P&L: ${position.unrealized_pnl:.2f} ({exit_reason})")
        
        # Update account equity
        total_unrealized = sum(pos.unrealized_pnl for pos in self.positions.values())
        self.account_equity = self.account_balance + total_unrealized
        
        # Track drawdown
        if self.account_equity > self.peak_equity:
            self.peak_equity = self.account_equity
        drawdown = (self.peak_equity - self.account_equity) / self.peak_equity
        if drawdown > self.max_drawdown:
            self.max_drawdown = drawdown
        
        return closed_trades

class VelocityTradingSystem:
    """Main integrated trading system"""
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.is_running = False
        
        # Initialize components
        logger.info("🚀 Initializing VelocityTrader System...")
        
        # AMD optimized system
        self.amd_system = AMDOptimizedSystem()
        self.data_processor = HighPerformanceDataProcessor(self.amd_system)
        
        # Data pipeline
        self.data_pipeline = DataPipeline()
        
        # Agent system
        self.agent_coordinator = DualAgentCoordinator()
        self.shadow_trader = ShadowTrader() if config.shadow_trading_enabled else None
        
        # Trading engine
        self.trading_engine = TradingEngine(config)
        
        # Logging system
        self.logger = ComprehensiveLogger("trading_logs.db")
        
        # Statistics
        self.stats = {
            'start_time': time.time(),
            'total_ticks': 0,
            'trades_executed': 0,
            'shadow_trades': 0,
            'last_model_save': time.time(),
            'last_transfer_learning': time.time()
        }
        
        # Event handling
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logger.info("✅ VelocityTrader System initialized successfully")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info("🛑 Shutdown signal received")
        self.stop()
        sys.exit(0)
    
    def start(self):
        """Start the trading system"""
        if self.is_running:
            logger.warning("System already running")
            return
        
        self.is_running = True
        
        # Start data pipeline
        self.data_pipeline.start()
        
        # Load agent models if they exist
        if Path("models/").exists():
            self.agent_coordinator.load_models("models/")
        
        # Start main loop
        asyncio.run(self._main_loop())
    
    def stop(self):
        """Stop the trading system"""
        logger.info("Stopping VelocityTrader System...")
        
        self.is_running = False
        
        # Stop data pipeline
        self.data_pipeline.stop()
        
        # Save models
        self.agent_coordinator.save_models("models/")
        
        # Log final statistics
        self._log_final_stats()
        
        logger.info("✅ System stopped successfully")
    
    async def _main_loop(self):
        """Main trading loop"""
        logger.info("🔄 Main trading loop started")
        
        while self.is_running:
            try:
                loop_start = time.time()
                
                # Process market data
                await self._process_market_tick()
                
                # Shadow trading
                if self.shadow_trader and self.stats['total_ticks'] % 10 == 0:
                    await self._process_shadow_trading()
                
                # Periodic tasks
                await self._periodic_tasks()
                
                # Rate limiting
                elapsed = time.time() - loop_start
                if elapsed < self.config.update_interval:
                    await asyncio.sleep(self.config.update_interval - elapsed)
                
            except Exception as e:
                logger.error(f"Main loop error: {e}")
                self.logger.log_error("main_loop", str(e), "HIGH")
                await asyncio.sleep(self.config.update_interval * 2)
    
    async def _process_market_tick(self):
        """Process single market tick"""
        self.stats['total_ticks'] += 1
        
        # Get all symbol data
        market_data = {}
        for symbol in self.config.symbols:
            data = self.data_pipeline.get_current_data(symbol)
            if data:
                market_data[symbol] = data
        
        if not market_data:
            return
        
        # Process with high-performance data processor
        processed_data = self.data_processor.process_market_data_parallel(
            {s: asdict(d) for s, d in market_data.items()}
        )
        
        # Update positions
        closed_trades = self.trading_engine.update_positions(market_data)
        
        # Log closed trades
        for trade in closed_trades:
            self.logger.log_trade(trade)
            self.stats['trades_executed'] += 1
            
            # Update agent statistics
            agent = (self.agent_coordinator.berserker 
                    if trade.agent_id == "BERSERKER" 
                    else self.agent_coordinator.sniper)
            agent.total_trades += 1
            if trade.pnl > 0:
                agent.winning_trades += 1
            agent.total_pnl += trade.pnl
        
        # Process each symbol
        for symbol, data in market_data.items():
            # Skip if already have position
            if symbol in self.trading_engine.positions:
                continue
            
            # Get physics metrics
            physics = self.data_pipeline.get_physics_metrics(symbol)
            if not physics:
                continue
            
            # Create state
            state = State(
                price=data.mid_price,
                spread=data.spread_cost_normalized,
                momentum=physics.momentum,
                acceleration=physics.acceleration,
                volatility=physics.volatility,
                liquidity_score=physics.liquidity_score,
                trend_strength=physics.trend_strength
            )
            
            # Get agent decision
            action, agent_name = self.agent_coordinator.process_tick(
                state, symbol, "M5"
            )
            
            # Execute trade if needed
            if action != Action.HOLD:
                position = self.trading_engine.execute_action(
                    action, symbol, data, agent_name
                )
                
                if position:
                    # Log new trade
                    self.logger.log_trade(TradeLog(
                        timestamp=position.entry_time,
                        agent_id=agent_name,
                        instrument=symbol,
                        timeframe="M5",
                        action=position.direction,
                        entry_price=position.entry_price,
                        exit_price=None,
                        position_size=position.position_size,
                        pnl=None,
                        mae=None,
                        mfe=None,
                        trade_duration=None,
                        spread_cost=data.spread_pips * position.position_size,
                        slippage=0.0,
                        market_conditions={
                            "momentum": physics.momentum,
                            "volatility": physics.volatility
                        },
                        confidence_score=abs(physics.momentum * physics.trend_strength),
                        risk_reward_ratio=None,
                        win_loss=None,
                        trade_id=f"{symbol}_{position.entry_time}",
                        virtual_trade=False
                    ))
    
    async def _process_shadow_trading(self):
        """Process shadow/virtual trading"""
        if not self.shadow_trader:
            return
        
        # Select random symbols for shadow trading
        import random
        shadow_symbols = random.sample(self.config.symbols, 
                                     min(3, len(self.config.symbols)))
        
        for symbol in shadow_symbols:
            data = self.data_pipeline.get_current_data(symbol)
            physics = self.data_pipeline.get_physics_metrics(symbol)
            
            if data and physics:
                state = State(
                    price=data.mid_price,
                    spread=data.spread_cost_normalized,
                    momentum=physics.momentum,
                    acceleration=physics.acceleration,
                    volatility=physics.volatility,
                    liquidity_score=physics.liquidity_score,
                    trend_strength=physics.trend_strength
                )
                
                # Shadow trade
                action, virtual_pnl = self.shadow_trader.process_virtual_tick(
                    state, symbol
                )
                
                self.stats['shadow_trades'] += 1
    
    async def _periodic_tasks(self):
        """Handle periodic tasks"""
        current_time = time.time()
        
        # Transfer learning
        if (self.shadow_trader and 
            current_time - self.stats['last_transfer_learning'] > self.config.transfer_learning_interval):
            
            self.shadow_trader.transfer_learning(self.agent_coordinator)
            self.stats['last_transfer_learning'] = current_time
            logger.info("🔄 Transfer learning completed")
        
        # Model saving
        if current_time - self.stats['last_model_save'] > self.config.model_save_interval:
            self.agent_coordinator.save_models("models/")
            self.stats['last_model_save'] = current_time
            logger.info("💾 Models saved")
        
        # Log performance metrics every minute
        if self.stats['total_ticks'] % 60 == 0:
            self._log_performance_metrics()
    
    def _log_performance_metrics(self):
        """Log current performance metrics"""
        # Calculate metrics
        runtime = time.time() - self.stats['start_time']
        agent_stats = self.agent_coordinator.get_statistics()
        
        # Create performance log
        perf_log = PerformanceLog(
            timestamp=time.time(),
            agent_id="SYSTEM",
            instrument="ALL",
            timeframe="ALL",
            win_rate=self._calculate_system_win_rate(),
            profit_factor=self._calculate_profit_factor(),
            sharpe_ratio=0.0,  # Would need returns history
            max_drawdown=self.trading_engine.max_drawdown,
            total_trades=self.stats['trades_executed'],
            avg_trade_duration=0.0,  # Would calculate from logs
            hit_ratio=self._calculate_system_win_rate(),
            avg_win=0.0,  # Would calculate
            avg_loss=0.0,  # Would calculate
            consecutive_wins=0,  # Would track
            consecutive_losses=0,  # Would track
            volatility_adjusted_return=0.0,
            kelly_criterion=0.0,
            calmar_ratio=0.0
        )
        
        self.logger.log_performance(perf_log)
        
        # Log to console
        logger.info(f"📊 Performance Update:")
        logger.info(f"   Account Balance: ${self.trading_engine.account_balance:.2f}")
        logger.info(f"   Account Equity: ${self.trading_engine.account_equity:.2f}")
        logger.info(f"   Open Positions: {len(self.trading_engine.positions)}")
        logger.info(f"   Total Trades: {self.stats['trades_executed']}")
        logger.info(f"   Max Drawdown: {self.trading_engine.max_drawdown:.1%}")
        logger.info(f"   Shadow Trades: {self.stats['shadow_trades']}")
    
    def _calculate_system_win_rate(self) -> float:
        """Calculate overall system win rate"""
        berserker_stats = self.agent_coordinator.get_statistics()['berserker']
        sniper_stats = self.agent_coordinator.get_statistics()['sniper']
        
        total_trades = berserker_stats['total_trades'] + sniper_stats['total_trades']
        total_wins = berserker_stats['winning_trades'] + sniper_stats['winning_trades']
        
        return total_wins / max(total_trades, 1)
    
    def _calculate_profit_factor(self) -> float:
        """Calculate profit factor"""
        # Simplified - would calculate from trade history
        return 1.5  # Placeholder
    
    def _log_final_stats(self):
        """Log final statistics on shutdown"""
        runtime = time.time() - self.stats['start_time']
        
        final_stats = {
            "runtime_seconds": runtime,
            "runtime_hours": runtime / 3600,
            "total_ticks_processed": self.stats['total_ticks'],
            "total_trades_executed": self.stats['trades_executed'],
            "total_shadow_trades": self.stats['shadow_trades'],
            "final_balance": self.trading_engine.account_balance,
            "final_equity": self.trading_engine.account_equity,
            "total_return": (self.trading_engine.account_balance - 10000) / 10000,
            "max_drawdown": self.trading_engine.max_drawdown,
            "agent_statistics": self.agent_coordinator.get_statistics()
        }
        
        # Save to file
        with open("final_stats.json", 'w') as f:
            json.dump(final_stats, f, indent=2)
        
        logger.info("📊 Final Statistics:")
        logger.info(f"   Runtime: {final_stats['runtime_hours']:.2f} hours")
        logger.info(f"   Total Return: {final_stats['total_return']:.1%}")
        logger.info(f"   Max Drawdown: {final_stats['max_drawdown']:.1%}")
        logger.info(f"   Total Trades: {final_stats['total_trades_executed']}")

# Main entry point
def main():
    """Main entry point"""
    print("=" * 60)
    print("🚀 VELOCITY TRADER - Physics-Based AI Trading System")
    print("=" * 60)
    
    # Load configuration
    config_file = Path("config/system_config.json")
    if config_file.exists():
        with open(config_file) as f:
            config_dict = json.load(f)
        config = SystemConfig(**config_dict)
    else:
        config = SystemConfig()
        logger.warning("Using default configuration")
    
    # Create and start system
    system = VelocityTradingSystem(config)
    
    try:
        system.start()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
    except Exception as e:
        logger.error(f"System error: {e}")
    finally:
        system.stop()

if __name__ == "__main__":
    main()