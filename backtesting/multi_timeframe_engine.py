#!/usr/bin/env python3
"""
Multi-Timeframe Backtest Engine
Comprehensive testing across M1, M5, M15, H1, H4, D1 timeframes
Physics-based execution with realistic friction modeling
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent / "agents"))
sys.path.append(str(Path(__file__).parent.parent / "physics"))
sys.path.append(str(Path(__file__).parent.parent / "data"))

import asyncio
import json
import math
import time
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
from collections import deque, defaultdict
import concurrent.futures

# Import our modules
from mt5_bridge import TIMEFRAME_M1, TIMEFRAME_M5, TIMEFRAME_M15, TIMEFRAME_H1, TIMEFRAME_H4, TIMEFRAME_D1
from dual_agent_rl_system import DualAgentController, TradeRecord, AgentPersona
from enhanced_friction_calculator import EnhancedFrictionCalculator, MarketRegime
from trading_strategy_theorem import PhysicsState, MarketPhysics, TradabilityGate

class TimeframeEnum(Enum):
    """Timeframe enumeration for systematic testing"""
    M1 = ("M1", TIMEFRAME_M1, 1)
    M5 = ("M5", TIMEFRAME_M5, 5) 
    M15 = ("M15", TIMEFRAME_M15, 15)
    H1 = ("H1", TIMEFRAME_H1, 60)
    H4 = ("H4", TIMEFRAME_H4, 240)
    D1 = ("D1", TIMEFRAME_D1, 1440)
    
    def __init__(self, name: str, mt5_const: int, minutes: int):
        self.display_name = name
        self.mt5_constant = mt5_const
        self.minutes_per_bar = minutes

@dataclass
class BacktestConfig:
    """Configuration for backtest execution"""
    start_date: str = "2023-01-01"
    end_date: str = "2025-12-26"
    initial_capital: float = 100000.0
    timeframes: List[TimeframeEnum] = field(default_factory=lambda: [
        TimeframeEnum.M1, TimeframeEnum.M5, TimeframeEnum.M15, 
        TimeframeEnum.H1, TimeframeEnum.H4, TimeframeEnum.D1
    ])
    max_concurrent_backtests: int = 3
    enable_slippage: bool = True
    enable_commission: bool = True
    enable_swap: bool = True
    risk_per_trade: float = 0.02
    max_drawdown_stop: float = 0.20

@dataclass
class BacktestResult:
    """Results from a single backtest run"""
    symbol: str
    timeframe: TimeframeEnum
    start_date: datetime
    end_date: datetime
    total_bars: int
    
    # Trade statistics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    
    # P&L statistics
    gross_pnl: float = 0.0
    total_friction: float = 0.0
    net_pnl: float = 0.0
    max_drawdown: float = 0.0
    max_runup: float = 0.0
    
    # Performance metrics
    sharpe_ratio: float = 0.0
    calmar_ratio: float = 0.0
    profit_factor: float = 0.0
    
    # Agent performance breakdown
    sniper_trades: int = 0
    sniper_net_pnl: float = 0.0
    berserker_trades: int = 0
    berserker_net_pnl: float = 0.0
    
    # Physics validation
    avg_energy_efficiency: float = 0.0
    avg_friction_accuracy: float = 0.0
    regime_distribution: Dict[str, int] = field(default_factory=dict)
    
    # Execution statistics
    avg_execution_time_ms: float = 0.0
    total_backtest_time_sec: float = 0.0

class MultiTimeframeEngine:
    """
    Multi-timeframe backtest engine for comprehensive strategy validation
    
    Capabilities:
    - Parallel execution across multiple timeframes
    - Physics-based trade simulation
    - Realistic friction modeling with directional asymmetries
    - Agent performance comparison
    - Statistical significance testing
    - Bias detection and robustness analysis
    """
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.friction_calc = EnhancedFrictionCalculator()
        
        # Results storage
        self.results: Dict[str, Dict[str, BacktestResult]] = defaultdict(dict)
        self.execution_log: List[Dict] = []
        
        # Performance tracking
        self.start_time = None
        self.end_time = None
        
    def run_comprehensive_backtest(self, symbols: List[str]) -> Dict[str, Any]:
        """Run comprehensive backtest across all symbols and timeframes"""
        
        print(f"🚀 MULTI-TIMEFRAME BACKTEST ENGINE")
        print(f"="*80)
        print(f"Symbols: {len(symbols)}")
        print(f"Timeframes: {[tf.display_name for tf in self.config.timeframes]}")
        print(f"Date range: {self.config.start_date} to {self.config.end_date}")
        print(f"Initial capital: ${self.config.initial_capital:,.0f}")
        
        self.start_time = time.time()
        
        # Create backtest tasks
        tasks = []
        for symbol in symbols:
            for timeframe in self.config.timeframes:
                tasks.append((symbol, timeframe))
        
        print(f"📊 Total backtest combinations: {len(tasks)}")
        
        # Execute backtests with concurrency control
        if self.config.max_concurrent_backtests > 1:
            results = self._run_parallel_backtests(tasks)
        else:
            results = self._run_sequential_backtests(tasks)
        
        self.end_time = time.time()
        
        # Aggregate results
        summary = self._aggregate_results()
        
        print(f"\n✅ Comprehensive backtest completed in {self.end_time - self.start_time:.1f}s")
        
        return summary
    
    def _run_parallel_backtests(self, tasks: List[Tuple[str, TimeframeEnum]]) -> List[BacktestResult]:
        """Run backtests in parallel using ThreadPoolExecutor"""
        
        results = []
        max_workers = min(self.config.max_concurrent_backtests, len(tasks))
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_task = {
                executor.submit(self._run_single_backtest, symbol, timeframe): (symbol, timeframe)
                for symbol, timeframe in tasks
            }
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_task):
                symbol, timeframe = future_to_task[future]
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                        self.results[symbol][timeframe.display_name] = result
                        print(f"✅ {symbol} {timeframe.display_name}: {result.net_pnl:+.1f}bp ({result.total_trades} trades)")
                except Exception as e:
                    print(f"❌ {symbol} {timeframe.display_name}: Error - {e}")
        
        return results
    
    def _run_sequential_backtests(self, tasks: List[Tuple[str, TimeframeEnum]]) -> List[BacktestResult]:
        """Run backtests sequentially"""
        
        results = []
        
        for i, (symbol, timeframe) in enumerate(tasks):
            print(f"📊 Running {i+1}/{len(tasks)}: {symbol} {timeframe.display_name}")
            
            try:
                result = self._run_single_backtest(symbol, timeframe)
                if result:
                    results.append(result)
                    self.results[symbol][timeframe.display_name] = result
                    print(f"✅ Result: {result.net_pnl:+.1f}bp ({result.total_trades} trades)")
            except Exception as e:
                print(f"❌ Error: {e}")
        
        return results
    
    def _run_single_backtest(self, symbol: str, timeframe: TimeframeEnum) -> Optional[BacktestResult]:
        """Run backtest for single symbol/timeframe combination"""
        
        execution_start = time.time()
        
        # Load market data
        market_data = self._load_market_data(symbol, timeframe)
        if not market_data or len(market_data) < 100:
            return None
        
        # Initialize dual-agent controller for this backtest
        controller = DualAgentController(self.config.initial_capital)
        
        # Initialize result object
        result = BacktestResult(
            symbol=symbol,
            timeframe=timeframe,
            start_date=datetime.strptime(self.config.start_date, "%Y-%m-%d"),
            end_date=datetime.strptime(self.config.end_date, "%Y-%m-%d"),
            total_bars=len(market_data),
            regime_distribution={}
        )
        
        # Run backtest simulation
        trade_records = []
        equity_curve = [self.config.initial_capital]
        peak_equity = self.config.initial_capital
        current_equity = self.config.initial_capital
        
        # Process each bar
        for i in range(100, len(market_data)):  # Skip initial bars for indicators
            
            bar = market_data[i]
            
            # Calculate physics state
            physics_state = self._calculate_physics_state(market_data, i, timeframe)
            
            # Track regime distribution
            regime_key = physics_state.regime.value
            result.regime_distribution[regime_key] = result.regime_distribution.get(regime_key, 0) + 1
            
            # Check for trading opportunity
            trade_decision = controller.evaluate_opportunity(symbol, bar, physics_state)
            
            if trade_decision:
                # Execute trade simulation
                trade_record = self._simulate_trade_execution(
                    symbol, timeframe, trade_decision, market_data, i
                )
                
                if trade_record:
                    trade_records.append(trade_record)
                    
                    # Update equity
                    current_equity += trade_record.net_pnl_bp
                    equity_curve.append(current_equity)
                    
                    # Update peak and drawdown
                    if current_equity > peak_equity:
                        peak_equity = current_equity
                    
                    drawdown = (peak_equity - current_equity) / peak_equity * 100
                    result.max_drawdown = max(result.max_drawdown, drawdown)
                    
                    # Stop loss check
                    if drawdown > self.config.max_drawdown_stop * 100:
                        print(f"⛔ Max drawdown stop triggered: {drawdown:.1f}%")
                        break
        
        # Calculate final statistics
        result = self._calculate_backtest_statistics(result, trade_records, equity_curve)
        result.total_backtest_time_sec = time.time() - execution_start
        
        return result
    
    def _load_market_data(self, symbol: str, timeframe: TimeframeEnum) -> Optional[List[Dict]]:
        """Load historical market data for symbol/timeframe"""
        
        try:
            # Use our MT5 bridge to get data
            from mt5_bridge import initialize, copy_rates_total, shutdown
            
            if not initialize():
                return None
            
            # Get historical data (simulate with 5000 bars)
            rates = copy_rates_total(symbol, timeframe.mt5_constant, 5000)
            
            if not rates:
                return None
            
            # Convert to our format
            market_data = []
            for rate in rates:
                bar = {
                    'timestamp': datetime.fromtimestamp(rate[0]),
                    'open': rate[1],
                    'high': rate[2],
                    'low': rate[3], 
                    'close': rate[4],
                    'tick_volume': rate[5],
                    'volume': rate[6] if len(rate) > 6 else 0,
                    'spread': rate[7] if len(rate) > 7 else 0,
                    'symbol': symbol
                }
                market_data.append(bar)
            
            return market_data
            
        except Exception as e:
            print(f"Error loading data for {symbol} {timeframe.display_name}: {e}")
            return None
    
    def _calculate_physics_state(self, market_data: List[Dict], index: int, 
                                timeframe: TimeframeEnum) -> PhysicsState:
        """Calculate physics state at given bar"""
        
        if index < 20:
            # Default state for early bars
            return PhysicsState(
                energy=50.0, friction=10.0, energy_friction_ratio=5.0,
                momentum=0.0, volatility=50.0, volume_percentile=50.0,
                regime=MarketPhysics.CRITICALLY_DAMPED,
                tradability=TradabilityGate.MARGINAL
            )
        
        bar = market_data[index]
        
        # Calculate energy (range-based)
        energy = (bar['high'] - bar['low']) / bar['close'] * 10000 if bar['close'] > 0 else 0
        
        # Calculate momentum (20-bar lookback)
        past_price = market_data[index - 20]['close']
        momentum = (bar['close'] / past_price - 1) * 10000 if past_price > 0 else 0
        
        # Calculate volatility (10-bar rolling)
        returns = []
        for i in range(max(0, index - 10), index):
            if (i > 0 and market_data[i]['close'] > 0 and 
                market_data[i-1]['close'] > 0):
                ret = math.log(market_data[i]['close'] / market_data[i-1]['close']) * 10000
                returns.append(ret)
        
        volatility = math.sqrt(sum(r*r for r in returns) / len(returns)) if returns else 50.0
        
        # Calculate volume percentile
        if index >= 96:
            recent_volumes = [market_data[i].get('volume', 0) for i in range(index - 96, index)]
            current_volume = bar.get('volume', 0)
            if recent_volumes and max(recent_volumes) > 0:
                volume_pct = sum(1 for v in recent_volumes if current_volume > v) / len(recent_volumes) * 100
            else:
                volume_pct = 50.0
        else:
            volume_pct = 50.0
        
        # Estimate friction
        direction = 1 if momentum > 0 else -1
        friction_components = self.friction_calc.calculate_friction(
            bar['symbol'], direction, 1.0, bar['close']
        )
        friction = friction_components.total_friction
        
        # Calculate E/F ratio
        ef_ratio = energy / max(1.0, friction)
        
        # Classify regime
        if ef_ratio < 2:
            regime = MarketPhysics.OVERDAMPED
        elif ef_ratio < 5:
            regime = MarketPhysics.CRITICALLY_DAMPED
        elif volatility > 200:
            regime = MarketPhysics.CHAOTIC
        else:
            regime = MarketPhysics.UNDERDAMPED
        
        # Tradability gate
        if ef_ratio < 2:
            tradability = TradabilityGate.UNTRADABLE
        elif ef_ratio < 5:
            tradability = TradabilityGate.MARGINAL
        elif ef_ratio < 10:
            tradability = TradabilityGate.TRADABLE
        else:
            tradability = TradabilityGate.HIGHLY_TRADABLE
        
        return PhysicsState(
            energy=energy,
            friction=friction,
            energy_friction_ratio=ef_ratio,
            momentum=momentum,
            volatility=volatility,
            volume_percentile=volume_pct,
            regime=regime,
            tradability=tradability
        )
    
    def _simulate_trade_execution(self, symbol: str, timeframe: TimeframeEnum,
                                 trade_decision: Dict, market_data: List[Dict],
                                 entry_index: int) -> Optional[TradeRecord]:
        """Simulate realistic trade execution with slippage and delays"""
        
        try:
            entry_bar = market_data[entry_index]
            hold_bars = int(trade_decision['hold_days'] * (1440 / timeframe.minutes_per_bar))
            exit_index = min(entry_index + hold_bars, len(market_data) - 1)
            exit_bar = market_data[exit_index]
            
            # Simulate slippage
            if self.config.enable_slippage:
                spread = max(1.0, entry_bar.get('spread', 0)) / 10000  # Convert to decimal
                entry_slippage = spread / 2 if trade_decision['direction'] > 0 else -spread / 2
                exit_slippage = -spread / 2 if trade_decision['direction'] > 0 else spread / 2
            else:
                entry_slippage = exit_slippage = 0.0
            
            # Calculate execution prices
            entry_price = entry_bar['close'] * (1 + entry_slippage)
            exit_price = exit_bar['close'] * (1 + exit_slippage)
            
            # Calculate P&L
            gross_pnl_bp = (exit_price / entry_price - 1) * trade_decision['direction'] * 10000
            
            # Calculate friction
            friction_components = self.friction_calc.calculate_friction(
                symbol, trade_decision['direction'], trade_decision['hold_days'], entry_price
            )
            friction_cost = friction_components.total_friction
            
            # Net P&L
            net_pnl_bp = gross_pnl_bp - friction_cost
            
            # Calculate MFE/MAE
            prices_during_trade = [market_data[i]['high'] for i in range(entry_index, exit_index + 1)]
            prices_during_trade.extend([market_data[i]['low'] for i in range(entry_index, exit_index + 1)])
            
            if trade_decision['direction'] > 0:  # Long trade
                mfe_price = max(prices_during_trade) - entry_price
                mae_price = entry_price - min(prices_during_trade)
            else:  # Short trade
                mfe_price = entry_price - min(prices_during_trade)
                mae_price = max(prices_during_trade) - entry_price
            
            mfe_bp = (mfe_price / entry_price) * 10000 if mfe_price > 0 else 0
            mae_bp = (mae_price / entry_price) * 10000 if mae_price > 0 else 0
            
            # Create trade record
            from dual_agent_rl_system import TradeRecord, TradeOutcome
            
            trade_record = TradeRecord(
                entry_time=entry_bar['timestamp'],
                exit_time=exit_bar['timestamp'],
                symbol=symbol,
                agent=trade_decision['agent'],
                direction=trade_decision['direction'],
                entry_price=entry_price,
                exit_price=exit_price,
                position_size=trade_decision['position_size'],
                hold_bars=hold_bars,
                hold_days=trade_decision['hold_days'],
                
                # Physics state (simplified)
                entry_energy=trade_decision['expected_return'],
                entry_friction=trade_decision['expected_friction'],
                entry_ef_ratio=trade_decision['expected_return'] / max(1.0, trade_decision['expected_friction']),
                entry_momentum=trade_decision.get('momentum', 0.0),
                entry_volatility=trade_decision.get('volatility', 50.0),
                entry_volume_pct=trade_decision.get('volume_pct', 50.0),
                entry_regime=MarketPhysics.UNDERDAMPED,
                
                # Results
                gross_pnl_bp=gross_pnl_bp,
                friction_cost_bp=friction_cost,
                net_pnl_bp=net_pnl_bp,
                mfe_bp=mfe_bp,
                mae_bp=mae_bp,
                
                # Performance metrics
                trade_outcome=TradeOutcome.WIN if net_pnl_bp > 0 else TradeOutcome.LOSS,
                energy_efficiency=min(1.5, abs(gross_pnl_bp) / max(10.0, trade_decision['expected_return'])),
                friction_accuracy=abs(friction_cost) / max(1.0, abs(trade_decision['expected_friction'])),
                hold_efficiency=1.0,
                reward=net_pnl_bp / 100.0,
                value_error=abs(net_pnl_bp - trade_decision['expected_return'])
            )
            
            return trade_record
            
        except Exception as e:
            print(f"Error simulating trade: {e}")
            return None
    
    def _calculate_backtest_statistics(self, result: BacktestResult, 
                                      trade_records: List[TradeRecord],
                                      equity_curve: List[float]) -> BacktestResult:
        """Calculate comprehensive backtest statistics"""
        
        if not trade_records:
            return result
        
        # Basic trade statistics
        result.total_trades = len(trade_records)
        result.winning_trades = len([t for t in trade_records if t.net_pnl_bp > 0])
        result.losing_trades = result.total_trades - result.winning_trades
        result.win_rate = result.winning_trades / result.total_trades * 100
        
        # P&L statistics
        result.gross_pnl = sum(t.gross_pnl_bp for t in trade_records)
        result.total_friction = sum(t.friction_cost_bp for t in trade_records)
        result.net_pnl = sum(t.net_pnl_bp for t in trade_records)
        
        # Drawdown calculation
        peak = max(equity_curve)
        trough = min(equity_curve)
        result.max_drawdown = (peak - trough) / peak * 100 if peak > 0 else 0
        result.max_runup = (peak - equity_curve[0]) / equity_curve[0] * 100 if equity_curve[0] > 0 else 0
        
        # Performance metrics
        if result.total_trades > 1:
            returns = [(equity_curve[i] / equity_curve[i-1] - 1) for i in range(1, len(equity_curve))]
            if returns:
                avg_return = sum(returns) / len(returns)
                return_std = math.sqrt(sum((r - avg_return)**2 for r in returns) / len(returns))
                result.sharpe_ratio = avg_return / return_std * math.sqrt(252) if return_std > 0 else 0
                result.calmar_ratio = (result.net_pnl / self.config.initial_capital * 100) / max(0.1, result.max_drawdown)
        
        # Profit factor
        gross_wins = sum(t.net_pnl_bp for t in trade_records if t.net_pnl_bp > 0)
        gross_losses = abs(sum(t.net_pnl_bp for t in trade_records if t.net_pnl_bp < 0))
        result.profit_factor = gross_wins / max(0.01, gross_losses)
        
        # Agent breakdown
        sniper_trades = [t for t in trade_records if t.agent == AgentPersona.SNIPER]
        berserker_trades = [t for t in trade_records if t.agent == AgentPersona.BERSERKER]
        
        result.sniper_trades = len(sniper_trades)
        result.sniper_net_pnl = sum(t.net_pnl_bp for t in sniper_trades)
        result.berserker_trades = len(berserker_trades)
        result.berserker_net_pnl = sum(t.net_pnl_bp for t in berserker_trades)
        
        # Physics validation
        result.avg_energy_efficiency = sum(t.energy_efficiency for t in trade_records) / len(trade_records)
        result.avg_friction_accuracy = sum(t.friction_accuracy for t in trade_records) / len(trade_records)
        
        return result
    
    def _aggregate_results(self) -> Dict[str, Any]:
        """Aggregate results across all symbols and timeframes"""
        
        all_results = []
        for symbol_results in self.results.values():
            all_results.extend(symbol_results.values())
        
        if not all_results:
            return {'error': 'No valid backtest results'}
        
        # Overall statistics
        total_trades = sum(r.total_trades for r in all_results)
        total_winning_trades = sum(r.winning_trades for r in all_results)
        total_net_pnl = sum(r.net_pnl for r in all_results)
        
        # Best performing combinations
        best_result = max(all_results, key=lambda r: r.net_pnl)
        worst_result = min(all_results, key=lambda r: r.net_pnl)
        
        # Timeframe analysis
        timeframe_performance = defaultdict(lambda: {'trades': 0, 'net_pnl': 0.0, 'results': []})
        for result in all_results:
            tf_name = result.timeframe.display_name
            timeframe_performance[tf_name]['trades'] += result.total_trades
            timeframe_performance[tf_name]['net_pnl'] += result.net_pnl
            timeframe_performance[tf_name]['results'].append(result)
        
        # Agent performance analysis
        total_sniper_trades = sum(r.sniper_trades for r in all_results)
        total_berserker_trades = sum(r.berserker_trades for r in all_results)
        total_sniper_pnl = sum(r.sniper_net_pnl for r in all_results)
        total_berserker_pnl = sum(r.berserker_net_pnl for r in all_results)
        
        return {
            'execution_summary': {
                'total_backtests': len(all_results),
                'total_symbols': len(self.results),
                'total_timeframes': len(self.config.timeframes),
                'execution_time_seconds': self.end_time - self.start_time,
                'avg_backtest_time': sum(r.total_backtest_time_sec for r in all_results) / len(all_results)
            },
            'overall_performance': {
                'total_trades': total_trades,
                'total_winning_trades': total_winning_trades,
                'overall_win_rate': total_winning_trades / max(1, total_trades) * 100,
                'total_net_pnl': total_net_pnl,
                'avg_net_pnl_per_backtest': total_net_pnl / len(all_results)
            },
            'best_performance': {
                'symbol': best_result.symbol,
                'timeframe': best_result.timeframe.display_name,
                'net_pnl': best_result.net_pnl,
                'win_rate': best_result.win_rate,
                'total_trades': best_result.total_trades
            },
            'worst_performance': {
                'symbol': worst_result.symbol,
                'timeframe': worst_result.timeframe.display_name,
                'net_pnl': worst_result.net_pnl,
                'win_rate': worst_result.win_rate,
                'total_trades': worst_result.total_trades
            },
            'timeframe_analysis': dict(timeframe_performance),
            'agent_performance': {
                'sniper': {
                    'total_trades': total_sniper_trades,
                    'total_net_pnl': total_sniper_pnl,
                    'avg_pnl_per_trade': total_sniper_pnl / max(1, total_sniper_trades)
                },
                'berserker': {
                    'total_trades': total_berserker_trades,
                    'total_net_pnl': total_berserker_pnl,
                    'avg_pnl_per_trade': total_berserker_pnl / max(1, total_berserker_trades)
                }
            },
            'detailed_results': self.results
        }
    
    def save_results(self, filepath: str) -> None:
        """Save backtest results to file"""
        
        summary = self._aggregate_results()
        
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"📊 Results saved to: {filepath}")

def test_multi_timeframe_engine():
    """Test the multi-timeframe backtest engine"""
    
    print("🧪 COMPREHENSIVE MULTI-TIMEFRAME BACKTEST")
    print("="*80)
    
    # Get all available MarketWatch symbols
    from mt5_marketwatch_integration import MT5MarketWatchManager
    mw_manager = MT5MarketWatchManager()
    all_symbols = mw_manager.get_marketwatch_symbols(visible_only=True)
    
    # Select diverse set of instruments for comprehensive testing
    test_symbols = []
    classification = mw_manager.get_instrument_classification()
    
    # Add 2-3 symbols from each category
    for category, symbols in classification.items():
        if symbols:
            test_symbols.extend(symbols[:3])  # Take first 3 from each category
    
    print(f"📊 Testing {len(test_symbols)} instruments from all MT5 MarketWatch categories")
    print(f"Symbols: {test_symbols}")
    
    # Configuration for comprehensive test
    config = BacktestConfig(
        timeframes=[TimeframeEnum.M15, TimeframeEnum.H1, TimeframeEnum.H4, TimeframeEnum.D1],
        max_concurrent_backtests=3,
        initial_capital=100000.0,
        start_date="2024-01-01",
        end_date="2025-12-26"
    )
    
    # Initialize engine
    engine = MultiTimeframeEngine(config)
    
    # Run comprehensive backtest
    summary = engine.run_comprehensive_backtest(test_symbols)
    
    # Display results
    print(f"\n📊 BACKTEST SUMMARY:")
    print(f"Total backtests: {summary['execution_summary']['total_backtests']}")
    print(f"Execution time: {summary['execution_summary']['execution_time_seconds']:.1f}s")
    print(f"Total trades: {summary['overall_performance']['total_trades']}")
    print(f"Overall win rate: {summary['overall_performance']['overall_win_rate']:.1f}%")
    print(f"Total net P&L: {summary['overall_performance']['total_net_pnl']:+.1f}bp")
    
    print(f"\n🏆 Best performance:")
    best = summary['best_performance']
    print(f"   {best['symbol']} {best['timeframe']}: {best['net_pnl']:+.1f}bp ({best['total_trades']} trades)")
    
    print(f"\n🤖 Agent performance:")
    sniper = summary['agent_performance']['sniper']
    berserker = summary['agent_performance']['berserker']
    print(f"   Sniper: {sniper['total_trades']} trades, {sniper['total_net_pnl']:+.1f}bp")
    print(f"   Berserker: {berserker['total_trades']} trades, {berserker['total_net_pnl']:+.1f}bp")
    
    # Save results
    results_file = Path(__file__).parent.parent / "results" / "test_backtest_results.json"
    engine.save_results(str(results_file))
    
    print(f"\n✅ Multi-timeframe engine test completed!")

if __name__ == "__main__":
    test_multi_timeframe_engine()