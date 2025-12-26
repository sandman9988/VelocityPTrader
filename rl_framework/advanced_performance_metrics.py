#!/usr/bin/env python3
"""
Advanced Performance Metrics Calculator
Comprehensive trading performance analysis including:
- Maximum Adverse Excursion (MAE)
- Maximum Favorable Excursion (MFE) 
- Omega Ratio
- Z-Factor
- Efficiency Ratio
- Kelly Criterion
- Ulcer Index
- Martin Ratio
- And many more advanced metrics
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import math
import statistics
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

@dataclass
class TradeMetrics:
    """Individual trade performance metrics"""
    trade_id: str
    entry_time: datetime
    exit_time: datetime
    symbol: str
    direction: int  # 1 for long, -1 for short
    
    # Basic trade data
    entry_price: float
    exit_price: float
    quantity: float
    
    # Price excursions during trade
    highest_price: float  # Best price reached
    lowest_price: float   # Worst price reached
    
    # P&L metrics
    gross_pnl: float
    commission: float
    slippage: float
    net_pnl: float
    
    # Calculated metrics
    mae: float = 0.0  # Maximum Adverse Excursion
    mfe: float = 0.0  # Maximum Favorable Excursion
    efficiency: float = 0.0  # MFE/MAE ratio
    
    # Time metrics
    bars_held: int = 0
    hold_time_minutes: float = 0.0
    
    # Risk metrics
    initial_risk: float = 0.0
    realized_risk: float = 0.0
    
    def __post_init__(self):
        """Calculate derived metrics"""
        self.calculate_excursions()
        self.calculate_efficiency()
        self.calculate_time_metrics()
    
    def calculate_excursions(self):
        """Calculate MAE and MFE"""
        if self.direction == 1:  # Long position
            # MAE: Maximum loss from entry price
            self.mae = max(0, self.entry_price - self.lowest_price)
            # MFE: Maximum gain from entry price  
            self.mfe = max(0, self.highest_price - self.entry_price)
        else:  # Short position
            # MAE: Maximum loss from entry price
            self.mae = max(0, self.highest_price - self.entry_price)
            # MFE: Maximum gain from entry price
            self.mfe = max(0, self.entry_price - self.lowest_price)
    
    def calculate_efficiency(self):
        """Calculate trade efficiency (MFE/MAE ratio)"""
        if self.mae > 0:
            self.efficiency = self.mfe / self.mae
        else:
            self.efficiency = float('inf') if self.mfe > 0 else 1.0
    
    def calculate_time_metrics(self):
        """Calculate time-based metrics"""
        time_diff = self.exit_time - self.entry_time
        self.hold_time_minutes = time_diff.total_seconds() / 60.0

@dataclass 
class InstrumentPerformance:
    """Comprehensive performance metrics for a single instrument"""
    symbol: str
    timeframe: str
    total_trades: int = 0
    
    # Basic performance
    total_pnl: float = 0.0
    gross_pnl: float = 0.0
    total_commission: float = 0.0
    win_rate: float = 0.0
    
    # Trade statistics
    winning_trades: int = 0
    losing_trades: int = 0
    breakeven_trades: int = 0
    
    # P&L statistics
    avg_win: float = 0.0
    avg_loss: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    
    # Advanced ratios
    profit_factor: float = 0.0
    expectancy: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    omega_ratio: float = 0.0
    
    # MAE/MFE analysis
    avg_mae: float = 0.0
    avg_mfe: float = 0.0
    mae_efficiency: float = 0.0  # Avg MFE / Avg MAE
    mfe_realization: float = 0.0  # Realized P&L / MFE
    
    # Risk metrics
    max_drawdown: float = 0.0
    max_runup: float = 0.0
    ulcer_index: float = 0.0
    value_at_risk_95: float = 0.0
    conditional_var_95: float = 0.0
    
    # Consistency metrics
    z_factor: float = 0.0
    k_factor: float = 0.0
    lake_ratio: float = 0.0
    martin_ratio: float = 0.0
    
    # Efficiency metrics
    efficiency_ratio: float = 0.0
    trade_efficiency: float = 0.0
    system_quality_number: float = 0.0
    
    # Kelly criterion
    optimal_f: float = 0.0
    kelly_percentage: float = 0.0
    
    # Time analysis
    avg_hold_time: float = 0.0
    win_hold_time: float = 0.0
    loss_hold_time: float = 0.0
    
    # Trade distribution
    consecutive_wins: int = 0
    consecutive_losses: int = 0
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0

class AdvancedPerformanceCalculator:
    """
    Advanced performance metrics calculator with institutional-grade analytics
    
    Calculates comprehensive performance metrics for trading systems including:
    - Traditional metrics (Sharpe, Sortino, Calmar)
    - MAE/MFE analysis for trade efficiency
    - Omega ratio for complete return distribution analysis
    - Z-Factor for system robustness
    - Kelly Criterion for optimal position sizing
    - Ulcer Index for drawdown pain measurement
    - Martin Ratio for recovery analysis
    """
    
    def __init__(self):
        self.trade_data: Dict[str, List[TradeMetrics]] = {}
        self.performance_cache: Dict[str, InstrumentPerformance] = {}
    
    def add_trade(self, trade: TradeMetrics) -> None:
        """Add a trade for performance analysis"""
        
        if trade.symbol not in self.trade_data:
            self.trade_data[trade.symbol] = []
        
        self.trade_data[trade.symbol].append(trade)
        
        # Invalidate cache for this symbol
        if trade.symbol in self.performance_cache:
            del self.performance_cache[trade.symbol]
    
    def calculate_instrument_performance(self, symbol: str, timeframe: str = "ALL") -> InstrumentPerformance:
        """Calculate comprehensive performance metrics for an instrument"""
        
        cache_key = f"{symbol}_{timeframe}"
        if cache_key in self.performance_cache:
            return self.performance_cache[cache_key]
        
        if symbol not in self.trade_data or not self.trade_data[symbol]:
            return InstrumentPerformance(symbol=symbol, timeframe=timeframe)
        
        trades = self.trade_data[symbol]
        performance = InstrumentPerformance(symbol=symbol, timeframe=timeframe)
        
        # Basic statistics
        performance.total_trades = len(trades)
        performance.total_pnl = sum(t.net_pnl for t in trades)
        performance.gross_pnl = sum(t.gross_pnl for t in trades)
        performance.total_commission = sum(t.commission for t in trades)
        
        # Win/Loss statistics
        winning_trades = [t for t in trades if t.net_pnl > 0]
        losing_trades = [t for t in trades if t.net_pnl < 0]
        breakeven_trades = [t for t in trades if t.net_pnl == 0]
        
        performance.winning_trades = len(winning_trades)
        performance.losing_trades = len(losing_trades)
        performance.breakeven_trades = len(breakeven_trades)
        performance.win_rate = len(winning_trades) / len(trades) * 100 if trades else 0
        
        # P&L statistics
        if winning_trades:
            performance.avg_win = statistics.mean(t.net_pnl for t in winning_trades)
            performance.largest_win = max(t.net_pnl for t in winning_trades)
        
        if losing_trades:
            performance.avg_loss = statistics.mean(t.net_pnl for t in losing_trades)
            performance.largest_loss = min(t.net_pnl for t in losing_trades)
        
        # Calculate advanced metrics
        self._calculate_ratios(performance, trades)
        self._calculate_mae_mfe_metrics(performance, trades)
        self._calculate_risk_metrics(performance, trades)
        self._calculate_consistency_metrics(performance, trades)
        self._calculate_efficiency_metrics(performance, trades)
        self._calculate_kelly_criterion(performance, trades)
        self._calculate_time_metrics(performance, trades)
        self._calculate_sequence_metrics(performance, trades)
        
        # Cache result
        self.performance_cache[cache_key] = performance
        
        return performance
    
    def _calculate_ratios(self, performance: InstrumentPerformance, trades: List[TradeMetrics]) -> None:
        """Calculate basic performance ratios"""
        
        returns = [t.net_pnl for t in trades]
        
        # Expectancy
        performance.expectancy = statistics.mean(returns) if returns else 0
        
        # Profit Factor
        gross_profit = sum(r for r in returns if r > 0)
        gross_loss = abs(sum(r for r in returns if r < 0))
        performance.profit_factor = gross_profit / max(0.01, gross_loss)
        
        # Sharpe Ratio (annualized, assuming daily returns)
        if len(returns) > 1:
            std_dev = statistics.stdev(returns)
            performance.sharpe_ratio = (performance.expectancy / std_dev * math.sqrt(252)) if std_dev > 0 else 0
        
        # Sortino Ratio
        negative_returns = [r for r in returns if r < 0]
        if negative_returns and len(negative_returns) > 1:
            downside_dev = statistics.stdev(negative_returns)
            performance.sortino_ratio = (performance.expectancy / downside_dev * math.sqrt(252)) if downside_dev > 0 else 0
        
        # Omega Ratio (threshold = 0)
        positive_returns = [r for r in returns if r > 0]
        omega_numerator = sum(positive_returns)
        omega_denominator = abs(sum(r for r in returns if r < 0))
        performance.omega_ratio = omega_numerator / max(0.01, omega_denominator)
    
    def _calculate_mae_mfe_metrics(self, performance: InstrumentPerformance, trades: List[TradeMetrics]) -> None:
        """Calculate MAE/MFE efficiency metrics"""
        
        if not trades:
            return
        
        # Average MAE and MFE
        performance.avg_mae = statistics.mean(t.mae for t in trades)
        performance.avg_mfe = statistics.mean(t.mfe for t in trades)
        
        # MAE Efficiency (how well we captured favorable moves vs adverse)
        performance.mae_efficiency = performance.avg_mfe / max(0.01, performance.avg_mae)
        
        # MFE Realization (how much of favorable moves we actually captured)
        total_mfe = sum(t.mfe for t in trades)
        if total_mfe > 0:
            performance.mfe_realization = performance.gross_pnl / total_mfe
        
        # Trade Efficiency (average efficiency across all trades)
        efficiencies = [t.efficiency for t in trades if math.isfinite(t.efficiency)]
        performance.trade_efficiency = statistics.mean(efficiencies) if efficiencies else 0
    
    def _calculate_risk_metrics(self, performance: InstrumentPerformance, trades: List[TradeMetrics]) -> None:
        """Calculate risk and drawdown metrics"""
        
        returns = [t.net_pnl for t in trades]
        
        if not returns:
            return
        
        # Calculate equity curve for drawdown analysis
        equity_curve = []
        running_total = 0
        for ret in returns:
            running_total += ret
            equity_curve.append(running_total)
        
        # Maximum Drawdown and Runup
        peak = equity_curve[0]
        max_dd = 0
        max_ru = 0
        
        for equity in equity_curve:
            if equity > peak:
                peak = equity
                max_ru = max(max_ru, equity - equity_curve[0])
            else:
                drawdown = peak - equity
                max_dd = max(max_dd, drawdown)
        
        performance.max_drawdown = max_dd
        performance.max_runup = max_ru
        
        # Calmar Ratio
        annual_return = performance.expectancy * 252
        performance.calmar_ratio = annual_return / max(0.01, max_dd) if max_dd > 0 else 0
        
        # Ulcer Index (drawdown pain)
        if len(equity_curve) > 1:
            drawdown_squares = []
            peak = equity_curve[0]
            
            for equity in equity_curve:
                peak = max(peak, equity)
                drawdown_pct = ((peak - equity) / max(0.01, abs(peak))) * 100
                drawdown_squares.append(drawdown_pct ** 2)
            
            performance.ulcer_index = math.sqrt(statistics.mean(drawdown_squares))
        
        # Value at Risk (95th percentile)
        sorted_returns = sorted(returns)
        var_index = int(len(sorted_returns) * 0.05)
        performance.value_at_risk_95 = sorted_returns[var_index] if var_index < len(sorted_returns) else 0
        
        # Conditional VaR (average of worst 5%)
        worst_returns = sorted_returns[:var_index + 1]
        performance.conditional_var_95 = statistics.mean(worst_returns) if worst_returns else 0
    
    def _calculate_consistency_metrics(self, performance: InstrumentPerformance, trades: List[TradeMetrics]) -> None:
        """Calculate consistency and robustness metrics"""
        
        returns = [t.net_pnl for t in trades]
        
        if len(returns) < 2:
            return
        
        # Z-Factor (modified for trading)
        # Z-Factor = (Win% - Loss%) / sqrt(Win% * Loss% / N)
        win_rate_decimal = performance.win_rate / 100
        loss_rate_decimal = 1 - win_rate_decimal
        
        if win_rate_decimal > 0 and loss_rate_decimal > 0:
            z_denominator = math.sqrt(win_rate_decimal * loss_rate_decimal / len(returns))
            performance.z_factor = (win_rate_decimal - loss_rate_decimal) / z_denominator
        
        # K-Factor (Kestner's metric for consistency)
        # K = Slope of Equity Curve / Standard Error of Slope
        x_values = list(range(len(returns)))
        equity_curve = []
        running_total = 0
        for ret in returns:
            running_total += ret
            equity_curve.append(running_total)
        
        if len(equity_curve) > 1:
            # Linear regression slope
            n = len(x_values)
            sum_x = sum(x_values)
            sum_y = sum(equity_curve)
            sum_xy = sum(x * y for x, y in zip(x_values, equity_curve))
            sum_x2 = sum(x * x for x in x_values)
            
            slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
            
            # Standard error of slope
            residuals = [equity_curve[i] - (slope * x_values[i]) for i in range(n)]
            residual_sum_squares = sum(r * r for r in residuals)
            slope_variance = residual_sum_squares / ((n - 2) * (sum_x2 - sum_x * sum_x / n))
            slope_std_error = math.sqrt(slope_variance)
            
            performance.k_factor = slope / max(0.01, slope_std_error)
        
        # Lake Ratio (time above previous peak)
        equity_curve_with_peaks = []
        peak = equity_curve[0] if equity_curve else 0
        
        for equity in equity_curve:
            peak = max(peak, equity)
            equity_curve_with_peaks.append((equity, peak))
        
        underwater_periods = sum(1 for equity, peak in equity_curve_with_peaks if equity < peak)
        performance.lake_ratio = underwater_periods / len(equity_curve) if equity_curve else 0
        
        # Martin Ratio (Ulcer Performance Index)
        if performance.ulcer_index > 0:
            annual_return = performance.expectancy * 252
            performance.martin_ratio = annual_return / performance.ulcer_index
    
    def _calculate_efficiency_metrics(self, performance: InstrumentPerformance, trades: List[TradeMetrics]) -> None:
        """Calculate various efficiency metrics"""
        
        if not trades:
            return
        
        returns = [t.net_pnl for t in trades]
        
        # Efficiency Ratio (trend strength)
        if len(returns) > 1:
            price_change = abs(returns[-1] - returns[0])
            total_movement = sum(abs(returns[i] - returns[i-1]) for i in range(1, len(returns)))
            performance.efficiency_ratio = price_change / max(0.01, total_movement)
        
        # System Quality Number (Van Tharp's SQN)
        if len(returns) > 1:
            expectancy = statistics.mean(returns)
            std_dev = statistics.stdev(returns)
            performance.system_quality_number = (math.sqrt(len(returns)) * expectancy) / std_dev if std_dev > 0 else 0
    
    def _calculate_kelly_criterion(self, performance: InstrumentPerformance, trades: List[TradeMetrics]) -> None:
        """Calculate Kelly Criterion for optimal position sizing"""
        
        if performance.winning_trades == 0 or performance.losing_trades == 0:
            return
        
        win_rate = performance.win_rate / 100
        avg_win_ratio = abs(performance.avg_win / performance.avg_loss) if performance.avg_loss != 0 else 1
        
        # Kelly percentage
        performance.kelly_percentage = win_rate - ((1 - win_rate) / avg_win_ratio)
        performance.kelly_percentage = max(0, min(1, performance.kelly_percentage))  # Clamp to [0,1]
        
        # Optimal f (Ralph Vince)
        returns = [t.net_pnl for t in trades if t.symbol == performance.symbol]
        if returns:
            largest_loss = abs(min(returns))
            if largest_loss > 0:
                performance.optimal_f = self._calculate_optimal_f(returns, largest_loss)
    
    def _calculate_optimal_f(self, returns: List[float], largest_loss: float) -> float:
        """Calculate optimal f using geometric mean maximization"""
        
        best_f = 0
        best_geom_mean = 0
        
        # Test f values from 0 to 1 in increments
        for f_test in [i/100.0 for i in range(0, 101, 5)]:
            if f_test == 0:
                continue
            
            # Calculate geometric mean of returns for this f
            geom_returns = []
            for ret in returns:
                # Normalize return by largest loss and apply f
                normalized_return = 1 + (f_test * ret / largest_loss)
                if normalized_return <= 0:
                    geom_returns = []  # Invalid f, would cause ruin
                    break
                geom_returns.append(normalized_return)
            
            if geom_returns:
                geom_mean = math.exp(sum(math.log(r) for r in geom_returns) / len(geom_returns))
                if geom_mean > best_geom_mean:
                    best_geom_mean = geom_mean
                    best_f = f_test
        
        return best_f
    
    def _calculate_time_metrics(self, performance: InstrumentPerformance, trades: List[TradeMetrics]) -> None:
        """Calculate time-based performance metrics"""
        
        if not trades:
            return
        
        # Average hold times
        performance.avg_hold_time = statistics.mean(t.hold_time_minutes for t in trades)
        
        winning_trades = [t for t in trades if t.net_pnl > 0]
        losing_trades = [t for t in trades if t.net_pnl < 0]
        
        if winning_trades:
            performance.win_hold_time = statistics.mean(t.hold_time_minutes for t in winning_trades)
        
        if losing_trades:
            performance.loss_hold_time = statistics.mean(t.hold_time_minutes for t in losing_trades)
    
    def _calculate_sequence_metrics(self, performance: InstrumentPerformance, trades: List[TradeMetrics]) -> None:
        """Calculate consecutive win/loss sequences"""
        
        if not trades:
            return
        
        current_wins = 0
        current_losses = 0
        max_wins = 0
        max_losses = 0
        
        for trade in trades:
            if trade.net_pnl > 0:
                current_wins += 1
                current_losses = 0
                max_wins = max(max_wins, current_wins)
            elif trade.net_pnl < 0:
                current_losses += 1
                current_wins = 0
                max_losses = max(max_losses, current_losses)
            else:
                current_wins = 0
                current_losses = 0
        
        performance.max_consecutive_wins = max_wins
        performance.max_consecutive_losses = max_losses
        performance.consecutive_wins = current_wins
        performance.consecutive_losses = current_losses
    
    def get_all_instruments_performance(self) -> Dict[str, InstrumentPerformance]:
        """Get performance metrics for all instruments"""
        
        results = {}
        for symbol in self.trade_data.keys():
            results[symbol] = self.calculate_instrument_performance(symbol)
        
        return results

def test_advanced_performance_calculator():
    """Test the advanced performance calculator with sample data"""
    
    print("🧪 ADVANCED PERFORMANCE METRICS CALCULATOR TEST")
    print("="*80)
    
    # Initialize calculator
    calc = AdvancedPerformanceCalculator()
    
    # Create sample trades
    sample_trades = [
        TradeMetrics(
            trade_id="EURUSD_001",
            entry_time=datetime.now() - timedelta(hours=24),
            exit_time=datetime.now() - timedelta(hours=20),
            symbol="EURUSD",
            direction=1,
            entry_price=1.0500,
            exit_price=1.0520,
            quantity=100000,
            highest_price=1.0535,
            lowest_price=1.0485,
            gross_pnl=200.0,
            commission=5.0,
            slippage=2.0,
            net_pnl=193.0
        ),
        TradeMetrics(
            trade_id="EURUSD_002", 
            entry_time=datetime.now() - timedelta(hours=20),
            exit_time=datetime.now() - timedelta(hours=16),
            symbol="EURUSD",
            direction=-1,
            entry_price=1.0520,
            exit_price=1.0495,
            quantity=100000,
            highest_price=1.0530,
            lowest_price=1.0480,
            gross_pnl=250.0,
            commission=5.0,
            slippage=3.0,
            net_pnl=242.0
        )
    ]
    
    # Add trades to calculator
    for trade in sample_trades:
        calc.add_trade(trade)
    
    # Calculate performance
    performance = calc.calculate_instrument_performance("EURUSD")
    
    # Display results
    print(f"\n📊 EURUSD PERFORMANCE METRICS:")
    print(f"="*60)
    
    print(f"\n🎯 BASIC METRICS:")
    print(f"   Total Trades: {performance.total_trades}")
    print(f"   Win Rate: {performance.win_rate:.1f}%")
    print(f"   Total P&L: {performance.total_pnl:+.2f}")
    print(f"   Expectancy: {performance.expectancy:+.2f}")
    
    print(f"\n📈 ADVANCED RATIOS:")
    print(f"   Profit Factor: {performance.profit_factor:.2f}")
    print(f"   Sharpe Ratio: {performance.sharpe_ratio:.2f}")
    print(f"   Sortino Ratio: {performance.sortino_ratio:.2f}")
    print(f"   Omega Ratio: {performance.omega_ratio:.2f}")
    print(f"   Calmar Ratio: {performance.calmar_ratio:.2f}")
    
    print(f"\n🎯 MAE/MFE ANALYSIS:")
    print(f"   Average MAE: {performance.avg_mae:.2f}")
    print(f"   Average MFE: {performance.avg_mfe:.2f}")
    print(f"   MAE Efficiency: {performance.mae_efficiency:.2f}")
    print(f"   MFE Realization: {performance.mfe_realization:.2f}")
    print(f"   Trade Efficiency: {performance.trade_efficiency:.2f}")
    
    print(f"\n⚠️  RISK METRICS:")
    print(f"   Max Drawdown: {performance.max_drawdown:.2f}")
    print(f"   Ulcer Index: {performance.ulcer_index:.2f}")
    print(f"   VaR (95%): {performance.value_at_risk_95:.2f}")
    print(f"   CVaR (95%): {performance.conditional_var_95:.2f}")
    
    print(f"\n🔍 CONSISTENCY METRICS:")
    print(f"   Z-Factor: {performance.z_factor:.2f}")
    print(f"   K-Factor: {performance.k_factor:.2f}")
    print(f"   Lake Ratio: {performance.lake_ratio:.2f}")
    print(f"   System Quality Number: {performance.system_quality_number:.2f}")
    
    print(f"\n💰 KELLY CRITERION:")
    print(f"   Kelly Percentage: {performance.kelly_percentage:.1%}")
    print(f"   Optimal f: {performance.optimal_f:.1%}")
    
    print(f"\n⏱️  TIME ANALYSIS:")
    print(f"   Avg Hold Time: {performance.avg_hold_time:.1f} minutes")
    print(f"   Win Hold Time: {performance.win_hold_time:.1f} minutes")
    print(f"   Loss Hold Time: {performance.loss_hold_time:.1f} minutes")
    
    print(f"\n🔄 SEQUENCE ANALYSIS:")
    print(f"   Max Consecutive Wins: {performance.max_consecutive_wins}")
    print(f"   Max Consecutive Losses: {performance.max_consecutive_losses}")
    
    print(f"\n✅ Advanced performance calculator test completed!")

if __name__ == "__main__":
    test_advanced_performance_calculator()