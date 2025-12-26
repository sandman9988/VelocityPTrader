#!/usr/bin/env python3
"""
Advanced Performance Metrics Calculator
Comprehensive trading performance analysis and metrics calculation

Features:
- Multi-dimensional performance analysis
- Risk-adjusted metrics
- Regime-specific performance
- Agent comparison metrics
"""

import math
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from collections import defaultdict
import statistics

@dataclass
class PerformanceSnapshot:
    """Snapshot of performance metrics at a point in time"""
    timestamp: datetime
    total_trades: int
    winning_trades: int
    losing_trades: int
    total_pnl: float
    gross_profit: float
    gross_loss: float
    max_drawdown: float
    max_runup: float
    win_rate: float
    profit_factor: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    expectancy: float

class AdvancedPerformanceCalculator:
    """Advanced performance metrics calculator for trading systems"""
    
    def __init__(self):
        self.trade_history: List[Dict] = []
        self.equity_curve: List[Tuple[datetime, float]] = []
        self.performance_snapshots: List[PerformanceSnapshot] = []
        
        # Risk-free rate for Sharpe calculation
        self.risk_free_rate = 0.02  # 2% annual
        
        print("📊 Advanced Performance Calculator initialized")
    
    def add_trade(self, trade_data: Dict[str, Any]):
        """Add trade to performance tracking"""
        
        trade = {
            'timestamp': datetime.fromisoformat(trade_data['timestamp']) if isinstance(trade_data['timestamp'], str) else trade_data['timestamp'],
            'symbol': trade_data['symbol'],
            'direction': trade_data['direction'],
            'size': trade_data['size'],
            'entry_price': trade_data['entry_price'],
            'exit_price': trade_data.get('exit_price', trade_data['entry_price']),
            'pnl': trade_data['pnl'],
            'commission': trade_data.get('commission', 0),
            'swap': trade_data.get('swap', 0),
            'duration_seconds': trade_data.get('duration_seconds', 0),
            'agent': trade_data.get('agent', 'Unknown'),
            'regime': trade_data.get('regime', 'Unknown'),
            'confidence': trade_data.get('confidence', 0.5)
        }
        
        self.trade_history.append(trade)
        
        # Update equity curve
        total_pnl = sum(t['pnl'] for t in self.trade_history)
        self.equity_curve.append((trade['timestamp'], total_pnl))
        
        # Create performance snapshot
        if len(self.trade_history) % 10 == 0:  # Every 10 trades
            snapshot = self._create_performance_snapshot()
            self.performance_snapshots.append(snapshot)
    
    def calculate_comprehensive_metrics(self) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics"""
        
        if not self.trade_history:
            return self._empty_metrics()
        
        # Basic metrics
        total_trades = len(self.trade_history)
        winning_trades = [t for t in self.trade_history if t['pnl'] > 0]
        losing_trades = [t for t in self.trade_history if t['pnl'] < 0]
        breakeven_trades = [t for t in self.trade_history if t['pnl'] == 0]
        
        total_pnl = sum(t['pnl'] for t in self.trade_history)
        gross_profit = sum(t['pnl'] for t in winning_trades)
        gross_loss = abs(sum(t['pnl'] for t in losing_trades))
        
        # Ratio metrics
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
        loss_rate = len(losing_trades) / total_trades if total_trades > 0 else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf') if gross_profit > 0 else 0
        
        # Average metrics
        avg_win = gross_profit / len(winning_trades) if winning_trades else 0
        avg_loss = gross_loss / len(losing_trades) if losing_trades else 0
        avg_trade = total_pnl / total_trades if total_trades > 0 else 0
        
        # Risk metrics
        max_drawdown = self._calculate_max_drawdown()
        max_runup = self._calculate_max_runup()
        
        # Statistical metrics
        sharpe_ratio = self._calculate_sharpe_ratio()
        sortino_ratio = self._calculate_sortino_ratio()
        calmar_ratio = self._calculate_calmar_ratio(total_pnl, max_drawdown)
        
        # Expectancy
        expectancy = (win_rate * avg_win) - (loss_rate * avg_loss)
        
        # Advanced metrics
        consecutive_wins, consecutive_losses = self._calculate_consecutive_runs()
        recovery_factor = abs(total_pnl / max_drawdown) if max_drawdown != 0 else float('inf')
        
        return {
            'basic_metrics': {
                'total_trades': total_trades,
                'winning_trades': len(winning_trades),
                'losing_trades': len(losing_trades),
                'breakeven_trades': len(breakeven_trades),
                'win_rate': win_rate,
                'loss_rate': loss_rate,
                'total_pnl': total_pnl,
                'gross_profit': gross_profit,
                'gross_loss': gross_loss,
                'profit_factor': profit_factor
            },
            'average_metrics': {
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'avg_trade': avg_trade,
                'avg_trade_duration': self._calculate_avg_duration()
            },
            'risk_metrics': {
                'max_drawdown': max_drawdown,
                'max_runup': max_runup,
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio,
                'calmar_ratio': calmar_ratio,
                'recovery_factor': recovery_factor
            },
            'distribution_metrics': {
                'expectancy': expectancy,
                'largest_win': max((t['pnl'] for t in winning_trades), default=0),
                'largest_loss': min((t['pnl'] for t in losing_trades), default=0),
                'consecutive_wins': consecutive_wins,
                'consecutive_losses': consecutive_losses
            },
            'agent_breakdown': self._calculate_agent_breakdown(),
            'regime_breakdown': self._calculate_regime_breakdown(),
            'symbol_breakdown': self._calculate_symbol_breakdown(),
            'monthly_breakdown': self._calculate_monthly_breakdown()
        }
    
    def calculate_agent_comparison(self) -> Dict[str, Any]:
        """Compare performance between different agents"""
        
        agents = set(t['agent'] for t in self.trade_history)
        agent_comparison = {}
        
        for agent in agents:
            agent_trades = [t for t in self.trade_history if t['agent'] == agent]
            
            if agent_trades:
                agent_metrics = self._calculate_metrics_for_trades(agent_trades)
                agent_comparison[agent] = agent_metrics
        
        return agent_comparison
    
    def calculate_regime_performance(self) -> Dict[str, Any]:
        """Calculate performance by market regime"""
        
        regimes = set(t['regime'] for t in self.trade_history)
        regime_performance = {}
        
        for regime in regimes:
            regime_trades = [t for t in self.trade_history if t['regime'] == regime]
            
            if regime_trades:
                regime_metrics = self._calculate_metrics_for_trades(regime_trades)
                regime_performance[regime] = regime_metrics
        
        return regime_performance
    
    def _create_performance_snapshot(self) -> PerformanceSnapshot:
        """Create performance snapshot at current moment"""
        
        metrics = self.calculate_comprehensive_metrics()
        basic = metrics['basic_metrics']
        risk = metrics['risk_metrics']
        
        return PerformanceSnapshot(
            timestamp=datetime.now(),
            total_trades=basic['total_trades'],
            winning_trades=basic['winning_trades'],
            losing_trades=basic['losing_trades'],
            total_pnl=basic['total_pnl'],
            gross_profit=basic['gross_profit'],
            gross_loss=basic['gross_loss'],
            max_drawdown=risk['max_drawdown'],
            max_runup=risk['max_runup'],
            win_rate=basic['win_rate'],
            profit_factor=basic['profit_factor'],
            sharpe_ratio=risk['sharpe_ratio'],
            sortino_ratio=risk['sortino_ratio'],
            calmar_ratio=risk['calmar_ratio'],
            expectancy=metrics['distribution_metrics']['expectancy']
        )
    
    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown from equity curve"""
        
        if len(self.equity_curve) < 2:
            return 0.0
        
        peak = self.equity_curve[0][1]
        max_dd = 0.0
        
        for timestamp, equity in self.equity_curve:
            if equity > peak:
                peak = equity
            else:
                drawdown = peak - equity
                max_dd = max(max_dd, drawdown)
        
        return max_dd
    
    def _calculate_max_runup(self) -> float:
        """Calculate maximum runup from equity curve"""
        
        if len(self.equity_curve) < 2:
            return 0.0
        
        trough = self.equity_curve[0][1]
        max_runup = 0.0
        
        for timestamp, equity in self.equity_curve:
            if equity < trough:
                trough = equity
            else:
                runup = equity - trough
                max_runup = max(max_runup, runup)
        
        return max_runup
    
    def _calculate_sharpe_ratio(self) -> float:
        """Calculate Sharpe ratio"""
        
        if len(self.trade_history) < 2:
            return 0.0
        
        returns = [t['pnl'] for t in self.trade_history]
        avg_return = statistics.mean(returns)
        std_return = statistics.stdev(returns) if len(returns) > 1 else 0
        
        if std_return == 0:
            return 0.0
        
        # Annualized Sharpe ratio
        risk_free_daily = self.risk_free_rate / 365
        excess_return = avg_return - risk_free_daily
        
        return excess_return / std_return * math.sqrt(252)  # Assuming daily returns
    
    def _calculate_sortino_ratio(self) -> float:
        """Calculate Sortino ratio (downside deviation)"""
        
        if len(self.trade_history) < 2:
            return 0.0
        
        returns = [t['pnl'] for t in self.trade_history]
        negative_returns = [r for r in returns if r < 0]
        
        if not negative_returns:
            return float('inf')
        
        avg_return = statistics.mean(returns)
        downside_deviation = statistics.stdev(negative_returns)
        
        if downside_deviation == 0:
            return 0.0
        
        return avg_return / downside_deviation * math.sqrt(252)
    
    def _calculate_calmar_ratio(self, total_return: float, max_drawdown: float) -> float:
        """Calculate Calmar ratio"""
        
        if max_drawdown == 0:
            return float('inf') if total_return > 0 else 0.0
        
        return abs(total_return / max_drawdown)
    
    def _calculate_consecutive_runs(self) -> Tuple[int, int]:
        """Calculate maximum consecutive wins and losses"""
        
        if not self.trade_history:
            return 0, 0
        
        max_wins = 0
        max_losses = 0
        current_wins = 0
        current_losses = 0
        
        for trade in self.trade_history:
            if trade['pnl'] > 0:
                current_wins += 1
                current_losses = 0
                max_wins = max(max_wins, current_wins)
            elif trade['pnl'] < 0:
                current_losses += 1
                current_wins = 0
                max_losses = max(max_losses, current_losses)
            else:
                current_wins = 0
                current_losses = 0
        
        return max_wins, max_losses
    
    def _calculate_avg_duration(self) -> float:
        """Calculate average trade duration in hours"""
        
        durations = [t['duration_seconds'] for t in self.trade_history if t['duration_seconds'] > 0]
        
        if not durations:
            return 0.0
        
        avg_seconds = sum(durations) / len(durations)
        return avg_seconds / 3600.0  # Convert to hours
    
    def _calculate_agent_breakdown(self) -> Dict[str, Dict]:
        """Calculate performance breakdown by agent"""
        
        agents = defaultdict(list)
        for trade in self.trade_history:
            agents[trade['agent']].append(trade)
        
        breakdown = {}
        for agent, trades in agents.items():
            breakdown[agent] = self._calculate_metrics_for_trades(trades)
        
        return breakdown
    
    def _calculate_regime_breakdown(self) -> Dict[str, Dict]:
        """Calculate performance breakdown by market regime"""
        
        regimes = defaultdict(list)
        for trade in self.trade_history:
            regimes[trade['regime']].append(trade)
        
        breakdown = {}
        for regime, trades in regimes.items():
            breakdown[regime] = self._calculate_metrics_for_trades(trades)
        
        return breakdown
    
    def _calculate_symbol_breakdown(self) -> Dict[str, Dict]:
        """Calculate performance breakdown by symbol"""
        
        symbols = defaultdict(list)
        for trade in self.trade_history:
            symbols[trade['symbol']].append(trade)
        
        breakdown = {}
        for symbol, trades in symbols.items():
            breakdown[symbol] = self._calculate_metrics_for_trades(trades)
        
        return breakdown
    
    def _calculate_monthly_breakdown(self) -> Dict[str, Dict]:
        """Calculate performance breakdown by month"""
        
        months = defaultdict(list)
        for trade in self.trade_history:
            month_key = trade['timestamp'].strftime('%Y-%m')
            months[month_key].append(trade)
        
        breakdown = {}
        for month, trades in months.items():
            breakdown[month] = self._calculate_metrics_for_trades(trades)
        
        return breakdown
    
    def _calculate_metrics_for_trades(self, trades: List[Dict]) -> Dict[str, Any]:
        """Calculate metrics for a subset of trades"""
        
        if not trades:
            return {}
        
        total_trades = len(trades)
        winning_trades = [t for t in trades if t['pnl'] > 0]
        losing_trades = [t for t in trades if t['pnl'] < 0]
        
        total_pnl = sum(t['pnl'] for t in trades)
        gross_profit = sum(t['pnl'] for t in winning_trades)
        gross_loss = abs(sum(t['pnl'] for t in losing_trades))
        
        win_rate = len(winning_trades) / total_trades
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf') if gross_profit > 0 else 0
        
        return {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'gross_profit': gross_profit,
            'gross_loss': gross_loss,
            'profit_factor': profit_factor,
            'avg_trade': total_pnl / total_trades,
            'largest_win': max((t['pnl'] for t in winning_trades), default=0),
            'largest_loss': min((t['pnl'] for t in losing_trades), default=0)
        }
    
    def _empty_metrics(self) -> Dict[str, Any]:
        """Return empty metrics structure"""
        
        return {
            'basic_metrics': {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'breakeven_trades': 0,
                'win_rate': 0.0,
                'loss_rate': 0.0,
                'total_pnl': 0.0,
                'gross_profit': 0.0,
                'gross_loss': 0.0,
                'profit_factor': 0.0
            },
            'average_metrics': {
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'avg_trade': 0.0,
                'avg_trade_duration': 0.0
            },
            'risk_metrics': {
                'max_drawdown': 0.0,
                'max_runup': 0.0,
                'sharpe_ratio': 0.0,
                'sortino_ratio': 0.0,
                'calmar_ratio': 0.0,
                'recovery_factor': 0.0
            },
            'distribution_metrics': {
                'expectancy': 0.0,
                'largest_win': 0.0,
                'largest_loss': 0.0,
                'consecutive_wins': 0,
                'consecutive_losses': 0
            },
            'agent_breakdown': {},
            'regime_breakdown': {},
            'symbol_breakdown': {},
            'monthly_breakdown': {}
        }

def demonstrate_performance_calculator():
    """Demonstrate advanced performance calculator"""
    
    print("📊 ADVANCED PERFORMANCE CALCULATOR DEMONSTRATION")
    print("=" * 60)
    
    calculator = AdvancedPerformanceCalculator()
    
    # Add sample trades
    import random
    
    agents = ['BERSERKER', 'SNIPER']
    symbols = ['EURUSD+', 'GBPUSD+', 'BTCUSD+']
    regimes = ['CHAOTIC', 'UNDERDAMPED', 'CRITICALLY_DAMPED']
    
    for i in range(50):
        trade_data = {
            'timestamp': datetime.now() - timedelta(hours=50-i),
            'symbol': random.choice(symbols),
            'direction': random.choice(['BUY', 'SELL']),
            'size': random.uniform(0.01, 0.20),
            'entry_price': 1.1000 + random.uniform(-0.01, 0.01),
            'exit_price': 1.1000 + random.uniform(-0.015, 0.015),
            'pnl': random.uniform(-50, 100),
            'duration_seconds': random.randint(300, 14400),
            'agent': random.choice(agents),
            'regime': random.choice(regimes),
            'confidence': random.uniform(0.3, 0.9)
        }
        
        calculator.add_trade(trade_data)
        
        if i % 10 == 0:
            print(f"📈 Added trade {i+1}: {trade_data['symbol']} {trade_data['agent']} P&L: {trade_data['pnl']:+.1f}")
    
    # Calculate comprehensive metrics
    metrics = calculator.calculate_comprehensive_metrics()
    
    print(f"\n📊 PERFORMANCE SUMMARY:")
    basic = metrics['basic_metrics']
    print(f"   Total Trades: {basic['total_trades']}")
    print(f"   Win Rate: {basic['win_rate']:.1%}")
    print(f"   Total P&L: ${basic['total_pnl']:+,.0f}")
    print(f"   Profit Factor: {basic['profit_factor']:.2f}")
    
    risk = metrics['risk_metrics']
    print(f"\n🎯 RISK METRICS:")
    print(f"   Max Drawdown: ${risk['max_drawdown']:,.0f}")
    print(f"   Sharpe Ratio: {risk['sharpe_ratio']:.3f}")
    print(f"   Sortino Ratio: {risk['sortino_ratio']:.3f}")
    print(f"   Recovery Factor: {risk['recovery_factor']:.2f}")
    
    print(f"\n🤖 AGENT COMPARISON:")
    agent_breakdown = metrics['agent_breakdown']
    for agent, agent_metrics in agent_breakdown.items():
        print(f"   {agent}:")
        print(f"     Trades: {agent_metrics['total_trades']}")
        print(f"     Win Rate: {agent_metrics['win_rate']:.1%}")
        print(f"     Total P&L: ${agent_metrics['total_pnl']:+,.0f}")
        print(f"     Profit Factor: {agent_metrics['profit_factor']:.2f}")
    
    print(f"\n🌊 REGIME PERFORMANCE:")
    regime_breakdown = metrics['regime_breakdown']
    for regime, regime_metrics in regime_breakdown.items():
        print(f"   {regime}:")
        print(f"     Trades: {regime_metrics['total_trades']}")
        print(f"     Win Rate: {regime_metrics['win_rate']:.1%}")
        print(f"     Avg Trade: ${regime_metrics['avg_trade']:+.1f}")
    
    print("\n✅ Advanced Performance Calculator demonstration completed!")

if __name__ == "__main__":
    demonstrate_performance_calculator()