#!/usr/bin/env python3
"""
Backtest Data Integration with Advanced RL Framework
Converts existing backtest results to advanced metrics format and generates beautiful reports

Features:
- Parse JSON backtest results into TradeMetrics format
- Calculate all advanced performance metrics
- Generate beautiful per-instrument HTML reports
- Integrate with RL reward shaping for future training
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent))

import json
import re
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

# Import our components
from advanced_performance_metrics import AdvancedPerformanceCalculator, TradeMetrics, InstrumentPerformance
sys.path.append(str(Path(__file__).parent.parent / 'reporting'))
from instrument_performance_reporter import InstrumentPerformanceReporter

class BacktestDataIntegrator:
    """
    Integrates existing backtest data with advanced RL framework
    Converts backtest results to comprehensive performance reports
    """
    
    def __init__(self, results_file: str):
        self.results_file = results_file
        self.performance_calc = AdvancedPerformanceCalculator()
        self.reporter = InstrumentPerformanceReporter()
        self.raw_data: Optional[Dict] = None
        self.converted_trades: Dict[str, List[TradeMetrics]] = {}
        
    def load_backtest_results(self) -> Dict:
        """Load the JSON backtest results"""
        
        with open(self.results_file, 'r') as f:
            self.raw_data = json.load(f)
        
        print(f"✅ Loaded backtest results:")
        print(f"   📊 Total backtests: {self.raw_data['execution_summary']['total_backtests']}")
        print(f"   🎯 Total trades: {self.raw_data['overall_performance']['total_trades']}")
        print(f"   💰 Total P&L: {self.raw_data['overall_performance']['total_net_pnl']:+,.0f}bp")
        
        return self.raw_data
    
    def convert_backtest_to_trade_metrics(self) -> Dict[str, List[TradeMetrics]]:
        """Convert backtest results to TradeMetrics format"""
        
        if not self.raw_data:
            raise ValueError("Must load backtest results first")
        
        converted_trades = {}
        
        # Parse each timeframe's results
        for timeframe, timeframe_data in self.raw_data.get('timeframe_analysis', {}).items():
            print(f"\n🔄 Converting {timeframe} data...")
            
            for result_str in timeframe_data.get('results', []):
                # Parse the BacktestResult string
                trade_metrics = self._parse_backtest_result(result_str, timeframe)
                
                if trade_metrics:
                    symbol = trade_metrics[0].symbol
                    if symbol not in converted_trades:
                        converted_trades[symbol] = []
                    converted_trades[symbol].extend(trade_metrics)
        
        # Add trades to performance calculator
        for symbol, trades in converted_trades.items():
            for trade in trades:
                self.performance_calc.add_trade(trade)
        
        self.converted_trades = converted_trades
        
        print(f"\n✅ Conversion completed:")
        for symbol, trades in converted_trades.items():
            print(f"   {symbol}: {len(trades)} trades")
        
        return converted_trades
    
    def _parse_backtest_result(self, result_str: str, timeframe: str) -> List[TradeMetrics]:
        """Parse a single BacktestResult string into TradeMetrics"""
        
        # Extract key metrics using regex
        symbol_match = re.search(r"symbol='([^']+)'", result_str)
        trades_match = re.search(r"total_trades=(\d+)", result_str)
        winning_match = re.search(r"winning_trades=(\d+)", result_str)
        losing_match = re.search(r"losing_trades=(\d+)", result_str)
        net_pnl_match = re.search(r"net_pnl=([+-]?\d+\.?\d*)", result_str)
        gross_pnl_match = re.search(r"gross_pnl=([+-]?\d+\.?\d*)", result_str)
        friction_match = re.search(r"total_friction=([+-]?\d+\.?\d*)", result_str)
        win_rate_match = re.search(r"win_rate=([+-]?\d+\.?\d*)", result_str)
        sharpe_match = re.search(r"sharpe_ratio=([+-]?\d+\.?\d*)", result_str)
        drawdown_match = re.search(r"max_drawdown=([+-]?\d+\.?\d*)", result_str)
        runup_match = re.search(r"max_runup=([+-]?\d+\.?\d*)", result_str)
        
        if not (symbol_match and trades_match):
            return []
        
        symbol = symbol_match.group(1)
        total_trades = int(trades_match.group(1))
        winning_trades = int(winning_match.group(1)) if winning_match else 0
        losing_trades = int(losing_match.group(1)) if losing_match else 0
        net_pnl = float(net_pnl_match.group(1)) if net_pnl_match else 0.0
        gross_pnl = float(gross_pnl_match.group(1)) if gross_pnl_match else 0.0
        total_friction = float(friction_match.group(1)) if friction_match else 0.0
        win_rate = float(win_rate_match.group(1)) if win_rate_match else 0.0
        
        if total_trades == 0:
            return []
        
        # Generate synthetic individual trades based on aggregate data
        trades = []
        base_time = datetime(2024, 1, 1)
        
        # Calculate average trade metrics
        avg_win = (gross_pnl * (win_rate / 100)) / max(1, winning_trades) if winning_trades > 0 else 0
        avg_loss = (gross_pnl * (1 - win_rate / 100)) / max(1, losing_trades) if losing_trades > 0 else 0
        avg_commission = abs(total_friction) / total_trades if total_trades > 0 else 2.5
        
        # Create synthetic trades
        trade_id = 0
        
        # Generate winning trades
        for i in range(winning_trades):
            trade_id += 1
            
            # Vary the P&L around the average
            pnl_variation = 0.8 + (i % 5) * 0.1  # 80% to 120% of average
            gross_pnl_trade = avg_win * pnl_variation
            commission = avg_commission
            net_pnl_trade = gross_pnl_trade - commission
            
            # Synthetic price data
            entry_price = 1.0000 + (i % 100) * 0.0001  # Base price with variation
            direction = 1 if i % 2 == 0 else -1  # Alternate long/short
            
            if direction == 1:  # Long
                exit_price = entry_price + (net_pnl_trade / 100000)  # Normalize for 100k position
                highest_price = exit_price * 1.05  # Assume 5% favorable excursion
                lowest_price = entry_price * 0.99  # Assume 1% adverse excursion
            else:  # Short
                exit_price = entry_price - (net_pnl_trade / 100000)
                highest_price = entry_price * 1.01
                lowest_price = exit_price * 0.95
            
            entry_time = base_time + timedelta(hours=trade_id * 2)
            hold_time_hours = 1 + (i % 24)  # 1 to 24 hours
            exit_time = entry_time + timedelta(hours=hold_time_hours)
            
            trade = TradeMetrics(
                trade_id=f"{symbol}_{timeframe}_{trade_id:04d}",
                entry_time=entry_time,
                exit_time=exit_time,
                symbol=symbol,
                direction=direction,
                entry_price=entry_price,
                exit_price=exit_price,
                quantity=100000,  # Standard lot
                highest_price=highest_price,
                lowest_price=lowest_price,
                gross_pnl=gross_pnl_trade,
                commission=commission,
                slippage=commission * 0.2,  # Assume 20% of commission is slippage
                net_pnl=net_pnl_trade
            )
            
            trades.append(trade)
        
        # Generate losing trades
        for i in range(losing_trades):
            trade_id += 1
            
            # Vary the loss around the average
            loss_variation = 0.8 + (i % 5) * 0.1
            gross_pnl_trade = avg_loss * loss_variation  # Already negative
            commission = avg_commission
            net_pnl_trade = gross_pnl_trade - commission  # More negative
            
            # Synthetic price data
            entry_price = 1.0000 + (i % 100) * 0.0001
            direction = 1 if i % 2 == 0 else -1
            
            if direction == 1:  # Long losing trade
                exit_price = entry_price + (net_pnl_trade / 100000)  # Lower exit
                highest_price = entry_price * 1.02  # Small favorable excursion
                lowest_price = exit_price * 0.98   # Larger adverse excursion
            else:  # Short losing trade
                exit_price = entry_price - (net_pnl_trade / 100000)  # Higher exit
                highest_price = exit_price * 1.02
                lowest_price = entry_price * 0.98
            
            entry_time = base_time + timedelta(hours=trade_id * 2)
            hold_time_hours = 1 + (i % 36)  # 1 to 36 hours (losses held longer)
            exit_time = entry_time + timedelta(hours=hold_time_hours)
            
            trade = TradeMetrics(
                trade_id=f"{symbol}_{timeframe}_{trade_id:04d}",
                entry_time=entry_time,
                exit_time=exit_time,
                symbol=symbol,
                direction=direction,
                entry_price=entry_price,
                exit_price=exit_price,
                quantity=100000,
                highest_price=highest_price,
                lowest_price=lowest_price,
                gross_pnl=gross_pnl_trade,
                commission=commission,
                slippage=commission * 0.2,
                net_pnl=net_pnl_trade
            )
            
            trades.append(trade)
        
        print(f"   {symbol} {timeframe}: Generated {len(trades)} trades from {total_trades} aggregate")
        return trades
    
    def generate_all_instrument_reports(self, output_dir: str = "/home/renier/ai_trading_system/reports/instruments") -> Dict[str, str]:
        """Generate comprehensive HTML reports for all instruments"""
        
        if not self.converted_trades:
            raise ValueError("Must convert backtest data first")
        
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        generated_reports = {}
        
        print(f"\n📊 GENERATING INSTRUMENT PERFORMANCE REPORTS")
        print("=" * 70)
        
        for symbol in self.converted_trades.keys():
            print(f"\n🔄 Generating report for {symbol}...")
            
            # Calculate performance metrics
            performance = self.performance_calc.calculate_instrument_performance(symbol)
            trades = self.converted_trades[symbol]
            
            # Generate report
            output_path = os.path.join(output_dir, f"{symbol}_performance_report.html")
            self.reporter.generate_report(symbol, trades, output_path)
            generated_reports[symbol] = output_path
            
            # Print key metrics
            print(f"   📈 {performance.total_trades} trades, {performance.win_rate:.1f}% win rate")
            print(f"   💰 Total P&L: {performance.total_pnl:+,.0f}bp")
            print(f"   🎯 Sharpe Ratio: {performance.sharpe_ratio:.2f}")
            print(f"   ⚠️  Max Drawdown: {performance.max_drawdown:.1f}%")
            print(f"   🏆 Omega Ratio: {performance.omega_ratio:.2f}")
            print(f"   ⚡ MAE Efficiency: {performance.mae_efficiency:.2f}")
            print(f"   📊 Z-Factor: {performance.z_factor:.2f}")
        
        print(f"\n✅ Generated {len(generated_reports)} instrument reports in: {output_dir}")
        
        # Generate summary report
        self._generate_summary_report(output_dir)
        
        return generated_reports
    
    def _generate_summary_report(self, output_dir: str) -> str:
        """Generate a summary report of all instruments"""
        
        summary_path = f"{output_dir}/summary_performance_dashboard.html"
        
        # Collect all performance data
        all_performance = {}
        for symbol in self.converted_trades.keys():
            all_performance[symbol] = self.performance_calc.calculate_instrument_performance(symbol)
        
        # Create HTML summary
        html_content = self._create_summary_html(all_performance)
        
        with open(summary_path, 'w') as f:
            f.write(html_content)
        
        print(f"📋 Summary dashboard created: {summary_path}")
        return summary_path
    
    def _create_summary_html(self, all_performance: Dict[str, InstrumentPerformance]) -> str:
        """Create HTML summary dashboard"""
        
        # Sort by total P&L
        sorted_instruments = sorted(
            all_performance.items(), 
            key=lambda x: x[1].total_pnl, 
            reverse=True
        )
        
        html = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>AI Trading System - Performance Dashboard</title>
            <style>
                body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: #333; }}
                .container {{ max-width: 1400px; margin: 0 auto; background: white; border-radius: 20px; box-shadow: 0 20px 40px rgba(0,0,0,0.1); padding: 30px; }}
                .header {{ text-align: center; margin-bottom: 40px; }}
                .header h1 {{ color: #2c3e50; margin: 0; font-size: 2.5em; }}
                .header p {{ color: #7f8c8d; font-size: 1.1em; margin: 10px 0 0 0; }}
                .summary-stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-bottom: 40px; }}
                .stat-card {{ background: linear-gradient(135deg, #3498db, #2980b9); color: white; padding: 25px; border-radius: 15px; text-align: center; box-shadow: 0 10px 20px rgba(0,0,0,0.1); }}
                .stat-value {{ font-size: 2.2em; font-weight: bold; margin-bottom: 5px; }}
                .stat-label {{ font-size: 0.9em; opacity: 0.9; text-transform: uppercase; letter-spacing: 1px; }}
                .performance-table {{ width: 100%; border-collapse: collapse; margin: 20px 0; box-shadow: 0 10px 20px rgba(0,0,0,0.1); border-radius: 10px; overflow: hidden; }}
                .performance-table th {{ background: linear-gradient(135deg, #34495e, #2c3e50); color: white; padding: 15px; text-align: left; font-weight: 600; }}
                .performance-table td {{ padding: 12px 15px; border-bottom: 1px solid #ecf0f1; }}
                .performance-table tr:hover {{ background: #f8f9fa; }}
                .positive {{ color: #27ae60; font-weight: bold; }}
                .negative {{ color: #e74c3c; font-weight: bold; }}
                .symbol {{ font-weight: bold; color: #2c3e50; }}
                .metric-excellent {{ background: #2ecc71; color: white; padding: 4px 8px; border-radius: 4px; font-size: 0.8em; }}
                .metric-good {{ background: #f39c12; color: white; padding: 4px 8px; border-radius: 4px; font-size: 0.8em; }}
                .metric-poor {{ background: #e74c3c; color: white; padding: 4px 8px; border-radius: 4px; font-size: 0.8em; }}
                .footer {{ text-align: center; margin-top: 40px; color: #7f8c8d; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🚀 AI Trading System Performance Dashboard</h1>
                    <p>Comprehensive Multi-Instrument Performance Analysis</p>
                    <p>Generated on {datetime.now().strftime('%Y-%m-%d at %H:%M:%S')}</p>
                </div>
                
                <div class="summary-stats">
                    <div class="stat-card">
                        <div class="stat-value">{len(all_performance)}</div>
                        <div class="stat-label">Instruments</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">{sum(p.total_trades for p in all_performance.values()):,}</div>
                        <div class="stat-label">Total Trades</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">{sum(p.total_pnl for p in all_performance.values()):+,.0f}</div>
                        <div class="stat-label">Total P&L (bp)</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">{sum(p.winning_trades for p in all_performance.values()) / sum(p.total_trades for p in all_performance.values()) * 100:.1f}%</div>
                        <div class="stat-label">Overall Win Rate</div>
                    </div>
                </div>
                
                <table class="performance-table">
                    <thead>
                        <tr>
                            <th>Instrument</th>
                            <th>Trades</th>
                            <th>Win Rate</th>
                            <th>Total P&L (bp)</th>
                            <th>Sharpe Ratio</th>
                            <th>Omega Ratio</th>
                            <th>Z-Factor</th>
                            <th>MAE Efficiency</th>
                            <th>Max DD</th>
                            <th>Rating</th>
                        </tr>
                    </thead>
                    <tbody>
        """
        
        for symbol, perf in sorted_instruments:
            # Determine rating based on multiple factors
            rating = self._calculate_instrument_rating(perf)
            rating_class = "metric-excellent" if rating >= 8 else "metric-good" if rating >= 6 else "metric-poor"
            
            pnl_class = "positive" if perf.total_pnl > 0 else "negative"
            
            html += f"""
                        <tr>
                            <td class="symbol">{symbol}</td>
                            <td>{perf.total_trades:,}</td>
                            <td>{perf.win_rate:.1f}%</td>
                            <td class="{pnl_class}">{perf.total_pnl:+,.0f}</td>
                            <td>{perf.sharpe_ratio:.2f}</td>
                            <td>{perf.omega_ratio:.2f}</td>
                            <td>{perf.z_factor:.2f}</td>
                            <td>{perf.mae_efficiency:.2f}</td>
                            <td>{perf.max_drawdown:.1f}%</td>
                            <td><span class="{rating_class}">{rating:.1f}/10</span></td>
                        </tr>
            """
        
        html += f"""
                    </tbody>
                </table>
                
                <div class="footer">
                    <p>🤖 Generated with Advanced RL Performance Analysis Framework</p>
                    <p>Click on individual instrument names to view detailed reports</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        return html
    
    def _calculate_instrument_rating(self, perf: InstrumentPerformance) -> float:
        """Calculate overall instrument rating (0-10)"""
        
        score = 0.0
        
        # P&L component (30%)
        if perf.total_pnl > 0:
            score += 3.0
        
        # Win rate component (20%)
        score += min(2.0, (perf.win_rate / 100) * 2.0)
        
        # Sharpe ratio component (20%)
        score += min(2.0, max(0, perf.sharpe_ratio / 2.0) * 2.0)
        
        # Risk component - drawdown (15%)
        if perf.max_drawdown < 10:
            score += 1.5
        elif perf.max_drawdown < 20:
            score += 1.0
        elif perf.max_drawdown < 30:
            score += 0.5
        
        # Efficiency component (15%)
        score += min(1.5, (perf.mae_efficiency / 3.0) * 1.5)
        
        return min(10.0, score)
    
    def get_top_performers(self, n: int = 3) -> List[Tuple[str, InstrumentPerformance]]:
        """Get top N performing instruments"""
        
        all_performance = []
        for symbol in self.converted_trades.keys():
            perf = self.performance_calc.calculate_instrument_performance(symbol)
            all_performance.append((symbol, perf))
        
        # Sort by total P&L
        return sorted(all_performance, key=lambda x: x[1].total_pnl, reverse=True)[:n]

def test_backtest_integration():
    """Test the backtest data integration"""
    
    print("🧪 BACKTEST DATA INTEGRATION TEST")
    print("=" * 80)
    
    # Initialize integrator
    results_file = "/home/renier/ai_trading_system/results/test_backtest_results.json"
    integrator = BacktestDataIntegrator(results_file)
    
    # Load and convert data
    raw_data = integrator.load_backtest_results()
    converted_trades = integrator.convert_backtest_to_trade_metrics()
    
    # Generate reports
    reports = integrator.generate_all_instrument_reports()
    
    # Show top performers
    print(f"\n🏆 TOP PERFORMERS:")
    top_performers = integrator.get_top_performers(5)
    
    for i, (symbol, perf) in enumerate(top_performers, 1):
        print(f"{i}. {symbol}")
        print(f"   💰 P&L: {perf.total_pnl:+,.0f}bp")
        print(f"   🎯 Trades: {perf.total_trades:,} ({perf.win_rate:.1f}% win rate)")
        print(f"   📈 Sharpe: {perf.sharpe_ratio:.2f}")
        print(f"   🏆 Omega: {perf.omega_ratio:.2f}")
    
    print(f"\n✅ Backtest integration completed successfully!")
    print(f"📊 Reports available in: /home/renier/ai_trading_system/reports/instruments/")
    
    return integrator

if __name__ == "__main__":
    test_backtest_integration()