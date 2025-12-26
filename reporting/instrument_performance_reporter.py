#!/usr/bin/env python3
"""
Beautiful Per-Instrument Performance Reporter
Generates comprehensive, visually appealing performance reports for each instrument
Features:
- Detailed MAE/MFE analysis with charts
- Advanced metrics (Omega, Z-Factor, Kelly Criterion)  
- Risk analytics and drawdown analysis
- HTML reports with charts and tables
- Efficiency analysis and trade quality metrics
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import math
import statistics
import json
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import html

# Import our advanced components
from rl_framework.advanced_performance_metrics import AdvancedPerformanceCalculator, TradeMetrics, InstrumentPerformance

@dataclass
class ReportConfig:
    """Configuration for report generation"""
    include_charts: bool = True
    chart_width: int = 800
    chart_height: int = 400
    theme: str = "professional"  # professional, dark, minimal
    currency: str = "USD"
    decimal_places: int = 2
    
class InstrumentPerformanceReporter:
    """
    Beautiful performance reporter for individual instruments
    
    Generates comprehensive reports with:
    - Executive summary with key metrics
    - Detailed MAE/MFE efficiency analysis  
    - Advanced risk and consistency metrics
    - Trade quality and timing analysis
    - Interactive charts and visualizations
    - Recommendations and insights
    """
    
    def __init__(self, config: Optional[ReportConfig] = None):
        self.config = config or ReportConfig()
        self.performance_calc = AdvancedPerformanceCalculator()
        
        # CSS styles for different themes
        self.themes = {
            "professional": {
                "primary": "#2E86AB",
                "secondary": "#A23B72", 
                "success": "#F18F01",
                "warning": "#C73E1D",
                "background": "#FFFFFF",
                "text": "#333333",
                "border": "#E0E0E0"
            },
            "dark": {
                "primary": "#3B82F6",
                "secondary": "#8B5CF6",
                "success": "#10B981", 
                "warning": "#F59E0B",
                "background": "#1F2937",
                "text": "#F9FAFB",
                "border": "#374151"
            },
            "minimal": {
                "primary": "#000000",
                "secondary": "#666666",
                "success": "#4F46E5",
                "warning": "#DC2626", 
                "background": "#FFFFFF",
                "text": "#111827",
                "border": "#D1D5DB"
            }
        }
        
        print("📊 Instrument Performance Reporter initialized")
        print(f"   🎨 Theme: {self.config.theme}")
        print(f"   📈 Charts enabled: {self.config.include_charts}")
    
    def generate_report(self, symbol: str, trades: List[TradeMetrics], 
                       output_path: str) -> str:
        """Generate comprehensive performance report for instrument"""
        
        print(f"\n📊 Generating performance report for {symbol}")
        print(f"   📈 Analyzing {len(trades)} trades")
        print(f"   💾 Output: {output_path}")
        
        # Add trades to calculator
        for trade in trades:
            self.performance_calc.add_trade(trade)
        
        # Calculate performance metrics
        performance = self.performance_calc.calculate_instrument_performance(symbol)
        
        # Generate HTML report
        html_content = self._generate_html_report(symbol, performance, trades)
        
        # Write to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"✅ Report generated: {output_path}")
        return output_path
    
    def _generate_html_report(self, symbol: str, performance: InstrumentPerformance,
                             trades: List[TradeMetrics]) -> str:
        """Generate complete HTML report"""
        
        theme = self.themes[self.config.theme]
        
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{symbol} - Trading Performance Report</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        {self._generate_css(theme)}
    </style>
</head>
<body>
    <div class="container">
        {self._generate_header(symbol, performance)}
        {self._generate_executive_summary(performance)}
        {self._generate_key_metrics_grid(performance)}
        {self._generate_mae_mfe_section(performance, trades)}
        {self._generate_risk_analysis(performance)}
        {self._generate_efficiency_analysis(performance)}
        {self._generate_trade_analysis(trades)}
        {self._generate_recommendations(performance)}
        {self._generate_footer()}
    </div>
    
    {self._generate_javascript(trades, performance) if self.config.include_charts else ""}
</body>
</html>
        """
        
        return html_content
    
    def _generate_css(self, theme: Dict[str, str]) -> str:
        """Generate CSS styles"""
        
        return f"""
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: {theme['background']};
            color: {theme['text']};
            line-height: 1.6;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }}
        
        .header {{
            background: linear-gradient(135deg, {theme['primary']}, {theme['secondary']});
            color: white;
            padding: 30px;
            border-radius: 15px;
            margin-bottom: 30px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        }}
        
        .header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
            font-weight: 700;
        }}
        
        .header .subtitle {{
            font-size: 1.2em;
            opacity: 0.9;
        }}
        
        .section {{
            background: {theme['background']};
            border: 1px solid {theme['border']};
            border-radius: 12px;
            padding: 25px;
            margin-bottom: 25px;
            box-shadow: 0 4px 16px rgba(0,0,0,0.05);
        }}
        
        .section h2 {{
            color: {theme['primary']};
            font-size: 1.8em;
            margin-bottom: 20px;
            border-bottom: 3px solid {theme['primary']};
            padding-bottom: 10px;
        }}
        
        .section h3 {{
            color: {theme['secondary']};
            font-size: 1.4em;
            margin: 20px 0 15px 0;
        }}
        
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 25px;
        }}
        
        .metric-card {{
            background: {theme['background']};
            border: 2px solid {theme['border']};
            border-radius: 10px;
            padding: 20px;
            text-align: center;
            transition: transform 0.2s, box-shadow 0.2s;
        }}
        
        .metric-card:hover {{
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(0,0,0,0.1);
        }}
        
        .metric-value {{
            font-size: 2.2em;
            font-weight: bold;
            margin-bottom: 8px;
        }}
        
        .metric-label {{
            font-size: 0.9em;
            color: {theme['secondary']};
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        
        .positive {{ color: {theme['success']}; }}
        .negative {{ color: {theme['warning']}; }}
        .neutral {{ color: {theme['secondary']}; }}
        
        .chart-container {{
            margin: 20px 0;
            padding: 20px;
            background: {theme['background']};
            border-radius: 8px;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid {theme['border']};
        }}
        
        th {{
            background: {theme['primary']};
            color: white;
            font-weight: 600;
        }}
        
        tr:nth-child(even) {{
            background: rgba(0,0,0,0.02);
        }}
        
        .recommendation {{
            background: linear-gradient(45deg, {theme['primary']}15, {theme['secondary']}15);
            border-left: 4px solid {theme['primary']};
            padding: 20px;
            margin: 15px 0;
            border-radius: 5px;
        }}
        
        .recommendation-icon {{
            font-size: 1.2em;
            margin-right: 10px;
        }}
        
        .footer {{
            text-align: center;
            padding: 30px;
            color: {theme['secondary']};
            border-top: 1px solid {theme['border']};
            margin-top: 40px;
        }}
        
        .badge {{
            display: inline-block;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.8em;
            font-weight: bold;
            margin: 2px;
        }}
        
        .badge-excellent {{ background: {theme['success']}; color: white; }}
        .badge-good {{ background: {theme['primary']}; color: white; }}
        .badge-warning {{ background: {theme['warning']}; color: white; }}
        .badge-poor {{ background: {theme['text']}; color: white; }}
        
        .progress-bar {{
            background: {theme['border']};
            height: 8px;
            border-radius: 4px;
            overflow: hidden;
            margin: 10px 0;
        }}
        
        .progress-fill {{
            height: 100%;
            background: linear-gradient(90deg, {theme['primary']}, {theme['secondary']});
            transition: width 0.3s ease;
        }}
        
        @media (max-width: 768px) {{
            .container {{
                padding: 10px;
            }}
            
            .header h1 {{
                font-size: 2em;
            }}
            
            .metrics-grid {{
                grid-template-columns: 1fr;
            }}
        }}
        """
    
    def _generate_header(self, symbol: str, performance: InstrumentPerformance) -> str:
        """Generate report header"""
        
        # Determine overall performance grade
        if performance.total_pnl > 0 and performance.win_rate > 60 and performance.sharpe_ratio > 1:
            grade = "A+"
            grade_class = "excellent"
        elif performance.total_pnl > 0 and performance.win_rate > 50:
            grade = "B+"
            grade_class = "good"  
        elif performance.total_pnl > 0:
            grade = "C+"
            grade_class = "warning"
        else:
            grade = "D"
            grade_class = "poor"
        
        return f"""
        <div class="header">
            <h1>{symbol} Trading Performance</h1>
            <div class="subtitle">
                Comprehensive Analysis Report • Generated {datetime.now().strftime('%B %d, %Y at %I:%M %p')}
                <span class="badge badge-{grade_class}" style="margin-left: 20px; font-size: 1.1em;">Grade: {grade}</span>
            </div>
        </div>
        """
    
    def _generate_executive_summary(self, performance: InstrumentPerformance) -> str:
        """Generate executive summary"""
        
        # Calculate key insights
        risk_assessment = "Low Risk" if performance.max_drawdown < 10 else "Moderate Risk" if performance.max_drawdown < 20 else "High Risk"
        consistency = "Excellent" if performance.win_rate > 70 else "Good" if performance.win_rate > 55 else "Moderate" if performance.win_rate > 45 else "Poor"
        efficiency = "Excellent" if performance.mae_efficiency > 3 else "Good" if performance.mae_efficiency > 2 else "Moderate" if performance.mae_efficiency > 1 else "Poor"
        
        return f"""
        <div class="section">
            <h2>📊 Executive Summary</h2>
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-value {'positive' if performance.total_pnl > 0 else 'negative'}">
                        {performance.total_pnl:+.2f}bp
                    </div>
                    <div class="metric-label">Total P&L</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value neutral">{performance.total_trades:,}</div>
                    <div class="metric-label">Total Trades</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value {'positive' if performance.win_rate > 50 else 'negative'}">
                        {performance.win_rate:.1f}%
                    </div>
                    <div class="metric-label">Win Rate</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value {'positive' if performance.sharpe_ratio > 1 else 'negative'}">
                        {performance.sharpe_ratio:.2f}
                    </div>
                    <div class="metric-label">Sharpe Ratio</div>
                </div>
            </div>
            
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin-top: 20px;">
                <div>
                    <h4>🎯 Performance Assessment</h4>
                    <ul style="margin: 10px 0; padding-left: 20px;">
                        <li><strong>Risk Level:</strong> <span class="badge badge-{'good' if risk_assessment == 'Low Risk' else 'warning' if risk_assessment == 'Moderate Risk' else 'poor'}">{risk_assessment}</span></li>
                        <li><strong>Consistency:</strong> <span class="badge badge-{'excellent' if consistency == 'Excellent' else 'good' if consistency == 'Good' else 'warning' if consistency == 'Moderate' else 'poor'}">{consistency}</span></li>
                        <li><strong>Trade Efficiency:</strong> <span class="badge badge-{'excellent' if efficiency == 'Excellent' else 'good' if efficiency == 'Good' else 'warning' if efficiency == 'Moderate' else 'poor'}">{efficiency}</span></li>
                    </ul>
                </div>
                <div>
                    <h4>📈 Key Highlights</h4>
                    <ul style="margin: 10px 0; padding-left: 20px;">
                        <li>Expectancy: <strong>{performance.expectancy:+.2f}bp per trade</strong></li>
                        <li>Profit Factor: <strong>{performance.profit_factor:.2f}</strong></li>
                        <li>Max Drawdown: <strong>{performance.max_drawdown:.1f}%</strong></li>
                        <li>MAE Efficiency: <strong>{performance.mae_efficiency:.2f}</strong></li>
                    </ul>
                </div>
            </div>
        </div>
        """
    
    def _generate_key_metrics_grid(self, performance: InstrumentPerformance) -> str:
        """Generate key metrics grid"""
        
        return f"""
        <div class="section">
            <h2>🔍 Advanced Performance Metrics</h2>
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-value {'positive' if performance.omega_ratio > 1 else 'negative'}">
                        {performance.omega_ratio:.2f}
                    </div>
                    <div class="metric-label">Omega Ratio</div>
                    <div style="font-size: 0.8em; margin-top: 5px; color: #666;">
                        {'Excellent' if performance.omega_ratio > 2 else 'Good' if performance.omega_ratio > 1.5 else 'Fair' if performance.omega_ratio > 1 else 'Poor'}
                    </div>
                </div>
                
                <div class="metric-card">
                    <div class="metric-value {'positive' if performance.z_factor > 1.96 else 'negative'}">
                        {performance.z_factor:.2f}
                    </div>
                    <div class="metric-label">Z-Factor</div>
                    <div style="font-size: 0.8em; margin-top: 5px; color: #666;">
                        {'Significant' if abs(performance.z_factor) > 1.96 else 'Not Significant'}
                    </div>
                </div>
                
                <div class="metric-card">
                    <div class="metric-value neutral">
                        {performance.kelly_percentage:.1%}
                    </div>
                    <div class="metric-label">Kelly %</div>
                    <div style="font-size: 0.8em; margin-top: 5px; color: #666;">
                        Optimal Position Size
                    </div>
                </div>
                
                <div class="metric-card">
                    <div class="metric-value {'positive' if performance.ulcer_index < 5 else 'negative'}">
                        {performance.ulcer_index:.2f}
                    </div>
                    <div class="metric-label">Ulcer Index</div>
                    <div style="font-size: 0.8em; margin-top: 5px; color: #666;">
                        Drawdown Pain
                    </div>
                </div>
                
                <div class="metric-card">
                    <div class="metric-value {'positive' if performance.system_quality_number > 2 else 'negative'}">
                        {performance.system_quality_number:.2f}
                    </div>
                    <div class="metric-label">SQN</div>
                    <div style="font-size: 0.8em; margin-top: 5px; color: #666;">
                        System Quality Number
                    </div>
                </div>
                
                <div class="metric-card">
                    <div class="metric-value {'positive' if performance.martin_ratio > 1 else 'negative'}">
                        {performance.martin_ratio:.2f}
                    </div>
                    <div class="metric-label">Martin Ratio</div>
                    <div style="font-size: 0.8em; margin-top: 5px; color: #666;">
                        UPI Ratio
                    </div>
                </div>
            </div>
        </div>
        """
    
    def _generate_mae_mfe_section(self, performance: InstrumentPerformance, trades: List[TradeMetrics]) -> str:
        """Generate MAE/MFE analysis section"""
        
        return f"""
        <div class="section">
            <h2>🎯 MAE/MFE Efficiency Analysis</h2>
            
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 25px;">
                <div>
                    <h3>Excursion Metrics</h3>
                    <table>
                        <tr><th>Metric</th><th>Value</th><th>Assessment</th></tr>
                        <tr>
                            <td>Average MAE</td>
                            <td>{performance.avg_mae:.2f}bp</td>
                            <td><span class="badge badge-{'good' if performance.avg_mae < 50 else 'warning' if performance.avg_mae < 100 else 'poor'}">
                                {'Low' if performance.avg_mae < 50 else 'Moderate' if performance.avg_mae < 100 else 'High'}
                            </span></td>
                        </tr>
                        <tr>
                            <td>Average MFE</td>
                            <td>{performance.avg_mfe:.2f}bp</td>
                            <td><span class="badge badge-{'excellent' if performance.avg_mfe > 100 else 'good' if performance.avg_mfe > 50 else 'warning'}">
                                {'Excellent' if performance.avg_mfe > 100 else 'Good' if performance.avg_mfe > 50 else 'Moderate'}
                            </span></td>
                        </tr>
                        <tr>
                            <td>MAE Efficiency</td>
                            <td>{performance.mae_efficiency:.2f}</td>
                            <td><span class="badge badge-{'excellent' if performance.mae_efficiency > 3 else 'good' if performance.mae_efficiency > 2 else 'warning' if performance.mae_efficiency > 1 else 'poor'}">
                                {'Excellent' if performance.mae_efficiency > 3 else 'Good' if performance.mae_efficiency > 2 else 'Fair' if performance.mae_efficiency > 1 else 'Poor'}
                            </span></td>
                        </tr>
                        <tr>
                            <td>MFE Realization</td>
                            <td>{performance.mfe_realization:.1%}</td>
                            <td><span class="badge badge-{'excellent' if performance.mfe_realization > 0.7 else 'good' if performance.mfe_realization > 0.5 else 'warning' if performance.mfe_realization > 0.3 else 'poor'}">
                                {'Excellent' if performance.mfe_realization > 0.7 else 'Good' if performance.mfe_realization > 0.5 else 'Fair' if performance.mfe_realization > 0.3 else 'Poor'}
                            </span></td>
                        </tr>
                    </table>
                </div>
                
                <div>
                    <h3>Trade Quality Distribution</h3>
                    <div class="chart-container">
                        <canvas id="maeChart" width="400" height="300"></canvas>
                    </div>
                </div>
            </div>
            
            <div style="margin-top: 25px;">
                <h3>💡 MAE/MFE Insights</h3>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(350px, 1fr)); gap: 15px;">
                    {"<div class='recommendation'><span class='recommendation-icon'>✅</span><strong>Excellent Efficiency:</strong> Your MAE efficiency of " + f"{performance.mae_efficiency:.2f}" + " indicates very good trade timing and risk management.</div>" if performance.mae_efficiency > 2.5 else ""}
                    {"<div class='recommendation'><span class='recommendation-icon'>⚠️</span><strong>Low Efficiency:</strong> MAE efficiency of " + f"{performance.mae_efficiency:.2f}" + " suggests room for improvement in entry timing and stop management.</div>" if performance.mae_efficiency < 1.5 else ""}
                    {"<div class='recommendation'><span class='recommendation-icon'>🎯</span><strong>Good Realization:</strong> You're capturing " + f"{performance.mfe_realization:.1%}" + " of favorable moves - solid profit taking.</div>" if performance.mfe_realization > 0.6 else ""}
                    {"<div class='recommendation'><span class='recommendation-icon'>📈</span><strong>Improve Exits:</strong> Only " + f"{performance.mfe_realization:.1%}" + " MFE realization suggests leaving profits on the table.</div>" if performance.mfe_realization < 0.4 else ""}
                </div>
            </div>
        </div>
        """
    
    def _generate_risk_analysis(self, performance: InstrumentPerformance) -> str:
        """Generate risk analysis section"""
        
        return f"""
        <div class="section">
            <h2>⚠️ Risk Analysis</h2>
            
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 25px;">
                <div>
                    <h3>Drawdown Metrics</h3>
                    <div style="margin: 15px 0;">
                        <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                            <span>Max Drawdown</span>
                            <span><strong>{performance.max_drawdown:.1f}%</strong></span>
                        </div>
                        <div class="progress-bar">
                            <div class="progress-fill" style="width: {min(100, performance.max_drawdown * 5)}%;"></div>
                        </div>
                    </div>
                    
                    <table>
                        <tr><th>Risk Metric</th><th>Value</th><th>Rating</th></tr>
                        <tr>
                            <td>Value at Risk (95%)</td>
                            <td>{performance.value_at_risk_95:.2f}bp</td>
                            <td><span class="badge badge-{'good' if abs(performance.value_at_risk_95) < 50 else 'warning' if abs(performance.value_at_risk_95) < 100 else 'poor'}">
                                {'Low' if abs(performance.value_at_risk_95) < 50 else 'Moderate' if abs(performance.value_at_risk_95) < 100 else 'High'}
                            </span></td>
                        </tr>
                        <tr>
                            <td>Conditional VaR</td>
                            <td>{performance.conditional_var_95:.2f}bp</td>
                            <td><span class="badge badge-{'good' if abs(performance.conditional_var_95) < 75 else 'warning' if abs(performance.conditional_var_95) < 150 else 'poor'}">
                                {'Low' if abs(performance.conditional_var_95) < 75 else 'Moderate' if abs(performance.conditional_var_95) < 150 else 'High'}
                            </span></td>
                        </tr>
                        <tr>
                            <td>Lake Ratio</td>
                            <td>{performance.lake_ratio:.1%}</td>
                            <td><span class="badge badge-{'excellent' if performance.lake_ratio < 0.3 else 'good' if performance.lake_ratio < 0.5 else 'warning' if performance.lake_ratio < 0.7 else 'poor'}">
                                {'Excellent' if performance.lake_ratio < 0.3 else 'Good' if performance.lake_ratio < 0.5 else 'Fair' if performance.lake_ratio < 0.7 else 'Poor'}
                            </span></td>
                        </tr>
                    </table>
                </div>
                
                <div>
                    <h3>Risk-Adjusted Returns</h3>
                    <div class="chart-container">
                        <canvas id="riskChart" width="400" height="300"></canvas>
                    </div>
                    
                    <div style="margin-top: 15px;">
                        <h4>Risk Assessment</h4>
                        <ul style="margin: 10px 0; padding-left: 20px;">
                            <li><strong>Sortino Ratio:</strong> {performance.sortino_ratio:.2f} 
                                <span class="badge badge-{'excellent' if performance.sortino_ratio > 2 else 'good' if performance.sortino_ratio > 1 else 'warning' if performance.sortino_ratio > 0 else 'poor'}">
                                    {'Excellent' if performance.sortino_ratio > 2 else 'Good' if performance.sortino_ratio > 1 else 'Fair' if performance.sortino_ratio > 0 else 'Poor'}
                                </span>
                            </li>
                            <li><strong>Calmar Ratio:</strong> {performance.calmar_ratio:.2f}
                                <span class="badge badge-{'excellent' if performance.calmar_ratio > 3 else 'good' if performance.calmar_ratio > 1 else 'warning' if performance.calmar_ratio > 0 else 'poor'}">
                                    {'Excellent' if performance.calmar_ratio > 3 else 'Good' if performance.calmar_ratio > 1 else 'Fair' if performance.calmar_ratio > 0 else 'Poor'}
                                </span>
                            </li>
                        </ul>
                    </div>
                </div>
            </div>
        </div>
        """
    
    def _generate_efficiency_analysis(self, performance: InstrumentPerformance) -> str:
        """Generate efficiency analysis section"""
        
        return f"""
        <div class="section">
            <h2>⚡ Efficiency Analysis</h2>
            
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(350px, 1fr)); gap: 25px;">
                <div>
                    <h3>Trading Efficiency</h3>
                    <table>
                        <tr><th>Efficiency Metric</th><th>Value</th><th>Grade</th></tr>
                        <tr>
                            <td>Trade Efficiency</td>
                            <td>{performance.trade_efficiency:.2f}</td>
                            <td><span class="badge badge-{'excellent' if performance.trade_efficiency > 2 else 'good' if performance.trade_efficiency > 1.5 else 'warning' if performance.trade_efficiency > 1 else 'poor'}">
                                {'A+' if performance.trade_efficiency > 2 else 'B+' if performance.trade_efficiency > 1.5 else 'C+' if performance.trade_efficiency > 1 else 'D'}
                            </span></td>
                        </tr>
                        <tr>
                            <td>Efficiency Ratio</td>
                            <td>{performance.efficiency_ratio:.2f}</td>
                            <td><span class="badge badge-{'excellent' if performance.efficiency_ratio > 0.8 else 'good' if performance.efficiency_ratio > 0.6 else 'warning' if performance.efficiency_ratio > 0.4 else 'poor'}">
                                {'A+' if performance.efficiency_ratio > 0.8 else 'B+' if performance.efficiency_ratio > 0.6 else 'C+' if performance.efficiency_ratio > 0.4 else 'D'}
                            </span></td>
                        </tr>
                        <tr>
                            <td>K-Factor</td>
                            <td>{performance.k_factor:.2f}</td>
                            <td><span class="badge badge-{'excellent' if performance.k_factor > 2 else 'good' if performance.k_factor > 1 else 'warning' if performance.k_factor > 0 else 'poor'}">
                                {'Excellent' if performance.k_factor > 2 else 'Good' if performance.k_factor > 1 else 'Fair' if performance.k_factor > 0 else 'Poor'}
                            </span></td>
                        </tr>
                    </table>
                </div>
                
                <div>
                    <h3>Time Analysis</h3>
                    <table>
                        <tr><th>Time Metric</th><th>Value</th><th>Note</th></tr>
                        <tr>
                            <td>Avg Hold Time</td>
                            <td>{performance.avg_hold_time/60:.1f}h</td>
                            <td>Overall average</td>
                        </tr>
                        <tr>
                            <td>Win Hold Time</td>
                            <td>{performance.win_hold_time/60:.1f}h</td>
                            <td>Winning trades</td>
                        </tr>
                        <tr>
                            <td>Loss Hold Time</td>
                            <td>{performance.loss_hold_time/60:.1f}h</td>
                            <td>Losing trades</td>
                        </tr>
                        <tr>
                            <td>Max Consecutive Wins</td>
                            <td>{performance.max_consecutive_wins}</td>
                            <td>Best streak</td>
                        </tr>
                        <tr>
                            <td>Max Consecutive Losses</td>
                            <td>{performance.max_consecutive_losses}</td>
                            <td>Worst streak</td>
                        </tr>
                    </table>
                </div>
            </div>
        </div>
        """
    
    def _generate_trade_analysis(self, trades: List[TradeMetrics]) -> str:
        """Generate detailed trade analysis"""
        
        if len(trades) == 0:
            return ""
        
        # Calculate trade statistics
        winning_trades = [t for t in trades if t.net_pnl > 0]
        losing_trades = [t for t in trades if t.net_pnl < 0]
        
        # Best and worst trades
        best_trade = max(trades, key=lambda t: t.net_pnl) if trades else None
        worst_trade = min(trades, key=lambda t: t.net_pnl) if trades else None
        
        return f"""
        <div class="section">
            <h2>📋 Trade Analysis</h2>
            
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 25px;">
                <div>
                    <h3>Trade Distribution</h3>
                    <div class="chart-container">
                        <canvas id="tradeDistChart" width="400" height="300"></canvas>
                    </div>
                </div>
                
                <div>
                    <h3>Notable Trades</h3>
                    <div style="margin-bottom: 20px;">
                        <h4 style="color: green;">🏆 Best Trade</h4>
                        {f"<p><strong>Trade ID:</strong> {best_trade.trade_id}<br>" +
                         f"<strong>P&L:</strong> +{best_trade.net_pnl:.2f}bp<br>" +
                         f"<strong>Efficiency:</strong> {best_trade.efficiency:.2f}<br>" +
                         f"<strong>Hold Time:</strong> {best_trade.hold_time_minutes/60:.1f}h</p>" if best_trade else "No data"}
                    </div>
                    
                    <div>
                        <h4 style="color: red;">📉 Worst Trade</h4>
                        {f"<p><strong>Trade ID:</strong> {worst_trade.trade_id}<br>" +
                         f"<strong>P&L:</strong> {worst_trade.net_pnl:.2f}bp<br>" +
                         f"<strong>MAE:</strong> {worst_trade.mae:.2f}bp<br>" +
                         f"<strong>Hold Time:</strong> {worst_trade.hold_time_minutes/60:.1f}h</p>" if worst_trade else "No data"}
                    </div>
                </div>
            </div>
            
            <div style="margin-top: 25px;">
                <h3>Recent Trades (Last 10)</h3>
                <table>
                    <tr>
                        <th>Trade ID</th>
                        <th>Entry Time</th>
                        <th>Direction</th>
                        <th>P&L (bp)</th>
                        <th>MAE</th>
                        <th>MFE</th>
                        <th>Efficiency</th>
                        <th>Hold Time</th>
                    </tr>
                    {self._generate_recent_trades_rows(trades[-10:] if len(trades) > 10 else trades)}
                </table>
            </div>
        </div>
        """
    
    def _generate_recent_trades_rows(self, recent_trades: List[TradeMetrics]) -> str:
        """Generate table rows for recent trades"""
        
        rows = []
        for trade in recent_trades:
            direction = "LONG" if trade.direction > 0 else "SHORT"
            pnl_class = "positive" if trade.net_pnl > 0 else "negative"
            
            row = f"""
            <tr>
                <td>{trade.trade_id}</td>
                <td>{trade.entry_time.strftime('%m/%d %H:%M')}</td>
                <td>{direction}</td>
                <td class="{pnl_class}">{trade.net_pnl:+.2f}</td>
                <td>{trade.mae:.2f}</td>
                <td>{trade.mfe:.2f}</td>
                <td>{trade.efficiency:.2f}</td>
                <td>{trade.hold_time_minutes/60:.1f}h</td>
            </tr>
            """
            rows.append(row)
        
        return "".join(rows)
    
    def _generate_recommendations(self, performance: InstrumentPerformance) -> str:
        """Generate actionable recommendations"""
        
        recommendations = []
        
        # Performance-based recommendations
        if performance.win_rate < 50:
            recommendations.append({
                "icon": "🎯",
                "title": "Improve Win Rate",
                "description": f"Current win rate of {performance.win_rate:.1f}% is below breakeven. Focus on better entry signals and market timing."
            })
        
        if performance.mae_efficiency < 1.5:
            recommendations.append({
                "icon": "⏰",
                "title": "Optimize Entry Timing",
                "description": f"MAE efficiency of {performance.mae_efficiency:.2f} suggests entries could be better timed. Consider tighter entry criteria."
            })
        
        if performance.mfe_realization < 0.5:
            recommendations.append({
                "icon": "💰",
                "title": "Improve Profit Taking",
                "description": f"Only {performance.mfe_realization:.1%} MFE realization indicates missed profit opportunities. Review exit strategy."
            })
        
        if performance.max_drawdown > 15:
            recommendations.append({
                "icon": "🛡️",
                "title": "Enhance Risk Management",
                "description": f"Max drawdown of {performance.max_drawdown:.1f}% is high. Consider tighter stop losses and position sizing."
            })
        
        if performance.avg_loss > abs(performance.avg_win * 2):
            recommendations.append({
                "icon": "✂️",
                "title": "Cut Losses Faster",
                "description": f"Average loss of {performance.avg_loss:.2f}bp is too large relative to average win. Implement stricter stop management."
            })
        
        if performance.kelly_percentage > 0.25:
            recommendations.append({
                "icon": "⚖️",
                "title": "Reduce Position Size",
                "description": f"Kelly criterion suggests {performance.kelly_percentage:.1%} allocation, but consider more conservative sizing for risk management."
            })
        
        # Positive reinforcements
        if performance.sharpe_ratio > 1.5:
            recommendations.append({
                "icon": "⭐",
                "title": "Excellent Risk-Adjusted Returns",
                "description": f"Sharpe ratio of {performance.sharpe_ratio:.2f} indicates strong risk-adjusted performance. Maintain current approach."
            })
        
        if performance.omega_ratio > 2:
            recommendations.append({
                "icon": "🚀",
                "title": "Superior Return Profile",
                "description": f"Omega ratio of {performance.omega_ratio:.2f} shows excellent return distribution. Consider scaling up gradually."
            })
        
        recommendations_html = ""
        for rec in recommendations:
            recommendations_html += f"""
            <div class='recommendation'>
                <span class='recommendation-icon'>{rec['icon']}</span>
                <strong>{rec['title']}:</strong> {rec['description']}
            </div>
            """
        
        return f"""
        <div class="section">
            <h2>💡 Recommendations & Action Items</h2>
            {recommendations_html if recommendations else "<p style='text-align: center; color: #666; font-style: italic;'>Performance metrics are within acceptable ranges. Continue monitoring and maintain current approach.</p>"}
        </div>
        """
    
    def _generate_footer(self) -> str:
        """Generate report footer"""
        
        return f"""
        <div class="footer">
            <p>Report generated by Advanced Performance Analytics Engine</p>
            <p>For questions or support, contact your trading system administrator</p>
            <p style="font-size: 0.8em; margin-top: 15px;">
                Disclaimer: This report is for informational purposes only. Past performance does not guarantee future results.
                Trading involves risk and may not be suitable for all investors.
            </p>
        </div>
        """
    
    def _generate_javascript(self, trades: List[TradeMetrics], performance: InstrumentPerformance) -> str:
        """Generate JavaScript for interactive charts"""
        
        # Prepare data for charts
        mae_data = [t.mae for t in trades]
        mfe_data = [t.mfe for t in trades]
        pnl_data = [t.net_pnl for t in trades]
        
        return f"""
        <script>
        // MAE/MFE Scatter Chart
        const maeCtx = document.getElementById('maeChart').getContext('2d');
        new Chart(maeCtx, {{
            type: 'scatter',
            data: {{
                datasets: [{{
                    label: 'Trades (MAE vs MFE)',
                    data: {[{"{'x': " + str(mae_data[i]) + ", 'y': " + str(mfe_data[i]) + "}"} for i in range(len(mae_data))]},
                    backgroundColor: 'rgba(46, 134, 171, 0.6)',
                    borderColor: 'rgba(46, 134, 171, 1)',
                    pointRadius: 4
                }}]
            }},
            options: {{
                responsive: true,
                plugins: {{
                    title: {{
                        display: true,
                        text: 'MAE vs MFE Distribution'
                    }}
                }},
                scales: {{
                    x: {{
                        title: {{
                            display: true,
                            text: 'MAE (bp)'
                        }}
                    }},
                    y: {{
                        title: {{
                            display: true,
                            text: 'MFE (bp)'
                        }}
                    }}
                }}
            }}
        }});
        
        // Risk Metrics Chart
        const riskCtx = document.getElementById('riskChart').getContext('2d');
        new Chart(riskCtx, {{
            type: 'radar',
            data: {{
                labels: ['Sharpe Ratio', 'Sortino Ratio', 'Calmar Ratio', 'Omega Ratio', 'K-Factor'],
                datasets: [{{
                    label: 'Performance Metrics',
                    data: [{performance.sharpe_ratio:.2f}, {performance.sortino_ratio:.2f}, {performance.calmar_ratio:.2f}, {performance.omega_ratio:.2f}, {performance.k_factor:.2f}],
                    backgroundColor: 'rgba(162, 59, 114, 0.2)',
                    borderColor: 'rgba(162, 59, 114, 1)',
                    pointBackgroundColor: 'rgba(162, 59, 114, 1)',
                    pointBorderColor: '#fff',
                    pointHoverBackgroundColor: '#fff',
                    pointHoverBorderColor: 'rgba(162, 59, 114, 1)'
                }}]
            }},
            options: {{
                responsive: true,
                plugins: {{
                    title: {{
                        display: true,
                        text: 'Risk-Adjusted Performance Radar'
                    }}
                }},
                scales: {{
                    r: {{
                        beginAtZero: true
                    }}
                }}
            }}
        }});
        
        // Trade Distribution Chart
        const tradeDistCtx = document.getElementById('tradeDistChart').getContext('2d');
        const pnlBuckets = {{
            'Large Loss (< -100bp)': {len([p for p in pnl_data if p < -100])},
            'Loss (-100 to 0bp)': {len([p for p in pnl_data if -100 <= p < 0])},
            'Small Win (0 to 50bp)': {len([p for p in pnl_data if 0 <= p < 50])},
            'Good Win (50 to 200bp)': {len([p for p in pnl_data if 50 <= p < 200])},
            'Large Win (> 200bp)': {len([p for p in pnl_data if p >= 200])}
        }};
        
        new Chart(tradeDistCtx, {{
            type: 'doughnut',
            data: {{
                labels: Object.keys(pnlBuckets),
                datasets: [{{
                    data: Object.values(pnlBuckets),
                    backgroundColor: [
                        '#C73E1D',
                        '#F18F01', 
                        '#A23B72',
                        '#2E86AB',
                        '#10B981'
                    ]
                }}]
            }},
            options: {{
                responsive: true,
                plugins: {{
                    title: {{
                        display: true,
                        text: 'P&L Distribution'
                    }},
                    legend: {{
                        position: 'bottom'
                    }}
                }}
            }}
        }});
        </script>
        """

def test_instrument_performance_reporter():
    """Test the performance reporter"""
    
    print("🧪 INSTRUMENT PERFORMANCE REPORTER TEST")
    print("="*80)
    
    # Create sample trades
    sample_trades = []
    base_time = datetime.now() - timedelta(days=30)
    
    for i in range(50):
        # Simulate realistic trading data
        direction = 1 if random.random() > 0.5 else -1
        entry_price = 95000 + random.gauss(0, 2000)
        
        # Simulate trade outcome
        if random.random() < 0.55:  # 55% win rate
            # Winning trade
            pnl = random.uniform(20, 200)
            exit_price = entry_price + (pnl * direction / 10)
            mfe = pnl * random.uniform(1.2, 3.0)
            mae = pnl * random.uniform(0.1, 0.8)
        else:
            # Losing trade
            pnl = -random.uniform(30, 150)
            exit_price = entry_price + (pnl * direction / 10)
            mae = abs(pnl) * random.uniform(1.0, 2.5)
            mfe = abs(pnl) * random.uniform(0.1, 0.6)
        
        trade = TradeMetrics(
            trade_id=f"BTCUSD_{i:03d}",
            entry_time=base_time + timedelta(hours=i*2),
            exit_time=base_time + timedelta(hours=i*2 + random.uniform(0.5, 24)),
            symbol="BTCUSD",
            direction=direction,
            entry_price=entry_price,
            exit_price=exit_price,
            quantity=random.uniform(0.1, 2.0),
            highest_price=entry_price + (mfe if direction > 0 else -mae),
            lowest_price=entry_price - (mae if direction > 0 else -mfe),
            gross_pnl=pnl * 1.1,  # Before commission
            commission=2.0,
            slippage=1.0,
            net_pnl=pnl
        )
        
        sample_trades.append(trade)
    
    # Initialize reporter
    config = ReportConfig(
        include_charts=True,
        theme="professional"
    )
    reporter = InstrumentPerformanceReporter(config)
    
    # Generate report
    output_path = "/tmp/BTCUSD_performance_report.html"
    generated_path = reporter.generate_report("BTCUSD", sample_trades, output_path)
    
    print(f"\n✅ Performance report generated!")
    print(f"📄 Report location: {generated_path}")
    print(f"🌐 Open in browser to view the interactive report")
    
    # Also try copying to user's Documents for easier access
    try:
        import shutil
        documents_path = "/mnt/c/Users/renie/Documents/BTCUSD_Performance_Report.html"
        shutil.copy(output_path, documents_path)
        print(f"📋 Report also copied to: {documents_path}")
    except Exception as e:
        print(f"⚠️ Could not copy to Documents: {e}")

if __name__ == "__main__":
    test_instrument_performance_reporter()