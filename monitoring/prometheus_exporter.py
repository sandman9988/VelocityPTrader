#!/usr/bin/env python3
"""
VelocityPTrader Prometheus Metrics Exporter

Exposes trading system metrics for Prometheus scraping.
Integrates with Grafana for comprehensive visualization.

Metrics Categories:
- Trading Performance: P&L, win rates, trade counts
- Agent Metrics: BERSERKER vs SNIPER comparison
- Market Regime: Physics-based regime states
- Risk Management: Drawdown, VaR, exposure
- System Health: Uptime, errors, latency
"""

import time
import threading
import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional
from http.server import HTTPServer, BaseHTTPRequestHandler
import logging

logger = logging.getLogger(__name__)

# Prometheus metrics format helper
class PrometheusMetric:
    """Helper class for Prometheus metric formatting"""

    GAUGE = "gauge"
    COUNTER = "counter"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"

    def __init__(self, name: str, help_text: str, metric_type: str = GAUGE):
        self.name = name
        self.help_text = help_text
        self.metric_type = metric_type
        self.values: Dict[str, float] = {}

    def set(self, value: float, labels: Dict[str, str] = None):
        """Set metric value with optional labels"""
        label_key = self._format_labels(labels) if labels else ""
        self.values[label_key] = value

    def inc(self, value: float = 1, labels: Dict[str, str] = None):
        """Increment counter metric"""
        label_key = self._format_labels(labels) if labels else ""
        self.values[label_key] = self.values.get(label_key, 0) + value

    def _format_labels(self, labels: Dict[str, str]) -> str:
        """Format labels for Prometheus"""
        if not labels:
            return ""
        parts = [f'{k}="{v}"' for k, v in sorted(labels.items())]
        return "{" + ",".join(parts) + "}"

    def to_prometheus(self) -> str:
        """Convert to Prometheus text format"""
        lines = []
        lines.append(f"# HELP {self.name} {self.help_text}")
        lines.append(f"# TYPE {self.name} {self.metric_type}")

        if not self.values:
            lines.append(f"{self.name} 0")
        else:
            for label_key, value in self.values.items():
                metric_name = f"{self.name}{label_key}"
                # Handle infinity and NaN
                if value == float('inf'):
                    value = 999999
                elif value == float('-inf'):
                    value = -999999
                elif value != value:  # NaN check
                    value = 0
                lines.append(f"{metric_name} {value}")

        return "\n".join(lines)


class TradingMetricsCollector:
    """Collects trading metrics from the VelocityPTrader system"""

    def __init__(self, db_path: str = "trading_logs.db"):
        self.db_path = db_path
        self.start_time = time.time()

        # Initialize all metrics
        self._init_metrics()

        logger.info("Trading Metrics Collector initialized")

    def _init_metrics(self):
        """Initialize all Prometheus metrics"""

        # === TRADING PERFORMANCE METRICS ===
        self.total_trades = PrometheusMetric(
            "velocitytrader_trades_total",
            "Total number of trades executed",
            PrometheusMetric.COUNTER
        )

        self.winning_trades = PrometheusMetric(
            "velocitytrader_trades_winning_total",
            "Total number of winning trades",
            PrometheusMetric.COUNTER
        )

        self.losing_trades = PrometheusMetric(
            "velocitytrader_trades_losing_total",
            "Total number of losing trades",
            PrometheusMetric.COUNTER
        )

        self.win_rate = PrometheusMetric(
            "velocitytrader_win_rate",
            "Current win rate percentage"
        )

        self.total_pnl = PrometheusMetric(
            "velocitytrader_pnl_total",
            "Total profit and loss in USD"
        )

        self.gross_profit = PrometheusMetric(
            "velocitytrader_gross_profit",
            "Total gross profit in USD"
        )

        self.gross_loss = PrometheusMetric(
            "velocitytrader_gross_loss",
            "Total gross loss in USD"
        )

        self.profit_factor = PrometheusMetric(
            "velocitytrader_profit_factor",
            "Profit factor (gross profit / gross loss)"
        )

        self.average_trade_pnl = PrometheusMetric(
            "velocitytrader_avg_trade_pnl",
            "Average P&L per trade in USD"
        )

        self.average_win = PrometheusMetric(
            "velocitytrader_avg_win",
            "Average winning trade P&L in USD"
        )

        self.average_loss = PrometheusMetric(
            "velocitytrader_avg_loss",
            "Average losing trade P&L in USD"
        )

        self.largest_win = PrometheusMetric(
            "velocitytrader_largest_win",
            "Largest single winning trade in USD"
        )

        self.largest_loss = PrometheusMetric(
            "velocitytrader_largest_loss",
            "Largest single losing trade in USD"
        )

        self.trade_duration_avg = PrometheusMetric(
            "velocitytrader_trade_duration_avg_seconds",
            "Average trade duration in seconds"
        )

        # === AGENT METRICS ===
        self.agent_trades = PrometheusMetric(
            "velocitytrader_agent_trades_total",
            "Total trades per agent",
            PrometheusMetric.COUNTER
        )

        self.agent_win_rate = PrometheusMetric(
            "velocitytrader_agent_win_rate",
            "Win rate per agent"
        )

        self.agent_pnl = PrometheusMetric(
            "velocitytrader_agent_pnl_total",
            "Total P&L per agent"
        )

        self.agent_avg_confidence = PrometheusMetric(
            "velocitytrader_agent_avg_confidence",
            "Average signal confidence per agent"
        )

        self.agent_signals_generated = PrometheusMetric(
            "velocitytrader_agent_signals_total",
            "Total signals generated per agent",
            PrometheusMetric.COUNTER
        )

        self.agent_active_positions = PrometheusMetric(
            "velocitytrader_agent_active_positions",
            "Current active positions per agent"
        )

        # === MARKET REGIME METRICS ===
        self.regime_current = PrometheusMetric(
            "velocitytrader_regime_current",
            "Current market regime (encoded: 1=CHAOTIC, 2=UNDERDAMPED, 3=CRITICALLY_DAMPED, 4=OVERDAMPED)"
        )

        self.regime_trades = PrometheusMetric(
            "velocitytrader_regime_trades_total",
            "Trades per market regime",
            PrometheusMetric.COUNTER
        )

        self.regime_win_rate = PrometheusMetric(
            "velocitytrader_regime_win_rate",
            "Win rate per market regime"
        )

        self.regime_pnl = PrometheusMetric(
            "velocitytrader_regime_pnl",
            "P&L per market regime"
        )

        self.regime_changes = PrometheusMetric(
            "velocitytrader_regime_changes_total",
            "Total regime changes detected",
            PrometheusMetric.COUNTER
        )

        # === PHYSICS INDICATORS ===
        self.physics_momentum = PrometheusMetric(
            "velocitytrader_physics_momentum",
            "Current market momentum indicator"
        )

        self.physics_velocity = PrometheusMetric(
            "velocitytrader_physics_velocity",
            "Current price velocity indicator"
        )

        self.physics_acceleration = PrometheusMetric(
            "velocitytrader_physics_acceleration",
            "Current price acceleration indicator"
        )

        self.physics_energy = PrometheusMetric(
            "velocitytrader_physics_energy",
            "Current market energy indicator"
        )

        self.physics_damping_ratio = PrometheusMetric(
            "velocitytrader_physics_damping_ratio",
            "Current damping ratio"
        )

        self.physics_natural_frequency = PrometheusMetric(
            "velocitytrader_physics_natural_frequency",
            "Natural frequency of price oscillation"
        )

        # === SYMBOL METRICS ===
        self.symbol_trades = PrometheusMetric(
            "velocitytrader_symbol_trades_total",
            "Trades per trading symbol",
            PrometheusMetric.COUNTER
        )

        self.symbol_pnl = PrometheusMetric(
            "velocitytrader_symbol_pnl",
            "P&L per trading symbol"
        )

        self.symbol_win_rate = PrometheusMetric(
            "velocitytrader_symbol_win_rate",
            "Win rate per trading symbol"
        )

        self.symbol_current_price = PrometheusMetric(
            "velocitytrader_symbol_price",
            "Current price per symbol"
        )

        self.symbol_spread = PrometheusMetric(
            "velocitytrader_symbol_spread",
            "Current spread per symbol in points"
        )

        # === RISK MANAGEMENT METRICS ===
        self.account_balance = PrometheusMetric(
            "velocitytrader_account_balance",
            "Current account balance in USD"
        )

        self.account_equity = PrometheusMetric(
            "velocitytrader_account_equity",
            "Current account equity in USD"
        )

        self.max_drawdown = PrometheusMetric(
            "velocitytrader_max_drawdown",
            "Maximum drawdown percentage"
        )

        self.current_drawdown = PrometheusMetric(
            "velocitytrader_current_drawdown",
            "Current drawdown percentage"
        )

        self.risk_exposure = PrometheusMetric(
            "velocitytrader_risk_exposure",
            "Current risk exposure percentage"
        )

        self.var_95 = PrometheusMetric(
            "velocitytrader_var_95",
            "Value at Risk (95% confidence) in USD"
        )

        self.sharpe_ratio = PrometheusMetric(
            "velocitytrader_sharpe_ratio",
            "Sharpe ratio"
        )

        self.sortino_ratio = PrometheusMetric(
            "velocitytrader_sortino_ratio",
            "Sortino ratio"
        )

        self.calmar_ratio = PrometheusMetric(
            "velocitytrader_calmar_ratio",
            "Calmar ratio"
        )

        self.recovery_factor = PrometheusMetric(
            "velocitytrader_recovery_factor",
            "Recovery factor (net profit / max drawdown)"
        )

        self.consecutive_wins = PrometheusMetric(
            "velocitytrader_consecutive_wins_max",
            "Maximum consecutive winning trades"
        )

        self.consecutive_losses = PrometheusMetric(
            "velocitytrader_consecutive_losses_max",
            "Maximum consecutive losing trades"
        )

        # === REINFORCEMENT LEARNING METRICS ===
        self.rl_exploration_rate = PrometheusMetric(
            "velocitytrader_rl_exploration_rate",
            "Current RL exploration rate (epsilon)"
        )

        self.rl_learning_progress = PrometheusMetric(
            "velocitytrader_rl_learning_progress",
            "RL learning progress (0-1)"
        )

        self.rl_experience_buffer_size = PrometheusMetric(
            "velocitytrader_rl_experience_buffer_size",
            "RL experience replay buffer size"
        )

        self.rl_model_confidence = PrometheusMetric(
            "velocitytrader_rl_model_confidence",
            "RL model confidence level"
        )

        self.rl_total_experiences = PrometheusMetric(
            "velocitytrader_rl_total_experiences",
            "Total RL experiences collected",
            PrometheusMetric.COUNTER
        )

        self.rl_training_episodes = PrometheusMetric(
            "velocitytrader_rl_training_episodes",
            "Total RL training episodes",
            PrometheusMetric.COUNTER
        )

        # === SYSTEM HEALTH METRICS ===
        self.system_uptime = PrometheusMetric(
            "velocitytrader_uptime_seconds",
            "System uptime in seconds",
            PrometheusMetric.COUNTER
        )

        self.system_status = PrometheusMetric(
            "velocitytrader_system_status",
            "System status (1=running, 0=stopped)"
        )

        self.ticks_processed = PrometheusMetric(
            "velocitytrader_ticks_processed_total",
            "Total market ticks processed",
            PrometheusMetric.COUNTER
        )

        self.signals_generated = PrometheusMetric(
            "velocitytrader_signals_generated_total",
            "Total trading signals generated",
            PrometheusMetric.COUNTER
        )

        self.errors_total = PrometheusMetric(
            "velocitytrader_errors_total",
            "Total errors encountered",
            PrometheusMetric.COUNTER
        )

        self.errors_critical = PrometheusMetric(
            "velocitytrader_errors_critical_total",
            "Critical errors encountered",
            PrometheusMetric.COUNTER
        )

        self.mt5_connection_status = PrometheusMetric(
            "velocitytrader_mt5_connection",
            "MT5 connection status (1=connected, 0=disconnected)"
        )

        self.active_symbols = PrometheusMetric(
            "velocitytrader_active_symbols",
            "Number of actively traded symbols"
        )

        self.websocket_clients = PrometheusMetric(
            "velocitytrader_websocket_clients",
            "Connected WebSocket clients"
        )

        self.db_query_latency = PrometheusMetric(
            "velocitytrader_db_query_latency_ms",
            "Database query latency in milliseconds"
        )

        # === TIMEFRAME METRICS ===
        self.timeframe_trades = PrometheusMetric(
            "velocitytrader_timeframe_trades_total",
            "Trades per timeframe",
            PrometheusMetric.COUNTER
        )

        self.timeframe_win_rate = PrometheusMetric(
            "velocitytrader_timeframe_win_rate",
            "Win rate per timeframe"
        )

        self.timeframe_pnl = PrometheusMetric(
            "velocitytrader_timeframe_pnl",
            "P&L per timeframe"
        )

    def collect_metrics(self):
        """Collect all current metrics from the system"""

        try:
            # Collect from database
            self._collect_from_database()

            # Collect from final_stats.json if exists
            self._collect_from_stats_file()

            # Update system metrics
            self._update_system_metrics()

        except Exception as e:
            logger.error(f"Error collecting metrics: {e}")
            self.errors_total.inc()

    def _collect_from_database(self):
        """Collect metrics from SQLite database"""

        if not Path(self.db_path).exists():
            return

        start_time = time.time()

        try:
            with sqlite3.connect(self.db_path) as conn:
                # Trade statistics
                trade_stats = conn.execute('''
                    SELECT
                        COUNT(*) as total,
                        SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins,
                        SUM(CASE WHEN pnl < 0 THEN 1 ELSE 0 END) as losses,
                        SUM(pnl) as total_pnl,
                        SUM(CASE WHEN pnl > 0 THEN pnl ELSE 0 END) as gross_profit,
                        SUM(CASE WHEN pnl < 0 THEN ABS(pnl) ELSE 0 END) as gross_loss,
                        AVG(pnl) as avg_pnl,
                        AVG(CASE WHEN pnl > 0 THEN pnl END) as avg_win,
                        AVG(CASE WHEN pnl < 0 THEN ABS(pnl) END) as avg_loss,
                        MAX(pnl) as largest_win,
                        MIN(pnl) as largest_loss,
                        AVG(trade_duration) as avg_duration
                    FROM trade_logs WHERE pnl IS NOT NULL
                ''').fetchone()

                if trade_stats and trade_stats[0] > 0:
                    total = trade_stats[0]
                    wins = trade_stats[1] or 0
                    losses = trade_stats[2] or 0

                    self.total_trades.set(total)
                    self.winning_trades.set(wins)
                    self.losing_trades.set(losses)
                    self.win_rate.set((wins / total * 100) if total > 0 else 0)
                    self.total_pnl.set(trade_stats[3] or 0)
                    self.gross_profit.set(trade_stats[4] or 0)
                    self.gross_loss.set(trade_stats[5] or 0)

                    gross_profit = trade_stats[4] or 0
                    gross_loss = trade_stats[5] or 1
                    self.profit_factor.set(gross_profit / gross_loss if gross_loss > 0 else 0)

                    self.average_trade_pnl.set(trade_stats[6] or 0)
                    self.average_win.set(trade_stats[7] or 0)
                    self.average_loss.set(trade_stats[8] or 0)
                    self.largest_win.set(trade_stats[9] or 0)
                    self.largest_loss.set(trade_stats[10] or 0)
                    self.trade_duration_avg.set(trade_stats[11] or 0)

                # Agent-specific statistics
                for agent in ['BERSERKER', 'SNIPER']:
                    agent_stats = conn.execute('''
                        SELECT
                            COUNT(*) as total,
                            SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins,
                            SUM(pnl) as total_pnl,
                            AVG(confidence) as avg_confidence
                        FROM trade_logs
                        WHERE agent_id = ? AND pnl IS NOT NULL
                    ''', (agent,)).fetchone()

                    if agent_stats and agent_stats[0] > 0:
                        labels = {"agent": agent.lower()}
                        total = agent_stats[0]
                        wins = agent_stats[1] or 0

                        self.agent_trades.set(total, labels)
                        self.agent_win_rate.set((wins / total * 100) if total > 0 else 0, labels)
                        self.agent_pnl.set(agent_stats[2] or 0, labels)
                        self.agent_avg_confidence.set(agent_stats[3] or 0, labels)

                # Symbol-specific statistics
                symbol_stats = conn.execute('''
                    SELECT
                        instrument,
                        COUNT(*) as total,
                        SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins,
                        SUM(pnl) as total_pnl
                    FROM trade_logs
                    WHERE pnl IS NOT NULL
                    GROUP BY instrument
                ''').fetchall()

                for row in symbol_stats:
                    symbol = row[0] or "UNKNOWN"
                    labels = {"symbol": symbol.replace("+", "")}
                    total = row[1]
                    wins = row[2] or 0

                    self.symbol_trades.set(total, labels)
                    self.symbol_win_rate.set((wins / total * 100) if total > 0 else 0, labels)
                    self.symbol_pnl.set(row[3] or 0, labels)

                # Error statistics
                error_stats = conn.execute('''
                    SELECT
                        COUNT(*) as total,
                        SUM(CASE WHEN severity = 'CRITICAL' THEN 1 ELSE 0 END) as critical
                    FROM error_logs
                ''').fetchone()

                if error_stats:
                    self.errors_total.set(error_stats[0] or 0)
                    self.errors_critical.set(error_stats[1] or 0)

                # Performance metrics from performance_logs
                perf_stats = conn.execute('''
                    SELECT * FROM performance_logs
                    ORDER BY timestamp DESC LIMIT 1
                ''').fetchone()

                if perf_stats and len(perf_stats) > 17:
                    self.sharpe_ratio.set(perf_stats[7] or 0)
                    self.max_drawdown.set((perf_stats[8] or 0) * 100)
                    self.calmar_ratio.set(perf_stats[17] or 0)

            # Track query latency
            latency_ms = (time.time() - start_time) * 1000
            self.db_query_latency.set(latency_ms)

        except Exception as e:
            logger.error(f"Database collection error: {e}")

    def _collect_from_stats_file(self):
        """Collect metrics from final_stats.json"""

        stats_file = Path("final_stats.json")
        if not stats_file.exists():
            return

        try:
            with open(stats_file) as f:
                stats = json.load(f)

            self.account_balance.set(stats.get('final_balance', 10000))
            self.account_equity.set(stats.get('final_equity', 10000))
            self.max_drawdown.set(stats.get('max_drawdown', 0) * 100)
            self.ticks_processed.set(stats.get('total_ticks_processed', 0))

            # Physics indicators if available
            if 'physics_state' in stats:
                physics = stats['physics_state']
                self.physics_momentum.set(physics.get('momentum', 0))
                self.physics_velocity.set(physics.get('velocity', 0))
                self.physics_acceleration.set(physics.get('acceleration', 0))
                self.physics_energy.set(physics.get('energy', 0))
                self.physics_damping_ratio.set(physics.get('damping_ratio', 0))

        except Exception as e:
            logger.error(f"Stats file collection error: {e}")

    def _update_system_metrics(self):
        """Update system-level metrics"""

        uptime = time.time() - self.start_time
        self.system_uptime.set(uptime)
        self.system_status.set(1)  # Running

    def get_metrics_output(self) -> str:
        """Generate Prometheus-formatted metrics output"""

        # Collect fresh metrics
        self.collect_metrics()

        # Build output
        metrics = [
            # Trading Performance
            self.total_trades,
            self.winning_trades,
            self.losing_trades,
            self.win_rate,
            self.total_pnl,
            self.gross_profit,
            self.gross_loss,
            self.profit_factor,
            self.average_trade_pnl,
            self.average_win,
            self.average_loss,
            self.largest_win,
            self.largest_loss,
            self.trade_duration_avg,

            # Agent Metrics
            self.agent_trades,
            self.agent_win_rate,
            self.agent_pnl,
            self.agent_avg_confidence,
            self.agent_signals_generated,
            self.agent_active_positions,

            # Market Regime
            self.regime_current,
            self.regime_trades,
            self.regime_win_rate,
            self.regime_pnl,
            self.regime_changes,

            # Physics Indicators
            self.physics_momentum,
            self.physics_velocity,
            self.physics_acceleration,
            self.physics_energy,
            self.physics_damping_ratio,
            self.physics_natural_frequency,

            # Symbol Metrics
            self.symbol_trades,
            self.symbol_pnl,
            self.symbol_win_rate,
            self.symbol_current_price,
            self.symbol_spread,

            # Risk Management
            self.account_balance,
            self.account_equity,
            self.max_drawdown,
            self.current_drawdown,
            self.risk_exposure,
            self.var_95,
            self.sharpe_ratio,
            self.sortino_ratio,
            self.calmar_ratio,
            self.recovery_factor,
            self.consecutive_wins,
            self.consecutive_losses,

            # RL Metrics
            self.rl_exploration_rate,
            self.rl_learning_progress,
            self.rl_experience_buffer_size,
            self.rl_model_confidence,
            self.rl_total_experiences,
            self.rl_training_episodes,

            # System Health
            self.system_uptime,
            self.system_status,
            self.ticks_processed,
            self.signals_generated,
            self.errors_total,
            self.errors_critical,
            self.mt5_connection_status,
            self.active_symbols,
            self.websocket_clients,
            self.db_query_latency,

            # Timeframe Metrics
            self.timeframe_trades,
            self.timeframe_win_rate,
            self.timeframe_pnl,
        ]

        output_lines = []
        for metric in metrics:
            output_lines.append(metric.to_prometheus())
            output_lines.append("")

        return "\n".join(output_lines)


class PrometheusHandler(BaseHTTPRequestHandler):
    """HTTP handler for Prometheus metrics endpoint"""

    collector = None

    def do_GET(self):
        if self.path == '/metrics':
            self.send_response(200)
            self.send_header('Content-Type', 'text/plain; charset=utf-8')
            self.end_headers()

            if self.collector:
                output = self.collector.get_metrics_output()
                self.wfile.write(output.encode('utf-8'))
        elif self.path == '/health':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"status": "healthy"}).encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass  # Suppress request logging


class PrometheusExporter:
    """Main Prometheus exporter service"""

    def __init__(self, port: int = 9090, db_path: str = "trading_logs.db"):
        self.port = port
        self.collector = TradingMetricsCollector(db_path)
        self.server = None
        self.server_thread = None

        logger.info(f"Prometheus Exporter initialized on port {port}")

    def start(self):
        """Start the Prometheus exporter server"""

        PrometheusHandler.collector = self.collector

        self.server = HTTPServer(('0.0.0.0', self.port), PrometheusHandler)
        self.server_thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.server_thread.start()

        logger.info(f"Prometheus metrics available at http://localhost:{self.port}/metrics")
        print(f"Prometheus metrics available at http://localhost:{self.port}/metrics")

    def stop(self):
        """Stop the Prometheus exporter server"""

        if self.server:
            self.server.shutdown()

        logger.info("Prometheus Exporter stopped")


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="VelocityPTrader Prometheus Exporter")
    parser.add_argument("--port", type=int, default=9090, help="Port to expose metrics on")
    parser.add_argument("--db", type=str, default="trading_logs.db", help="Path to trading database")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    exporter = PrometheusExporter(port=args.port, db_path=args.db)
    exporter.start()

    print(f"VelocityPTrader Prometheus Exporter running on port {args.port}")
    print("Press Ctrl+C to stop")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down...")
        exporter.stop()


if __name__ == "__main__":
    main()
