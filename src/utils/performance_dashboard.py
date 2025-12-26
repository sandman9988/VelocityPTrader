#!/usr/bin/env python3
"""
PERFORMANCE MONITORING DASHBOARD
Real-time monitoring of VelocityTrader system performance
"""

import asyncio
import json
import time
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging
from http.server import HTTPServer, SimpleHTTPRequestHandler
import socketserver
import threading
import websockets
import signal

logger = logging.getLogger(__name__)

class PerformanceDashboard:
    """Real-time performance monitoring dashboard"""
    
    def __init__(self, db_path: str = "trading_logs.db", port: int = 8080):
        self.db_path = db_path
        self.port = port
        self.is_running = False
        self.http_server = None
        self.websocket_server = None
        self.connected_clients = set()
        
        # Performance cache
        self.cache = {
            'last_update': 0,
            'cache_duration': 5,  # 5 seconds
            'data': {}
        }
        
        logger.info(f"📊 Performance Dashboard initialized on port {port}")
    
    def start(self):
        """Start the dashboard"""
        if self.is_running:
            return
        
        self.is_running = True
        
        # Start HTTP server in thread
        http_thread = threading.Thread(target=self._start_http_server, daemon=True)
        http_thread.start()
        
        # Start WebSocket server
        ws_thread = threading.Thread(target=self._start_websocket_server, daemon=True)
        ws_thread.start()
        
        logger.info("✅ Performance Dashboard started")
        logger.info(f"   📊 Dashboard: http://localhost:{self.port}")
        logger.info(f"   🔌 WebSocket: ws://localhost:{self.port + 1}")
    
    def stop(self):
        """Stop the dashboard"""
        self.is_running = False
        
        if self.http_server:
            self.http_server.shutdown()
        
        logger.info("⏹️ Performance Dashboard stopped")
    
    def _start_http_server(self):
        """Start HTTP server"""
        class DashboardHandler(SimpleHTTPRequestHandler):
            def __init__(self, *args, dashboard=None, **kwargs):
                self.dashboard = dashboard
                super().__init__(*args, **kwargs)
            
            def do_GET(self):
                if self.path == '/':
                    self.send_dashboard_html()
                elif self.path == '/api/stats':
                    self.send_performance_stats()
                elif self.path == '/api/trades':
                    self.send_trade_data()
                elif self.path == '/api/agents':
                    self.send_agent_stats()
                else:
                    self.send_error(404)
            
            def send_dashboard_html(self):
                """Send main dashboard HTML"""
                html = self.dashboard._generate_dashboard_html()
                self.send_response(200)
                self.send_header('Content-type', 'text/html')
                self.send_header('Cache-Control', 'no-cache')
                self.end_headers()
                self.wfile.write(html.encode('utf-8'))
            
            def send_performance_stats(self):
                """Send performance statistics"""
                stats = self.dashboard._get_performance_stats()
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps(stats, indent=2).encode('utf-8'))
            
            def send_trade_data(self):
                """Send recent trade data"""
                trades = self.dashboard._get_recent_trades()
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps(trades, indent=2).encode('utf-8'))
            
            def send_agent_stats(self):
                """Send agent statistics"""
                agents = self.dashboard._get_agent_stats()
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps(agents, indent=2).encode('utf-8'))
            
            def log_message(self, format, *args):
                # Suppress request logging
                pass
        
        try:
            with socketserver.TCPServer(("", self.port), 
                                      lambda *args, **kwargs: DashboardHandler(*args, dashboard=self, **kwargs)) as httpd:
                self.http_server = httpd
                httpd.serve_forever()
        except Exception as e:
            logger.error(f"HTTP server error: {e}")
    
    def _start_websocket_server(self):
        """Start WebSocket server for real-time updates"""
        async def handle_client(websocket, path):
            self.connected_clients.add(websocket)
            logger.info(f"Client connected: {websocket.remote_address}")
            
            try:
                await websocket.wait_closed()
            finally:
                self.connected_clients.remove(websocket)
                logger.info(f"Client disconnected: {websocket.remote_address}")
        
        async def broadcast_updates():
            """Broadcast updates to connected clients"""
            while self.is_running:
                try:
                    if self.connected_clients:
                        stats = self._get_performance_stats()
                        message = json.dumps({
                            'type': 'update',
                            'data': stats,
                            'timestamp': datetime.now().isoformat()
                        })
                        
                        # Send to all connected clients
                        disconnected = set()
                        for client in self.connected_clients:
                            try:
                                await client.send(message)
                            except websockets.exceptions.ConnectionClosed:
                                disconnected.add(client)
                        
                        # Remove disconnected clients
                        self.connected_clients -= disconnected
                    
                    await asyncio.sleep(1)  # Update every second
                except Exception as e:
                    logger.error(f"WebSocket broadcast error: {e}")
        
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            # Start WebSocket server
            start_server = websockets.serve(handle_client, "localhost", self.port + 1)
            
            # Run both server and broadcast task
            loop.run_until_complete(asyncio.gather(
                start_server,
                broadcast_updates()
            ))
        except Exception as e:
            logger.error(f"WebSocket server error: {e}")
    
    def _get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics"""
        current_time = time.time()
        
        # Check cache
        if (current_time - self.cache['last_update'] < self.cache['cache_duration'] and 
            self.cache['data']):
            return self.cache['data']
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                # System metrics
                system_stats = self._get_system_metrics(conn)
                
                # Trade statistics
                trade_stats = self._get_trade_statistics(conn)
                
                # Performance metrics
                performance_stats = self._get_performance_metrics(conn)
                
                # Agent statistics
                agent_stats = self._get_agent_statistics(conn)
                
                # Error statistics
                error_stats = self._get_error_statistics(conn)
                
                stats = {
                    'timestamp': datetime.now().isoformat(),
                    'system': system_stats,
                    'trading': trade_stats,
                    'performance': performance_stats,
                    'agents': agent_stats,
                    'errors': error_stats,
                    'status': 'running' if self.is_running else 'stopped'
                }
                
                # Update cache
                self.cache['data'] = stats
                self.cache['last_update'] = current_time
                
                return stats
                
        except Exception as e:
            logger.error(f"Error getting performance stats: {e}")
            return {'error': str(e), 'timestamp': datetime.now().isoformat()}
    
    def _get_system_metrics(self, conn) -> Dict[str, Any]:
        """Get system-level metrics"""
        # Get final_stats.json if it exists
        stats_file = Path("final_stats.json")
        if stats_file.exists():
            with open(stats_file) as f:
                final_stats = json.load(f)
            
            return {
                'uptime_hours': final_stats.get('runtime_hours', 0),
                'total_ticks': final_stats.get('total_ticks_processed', 0),
                'account_balance': final_stats.get('final_balance', 10000),
                'account_equity': final_stats.get('final_equity', 10000),
                'total_return_pct': final_stats.get('total_return', 0) * 100,
                'max_drawdown_pct': final_stats.get('max_drawdown', 0) * 100
            }
        
        return {
            'uptime_hours': 0,
            'total_ticks': 0,
            'account_balance': 10000,
            'account_equity': 10000,
            'total_return_pct': 0,
            'max_drawdown_pct': 0
        }
    
    def _get_trade_statistics(self, conn) -> Dict[str, Any]:
        """Get trading statistics"""
        try:
            # Recent trades (last 24 hours)
            yesterday = time.time() - 86400
            
            stats = conn.execute('''
                SELECT 
                    COUNT(*) as total_trades,
                    SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as winning_trades,
                    SUM(pnl) as total_pnl,
                    AVG(pnl) as avg_pnl,
                    AVG(trade_duration) as avg_duration,
                    COUNT(CASE WHEN virtual_trade = 0 THEN 1 END) as real_trades,
                    COUNT(CASE WHEN virtual_trade = 1 THEN 1 END) as virtual_trades
                FROM trade_logs 
                WHERE timestamp > ? AND pnl IS NOT NULL
            ''', (yesterday,)).fetchone()
            
            if stats and stats[0] > 0:
                return {
                    'total_trades': stats[0],
                    'winning_trades': stats[1],
                    'win_rate_pct': (stats[1] / stats[0]) * 100 if stats[0] > 0 else 0,
                    'total_pnl': stats[2] or 0,
                    'avg_pnl': stats[3] or 0,
                    'avg_duration_minutes': (stats[4] or 0) / 60,
                    'real_trades': stats[5],
                    'virtual_trades': stats[6]
                }
            
        except Exception as e:
            logger.error(f"Error getting trade statistics: {e}")
        
        return {
            'total_trades': 0,
            'winning_trades': 0,
            'win_rate_pct': 0,
            'total_pnl': 0,
            'avg_pnl': 0,
            'avg_duration_minutes': 0,
            'real_trades': 0,
            'virtual_trades': 0
        }
    
    def _get_performance_metrics(self, conn) -> Dict[str, Any]:
        """Get performance metrics"""
        try:
            # Get latest performance metrics
            latest = conn.execute('''
                SELECT * FROM performance_logs 
                ORDER BY timestamp DESC LIMIT 1
            ''').fetchone()
            
            if latest:
                return {
                    'win_rate': latest[5] * 100,
                    'profit_factor': latest[6],
                    'sharpe_ratio': latest[7],
                    'max_drawdown': latest[8] * 100,
                    'calmar_ratio': latest[17]
                }
        
        except Exception as e:
            logger.error(f"Error getting performance metrics: {e}")
        
        return {
            'win_rate': 0,
            'profit_factor': 0,
            'sharpe_ratio': 0,
            'max_drawdown': 0,
            'calmar_ratio': 0
        }
    
    def _get_agent_statistics(self, conn) -> Dict[str, Any]:
        """Get agent-specific statistics"""
        try:
            berserker_stats = conn.execute('''
                SELECT 
                    COUNT(*) as total_trades,
                    SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as winning_trades,
                    SUM(pnl) as total_pnl
                FROM trade_logs 
                WHERE agent_id = 'BERSERKER' AND pnl IS NOT NULL
            ''').fetchone()
            
            sniper_stats = conn.execute('''
                SELECT 
                    COUNT(*) as total_trades,
                    SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as winning_trades,
                    SUM(pnl) as total_pnl
                FROM trade_logs 
                WHERE agent_id = 'SNIPER' AND pnl IS NOT NULL
            ''').fetchone()
            
            return {
                'berserker': {
                    'total_trades': berserker_stats[0] if berserker_stats else 0,
                    'winning_trades': berserker_stats[1] if berserker_stats else 0,
                    'win_rate_pct': (berserker_stats[1] / berserker_stats[0] * 100 
                                   if berserker_stats and berserker_stats[0] > 0 else 0),
                    'total_pnl': berserker_stats[2] if berserker_stats else 0
                },
                'sniper': {
                    'total_trades': sniper_stats[0] if sniper_stats else 0,
                    'winning_trades': sniper_stats[1] if sniper_stats else 0,
                    'win_rate_pct': (sniper_stats[1] / sniper_stats[0] * 100 
                                   if sniper_stats and sniper_stats[0] > 0 else 0),
                    'total_pnl': sniper_stats[2] if sniper_stats else 0
                }
            }
        
        except Exception as e:
            logger.error(f"Error getting agent statistics: {e}")
            return {'berserker': {}, 'sniper': {}}
    
    def _get_error_statistics(self, conn) -> Dict[str, Any]:
        """Get error statistics"""
        try:
            yesterday = time.time() - 86400
            
            error_stats = conn.execute('''
                SELECT 
                    COUNT(*) as total_errors,
                    COUNT(CASE WHEN severity = 'CRITICAL' THEN 1 END) as critical_errors,
                    COUNT(CASE WHEN resolved = 1 THEN 1 END) as resolved_errors
                FROM error_logs 
                WHERE timestamp > ?
            ''', (yesterday,)).fetchone()
            
            if error_stats:
                return {
                    'total_errors': error_stats[0],
                    'critical_errors': error_stats[1],
                    'resolved_errors': error_stats[2],
                    'resolution_rate_pct': (error_stats[2] / error_stats[0] * 100 
                                          if error_stats[0] > 0 else 100)
                }
        
        except Exception as e:
            logger.error(f"Error getting error statistics: {e}")
        
        return {
            'total_errors': 0,
            'critical_errors': 0,
            'resolved_errors': 0,
            'resolution_rate_pct': 100
        }
    
    def _get_recent_trades(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent trades"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                trades = conn.execute('''
                    SELECT 
                        timestamp, agent_id, instrument, action, 
                        entry_price, exit_price, pnl, win_loss, virtual_trade
                    FROM trade_logs 
                    WHERE pnl IS NOT NULL
                    ORDER BY timestamp DESC 
                    LIMIT ?
                ''', (limit,)).fetchall()
                
                return [{
                    'timestamp': datetime.fromtimestamp(trade[0]).strftime('%H:%M:%S'),
                    'agent': trade[1],
                    'symbol': trade[2],
                    'action': trade[3],
                    'entry_price': trade[4],
                    'exit_price': trade[5],
                    'pnl': trade[6],
                    'result': trade[7],
                    'virtual': bool(trade[8])
                } for trade in trades]
        
        except Exception as e:
            logger.error(f"Error getting recent trades: {e}")
            return []
    
    def _get_agent_stats(self) -> Dict[str, Any]:
        """Get detailed agent statistics"""
        return self._get_performance_stats()['agents']
    
    def _generate_dashboard_html(self) -> str:
        """Generate main dashboard HTML"""
        return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>VelocityTrader Performance Dashboard</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
            background: #0d1421; 
            color: #e1e5e9; 
            overflow-x: auto;
        }
        .header { 
            background: linear-gradient(135deg, #1a2332, #2d3748); 
            padding: 20px; 
            border-bottom: 2px solid #4a5568; 
        }
        .header h1 { 
            color: #63b3ed; 
            font-size: 2.5em; 
            margin-bottom: 10px; 
        }
        .header .status { 
            color: #68d391; 
            font-size: 1.2em; 
            font-weight: bold; 
        }
        .container { 
            padding: 20px; 
            max-width: 1400px; 
            margin: 0 auto; 
        }
        .grid { 
            display: grid; 
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); 
            gap: 20px; 
            margin-bottom: 30px; 
        }
        .card { 
            background: linear-gradient(135deg, #2d3748, #4a5568); 
            border-radius: 10px; 
            padding: 20px; 
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3); 
            border: 1px solid #4a5568; 
        }
        .card h3 { 
            color: #63b3ed; 
            margin-bottom: 15px; 
            font-size: 1.3em; 
            border-bottom: 2px solid #4a5568; 
            padding-bottom: 10px; 
        }
        .metric { 
            display: flex; 
            justify-content: space-between; 
            margin: 10px 0; 
            padding: 5px 0; 
        }
        .metric-label { 
            color: #a0aec0; 
        }
        .metric-value { 
            font-weight: bold; 
            color: #e1e5e9; 
        }
        .positive { color: #68d391 !important; }
        .negative { color: #fc8181 !important; }
        .warning { color: #fbb651 !important; }
        .table-container { 
            overflow-x: auto; 
            margin-top: 20px; 
        }
        table { 
            width: 100%; 
            border-collapse: collapse; 
            background: #2d3748; 
            border-radius: 10px; 
            overflow: hidden; 
        }
        th, td { 
            padding: 12px; 
            text-align: left; 
            border-bottom: 1px solid #4a5568; 
        }
        th { 
            background: #1a2332; 
            color: #63b3ed; 
            font-weight: bold; 
        }
        .agent-berserker { color: #fc8181; font-weight: bold; }
        .agent-sniper { color: #68d391; font-weight: bold; }
        .update-time { 
            color: #a0aec0; 
            font-size: 0.9em; 
            text-align: center; 
            margin-top: 20px; 
        }
        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.5; }
            100% { opacity: 1; }
        }
        .live-indicator { 
            color: #68d391; 
            animation: pulse 2s infinite; 
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>⚡ VelocityTrader Dashboard</h1>
        <div class="status">
            <span class="live-indicator">● LIVE</span> 
            Physics-Based AI Trading System
        </div>
    </div>
    
    <div class="container">
        <div class="grid">
            <div class="card">
                <h3>💰 Account Overview</h3>
                <div class="metric">
                    <span class="metric-label">Balance:</span>
                    <span class="metric-value" id="balance">$10,000.00</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Equity:</span>
                    <span class="metric-value" id="equity">$10,000.00</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Total Return:</span>
                    <span class="metric-value" id="return">0.00%</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Max Drawdown:</span>
                    <span class="metric-value" id="drawdown">0.00%</span>
                </div>
            </div>
            
            <div class="card">
                <h3>📊 Trading Statistics</h3>
                <div class="metric">
                    <span class="metric-label">Total Trades:</span>
                    <span class="metric-value" id="total-trades">0</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Win Rate:</span>
                    <span class="metric-value" id="win-rate">0.00%</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Total P&L:</span>
                    <span class="metric-value" id="total-pnl">$0.00</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Avg Trade Duration:</span>
                    <span class="metric-value" id="avg-duration">0.0 min</span>
                </div>
            </div>
            
            <div class="card">
                <h3>⚔️ BERSERKER Agent</h3>
                <div class="metric">
                    <span class="metric-label">Trades:</span>
                    <span class="metric-value" id="berserker-trades">0</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Win Rate:</span>
                    <span class="metric-value" id="berserker-win-rate">0.00%</span>
                </div>
                <div class="metric">
                    <span class="metric-label">P&L:</span>
                    <span class="metric-value" id="berserker-pnl">$0.00</span>
                </div>
            </div>
            
            <div class="card">
                <h3>🎯 SNIPER Agent</h3>
                <div class="metric">
                    <span class="metric-label">Trades:</span>
                    <span class="metric-value" id="sniper-trades">0</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Win Rate:</span>
                    <span class="metric-value" id="sniper-win-rate">0.00%</span>
                </div>
                <div class="metric">
                    <span class="metric-label">P&L:</span>
                    <span class="metric-value" id="sniper-pnl">$0.00</span>
                </div>
            </div>
            
            <div class="card">
                <h3>⚙️ System Metrics</h3>
                <div class="metric">
                    <span class="metric-label">Uptime:</span>
                    <span class="metric-value" id="uptime">0.0 hours</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Ticks Processed:</span>
                    <span class="metric-value" id="ticks">0</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Errors (24h):</span>
                    <span class="metric-value" id="errors">0</span>
                </div>
            </div>
            
            <div class="card">
                <h3>📈 Performance Metrics</h3>
                <div class="metric">
                    <span class="metric-label">Profit Factor:</span>
                    <span class="metric-value" id="profit-factor">0.00</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Sharpe Ratio:</span>
                    <span class="metric-value" id="sharpe">0.00</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Calmar Ratio:</span>
                    <span class="metric-value" id="calmar">0.00</span>
                </div>
            </div>
        </div>
        
        <div class="card">
            <h3>📋 Recent Trades</h3>
            <div class="table-container">
                <table id="trades-table">
                    <thead>
                        <tr>
                            <th>Time</th>
                            <th>Agent</th>
                            <th>Symbol</th>
                            <th>Action</th>
                            <th>Entry</th>
                            <th>Exit</th>
                            <th>P&L</th>
                            <th>Result</th>
                            <th>Type</th>
                        </tr>
                    </thead>
                    <tbody id="trades-body">
                        <tr>
                            <td colspan="9" style="text-align: center; color: #a0aec0;">No trades yet</td>
                        </tr>
                    </tbody>
                </table>
            </div>
        </div>
        
        <div class="update-time" id="last-update">
            Last updated: Never
        </div>
    </div>

    <script>
        // WebSocket connection for real-time updates
        let ws = null;
        
        function connectWebSocket() {
            try {
                ws = new WebSocket(`ws://${window.location.hostname}:${parseInt(window.location.port) + 1}`);
                
                ws.onopen = function(event) {
                    console.log('WebSocket connected');
                };
                
                ws.onmessage = function(event) {
                    const data = JSON.parse(event.data);
                    if (data.type === 'update') {
                        updateDashboard(data.data);
                    }
                };
                
                ws.onclose = function(event) {
                    console.log('WebSocket disconnected');
                    setTimeout(connectWebSocket, 5000); // Reconnect after 5 seconds
                };
                
                ws.onerror = function(error) {
                    console.error('WebSocket error:', error);
                };
            } catch (error) {
                console.error('Failed to connect WebSocket:', error);
                // Fallback to HTTP polling
                setTimeout(fetchData, 1000);
            }
        }
        
        function fetchData() {
            fetch('/api/stats')
                .then(response => response.json())
                .then(data => updateDashboard(data))
                .catch(error => {
                    console.error('Error fetching data:', error);
                    setTimeout(fetchData, 5000);
                });
        }
        
        function updateDashboard(data) {
            if (data.error) {
                console.error('Dashboard data error:', data.error);
                return;
            }
            
            // System metrics
            if (data.system) {
                document.getElementById('balance').textContent = `$${data.system.account_balance?.toFixed(2) || '0.00'}`;
                document.getElementById('equity').textContent = `$${data.system.account_equity?.toFixed(2) || '0.00'}`;
                document.getElementById('return').textContent = `${data.system.total_return_pct?.toFixed(2) || '0.00'}%`;
                document.getElementById('drawdown').textContent = `${data.system.max_drawdown_pct?.toFixed(2) || '0.00'}%`;
                document.getElementById('uptime').textContent = `${data.system.uptime_hours?.toFixed(1) || '0.0'} hours`;
                document.getElementById('ticks').textContent = data.system.total_ticks || '0';
                
                // Color code return
                const returnElement = document.getElementById('return');
                const returnValue = data.system.total_return_pct || 0;
                returnElement.className = 'metric-value ' + (returnValue >= 0 ? 'positive' : 'negative');
            }
            
            // Trading statistics
            if (data.trading) {
                document.getElementById('total-trades').textContent = data.trading.total_trades || '0';
                document.getElementById('win-rate').textContent = `${data.trading.win_rate_pct?.toFixed(2) || '0.00'}%`;
                document.getElementById('total-pnl').textContent = `$${data.trading.total_pnl?.toFixed(2) || '0.00'}`;
                document.getElementById('avg-duration').textContent = `${data.trading.avg_duration_minutes?.toFixed(1) || '0.0'} min`;
                
                // Color code P&L
                const pnlElement = document.getElementById('total-pnl');
                const pnlValue = data.trading.total_pnl || 0;
                pnlElement.className = 'metric-value ' + (pnlValue >= 0 ? 'positive' : 'negative');
            }
            
            // Agent statistics
            if (data.agents) {
                // BERSERKER
                if (data.agents.berserker) {
                    document.getElementById('berserker-trades').textContent = data.agents.berserker.total_trades || '0';
                    document.getElementById('berserker-win-rate').textContent = `${data.agents.berserker.win_rate_pct?.toFixed(2) || '0.00'}%`;
                    document.getElementById('berserker-pnl').textContent = `$${data.agents.berserker.total_pnl?.toFixed(2) || '0.00'}`;
                    
                    const berserkerPnlElement = document.getElementById('berserker-pnl');
                    const berserkerPnl = data.agents.berserker.total_pnl || 0;
                    berserkerPnlElement.className = 'metric-value ' + (berserkerPnl >= 0 ? 'positive' : 'negative');
                }
                
                // SNIPER
                if (data.agents.sniper) {
                    document.getElementById('sniper-trades').textContent = data.agents.sniper.total_trades || '0';
                    document.getElementById('sniper-win-rate').textContent = `${data.agents.sniper.win_rate_pct?.toFixed(2) || '0.00'}%`;
                    document.getElementById('sniper-pnl').textContent = `$${data.agents.sniper.total_pnl?.toFixed(2) || '0.00'}`;
                    
                    const sniperPnlElement = document.getElementById('sniper-pnl');
                    const sniperPnl = data.agents.sniper.total_pnl || 0;
                    sniperPnlElement.className = 'metric-value ' + (sniperPnl >= 0 ? 'positive' : 'negative');
                }
            }
            
            // Performance metrics
            if (data.performance) {
                document.getElementById('profit-factor').textContent = data.performance.profit_factor?.toFixed(2) || '0.00';
                document.getElementById('sharpe').textContent = data.performance.sharpe_ratio?.toFixed(2) || '0.00';
                document.getElementById('calmar').textContent = data.performance.calmar_ratio?.toFixed(2) || '0.00';
            }
            
            // Error statistics
            if (data.errors) {
                document.getElementById('errors').textContent = data.errors.total_errors || '0';
                
                const errorsElement = document.getElementById('errors');
                const errorCount = data.errors.total_errors || 0;
                errorsElement.className = 'metric-value ' + (errorCount > 10 ? 'negative' : errorCount > 5 ? 'warning' : 'positive');
            }
            
            // Update timestamp
            const now = new Date();
            document.getElementById('last-update').textContent = `Last updated: ${now.toLocaleTimeString()}`;
            
            // Fetch and update recent trades
            fetchRecentTrades();
        }
        
        function fetchRecentTrades() {
            fetch('/api/trades')
                .then(response => response.json())
                .then(trades => {
                    const tbody = document.getElementById('trades-body');
                    
                    if (trades.length === 0) {
                        tbody.innerHTML = '<tr><td colspan="9" style="text-align: center; color: #a0aec0;">No trades yet</td></tr>';
                        return;
                    }
                    
                    tbody.innerHTML = trades.map(trade => `
                        <tr>
                            <td>${trade.timestamp}</td>
                            <td><span class="agent-${trade.agent.toLowerCase()}">${trade.agent}</span></td>
                            <td>${trade.symbol}</td>
                            <td>${trade.action}</td>
                            <td>${trade.entry_price?.toFixed(5) || 'N/A'}</td>
                            <td>${trade.exit_price?.toFixed(5) || 'N/A'}</td>
                            <td class="${trade.pnl >= 0 ? 'positive' : 'negative'}">$${trade.pnl?.toFixed(2) || '0.00'}</td>
                            <td class="${trade.result === 'WIN' ? 'positive' : 'negative'}">${trade.result || 'N/A'}</td>
                            <td>${trade.virtual ? 'Virtual' : 'Real'}</td>
                        </tr>
                    `).join('');
                })
                .catch(error => console.error('Error fetching trades:', error));
        }
        
        // Initialize dashboard
        document.addEventListener('DOMContentLoaded', function() {
            connectWebSocket();
            
            // Fallback: fetch data every 5 seconds if WebSocket fails
            setInterval(() => {
                if (!ws || ws.readyState !== WebSocket.OPEN) {
                    fetchData();
                }
            }, 5000);
        });
    </script>
</body>
</html>"""

# Main entry point
def main():
    """Main entry point for dashboard"""
    dashboard = PerformanceDashboard()
    
    try:
        dashboard.start()
        
        # Keep running
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("Dashboard shutdown requested")
    finally:
        dashboard.stop()

if __name__ == "__main__":
    main()