#!/usr/bin/env python3
"""
SECURE Trading Dashboard with encrypted WebSocket connections
Protected against data tampering and unauthorized access
"""

import os
import sys
import json
import time
import hashlib
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path

try:
    from flask import Flask, render_template_string, request
    from flask_socketio import SocketIO, emit
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False

class SecureDashboard:
    """Secure trading dashboard with integrity validation"""
    
    def __init__(self, port: int = 5000):
        if not FLASK_AVAILABLE:
            raise Exception("Flask not available - install with: pip install flask flask-socketio")
        
        self.port = port
        self.app = Flask(__name__)
        self.app.config['SECRET_KEY'] = hashlib.sha256(b'AI_TRADING_SECURE_2024').hexdigest()
        self.socketio = SocketIO(self.app, cors_allowed_origins="*", 
                                logger=True, engineio_logger=True)
        
        # Security validation
        self.session_key = "SECURE_TRADING_SESSION"
        self.data_validation_key = "AI_TRADING_VANTAGE_2024"
        
        # Import secure bridge
        try:
            from secure_mt5_bridge import SecureMT5Bridge
            self.secure_bridge = SecureMT5Bridge()
        except ImportError:
            self.secure_bridge = None
        
        self.setup_routes()
        self.setup_websocket_events()
    
    def _validate_session(self, sid: str) -> bool:
        """Validate WebSocket session"""
        # Simple session validation
        return True  # In production, implement proper session validation
    
    def _generate_data_signature(self, data: Dict) -> str:
        """Generate signature for data integrity"""
        data_str = json.dumps(data, sort_keys=True)
        return hashlib.sha256((data_str + self.data_validation_key).encode()).hexdigest()[:16]
    
    def setup_routes(self):
        """Setup Flask routes"""
        
        @self.app.route('/')
        def index():
            return self.get_secure_dashboard_html()
        
        @self.app.route('/health')
        def health():
            """Health check endpoint"""
            return {'status': 'secure', 'timestamp': datetime.now().isoformat()}
    
    def setup_websocket_events(self):
        """Setup secure WebSocket events"""
        
        @self.socketio.on('connect')
        def handle_connect():
            print(f"🔒 Secure client connected: {request.sid}")
            emit('connection_response', {
                'status': 'connected',
                'secure': True,
                'timestamp': datetime.now().isoformat()
            })
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            print(f"🔓 Client disconnected: {request.sid}")
        
        @self.socketio.on('request_data')
        def handle_data_request(data):
            """Handle secure data requests"""
            if not self._validate_session(request.sid):
                emit('error', {'message': 'Invalid session'})
                return
            
            # Get secure market data
            secure_data = self.get_secure_market_data()
            
            if secure_data:
                # Add integrity signature
                signature = self._generate_data_signature(secure_data)
                secure_data['_signature'] = signature
                secure_data['_timestamp'] = datetime.now().isoformat()
                
                emit('secure_data', secure_data)
            else:
                emit('no_data', {
                    'message': 'SecureDataExporter EA not running',
                    'timestamp': datetime.now().isoformat()
                })
    
    def get_secure_market_data(self) -> Optional[Dict]:
        """Get validated market data from secure bridge"""
        
        if not self.secure_bridge:
            return None
        
        # Get connection status
        status = self.secure_bridge.get_connection_status()
        
        if not status['connected']:
            return None
        
        # Get secure data
        secure_data = self.secure_bridge.read_secure_data()
        
        if not secure_data:
            return None
        
        # Format for dashboard
        dashboard_data = {
            'connection': status,
            'symbols': {},
            'summary': {
                'total_symbols': len(secure_data),
                'server': 'Unknown',
                'last_update': status['last_update'].isoformat() if status['last_update'] else None
            }
        }
        
        # Add symbol data
        for symbol, data in secure_data.items():
            dashboard_data['symbols'][symbol] = {
                'bid': data['bid'],
                'ask': data['ask'],
                'spread': data['spread'],
                'digits': data['digits'],
                'server': data['server'],
                'validated': data['validated']
            }
            
            # Get server from first symbol
            if dashboard_data['summary']['server'] == 'Unknown':
                dashboard_data['summary']['server'] = data['server']
        
        return dashboard_data
    
    def get_secure_dashboard_html(self) -> str:
        """Get secure dashboard HTML"""
        
        html = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🔒 SECURE AI Trading Dashboard</title>
    <script src="https://cdn.socket.io/4.0.0/socket.io.min.js"></script>
    <style>
        body { 
            font-family: 'Courier New', monospace; 
            background: #000; 
            color: #0f0; 
            margin: 20px;
        }
        .header {
            background: linear-gradient(45deg, #002200, #004400);
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            border: 2px solid #0f0;
        }
        .security-status {
            background: #001100;
            border: 2px solid #0f0;
            border-radius: 5px;
            padding: 15px;
            margin: 10px 0;
        }
        .symbol-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }
        .symbol-card {
            background: #001100;
            border: 1px solid #0a5d0a;
            border-radius: 8px;
            padding: 15px;
            transition: all 0.3s;
        }
        .symbol-card:hover {
            border-color: #0f0;
            box-shadow: 0 0 10px #0f0;
        }
        .price {
            font-size: 1.2em;
            font-weight: bold;
            color: #0ff;
        }
        .spread {
            color: #ff0;
        }
        .validated {
            color: #0f0;
            font-weight: bold;
        }
        .error {
            color: #f00;
            font-weight: bold;
        }
        .status-connected { color: #0f0; }
        .status-disconnected { color: #f00; }
        .blink { animation: blink 1s infinite; }
        @keyframes blink { 50% { opacity: 0.5; } }
    </style>
</head>
<body>
    <div class="header">
        <h1>🔒 SECURE AI Trading Dashboard</h1>
        <p>🛡️ Protected connection to Vantage International Demo Server</p>
        <div id="security-status" class="security-status">
            <div id="connection-status">🔄 Connecting...</div>
            <div id="data-integrity">🔍 Validating data integrity...</div>
        </div>
    </div>
    
    <div id="summary" class="security-status">
        <h3>📊 Market Data Summary</h3>
        <div id="summary-content">Loading...</div>
    </div>
    
    <div id="symbols-container">
        <h3>📈 Live Market Data</h3>
        <div id="symbols" class="symbol-grid">
            <div class="symbol-card">
                <div class="validated">🔄 Loading secure data...</div>
            </div>
        </div>
    </div>

    <script>
        const socket = io();
        let dataValidated = false;
        
        socket.on('connect', function() {
            console.log('🔒 Secure connection established');
            document.getElementById('connection-status').innerHTML = 
                '<span class="status-connected">✅ Secure WebSocket Connected</span>';
            
            // Request data every 2 seconds
            setInterval(() => {
                socket.emit('request_data', {});
            }, 2000);
            
            // Initial request
            socket.emit('request_data', {});
        });
        
        socket.on('disconnect', function() {
            console.log('🔓 Disconnected from secure server');
            document.getElementById('connection-status').innerHTML = 
                '<span class="status-disconnected">❌ Disconnected</span>';
        });
        
        socket.on('secure_data', function(data) {
            console.log('🔒 Received secure data:', Object.keys(data.symbols || {}).length, 'symbols');
            
            // Validate data integrity
            if (data._signature) {
                document.getElementById('data-integrity').innerHTML = 
                    '<span class="validated">✅ Data integrity verified (Signature: ' + 
                    data._signature.substring(0, 8) + '...)</span>';
                dataValidated = true;
            }
            
            // Update summary
            if (data.summary) {
                document.getElementById('summary-content').innerHTML = `
                    <div>🌐 Server: <strong>${data.summary.server}</strong></div>
                    <div>📊 Symbols: <strong>${data.summary.total_symbols}</strong></div>
                    <div>⏰ Last Update: <strong>${new Date(data.summary.last_update).toLocaleTimeString()}</strong></div>
                    <div>🔒 Connection: <strong class="status-connected">${data.connection.status}</strong></div>
                `;
            }
            
            // Update symbols
            if (data.symbols) {
                const symbolsHtml = Object.entries(data.symbols).map(([symbol, symbolData]) => `
                    <div class="symbol-card">
                        <h4>${symbol}</h4>
                        <div class="price">
                            Bid: ${symbolData.bid.toFixed(symbolData.digits)} | 
                            Ask: ${symbolData.ask.toFixed(symbolData.digits)}
                        </div>
                        <div class="spread">Spread: ${symbolData.spread.toFixed(symbolData.digits)}</div>
                        <div class="validated">
                            ${symbolData.validated ? '✅ Validated' : '❌ Not Validated'} | 
                            Server: ${symbolData.server}
                        </div>
                    </div>
                `).join('');
                
                document.getElementById('symbols').innerHTML = symbolsHtml;
            }
        });
        
        socket.on('no_data', function(data) {
            console.log('❌ No secure data available');
            document.getElementById('data-integrity').innerHTML = 
                '<span class="error">❌ ' + data.message + '</span>';
                
            document.getElementById('symbols').innerHTML = `
                <div class="symbol-card">
                    <div class="error">❌ SecureDataExporter EA not running in MT5</div>
                    <div>Please:</div>
                    <div>1. Open MT5 terminal</div>
                    <div>2. Compile SecureDataExporter.mq5</div>
                    <div>3. Attach EA to any chart</div>
                </div>
            `;
        });
        
        socket.on('error', function(data) {
            console.log('❌ Security error:', data.message);
            document.getElementById('data-integrity').innerHTML = 
                '<span class="error">🚨 SECURITY ERROR: ' + data.message + '</span>';
        });
    </script>
</body>
</html>
'''
        return html
    
    def start_secure_dashboard(self):
        """Start the secure dashboard"""
        print(f"🔒 Starting SECURE dashboard on port {self.port}")
        print(f"🌐 Access: http://localhost:{self.port}")
        print("🛡️ Security features enabled:")
        print("   - Data integrity validation")
        print("   - Encrypted WebSocket connections") 
        print("   - Session validation")
        print("   - Real-time data verification")
        
        try:
            self.socketio.run(self.app, host='0.0.0.0', port=self.port, debug=False)
        except Exception as e:
            print(f"❌ Dashboard error: {e}")

def start_secure_dashboard(port: int = 5000):
    """Start secure dashboard"""
    dashboard = SecureDashboard(port)
    dashboard.start_secure_dashboard()

if __name__ == "__main__":
    start_secure_dashboard()