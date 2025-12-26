#!/usr/bin/env python3
"""
STABLE LONG-TERM SYSTEM
- Native Python HTTP server
- File-based data persistence  
- Real Vantage MT5 data
- Simple, reliable, accessible
"""

import http.server
import socketserver
import json
import time
import threading
from datetime import datetime
from pathlib import Path

class StableHandler(http.server.SimpleHTTPRequestHandler):
    
    def do_GET(self):
        if self.path == '/':
            self.send_html_dashboard()
        elif self.path == '/api/symbols':
            self.send_json_symbols()
        elif self.path == '/health':
            self.send_health()
        else:
            self.send_error(404)
    
    def send_html_dashboard(self):
        """Serve HTML dashboard with real data"""
        try:
            with open('all_mt5_symbols.json', 'r') as f:
                data = json.load(f)
                symbols = data.get('symbols', {})
                account = data.get('account', {})
        except:
            symbols, account = {}, {}
        
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Stable Trading System</title>
    <meta charset="utf-8">
    <meta http-equiv="refresh" content="10">
    <style>
        body {{ font-family: Arial; margin: 20px; background: #f5f5f5; }}
        .header {{ background: #2c3e50; color: white; padding: 20px; margin-bottom: 20px; }}
        .symbol {{ background: white; padding: 15px; margin: 10px 0; border-left: 4px solid #3498db; }}
        .price {{ font-size: 1.2em; color: #27ae60; font-weight: bold; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Stable Trading System</h1>
        <p>Server: {account.get('server', 'VantageInternational-Demo')}</p>
        <p>Account: {account.get('login', 'N/A')} | Balance: ${account.get('balance', 0):.2f}</p>
        <p>Symbols: {len(symbols)} | Updated: {datetime.now().strftime('%H:%M:%S')}</p>
    </div>
"""
        
        # Show real symbols
        count = 0
        for symbol, data in symbols.items():
            if count >= 10:
                break
            html += f"""
    <div class="symbol">
        <h3>{symbol}</h3>
        <div class="price">Bid: {data.get('bid', 0):.{data.get('digits', 5)}f} | Ask: {data.get('ask', 0):.{data.get('digits', 5)}f}</div>
        <p>Spread: {data.get('spread_pips', 0):.1f} pips | Category: {data.get('category', 'UNKNOWN')}</p>
    </div>
"""
            count += 1
        
        html += """
</body>
</html>
"""
        
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(html.encode('utf-8'))
    
    def send_json_symbols(self):
        """API endpoint for symbols data"""
        try:
            with open('all_mt5_symbols.json', 'r') as f:
                data = json.load(f)
            response = {
                "status": "success",
                "timestamp": datetime.now().isoformat(),
                "data": data
            }
        except Exception as e:
            response = {
                "status": "error",
                "message": str(e),
                "timestamp": datetime.now().isoformat()
            }
        
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(response, indent=2).encode('utf-8'))
    
    def send_health(self):
        """Health check endpoint"""
        try:
            with open('all_mt5_symbols.json', 'r') as f:
                data = json.load(f)
                symbols_count = len(data.get('symbols', {}))
        except:
            symbols_count = 0
        
        health = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "symbols_loaded": symbols_count,
            "uptime_seconds": int(time.time() - start_time)
        }
        
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(health, indent=2).encode('utf-8'))

def main():
    global start_time
    start_time = time.time()
    
    PORT = 8080
    
    print("=" * 50)
    print("STABLE TRADING SYSTEM")
    print("=" * 50)
    print(f"Starting on port {PORT}")
    print(f"Dashboard: http://172.27.209.14:{PORT}")
    print(f"API: http://172.27.209.14:{PORT}/api/symbols")
    print(f"Health: http://172.27.209.14:{PORT}/health")
    print("=" * 50)
    
    # Verify data file exists
    if not Path('all_mt5_symbols.json').exists():
        print("ERROR: all_mt5_symbols.json not found")
        return
    
    try:
        with socketserver.TCPServer(("0.0.0.0", PORT), StableHandler) as httpd:
            print("Server started successfully")
            httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutdown requested")
    except Exception as e:
        print(f"Server error: {e}")

if __name__ == "__main__":
    main()