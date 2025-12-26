#!/usr/bin/env python3
"""
MINIMAL HTTP SERVER - No creativity, just works
Basic HTTP server that responds on port 8080
"""

import http.server
import socketserver
import json
from datetime import datetime

class MinimalHandler(http.server.SimpleHTTPRequestHandler):
    
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            html = """<!DOCTYPE html>
<html>
<head><title>Minimal Server</title></head>
<body>
<h1>Minimal Server Running</h1>
<p>Time: %s</p>
<p><a href="/health">Health Check</a></p>
<p><a href="/data">Real Data</a></p>
</body>
</html>""" % datetime.now()
            self.wfile.write(html.encode())
            
        elif self.path == '/health':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            response = {
                "status": "healthy", 
                "time": datetime.now().isoformat(),
                "server": "minimal"
            }
            self.wfile.write(json.dumps(response).encode())
            
        elif self.path == '/data':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            try:
                with open('all_mt5_symbols.json', 'r') as f:
                    data = json.load(f)
                    response = {
                        "status": "success",
                        "symbols_count": len(data.get('symbols', {})),
                        "server": data.get('account', {}).get('server', 'unknown')
                    }
            except:
                response = {"status": "error", "message": "Cannot load data"}
            
            self.wfile.write(json.dumps(response).encode())
        else:
            self.send_error(404)

if __name__ == "__main__":
    PORT = 8080
    print(f"Starting minimal server on port {PORT}")
    print(f"Access: http://172.27.209.14:{PORT}")
    
    with socketserver.TCPServer(("0.0.0.0", PORT), MinimalHandler) as httpd:
        httpd.serve_forever()