#!/usr/bin/env python3
"""
Simple HTTP Dashboard for Windows Access Testing
No HTTPS, just plain HTTP for testing connectivity
"""

import asyncio
import json
import sys
import os
from pathlib import Path
from datetime import datetime

# Add paths
sys.path.append(str(Path(__file__).parent / "core"))

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import uvicorn

class SimpleDashboard:
    """Simple HTTP dashboard for testing"""
    
    def __init__(self, port=8080):
        self.port = port
        self.app = FastAPI(title="Simple AI Trading Dashboard")
        self.setup_routes()
        
        # Load symbol data if available
        self.symbol_data = self.load_symbols()
    
    def load_symbols(self):
        """Load symbol data"""
        try:
            if os.path.exists("all_mt5_symbols.json"):
                with open("all_mt5_symbols.json", 'r') as f:
                    data = json.load(f)
                    return data.get('symbols', {})
        except:
            pass
        return {}
    
    def setup_routes(self):
        """Setup simple routes"""
        
        @self.app.get("/")
        async def dashboard():
            return HTMLResponse(self.get_simple_html())
        
        @self.app.get("/health")
        async def health():
            return {"status": "ok", "time": datetime.now().isoformat()}
        
        @self.app.get("/data")
        async def get_data():
            """Return simple market data"""
            return {
                "symbols": len(self.symbol_data),
                "data": dict(list(self.symbol_data.items())[:5]),  # First 5 symbols
                "timestamp": datetime.now().isoformat()
            }
    
    def get_simple_html(self):
        """Simple dashboard HTML"""
        return '''
<!DOCTYPE html>
<html>
<head>
    <title>AI Trading System - Simple Dashboard</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f0f0f0; }
        .container { max-width: 1200px; margin: 0 auto; }
        .header { background: #2c3e50; color: white; padding: 20px; border-radius: 5px; }
        .status { background: #27ae60; color: white; padding: 10px; margin: 10px 0; border-radius: 5px; }
        .symbol { background: white; padding: 15px; margin: 10px; border-radius: 5px; border-left: 4px solid #3498db; }
        .price { font-size: 1.2em; font-weight: bold; color: #e74c3c; }
        .loading { color: #7f8c8d; font-style: italic; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🏭 AI Trading System - Simple Dashboard</h1>
            <p>HTTP Version for Windows Access Testing</p>
        </div>
        
        <div class="status" id="status">
            🔄 Connecting...
        </div>
        
        <div id="data-container">
            <h3>📊 Market Data</h3>
            <div id="symbols" class="loading">Loading market data...</div>
        </div>
        
        <div style="background: white; padding: 15px; margin: 10px 0; border-radius: 5px;">
            <h3>🔗 Access Information</h3>
            <p><strong>Simple HTTP:</strong> http://''' + '''172.27.209.14:8080</p>
            <p><strong>Enterprise HTTPS:</strong> https://172.27.209.14:8443</p>
            <p><strong>Metrics:</strong> http://172.27.209.14:9090</p>
        </div>
    </div>
    
    <script>
        async function updateData() {
            try {
                // Test health
                const healthResponse = await fetch('/health');
                const health = await healthResponse.json();
                document.getElementById('status').innerHTML = 
                    '✅ Connected - ' + health.time;
                
                // Get data
                const dataResponse = await fetch('/data');
                const data = await dataResponse.json();
                
                if (data.symbols > 0) {
                    let html = '<h4>Found ' + data.symbols + ' symbols from Vantage MT5:</h4>';
                    
                    for (const [symbol, symbolData] of Object.entries(data.data)) {
                        html += '<div class="symbol">';
                        html += '<h4>' + symbol + '</h4>';
                        html += '<div class="price">Bid: ' + symbolData.bid.toFixed(symbolData.digits) + 
                               ' | Ask: ' + symbolData.ask.toFixed(symbolData.digits) + '</div>';
                        html += '<p>' + symbolData.description + '</p>';
                        html += '</div>';
                    }
                    
                    document.getElementById('symbols').innerHTML = html;
                } else {
                    document.getElementById('symbols').innerHTML = 
                        '<div class="symbol">No symbol data available</div>';
                }
                
            } catch (error) {
                document.getElementById('status').innerHTML = 
                    '❌ Connection Error: ' + error.message;
                document.getElementById('symbols').innerHTML = 
                    '<div class="symbol">Error loading data: ' + error.message + '</div>';
            }
        }
        
        // Update every 5 seconds
        setInterval(updateData, 5000);
        updateData(); // Initial load
    </script>
</body>
</html>
        '''
    
    async def start(self):
        """Start secure dashboard"""
        # Use mkcert certificates if available, otherwise generate self-signed
        mkcert_cert = "localhost+2.pem"
        mkcert_key = "localhost+2-key.pem"
        
        if os.path.exists(mkcert_cert) and os.path.exists(mkcert_key):
            cert_file = mkcert_cert
            key_file = mkcert_key
            print("🔐 Using mkcert trusted certificates")
        else:
            cert_file = "certs/simple_dashboard.crt"
            key_file = "certs/simple_dashboard.key"
            if not os.path.exists(cert_file) or not os.path.exists(key_file):
                self.generate_certificate(cert_file, key_file)
            print("🔐 Using self-signed certificates")
        
        config = uvicorn.Config(
            self.app,
            host="0.0.0.0",  # Bind to all interfaces
            port=8444,  # Use different port for HTTPS
            ssl_certfile=cert_file,
            ssl_keyfile=key_file,
            log_level="info"
        )
        server = uvicorn.Server(config)
        await server.serve()
    
    def generate_certificate(self, cert_file, key_file):
        """Generate self-signed certificate"""
        os.makedirs("certs", exist_ok=True)
        
        from cryptography import x509
        from cryptography.x509.oid import NameOID
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import rsa
        import ipaddress
        from datetime import datetime, timedelta
        
        # Generate private key
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
        )
        
        # Create certificate
        subject = issuer = x509.Name([
            x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
            x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, "WSL"),
            x509.NameAttribute(NameOID.LOCALITY_NAME, "Trading"),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "AI Trading System"),
            x509.NameAttribute(NameOID.COMMON_NAME, "simple.trading.local"),
        ])
        
        # Get WSL IP for SAN
        import subprocess
        try:
            result = subprocess.run(['hostname', '-I'], capture_output=True, text=True)
            wsl_ip = result.stdout.strip().split()[0]
        except:
            wsl_ip = "172.27.209.14"
        
        cert = x509.CertificateBuilder().subject_name(
            subject
        ).issuer_name(
            issuer
        ).public_key(
            private_key.public_key()
        ).serial_number(
            x509.random_serial_number()
        ).not_valid_before(
            datetime.utcnow()
        ).not_valid_after(
            datetime.utcnow() + timedelta(days=365)
        ).add_extension(
            x509.SubjectAlternativeName([
                x509.DNSName("localhost"),
                x509.DNSName("simple.trading.local"),
                x509.IPAddress(ipaddress.IPv4Address("127.0.0.1")),
                x509.IPAddress(ipaddress.IPv4Address(wsl_ip)),
            ]),
            critical=False,
        ).sign(private_key, hashes.SHA256())
        
        # Write certificate
        with open(cert_file, "wb") as f:
            f.write(cert.public_bytes(serialization.Encoding.PEM))
        
        # Write private key
        with open(key_file, "wb") as f:
            f.write(private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            ))
        
        print(f"✅ Generated TLS certificate: {cert_file}")
        print(f"✅ Generated private key: {key_file}")

async def main():
    print("🔐 STARTING ENCRYPTED SIMPLE DASHBOARD")
    print("=" * 60)
    print("This secure HTTPS dashboard uses TLS encryption")
    print("with self-signed certificates for Windows access")
    print()
    
    # Get WSL IP
    import subprocess
    try:
        result = subprocess.run(['hostname', '-I'], capture_output=True, text=True)
        wsl_ip = result.stdout.strip().split()[0]
        print(f"🖥️  WSL2 IP: {wsl_ip}")
        print(f"🔐 Secure Access: https://{wsl_ip}:8444")
    except:
        print("🔐 Secure Access: https://localhost:8444")
    
    print("=" * 60)
    
    dashboard = SimpleDashboard()
    await dashboard.start()

if __name__ == "__main__":
    asyncio.run(main())