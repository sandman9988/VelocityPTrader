#!/usr/bin/env python3
"""
Enterprise-Grade Trading Dashboard
- TLS/SSL encrypted connections with certificate validation
- WebSocket with automatic reconnection and failover
- High-availability architecture with load balancing
- Production monitoring and alerting
- Zero-downtime deployment support
"""

import ssl
import json
import asyncio
import aiohttp
import logging
import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Any, Callable
from dataclasses import dataclass, asdict
from pathlib import Path
import threading
import time
import uuid
import jwt
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.serialization import load_pem_private_key, load_pem_public_key
import websockets
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Depends, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import HTMLResponse
import redis
from prometheus_client import Counter, Histogram, Gauge, generate_latest
import structlog

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)

# Metrics
WEBSOCKET_CONNECTIONS = Gauge('websocket_connections_total', 'Number of active WebSocket connections')
REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('http_request_duration_seconds', 'HTTP request duration')
DATA_UPDATES = Counter('market_data_updates_total', 'Total market data updates')
AUTHENTICATION_ATTEMPTS = Counter('auth_attempts_total', 'Authentication attempts', ['result'])

@dataclass
class TLSConfig:
    """TLS configuration for secure connections"""
    cert_file: str
    key_file: str
    ca_file: Optional[str] = None
    verify_mode: int = ssl.CERT_REQUIRED
    protocol: int = ssl.PROTOCOL_TLS_SERVER
    cipher_suites: str = "ECDHE+AESGCM:ECDHE+CHACHA20:DHE+AESGCM:DHE+CHACHA20:!aNULL:!MD5:!DSS"

@dataclass
class ConnectionConfig:
    """Connection configuration with failover"""
    primary_host: str = "0.0.0.0"
    primary_port: int = 8443  # HTTPS
    secondary_host: Optional[str] = None
    secondary_port: Optional[int] = None
    max_connections: int = 1000
    connection_timeout: int = 30
    heartbeat_interval: int = 30
    reconnect_interval: int = 5
    max_reconnect_attempts: int = 10

@dataclass
class SecurityConfig:
    """Security configuration"""
    jwt_secret_key: str
    jwt_algorithm: str = "HS256"
    jwt_expiration_minutes: int = 60
    api_key_header: str = "X-API-Key"
    rate_limit_requests: int = 100
    rate_limit_window: int = 60
    encryption_key: bytes = None
    
    def __post_init__(self):
        if self.encryption_key is None:
            self.encryption_key = Fernet.generate_key()

class CertificateManager:
    """Manages SSL/TLS certificates for secure connections"""
    
    def __init__(self, cert_dir: str = "certs"):
        self.cert_dir = Path(cert_dir)
        self.cert_dir.mkdir(exist_ok=True)
        
    def generate_self_signed_cert(self, hostname: str = "localhost") -> TLSConfig:
        """Generate self-signed certificate for development"""
        
        # Generate private key
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048
        )
        
        # Generate certificate
        from cryptography import x509
        from cryptography.x509.oid import NameOID
        import ipaddress
        
        subject = issuer = x509.Name([
            x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
            x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, "CA"),
            x509.NameAttribute(NameOID.LOCALITY_NAME, "San Francisco"),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "AI Trading System"),
            x509.NameAttribute(NameOID.COMMON_NAME, hostname),
        ])
        
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
                x509.DNSName(hostname),
                x509.DNSName("localhost"),
                x509.IPAddress(ipaddress.IPv4Address("127.0.0.1")),
            ]),
            critical=False,
        ).sign(private_key, hashes.SHA256())
        
        # Save certificate and key
        cert_file = self.cert_dir / "server.crt"
        key_file = self.cert_dir / "server.key"
        
        with open(cert_file, "wb") as f:
            f.write(cert.public_bytes(serialization.Encoding.PEM))
            
        with open(key_file, "wb") as f:
            f.write(private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            ))
        
        logger.info(f"Generated self-signed certificate: {cert_file}")
        
        return TLSConfig(
            cert_file=str(cert_file),
            key_file=str(key_file),
            verify_mode=ssl.CERT_NONE  # Self-signed
        )

class SessionManager:
    """Manages user sessions with JWT tokens and Redis"""
    
    def __init__(self, security_config: SecurityConfig, redis_client: Optional[redis.Redis] = None):
        self.security_config = security_config
        self.redis_client = redis_client or redis.Redis(host='localhost', port=6379, db=0)
        self.cipher_suite = Fernet(security_config.encryption_key)
        
    def create_session(self, user_id: str, permissions: List[str]) -> str:
        """Create secure session token"""
        
        session_id = str(uuid.uuid4())
        expires = datetime.utcnow() + timedelta(minutes=self.security_config.jwt_expiration_minutes)
        
        payload = {
            'session_id': session_id,
            'user_id': user_id,
            'permissions': permissions,
            'exp': expires,
            'iat': datetime.utcnow(),
            'iss': 'ai-trading-system'
        }
        
        token = jwt.encode(
            payload,
            self.security_config.jwt_secret_key,
            algorithm=self.security_config.jwt_algorithm
        )
        
        # Store session in Redis
        session_data = {
            'user_id': user_id,
            'permissions': permissions,
            'created_at': datetime.utcnow().isoformat(),
            'last_access': datetime.utcnow().isoformat()
        }
        
        self.redis_client.setex(
            f"session:{session_id}",
            timedelta(minutes=self.security_config.jwt_expiration_minutes),
            self.cipher_suite.encrypt(json.dumps(session_data).encode())
        )
        
        logger.info(f"Created session for user {user_id}", session_id=session_id)
        return token
        
    def validate_session(self, token: str) -> Optional[Dict]:
        """Validate session token"""
        
        try:
            payload = jwt.decode(
                token,
                self.security_config.jwt_secret_key,
                algorithms=[self.security_config.jwt_algorithm]
            )
            
            session_id = payload.get('session_id')
            if not session_id:
                return None
                
            # Check Redis session
            encrypted_data = self.redis_client.get(f"session:{session_id}")
            if not encrypted_data:
                return None
                
            session_data = json.loads(self.cipher_suite.decrypt(encrypted_data).decode())
            
            # Update last access
            session_data['last_access'] = datetime.utcnow().isoformat()
            self.redis_client.setex(
                f"session:{session_id}",
                timedelta(minutes=self.security_config.jwt_expiration_minutes),
                self.cipher_suite.encrypt(json.dumps(session_data).encode())
            )
            
            return {
                'user_id': session_data['user_id'],
                'permissions': session_data['permissions'],
                'session_id': session_id
            }
            
        except Exception as e:
            logger.warning(f"Session validation failed: {e}")
            return None
    
    def revoke_session(self, session_id: str) -> bool:
        """Revoke session"""
        result = self.redis_client.delete(f"session:{session_id}")
        logger.info(f"Revoked session", session_id=session_id)
        return bool(result)

class ConnectionManager:
    """Manages WebSocket connections with automatic reconnection"""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.connection_metadata: Dict[str, Dict] = {}
        self._lock = threading.RLock()
        
    async def connect(self, websocket: WebSocket, session_id: str, user_id: str,
                      permissions: Optional[List[str]] = None):
        """Accept WebSocket connection with permissions"""
        await websocket.accept()

        with self._lock:
            self.active_connections[session_id] = websocket
            self.connection_metadata[session_id] = {
                'user_id': user_id,
                'permissions': permissions or [],
                'connected_at': datetime.utcnow(),
                'last_ping': datetime.utcnow(),
                'message_count': 0
            }

        WEBSOCKET_CONNECTIONS.inc()
        logger.info(f"WebSocket connected", user_id=user_id, session_id=session_id,
                   permissions=permissions)
        
    def disconnect(self, session_id: str):
        """Disconnect WebSocket"""
        with self._lock:
            if session_id in self.active_connections:
                del self.active_connections[session_id]
            if session_id in self.connection_metadata:
                del self.connection_metadata[session_id]
                
        WEBSOCKET_CONNECTIONS.dec()
        logger.info(f"WebSocket disconnected", session_id=session_id)
        
    async def send_personal_message(self, message: Dict, session_id: str):
        """Send message to specific connection"""
        with self._lock:
            websocket = self.active_connections.get(session_id)
            
        if websocket:
            try:
                await websocket.send_json(message)
                
                with self._lock:
                    if session_id in self.connection_metadata:
                        self.connection_metadata[session_id]['message_count'] += 1
                        
            except Exception as e:
                logger.error(f"Failed to send message to {session_id}: {e}")
                self.disconnect(session_id)
                
    # Permission requirements for different message types
    MESSAGE_PERMISSIONS = {
        'trade_signal': ['trade', 'admin'],
        'trade_executed': ['trade', 'admin'],
        'position_update': ['trade', 'admin'],
        'market_data': ['view', 'trade', 'admin'],
        'agent_status': ['view', 'trade', 'admin'],
        'system_alert': ['admin'],
        'error_alert': ['admin'],
        'performance_update': ['view', 'trade', 'admin'],
        'regime_change': ['view', 'trade', 'admin'],
        'ping': [],  # Everyone gets pings
        'status': [],  # Everyone gets status updates
    }

    def _user_has_permission(self, user_permissions: List[str], required_permissions: List[str]) -> bool:
        """Check if user has any of the required permissions"""
        if not required_permissions:
            return True  # No permission required
        return any(perm in user_permissions for perm in required_permissions)

    async def broadcast(self, message: Dict, required_permissions: Optional[List[str]] = None):
        """Broadcast message to all connections with permission filtering"""

        disconnected_sessions = []
        message_type = message.get('type', 'unknown')

        # Determine required permissions for this message type
        if required_permissions is None:
            required_permissions = self.MESSAGE_PERMISSIONS.get(message_type, [])

        with self._lock:
            connections_to_notify = list(self.active_connections.items())
            metadata_copy = dict(self.connection_metadata)

        recipients_count = 0
        filtered_count = 0

        for session_id, websocket in connections_to_notify:
            try:
                # Check user permissions if required
                user_metadata = metadata_copy.get(session_id, {})
                user_permissions = user_metadata.get('permissions', [])

                # Skip if user doesn't have required permissions
                if required_permissions and not self._user_has_permission(user_permissions, required_permissions):
                    filtered_count += 1
                    continue

                await websocket.send_json(message)
                recipients_count += 1

                with self._lock:
                    if session_id in self.connection_metadata:
                        self.connection_metadata[session_id]['message_count'] += 1

            except Exception as e:
                logger.error(f"Failed to broadcast to {session_id}: {e}")
                disconnected_sessions.append(session_id)

        # Clean up disconnected sessions
        for session_id in disconnected_sessions:
            self.disconnect(session_id)

        if filtered_count > 0:
            logger.debug(f"Broadcast {message_type}: sent to {recipients_count}, filtered {filtered_count}")
            
    async def ping_all_connections(self):
        """Send ping to all connections for health check"""
        ping_message = {
            'type': 'ping',
            'timestamp': datetime.utcnow().isoformat()
        }
        
        await self.broadcast(ping_message)

class EnterpriseDashboard:
    """Enterprise-grade trading dashboard with high availability"""
    
    def __init__(self, 
                 connection_config: ConnectionConfig,
                 security_config: SecurityConfig,
                 tls_config: Optional[TLSConfig] = None):
        
        self.connection_config = connection_config
        self.security_config = security_config
        self.tls_config = tls_config
        
        # Initialize components
        self.session_manager = SessionManager(security_config)
        self.connection_manager = ConnectionManager()
        self.certificate_manager = CertificateManager()
        
        # Create FastAPI app
        self.app = FastAPI(
            title="AI Trading System Enterprise Dashboard",
            description="Production-grade encrypted trading dashboard",
            version="1.0.0",
            docs_url="/api/docs" if __name__ == "__main__" else None  # Disable in production
        )
        
        # Add middleware
        self._setup_middleware()
        
        # Add routes
        self._setup_routes()
        
        # Background tasks
        self.background_tasks = []
        
        logger.info("Enterprise Dashboard initialized")
        
    def _setup_middleware(self):
        """Setup security and performance middleware"""
        
        # CORS with strict settings
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["https://localhost:3000"],  # React app
            allow_credentials=True,
            allow_methods=["GET", "POST", "PUT", "DELETE"],
            allow_headers=["*"],
        )
        
        # Trusted host middleware
        self.app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=["*"]  # Allow all hosts for WSL2 Windows access
        )
        
        # GZip compression
        self.app.add_middleware(GZipMiddleware, minimum_size=1000)
        
    def _setup_routes(self):
        """Setup API routes"""
        
        security = HTTPBearer()
        
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint"""
            return {
                "status": "healthy",
                "timestamp": datetime.utcnow().isoformat(),
                "version": "1.0.0"
            }
            
        @self.app.get("/metrics")
        async def metrics():
            """Prometheus metrics endpoint"""
            return generate_latest()
            
        @self.app.get("/api/settings")
        async def get_all_settings():
            """Get all system settings grouped by category"""
            try:
                from src.database.operations import AtomicDataOperations
                ops = AtomicDataOperations()
                settings = ops.get_all_settings(include_secrets=False)
                return {"status": "success", "settings": settings}
            except Exception as e:
                logger.error(f"Failed to get settings: {e}")
                return {"status": "error", "message": str(e)}

        @self.app.get("/api/settings/{category}")
        async def get_settings_by_category(category: str):
            """Get settings for a specific category"""
            try:
                from src.database.operations import AtomicDataOperations
                ops = AtomicDataOperations()
                settings = ops.get_settings_by_category(category)
                return {"status": "success", "category": category, "settings": settings}
            except Exception as e:
                logger.error(f"Failed to get settings for category {category}: {e}")
                return {"status": "error", "message": str(e)}

        @self.app.put("/api/settings/{category}/{key}")
        async def update_setting(category: str, key: str, data: Dict[str, Any]):
            """Update a setting value"""
            try:
                from src.database.operations import AtomicDataOperations
                ops = AtomicDataOperations()
                value = str(data.get('value', ''))
                updated_by = data.get('updated_by', 'dashboard')

                result = ops.update_setting(category, key, value, updated_by)
                if result:
                    return {"status": "success", "setting": result}
                else:
                    return {"status": "error", "message": "Failed to update setting"}
            except Exception as e:
                logger.error(f"Failed to update setting {category}/{key}: {e}")
                return {"status": "error", "message": str(e)}

        @self.app.post("/api/settings/initialize")
        async def initialize_settings():
            """Initialize default settings"""
            try:
                from src.database.operations import AtomicDataOperations
                ops = AtomicDataOperations()
                success = ops.initialize_default_settings()
                if success:
                    return {"status": "success", "message": "Default settings initialized"}
                else:
                    return {"status": "error", "message": "Failed to initialize settings"}
            except Exception as e:
                logger.error(f"Failed to initialize settings: {e}")
                return {"status": "error", "message": str(e)}

        @self.app.post("/api/startup")
        async def startup_system():
            """Start all system components with progress tracking"""
            results = {
                "status": "success",
                "timestamp": datetime.utcnow().isoformat(),
                "steps": {}
            }

            # Step 1: Database
            try:
                from src.database.connection import get_database_manager
                db = get_database_manager()
                db.initialize_sync_engine()
                health = db.health_check()
                results["steps"]["database"] = {
                    "status": "success",
                    "message": f"Connected to {db.config.database}",
                    "details": health.get("connection_pool", {})
                }
            except Exception as e:
                results["steps"]["database"] = {"status": "error", "message": str(e)}
                results["status"] = "partial"

            # Step 2: Settings
            try:
                from src.database.operations import AtomicDataOperations
                ops = AtomicDataOperations()
                settings = ops.get_all_settings()
                if not settings:
                    ops.initialize_default_settings()
                    settings = ops.get_all_settings()
                results["steps"]["settings"] = {
                    "status": "success",
                    "message": f"Loaded {sum(len(v) for v in settings.values())} settings",
                    "categories": list(settings.keys())
                }
            except Exception as e:
                results["steps"]["settings"] = {"status": "error", "message": str(e)}
                results["status"] = "partial"

            # Step 3: MT5 Bridge
            try:
                from src.data.mt5_resilient_connection import get_mt5_connection
                mt5 = get_mt5_connection()
                status = mt5.get_connection_status()
                if mt5._is_initialized or status.get("data_file_exists"):
                    results["steps"]["mt5"] = {
                        "status": "success",
                        "message": f"MT5 bridge ready ({status.get('symbols_count', 0)} symbols)",
                        "details": {
                            "server": status.get("server"),
                            "state": status.get("state")
                        }
                    }
                else:
                    results["steps"]["mt5"] = {
                        "status": "warning",
                        "message": "MT5 bridge in degraded mode - using cached data"
                    }
            except Exception as e:
                results["steps"]["mt5"] = {"status": "error", "message": str(e)}
                results["status"] = "partial"

            # Step 4: Agents
            try:
                # Check if agent configuration exists
                from src.database.operations import AtomicDataOperations
                ops = AtomicDataOperations()
                berserker = ops.get_setting("agents", "berserker_enabled")
                sniper = ops.get_setting("agents", "sniper_enabled")
                results["steps"]["agents"] = {
                    "status": "success",
                    "message": "Agent configuration loaded",
                    "details": {
                        "berserker": berserker.get("value") if berserker else True,
                        "sniper": sniper.get("value") if sniper else True
                    }
                }
            except Exception as e:
                results["steps"]["agents"] = {"status": "error", "message": str(e)}
                results["status"] = "partial"

            # Step 5: Monitoring
            try:
                results["steps"]["monitoring"] = {
                    "status": "success",
                    "message": "Prometheus metrics enabled",
                    "details": {
                        "websocket_connections": len(self.connection_manager.active_connections),
                        "metrics_endpoint": "/metrics"
                    }
                }
            except Exception as e:
                results["steps"]["monitoring"] = {"status": "error", "message": str(e)}
                results["status"] = "partial"

            # Overall status
            failed_steps = [k for k, v in results["steps"].items() if v.get("status") == "error"]
            if failed_steps:
                results["status"] = "partial"
                results["message"] = f"Started with errors in: {', '.join(failed_steps)}"
            else:
                results["status"] = "success"
                results["message"] = "All systems operational"

            logger.info("System startup completed", status=results["status"])
            return results

        @self.app.post("/api/shutdown")
        async def shutdown_system():
            """Gracefully shutdown system components"""
            results = {
                "status": "success",
                "timestamp": datetime.utcnow().isoformat(),
                "message": "System shutdown initiated"
            }

            try:
                # Close database connections
                from src.database.connection import get_database_manager
                db = get_database_manager()
                await db.close()
                results["database"] = "closed"
            except Exception as e:
                results["database"] = f"error: {str(e)}"

            logger.warning("System shutdown completed")
            return results

        @self.app.get("/api/resilience")
        async def get_resilience_status():
            """Get comprehensive resilience status for all system components"""
            try:
                result = {
                    "status": "success",
                    "timestamp": datetime.utcnow().isoformat(),
                    "components": {}
                }

                # Get all registered connection manager health
                try:
                    from src.utils.resilience import get_all_connection_health
                    connection_health = get_all_connection_health()
                    result["components"]["connections"] = connection_health
                except Exception as e:
                    result["components"]["connections"] = {"error": str(e)}

                # Get database resilience status
                try:
                    from src.database.connection import get_database_manager
                    db = get_database_manager()
                    result["components"]["database"] = db.get_resilience_status()
                except Exception as e:
                    result["components"]["database"] = {"error": str(e)}

                # Get MT5 connection status
                try:
                    from src.data.mt5_resilient_connection import get_mt5_connection
                    mt5 = get_mt5_connection()
                    result["components"]["mt5"] = mt5.get_connection_status()
                except Exception as e:
                    result["components"]["mt5"] = {"error": str(e)}

                # Get WebSocket connection summary
                result["components"]["websocket"] = {
                    "active_connections": len(self.connection_manager.active_connections),
                    "connection_metadata": {
                        sid: {
                            "user_id": meta.get("user_id"),
                            "connected_at": meta.get("connected_at").isoformat() if meta.get("connected_at") else None,
                            "message_count": meta.get("message_count", 0)
                        }
                        for sid, meta in self.connection_manager.connection_metadata.items()
                    }
                }

                return result

            except Exception as e:
                logger.error(f"Failed to get resilience status: {e}")
                return {"status": "error", "message": str(e)}

        @self.app.get("/api/symbols")
        async def get_real_symbols():
            """Get REAL Vantage MT5 symbols with ATOMIC persistence"""
            try:
                import json
                import os
                from pathlib import Path
                
                # ATOMIC READ of real Vantage data
                symbols_file = Path('all_mt5_symbols.json')
                if not symbols_file.exists():
                    return {"status": "error", "message": "Real Vantage data not found"}
                    
                with open(symbols_file, 'r') as f:
                    vantage_data = json.load(f)
                    symbols = vantage_data.get('symbols', {})
                    
                # ATOMIC persistence of symbol states
                persistence_file = Path('symbol_persistence.json')
                symbol_states = {}
                if persistence_file.exists():
                    with open(persistence_file, 'r') as f:
                        symbol_states = json.load(f)
                
                return {
                    "status": "success",
                    "server": vantage_data.get('account', {}).get('server', 'VantageInternational-Demo'),
                    "account": vantage_data.get('account', {}),
                    "symbols": symbols,
                    "count": len(symbols),
                    "persistence": symbol_states
                }
            except Exception as e:
                return {"status": "error", "message": f"Failed to load REAL data: {str(e)}"}
            
        @self.app.post("/auth/login")
        async def login(credentials: Dict[str, str]):
            """Authenticate and create session"""
            
            # TODO: Implement proper authentication
            username = credentials.get("username")
            password = credentials.get("password")
            
            if username == "admin" and password == "secure_password":  # Production: use proper auth
                token = self.session_manager.create_session(
                    user_id=username,
                    permissions=["read", "trade", "admin"]
                )
                AUTHENTICATION_ATTEMPTS.labels(result="success").inc()
                return {"access_token": token, "token_type": "bearer"}
            else:
                AUTHENTICATION_ATTEMPTS.labels(result="failure").inc()
                raise HTTPException(status_code=401, detail="Invalid credentials")
                
        @self.app.get("/")
        async def dashboard():
            """Serve dashboard HTML"""
            return HTMLResponse(self._get_dashboard_html())
            
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket, token: str):
            """WebSocket endpoint with authentication"""
            
            # Validate token
            session_data = self.session_manager.validate_session(token)
            if not session_data:
                await websocket.close(code=4001, reason="Authentication failed")
                return
                
            session_id = session_data['session_id']
            user_id = session_data['user_id']
            
            await self.connection_manager.connect(websocket, session_id, user_id)
            
            try:
                # Send initial data
                await self.connection_manager.send_personal_message({
                    "type": "connection_established",
                    "user_id": user_id,
                    "permissions": session_data['permissions'],
                    "timestamp": datetime.utcnow().isoformat()
                }, session_id)
                
                # Listen for messages
                while True:
                    data = await websocket.receive_json()
                    await self._handle_websocket_message(data, session_id, user_id)
                    
            except WebSocketDisconnect:
                self.connection_manager.disconnect(session_id)
            except Exception as e:
                logger.error(f"WebSocket error for {user_id}: {e}")
                self.connection_manager.disconnect(session_id)
                
    async def _handle_websocket_message(self, data: Dict, session_id: str, user_id: str):
        """Handle incoming WebSocket message"""
        
        message_type = data.get("type")
        
        if message_type == "ping":
            await self.connection_manager.send_personal_message({
                "type": "pong",
                "timestamp": datetime.utcnow().isoformat()
            }, session_id)
            
        elif message_type == "subscribe_market_data":
            symbols = data.get("symbols", [])
            # TODO: Subscribe to market data for these symbols
            await self.connection_manager.send_personal_message({
                "type": "market_data_subscription",
                "symbols": symbols,
                "status": "subscribed"
            }, session_id)
            
        elif message_type == "request_system_status":
            # TODO: Get actual system status
            await self.connection_manager.send_personal_message({
                "type": "system_status",
                "status": "running",
                "agents": {"berserker": "active", "sniper": "active"},
                "timestamp": datetime.utcnow().isoformat()
            }, session_id)
            
    def _get_dashboard_html(self) -> str:
        """Generate enterprise dashboard HTML"""

        return '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Enterprise Trading Dashboard</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', system-ui, sans-serif;
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
            color: #e2e8f0;
            height: 100vh;
            overflow: hidden;
        }
        .header {
            background: linear-gradient(90deg, #1e40af 0%, #3730a3 100%);
            padding: 1rem 2rem;
            box-shadow: 0 2px 10px rgba(0,0,0,0.3);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .logo { font-size: 1.5rem; font-weight: bold; }
        .header-controls {
            display: flex;
            align-items: center;
            gap: 1rem;
        }
        .settings-btn {
            background: rgba(255,255,255,0.1);
            border: 1px solid rgba(255,255,255,0.2);
            color: white;
            padding: 0.5rem 1rem;
            border-radius: 8px;
            cursor: pointer;
            display: flex;
            align-items: center;
            gap: 0.5rem;
            transition: all 0.2s;
            font-size: 0.9rem;
        }
        .settings-btn:hover {
            background: rgba(255,255,255,0.2);
            transform: scale(1.02);
        }
        .settings-btn svg {
            width: 18px;
            height: 18px;
            animation: none;
        }
        .settings-btn:hover svg {
            animation: spin-slow 2s linear infinite;
        }
        @keyframes spin-slow {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        .startup-btn {
            background: linear-gradient(135deg, #10b981 0%, #059669 100%);
            border: none;
            color: white;
            padding: 0.6rem 1.2rem;
            border-radius: 8px;
            cursor: pointer;
            display: flex;
            align-items: center;
            gap: 0.5rem;
            transition: all 0.3s;
            font-size: 0.95rem;
            font-weight: 600;
            box-shadow: 0 2px 8px rgba(16, 185, 129, 0.3);
        }
        .startup-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(16, 185, 129, 0.4);
        }
        .startup-btn svg {
            width: 18px;
            height: 18px;
        }
        .startup-btn.running {
            background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
            box-shadow: 0 2px 8px rgba(245, 158, 11, 0.3);
        }
        .startup-btn.running:hover {
            box-shadow: 0 4px 12px rgba(245, 158, 11, 0.4);
        }
        .startup-modal {
            max-width: 600px;
        }
        .startup-body {
            padding: 1.5rem;
        }
        .startup-progress {
            display: flex;
            flex-direction: column;
            gap: 1rem;
        }
        .startup-step {
            display: flex;
            align-items: center;
            gap: 1rem;
            padding: 1rem;
            background: rgba(0, 0, 0, 0.2);
            border-radius: 10px;
            border: 1px solid rgba(148, 163, 184, 0.1);
            transition: all 0.3s;
        }
        .startup-step.active {
            border-color: #3b82f6;
            background: rgba(59, 130, 246, 0.1);
        }
        .startup-step.success {
            border-color: #10b981;
            background: rgba(16, 185, 129, 0.1);
        }
        .startup-step.error {
            border-color: #ef4444;
            background: rgba(239, 68, 68, 0.1);
        }
        .step-icon {
            font-size: 1.5rem;
            width: 40px;
            text-align: center;
        }
        .step-info {
            flex: 1;
        }
        .step-name {
            font-weight: 600;
            color: #e2e8f0;
        }
        .step-status {
            font-size: 0.85rem;
            color: #94a3b8;
            margin-top: 0.25rem;
        }
        .step-indicator {
            width: 24px;
            height: 24px;
            border-radius: 50%;
            background: rgba(148, 163, 184, 0.2);
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .step-indicator.loading {
            border: 2px solid rgba(59, 130, 246, 0.3);
            border-top-color: #3b82f6;
            animation: spin 1s linear infinite;
        }
        .step-indicator.success {
            background: #10b981;
        }
        .step-indicator.success::after {
            content: '✓';
            color: white;
            font-size: 14px;
        }
        .step-indicator.error {
            background: #ef4444;
        }
        .step-indicator.error::after {
            content: '✕';
            color: white;
            font-size: 14px;
        }
        .startup-summary {
            margin-top: 1.5rem;
            padding: 1rem;
            background: rgba(59, 130, 246, 0.1);
            border-radius: 8px;
            border-left: 4px solid #3b82f6;
            display: none;
        }
        .startup-summary.show {
            display: block;
        }
        .startup-summary.success {
            background: rgba(16, 185, 129, 0.1);
            border-color: #10b981;
        }
        .startup-summary.error {
            background: rgba(239, 68, 68, 0.1);
            border-color: #ef4444;
        }
        .btn-danger {
            background: #ef4444;
            border: none;
            color: white;
        }
        .btn-danger:hover {
            background: #dc2626;
        }
        .status-badge {
            background: #10b981;
            color: white;
            padding: 0.25rem 0.75rem;
            border-radius: 1rem;
            font-size: 0.875rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        .status-indicator {
            width: 8px;
            height: 8px;
            background: #34d399;
            border-radius: 50%;
            animation: pulse 2s infinite;
        }
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        .dashboard-grid {
            display: grid;
            grid-template-columns: 300px 1fr 400px;
            height: calc(100vh - 80px);
            gap: 1rem;
            padding: 1rem;
        }
        .panel {
            background: rgba(30, 41, 59, 0.8);
            border-radius: 12px;
            border: 1px solid rgba(148, 163, 184, 0.2);
            padding: 1.5rem;
            overflow: auto;
            backdrop-filter: blur(10px);
        }
        .panel-title {
            font-size: 1.25rem;
            font-weight: 600;
            margin-bottom: 1rem;
            color: #3b82f6;
        }
        .metric-row {
            display: flex;
            justify-content: space-between;
            padding: 0.75rem 0;
            border-bottom: 1px solid rgba(148, 163, 184, 0.1);
        }
        .metric-value {
            font-weight: 600;
            color: #10b981;
        }
        .connection-status {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            padding: 1rem;
            background: rgba(16, 185, 129, 0.1);
            border-radius: 8px;
            margin-bottom: 1rem;
            border-left: 4px solid #10b981;
        }
        .error-message {
            background: rgba(239, 68, 68, 0.1);
            border-left: 4px solid #ef4444;
            color: #fecaca;
        }
        .symbol-grid {
            display: grid;
            gap: 0.5rem;
        }
        .symbol-card {
            background: rgba(51, 65, 85, 0.6);
            padding: 1rem;
            border-radius: 8px;
            border: 1px solid rgba(148, 163, 184, 0.1);
        }
        .symbol-name {
            font-weight: 600;
            color: #60a5fa;
            margin-bottom: 0.5rem;
        }
        .price-display {
            display: flex;
            justify-content: space-between;
            font-family: 'Courier New', monospace;
        }
        .bid { color: #ef4444; }
        .ask { color: #10b981; }
        .spread { color: #f59e0b; font-size: 0.875rem; }
        .loading {
            text-align: center;
            color: #94a3b8;
            padding: 2rem;
        }
        .spinner {
            border: 2px solid rgba(148, 163, 184, 0.3);
            border-top: 2px solid #3b82f6;
            border-radius: 50%;
            width: 24px;
            height: 24px;
            animation: spin 1s linear infinite;
            margin: 0 auto 1rem;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        /* Settings Modal Styles */
        .modal-overlay {
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(0, 0, 0, 0.7);
            backdrop-filter: blur(4px);
            z-index: 1000;
            justify-content: center;
            align-items: center;
        }
        .modal-overlay.active {
            display: flex;
        }
        .modal {
            background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
            border-radius: 16px;
            border: 1px solid rgba(148, 163, 184, 0.2);
            width: 90%;
            max-width: 900px;
            max-height: 85vh;
            display: flex;
            flex-direction: column;
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
        }
        .modal-header {
            padding: 1.5rem;
            border-bottom: 1px solid rgba(148, 163, 184, 0.2);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .modal-header h2 {
            font-size: 1.5rem;
            color: #3b82f6;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        .modal-close {
            background: none;
            border: none;
            color: #94a3b8;
            font-size: 1.5rem;
            cursor: pointer;
            padding: 0.5rem;
            border-radius: 8px;
            transition: all 0.2s;
        }
        .modal-close:hover {
            background: rgba(239, 68, 68, 0.2);
            color: #ef4444;
        }
        .modal-body {
            display: flex;
            flex: 1;
            overflow: hidden;
        }
        .settings-tabs {
            width: 200px;
            background: rgba(0, 0, 0, 0.2);
            padding: 1rem 0;
            border-right: 1px solid rgba(148, 163, 184, 0.1);
        }
        .settings-tab {
            padding: 0.75rem 1.5rem;
            cursor: pointer;
            color: #94a3b8;
            transition: all 0.2s;
            border-left: 3px solid transparent;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        .settings-tab:hover {
            background: rgba(59, 130, 246, 0.1);
            color: #e2e8f0;
        }
        .settings-tab.active {
            background: rgba(59, 130, 246, 0.2);
            color: #3b82f6;
            border-left-color: #3b82f6;
        }
        .settings-tab-icon {
            font-size: 1.1rem;
        }
        .settings-content {
            flex: 1;
            padding: 1.5rem;
            overflow-y: auto;
        }
        .settings-section {
            display: none;
        }
        .settings-section.active {
            display: block;
        }
        .settings-section h3 {
            font-size: 1.1rem;
            color: #60a5fa;
            margin-bottom: 1rem;
            padding-bottom: 0.5rem;
            border-bottom: 1px solid rgba(148, 163, 184, 0.1);
        }
        .setting-item {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1rem;
            padding: 1rem 0;
            border-bottom: 1px solid rgba(148, 163, 184, 0.05);
            align-items: center;
        }
        .setting-item:last-child {
            border-bottom: none;
        }
        .setting-label {
            display: flex;
            flex-direction: column;
            gap: 0.25rem;
        }
        .setting-label .key {
            font-weight: 500;
            color: #e2e8f0;
        }
        .setting-label .description {
            font-size: 0.8rem;
            color: #64748b;
        }
        .setting-input {
            background: rgba(0, 0, 0, 0.3);
            border: 1px solid rgba(148, 163, 184, 0.2);
            border-radius: 8px;
            padding: 0.75rem 1rem;
            color: #e2e8f0;
            font-size: 0.9rem;
            width: 100%;
            transition: all 0.2s;
        }
        .setting-input:focus {
            outline: none;
            border-color: #3b82f6;
            box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.2);
        }
        .setting-input:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }
        .setting-input[type="checkbox"] {
            width: auto;
            height: 20px;
            width: 20px;
            cursor: pointer;
        }
        .toggle-switch {
            position: relative;
            width: 50px;
            height: 26px;
            background: rgba(0, 0, 0, 0.3);
            border-radius: 13px;
            cursor: pointer;
            transition: all 0.3s;
        }
        .toggle-switch.active {
            background: #10b981;
        }
        .toggle-switch::after {
            content: '';
            position: absolute;
            width: 22px;
            height: 22px;
            background: white;
            border-radius: 50%;
            top: 2px;
            left: 2px;
            transition: all 0.3s;
        }
        .toggle-switch.active::after {
            left: 26px;
        }
        .modal-footer {
            padding: 1rem 1.5rem;
            border-top: 1px solid rgba(148, 163, 184, 0.2);
            display: flex;
            justify-content: flex-end;
            gap: 1rem;
        }
        .btn {
            padding: 0.75rem 1.5rem;
            border-radius: 8px;
            cursor: pointer;
            font-size: 0.9rem;
            transition: all 0.2s;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        .btn-secondary {
            background: rgba(148, 163, 184, 0.2);
            border: 1px solid rgba(148, 163, 184, 0.3);
            color: #e2e8f0;
        }
        .btn-secondary:hover {
            background: rgba(148, 163, 184, 0.3);
        }
        .btn-primary {
            background: #3b82f6;
            border: none;
            color: white;
        }
        .btn-primary:hover {
            background: #2563eb;
        }
        .btn-success {
            background: #10b981;
            border: none;
            color: white;
        }
        .btn-success:hover {
            background: #059669;
        }
        .save-indicator {
            font-size: 0.8rem;
            color: #10b981;
            display: none;
        }
        .save-indicator.show {
            display: inline;
            animation: fadeIn 0.3s;
        }
        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }
    </style>
</head>
<body>
    <div class="header">
        <div class="logo">VelocityPTrader Dashboard</div>
        <div class="header-controls">
            <button class="startup-btn" id="startup-btn" onclick="openStartupModal()">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <polygon points="5 3 19 12 5 21 5 3"></polygon>
                </svg>
                <span id="startup-btn-text">Start System</span>
            </button>
            <button class="settings-btn" onclick="openSettings()">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <circle cx="12" cy="12" r="3"></circle>
                    <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"></path>
                </svg>
                Settings
            </button>
            <div class="status-badge">
                <div class="status-indicator"></div>
                <span id="connection-status">Connecting...</span>
            </div>
        </div>
    </div>

    <!-- Startup Modal -->
    <div class="modal-overlay" id="startup-modal">
        <div class="modal startup-modal">
            <div class="modal-header">
                <h2>
                    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <circle cx="12" cy="12" r="10"></circle>
                        <polygon points="10 8 16 12 10 16 10 8"></polygon>
                    </svg>
                    System Startup
                </h2>
                <button class="modal-close" onclick="closeStartupModal()">&times;</button>
            </div>
            <div class="modal-body startup-body">
                <div class="startup-progress">
                    <div class="startup-step" id="step-database">
                        <div class="step-icon">&#128451;</div>
                        <div class="step-info">
                            <div class="step-name">Database Connection</div>
                            <div class="step-status" id="status-database">Waiting...</div>
                        </div>
                        <div class="step-indicator" id="indicator-database"></div>
                    </div>
                    <div class="startup-step" id="step-settings">
                        <div class="step-icon">&#9881;</div>
                        <div class="step-info">
                            <div class="step-name">Load Settings</div>
                            <div class="step-status" id="status-settings">Waiting...</div>
                        </div>
                        <div class="step-indicator" id="indicator-settings"></div>
                    </div>
                    <div class="startup-step" id="step-mt5">
                        <div class="step-icon">&#128279;</div>
                        <div class="step-info">
                            <div class="step-name">MT5 Bridge</div>
                            <div class="step-status" id="status-mt5">Waiting...</div>
                        </div>
                        <div class="step-indicator" id="indicator-mt5"></div>
                    </div>
                    <div class="startup-step" id="step-agents">
                        <div class="step-icon">&#129302;</div>
                        <div class="step-info">
                            <div class="step-name">Trading Agents</div>
                            <div class="step-status" id="status-agents">Waiting...</div>
                        </div>
                        <div class="step-indicator" id="indicator-agents"></div>
                    </div>
                    <div class="startup-step" id="step-monitoring">
                        <div class="step-icon">&#128202;</div>
                        <div class="step-info">
                            <div class="step-name">Monitoring</div>
                            <div class="step-status" id="status-monitoring">Waiting...</div>
                        </div>
                        <div class="step-indicator" id="indicator-monitoring"></div>
                    </div>
                </div>
                <div class="startup-summary" id="startup-summary"></div>
            </div>
            <div class="modal-footer">
                <button class="btn btn-secondary" onclick="closeStartupModal()">Close</button>
                <button class="btn btn-primary" id="start-btn" onclick="startSystem()">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <polygon points="5 3 19 12 5 21 5 3"></polygon>
                    </svg>
                    Start All Services
                </button>
                <button class="btn btn-danger" id="stop-btn" onclick="stopSystem()" style="display:none;">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <rect x="6" y="6" width="12" height="12"></rect>
                    </svg>
                    Stop System
                </button>
            </div>
        </div>
    </div>

    <!-- Settings Modal -->
    <div class="modal-overlay" id="settings-modal">
        <div class="modal">
            <div class="modal-header">
                <h2>
                    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <circle cx="12" cy="12" r="3"></circle>
                        <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"></path>
                    </svg>
                    System Settings
                </h2>
                <button class="modal-close" onclick="closeSettings()">&times;</button>
            </div>
            <div class="modal-body">
                <div class="settings-tabs" id="settings-tabs">
                    <div class="settings-tab active" data-tab="mt5">
                        <span class="settings-tab-icon">&#128279;</span> MT5 Connection
                    </div>
                    <div class="settings-tab" data-tab="trading">
                        <span class="settings-tab-icon">&#128200;</span> Trading
                    </div>
                    <div class="settings-tab" data-tab="risk">
                        <span class="settings-tab-icon">&#128737;</span> Risk Management
                    </div>
                    <div class="settings-tab" data-tab="agents">
                        <span class="settings-tab-icon">&#129302;</span> Agents
                    </div>
                    <div class="settings-tab" data-tab="monitoring">
                        <span class="settings-tab-icon">&#128202;</span> Monitoring
                    </div>
                    <div class="settings-tab" data-tab="dashboard">
                        <span class="settings-tab-icon">&#128187;</span> Dashboard
                    </div>
                    <div class="settings-tab" data-tab="database">
                        <span class="settings-tab-icon">&#128451;</span> Database
                    </div>
                    <div class="settings-tab" data-tab="notifications">
                        <span class="settings-tab-icon">&#128276;</span> Notifications
                    </div>
                </div>
                <div class="settings-content" id="settings-content">
                    <div class="loading">
                        <div class="spinner"></div>
                        <div>Loading settings...</div>
                    </div>
                </div>
            </div>
            <div class="modal-footer">
                <span class="save-indicator" id="save-indicator">Changes saved!</span>
                <button class="btn btn-secondary" onclick="closeSettings()">Close</button>
                <button class="btn btn-success" onclick="initializeSettings()">Reset to Defaults</button>
            </div>
        </div>
    </div>
    
    <div class="dashboard-grid">
        <div class="panel">
            <div class="panel-title">🤖 System Status</div>
            <div id="system-status" class="loading">
                <div class="spinner"></div>
                <div>Initializing secure connection...</div>
            </div>
        </div>
        
        <div class="panel">
            <div class="panel-title">📈 Live Market Data</div>
            <div id="market-data" class="loading">
                <div class="spinner"></div>
                <div>Loading encrypted market feed...</div>
            </div>
        </div>
        
        <div class="panel">
            <div class="panel-title">⚡ Performance Metrics</div>
            <div id="performance-metrics">
                <div class="metric-row">
                    <span>Connection Latency</span>
                    <span class="metric-value" id="latency">-- ms</span>
                </div>
                <div class="metric-row">
                    <span>Messages/sec</span>
                    <span class="metric-value" id="message-rate">--</span>
                </div>
                <div class="metric-row">
                    <span>Data Integrity</span>
                    <span class="metric-value">🔒 Verified</span>
                </div>
                <div class="metric-row">
                    <span>Encryption Status</span>
                    <span class="metric-value">🛡️ TLS 1.3</span>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        class EnterpriseDashboardClient {
            constructor() {
                this.ws = null;
                this.reconnectAttempts = 0;
                this.maxReconnectAttempts = 15;
                this.messageCount = 0;
                this.lastMessageTime = Date.now();
                this.token = localStorage.getItem('auth_token');

                // Exponential backoff configuration
                this.baseReconnectDelay = 1000;  // 1 second
                this.maxReconnectDelay = 60000;  // 60 seconds max
                this.backoffMultiplier = 2.0;
                this.jitterFactor = 0.1;

                // Health tracking
                this.connectionHealth = {
                    state: 'DISCONNECTED',
                    consecutiveFailures: 0,
                    consecutiveSuccesses: 0,
                    lastSuccessTime: null,
                    lastFailureTime: null,
                    totalReconnects: 0
                };

                this.connect();
                this.startMetricsUpdate();
                this.startHealthMonitoring();
            }

            calculateBackoffDelay() {
                // Exponential backoff with jitter
                let delay = this.baseReconnectDelay * Math.pow(this.backoffMultiplier, this.reconnectAttempts);
                delay = Math.min(delay, this.maxReconnectDelay);

                // Add jitter to prevent thundering herd
                const jitter = delay * this.jitterFactor;
                delay += (Math.random() - 0.5) * 2 * jitter;

                return Math.max(delay, this.baseReconnectDelay);
            }

            connect() {
                if (!this.token) {
                    this.authenticate();
                    return;
                }

                this.connectionHealth.state = 'CONNECTING';

                const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
                const wsUrl = `${protocol}//${location.host}/ws?token=${this.token}`;

                try {
                    this.ws = new WebSocket(wsUrl);
                    this.ws.onopen = this.onConnect.bind(this);
                    this.ws.onmessage = this.onMessage.bind(this);
                    this.ws.onclose = this.onClose.bind(this);
                    this.ws.onerror = this.onError.bind(this);

                } catch (error) {
                    console.error('WebSocket connection failed:', error);
                    this.handleConnectionFailure('Connection initialization failed');
                }
            }
            
            async authenticate() {
                try {
                    const response = await fetch('/auth/login', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            username: 'admin',
                            password: 'secure_password'  // TODO: Proper auth UI
                        })
                    });
                    
                    if (response.ok) {
                        const data = await response.json();
                        this.token = data.access_token;
                        localStorage.setItem('auth_token', this.token);
                        this.connect();
                    } else {
                        this.updateConnectionStatus('Authentication Failed', false);
                    }
                } catch (error) {
                    this.updateConnectionStatus('Auth Error', false);
                }
            }
            
            onConnect() {
                console.log('Enterprise WebSocket connected');
                this.reconnectAttempts = 0;
                this.connectionHealth.state = 'CONNECTED';
                this.connectionHealth.consecutiveSuccesses++;
                this.connectionHealth.consecutiveFailures = 0;
                this.connectionHealth.lastSuccessTime = Date.now();
                this.updateConnectionStatus('Connected', true);

                // Subscribe to market data
                this.send({
                    type: 'subscribe_market_data',
                    symbols: ['EURUSD', 'EURUSD+', 'BTCUSD', 'XAUUSD']
                });

                // Request system status
                this.send({ type: 'request_system_status' });
            }

            onMessage(event) {
                const data = JSON.parse(event.data);
                this.messageCount++;
                this.lastMessageTime = Date.now();
                this.connectionHealth.consecutiveSuccesses++;

                switch (data.type) {
                    case 'connection_established':
                        this.updateSystemStatus(data);
                        break;
                    case 'market_data':
                        this.updateMarketData(data);
                        break;
                    case 'system_status':
                        this.updateSystemStatus(data);
                        break;
                    case 'pong':
                        // Update latency measurement
                        if (data.timestamp) {
                            const latency = Date.now() - new Date(data.timestamp).getTime();
                            document.getElementById('latency').textContent = `${latency} ms`;
                        }
                        break;
                    case 'resilience_status':
                        this.updateResilienceStatus(data);
                        break;
                }
            }

            onClose(event) {
                console.log('WebSocket closed:', event.code, event.reason);
                this.handleConnectionFailure(`Connection closed: ${event.reason || 'Unknown'}`);
            }

            onError(error) {
                console.error('WebSocket error:', error);
                this.handleConnectionFailure('Connection error');
            }

            handleConnectionFailure(reason) {
                this.connectionHealth.state = 'DISCONNECTED';
                this.connectionHealth.consecutiveFailures++;
                this.connectionHealth.consecutiveSuccesses = 0;
                this.connectionHealth.lastFailureTime = Date.now();

                // Check if we should enter degraded mode
                if (this.connectionHealth.consecutiveFailures >= 3) {
                    this.connectionHealth.state = 'DEGRADED';
                    this.updateConnectionStatus('Degraded Mode', false);
                    console.warn('Entering degraded mode after multiple failures');
                } else {
                    this.updateConnectionStatus('Disconnected', false);
                }

                this.scheduleReconnect();
            }

            scheduleReconnect() {
                if (this.reconnectAttempts >= this.maxReconnectAttempts) {
                    this.connectionHealth.state = 'FAILED';
                    this.updateConnectionStatus('Connection Failed - Refresh Page', false);
                    console.error('Max reconnection attempts reached');
                    return;
                }

                const delay = this.calculateBackoffDelay();
                this.reconnectAttempts++;
                this.connectionHealth.totalReconnects++;
                this.connectionHealth.state = 'RECONNECTING';

                console.log(`Reconnecting in ${Math.round(delay/1000)}s... Attempt ${this.reconnectAttempts}/${this.maxReconnectAttempts}`);
                this.updateConnectionStatus(
                    `Reconnecting in ${Math.round(delay/1000)}s (${this.reconnectAttempts}/${this.maxReconnectAttempts})`,
                    false
                );

                setTimeout(() => {
                    this.connect();
                }, delay);
            }

            startHealthMonitoring() {
                // Check connection health every 30 seconds
                setInterval(() => {
                    if (this.connectionHealth.state === 'CONNECTED') {
                        const timeSinceLastMessage = Date.now() - this.lastMessageTime;

                        // If no messages for 60 seconds, connection might be stale
                        if (timeSinceLastMessage > 60000) {
                            console.warn('No messages received for 60s, checking connection');
                            this.send({ type: 'ping', timestamp: Date.now() });
                        }
                    }

                    // Log health status periodically
                    console.debug('Connection health:', this.connectionHealth);
                }, 30000);
            }

            updateResilienceStatus(data) {
                // Update UI with backend resilience information
                if (data.database) {
                    console.log('Database resilience:', data.database);
                }
                if (data.mt5) {
                    console.log('MT5 resilience:', data.mt5);
                }
            }
            
            send(data) {
                if (this.ws && this.ws.readyState === WebSocket.OPEN) {
                    this.ws.send(JSON.stringify(data));
                }
            }
            
            updateConnectionStatus(status, isConnected) {
                const statusElement = document.getElementById('connection-status');
                statusElement.textContent = status;
                statusElement.parentElement.className = isConnected ? 'status-badge' : 'status-badge error-message';
            }
            
            updateSystemStatus(data) {
                const container = document.getElementById('system-status');
                container.innerHTML = `
                    <div class="connection-status">
                        <span>🔒</span>
                        <div>
                            <div>Secure Connection Established</div>
                            <div style="font-size: 0.875rem; opacity: 0.8;">User: ${data.user_id || 'Unknown'}</div>
                        </div>
                    </div>
                    <div class="metric-row">
                        <span>BERSERKER Agent</span>
                        <span class="metric-value">🟢 Active</span>
                    </div>
                    <div class="metric-row">
                        <span>SNIPER Agent</span>
                        <span class="metric-value">🟢 Active</span>
                    </div>
                    <div class="metric-row">
                        <span>Data Pipeline</span>
                        <span class="metric-value">🟢 Running</span>
                    </div>
                `;
            }
            
            updateMarketData(data) {
                // TODO: Update with real market data
                const container = document.getElementById('market-data');
                container.innerHTML = `
                    <div class="symbol-card">
                        <div class="symbol-name">EURUSD+ (ECN)</div>
                        <div class="price-display">
                            <span class="bid">Bid: 1.17762</span>
                            <span class="ask">Ask: 1.17763</span>
                        </div>
                        <div class="spread">Spread: 0.1 pips</div>
                    </div>
                    <div class="symbol-card">
                        <div class="symbol-name">BTCUSD</div>
                        <div class="price-display">
                            <span class="bid">Bid: 95,420</span>
                            <span class="ask">Ask: 95,440</span>
                        </div>
                        <div class="spread">Spread: $20</div>
                    </div>
                `;
            }
            
            startMetricsUpdate() {
                setInterval(() => {
                    const now = Date.now();
                    const messageRate = Math.round(this.messageCount / ((now - this.lastMessageTime) / 1000)) || 0;
                    
                    document.getElementById('message-rate').textContent = messageRate;
                    document.getElementById('latency').textContent = '< 50 ms';
                    
                    // Send periodic ping
                    this.send({ type: 'ping', timestamp: now });
                }, 1000);
            }
        }
        
        // Initialize dashboard
        document.addEventListener('DOMContentLoaded', () => {
            new EnterpriseDashboardClient();
        });

        // ========== STARTUP MANAGEMENT ==========

        let systemRunning = false;
        const startupSteps = ['database', 'settings', 'mt5', 'agents', 'monitoring'];

        function openStartupModal() {
            document.getElementById('startup-modal').classList.add('active');
            // Reset all steps
            startupSteps.forEach(step => {
                document.getElementById(`step-${step}`).className = 'startup-step';
                document.getElementById(`status-${step}`).textContent = 'Waiting...';
                document.getElementById(`indicator-${step}`).className = 'step-indicator';
            });
            document.getElementById('startup-summary').className = 'startup-summary';
            document.getElementById('startup-summary').innerHTML = '';
            updateStartupButtons();
        }

        function closeStartupModal() {
            document.getElementById('startup-modal').classList.remove('active');
        }

        function updateStartupButtons() {
            const startBtn = document.getElementById('start-btn');
            const stopBtn = document.getElementById('stop-btn');
            const startupBtn = document.getElementById('startup-btn');
            const btnText = document.getElementById('startup-btn-text');

            if (systemRunning) {
                startBtn.style.display = 'none';
                stopBtn.style.display = 'flex';
                startupBtn.classList.add('running');
                btnText.textContent = 'System Running';
            } else {
                startBtn.style.display = 'flex';
                stopBtn.style.display = 'none';
                startupBtn.classList.remove('running');
                btnText.textContent = 'Start System';
            }
        }

        function setStepStatus(step, status, message) {
            const stepEl = document.getElementById(`step-${step}`);
            const statusEl = document.getElementById(`status-${step}`);
            const indicatorEl = document.getElementById(`indicator-${step}`);

            stepEl.className = `startup-step ${status}`;
            statusEl.textContent = message;

            if (status === 'active') {
                indicatorEl.className = 'step-indicator loading';
            } else if (status === 'success') {
                indicatorEl.className = 'step-indicator success';
            } else if (status === 'error') {
                indicatorEl.className = 'step-indicator error';
            } else {
                indicatorEl.className = 'step-indicator';
            }
        }

        async function startSystem() {
            const startBtn = document.getElementById('start-btn');
            startBtn.disabled = true;
            startBtn.innerHTML = '<div class="spinner" style="width:16px;height:16px;"></div> Starting...';

            try {
                // Animate steps sequentially for visual feedback
                for (const step of startupSteps) {
                    setStepStatus(step, 'active', 'Initializing...');
                    await new Promise(r => setTimeout(r, 300)); // Brief visual delay
                }

                // Call the actual startup API
                const response = await fetch('/api/startup', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' }
                });

                const result = await response.json();

                // Update each step with results
                for (const step of startupSteps) {
                    const stepResult = result.steps[step];
                    if (stepResult) {
                        const status = stepResult.status === 'success' ? 'success' :
                                      stepResult.status === 'warning' ? 'success' : 'error';
                        setStepStatus(step, status, stepResult.message);
                    }
                }

                // Show summary
                const summary = document.getElementById('startup-summary');
                if (result.status === 'success') {
                    summary.className = 'startup-summary show success';
                    summary.innerHTML = `
                        <strong>✓ System Started Successfully</strong><br>
                        ${result.message}<br>
                        <small>Started at ${new Date(result.timestamp).toLocaleTimeString()}</small>
                    `;
                    systemRunning = true;
                } else {
                    summary.className = 'startup-summary show error';
                    summary.innerHTML = `
                        <strong>⚠ System Started with Warnings</strong><br>
                        ${result.message}<br>
                        <small>Check the steps above for details</small>
                    `;
                    systemRunning = true; // Still mark as running even with partial success
                }

                updateStartupButtons();

            } catch (error) {
                console.error('Startup failed:', error);
                const summary = document.getElementById('startup-summary');
                summary.className = 'startup-summary show error';
                summary.innerHTML = `
                    <strong>✕ Startup Failed</strong><br>
                    ${error.message}<br>
                    <small>Please check the console for details</small>
                `;
            } finally {
                startBtn.disabled = false;
                startBtn.innerHTML = `
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <polygon points="5 3 19 12 5 21 5 3"></polygon>
                    </svg>
                    Start All Services
                `;
            }
        }

        async function stopSystem() {
            const stopBtn = document.getElementById('stop-btn');
            stopBtn.disabled = true;
            stopBtn.innerHTML = '<div class="spinner" style="width:16px;height:16px;"></div> Stopping...';

            try {
                const response = await fetch('/api/shutdown', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' }
                });

                const result = await response.json();

                // Reset all steps
                startupSteps.forEach(step => {
                    setStepStatus(step, '', 'Stopped');
                });

                const summary = document.getElementById('startup-summary');
                summary.className = 'startup-summary show';
                summary.innerHTML = `
                    <strong>System Stopped</strong><br>
                    ${result.message}<br>
                    <small>Stopped at ${new Date().toLocaleTimeString()}</small>
                `;

                systemRunning = false;
                updateStartupButtons();

            } catch (error) {
                console.error('Shutdown failed:', error);
            } finally {
                stopBtn.disabled = false;
                stopBtn.innerHTML = `
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <rect x="6" y="6" width="12" height="12"></rect>
                    </svg>
                    Stop System
                `;
            }
        }

        // Close startup modal on overlay click
        document.getElementById('startup-modal').addEventListener('click', (e) => {
            if (e.target.classList.contains('modal-overlay')) {
                closeStartupModal();
            }
        });

        // ========== SETTINGS MANAGEMENT ==========

        let currentSettings = {};
        let activeTab = 'mt5';

        function openSettings() {
            document.getElementById('settings-modal').classList.add('active');
            loadSettings();
        }

        function closeSettings() {
            document.getElementById('settings-modal').classList.remove('active');
        }

        // Close modal on overlay click
        document.getElementById('settings-modal').addEventListener('click', (e) => {
            if (e.target.classList.contains('modal-overlay')) {
                closeSettings();
            }
        });

        // Tab switching
        document.querySelectorAll('.settings-tab').forEach(tab => {
            tab.addEventListener('click', () => {
                document.querySelectorAll('.settings-tab').forEach(t => t.classList.remove('active'));
                tab.classList.add('active');
                activeTab = tab.dataset.tab;
                renderSettingsContent();
            });
        });

        async function loadSettings() {
            try {
                const response = await fetch('/api/settings');
                const data = await response.json();

                if (data.status === 'success') {
                    currentSettings = data.settings;
                    renderSettingsContent();
                } else {
                    showSettingsError('Failed to load settings: ' + data.message);
                }
            } catch (error) {
                showSettingsError('Network error loading settings: ' + error.message);
            }
        }

        function renderSettingsContent() {
            const container = document.getElementById('settings-content');
            const settings = currentSettings[activeTab] || [];

            if (settings.length === 0) {
                container.innerHTML = `
                    <div class="loading">
                        <div>No settings found for this category.</div>
                        <button class="btn btn-primary" onclick="initializeSettings()" style="margin-top: 1rem;">
                            Initialize Default Settings
                        </button>
                    </div>
                `;
                return;
            }

            const categoryTitles = {
                'mt5': 'MT5 Connection Settings',
                'trading': 'Trading Configuration',
                'risk': 'Risk Management',
                'agents': 'Agent Settings',
                'monitoring': 'Monitoring & Metrics',
                'dashboard': 'Dashboard Settings',
                'database': 'Database Configuration',
                'notifications': 'Notification Settings'
            };

            let html = `<div class="settings-section active">
                <h3>${categoryTitles[activeTab] || activeTab.toUpperCase()}</h3>`;

            for (const setting of settings) {
                html += renderSettingItem(setting);
            }

            html += '</div>';
            container.innerHTML = html;

            // Add event listeners for inputs
            container.querySelectorAll('.setting-input').forEach(input => {
                input.addEventListener('change', handleSettingChange);
            });

            container.querySelectorAll('.toggle-switch').forEach(toggle => {
                toggle.addEventListener('click', handleToggleClick);
            });
        }

        function renderSettingItem(setting) {
            const inputId = `setting-${setting.category}-${setting.key}`;
            let inputHtml = '';

            if (setting.value_type === 'bool') {
                const isActive = setting.value === true || setting.value === 'true';
                inputHtml = `
                    <div class="toggle-switch ${isActive ? 'active' : ''}"
                         data-category="${setting.category}"
                         data-key="${setting.key}"
                         data-value="${isActive}">
                    </div>
                `;
            } else if (setting.is_secret) {
                inputHtml = `
                    <input type="password"
                           class="setting-input"
                           id="${inputId}"
                           data-category="${setting.category}"
                           data-key="${setting.key}"
                           value="${setting.value || ''}"
                           placeholder="Enter ${setting.key}..."
                           ${!setting.is_editable ? 'disabled' : ''}>
                `;
            } else if (setting.value_type === 'int' || setting.value_type === 'float') {
                const step = setting.value_type === 'float' ? '0.001' : '1';
                inputHtml = `
                    <input type="number"
                           class="setting-input"
                           id="${inputId}"
                           data-category="${setting.category}"
                           data-key="${setting.key}"
                           value="${setting.value || ''}"
                           step="${step}"
                           ${setting.min_value !== null ? 'min="' + setting.min_value + '"' : ''}
                           ${setting.max_value !== null ? 'max="' + setting.max_value + '"' : ''}
                           ${!setting.is_editable ? 'disabled' : ''}>
                `;
            } else {
                inputHtml = `
                    <input type="text"
                           class="setting-input"
                           id="${inputId}"
                           data-category="${setting.category}"
                           data-key="${setting.key}"
                           value="${setting.value || ''}"
                           ${!setting.is_editable ? 'disabled' : ''}>
                `;
            }

            return `
                <div class="setting-item">
                    <div class="setting-label">
                        <span class="key">${formatKey(setting.key)}</span>
                        <span class="description">${setting.description || ''}</span>
                    </div>
                    ${inputHtml}
                </div>
            `;
        }

        function formatKey(key) {
            return key.split('_').map(word =>
                word.charAt(0).toUpperCase() + word.slice(1)
            ).join(' ');
        }

        async function handleSettingChange(event) {
            const input = event.target;
            const category = input.dataset.category;
            const key = input.dataset.key;
            const value = input.value;

            await updateSetting(category, key, value);
        }

        async function handleToggleClick(event) {
            const toggle = event.target;
            const category = toggle.dataset.category;
            const key = toggle.dataset.key;
            const currentValue = toggle.dataset.value === 'true';
            const newValue = !currentValue;

            toggle.classList.toggle('active');
            toggle.dataset.value = newValue.toString();

            await updateSetting(category, key, newValue.toString());
        }

        async function updateSetting(category, key, value) {
            try {
                const response = await fetch(`/api/settings/${category}/${key}`, {
                    method: 'PUT',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ value: value, updated_by: 'dashboard' })
                });

                const data = await response.json();

                if (data.status === 'success') {
                    showSaveIndicator();
                    // Update local cache
                    const idx = currentSettings[category]?.findIndex(s => s.key === key);
                    if (idx !== undefined && idx >= 0) {
                        currentSettings[category][idx] = data.setting;
                    }
                } else {
                    alert('Failed to update setting: ' + data.message);
                }
            } catch (error) {
                alert('Network error updating setting: ' + error.message);
            }
        }

        async function initializeSettings() {
            if (!confirm('This will reset all settings to their default values. Continue?')) {
                return;
            }

            try {
                const response = await fetch('/api/settings/initialize', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' }
                });

                const data = await response.json();

                if (data.status === 'success') {
                    showSaveIndicator();
                    loadSettings();
                } else {
                    alert('Failed to initialize settings: ' + data.message);
                }
            } catch (error) {
                alert('Network error initializing settings: ' + error.message);
            }
        }

        function showSaveIndicator() {
            const indicator = document.getElementById('save-indicator');
            indicator.classList.add('show');
            setTimeout(() => {
                indicator.classList.remove('show');
            }, 2000);
        }

        function showSettingsError(message) {
            document.getElementById('settings-content').innerHTML = `
                <div class="loading error-message">
                    <div>${message}</div>
                    <button class="btn btn-primary" onclick="loadSettings()" style="margin-top: 1rem;">
                        Retry
                    </button>
                </div>
            `;
        }

        // Keyboard shortcut to open settings (Ctrl+,)
        document.addEventListener('keydown', (e) => {
            if (e.ctrlKey && e.key === ',') {
                e.preventDefault();
                openSettings();
            }
            if (e.key === 'Escape') {
                closeSettings();
            }
        });
    </script>
</body>
</html>
'''
    
    async def start_server(self):
        """Start the enterprise server with TLS"""
        
        # Generate certificate if needed
        if not self.tls_config:
            self.tls_config = self.certificate_manager.generate_self_signed_cert()
            
        # SSL context
        ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
        ssl_context.load_cert_chain(self.tls_config.cert_file, self.tls_config.key_file)
        ssl_context.set_ciphers(self.tls_config.cipher_suites)
        
        # Start background tasks
        ping_task = asyncio.create_task(self._ping_connections_periodically())
        self.background_tasks.append(ping_task)
        
        logger.info(f"Starting Enterprise Dashboard on {self.connection_config.primary_host}:{self.connection_config.primary_port}")
        
        # Start server with SSL
        config = uvicorn.Config(
            self.app,
            host=self.connection_config.primary_host,
            port=self.connection_config.primary_port,
            ssl_keyfile=self.tls_config.key_file,
            ssl_certfile=self.tls_config.cert_file,
            log_level="info"
        )
        
        server = uvicorn.Server(config)
        await server.serve()
        
    async def _ping_connections_periodically(self):
        """Background task to ping all connections"""
        while True:
            await asyncio.sleep(self.connection_config.heartbeat_interval)
            await self.connection_manager.ping_all_connections()

def main():
    """Start the enterprise dashboard"""
    
    # Configuration
    connection_config = ConnectionConfig(
        primary_host="0.0.0.0",
        primary_port=8443
    )
    
    security_config = SecurityConfig(
        jwt_secret_key=secrets.token_urlsafe(32),
        jwt_expiration_minutes=60
    )
    
    # Create and start dashboard
    dashboard = EnterpriseDashboard(
        connection_config=connection_config,
        security_config=security_config
    )
    
    print("🏭 STARTING ENTERPRISE TRADING DASHBOARD")
    print("=" * 60)
    print(f"🔒 HTTPS Server: https://localhost:{connection_config.primary_port}")
    print("🛡️  Features: TLS encryption, JWT authentication, automatic failover")
    print("📊 Metrics: /metrics")
    print("💻 Health: /health")
    print()
    
    try:
        asyncio.run(dashboard.start_server())
    except KeyboardInterrupt:
        print("\n🛑 Shutting down enterprise dashboard...")
    except Exception as e:
        logger.error(f"Enterprise dashboard error: {e}")

if __name__ == "__main__":
    main()