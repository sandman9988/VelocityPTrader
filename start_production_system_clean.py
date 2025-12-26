#!/usr/bin/env python3
"""
Production AI Trading System - Clean Startup with Port Management
"""

import asyncio
import sys
import os
import signal
import secrets
from pathlib import Path
from datetime import datetime

# Add paths
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent / "core"))
sys.path.append(str(Path(__file__).parent / "dashboard"))

import structlog
from prometheus_client import start_http_server

# Import components
from core.symbol_manager import ProductionSymbolManager, InstrumentType
from core.mt5_data_pipeline import ProductionMT5Pipeline, MarketTick, DataSourceType
from dashboard.enterprise_dashboard import EnterpriseDashboard, ConnectionConfig, SecurityConfig
from port_manager import PortManager

# Configure logging
logging = structlog.get_logger(__name__)

class CleanProductionSystem:
    """Production system with proper port management"""
    
    def __init__(self):
        self.port_manager = PortManager()
        self.symbol_manager = None
        self.data_pipeline = None
        self.dashboard = None
        self.running = False
        self.start_time = datetime.now()
        
        # Use standard ports
        self.dashboard_port = 8443
        self.metrics_port = 9090
        
    async def initialize(self):
        """Initialize with port cleanup"""
        
        print("🏭 INITIALIZING CLEAN PRODUCTION SYSTEM")
        print("=" * 60)
        
        # Clean up ports first
        print("🧹 Ensuring ports are available...")
        self.port_manager.kill_port(self.dashboard_port, force=True)
        self.port_manager.kill_port(self.metrics_port, force=True)
        
        # Initialize symbol manager
        print("📊 Initializing Symbol Manager...")
        self.symbol_manager = ProductionSymbolManager()
        status = self.symbol_manager.get_system_status()
        print(f"   ✅ {status['total_instruments']} instruments loaded")
        
        # Get optimal symbols
        forex_instruments = self.symbol_manager.get_instruments_by_type(InstrumentType.FOREX)[:15]
        crypto_instruments = self.symbol_manager.get_instruments_by_type(InstrumentType.CRYPTO)[:3]
        index_instruments = self.symbol_manager.get_instruments_by_type(InstrumentType.INDEX)[:2]
        
        # Prefer ECN versions
        symbols = []
        for inst in forex_instruments + crypto_instruments + index_instruments:
            if inst.is_ecn:
                symbols.append(inst.symbol)
            elif self.symbol_manager.get_ecn_equivalent(inst.symbol):
                symbols.append(self.symbol_manager.get_ecn_equivalent(inst.symbol))
            else:
                symbols.append(inst.symbol)
        
        symbols = symbols[:20]  # Limit for performance
        
        # Initialize data pipeline
        print("📡 Initializing MT5 Data Pipeline...")
        self.data_pipeline = ProductionMT5Pipeline(symbols)
        self.data_pipeline.on_data_update = self._handle_data_update
        await self.data_pipeline.initialize()
        print(f"   ✅ Pipeline ready with {len(symbols)} symbols")
        
        # Start metrics server
        print(f"📈 Starting metrics server on port {self.metrics_port}...")
        start_http_server(self.metrics_port)
        self.port_manager.save_pid('metrics_server', os.getpid())
        
        # Initialize dashboard
        print(f"🌐 Initializing dashboard on port {self.dashboard_port}...")
        
        connection_config = ConnectionConfig(
            primary_host="0.0.0.0",
            primary_port=self.dashboard_port,
            max_connections=100
        )
        
        security_config = SecurityConfig(
            jwt_secret_key=secrets.token_urlsafe(32),
            jwt_expiration_minutes=120
        )
        
        self.dashboard = EnterpriseDashboard(
            connection_config=connection_config,
            security_config=security_config
        )
        
        print("✅ SYSTEM INITIALIZED")
        print("=" * 60)
        print(f"🌐 Dashboard: https://localhost:{self.dashboard_port}")
        print(f"📊 Metrics:   http://localhost:{self.metrics_port}")
        
        return True
    
    def _handle_data_update(self, ticks):
        """Handle data updates"""
        # Update symbol manager
        if self.symbol_manager:
            for symbol, tick in ticks.items():
                tick_data = {
                    'bid': tick.bid,
                    'ask': tick.ask,
                    'last': tick.last,
                    'volume': tick.volume,
                    'timestamp': tick.timestamp.timestamp()
                }
                self.symbol_manager.update_market_data(symbol, tick_data)
    
    async def start(self):
        """Start the system"""
        
        if not all([self.symbol_manager, self.data_pipeline, self.dashboard]):
            await self.initialize()
        
        self.running = True
        
        # Save PID for management
        self.port_manager.save_pid('production_system', os.getpid())
        
        print("🚀 STARTING PRODUCTION SYSTEM")
        
        try:
            # Start components
            data_task = asyncio.create_task(self.data_pipeline.start_continuous_updates())
            dashboard_task = asyncio.create_task(self.dashboard.start_server())
            monitoring_task = asyncio.create_task(self._monitoring_loop())
            
            print("✅ System running - all components active")
            
            # Wait for tasks
            await asyncio.gather(data_task, dashboard_task, monitoring_task)
            
        except Exception as e:
            print(f"❌ System error: {e}")
            await self.cleanup()
            raise
    
    async def _monitoring_loop(self):
        """System monitoring"""
        while self.running:
            try:
                # Simple health check every 60 seconds
                await asyncio.sleep(60)
                
                uptime = datetime.now() - self.start_time
                print(f"📊 System health: Uptime {int(uptime.total_seconds())}s")
                
            except Exception as e:
                print(f"⚠️ Monitoring error: {e}")
                await asyncio.sleep(10)
    
    async def cleanup(self):
        """Clean shutdown"""
        print("🛑 Shutting down system...")
        
        self.running = False
        
        if self.data_pipeline:
            await self.data_pipeline.stop()
        
        print("✅ Shutdown complete")
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        print(f"\n🛑 Received signal {signum}, shutting down...")
        
        # Create new event loop if none exists
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        # Schedule cleanup
        task = loop.create_task(self.cleanup())
        loop.run_until_complete(task)
        
        sys.exit(0)

async def main():
    """Main entry point"""
    
    # Create system
    system = CleanProductionSystem()
    
    # Setup signal handlers
    signal.signal(signal.SIGINT, system.signal_handler)
    signal.signal(signal.SIGTERM, system.signal_handler)
    
    try:
        await system.start()
    except KeyboardInterrupt:
        print("\n🛑 Keyboard interrupt")
        await system.cleanup()
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        await system.cleanup()
        return 1
    
    return 0

if __name__ == "__main__":
    # Ensure directories exist
    Path("logs").mkdir(exist_ok=True)
    Path("pids").mkdir(exist_ok=True)
    Path("certs").mkdir(exist_ok=True)
    
    exit_code = asyncio.run(main())
    sys.exit(exit_code)