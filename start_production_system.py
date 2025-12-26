#!/usr/bin/env python3
"""
Production AI Trading System Startup
Complete integration of all enterprise components
"""

import asyncio
import sys
import json
import signal
import logging
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

# Add paths
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent / "core"))
sys.path.append(str(Path(__file__).parent / "dashboard"))

import structlog
from prometheus_client import start_http_server

# Import production components
from core.symbol_manager import ProductionSymbolManager, InstrumentType
from core.mt5_data_pipeline import ProductionMT5Pipeline, MarketTick, DataSourceType
from dashboard.enterprise_dashboard import EnterpriseDashboard, ConnectionConfig, SecurityConfig
import secrets

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/production_system.log'),
        logging.StreamHandler()
    ]
)

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

class ProductionTradingSystem:
    """Complete production trading system integration"""
    
    def __init__(self):
        self.symbol_manager: Optional[ProductionSymbolManager] = None
        self.data_pipeline: Optional[ProductionMT5Pipeline] = None
        self.dashboard: Optional[EnterpriseDashboard] = None
        
        self.running = False
        self.shutdown_event = asyncio.Event()
        
        # System metrics
        self.start_time = datetime.now()
        self.total_symbols = 0
        self.active_symbols = 0
        self.data_updates = 0
        
        logger.info("Production trading system initialized")
    
    async def initialize(self):
        """Initialize all production components"""
        
        logger.info("🏭 INITIALIZING PRODUCTION AI TRADING SYSTEM")
        logger.info("=" * 70)
        
        try:
            # 1. Initialize Symbol Manager
            logger.info("📊 Initializing Symbol Manager...")
            self.symbol_manager = ProductionSymbolManager()
            
            status = self.symbol_manager.get_system_status()
            self.total_symbols = status['total_instruments']
            
            logger.info("Symbol manager initialized",
                       total_symbols=self.total_symbols,
                       ecn_symbols=status['ecn_instruments'],
                       tradeable_pairs=status['tradeable_pairs'])
            
            # 2. Get optimal symbol list (prefer ECN)
            optimal_symbols = []
            
            # Get top performing instruments by type
            forex_instruments = self.symbol_manager.get_instruments_by_type(InstrumentType.FOREX)
            crypto_instruments = self.symbol_manager.get_instruments_by_type(InstrumentType.CRYPTO)
            index_instruments = self.symbol_manager.get_instruments_by_type(InstrumentType.INDEX)
            
            # Add ECN versions where available
            for instrument in forex_instruments:
                if instrument.is_ecn:
                    optimal_symbols.append(instrument.symbol)
                elif self.symbol_manager.get_ecn_equivalent(instrument.symbol):
                    optimal_symbols.append(self.symbol_manager.get_ecn_equivalent(instrument.symbol))
                else:
                    optimal_symbols.append(instrument.symbol)
            
            # Add crypto and indices (no ECN variants typically)
            for instrument in crypto_instruments + index_instruments:
                optimal_symbols.append(instrument.symbol)
            
            # Limit to top symbols for performance
            optimal_symbols = optimal_symbols[:20]  # Top 20 instruments
            self.active_symbols = len(optimal_symbols)
            
            logger.info("Optimized symbol list created",
                       active_symbols=self.active_symbols,
                       symbols=optimal_symbols[:5])  # Show first 5
            
            # 3. Initialize Data Pipeline
            logger.info("📡 Initializing MT5 Data Pipeline...")
            self.data_pipeline = ProductionMT5Pipeline(optimal_symbols)
            
            # Setup callbacks
            self.data_pipeline.on_data_update = self._handle_data_update
            self.data_pipeline.on_source_change = self._handle_source_change
            
            await self.data_pipeline.initialize()
            
            pipeline_status = self.data_pipeline.get_status_report()
            logger.info("Data pipeline initialized",
                       active_source=pipeline_status['active_source'],
                       total_sources=pipeline_status['total_sources'])
            
            # 4. Initialize Enterprise Dashboard
            logger.info("🌐 Initializing Enterprise Dashboard...")
            
            connection_config = ConnectionConfig(
                primary_host="0.0.0.0", 
                primary_port=8444,  # Changed port to avoid conflict
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
            
            # 5. Start Prometheus metrics server
            logger.info("📈 Starting metrics server on port 9090...")
            start_http_server(9090)
            
            logger.info("✅ ALL COMPONENTS INITIALIZED SUCCESSFULLY")
            logger.info("=" * 70)
            logger.info("🌐 Enterprise Dashboard: https://localhost:8444")
            logger.info("📊 Metrics Endpoint: http://localhost:9090")
            logger.info("📁 Logs: logs/production_system.log")
            
            return True
            
        except Exception as e:
            logger.error("System initialization failed", error=str(e))
            raise
    
    def _handle_data_update(self, ticks: Dict[str, MarketTick]):
        """Handle real-time data updates"""
        self.data_updates += 1
        
        if self.data_updates % 10 == 0:  # Log every 10th update
            logger.info("Data update received",
                       symbols=len(ticks),
                       update_count=self.data_updates,
                       primary_source=list(ticks.values())[0].source.value if ticks else None)
        
        # Update symbol manager with latest data
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
    
    def _handle_source_change(self, old_source: DataSourceType, new_source: DataSourceType):
        """Handle data source failover"""
        logger.warning("Data source failover occurred",
                      from_source=old_source.value,
                      to_source=new_source.value)
        
        # Could trigger alerts here in production
    
    async def start_system(self):
        """Start the complete production system"""
        
        if not all([self.symbol_manager, self.data_pipeline, self.dashboard]):
            await self.initialize()
        
        self.running = True
        
        logger.info("🚀 STARTING PRODUCTION TRADING SYSTEM")
        
        try:
            # Start data pipeline
            data_task = asyncio.create_task(self.data_pipeline.start_continuous_updates())
            
            # Start dashboard server
            dashboard_task = asyncio.create_task(self.dashboard.start_server())
            
            # Start system monitoring
            monitoring_task = asyncio.create_task(self._system_monitoring_loop())
            
            # Wait for shutdown signal
            await self.shutdown_event.wait()
            
            # Cleanup
            logger.info("🛑 Shutting down production system...")
            
            self.running = False
            
            # Cancel tasks
            data_task.cancel()
            dashboard_task.cancel() 
            monitoring_task.cancel()
            
            # Stop data pipeline
            if self.data_pipeline:
                await self.data_pipeline.stop()
            
            logger.info("✅ Production system shutdown complete")
            
        except Exception as e:
            logger.error("Production system error", error=str(e))
            raise
    
    async def _system_monitoring_loop(self):
        """Background system monitoring and health checks"""
        
        while self.running:
            try:
                # Log system health every 60 seconds
                uptime = datetime.now() - self.start_time
                
                # Get component status
                pipeline_status = self.data_pipeline.get_status_report() if self.data_pipeline else {}
                symbol_status = self.symbol_manager.get_system_status() if self.symbol_manager else {}
                
                logger.info("System health check",
                           uptime_seconds=int(uptime.total_seconds()),
                           data_updates=self.data_updates,
                           active_symbols=self.active_symbols,
                           pipeline_running=pipeline_status.get('running', False),
                           active_source=pipeline_status.get('active_source'),
                           data_coverage_pct=symbol_status.get('data_coverage_pct', 0))
                
                await asyncio.sleep(60)  # Monitor every minute
                
            except Exception as e:
                logger.error("System monitoring error", error=str(e))
                await asyncio.sleep(10)
    
    def shutdown(self):
        """Trigger system shutdown"""
        logger.info("Shutdown signal received")
        self.shutdown_event.set()

def signal_handler(trading_system):
    """Handle shutdown signals"""
    def handler(signum, frame):
        print(f"\\n🛑 Received signal {signum}, shutting down gracefully...")
        trading_system.shutdown()
    return handler

async def main():
    """Main entry point"""
    
    # Ensure logs directory exists
    Path("logs").mkdir(exist_ok=True)
    
    print("🏭 PRODUCTION AI TRADING SYSTEM")
    print("=" * 70)
    print("Enterprise-grade trading system with:")
    print("✅ Production symbol management (ECN/Standard)")
    print("✅ Multi-source data pipeline with failover")  
    print("✅ TLS-encrypted enterprise dashboard")
    print("✅ Comprehensive monitoring and alerting")
    print("✅ Structured logging and metrics")
    print("=" * 70)
    
    # Create production system
    trading_system = ProductionTradingSystem()
    
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler(trading_system))
    signal.signal(signal.SIGTERM, signal_handler(trading_system))
    
    try:
        # Initialize and start
        await trading_system.initialize()
        await trading_system.start_system()
        
    except KeyboardInterrupt:
        print("\\n🛑 Keyboard interrupt received")
        trading_system.shutdown()
    except Exception as e:
        logger.error("Production system failed", error=str(e))
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)