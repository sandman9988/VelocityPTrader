#!/usr/bin/env python3
"""
Production MT5 Data Pipeline
High-reliability market data pipeline with multiple failover mechanisms
"""

import asyncio
import json
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any, NamedTuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from collections import deque, defaultdict
import structlog
from prometheus_client import Counter, Histogram, Gauge, Summary
import subprocess
import hashlib
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
import redis

# Setup structured logging
logger = structlog.get_logger(__name__)

# Metrics
DATA_SOURCE_ATTEMPTS = Counter('mt5_data_source_attempts_total', 'Data source connection attempts', ['source', 'result'])
DATA_LATENCY = Histogram('mt5_data_latency_seconds', 'Data retrieval latency', ['source'])
DATA_QUALITY = Gauge('mt5_data_quality_score', 'Data quality score 0-1', ['symbol'])
FAILOVER_EVENTS = Counter('mt5_failover_events_total', 'Failover events', ['from_source', 'to_source'])
PIPELINE_HEALTH = Gauge('mt5_pipeline_health_score', 'Overall pipeline health 0-1')

class DataSourceType(Enum):
    """Available data sources in priority order"""
    DIRECT_WINDOWS_MT5 = "direct_windows_mt5"  # Highest priority
    SECURE_FILE_BRIDGE = "secure_file_bridge"  # Medium priority  
    HTTP_API = "http_api"  # Lowest priority
    CACHED_DATA = "cached_data"  # Emergency fallback

class DataSourceStatus(Enum):
    """Data source status"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILED = "failed"
    UNKNOWN = "unknown"

@dataclass
class DataSourceConfig:
    """Configuration for a data source"""
    source_type: DataSourceType
    priority: int
    timeout_seconds: float
    retry_attempts: int
    health_check_interval: float
    enabled: bool = True
    config: Dict[str, Any] = field(default_factory=dict)

@dataclass
class MarketTick:
    """Standardized market tick data"""
    symbol: str
    bid: float
    ask: float
    last: float
    volume: int
    timestamp: datetime
    source: DataSourceType
    quality_score: float
    latency_ms: float
    
    def __post_init__(self):
        """Validate tick data"""
        if self.bid <= 0 or self.ask <= 0:
            raise ValueError(f"Invalid prices: bid={self.bid}, ask={self.ask}")
        if self.bid >= self.ask:
            raise ValueError(f"Bid >= Ask: {self.bid} >= {self.ask}")
        if self.quality_score < 0 or self.quality_score > 1:
            raise ValueError(f"Quality score must be 0-1: {self.quality_score}")

@dataclass 
class DataSourceHealth:
    """Health metrics for a data source"""
    source_type: DataSourceType
    status: DataSourceStatus
    last_successful_update: datetime
    success_rate: float
    avg_latency_ms: float
    error_count: int
    last_error: Optional[str]
    consecutive_failures: int

class DataSourceInterface:
    """Base interface for all data sources"""
    
    def __init__(self, config: DataSourceConfig):
        self.config = config
        self.health = DataSourceHealth(
            source_type=config.source_type,
            status=DataSourceStatus.UNKNOWN,
            last_successful_update=datetime.min,
            success_rate=0.0,
            avg_latency_ms=0.0,
            error_count=0,
            last_error=None,
            consecutive_failures=0
        )
        self._metrics_window = deque(maxlen=100)  # Last 100 operations
    
    async def initialize(self) -> bool:
        """Initialize the data source"""
        raise NotImplementedError
    
    async def get_market_data(self, symbols: List[str]) -> Dict[str, MarketTick]:
        """Get current market data for symbols"""
        raise NotImplementedError
    
    async def health_check(self) -> bool:
        """Check if data source is healthy"""
        raise NotImplementedError
    
    async def cleanup(self):
        """Cleanup resources"""
        pass
    
    def _record_operation(self, success: bool, latency_ms: float, error: Optional[str] = None):
        """Record operation metrics"""
        self._metrics_window.append({
            'success': success,
            'latency_ms': latency_ms,
            'timestamp': datetime.now(),
            'error': error
        })
        
        # Update health metrics
        if success:
            self.health.last_successful_update = datetime.now()
            self.health.consecutive_failures = 0
        else:
            self.health.error_count += 1
            self.health.last_error = error
            self.health.consecutive_failures += 1
        
        # Calculate success rate and avg latency
        if self._metrics_window:
            successes = sum(1 for m in self._metrics_window if m['success'])
            self.health.success_rate = successes / len(self._metrics_window)
            
            latencies = [m['latency_ms'] for m in self._metrics_window if m['success']]
            self.health.avg_latency_ms = sum(latencies) / len(latencies) if latencies else 0
        
        # Update status
        if self.health.consecutive_failures >= 3:
            self.health.status = DataSourceStatus.FAILED
        elif self.health.success_rate < 0.8:
            self.health.status = DataSourceStatus.DEGRADED
        else:
            self.health.status = DataSourceStatus.HEALTHY

class DirectWindowsMT5Source(DataSourceInterface):
    """Direct connection to Windows MT5 via MetaTrader5 package"""
    
    async def initialize(self) -> bool:
        """Initialize direct MT5 connection"""
        start_time = time.time()
        
        try:
            # Run the Windows MT5 fetcher
            result = await asyncio.get_event_loop().run_in_executor(
                None, self._fetch_mt5_data
            )
            
            latency_ms = (time.time() - start_time) * 1000
            
            if result and result.get('success'):
                self._record_operation(True, latency_ms)
                DATA_SOURCE_ATTEMPTS.labels(source='direct_mt5', result='success').inc()
                logger.info("Direct MT5 source initialized", 
                          symbols=result.get('processed_symbols', 0),
                          server=result.get('account', {}).get('server'))
                return True
            else:
                error = result.get('error', 'Unknown error') if result else 'No response'
                self._record_operation(False, latency_ms, error)
                DATA_SOURCE_ATTEMPTS.labels(source='direct_mt5', result='failure').inc()
                logger.error("Direct MT5 initialization failed", error=error)
                return False
                
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            self._record_operation(False, latency_ms, str(e))
            DATA_SOURCE_ATTEMPTS.labels(source='direct_mt5', result='error').inc()
            logger.error("Direct MT5 initialization error", error=str(e))
            return False
    
    def _fetch_mt5_data(self) -> Optional[Dict]:
        """Fetch data using Windows Python process"""
        try:
            result = subprocess.run([
                'python.exe', 
                'C:\\\\temp\\\\fetch_mt5_symbols.py'
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                return json.loads(result.stdout)
            else:
                return {'error': f'Process failed: {result.stderr}'}
                
        except subprocess.TimeoutExpired:
            return {'error': 'Timeout waiting for MT5 data'}
        except Exception as e:
            return {'error': str(e)}
    
    async def get_market_data(self, symbols: List[str]) -> Dict[str, MarketTick]:
        """Get current market data via direct MT5"""
        start_time = time.time()
        
        try:
            data = await asyncio.get_event_loop().run_in_executor(
                None, self._fetch_mt5_data
            )
            
            latency_ms = (time.time() - start_time) * 1000
            
            if not data or not data.get('success'):
                error = data.get('error', 'No data') if data else 'No response'
                self._record_operation(False, latency_ms, error)
                return {}
            
            # Convert to MarketTick objects
            ticks = {}
            for symbol, symbol_data in data.get('symbols', {}).items():
                if symbol in symbols:
                    try:
                        tick = MarketTick(
                            symbol=symbol,
                            bid=symbol_data['bid'],
                            ask=symbol_data['ask'],
                            last=symbol_data.get('last_price', (symbol_data['bid'] + symbol_data['ask']) / 2),
                            volume=0,  # Not provided by this source
                            timestamp=datetime.now(),
                            source=DataSourceType.DIRECT_WINDOWS_MT5,
                            quality_score=1.0,  # Direct source = highest quality
                            latency_ms=latency_ms
                        )
                        ticks[symbol] = tick
                        DATA_QUALITY.labels(symbol=symbol).set(1.0)
                        
                    except Exception as e:
                        logger.warning("Failed to create tick for symbol", symbol=symbol, error=str(e))
            
            self._record_operation(True, latency_ms)
            DATA_LATENCY.labels(source='direct_mt5').observe(latency_ms / 1000)
            
            return ticks
            
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            self._record_operation(False, latency_ms, str(e))
            logger.error("Direct MT5 data fetch failed", error=str(e))
            return {}
    
    async def health_check(self) -> bool:
        """Check MT5 connection health"""
        try:
            # Quick health check with minimal data
            test_data = await self.get_market_data(['EURUSD'])
            return len(test_data) > 0
        except Exception:
            return False

class SecureFileBridgeSource(DataSourceInterface):
    """Secure file-based bridge to MT5 terminal"""
    
    def __init__(self, config: DataSourceConfig):
        super().__init__(config)
        self.bridge_path = Path("/mnt/c/DevCenter/MT5-Unified/MT5-Core/Terminal/MQL5/Files/SecureBridge")
        self.validation_key = "AI_TRADING_VANTAGE_2024"
    
    async def initialize(self) -> bool:
        """Initialize secure bridge"""
        if not self.bridge_path.exists():
            logger.error("Bridge directory not found", path=str(self.bridge_path))
            return False
        
        logger.info("Secure file bridge initialized", path=str(self.bridge_path))
        return True
    
    def _validate_data_integrity(self, data: str, checksum: str) -> bool:
        """Validate data integrity with checksum"""
        combined = data + self.validation_key
        expected_checksum = str(sum(ord(c) for c in combined) % 65536)
        return expected_checksum == checksum
    
    async def get_market_data(self, symbols: List[str]) -> Dict[str, MarketTick]:
        """Get market data from secure bridge files"""
        start_time = time.time()
        
        try:
            data_file = self.bridge_path / "secure_data.txt"
            checksum_file = self.bridge_path / "checksum.txt"
            heartbeat_file = self.bridge_path / "heartbeat.txt"
            
            # Check file freshness
            if not heartbeat_file.exists():
                self._record_operation(False, 0, "No heartbeat file")
                return {}
            
            heartbeat_time = int(heartbeat_file.read_text().strip())
            age = int(time.time()) - heartbeat_time
            
            if age > 10:  # Data too old
                self._record_operation(False, 0, f"Stale data: {age}s old")
                return {}
            
            # Read and validate data
            if not data_file.exists() or not checksum_file.exists():
                self._record_operation(False, 0, "Missing data files")
                return {}
            
            data_content = data_file.read_text()
            provided_checksum = checksum_file.read_text().strip()
            
            if not self._validate_data_integrity(data_content, provided_checksum):
                self._record_operation(False, 0, "Data integrity check failed")
                return {}
            
            # Parse data
            ticks = {}
            lines = data_content.strip().split('\\n')
            
            for line in lines[1:]:  # Skip metadata line
                if not line.strip():
                    continue
                    
                parts = line.split('|')
                if len(parts) >= 7:
                    symbol = parts[0]
                    if symbol in symbols:
                        try:
                            tick = MarketTick(
                                symbol=symbol,
                                bid=float(parts[1]),
                                ask=float(parts[2]),
                                last=(float(parts[1]) + float(parts[2])) / 2,
                                volume=0,
                                timestamp=datetime.fromtimestamp(int(parts[6])),
                                source=DataSourceType.SECURE_FILE_BRIDGE,
                                quality_score=0.9,  # High quality, validated data
                                latency_ms=(time.time() - start_time) * 1000
                            )
                            ticks[symbol] = tick
                            DATA_QUALITY.labels(symbol=symbol).set(0.9)
                            
                        except Exception as e:
                            logger.warning("Failed to parse tick data", symbol=symbol, error=str(e))
            
            latency_ms = (time.time() - start_time) * 1000
            self._record_operation(True, latency_ms)
            DATA_LATENCY.labels(source='secure_bridge').observe(latency_ms / 1000)
            
            return ticks
            
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            self._record_operation(False, latency_ms, str(e))
            logger.error("Secure bridge data fetch failed", error=str(e))
            return {}
    
    async def health_check(self) -> bool:
        """Check bridge health"""
        try:
            heartbeat_file = self.bridge_path / "heartbeat.txt"
            if not heartbeat_file.exists():
                return False
                
            heartbeat_time = int(heartbeat_file.read_text().strip())
            age = int(time.time()) - heartbeat_time
            return age <= 10  # Fresh data
            
        except Exception:
            return False

class CachedDataSource(DataSourceInterface):
    """Emergency fallback using cached/historical data"""
    
    def __init__(self, config: DataSourceConfig):
        super().__init__(config)
        self.redis_client = redis.Redis(host='localhost', port=6379, db=1, decode_responses=True)
        self.cache_ttl = 300  # 5 minutes
    
    async def initialize(self) -> bool:
        """Initialize cache connection"""
        try:
            self.redis_client.ping()
            logger.info("Cache data source initialized")
            return True
        except Exception as e:
            logger.error("Cache initialization failed", error=str(e))
            return False
    
    async def get_market_data(self, symbols: List[str]) -> Dict[str, MarketTick]:
        """Get cached market data (emergency fallback)"""
        start_time = time.time()
        
        try:
            ticks = {}
            for symbol in symbols:
                cache_key = f"market_tick:{symbol}"
                cached_data = self.redis_client.get(cache_key)
                
                if cached_data:
                    tick_data = json.loads(cached_data)
                    
                    # Check data age
                    cached_time = datetime.fromisoformat(tick_data['timestamp'])
                    age = (datetime.now() - cached_time).total_seconds()
                    
                    if age <= self.cache_ttl:
                        tick = MarketTick(
                            symbol=symbol,
                            bid=tick_data['bid'],
                            ask=tick_data['ask'],
                            last=tick_data['last'],
                            volume=tick_data['volume'],
                            timestamp=cached_time,
                            source=DataSourceType.CACHED_DATA,
                            quality_score=max(0.1, 0.8 - (age / self.cache_ttl) * 0.7),  # Decreasing quality over time
                            latency_ms=(time.time() - start_time) * 1000
                        )
                        ticks[symbol] = tick
                        DATA_QUALITY.labels(symbol=symbol).set(tick.quality_score)
            
            latency_ms = (time.time() - start_time) * 1000
            self._record_operation(True, latency_ms)
            
            if ticks:
                logger.warning("Using cached data", symbols=list(ticks.keys()), count=len(ticks))
            
            return ticks
            
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            self._record_operation(False, latency_ms, str(e))
            logger.error("Cache data fetch failed", error=str(e))
            return {}
    
    async def health_check(self) -> bool:
        """Check cache health"""
        try:
            self.redis_client.ping()
            return True
        except Exception:
            return False
    
    def cache_tick_data(self, tick: MarketTick):
        """Cache tick data for emergency fallback"""
        try:
            cache_key = f"market_tick:{tick.symbol}"
            tick_data = {
                'symbol': tick.symbol,
                'bid': tick.bid,
                'ask': tick.ask,
                'last': tick.last,
                'volume': tick.volume,
                'timestamp': tick.timestamp.isoformat(),
                'source': tick.source.value,
                'quality_score': tick.quality_score
            }
            
            self.redis_client.setex(
                cache_key,
                self.cache_ttl,
                json.dumps(tick_data)
            )
            
        except Exception as e:
            logger.warning("Failed to cache tick data", symbol=tick.symbol, error=str(e))

class ProductionMT5Pipeline:
    """Production-grade MT5 data pipeline with automatic failover"""
    
    def __init__(self, symbols: List[str]):
        self.symbols = symbols
        self.data_sources: List[DataSourceInterface] = []
        self.active_source: Optional[DataSourceInterface] = None
        self.cache_source: Optional[CachedDataSource] = None
        
        # Threading
        self._lock = threading.RLock()
        self._running = False
        self._health_check_task = None
        
        # Callbacks
        self.on_data_update: Optional[Callable[[Dict[str, MarketTick]], None]] = None
        self.on_source_change: Optional[Callable[[DataSourceType, DataSourceType], None]] = None
        self.on_health_change: Optional[Callable[[Dict[str, DataSourceHealth]], None]] = None
        
        # Performance tracking
        self.update_interval = 1.0  # seconds
        self.last_update = datetime.min
        self.update_count = 0
        
        logger.info("Production MT5 pipeline initialized", symbols=len(symbols))
    
    async def initialize(self):
        """Initialize all data sources"""
        logger.info("Initializing MT5 data pipeline")
        
        # Initialize data sources in priority order
        configs = [
            DataSourceConfig(
                source_type=DataSourceType.DIRECT_WINDOWS_MT5,
                priority=1,
                timeout_seconds=30.0,
                retry_attempts=3,
                health_check_interval=60.0
            ),
            DataSourceConfig(
                source_type=DataSourceType.SECURE_FILE_BRIDGE,
                priority=2,
                timeout_seconds=5.0,
                retry_attempts=2,
                health_check_interval=30.0
            )
        ]
        
        # Create data sources
        for config in configs:
            if config.source_type == DataSourceType.DIRECT_WINDOWS_MT5:
                source = DirectWindowsMT5Source(config)
            elif config.source_type == DataSourceType.SECURE_FILE_BRIDGE:
                source = SecureFileBridgeSource(config)
            else:
                continue
            
            # Try to initialize
            try:
                if await source.initialize():
                    self.data_sources.append(source)
                    logger.info("Data source initialized", source=config.source_type.value)
                else:
                    logger.warning("Data source initialization failed", source=config.source_type.value)
            except Exception as e:
                logger.error("Data source initialization error", source=config.source_type.value, error=str(e))
        
        # Initialize cache source (always available)
        cache_config = DataSourceConfig(
            source_type=DataSourceType.CACHED_DATA,
            priority=999,  # Lowest priority
            timeout_seconds=1.0,
            retry_attempts=1,
            health_check_interval=10.0
        )
        self.cache_source = CachedDataSource(cache_config)
        await self.cache_source.initialize()
        
        # Set initial active source
        await self._select_best_source()
        
        if not self.active_source:
            raise Exception("No data sources available")
        
        logger.info("MT5 pipeline initialized", 
                   active_source=self.active_source.config.source_type.value,
                   total_sources=len(self.data_sources))
    
    async def _select_best_source(self) -> bool:
        """Select the best available data source"""
        
        best_source = None
        best_score = -1
        
        for source in self.data_sources:
            if not source.config.enabled:
                continue
            
            # Health check
            is_healthy = await source.health_check()
            
            if is_healthy:
                # Calculate source score (priority + health metrics)
                health_score = source.health.success_rate
                priority_score = 1.0 / source.config.priority  # Lower priority number = higher score
                overall_score = (health_score * 0.7) + (priority_score * 0.3)
                
                if overall_score > best_score:
                    best_score = overall_score
                    best_source = source
        
        # Switch source if needed
        if best_source != self.active_source:
            old_source = self.active_source.config.source_type if self.active_source else None
            new_source = best_source.config.source_type
            
            self.active_source = best_source
            
            if old_source:
                FAILOVER_EVENTS.labels(from_source=old_source.value, to_source=new_source.value).inc()
                logger.info("Switched data source", from_source=old_source.value, to_source=new_source.value)
                
                if self.on_source_change:
                    self.on_source_change(old_source, new_source)
            
            return True
        
        return best_source is not None
    
    async def get_live_data(self) -> Dict[str, MarketTick]:
        """Get live market data with automatic failover"""
        
        if not self.active_source:
            await self._select_best_source()
            if not self.active_source:
                logger.error("No active data source available")
                return {}
        
        try:
            # Get data from active source
            ticks = await self.active_source.get_market_data(self.symbols)
            
            # Cache successful data
            if ticks and self.cache_source:
                for tick in ticks.values():
                    self.cache_source.cache_tick_data(tick)
            
            # Update metrics
            self.last_update = datetime.now()
            self.update_count += 1
            
            # Trigger callback
            if self.on_data_update and ticks:
                self.on_data_update(ticks)
            
            return ticks
            
        except Exception as e:
            logger.error("Data fetch failed from active source", 
                        source=self.active_source.config.source_type.value, 
                        error=str(e))
            
            # Try failover
            await self._select_best_source()
            
            # If still no source, try cache
            if not self.active_source and self.cache_source:
                logger.warning("Using cached data as emergency fallback")
                return await self.cache_source.get_market_data(self.symbols)
            
            return {}
    
    async def start_continuous_updates(self):
        """Start continuous data updates"""
        self._running = True
        
        # Start health check task
        self._health_check_task = asyncio.create_task(self._health_check_loop())
        
        logger.info("Started continuous data updates", interval=self.update_interval)
        
        try:
            while self._running:
                start_time = time.time()
                
                # Get data
                await self.get_live_data()
                
                # Update pipeline health metric
                health_scores = [source.health.success_rate for source in self.data_sources]
                if health_scores:
                    PIPELINE_HEALTH.set(sum(health_scores) / len(health_scores))
                
                # Sleep until next update
                elapsed = time.time() - start_time
                sleep_time = max(0, self.update_interval - elapsed)
                await asyncio.sleep(sleep_time)
                
        except Exception as e:
            logger.error("Continuous updates failed", error=str(e))
        finally:
            self._running = False
            if self._health_check_task:
                self._health_check_task.cancel()
    
    async def _health_check_loop(self):
        """Background health checking for all sources"""
        while self._running:
            try:
                # Check all sources
                health_data = {}
                
                for source in self.data_sources:
                    is_healthy = await source.health_check()
                    health_data[source.config.source_type.value] = source.health
                    
                    logger.debug("Health check completed",
                               source=source.config.source_type.value,
                               healthy=is_healthy,
                               success_rate=source.health.success_rate)
                
                # Trigger callback
                if self.on_health_change:
                    self.on_health_change(health_data)
                
                # Consider source switching
                await self._select_best_source()
                
                await asyncio.sleep(30)  # Health check every 30s
                
            except Exception as e:
                logger.error("Health check loop error", error=str(e))
                await asyncio.sleep(10)  # Short delay on error
    
    async def stop(self):
        """Stop the pipeline"""
        logger.info("Stopping MT5 data pipeline")
        
        self._running = False
        
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
        
        # Cleanup data sources
        for source in self.data_sources:
            try:
                await source.cleanup()
            except Exception as e:
                logger.warning("Source cleanup error", 
                             source=source.config.source_type.value, 
                             error=str(e))
    
    def get_status_report(self) -> Dict[str, Any]:
        """Get comprehensive pipeline status"""
        return {
            'active_source': self.active_source.config.source_type.value if self.active_source else None,
            'total_sources': len(self.data_sources),
            'symbols_tracked': len(self.symbols),
            'last_update': self.last_update.isoformat(),
            'update_count': self.update_count,
            'running': self._running,
            'source_health': {
                source.config.source_type.value: {
                    'status': source.health.status.value,
                    'success_rate': source.health.success_rate,
                    'avg_latency_ms': source.health.avg_latency_ms,
                    'consecutive_failures': source.health.consecutive_failures,
                    'last_error': source.health.last_error
                }
                for source in self.data_sources
            }
        }

async def main():
    """Test the production MT5 pipeline"""
    
    print("🏭 PRODUCTION MT5 DATA PIPELINE TEST")
    print("=" * 60)
    
    # Test symbols
    symbols = ['EURUSD', 'EURUSD+', 'BTCUSD', 'XAUUSD']
    
    # Create pipeline
    pipeline = ProductionMT5Pipeline(symbols)
    
    # Setup callbacks
    def on_data_update(ticks: Dict[str, MarketTick]):
        print(f"\n📊 Data Update: {len(ticks)} ticks received")
        for symbol, tick in list(ticks.items())[:3]:  # Show first 3
            print(f"  {symbol}: {tick.bid:.5f}/{tick.ask:.5f} "
                  f"(Source: {tick.source.value}, Quality: {tick.quality_score:.2f})")
    
    def on_source_change(old_source: DataSourceType, new_source: DataSourceType):
        print(f"🔄 Source changed: {old_source.value} → {new_source.value}")
    
    pipeline.on_data_update = on_data_update
    pipeline.on_source_change = on_source_change
    
    try:
        # Initialize
        await pipeline.initialize()
        print(f"✅ Pipeline initialized with {len(pipeline.data_sources)} sources")
        
        # Get status
        status = pipeline.get_status_report()
        print(f"\n📋 Pipeline Status:")
        print(f"   Active source: {status['active_source']}")
        print(f"   Symbols: {status['symbols_tracked']}")
        
        # Test single update
        print(f"\n🧪 Testing single data fetch...")
        ticks = await pipeline.get_live_data()
        print(f"Received {len(ticks)} ticks")
        
        # Run continuous updates for a short test
        print(f"\n🔄 Starting continuous updates (10 seconds)...")
        update_task = asyncio.create_task(pipeline.start_continuous_updates())
        
        await asyncio.sleep(10)
        
        await pipeline.stop()
        update_task.cancel()
        
        print(f"\n✅ Production pipeline test completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Pipeline test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())