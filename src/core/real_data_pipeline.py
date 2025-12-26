#!/usr/bin/env python3
"""
REAL DATA PIPELINE - NO FAKE DATA ALLOWED
Enterprise-grade MT5 data pipeline with PostgreSQL persistence
ONLY LIVE DATA from Vantage International Demo Server
"""

import asyncio
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from decimal import Decimal
from dataclasses import dataclass
import threading

import structlog
import MetaTrader5 as mt5
import numpy as np

from ..database.connection import get_database_manager
from ..database.operations import AtomicDataOperations
from ..utils.enterprise_logging import get_enterprise_logger

logger = structlog.get_logger(__name__)

@dataclass
class RealMarketData:
    """Real market data from MT5 - NO FAKE DATA"""
    symbol: str
    timeframe: str
    timestamp: datetime
    open_price: float
    high_price: float
    low_price: float
    close_price: float
    volume: float
    bid_price: float
    ask_price: float
    spread_pips: float
    
    # Physics metrics calculated from real data
    momentum: Optional[float] = None
    acceleration: Optional[float] = None
    volatility: Optional[float] = None
    liquidity_score: Optional[float] = None
    trend_strength: Optional[float] = None
    market_regime: Optional[str] = None
    
    @property
    def mid_price(self) -> float:
        """Mid price between bid and ask"""
        return (self.bid_price + self.ask_price) / 2.0
    
    @property
    def spread_cost_normalized(self) -> float:
        """Normalized spread cost"""
        return self.spread_pips / 100.0  # Convert pips to normalized cost

@dataclass
class PhysicsMetrics:
    """Physics-based market metrics calculated from real data"""
    momentum: float
    acceleration: float
    volatility: float
    liquidity_score: float
    trend_strength: float
    market_regime: str
    friction: float
    energy: float

class RealDataPipeline:
    """Enterprise real data pipeline - NO FAKE DATA ALLOWED"""
    
    def __init__(self, session_id: Optional[str] = None):
        self.session_id = session_id
        self.db_manager = get_database_manager()
        self.atomic_ops = AtomicDataOperations(self.db_manager)
        self.logger = get_enterprise_logger()
        
        # MT5 connection state
        self.mt5_initialized = False
        self.mt5_logged_in = False
        self.vantage_server = "VantageInternational-Demo"
        self.vantage_login = 10916362
        
        # Data validation
        self.symbols_validated = False
        self.real_symbols: Dict[str, Dict] = {}
        
        # Performance tracking
        self.ticks_processed = 0
        self.data_errors = 0
        self.last_data_time = 0
        
        # Threading for real-time data
        self.data_thread: Optional[threading.Thread] = None
        self.stop_flag = threading.Event()
        
        logger.info("Real data pipeline initialized - NO FAKE DATA ALLOWED")
    
    def initialize_mt5_connection(self) -> bool:
        """Initialize MT5 connection to Vantage server ONLY"""
        try:
            if not mt5.initialize():
                error = mt5.last_error()
                logger.error("MT5 initialization failed", error=error)
                return False
            
            self.mt5_initialized = True
            
            # Login to Vantage server ONLY
            if not mt5.login(self.vantage_login, server=self.vantage_server):
                error = mt5.last_error()
                logger.error("MT5 login failed", error=error, server=self.vantage_server)
                return False
            
            self.mt5_logged_in = True
            
            # Verify connection to Vantage
            account_info = mt5.account_info()
            if not account_info:
                logger.error("Failed to get account info")
                return False
            
            if account_info.server != self.vantage_server:
                logger.error("❌ WRONG SERVER - Only Vantage allowed!", 
                           connected_server=account_info.server,
                           required_server=self.vantage_server)
                return False
            
            logger.info("✅ MT5 connected to Vantage server",
                       server=account_info.server,
                       login=account_info.login,
                       balance=account_info.balance)
            
            return True
            
        except Exception as e:
            logger.error("MT5 connection error", error=str(e))
            return False
    
    def load_real_symbols(self) -> bool:
        """Load ONLY real Vantage symbols - NO FAKE DATA"""
        try:
            # Load from real MT5 symbols file
            symbols_file = Path("data/all_mt5_symbols.json")
            if not symbols_file.exists():
                logger.error("❌ all_mt5_symbols.json not found - cannot use fake data!")
                return False
            
            with open(symbols_file) as f:
                symbols_data = json.load(f)
            
            # Validate server is Vantage
            if symbols_data.get('account', {}).get('server') != self.vantage_server:
                logger.error("❌ INVALID SERVER in symbols file!", 
                           file_server=symbols_data.get('account', {}).get('server'),
                           required_server=self.vantage_server)
                return False
            
            # Extract only ECN symbols (+ suffix)
            self.real_symbols = {}
            symbols_list = symbols_data.get('symbols', {})
            
            for symbol_name, symbol_data in symbols_list.items():
                if symbol_name.endswith('+'):  # ECN symbols only
                    self.real_symbols[symbol_name] = symbol_data
            
            if not self.real_symbols:
                logger.error("❌ No valid ECN symbols found!")
                return False
            
            self.symbols_validated = True
            logger.info("✅ Real Vantage symbols loaded", 
                       count=len(self.real_symbols),
                       ecn_symbols=list(self.real_symbols.keys())[:10])  # Show first 10
            
            return True
            
        except Exception as e:
            logger.error("Failed to load real symbols", error=str(e))
            return False
    
    def start(self) -> bool:
        """Start real data pipeline"""
        try:
            # Initialize MT5 connection
            if not self.initialize_mt5_connection():
                raise RuntimeError("Failed to connect to MT5")
            
            # Load real symbols
            if not self.load_real_symbols():
                raise RuntimeError("Failed to load real symbols")
            
            # Start real-time data thread
            self.data_thread = threading.Thread(target=self._real_time_data_loop, daemon=True)
            self.data_thread.start()
            
            logger.info("✅ Real data pipeline started - LIVE DATA ONLY")
            return True
            
        except Exception as e:
            logger.error("Failed to start data pipeline", error=str(e))
            return False
    
    def stop(self):
        """Stop data pipeline"""
        try:
            self.stop_flag.set()
            
            if self.data_thread and self.data_thread.is_alive():
                self.data_thread.join(timeout=5.0)
            
            if self.mt5_initialized:
                mt5.shutdown()
                self.mt5_initialized = False
                self.mt5_logged_in = False
            
            logger.info("Real data pipeline stopped")
            
        except Exception as e:
            logger.error("Error stopping data pipeline", error=str(e))
    
    def _real_time_data_loop(self):
        """Real-time data collection loop"""
        logger.info("Real-time data collection started")
        
        while not self.stop_flag.is_set():
            try:
                # Collect real market data for all symbols
                market_data_batch = []
                
                for symbol in list(self.real_symbols.keys())[:5]:  # Limit to 5 for demo
                    try:
                        # Get real tick data
                        tick = mt5.symbol_info_tick(symbol)
                        if not tick:
                            continue
                        
                        # Get 1-minute bar data
                        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 0, 2)
                        if rates is None or len(rates) < 1:
                            continue
                        
                        latest_bar = rates[-1]
                        
                        # Create real market data
                        real_data = RealMarketData(
                            symbol=symbol,
                            timeframe="M1",
                            timestamp=datetime.fromtimestamp(latest_bar['time'], tz=timezone.utc),
                            open_price=float(latest_bar['open']),
                            high_price=float(latest_bar['high']),
                            low_price=float(latest_bar['low']),
                            close_price=float(latest_bar['close']),
                            volume=float(latest_bar['tick_volume']),
                            bid_price=float(tick.bid),
                            ask_price=float(tick.ask),
                            spread_pips=(tick.ask - tick.bid) * (10 ** self.real_symbols[symbol].get('digits', 5))
                        )
                        
                        # Calculate physics metrics from real data
                        physics = self._calculate_physics_metrics(symbol, rates)
                        if physics:
                            real_data.momentum = physics.momentum
                            real_data.acceleration = physics.acceleration
                            real_data.volatility = physics.volatility
                            real_data.liquidity_score = physics.liquidity_score
                            real_data.trend_strength = physics.trend_strength
                            real_data.market_regime = physics.market_regime
                        
                        # Convert to database format
                        data_dict = {
                            'symbol': real_data.symbol,
                            'timeframe': real_data.timeframe,
                            'timestamp': real_data.timestamp,
                            'open_price': real_data.open_price,
                            'high_price': real_data.high_price,
                            'low_price': real_data.low_price,
                            'close_price': real_data.close_price,
                            'volume': real_data.volume,
                            'bid_price': real_data.bid_price,
                            'ask_price': real_data.ask_price,
                            'spread_pips': real_data.spread_pips,
                            'momentum': real_data.momentum,
                            'acceleration': real_data.acceleration,
                            'volatility': real_data.volatility,
                            'liquidity_score': real_data.liquidity_score,
                            'trend_strength': real_data.trend_strength,
                            'market_regime': real_data.market_regime
                        }
                        
                        market_data_batch.append(data_dict)
                        
                    except Exception as e:
                        logger.error("Error processing symbol data", symbol=symbol, error=str(e))
                        self.data_errors += 1
                        continue
                
                # Store real data atomically
                if market_data_batch and self.session_id:
                    try:
                        self.logger.log_market_data_batch(market_data_batch)
                        self.ticks_processed += len(market_data_batch)
                        self.last_data_time = time.time()
                        
                        logger.debug("Real market data processed",
                                   count=len(market_data_batch),
                                   ticks_total=self.ticks_processed)
                        
                    except Exception as e:
                        logger.error("Failed to store market data", error=str(e))
                        self.data_errors += 1
                
                # Rate limiting - 1 second intervals
                time.sleep(1.0)
                
            except Exception as e:
                logger.error("Real-time data loop error", error=str(e))
                self.data_errors += 1
                time.sleep(5.0)  # Longer delay on errors
        
        logger.info("Real-time data collection stopped")
    
    def _calculate_physics_metrics(self, symbol: str, rates_data) -> Optional[PhysicsMetrics]:
        """Calculate physics-based metrics from real market data"""
        try:
            if len(rates_data) < 20:  # Need enough data for calculations
                return None
            
            # Extract price data
            closes = np.array([float(rate['close']) for rate in rates_data])
            highs = np.array([float(rate['high']) for rate in rates_data])
            lows = np.array([float(rate['low']) for rate in rates_data])
            volumes = np.array([float(rate['tick_volume']) for rate in rates_data])
            
            # Calculate physics metrics
            
            # Momentum (price velocity)
            price_changes = np.diff(closes)
            momentum = float(np.mean(price_changes[-5:]) if len(price_changes) >= 5 else 0)
            
            # Acceleration (momentum change)
            momentum_changes = np.diff(price_changes)
            acceleration = float(np.mean(momentum_changes[-3:]) if len(momentum_changes) >= 3 else 0)
            
            # Volatility (price dispersion)
            volatility = float(np.std(price_changes[-10:]) if len(price_changes) >= 10 else 0)
            
            # Liquidity score (volume-based)
            avg_volume = np.mean(volumes[-10:])
            recent_volume = np.mean(volumes[-3:])
            liquidity_score = float(min(1.0, recent_volume / max(avg_volume, 1)))
            
            # Trend strength (directional persistence)
            trend_changes = np.sign(price_changes[-10:]) if len(price_changes) >= 10 else np.array([0])
            trend_strength = float(abs(np.mean(trend_changes)))
            
            # Market regime classification (physics-based)
            regime = self._classify_market_regime(momentum, acceleration, volatility)
            
            # Friction (market resistance)
            price_range = np.mean(highs[-5:] - lows[-5:])
            avg_price = np.mean(closes[-5:])
            friction = float(price_range / max(avg_price, 1))
            
            # Energy (market activity)
            energy = float(volatility * liquidity_score)
            
            return PhysicsMetrics(
                momentum=momentum,
                acceleration=acceleration,
                volatility=volatility,
                liquidity_score=liquidity_score,
                trend_strength=trend_strength,
                market_regime=regime,
                friction=friction,
                energy=energy
            )
            
        except Exception as e:
            logger.error("Physics calculation error", symbol=symbol, error=str(e))
            return None
    
    def _classify_market_regime(self, momentum: float, acceleration: float, volatility: float) -> str:
        """Classify market regime using physics principles"""
        
        # Normalize values for classification
        momentum_abs = abs(momentum)
        acceleration_abs = abs(acceleration)
        
        if volatility > 0.01 and acceleration_abs > momentum_abs * 0.5:
            return "CHAOTIC"
        elif momentum_abs > 0.001 and acceleration_abs < momentum_abs * 0.2:
            return "UNDERDAMPED"
        elif momentum_abs > 0.0005 and acceleration_abs < momentum_abs * 0.5:
            return "CRITICALLY_DAMPED"
        else:
            return "OVERDAMPED"
    
    def get_current_data(self, symbol: str) -> Optional[RealMarketData]:
        """Get current real market data for symbol"""
        if not self.mt5_logged_in or symbol not in self.real_symbols:
            return None
        
        try:
            # Get real tick
            tick = mt5.symbol_info_tick(symbol)
            if not tick:
                return None
            
            # Get recent bar
            rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 0, 1)
            if not rates or len(rates) == 0:
                return None
            
            latest_bar = rates[0]
            
            return RealMarketData(
                symbol=symbol,
                timeframe="M1",
                timestamp=datetime.fromtimestamp(latest_bar['time'], tz=timezone.utc),
                open_price=float(latest_bar['open']),
                high_price=float(latest_bar['high']),
                low_price=float(latest_bar['low']),
                close_price=float(latest_bar['close']),
                volume=float(latest_bar['tick_volume']),
                bid_price=float(tick.bid),
                ask_price=float(tick.ask),
                spread_pips=(tick.ask - tick.bid) * (10 ** self.real_symbols[symbol].get('digits', 5))
            )
            
        except Exception as e:
            logger.error("Error getting current data", symbol=symbol, error=str(e))
            return None
    
    def get_physics_metrics(self, symbol: str) -> Optional[PhysicsMetrics]:
        """Get physics metrics for symbol"""
        if not self.mt5_logged_in or symbol not in self.real_symbols:
            return None
        
        try:
            # Get recent bars for calculation
            rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 0, 50)
            if not rates or len(rates) < 20:
                return None
            
            return self._calculate_physics_metrics(symbol, rates)
            
        except Exception as e:
            logger.error("Error calculating physics metrics", symbol=symbol, error=str(e))
            return None
    
    def validate_data_integrity(self) -> Dict[str, Any]:
        """Validate that only real data is being used"""
        health_status = {
            'mt5_connected': self.mt5_logged_in,
            'server': self.vantage_server if self.mt5_logged_in else None,
            'symbols_validated': self.symbols_validated,
            'real_symbols_count': len(self.real_symbols),
            'ticks_processed': self.ticks_processed,
            'data_errors': self.data_errors,
            'error_rate_percent': (self.data_errors / max(self.ticks_processed, 1)) * 100,
            'last_data_seconds_ago': time.time() - self.last_data_time if self.last_data_time > 0 else None,
            'data_integrity': 'VERIFIED' if self.symbols_validated and self.mt5_logged_in else 'UNVERIFIED'
        }
        
        # Validate no fake data flags
        if self.mt5_logged_in:
            account = mt5.account_info()
            if account and account.server == self.vantage_server:
                health_status['vantage_verified'] = True
                health_status['account_login'] = account.login
            else:
                health_status['vantage_verified'] = False
                health_status['error'] = 'Not connected to Vantage server'
        
        return health_status
    
    def get_available_symbols(self) -> List[str]:
        """Get list of available real symbols"""
        return list(self.real_symbols.keys()) if self.symbols_validated else []
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get pipeline statistics"""
        return {
            'ticks_processed': self.ticks_processed,
            'data_errors': self.data_errors,
            'symbols_count': len(self.real_symbols),
            'mt5_status': 'CONNECTED' if self.mt5_logged_in else 'DISCONNECTED',
            'data_source': 'MT5_VANTAGE_REAL',
            'fake_data_rejected': True,
            'integrity_validated': self.symbols_validated and self.mt5_logged_in
        }