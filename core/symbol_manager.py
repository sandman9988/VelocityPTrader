#!/usr/bin/env python3
"""
Production Symbol Manager
Handles Standard vs ECN instruments with proper classification and management
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple, NamedTuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import threading
from collections import defaultdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/symbol_manager.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class InstrumentType(Enum):
    """Instrument type classification"""
    FOREX = "FOREX"
    CRYPTO = "CRYPTO"
    INDEX = "INDEX"
    METAL = "METAL"
    ENERGY = "ENERGY"
    BOND = "BOND"
    COMMODITY = "COMMODITY"
    UNKNOWN = "UNKNOWN"

class TradingSession(Enum):
    """Trading session classification"""
    ASIAN = "ASIAN"
    EUROPEAN = "EUROPEAN"
    AMERICAN = "AMERICAN"
    OVERNIGHT = "OVERNIGHT"

@dataclass(frozen=True)
class InstrumentSpecs:
    """Immutable instrument specifications"""
    symbol: str
    description: str
    instrument_type: InstrumentType
    is_ecn: bool
    digits: int
    point: float
    contract_size: float
    margin_initial: float
    margin_maintenance: float
    min_volume: float
    max_volume: float
    volume_step: float
    swap_long: float
    swap_short: float
    trading_sessions: List[TradingSession]
    base_currency: str
    quote_currency: str
    server: str
    last_updated: datetime

@dataclass
class MarketData:
    """Real-time market data"""
    symbol: str
    bid: float
    ask: float
    last: float
    volume: int
    spread_points: float
    spread_pips: float
    timestamp: datetime
    
    def __post_init__(self):
        """Validate market data"""
        if self.bid <= 0 or self.ask <= 0:
            raise ValueError(f"Invalid prices for {self.symbol}: bid={self.bid}, ask={self.ask}")
        if self.bid >= self.ask:
            raise ValueError(f"Bid >= Ask for {self.symbol}: {self.bid} >= {self.ask}")

class SymbolClassifier:
    """Production-grade symbol classification engine"""
    
    # Classification rules
    FOREX_PATTERNS = [
        # Major pairs
        "EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "USDCAD", "NZDUSD",
        # Minor pairs
        "EURGBP", "EURJPY", "GBPJPY", "EURAUD", "GBPAUD", "EURNZD",
        # Exotic pairs (partial matches)
        "USD", "EUR", "GBP", "AUD", "CAD", "CHF", "JPY", "NZD"
    ]
    
    CRYPTO_PATTERNS = ["BTC", "ETH", "LTC", "XRP", "ADA", "DOT", "LINK", "BCH"]
    INDEX_PATTERNS = ["US30", "NAS100", "SPX500", "GER40", "UK100", "AUS200", "JP225", "DJ30", "SP500", "SPI200", "Nikkei"]
    METAL_PATTERNS = ["XAU", "XAG", "GOLD", "SILVER", "COPPER", "PLATINUM", "PALLADIUM"]
    ENERGY_PATTERNS = ["OIL", "BRENT", "NGAS", "HEATING", "CRUDE"]
    BOND_PATTERNS = ["BOND", "NOTE", "GILT", "BUND"]
    
    @classmethod
    def classify_instrument(cls, symbol: str) -> InstrumentType:
        """Classify instrument type based on symbol"""
        symbol_upper = symbol.upper().replace("+", "")  # Remove ECN suffix
        
        # Crypto (highest priority - specific patterns)
        if any(pattern in symbol_upper for pattern in cls.CRYPTO_PATTERNS):
            return InstrumentType.CRYPTO
            
        # Metals (before forex to catch XAUUSD etc.)
        if any(pattern in symbol_upper for pattern in cls.METAL_PATTERNS):
            return InstrumentType.METAL
            
        # Energy
        if any(pattern in symbol_upper for pattern in cls.ENERGY_PATTERNS):
            return InstrumentType.ENERGY
            
        # Indices
        if any(pattern in symbol_upper for pattern in cls.INDEX_PATTERNS):
            return InstrumentType.INDEX
            
        # Bonds
        if any(pattern in symbol_upper for pattern in cls.BOND_PATTERNS):
            return InstrumentType.BOND
            
        # Forex (most common, check length and currency patterns)
        if len(symbol_upper.replace("+", "")) in [6, 7]:  # EURUSD, EURUSD+
            forex_currencies = ["USD", "EUR", "GBP", "AUD", "CAD", "CHF", "JPY", "NZD", "SEK", "NOK", "MXN", "ZAR"]
            if any(curr in symbol_upper for curr in forex_currencies):
                return InstrumentType.FOREX
                
        return InstrumentType.UNKNOWN
    
    @classmethod
    def is_ecn_instrument(cls, symbol: str) -> bool:
        """Check if instrument is ECN (has + suffix)"""
        return symbol.endswith("+")
    
    @classmethod
    def get_base_symbol(cls, symbol: str) -> str:
        """Get base symbol without ECN suffix"""
        return symbol.replace("+", "")
    
    @classmethod
    def parse_currency_pair(cls, symbol: str) -> Tuple[str, str]:
        """Parse currency pair into base and quote currencies"""
        clean_symbol = cls.get_base_symbol(symbol).upper()
        
        # Common currency codes
        currencies = ["USD", "EUR", "GBP", "AUD", "CAD", "CHF", "JPY", "NZD", "SEK", "NOK", "MXN", "ZAR"]
        
        # Try to split 6-character pairs
        if len(clean_symbol) == 6:
            base = clean_symbol[:3]
            quote = clean_symbol[3:]
            if base in currencies and quote in currencies:
                return base, quote
                
        # Default for non-forex or unparseable symbols
        return clean_symbol, "USD"

class ProductionSymbolManager:
    """Production-grade symbol management system"""
    
    def __init__(self, data_source_path: Optional[str] = None):
        """Initialize symbol manager"""
        self.data_source_path = data_source_path or "all_mt5_symbols.json"
        self.instruments: Dict[str, InstrumentSpecs] = {}
        self.market_data: Dict[str, MarketData] = {}
        self.symbol_groups: Dict[InstrumentType, Set[str]] = defaultdict(set)
        self.ecn_mappings: Dict[str, str] = {}  # standard -> ecn mapping
        self.standard_mappings: Dict[str, str] = {}  # ecn -> standard mapping
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Logging
        Path("logs").mkdir(exist_ok=True)
        
        logger.info("Initializing Production Symbol Manager")
        self._load_instruments()
        self._build_mappings()
        
    def _load_instruments(self) -> None:
        """Load instrument specifications from data source"""
        try:
            with open(self.data_source_path, 'r') as f:
                data = json.load(f)
                
            if not data.get('success'):
                raise ValueError("Invalid data source format")
                
            logger.info(f"Loading instruments from {data['account']['server']}")
            logger.info(f"Account: {data['account']['login']} (${data['account']['balance']:,.2f})")
            
            symbols_data = data['symbols']
            loaded_count = 0
            
            with self._lock:
                for symbol_name, symbol_data in symbols_data.items():
                    try:
                        specs = self._create_instrument_specs(symbol_name, symbol_data)
                        self.instruments[symbol_name] = specs
                        self.symbol_groups[specs.instrument_type].add(symbol_name)
                        loaded_count += 1
                        
                    except Exception as e:
                        logger.error(f"Failed to load instrument {symbol_name}: {e}")
                        
            logger.info(f"Successfully loaded {loaded_count} instruments")
            
            # Log breakdown by type
            for instrument_type, symbols in self.symbol_groups.items():
                logger.info(f"{instrument_type.value}: {len(symbols)} instruments")
                
        except Exception as e:
            logger.error(f"Failed to load instruments: {e}")
            raise
            
    def _create_instrument_specs(self, symbol: str, data: Dict) -> InstrumentSpecs:
        """Create instrument specifications from raw data"""
        
        # Classify instrument
        instrument_type = SymbolClassifier.classify_instrument(symbol)
        is_ecn = SymbolClassifier.is_ecn_instrument(symbol)
        base_currency, quote_currency = SymbolClassifier.parse_currency_pair(symbol)
        
        # Determine trading sessions based on instrument type
        sessions = self._get_trading_sessions(instrument_type, base_currency, quote_currency)
        
        return InstrumentSpecs(
            symbol=symbol,
            description=data.get('description', ''),
            instrument_type=instrument_type,
            is_ecn=is_ecn,
            digits=data.get('digits', 5),
            point=data.get('point', 0.00001),
            contract_size=data.get('contract_size', 100000.0),
            margin_initial=data.get('margin_initial', 0.0),
            margin_maintenance=data.get('margin_initial', 0.0),  # Fallback
            min_volume=data.get('min_volume', 0.01),
            max_volume=data.get('max_volume', 100.0),
            volume_step=data.get('volume_step', 0.01),
            swap_long=data.get('swap_long', 0.0),
            swap_short=data.get('swap_short', 0.0),
            trading_sessions=sessions,
            base_currency=base_currency,
            quote_currency=quote_currency,
            server=data.get('server', 'Unknown'),
            last_updated=datetime.now()
        )
        
    def _get_trading_sessions(self, instrument_type: InstrumentType, 
                            base_currency: str, quote_currency: str) -> List[TradingSession]:
        """Determine trading sessions for instrument"""
        
        if instrument_type == InstrumentType.FOREX:
            # Forex trades 24/5
            return [TradingSession.ASIAN, TradingSession.EUROPEAN, TradingSession.AMERICAN]
        elif instrument_type == InstrumentType.CRYPTO:
            # Crypto trades 24/7
            return [TradingSession.ASIAN, TradingSession.EUROPEAN, TradingSession.AMERICAN, TradingSession.OVERNIGHT]
        elif instrument_type == InstrumentType.INDEX:
            # Indices typically trade during their regional sessions
            if any(region in base_currency for region in ["US", "SPX", "NAS", "DJ"]):
                return [TradingSession.AMERICAN]
            elif any(region in base_currency for region in ["GER", "UK", "EU"]):
                return [TradingSession.EUROPEAN]
            elif any(region in base_currency for region in ["JP", "AUS", "SPI"]):
                return [TradingSession.ASIAN]
            else:
                return [TradingSession.EUROPEAN]  # Default
        else:
            # Commodities, metals, energy
            return [TradingSession.ASIAN, TradingSession.EUROPEAN, TradingSession.AMERICAN]
            
    def _build_mappings(self) -> None:
        """Build ECN/Standard instrument mappings"""
        with self._lock:
            for symbol in self.instruments.keys():
                if SymbolClassifier.is_ecn_instrument(symbol):
                    # ECN instrument
                    base_symbol = SymbolClassifier.get_base_symbol(symbol)
                    self.standard_mappings[symbol] = base_symbol
                    
                    # Check if standard version exists
                    if base_symbol in self.instruments:
                        self.ecn_mappings[base_symbol] = symbol
                else:
                    # Standard instrument - check if ECN version exists
                    ecn_symbol = symbol + "+"
                    if ecn_symbol in self.instruments:
                        self.ecn_mappings[symbol] = ecn_symbol
                        self.standard_mappings[ecn_symbol] = symbol
                        
        logger.info(f"Built mappings: {len(self.ecn_mappings)} standard->ECN, {len(self.standard_mappings)} ECN->standard")
        
    def get_instrument(self, symbol: str) -> Optional[InstrumentSpecs]:
        """Get instrument specifications"""
        with self._lock:
            return self.instruments.get(symbol)
            
    def get_ecn_equivalent(self, symbol: str) -> Optional[str]:
        """Get ECN equivalent of standard instrument"""
        with self._lock:
            return self.ecn_mappings.get(symbol)
            
    def get_standard_equivalent(self, symbol: str) -> Optional[str]:
        """Get standard equivalent of ECN instrument"""
        with self._lock:
            return self.standard_mappings.get(symbol)
            
    def get_optimal_instrument(self, base_symbol: str, prefer_ecn: bool = True) -> Optional[str]:
        """Get optimal instrument (ECN preferred for better spreads)"""
        
        # Remove + suffix if present
        clean_base = SymbolClassifier.get_base_symbol(base_symbol)
        
        with self._lock:
            ecn_symbol = clean_base + "+"
            standard_symbol = clean_base
            
            if prefer_ecn and ecn_symbol in self.instruments:
                return ecn_symbol
            elif standard_symbol in self.instruments:
                return standard_symbol
            elif not prefer_ecn and ecn_symbol in self.instruments:
                return ecn_symbol
                
        return None
        
    def get_instruments_by_type(self, instrument_type: InstrumentType) -> List[InstrumentSpecs]:
        """Get all instruments of specified type"""
        with self._lock:
            symbols = self.symbol_groups.get(instrument_type, set())
            return [self.instruments[symbol] for symbol in symbols]
            
    def get_tradeable_pairs(self) -> List[Tuple[str, str]]:
        """Get list of instruments that have both standard and ECN versions"""
        pairs = []
        with self._lock:
            for standard, ecn in self.ecn_mappings.items():
                pairs.append((standard, ecn))
        return pairs
        
    def update_market_data(self, symbol: str, tick_data: Dict) -> None:
        """Update market data for symbol"""
        try:
            specs = self.get_instrument(symbol)
            if not specs:
                logger.warning(f"No specs found for symbol {symbol}")
                return
                
            # Calculate spread
            bid = float(tick_data['bid'])
            ask = float(tick_data['ask'])
            spread_points = ask - bid
            spread_pips = spread_points / specs.point if specs.point > 0 else 0
            
            market_data = MarketData(
                symbol=symbol,
                bid=bid,
                ask=ask,
                last=float(tick_data.get('last', (bid + ask) / 2)),
                volume=int(tick_data.get('volume', 0)),
                spread_points=spread_points,
                spread_pips=spread_pips,
                timestamp=datetime.now()
            )
            
            with self._lock:
                self.market_data[symbol] = market_data
                
        except Exception as e:
            logger.error(f"Failed to update market data for {symbol}: {e}")
            
    def get_market_data(self, symbol: str) -> Optional[MarketData]:
        """Get current market data for symbol"""
        with self._lock:
            return self.market_data.get(symbol)
            
    def get_spread_comparison(self) -> Dict[str, Dict]:
        """Get spread comparison between standard and ECN instruments"""
        comparisons = {}
        
        with self._lock:
            for standard, ecn in self.ecn_mappings.items():
                std_data = self.market_data.get(standard)
                ecn_data = self.market_data.get(ecn)
                
                if std_data and ecn_data:
                    comparisons[standard] = {
                        'standard_spread': std_data.spread_pips,
                        'ecn_spread': ecn_data.spread_pips,
                        'difference': std_data.spread_pips - ecn_data.spread_pips,
                        'savings_pct': ((std_data.spread_pips - ecn_data.spread_pips) / std_data.spread_pips) * 100
                    }
                    
        return comparisons
        
    def get_system_status(self) -> Dict:
        """Get comprehensive system status"""
        with self._lock:
            total_instruments = len(self.instruments)
            ecn_instruments = sum(1 for specs in self.instruments.values() if specs.is_ecn)
            standard_instruments = total_instruments - ecn_instruments
            
            instruments_with_data = len(self.market_data)
            data_coverage = (instruments_with_data / total_instruments) * 100 if total_instruments > 0 else 0
            
            return {
                'total_instruments': total_instruments,
                'ecn_instruments': ecn_instruments,
                'standard_instruments': standard_instruments,
                'tradeable_pairs': len(self.ecn_mappings),
                'instruments_with_market_data': instruments_with_data,
                'data_coverage_pct': round(data_coverage, 1),
                'instrument_breakdown': {
                    instrument_type.value: len(symbols) 
                    for instrument_type, symbols in self.symbol_groups.items()
                },
                'last_updated': max(
                    [data.timestamp for data in self.market_data.values()],
                    default=datetime.now()
                ).isoformat()
            }

def main():
    """Test the production symbol manager"""
    
    print("🏭 PRODUCTION SYMBOL MANAGER TEST")
    print("=" * 60)
    
    try:
        manager = ProductionSymbolManager()
        
        # System status
        status = manager.get_system_status()
        print(f"\n📊 System Status:")
        print(f"   Total instruments: {status['total_instruments']}")
        print(f"   ECN instruments: {status['ecn_instruments']}")
        print(f"   Standard instruments: {status['standard_instruments']}")
        print(f"   Tradeable pairs: {status['tradeable_pairs']}")
        
        # Show some examples
        print(f"\n🔍 Example Instruments:")
        for instrument_type in [InstrumentType.FOREX, InstrumentType.CRYPTO, InstrumentType.INDEX]:
            instruments = manager.get_instruments_by_type(instrument_type)[:3]
            print(f"\n{instrument_type.value}:")
            for inst in instruments:
                ecn_equiv = manager.get_ecn_equivalent(inst.symbol) if not inst.is_ecn else None
                std_equiv = manager.get_standard_equivalent(inst.symbol) if inst.is_ecn else None
                equiv_info = f" (ECN: {ecn_equiv})" if ecn_equiv else f" (Standard: {std_equiv})" if std_equiv else ""
                print(f"  {inst.symbol:<15} | {inst.description[:30]:<30} | ECN: {inst.is_ecn}{equiv_info}")
                
        # Show tradeable pairs
        print(f"\n💱 Tradeable Pairs (Standard ↔ ECN):")
        pairs = manager.get_tradeable_pairs()[:10]
        for standard, ecn in pairs:
            std_specs = manager.get_instrument(standard)
            ecn_specs = manager.get_instrument(ecn)
            print(f"  {standard:<12} ↔ {ecn:<12} | {std_specs.description if std_specs else 'Unknown'}")
            
        print(f"\n✅ Production Symbol Manager working correctly!")
        
    except Exception as e:
        logger.error(f"Symbol manager test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()