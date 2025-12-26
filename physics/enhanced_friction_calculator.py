#!/usr/bin/env python3
"""
Enhanced Friction Calculator with Directional Asymmetries
Real-world friction modeling including swap rates, weekend penalties, regime adjustments
Based on actual MT5 trading conditions discovered in research
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "data"))

import math
import json
from typing import Dict, Tuple, Optional
from enum import Enum
from dataclasses import dataclass
from datetime import datetime, timedelta

class InstrumentClass(Enum):
    """Instrument classification for friction modeling"""
    FOREX = "FOREX"
    CRYPTO = "CRYPTO" 
    INDEX = "INDEX"
    COMMODITY = "COMMODITY"
    STOCK = "STOCK"

class MarketRegime(Enum):
    """Market regime affects friction costs"""
    NORMAL = "NORMAL"           # Standard conditions
    VOLATILE = "VOLATILE"       # High volatility increases spreads
    ILLIQUID = "ILLIQUID"       # Low volume increases impact
    NEWS_EVENT = "NEWS_EVENT"   # Major events spike spreads
    WEEKEND = "WEEKEND"         # Weekend gaps and rollover

@dataclass
class FrictionComponents:
    """Breakdown of all friction components"""
    spread_cost: float          # Bid-ask spread cost (bp)
    commission_cost: float      # Broker commission (bp)
    swap_cost: float           # Overnight financing (bp)
    market_impact_cost: float  # Price impact from execution (bp)
    weekend_penalty: float     # Weekend rollover penalty (bp)
    regime_adjustment: float   # Regime-specific adjustments (bp)
    total_friction: float      # Sum of all components (bp)

class EnhancedFrictionCalculator:
    """
    Enhanced friction calculator incorporating real trading costs
    
    Key discoveries from research:
    - Crypto: -18%/day long vs +4%/day short asymmetry
    - Index CFDs: High negative carry (-8.5% annually)
    - Friday 3x swap penalties for crypto
    - Wednesday 3x swap penalties for forex
    - Spread variations by regime (2x-5x in volatile conditions)
    """
    
    def __init__(self):
        # Load instrument-specific parameters
        self.instrument_parameters = self._initialize_instrument_parameters()
        
        # Load current market conditions
        self.current_regime = MarketRegime.NORMAL
        self.volatility_multiplier = 1.0
        self.liquidity_multiplier = 1.0
    
    def _initialize_instrument_parameters(self) -> Dict:
        """Initialize realistic instrument parameters based on research"""
        
        return {
            # FOREX PAIRS
            'EURUSD': {
                'class': InstrumentClass.FOREX,
                'base_spread_bp': 1.5,
                'commission_bp': 0.0,          # Usually spread-only
                'swap_long_annual': -1.2,      # Interest rate differential
                'swap_short_annual': 0.8,
                'contract_size': 100000,
                'point_value': 1.0,
                'weekend_multiplier': 3.0,     # Wednesday 3x swap
                'volatility_spread_factor': 2.0
            },
            'GBPUSD': {
                'class': InstrumentClass.FOREX,
                'base_spread_bp': 1.8,
                'commission_bp': 0.0,
                'swap_long_annual': -2.1,
                'swap_short_annual': 1.2,
                'contract_size': 100000,
                'point_value': 1.0,
                'weekend_multiplier': 3.0,
                'volatility_spread_factor': 2.5
            },
            'USDJPY': {
                'class': InstrumentClass.FOREX,
                'base_spread_bp': 1.2,
                'commission_bp': 0.0,
                'swap_long_annual': 2.8,       # Positive carry trade
                'swap_short_annual': -3.5,     # Negative for shorts
                'contract_size': 100000,
                'point_value': 1.0,
                'weekend_multiplier': 3.0,
                'volatility_spread_factor': 2.0
            },
            
            # CRYPTO
            'BTCUSD': {
                'class': InstrumentClass.CRYPTO,
                'base_spread_bp': 20.0,        # Wide crypto spreads
                'commission_bp': 0.1,          # Small commission %
                'swap_long_annual': -18.0 * 365,  # -18% PER DAY!
                'swap_short_annual': 4.0 * 365,   # +4% PER DAY
                'contract_size': 1.0,
                'point_value': 1.0,
                'weekend_multiplier': 3.0,     # Friday 3x penalty
                'volatility_spread_factor': 3.0
            },
            'ETHUSD': {
                'class': InstrumentClass.CRYPTO,
                'base_spread_bp': 25.0,
                'commission_bp': 0.15,
                'swap_long_annual': -15.0 * 365,
                'swap_short_annual': 3.0 * 365,
                'contract_size': 1.0,
                'point_value': 1.0,
                'weekend_multiplier': 3.0,
                'volatility_spread_factor': 3.5
            },
            
            # INDICES
            'NAS100': {
                'class': InstrumentClass.INDEX,
                'base_spread_bp': 10.0,
                'commission_bp': 2.0,          # Index CFD commission
                'swap_long_annual': -8.5,      # Negative carry
                'swap_short_annual': -8.0,     # Also negative (slightly better)
                'contract_size': 1.0,
                'point_value': 1.0,
                'weekend_multiplier': 1.4,     # Less weekend impact
                'volatility_spread_factor': 4.0  # High volatility impact
            },
            'SPX500': {
                'class': InstrumentClass.INDEX,
                'base_spread_bp': 8.0,
                'commission_bp': 1.8,
                'swap_long_annual': -7.2,
                'swap_short_annual': -6.8,
                'contract_size': 1.0,
                'point_value': 1.0,
                'weekend_multiplier': 1.4,
                'volatility_spread_factor': 3.5
            },
            'GER40': {
                'class': InstrumentClass.INDEX,
                'base_spread_bp': 6.0,
                'commission_bp': 1.5,
                'swap_long_annual': -5.5,      # European indices cheaper
                'swap_short_annual': -5.2,
                'contract_size': 1.0,
                'point_value': 1.0,
                'weekend_multiplier': 1.4,
                'volatility_spread_factor': 3.0
            },
            
            # COMMODITIES
            'XAUUSD': {
                'class': InstrumentClass.COMMODITY,
                'base_spread_bp': 3.0,
                'commission_bp': 0.5,
                'swap_long_annual': -2.8,      # Storage costs
                'swap_short_annual': -1.2,     # Less negative for shorts
                'contract_size': 100.0,
                'point_value': 1.0,
                'weekend_multiplier': 1.2,
                'volatility_spread_factor': 2.5
            },
            'XAGUSD': {
                'class': InstrumentClass.COMMODITY,
                'base_spread_bp': 4.0,
                'commission_bp': 0.8,
                'swap_long_annual': -3.5,
                'swap_short_annual': -1.8,
                'contract_size': 5000.0,
                'point_value': 1.0,
                'weekend_multiplier': 1.2,
                'volatility_spread_factor': 3.0
            },
            'USOIL': {
                'class': InstrumentClass.COMMODITY,
                'base_spread_bp': 5.0,
                'commission_bp': 1.0,
                'swap_long_annual': -4.2,      # Storage costs
                'swap_short_annual': -2.8,
                'contract_size': 1000.0,
                'point_value': 1.0,
                'weekend_multiplier': 1.3,
                'volatility_spread_factor': 4.0
            }
        }
    
    def calculate_friction(self, 
                          symbol: str, 
                          direction: int,  # 1 for long, -1 for short
                          hold_days: float,
                          entry_price: float = 0,
                          position_size: float = 1.0,
                          current_regime: MarketRegime = MarketRegime.NORMAL) -> FrictionComponents:
        """
        Calculate complete friction breakdown for a trade
        
        Args:
            symbol: Trading instrument
            direction: 1 for long, -1 for short
            hold_days: Number of days to hold position
            entry_price: Entry price (for percentage-based calculations)
            position_size: Position size multiplier
            current_regime: Current market regime
            
        Returns:
            FrictionComponents with complete breakdown
        """
        
        # Get instrument parameters
        params = self._get_instrument_params(symbol)
        
        # 1. SPREAD COST (entry + exit)
        base_spread = params['base_spread_bp']
        regime_spread_multiplier = self._get_regime_spread_multiplier(current_regime)
        volatility_multiplier = params.get('volatility_spread_factor', 1.0)
        
        # Spread cost depends on regime and volatility
        if current_regime in [MarketRegime.VOLATILE, MarketRegime.NEWS_EVENT]:
            spread_multiplier = regime_spread_multiplier * volatility_multiplier
        else:
            spread_multiplier = regime_spread_multiplier
        
        spread_cost = base_spread * spread_multiplier
        
        # 2. COMMISSION COST (one-time or per-day)
        base_commission = params['commission_bp']
        
        # Some brokers charge daily commission for CFDs
        if params['class'] == InstrumentClass.INDEX:
            commission_cost = base_commission * max(1.0, hold_days * 0.1)  # Small daily fee
        else:
            commission_cost = base_commission  # One-time
        
        # 3. SWAP COST (the critical component for extended holds)
        daily_swap_rate = self._calculate_daily_swap_rate(params, direction)
        base_swap_cost = daily_swap_rate * hold_days
        
        # Weekend/holiday multipliers
        weekend_penalty = self._calculate_weekend_penalty(params, hold_days)
        total_swap_cost = base_swap_cost + weekend_penalty
        
        # 4. MARKET IMPACT COST (decreases with patience)
        market_impact = self._calculate_market_impact(
            params, position_size, hold_days, current_regime
        )
        
        # 5. REGIME ADJUSTMENTS
        regime_adjustment = self._calculate_regime_adjustments(
            params, current_regime, hold_days
        )
        
        # Total friction
        total_friction = (spread_cost + commission_cost + total_swap_cost + 
                         market_impact + regime_adjustment)
        
        return FrictionComponents(
            spread_cost=spread_cost,
            commission_cost=commission_cost,
            swap_cost=total_swap_cost,
            market_impact_cost=market_impact,
            weekend_penalty=weekend_penalty,
            regime_adjustment=regime_adjustment,
            total_friction=total_friction
        )
    
    def _get_instrument_params(self, symbol: str) -> Dict:
        """Get parameters for instrument, with fallback defaults"""
        
        # Try exact match first
        if symbol in self.instrument_parameters:
            return self.instrument_parameters[symbol]
        
        # Try partial matches
        symbol_upper = symbol.upper()
        for key, params in self.instrument_parameters.items():
            if key in symbol_upper or symbol_upper in key:
                return params
        
        # Fallback based on symbol characteristics
        if len(symbol) == 6 and symbol[:3] != symbol[3:]:  # Likely forex
            return {
                'class': InstrumentClass.FOREX,
                'base_spread_bp': 2.0,
                'commission_bp': 0.0,
                'swap_long_annual': -1.0,
                'swap_short_annual': 0.5,
                'contract_size': 100000,
                'point_value': 1.0,
                'weekend_multiplier': 3.0,
                'volatility_spread_factor': 2.0
            }
        elif any(crypto in symbol_upper for crypto in ['BTC', 'ETH', 'CRYPTO']):
            return {
                'class': InstrumentClass.CRYPTO,
                'base_spread_bp': 30.0,
                'commission_bp': 0.2,
                'swap_long_annual': -20.0 * 365,  # Very expensive longs
                'swap_short_annual': 5.0 * 365,   # Beneficial shorts
                'contract_size': 1.0,
                'point_value': 1.0,
                'weekend_multiplier': 3.0,
                'volatility_spread_factor': 4.0
            }
        else:  # Default to index/commodity
            return {
                'class': InstrumentClass.INDEX,
                'base_spread_bp': 8.0,
                'commission_bp': 2.0,
                'swap_long_annual': -6.0,
                'swap_short_annual': -5.5,
                'contract_size': 1.0,
                'point_value': 1.0,
                'weekend_multiplier': 1.5,
                'volatility_spread_factor': 3.0
            }
    
    def _calculate_daily_swap_rate(self, params: Dict, direction: int) -> float:
        """Calculate daily swap rate in basis points"""
        
        if direction > 0:  # Long position
            annual_rate = params['swap_long_annual']
        else:  # Short position
            annual_rate = params['swap_short_annual']
        
        # Convert annual % to daily basis points
        daily_rate = annual_rate / 365 * 100  # Convert to basis points
        
        return daily_rate
    
    def _calculate_weekend_penalty(self, params: Dict, hold_days: float) -> float:
        """Calculate weekend/holiday rollover penalties"""
        
        weekend_multiplier = params.get('weekend_multiplier', 1.0)
        
        if weekend_multiplier <= 1.0:
            return 0.0
        
        # Estimate number of weekends in hold period
        weeks_held = hold_days / 7.0
        
        # Weekend penalty = extra swap charges
        extra_days = (weekend_multiplier - 1.0)  # e.g., 3x = 2 extra days
        weekend_penalty_days = weeks_held * extra_days
        
        # Calculate penalty based on daily swap rate
        direction = 1  # Assume worst case (long for crypto)
        daily_swap = self._calculate_daily_swap_rate(params, direction)
        
        return weekend_penalty_days * abs(daily_swap)
    
    def _calculate_market_impact(self, params: Dict, position_size: float, 
                                hold_days: float, regime: MarketRegime) -> float:
        """Calculate market impact based on execution urgency"""
        
        base_impact = 2.0  # Base impact in bp
        
        # Size impact
        size_multiplier = 1.0 + math.log(max(1.0, position_size))
        
        # Urgency impact (decreases with longer holds = more patient execution)
        if hold_days < 0.1:  # Intraday
            urgency_multiplier = 3.0  # Rush penalty
        elif hold_days < 1.0:  # Same day
            urgency_multiplier = 2.0
        elif hold_days < 7.0:  # Week
            urgency_multiplier = 1.0
        else:  # Long term
            urgency_multiplier = 0.5  # Patient execution bonus
        
        # Regime impact
        regime_multipliers = {
            MarketRegime.NORMAL: 1.0,
            MarketRegime.VOLATILE: 2.0,
            MarketRegime.ILLIQUID: 3.0,
            MarketRegime.NEWS_EVENT: 4.0,
            MarketRegime.WEEKEND: 1.5
        }
        
        regime_multiplier = regime_multipliers.get(regime, 1.0)
        
        return base_impact * size_multiplier * urgency_multiplier * regime_multiplier
    
    def _calculate_regime_adjustments(self, params: Dict, regime: MarketRegime, 
                                    hold_days: float) -> float:
        """Calculate regime-specific friction adjustments"""
        
        adjustments = {
            MarketRegime.NORMAL: 0.0,
            MarketRegime.VOLATILE: 1.0,      # Higher execution costs
            MarketRegime.ILLIQUID: 2.0,      # Poor fills
            MarketRegime.NEWS_EVENT: 5.0,    # Major event premium
            MarketRegime.WEEKEND: 1.0        # Weekend gap risk
        }
        
        base_adjustment = adjustments.get(regime, 0.0)
        
        # Longer holds reduce regime impact (time diversification)
        time_reduction = min(0.8, hold_days * 0.1)
        
        return base_adjustment * (1.0 - time_reduction)
    
    def _get_regime_spread_multiplier(self, regime: MarketRegime) -> float:
        """Get spread multiplier for current regime"""
        
        multipliers = {
            MarketRegime.NORMAL: 1.0,
            MarketRegime.VOLATILE: 2.5,      # Spreads widen significantly
            MarketRegime.ILLIQUID: 3.0,      # Very wide spreads
            MarketRegime.NEWS_EVENT: 4.0,    # Extreme spread widening
            MarketRegime.WEEKEND: 1.5        # Moderate widening
        }
        
        return multipliers.get(regime, 1.0)
    
    def get_breakeven_move(self, symbol: str, direction: int, hold_days: float,
                          regime: MarketRegime = MarketRegime.NORMAL) -> float:
        """Calculate minimum move required to break even after friction"""
        
        friction = self.calculate_friction(symbol, direction, hold_days, 0, 1.0, regime)
        return friction.total_friction
    
    def compare_hold_periods(self, symbol: str, direction: int, 
                           max_days: int = 30) -> Dict[int, FrictionComponents]:
        """Compare friction costs across different hold periods"""
        
        results = {}
        
        for days in [1, 2, 3, 7, 14, 21, 30]:
            if days <= max_days:
                friction = self.calculate_friction(symbol, direction, days)
                results[days] = friction
        
        return results
    
    def analyze_directional_asymmetry(self, symbol: str, 
                                    hold_days: float = 7.0) -> Dict[str, float]:
        """Analyze long vs short friction asymmetry"""
        
        long_friction = self.calculate_friction(symbol, 1, hold_days)
        short_friction = self.calculate_friction(symbol, -1, hold_days)
        
        return {
            'long_total': long_friction.total_friction,
            'short_total': short_friction.total_friction,
            'asymmetry_bp': long_friction.total_friction - short_friction.total_friction,
            'asymmetry_ratio': long_friction.total_friction / short_friction.total_friction,
            'long_swap': long_friction.swap_cost,
            'short_swap': short_friction.swap_cost,
            'swap_asymmetry': long_friction.swap_cost - short_friction.swap_cost
        }

def test_enhanced_friction_calculator():
    """Test the enhanced friction calculator with real scenarios"""
    
    print("🧪 ENHANCED FRICTION CALCULATOR TEST")
    print("="*80)
    
    calc = EnhancedFrictionCalculator()
    
    # Test scenarios from our research
    test_scenarios = [
        ('BTCUSD', 1, 1.0, "BTC Long (1 day)"),
        ('BTCUSD', -1, 7.0, "BTC Short (7 days)"),
        ('BTCUSD', 1, 7.0, "BTC Long (7 days) - Should be expensive!"),
        ('EURUSD', 1, 30.0, "EUR Long (30 days)"),
        ('USDJPY', 1, 30.0, "JPY Carry Trade Long (30 days)"),
        ('NAS100', 1, 14.0, "NAS100 Long (14 days)"),
        ('XAUUSD', -1, 7.0, "Gold Short (7 days)")
    ]
    
    print(f"\n📊 FRICTION ANALYSIS RESULTS:")
    print(f"{'Scenario':<25} {'Total':<10} {'Spread':<8} {'Swap':<10} {'Impact':<8} {'Viable?'}")
    print("-" * 80)
    
    for symbol, direction, days, description in test_scenarios:
        friction = calc.calculate_friction(symbol, direction, days)
        
        # Determine if trade is viable (arbitrary threshold: <100bp total friction)
        viable = "✅" if friction.total_friction < 100 else "❌"
        
        print(f"{description:<25} {friction.total_friction:<10.1f}bp {friction.spread_cost:<8.1f}bp "
              f"{friction.swap_cost:<10.1f}bp {friction.market_impact_cost:<8.1f}bp {viable}")
    
    # Test directional asymmetry
    print(f"\n⚖️  DIRECTIONAL ASYMMETRY ANALYSIS:")
    
    asymmetry_tests = ['BTCUSD', 'EURUSD', 'USDJPY', 'NAS100']
    
    for symbol in asymmetry_tests:
        asymmetry = calc.analyze_directional_asymmetry(symbol, 7.0)
        
        print(f"\n{symbol} (7-day hold):")
        print(f"   Long friction:  {asymmetry['long_total']:+.1f}bp")
        print(f"   Short friction: {asymmetry['short_total']:+.1f}bp")
        print(f"   Asymmetry:      {asymmetry['asymmetry_bp']:+.1f}bp")
        print(f"   Ratio:          {asymmetry['asymmetry_ratio']:.2f}x")
        
        if asymmetry['asymmetry_bp'] > 50:
            print(f"   💡 Strong bias toward SHORT positions")
        elif asymmetry['asymmetry_bp'] < -50:
            print(f"   💡 Strong bias toward LONG positions")
        else:
            print(f"   💡 Relatively neutral direction bias")
    
    # Test regime impact
    print(f"\n🌪️  REGIME IMPACT ANALYSIS:")
    
    regimes = [MarketRegime.NORMAL, MarketRegime.VOLATILE, MarketRegime.NEWS_EVENT]
    
    for regime in regimes:
        friction = calc.calculate_friction('EURUSD', 1, 1.0, current_regime=regime)
        print(f"   {regime.value:<12}: {friction.total_friction:.1f}bp total friction")
    
    print(f"\n✅ Enhanced friction calculator validation complete!")

if __name__ == "__main__":
    test_enhanced_friction_calculator()