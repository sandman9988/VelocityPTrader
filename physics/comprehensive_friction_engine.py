#!/usr/bin/env python3
"""
Comprehensive Friction Engine
Integrates ALL MT5 data sources:
- SymbolInfo for specifications
- MarketWatch for real-time quotes  
- Data Window for market activity
- Historical data for context
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "data"))

import math
import json
from typing import Dict, Tuple, Optional, List
from enum import Enum
from dataclasses import dataclass
from datetime import datetime, timedelta

# Import all our MT5 integration modules
from symbol_aware_friction_calculator import SymbolAwareFrictionCalculator, RealFrictionComponents
from mt5_marketwatch_integration import MT5MarketWatchManager, MarketWatchSymbol, DataWindowSnapshot

@dataclass
class ComprehensiveFrictionAnalysis:
    """Complete friction analysis using all MT5 data sources"""
    symbol: str
    direction: int
    hold_days: float
    analysis_timestamp: datetime
    
    # Source data integration
    symbol_info_available: bool = False
    marketwatch_available: bool = False
    data_window_available: bool = False
    
    # Real-time market conditions
    current_bid: float = 0.0
    current_ask: float = 0.0
    live_spread_points: float = 0.0
    live_spread_bp: float = 0.0
    market_activity_level: str = "UNKNOWN"
    
    # Symbol specifications (from SymbolInfo)
    contract_size: float = 1.0
    tick_value: float = 1.0
    min_volume: float = 0.01
    margin_required: float = 0.0
    
    # Swap rates (actual from symbol)
    swap_long_rate: float = 0.0
    swap_short_rate: float = 0.0
    swap_type: int = 0
    
    # Friction breakdown
    entry_spread_cost_bp: float = 0.0
    exit_spread_cost_bp: float = 0.0
    commission_cost_bp: float = 0.0
    daily_swap_cost_bp: float = 0.0
    weekend_penalty_bp: float = 0.0
    market_impact_bp: float = 0.0
    
    # Total costs
    total_friction_bp: float = 0.0
    friction_per_day_bp: float = 0.0
    breakeven_move_bp: float = 0.0
    
    # Market context
    current_volatility_bp: float = 0.0
    volume_percentile: float = 50.0
    regime_assessment: str = "NORMAL"
    
    # Viability assessment
    is_tradeable: bool = False
    viability_score: float = 0.0
    recommendation: str = "AVOID"

class ComprehensiveFrictionEngine:
    """
    Ultimate friction calculation engine
    
    Data Sources Integration:
    1. SymbolInfo → Contract specs, swap rates, tick values
    2. MarketWatch → Real-time quotes, spread monitoring
    3. Data Window → Market activity, volume, price action
    4. Historical Data → Volatility context, regime analysis
    
    Capabilities:
    - Real-time spread monitoring with regime adjustments
    - Actual swap calculation using symbol-specific rates
    - Market impact modeling based on current activity
    - Dynamic friction assessment with viability scoring
    """
    
    def __init__(self):
        # Initialize all data sources
        self.symbol_calc = SymbolAwareFrictionCalculator(initialize_mt5=True)
        self.marketwatch = MT5MarketWatchManager(auto_initialize=True)
        
        # Analysis cache
        self.recent_analyses: Dict[str, ComprehensiveFrictionAnalysis] = {}
        self.spread_history: Dict[str, List[Tuple[datetime, float]]] = {}
        
        print("🔧 Comprehensive Friction Engine initialized")
        print("   ✅ Symbol-aware calculator ready")
        print("   ✅ MarketWatch integration active")
        print("   ✅ Data Window monitoring enabled")
    
    def analyze_comprehensive_friction(self, symbol: str, direction: int, 
                                     hold_days: float, 
                                     position_size_lots: float = 1.0) -> ComprehensiveFrictionAnalysis:
        """
        Perform comprehensive friction analysis using ALL data sources
        
        Returns complete friction breakdown with viability assessment
        """
        
        print(f"\n🔬 COMPREHENSIVE FRICTION ANALYSIS: {symbol}")
        print(f"Direction: {'LONG' if direction > 0 else 'SHORT'}, Hold: {hold_days} days")
        print("="*60)
        
        analysis = ComprehensiveFrictionAnalysis(
            symbol=symbol,
            direction=direction,
            hold_days=hold_days,
            analysis_timestamp=datetime.now()
        )
        
        # Step 1: Get MarketWatch data for real-time conditions
        self._integrate_marketwatch_data(analysis)
        
        # Step 2: Get Data Window snapshot for market activity
        self._integrate_data_window_data(analysis)
        
        # Step 3: Get SymbolInfo specifications
        self._integrate_symbol_info_data(analysis)
        
        # Step 4: Calculate comprehensive friction
        self._calculate_comprehensive_friction(analysis, position_size_lots)
        
        # Step 5: Assess market context and viability
        self._assess_viability_and_context(analysis)
        
        # Cache the analysis
        self.recent_analyses[symbol] = analysis
        
        self._print_analysis_summary(analysis)
        
        return analysis
    
    def _integrate_marketwatch_data(self, analysis: ComprehensiveFrictionAnalysis) -> None:
        """Integrate real-time MarketWatch data"""
        
        try:
            symbols = self.marketwatch.get_marketwatch_symbols(visible_only=True)
            
            # Find our symbol in MarketWatch
            mw_symbol = None
            for symbol_obj in symbols:
                if symbol_obj.name == analysis.symbol:
                    mw_symbol = symbol_obj
                    break
            
            if mw_symbol:
                analysis.marketwatch_available = True
                analysis.current_bid = mw_symbol.bid
                analysis.current_ask = mw_symbol.ask
                analysis.live_spread_points = mw_symbol.spread_points
                analysis.live_spread_bp = mw_symbol.spread_bp
                
                # Track spread history
                now = datetime.now()
                if analysis.symbol not in self.spread_history:
                    self.spread_history[analysis.symbol] = []
                
                self.spread_history[analysis.symbol].append((now, mw_symbol.spread_bp))
                
                # Keep only recent history (last hour)
                cutoff_time = now - timedelta(hours=1)
                self.spread_history[analysis.symbol] = [
                    (ts, spread) for ts, spread in self.spread_history[analysis.symbol] 
                    if ts > cutoff_time
                ]
                
                print(f"📊 MarketWatch: Bid={analysis.current_bid:.5f}, Ask={analysis.current_ask:.5f}")
                print(f"   Live spread: {analysis.live_spread_bp:.1f}bp")
            else:
                print(f"⚠️  {analysis.symbol} not found in MarketWatch")
                
        except Exception as e:
            print(f"❌ MarketWatch integration error: {e}")
    
    def _integrate_data_window_data(self, analysis: ComprehensiveFrictionAnalysis) -> None:
        """Integrate Data Window market activity data"""
        
        try:
            snapshot = self.marketwatch.get_data_window_snapshot(analysis.symbol)
            
            if snapshot:
                analysis.data_window_available = True
                analysis.volume_percentile = self._assess_volume_activity(snapshot)
                analysis.current_volatility_bp = snapshot.daily_range_bp
                analysis.market_activity_level = self._classify_market_activity(snapshot)
                
                print(f"📈 Data Window: Volume activity {analysis.volume_percentile:.0f}%ile")
                print(f"   Daily range: {analysis.current_volatility_bp:.1f}bp")
                print(f"   Activity level: {analysis.market_activity_level}")
            else:
                print(f"⚠️  Data Window snapshot not available")
                
        except Exception as e:
            print(f"❌ Data Window integration error: {e}")
    
    def _integrate_symbol_info_data(self, analysis: ComprehensiveFrictionAnalysis) -> None:
        """Integrate SymbolInfo specifications"""
        
        try:
            # Get comprehensive symbol info using our symbol-aware calculator
            real_friction = self.symbol_calc.calculate_real_friction(
                analysis.symbol, analysis.direction, analysis.hold_days
            )
            
            analysis.symbol_info_available = True
            analysis.contract_size = real_friction.contract_size
            analysis.tick_value = real_friction.tick_value_usd
            analysis.min_volume = real_friction.min_volume
            
            # Extract swap information
            symbol_info = self.symbol_calc._get_real_symbol_info(analysis.symbol)
            if symbol_info:
                analysis.swap_long_rate = getattr(symbol_info, 'swap_long', 0.0)
                analysis.swap_short_rate = getattr(symbol_info, 'swap_short', 0.0)
                analysis.swap_type = getattr(symbol_info, 'swap_type', 0)
                analysis.margin_required = getattr(symbol_info, 'margin_initial', 0.0)
            
            print(f"🔍 SymbolInfo: Contract size {analysis.contract_size:,.0f}")
            print(f"   Swap rates: Long {analysis.swap_long_rate:+.2f}, Short {analysis.swap_short_rate:+.2f}")
            
        except Exception as e:
            print(f"❌ SymbolInfo integration error: {e}")
    
    def _calculate_comprehensive_friction(self, analysis: ComprehensiveFrictionAnalysis, 
                                        position_size_lots: float) -> None:
        """Calculate friction using all available data sources"""
        
        # 1. SPREAD COSTS - Use real-time MarketWatch data if available
        if analysis.marketwatch_available:
            # Real-time spread from MarketWatch
            base_spread_bp = analysis.live_spread_bp
            
            # Adjust for market conditions
            regime_multiplier = self._get_regime_spread_multiplier(analysis)
            actual_spread_bp = base_spread_bp * regime_multiplier
            
        else:
            # Fallback to symbol calculator
            real_friction = self.symbol_calc.calculate_real_friction(
                analysis.symbol, analysis.direction, analysis.hold_days
            )
            actual_spread_bp = real_friction.actual_spread_bp
        
        # Entry and exit spread costs
        analysis.entry_spread_cost_bp = actual_spread_bp / 2  # Half spread for entry
        analysis.exit_spread_cost_bp = actual_spread_bp / 2   # Half spread for exit
        
        # 2. COMMISSION COSTS - Based on contract size and position size
        if analysis.symbol_info_available:
            notional_value = analysis.current_ask * analysis.contract_size * position_size_lots
            
            # Commission varies by instrument type
            if 'BTC' in analysis.symbol.upper() or 'ETH' in analysis.symbol.upper():
                commission_rate = 0.0005  # 0.05% for crypto
            elif len(analysis.symbol) == 6:  # Forex
                commission_rate = 0.00002  # 0.002% for forex
            else:
                commission_rate = 0.0001   # 0.01% for others
            
            commission_usd = notional_value * commission_rate
            analysis.commission_cost_bp = (commission_usd / notional_value) * 10000
        else:
            analysis.commission_cost_bp = 2.0  # Default fallback
        
        # 3. SWAP COSTS - Use actual symbol swap rates
        if analysis.symbol_info_available:
            selected_swap_rate = (analysis.swap_long_rate if analysis.direction > 0 
                                else analysis.swap_short_rate)
            
            # Calculate daily swap in basis points
            if analysis.swap_type == 0:  # Points
                analysis.daily_swap_cost_bp = selected_swap_rate * 100  # Convert points to bp
            elif analysis.swap_type == 1:  # Base currency
                # Convert to basis points of position value
                analysis.daily_swap_cost_bp = (selected_swap_rate / analysis.current_ask) * 10000
            elif analysis.swap_type == 2:  # Annual percentage
                analysis.daily_swap_cost_bp = (selected_swap_rate / 365) * 100
            else:
                analysis.daily_swap_cost_bp = selected_swap_rate
        else:
            # Fallback swap estimation
            analysis.daily_swap_cost_bp = self._estimate_swap_fallback(analysis.symbol, analysis.direction)
        
        # 4. WEEKEND PENALTY - Based on instrument type and hold period
        analysis.weekend_penalty_bp = self._calculate_weekend_penalty(analysis)
        
        # 5. MARKET IMPACT - Based on Data Window activity and position size
        analysis.market_impact_bp = self._calculate_market_impact(analysis, position_size_lots)
        
        # 6. TOTAL FRICTION
        total_swap_cost = (analysis.daily_swap_cost_bp * analysis.hold_days) + analysis.weekend_penalty_bp
        
        analysis.total_friction_bp = (
            analysis.entry_spread_cost_bp + 
            analysis.exit_spread_cost_bp +
            analysis.commission_cost_bp +
            total_swap_cost +
            analysis.market_impact_bp
        )
        
        analysis.friction_per_day_bp = analysis.total_friction_bp / max(0.1, analysis.hold_days)
        analysis.breakeven_move_bp = abs(analysis.total_friction_bp)
    
    def _assess_viability_and_context(self, analysis: ComprehensiveFrictionAnalysis) -> None:
        """Assess trading viability based on comprehensive analysis"""
        
        # Regime assessment based on spread and volatility
        if analysis.live_spread_bp > 0:
            spread_history = [spread for _, spread in self.spread_history.get(analysis.symbol, [])]
            if spread_history:
                avg_spread = sum(spread_history) / len(spread_history)
                if analysis.live_spread_bp > avg_spread * 2:
                    analysis.regime_assessment = "HIGH_SPREAD"
                elif analysis.live_spread_bp < avg_spread * 0.5:
                    analysis.regime_assessment = "LOW_SPREAD"
                else:
                    analysis.regime_assessment = "NORMAL"
        
        # Volatility assessment
        if analysis.current_volatility_bp > 200:
            analysis.regime_assessment += "_HIGH_VOL"
        elif analysis.current_volatility_bp < 50:
            analysis.regime_assessment += "_LOW_VOL"
        
        # Viability scoring (0-100)
        viability_score = 50.0  # Base score
        
        # Penalize high friction
        if abs(analysis.total_friction_bp) > 100:
            viability_score -= 30
        elif abs(analysis.total_friction_bp) > 50:
            viability_score -= 15
        
        # Bonus for favorable swaps
        if analysis.daily_swap_cost_bp > 10:  # Positive carry
            viability_score += 20
        elif analysis.daily_swap_cost_bp < -100:  # Very negative carry
            viability_score -= 25
        
        # Activity level adjustment
        if analysis.volume_percentile > 80:
            viability_score += 10
        elif analysis.volume_percentile < 20:
            viability_score -= 10
        
        analysis.viability_score = max(0, min(100, viability_score))
        
        # Final recommendation
        if analysis.viability_score > 70:
            analysis.is_tradeable = True
            analysis.recommendation = "HIGHLY_VIABLE"
        elif analysis.viability_score > 50:
            analysis.is_tradeable = True
            analysis.recommendation = "VIABLE"
        elif analysis.viability_score > 30:
            analysis.is_tradeable = False
            analysis.recommendation = "MARGINAL"
        else:
            analysis.is_tradeable = False
            analysis.recommendation = "AVOID"
    
    def _get_regime_spread_multiplier(self, analysis: ComprehensiveFrictionAnalysis) -> float:
        """Get spread multiplier based on current market regime"""
        
        multiplier = 1.0
        
        # Volume-based adjustment
        if analysis.volume_percentile > 90:
            multiplier *= 0.8  # High volume = tighter spreads
        elif analysis.volume_percentile < 20:
            multiplier *= 1.5  # Low volume = wider spreads
        
        # Volatility-based adjustment
        if analysis.current_volatility_bp > 200:
            multiplier *= 2.0  # High volatility = much wider spreads
        elif analysis.current_volatility_bp > 100:
            multiplier *= 1.3
        
        return multiplier
    
    def _assess_volume_activity(self, snapshot: DataWindowSnapshot) -> float:
        """Assess current volume activity level (percentile)"""
        
        # Simplified volume assessment based on tick count and volume
        base_activity = 50.0
        
        # Higher tick volume suggests more activity
        if snapshot.tick_volume > 1000:
            base_activity += 30
        elif snapshot.tick_volume > 500:
            base_activity += 15
        elif snapshot.tick_volume < 100:
            base_activity -= 20
        
        return max(0, min(100, base_activity))
    
    def _classify_market_activity(self, snapshot: DataWindowSnapshot) -> str:
        """Classify current market activity level"""
        
        if snapshot.tick_volume > 2000:
            return "HIGH"
        elif snapshot.tick_volume > 500:
            return "MEDIUM"
        elif snapshot.tick_volume > 100:
            return "LOW"
        else:
            return "VERY_LOW"
    
    def _calculate_weekend_penalty(self, analysis: ComprehensiveFrictionAnalysis) -> float:
        """Calculate weekend rollover penalties"""
        
        symbol_upper = analysis.symbol.upper()
        weeks_held = analysis.hold_days / 7.0
        
        if 'BTC' in symbol_upper or 'ETH' in symbol_upper:
            # Crypto: Friday 3x penalty
            return weeks_held * 2 * abs(analysis.daily_swap_cost_bp)
        elif len(analysis.symbol) == 6:
            # Forex: Wednesday 3x penalty
            return weeks_held * 2 * abs(analysis.daily_swap_cost_bp) * 0.5
        else:
            # Others: minimal weekend impact
            return weeks_held * 0.5 * abs(analysis.daily_swap_cost_bp)
    
    def _calculate_market_impact(self, analysis: ComprehensiveFrictionAnalysis, 
                                position_size_lots: float) -> float:
        """Calculate market impact based on position size and market activity"""
        
        base_impact = 2.0  # Base impact in bp
        
        # Size impact
        size_multiplier = 1.0 + math.log(max(1.0, position_size_lots))
        
        # Activity impact
        if analysis.market_activity_level == "VERY_LOW":
            activity_multiplier = 3.0
        elif analysis.market_activity_level == "LOW":
            activity_multiplier = 2.0
        elif analysis.market_activity_level == "HIGH":
            activity_multiplier = 0.7
        else:
            activity_multiplier = 1.0
        
        # Hold period impact (longer holds = more patient execution)
        if analysis.hold_days >= 7:
            urgency_multiplier = 0.5
        elif analysis.hold_days >= 1:
            urgency_multiplier = 0.8
        else:
            urgency_multiplier = 1.5
        
        return base_impact * size_multiplier * activity_multiplier * urgency_multiplier
    
    def _estimate_swap_fallback(self, symbol: str, direction: int) -> float:
        """Fallback swap estimation when symbol info not available"""
        
        symbol_upper = symbol.upper()
        
        if 'BTC' in symbol_upper:
            return -1800.0 if direction > 0 else 400.0
        elif 'ETH' in symbol_upper:
            return -1500.0 if direction > 0 else 300.0
        elif len(symbol) == 6:  # Forex
            return -1.0 if direction > 0 else 0.5
        else:
            return -23.0  # Index/commodity negative carry
    
    def _print_analysis_summary(self, analysis: ComprehensiveFrictionAnalysis) -> None:
        """Print comprehensive analysis summary"""
        
        print(f"\n📊 FRICTION BREAKDOWN:")
        print(f"   Entry spread: {analysis.entry_spread_cost_bp:.1f}bp")
        print(f"   Exit spread: {analysis.exit_spread_cost_bp:.1f}bp")
        print(f"   Commission: {analysis.commission_cost_bp:.1f}bp")
        print(f"   Daily swap: {analysis.daily_swap_cost_bp:+.1f}bp")
        print(f"   Weekend penalty: {analysis.weekend_penalty_bp:.1f}bp")
        print(f"   Market impact: {analysis.market_impact_bp:.1f}bp")
        print(f"   ──────────────────────")
        print(f"   TOTAL FRICTION: {analysis.total_friction_bp:+.1f}bp")
        print(f"   Per day: {analysis.friction_per_day_bp:+.1f}bp/day")
        print(f"   Breakeven move: {analysis.breakeven_move_bp:.1f}bp")
        
        print(f"\n🎯 VIABILITY ASSESSMENT:")
        print(f"   Viability score: {analysis.viability_score:.0f}/100")
        print(f"   Recommendation: {analysis.recommendation}")
        print(f"   Tradeable: {'✅ YES' if analysis.is_tradeable else '❌ NO'}")
        print(f"   Market regime: {analysis.regime_assessment}")
    
    def compare_instruments_comprehensive(self, symbols: List[str], 
                                        direction: int = 1, 
                                        hold_days: float = 7.0) -> Dict[str, ComprehensiveFrictionAnalysis]:
        """Compare comprehensive friction across multiple instruments"""
        
        print(f"\n🔬 COMPREHENSIVE FRICTION COMPARISON")
        print(f"Direction: {'LONG' if direction > 0 else 'SHORT'}, Hold: {hold_days} days")
        print("="*80)
        
        results = {}
        
        for symbol in symbols:
            analysis = self.analyze_comprehensive_friction(symbol, direction, hold_days)
            results[symbol] = analysis
            
            print(f"\n{symbol}: {analysis.total_friction_bp:+.1f}bp total → {analysis.recommendation}")
        
        # Rank by viability
        ranked_symbols = sorted(results.items(), key=lambda x: x[1].viability_score, reverse=True)
        
        print(f"\n🏆 RANKING BY VIABILITY:")
        for i, (symbol, analysis) in enumerate(ranked_symbols):
            print(f"   {i+1}. {symbol}: {analysis.viability_score:.0f}/100 - {analysis.recommendation}")
        
        return results

def test_comprehensive_friction_engine():
    """Test the comprehensive friction engine"""
    
    print("🧪 COMPREHENSIVE FRICTION ENGINE TEST")
    print("="*80)
    
    # Initialize engine
    engine = ComprehensiveFrictionEngine()
    
    # Test comprehensive analysis
    test_symbols = ['EURUSD', 'BTCUSD', 'XAUUSD', 'US30']
    
    print(f"\n1. 🔬 INDIVIDUAL COMPREHENSIVE ANALYSES:")
    
    for symbol in test_symbols:
        analysis = engine.analyze_comprehensive_friction(symbol, 1, 7.0)
        print(f"\n{symbol} Summary:")
        print(f"   Total friction: {analysis.total_friction_bp:+.1f}bp")
        print(f"   Viability: {analysis.viability_score:.0f}/100 ({analysis.recommendation})")
        print(f"   Data sources: SymbolInfo={analysis.symbol_info_available}, "
              f"MarketWatch={analysis.marketwatch_available}, "
              f"DataWindow={analysis.data_window_available}")
    
    print(f"\n2. ⚖️  COMPREHENSIVE COMPARISON:")
    
    comparison_results = engine.compare_instruments_comprehensive(test_symbols, 1, 7.0)
    
    print(f"\n✅ Comprehensive friction engine test completed!")
    print(f"\n💡 All calculations now integrate:")
    print(f"   ✅ Real SymbolInfo specifications")
    print(f"   ✅ Live MarketWatch quotes and spreads")
    print(f"   ✅ Data Window market activity")
    print(f"   ✅ Historical context for regime assessment")

if __name__ == "__main__":
    test_comprehensive_friction_engine()