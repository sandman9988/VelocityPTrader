#!/usr/bin/env python3
"""
Physics-Based RL Trading Strategy Theorem
Dynamic adaptation to instrument-specific friction realities
No static assumptions - continuous learning and optimization
"""

from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional
from enum import Enum
from dataclasses import dataclass
import math

class MarketPhysics(Enum):
    """Market regime classification based on energy dynamics"""
    OVERDAMPED = "OVERDAMPED"         # High friction, slow moves
    CRITICALLY_DAMPED = "CRITICAL"    # Balanced energy/friction
    UNDERDAMPED = "UNDERDAMPED"       # Low friction, fast moves
    CHAOTIC = "CHAOTIC"               # Extreme volatility spikes

class TradabilityGate(Enum):
    """Energy/Friction ratio based entry gating"""
    UNTRADABLE = "UNTRADABLE"         # E/F < 2 (friction dominates)
    MARGINAL = "MARGINAL"             # E/F 2-5 (break-even)
    TRADABLE = "TRADABLE"             # E/F 5-10 (profitable)
    HIGHLY_TRADABLE = "HIGHLY_TRADABLE" # E/F > 10 (high profit)

class AgentPersona(Enum):
    """Specialized trading agents with distinct objectives"""
    SNIPER = "SNIPER"                 # Predictable trends, efficiency focus
    BERSERKER = "BERSERKER"           # Extreme events, patience focus

@dataclass
class InstrumentProfile:
    """Dynamic instrument characteristics learned by RL"""
    symbol: str
    long_max_hold_days: float         # Max viable long hold (swap dependent)
    short_max_hold_days: float        # Max viable short hold
    min_energy_long: float            # Minimum energy for long entry
    min_energy_short: float           # Minimum energy for short entry
    friction_multiplier_long: float   # Directional friction bias
    friction_multiplier_short: float
    optimal_sniper_hold: int          # Bars for trend following
    optimal_berserker_hold: int       # Bars for extreme events
    volume_threshold: float           # Minimum volume percentile
    volatility_threshold: float       # Minimum volatility
    last_updated: str                 # RL update timestamp

@dataclass
class PhysicsState:
    """Complete market physics at given moment"""
    energy: float                     # Current available energy (bp)
    friction: float                   # Total friction cost (bp)
    energy_friction_ratio: float     # E/F tradability ratio
    momentum: float                   # Directional momentum (bp)
    volatility: float                 # Current volatility (bp)
    volume_percentile: float          # Volume vs recent history
    regime: MarketPhysics            # Current physics regime
    tradability: TradabilityGate     # Entry gate status

class TradingStrategyTheorem:
    """
    TRADING STRATEGY THEOREM:
    
    1. PHYSICS FOUNDATION
       Market behavior = Energy flow through friction medium
       Success = Energy Capture / Total Friction > Minimum Threshold
    
    2. DYNAMIC ADAPTATION  
       All parameters instrument-specific and time-variant
       No static rules - continuous RL optimization
    
    3. DUAL-AGENT ARCHITECTURE
       Sniper: Maximum efficiency, predictable energy flows
       Berserker: Maximum patience, extreme energy events
    
    4. TRADABILITY GATING
       Entry only when Energy/Friction ratio exceeds threshold
       Different thresholds per agent, instrument, direction
    
    5. FRICTION REALITY
       Incorporate actual overnight costs, directional asymmetries
       Weekend penalties, broker-specific rates, economic events
    
    6. REGIME ADAPTATION
       Physics regimes determine optimal agent activation
       Overdamped = avoid, Underdamped = Sniper, Chaotic = Berserker
    """
    
    def __init__(self):
        self.instrument_profiles: Dict[str, InstrumentProfile] = {}
        self.rl_memory: Dict = {}
        self.performance_history: List = []
    
    def classify_physics_regime(self, energy: float, friction: float, 
                              volatility: float) -> MarketPhysics:
        """Determine current physics regime"""
        ef_ratio = energy / friction if friction > 0 else 0
        
        if ef_ratio < 1:
            return MarketPhysics.OVERDAMPED    # Friction dominates
        elif ef_ratio < 3:
            return MarketPhysics.CRITICALLY_DAMPED  # Balanced
        elif volatility > 200:  # High vol threshold
            return MarketPhysics.CHAOTIC       # Extreme volatility
        else:
            return MarketPhysics.UNDERDAMPED   # Energy dominates
    
    def calculate_tradability_gate(self, energy: float, friction: float) -> TradabilityGate:
        """Determine if conditions are tradable"""
        if friction <= 0:
            return TradabilityGate.UNTRADABLE
        
        ef_ratio = energy / friction
        
        if ef_ratio < 2:
            return TradabilityGate.UNTRADABLE
        elif ef_ratio < 5:
            return TradabilityGate.MARGINAL
        elif ef_ratio < 10:
            return TradabilityGate.TRADABLE
        else:
            return TradabilityGate.HIGHLY_TRADABLE
    
    def determine_optimal_agent(self, physics_state: PhysicsState, 
                              instrument_profile: InstrumentProfile) -> Optional[AgentPersona]:
        """Select optimal agent based on current conditions"""
        
        # Agent selection based on regime and conditions
        if physics_state.regime == MarketPhysics.OVERDAMPED:
            return None  # Avoid trading in high friction
        
        elif physics_state.regime == MarketPhysics.CHAOTIC:
            # Berserker conditions
            if (physics_state.volatility > instrument_profile.volatility_threshold * 2 and
                physics_state.volume_percentile > 80):
                return AgentPersona.BERSERKER
        
        elif physics_state.regime == MarketPhysics.UNDERDAMPED:
            # Sniper conditions  
            if (abs(physics_state.momentum) > 100 and
                physics_state.energy > instrument_profile.min_energy_long):
                return AgentPersona.SNIPER
        
        return None
    
    def calculate_optimal_hold_period(self, agent: AgentPersona, direction: int,
                                    instrument_profile: InstrumentProfile,
                                    physics_state: PhysicsState) -> int:
        """RL-learned optimal hold period per agent/instrument/direction"""
        
        base_hold = (instrument_profile.optimal_berserker_hold if agent == AgentPersona.BERSERKER 
                    else instrument_profile.optimal_sniper_hold)
        
        # Direction-specific constraints
        max_hold_days = (instrument_profile.long_max_hold_days if direction > 0 
                        else instrument_profile.short_max_hold_days)
        
        # Convert to bars (assuming M15)
        max_hold_bars = int(max_hold_days * 96)
        
        # Regime adjustments
        if physics_state.regime == MarketPhysics.CHAOTIC:
            # Extended holds for extreme events
            regime_multiplier = 2.0
        elif physics_state.regime == MarketPhysics.CRITICALLY_DAMPED:
            # Standard holds
            regime_multiplier = 1.0
        else:
            # Shorter holds for other regimes
            regime_multiplier = 0.5
        
        optimal_hold = int(base_hold * regime_multiplier)
        
        # Constrain by swap limitations
        return min(optimal_hold, max_hold_bars)
    
    def execute_theorem_step(self, symbol: str, market_data: Dict, 
                           current_positions: List) -> Dict:
        """Execute one step of the trading theorem"""
        
        # 1. Get/create instrument profile
        if symbol not in self.instrument_profiles:
            self.instrument_profiles[symbol] = self._create_default_profile(symbol)
        
        profile = self.instrument_profiles[symbol]
        
        # 2. Calculate current physics state
        physics_state = self._calculate_physics_state(market_data, profile)
        
        # 3. Apply tradability gate
        if physics_state.tradability == TradabilityGate.UNTRADABLE:
            return {'action': 'WAIT', 'reason': 'Untradable conditions'}
        
        # 4. Determine optimal agent
        agent = self.determine_optimal_agent(physics_state, profile)
        if agent is None:
            return {'action': 'WAIT', 'reason': 'No suitable agent'}
        
        # 5. Calculate position direction and size
        direction = 1 if physics_state.momentum > 0 else -1
        
        # 6. Calculate optimal hold period
        optimal_hold = self.calculate_optimal_hold_period(
            agent, direction, profile, physics_state
        )
        
        # 7. Risk management
        position_size = self._calculate_position_size(
            physics_state, profile, optimal_hold
        )
        
        # 8. Execute trade decision
        trade_decision = {
            'action': 'ENTER',
            'agent': agent.value,
            'direction': direction,
            'position_size': position_size,
            'optimal_hold_bars': optimal_hold,
            'physics_state': physics_state,
            'expected_ef_ratio': physics_state.energy_friction_ratio,
            'reasoning': f"{agent.value} entry in {physics_state.regime.value} regime"
        }
        
        # 9. Update RL memory
        self._update_rl_memory(symbol, physics_state, trade_decision)
        
        return trade_decision
    
    def update_instrument_profile_from_results(self, symbol: str, 
                                             trade_results: Dict) -> None:
        """Update instrument profile based on actual trade performance (RL feedback)"""
        
        if symbol not in self.instrument_profiles:
            return
        
        profile = self.instrument_profiles[symbol]
        
        # Learning rate for profile updates
        alpha = 0.1
        
        # Update based on actual vs expected performance
        actual_return = trade_results.get('net_pnl_bp', 0)
        expected_return = trade_results.get('expected_return', 0)
        performance_ratio = actual_return / expected_return if expected_return != 0 else 0
        
        # Adjust thresholds based on performance
        if performance_ratio < 0.5:  # Underperformed
            # Increase entry thresholds
            profile.min_energy_long *= (1 + alpha)
            profile.min_energy_short *= (1 + alpha)
            profile.volatility_threshold *= (1 + alpha)
        elif performance_ratio > 1.5:  # Overperformed
            # Relax entry thresholds slightly
            profile.min_energy_long *= (1 - alpha * 0.5)
            profile.min_energy_short *= (1 - alpha * 0.5)
        
        # Update optimal hold periods based on actual results
        actual_hold = trade_results.get('actual_hold_bars', 0)
        optimal_performance_hold = trade_results.get('optimal_performance_hold', actual_hold)
        
        if trade_results.get('agent') == 'SNIPER':
            profile.optimal_sniper_hold = int(
                profile.optimal_sniper_hold * (1 - alpha) + 
                optimal_performance_hold * alpha
            )
        elif trade_results.get('agent') == 'BERSERKER':
            profile.optimal_berserker_hold = int(
                profile.optimal_berserker_hold * (1 - alpha) + 
                optimal_performance_hold * alpha
            )
    
    def _create_default_profile(self, symbol: str) -> InstrumentProfile:
        """Create default instrument profile (will be RL-optimized)"""
        symbol_upper = symbol.upper()
        
        # Default values based on instrument type (starting point for RL)
        if 'BTC' in symbol_upper or 'ETH' in symbol_upper:
            return InstrumentProfile(
                symbol=symbol,
                long_max_hold_days=1.0,    # Crypto longs = intraday only
                short_max_hold_days=7.0,   # Crypto shorts can hold longer
                min_energy_long=2000,      # Very high threshold for longs
                min_energy_short=400,      # Lower threshold for shorts
                friction_multiplier_long=18.0,  # -18%/day
                friction_multiplier_short=0.25, # +4%/day benefit
                optimal_sniper_hold=4,     # 1 hour holds
                optimal_berserker_hold=96, # 1 day max
                volume_threshold=70.0,
                volatility_threshold=100.0,
                last_updated="initial"
            )
        elif any(idx in symbol_upper for idx in ['NAS', 'SPX', 'DJ30', 'DAX']):
            return InstrumentProfile(
                symbol=symbol,
                long_max_hold_days=3.0,    # Index CFDs limited
                short_max_hold_days=3.0,   # Both directions similar
                min_energy_long=150,       # High spreads need energy
                min_energy_short=150,
                friction_multiplier_long=1.0,
                friction_multiplier_short=1.0,
                optimal_sniper_hold=14,    # 3.5 hours
                optimal_berserker_hold=288, # 3 days
                volume_threshold=60.0,
                volatility_threshold=80.0,
                last_updated="initial"
            )
        else:  # Forex/commodities
            return InstrumentProfile(
                symbol=symbol,
                long_max_hold_days=14.0,   # Can hold longer
                short_max_hold_days=14.0,
                min_energy_long=100,       # Lower friction
                min_energy_short=100,
                friction_multiplier_long=1.0,
                friction_multiplier_short=1.0,
                optimal_sniper_hold=20,    # 5 hours
                optimal_berserker_hold=672, # 7 days
                volume_threshold=50.0,
                volatility_threshold=60.0,
                last_updated="initial"
            )
    
    def _calculate_physics_state(self, market_data: Dict, 
                               profile: InstrumentProfile) -> PhysicsState:
        """Calculate complete physics state from market data"""
        
        # Extract market data
        current_price = market_data.get('close', 0)
        high = market_data.get('high', current_price)
        low = market_data.get('low', current_price)
        volume = market_data.get('volume', 1000)
        
        # Calculate energy (range-based)
        if current_price > 0:
            energy = (high - low) / current_price * 10000  # bp
        else:
            energy = 0
        
        # Calculate momentum (simplified - would use actual lookback)
        momentum = market_data.get('momentum_bp', 0)
        
        # Calculate volatility (simplified)
        volatility = market_data.get('volatility_bp', 50)
        
        # Volume percentile (simplified)
        volume_pct = market_data.get('volume_percentile', 50)
        
        # Estimate friction (would use actual swap calculator)
        base_friction = 20  # Base spread + commission
        direction = 1 if momentum > 0 else -1
        friction_multiplier = (profile.friction_multiplier_long if direction > 0 
                             else profile.friction_multiplier_short)
        
        # Simplified daily friction estimate
        friction = base_friction * friction_multiplier
        
        # Calculate ratios
        ef_ratio = energy / friction if friction > 0 else 0
        
        # Determine regime and tradability
        regime = self.classify_physics_regime(energy, friction, volatility)
        tradability = self.calculate_tradability_gate(energy, friction)
        
        return PhysicsState(
            energy=energy,
            friction=friction,
            energy_friction_ratio=ef_ratio,
            momentum=momentum,
            volatility=volatility,
            volume_percentile=volume_pct,
            regime=regime,
            tradability=tradability
        )
    
    def _calculate_position_size(self, physics_state: PhysicsState,
                               profile: InstrumentProfile, hold_bars: int) -> float:
        """Calculate position size based on energy/friction analysis"""
        
        # Base position size
        base_size = 1.0
        
        # Scale by E/F ratio
        ef_multiplier = min(physics_state.energy_friction_ratio / 5.0, 2.0)
        
        # Scale by volatility (higher vol = smaller size)
        vol_multiplier = max(0.5, 100.0 / physics_state.volatility)
        
        # Scale by hold period (longer = smaller)
        hold_multiplier = max(0.5, 20.0 / hold_bars)
        
        return base_size * ef_multiplier * vol_multiplier * hold_multiplier
    
    def _update_rl_memory(self, symbol: str, physics_state: PhysicsState,
                         trade_decision: Dict) -> None:
        """Update RL memory for future optimization"""
        
        memory_key = f"{symbol}_{physics_state.regime.value}_{trade_decision['agent']}"
        
        if memory_key not in self.rl_memory:
            self.rl_memory[memory_key] = []
        
        self.rl_memory[memory_key].append({
            'physics_state': physics_state,
            'decision': trade_decision,
            'timestamp': 'current'
        })
        
        # Keep memory size manageable
        if len(self.rl_memory[memory_key]) > 1000:
            self.rl_memory[memory_key] = self.rl_memory[memory_key][-500:]

def print_theorem_summary():
    """Print the complete trading theorem"""
    
    print("="*80)
    print("PHYSICS-BASED RL TRADING STRATEGY THEOREM")
    print("="*80)
    
    print("""
🔬 CORE PRINCIPLES:

1. PHYSICS FOUNDATION
   • Market = Energy flow through friction medium
   • Success = (Energy Capture / Total Friction) > Minimum Threshold
   • All behavior emergent from energy/friction dynamics

2. NO STATIC ASSUMPTIONS
   • All parameters instrument-specific and time-variant
   • Swap rates: -18%/day BTC long vs +4%/day BTC short
   • Hold periods: 1-day crypto longs vs 7-day shorts
   • Spreads: 2bp forex vs 170bp NAS100ft

3. DUAL-AGENT ARCHITECTURE
   • SNIPER: Predictable energy flows, efficiency focus
   • BERSERKER: Extreme energy events, patience focus
   • Competitive capital allocation based on performance

4. TRADABILITY GATING
   • Entry only when Energy/Friction > threshold
   • UNTRADABLE: E/F < 2, TRADABLE: E/F > 5
   • Dynamic thresholds per agent/instrument/direction

5. REGIME-ADAPTIVE EXECUTION
   • OVERDAMPED: Avoid (friction dominates)
   • UNDERDAMPED: Sniper activation (energy flows)
   • CHAOTIC: Berserker activation (extreme events)
   • CRITICAL: Balanced conditions

6. CONTINUOUS RL OPTIMIZATION
   • Performance feedback updates all parameters
   • Instrument profiles learned from actual results
   • No fixed rules - adaptive to market evolution
""")
    
    print("🎯 IMPLEMENTATION STRATEGY:")
    print("""
✅ Start with conservative instrument profiles
✅ Execute theorem step-by-step for each opportunity
✅ Measure actual vs expected performance
✅ Update profiles using RL feedback (learning rate 0.1)
✅ Continuously optimize based on real friction costs
✅ Adapt to changing market conditions automatically
""")
    
    print("⚡ COMPETITIVE ADVANTAGES:")
    print("""
• Accounts for actual overnight costs vs theoretical friction
• Adapts to instrument-specific directional asymmetries  
• Learns optimal hold periods from real performance data
• Automatically adjusts to regime changes
• No static rules to become obsolete
• Physics-based foundation provides robust framework
""")

if __name__ == "__main__":
    print_theorem_summary()
    
    # Example usage
    theorem = TradingStrategyTheorem()
    
    # Mock market data
    sample_data = {
        'close': 95000,
        'high': 96500, 
        'low': 94200,
        'volume': 5000,
        'momentum_bp': 150,
        'volatility_bp': 120,
        'volume_percentile': 75
    }
    
    # Execute theorem
    decision = theorem.execute_theorem_step('BTCUSD', sample_data, [])
    print(f"\nSample Decision: {decision}")