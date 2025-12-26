#!/usr/bin/env python3
"""
Dual-Agent RL System: Sniper vs Berserker
Competitive capital allocation with reinforcement learning
Physics-based decision making with realistic friction awareness
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent / "physics"))
sys.path.append(str(Path(__file__).parent.parent / "data"))

import math
import json
import random
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from collections import deque, defaultdict

# Import our custom modules
from enhanced_friction_calculator import EnhancedFrictionCalculator, MarketRegime, InstrumentClass
from trading_strategy_theorem import TradingStrategyTheorem, PhysicsState, MarketPhysics, AgentPersona

class TradeOutcome(Enum):
    """Trade outcome classification"""
    WIN = "WIN"
    LOSS = "LOSS"
    BREAKEVEN = "BREAKEVEN"
    STOPPED = "STOPPED"

@dataclass 
class TradeRecord:
    """Complete trade record for RL learning"""
    entry_time: datetime
    exit_time: datetime
    symbol: str
    agent: AgentPersona
    direction: int                  # 1 for long, -1 for short
    entry_price: float
    exit_price: float
    position_size: float
    hold_bars: int
    hold_days: float
    
    # Physics state at entry
    entry_energy: float            # Available energy (bp)
    entry_friction: float          # Expected friction (bp)
    entry_ef_ratio: float          # Energy/friction ratio
    entry_momentum: float          # Momentum (bp)
    entry_volatility: float        # Volatility (bp)
    entry_volume_pct: float        # Volume percentile
    entry_regime: MarketPhysics    # Physics regime
    
    # Trade results
    gross_pnl_bp: float           # Gross P&L (bp)
    friction_cost_bp: float       # Actual friction cost (bp)
    net_pnl_bp: float             # Net P&L after friction (bp)
    mfe_bp: float                 # Maximum favorable excursion (bp)
    mae_bp: float                 # Maximum adverse excursion (bp)
    
    # Performance metrics
    trade_outcome: TradeOutcome
    energy_efficiency: float      # Actual/predicted energy capture
    friction_accuracy: float      # Predicted/actual friction ratio
    hold_efficiency: float        # Optimal hold vs actual hold
    
    # RL feedback
    reward: float                 # RL reward for this trade
    value_error: float            # Prediction error for learning

@dataclass
class AgentPerformance:
    """Agent performance tracking"""
    total_trades: int = 0
    winning_trades: int = 0
    total_pnl_bp: float = 0.0
    total_friction_bp: float = 0.0
    net_pnl_bp: float = 0.0
    
    max_drawdown_bp: float = 0.0
    max_runup_bp: float = 0.0
    current_streak: int = 0
    longest_win_streak: int = 0
    longest_loss_streak: int = 0
    
    sharpe_ratio: float = 0.0
    calmar_ratio: float = 0.0
    profit_factor: float = 0.0
    
    avg_hold_days: float = 0.0
    avg_energy_efficiency: float = 0.0
    
    # RL metrics
    learning_rate: float = 0.01
    exploration_rate: float = 0.1
    confidence_level: float = 0.5

class SniperAgent:
    """
    Sniper Agent: Efficiency-focused trend following
    
    Philosophy:
    - Predictable energy flows with high efficiency
    - Avoids churn and ranging markets
    - Short to medium holds (hours to days)
    - High win rate target (>65%)
    - Consistent moderate returns
    """
    
    def __init__(self, initial_capital: float = 50000.0):
        self.capital = initial_capital
        self.performance = AgentPerformance()
        self.friction_calc = EnhancedFrictionCalculator()
        
        # Sniper-specific parameters (RL-adjustable)
        self.min_ef_ratio = 8.0           # Higher efficiency requirement
        self.min_trend_strength = 80.0     # Strong trend requirement (bp)
        self.max_volatility = 150.0        # Avoid high volatility
        self.min_volume_pct = 60.0         # Decent volume required
        self.target_hold_hours = 8.0       # Target ~8 hour holds
        self.max_hold_days = 3.0           # Maximum hold period
        
        # Risk management
        self.max_risk_per_trade = 0.015    # 1.5% risk per trade
        self.max_drawdown_limit = 0.08     # 8% max drawdown
        
        # RL memory
        self.trade_memory = deque(maxlen=1000)
        self.prediction_errors = deque(maxlen=100)
        
    def evaluate_opportunity(self, symbol: str, market_data: Dict, 
                           physics_state: PhysicsState) -> Optional[Dict]:
        """Evaluate if opportunity meets Sniper criteria"""
        
        # Sniper entry criteria
        criteria_met = {
            'energy_friction_ratio': physics_state.energy_friction_ratio >= self.min_ef_ratio,
            'trend_strength': abs(physics_state.momentum) >= self.min_trend_strength,
            'volatility_acceptable': physics_state.volatility <= self.max_volatility,
            'volume_adequate': physics_state.volume_percentile >= self.min_volume_pct,
            'regime_suitable': physics_state.regime in [MarketPhysics.UNDERDAMPED, MarketPhysics.CRITICALLY_DAMPED]
        }
        
        # All criteria must be met for Sniper
        if not all(criteria_met.values()):
            return None
        
        # Calculate optimal position
        direction = 1 if physics_state.momentum > 0 else -1
        hold_days = min(self.target_hold_hours / 24.0, self.max_hold_days)
        
        # Calculate expected friction
        friction = self.friction_calc.calculate_friction(
            symbol, direction, hold_days, market_data.get('close', 0)
        )
        
        # Verify still profitable after friction
        expected_return = physics_state.energy * 0.6  # Conservative capture assumption
        if expected_return <= friction.total_friction:
            return None
        
        # Position sizing based on Kelly criterion (simplified)
        win_prob = self._estimate_win_probability(physics_state)
        risk_reward = expected_return / friction.total_friction
        kelly_fraction = (win_prob * risk_reward - (1 - win_prob)) / risk_reward
        kelly_fraction = max(0.0, min(0.25, kelly_fraction))  # Cap at 25%
        
        position_size = (self.capital * self.max_risk_per_trade * kelly_fraction) / friction.total_friction
        
        return {
            'agent': AgentPersona.SNIPER,
            'direction': direction,
            'hold_days': hold_days,
            'position_size': position_size,
            'expected_return': expected_return,
            'expected_friction': friction.total_friction,
            'win_probability': win_prob,
            'criteria_scores': criteria_met,
            'confidence': self.performance.confidence_level
        }
    
    def _estimate_win_probability(self, physics_state: PhysicsState) -> float:
        """Estimate win probability based on current conditions"""
        
        # Base probability from historical performance
        base_prob = 0.65 if self.performance.total_trades == 0 else (
            self.performance.winning_trades / max(1, self.performance.total_trades)
        )
        
        # Adjust based on physics conditions
        ef_bonus = min(0.15, (physics_state.energy_friction_ratio - self.min_ef_ratio) * 0.01)
        trend_bonus = min(0.1, abs(physics_state.momentum - self.min_trend_strength) * 0.001)
        vol_penalty = max(0.0, (physics_state.volatility - self.max_volatility) * 0.001)
        
        adjusted_prob = base_prob + ef_bonus + trend_bonus - vol_penalty
        return max(0.1, min(0.9, adjusted_prob))
    
    def update_from_trade(self, trade_record: TradeRecord) -> None:
        """Update agent parameters based on trade outcome"""
        
        # Update performance metrics
        self.performance.total_trades += 1
        self.performance.total_pnl_bp += trade_record.gross_pnl_bp
        self.performance.total_friction_bp += trade_record.friction_cost_bp
        self.performance.net_pnl_bp += trade_record.net_pnl_bp
        
        if trade_record.net_pnl_bp > 0:
            self.performance.winning_trades += 1
            self.performance.current_streak = max(0, self.performance.current_streak) + 1
            self.performance.longest_win_streak = max(
                self.performance.longest_win_streak, self.performance.current_streak
            )
        else:
            self.performance.current_streak = min(0, self.performance.current_streak) - 1
            self.performance.longest_loss_streak = max(
                self.performance.longest_loss_streak, abs(self.performance.current_streak)
            )
        
        # RL parameter updates
        prediction_error = abs(trade_record.net_pnl_bp - 
                              (trade_record.entry_energy * 0.6 - trade_record.friction_cost_bp))
        self.prediction_errors.append(prediction_error)
        
        # Adaptive parameter adjustment
        if trade_record.net_pnl_bp > 0:
            # Successful trade - slightly relax criteria
            self.min_ef_ratio *= 0.995
            self.performance.confidence_level = min(0.95, self.performance.confidence_level + 0.01)
        else:
            # Failed trade - tighten criteria
            self.min_ef_ratio *= 1.005
            self.performance.confidence_level = max(0.1, self.performance.confidence_level - 0.02)
        
        # Store trade for learning
        self.trade_memory.append(trade_record)

class BerserkerAgent:
    """
    Berserker Agent: Extreme event capture specialist
    
    Philosophy:
    - Wait for perfect storms, then ride them fully
    - High patience, extended holds (days to weeks)
    - Low frequency, high impact trades
    - Targets 75th-95th percentile extremes
    - Massive moves with high probability of success
    """
    
    def __init__(self, initial_capital: float = 50000.0):
        self.capital = initial_capital
        self.performance = AgentPerformance()
        self.friction_calc = EnhancedFrictionCalculator()
        
        # Berserker-specific parameters (RL-adjustable)
        self.min_ef_ratio = 15.0          # Very high efficiency requirement
        self.min_volatility_spike = 3.0    # 3x normal volatility
        self.min_volume_pct = 85.0         # Very high volume required
        self.min_extreme_momentum = 200.0  # 2% minimum move
        self.target_hold_days = 14.0       # Target 2-week holds
        self.max_hold_days = 60.0          # Up to 2 months for extreme events
        
        # Risk management (different from Sniper)
        self.max_risk_per_trade = 0.025    # 2.5% risk per trade (higher)
        self.max_drawdown_limit = 0.15     # 15% max drawdown (higher tolerance)
        
        # Patience parameters
        self.trades_per_month_target = 2    # Very selective
        self.last_trade_time = None
        self.min_rest_days = 7              # Minimum rest between trades
        
        # RL memory
        self.trade_memory = deque(maxlen=500)  # Fewer trades to remember
        self.extreme_events = deque(maxlen=100) # Remember extreme conditions
    
    def evaluate_opportunity(self, symbol: str, market_data: Dict, 
                           physics_state: PhysicsState) -> Optional[Dict]:
        """Evaluate if opportunity meets Berserker criteria"""
        
        # Berserker needs extreme conditions
        extreme_criteria = {
            'extreme_ef_ratio': physics_state.energy_friction_ratio >= self.min_ef_ratio,
            'volatility_spike': physics_state.volatility >= 200.0,  # High volatility threshold
            'volume_surge': physics_state.volume_percentile >= self.min_volume_pct,
            'massive_momentum': abs(physics_state.momentum) >= self.min_extreme_momentum,
            'chaos_regime': physics_state.regime == MarketPhysics.CHAOTIC,
            'patience_ready': self._is_patience_criteria_met()
        }
        
        # Need most criteria for Berserker (not all - extreme events are rare)
        criteria_count = sum(extreme_criteria.values())
        if criteria_count < 4:  # Need at least 4/6 criteria
            return None
        
        # Special crypto directional logic
        direction = self._determine_optimal_direction(symbol, physics_state)
        
        # Calculate optimal hold for extreme event
        hold_days = self._calculate_extreme_hold_period(symbol, physics_state)
        
        # Calculate friction for extended hold
        friction = self.friction_calc.calculate_friction(
            symbol, direction, hold_days, market_data.get('close', 0)
        )
        
        # For extreme events, accept higher friction if event is truly exceptional
        expected_return = physics_state.energy * 0.8  # Aggressive capture assumption
        
        # Berserker accepts negative friction if short-term for extreme reversal potential
        if symbol.upper() == 'BTCUSD' and direction == 1 and hold_days <= 1.0:
            # Special case: BTC long intraday during extreme events
            pass
        elif expected_return <= friction.total_friction:
            return None
        
        # Position sizing for extreme events (larger positions)
        win_prob = self._estimate_extreme_win_probability(physics_state, extreme_criteria)
        risk_reward = abs(expected_return) / max(abs(friction.total_friction), 10.0)
        kelly_fraction = (win_prob * risk_reward - (1 - win_prob)) / max(risk_reward, 1.0)
        kelly_fraction = max(0.0, min(0.4, kelly_fraction))  # Cap at 40%
        
        position_size = (self.capital * self.max_risk_per_trade * kelly_fraction) / max(abs(friction.total_friction), 10.0)
        
        return {
            'agent': AgentPersona.BERSERKER,
            'direction': direction,
            'hold_days': hold_days,
            'position_size': position_size,
            'expected_return': expected_return,
            'expected_friction': friction.total_friction,
            'win_probability': win_prob,
            'criteria_scores': extreme_criteria,
            'confidence': self.performance.confidence_level,
            'extremity_level': criteria_count / 6.0
        }
    
    def _determine_optimal_direction(self, symbol: str, physics_state: PhysicsState) -> int:
        """Determine optimal direction considering asymmetric friction"""
        
        # For crypto, strongly bias toward shorts for extended holds
        if 'BTC' in symbol.upper() or 'ETH' in symbol.upper():
            # Crypto: prefer shorts for extended holds, longs only for intraday
            if physics_state.momentum < 0:
                return -1  # Short on down momentum (with positive carry)
            else:
                return 1   # Long only if very strong up momentum
        else:
            # For other instruments, follow momentum
            return 1 if physics_state.momentum > 0 else -1
    
    def _calculate_extreme_hold_period(self, symbol: str, physics_state: PhysicsState) -> float:
        """Calculate hold period for extreme event capture"""
        
        # Base hold period depends on extremity
        extremity_factor = min(3.0, physics_state.energy_friction_ratio / 10.0)
        base_hold = self.target_hold_days * extremity_factor
        
        # Instrument-specific adjustments
        if 'BTC' in symbol.upper() or 'ETH' in symbol.upper():
            # Crypto: much shorter holds for longs
            if physics_state.momentum > 0:  # Long
                return min(1.0, base_hold * 0.1)  # Maximum 1 day for crypto longs
            else:  # Short
                return min(30.0, base_hold)       # Up to 30 days for crypto shorts
        else:
            return min(self.max_hold_days, base_hold)
    
    def _is_patience_criteria_met(self) -> bool:
        """Check if enough time has passed since last trade"""
        
        if self.last_trade_time is None:
            return True
        
        days_since_last = (datetime.now() - self.last_trade_time).days
        return days_since_last >= self.min_rest_days
    
    def _estimate_extreme_win_probability(self, physics_state: PhysicsState, 
                                        criteria: Dict[str, bool]) -> float:
        """Estimate win probability for extreme events"""
        
        # Base probability higher for extreme events
        base_prob = 0.75 if self.performance.total_trades == 0 else min(0.9, (
            self.performance.winning_trades / max(1, self.performance.total_trades)
        ) + 0.1)
        
        # Bonus for meeting more extreme criteria
        criteria_bonus = sum(criteria.values()) * 0.05
        
        # Volatility spike bonus
        vol_bonus = min(0.1, (physics_state.volatility - 200.0) * 0.001)
        
        # Energy/friction ratio bonus
        ef_bonus = min(0.15, (physics_state.energy_friction_ratio - self.min_ef_ratio) * 0.005)
        
        adjusted_prob = base_prob + criteria_bonus + vol_bonus + ef_bonus
        return max(0.3, min(0.95, adjusted_prob))
    
    def update_from_trade(self, trade_record: TradeRecord) -> None:
        """Update Berserker parameters based on trade outcome"""
        
        # Update performance metrics (similar to Sniper)
        self.performance.total_trades += 1
        self.performance.total_pnl_bp += trade_record.gross_pnl_bp
        self.performance.total_friction_bp += trade_record.friction_cost_bp
        self.performance.net_pnl_bp += trade_record.net_pnl_bp
        
        if trade_record.net_pnl_bp > 0:
            self.performance.winning_trades += 1
            self.performance.current_streak = max(0, self.performance.current_streak) + 1
        else:
            self.performance.current_streak = min(0, self.performance.current_streak) - 1
        
        # RL parameter updates (more conservative for Berserker)
        if trade_record.net_pnl_bp > 500:  # Big win
            # Major success - slightly relax criteria
            self.min_ef_ratio *= 0.99
            self.min_extreme_momentum *= 0.99
            self.performance.confidence_level = min(0.95, self.performance.confidence_level + 0.02)
        elif trade_record.net_pnl_bp < -200:  # Significant loss
            # Failed extreme event - tighten criteria significantly
            self.min_ef_ratio *= 1.02
            self.min_extreme_momentum *= 1.02
            self.performance.confidence_level = max(0.2, self.performance.confidence_level - 0.05)
        
        # Update timing
        self.last_trade_time = trade_record.exit_time
        
        # Store trade and extreme event for learning
        self.trade_memory.append(trade_record)
        if trade_record.entry_volatility > 200.0:
            self.extreme_events.append({
                'volatility': trade_record.entry_volatility,
                'momentum': trade_record.entry_momentum,
                'ef_ratio': trade_record.entry_ef_ratio,
                'outcome': trade_record.net_pnl_bp
            })

class DualAgentController:
    """
    Dual-Agent Controller managing competitive capital allocation
    Sniper vs Berserker with performance-based resource allocation
    """
    
    def __init__(self, total_capital: float = 100000.0):
        self.total_capital = total_capital
        
        # Initialize agents with equal allocation
        initial_allocation = total_capital / 2.0
        self.sniper = SniperAgent(initial_allocation)
        self.berserker = BerserkerAgent(initial_allocation)
        
        # Capital allocation tracking
        self.allocation_history = []
        self.rebalance_frequency = 50  # Rebalance every 50 trades
        self.trades_since_rebalance = 0
        
        # Performance comparison
        self.comparative_metrics = {
            'sniper_sharpe': 0.0,
            'berserker_sharpe': 0.0,
            'sniper_calmar': 0.0,
            'berserker_calmar': 0.0,
            'optimal_allocation': {'sniper': 0.5, 'berserker': 0.5}
        }
    
    def evaluate_opportunity(self, symbol: str, market_data: Dict, 
                           physics_state: PhysicsState) -> Optional[Dict]:
        """Let both agents evaluate opportunity, choose best option"""
        
        # Get evaluations from both agents
        sniper_eval = self.sniper.evaluate_opportunity(symbol, market_data, physics_state)
        berserker_eval = self.berserker.evaluate_opportunity(symbol, market_data, physics_state)
        
        # If neither agent interested, pass
        if not sniper_eval and not berserker_eval:
            return None
        
        # If only one agent interested, use that one
        if sniper_eval and not berserker_eval:
            return sniper_eval
        elif berserker_eval and not sniper_eval:
            return berserker_eval
        
        # If both interested, choose based on expected value and confidence
        sniper_expected_value = (sniper_eval['expected_return'] - sniper_eval['expected_friction']) * sniper_eval['confidence']
        berserker_expected_value = (berserker_eval['expected_return'] - berserker_eval['expected_friction']) * berserker_eval['confidence']
        
        # Add agent performance bias
        sniper_performance_multiplier = 1.0 + (self.sniper.performance.net_pnl_bp / max(1000.0, self.total_capital)) * 0.1
        berserker_performance_multiplier = 1.0 + (self.berserker.performance.net_pnl_bp / max(1000.0, self.total_capital)) * 0.1
        
        sniper_score = sniper_expected_value * sniper_performance_multiplier
        berserker_score = berserker_expected_value * berserker_performance_multiplier
        
        # Choose agent with higher score
        if sniper_score > berserker_score:
            return sniper_eval
        else:
            return berserker_eval
    
    def execute_trade(self, trade_decision: Dict, market_data: Dict) -> TradeRecord:
        """Execute trade and create trade record"""
        
        # This would interface with actual execution in real system
        # For now, simulate the trade execution
        
        agent_type = trade_decision['agent']
        
        # Create simulated trade record
        trade_record = TradeRecord(
            entry_time=datetime.now(),
            exit_time=datetime.now() + timedelta(days=trade_decision['hold_days']),
            symbol=market_data.get('symbol', 'TEST'),
            agent=agent_type,
            direction=trade_decision['direction'],
            entry_price=market_data.get('close', 100.0),
            exit_price=market_data.get('close', 100.0) * (1 + random.gauss(0, 0.01)),  # Simulate price movement
            position_size=trade_decision['position_size'],
            hold_bars=int(trade_decision['hold_days'] * 96),  # Assume M15 bars
            hold_days=trade_decision['hold_days'],
            
            # Physics state (would be captured from actual state)
            entry_energy=trade_decision['expected_return'],
            entry_friction=trade_decision['expected_friction'], 
            entry_ef_ratio=trade_decision['expected_return'] / max(1.0, trade_decision['expected_friction']),
            entry_momentum=market_data.get('momentum', 100.0),
            entry_volatility=market_data.get('volatility', 50.0),
            entry_volume_pct=market_data.get('volume_pct', 70.0),
            entry_regime=MarketPhysics.UNDERDAMPED,
            
            # Results (simulated)
            gross_pnl_bp=random.gauss(trade_decision['expected_return'], abs(trade_decision['expected_return']) * 0.5),
            friction_cost_bp=trade_decision['expected_friction'],
            net_pnl_bp=0.0,  # Will be calculated
            mfe_bp=abs(random.gauss(trade_decision['expected_return'] * 1.5, 50)),
            mae_bp=abs(random.gauss(50, 25)),
            
            trade_outcome=TradeOutcome.WIN,  # Will be determined
            energy_efficiency=random.uniform(0.3, 1.2),
            friction_accuracy=random.uniform(0.8, 1.2),
            hold_efficiency=random.uniform(0.7, 1.1),
            reward=0.0,  # Will be calculated
            value_error=0.0  # Will be calculated
        )
        
        # Calculate derived fields
        trade_record.net_pnl_bp = trade_record.gross_pnl_bp - trade_record.friction_cost_bp
        trade_record.trade_outcome = TradeOutcome.WIN if trade_record.net_pnl_bp > 0 else TradeOutcome.LOSS
        trade_record.reward = trade_record.net_pnl_bp / 100.0  # Simple reward function
        
        # Update appropriate agent
        if agent_type == AgentPersona.SNIPER:
            self.sniper.update_from_trade(trade_record)
        else:
            self.berserker.update_from_trade(trade_record)
        
        # Check for rebalancing
        self.trades_since_rebalance += 1
        if self.trades_since_rebalance >= self.rebalance_frequency:
            self.rebalance_capital()
        
        return trade_record
    
    def rebalance_capital(self) -> None:
        """Rebalance capital between agents based on performance"""
        
        print(f"\n🔄 Rebalancing capital after {self.trades_since_rebalance} trades...")
        
        # Calculate performance metrics
        sniper_trades = max(1, self.sniper.performance.total_trades)
        berserker_trades = max(1, self.berserker.performance.total_trades)
        
        sniper_avg_return = self.sniper.performance.net_pnl_bp / sniper_trades
        berserker_avg_return = self.berserker.performance.net_pnl_bp / berserker_trades
        
        sniper_win_rate = self.sniper.performance.winning_trades / sniper_trades
        berserker_win_rate = self.berserker.performance.winning_trades / berserker_trades
        
        # Simple allocation based on average return and win rate
        sniper_score = sniper_avg_return * sniper_win_rate
        berserker_score = berserker_avg_return * berserker_win_rate
        
        total_score = abs(sniper_score) + abs(berserker_score)
        
        if total_score > 0:
            sniper_allocation = max(0.2, min(0.8, abs(sniper_score) / total_score))
            berserker_allocation = 1.0 - sniper_allocation
        else:
            # Default equal allocation if no clear winner
            sniper_allocation = 0.5
            berserker_allocation = 0.5
        
        # Update agent capital
        self.sniper.capital = self.total_capital * sniper_allocation
        self.berserker.capital = self.total_capital * berserker_allocation
        
        print(f"   Sniper: {sniper_allocation:.1%} (${self.sniper.capital:,.0f})")
        print(f"   Berserker: {berserker_allocation:.1%} (${self.berserker.capital:,.0f})")
        
        # Reset counter
        self.trades_since_rebalance = 0
        
        # Store allocation history
        self.allocation_history.append({
            'sniper_allocation': sniper_allocation,
            'berserker_allocation': berserker_allocation,
            'sniper_performance': sniper_avg_return,
            'berserker_performance': berserker_avg_return
        })
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        
        return {
            'total_capital': self.total_capital,
            'sniper': {
                'capital': self.sniper.capital,
                'trades': self.sniper.performance.total_trades,
                'win_rate': self.sniper.performance.winning_trades / max(1, self.sniper.performance.total_trades),
                'net_pnl': self.sniper.performance.net_pnl_bp,
                'confidence': self.sniper.performance.confidence_level
            },
            'berserker': {
                'capital': self.berserker.capital,
                'trades': self.berserker.performance.total_trades,
                'win_rate': self.berserker.performance.winning_trades / max(1, self.berserker.performance.total_trades),
                'net_pnl': self.berserker.performance.net_pnl_bp,
                'confidence': self.berserker.performance.confidence_level
            },
            'combined_performance': {
                'total_trades': self.sniper.performance.total_trades + self.berserker.performance.total_trades,
                'total_net_pnl': self.sniper.performance.net_pnl_bp + self.berserker.performance.net_pnl_bp
            }
        }

def test_dual_agent_system():
    """Test the dual-agent RL system"""
    
    print("🤖 DUAL-AGENT RL SYSTEM TEST")
    print("="*80)
    
    # Initialize controller
    controller = DualAgentController(100000.0)
    
    # Test scenarios
    test_scenarios = [
        {
            'symbol': 'EURUSD',
            'market_data': {
                'symbol': 'EURUSD',
                'close': 1.0500,
                'momentum': 120.0,    # Strong trend (Sniper territory)
                'volatility': 80.0,   # Moderate volatility
                'volume_pct': 70.0,
                'energy': 150.0,
                'friction': 15.0
            },
            'description': 'EURUSD Strong Trend (Sniper Opportunity)'
        },
        {
            'symbol': 'BTCUSD',
            'market_data': {
                'symbol': 'BTCUSD',
                'close': 95000.0,
                'momentum': -350.0,   # Extreme crash (Berserker territory)
                'volatility': 280.0,  # Very high volatility
                'volume_pct': 95.0,   # Volume spike
                'energy': 500.0,
                'friction': 25.0
            },
            'description': 'BTCUSD Extreme Crash (Berserker Opportunity)'
        },
        {
            'symbol': 'NAS100',
            'market_data': {
                'symbol': 'NAS100',
                'close': 25000.0,
                'momentum': 80.0,     # Moderate trend
                'volatility': 120.0,  # Medium volatility
                'volume_pct': 55.0,   # Below average volume
                'energy': 100.0,
                'friction': 20.0
            },
            'description': 'NAS100 Moderate Conditions (Low Interest)'
        }
    ]
    
    print(f"Testing {len(test_scenarios)} market scenarios...")
    
    for i, scenario in enumerate(test_scenarios):
        print(f"\n📊 Scenario {i+1}: {scenario['description']}")
        
        # Create physics state
        market_data = scenario['market_data']
        physics_state = PhysicsState(
            energy=market_data['energy'],
            friction=market_data['friction'],
            energy_friction_ratio=market_data['energy'] / market_data['friction'],
            momentum=market_data['momentum'],
            volatility=market_data['volatility'],
            volume_percentile=market_data['volume_pct'],
            regime=MarketPhysics.CHAOTIC if market_data['volatility'] > 200 else MarketPhysics.UNDERDAMPED,
            tradability=None
        )
        
        # Get agent evaluations
        sniper_eval = controller.sniper.evaluate_opportunity(
            scenario['symbol'], market_data, physics_state
        )
        berserker_eval = controller.berserker.evaluate_opportunity(
            scenario['symbol'], market_data, physics_state
        )
        
        print(f"   Sniper evaluation: {'✅ Interested' if sniper_eval else '❌ Not interested'}")
        if sniper_eval:
            print(f"      Expected return: {sniper_eval['expected_return']:.1f}bp")
            print(f"      Win probability: {sniper_eval['win_probability']:.1%}")
            print(f"      Hold period: {sniper_eval['hold_days']:.1f} days")
        
        print(f"   Berserker evaluation: {'✅ Interested' if berserker_eval else '❌ Not interested'}")
        if berserker_eval:
            print(f"      Expected return: {berserker_eval['expected_return']:.1f}bp")
            print(f"      Win probability: {berserker_eval['win_probability']:.1%}")
            print(f"      Hold period: {berserker_eval['hold_days']:.1f} days")
        
        # Controller decision
        decision = controller.evaluate_opportunity(scenario['symbol'], market_data, physics_state)
        if decision:
            print(f"   🎯 Controller decision: {decision['agent'].value} agent selected")
            
            # Execute simulated trade
            trade_record = controller.execute_trade(decision, market_data)
            print(f"      Trade result: {trade_record.net_pnl_bp:+.1f}bp")
        else:
            print(f"   ⏸️  Controller decision: No trade")
    
    # Performance summary
    print(f"\n📈 PERFORMANCE SUMMARY:")
    summary = controller.get_performance_summary()
    
    print(f"Sniper Agent:")
    print(f"   Capital: ${summary['sniper']['capital']:,.0f}")
    print(f"   Trades: {summary['sniper']['trades']}")
    print(f"   Win rate: {summary['sniper']['win_rate']:.1%}")
    print(f"   Net P&L: {summary['sniper']['net_pnl']:+.1f}bp")
    print(f"   Confidence: {summary['sniper']['confidence']:.1%}")
    
    print(f"\nBerserker Agent:")
    print(f"   Capital: ${summary['berserker']['capital']:,.0f}")
    print(f"   Trades: {summary['berserker']['trades']}")
    print(f"   Win rate: {summary['berserker']['win_rate']:.1%}")
    print(f"   Net P&L: {summary['berserker']['net_pnl']:+.1f}bp")
    print(f"   Confidence: {summary['berserker']['confidence']:.1%}")
    
    print(f"\nCombined Performance:")
    print(f"   Total trades: {summary['combined_performance']['total_trades']}")
    print(f"   Total net P&L: {summary['combined_performance']['total_net_pnl']:+.1f}bp")
    
    print(f"\n✅ Dual-agent system test completed!")

if __name__ == "__main__":
    test_dual_agent_system()