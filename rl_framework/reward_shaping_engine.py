#!/usr/bin/env python3
"""
Reward Shaping Engine for RL Trading System
Advanced reward shaping techniques for optimal RL agent training including:
- Multi-objective reward functions
- Intrinsic motivation mechanisms  
- Curriculum learning schedules
- Risk-adjusted reward shaping
- Performance-based dynamic rewards
- Exploration bonuses
- Skill acquisition rewards
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import math
import random
import statistics
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json

class RewardType(Enum):
    """Types of rewards in the system"""
    PROFIT = "profit"                    # Raw P&L reward
    RISK_ADJUSTED = "risk_adjusted"      # Sharpe-like risk adjustment
    EFFICIENCY = "efficiency"            # MAE/MFE efficiency reward
    CONSISTENCY = "consistency"          # Consistency bonus/penalty
    EXPLORATION = "exploration"          # Exploration bonus
    SKILL_ACQUISITION = "skill"          # Learning new skills
    DRAWDOWN_PENALTY = "drawdown"        # Drawdown penalty
    REGIME_ADAPTATION = "regime"         # Regime adaptation reward
    DIVERSIFICATION = "diversification"  # Portfolio diversification

@dataclass
class RewardComponent:
    """Individual reward component"""
    reward_type: RewardType
    base_value: float
    weight: float
    decay_factor: float = 1.0
    threshold: float = 0.0
    enabled: bool = True

@dataclass
class TradingState:
    """Current trading state for reward calculation"""
    # Market data
    symbol: str
    current_price: float
    volatility: float
    volume_percentile: float
    regime: str
    
    # Position data
    position_size: float
    entry_price: float
    unrealized_pnl: float
    time_in_position: int
    
    # Performance metrics
    mae: float
    mfe: float
    efficiency: float
    drawdown_from_peak: float
    
    # Historical context
    recent_pnl: List[float] = field(default_factory=list)
    recent_trades: int = 0
    win_streak: int = 0
    loss_streak: int = 0
    
    # Agent context
    agent_type: str = "SNIPER"
    confidence: float = 0.5
    exploration_level: float = 0.1

@dataclass
class RewardResult:
    """Complete reward breakdown"""
    total_reward: float
    components: Dict[str, float] = field(default_factory=dict)
    bonuses: Dict[str, float] = field(default_factory=dict)
    penalties: Dict[str, float] = field(default_factory=dict)
    meta_info: Dict[str, Any] = field(default_factory=dict)

class RewardShapingEngine:
    """
    Advanced reward shaping engine for RL trading agents
    
    Features:
    - Multi-objective reward functions with dynamic weights
    - Intrinsic motivation for exploration and skill development
    - Curriculum learning with progressive difficulty
    - Risk-adjusted rewards based on market conditions
    - Dynamic reward adaptation based on performance
    - Exploration bonuses for trying new strategies
    - Skill acquisition rewards for learning specific patterns
    """
    
    def __init__(self, config_file: Optional[str] = None):
        self.reward_components: Dict[RewardType, RewardComponent] = {}
        self.curriculum_stage = 0
        self.training_episodes = 0
        self.performance_history: List[float] = []
        
        # Reward shaping parameters
        self.risk_adjustment_factor = 1.0
        self.exploration_bonus_decay = 0.995
        self.skill_learning_rate = 0.1
        self.consistency_window = 20
        
        # Performance tracking
        self.recent_sharpe = 0.0
        self.recent_drawdown = 0.0
        self.skill_metrics: Dict[str, float] = {}
        
        # Initialize default reward components
        self._initialize_reward_components()
        
        if config_file:
            self.load_config(config_file)
        
        print("🎯 Reward Shaping Engine initialized")
        print(f"   📊 {len(self.reward_components)} reward components active")
        print(f"   🎓 Curriculum learning enabled")
        print(f"   🔍 Exploration bonuses configured")
    
    def _initialize_reward_components(self):
        """Initialize default reward components"""
        
        # Primary profit reward
        self.reward_components[RewardType.PROFIT] = RewardComponent(
            reward_type=RewardType.PROFIT,
            base_value=1.0,
            weight=1.0,
            threshold=0.0
        )
        
        # Risk-adjusted reward (Sharpe-like)
        self.reward_components[RewardType.RISK_ADJUSTED] = RewardComponent(
            reward_type=RewardType.RISK_ADJUSTED,
            base_value=0.5,
            weight=0.8,
            threshold=0.5
        )
        
        # Efficiency reward (MAE/MFE based)
        self.reward_components[RewardType.EFFICIENCY] = RewardComponent(
            reward_type=RewardType.EFFICIENCY,
            base_value=0.3,
            weight=0.6,
            threshold=1.0
        )
        
        # Consistency reward
        self.reward_components[RewardType.CONSISTENCY] = RewardComponent(
            reward_type=RewardType.CONSISTENCY,
            base_value=0.2,
            weight=0.4,
            threshold=0.6
        )
        
        # Exploration bonus
        self.reward_components[RewardType.EXPLORATION] = RewardComponent(
            reward_type=RewardType.EXPLORATION,
            base_value=0.1,
            weight=0.3,
            decay_factor=0.995
        )
        
        # Skill acquisition reward
        self.reward_components[RewardType.SKILL_ACQUISITION] = RewardComponent(
            reward_type=RewardType.SKILL_ACQUISITION,
            base_value=0.4,
            weight=0.5,
            threshold=0.1
        )
        
        # Drawdown penalty
        self.reward_components[RewardType.DRAWDOWN_PENALTY] = RewardComponent(
            reward_type=RewardType.DRAWDOWN_PENALTY,
            base_value=-2.0,
            weight=1.5,
            threshold=0.05  # 5% drawdown threshold
        )
        
        # Regime adaptation reward
        self.reward_components[RewardType.REGIME_ADAPTATION] = RewardComponent(
            reward_type=RewardType.REGIME_ADAPTATION,
            base_value=0.3,
            weight=0.4,
            threshold=0.0
        )
    
    def calculate_reward(self, state: TradingState, action_taken: str, 
                        trade_result: Optional[Dict] = None) -> RewardResult:
        """
        Calculate comprehensive reward with multiple components
        
        Args:
            state: Current trading state
            action_taken: Action taken by agent
            trade_result: Results if trade was completed
            
        Returns:
            RewardResult with breakdown of all reward components
        """
        
        result = RewardResult(total_reward=0.0)
        
        # Calculate each reward component
        for reward_type, component in self.reward_components.items():
            if not component.enabled:
                continue
            
            component_reward = self._calculate_component_reward(
                reward_type, component, state, action_taken, trade_result
            )
            
            result.components[reward_type.value] = component_reward
            result.total_reward += component_reward * component.weight
        
        # Apply curriculum learning adjustments
        result.total_reward = self._apply_curriculum_adjustment(result.total_reward, state)
        
        # Apply exploration bonuses
        exploration_bonus = self._calculate_exploration_bonus(state, action_taken)
        result.bonuses["exploration"] = exploration_bonus
        result.total_reward += exploration_bonus
        
        # Apply skill acquisition bonuses
        skill_bonus = self._calculate_skill_bonus(state, action_taken, trade_result)
        result.bonuses["skill_acquisition"] = skill_bonus
        result.total_reward += skill_bonus
        
        # Apply dynamic weight adjustments
        result.total_reward = self._apply_dynamic_adjustments(result.total_reward, state)
        
        # Update internal state
        self._update_internal_state(state, action_taken, result.total_reward)
        
        # Store meta information
        result.meta_info = {
            "curriculum_stage": self.curriculum_stage,
            "training_episode": self.training_episodes,
            "exploration_level": state.exploration_level,
            "risk_adjustment": self.risk_adjustment_factor,
            "recent_performance": statistics.mean(self.performance_history[-10:]) if self.performance_history else 0
        }
        
        return result
    
    def _calculate_component_reward(self, reward_type: RewardType, component: RewardComponent,
                                   state: TradingState, action: str, 
                                   trade_result: Optional[Dict]) -> float:
        """Calculate reward for specific component"""
        
        if reward_type == RewardType.PROFIT:
            return self._calculate_profit_reward(component, state, trade_result)
        
        elif reward_type == RewardType.RISK_ADJUSTED:
            return self._calculate_risk_adjusted_reward(component, state, trade_result)
        
        elif reward_type == RewardType.EFFICIENCY:
            return self._calculate_efficiency_reward(component, state, trade_result)
        
        elif reward_type == RewardType.CONSISTENCY:
            return self._calculate_consistency_reward(component, state, trade_result)
        
        elif reward_type == RewardType.DRAWDOWN_PENALTY:
            return self._calculate_drawdown_penalty(component, state, trade_result)
        
        elif reward_type == RewardType.REGIME_ADAPTATION:
            return self._calculate_regime_reward(component, state, action)
        
        else:
            return 0.0
    
    def _calculate_profit_reward(self, component: RewardComponent, state: TradingState,
                                trade_result: Optional[Dict]) -> float:
        """Calculate base profit reward"""
        
        if trade_result and 'net_pnl' in trade_result:
            # Completed trade - use actual P&L
            pnl = trade_result['net_pnl']
            # Normalize by position size and volatility
            normalized_pnl = pnl / (abs(state.position_size) * state.volatility + 1e-6)
            return component.base_value * normalized_pnl
        else:
            # Ongoing trade - use unrealized P&L with time decay
            time_decay = max(0.1, 1.0 - (state.time_in_position / 1000))
            normalized_unrealized = state.unrealized_pnl / (abs(state.position_size) * state.volatility + 1e-6)
            return component.base_value * normalized_unrealized * time_decay
    
    def _calculate_risk_adjusted_reward(self, component: RewardComponent, state: TradingState,
                                       trade_result: Optional[Dict]) -> float:
        """Calculate risk-adjusted reward (Sharpe-like)"""
        
        if not state.recent_pnl or len(state.recent_pnl) < 2:
            return 0.0
        
        # Calculate recent Sharpe ratio
        mean_return = statistics.mean(state.recent_pnl)
        std_return = statistics.stdev(state.recent_pnl)
        
        if std_return == 0:
            sharpe_like = 0.0
        else:
            sharpe_like = mean_return / std_return
        
        # Reward above threshold
        excess_sharpe = max(0, sharpe_like - component.threshold)
        return component.base_value * excess_sharpe
    
    def _calculate_efficiency_reward(self, component: RewardComponent, state: TradingState,
                                    trade_result: Optional[Dict]) -> float:
        """Calculate efficiency reward based on MAE/MFE"""
        
        if state.mae == 0:
            return 0.0
        
        efficiency = state.mfe / state.mae
        
        # Bonus for high efficiency
        if efficiency > component.threshold:
            efficiency_bonus = (efficiency - component.threshold) / component.threshold
            return component.base_value * efficiency_bonus
        else:
            # Penalty for low efficiency
            efficiency_penalty = (component.threshold - efficiency) / component.threshold
            return -component.base_value * 0.5 * efficiency_penalty
    
    def _calculate_consistency_reward(self, component: RewardComponent, state: TradingState,
                                     trade_result: Optional[Dict]) -> float:
        """Calculate consistency reward"""
        
        if len(state.recent_pnl) < self.consistency_window:
            return 0.0
        
        recent_returns = state.recent_pnl[-self.consistency_window:]
        
        # Calculate consistency metrics
        positive_periods = sum(1 for r in recent_returns if r > 0)
        consistency_ratio = positive_periods / len(recent_returns)
        
        # Calculate volatility of returns
        volatility_penalty = statistics.stdev(recent_returns) if len(recent_returns) > 1 else 0
        
        # Reward consistency above threshold
        if consistency_ratio > component.threshold:
            consistency_bonus = (consistency_ratio - component.threshold) * component.base_value
            volatility_adjustment = max(0.1, 1.0 - volatility_penalty / 100)
            return consistency_bonus * volatility_adjustment
        else:
            return -component.base_value * 0.3 * (component.threshold - consistency_ratio)
    
    def _calculate_drawdown_penalty(self, component: RewardComponent, state: TradingState,
                                   trade_result: Optional[Dict]) -> float:
        """Calculate drawdown penalty"""
        
        if state.drawdown_from_peak > component.threshold:
            # Exponential penalty for large drawdowns
            excess_dd = state.drawdown_from_peak - component.threshold
            penalty_multiplier = math.exp(excess_dd * 10) - 1
            return component.base_value * penalty_multiplier
        
        return 0.0
    
    def _calculate_regime_reward(self, component: RewardComponent, state: TradingState,
                                action: str) -> float:
        """Calculate reward for regime-appropriate actions"""
        
        # Define optimal actions per regime
        optimal_actions = {
            "CHAOTIC": ["BERSERKER_ENTER", "HIGH_VOLATILITY_STRATEGY"],
            "UNDERDAMPED": ["SNIPER_ENTER", "TREND_FOLLOWING"],
            "CRITICALLY_DAMPED": ["MIXED_STRATEGY", "MODERATE_POSITION"],
            "OVERDAMPED": ["WAIT", "NO_POSITION", "MARKET_MAKING"]
        }
        
        if state.regime in optimal_actions and action in optimal_actions[state.regime]:
            return component.base_value
        elif state.regime == "OVERDAMPED" and action in ["AGGRESSIVE_ENTRY", "LARGE_POSITION"]:
            return -component.base_value  # Penalty for poor regime matching
        
        return 0.0
    
    def _calculate_exploration_bonus(self, state: TradingState, action: str) -> float:
        """Calculate exploration bonus for trying new strategies"""
        
        exploration_component = self.reward_components.get(RewardType.EXPLORATION)
        if not exploration_component or not exploration_component.enabled:
            return 0.0
        
        # Base exploration bonus
        base_bonus = exploration_component.base_value * state.exploration_level
        
        # Bonus for exploring in appropriate conditions
        if state.regime == "CHAOTIC" and action in ["BERSERKER_ENTER", "HIGH_RISK_STRATEGY"]:
            base_bonus *= 2.0  # Double bonus for exploring in chaos
        
        # Apply decay over time
        decayed_bonus = base_bonus * (exploration_component.decay_factor ** self.training_episodes)
        
        return decayed_bonus
    
    def _calculate_skill_bonus(self, state: TradingState, action: str, 
                              trade_result: Optional[Dict]) -> float:
        """Calculate skill acquisition bonus"""
        
        skill_component = self.reward_components.get(RewardType.SKILL_ACQUISITION)
        if not skill_component or not skill_component.enabled:
            return 0.0
        
        skill_bonus = 0.0
        
        # Skill: High efficiency trading
        if state.efficiency > 2.0:
            skill_key = "high_efficiency"
            current_skill = self.skill_metrics.get(skill_key, 0.0)
            skill_improvement = min(0.1, state.efficiency / 10.0 - current_skill)
            if skill_improvement > 0:
                self.skill_metrics[skill_key] = current_skill + skill_improvement * self.skill_learning_rate
                skill_bonus += skill_component.base_value * skill_improvement
        
        # Skill: Regime adaptation
        if action.startswith(state.regime.lower()):  # Action matches regime
            skill_key = f"regime_{state.regime.lower()}"
            current_skill = self.skill_metrics.get(skill_key, 0.0)
            skill_improvement = min(0.05, 0.1 - current_skill)
            if skill_improvement > 0:
                self.skill_metrics[skill_key] = current_skill + skill_improvement * self.skill_learning_rate
                skill_bonus += skill_component.base_value * skill_improvement * 2
        
        # Skill: Drawdown control
        if state.drawdown_from_peak < 0.02:  # Low drawdown
            skill_key = "drawdown_control"
            current_skill = self.skill_metrics.get(skill_key, 0.0)
            skill_improvement = min(0.05, 0.1 - current_skill)
            if skill_improvement > 0:
                self.skill_metrics[skill_key] = current_skill + skill_improvement * self.skill_learning_rate
                skill_bonus += skill_component.base_value * skill_improvement
        
        return skill_bonus
    
    def _apply_curriculum_adjustment(self, base_reward: float, state: TradingState) -> float:
        """Apply curriculum learning adjustments"""
        
        # Stage 0: Focus on basic profitability
        if self.curriculum_stage == 0:
            if self.training_episodes > 1000 and statistics.mean(self.performance_history[-100:]) > 0:
                self.curriculum_stage = 1
                print(f"🎓 Advanced to curriculum stage 1: Risk management focus")
            return base_reward
        
        # Stage 1: Add risk management emphasis
        elif self.curriculum_stage == 1:
            risk_multiplier = 1.2 if state.drawdown_from_peak < 0.05 else 0.8
            if self.training_episodes > 2000 and self.recent_sharpe > 1.0:
                self.curriculum_stage = 2
                print(f"🎓 Advanced to curriculum stage 2: Consistency and efficiency focus")
            return base_reward * risk_multiplier
        
        # Stage 2: Focus on consistency and efficiency
        elif self.curriculum_stage == 2:
            efficiency_multiplier = 1.0 + (state.efficiency - 1.0) * 0.2
            consistency_multiplier = 1.0 + (len([r for r in state.recent_pnl[-20:] if r > 0]) / 20 - 0.5)
            return base_reward * efficiency_multiplier * consistency_multiplier
        
        return base_reward
    
    def _apply_dynamic_adjustments(self, reward: float, state: TradingState) -> float:
        """Apply dynamic reward adjustments based on market conditions"""
        
        # Market volatility adjustment
        vol_adjustment = 1.0
        if state.volatility > 200:  # High volatility
            vol_adjustment = 1.3 if state.agent_type == "BERSERKER" else 0.7
        elif state.volatility < 50:  # Low volatility  
            vol_adjustment = 0.8 if state.agent_type == "BERSERKER" else 1.1
        
        # Volume adjustment
        volume_adjustment = 1.0
        if state.volume_percentile > 80:
            volume_adjustment = 1.2  # High volume = better execution
        elif state.volume_percentile < 20:
            volume_adjustment = 0.9  # Low volume = worse execution
        
        # Performance-based adjustment
        perf_adjustment = 1.0
        if len(self.performance_history) > 50:
            recent_performance = statistics.mean(self.performance_history[-50:])
            if recent_performance > 10:  # Good recent performance
                perf_adjustment = 1.1
            elif recent_performance < -5:  # Poor recent performance
                perf_adjustment = 0.9
        
        return reward * vol_adjustment * volume_adjustment * perf_adjustment
    
    def _update_internal_state(self, state: TradingState, action: str, total_reward: float):
        """Update internal state for next reward calculation"""
        
        self.training_episodes += 1
        self.performance_history.append(total_reward)
        
        # Keep history manageable
        if len(self.performance_history) > 10000:
            self.performance_history = self.performance_history[-5000:]
        
        # Update recent metrics
        if len(self.performance_history) > 20:
            self.recent_sharpe = statistics.mean(self.performance_history[-20:]) / (
                statistics.stdev(self.performance_history[-20:]) + 1e-6)
        
        # Decay exploration bonus
        exploration_component = self.reward_components.get(RewardType.EXPLORATION)
        if exploration_component:
            exploration_component.base_value *= exploration_component.decay_factor
    
    def update_reward_weights(self, performance_feedback: Dict[str, float]):
        """Dynamically update reward component weights based on performance"""
        
        learning_rate = 0.01
        
        for component_name, performance_score in performance_feedback.items():
            for reward_type, component in self.reward_components.items():
                if reward_type.value == component_name:
                    # Increase weight if component led to good performance
                    if performance_score > 0.7:
                        component.weight *= (1 + learning_rate)
                    elif performance_score < 0.3:
                        component.weight *= (1 - learning_rate)
                    
                    # Keep weights in reasonable range
                    component.weight = max(0.1, min(2.0, component.weight))
    
    def get_reward_summary(self) -> Dict[str, Any]:
        """Get summary of current reward configuration"""
        
        return {
            "curriculum_stage": self.curriculum_stage,
            "training_episodes": self.training_episodes,
            "performance_history_length": len(self.performance_history),
            "recent_performance": statistics.mean(self.performance_history[-100:]) if self.performance_history else 0,
            "skill_metrics": dict(self.skill_metrics),
            "component_weights": {rt.value: comp.weight for rt, comp in self.reward_components.items()},
            "active_components": len([c for c in self.reward_components.values() if c.enabled])
        }
    
    def save_config(self, filename: str):
        """Save current reward configuration"""
        
        config = {
            "curriculum_stage": self.curriculum_stage,
            "training_episodes": self.training_episodes,
            "skill_metrics": self.skill_metrics,
            "reward_components": {
                rt.value: {
                    "base_value": comp.base_value,
                    "weight": comp.weight,
                    "decay_factor": comp.decay_factor,
                    "threshold": comp.threshold,
                    "enabled": comp.enabled
                }
                for rt, comp in self.reward_components.items()
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(config, f, indent=2)
    
    def load_config(self, filename: str):
        """Load reward configuration"""
        
        try:
            with open(filename, 'r') as f:
                config = json.load(f)
            
            self.curriculum_stage = config.get("curriculum_stage", 0)
            self.training_episodes = config.get("training_episodes", 0)
            self.skill_metrics = config.get("skill_metrics", {})
            
            # Update component configurations
            component_configs = config.get("reward_components", {})
            for component_name, component_config in component_configs.items():
                for rt in RewardType:
                    if rt.value == component_name and rt in self.reward_components:
                        comp = self.reward_components[rt]
                        comp.base_value = component_config.get("base_value", comp.base_value)
                        comp.weight = component_config.get("weight", comp.weight)
                        comp.decay_factor = component_config.get("decay_factor", comp.decay_factor)
                        comp.threshold = component_config.get("threshold", comp.threshold)
                        comp.enabled = component_config.get("enabled", comp.enabled)
            
            print(f"✅ Loaded reward configuration from {filename}")
            
        except Exception as e:
            print(f"⚠️ Error loading config: {e}")

def test_reward_shaping_engine():
    """Test the reward shaping engine"""
    
    print("🧪 REWARD SHAPING ENGINE TEST")
    print("="*80)
    
    # Initialize engine
    engine = RewardShapingEngine()
    
    # Create test trading state
    test_state = TradingState(
        symbol="BTCUSD",
        current_price=95000.0,
        volatility=150.0,
        volume_percentile=75.0,
        regime="CHAOTIC",
        position_size=1.0,
        entry_price=94500.0,
        unrealized_pnl=500.0,
        time_in_position=24,
        mae=200.0,
        mfe=800.0,
        efficiency=4.0,
        drawdown_from_peak=0.02,
        recent_pnl=[50.0, -20.0, 120.0, 80.0, -10.0, 200.0],
        recent_trades=6,
        agent_type="BERSERKER",
        confidence=0.8,
        exploration_level=0.3
    )
    
    # Test different actions
    test_actions = [
        "BERSERKER_ENTER",
        "SNIPER_ENTER", 
        "WAIT",
        "EXIT_POSITION"
    ]
    
    print(f"\n🎯 TESTING REWARD CALCULATIONS:")
    print(f"State: {test_state.symbol} in {test_state.regime} regime")
    print(f"Position: {test_state.position_size} @ {test_state.entry_price}")
    print(f"MAE: {test_state.mae:.1f}, MFE: {test_state.mfe:.1f}, Efficiency: {test_state.efficiency:.1f}")
    
    for action in test_actions:
        print(f"\n{'='*50}")
        print(f"ACTION: {action}")
        print(f"{'='*50}")
        
        # Calculate reward
        reward_result = engine.calculate_reward(test_state, action)
        
        print(f"🎁 TOTAL REWARD: {reward_result.total_reward:.3f}")
        
        print(f"\n📊 COMPONENT BREAKDOWN:")
        for component, value in reward_result.components.items():
            print(f"   {component}: {value:.3f}")
        
        print(f"\n🎉 BONUSES:")
        for bonus, value in reward_result.bonuses.items():
            print(f"   {bonus}: {value:.3f}")
        
        print(f"\n📋 META INFO:")
        for key, value in reward_result.meta_info.items():
            print(f"   {key}: {value}")
    
    # Test curriculum advancement
    print(f"\n📚 CURRICULUM LEARNING:")
    print(f"Current stage: {engine.curriculum_stage}")
    print(f"Training episodes: {engine.training_episodes}")
    
    # Simulate learning progress
    for i in range(100):
        engine.performance_history.append(random.gauss(5.0, 2.0))  # Positive trend
        engine.training_episodes += 1
    
    # Test reward with improved performance
    reward_result = engine.calculate_reward(test_state, "BERSERKER_ENTER")
    print(f"After 100 episodes: Total reward = {reward_result.total_reward:.3f}")
    print(f"Curriculum stage: {engine.curriculum_stage}")
    
    # Test configuration save/load
    config_file = "/tmp/reward_config_test.json"
    engine.save_config(config_file)
    
    # Get summary
    summary = engine.get_reward_summary()
    print(f"\n📈 REWARD ENGINE SUMMARY:")
    for key, value in summary.items():
        if key != "skill_metrics":
            print(f"   {key}: {value}")
    
    print(f"\n🎨 SKILL METRICS:")
    for skill, level in summary["skill_metrics"].items():
        print(f"   {skill}: {level:.3f}")
    
    print(f"\n✅ Reward shaping engine test completed!")

if __name__ == "__main__":
    test_reward_shaping_engine()