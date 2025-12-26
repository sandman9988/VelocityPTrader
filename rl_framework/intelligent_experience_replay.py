#!/usr/bin/env python3
"""
Intelligent Experience Replay System
Advanced ML/RL-based learning focused on journey efficiency and trade quality

Key Features:
- Asymmetrical reward shaping based on trade journey quality
- Intelligent experience prioritization for faster learning
- Dynamic reward adjustment based on efficiency patterns
- Journey-focused learning that values the path to profit, not just profit
- Exploration-exploitation balance with efficiency bias
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent))

import math
import random
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from collections import deque, defaultdict
from enum import Enum
import heapq

@dataclass
class TradeJourney:
    """Represents the complete journey of a trade with efficiency metrics"""
    trade_id: str
    symbol: str
    agent_type: str
    
    # Journey states (sequence of market states during trade)
    entry_state: Dict[str, float]
    intermediate_states: List[Dict[str, float]]
    exit_state: Dict[str, float]
    
    # Journey metrics
    duration_steps: int
    total_pnl: float
    max_adverse_excursion: float
    max_favorable_excursion: float
    
    # Actions taken during journey
    entry_action: int
    intermediate_actions: List[int]
    exit_action: int
    
    # Context information
    market_regime: str
    volatility_percentile: float
    volume_percentile: float
    
    # Journey efficiency scores
    efficiency_score: float = 0.0
    learning_value: float = 0.0
    exploration_reward: float = 0.0
    
    # Learning metadata
    replay_count: int = 0
    last_replay_time: datetime = field(default_factory=datetime.now)
    priority_score: float = 0.0

@dataclass
class EfficiencyPattern:
    """Pattern representing efficient trade characteristics"""
    mae_to_pnl_ratio: float
    mfe_to_pnl_ratio: float
    duration_efficiency: float
    action_sequence_quality: float
    regime_appropriateness: float
    
    def calculate_overall_score(self) -> float:
        """Calculate overall efficiency score"""
        weights = {
            'mae_pnl': 0.3,    # How well adverse moves were managed
            'mfe_pnl': 0.25,   # How well favorable moves were captured  
            'duration': 0.2,   # Time efficiency
            'actions': 0.15,   # Action sequence quality
            'regime': 0.1      # Regime appropriateness
        }
        
        return (
            weights['mae_pnl'] * (1.0 / max(0.01, self.mae_to_pnl_ratio)) +
            weights['mfe_pnl'] * self.mfe_to_pnl_ratio +
            weights['duration'] * self.duration_efficiency +
            weights['actions'] * self.action_sequence_quality +
            weights['regime'] * self.regime_appropriateness
        )

class JourneyEfficiencyAnalyzer:
    """Analyzes trade journeys to identify efficiency patterns"""
    
    def __init__(self):
        self.efficiency_patterns: Dict[str, List[EfficiencyPattern]] = {}
        self.journey_database: List[TradeJourney] = []
        self.pattern_thresholds = {
            'excellent': 0.8,
            'good': 0.6, 
            'poor': 0.3
        }
        
    def analyze_journey_efficiency(self, journey: TradeJourney) -> EfficiencyPattern:
        """Analyze the efficiency of a complete trade journey"""
        
        # Calculate MAE to P&L ratio (lower is better)
        mae_ratio = journey.max_adverse_excursion / max(abs(journey.total_pnl), 1.0)
        
        # Calculate MFE to P&L ratio (higher is better for winners)
        mfe_ratio = journey.max_favorable_excursion / max(abs(journey.total_pnl), 1.0)
        
        # Duration efficiency (faster profitable exits are better)
        expected_duration = self._get_expected_duration(journey.symbol, journey.market_regime)
        duration_efficiency = max(0.1, expected_duration / max(journey.duration_steps, 1))
        
        # Action sequence quality
        action_quality = self._evaluate_action_sequence(journey)
        
        # Regime appropriateness
        regime_score = self._evaluate_regime_appropriateness(journey)
        
        pattern = EfficiencyPattern(
            mae_to_pnl_ratio=mae_ratio,
            mfe_to_pnl_ratio=mfe_ratio,
            duration_efficiency=min(2.0, duration_efficiency),
            action_sequence_quality=action_quality,
            regime_appropriateness=regime_score
        )
        
        journey.efficiency_score = pattern.calculate_overall_score()
        return pattern
    
    def _get_expected_duration(self, symbol: str, regime: str) -> float:
        """Get expected trade duration for symbol/regime combination"""
        
        base_durations = {
            'CHAOTIC': 30,      # Fast moves in chaos
            'UNDERDAMPED': 60,  # Medium trending moves
            'CRITICALLY_DAMPED': 90,  # Slower moves
            'OVERDAMPED': 120   # Very slow moves
        }
        
        symbol_multipliers = {
            'BTCUSD': 0.8,   # Crypto moves faster
            'ETHUSD': 0.8,
            'XAUUSD': 1.1,   # Gold slightly slower
            'EURUSD': 1.3,   # Forex slower
            'GBPUSD': 1.3
        }
        
        base = base_durations.get(regime, 60)
        multiplier = symbol_multipliers.get(symbol, 1.0)
        return base * multiplier
    
    def _evaluate_action_sequence(self, journey: TradeJourney) -> float:
        """Evaluate quality of action sequence during journey"""
        
        if not journey.intermediate_actions:
            return 0.5  # Neutral for immediate exit
        
        # Count beneficial actions vs detrimental ones
        beneficial_count = 0
        total_actions = len(journey.intermediate_actions)
        
        for i, action in enumerate(journey.intermediate_actions):
            # Evaluate if action was beneficial at that point
            if self._is_action_beneficial(action, journey, i):
                beneficial_count += 1
        
        return beneficial_count / max(total_actions, 1)
    
    def _is_action_beneficial(self, action: int, journey: TradeJourney, step: int) -> bool:
        """Determine if an action was beneficial at a specific step"""
        
        # Simplified evaluation - in practice would use more sophisticated logic
        if step < len(journey.intermediate_states):
            state = journey.intermediate_states[step]
            unrealized_pnl = state.get('unrealized_pnl', 0)
            
            # Beneficial actions:
            # - Taking profit when ahead
            # - Cutting losses when behind
            # - Position sizing adjustments based on momentum
            
            if action in [7, 8] and unrealized_pnl > 0:  # Exit/reduce when profitable
                return True
            elif action in [7, 8] and unrealized_pnl < -50:  # Exit when losing badly
                return True
            elif action == 0 and abs(unrealized_pnl) < 20:  # Wait when small moves
                return True
        
        return False
    
    def _evaluate_regime_appropriateness(self, journey: TradeJourney) -> float:
        """Evaluate how appropriate the trade was for the market regime"""
        
        regime_agent_scores = {
            'CHAOTIC': {'BERSERKER': 1.0, 'SNIPER': 0.6},
            'UNDERDAMPED': {'BERSERKER': 0.8, 'SNIPER': 0.9},
            'CRITICALLY_DAMPED': {'BERSERKER': 0.5, 'SNIPER': 0.8},
            'OVERDAMPED': {'BERSERKER': 0.3, 'SNIPER': 0.6}
        }
        
        return regime_agent_scores.get(journey.market_regime, {}).get(journey.agent_type, 0.5)

class AsymmetricalRewardShaper:
    """
    Advanced reward shaping that emphasizes journey efficiency over just profit
    Uses asymmetrical rewards that heavily weight learning and exploration
    """
    
    def __init__(self):
        self.efficiency_analyzer = JourneyEfficiencyAnalyzer()
        self.learning_curves: Dict[str, List[float]] = defaultdict(list)
        self.exploration_bonus_decay = 0.998
        self.current_exploration_rate = 1.0
        
        # Asymmetrical reward weights
        self.reward_weights = {
            'profit': 0.3,           # Base profit component
            'efficiency': 0.35,      # Journey efficiency (most important)
            'exploration': 0.15,     # Exploration bonus
            'learning_value': 0.2    # How much this experience teaches us
        }
        
    def shape_reward(self, journey: TradeJourney, raw_pnl: float) -> Dict[str, float]:
        """
        Shape reward based on journey efficiency and learning value
        Returns breakdown of reward components
        """
        
        # Analyze journey efficiency
        efficiency_pattern = self.efficiency_analyzer.analyze_journey_efficiency(journey)
        
        # Calculate base profit component (normalized)
        profit_component = self._calculate_profit_component(raw_pnl, journey)
        
        # Calculate efficiency component (asymmetrically weighted)
        efficiency_component = self._calculate_efficiency_component(efficiency_pattern, journey)
        
        # Calculate exploration component
        exploration_component = self._calculate_exploration_component(journey)
        
        # Calculate learning value component
        learning_component = self._calculate_learning_value(journey)
        
        # Combine components with asymmetrical weighting
        total_reward = (
            self.reward_weights['profit'] * profit_component +
            self.reward_weights['efficiency'] * efficiency_component +
            self.reward_weights['exploration'] * exploration_component +
            self.reward_weights['learning_value'] * learning_component
        )
        
        # Apply asymmetrical scaling for efficient vs inefficient trades
        if efficiency_pattern.calculate_overall_score() > 0.7:
            total_reward *= 1.5  # Bonus for highly efficient trades
        elif efficiency_pattern.calculate_overall_score() < 0.3:
            total_reward *= 0.5  # Penalty for highly inefficient trades
        
        return {
            'total_reward': total_reward,
            'profit_component': profit_component,
            'efficiency_component': efficiency_component,
            'exploration_component': exploration_component,
            'learning_component': learning_component,
            'efficiency_multiplier': 1.5 if efficiency_pattern.calculate_overall_score() > 0.7 else 1.0
        }
    
    def _calculate_profit_component(self, raw_pnl: float, journey: TradeJourney) -> float:
        """Calculate profit component with diminishing returns for large wins"""
        
        # Normalize P&L by symbol typical range
        symbol_normalizers = {
            'BTCUSD': 10000,  # High volatility
            'ETHUSD': 5000,
            'XAUUSD': 500,
            'EURUSD': 100,    # Lower volatility
            'GBPUSD': 100
        }
        
        normalizer = symbol_normalizers.get(journey.symbol, 1000)
        normalized_pnl = raw_pnl / normalizer
        
        # Apply diminishing returns to prevent overfitting on large wins
        if normalized_pnl > 0:
            return math.tanh(normalized_pnl / 2.0)  # Caps at ~1.0
        else:
            return -math.sqrt(abs(normalized_pnl))  # Linear penalty for losses
    
    def _calculate_efficiency_component(self, pattern: EfficiencyPattern, journey: TradeJourney) -> float:
        """Calculate efficiency component - most important for learning"""
        
        base_efficiency = pattern.calculate_overall_score()
        
        # Bonus for improvement over agent's historical efficiency
        agent_key = f"{journey.symbol}_{journey.agent_type}"
        historical_efficiency = self.learning_curves.get(agent_key, [0.4])
        
        if historical_efficiency:
            avg_historical = statistics.mean(historical_efficiency[-20:])  # Recent average
            improvement_bonus = max(0, base_efficiency - avg_historical) * 2.0
        else:
            improvement_bonus = 0
        
        # Store current efficiency for future comparisons
        self.learning_curves[agent_key].append(base_efficiency)
        
        return base_efficiency + improvement_bonus
    
    def _calculate_exploration_component(self, journey: TradeJourney) -> float:
        """Calculate exploration bonus with intelligent decay"""
        
        # Base exploration bonus
        base_bonus = self.current_exploration_rate * 0.1
        
        # Bonus for exploring new regime/agent combinations
        combo_key = f"{journey.market_regime}_{journey.agent_type}_{journey.symbol}"
        exploration_history = getattr(self, '_exploration_history', set())
        
        if combo_key not in exploration_history:
            exploration_history.add(combo_key)
            self._exploration_history = exploration_history
            base_bonus *= 2.0  # Double bonus for new combinations
        
        # Bonus for exploring during high uncertainty periods
        if journey.volatility_percentile > 80 or journey.volatility_percentile < 20:
            base_bonus *= 1.3  # Bonus for extreme conditions
        
        # Decay exploration rate over time
        self.current_exploration_rate *= self.exploration_bonus_decay
        
        return base_bonus
    
    def _calculate_learning_value(self, journey: TradeJourney) -> float:
        """Calculate how valuable this experience is for learning"""
        
        learning_value = 0.0
        
        # High learning value for:
        # 1. Surprising outcomes (high efficiency with loss, or low efficiency with win)
        efficiency_score = journey.efficiency_score
        pnl_positive = journey.total_pnl > 0
        
        if pnl_positive and efficiency_score > 0.8:
            learning_value += 0.8  # Good example to reinforce
        elif not pnl_positive and efficiency_score > 0.6:
            learning_value += 1.0  # Important lesson: good process, bad outcome
        elif pnl_positive and efficiency_score < 0.4:
            learning_value += 0.9  # Important lesson: bad process, lucky outcome
        elif not pnl_positive and efficiency_score < 0.3:
            learning_value += 0.5  # Clear negative example
        
        # 2. Rare market conditions
        if journey.market_regime in ['CHAOTIC', 'OVERDAMPED']:
            learning_value += 0.2  # Rare conditions are valuable to learn from
        
        # 3. Edge cases in MAE/MFE ratios
        mae_ratio = journey.max_adverse_excursion / max(abs(journey.total_pnl), 1.0)
        mfe_ratio = journey.max_favorable_excursion / max(abs(journey.total_pnl), 1.0)
        
        if mae_ratio > 3.0 or mfe_ratio > 5.0:  # Extreme ratios
            learning_value += 0.3
        
        return min(1.0, learning_value)

class IntelligentExperienceReplay:
    """
    Intelligent experience replay system that prioritizes learning experiences
    based on journey efficiency and learning value
    """
    
    def __init__(self, capacity: int = 50000):
        self.capacity = capacity
        self.experiences: List[TradeJourney] = []
        self.priority_heap: List[Tuple[float, int]] = []  # (negative_priority, index)
        self.reward_shaper = AsymmetricalRewardShaper()
        
        # Learning statistics
        self.replay_counts: Dict[str, int] = defaultdict(int)
        self.learning_progress: Dict[str, List[float]] = defaultdict(list)
        
    def add_experience(self, journey: TradeJourney):
        """Add new trading experience to replay buffer"""
        
        # Shape rewards for this journey
        reward_breakdown = self.reward_shaper.shape_reward(journey, journey.total_pnl)
        journey.learning_value = reward_breakdown['learning_component']
        
        # Calculate priority based on learning value and efficiency
        priority = self._calculate_experience_priority(journey, reward_breakdown)
        journey.priority_score = priority
        
        # Add to buffer
        if len(self.experiences) < self.capacity:
            self.experiences.append(journey)
            heapq.heappush(self.priority_heap, (-priority, len(self.experiences) - 1))
        else:
            # Replace lowest priority experience
            if self.priority_heap:
                _, lowest_idx = heapq.heappop(self.priority_heap)
                self.experiences[lowest_idx] = journey
                heapq.heappush(self.priority_heap, (-priority, lowest_idx))
        
        print(f"📚 Added experience: {journey.trade_id}")
        print(f"   ⚡ Efficiency: {journey.efficiency_score:.3f}")
        print(f"   🎯 Learning Value: {journey.learning_value:.3f}")
        print(f"   🏆 Priority: {priority:.3f}")
    
    def _calculate_experience_priority(self, journey: TradeJourney, reward_breakdown: Dict[str, float]) -> float:
        """Calculate priority for experience replay"""
        
        priority = 0.0
        
        # Base priority from learning value
        priority += reward_breakdown['learning_component'] * 0.4
        
        # Priority from efficiency (both high and low efficiency are valuable)
        eff_score = journey.efficiency_score
        if eff_score > 0.8 or eff_score < 0.3:  # Extreme cases
            priority += 0.3
        else:
            priority += 0.1
        
        # Priority from rarity
        combo_key = f"{journey.market_regime}_{journey.agent_type}_{journey.symbol}"
        rarity_bonus = 1.0 / max(1, self.replay_counts.get(combo_key, 0) + 1)
        priority += rarity_bonus * 0.2
        
        # Recency bonus (newer experiences get slight priority)
        time_diff = (datetime.now() - journey.last_replay_time).total_seconds() / 3600  # Hours
        recency_bonus = min(0.1, time_diff / 24.0)  # Max 0.1 for day-old experiences
        priority += recency_bonus
        
        return priority
    
    def sample_for_learning(self, batch_size: int = 32, focus_type: str = 'mixed') -> List[TradeJourney]:
        """
        Sample experiences for learning with intelligent prioritization
        
        focus_type: 'efficient', 'inefficient', 'mixed', 'rare'
        """
        
        if len(self.experiences) < batch_size:
            return self.experiences.copy()
        
        sampled = []
        
        if focus_type == 'efficient':
            # Focus on highly efficient trades to reinforce good patterns
            candidates = [j for j in self.experiences if j.efficiency_score > 0.7]
            sampled = random.sample(candidates, min(batch_size, len(candidates)))
            
        elif focus_type == 'inefficient':
            # Focus on inefficient trades to learn what to avoid
            candidates = [j for j in self.experiences if j.efficiency_score < 0.4]
            sampled = random.sample(candidates, min(batch_size, len(candidates)))
            
        elif focus_type == 'rare':
            # Focus on rare market conditions
            candidates = [j for j in self.experiences if j.market_regime in ['CHAOTIC', 'OVERDAMPED']]
            sampled = random.sample(candidates, min(batch_size, len(candidates)))
            
        else:  # mixed
            # Use priority-based sampling with some randomness
            high_priority = int(batch_size * 0.6)  # 60% high priority
            random_sample = batch_size - high_priority
            
            # Sample high priority experiences
            priority_candidates = sorted(self.experiences, key=lambda x: x.priority_score, reverse=True)
            sampled.extend(priority_candidates[:high_priority])
            
            # Add some random experiences for diversity
            remaining = [j for j in self.experiences if j not in sampled]
            if remaining:
                sampled.extend(random.sample(remaining, min(random_sample, len(remaining))))
        
        # Update replay counts
        for journey in sampled:
            combo_key = f"{journey.market_regime}_{journey.agent_type}_{journey.symbol}"
            self.replay_counts[combo_key] += 1
            journey.replay_count += 1
            journey.last_replay_time = datetime.now()
        
        return sampled
    
    def get_learning_insights(self) -> Dict[str, Any]:
        """Get insights about what the system is learning"""
        
        if not self.experiences:
            return {}
        
        # Analyze efficiency trends
        efficiency_scores = [j.efficiency_score for j in self.experiences]
        
        # Most/least efficient patterns
        most_efficient = max(self.experiences, key=lambda x: x.efficiency_score)
        least_efficient = min(self.experiences, key=lambda x: x.efficiency_score)
        
        # Learning progress by agent/symbol combination
        learning_progress = {}
        for journey in self.experiences:
            key = f"{journey.agent_type}_{journey.symbol}"
            if key not in learning_progress:
                learning_progress[key] = []
            learning_progress[key].append(journey.efficiency_score)
        
        # Calculate trends
        trends = {}
        for key, scores in learning_progress.items():
            if len(scores) > 5:
                recent_avg = statistics.mean(scores[-5:])
                older_avg = statistics.mean(scores[:-5])
                trends[key] = recent_avg - older_avg
        
        return {
            'total_experiences': len(self.experiences),
            'avg_efficiency': statistics.mean(efficiency_scores),
            'efficiency_std': statistics.stdev(efficiency_scores) if len(efficiency_scores) > 1 else 0,
            'most_efficient_pattern': {
                'trade_id': most_efficient.trade_id,
                'efficiency': most_efficient.efficiency_score,
                'agent': most_efficient.agent_type,
                'regime': most_efficient.market_regime,
                'symbol': most_efficient.symbol
            },
            'least_efficient_pattern': {
                'trade_id': least_efficient.trade_id,
                'efficiency': least_efficient.efficiency_score,
                'agent': least_efficient.agent_type,
                'regime': least_efficient.market_regime,
                'symbol': least_efficient.symbol
            },
            'learning_trends': trends,
            'replay_statistics': dict(self.replay_counts)
        }

def demonstrate_intelligent_replay():
    """Demonstrate the intelligent experience replay system"""
    
    print("🧠 INTELLIGENT EXPERIENCE REPLAY DEMONSTRATION")
    print("=" * 80)
    
    # Initialize replay system
    replay_system = IntelligentExperienceReplay(capacity=1000)
    
    # Create sample trading experiences
    sample_experiences = [
        # Highly efficient winning trade
        TradeJourney(
            trade_id="EFFICIENT_WIN_001",
            symbol="BTCUSD",
            agent_type="BERSERKER",
            entry_state={'price': 95000, 'volatility': 120, 'momentum': 15},
            intermediate_states=[
                {'unrealized_pnl': 50, 'mae': 10, 'mfe': 100},
                {'unrealized_pnl': 150, 'mae': 10, 'mfe': 200}
            ],
            exit_state={'price': 97000, 'final_pnl': 200},
            duration_steps=25,
            total_pnl=20000,
            max_adverse_excursion=500,
            max_favorable_excursion=25000,
            entry_action=2,
            intermediate_actions=[0, 8],  # Wait, then reduce
            exit_action=7,
            market_regime="CHAOTIC",
            volatility_percentile=85,
            volume_percentile=75
        ),
        
        # Inefficient losing trade (important for learning)
        TradeJourney(
            trade_id="INEFFICIENT_LOSS_001",
            symbol="EURUSD",
            agent_type="BERSERKER",  # Wrong agent for regime
            entry_state={'price': 1.0500, 'volatility': 30, 'momentum': 2},
            intermediate_states=[
                {'unrealized_pnl': -10, 'mae': 25, 'mfe': 5},
                {'unrealized_pnl': -45, 'mae': 60, 'mfe': 5},
                {'unrealized_pnl': -80, 'mae': 95, 'mfe': 5}
            ],
            exit_state={'price': 1.0420, 'final_pnl': -80},
            duration_steps=120,  # Held too long
            total_pnl=-8000,
            max_adverse_excursion=9500,
            max_favorable_excursion=500,
            entry_action=3,  # Large position
            intermediate_actions=[0, 0, 0, 9],  # Wait too long, then increase
            exit_action=7,
            market_regime="OVERDAMPED",  # Bad regime for BERSERKER
            volatility_percentile=15,
            volume_percentile=25
        ),
        
        # Lucky win with poor process
        TradeJourney(
            trade_id="LUCKY_WIN_001",
            symbol="XAUUSD",
            agent_type="SNIPER",
            entry_state={'price': 2000, 'volatility': 60, 'momentum': 8},
            intermediate_states=[
                {'unrealized_pnl': -30, 'mae': 40, 'mfe': 10},
                {'unrealized_pnl': -60, 'mae': 80, 'mfe': 10},
                {'unrealized_pnl': 50, 'mae': 80, 'mfe': 120}  # Lucky reversal
            ],
            exit_state={'price': 2050, 'final_pnl': 50},
            duration_steps=180,  # Very long hold
            total_pnl=5000,
            max_adverse_excursion=8000,
            max_favorable_excursion=12000,
            entry_action=1,
            intermediate_actions=[0, 0, 0, 0, 0],  # Just waited
            exit_action=7,
            market_regime="CRITICALLY_DAMPED",
            volatility_percentile=55,
            volume_percentile=40
        )
    ]
    
    # Add experiences to replay system
    print("\n📚 Adding sample experiences...")
    for experience in sample_experiences:
        replay_system.add_experience(experience)
    
    # Sample for different learning focuses
    print("\n🎯 SAMPLING FOR DIFFERENT LEARNING FOCUSES:")
    
    focus_types = ['efficient', 'inefficient', 'mixed', 'rare']
    for focus in focus_types:
        print(f"\n{focus.upper()} Focus:")
        samples = replay_system.sample_for_learning(batch_size=2, focus_type=focus)
        for sample in samples:
            print(f"   📊 {sample.trade_id}: Efficiency={sample.efficiency_score:.3f}, "
                  f"P&L={sample.total_pnl:+.0f}, Agent={sample.agent_type}, "
                  f"Regime={sample.market_regime}")
    
    # Get learning insights
    print(f"\n🔍 LEARNING INSIGHTS:")
    insights = replay_system.get_learning_insights()
    
    for key, value in insights.items():
        if key not in ['most_efficient_pattern', 'least_efficient_pattern', 'learning_trends']:
            print(f"   {key}: {value}")
    
    print(f"\n🏆 Most Efficient Pattern:")
    most_eff = insights['most_efficient_pattern']
    print(f"   {most_eff['trade_id']}: {most_eff['efficiency']:.3f} efficiency")
    print(f"   Agent: {most_eff['agent']}, Regime: {most_eff['regime']}, Symbol: {most_eff['symbol']}")
    
    print(f"\n❌ Least Efficient Pattern:")
    least_eff = insights['least_efficient_pattern']
    print(f"   {least_eff['trade_id']}: {least_eff['efficiency']:.3f} efficiency")
    print(f"   Agent: {least_eff['agent']}, Regime: {least_eff['regime']}, Symbol: {least_eff['symbol']}")
    
    print(f"\n✅ Intelligent Experience Replay system demonstrated!")
    print(f"🎯 Key Features:")
    print(f"   • Asymmetrical reward shaping emphasizing journey efficiency")
    print(f"   • Intelligent prioritization of learning experiences") 
    print(f"   • Dynamic sampling based on learning focus")
    print(f"   • Continuous learning progress tracking")
    
    return replay_system

if __name__ == "__main__":
    demonstrate_intelligent_replay()