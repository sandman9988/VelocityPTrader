#!/usr/bin/env python3
"""
RL Learning Integration System
Integrates intelligent experience replay with the existing trading system
for continuous learning and optimization

Features:
- Real-time trade journey capture
- Continuous learning from trade outcomes  
- Adaptive strategy optimization
- Journey efficiency feedback loops
- Dynamic agent allocation based on learning
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent))

import json
import time
import random
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict, deque

# Import our intelligent replay system
from intelligent_experience_replay import (
    IntelligentExperienceReplay, 
    TradeJourney, 
    AsymmetricalRewardShaper,
    JourneyEfficiencyAnalyzer
)

@dataclass
class LiveTradingState:
    """Real-time trading state for learning"""
    symbol: str
    agent_type: str
    entry_time: datetime
    entry_price: float
    entry_action: int
    position_size: float
    
    # Running metrics
    current_price: float
    unrealized_pnl: float
    max_adverse_excursion: float = 0.0
    max_favorable_excursion: float = 0.0
    
    # Market context
    market_regime: str = "UNDERDAMPED"
    volatility_percentile: float = 50.0
    volume_percentile: float = 50.0
    
    # Journey tracking
    state_history: List[Dict[str, float]] = field(default_factory=list)
    action_history: List[int] = field(default_factory=list)
    
    # Learning metadata
    is_exploration_trade: bool = False
    confidence_level: float = 0.5

class RLLearningEngine:
    """
    Main engine that integrates RL learning with live trading
    Continuously learns from trade outcomes and optimizes strategies
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        
        # Initialize intelligent replay system
        self.replay_system = IntelligentExperienceReplay(capacity=10000)
        self.efficiency_analyzer = JourneyEfficiencyAnalyzer()
        
        # Live trading state tracking
        self.active_trades: Dict[str, LiveTradingState] = {}
        self.completed_journeys: List[TradeJourney] = []
        
        # Learning progress tracking
        self.learning_stats = {
            'total_journeys_learned': 0,
            'avg_efficiency_trend': deque(maxlen=100),
            'agent_performance_trends': defaultdict(lambda: deque(maxlen=50)),
            'regime_success_rates': defaultdict(float),
            'last_learning_session': None
        }
        
        # Adaptive strategy parameters
        self.strategy_weights = {
            'BERSERKER': {'CHAOTIC': 1.0, 'UNDERDAMPED': 0.8, 'CRITICALLY_DAMPED': 0.5, 'OVERDAMPED': 0.3},
            'SNIPER': {'CHAOTIC': 0.6, 'UNDERDAMPED': 0.9, 'CRITICALLY_DAMPED': 0.8, 'OVERDAMPED': 0.6}
        }
        
        # Exploration strategy
        self.exploration_rate = 0.15  # 15% of trades for exploration
        self.exploration_decay = 0.999
        
        print("🧠 RL Learning Engine initialized")
        print(f"   📚 Experience replay capacity: {self.replay_system.capacity:,}")
        print(f"   🎯 Initial exploration rate: {self.exploration_rate:.1%}")
        print(f"   ⚡ Journey efficiency focus enabled")
    
    def on_trade_entry(self, trade_entry: Dict[str, Any]) -> Dict[str, Any]:
        """
        Called when a new trade is entered
        Returns: Learning-based modifications to trade parameters
        """
        
        symbol = trade_entry['symbol']
        agent_type = trade_entry['agent_type']
        market_regime = trade_entry.get('market_regime', 'UNDERDAMPED')
        
        # Generate unique trade ID
        trade_id = f"{symbol}_{agent_type}_{int(time.time())}"
        
        # Determine if this should be an exploration trade
        is_exploration = self._should_explore(symbol, agent_type, market_regime)
        
        # Create live trading state
        live_state = LiveTradingState(
            symbol=symbol,
            agent_type=agent_type,
            entry_time=datetime.now(),
            entry_price=trade_entry['entry_price'],
            entry_action=trade_entry['action'],
            position_size=trade_entry['position_size'],
            current_price=trade_entry['entry_price'],
            unrealized_pnl=0.0,
            market_regime=market_regime,
            volatility_percentile=trade_entry.get('volatility_percentile', 50),
            volume_percentile=trade_entry.get('volume_percentile', 50),
            is_exploration_trade=is_exploration,
            confidence_level=self._calculate_trade_confidence(symbol, agent_type, market_regime)
        )
        
        # Store active trade
        self.active_trades[trade_id] = live_state
        
        # Get learning-based trade modifications
        modifications = self._get_learning_based_modifications(live_state)
        
        print(f"📈 Trade entered: {trade_id}")
        print(f"   🎯 Confidence: {live_state.confidence_level:.2f}")
        print(f"   🔍 Exploration: {is_exploration}")
        if modifications:
            print(f"   🔧 Modifications: {modifications}")
        
        return {
            'trade_id': trade_id,
            'modifications': modifications,
            'is_exploration': is_exploration,
            'confidence': live_state.confidence_level
        }
    
    def on_trade_update(self, trade_id: str, market_update: Dict[str, Any]):
        """Called on each market update for active trades"""
        
        if trade_id not in self.active_trades:
            return
        
        state = self.active_trades[trade_id]
        
        # Update current state
        state.current_price = market_update['current_price']
        state.unrealized_pnl = self._calculate_unrealized_pnl(state, market_update['current_price'])
        
        # Update MAE/MFE
        if state.unrealized_pnl < 0:
            state.max_adverse_excursion = max(state.max_adverse_excursion, abs(state.unrealized_pnl))
        else:
            state.max_favorable_excursion = max(state.max_favorable_excursion, state.unrealized_pnl)
        
        # Record state snapshot
        state_snapshot = {
            'timestamp': time.time(),
            'price': state.current_price,
            'unrealized_pnl': state.unrealized_pnl,
            'mae': state.max_adverse_excursion,
            'mfe': state.max_favorable_excursion,
            'volatility': market_update.get('volatility', 50),
            'momentum': market_update.get('momentum', 0)
        }
        state.state_history.append(state_snapshot)
        
        # Keep history manageable
        if len(state.state_history) > 1000:
            state.state_history = state.state_history[-500:]
    
    def on_trade_action(self, trade_id: str, action: int, action_context: Dict[str, Any]):
        """Called when an action is taken on an active trade"""
        
        if trade_id not in self.active_trades:
            return
        
        state = self.active_trades[trade_id]
        state.action_history.append(action)
        
        # Learn from intermediate actions immediately
        self._learn_from_intermediate_action(state, action, action_context)
    
    def on_trade_exit(self, trade_id: str, exit_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Called when trade is closed
        Returns: Learning insights from completed trade
        """
        
        if trade_id not in self.active_trades:
            return {}
        
        state = self.active_trades[trade_id]
        
        # Create complete trade journey
        journey = TradeJourney(
            trade_id=trade_id,
            symbol=state.symbol,
            agent_type=state.agent_type,
            entry_state={'price': state.entry_price, 'volatility': state.volatility_percentile},
            intermediate_states=state.state_history,
            exit_state={'price': exit_data['exit_price'], 'final_pnl': exit_data['net_pnl']},
            duration_steps=len(state.state_history),
            total_pnl=exit_data['net_pnl'],
            max_adverse_excursion=state.max_adverse_excursion,
            max_favorable_excursion=state.max_favorable_excursion,
            entry_action=state.entry_action,
            intermediate_actions=state.action_history,
            exit_action=exit_data['exit_action'],
            market_regime=state.market_regime,
            volatility_percentile=state.volatility_percentile,
            volume_percentile=state.volume_percentile
        )
        
        # Add to experience replay for learning
        self.replay_system.add_experience(journey)
        self.completed_journeys.append(journey)
        
        # Update learning statistics
        self._update_learning_stats(journey)
        
        # Learn immediately from this completed journey
        learning_insights = self._learn_from_completed_journey(journey)
        
        # Remove from active trades
        del self.active_trades[trade_id]
        
        print(f"📉 Trade completed: {trade_id}")
        print(f"   💰 P&L: {exit_data['net_pnl']:+.0f}")
        print(f"   ⚡ Efficiency: {journey.efficiency_score:.3f}")
        print(f"   📚 Learning value: {journey.learning_value:.3f}")
        
        return {
            'efficiency_score': journey.efficiency_score,
            'learning_value': journey.learning_value,
            'insights': learning_insights
        }
    
    def _should_explore(self, symbol: str, agent_type: str, regime: str) -> bool:
        """Determine if trade should be exploration-focused"""
        
        # Base exploration rate
        if random.random() < self.exploration_rate:
            return True
        
        # Explore more in rare conditions
        rare_conditions = ['CHAOTIC', 'OVERDAMPED']
        if regime in rare_conditions:
            return random.random() < self.exploration_rate * 1.5
        
        # Explore more for underperforming combinations
        combo_key = f"{agent_type}_{symbol}_{regime}"
        recent_performance = self.learning_stats['agent_performance_trends'][combo_key]
        
        if recent_performance and statistics.mean(recent_performance) < 0.4:
            return random.random() < self.exploration_rate * 2.0
        
        return False
    
    def _calculate_trade_confidence(self, symbol: str, agent_type: str, regime: str) -> float:
        """Calculate confidence level for trade based on historical learning"""
        
        # Base confidence from strategy weights
        base_confidence = self.strategy_weights.get(agent_type, {}).get(regime, 0.5)
        
        # Adjust based on recent learning
        combo_key = f"{agent_type}_{symbol}_{regime}"
        recent_performance = self.learning_stats['agent_performance_trends'][combo_key]
        
        if recent_performance:
            recent_avg = statistics.mean(recent_performance)
            # Boost confidence if recent performance is good
            if recent_avg > 0.7:
                base_confidence *= 1.2
            elif recent_avg < 0.3:
                base_confidence *= 0.8
        
        return min(1.0, max(0.1, base_confidence))
    
    def _get_learning_based_modifications(self, state: LiveTradingState) -> Dict[str, Any]:
        """Get trade modifications based on learning"""
        
        modifications = {}
        
        # Position size modification based on confidence
        if state.confidence_level > 0.8:
            modifications['position_multiplier'] = 1.2  # Increase size for high confidence
        elif state.confidence_level < 0.4:
            modifications['position_multiplier'] = 0.7  # Reduce size for low confidence
        
        # Stop loss modification based on regime learning
        if state.market_regime == 'CHAOTIC' and state.agent_type == 'BERSERKER':
            modifications['stop_multiplier'] = 1.5  # Wider stops in chaos for BERSERKER
        elif state.market_regime == 'OVERDAMPED':
            modifications['stop_multiplier'] = 0.8  # Tighter stops in low volatility
        
        # Exploration modifications
        if state.is_exploration_trade:
            modifications['exploration_mode'] = True
            modifications['take_profit_multiplier'] = 1.3  # Let winners run more in exploration
        
        return modifications
    
    def _calculate_unrealized_pnl(self, state: LiveTradingState, current_price: float) -> float:
        """Calculate unrealized P&L for position"""
        
        price_diff = current_price - state.entry_price
        direction = 1 if state.position_size > 0 else -1  # Assume positive size = long
        
        return direction * price_diff * abs(state.position_size)
    
    def _learn_from_intermediate_action(self, state: LiveTradingState, action: int, context: Dict):
        """Learn from actions taken during trade"""
        
        # Evaluate if action was appropriate given current state
        current_efficiency = 0.0
        if state.max_adverse_excursion > 0:
            current_efficiency = state.max_favorable_excursion / state.max_adverse_excursion
        
        # Store immediate feedback for this action
        # This could be used for real-time strategy adjustment
        action_feedback = {
            'action': action,
            'efficiency_at_time': current_efficiency,
            'unrealized_pnl': state.unrealized_pnl,
            'regime': state.market_regime,
            'agent': state.agent_type
        }
        
        # Use this for immediate learning updates if needed
        pass
    
    def _learn_from_completed_journey(self, journey: TradeJourney) -> Dict[str, Any]:
        """Learn from completed trade journey"""
        
        insights = {}
        
        # Analyze journey efficiency vs historical patterns
        efficiency_score = journey.efficiency_score
        
        # Update agent/regime performance tracking
        combo_key = f"{journey.agent_type}_{journey.symbol}_{journey.market_regime}"
        self.learning_stats['agent_performance_trends'][combo_key].append(efficiency_score)
        
        # Identify key insights
        if efficiency_score > 0.8:
            insights['pattern'] = 'highly_efficient'
            insights['recommendation'] = f"Increase {journey.agent_type} allocation in {journey.market_regime} regime for {journey.symbol}"
        elif efficiency_score < 0.3:
            insights['pattern'] = 'highly_inefficient'
            insights['recommendation'] = f"Avoid {journey.agent_type} in {journey.market_regime} regime for {journey.symbol}"
        
        # Duration efficiency insights
        duration_steps = journey.duration_steps
        expected_duration = self.efficiency_analyzer._get_expected_duration(journey.symbol, journey.market_regime)
        
        if duration_steps > expected_duration * 2:
            insights['duration_issue'] = f"Trade held {duration_steps/expected_duration:.1f}x longer than expected"
        
        return insights
    
    def _update_learning_stats(self, journey: TradeJourney):
        """Update overall learning statistics"""
        
        self.learning_stats['total_journeys_learned'] += 1
        self.learning_stats['avg_efficiency_trend'].append(journey.efficiency_score)
        self.learning_stats['last_learning_session'] = datetime.now()
        
        # Update regime success rates
        regime = journey.market_regime
        was_profitable = journey.total_pnl > 0
        
        current_rate = self.learning_stats['regime_success_rates'].get(regime, 0.5)
        # Simple moving average update
        alpha = 0.1  # Learning rate
        new_rate = current_rate + alpha * (1.0 if was_profitable else 0.0 - current_rate)
        self.learning_stats['regime_success_rates'][regime] = new_rate
    
    def run_learning_session(self, focus_type: str = 'mixed', batch_size: int = 32) -> Dict[str, Any]:
        """
        Run focused learning session on historical experiences
        
        focus_type: 'efficient', 'inefficient', 'mixed', 'rare'
        """
        
        if not self.replay_system.experiences:
            return {'error': 'No experiences available for learning'}
        
        print(f"\n🧠 LEARNING SESSION: {focus_type.upper()} FOCUS")
        print("=" * 50)
        
        # Sample experiences for learning
        experiences = self.replay_system.sample_for_learning(batch_size, focus_type)
        
        # Analyze patterns in sampled experiences
        learning_results = self._analyze_experience_patterns(experiences)
        
        # Update strategy weights based on learning
        self._update_strategy_weights(learning_results)
        
        # Get learning insights
        insights = self.replay_system.get_learning_insights()
        
        print(f"📚 Processed {len(experiences)} experiences")
        print(f"🎯 Focus: {focus_type}")
        print(f"📈 Avg efficiency: {insights['avg_efficiency']:.3f}")
        
        if insights['learning_trends']:
            print(f"📊 Learning trends:")
            for combo, trend in insights['learning_trends'].items():
                print(f"   {combo}: {trend:+.3f}")
        
        return {
            'experiences_processed': len(experiences),
            'learning_results': learning_results,
            'insights': insights,
            'updated_weights': dict(self.strategy_weights)
        }
    
    def _analyze_experience_patterns(self, experiences: List[TradeJourney]) -> Dict[str, Any]:
        """Analyze patterns in sampled experiences"""
        
        patterns = {
            'agent_efficiency': defaultdict(list),
            'regime_efficiency': defaultdict(list), 
            'symbol_efficiency': defaultdict(list),
            'duration_patterns': defaultdict(list)
        }
        
        for exp in experiences:
            patterns['agent_efficiency'][exp.agent_type].append(exp.efficiency_score)
            patterns['regime_efficiency'][exp.market_regime].append(exp.efficiency_score)
            patterns['symbol_efficiency'][exp.symbol].append(exp.efficiency_score)
            patterns['duration_patterns'][f"{exp.agent_type}_{exp.market_regime}"].append(exp.duration_steps)
        
        # Calculate averages
        results = {}
        for pattern_type, data in patterns.items():
            results[pattern_type] = {}
            for key, values in data.items():
                if values:
                    results[pattern_type][key] = {
                        'avg': statistics.mean(values),
                        'count': len(values),
                        'std': statistics.stdev(values) if len(values) > 1 else 0
                    }
        
        return results
    
    def _update_strategy_weights(self, learning_results: Dict[str, Any]):
        """Update strategy weights based on learning results"""
        
        # Update agent/regime weights based on efficiency
        regime_efficiency = learning_results.get('regime_efficiency', {})
        agent_efficiency = learning_results.get('agent_efficiency', {})
        
        learning_rate = 0.05  # Conservative learning rate
        
        for agent in ['BERSERKER', 'SNIPER']:
            for regime in ['CHAOTIC', 'UNDERDAMPED', 'CRITICALLY_DAMPED', 'OVERDAMPED']:
                current_weight = self.strategy_weights[agent][regime]
                
                # Get efficiency data for this combination
                regime_eff = regime_efficiency.get(regime, {}).get('avg', 0.5)
                agent_eff = agent_efficiency.get(agent, {}).get('avg', 0.5)
                
                # Combined efficiency signal
                combined_eff = (regime_eff + agent_eff) / 2.0
                
                # Update weight
                target_weight = combined_eff
                new_weight = current_weight + learning_rate * (target_weight - current_weight)
                self.strategy_weights[agent][regime] = max(0.1, min(1.5, new_weight))
        
        # Decay exploration rate
        self.exploration_rate *= self.exploration_decay
    
    def get_learning_summary(self) -> Dict[str, Any]:
        """Get comprehensive learning summary"""
        
        total_experiences = len(self.replay_system.experiences)
        if total_experiences == 0:
            return {'status': 'no_learning_data'}
        
        # Recent efficiency trend
        recent_efficiency = list(self.learning_stats['avg_efficiency_trend'])
        efficiency_trend = 'improving' if len(recent_efficiency) > 10 and recent_efficiency[-5:] > recent_efficiency[-10:-5] else 'stable'
        
        # Best/worst performing combinations
        performance_data = []
        for combo, trend in self.learning_stats['agent_performance_trends'].items():
            if trend:
                performance_data.append((combo, statistics.mean(trend)))
        
        performance_data.sort(key=lambda x: x[1], reverse=True)
        best_combo = performance_data[0] if performance_data else None
        worst_combo = performance_data[-1] if performance_data else None
        
        return {
            'total_experiences': total_experiences,
            'total_journeys_learned': self.learning_stats['total_journeys_learned'],
            'current_exploration_rate': self.exploration_rate,
            'efficiency_trend': efficiency_trend,
            'avg_recent_efficiency': statistics.mean(recent_efficiency) if recent_efficiency else 0,
            'best_performing_combo': best_combo,
            'worst_performing_combo': worst_combo,
            'regime_success_rates': dict(self.learning_stats['regime_success_rates']),
            'current_strategy_weights': dict(self.strategy_weights),
            'active_trades': len(self.active_trades),
            'last_learning_session': self.learning_stats['last_learning_session']
        }

def demonstrate_rl_integration():
    """Demonstrate the RL learning integration system"""
    
    print("🚀 RL LEARNING INTEGRATION DEMONSTRATION")
    print("=" * 80)
    
    # Initialize learning engine
    engine = RLLearningEngine()
    
    # Simulate a few trades
    print(f"\n📈 SIMULATING TRADING SEQUENCE")
    
    # Trade 1: Successful BERSERKER in CHAOTIC
    entry_data = {
        'symbol': 'BTCUSD',
        'agent_type': 'BERSERKER', 
        'action': 2,
        'entry_price': 95000,
        'position_size': 1.0,
        'market_regime': 'CHAOTIC',
        'volatility_percentile': 85,
        'volume_percentile': 75
    }
    
    trade1_result = engine.on_trade_entry(entry_data)
    trade1_id = trade1_result['trade_id']
    
    # Simulate price updates
    for i, price in enumerate([95100, 95300, 95800, 96200, 96500]):
        engine.on_trade_update(trade1_id, {'current_price': price})
        if i == 2:  # Take action at step 3
            engine.on_trade_action(trade1_id, 8, {'action_type': 'reduce_position'})
    
    # Close trade
    engine.on_trade_exit(trade1_id, {
        'exit_price': 96500,
        'net_pnl': 15000,
        'exit_action': 7
    })
    
    # Trade 2: Poor BERSERKER in OVERDAMPED
    entry_data2 = {
        'symbol': 'EURUSD',
        'agent_type': 'BERSERKER',
        'action': 3,
        'entry_price': 1.0500,
        'position_size': 100000,
        'market_regime': 'OVERDAMPED',
        'volatility_percentile': 25,
        'volume_percentile': 30
    }
    
    trade2_result = engine.on_trade_entry(entry_data2)
    trade2_id = trade2_result['trade_id']
    
    # Simulate losing trade
    for price in [1.0490, 1.0475, 1.0460, 1.0440, 1.0420]:
        engine.on_trade_update(trade2_id, {'current_price': price})
    
    engine.on_trade_exit(trade2_id, {
        'exit_price': 1.0420,
        'net_pnl': -8000,
        'exit_action': 7
    })
    
    # Run learning session
    learning_results = engine.run_learning_session(focus_type='mixed')
    
    # Get learning summary
    print(f"\n📊 LEARNING SUMMARY:")
    summary = engine.get_learning_summary()
    
    for key, value in summary.items():
        if key not in ['current_strategy_weights', 'regime_success_rates']:
            print(f"   {key}: {value}")
    
    print(f"\n🎯 UPDATED STRATEGY WEIGHTS:")
    for agent, regimes in summary['current_strategy_weights'].items():
        print(f"   {agent}:")
        for regime, weight in regimes.items():
            print(f"      {regime}: {weight:.3f}")
    
    print(f"\n✅ RL Learning Integration demonstration completed!")
    print(f"🧠 System is continuously learning from trade outcomes")
    print(f"⚡ Focus on journey efficiency enables better decision making")
    print(f"🎯 Asymmetrical rewards value the path to profit, not just profit")
    
    return engine

if __name__ == "__main__":
    demonstrate_rl_integration()