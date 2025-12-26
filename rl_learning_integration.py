#!/usr/bin/env python3
"""
RL Learning Integration Module
Provides core RL learning components for the trading system

Features:
- RL learning engine with multiple algorithms
- Live trading state management
- Experience replay and learning optimization
- Performance tracking and metrics
"""

import json
import time
import random
import math
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import deque
import logging

# Import defensive safety functions
from defensive_safety import safe_divide, safe_percentage, validate_numeric, safe_array_mean, safe_array_std

@dataclass
class LiveTradingState:
    """Real-time trading state for RL learning"""
    trade_id: str
    symbol: str
    entry_price: float
    position_size: float
    direction: str  # BUY/SELL
    entry_time: datetime
    market_regime: str
    agent_type: str
    
    # Dynamic state
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    max_favorable_excursion: float = 0.0
    max_adverse_excursion: float = 0.0
    duration_seconds: int = 0
    
    # RL specific
    state_vector: List[float] = field(default_factory=list)
    action_taken: int = 0
    confidence: float = 0.0
    is_exploration: bool = False

@dataclass
class RLExperience:
    """Single RL experience for replay learning"""
    state: List[float]
    action: int
    reward: float
    next_state: List[float]
    done: bool
    metadata: Dict[str, Any] = field(default_factory=dict)

class RLLearningEngine:
    """Core RL learning engine for trading agents"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        
        # RL Parameters
        self.learning_rate = self.config.get('learning_rate', 0.001)
        self.gamma = self.config.get('gamma', 0.95)
        self.epsilon = self.config.get('initial_epsilon', 0.1)
        self.epsilon_decay = self.config.get('epsilon_decay', 0.995)
        self.epsilon_min = self.config.get('epsilon_min', 0.01)
        
        # Experience replay
        self.memory_size = self.config.get('memory_size', 10000)
        self.batch_size = self.config.get('batch_size', 32)
        self.experience_buffer = deque(maxlen=self.memory_size)
        
        # State tracking
        self.active_trades: Dict[str, LiveTradingState] = {}
        self.completed_trades: List[Dict] = []
        
        # Performance metrics
        self.total_episodes = 0
        self.total_rewards = 0.0
        self.recent_rewards = deque(maxlen=100)
        self.learning_history = []
        
        # Disabled verbose logging
    
    def on_trade_entry(self, trade_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process new trade entry for RL learning"""
        
        trade_id = f"{trade_data['agent']}_{trade_data['symbol']}_{trade_data['timeframe']}_{int(time.time())}"
        
        # Create live trading state
        trading_state = LiveTradingState(
            trade_id=trade_id,
            symbol=trade_data['symbol'],
            entry_price=trade_data['entry_price'],
            position_size=trade_data['position_size'],
            direction=trade_data['signal_type'],
            entry_time=datetime.fromisoformat(trade_data['timestamp']),
            market_regime=trade_data.get('regime', 'UNKNOWN'),
            agent_type=trade_data['agent'],
            current_price=trade_data['entry_price'],
            confidence=trade_data['confidence'],
            is_exploration=trade_data['confidence'] < 0.6  # Low confidence = exploration
        )
        
        # Generate state vector for RL
        state_vector = self._create_state_vector(trade_data)
        trading_state.state_vector = state_vector
        trading_state.action_taken = self._map_action(trade_data['signal_type'])
        
        # Store active trade
        self.active_trades[trade_id] = trading_state
        
        # Increment episodes
        self.total_episodes += 1
        
        return {
            'trade_id': trade_id,
            'confidence': trading_state.confidence,
            'is_exploration': trading_state.is_exploration,
            'state_vector_size': len(state_vector),
            'action': trading_state.action_taken
        }
    
    def on_trade_update(self, trade_id: str, update_data: Dict[str, Any]):
        """Update existing trade state"""
        
        if trade_id not in self.active_trades:
            return
        
        trade_state = self.active_trades[trade_id]
        
        # Update current state
        trade_state.current_price = update_data['current_price']
        trade_state.duration_seconds = (datetime.now() - trade_state.entry_time).total_seconds()
        
        # Calculate P&L
        if trade_state.direction == 'BUY':
            pnl = (trade_state.current_price - trade_state.entry_price) * trade_state.position_size
        else:
            pnl = (trade_state.entry_price - trade_state.current_price) * trade_state.position_size
        
        trade_state.unrealized_pnl = pnl
        
        # Update excursions
        if pnl > trade_state.max_favorable_excursion:
            trade_state.max_favorable_excursion = pnl
        
        if pnl < 0 and abs(pnl) > trade_state.max_adverse_excursion:
            trade_state.max_adverse_excursion = abs(pnl)
    
    def on_trade_exit(self, trade_id: str, exit_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process trade exit and create learning experience"""
        
        if trade_id not in self.active_trades:
            return {'error': 'Trade not found'}
        
        trade_state = self.active_trades.pop(trade_id)
        
        # Calculate final metrics
        final_pnl = exit_data['net_pnl']
        duration = (datetime.now() - trade_state.entry_time).total_seconds()
        
        # Calculate reward for RL
        reward = self._calculate_reward(trade_state, final_pnl, duration)
        
        # Create next state (post-trade)
        next_state = self._create_post_trade_state(trade_state, exit_data)
        
        # Create RL experience
        experience = RLExperience(
            state=trade_state.state_vector,
            action=trade_state.action_taken,
            reward=reward,
            next_state=next_state,
            done=True,
            metadata={
                'symbol': trade_state.symbol,
                'agent': trade_state.agent_type,
                'regime': trade_state.market_regime,
                'pnl': final_pnl,
                'duration': duration
            }
        )
        
        # Add to experience buffer
        self.experience_buffer.append(experience)
        
        # Update metrics
        self.total_rewards += reward
        self.recent_rewards.append(reward)
        
        # Calculate efficiency score
        efficiency_score = self._calculate_efficiency_score(trade_state, final_pnl, duration)
        
        # Store completed trade
        completed_trade = {
            'trade_id': trade_id,
            'symbol': trade_state.symbol,
            'agent': trade_state.agent_type,
            'pnl': final_pnl,
            'duration': duration,
            'reward': reward,
            'efficiency': efficiency_score,
            'regime': trade_state.market_regime,
            'timestamp': datetime.now().isoformat()
        }
        self.completed_trades.append(completed_trade)
        
        return {
            'trade_id': trade_id,
            'reward': reward,
            'efficiency_score': efficiency_score,
            'learning_value': abs(reward),  # Higher rewards = more learning value
            'experience_added': True
        }
    
    def run_learning_session(self, focus_type: str = 'mixed', batch_size: Optional[int] = None) -> Dict[str, Any]:
        """Run a learning session using experience replay"""
        
        if len(self.experience_buffer) < (batch_size or self.batch_size):
            return {'status': 'insufficient_data', 'buffer_size': len(self.experience_buffer)}
        
        batch_size = batch_size or self.batch_size
        
        # Sample experiences based on focus type
        if focus_type == 'recent':
            # Focus on recent experiences
            experiences = list(self.experience_buffer)[-batch_size:]
        elif focus_type == 'efficient':
            # Focus on high-reward experiences
            experiences = sorted(self.experience_buffer, key=lambda x: abs(x.reward), reverse=True)[:batch_size]
        else:  # mixed
            # Random sampling
            experiences = random.sample(list(self.experience_buffer), batch_size)
        
        # Simulate learning process
        total_loss = 0.0
        for experience in experiences:
            # Simple Q-learning update simulation
            q_target = experience.reward + self.gamma * self._estimate_q_value(experience.next_state)
            q_current = self._estimate_q_value(experience.state)
            
            loss = (q_target - q_current) ** 2
            total_loss += loss
        
        avg_loss = total_loss / len(experiences)
        
        # Update exploration rate
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        # Record learning history
        learning_record = {
            'timestamp': datetime.now().isoformat(),
            'experiences_processed': len(experiences),
            'avg_loss': avg_loss,
            'exploration_rate': self.epsilon,
            'buffer_size': len(self.experience_buffer),
            'focus_type': focus_type
        }
        self.learning_history.append(learning_record)
        
        return {
            'experiences_processed': len(experiences),
            'avg_loss': avg_loss,
            'exploration_rate': self.epsilon,
            'learning_progress': min(1.0, len(self.experience_buffer) / self.memory_size),
            'status': 'completed'
        }
    
    def get_learning_summary(self) -> Dict[str, Any]:
        """Get comprehensive learning summary"""
        
        avg_reward = sum(self.recent_rewards) / len(self.recent_rewards) if self.recent_rewards else 0.0
        
        # Agent performance by type
        agent_performance = {}
        for trade in self.completed_trades[-100:]:  # Last 100 trades
            agent = trade['agent']
            if agent not in agent_performance:
                agent_performance[agent] = {'trades': 0, 'total_pnl': 0, 'total_reward': 0}
            
            agent_performance[agent]['trades'] += 1
            agent_performance[agent]['total_pnl'] += trade['pnl']
            agent_performance[agent]['total_reward'] += trade['reward']
        
        return {
            'total_episodes': self.total_episodes,
            'total_experiences': len(self.experience_buffer),
            'current_exploration_rate': self.epsilon,
            'average_recent_reward': avg_reward,
            'total_journeys_learned': len(self.completed_trades),
            'learning_sessions': len(self.learning_history),
            'agent_performance': agent_performance,
            'memory_utilization': len(self.experience_buffer) / self.memory_size
        }
    
    def _create_state_vector(self, trade_data: Dict) -> List[float]:
        """Create state vector for RL input"""
        
        # Simplified state vector (would be more complex in real implementation)
        state = [
            float(trade_data['confidence']),
            1.0 if trade_data['signal_type'] == 'BUY' else 0.0,
            float(trade_data['position_size']),
            hash(trade_data.get('regime', 'UNKNOWN')) % 4 / 4.0,  # Regime encoding
            float(hash(trade_data['symbol']) % 10) / 10.0,  # Symbol encoding
            float(time.time() % 86400) / 86400.0,  # Time of day
        ]
        
        return state
    
    def _create_post_trade_state(self, trade_state: LiveTradingState, exit_data: Dict) -> List[float]:
        """Create post-trade state vector"""
        
        # Simplified next state
        next_state = [
            exit_data['net_pnl'] / 1000.0,  # Normalized P&L
            trade_state.duration_seconds / 3600.0,  # Duration in hours
            trade_state.max_favorable_excursion / 100.0,
            trade_state.max_adverse_excursion / 100.0,
            1.0 if exit_data['net_pnl'] > 0 else 0.0,  # Win/Loss
            random.random()  # Market randomness
        ]
        
        return next_state
    
    def _map_action(self, signal_type: str) -> int:
        """Map trading signal to action integer"""
        
        action_map = {'BUY': 0, 'SELL': 1, 'HOLD': 2, 'CLOSE': 3}
        return action_map.get(signal_type, 2)
    
    def _calculate_reward(self, trade_state: LiveTradingState, final_pnl: float, duration: float) -> float:
        """Calculate RL reward for trade outcome"""
        
        # Base reward from P&L
        pnl_reward = final_pnl / 100.0  # Normalize
        
        # Duration penalty (prefer shorter trades for efficiency)
        duration_penalty = -duration / 3600.0 * 0.1  # Small penalty for duration
        
        # Confidence bonus (reward confident decisions)
        confidence_bonus = trade_state.confidence * 0.5 if final_pnl > 0 else 0
        
        # Risk-adjusted reward
        risk_adjustment = -trade_state.max_adverse_excursion / 200.0 if trade_state.max_adverse_excursion > 0 else 0
        
        total_reward = pnl_reward + duration_penalty + confidence_bonus + risk_adjustment
        
        return float(total_reward)
    
    def _calculate_efficiency_score(self, trade_state: LiveTradingState, final_pnl: float, duration: float) -> float:
        """Calculate trade efficiency score"""
        
        if duration == 0:
            return 0.0
        
        # Efficiency = (P&L / duration) * confidence factor
        base_efficiency = (final_pnl / (duration / 3600.0)) if duration > 0 else 0  # P&L per hour
        confidence_factor = trade_state.confidence
        
        efficiency = base_efficiency * confidence_factor
        
        # Normalize to 0-1 range
        return max(0.0, min(1.0, efficiency / 100.0))
    
    def _estimate_q_value(self, state: List[float]) -> float:
        """Estimate Q-value for given state (simplified)"""
        
        # Simple linear combination (would use neural network in real implementation)
        if not state:
            return 0.0
        
        weights = [0.5, 0.3, 0.2, 0.1, 0.1, 0.1][:len(state)]
        q_value = sum(s * w for s, w in zip(state, weights))
        
        return float(q_value)

def demonstrate_rl_learning():
    """Demonstrate RL learning integration"""
    
    print("🧠 RL LEARNING INTEGRATION DEMONSTRATION")
    print("=" * 60)
    
    # Create RL engine
    engine = RLLearningEngine({
        'learning_rate': 0.001,
        'initial_epsilon': 0.2,
        'memory_size': 1000
    })
    
    # Simulate several trades
    trade_ids = []
    
    for i in range(10):
        # Create mock trade entry
        trade_data = {
            'agent': 'BERSERKER' if i % 2 == 0 else 'SNIPER',
            'symbol': ['EURUSD+', 'GBPUSD+', 'BTCUSD+'][i % 3],
            'timeframe': 'M15',
            'signal_type': 'BUY' if i % 2 == 0 else 'SELL',
            'entry_price': 1.1000 + random.uniform(-0.01, 0.01),
            'position_size': 0.1,
            'confidence': 0.3 + random.random() * 0.6,
            'timestamp': datetime.now().isoformat(),
            'regime': ['CHAOTIC', 'UNDERDAMPED', 'CRITICALLY_DAMPED'][i % 3]
        }
        
        # Process entry
        result = engine.on_trade_entry(trade_data)
        trade_ids.append(result['trade_id'])
        
        print(f"📈 Trade {i+1}: {result['trade_id'][:20]}... (confidence: {result['confidence']:.3f})")
    
    # Simulate trade updates and exits
    for i, trade_id in enumerate(trade_ids):
        # Simulate some updates
        for j in range(3):
            update_data = {
                'current_price': 1.1000 + random.uniform(-0.005, 0.005),
                'timestamp': datetime.now(),
                'volatility': random.random(),
                'momentum': random.uniform(-1, 1)
            }
            engine.on_trade_update(trade_id, update_data)
        
        # Simulate exit
        exit_data = {
            'net_pnl': random.uniform(-50, 100),
            'exit_price': 1.1000 + random.uniform(-0.01, 0.01),
            'exit_reason': 'TP_HIT' if random.random() > 0.5 else 'MANUAL'
        }
        
        exit_result = engine.on_trade_exit(trade_id, exit_data)
        print(f"📉 Exit {i+1}: P&L {exit_data['net_pnl']:+.0f}, Reward: {exit_result['reward']:+.3f}")
    
    # Run learning session
    print(f"\n🧠 Running learning session...")
    learning_result = engine.run_learning_session('mixed', batch_size=8)
    print(f"   Processed: {learning_result['experiences_processed']} experiences")
    print(f"   Avg Loss: {learning_result['avg_loss']:.4f}")
    print(f"   Exploration Rate: {learning_result['exploration_rate']:.3f}")
    
    # Get summary
    summary = engine.get_learning_summary()
    print(f"\n📊 LEARNING SUMMARY:")
    print(f"   Total Episodes: {summary['total_episodes']}")
    print(f"   Experience Buffer: {summary['total_experiences']}/{engine.memory_size}")
    print(f"   Average Reward: {summary['average_recent_reward']:+.3f}")
    print(f"   Learning Sessions: {summary['learning_sessions']}")
    
    if summary['agent_performance']:
        print(f"   Agent Performance:")
        for agent, perf in summary['agent_performance'].items():
            avg_pnl = perf['total_pnl'] / perf['trades'] if perf['trades'] > 0 else 0
            print(f"     {agent}: {perf['trades']} trades, avg P&L: {avg_pnl:+.1f}")
    
    print("\n✅ RL Learning Integration demonstration completed!")

if __name__ == "__main__":
    demonstrate_rl_learning()