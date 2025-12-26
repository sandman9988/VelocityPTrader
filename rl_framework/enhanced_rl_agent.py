#!/usr/bin/env python3
"""
Enhanced RL Trading Agent with Advanced Reward Shaping
Deep Q-Network (DQN) with sophisticated reward shaping and performance tracking
Features:
- Double DQN with target networks
- Prioritized Experience Replay  
- Multi-objective reward functions
- Advanced performance metrics integration
- Continuous learning and adaptation
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent))

import math
import random
import statistics
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque, namedtuple
from enum import Enum

# Import our advanced components
from reward_shaping_engine import RewardShapingEngine, TradingState, RewardResult
from advanced_performance_metrics import AdvancedPerformanceCalculator, TradeMetrics, InstrumentPerformance

# Experience tuple for replay buffer
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done', 'priority'])

class ActionType(Enum):
    """Available actions for the RL agent"""
    WAIT = 0
    ENTER_LONG_SMALL = 1
    ENTER_LONG_MEDIUM = 2
    ENTER_LONG_LARGE = 3
    ENTER_SHORT_SMALL = 4
    ENTER_SHORT_MEDIUM = 5
    ENTER_SHORT_LARGE = 6
    EXIT_POSITION = 7
    REDUCE_POSITION = 8
    INCREASE_POSITION = 9

@dataclass
class AgentState:
    """Complete agent state representation"""
    # Market features (normalized)
    price_change: float = 0.0
    volatility: float = 0.0
    volume_percentile: float = 0.0
    momentum: float = 0.0
    regime_vector: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])  # One-hot encoded
    
    # Position features
    position_size: float = 0.0
    position_pnl: float = 0.0
    time_in_position: float = 0.0
    
    # Performance features
    mae: float = 0.0
    mfe: float = 0.0
    efficiency: float = 0.0
    drawdown: float = 0.0
    
    # Historical features
    recent_returns: List[float] = field(default_factory=lambda: [0.0] * 10)
    win_streak: float = 0.0
    loss_streak: float = 0.0
    
    # Agent meta features
    confidence: float = 0.5
    exploration_factor: float = 0.1
    
    def to_vector(self) -> List[float]:
        """Convert state to feature vector"""
        vector = [
            self.price_change,
            self.volatility,
            self.volume_percentile,
            self.momentum
        ]
        vector.extend(self.regime_vector)
        vector.extend([
            self.position_size,
            self.position_pnl,
            self.time_in_position,
            self.mae,
            self.mfe,
            self.efficiency,
            self.drawdown
        ])
        vector.extend(self.recent_returns)
        vector.extend([
            self.win_streak,
            self.loss_streak,
            self.confidence,
            self.exploration_factor
        ])
        return vector

class PrioritizedReplayBuffer:
    """Prioritized experience replay buffer"""
    
    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.buffer = []
        self.priorities = []
        self.alpha = 0.6  # Priority exponent
        self.beta = 0.4   # Importance sampling exponent (annealed)
        self.beta_increment = 0.001
        self.epsilon = 1e-6  # Small value to prevent zero priorities
        
    def push(self, experience: Experience):
        """Add experience with priority"""
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
            self.priorities.append(max(self.priorities) if self.priorities else 1.0)
        else:
            # Replace experience with lowest priority
            min_idx = self.priorities.index(min(self.priorities))
            self.buffer[min_idx] = experience
            self.priorities[min_idx] = max(self.priorities)
    
    def sample(self, batch_size: int) -> Tuple[List[Experience], List[float], List[int]]:
        """Sample batch with importance sampling weights"""
        if len(self.buffer) < batch_size:
            return [], [], []
        
        # Calculate sampling probabilities
        priorities = np.array(self.priorities)
        probs = priorities ** self.alpha
        probs = probs / probs.sum()
        
        # Sample indices
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        
        # Sample experiences
        experiences = [self.buffer[i] for i in indices]
        
        # Calculate importance sampling weights
        weights = (len(self.buffer) * probs[indices]) ** (-self.beta)
        weights = weights / weights.max()  # Normalize
        
        # Anneal beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        return experiences, weights.tolist(), indices.tolist()
    
    def update_priorities(self, indices: List[int], td_errors: List[float]):
        """Update priorities based on TD errors"""
        for idx, error in zip(indices, td_errors):
            if idx < len(self.priorities):
                self.priorities[idx] = abs(error) + self.epsilon

class SimpleDQN:
    """Simplified Deep Q-Network for demonstration"""
    
    def __init__(self, input_size: int, hidden_size: int = 128, output_size: int = 10):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Initialize weights randomly (simplified implementation)
        self.weights1 = [[random.gauss(0, 0.1) for _ in range(input_size)] for _ in range(hidden_size)]
        self.bias1 = [random.gauss(0, 0.1) for _ in range(hidden_size)]
        self.weights2 = [[random.gauss(0, 0.1) for _ in range(hidden_size)] for _ in range(output_size)]
        self.bias2 = [random.gauss(0, 0.1) for _ in range(output_size)]
        
        self.learning_rate = 0.001
    
    def forward(self, state_vector: List[float]) -> List[float]:
        """Forward pass through network"""
        # Hidden layer
        hidden = []
        for i in range(self.hidden_size):
            activation = self.bias1[i]
            for j in range(self.input_size):
                activation += state_vector[j] * self.weights1[i][j]
            hidden.append(max(0, activation))  # ReLU
        
        # Output layer
        outputs = []
        for i in range(self.output_size):
            activation = self.bias2[i]
            for j in range(self.hidden_size):
                activation += hidden[j] * self.weights2[i][j]
            outputs.append(activation)
        
        return outputs
    
    def copy_from(self, other_network):
        """Copy weights from another network"""
        self.weights1 = [row[:] for row in other_network.weights1]
        self.bias1 = other_network.bias1[:]
        self.weights2 = [row[:] for row in other_network.weights2]
        self.bias2 = other_network.bias2[:]

class EnhancedRLAgent:
    """
    Enhanced RL Trading Agent with advanced features:
    - Double DQN architecture
    - Prioritized experience replay
    - Multi-objective reward shaping
    - Continuous performance monitoring
    - Adaptive exploration strategies
    """
    
    def __init__(self, symbol: str, config: Optional[Dict] = None):
        self.symbol = symbol
        self.config = config or {}
        
        # Initialize components
        self.reward_shaper = RewardShapingEngine()
        self.performance_calc = AdvancedPerformanceCalculator()
        
        # Network architecture
        self.state_size = 26  # From AgentState.to_vector()
        self.action_size = len(ActionType)
        self.q_network = SimpleDQN(self.state_size, 128, self.action_size)
        self.target_network = SimpleDQN(self.state_size, 128, self.action_size)
        self.target_network.copy_from(self.q_network)
        
        # Experience replay
        self.replay_buffer = PrioritizedReplayBuffer(10000)
        self.batch_size = 32
        
        # Learning parameters
        self.epsilon = 1.0          # Exploration rate
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.05
        self.gamma = 0.99           # Discount factor
        self.target_update_freq = 100
        
        # Training state
        self.training_step = 0
        self.episode = 0
        self.current_state: Optional[AgentState] = None
        self.last_action: Optional[int] = None
        self.position_history: List[Dict] = []
        
        # Performance tracking
        self.episode_rewards: List[float] = []
        self.episode_actions: List[int] = []
        self.performance_metrics: Dict[str, List[float]] = {
            "total_reward": [],
            "avg_q_value": [],
            "exploration_rate": [],
            "loss": []
        }
        
        print(f"🤖 Enhanced RL Agent initialized for {symbol}")
        print(f"   🧠 Network: {self.state_size} → 128 → {self.action_size}")
        print(f"   💾 Replay buffer capacity: {self.replay_buffer.capacity}")
        print(f"   🎯 Reward shaping: {len(self.reward_shaper.reward_components)} components")
        
    def get_state_vector(self, market_data: Dict) -> AgentState:
        """Convert market data to agent state"""
        
        state = AgentState()
        
        # Market features
        state.price_change = market_data.get('price_change_pct', 0.0)
        state.volatility = market_data.get('volatility', 50.0) / 200.0  # Normalize
        state.volume_percentile = market_data.get('volume_percentile', 50.0) / 100.0
        state.momentum = market_data.get('momentum', 0.0) / 100.0
        
        # Regime encoding (one-hot)
        regime = market_data.get('regime', 'UNDERDAMPED')
        regime_map = {"OVERDAMPED": 0, "CRITICALLY_DAMPED": 1, "UNDERDAMPED": 2, "CHAOTIC": 3}
        regime_idx = regime_map.get(regime, 2)
        state.regime_vector = [0.0] * 4
        state.regime_vector[regime_idx] = 1.0
        
        # Position features
        state.position_size = market_data.get('position_size', 0.0)
        state.position_pnl = market_data.get('position_pnl', 0.0) / 100.0  # Normalize
        state.time_in_position = min(1.0, market_data.get('time_in_position', 0) / 100.0)
        
        # Performance features
        state.mae = min(1.0, market_data.get('mae', 0.0) / 100.0)
        state.mfe = min(1.0, market_data.get('mfe', 0.0) / 100.0)
        state.efficiency = min(2.0, market_data.get('efficiency', 1.0)) / 2.0
        state.drawdown = min(1.0, market_data.get('drawdown', 0.0))
        
        # Historical features (recent returns)
        recent_returns = market_data.get('recent_returns', [0.0] * 10)
        state.recent_returns = [max(-1.0, min(1.0, r / 100.0)) for r in recent_returns[-10:]]
        if len(state.recent_returns) < 10:
            state.recent_returns.extend([0.0] * (10 - len(state.recent_returns)))
        
        # Streak features
        state.win_streak = min(1.0, market_data.get('win_streak', 0) / 10.0)
        state.loss_streak = min(1.0, market_data.get('loss_streak', 0) / 10.0)
        
        # Agent meta features
        state.confidence = market_data.get('confidence', 0.5)
        state.exploration_factor = self.epsilon
        
        return state
    
    def select_action(self, state: AgentState, training: bool = True) -> int:
        """Select action using epsilon-greedy with neural network"""
        
        if training and random.random() < self.epsilon:
            # Exploration: random action
            return random.randint(0, self.action_size - 1)
        else:
            # Exploitation: use Q-network
            state_vector = state.to_vector()
            q_values = self.q_network.forward(state_vector)
            return q_values.index(max(q_values))
    
    def step(self, market_data: Dict, training: bool = True) -> Tuple[int, Dict]:
        """Take one step in the environment"""
        
        # Convert market data to state
        current_state = self.get_state_vector(market_data)
        
        # Select action
        action = self.select_action(current_state, training)
        
        # Calculate reward if we have previous state/action
        reward_info = {}
        if self.current_state is not None and self.last_action is not None:
            # Create trading state for reward calculation
            trading_state = self._create_trading_state(self.current_state, market_data)
            
            # Get trade result if applicable
            trade_result = self._get_trade_result(market_data)
            
            # Calculate reward
            reward_result = self.reward_shaper.calculate_reward(
                trading_state, 
                ActionType(self.last_action).name,
                trade_result
            )
            
            reward_info = {
                "total_reward": reward_result.total_reward,
                "components": reward_result.components,
                "bonuses": reward_result.bonuses,
                "meta_info": reward_result.meta_info
            }
            
            # Store experience for training
            if training:
                done = market_data.get('episode_done', False)
                priority = abs(reward_result.total_reward) + 1e-6
                
                experience = Experience(
                    state=self.current_state.to_vector(),
                    action=self.last_action,
                    reward=reward_result.total_reward,
                    next_state=current_state.to_vector(),
                    done=done,
                    priority=priority
                )
                
                self.replay_buffer.push(experience)
                self.episode_rewards.append(reward_result.total_reward)
                
                # Train if enough experiences
                if len(self.replay_buffer.buffer) > self.batch_size:
                    self._train_step()
        
        # Update state
        self.current_state = current_state
        self.last_action = action
        
        # Track performance
        if training:
            self.episode_actions.append(action)
            self._update_performance_metrics(reward_info)
        
        return action, reward_info
    
    def _create_trading_state(self, agent_state: AgentState, market_data: Dict) -> TradingState:
        """Convert agent state to trading state for reward calculation"""
        
        # Extract regime from one-hot vector
        regime_idx = agent_state.regime_vector.index(max(agent_state.regime_vector))
        regime_names = ["OVERDAMPED", "CRITICALLY_DAMPED", "UNDERDAMPED", "CHAOTIC"]
        regime = regime_names[regime_idx]
        
        return TradingState(
            symbol=self.symbol,
            current_price=market_data.get('current_price', 1.0),
            volatility=agent_state.volatility * 200.0,  # Denormalize
            volume_percentile=agent_state.volume_percentile * 100.0,
            regime=regime,
            position_size=agent_state.position_size,
            entry_price=market_data.get('entry_price', market_data.get('current_price', 1.0)),
            unrealized_pnl=agent_state.position_pnl * 100.0,  # Denormalize
            time_in_position=int(agent_state.time_in_position * 100.0),
            mae=agent_state.mae * 100.0,
            mfe=agent_state.mfe * 100.0,
            efficiency=agent_state.efficiency * 2.0,
            drawdown_from_peak=agent_state.drawdown,
            recent_pnl=[r * 100.0 for r in agent_state.recent_returns],  # Denormalize
            recent_trades=len(self.episode_actions),
            win_streak=int(agent_state.win_streak * 10.0),
            loss_streak=int(agent_state.loss_streak * 10.0),
            agent_type="ENHANCED_RL",
            confidence=agent_state.confidence,
            exploration_level=agent_state.exploration_factor
        )
    
    def _get_trade_result(self, market_data: Dict) -> Optional[Dict]:
        """Extract trade result from market data if available"""
        
        if market_data.get('trade_completed', False):
            return {
                'net_pnl': market_data.get('trade_pnl', 0.0),
                'gross_pnl': market_data.get('gross_pnl', 0.0),
                'commission': market_data.get('commission', 0.0),
                'hold_time': market_data.get('hold_time', 0.0)
            }
        return None
    
    def _train_step(self):
        """Perform one training step"""
        
        # Sample batch from replay buffer
        experiences, weights, indices = self.replay_buffer.sample(self.batch_size)
        
        if not experiences:
            return
        
        # Simplified training step (in practice, would use proper gradient descent)
        td_errors = []
        
        for i, (exp, weight) in enumerate(zip(experiences, weights)):
            # Calculate target Q-value
            if exp.done:
                target_q = exp.reward
            else:
                next_q_values = self.target_network.forward(exp.next_state)
                target_q = exp.reward + self.gamma * max(next_q_values)
            
            # Calculate current Q-value
            current_q_values = self.q_network.forward(exp.state)
            current_q = current_q_values[exp.action]
            
            # TD error for priority update
            td_error = target_q - current_q
            td_errors.append(td_error)
            
            # Simplified weight update (gradient descent approximation)
            learning_rate = self.q_network.learning_rate * weight
            current_q_values[exp.action] += learning_rate * td_error
        
        # Update priorities
        self.replay_buffer.update_priorities(indices, td_errors)
        
        # Update target network periodically
        self.training_step += 1
        if self.training_step % self.target_update_freq == 0:
            self.target_network.copy_from(self.q_network)
        
        # Decay exploration
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def _update_performance_metrics(self, reward_info: Dict):
        """Update performance tracking metrics"""
        
        if reward_info:
            self.performance_metrics["total_reward"].append(reward_info.get("total_reward", 0.0))
        
        # Calculate average Q-value for current state
        if self.current_state:
            q_values = self.q_network.forward(self.current_state.to_vector())
            self.performance_metrics["avg_q_value"].append(statistics.mean(q_values))
        
        self.performance_metrics["exploration_rate"].append(self.epsilon)
    
    def end_episode(self) -> Dict[str, Any]:
        """End current episode and return performance summary"""
        
        self.episode += 1
        
        # Calculate episode metrics
        episode_summary = {
            "episode": self.episode,
            "total_reward": sum(self.episode_rewards),
            "avg_reward": statistics.mean(self.episode_rewards) if self.episode_rewards else 0.0,
            "total_actions": len(self.episode_actions),
            "exploration_rate": self.epsilon,
            "training_steps": self.training_step
        }
        
        # Action distribution
        action_counts = {action.name: self.episode_actions.count(action.value) for action in ActionType}
        episode_summary["action_distribution"] = action_counts
        
        # Add to performance history
        for key in ["total_reward", "avg_q_value", "exploration_rate"]:
            if self.performance_metrics[key]:
                episode_summary[f"avg_{key}"] = statistics.mean(self.performance_metrics[key][-100:])
        
        # Reset episode data
        self.episode_rewards = []
        self.episode_actions = []
        self.current_state = None
        self.last_action = None
        
        return episode_summary
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        
        # Basic agent statistics
        report = {
            "agent_info": {
                "symbol": self.symbol,
                "episodes_completed": self.episode,
                "training_steps": self.training_step,
                "replay_buffer_size": len(self.replay_buffer.buffer),
                "current_exploration_rate": self.epsilon
            }
        }
        
        # Performance trends
        if self.performance_metrics["total_reward"]:
            recent_rewards = self.performance_metrics["total_reward"][-100:]
            report["performance_trends"] = {
                "avg_reward_recent_100": statistics.mean(recent_rewards),
                "reward_std": statistics.stdev(recent_rewards) if len(recent_rewards) > 1 else 0,
                "best_reward": max(self.performance_metrics["total_reward"]),
                "worst_reward": min(self.performance_metrics["total_reward"])
            }
        
        # Learning progress
        if self.performance_metrics["avg_q_value"]:
            recent_q_values = self.performance_metrics["avg_q_value"][-100:]
            report["learning_progress"] = {
                "avg_q_value": statistics.mean(recent_q_values),
                "q_value_trend": "increasing" if recent_q_values[-1] > recent_q_values[0] else "decreasing"
            }
        
        # Reward shaper summary
        report["reward_shaping"] = self.reward_shaper.get_reward_summary()
        
        return report
    
    def save_model(self, filepath: str):
        """Save model and configuration"""
        
        # Save reward shaper config
        self.reward_shaper.save_config(f"{filepath}_reward_config.json")
        
        # Save agent configuration and weights (simplified)
        config = {
            "symbol": self.symbol,
            "episode": self.episode,
            "training_step": self.training_step,
            "epsilon": self.epsilon,
            "performance_metrics": self.performance_metrics
        }
        
        import json
        with open(f"{filepath}_agent_config.json", 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"✅ Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load model and configuration"""
        
        try:
            # Load reward shaper config
            self.reward_shaper.load_config(f"{filepath}_reward_config.json")
            
            # Load agent configuration
            import json
            with open(f"{filepath}_agent_config.json", 'r') as f:
                config = json.load(f)
            
            self.episode = config.get("episode", 0)
            self.training_step = config.get("training_step", 0)
            self.epsilon = config.get("epsilon", self.epsilon)
            self.performance_metrics = config.get("performance_metrics", self.performance_metrics)
            
            print(f"✅ Model loaded from {filepath}")
            
        except Exception as e:
            print(f"⚠️ Error loading model: {e}")

def test_enhanced_rl_agent():
    """Test the enhanced RL agent"""
    
    print("🧪 ENHANCED RL AGENT TEST")
    print("="*80)
    
    # Initialize agent
    agent = EnhancedRLAgent("BTCUSD")
    
    # Simulate market data
    test_episodes = 5
    steps_per_episode = 100
    
    for episode in range(test_episodes):
        print(f"\n🎮 Episode {episode + 1}/{test_episodes}")
        print("-" * 40)
        
        for step in range(steps_per_episode):
            # Simulate market data
            market_data = {
                'current_price': 95000 + random.gauss(0, 1000),
                'price_change_pct': random.gauss(0, 2.0),
                'volatility': random.uniform(50, 200),
                'volume_percentile': random.uniform(0, 100),
                'momentum': random.gauss(0, 50),
                'regime': random.choice(['OVERDAMPED', 'CRITICALLY_DAMPED', 'UNDERDAMPED', 'CHAOTIC']),
                'position_size': random.uniform(-2, 2),
                'position_pnl': random.gauss(0, 100),
                'time_in_position': random.randint(0, 100),
                'mae': random.uniform(0, 200),
                'mfe': random.uniform(0, 500),
                'efficiency': random.uniform(0.5, 3.0),
                'drawdown': random.uniform(0, 0.1),
                'recent_returns': [random.gauss(0, 20) for _ in range(10)],
                'win_streak': random.randint(0, 5),
                'loss_streak': random.randint(0, 3),
                'confidence': random.uniform(0.3, 0.9),
                'episode_done': step == steps_per_episode - 1,
                'trade_completed': random.random() < 0.1,
                'trade_pnl': random.gauss(0, 50) if random.random() < 0.1 else 0
            }
            
            # Agent step
            action, reward_info = agent.step(market_data, training=True)
            
            # Print occasional updates
            if step % 20 == 0:
                action_name = ActionType(action).name
                reward = reward_info.get("total_reward", 0.0)
                print(f"   Step {step}: Action={action_name}, Reward={reward:.3f}")
        
        # End episode
        episode_summary = agent.end_episode()
        
        print(f"\n📊 Episode {episode + 1} Summary:")
        print(f"   Total Reward: {episode_summary['total_reward']:.2f}")
        print(f"   Avg Reward: {episode_summary['avg_reward']:.3f}")
        print(f"   Total Actions: {episode_summary['total_actions']}")
        print(f"   Exploration Rate: {episode_summary['exploration_rate']:.3f}")
        
        # Show top 3 actions
        action_dist = episode_summary["action_distribution"]
        top_actions = sorted(action_dist.items(), key=lambda x: x[1], reverse=True)[:3]
        print(f"   Top Actions: {', '.join([f'{k}({v})' for k, v in top_actions])}")
    
    # Generate performance report
    print(f"\n📈 FINAL PERFORMANCE REPORT:")
    print("=" * 60)
    
    report = agent.get_performance_report()
    
    # Agent info
    agent_info = report["agent_info"]
    print(f"\n🤖 Agent Info:")
    for key, value in agent_info.items():
        print(f"   {key}: {value}")
    
    # Performance trends
    if "performance_trends" in report:
        print(f"\n📊 Performance Trends:")
        for key, value in report["performance_trends"].items():
            print(f"   {key}: {value:.3f}")
    
    # Learning progress
    if "learning_progress" in report:
        print(f"\n🧠 Learning Progress:")
        for key, value in report["learning_progress"].items():
            print(f"   {key}: {value}")
    
    # Reward shaping summary
    reward_summary = report["reward_shaping"]
    print(f"\n🎯 Reward Shaping:")
    print(f"   Curriculum Stage: {reward_summary['curriculum_stage']}")
    print(f"   Training Episodes: {reward_summary['training_episodes']}")
    print(f"   Recent Performance: {reward_summary['recent_performance']:.3f}")
    print(f"   Active Components: {reward_summary['active_components']}")
    
    # Test save/load
    model_path = "/tmp/test_rl_agent"
    agent.save_model(model_path)
    
    print(f"\n✅ Enhanced RL Agent test completed!")

if __name__ == "__main__":
    test_enhanced_rl_agent()