#!/usr/bin/env python3
"""
PHASE 3: DUAL AGENT REINFORCEMENT LEARNING SYSTEM
BERSERKER (Aggressive) and SNIPER (Precision) agents with physics-based trading
"""

import numpy as np
import time
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import json
import logging
from pathlib import Path
import pickle
from concurrent.futures import ThreadPoolExecutor
import threading
import queue

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Action(Enum):
    """Trading actions"""
    HOLD = 0
    BUY = 1
    SELL = 2

class AgentType(Enum):
    """Agent personalities"""
    BERSERKER = "BERSERKER"  # Aggressive, high-frequency, volatility-seeking
    SNIPER = "SNIPER"        # Precision, patient, high-probability

@dataclass
class AgentConfig:
    """Agent configuration"""
    agent_type: AgentType
    learning_rate: float = 0.001
    epsilon: float = 0.1  # Exploration rate
    epsilon_decay: float = 0.995
    epsilon_min: float = 0.01
    gamma: float = 0.95  # Discount factor
    memory_size: int = 10000
    batch_size: int = 32
    update_frequency: int = 100
    
    # Physics thresholds (different for each agent type)
    momentum_threshold: float = 0.5
    volatility_threshold: float = 0.3
    liquidity_threshold: float = 0.2
    confidence_threshold: float = 0.7
    
    # Risk parameters
    max_position_size: float = 0.1
    stop_loss_pips: float = 20
    take_profit_pips: float = 40
    max_drawdown: float = 0.15

@dataclass
class State:
    """Market state representation"""
    price: float
    spread: float
    momentum: float
    acceleration: float
    volatility: float
    liquidity_score: float
    trend_strength: float
    position: float = 0  # Current position size
    unrealized_pnl: float = 0
    
    def to_vector(self) -> np.ndarray:
        """Convert state to neural network input vector"""
        return np.array([
            self.price,
            self.spread,
            self.momentum,
            self.acceleration,
            self.volatility,
            self.liquidity_score,
            self.trend_strength,
            self.position,
            self.unrealized_pnl
        ])

@dataclass 
class Experience:
    """Single experience for replay memory"""
    state: State
    action: Action
    reward: float
    next_state: State
    done: bool

class ReplayMemory:
    """Experience replay memory for stable learning"""
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.memory: List[Experience] = []
        self.position = 0
    
    def push(self, experience: Experience):
        """Store experience"""
        if len(self.memory) < self.capacity:
            self.memory.append(experience)
        else:
            self.memory[self.position] = experience
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int) -> List[Experience]:
        """Random sample for training"""
        import random
        return random.sample(self.memory, min(batch_size, len(self.memory)))
    
    def __len__(self) -> int:
        return len(self.memory)

class SimpleNeuralNetwork:
    """Simple neural network for Q-learning"""
    
    def __init__(self, input_size: int, hidden_size: int = 128, output_size: int = 3):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Initialize weights
        self.weights1 = np.random.randn(input_size, hidden_size) * 0.1
        self.bias1 = np.zeros((1, hidden_size))
        self.weights2 = np.random.randn(hidden_size, output_size) * 0.1
        self.bias2 = np.zeros((1, output_size))
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass"""
        # Hidden layer with ReLU
        hidden = np.maximum(0, np.dot(x, self.weights1) + self.bias1)
        # Output layer
        output = np.dot(hidden, self.weights2) + self.bias2
        return output
    
    def update_weights(self, gradients: Dict[str, np.ndarray], learning_rate: float):
        """Update weights with gradients"""
        self.weights1 -= learning_rate * gradients['w1']
        self.bias1 -= learning_rate * gradients['b1']
        self.weights2 -= learning_rate * gradients['w2']
        self.bias2 -= learning_rate * gradients['b2']

class RLAgent:
    """Base reinforcement learning agent"""
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.memory = ReplayMemory(config.memory_size)
        self.q_network = SimpleNeuralNetwork(input_size=9)  # State vector size
        self.target_network = SimpleNeuralNetwork(input_size=9)
        self.update_target_network()
        
        # Statistics
        self.total_trades = 0
        self.winning_trades = 0
        self.total_pnl = 0.0
        self.training_steps = 0
        
        # Thread safety
        self.lock = threading.Lock()
        
        logger.info(f"🤖 {config.agent_type.value} Agent initialized")
    
    def update_target_network(self):
        """Copy weights from Q-network to target network"""
        self.target_network.weights1 = self.q_network.weights1.copy()
        self.target_network.bias1 = self.q_network.bias1.copy()
        self.target_network.weights2 = self.q_network.weights2.copy()
        self.target_network.bias2 = self.q_network.bias2.copy()
    
    def act(self, state: State) -> Action:
        """Choose action using epsilon-greedy policy"""
        # Exploration vs exploitation
        if np.random.random() < self.config.epsilon:
            return np.random.choice(list(Action))
        
        # Get Q-values
        state_vector = state.to_vector().reshape(1, -1)
        q_values = self.q_network.forward(state_vector)
        
        # Apply agent-specific decision biases
        if self.config.agent_type == AgentType.BERSERKER:
            # BERSERKER: Bias towards action in volatile markets
            if state.volatility > self.config.volatility_threshold:
                q_values[0, Action.BUY.value] *= 1.2
                q_values[0, Action.SELL.value] *= 1.2
        else:  # SNIPER
            # SNIPER: Bias towards high-confidence setups
            confidence = abs(state.momentum) * state.trend_strength
            if confidence < self.config.confidence_threshold:
                q_values[0, Action.HOLD.value] *= 1.5
        
        # Choose action with highest Q-value
        return Action(np.argmax(q_values[0]))
    
    def remember(self, experience: Experience):
        """Store experience in replay memory"""
        with self.lock:
            self.memory.push(experience)
    
    def replay(self):
        """Train on batch from replay memory"""
        if len(self.memory) < self.config.batch_size:
            return
        
        with self.lock:
            batch = self.memory.sample(self.config.batch_size)
        
        # Prepare batch data
        states = np.array([exp.state.to_vector() for exp in batch])
        actions = np.array([exp.action.value for exp in batch])
        rewards = np.array([exp.reward for exp in batch])
        next_states = np.array([exp.next_state.to_vector() for exp in batch])
        dones = np.array([exp.done for exp in batch])
        
        # Current Q-values
        current_q = self.q_network.forward(states)
        
        # Next Q-values from target network
        next_q = self.target_network.forward(next_states)
        max_next_q = np.max(next_q, axis=1)
        
        # Calculate targets
        targets = current_q.copy()
        for i in range(len(batch)):
            if dones[i]:
                targets[i, actions[i]] = rewards[i]
            else:
                targets[i, actions[i]] = rewards[i] + self.config.gamma * max_next_q[i]
        
        # Simple gradient calculation (simplified for demonstration)
        # In production, use proper backpropagation
        loss = np.mean((targets - current_q) ** 2)
        
        # Update epsilon
        self.config.epsilon = max(self.config.epsilon_min, 
                                 self.config.epsilon * self.config.epsilon_decay)
        
        self.training_steps += 1
        
        # Update target network periodically
        if self.training_steps % self.config.update_frequency == 0:
            self.update_target_network()
    
    def calculate_reward(self, state: State, action: Action, 
                        next_state: State, pnl: float) -> float:
        """Calculate shaped reward based on agent personality"""
        base_reward = pnl * 100  # Scale PnL
        
        if self.config.agent_type == AgentType.BERSERKER:
            # BERSERKER rewards
            # Reward volatility capture
            volatility_reward = state.volatility * 10 if action != Action.HOLD else 0
            
            # Reward momentum following
            momentum_reward = abs(state.momentum) * 5 if action != Action.HOLD else -2
            
            # Penalize inaction in volatile markets
            inaction_penalty = -5 if action == Action.HOLD and state.volatility > 0.5 else 0
            
            shaped_reward = base_reward + volatility_reward + momentum_reward + inaction_penalty
            
        else:  # SNIPER
            # SNIPER rewards
            # Reward precision (high confidence trades)
            confidence = abs(state.momentum) * state.trend_strength
            precision_reward = confidence * 20 if action != Action.HOLD else 0
            
            # Reward patience
            patience_reward = 2 if action == Action.HOLD and confidence < 0.7 else 0
            
            # Penalize low-quality trades
            quality_penalty = -10 if action != Action.HOLD and confidence < 0.5 else 0
            
            shaped_reward = base_reward + precision_reward + patience_reward + quality_penalty
        
        return shaped_reward

class BerserkerAgent(RLAgent):
    """Aggressive trading agent for volatile markets"""
    
    def __init__(self):
        config = AgentConfig(
            agent_type=AgentType.BERSERKER,
            learning_rate=0.002,
            epsilon=0.15,
            momentum_threshold=0.3,
            volatility_threshold=0.2,  # Lower threshold - seeks volatility
            confidence_threshold=0.5,  # Lower confidence required
            max_position_size=0.15,    # Larger positions
            stop_loss_pips=30,
            take_profit_pips=50
        )
        super().__init__(config)
        logger.info("⚔️ BERSERKER Agent: Volatility Hunter Activated")

class SniperAgent(RLAgent):
    """Precision trading agent for high-probability setups"""
    
    def __init__(self):
        config = AgentConfig(
            agent_type=AgentType.SNIPER,
            learning_rate=0.001,
            epsilon=0.05,  # Less exploration
            momentum_threshold=0.6,     # Higher threshold
            volatility_threshold=0.4,
            confidence_threshold=0.8,   # High confidence required
            max_position_size=0.08,     # Smaller, precise positions
            stop_loss_pips=15,          # Tighter stops
            take_profit_pips=30
        )
        super().__init__(config)
        logger.info("🎯 SNIPER Agent: Precision Trader Activated")

class DualAgentCoordinator:
    """Coordinates BERSERKER and SNIPER agents"""
    
    def __init__(self):
        self.berserker = BerserkerAgent()
        self.sniper = SniperAgent()
        self.market_regime = "NEUTRAL"  # VOLATILE, TRENDING, RANGING, NEUTRAL
        self.active_positions: Dict[str, Any] = {}
        
        logger.info("🎮 Dual Agent Coordinator initialized")
    
    def detect_market_regime(self, state: State) -> str:
        """Detect current market regime"""
        if state.volatility > 0.5:
            return "VOLATILE"
        elif state.trend_strength > 0.7 and abs(state.momentum) > 0.4:
            return "TRENDING"
        elif state.volatility < 0.2 and state.trend_strength < 0.3:
            return "RANGING"
        else:
            return "NEUTRAL"
    
    def select_agent(self, state: State) -> RLAgent:
        """Select appropriate agent based on market conditions"""
        regime = self.detect_market_regime(state)
        self.market_regime = regime
        
        if regime == "VOLATILE":
            logger.debug("Market regime: VOLATILE - Selecting BERSERKER")
            return self.berserker
        elif regime == "TRENDING" and abs(state.momentum) > 0.6:
            logger.debug("Market regime: TRENDING - Selecting SNIPER")
            return self.sniper
        elif regime == "RANGING":
            # Both agents can work in ranging markets
            if state.liquidity_score > 0.7:
                return self.sniper  # High liquidity favors precision
            else:
                return self.berserker  # Low liquidity needs aggressive fills
        else:
            # Neutral regime - choose based on confidence
            confidence = abs(state.momentum) * state.trend_strength
            return self.sniper if confidence > 0.6 else self.berserker
    
    def process_tick(self, state: State, symbol: str, 
                    timeframe: str) -> Tuple[Action, str]:
        """Process market tick and return action with agent name"""
        # Select appropriate agent
        agent = self.select_agent(state)
        
        # Get action
        action = agent.act(state)
        
        # Log decision
        agent_name = agent.config.agent_type.value
        logger.info(f"📊 {symbol}/{timeframe} - {agent_name} decides: {action.name}")
        
        return action, agent_name
    
    def train_agents(self, experience: Experience, agent_name: str):
        """Train specific agent with experience"""
        agent = self.berserker if agent_name == "BERSERKER" else self.sniper
        agent.remember(experience)
        agent.replay()
    
    def save_models(self, path: str = "models/"):
        """Save both agent models"""
        Path(path).mkdir(exist_ok=True)
        
        # Save BERSERKER
        with open(f"{path}/berserker_model.pkl", 'wb') as f:
            pickle.dump({
                'q_network': self.berserker.q_network,
                'config': self.berserker.config,
                'stats': {
                    'total_trades': self.berserker.total_trades,
                    'winning_trades': self.berserker.winning_trades,
                    'total_pnl': self.berserker.total_pnl
                }
            }, f)
        
        # Save SNIPER
        with open(f"{path}/sniper_model.pkl", 'wb') as f:
            pickle.dump({
                'q_network': self.sniper.q_network,
                'config': self.sniper.config,
                'stats': {
                    'total_trades': self.sniper.total_trades,
                    'winning_trades': self.sniper.winning_trades,
                    'total_pnl': self.sniper.total_pnl
                }
            }, f)
        
        logger.info("💾 Models saved successfully")
    
    def load_models(self, path: str = "models/"):
        """Load both agent models"""
        try:
            # Load BERSERKER
            with open(f"{path}/berserker_model.pkl", 'rb') as f:
                berserker_data = pickle.load(f)
                self.berserker.q_network = berserker_data['q_network']
                self.berserker.config = berserker_data['config']
                self.berserker.total_trades = berserker_data['stats']['total_trades']
                self.berserker.winning_trades = berserker_data['stats']['winning_trades']
                self.berserker.total_pnl = berserker_data['stats']['total_pnl']
            
            # Load SNIPER
            with open(f"{path}/sniper_model.pkl", 'rb') as f:
                sniper_data = pickle.load(f)
                self.sniper.q_network = sniper_data['q_network']
                self.sniper.config = sniper_data['config']
                self.sniper.total_trades = sniper_data['stats']['total_trades']
                self.sniper.winning_trades = sniper_data['stats']['winning_trades']
                self.sniper.total_pnl = sniper_data['stats']['total_pnl']
            
            logger.info("💾 Models loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get combined statistics from both agents"""
        berserker_stats = {
            'total_trades': self.berserker.total_trades,
            'winning_trades': self.berserker.winning_trades,
            'win_rate': self.berserker.winning_trades / max(self.berserker.total_trades, 1),
            'total_pnl': self.berserker.total_pnl,
            'epsilon': self.berserker.config.epsilon,
            'training_steps': self.berserker.training_steps
        }
        
        sniper_stats = {
            'total_trades': self.sniper.total_trades,
            'winning_trades': self.sniper.winning_trades,
            'win_rate': self.sniper.winning_trades / max(self.sniper.total_trades, 1),
            'total_pnl': self.sniper.total_pnl,
            'epsilon': self.sniper.config.epsilon,
            'training_steps': self.sniper.training_steps
        }
        
        return {
            'market_regime': self.market_regime,
            'berserker': berserker_stats,
            'sniper': sniper_stats,
            'combined': {
                'total_trades': berserker_stats['total_trades'] + sniper_stats['total_trades'],
                'total_pnl': berserker_stats['total_pnl'] + sniper_stats['total_pnl'],
                'active_positions': len(self.active_positions)
            }
        }

# Shadow agents for continuous learning
class ShadowTrader:
    """Manages shadow agents that trade virtually for accelerated learning"""
    
    def __init__(self):
        self.shadow_berserker = BerserkerAgent()
        self.shadow_sniper = SniperAgent()
        self.virtual_balance = 10000.0
        self.virtual_positions: Dict[str, Any] = {}
        
        logger.info("👥 Shadow Trader initialized for virtual trading")
    
    def process_virtual_tick(self, state: State, symbol: str) -> Tuple[Action, float]:
        """Process tick with shadow agents"""
        # Randomly choose shadow agent
        agent = self.shadow_berserker if np.random.random() > 0.5 else self.shadow_sniper
        
        # Get action
        action = agent.act(state)
        
        # Simulate virtual trade
        virtual_pnl = self._simulate_trade(state, action, symbol)
        
        # Create virtual experience
        next_state = state  # Simplified - in reality would be next tick
        experience = Experience(
            state=state,
            action=action,
            reward=agent.calculate_reward(state, action, next_state, virtual_pnl),
            next_state=next_state,
            done=False
        )
        
        # Train shadow agent
        agent.remember(experience)
        agent.replay()
        
        return action, virtual_pnl
    
    def _simulate_trade(self, state: State, action: Action, symbol: str) -> float:
        """Simulate virtual trade execution"""
        if action == Action.HOLD:
            return 0.0
        
        # Simple PnL simulation based on momentum
        position_size = 0.1
        if action == Action.BUY:
            # Assume price moves with momentum
            price_change = state.momentum * 0.0001
            return position_size * price_change * 10000  # Pip value
        else:  # SELL
            price_change = -state.momentum * 0.0001
            return position_size * price_change * 10000
    
    def transfer_learning(self, main_coordinator: DualAgentCoordinator, 
                         transfer_rate: float = 0.1):
        """Transfer shadow learning to main agents"""
        # Transfer BERSERKER learning
        if self.shadow_berserker.training_steps > 1000:
            # Blend weights (simplified)
            main_coordinator.berserker.q_network.weights1 = (
                (1 - transfer_rate) * main_coordinator.berserker.q_network.weights1 +
                transfer_rate * self.shadow_berserker.q_network.weights1
            )
            logger.info("🔄 Transferred shadow BERSERKER learning")
        
        # Transfer SNIPER learning
        if self.shadow_sniper.training_steps > 1000:
            main_coordinator.sniper.q_network.weights1 = (
                (1 - transfer_rate) * main_coordinator.sniper.q_network.weights1 +
                transfer_rate * self.shadow_sniper.q_network.weights1
            )
            logger.info("🔄 Transferred shadow SNIPER learning")

# Main execution for testing
if __name__ == "__main__":
    print("🧪 TESTING PHASE 3: DUAL AGENT SYSTEM")
    print("=" * 50)
    
    # Create coordinator
    coordinator = DualAgentCoordinator()
    shadow_trader = ShadowTrader()
    
    # Create test state
    test_state = State(
        price=1.1750,
        spread=0.0001,
        momentum=0.5,
        acceleration=0.1,
        volatility=0.4,
        liquidity_score=0.8,
        trend_strength=0.6
    )
    
    # Test agent selection
    agent = coordinator.select_agent(test_state)
    print(f"Selected agent: {agent.config.agent_type.value}")
    print(f"Market regime: {coordinator.market_regime}")
    
    # Test action selection
    action, agent_name = coordinator.process_tick(test_state, "EURUSD", "M5")
    print(f"Action: {action.name} by {agent_name}")
    
    # Test shadow trading
    shadow_action, virtual_pnl = shadow_trader.process_virtual_tick(test_state, "EURUSD")
    print(f"Shadow action: {shadow_action.name}, Virtual PnL: ${virtual_pnl:.2f}")
    
    # Get statistics
    stats = coordinator.get_statistics()
    print(f"\nStatistics: {json.dumps(stats, indent=2)}")
    
    print("\n✅ Phase 3 Dual Agent System ready for integration tests")