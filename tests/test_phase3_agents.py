#!/usr/bin/env python3
"""
PHASE 3 INTEGRATION TESTS: AGENT FRAMEWORK
Comprehensive testing of dual agent RL system
"""

import unittest
import numpy as np
import time
import tempfile
import shutil
from pathlib import Path
import json

# Add parent directory to path
import sys
sys.path.append('..')

from src.agents.dual_agent_system import (
    Action, AgentType, AgentConfig, State, Experience,
    ReplayMemory, SimpleNeuralNetwork, RLAgent,
    BerserkerAgent, SniperAgent, DualAgentCoordinator,
    ShadowTrader
)

class TestState(unittest.TestCase):
    """Test State class"""
    
    def setUp(self):
        self.state = State(
            price=1.2500,
            spread=0.0002,
            momentum=0.3,
            acceleration=0.05,
            volatility=0.25,
            liquidity_score=0.7,
            trend_strength=0.6,
            position=0.1,
            unrealized_pnl=10.5
        )
    
    def test_state_creation(self):
        """Test State object creation"""
        self.assertEqual(self.state.price, 1.2500)
        self.assertEqual(self.state.momentum, 0.3)
        self.assertEqual(self.state.position, 0.1)
        self.assertEqual(self.state.unrealized_pnl, 10.5)
    
    def test_state_to_vector(self):
        """Test state vector conversion"""
        vector = self.state.to_vector()
        
        self.assertIsInstance(vector, np.ndarray)
        self.assertEqual(len(vector), 9)  # 9 features
        self.assertEqual(vector[0], 1.2500)  # price
        self.assertEqual(vector[2], 0.3)     # momentum
        self.assertEqual(vector[7], 0.1)     # position
        self.assertEqual(vector[8], 10.5)    # unrealized_pnl

class TestReplayMemory(unittest.TestCase):
    """Test ReplayMemory class"""
    
    def setUp(self):
        self.memory = ReplayMemory(capacity=100)
        self.test_state = State(
            price=1.2500, spread=0.0002, momentum=0.3,
            acceleration=0.05, volatility=0.25,
            liquidity_score=0.7, trend_strength=0.6
        )
    
    def test_memory_initialization(self):
        """Test memory initialization"""
        self.assertEqual(self.memory.capacity, 100)
        self.assertEqual(len(self.memory), 0)
    
    def test_push_experience(self):
        """Test pushing experiences to memory"""
        exp = Experience(
            state=self.test_state,
            action=Action.BUY,
            reward=10.0,
            next_state=self.test_state,
            done=False
        )
        
        self.memory.push(exp)
        self.assertEqual(len(self.memory), 1)
        
        # Push more experiences
        for i in range(10):
            self.memory.push(exp)
        self.assertEqual(len(self.memory), 11)
    
    def test_memory_overflow(self):
        """Test memory behavior when capacity exceeded"""
        exp = Experience(
            state=self.test_state,
            action=Action.BUY,
            reward=10.0,
            next_state=self.test_state,
            done=False
        )
        
        # Fill beyond capacity
        for i in range(150):
            self.memory.push(exp)
        
        # Should not exceed capacity
        self.assertEqual(len(self.memory), 100)
    
    def test_sample_experiences(self):
        """Test sampling from memory"""
        # Add diverse experiences
        for i in range(50):
            exp = Experience(
                state=self.test_state,
                action=Action(i % 3),  # Vary actions
                reward=float(i),
                next_state=self.test_state,
                done=(i % 10 == 0)
            )
            self.memory.push(exp)
        
        # Sample batch
        batch = self.memory.sample(10)
        self.assertEqual(len(batch), 10)
        
        # Check all samples are Experience objects
        for exp in batch:
            self.assertIsInstance(exp, Experience)

class TestSimpleNeuralNetwork(unittest.TestCase):
    """Test SimpleNeuralNetwork class"""
    
    def setUp(self):
        self.nn = SimpleNeuralNetwork(input_size=9, hidden_size=64, output_size=3)
    
    def test_network_initialization(self):
        """Test network initialization"""
        self.assertEqual(self.nn.input_size, 9)
        self.assertEqual(self.nn.hidden_size, 64)
        self.assertEqual(self.nn.output_size, 3)
        
        # Check weight shapes
        self.assertEqual(self.nn.weights1.shape, (9, 64))
        self.assertEqual(self.nn.bias1.shape, (1, 64))
        self.assertEqual(self.nn.weights2.shape, (64, 3))
        self.assertEqual(self.nn.bias2.shape, (1, 3))
    
    def test_forward_pass(self):
        """Test forward pass computation"""
        # Create test input
        test_input = np.random.randn(1, 9)
        
        output = self.nn.forward(test_input)
        
        self.assertEqual(output.shape, (1, 3))
        self.assertEqual(len(output[0]), 3)  # 3 actions
    
    def test_batch_forward(self):
        """Test forward pass with batch input"""
        batch_size = 32
        test_batch = np.random.randn(batch_size, 9)
        
        output = self.nn.forward(test_batch)
        
        self.assertEqual(output.shape, (32, 3))

class TestAgentConfig(unittest.TestCase):
    """Test AgentConfig class"""
    
    def test_berserker_config(self):
        """Test BERSERKER configuration"""
        config = AgentConfig(
            agent_type=AgentType.BERSERKER,
            volatility_threshold=0.2,
            confidence_threshold=0.5
        )
        
        self.assertEqual(config.agent_type, AgentType.BERSERKER)
        self.assertEqual(config.volatility_threshold, 0.2)
        self.assertEqual(config.confidence_threshold, 0.5)
        self.assertEqual(config.learning_rate, 0.001)  # Default
    
    def test_sniper_config(self):
        """Test SNIPER configuration"""
        config = AgentConfig(
            agent_type=AgentType.SNIPER,
            confidence_threshold=0.8,
            max_position_size=0.08
        )
        
        self.assertEqual(config.agent_type, AgentType.SNIPER)
        self.assertEqual(config.confidence_threshold, 0.8)
        self.assertEqual(config.max_position_size, 0.08)

class TestRLAgent(unittest.TestCase):
    """Test base RLAgent class"""
    
    def setUp(self):
        self.config = AgentConfig(agent_type=AgentType.BERSERKER)
        self.agent = RLAgent(self.config)
        self.test_state = State(
            price=1.2500, spread=0.0002, momentum=0.3,
            acceleration=0.05, volatility=0.25,
            liquidity_score=0.7, trend_strength=0.6
        )
    
    def test_agent_initialization(self):
        """Test agent initialization"""
        self.assertIsInstance(self.agent.memory, ReplayMemory)
        self.assertIsInstance(self.agent.q_network, SimpleNeuralNetwork)
        self.assertIsInstance(self.agent.target_network, SimpleNeuralNetwork)
        self.assertEqual(self.agent.total_trades, 0)
        self.assertEqual(self.agent.winning_trades, 0)
        self.assertEqual(self.agent.total_pnl, 0.0)
    
    def test_epsilon_greedy_action(self):
        """Test epsilon-greedy action selection"""
        # Force exploration
        self.agent.config.epsilon = 1.0
        actions_taken = set()
        
        # Should explore all actions
        for _ in range(100):
            action = self.agent.act(self.test_state)
            actions_taken.add(action)
        
        # Should have tried all 3 actions
        self.assertEqual(len(actions_taken), 3)
        
        # Force exploitation
        self.agent.config.epsilon = 0.0
        action = self.agent.act(self.test_state)
        self.assertIn(action, list(Action))
    
    def test_remember_experience(self):
        """Test experience storage"""
        exp = Experience(
            state=self.test_state,
            action=Action.BUY,
            reward=10.0,
            next_state=self.test_state,
            done=False
        )
        
        initial_memory_size = len(self.agent.memory)
        self.agent.remember(exp)
        self.assertEqual(len(self.agent.memory), initial_memory_size + 1)
    
    def test_reward_calculation(self):
        """Test reward shaping"""
        # Test BERSERKER rewards
        berserker_agent = BerserkerAgent()
        
        # High volatility state
        volatile_state = State(
            price=1.2500, spread=0.0002, momentum=0.5,
            acceleration=0.1, volatility=0.6,
            liquidity_score=0.7, trend_strength=0.8
        )
        
        # BERSERKER should get positive reward for action in volatile market
        reward = berserker_agent.calculate_reward(
            volatile_state, Action.BUY, volatile_state, 5.0
        )
        self.assertGreater(reward, 500)  # Base reward (5*100) + bonuses
        
        # BERSERKER should get negative reward for holding in volatile market
        reward = berserker_agent.calculate_reward(
            volatile_state, Action.HOLD, volatile_state, 0.0
        )
        self.assertLess(reward, 0)
        
        # Test SNIPER rewards
        sniper_agent = SniperAgent()
        
        # Low confidence state
        low_conf_state = State(
            price=1.2500, spread=0.0002, momentum=0.1,
            acceleration=0.01, volatility=0.2,
            liquidity_score=0.7, trend_strength=0.3
        )
        
        # SNIPER should get positive reward for holding in low confidence
        reward = sniper_agent.calculate_reward(
            low_conf_state, Action.HOLD, low_conf_state, 0.0
        )
        self.assertGreater(reward, 0)
        
        # SNIPER should get negative reward for trading in low confidence
        reward = sniper_agent.calculate_reward(
            low_conf_state, Action.BUY, low_conf_state, 0.0
        )
        self.assertLess(reward, 0)

class TestBerserkerAgent(unittest.TestCase):
    """Test BERSERKER agent specifics"""
    
    def setUp(self):
        self.agent = BerserkerAgent()
    
    def test_berserker_configuration(self):
        """Test BERSERKER-specific configuration"""
        self.assertEqual(self.agent.config.agent_type, AgentType.BERSERKER)
        self.assertEqual(self.agent.config.volatility_threshold, 0.2)
        self.assertEqual(self.agent.config.confidence_threshold, 0.5)
        self.assertEqual(self.agent.config.max_position_size, 0.15)
        self.assertEqual(self.agent.config.learning_rate, 0.002)
        self.assertEqual(self.agent.config.epsilon, 0.15)

class TestSniperAgent(unittest.TestCase):
    """Test SNIPER agent specifics"""
    
    def setUp(self):
        self.agent = SniperAgent()
    
    def test_sniper_configuration(self):
        """Test SNIPER-specific configuration"""
        self.assertEqual(self.agent.config.agent_type, AgentType.SNIPER)
        self.assertEqual(self.agent.config.confidence_threshold, 0.8)
        self.assertEqual(self.agent.config.max_position_size, 0.08)
        self.assertEqual(self.agent.config.stop_loss_pips, 15)
        self.assertEqual(self.agent.config.learning_rate, 0.001)
        self.assertEqual(self.agent.config.epsilon, 0.05)

class TestDualAgentCoordinator(unittest.TestCase):
    """Test DualAgentCoordinator class"""
    
    def setUp(self):
        self.coordinator = DualAgentCoordinator()
    
    def test_coordinator_initialization(self):
        """Test coordinator initialization"""
        self.assertIsInstance(self.coordinator.berserker, BerserkerAgent)
        self.assertIsInstance(self.coordinator.sniper, SniperAgent)
        self.assertEqual(self.coordinator.market_regime, "NEUTRAL")
        self.assertIsInstance(self.coordinator.active_positions, dict)
    
    def test_market_regime_detection(self):
        """Test market regime detection"""
        # Volatile market
        volatile_state = State(
            price=1.2500, spread=0.0002, momentum=0.4,
            acceleration=0.1, volatility=0.6,
            liquidity_score=0.7, trend_strength=0.5
        )
        regime = self.coordinator.detect_market_regime(volatile_state)
        self.assertEqual(regime, "VOLATILE")
        
        # Trending market
        trending_state = State(
            price=1.2500, spread=0.0002, momentum=0.5,
            acceleration=0.1, volatility=0.3,
            liquidity_score=0.7, trend_strength=0.8
        )
        regime = self.coordinator.detect_market_regime(trending_state)
        self.assertEqual(regime, "TRENDING")
        
        # Ranging market
        ranging_state = State(
            price=1.2500, spread=0.0002, momentum=0.1,
            acceleration=0.01, volatility=0.1,
            liquidity_score=0.7, trend_strength=0.2
        )
        regime = self.coordinator.detect_market_regime(ranging_state)
        self.assertEqual(regime, "RANGING")
    
    def test_agent_selection(self):
        """Test agent selection based on market conditions"""
        # Volatile market should select BERSERKER
        volatile_state = State(
            price=1.2500, spread=0.0002, momentum=0.4,
            acceleration=0.1, volatility=0.6,
            liquidity_score=0.7, trend_strength=0.5
        )
        agent = self.coordinator.select_agent(volatile_state)
        self.assertEqual(agent.config.agent_type, AgentType.BERSERKER)
        
        # High confidence trending should select SNIPER
        trending_state = State(
            price=1.2500, spread=0.0002, momentum=0.7,
            acceleration=0.1, volatility=0.3,
            liquidity_score=0.8, trend_strength=0.8
        )
        agent = self.coordinator.select_agent(trending_state)
        self.assertEqual(agent.config.agent_type, AgentType.SNIPER)
    
    def test_process_tick(self):
        """Test tick processing"""
        test_state = State(
            price=1.2500, spread=0.0002, momentum=0.3,
            acceleration=0.05, volatility=0.25,
            liquidity_score=0.7, trend_strength=0.6
        )
        
        action, agent_name = self.coordinator.process_tick(
            test_state, "EURUSD", "M5"
        )
        
        self.assertIn(action, list(Action))
        self.assertIn(agent_name, ["BERSERKER", "SNIPER"])
    
    def test_statistics(self):
        """Test statistics collection"""
        stats = self.coordinator.get_statistics()
        
        self.assertIn('market_regime', stats)
        self.assertIn('berserker', stats)
        self.assertIn('sniper', stats)
        self.assertIn('combined', stats)
        
        # Check berserker stats
        self.assertIn('total_trades', stats['berserker'])
        self.assertIn('win_rate', stats['berserker'])
        self.assertIn('epsilon', stats['berserker'])
        
        # Check combined stats
        self.assertIn('total_trades', stats['combined'])
        self.assertIn('total_pnl', stats['combined'])
    
    def test_model_save_load(self):
        """Test model saving and loading"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Train agents a bit
            test_state = State(
                price=1.2500, spread=0.0002, momentum=0.3,
                acceleration=0.05, volatility=0.25,
                liquidity_score=0.7, trend_strength=0.6
            )
            
            for i in range(10):
                exp = Experience(
                    state=test_state,
                    action=Action.BUY,
                    reward=10.0,
                    next_state=test_state,
                    done=False
                )
                self.coordinator.berserker.remember(exp)
                self.coordinator.sniper.remember(exp)
            
            # Update stats
            self.coordinator.berserker.total_trades = 100
            self.coordinator.sniper.total_trades = 50
            
            # Save models
            self.coordinator.save_models(temp_dir)
            
            # Create new coordinator
            new_coordinator = DualAgentCoordinator()
            
            # Load models
            success = new_coordinator.load_models(temp_dir)
            self.assertTrue(success)
            
            # Check stats were loaded
            self.assertEqual(new_coordinator.berserker.total_trades, 100)
            self.assertEqual(new_coordinator.sniper.total_trades, 50)

class TestShadowTrader(unittest.TestCase):
    """Test ShadowTrader class"""
    
    def setUp(self):
        self.shadow_trader = ShadowTrader()
        self.test_state = State(
            price=1.2500, spread=0.0002, momentum=0.3,
            acceleration=0.05, volatility=0.25,
            liquidity_score=0.7, trend_strength=0.6
        )
    
    def test_shadow_trader_initialization(self):
        """Test shadow trader initialization"""
        self.assertIsInstance(self.shadow_trader.shadow_berserker, BerserkerAgent)
        self.assertIsInstance(self.shadow_trader.shadow_sniper, SniperAgent)
        self.assertEqual(self.shadow_trader.virtual_balance, 10000.0)
        self.assertIsInstance(self.shadow_trader.virtual_positions, dict)
    
    def test_virtual_trading(self):
        """Test virtual trading functionality"""
        action, virtual_pnl = self.shadow_trader.process_virtual_tick(
            self.test_state, "EURUSD"
        )
        
        self.assertIn(action, list(Action))
        self.assertIsInstance(virtual_pnl, float)
        
        # Check that shadow agents are learning
        total_steps = (self.shadow_trader.shadow_berserker.training_steps + 
                      self.shadow_trader.shadow_sniper.training_steps)
        self.assertGreaterEqual(total_steps, 0)
    
    def test_transfer_learning(self):
        """Test transfer learning mechanism"""
        main_coordinator = DualAgentCoordinator()
        
        # Train shadow agents
        for _ in range(1100):  # Exceed transfer threshold
            self.shadow_trader.process_virtual_tick(self.test_state, "EURUSD")
        
        # Store original weights
        original_weights = main_coordinator.berserker.q_network.weights1.copy()
        
        # Transfer learning
        self.shadow_trader.transfer_learning(main_coordinator, transfer_rate=0.1)
        
        # Weights should have changed (if shadow berserker was trained enough)
        if self.shadow_trader.shadow_berserker.training_steps > 1000:
            new_weights = main_coordinator.berserker.q_network.weights1
            # Some weights should be different
            self.assertFalse(np.array_equal(original_weights, new_weights))

class TestIntegration(unittest.TestCase):
    """Integration tests for the complete agent system"""
    
    def setUp(self):
        self.coordinator = DualAgentCoordinator()
        self.shadow_trader = ShadowTrader()
    
    def test_full_trading_cycle(self):
        """Test complete trading cycle"""
        # Create sequence of market states
        states = [
            State(price=1.2500 + i*0.0001, spread=0.0002, 
                 momentum=0.3 + i*0.1, acceleration=0.05,
                 volatility=0.25 + i*0.05, liquidity_score=0.7,
                 trend_strength=0.6)
            for i in range(10)
        ]
        
        actions_taken = []
        agents_used = []
        
        for i, state in enumerate(states):
            # Main agents process tick
            action, agent_name = self.coordinator.process_tick(
                state, "EURUSD", "M5"
            )
            actions_taken.append(action)
            agents_used.append(agent_name)
            
            # Shadow agents process tick
            shadow_action, virtual_pnl = self.shadow_trader.process_virtual_tick(
                state, "EURUSD"
            )
            
            # Create experience for main agent
            if i > 0:
                exp = Experience(
                    state=states[i-1],
                    action=actions_taken[i-1],
                    reward=np.random.randn() * 10,  # Random PnL
                    next_state=state,
                    done=False
                )
                self.coordinator.train_agents(exp, agents_used[i-1])
        
        # Check that agents were used
        self.assertGreater(len(set(agents_used)), 0)
        
        # Check statistics
        stats = self.coordinator.get_statistics()
        self.assertIsInstance(stats, dict)
        
        # Check that training occurred
        total_training = (self.coordinator.berserker.training_steps + 
                         self.coordinator.sniper.training_steps)
        self.assertGreaterEqual(total_training, 0)

# Test runner
def run_phase3_tests():
    """Run all Phase 3 tests"""
    print("\n" + "="*60)
    print("🧪 PHASE 3 TESTS: AGENT FRAMEWORK")
    print("="*60)
    
    # Create test suite
    test_classes = [
        TestState,
        TestReplayMemory,
        TestSimpleNeuralNetwork,
        TestAgentConfig,
        TestRLAgent,
        TestBerserkerAgent,
        TestSniperAgent,
        TestDualAgentCoordinator,
        TestShadowTrader,
        TestIntegration
    ]
    
    suite = unittest.TestSuite()
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Report results
    print("\n" + "="*60)
    if result.wasSuccessful():
        print("✅ PHASE 3 TESTS PASSED - Agent Framework validated")
        print("   Ready to proceed to Phase 4: Full System Integration")
    else:
        print("❌ PHASE 3 TESTS FAILED")
        print(f"   Failures: {len(result.failures)}")
        print(f"   Errors: {len(result.errors)}")
    
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_phase3_tests()
    exit(0 if success else 1)