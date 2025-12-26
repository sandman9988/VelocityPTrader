#!/usr/bin/env python3
"""
PHASE 4 INTEGRATION TESTS: FULL SYSTEM INTEGRATION
End-to-end testing of complete VelocityTrader system
"""

import unittest
import asyncio
import tempfile
import shutil
import json
import time
import threading
from pathlib import Path
from unittest.mock import patch, MagicMock
import subprocess
import sys

# Add parent directory to path
sys.path.append('..')

# Import all components
from src.core.data_pipeline import DataPipeline, MarketData
from src.agents.dual_agent_system import DualAgentCoordinator, State, Action
from src.core.integrated_system import VelocityTradingSystem, SystemConfig, TradingEngine
from src.utils.logging_system import ComprehensiveLogger
from src.utils.performance_dashboard import PerformanceDashboard

class TestSystemConfig(unittest.TestCase):
    """Test SystemConfig class"""
    
    def test_default_config(self):
        """Test default configuration"""
        config = SystemConfig()
        
        self.assertEqual(config.mt5_login, 10916362)
        self.assertEqual(config.mt5_server, "VantageInternational-Demo")
        self.assertIsInstance(config.symbols, list)
        self.assertGreater(len(config.symbols), 5)
        self.assertIn("EURUSD+", config.symbols)
        self.assertIn("BTCUSD+", config.symbols)
    
    def test_config_from_dict(self):
        """Test configuration from dictionary"""
        config_dict = {
            "mt5_login": 12345,
            "mt5_server": "TestServer",
            "symbols": ["EURUSD+", "GBPUSD+"],
            "max_positions": 5,
            "risk_per_trade": 0.01
        }
        
        config = SystemConfig(**config_dict)
        
        self.assertEqual(config.mt5_login, 12345)
        self.assertEqual(config.mt5_server, "TestServer")
        self.assertEqual(config.symbols, ["EURUSD+", "GBPUSD+"])
        self.assertEqual(config.max_positions, 5)
        self.assertEqual(config.risk_per_trade, 0.01)

class TestTradingEngine(unittest.TestCase):
    """Test TradingEngine class"""
    
    def setUp(self):
        self.config = SystemConfig()
        self.engine = TradingEngine(self.config)
        self.test_market_data = MarketData(
            symbol="EURUSD+",
            bid=1.17756,
            ask=1.17769,
            spread_pips=1.3,
            digits=5,
            category="FOREX",
            timestamp=time.time(),
            timeframe="M5"
        )
    
    def test_engine_initialization(self):
        """Test trading engine initialization"""
        self.assertEqual(self.engine.account_balance, 10000.0)
        self.assertEqual(self.engine.account_equity, 10000.0)
        self.assertEqual(len(self.engine.positions), 0)
        self.assertTrue(self.engine.can_open_position())
    
    def test_position_size_calculation(self):
        """Test position size calculation"""
        position_size = self.engine.calculate_position_size("EURUSD+", 20.0)
        
        # Should be reasonable position size based on risk
        self.assertGreater(position_size, 0)
        self.assertLessEqual(position_size, 1.0)  # Max 1 lot
    
    def test_position_opening(self):
        """Test position opening"""
        # Test BUY position
        position = self.engine.execute_action(
            Action.BUY, "EURUSD+", self.test_market_data, "BERSERKER"
        )
        
        self.assertIsNotNone(position)
        self.assertEqual(position.symbol, "EURUSD+")
        self.assertEqual(position.direction, "BUY")
        self.assertEqual(position.agent_name, "BERSERKER")
        self.assertGreater(position.position_size, 0)
        
        # Check position is tracked
        self.assertIn("EURUSD+", self.engine.positions)
        # Note: can_open_position checks max_positions, not duplicate symbols
        self.assertEqual(len(self.engine.positions), 1)
    
    def test_position_updates(self):
        """Test position updates and closing"""
        # Open position
        position = self.engine.execute_action(
            Action.BUY, "EURUSD+", self.test_market_data, "BERSERKER"
        )
        self.assertIsNotNone(position)
        
        # Create updated market data that should trigger stop loss
        updated_data = MarketData(
            symbol="EURUSD+",
            bid=1.17000,  # Much lower - should trigger stop loss
            ask=1.17010,
            spread_pips=1.0,
            digits=5,
            category="FOREX",
            timestamp=time.time() + 60,
            timeframe="M5"
        )
        
        # Update positions
        closed_trades = self.engine.update_positions({"EURUSD+": updated_data})
        
        # Should have closed the position
        self.assertEqual(len(closed_trades), 1)
        self.assertEqual(closed_trades[0].win_loss, "LOSS")
        self.assertNotIn("EURUSD+", self.engine.positions)
    
    def test_max_positions_limit(self):
        """Test maximum positions limit"""
        # Fill up to max positions
        symbols = [f"SYMBOL{i}" for i in range(self.config.max_positions + 2)]
        
        positions_opened = 0
        for i, symbol in enumerate(symbols):
            market_data = MarketData(
                symbol=symbol,
                bid=1.0 + i * 0.001,
                ask=1.001 + i * 0.001,
                spread_pips=1.0,
                digits=3,
                category="TEST",
                timestamp=time.time(),
                timeframe="M5"
            )
            
            position = self.engine.execute_action(
                Action.BUY, symbol, market_data, "BERSERKER"
            )
            
            if position:
                positions_opened += 1
        
        # Should only open up to max_positions
        self.assertEqual(positions_opened, self.config.max_positions)
        self.assertFalse(self.engine.can_open_position())

class TestVelocityTradingSystem(unittest.TestCase):
    """Test integrated VelocityTradingSystem"""
    
    def setUp(self):
        self.config = SystemConfig()
        # Reduce intervals for testing
        self.config.update_interval = 0.1
        self.config.model_save_interval = 1
        self.config.transfer_learning_interval = 2
        self.config.use_multiprocessing = False  # Simpler for tests
    
    def test_system_initialization(self):
        """Test system initialization"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Change to temp directory
            original_cwd = Path.cwd()
            import os
            os.chdir(temp_dir)
            
            try:
                system = VelocityTradingSystem(self.config)
                
                # Check components are initialized
                self.assertIsNotNone(system.data_pipeline)
                self.assertIsNotNone(system.agent_coordinator)
                self.assertIsNotNone(system.trading_engine)
                self.assertIsNotNone(system.logger)
                
                # Check configuration
                self.assertEqual(system.config.mt5_login, self.config.mt5_login)
                self.assertFalse(system.is_running)
                
            finally:
                os.chdir(original_cwd)
    
    def test_system_with_simple_config(self):
        """Test system with simplified configuration"""
        with tempfile.TemporaryDirectory() as temp_dir:
            original_cwd = Path.cwd()
            import os
            os.chdir(temp_dir)
            
            try:
                # Use simplified config to avoid multiprocessing issues
                simple_config = SystemConfig()
                simple_config.use_multiprocessing = False
                simple_config.symbols = ["EURUSD+"]
                
                system = VelocityTradingSystem(simple_config)
                
                # Check system components
                self.assertIsNotNone(system.data_pipeline)
                self.assertIsNotNone(system.agent_coordinator)
                self.assertIsNotNone(system.trading_engine)
                
            finally:
                os.chdir(original_cwd)

class TestPerformanceDashboard(unittest.TestCase):
    """Test PerformanceDashboard class"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "test_logs.db"
        
        # Create test database with some data
        logger = ComprehensiveLogger(str(self.db_path))
        
        # Add some test data
        from src.utils.logging_system import TradeLog, PerformanceLog
        
        test_trade = TradeLog(
            timestamp=time.time(),
            agent_id="BERSERKER",
            instrument="EURUSD+",
            timeframe="M5",
            action="BUY",
            entry_price=1.17756,
            exit_price=1.17856,
            position_size=0.1,
            pnl=10.0,
            mae=-5.0,
            mfe=15.0,
            trade_duration=300.0,
            spread_cost=1.3,
            slippage=0.0,
            market_conditions={"volatility": "medium"},
            confidence_score=0.8,
            risk_reward_ratio=3.0,
            win_loss="WIN",
            trade_id="TEST_001",
            virtual_trade=False
        )
        
        logger.log_trade(test_trade)
        
        self.dashboard = PerformanceDashboard(str(self.db_path), port=8081)
    
    def tearDown(self):
        self.dashboard.stop()
        shutil.rmtree(self.temp_dir)
    
    def test_dashboard_initialization(self):
        """Test dashboard initialization"""
        self.assertEqual(self.dashboard.port, 8081)
        self.assertFalse(self.dashboard.is_running)
        self.assertEqual(self.dashboard.db_path, str(self.db_path))
    
    def test_performance_stats_generation(self):
        """Test performance statistics generation"""
        stats = self.dashboard._get_performance_stats()
        
        self.assertIn('timestamp', stats)
        self.assertIn('system', stats)
        self.assertIn('trading', stats)
        self.assertIn('agents', stats)
        
        # Check trading stats
        if stats['trading']['total_trades'] > 0:
            self.assertGreater(stats['trading']['win_rate_pct'], 0)
    
    def test_recent_trades_retrieval(self):
        """Test recent trades retrieval"""
        trades = self.dashboard._get_recent_trades(10)
        
        self.assertIsInstance(trades, list)
        if len(trades) > 0:
            trade = trades[0]
            self.assertIn('timestamp', trade)
            self.assertIn('agent', trade)
            self.assertIn('symbol', trade)
            self.assertIn('pnl', trade)
    
    def test_html_generation(self):
        """Test HTML dashboard generation"""
        html = self.dashboard._generate_dashboard_html()
        
        self.assertIn('<html', html)
        self.assertIn('VelocityTrader', html)
        self.assertIn('dashboard', html.lower())
        self.assertIn('</html>', html)

class TestEndToEndIntegration(unittest.TestCase):
    """End-to-end integration tests"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.original_cwd = Path.cwd()
        
        # Create test configuration
        self.config = SystemConfig()
        self.config.symbols = ["EURUSD+", "GBPUSD+"]  # Smaller set for testing
        self.config.update_interval = 0.1
        self.config.model_save_interval = 1
        self.config.use_multiprocessing = False
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
    
    def test_component_integration(self):
        """Test integration between all major components"""
        import os
        os.chdir(self.temp_dir)
        
        try:
            # Initialize all components
            from src.core.data_pipeline import DataPipeline
            from src.agents.dual_agent_system import DualAgentCoordinator
            from src.utils.logging_system import ComprehensiveLogger
            
            # Create mock data pipeline
            data_pipeline = DataPipeline()
            agent_coordinator = DualAgentCoordinator()
            logger = ComprehensiveLogger()
            
            # Test data flow
            test_data = MarketData(
                symbol="EURUSD+",
                bid=1.17756,
                ask=1.17769,
                spread_pips=1.3,
                digits=5,
                category="FOREX",
                timestamp=time.time(),
                timeframe="M5"
            )
            
            # Create state from market data
            state = State(
                price=test_data.mid_price,
                spread=test_data.spread_cost_normalized,
                momentum=0.3,
                acceleration=0.05,
                volatility=0.25,
                liquidity_score=0.7,
                trend_strength=0.6
            )
            
            # Get agent decision
            action, agent_name = agent_coordinator.process_tick(
                state, "EURUSD+", "M5"
            )
            
            # Verify integration
            self.assertIn(action, [Action.HOLD, Action.BUY, Action.SELL])
            self.assertIn(agent_name, ["BERSERKER", "SNIPER"])
            
        finally:
            os.chdir(self.original_cwd)
    
    def test_configuration_loading(self):
        """Test configuration file loading"""
        import os
        os.chdir(self.temp_dir)
        
        try:
            # Create test config file
            config_data = {
                "mt5_login": 12345,
                "symbols": ["EURUSD+", "GBPUSD+"],
                "max_positions": 5
            }
            
            config_dir = Path("config")
            config_dir.mkdir()
            
            config_file = config_dir / "system_config.json"
            with open(config_file, 'w') as f:
                json.dump(config_data, f)
            
            # Test loading
            if config_file.exists():
                with open(config_file) as f:
                    loaded_config = json.load(f)
                
                config = SystemConfig(**loaded_config)
                
                self.assertEqual(config.mt5_login, 12345)
                self.assertEqual(config.symbols, ["EURUSD+", "GBPUSD+"])
                self.assertEqual(config.max_positions, 5)
            
        finally:
            os.chdir(self.original_cwd)
    
    def test_model_persistence(self):
        """Test model saving and loading"""
        import os
        os.chdir(self.temp_dir)
        
        try:
            # Create agent coordinator
            coordinator1 = DualAgentCoordinator()
            
            # Train agents a bit
            for i in range(10):
                coordinator1.berserker.total_trades += 1
                coordinator1.sniper.total_trades += 1
            
            # Save models
            coordinator1.save_models("models/")
            
            # Create new coordinator and load models
            coordinator2 = DualAgentCoordinator()
            success = coordinator2.load_models("models/")
            
            self.assertTrue(success)
            self.assertEqual(coordinator2.berserker.total_trades, 
                           coordinator1.berserker.total_trades)
            self.assertEqual(coordinator2.sniper.total_trades, 
                           coordinator1.sniper.total_trades)
            
        finally:
            os.chdir(self.original_cwd)

class TestSystemLauncher(unittest.TestCase):
    """Test system launcher functionality"""
    
    def test_dependency_check(self):
        """Test dependency checking"""
        from velocity_trader import VelocityTraderLauncher
        
        launcher = VelocityTraderLauncher()
        
        # This should pass in our environment
        has_deps = launcher.check_dependencies()
        
        # Should at least detect basic packages
        self.assertIsInstance(has_deps, bool)
    
    def test_test_running(self):
        """Test running system tests"""
        from velocity_trader import VelocityTraderLauncher
        
        launcher = VelocityTraderLauncher()
        
        # Run tests (this will actually run our test suite)
        # Note: This is a meta-test - testing our test runner
        result = launcher.run_tests()
        
        # Should complete without crashing
        self.assertIsInstance(result, bool)

# Integration test runner
def run_phase4_integration_tests():
    """Run all Phase 4 integration tests"""
    print("\n" + "="*60)
    print("🧪 PHASE 4 TESTS: FULL SYSTEM INTEGRATION")
    print("="*60)
    
    # Create test suite
    test_classes = [
        TestSystemConfig,
        TestTradingEngine,
        TestVelocityTradingSystem,
        TestPerformanceDashboard,
        TestEndToEndIntegration,
        TestSystemLauncher
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
        print("✅ PHASE 4 TESTS PASSED - Full System Integration Complete")
        print("   🎉 VelocityTrader ready for production deployment!")
    else:
        print("❌ PHASE 4 TESTS FAILED")
        print(f"   Failures: {len(result.failures)}")
        print(f"   Errors: {len(result.errors)}")
        
        for failure in result.failures:
            print(f"   FAILURE: {failure[0]}")
        
        for error in result.errors:
            print(f"   ERROR: {error[0]}")
    
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_phase4_integration_tests()
    exit(0 if success else 1)