#!/usr/bin/env python3
"""
VELOCITYTRADER LAUNCHER
Main entry point for the VelocityTrader system
"""

import argparse
import sys
import os
import time
import signal
import subprocess
import threading
from pathlib import Path
from typing import Optional

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('velocity_trader.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class VelocityTraderLauncher:
    """Main launcher for VelocityTrader system"""
    
    def __init__(self):
        self.trading_process: Optional[subprocess.Popen] = None
        self.dashboard_process: Optional[subprocess.Popen] = None
        self.is_running = False
        
        # Signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info("🛑 Shutdown signal received")
        self.stop_all()
        sys.exit(0)
    
    def check_dependencies(self) -> bool:
        """Check system dependencies"""
        logger.info("🔍 Checking system dependencies...")
        
        # Check Python packages
        required_packages = [
            'numpy', 'psutil', 'torch', 'structlog'
        ]
        
        missing_packages = []
        for package in required_packages:
            try:
                __import__(package)
                logger.info(f"   ✅ {package}")
            except ImportError:
                missing_packages.append(package)
                logger.error(f"   ❌ {package}")
        
        if missing_packages:
            logger.error(f"Missing packages: {missing_packages}")
            logger.error("Run: pip install -r requirements.txt")
            return False
        
        # Check data files
        data_file = Path("data/all_mt5_symbols.json")
        if not data_file.exists():
            logger.warning("⚠️ MT5 symbols data not found - will use simulation mode")
        else:
            logger.info("✅ MT5 data file found")
        
        # Check models directory
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        logger.info("✅ Models directory ready")
        
        return True
    
    def start_trading_system(self, config_file: str = "config/system_config.json") -> bool:
        """Start the main trading system"""
        try:
            logger.info("🚀 Starting VelocityTrader system...")
            
            # Import and start the integrated system
            from src.core.integrated_system import main as trading_main
            
            # Start in separate process
            self.trading_process = subprocess.Popen([
                sys.executable, "-c",
                "from src.core.integrated_system import main; main()"
            ])
            
            logger.info("✅ Trading system started")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to start trading system: {e}")
            return False
    
    def start_dashboard(self, port: int = 8080) -> bool:
        """Start the performance dashboard"""
        try:
            logger.info(f"📊 Starting performance dashboard on port {port}...")
            
            # Start dashboard in separate process
            self.dashboard_process = subprocess.Popen([
                sys.executable, "-c",
                f"from src.utils.performance_dashboard import main; main()"
            ])
            
            logger.info(f"✅ Dashboard started: http://localhost:{port}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to start dashboard: {e}")
            return False
    
    def stop_all(self):
        """Stop all processes"""
        logger.info("⏹️ Stopping all processes...")
        
        if self.trading_process:
            self.trading_process.terminate()
            try:
                self.trading_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.trading_process.kill()
            self.trading_process = None
            logger.info("✅ Trading system stopped")
        
        if self.dashboard_process:
            self.dashboard_process.terminate()
            try:
                self.dashboard_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.dashboard_process.kill()
            self.dashboard_process = None
            logger.info("✅ Dashboard stopped")
        
        self.is_running = False
    
    def monitor_processes(self):
        """Monitor running processes"""
        while self.is_running:
            try:
                # Check trading system
                if self.trading_process and self.trading_process.poll() is not None:
                    logger.error("❌ Trading system crashed, restarting...")
                    self.start_trading_system()
                
                # Check dashboard
                if self.dashboard_process and self.dashboard_process.poll() is not None:
                    logger.error("❌ Dashboard crashed, restarting...")
                    self.start_dashboard()
                
                time.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                logger.error(f"Monitor error: {e}")
                time.sleep(10)
    
    def run_tests(self) -> bool:
        """Run system tests"""
        logger.info("🧪 Running system tests...")
        
        test_commands = [
            ("Phase 1 Tests", "python tests/test_framework.py"),
            ("Phase 2 Tests", "python tests/test_phase2_pipeline.py"),
            ("Phase 3 Tests", "python tests/test_phase3_agents.py"),
        ]
        
        all_passed = True
        
        for name, command in test_commands:
            logger.info(f"Running {name}...")
            try:
                result = subprocess.run(
                    command.split(),
                    capture_output=True,
                    text=True,
                    timeout=120
                )
                
                if result.returncode == 0:
                    logger.info(f"✅ {name} passed")
                else:
                    logger.error(f"❌ {name} failed:")
                    logger.error(result.stderr)
                    all_passed = False
                    
            except subprocess.TimeoutExpired:
                logger.error(f"❌ {name} timed out")
                all_passed = False
            except Exception as e:
                logger.error(f"❌ {name} error: {e}")
                all_passed = False
        
        if all_passed:
            logger.info("✅ All tests passed")
        else:
            logger.error("❌ Some tests failed")
        
        return all_passed
    
    def start_full_system(self, with_dashboard: bool = True, run_tests_first: bool = False):
        """Start the complete VelocityTrader system"""
        logger.info("=" * 60)
        logger.info("🚀 VELOCITYTRADER - Physics-Based AI Trading System")
        logger.info("=" * 60)
        
        # Check dependencies
        if not self.check_dependencies():
            logger.error("❌ Dependency check failed")
            return False
        
        # Run tests if requested
        if run_tests_first:
            if not self.run_tests():
                logger.error("❌ Tests failed - aborting startup")
                return False
        
        self.is_running = True
        
        # Start trading system
        if not self.start_trading_system():
            return False
        
        # Start dashboard
        if with_dashboard:
            if not self.start_dashboard():
                logger.warning("⚠️ Dashboard failed to start, continuing without it")
        
        # Start monitoring
        monitor_thread = threading.Thread(target=self.monitor_processes, daemon=True)
        monitor_thread.start()
        
        logger.info("✅ VelocityTrader system fully operational")
        logger.info("   📈 Trading system: Running")
        if with_dashboard:
            logger.info("   📊 Dashboard: http://localhost:8080")
        logger.info("   Press Ctrl+C to stop")
        
        try:
            # Keep main thread alive
            while self.is_running:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")
        finally:
            self.stop_all()
        
        return True

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='VelocityTrader - Physics-Based AI Trading System')
    parser.add_argument('--config', '-c', default='config/system_config.json',
                       help='Configuration file path')
    parser.add_argument('--no-dashboard', action='store_true',
                       help='Start without dashboard')
    parser.add_argument('--test', action='store_true',
                       help='Run tests before starting')
    parser.add_argument('--test-only', action='store_true',
                       help='Only run tests, do not start system')
    parser.add_argument('--dashboard-only', action='store_true',
                       help='Only start dashboard')
    parser.add_argument('--port', '-p', type=int, default=8080,
                       help='Dashboard port (default: 8080)')
    
    args = parser.parse_args()
    
    launcher = VelocityTraderLauncher()
    
    try:
        if args.test_only:
            success = launcher.run_tests()
            sys.exit(0 if success else 1)
        
        elif args.dashboard_only:
            success = launcher.start_dashboard(args.port)
            if success:
                logger.info("Dashboard running. Press Ctrl+C to stop.")
                try:
                    while True:
                        time.sleep(1)
                except KeyboardInterrupt:
                    pass
                finally:
                    launcher.stop_all()
        
        else:
            success = launcher.start_full_system(
                with_dashboard=not args.no_dashboard,
                run_tests_first=args.test
            )
            sys.exit(0 if success else 1)
    
    except Exception as e:
        logger.error(f"Launcher error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()