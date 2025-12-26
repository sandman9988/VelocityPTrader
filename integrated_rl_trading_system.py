#!/usr/bin/env python3
"""
Integrated Real-Time RL Trading System
Complete integration of RL learning, real-time trading, and monitoring dashboard

Features:
- Real-time RL learning and adaptation
- Live MT5 data integration
- Professional monitoring dashboard
- Comprehensive trade management
- Risk monitoring and alerts
- Performance tracking and optimization
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import json
import time
import threading
import signal
from datetime import datetime
from typing import Dict, List, Optional, Any

# Import our system components
from real_time_rl_system import RealTimeRLTradingSystem, LiveTradeSignal
from monitoring_dashboard import RLTradingDashboard, create_monitoring_dashboard
from rl_learning_integration import RLLearningEngine

class IntegratedRLTradingSystem:
    """Complete integrated RL trading system with monitoring"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._get_default_config()
        
        # Initialize core components
        print("🚀 INITIALIZING INTEGRATED RL TRADING SYSTEM")
        print("=" * 70)
        
        # Initialize RL trading system
        print("🧠 Initializing RL trading engine...")
        self.rl_system = RealTimeRLTradingSystem(self.config)
        
        # Initialize monitoring dashboard
        print("📊 Initializing monitoring dashboard...")
        self.dashboard = create_monitoring_dashboard(
            self.rl_system, 
            port=self.config.get('dashboard_port', 5000)
        )
        
        # System state
        self.is_running = False
        self.dashboard_thread = None
        
        # Setup callbacks
        self._setup_trade_callbacks()
        
        print("✅ Integrated RL Trading System initialized")
        print(f"   📊 Monitoring {len(self.config['symbols'])} symbols")
        print(f"   🌐 Dashboard will be available on port {self.config.get('dashboard_port', 5000)}")
    
    def _get_default_config(self) -> Dict:
        """Get default system configuration"""
        return {
            'symbols': ['EURUSD', 'GBPUSD', 'USDJPY', 'BTCUSD', 'XAUUSD'],
            'learning_frequency': 60,  # Learn every 60 seconds
            'signal_frequency': 10,    # Generate signals every 10 seconds
            'dashboard_port': 5000,
            'risk_management': {
                'max_positions': 5,
                'max_risk_per_trade': 0.02,  # 2%
                'max_total_risk': 0.10,      # 10%
                'max_drawdown_limit': 0.15   # 15%
            },
            'learning_config': {
                'experience_buffer_size': 10000,
                'batch_size': 32,
                'learning_rate': 0.001,
                'exploration_decay': 0.999
            }
        }
    
    def _setup_trade_callbacks(self):
        """Setup callbacks for trade signals and position updates"""
        
        def on_trade_signal(signal: LiveTradeSignal):
            """Handle new trading signals from RL system"""
            
            print(f"\n🔔 NEW TRADING SIGNAL")
            print(f"   Symbol: {signal.symbol}")
            print(f"   Signal: {signal.signal_type}")
            print(f"   Confidence: {signal.confidence:.3f}")
            print(f"   Agent: {signal.agent_recommendation}")
            print(f"   Size: {signal.position_size:.2f}")
            print(f"   Regime: {signal.regime_context}")
            print(f"   Reasoning: {signal.reasoning}")
            
            # Apply risk management
            if self._should_execute_signal(signal):
                self._execute_trading_signal(signal)
            else:
                print(f"   ⚠️  Signal rejected by risk management")
        
        def on_position_update(updates: List[Dict]):
            """Handle position updates"""
            
            for update in updates:
                # Monitor for risk limits
                unrealized_pnl = update.get('unrealized_pnl', 0)
                if unrealized_pnl < -1000:  # Example threshold
                    print(f"⚠️  Large unrealized loss: {update['symbol']} ${unrealized_pnl:.2f}")
        
        self.rl_system.set_trade_signal_callback(on_trade_signal)
        self.rl_system.set_position_update_callback(on_position_update)
    
    def _should_execute_signal(self, signal: LiveTradeSignal) -> bool:
        """Apply risk management to determine if signal should be executed"""
        
        risk_config = self.config['risk_management']
        
        # Check confidence threshold
        if signal.confidence < 0.4:
            return False
        
        # Check maximum positions
        active_positions = len(self.rl_system.active_positions)
        if active_positions >= risk_config['max_positions']:
            return False
        
        # Check symbol exposure (avoid too much exposure to one symbol)
        symbol_positions = sum(1 for pos in self.rl_system.active_positions.values() 
                              if pos.symbol == signal.symbol)
        if symbol_positions >= 2:  # Max 2 positions per symbol
            return False
        
        return True
    
    def _execute_trading_signal(self, signal: LiveTradeSignal):
        """Execute trading signal (integrate with actual broker API)"""
        
        # This is where you would integrate with your broker's API
        # For demonstration, we'll simulate trade execution
        
        trade_data = {
            'symbol': signal.symbol,
            'agent_type': signal.agent_recommendation,
            'action': 2 if signal.signal_type == 'BUY' else 3,
            'entry_price': 1.0500,  # Would get from current market price
            'position_size': signal.position_size,
            'market_regime': signal.regime_context,
            'volatility_percentile': 65,
            'volume_percentile': 55,
            'confidence': signal.confidence
        }
        
        # Register trade with RL system for learning
        trade_id = self.rl_system.on_trade_executed(trade_data)
        
        print(f"   ✅ Trade executed: {trade_id}")
        print(f"   📈 Registered with RL system for learning")
        
        # In a real system, you would:
        # 1. Submit order to broker API
        # 2. Handle order confirmation
        # 3. Monitor position
        # 4. Update RL system with actual execution data
    
    def start_system(self):
        """Start the complete integrated trading system"""
        
        if self.is_running:
            print("⚠️  System already running")
            return
        
        print("\n🚀 STARTING INTEGRATED RL TRADING SYSTEM")
        print("=" * 70)
        
        self.is_running = True
        
        # Start dashboard in separate thread
        self.dashboard_thread = threading.Thread(
            target=self._run_dashboard, 
            daemon=True
        )
        self.dashboard_thread.start()
        
        # Wait a moment for dashboard to initialize
        time.sleep(2)
        
        # Start RL trading system (this will run the main loop)
        try:
            print("🧠 Starting RL trading engine...")
            self.rl_system.start_real_time_trading()
        except KeyboardInterrupt:
            print("\n🛑 Shutdown signal received")
            self.stop_system()
        except Exception as e:
            print(f"\n❌ Critical system error: {e}")
            self.stop_system()
    
    def _run_dashboard(self):
        """Run dashboard in separate thread"""
        
        try:
            print(f"🌐 Dashboard starting on http://localhost:{self.config['dashboard_port']}")
            self.dashboard.start_dashboard()
        except Exception as e:
            print(f"❌ Dashboard error: {e}")
    
    def stop_system(self):
        """Stop the integrated trading system"""
        
        if not self.is_running:
            return
        
        print("\n⏹️  STOPPING INTEGRATED RL TRADING SYSTEM")
        print("=" * 70)
        
        self.is_running = False
        
        # Stop RL system
        print("🧠 Stopping RL trading engine...")
        self.rl_system.stop_trading()
        
        # Stop dashboard
        print("📊 Stopping monitoring dashboard...")
        self.dashboard.stop_dashboard()
        
        print("✅ Integrated RL Trading System stopped")
    
    def get_system_performance(self) -> Dict[str, Any]:
        """Get comprehensive system performance metrics"""
        
        rl_status = self.rl_system.get_system_status()
        learning_summary = rl_status.get('learning_summary', {})
        
        return {
            'system_uptime': time.time(),
            'active_symbols': rl_status.get('active_symbols', 0),
            'active_positions': rl_status.get('active_positions', 0),
            'total_signals_generated': len(getattr(self.rl_system, 'latest_signals', {})),
            'learning_metrics': {
                'total_experiences': learning_summary.get('total_experiences', 0),
                'current_exploration_rate': learning_summary.get('current_exploration_rate', 0.1),
                'avg_recent_efficiency': learning_summary.get('avg_recent_efficiency', 0),
                'total_journeys_learned': learning_summary.get('total_journeys_learned', 0)
            },
            'risk_metrics': {
                'max_positions': self.config['risk_management']['max_positions'],
                'positions_used': len(self.rl_system.active_positions),
                'risk_utilization': len(self.rl_system.active_positions) / self.config['risk_management']['max_positions']
            }
        }
    
    def run_system_diagnostics(self) -> Dict[str, Any]:
        """Run comprehensive system diagnostics"""
        
        print("\n🔍 RUNNING SYSTEM DIAGNOSTICS")
        print("=" * 50)
        
        diagnostics = {
            'timestamp': datetime.now().isoformat(),
            'components': {},
            'health_status': 'HEALTHY',
            'issues': []
        }
        
        # Test RL engine
        try:
            rl_status = self.rl_system.get_system_status()
            diagnostics['components']['rl_engine'] = {
                'status': 'HEALTHY',
                'active_symbols': rl_status.get('active_symbols', 0),
                'learning_progress': rl_status.get('learning_summary', {})
            }
            print("✅ RL Engine: HEALTHY")
        except Exception as e:
            diagnostics['components']['rl_engine'] = {
                'status': 'ERROR',
                'error': str(e)
            }
            diagnostics['issues'].append(f"RL Engine error: {e}")
            print(f"❌ RL Engine: ERROR - {e}")
        
        # Test market data feed
        try:
            latest_data = self.rl_system.latest_market_data
            diagnostics['components']['market_data'] = {
                'status': 'HEALTHY',
                'symbols_receiving_data': len(latest_data),
                'last_update': max([data.timestamp for data in latest_data.values()]).isoformat() if latest_data else None
            }
            print(f"✅ Market Data: HEALTHY ({len(latest_data)} symbols)")
        except Exception as e:
            diagnostics['components']['market_data'] = {
                'status': 'ERROR',
                'error': str(e)
            }
            diagnostics['issues'].append(f"Market data error: {e}")
            print(f"❌ Market Data: ERROR - {e}")
        
        # Test regime detection
        try:
            regime_states = {}
            for symbol in self.rl_system.symbols:
                regime_state = self.rl_system.regime_detector.get_current_regime(symbol)
                if regime_state:
                    regime_states[symbol] = regime_state.regime
            
            diagnostics['components']['regime_detection'] = {
                'status': 'HEALTHY',
                'detected_regimes': regime_states
            }
            print(f"✅ Regime Detection: HEALTHY ({len(regime_states)} regimes detected)")
        except Exception as e:
            diagnostics['components']['regime_detection'] = {
                'status': 'ERROR',
                'error': str(e)
            }
            diagnostics['issues'].append(f"Regime detection error: {e}")
            print(f"❌ Regime Detection: ERROR - {e}")
        
        # Determine overall health
        if diagnostics['issues']:
            diagnostics['health_status'] = 'DEGRADED' if len(diagnostics['issues']) < 3 else 'CRITICAL'
        
        print(f"\n📊 Overall System Health: {diagnostics['health_status']}")
        if diagnostics['issues']:
            print("⚠️  Issues detected:")
            for issue in diagnostics['issues']:
                print(f"   • {issue}")
        
        return diagnostics

def main():
    """Main entry point for the integrated system"""
    
    print("🎯 INTEGRATED RL TRADING SYSTEM")
    print("=" * 80)
    print("Real-time reinforcement learning with professional monitoring")
    print()
    
    # Configuration
    config = {
        'symbols': ['EURUSD', 'GBPUSD', 'USDJPY', 'BTCUSD', 'XAUUSD'],
        'learning_frequency': 30,  # Faster learning for demo
        'signal_frequency': 5,     # More frequent signals
        'dashboard_port': 5000,
        'risk_management': {
            'max_positions': 3,
            'max_risk_per_trade': 0.02,
            'max_total_risk': 0.08,
            'max_drawdown_limit': 0.12
        }
    }
    
    # Initialize system
    system = IntegratedRLTradingSystem(config)
    
    # Setup signal handlers for clean shutdown
    def signal_handler(signum, frame):
        print("\n🛑 Shutdown signal received")
        system.stop_system()
        exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Run diagnostics
    diagnostics = system.run_system_diagnostics()
    
    if diagnostics['health_status'] != 'CRITICAL':
        print(f"\n🚀 System health check passed - starting trading system")
        print(f"🌐 Monitor at: http://localhost:{config['dashboard_port']}")
        print(f"💡 Press Ctrl+C to stop the system")
        
        try:
            # Start the integrated system
            system.start_system()
        except KeyboardInterrupt:
            print("\n🛑 Manual shutdown initiated")
        finally:
            system.stop_system()
    else:
        print(f"\n❌ System health check failed - cannot start")
        print(f"   Fix the issues above and try again")

if __name__ == "__main__":
    main()