#!/usr/bin/env python3
"""
Real-Time RL Trading System
Continuous learning and adaptation for live trading with MT5 integration

Features:
- Real-time market data processing
- Live trade monitoring and learning
- Dynamic model updates based on trade outcomes
- Real-time regime detection
- Continuous experience replay optimization
- Live signal generation with confidence scoring
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent / "rl_framework"))
sys.path.append(str(Path(__file__).parent / "data"))
sys.path.append(str(Path(__file__).parent / "agents"))

import json
import time
import random
import threading
import asyncio
# Use math instead of numpy for compatibility
import math
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from collections import deque, defaultdict
from queue import Queue, Empty
import logging

# Import our RL components
from rl_learning_integration import RLLearningEngine, LiveTradingState
from intelligent_experience_replay import IntelligentExperienceReplay, TradeJourney
from advanced_performance_metrics import AdvancedPerformanceCalculator

# Try to import MT5
try:
    from mt5_bridge import (
        initialize, shutdown, symbols_get, copy_rates_total, 
        symbol_info_tick, copy_rates_from,
        TIMEFRAME_M1, TIMEFRAME_M5, TIMEFRAME_M15, TIMEFRAME_H1
    )
    MT5_AVAILABLE = True
except ImportError:
    print("⚠️  MT5 Bridge not available - using simulated mode")
    MT5_AVAILABLE = False

@dataclass
class LiveMarketData:
    """Real-time market data structure"""
    symbol: str
    timestamp: datetime
    bid: float
    ask: float
    spread: float
    volume: int
    
    # Technical indicators (calculated in real-time)
    volatility: float = 0.0
    momentum: float = 0.0
    trend_strength: float = 0.0
    volume_profile: float = 0.0

@dataclass
class MarketRegimeState:
    """Current market regime detection"""
    symbol: str
    regime: str  # CHAOTIC, UNDERDAMPED, CRITICALLY_DAMPED, OVERDAMPED
    confidence: float
    volatility_percentile: float
    trend_clarity: float
    last_update: datetime

@dataclass
class LiveTradeSignal:
    """Real-time trading signal from RL system"""
    symbol: str
    signal_type: str  # BUY, SELL, CLOSE, HOLD
    confidence: float
    agent_recommendation: str  # BERSERKER, SNIPER
    position_size: float
    stop_loss: float
    take_profit: float
    regime_context: str
    reasoning: str
    timestamp: datetime

class RealTimeRegimeDetector:
    """Real-time market regime detection system"""
    
    def __init__(self, lookback_periods: int = 100):
        self.lookback_periods = lookback_periods
        self.price_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=lookback_periods))
        self.volume_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=lookback_periods))
        self.regime_cache: Dict[str, MarketRegimeState] = {}
        
    def update_market_data(self, market_data: LiveMarketData):
        """Update market data for regime detection"""
        
        symbol = market_data.symbol
        mid_price = (market_data.bid + market_data.ask) / 2
        
        self.price_history[symbol].append(mid_price)
        self.volume_history[symbol].append(market_data.volume)
        
        # Calculate regime if we have enough data
        if len(self.price_history[symbol]) >= 20:
            regime_state = self._calculate_regime(symbol)
    
    def detect_regime(self, symbol: str) -> MarketRegimeState:
        """Detect current market regime for symbol"""
        if symbol in self.regime_cache:
            return self.regime_cache[symbol]
        
        # Return default regime if no data
        return MarketRegimeState(
            symbol=symbol,
            regime='CRITICALLY_DAMPED',
            confidence=0.5,
            volatility_percentile=50.0,
            trend_clarity=0.5,
            last_update=datetime.now()
        )
    
    def _calculate_regime(self, symbol: str) -> MarketRegimeState:
        """Calculate current market regime based on price action"""
        
        prices = list(self.price_history[symbol])
        volumes = list(self.volume_history[symbol])
        
        # Calculate volatility (rolling standard deviation)
        if len(prices) >= 10:
            recent_returns = [math.log(prices[i]/prices[i-1]) for i in range(1, len(prices))]
            mean_return = sum(recent_returns) / len(recent_returns)
            variance = sum((r - mean_return) ** 2 for r in recent_returns) / len(recent_returns)
            volatility = math.sqrt(variance) * math.sqrt(252 * 24)  # Annualized
        else:
            volatility = 0.0
        
        # Calculate momentum
        if len(prices) >= 20:
            short_ma = sum(prices[-5:]) / 5
            long_ma = sum(prices[-20:]) / 20
            momentum = (short_ma - long_ma) / long_ma if long_ma != 0 else 0
        else:
            momentum = 0.0
        
        # Calculate trend strength
        if len(prices) >= 10:
            price_changes = [prices[i] - prices[i-1] for i in range(1, len(prices))]
            positive_moves = sum(1 for change in price_changes if change > 0)
            trend_strength = abs(2 * positive_moves / len(price_changes) - 1)  # 0 = choppy, 1 = strong trend
        else:
            trend_strength = 0.0
        
        # Determine regime based on volatility and trend characteristics
        volatility_percentile = self._get_volatility_percentile(symbol, volatility)
        
        if volatility_percentile > 80 and trend_strength < 0.3:
            regime = "CHAOTIC"
            confidence = 0.8
        elif volatility_percentile > 60 and trend_strength > 0.6:
            regime = "UNDERDAMPED" 
            confidence = 0.7
        elif volatility_percentile < 40 and trend_strength > 0.4:
            regime = "CRITICALLY_DAMPED"
            confidence = 0.6
        elif volatility_percentile < 30:
            regime = "OVERDAMPED"
            confidence = 0.7
        else:
            regime = "UNDERDAMPED"  # Default
            confidence = 0.4
        
        return MarketRegimeState(
            symbol=symbol,
            regime=regime,
            confidence=confidence,
            volatility_percentile=volatility_percentile,
            trend_clarity=trend_strength,
            last_update=datetime.now()
        )
    
    def _get_volatility_percentile(self, symbol: str, current_vol: float) -> float:
        """Get volatility percentile relative to recent history"""
        
        if len(self.price_history[symbol]) < 50:
            return 50.0
        
        prices = list(self.price_history[symbol])
        historical_vols = []
        
        # Calculate rolling volatilities
        for i in range(10, len(prices)):
            window_prices = prices[i-10:i]
            returns = [math.log(window_prices[j]/window_prices[j-1]) for j in range(1, len(window_prices))]
            mean_return = sum(returns) / len(returns)
            variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
            vol = math.sqrt(variance) * math.sqrt(252 * 24)
            historical_vols.append(vol)
        
        if not historical_vols:
            return 50.0
        
        # Calculate percentile
        historical_vols.sort()
        position = sum(1 for v in historical_vols if v <= current_vol)
        percentile = (position / len(historical_vols)) * 100
        
        return percentile
    
    def get_current_regime(self, symbol: str) -> Optional[MarketRegimeState]:
        """Get current regime for symbol"""
        return self.regime_cache.get(symbol)

class LiveMarketDataFeed:
    """Real-time market data feed with MT5 integration"""
    
    def __init__(self, symbols: List[str], update_frequency: float = 1.0):
        self.symbols = symbols
        self.update_frequency = update_frequency  # seconds
        self.data_queue = Queue()
        self.is_running = False
        self.data_thread = None
        
        # FORCE REAL MT5 connection - NO SIMULATION ALLOWED!
        self.mt5_connected = False
        self.use_simulation = False  # NEVER use simulation
        
        # FORCE real data validation
        def validate_real_data(data):
            if not data or data.bid == 0 or data.ask == 0:
                return None  # Invalid data
            if data.bid > data.ask:  # Sanity check
                return None
            return data
        
        self.validate_data = validate_real_data
        
        if not MT5_AVAILABLE:
            raise Exception("❌ MT5 bridge not available - REAL data required for training!")
            
        try:
            if not initialize():
                raise Exception("❌ Failed to initialize MT5 - Is your terminal running?")
                
            # Verify we have real symbols
            test_symbols = symbols_get()
            if not test_symbols or len(test_symbols) == 0:
                raise Exception("❌ No symbols in MarketWatch - Open your MT5 terminal!")
                
            self.mt5_connected = True
            self.real_symbols = [s.name for s in test_symbols]
            print(f"✅ REAL MT5 DATA: Connected with {len(test_symbols)} live symbols")
            print(f"📊 MarketWatch symbols: {', '.join(self.real_symbols)}")
            
        except Exception as e:
            print(f"🛑 CRITICAL ERROR: {e}")
            print("🔧 Please ensure your MT5 terminal is running and has symbols in MarketWatch")
            raise e
        
        print(f"📡 Market data feed initialized for {len(symbols)} symbols")
        print(f"📈 Full MarketWatch: {', '.join(symbols)}")
        print(f"   Update frequency: {update_frequency}s")
        print(f"   MT5 connected: {self.mt5_connected}")
    
    def start_feed(self):
        """Start the market data feed"""
        
        if self.is_running:
            print("⚠️  Data feed already running")
            return
        
        self.is_running = True
        self.data_thread = threading.Thread(target=self._data_feed_worker, daemon=True)
        self.data_thread.start()
        
        print("🚀 Live market data feed started")
    
    def stop_feed(self):
        """Stop the market data feed"""
        
        self.is_running = False
        if self.data_thread:
            self.data_thread.join()
        
        if self.mt5_connected:
            shutdown()
        
        print("⏹️  Market data feed stopped")
    
    def _data_feed_worker(self):
        """Background worker for fetching market data"""
        
        while self.is_running:
            try:
                for symbol in self.symbols:
                    market_data = self._fetch_symbol_data(symbol)
                    if market_data:
                        self.data_queue.put(market_data)
                
                time.sleep(self.update_frequency)
                
            except Exception as e:
                print(f"❌ Data feed error: {e}")
                time.sleep(5)  # Wait before retrying
    
    def _fetch_symbol_data(self, symbol: str) -> Optional[LiveMarketData]:
        """Fetch real-time data for a symbol"""
        
        try:
            if self.mt5_connected:
                # Get real tick data from MT5
                tick = symbol_info_tick(symbol)
                if tick is None:
                    return None
                
                return LiveMarketData(
                    symbol=symbol,
                    timestamp=datetime.now(),
                    bid=tick.bid,
                    ask=tick.ask,
                    spread=tick.ask - tick.bid,
                    volume=tick.volume if hasattr(tick, 'volume') else 0
                )
            else:
                # NO SIMULATION - REAL DATA ONLY!
                raise Exception(f"❌ MT5 not connected - cannot get REAL data for {symbol}")
                
        except Exception as e:
            print(f"❌ Error fetching data for {symbol}: {e}")
            return None
    
    def _simulate_market_data(self, symbol: str) -> LiveMarketData:
        """Simulate market data for testing purposes"""
        
        # Base prices for different symbols
        base_prices = {
            'EURUSD': 1.0500,
            'GBPUSD': 1.2500,
            'USDJPY': 150.00,
            'BTCUSD': 95000.0,
            'XAUUSD': 2000.0
        }
        
        base_price = base_prices.get(symbol, 1.0000)
        
        # Add some random movement
        price_change = random.gauss(0, base_price * 0.0001)
        current_price = base_price + price_change
        
        spread = base_price * random.uniform(0.00001, 0.0001)
        
        return LiveMarketData(
            symbol=symbol,
            timestamp=datetime.now(),
            bid=current_price - spread/2,
            ask=current_price + spread/2,
            spread=spread,
            volume=random.randint(100, 1000)
        )
    
    def get_latest_data(self) -> List[LiveMarketData]:
        """Get all available latest market data"""
        
        data = []
        try:
            while True:
                market_data = self.data_queue.get_nowait()
                data.append(market_data)
        except Empty:
            pass
        
        return data

class RealTimeRLTradingSystem:
    """Main real-time RL trading system"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        
        # Get REAL symbols from your actual MT5 terminal
        try:
            if MT5_AVAILABLE and initialize():
                real_symbols = symbols_get()
                if real_symbols and len(real_symbols) > 0:
                    # Use ALL your actual symbols from MarketWatch
                    self.symbols = [s.name for s in real_symbols]  # Use ALL symbols
                    print(f"🎯 Using ALL REAL symbols from your terminal ({len(self.symbols)} symbols)")
                else:
                    raise Exception("No symbols found in your MarketWatch")
            else:
                raise Exception("Cannot connect to your MT5 terminal")
        except Exception as e:
            print(f"🛑 CRITICAL: Cannot get REAL symbols: {e}")
            print("Using fallback symbols - but REAL data connection still required!")
            self.symbols = self.config.get('symbols', [
                'EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD'  # Your actual symbols
            ])
        
        # Initialize components
        self.rl_engine = RLLearningEngine()
        self.regime_detector = RealTimeRegimeDetector()
        self.market_feed = LiveMarketDataFeed(
            symbols=self.symbols,
            update_frequency=1.0  # 1 second updates
        )
        
        # Real-time state tracking
        self.active_positions: Dict[str, LiveTradingState] = {}
        self.latest_market_data: Dict[str, LiveMarketData] = {}
        self.latest_signals: Dict[str, LiveTradeSignal] = {}
        
        # Learning configuration
        self.learning_frequency = 30  # Learn every 30 seconds
        self.signal_generation_frequency = 3  # Generate signals every 3 seconds
        self.last_learning_time = time.time()
        self.last_signal_time = time.time()
        
        # Performance tracking
        self.performance_tracker = AdvancedPerformanceCalculator()
        
        # Callbacks for trade execution (to be set by user)
        self.on_trade_signal_callback: Optional[Callable] = None
        self.on_position_update_callback: Optional[Callable] = None
        
        print("🧠 Real-Time RL Trading System initialized")
        print(f"   📊 Monitoring {len(self.symbols)} symbols: {', '.join(self.symbols)}")
        print(f"   🎯 Learning frequency: {self.learning_frequency}s")
        print(f"   📡 Signal frequency: {self.signal_generation_frequency}s")
    
    def set_trade_signal_callback(self, callback: Callable):
        """Set callback for when new trade signals are generated"""
        self.on_trade_signal_callback = callback
    
    def set_position_update_callback(self, callback: Callable):
        """Set callback for position updates"""
        self.on_position_update_callback = callback
    
    def start_real_time_trading(self):
        """Start the real-time trading system"""
        
        print("\n🚀 STARTING REAL-TIME RL TRADING SYSTEM")
        print("=" * 60)
        
        # Start market data feed
        self.market_feed.start_feed()
        
        # Start main processing loop
        self._run_main_loop()
    
    def stop_trading(self):
        """Stop the real-time trading system"""
        
        print("\n⏹️  STOPPING REAL-TIME TRADING SYSTEM")
        
        # Stop market data feed
        self.market_feed.stop_feed()
        
        # Close any remaining positions (would implement actual closing logic)
        print(f"📊 Active positions to close: {len(self.active_positions)}")
        
        # Final learning session
        self._run_learning_session('final')
        
        print("✅ Real-time trading system stopped")
    
    def _run_main_loop(self):
        """Main processing loop for real-time trading"""
        
        try:
            while True:
                current_time = time.time()
                
                # Process latest market data
                self._process_market_updates()
                
                # Generate trading signals
                if current_time - self.last_signal_time >= self.signal_generation_frequency:
                    self._generate_trading_signals()
                    self.last_signal_time = current_time
                
                # Run learning sessions
                if current_time - self.last_learning_time >= self.learning_frequency:
                    self._run_learning_session('periodic')
                    self.last_learning_time = current_time
                
                # Update active positions
                self._update_active_positions()
                
                # Brief sleep to prevent excessive CPU usage
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            print("\n🛑 Received stop signal")
            self.stop_trading()
        except Exception as e:
            print(f"\n❌ Critical error in main loop: {e}")
            self.stop_trading()
    
    def _process_market_updates(self):
        """Process latest market data updates"""
        
        latest_data = self.market_feed.get_latest_data()
        
        for market_data in latest_data:
            symbol = market_data.symbol
            
            # Store latest data
            self.latest_market_data[symbol] = market_data
            
            # Update regime detection
            self.regime_detector.update_market_data(market_data)
            
            # Update any active positions for this symbol
            self._update_symbol_positions(symbol, market_data)
    
    def _update_symbol_positions(self, symbol: str, market_data: LiveMarketData):
        """Update positions for a specific symbol"""
        
        # Find active positions for this symbol
        symbol_positions = {
            trade_id: state for trade_id, state in self.active_positions.items() 
            if state.symbol == symbol
        }
        
        for trade_id, position in symbol_positions.items():
            # Calculate current P&L
            mid_price = (market_data.bid + market_data.ask) / 2
            
            # Update position state
            self.rl_engine.on_trade_update(trade_id, {
                'current_price': mid_price,
                'timestamp': market_data.timestamp,
                'volatility': market_data.volatility,
                'momentum': market_data.momentum
            })
            
            # Check for exit conditions (this would integrate with actual position management)
            self._check_exit_conditions(trade_id, position, market_data)
    
    def _check_exit_conditions(self, trade_id: str, position: LiveTradingState, market_data: LiveMarketData):
        """Check if position should be closed based on RL learning"""
        
        # This is where you'd implement sophisticated exit logic
        # based on RL learning, regime changes, etc.
        
        current_regime = self.regime_detector.get_current_regime(position.symbol)
        
        # Example: Exit if regime changes significantly
        if current_regime and current_regime.regime != position.market_regime:
            if current_regime.confidence > 0.7:
                print(f"🔄 Regime change detected for {trade_id}: {position.market_regime} -> {current_regime.regime}")
                # Would trigger actual position close here
    
    def _generate_trading_signals(self):
        """Generate trading signals based on current market conditions and RL"""
        
        for symbol in self.symbols:
            if symbol not in self.latest_market_data:
                continue
            
            signal = self._generate_signal_for_symbol(symbol)
            if signal:
                self.latest_signals[symbol] = signal
                
                # Call user callback if set
                if self.on_trade_signal_callback:
                    self.on_trade_signal_callback(signal)
                
                print(f"📡 Signal: {signal.symbol} {signal.signal_type} "
                      f"(Confidence: {signal.confidence:.3f}, Agent: {signal.agent_recommendation})")
    
    def _generate_signal_for_symbol(self, symbol: str) -> Optional[LiveTradeSignal]:
        """Generate trading signal for specific symbol"""
        
        market_data = self.latest_market_data[symbol]
        regime_state = self.regime_detector.get_current_regime(symbol)
        
        if not regime_state:
            return None
        
        # Determine optimal agent for current regime
        agent_scores = {
            'BERSERKER': self._get_agent_regime_score('BERSERKER', regime_state.regime),
            'SNIPER': self._get_agent_regime_score('SNIPER', regime_state.regime)
        }
        
        best_agent = max(agent_scores.keys(), key=lambda k: agent_scores[k])
        confidence = agent_scores[best_agent]
        
        # Only generate signals if confidence is reasonable (lowered for more learning activity)
        if confidence < 0.15:
            return None
        
        # Determine signal type based on momentum and RL learning
        signal_type = self._determine_signal_type(symbol, market_data, regime_state)
        
        if signal_type == "HOLD":
            return None
        
        # Calculate position sizing and risk parameters
        position_size = self._calculate_position_size(symbol, confidence, regime_state)
        stop_loss, take_profit = self._calculate_risk_parameters(symbol, signal_type, market_data, regime_state)
        
        return LiveTradeSignal(
            symbol=symbol,
            signal_type=signal_type,
            confidence=confidence,
            agent_recommendation=best_agent,
            position_size=position_size,
            stop_loss=stop_loss,
            take_profit=take_profit,
            regime_context=regime_state.regime,
            reasoning=f"{best_agent} in {regime_state.regime} (conf: {confidence:.3f})",
            timestamp=datetime.now()
        )
    
    def _get_agent_regime_score(self, agent: str, regime: str) -> float:
        """Get agent performance score for specific regime"""
        
        # Base scores from our analysis
        base_scores = {
            'BERSERKER': {
                'CHAOTIC': 0.9,
                'UNDERDAMPED': 0.7,
                'CRITICALLY_DAMPED': 0.4,
                'OVERDAMPED': 0.2
            },
            'SNIPER': {
                'CHAOTIC': 0.4,
                'UNDERDAMPED': 0.8,
                'CRITICALLY_DAMPED': 0.7,
                'OVERDAMPED': 0.5
            }
        }
        
        base_score = base_scores.get(agent, {}).get(regime, 0.5)
        
        # Adjust based on recent learning (would use actual RL performance here)
        learning_adjustment = random.gauss(0, 0.1)  # Simulate learning adjustment
        
        return max(0.0, min(1.0, base_score + learning_adjustment))
    
    def _determine_signal_type(self, symbol: str, market_data: LiveMarketData, regime_state: MarketRegimeState) -> str:
        """Determine signal type based on market conditions"""
        
        # Simple momentum-based logic (would be replaced with sophisticated RL decision)
        momentum = market_data.momentum if hasattr(market_data, 'momentum') else random.gauss(0, 0.1)
        
        # Adjust thresholds based on regime
        if regime_state.regime == "CHAOTIC":
            threshold = 0.02  # Higher threshold in chaotic conditions
        else:
            threshold = 0.01
        
        if momentum > threshold:
            return "BUY"
        elif momentum < -threshold:
            return "SELL"
        else:
            return "HOLD"
    
    def _calculate_position_size(self, symbol: str, confidence: float, regime_state: MarketRegimeState) -> float:
        """Calculate position size based on confidence and regime"""
        
        # Base position size varies by symbol type for learning diversity (ECN symbols)
        symbol_sizes = {
            'EURUSD+': 0.1,   # Standard lot fraction
            'GBPUSD+': 0.08,  # Slightly smaller
            'USDJPY+': 0.12,  # Slightly larger  
            'BTCUSD+': 0.01,  # Much smaller for crypto
            'XAUUSD+': 0.05   # Small for gold
        }
        base_size = symbol_sizes.get(symbol, 0.1)
        
        # Adjust for confidence (more varied)
        confidence_multiplier = max(0.5, confidence * 2.0)  # Range 0.5 to 2.0
        
        # Adjust for regime volatility
        if regime_state.regime == "CHAOTIC":
            volatility_multiplier = random.uniform(0.3, 0.7)  # Random smaller positions
        elif regime_state.regime == "OVERDAMPED":
            volatility_multiplier = random.uniform(1.0, 2.5)  # Random larger positions
        else:
            volatility_multiplier = random.uniform(0.8, 1.5)  # Random moderate positions
        
        final_size = base_size * confidence_multiplier * volatility_multiplier
        return round(final_size, 2)  # Round to 2 decimal places
    
    def _calculate_risk_parameters(self, symbol: str, signal_type: str, market_data: LiveMarketData, regime_state: MarketRegimeState) -> Tuple[float, float]:
        """Calculate stop loss and take profit levels"""
        
        mid_price = (market_data.bid + market_data.ask) / 2
        
        # Base risk as percentage of price
        base_risk = 0.002  # 0.2%
        
        # Adjust for regime
        if regime_state.regime == "CHAOTIC":
            risk_multiplier = 2.0  # Wider stops in chaotic markets
        elif regime_state.regime == "OVERDAMPED":
            risk_multiplier = 0.5  # Tighter stops in stable markets
        else:
            risk_multiplier = 1.0
        
        risk_amount = mid_price * base_risk * risk_multiplier
        
        if signal_type == "BUY":
            stop_loss = mid_price - risk_amount
            take_profit = mid_price + risk_amount * 2  # 2:1 reward/risk
        else:  # SELL
            stop_loss = mid_price + risk_amount
            take_profit = mid_price - risk_amount * 2
        
        return stop_loss, take_profit
    
    def _update_active_positions(self):
        """Update all active positions and call callbacks"""
        
        if self.on_position_update_callback and self.active_positions:
            # Prepare position updates
            position_updates = []
            for trade_id, position in self.active_positions.items():
                if position.symbol in self.latest_market_data:
                    market_data = self.latest_market_data[position.symbol]
                    mid_price = (market_data.bid + market_data.ask) / 2
                    
                    position_updates.append({
                        'trade_id': trade_id,
                        'symbol': position.symbol,
                        'current_price': mid_price,
                        'unrealized_pnl': position.unrealized_pnl,
                        'mae': position.max_adverse_excursion,
                        'mfe': position.max_favorable_excursion
                    })
            
            if position_updates:
                self.on_position_update_callback(position_updates)
    
    def _run_learning_session(self, session_type: str):
        """Run periodic learning session"""
        
        print(f"\n🧠 Running {session_type} learning session...")
        
        try:
            # Run different types of learning based on session type
            if session_type == 'periodic':
                # Focus on recent experiences
                results = self.rl_engine.run_learning_session(focus_type='mixed', batch_size=32)
            elif session_type == 'final':
                # Comprehensive learning session
                results = self.rl_engine.run_learning_session(focus_type='efficient', batch_size=64)
            
            if results and 'experiences_processed' in results:
                print(f"   📚 Processed {results['experiences_processed']} experiences")
                print(f"   🎯 Learning insights updated")
            
        except Exception as e:
            print(f"   ❌ Learning session error: {e}")
    
    def on_trade_executed(self, trade_data: Dict[str, Any]) -> str:
        """Called when a trade is actually executed (by external system)"""
        
        # Create RL learning entry for the trade
        rl_result = self.rl_engine.on_trade_entry(trade_data)
        trade_id = rl_result['trade_id']
        
        print(f"📈 Trade executed: {trade_id}")
        print(f"   🎯 Confidence: {rl_result['confidence']:.3f}")
        print(f"   🔍 Exploration: {rl_result['is_exploration']}")
        
        return trade_id
    
    def on_trade_closed(self, trade_id: str, exit_data: Dict[str, Any]):
        """Called when a trade is closed"""
        
        # Complete the RL learning cycle
        rl_result = self.rl_engine.on_trade_exit(trade_id, exit_data)
        
        print(f"📉 Trade closed: {trade_id}")
        print(f"   💰 P&L: {exit_data['net_pnl']:+.0f}")
        print(f"   ⚡ Efficiency: {rl_result['efficiency_score']:.3f}")
        print(f"   📚 Learning value: {rl_result['learning_value']:.3f}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        
        return {
            'active_symbols': len(self.latest_market_data),
            'active_positions': len(self.active_positions),
            'latest_signals': {s: sig.signal_type for s, sig in self.latest_signals.items()},
            'regime_states': {s: self.regime_detector.get_current_regime(s).regime 
                             for s in self.symbols if self.regime_detector.get_current_regime(s)},
            'learning_summary': self.rl_engine.get_learning_summary(),
            'uptime': datetime.now().isoformat()
        }

def demonstrate_real_time_rl():
    """Demonstrate the real-time RL trading system"""
    
    print("🚀 REAL-TIME RL TRADING SYSTEM DEMONSTRATION")
    print("=" * 80)
    
    # Initialize system
    config = {
        'symbols': ['EURUSD', 'GBPUSD', 'BTCUSD'],
        'learning_frequency': 30,  # Faster learning for demo
        'signal_frequency': 3      # More frequent signals for demo
    }
    
    system = RealTimeRLTradingSystem(config)
    
    # Set up callbacks
    def on_trade_signal(signal: LiveTradeSignal):
        print(f"🔔 NEW SIGNAL: {signal.symbol} {signal.signal_type} "
              f"@ {signal.confidence:.3f} confidence")
        print(f"   Agent: {signal.agent_recommendation}")
        print(f"   Size: {signal.position_size:.2f}")
        print(f"   Regime: {signal.regime_context}")
        print(f"   Reasoning: {signal.reasoning}")
    
    def on_position_update(updates: List[Dict]):
        for update in updates[:2]:  # Show first 2 updates
            print(f"📊 Position Update: {update['symbol']} "
                  f"P&L: {update['unrealized_pnl']:+.0f}")
    
    system.set_trade_signal_callback(on_trade_signal)
    system.set_position_update_callback(on_position_update)
    
    # Run system for demonstration (would run indefinitely in production)
    try:
        print("\n🎯 Starting real-time system (demo for 60 seconds)...")
        
        # Start the system in a separate thread for demo
        import threading
        
        def run_system():
            system.start_real_time_trading()
        
        system_thread = threading.Thread(target=run_system, daemon=True)
        system_thread.start()
        
        # Let it run for demo period
        demo_duration = 60  # seconds
        start_time = time.time()
        
        while time.time() - start_time < demo_duration:
            # Show periodic status
            if int(time.time() - start_time) % 15 == 0:
                status = system.get_system_status()
                print(f"\n📊 System Status:")
                print(f"   Active symbols: {status['active_symbols']}")
                print(f"   Latest signals: {status['latest_signals']}")
                print(f"   Regime states: {status['regime_states']}")
            
            time.sleep(1)
        
        print("\n⏹️  Demo completed - stopping system")
        system.stop_trading()
        
    except KeyboardInterrupt:
        print("\n🛑 Demo interrupted - stopping system")
        system.stop_trading()
    
    print("\n✅ Real-time RL demonstration completed!")
    print("🎯 System is ready for live trading integration")

if __name__ == "__main__":
    demonstrate_real_time_rl()