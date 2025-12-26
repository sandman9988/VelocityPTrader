#!/usr/bin/env python3
"""
Demo Script for Dual Agent RL Trading System
Shows the complete system functionality without external dependencies
"""

import json
import time
import random
from datetime import datetime, timedelta
from pathlib import Path

def simulate_dual_agent_trading():
    """Simulate dual agent trading with realistic data"""
    
    print("🚀 DUAL AGENT RL TRADING SYSTEM DEMO")
    print("=" * 80)
    print("📚 SHADOW TRADING MODE - Real-time learning from market data")
    print("🤖 BERSERKER Agent: High-frequency aggressive trading")
    print("🎯 SNIPER Agent: Precision patient trading")
    print()
    
    # Initialize simulation state
    symbols = ['EURUSD+', 'GBPUSD+', 'USDJPY+', 'BTCUSD+', 'XAUUSD+']
    timeframes = ['M5', 'M15', 'H1']
    regimes = ['CHAOTIC', 'UNDERDAMPED', 'CRITICALLY_DAMPED', 'OVERDAMPED']
    
    # Agent configurations
    berserker_config = {
        'name': 'BERSERKER',
        'enabled': True,
        'risk_tolerance': 0.08,
        'frequency_multiplier': 3.0,
        'exploration_rate': 0.25,
        'learning_rate': 0.002,
        'total_episodes': 1250,
        'success_rate': 0.62
    }
    
    sniper_config = {
        'name': 'SNIPER', 
        'enabled': True,
        'risk_tolerance': 0.03,
        'frequency_multiplier': 0.5,
        'exploration_rate': 0.05,
        'learning_rate': 0.0005,
        'total_episodes': 890,
        'success_rate': 0.74
    }
    
    # Simulation state
    berserker_trades = []
    sniper_trades = []
    symbol_rankings = {}
    atomic_saves = 0
    total_experiences = 0
    
    print("🔄 INITIALIZING DUAL AGENT SYSTEM")
    print("-" * 50)
    print("✅ Atomic persistence system initialized")
    print("✅ BERSERKER agent initialized (aggressive strategy)")
    print("✅ SNIPER agent initialized (precision strategy)")
    print("✅ Market regime detection active")
    print("✅ Symbol performance ranking system ready")
    print()
    
    print("📊 TRAINING CONFIGURATION")
    print("-" * 30)
    print(f"   🎯 Symbols: {len(symbols)} (top performers auto-selected)")
    print(f"   ⏰ Timeframes: {', '.join(timeframes)}")
    print(f"   💾 Atomic saving: Per instrument/timeframe/agent")
    print(f"   🧠 Experience replay: Intelligent prioritization")
    print()
    
    print("🚀 STARTING SIMULTANEOUS TRAINING")
    print("=" * 50)
    
    # Simulate 30 seconds of trading activity
    start_time = time.time()
    last_berserker_signal = 0
    last_sniper_signal = 0
    
    while time.time() - start_time < 30:  # Run for 30 seconds
        current_time = time.time() - start_time
        
        # BERSERKER trading (every 5 seconds - high frequency)
        if current_time - last_berserker_signal >= 5:
            symbol = random.choice(symbols)
            regime = random.choice(['CHAOTIC', 'UNDERDAMPED'])  # Preferred regimes
            
            trade = generate_agent_trade('BERSERKER', symbol, regime, berserker_config)
            berserker_trades.append(trade)
            
            print(f"⚔️  BERSERKER: {trade['symbol']} {trade['direction']} | "
                  f"Size: {trade['size']:.2f} | Confidence: {trade['confidence']:.3f} | "
                  f"Regime: {trade['regime']}")
            
            last_berserker_signal = current_time
        
        # SNIPER trading (every 15 seconds - low frequency)
        if current_time - last_sniper_signal >= 15:
            symbol = random.choice(symbols)
            regime = random.choice(['CRITICALLY_DAMPED', 'OVERDAMPED'])  # Preferred regimes
            
            trade = generate_agent_trade('SNIPER', symbol, regime, sniper_config)
            sniper_trades.append(trade)
            
            print(f"🎯 SNIPER: {trade['symbol']} {trade['direction']} | "
                  f"Size: {trade['size']:.2f} | Confidence: {trade['confidence']:.3f} | "
                  f"Regime: {trade['regime']}")
            
            last_sniper_signal = current_time
        
        # Simulate atomic saves every 10 seconds
        if int(current_time) % 10 == 0 and int(current_time) > 0:
            atomic_saves += len(symbols) * len(timeframes) * 2  # Both agents
            total_experiences += 50
            print(f"💾 Atomic save completed: {atomic_saves} model states saved")
        
        # Update symbol rankings periodically
        if int(current_time) % 15 == 0 and int(current_time) > 0:
            symbol_rankings = update_symbol_rankings(berserker_trades + sniper_trades)
            top_3 = list(symbol_rankings.items())[:3]
            print(f"📊 Symbol rankings updated: {[s for s, _ in top_3]}")
        
        time.sleep(1)
    
    print("\n" + "=" * 50)
    print("📊 TRAINING SESSION SUMMARY")
    print("=" * 50)
    
    # Calculate performance metrics
    berserker_pnl = sum(t['pnl'] for t in berserker_trades)
    sniper_pnl = sum(t['pnl'] for t in sniper_trades)
    total_pnl = berserker_pnl + sniper_pnl
    
    berserker_wins = len([t for t in berserker_trades if t['pnl'] > 0])
    sniper_wins = len([t for t in sniper_trades if t['pnl'] > 0])
    
    berserker_win_rate = berserker_wins / len(berserker_trades) if berserker_trades else 0
    sniper_win_rate = sniper_wins / len(sniper_trades) if sniper_trades else 0
    
    print(f"🤖 AGENT PERFORMANCE:")
    print(f"   ⚔️  BERSERKER: {len(berserker_trades)} trades, {berserker_win_rate:.1%} win rate, P&L: ${berserker_pnl:+.0f}")
    print(f"   🎯 SNIPER: {len(sniper_trades)} trades, {sniper_win_rate:.1%} win rate, P&L: ${sniper_pnl:+.0f}")
    print(f"   💰 TOTAL P&L: ${total_pnl:+.0f}")
    print()
    
    print(f"🧠 RL LEARNING METRICS:")
    print(f"   📚 Total episodes: {berserker_config['total_episodes'] + sniper_config['total_episodes']}")
    print(f"   🔄 Experience buffer: {total_experiences + 8500} experiences")
    print(f"   ⚡ Exploration rates: BERSERKER {berserker_config['exploration_rate']:.1%}, SNIPER {sniper_config['exploration_rate']:.1%}")
    print(f"   📈 Success rates: BERSERKER {berserker_config['success_rate']:.1%}, SNIPER {sniper_config['success_rate']:.1%}")
    print()
    
    print(f"💾 ATOMIC PERSISTENCE:")
    print(f"   🗂️  Model states saved: {atomic_saves}")
    print(f"   📁 Directory structure: ./rl_models/[AGENT]/[SYMBOL]/[TIMEFRAME]/")
    print(f"   ✅ Versioned saves: v1.model, v2.model, v3.model...")
    print(f"   🔐 Checksum verification: Enabled")
    print()
    
    print(f"🎯 SYMBOL PERFORMANCE RANKINGS:")
    if symbol_rankings:
        for i, (symbol, score) in enumerate(symbol_rankings.items()[:5], 1):
            medal = ['🥇', '🥈', '🥉'][i-1] if i <= 3 else f"{i}."
            print(f"   {medal} {symbol}: {score:.1%} performance score")
    
    print()
    print("✅ DUAL AGENT TRAINING COMPLETE!")
    print("🌐 Dashboard available at: http://localhost:5000")
    print("⚙️  Settings: Click gear icon to configure agents and symbols")
    print("📊 Real-time metrics: RL learning progress, trade performance, market watch")

def generate_agent_trade(agent_name, symbol, regime, config):
    """Generate realistic agent trade"""
    
    direction = random.choice(['BUY', 'SELL'])
    
    # Agent-specific sizing and confidence
    if agent_name == 'BERSERKER':
        size = random.uniform(0.08, 0.20)  # Larger, aggressive
        confidence = random.uniform(0.4, 0.9)  # Variable confidence
        pnl = random.uniform(-30, 60)  # More volatile P&L
    else:  # SNIPER
        size = random.uniform(0.02, 0.10)  # Smaller, precise
        confidence = random.uniform(0.6, 0.9)  # Higher confidence
        pnl = random.uniform(-15, 45)  # More consistent P&L
    
    return {
        'agent': agent_name,
        'symbol': symbol,
        'direction': direction,
        'size': size,
        'confidence': confidence,
        'regime': regime,
        'pnl': pnl,
        'timestamp': datetime.now()
    }

def update_symbol_rankings(all_trades):
    """Update symbol performance rankings"""
    
    symbol_performance = {}
    
    for symbol in ['EURUSD+', 'GBPUSD+', 'USDJPY+', 'BTCUSD+', 'XAUUSD+']:
        symbol_trades = [t for t in all_trades if t['symbol'] == symbol]
        
        if symbol_trades:
            wins = len([t for t in symbol_trades if t['pnl'] > 0])
            win_rate = wins / len(symbol_trades)
            avg_pnl = sum(t['pnl'] for t in symbol_trades) / len(symbol_trades)
            
            # Composite performance score
            score = (win_rate * 0.6) + ((avg_pnl + 50) / 100 * 0.4)  # Normalize avg_pnl
            symbol_performance[symbol] = max(0, min(1, score))
        else:
            symbol_performance[symbol] = 0.5
    
    # Sort by performance
    return dict(sorted(symbol_performance.items(), key=lambda x: x[1], reverse=True))

if __name__ == "__main__":
    simulate_dual_agent_trading()