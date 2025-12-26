#!/usr/bin/env python3
"""
Start trading system with REAL MT5 symbols from Vantage International Demo
Uses actual symbol list and real-time data
"""

import sys
import json
from pathlib import Path

# Load REAL symbols from MT5
def load_real_symbols():
    """Load real symbols from the fetched MT5 data"""
    
    symbols_file = "all_mt5_symbols.json"
    if not Path(symbols_file).exists():
        print("❌ Real symbols file not found. Run fetch_all_mt5_symbols.py first")
        return []
    
    with open(symbols_file, 'r') as f:
        data = json.load(f)
    
    if not data.get('success'):
        print("❌ Invalid symbols data")
        return []
    
    # Extract symbol names
    symbols = list(data['symbols'].keys())
    
    print(f"✅ Loaded {len(symbols)} REAL symbols from Vantage International Demo:")
    print(f"   Server: {data['account']['server']}")
    print(f"   Account: {data['account']['login']}")
    print(f"   Balance: ${data['account']['balance']:,.2f}")
    
    # Show symbol breakdown
    by_type = {}
    for symbol_name, symbol_data in data['symbols'].items():
        symbol_type = symbol_data['type']
        if symbol_type not in by_type:
            by_type[symbol_type] = []
        by_type[symbol_type].append(symbol_name)
    
    for symbol_type, type_symbols in by_type.items():
        print(f"   {symbol_type}: {len(type_symbols)} symbols")
    
    return symbols

def main():
    """Start the system with real Vantage symbols"""
    
    print("🚀 STARTING AI TRADING SYSTEM WITH REAL VANTAGE DATA")
    print("=" * 70)
    
    # Load real symbols
    real_symbols = load_real_symbols()
    if not real_symbols:
        return
    
    # Configuration with REAL symbols
    config = {
        'berserker_enabled': True,
        'sniper_enabled': True,
        'symbols': real_symbols,  # ALL 42 real symbols from your MT5
        'timeframes': ['M5', 'M15', 'H1'],
        'top_symbol_count': len(real_symbols),  # Use ALL symbols
        'learning_frequency': 30,
        'signal_frequency': 5,
        'dashboard_port': 5000,
        'model_directory': './rl_models',
        'save_interval': 30,
        'max_versions': 10,
        'trading_mode': 'shadow',
        'terminal_path': '/mnt/c/DevCenter/MT5-Unified/MT5-Core/Terminal',
        'broker': 'vantage_international_demo',
        'account_server': 'VantageInternational-Demo',
        'atomic_saving': {
            'per_instrument': True,
            'per_timeframe': True,
            'per_agent': True,
            'per_session': True
        }
    }
    
    print("✅ Configuration loaded:")
    print(f"   🤖 Agents: BERSERKER + SNIPER")
    print(f"   📊 Symbols: {len(real_symbols)} (all from MarketWatch)")
    print(f"   ⏰ Timeframes: {', '.join(config['timeframes'])}")
    print(f"   🏦 Broker: {config['broker']}")
    print(f"   🌐 Dashboard: http://localhost:{config['dashboard_port']}")
    
    # Import and start the integrated system
    try:
        print("\n🔄 Initializing integrated trading system...")
        from integrated_dual_agent_system import IntegratedDualAgentSystem
        
        system = IntegratedDualAgentSystem(config)
        
        print("\n🚀 Starting system...")
        system.start_system()
        
        print("\n✅ System running! Check the dashboard for real-time data")
        print("💡 Press Ctrl+C to stop")
        
        # Keep running
        import time
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n🛑 Stopping system...")
            system.stop_system()
            print("✅ System stopped")
            
    except Exception as e:
        print(f"❌ System error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()