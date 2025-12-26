#!/usr/bin/env python3
"""
Start the system with EXACT Vantage International Demo Server configuration
NO FAKE DATA - REAL VANTAGE SYMBOLS AND PRICES ONLY!
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

# Force the correct Vantage symbols (NO + suffix)
VANTAGE_SYMBOLS = [
    'EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD', 
    'USDCHF', 'NZDUSD', 'EURGBP', 'EURJPY', 'GBPJPY',
    'XAUUSD', 'XAGUSD', 'BTCUSD', 'ETHUSD', 'US30',
    'NAS100', 'SPX500', 'GER40', 'USOIL', 'UKBRENT'
]

# Override the config
config = {
    'berserker_enabled': True,
    'sniper_enabled': True,
    'symbols': VANTAGE_SYMBOLS,  # EXACT Vantage symbols
    'timeframes': ['M5', 'M15', 'H1'],
    'top_symbol_count': 20,  # Use ALL symbols
    'learning_frequency': 30,
    'signal_frequency': 3,
    'dashboard_port': 5000,
    'model_directory': './rl_models',
    'save_interval': 30,
    'max_versions': 10,
    'trading_mode': 'shadow',
    'terminal_path': '/mnt/c/DevCenter/MT5-Unified/MT5-Core/Terminal',
    'broker': 'vantage',
    'atomic_saving': {
        'per_instrument': True,
        'per_timeframe': True,
        'per_agent': True,
        'per_session': True
    }
}

print("🚀 STARTING WITH VANTAGE INTERNATIONAL DEMO CONFIGURATION")
print("=" * 60)
print(f"✅ Using EXACT Vantage symbols: {', '.join(VANTAGE_SYMBOLS[:5])}...")
print("✅ NO fake data - REAL Vantage prices only")
print("✅ NO symbol suffix modifications")
print("=" * 60)

# Import and start the system
from integrated_dual_agent_system import IntegratedDualAgentSystem

# Start with Vantage config
system = IntegratedDualAgentSystem(config)
system.start_system()

try:
    # Keep running
    import time
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("\n🛑 Stopping...")
    system.stop_system()