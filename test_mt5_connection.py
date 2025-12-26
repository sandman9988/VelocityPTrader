#!/usr/bin/env python3
"""
Test MT5 Connection to Vantage International Terminal
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

try:
    from data.mt5_bridge import (
        initialize, shutdown, symbols_get, 
        symbol_info_tick, copy_rates_from
    )
    MT5_AVAILABLE = True
    print("✅ MT5 Bridge imported successfully")
except ImportError as e:
    print(f"❌ MT5 Bridge import failed: {e}")
    MT5_AVAILABLE = False

def test_mt5_connection():
    """Test connection to Vantage International terminal"""
    
    print("\n🔌 TESTING MT5 CONNECTION TO VANTAGE INTERNATIONAL")
    print("=" * 60)
    
    if not MT5_AVAILABLE:
        print("❌ MT5 Bridge not available")
        return False
    
    try:
        # Try to initialize
        print("🔄 Initializing MT5 connection...")
        success = initialize()
        
        if not success:
            print("❌ Failed to initialize MT5")
            return False
        
        print("✅ MT5 initialized successfully")
        
        # Get symbols
        print("🔄 Fetching symbols from terminal...")
        symbols = symbols_get()
        
        if symbols is None:
            print("❌ No symbols returned from terminal")
            return False
        
        print(f"✅ Found {len(symbols)} symbols in MarketWatch")
        
        # Show first 10 symbols
        print("\n📊 SYMBOLS IN YOUR VANTAGE TERMINAL:")
        print("-" * 40)
        for i, symbol in enumerate(symbols[:10]):
            print(f"  {i+1}. {symbol.name} - {symbol.description}")
        
        if len(symbols) > 10:
            print(f"  ... and {len(symbols) - 10} more symbols")
        
        # Test getting tick data for a few symbols
        print("\n💹 TESTING REAL TICK DATA:")
        print("-" * 30)
        
        test_symbols = []
        for symbol in symbols[:5]:
            test_symbols.append(symbol.name)
        
        for symbol_name in test_symbols:
            try:
                tick = symbol_info_tick(symbol_name)
                if tick:
                    spread = tick.ask - tick.bid
                    print(f"  📈 {symbol_name}:")
                    print(f"     Bid: {tick.bid}")
                    print(f"     Ask: {tick.ask}")
                    print(f"     Spread: {spread}")
                    print(f"     Time: {tick.time}")
                else:
                    print(f"  ❌ {symbol_name}: No tick data")
            except Exception as e:
                print(f"  ❌ {symbol_name}: Error - {e}")
        
        print(f"\n✅ MT5 CONNECTION TO VANTAGE TERMINAL SUCCESSFUL!")
        print(f"✅ Ready to use REAL market data in RL system")
        
        return True
        
    except Exception as e:
        print(f"❌ Connection error: {e}")
        return False
    
    finally:
        try:
            shutdown()
            print("🔌 MT5 connection closed")
        except:
            pass

if __name__ == "__main__":
    test_mt5_connection()