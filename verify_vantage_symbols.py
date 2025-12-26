#!/usr/bin/env python3
"""
Verify EXACT symbols and pricing from Vantage International Demo Server
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from data.mt5_marketwatch_integration import MT5MarketWatchManager

def verify_vantage_symbols():
    """Get and display EXACT Vantage symbols and current prices"""
    
    print("🔍 VERIFYING VANTAGE INTERNATIONAL DEMO SERVER SYMBOLS...")
    print("=" * 60)
    
    try:
        # Initialize MT5 connection
        mw_manager = MT5MarketWatchManager()
        
        # Get ALL symbols from MarketWatch
        all_symbols = mw_manager.get_marketwatch_symbols(visible_only=True)
        
        print(f"\n✅ Found {len(all_symbols)} symbols in Vantage MarketWatch:\n")
        
        # Get current prices for each symbol
        for i, symbol in enumerate(all_symbols):
            if hasattr(symbol, 'name'):
                sym_name = symbol.name
            else:
                sym_name = str(symbol)
            
            # Get current tick data using proper method
            from data.mt5_bridge import symbol_info_tick
            tick = symbol_info_tick(sym_name)
            if tick:
                bid = tick.bid
                ask = tick.ask
                spread_points = round((ask - bid) * (10 ** symbol.digits), 1)
                
                print(f"{i+1:2}. {sym_name:15} Bid: {bid:.5f}  Ask: {ask:.5f}  Spread: {spread_points} points")
            else:
                print(f"{i+1:2}. {sym_name:15} [No tick data]")
        
        # Show symbol info
        print("\nSYMBOL SPECIFICATIONS:")
        print("-" * 60)
        
        for symbol in all_symbols[:5]:  # Show first 5 as examples
            if hasattr(symbol, 'name'):
                from data.mt5_bridge import symbol_info
                info = symbol_info(symbol.name)
                if info:
                    print(f"\n{symbol.name}:")
                    print(f"  Digits: {info.digits}")
                    print(f"  Point: {info.point}")
                    print(f"  Tick size: {info.trade_tick_size}")
                    print(f"  Tick value: {info.trade_tick_value}")
                    print(f"  Spread (current): {info.spread}")
                    print(f"  Contract size: {info.trade_contract_size}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nMake sure:")
        print("1. MT5 terminal is running")
        print("2. Logged into Vantage International Demo")
        print("3. MarketWatch panel is open with symbols")

if __name__ == "__main__":
    verify_vantage_symbols()