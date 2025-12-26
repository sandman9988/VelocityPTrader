#!/usr/bin/env python3
"""
Direct MT5 Symbol Retrieval - Get ALL MetaTrader 5 symbols in minutes!
Based on the YouTube tutorial approach
"""

import sys
import time
from datetime import datetime
from typing import List, Dict, Optional

def install_mt5_package():
    """Install MetaTrader5 package if needed"""
    try:
        import MetaTrader5 as mt5
        return True
    except ImportError:
        print("📦 Installing MetaTrader5 package...")
        import subprocess
        result = subprocess.run([sys.executable, '-m', 'pip', 'install', 'MetaTrader5'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ MetaTrader5 package installed successfully!")
            return True
        else:
            print(f"❌ Failed to install MetaTrader5: {result.stderr}")
            return False

def get_all_mt5_symbols() -> Dict[str, Dict]:
    """Get ALL symbols from MetaTrader 5 - the fast way!"""
    
    # Try to install/import MT5 package
    if not install_mt5_package():
        return {}
    
    try:
        import MetaTrader5 as mt5
    except ImportError:
        print("❌ MetaTrader5 package still not available")
        return {}
    
    print("🚀 GETTING ALL MT5 SYMBOLS - THE FAST WAY!")
    print("=" * 60)
    
    # Initialize connection to MT5
    if not mt5.initialize():
        print("❌ Failed to initialize MT5 connection")
        print("Make sure your MT5 terminal is running and logged in")
        return {}
    
    print("✅ MT5 connection established")
    
    # Get account info
    account_info = mt5.account_info()
    if account_info:
        print(f"🏦 Connected to: {account_info.server}")
        print(f"👤 Account: {account_info.login}")
        print(f"💰 Balance: ${account_info.balance:,.2f}")
        print(f"🏢 Company: {account_info.company}")
    
    print("\n📊 Retrieving ALL symbols...")
    
    # Method 1: Get ALL available symbols
    print("\n1️⃣ Getting ALL available symbols...")
    all_symbols = mt5.symbols_get()
    if all_symbols:
        print(f"   Found {len(all_symbols)} total symbols available")
    else:
        print("   ❌ No symbols found")
        
    # Method 2: Get MarketWatch symbols (visible symbols)
    print("\n2️⃣ Getting MarketWatch symbols...")
    marketwatch_symbols = mt5.symbols_get(group="*")  # All symbols
    visible_symbols = [s for s in marketwatch_symbols if s.visible] if marketwatch_symbols else []
    print(f"   Found {len(visible_symbols)} visible symbols in MarketWatch")
    
    # Method 3: Get specific symbol groups
    print("\n3️⃣ Getting symbols by groups...")
    
    # Common symbol groups
    groups = [
        ("Forex", "*USD*,*EUR*,*GBP*,*AUD*,*CAD*,*CHF*,*JPY*,*NZD*"),
        ("Crypto", "*BTC*,*ETH*,*LTC*,*XRP*"),
        ("Indices", "*US30*,*NAS100*,*SPX500*,*GER40*,*UK100*"),
        ("Metals", "*XAU*,*XAG*,*GOLD*,*SILVER*"),
        ("Energy", "*OIL*,*BRENT*,*NGAS*")
    ]
    
    all_grouped_symbols = {}
    
    for group_name, group_filter in groups:
        group_symbols = mt5.symbols_get(group=group_filter)
        if group_symbols:
            # Filter for visible only
            group_visible = [s for s in group_symbols if s.visible]
            all_grouped_symbols[group_name] = group_visible
            print(f"   {group_name}: {len(group_visible)} symbols")
    
    # Compile final symbol list
    print("\n📋 COMPILING FINAL SYMBOL LIST...")
    
    final_symbols = {}
    
    # Use visible MarketWatch symbols as primary source
    if visible_symbols:
        for symbol in visible_symbols:
            # Get current tick data
            tick = mt5.symbol_info_tick(symbol.name)
            
            # Get detailed symbol info
            symbol_info = mt5.symbol_info(symbol.name)
            
            if tick and symbol_info:
                # Calculate spread
                spread_points = tick.ask - tick.bid
                spread_pips = spread_points / symbol_info.point if symbol_info.point > 0 else 0
                
                # Determine symbol type
                symbol_type = "UNKNOWN"
                if any(curr in symbol.name for curr in ["USD", "EUR", "GBP", "AUD", "CAD", "CHF", "JPY", "NZD"]):
                    symbol_type = "FOREX"
                elif any(crypto in symbol.name for crypto in ["BTC", "ETH", "LTC", "XRP"]):
                    symbol_type = "CRYPTO"
                elif any(idx in symbol.name for idx in ["US30", "NAS100", "SPX500", "GER40", "UK100"]):
                    symbol_type = "INDEX"
                elif any(metal in symbol.name for metal in ["XAU", "XAG", "GOLD", "SILVER"]):
                    symbol_type = "METAL"
                elif any(energy in symbol.name for energy in ["OIL", "BRENT", "NGAS"]):
                    symbol_type = "ENERGY"
                
                final_symbols[symbol.name] = {
                    'name': symbol.name,
                    'description': symbol.description,
                    'type': symbol_type,
                    'visible': symbol.visible,
                    'digits': symbol_info.digits,
                    'point': symbol_info.point,
                    'spread': spread_points,
                    'spread_pips': spread_pips,
                    'bid': tick.bid,
                    'ask': tick.ask,
                    'last_update': datetime.fromtimestamp(tick.time),
                    'contract_size': symbol_info.trade_contract_size,
                    'margin_required': symbol_info.margin_initial,
                    'swap_long': symbol_info.swap_long,
                    'swap_short': symbol_info.swap_short,
                    'server': account_info.server if account_info else 'Unknown'
                }
    
    # Clean up
    mt5.shutdown()
    
    print(f"\n✅ SUCCESSFULLY RETRIEVED {len(final_symbols)} REAL SYMBOLS!")
    return final_symbols

def display_symbols_summary(symbols: Dict[str, Dict]):
    """Display a nice summary of all symbols"""
    
    if not symbols:
        print("❌ No symbols to display")
        return
    
    print(f"\n📊 SYMBOL SUMMARY ({len(symbols)} symbols)")
    print("=" * 80)
    
    # Group by type
    by_type = {}
    for symbol, data in symbols.items():
        symbol_type = data['type']
        if symbol_type not in by_type:
            by_type[symbol_type] = []
        by_type[symbol_type].append((symbol, data))
    
    # Display by type
    for symbol_type, type_symbols in by_type.items():
        print(f"\n{symbol_type} ({len(type_symbols)} symbols):")
        print("-" * 60)
        
        for symbol, data in sorted(type_symbols)[:10]:  # Show first 10 of each type
            print(f"{symbol:12} | Bid: {data['bid']:10.{data['digits']}f} | "
                  f"Ask: {data['ask']:10.{data['digits']}f} | "
                  f"Spread: {data['spread_pips']:6.1f} pips")
        
        if len(type_symbols) > 10:
            print(f"... and {len(type_symbols) - 10} more {symbol_type} symbols")
    
    print("\n" + "=" * 80)
    print(f"🎯 Server: {list(symbols.values())[0]['server']}")
    print(f"⏰ Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

def save_symbols_to_json(symbols: Dict[str, Dict], filename: str = "mt5_symbols.json"):
    """Save symbols to JSON file"""
    
    import json
    from datetime import datetime
    
    # Convert datetime objects to strings for JSON serialization
    json_symbols = {}
    for symbol, data in symbols.items():
        json_data = data.copy()
        if 'last_update' in json_data:
            json_data['last_update'] = json_data['last_update'].isoformat()
        json_symbols[symbol] = json_data
    
    # Add metadata
    export_data = {
        'metadata': {
            'export_time': datetime.now().isoformat(),
            'total_symbols': len(symbols),
            'source': 'MetaTrader5 Direct API'
        },
        'symbols': json_symbols
    }
    
    with open(filename, 'w') as f:
        json.dump(export_data, f, indent=2)
    
    print(f"💾 Symbols saved to: {filename}")

def main():
    """Main function - get all MT5 symbols quickly!"""
    
    print("🚀 MT5 SYMBOL EXTRACTOR - GET ALL SYMBOLS IN MINUTES!")
    print("Based on YouTube tutorial method")
    print("=" * 80)
    
    # Get all symbols
    symbols = get_all_mt5_symbols()
    
    if symbols:
        # Display summary
        display_symbols_summary(symbols)
        
        # Save to file
        save_symbols_to_json(symbols)
        
        print("\n🎉 SUCCESS! All symbols extracted successfully!")
        print(f"📊 Total symbols: {len(symbols)}")
        print("💾 Data saved to mt5_symbols.json")
        
        # Show how to use in trading system
        print("\n🔗 TO USE IN YOUR TRADING SYSTEM:")
        print("1. Import the symbols: symbols = json.load(open('mt5_symbols.json'))")
        print("2. Use symbol names: symbol_list = list(symbols['symbols'].keys())")
        print("3. Get real-time data for any symbol using MetaTrader5 package")
        
    else:
        print("\n❌ FAILED TO GET SYMBOLS")
        print("Make sure:")
        print("1. MT5 terminal is running")
        print("2. You're logged into your broker account")
        print("3. MarketWatch panel has symbols visible")
        print("4. MetaTrader5 package is installed")

if __name__ == "__main__":
    main()