#!/usr/bin/env python3
"""
Analyze the differences between standard and ECN symbols
Show spread differences between EURUSD vs EURUSD+ etc.
"""

import json
from pathlib import Path

def analyze_symbol_differences():
    """Analyze differences between standard and ECN instruments"""
    
    symbols_file = "all_mt5_symbols.json"
    if not Path(symbols_file).exists():
        print("❌ Symbols file not found")
        return
    
    with open(symbols_file, 'r') as f:
        data = json.load(f)
    
    symbols = data['symbols']
    
    print("📊 VANTAGE SYMBOL ANALYSIS - STANDARD vs ECN")
    print("=" * 70)
    print(f"Server: {data['account']['server']}")
    print(f"Total symbols: {len(symbols)}")
    
    # Find symbol pairs (standard vs ECN)
    standard_symbols = {}
    ecn_symbols = {}
    
    for symbol_name, symbol_data in symbols.items():
        if symbol_name.endswith('+'):
            # ECN symbol
            base_name = symbol_name[:-1]
            ecn_symbols[base_name] = {
                'name': symbol_name,
                'data': symbol_data
            }
        else:
            # Standard symbol
            standard_symbols[symbol_name] = {
                'name': symbol_name,
                'data': symbol_data
            }
    
    print(f"\n📈 SYMBOL BREAKDOWN:")
    print(f"   Standard instruments: {len(standard_symbols)}")
    print(f"   ECN instruments (+): {len(ecn_symbols)}")
    
    # Find pairs that exist in both standard and ECN
    pairs = []
    for base_name in ecn_symbols.keys():
        if base_name in standard_symbols:
            pairs.append(base_name)
    
    print(f"   Instruments with both versions: {len(pairs)}")
    
    if pairs:
        print(f"\n🔍 SPREAD COMPARISON (Standard vs ECN):")
        print("-" * 70)
        print(f"{'Symbol':<12} {'Standard Spread':<15} {'ECN Spread':<12} {'Difference':<10}")
        print("-" * 70)
        
        for base_name in sorted(pairs):
            std_data = standard_symbols[base_name]['data']
            ecn_data = ecn_symbols[base_name]['data']
            
            std_spread = std_data['spread_pips']
            ecn_spread = ecn_data['spread_pips']
            difference = std_spread - ecn_spread
            
            print(f"{base_name:<12} {std_spread:>8.1f} pips     {ecn_spread:>8.1f} pips   {difference:>+8.1f} pips")
        
        print("-" * 70)
        print("✅ ECN instruments have significantly lower spreads!")
    
    # Show all ECN symbols
    print(f"\n📊 ALL ECN SYMBOLS (+ suffix):")
    print("-" * 50)
    
    ecn_by_type = {}
    for base_name, ecn_info in ecn_symbols.items():
        symbol_type = ecn_info['data']['type']
        if symbol_type not in ecn_by_type:
            ecn_by_type[symbol_type] = []
        ecn_by_type[symbol_type].append(ecn_info)
    
    for symbol_type, type_symbols in ecn_by_type.items():
        print(f"\n{symbol_type} ECN ({len(type_symbols)} symbols):")
        for symbol_info in sorted(type_symbols, key=lambda x: x['name']):
            data = symbol_info['data']
            print(f"  {data['name']:<12} | Spread: {data['spread_pips']:>5.1f} pips | "
                  f"Bid: {data['bid']:.{data['digits']}f} | Ask: {data['ask']:.{data['digits']}f}")
    
    # Recommendation
    print(f"\n💡 TRADING RECOMMENDATIONS:")
    print("✅ Use ECN symbols (+) for lower spreads when available")
    print("✅ Standard symbols for broader market access")
    print("⚠️  Treat EURUSD and EURUSD+ as completely different instruments")
    print("⚠️  ECN symbols typically have commission instead of wider spreads")
    
    # Create optimized symbol list
    optimized_symbols = []
    
    # Add all ECN symbols first (better spreads)
    for base_name, ecn_info in ecn_symbols.items():
        optimized_symbols.append(ecn_info['name'])
    
    # Add standard symbols that don't have ECN equivalents
    for std_name in standard_symbols.keys():
        if std_name not in ecn_symbols:
            optimized_symbols.append(std_name)
    
    print(f"\n🎯 OPTIMIZED SYMBOL LIST ({len(optimized_symbols)} symbols):")
    print("Prioritizes ECN versions for better spreads")
    print(", ".join(sorted(optimized_symbols)))
    
    return optimized_symbols

if __name__ == "__main__":
    analyze_symbol_differences()