#!/usr/bin/env python3
"""
Test REAL MT5 data connection to ensure we get ACTUAL Vantage prices
"""

import MetaTrader5 as mt5
import time

print("🔍 TESTING REAL MT5 DATA CONNECTION TO VANTAGE...")
print("=" * 60)

# Initialize MT5 connection
if not mt5.initialize():
    print("❌ Failed to initialize MT5")
    quit()

print("✅ MT5 initialized")

# Get account info to verify Vantage connection
account = mt5.account_info()
if account:
    print(f"\n📊 Connected to: {account.server}")
    print(f"👤 Account: {account.login}")
    print(f"💰 Balance: ${account.balance}")
    print(f"🏢 Company: {account.company}")

# Get REAL tick data for key symbols
symbols = ['EURUSD', 'GBPUSD', 'BTCUSD', 'XAUUSD', 'US30']

print(f"\n📈 REAL VANTAGE TICK DATA:")
print("-" * 60)

for symbol in symbols:
    tick = mt5.symbol_info_tick(symbol)
    if tick:
        info = mt5.symbol_info(symbol)
        spread_points = tick.ask - tick.bid
        
        if info:
            spread_money = spread_points / info.point
            
            print(f"\n{symbol}:")
            print(f"  Bid: {tick.bid}")
            print(f"  Ask: {tick.ask}")
            print(f"  Spread (points): {spread_points:.5f}")
            print(f"  Spread (pips): {spread_money:.1f}")
            print(f"  Time: {time.ctime(tick.time)}")
            print(f"  Digits: {info.digits}")
    else:
        print(f"\n{symbol}: NO TICK DATA")

mt5.shutdown()
print("\n✅ Test complete")