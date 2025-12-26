#!/usr/bin/env python3
"""
Fetch ALL MT5 symbols using Windows execution
Works from WSL2 by executing Windows Python with MetaTrader5 package
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

class WindowsMT5SymbolFetcher:
    """Fetch symbols by running Windows Python from WSL2"""
    
    def __init__(self):
        self.windows_python_paths = [
            "python.exe",
            "py.exe", 
            "/mnt/c/Python*/python.exe",
            "/mnt/c/Users/*/AppData/Local/Programs/Python/*/python.exe"
        ]
        
    def create_windows_mt5_script(self) -> str:
        """Create Windows Python script to fetch MT5 symbols"""
        
        script_content = '''
import sys
import json
import subprocess
from datetime import datetime

def install_mt5():
    """Install MetaTrader5 package"""
    try:
        import MetaTrader5 as mt5
        return True
    except ImportError:
        print("Installing MetaTrader5...")
        result = subprocess.run([sys.executable, '-m', 'pip', 'install', 'MetaTrader5'], 
                              capture_output=True, text=True)
        return result.returncode == 0

def fetch_all_symbols():
    """Fetch all MT5 symbols"""
    
    if not install_mt5():
        return {"error": "Failed to install MetaTrader5 package"}
    
    try:
        import MetaTrader5 as mt5
    except ImportError:
        return {"error": "MetaTrader5 package not available"}
    
    # Initialize MT5
    if not mt5.initialize():
        return {"error": "Failed to connect to MT5. Make sure terminal is running and logged in."}
    
    try:
        # Get account info
        account_info = mt5.account_info()
        account_data = {
            "server": account_info.server if account_info else "Unknown",
            "login": account_info.login if account_info else 0,
            "company": account_info.company if account_info else "Unknown",
            "balance": account_info.balance if account_info else 0
        }
        
        # Get ALL symbols
        all_symbols = mt5.symbols_get()
        if not all_symbols:
            return {"error": "No symbols found"}
        
        # Get MarketWatch (visible) symbols  
        visible_symbols = [s for s in all_symbols if s.visible]
        
        # Process symbols
        symbol_data = {}
        
        for symbol in visible_symbols:
            try:
                # Get tick data
                tick = mt5.symbol_info_tick(symbol.name)
                if not tick:
                    continue
                
                # Get symbol info
                info = mt5.symbol_info(symbol.name)
                if not info:
                    continue
                
                # Calculate spread
                spread_points = tick.ask - tick.bid
                spread_pips = spread_points / info.point if info.point > 0 else 0
                
                # Determine type
                symbol_type = "UNKNOWN"
                name = symbol.name.upper()
                if any(curr in name for curr in ["USD", "EUR", "GBP", "AUD", "CAD", "CHF", "JPY", "NZD"]) and len(symbol.name) <= 8:
                    symbol_type = "FOREX"
                elif any(crypto in name for crypto in ["BTC", "ETH", "LTC", "XRP", "ADA", "DOT"]):
                    symbol_type = "CRYPTO"  
                elif any(idx in name for idx in ["US30", "NAS100", "SPX500", "GER40", "UK100", "AUS200", "JP225"]):
                    symbol_type = "INDEX"
                elif any(metal in name for metal in ["XAU", "XAG", "GOLD", "SILVER", "COPPER", "PLATINUM"]):
                    symbol_type = "METAL"
                elif any(energy in name for energy in ["OIL", "BRENT", "NGAS", "HEATING"]):
                    symbol_type = "ENERGY"
                elif any(bond in name for bond in ["BOND", "NOTE", "GILT"]):
                    symbol_type = "BOND"
                
                symbol_data[symbol.name] = {
                    "name": symbol.name,
                    "description": symbol.description,
                    "type": symbol_type,
                    "visible": symbol.visible,
                    "digits": info.digits,
                    "point": info.point,
                    "spread_points": spread_points,
                    "spread_pips": round(spread_pips, 1),
                    "bid": tick.bid,
                    "ask": tick.ask,
                    "last_price": tick.last,
                    "time": tick.time,
                    "contract_size": info.trade_contract_size,
                    "margin_initial": info.margin_initial,
                    "swap_long": info.swap_long,
                    "swap_short": info.swap_short,
                    "min_volume": info.volume_min,
                    "max_volume": info.volume_max,
                    "volume_step": info.volume_step
                }
                
            except Exception as e:
                continue
        
        result = {
            "success": True,
            "account": account_data,
            "total_symbols": len(all_symbols),
            "visible_symbols": len(visible_symbols),
            "processed_symbols": len(symbol_data),
            "symbols": symbol_data,
            "fetch_time": datetime.now().isoformat()
        }
        
        return result
        
    except Exception as e:
        return {"error": f"Error fetching symbols: {str(e)}"}
    
    finally:
        mt5.shutdown()

if __name__ == "__main__":
    result = fetch_all_symbols()
    print(json.dumps(result, indent=2))
'''
        
        # Write to Windows-accessible location
        script_path = "/mnt/c/temp/fetch_mt5_symbols.py"
        os.makedirs("/mnt/c/temp", exist_ok=True)
        
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        return script_path.replace("/mnt/c/", "C:\\\\").replace("/", "\\\\")
    
    def find_windows_python(self) -> Optional[str]:
        """Find Windows Python executable"""
        
        # Try common Python locations
        python_paths = [
            "python.exe",
            "py.exe",
            "C:\\\\Python310\\\\python.exe",
            "C:\\\\Python311\\\\python.exe", 
            "C:\\\\Python312\\\\python.exe",
            "C:\\\\Program Files\\\\Python310\\\\python.exe",
            "C:\\\\Program Files\\\\Python311\\\\python.exe",
            "C:\\\\Program Files\\\\Python312\\\\python.exe"
        ]
        
        for python_path in python_paths:
            try:
                # Test if Python works
                result = subprocess.run([python_path, "--version"], 
                                      capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    print(f"✅ Found Windows Python: {python_path}")
                    return python_path
            except Exception:
                continue
        
        return None
    
    def fetch_symbols_via_windows(self) -> Dict:
        """Fetch symbols by executing Windows Python"""
        
        print("🔍 Looking for Windows Python...")
        
        # Find Windows Python
        python_exe = self.find_windows_python()
        if not python_exe:
            return {"error": "Windows Python not found. Please install Python on Windows."}
        
        # Create Windows script
        print("📝 Creating Windows MT5 script...")
        windows_script_path = self.create_windows_mt5_script()
        
        print(f"🚀 Executing Windows script: {windows_script_path}")
        print("⏳ This may take a moment to install MetaTrader5 package...")
        
        try:
            # Execute Windows Python script
            result = subprocess.run([
                python_exe, windows_script_path
            ], capture_output=True, text=True, timeout=120)  # 2 minute timeout
            
            if result.returncode == 0:
                # Parse JSON result
                try:
                    symbol_data = json.loads(result.stdout)
                    return symbol_data
                except json.JSONDecodeError as e:
                    return {"error": f"Failed to parse result: {e}\\nOutput: {result.stdout[:500]}"}
            else:
                return {"error": f"Script failed: {result.stderr}\\nOutput: {result.stdout}"}
                
        except subprocess.TimeoutExpired:
            return {"error": "Script timed out. MT5 may not be running or responding."}
        except Exception as e:
            return {"error": f"Execution error: {e}"}

def display_symbol_results(data: Dict):
    """Display the symbol fetching results"""
    
    if "error" in data:
        print(f"❌ ERROR: {data['error']}")
        return
    
    if not data.get("success"):
        print("❌ Failed to fetch symbols")
        return
    
    print("✅ SUCCESS! Retrieved MT5 symbols")
    print("=" * 60)
    
    # Account info
    account = data.get("account", {})
    print(f"🏦 Server: {account.get('server', 'Unknown')}")
    print(f"👤 Login: {account.get('login', 'Unknown')}")  
    print(f"🏢 Company: {account.get('company', 'Unknown')}")
    print(f"💰 Balance: ${account.get('balance', 0):,.2f}")
    
    # Symbol counts
    print(f"\\n📊 Symbol counts:")
    print(f"   Total available: {data.get('total_symbols', 0)}")
    print(f"   Visible in MarketWatch: {data.get('visible_symbols', 0)}")
    print(f"   Successfully processed: {data.get('processed_symbols', 0)}")
    
    # Symbols by type
    symbols = data.get("symbols", {})
    if symbols:
        by_type = {}
        for symbol_name, symbol_data in symbols.items():
            symbol_type = symbol_data.get("type", "UNKNOWN")
            if symbol_type not in by_type:
                by_type[symbol_type] = []
            by_type[symbol_type].append((symbol_name, symbol_data))
        
        print(f"\\n📈 Symbols by type:")
        for symbol_type, type_symbols in sorted(by_type.items()):
            print(f"\\n{symbol_type} ({len(type_symbols)} symbols):")
            print("-" * 50)
            
            # Show first 5 of each type
            for symbol_name, symbol_data in sorted(type_symbols)[:5]:
                bid = symbol_data.get("bid", 0)
                ask = symbol_data.get("ask", 0)
                digits = symbol_data.get("digits", 5)
                spread_pips = symbol_data.get("spread_pips", 0)
                
                print(f"  {symbol_name:12} | "
                      f"Bid: {bid:10.{digits}f} | "
                      f"Ask: {ask:10.{digits}f} | "
                      f"Spread: {spread_pips:6.1f} pips")
            
            if len(type_symbols) > 5:
                print(f"  ... and {len(type_symbols) - 5} more {symbol_type} symbols")
    
    # Save to file
    output_file = "all_mt5_symbols.json"
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"\\n💾 Full data saved to: {output_file}")
    print(f"⏰ Fetch time: {data.get('fetch_time', 'Unknown')}")

def main():
    """Main function"""
    
    print("🚀 MT5 SYMBOL FETCHER - GET ALL SYMBOLS FROM WINDOWS!")
    print("=" * 70)
    print("This script uses Windows Python to fetch symbols via MetaTrader5 package")
    print()
    
    fetcher = WindowsMT5SymbolFetcher()
    result = fetcher.fetch_symbols_via_windows()
    
    display_symbol_results(result)

if __name__ == "__main__":
    main()