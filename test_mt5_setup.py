#!/usr/bin/env python3
"""
Test MT5 Custom Installation
"""

import sys
import os
import json
from pathlib import Path

def test_custom_mt5_setup():
    print("🧪 Testing Custom MT5 Setup")
    print("="*50)
    
    # Load configuration
    config_file = Path.home() / "ai_trading_system" / "config" / "mt5_config.json"
    
    if config_file.exists():
        with open(config_file) as f:
            config = json.load(f)
        
        print("📋 Configuration loaded:")
        print(f"   MT5 Path: {config['mt5_path']}")
        print(f"   WSL Path: {config['wsl_mt5_path']}")
        
        # Test path access
        wsl_path = config['wsl_mt5_path']
        if os.path.exists(wsl_path):
            print(f"✅ MT5 directory accessible")
        else:
            print(f"❌ MT5 directory not accessible")
        
        # Test MT5 module
        try:
            import MetaTrader5 as mt5
            print(f"✅ MetaTrader5 module imported")
            
            # Test initialization
            if mt5.initialize(path=config['mt5_path']):
                print(f"✅ MT5 initialized successfully")
                
                # Get terminal info
                terminal_info = mt5.terminal_info()
                if terminal_info:
                    print(f"   Company: {terminal_info.company}")
                    print(f"   Terminal: {terminal_info.name}")
                    print(f"   Build: {terminal_info.build}")
                
                mt5.shutdown()
            else:
                print(f"⚠️  MT5 initialization failed: {mt5.last_error()}")
                
        except ImportError:
            print(f"❌ MetaTrader5 module not available")
    else:
        print(f"❌ Configuration file not found: {config_file}")

if __name__ == "__main__":
    test_custom_mt5_setup()
