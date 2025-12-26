#!/usr/bin/env python3
"""
Test SECURE MT5 connection status
Check if SecureDataExporter EA is running and providing real data
"""

import os
import time
from datetime import datetime
from secure_mt5_bridge import SecureMT5Bridge

def test_secure_connection():
    """Test the secure connection and show current status"""
    
    print("🔒 TESTING SECURE MT5 CONNECTION")
    print("=" * 60)
    
    # Initialize secure bridge
    bridge = SecureMT5Bridge()
    
    # Check connection status
    print("\n📊 Connection Status:")
    status = bridge.get_connection_status()
    
    print(f"Connected: {'✅' if status['connected'] else '❌'} {status['connected']}")
    print(f"Status: {status['status']}")
    print(f"Last Update: {status['last_update']}")
    print(f"Data Age: {status['age']}s" if status['age'] is not None else "Data Age: N/A")
    
    # Try to read secure data
    print("\n🔍 Reading Secure Data:")
    
    if status['connected']:
        secure_data = bridge.read_secure_data()
        
        if secure_data:
            print(f"✅ SUCCESS! Retrieved {len(secure_data)} validated symbols:")
            print("-" * 60)
            
            for symbol, data in list(secure_data.items())[:10]:  # Show first 10
                spread_pips = data['spread'] / data['point'] if data['point'] > 0 else 0
                print(f"{symbol:10} | Bid: {data['bid']:10.{data['digits']}f} | "
                      f"Ask: {data['ask']:10.{data['digits']}f} | "
                      f"Spread: {spread_pips:6.1f} pips | "
                      f"Server: {data['server']}")
            
            if len(secure_data) > 10:
                print(f"... and {len(secure_data) - 10} more symbols")
                
            print("-" * 60)
            print(f"✅ ALL DATA VALIDATED AND SECURE")
            
        else:
            print("❌ No secure data available")
            
    else:
        print("❌ SecureDataExporter EA not running")
    
    # Show setup instructions
    print("\n🚀 TO GET REAL DATA:")
    print("1. Open your MT5 terminal")
    print("2. Open MetaEditor (F4)")
    print("3. Find SecureDataExporter.mq5 in Experts folder")
    print("4. Compile it (F7)")
    print("5. Drag it to any chart")
    print("6. Check 'Allow DLL imports' in settings")
    print("7. Re-run this test")
    
    # Check file paths
    print("\n📁 File Status:")
    bridge_path = "/mnt/c/DevCenter/MT5-Unified/MT5-Core/Terminal/MQL5/Files/SecureBridge"
    data_file = f"{bridge_path}/secure_data.txt"
    checksum_file = f"{bridge_path}/checksum.txt"
    heartbeat_file = f"{bridge_path}/heartbeat.txt"
    
    print(f"Bridge directory: {'✅' if os.path.exists(bridge_path) else '❌'} {bridge_path}")
    print(f"Data file: {'✅' if os.path.exists(data_file) else '❌'} {data_file}")
    print(f"Checksum file: {'✅' if os.path.exists(checksum_file) else '❌'} {checksum_file}")
    print(f"Heartbeat file: {'✅' if os.path.exists(heartbeat_file) else '❌'} {heartbeat_file}")
    
    return status['connected'] and len(secure_data) > 0 if status['connected'] else False

if __name__ == "__main__":
    success = test_secure_connection()
    
    if success:
        print("\n🎉 SECURE CONNECTION WORKING!")
        print("Your system is ready to trade with REAL Vantage data")
    else:
        print("\n⚠️  SECURE CONNECTION NOT READY")
        print("Please set up SecureDataExporter EA in MT5")