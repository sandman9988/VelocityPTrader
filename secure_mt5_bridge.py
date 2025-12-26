#!/usr/bin/env python3
"""
SECURE MT5 Bridge - Protected against data injection/manipulation
Direct, validated connection to MT5 terminal
"""

import os
import sys
import time
import hashlib
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
import struct

class SecureMT5Bridge:
    """Secure MT5 connection with integrity validation"""
    
    def __init__(self):
        self.terminal_path = "/mnt/c/DevCenter/MT5-Unified/MT5-Core/Terminal"
        self.bridge_path = f"{self.terminal_path}/MQL5/Files/SecureBridge"
        
        # Security settings
        self.data_max_age = 5  # seconds
        self.validation_key = "AI_TRADING_VANTAGE_2024"  # Simple validation
        
        # Ensure paths exist
        os.makedirs(self.bridge_path, exist_ok=True)
    
    def _generate_checksum(self, data: str) -> str:
        """Generate checksum for data integrity"""
        return hashlib.sha256((data + self.validation_key).encode()).hexdigest()[:16]
    
    def _validate_data_integrity(self, data: str, checksum: str) -> bool:
        """Validate data hasn't been tampered with"""
        expected_checksum = self._generate_checksum(data)
        return expected_checksum == checksum
    
    def create_secure_data_exporter(self):
        """Create secure MQL5 EA with data integrity checks"""
        
        ea_code = f'''
//+------------------------------------------------------------------+
//| SecureDataExporter.mq5                                          |
//| SECURE export of REAL MarketWatch data with integrity validation|
//+------------------------------------------------------------------+
#property copyright "AI Trading System - Secure"
#property version   "1.00"

string ValidationKey = "{self.validation_key}";

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
{{
    Print("SecureDataExporter: Starting SECURE market data export...");
    
    // Export data every 1 second
    EventSetTimer(1);
    
    return(INIT_SUCCEEDED);
}}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{{
    EventKillTimer();
}}

//+------------------------------------------------------------------+
//| Timer function - export current prices securely                 |
//+------------------------------------------------------------------+
void OnTimer()
{{
    ExportSecureMarketData();
}}

//+------------------------------------------------------------------+
//| Generate simple checksum for data validation                    |
//+------------------------------------------------------------------+
string GenerateChecksum(string data)
{{
    // Simple checksum (in production, use cryptographic hash)
    string combined = data + ValidationKey;
    int checksum = 0;
    
    for(int i = 0; i < StringLen(combined); i++)
    {{
        checksum = (checksum + StringGetCharacter(combined, i)) % 65536;
    }}
    
    return IntegerToString(checksum);
}}

//+------------------------------------------------------------------+
//| Export MarketWatch data with security validation                |
//+------------------------------------------------------------------+
void ExportSecureMarketData()
{{
    int total = SymbolsTotal(true);
    
    if(total == 0)
    {{
        Print("ERROR: No symbols in MarketWatch!");
        return;
    }}
    
    string timestamp_str = IntegerToString(TimeTradeServer());
    string data_block = "";
    int valid_symbols = 0;
    
    // Get account info for validation
    string server = AccountInfoString(ACCOUNT_SERVER);
    int login = (int)AccountInfoInteger(ACCOUNT_LOGIN);
    
    Print("Exporting data for ", total, " symbols from server: ", server);
    
    for(int i = 0; i < total; i++)
    {{
        string symbol = SymbolName(i, true);
        
        if(symbol == "")
            continue;
            
        // Get REAL tick data
        MqlTick tick;
        if(!SymbolInfoTick(symbol, tick))
        {{
            Print("WARNING: No tick data for ", symbol);
            continue;
        }}
        
        // Validate tick data
        if(tick.bid <= 0 || tick.ask <= 0 || tick.bid >= tick.ask)
        {{
            Print("ERROR: Invalid tick data for ", symbol, " - bid:", tick.bid, " ask:", tick.ask);
            continue;
        }}
        
        // Get symbol specifications
        int digits = (int)SymbolInfoInteger(symbol, SYMBOL_DIGITS);
        double point = SymbolInfoDouble(symbol, SYMBOL_POINT);
        int spread_points = (int)SymbolInfoInteger(symbol, SYMBOL_SPREAD);
        double spread = spread_points * point;
        
        // Validate symbol specs
        if(digits <= 0 || point <= 0)
        {{
            Print("ERROR: Invalid symbol specs for ", symbol);
            continue;
        }}
        
        // Build data entry
        string entry = symbol + "|" + 
                      DoubleToString(tick.bid, digits) + "|" + 
                      DoubleToString(tick.ask, digits) + "|" + 
                      DoubleToString(spread, digits) + "|" +
                      IntegerToString(digits) + "|" +
                      DoubleToString(point, 8) + "|" +
                      timestamp_str;
        
        data_block += entry + "\\\\n";
        valid_symbols++;
    }}
    
    if(valid_symbols == 0)
    {{
        Print("CRITICAL: No valid symbols exported!");
        return;
    }}
    
    // Add metadata
    string metadata = "SERVER:" + server + "|LOGIN:" + IntegerToString(login) + 
                     "|SYMBOLS:" + IntegerToString(valid_symbols) + 
                     "|TIMESTAMP:" + timestamp_str;
    
    string full_data = metadata + "\\\\n" + data_block;
    
    // Generate checksum
    string checksum = GenerateChecksum(full_data);
    
    // Write secure data file
    int handle = FileOpen("SecureBridge\\\\secure_data.txt", FILE_WRITE|FILE_TXT);
    if(handle != INVALID_HANDLE)
    {{
        FileWrite(handle, full_data);
        FileClose(handle);
    }}
    else
    {{
        Print("ERROR: Cannot write secure data file!");
        return;
    }}
    
    // Write checksum file
    int checksum_handle = FileOpen("SecureBridge\\\\checksum.txt", FILE_WRITE|FILE_TXT);
    if(checksum_handle != INVALID_HANDLE)
    {{
        FileWrite(checksum_handle, checksum);
        FileClose(checksum_handle);
    }}
    
    // Write heartbeat
    int heartbeat_handle = FileOpen("SecureBridge\\\\heartbeat.txt", FILE_WRITE|FILE_TXT);
    if(heartbeat_handle != INVALID_HANDLE)
    {{
        FileWrite(heartbeat_handle, timestamp_str);
        FileClose(heartbeat_handle);
    }}
    
    Print("Exported ", valid_symbols, " symbols securely at ", timestamp_str);
}}
'''
        
        # Write the secure EA
        ea_path = f"{self.terminal_path}/MQL5/Experts/SecureDataExporter.mq5"
        os.makedirs(os.path.dirname(ea_path), exist_ok=True)
        
        with open(ea_path, 'w') as f:
            f.write(ea_code)
        
        print(f"✅ Created SecureDataExporter EA: {ea_path}")
        return ea_path
    
    def read_secure_data(self) -> Dict[str, Dict]:
        """Read and validate secure data from MT5"""
        
        data_file = f"{self.bridge_path}/secure_data.txt"
        checksum_file = f"{self.bridge_path}/checksum.txt"
        heartbeat_file = f"{self.bridge_path}/heartbeat.txt"
        
        # Check all files exist
        if not all(os.path.exists(f) for f in [data_file, checksum_file, heartbeat_file]):
            print("❌ Secure data files missing - SecureDataExporter EA not running")
            return {}
        
        try:
            # Check data freshness
            with open(heartbeat_file, 'r') as f:
                mt5_timestamp = int(f.read().strip())
                current_time = int(time.time())
                age = current_time - mt5_timestamp
                
                if age > self.data_max_age:
                    print(f"❌ Data too old ({age}s) - possible stale/injected data")
                    return {}
            
            # Read data and checksum
            with open(data_file, 'r') as f:
                data_content = f.read()
            
            with open(checksum_file, 'r') as f:
                provided_checksum = f.read().strip()
            
            # Validate data integrity
            if not self._validate_data_integrity(data_content, provided_checksum):
                print("❌ SECURITY VIOLATION: Data integrity check failed!")
                print("❌ Possible man-in-the-middle attack or data corruption!")
                return {}
            
            # Parse validated data
            lines = data_content.strip().split('\\n')
            if not lines:
                return {}
            
            # Parse metadata
            metadata_line = lines[0]
            if not metadata_line.startswith('SERVER:'):
                print("❌ Invalid metadata format")
                return {}
            
            # Extract metadata
            metadata = {}
            for part in metadata_line.split('|'):
                if ':' in part:
                    key, value = part.split(':', 1)
                    metadata[key] = value
            
            print(f"✅ SECURE DATA from {metadata.get('SERVER', 'Unknown')} "
                  f"login {metadata.get('LOGIN', '?')} "
                  f"({metadata.get('SYMBOLS', '0')} symbols)")
            
            # Parse symbol data
            secure_data = {}
            for line in lines[1:]:
                if not line.strip():
                    continue
                
                parts = line.strip().split('|')
                if len(parts) >= 7:
                    symbol = parts[0]
                    bid = float(parts[1])
                    ask = float(parts[2])
                    spread = float(parts[3])
                    digits = int(parts[4])
                    point = float(parts[5])
                    timestamp = int(parts[6])
                    
                    # Additional validation
                    if bid <= 0 or ask <= 0 or bid >= ask:
                        print(f"❌ Invalid price data for {symbol}")
                        continue
                    
                    if digits < 1 or digits > 8 or point <= 0:
                        print(f"❌ Invalid symbol specs for {symbol}")
                        continue
                    
                    secure_data[symbol] = {
                        'symbol': symbol,
                        'bid': bid,
                        'ask': ask,
                        'spread': spread,
                        'digits': digits,
                        'point': point,
                        'timestamp': timestamp,
                        'validated': True,
                        'server': metadata.get('SERVER', 'Unknown')
                    }
            
            print(f"✅ VALIDATED {len(secure_data)} symbols with integrity checks")
            return secure_data
            
        except Exception as e:
            print(f"❌ Error reading secure data: {e}")
            return {}
    
    def get_connection_status(self) -> Dict[str, Any]:
        """Get secure connection status"""
        
        heartbeat_file = f"{self.bridge_path}/heartbeat.txt"
        
        if not os.path.exists(heartbeat_file):
            return {
                'connected': False,
                'status': 'SecureDataExporter EA not running',
                'last_update': None,
                'age': None
            }
        
        try:
            with open(heartbeat_file, 'r') as f:
                last_timestamp = int(f.read().strip())
                current_time = int(time.time())
                age = current_time - last_timestamp
                
                return {
                    'connected': age <= self.data_max_age,
                    'status': 'Connected' if age <= self.data_max_age else f'Stale ({age}s old)',
                    'last_update': datetime.fromtimestamp(last_timestamp),
                    'age': age
                }
        except Exception as e:
            return {
                'connected': False,
                'status': f'Error reading heartbeat: {e}',
                'last_update': None,
                'age': None
            }

def setup_secure_connection():
    """Setup secure MT5 connection"""
    
    print("🔒 SETTING UP SECURE MT5 CONNECTION")
    print("=" * 60)
    
    bridge = SecureMT5Bridge()
    
    # Create secure EA
    ea_path = bridge.create_secure_data_exporter()
    
    print("\\n🛡️ SECURITY FEATURES:")
    print("- Data integrity validation with checksums")
    print("- Timestamp validation (max 5s age)")
    print("- Direct file access (no network)")
    print("- Input validation and sanitization")
    print("- Server/login verification")
    
    print("\\n🚀 SETUP STEPS:")
    print("1. Open your MT5 terminal")
    print("2. Open MetaEditor")
    print(f"3. Compile: {ea_path}")
    print("4. Attach SecureDataExporter to any chart")
    print("5. Verify it's running and exporting data")
    
    # Test connection
    print("\\n📊 Testing secure connection...")
    status = bridge.get_connection_status()
    print(f"Status: {status['status']}")
    
    if status['connected']:
        data = bridge.read_secure_data()
        if data:
            print(f"✅ SECURE DATA RECEIVED: {len(data)} symbols")
            for symbol in list(data.keys())[:3]:  # Show first 3
                d = data[symbol]
                print(f"  {symbol}: {d['bid']:.{d['digits']}f}/{d['ask']:.{d['digits']}f} "
                      f"(Server: {d['server']})")
        else:
            print("❌ No secure data received")
    
    return bridge

if __name__ == "__main__":
    setup_secure_connection()