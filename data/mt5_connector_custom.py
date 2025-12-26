#!/usr/bin/env python3
"""
Updated MT5 Data Connector with Custom Path
Configured for: C:\DevCenter\MT5-Unified\MT5-Core\Terminal
"""

import sys
import os
from pathlib import Path

# Custom MT5 path configuration
CUSTOM_MT5_PATH = "/mnt/c/DevCenter/MT5-Unified/MT5-Core/Terminal"
WINDOWS_MT5_PATH = "C:\DevCenter\MT5-Unified\MT5-Core\Terminal"

# Add custom MT5 Python path if it exists
possible_python_paths = [
    os.path.join(CUSTOM_MT5_PATH, "Python"),
    os.path.join(CUSTOM_MT5_PATH, "MQL5", "Python"),
    "/mnt/c/Program Files/MetaTrader 5/Python"  # Fallback
]

for python_path in possible_python_paths:
    if os.path.exists(python_path):
        sys.path.append(python_path)
        print(f"📍 Added MT5 Python path: {python_path}")
        break

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
    print("✅ MetaTrader5 module loaded successfully")
except ImportError:
    # Use our custom MT5 bridge
    try:
        from mt5_bridge import *
        import mt5_bridge as mt5
        MT5_AVAILABLE = True
        print("✅ Custom MT5 Bridge loaded successfully")
    except ImportError as e:
        print(f"❌ Neither MetaTrader5 nor MT5 Bridge available: {e}")
        MT5_AVAILABLE = False

# Now we have either the official MT5 module or our custom bridge available as 'mt5'
