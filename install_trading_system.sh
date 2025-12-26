#!/bin/bash
# MT5 Trading System Installation Script
# Custom path: C:\DevCenter\MT5-Unified\MT5-Core\Terminal

echo "🚀 Installing MT5 Trading System Dependencies"
echo "=============================================="

# Create virtual environment
cd /home/renier/ai_trading_system
python3 -m venv trading_env
source trading_env/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install core packages
echo "📦 Installing core packages..."
pip install pandas numpy matplotlib plotly scikit-learn

# Install MT5 package
echo "📦 Installing MetaTrader5 package..."
pip install MetaTrader5

# Install optional ML packages
echo "📦 Installing ML packages..."
pip install tensorflow asyncio aiofiles

echo "✅ Installation complete!"
echo ""
echo "To activate environment:"
echo "cd /home/renier/ai_trading_system"
echo "source trading_env/bin/activate"
echo ""
echo "To test MT5 connection:"
echo "python3 data/mt5_connector_custom.py"
