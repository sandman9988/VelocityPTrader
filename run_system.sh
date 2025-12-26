#!/bin/bash

echo "🏭 Starting Production AI Trading System..."
echo "========================================"

# Activate virtual environment
source trading_env/bin/activate

# Create required directories
mkdir -p logs certs

# Kill any existing instances
pkill -f "start_production_system.py" 2>/dev/null || true

# Start the system
echo "Starting system components..."
python3 start_production_system.py

echo ""
echo "🌐 Access the dashboard at: https://localhost:8444"
echo "📊 View metrics at: http://localhost:9090"
echo "📁 Check logs: tail -f logs/production_system.log"