#!/bin/bash
# Start the AI Trading System with proper environment

echo "🚀 Starting AI Trading System with ALL MarketWatch symbols..."

# Activate virtual environment if it exists
if [ -d "trading_env" ]; then
    echo "📦 Activating virtual environment..."
    source trading_env/bin/activate
fi

# Start the integrated dual agent system
echo "🤖 Launching integrated dual agent system..."
python integrated_dual_agent_system.py

# Keep script running
read -p "Press Enter to exit..."