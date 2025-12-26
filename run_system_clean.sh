#!/bin/bash

echo "🏭 Production AI Trading System - Clean Startup"
echo "==============================================="

# Activate virtual environment
source trading_env/bin/activate

# Create required directories
mkdir -p logs certs pids

echo "🧹 Cleaning up existing processes and ports..."

# Clean up previous instances
python3 port_manager.py --all --force
python3 port_manager.py --python

echo "✅ Cleanup complete"
echo ""

echo "🚀 Starting fresh system..."

# Start the system with proper PID tracking
python3 start_production_system_clean.py &
SYSTEM_PID=$!

# Save PID for management
echo $SYSTEM_PID > pids/production_system.pid
echo "📝 System PID: $SYSTEM_PID"

# Wait a moment for startup
sleep 3

echo ""
echo "📊 Checking system status..."
python3 port_manager.py --status

echo ""
echo "🎉 SYSTEM READY!"
echo "================================"
echo "🌐 Dashboard: https://localhost:8443"
echo "📊 Metrics:   http://localhost:9090"
echo "📁 Logs:      tail -f logs/production_system.log"
echo ""
echo "🛑 To stop system:"
echo "   python3 port_manager.py --services"
echo "   python3 port_manager.py --all"
echo ""
echo "Press Ctrl+C to stop this session (system will continue in background)"

# Keep script running to show it's active
trap 'echo "Script terminated, system continues in background"; exit 0' INT

wait $SYSTEM_PID