#!/bin/bash

# Push VelocityPTrader to GitHub
echo "==================================="
echo "Pushing VelocityPTrader to GitHub"
echo "==================================="

# Check if remote exists
if git remote | grep -q "origin"; then
    echo "✅ Remote 'origin' already configured"
else
    echo "Adding GitHub remote..."
    git remote add origin https://github.com/sandman9988/VelocityPTrader.git
fi

# Push to main branch
echo "Pushing to GitHub..."
git push -u origin main

echo ""
echo "✅ Repository structure created and ready!"
echo ""
echo "Next steps:"
echo "1. Set up GitHub authentication (PAT or SSH key)"
echo "2. Run: ./push_to_github.sh"
echo "3. Your code will be live at: https://github.com/sandman9988/VelocityPTrader"
echo ""
echo "Repository includes:"
echo "- Phase 1: Hardware validation (COMPLETE)"
echo "- Phase 2: Core data pipeline (IN PROGRESS)"
echo "- Physics-based market analysis"
echo "- Real MT5 data integration"
echo "- AMD optimized system"
echo "- Comprehensive test framework"