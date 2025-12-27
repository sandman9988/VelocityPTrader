# VelocityPTrader

A physics-based algorithmic trading system using reinforcement learning and real-time market data from MT5.

## Overview

VelocityPTrader implements a sophisticated trading system based on physics principles, treating market dynamics as physical systems with momentum, acceleration, friction, and energy. The system uses dual reinforcement learning agents (BERSERKER and SNIPER) for different market conditions.

## Key Features

- **Physics-Based Market Analysis**: Treats price movements as physical systems with momentum, acceleration, and friction
- **Dual Agent System**: 
  - BERSERKER: Aggressive trading for high volatility markets
  - SNIPER: Precision trading for stable conditions
- **Real MT5 Integration**: Live data from Vantage International Demo Server
- **Multi-Timeframe Analysis**: M1, M5, M15, H1 simultaneous processing
- **AMD Optimized**: Leverages 16-core AMD 5950X CPU with 128GB RAM
- **Comprehensive Logging**: Defense-in-depth logging for analysis and debugging
- **Shadow Trading**: Continuous virtual trading for accelerated learning
- **🤖 AI-Powered Workflow Auto-Recovery**: Automatically detects, analyzes, and fixes CI/CD failures


## System Requirements

- Python 3.8+
- MetaTrader 5 Terminal
- 32GB+ RAM (optimized for 128GB)
- Multi-core CPU (optimized for AMD 5950X)
- WSL2 (for Windows users)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/sandman9988/VelocityPTrader.git
cd VelocityPTrader
```

2. Create virtual environment:
```bash
python3 -m venv trading_env
source trading_env/bin/activate  # On Windows: trading_env\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure MT5 connection in `config/mt5_config.json`

## Architecture

```
VelocityPTrader/
├── src/
│   ├── core/           # Core data pipeline and processing
│   ├── agents/         # RL agents (BERSERKER, SNIPER)
│   ├── physics/        # Physics-based market analysis
│   └── utils/          # Utilities and helpers
├── tests/              # Comprehensive test suite
├── config/             # Configuration files
├── data/               # Market data and logs
└── docs/               # Documentation
```

## Physics Model

The system models markets using physics principles:

- **Momentum (p)**: Rate of price change over time
- **Acceleration (a)**: Change in momentum indicating trend strength
- **Friction (f)**: Market resistance (spread + volatility)
- **Energy (E)**: Market volatility as kinetic energy proxy
- **Liquidity (L)**: Inverse of friction, indicating ease of trading

## Testing

Run the comprehensive test suite:

```bash
# Phase 1: Hardware validation
python tests/test_framework.py

# Phase 2: Core data pipeline
python tests/test_phase2_pipeline.py

# All tests
python -m pytest tests/
```

## Development Phases

1. **Phase 1**: Hardware validation and baseline testing ✅
2. **Phase 2**: Core data pipeline with unit tests (in progress)
3. **Phase 3**: Agent framework with integration tests
4. **Phase 4**: Full system integration and performance tests

## 🤖 Workflow Auto-Recovery System

VelocityPTrader includes an advanced AI-powered workflow auto-recovery system that automatically detects, analyzes, and fixes CI/CD failures.

### Features

- **Automatic Failure Detection**: Monitors all workflow runs and triggers on failures
- **Intelligent Categorization**: Classifies failures into 12 categories (linting, dependencies, tests, security, etc.)
- **AI-Powered Analysis**: Uses pattern matching and heuristics to diagnose issues
- **Automated Fixes**: Automatically fixes safe issues like linting errors and formatting
- **PR Creation**: Creates pull requests with automated fixes for review
- **Issue Creation**: Opens issues for manual intervention when needed
- **CLI Tool**: Local analysis and recovery tool for developers

### Quick Start

The auto-recovery system is already active and monitoring these workflows:
- Comprehensive CI/CD Pipeline
- PostgreSQL Enterprise Testing
- CodeQL Security Analysis

When a workflow fails, the system will:
1. Analyze the failure automatically
2. Categorize the problem
3. Apply automated fixes (if safe), OR
4. Create an issue with detailed analysis

### Manual Analysis

Use the CLI tool for local failure analysis:

```bash
# Analyze a workflow log
python workflow_recovery_cli.py analyze --log workflow.log --workflow "CI/CD" --job "lint" --step "Black"

# Generate recovery actions
python workflow_recovery_cli.py recover --log workflow.log --workflow "CI/CD" --job "lint" --step "Black"

# Execute recovery (dry-run first!)
python workflow_recovery_cli.py execute --log workflow.log --workflow "CI/CD" --job "lint" --step "Black" --dry-run
```

### Documentation

- **Quick Start**: [docs/WORKFLOW_AUTO_RECOVERY_QUICKSTART.md](docs/WORKFLOW_AUTO_RECOVERY_QUICKSTART.md)
- **Full Documentation**: [docs/WORKFLOW_AUTO_RECOVERY.md](docs/WORKFLOW_AUTO_RECOVERY.md)

## Contributing

Please read [CONTRIBUTING.md](docs/CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## License

This project is proprietary software. All rights reserved.

## Acknowledgments

- Built with reinforcement learning principles
- Optimized for AMD hardware architecture
- Real-time data from Vantage International Demo Server