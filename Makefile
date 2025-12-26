# VelocityTrader Development Makefile
# Comprehensive automation for development, testing, and deployment

.DEFAULT_GOAL := help
PYTHON := python3
PIP := pip3
PYTEST := pytest
VENV := venv
SOURCE_FILES := src tests velocity_trader.py

# Colors for output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[0;33m
BLUE := \033[0;34m
PURPLE := \033[0;35m
CYAN := \033[0;36m
WHITE := \033[0;37m
RESET := \033[0m

# Help target
.PHONY: help
help: ## 📖 Show this help message
	@echo "$(CYAN)🚀 VelocityTrader Development Commands$(RESET)"
	@echo "$(YELLOW)======================================$(RESET)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "$(WHITE)%-20s$(RESET) %s\n", $$1, $$2}'

# Environment setup
.PHONY: setup
setup: ## 🛠️  Set up development environment
	@echo "$(BLUE)Setting up development environment...$(RESET)"
	$(PYTHON) -m venv $(VENV)
	./$(VENV)/bin/pip install --upgrade pip setuptools wheel
	./$(VENV)/bin/pip install -r requirements.txt
	./$(VENV)/bin/pre-commit install
	@echo "$(GREEN)✅ Environment setup complete!$(RESET)"

.PHONY: clean-setup
clean-setup: clean setup ## 🧹 Clean and rebuild environment

# Dependency management
.PHONY: install
install: ## 📦 Install dependencies
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

.PHONY: install-dev
install-dev: ## 📦 Install development dependencies
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	$(PIP) install -e .

.PHONY: update-deps
update-deps: ## 📈 Update dependencies
	$(PIP) install --upgrade pip
	$(PIP) install --upgrade -r requirements.txt
	$(PIP) freeze > requirements-freeze.txt

# Code quality and linting
.PHONY: format
format: ## 🎨 Format code with black and isort
	@echo "$(BLUE)Formatting code...$(RESET)"
	black $(SOURCE_FILES)
	isort $(SOURCE_FILES)
	@echo "$(GREEN)✅ Code formatted!$(RESET)"

.PHONY: format-check
format-check: ## 🔍 Check code formatting
	@echo "$(BLUE)Checking code formatting...$(RESET)"
	black --check --diff $(SOURCE_FILES)
	isort --check-only --diff $(SOURCE_FILES)

.PHONY: lint
lint: ## 🔍 Run all linting tools
	@echo "$(BLUE)Running linting tools...$(RESET)"
	ruff check $(SOURCE_FILES)
	flake8 $(SOURCE_FILES)
	pylint src/ || true
	@echo "$(GREEN)✅ Linting complete!$(RESET)"

.PHONY: lint-fix
lint-fix: ## 🔧 Fix linting issues automatically
	@echo "$(BLUE)Fixing linting issues...$(RESET)"
	ruff check --fix $(SOURCE_FILES)
	autoflake --remove-all-unused-imports --remove-unused-variables --in-place --recursive $(SOURCE_FILES)

.PHONY: type-check
type-check: ## 🏷️  Run type checking with mypy
	@echo "$(BLUE)Running type checking...$(RESET)"
	mypy src/ --show-error-codes --show-error-context --pretty
	@echo "$(GREEN)✅ Type checking complete!$(RESET)"

.PHONY: quality
quality: format-check lint type-check ## 🌟 Run all code quality checks

# Security scanning
.PHONY: security
security: ## 🔒 Run security scans
	@echo "$(BLUE)Running security scans...$(RESET)"
	bandit -r src/ -f json -o bandit-report.json || true
	bandit -r src/
	safety check
	semgrep --config=auto src/ || true
	@echo "$(GREEN)✅ Security scan complete!$(RESET)"

.PHONY: security-report
security-report: ## 📊 Generate detailed security report
	@echo "$(BLUE)Generating security report...$(RESET)"
	bandit -r src/ -f html -o security-report.html
	safety check --json --output safety-report.json || true
	@echo "$(GREEN)✅ Security report generated!$(RESET)"

# Testing
.PHONY: test
test: ## 🧪 Run all tests
	@echo "$(BLUE)Running test suite...$(RESET)"
	$(PYTEST) -v --tb=short

.PHONY: test-fast
test-fast: ## ⚡ Run fast tests only
	@echo "$(BLUE)Running fast tests...$(RESET)"
	$(PYTEST) -v --tb=short -m "not slow"

.PHONY: test-unit
test-unit: ## 🔬 Run unit tests
	@echo "$(BLUE)Running unit tests...$(RESET)"
	$(PYTEST) -v --tb=short -m "unit"

.PHONY: test-integration
test-integration: ## 🔗 Run integration tests
	@echo "$(BLUE)Running integration tests...$(RESET)"
	$(PYTEST) -v --tb=short -m "integration"

.PHONY: test-phase1
test-phase1: ## 🏗️  Run Phase 1 tests (Hardware)
	@echo "$(BLUE)Running Phase 1 tests...$(RESET)"
	$(PYTHON) tests/test_framework.py

.PHONY: test-phase2
test-phase2: ## 📊 Run Phase 2 tests (Data Pipeline)
	@echo "$(BLUE)Running Phase 2 tests...$(RESET)"
	$(PYTHON) tests/test_phase2_pipeline.py

.PHONY: test-phase3
test-phase3: ## 🤖 Run Phase 3 tests (Agents)
	@echo "$(BLUE)Running Phase 3 tests...$(RESET)"
	$(PYTHON) tests/test_phase3_agents.py

.PHONY: test-phase4
test-phase4: ## 🚀 Run Phase 4 tests (Integration)
	@echo "$(BLUE)Running Phase 4 tests...$(RESET)"
	$(PYTHON) tests/test_phase4_integration.py

.PHONY: test-all-phases
test-all-phases: test-phase1 test-phase2 test-phase3 test-phase4 ## 🎯 Run all phase tests

.PHONY: test-cov
test-cov: ## 📈 Run tests with coverage
	@echo "$(BLUE)Running tests with coverage...$(RESET)"
	$(PYTEST) --cov=src --cov-report=html --cov-report=xml --cov-report=term-missing

.PHONY: test-performance
test-performance: ## ⚡ Run performance benchmarks
	@echo "$(BLUE)Running performance benchmarks...$(RESET)"
	$(PYTHON) -c "
	import time
	import psutil
	from src.core.data_pipeline import DataPipeline
	from src.agents.dual_agent_system import DualAgentCoordinator
	
	print('🚀 VelocityTrader Performance Benchmark')
	print('=' * 50)
	
	# System info
	print(f'CPU Count: {psutil.cpu_count()} cores')
	print(f'Memory: {psutil.virtual_memory().total / 1024**3:.1f} GB')
	
	# Benchmarks
	start = time.time()
	pipeline = DataPipeline()
	print(f'Data Pipeline Init: {time.time() - start:.3f}s')
	
	start = time.time()
	coordinator = DualAgentCoordinator()
	print(f'Agent System Init: {time.time() - start:.3f}s')
	
	process = psutil.Process()
	print(f'Memory Usage: {process.memory_info().rss / 1024**2:.1f} MB')
	print('✅ Benchmarks completed')
	"

# VelocityTrader specific commands
.PHONY: run
run: ## 🚀 Run VelocityTrader system
	@echo "$(BLUE)Starting VelocityTrader...$(RESET)"
	$(PYTHON) velocity_trader.py

.PHONY: run-test-mode
run-test-mode: ## 🧪 Run VelocityTrader in test mode
	@echo "$(BLUE)Starting VelocityTrader in test mode...$(RESET)"
	$(PYTHON) velocity_trader.py --test-only

.PHONY: run-dashboard
run-dashboard: ## 📊 Run performance dashboard only
	@echo "$(BLUE)Starting dashboard...$(RESET)"
	$(PYTHON) velocity_trader.py --dashboard-only

.PHONY: validate-config
validate-config: ## ⚙️  Validate system configuration
	@echo "$(BLUE)Validating configuration...$(RESET)"
	$(PYTHON) -c "
	import json
	from pathlib import Path
	from src.core.integrated_system import SystemConfig
	
	config_file = Path('config/system_config.json')
	if config_file.exists():
	    with open(config_file) as f:
	        config_dict = json.load(f)
	    config = SystemConfig(**config_dict)
	    print('✅ Configuration valid')
	    print(f'   MT5 Server: {config.mt5_server}')
	    print(f'   Symbols: {len(config.symbols)}')
	    print(f'   Max Positions: {config.max_positions}')
	else:
	    print('❌ Configuration file not found')
	"

# Pre-commit and git hooks
.PHONY: pre-commit
pre-commit: ## 🪝 Run pre-commit hooks
	@echo "$(BLUE)Running pre-commit hooks...$(RESET)"
	pre-commit run --all-files

.PHONY: pre-commit-update
pre-commit-update: ## 📈 Update pre-commit hooks
	pre-commit autoupdate

# Documentation
.PHONY: docs
docs: ## 📚 Generate documentation
	@echo "$(BLUE)Generating documentation...$(RESET)"
	# Add documentation generation here
	@echo "$(YELLOW)⚠️  Documentation generation not yet implemented$(RESET)"

# Deployment and release
.PHONY: build
build: ## 📦 Build distribution packages
	@echo "$(BLUE)Building packages...$(RESET)"
	$(PYTHON) -m build
	@echo "$(GREEN)✅ Build complete!$(RESET)"

.PHONY: release-check
release-check: quality test security ## 🚀 Run all checks for release
	@echo "$(GREEN)✅ Release checks passed!$(RESET)"

.PHONY: deploy-check
deploy-check: release-check validate-config ## 🎯 Comprehensive deployment check
	@echo "$(BLUE)Running deployment readiness check...$(RESET)"
	$(PYTHON) velocity_trader.py --test-only
	@echo "$(GREEN)✅ Deployment ready!$(RESET)"

# Cleanup
.PHONY: clean
clean: ## 🧹 Clean temporary files and caches
	@echo "$(BLUE)Cleaning temporary files...$(RESET)"
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	find . -type f -name ".coverage" -delete 2>/dev/null || true
	rm -rf .coverage.* htmlcov/ .pytest_cache/ .mypy_cache/ .ruff_cache/
	rm -rf build/ dist/ *.egg-info/
	rm -rf bandit-report.json safety-report.json security-report.html
	@echo "$(GREEN)✅ Cleanup complete!$(RESET)"

.PHONY: clean-all
clean-all: clean ## 🗑️  Clean everything including venv
	rm -rf $(VENV)/
	rm -rf models/ rl_models/
	rm -rf logs/ results/ reports/

# CI/CD simulation
.PHONY: ci
ci: quality test security ## 🔄 Simulate CI pipeline locally
	@echo "$(GREEN)✅ CI pipeline simulation complete!$(RESET)"

.PHONY: ci-full
ci-full: clean install quality test-cov security performance-test deploy-check ## 🔄 Full CI pipeline simulation
	@echo "$(GREEN)✅ Full CI pipeline simulation complete!$(RESET)"

# Development workflow
.PHONY: dev
dev: format lint test ## 💻 Standard development workflow
	@echo "$(GREEN)✅ Development workflow complete!$(RESET)"

.PHONY: commit-ready
commit-ready: format lint test-fast security ## 📝 Prepare for commit
	@echo "$(GREEN)✅ Ready to commit!$(RESET)"

# Information
.PHONY: info
info: ## ℹ️  Show system information
	@echo "$(CYAN)📊 VelocityTrader System Information$(RESET)"
	@echo "$(YELLOW)=================================$(RESET)"
	@echo "Python: $(shell $(PYTHON) --version)"
	@echo "Pip: $(shell $(PIP) --version)"
	@echo "Git: $(shell git --version 2>/dev/null || echo 'Not installed')"
	@echo "Pre-commit: $(shell pre-commit --version 2>/dev/null || echo 'Not installed')"
	@echo "CPU Cores: $(shell $(PYTHON) -c 'import psutil; print(psutil.cpu_count())')"
	@echo "Memory: $(shell $(PYTHON) -c 'import psutil; print(f\"{psutil.virtual_memory().total / 1024**3:.1f} GB\")')"
	@echo "Platform: $(shell $(PYTHON) -c 'import platform; print(platform.system(), platform.release())')"

# Watch mode for development
.PHONY: watch-test
watch-test: ## 👀 Watch for changes and run tests
	@echo "$(BLUE)Watching for changes...$(RESET)"
	@echo "$(YELLOW)Press Ctrl+C to stop$(RESET)"
	while true; do \
		find src tests -name "*.py" | entr -d make test-fast; \
	done

# Aliases for convenience
.PHONY: t
t: test ## 🧪 Alias for test

.PHONY: l
l: lint ## 🔍 Alias for lint

.PHONY: f
f: format ## 🎨 Alias for format

.PHONY: s
s: security ## 🔒 Alias for security

.PHONY: c
c: clean ## 🧹 Alias for clean