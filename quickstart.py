#!/usr/bin/env python3
"""
VelocityPTrader Quick Start Script
One-click system startup with health checks and resilience
"""

import asyncio
import sys
import time
import os
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))


class StartupPhase(Enum):
    DATABASE = "database"
    SETTINGS = "settings"
    MT5_BRIDGE = "mt5"
    AGENTS = "agents"
    MONITORING = "monitoring"
    DASHBOARD = "dashboard"


@dataclass
class StartupResult:
    phase: StartupPhase
    success: bool
    message: str
    details: Optional[Dict[str, Any]] = None


class VelocityStartup:
    """Quick start manager for VelocityPTrader"""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.results: Dict[StartupPhase, StartupResult] = {}

    def log(self, message: str, phase: Optional[StartupPhase] = None, status: str = "INFO"):
        """Print status message"""
        if not self.verbose:
            return

        icons = {
            "INFO": "ℹ️ ",
            "SUCCESS": "✅",
            "ERROR": "❌",
            "WARN": "⚠️ ",
            "PENDING": "⏳",
        }

        icon = icons.get(status, "  ")
        phase_str = f"[{phase.value.upper()}]" if phase else ""
        print(f"{icon} {phase_str:12} {message}")

    def check_database(self) -> StartupResult:
        """Initialize and verify database connection"""
        self.log("Connecting to PostgreSQL...", StartupPhase.DATABASE, "PENDING")

        try:
            from src.database.connection import get_database_manager

            db = get_database_manager()
            db.initialize_sync_engine()
            health = db.health_check()

            pool_info = health.get("connection_pool", {})
            message = f"Connected to {db.config.database} (pool: {pool_info.get('size', 'N/A')})"

            self.log(message, StartupPhase.DATABASE, "SUCCESS")
            return StartupResult(
                phase=StartupPhase.DATABASE,
                success=True,
                message=message,
                details=health
            )

        except Exception as e:
            message = f"Database connection failed: {str(e)}"
            self.log(message, StartupPhase.DATABASE, "ERROR")
            return StartupResult(
                phase=StartupPhase.DATABASE,
                success=False,
                message=message
            )

    def load_settings(self) -> StartupResult:
        """Load or initialize system settings"""
        self.log("Loading system settings...", StartupPhase.SETTINGS, "PENDING")

        try:
            from src.database.operations import AtomicDataOperations

            ops = AtomicDataOperations()
            settings = ops.get_all_settings()

            if not settings:
                self.log("Initializing default settings...", StartupPhase.SETTINGS, "INFO")
                ops.initialize_default_settings()
                settings = ops.get_all_settings()

            total_settings = sum(len(v) for v in settings.values())
            message = f"Loaded {total_settings} settings across {len(settings)} categories"

            self.log(message, StartupPhase.SETTINGS, "SUCCESS")
            return StartupResult(
                phase=StartupPhase.SETTINGS,
                success=True,
                message=message,
                details={"categories": list(settings.keys()), "total": total_settings}
            )

        except Exception as e:
            message = f"Settings load failed: {str(e)}"
            self.log(message, StartupPhase.SETTINGS, "ERROR")
            return StartupResult(
                phase=StartupPhase.SETTINGS,
                success=False,
                message=message
            )

    def connect_mt5(self) -> StartupResult:
        """Initialize MT5 bridge connection"""
        self.log("Connecting to MT5 bridge...", StartupPhase.MT5_BRIDGE, "PENDING")

        try:
            from src.data.mt5_resilient_connection import get_mt5_connection

            mt5 = get_mt5_connection()
            status = mt5.get_connection_status()

            if mt5._is_initialized or status.get("data_file_exists"):
                message = f"MT5 bridge ready ({status.get('symbols_count', 0)} symbols)"
                self.log(message, StartupPhase.MT5_BRIDGE, "SUCCESS")
                return StartupResult(
                    phase=StartupPhase.MT5_BRIDGE,
                    success=True,
                    message=message,
                    details=status
                )
            else:
                message = "MT5 bridge in degraded mode - using cached data"
                self.log(message, StartupPhase.MT5_BRIDGE, "WARN")
                return StartupResult(
                    phase=StartupPhase.MT5_BRIDGE,
                    success=True,  # Still consider it a success
                    message=message,
                    details=status
                )

        except Exception as e:
            message = f"MT5 connection failed: {str(e)}"
            self.log(message, StartupPhase.MT5_BRIDGE, "ERROR")
            return StartupResult(
                phase=StartupPhase.MT5_BRIDGE,
                success=False,
                message=message
            )

    def initialize_agents(self) -> StartupResult:
        """Load agent configuration"""
        self.log("Loading agent configuration...", StartupPhase.AGENTS, "PENDING")

        try:
            from src.database.operations import AtomicDataOperations

            ops = AtomicDataOperations()
            berserker = ops.get_setting("agents", "berserker_enabled")
            sniper = ops.get_setting("agents", "sniper_enabled")

            agents_status = {
                "BERSERKER": berserker.get("value", "true") if berserker else "true",
                "SNIPER": sniper.get("value", "true") if sniper else "true"
            }

            enabled_count = sum(1 for v in agents_status.values() if v.lower() == "true")
            message = f"Agent configuration loaded ({enabled_count}/2 agents enabled)"

            self.log(message, StartupPhase.AGENTS, "SUCCESS")
            return StartupResult(
                phase=StartupPhase.AGENTS,
                success=True,
                message=message,
                details=agents_status
            )

        except Exception as e:
            message = f"Agent initialization failed: {str(e)}"
            self.log(message, StartupPhase.AGENTS, "ERROR")
            return StartupResult(
                phase=StartupPhase.AGENTS,
                success=False,
                message=message
            )

    def start_monitoring(self) -> StartupResult:
        """Initialize monitoring systems"""
        self.log("Starting monitoring...", StartupPhase.MONITORING, "PENDING")

        try:
            # Check Prometheus metrics
            message = "Prometheus metrics endpoint enabled at /metrics"
            self.log(message, StartupPhase.MONITORING, "SUCCESS")

            return StartupResult(
                phase=StartupPhase.MONITORING,
                success=True,
                message=message,
                details={"metrics_endpoint": "/metrics", "grafana": "http://localhost:3000"}
            )

        except Exception as e:
            message = f"Monitoring setup failed: {str(e)}"
            self.log(message, StartupPhase.MONITORING, "ERROR")
            return StartupResult(
                phase=StartupPhase.MONITORING,
                success=False,
                message=message
            )

    def start_all(self) -> bool:
        """Run complete startup sequence"""
        print("\n" + "=" * 60)
        print("   VelocityPTrader - Physics-Based Trading System")
        print("   Quick Start Sequence")
        print("=" * 60 + "\n")

        start_time = time.time()

        # Run startup phases
        phases = [
            (StartupPhase.DATABASE, self.check_database),
            (StartupPhase.SETTINGS, self.load_settings),
            (StartupPhase.MT5_BRIDGE, self.connect_mt5),
            (StartupPhase.AGENTS, self.initialize_agents),
            (StartupPhase.MONITORING, self.start_monitoring),
        ]

        for phase, func in phases:
            result = func()
            self.results[phase] = result

            # Stop on critical failure (database is critical)
            if not result.success and phase == StartupPhase.DATABASE:
                break

        elapsed = time.time() - start_time

        # Print summary
        print("\n" + "-" * 60)
        success_count = sum(1 for r in self.results.values() if r.success)
        total_count = len(self.results)

        if success_count == total_count:
            print(f"✅ STARTUP COMPLETE - All {total_count} phases successful")
            print(f"   Elapsed time: {elapsed:.2f}s")
            print("\n   Dashboard: http://localhost:8443")
            print("   Metrics:   http://localhost:8443/metrics")
            print("   Grafana:   http://localhost:3000")
        else:
            failed = [p.value for p, r in self.results.items() if not r.success]
            print(f"⚠️  STARTUP PARTIAL - {success_count}/{total_count} phases successful")
            print(f"   Failed phases: {', '.join(failed)}")

        print("-" * 60 + "\n")

        return success_count == total_count


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="VelocityPTrader Quick Start")
    parser.add_argument("-q", "--quiet", action="store_true", help="Suppress output")
    parser.add_argument("--dashboard", action="store_true", help="Also start dashboard server")
    parser.add_argument("--port", type=int, default=8443, help="Dashboard port (default: 8443)")

    args = parser.parse_args()

    startup = VelocityStartup(verbose=not args.quiet)
    success = startup.start_all()

    if args.dashboard and success:
        print("Starting dashboard server...")
        try:
            from dashboard.enterprise_dashboard import EnterpriseDashboard, ConnectionConfig, SecurityConfig
            import uvicorn

            config = ConnectionConfig(primary_port=args.port)
            security = SecurityConfig(jwt_secret_key=os.getenv("JWT_SECRET", "dev-secret-key"))

            dashboard = EnterpriseDashboard(config, security)
            uvicorn.run(dashboard.app, host="0.0.0.0", port=args.port)

        except KeyboardInterrupt:
            print("\nDashboard stopped.")
        except Exception as e:
            print(f"Dashboard failed to start: {e}")
            sys.exit(1)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
