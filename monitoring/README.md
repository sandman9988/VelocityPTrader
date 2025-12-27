# VelocityPTrader Monitoring Stack

Comprehensive Grafana + Prometheus monitoring solution for the VelocityPTrader physics-based algorithmic trading system.

## Quick Start

### 1. Start the Prometheus Metrics Exporter

```bash
# From the project root directory
python monitoring/prometheus_exporter.py --port 9090
```

### 2. Start the Monitoring Stack

```bash
cd monitoring
docker-compose up -d
```

### 3. Access Dashboards

- **Grafana**: http://localhost:3000 (admin/velocitytrader)
- **Prometheus**: http://localhost:9091
- **Alertmanager**: http://localhost:9093

## Available Dashboards

### 1. Trading Performance
**UID**: `velocitytrader-performance`

Key metrics visualized:
- Total P&L with trend
- Win rate gauge
- Profit factor
- Trade statistics (wins/losses)
- Largest win/loss
- Average trade duration

### 2. Dual Agent Comparison
**UID**: `velocitytrader-agents`

Compare BERSERKER vs SNIPER agents:
- P&L comparison over time
- Win rate comparison
- Trade frequency
- Signal confidence levels
- Trade distribution pie chart

### 3. Market Regime & Physics
**UID**: `velocitytrader-physics`

Physics-based indicators:
- Current market regime (CHAOTIC/UNDERDAMPED/CRITICALLY_DAMPED/OVERDAMPED)
- Momentum, velocity, acceleration
- Market energy
- Natural frequency
- Damping ratio
- Performance by regime

### 4. Risk Management
**UID**: `velocitytrader-risk`

Risk metrics dashboard:
- Account balance/equity
- Drawdown analysis (current vs max)
- Sharpe, Sortino, Calmar ratios
- Recovery factor
- VaR (95%)
- Consecutive wins/losses

### 5. System Health & RL
**UID**: `velocitytrader-system`

System monitoring:
- System/MT5 connection status
- Uptime
- Error rates
- Database latency
- RL learning progress
- Exploration rate
- Model confidence

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    VelocityPTrader                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │  BERSERKER  │  │   SNIPER    │  │  Physics Engine     │ │
│  │   Agent     │  │   Agent     │  │  (Regime Detection) │ │
│  └─────────────┘  └─────────────┘  └─────────────────────┘ │
│          │               │                   │              │
│          └───────────────┴───────────────────┘              │
│                          │                                  │
│                  ┌───────▼───────┐                         │
│                  │  SQLite DB    │                         │
│                  │ trading_logs  │                         │
│                  └───────────────┘                         │
└──────────────────────────┬──────────────────────────────────┘
                           │
                   ┌───────▼───────┐
                   │   Prometheus  │
                   │   Exporter    │
                   │   :9090       │
                   └───────────────┘
                           │
              ┌────────────┼────────────┐
              │            │            │
      ┌───────▼───────┐    │    ┌───────▼───────┐
      │  Prometheus   │    │    │  Alertmanager │
      │    :9091      │◄───┘    │    :9093      │
      └───────────────┘         └───────────────┘
              │
      ┌───────▼───────┐
      │    Grafana    │
      │    :3000      │
      └───────────────┘
```

## Metrics Exposed

### Trading Metrics
| Metric | Type | Description |
|--------|------|-------------|
| velocitytrader_trades_total | Counter | Total trades executed |
| velocitytrader_win_rate | Gauge | Current win rate % |
| velocitytrader_pnl_total | Gauge | Total P&L in USD |
| velocitytrader_profit_factor | Gauge | Gross profit / loss ratio |
| velocitytrader_sharpe_ratio | Gauge | Risk-adjusted return |

### Agent Metrics
| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| velocitytrader_agent_trades_total | Counter | agent | Trades per agent |
| velocitytrader_agent_win_rate | Gauge | agent | Win rate per agent |
| velocitytrader_agent_pnl_total | Gauge | agent | P&L per agent |

### Physics Metrics
| Metric | Type | Description |
|--------|------|-------------|
| velocitytrader_regime_current | Gauge | Current regime (1-4) |
| velocitytrader_physics_momentum | Gauge | Market momentum |
| velocitytrader_physics_velocity | Gauge | Price velocity |
| velocitytrader_physics_energy | Gauge | Market energy |

### Risk Metrics
| Metric | Type | Description |
|--------|------|-------------|
| velocitytrader_max_drawdown | Gauge | Maximum drawdown % |
| velocitytrader_var_95 | Gauge | Value at Risk (95%) |
| velocitytrader_calmar_ratio | Gauge | Calmar ratio |

### System Metrics
| Metric | Type | Description |
|--------|------|-------------|
| velocitytrader_system_status | Gauge | 1=running, 0=stopped |
| velocitytrader_uptime_seconds | Counter | System uptime |
| velocitytrader_errors_total | Counter | Total errors |

## Alert Rules

Pre-configured alerts for:
- **Critical**: Drawdown > 25%, System down, MT5 disconnected
- **Warning**: Drawdown > 15%, Low win rate, High error rate
- **Agent**: Individual agent underperformance
- **RL**: Low model confidence, high exploration rate

## Customization

### Adding Custom Dashboards
1. Create JSON dashboard file in `grafana/dashboards/`
2. Restart Grafana or wait for auto-reload (30s)

### Modifying Alerts
Edit `prometheus/alert_rules.yml` and reload Prometheus:
```bash
curl -X POST http://localhost:9091/-/reload
```

### Configuring Notifications
Edit `prometheus/alertmanager.yml` to add:
- Slack webhooks
- Email notifications
- PagerDuty integration

## Troubleshooting

### No Data in Dashboards
1. Verify exporter is running: `curl http://localhost:9090/metrics`
2. Check Prometheus targets: http://localhost:9091/targets
3. Verify database exists: `ls -la trading_logs.db`

### Connection Issues
```bash
# Check container status
docker-compose ps

# View logs
docker-compose logs grafana
docker-compose logs prometheus
```

### Reset Grafana Password
```bash
docker exec -it velocitytrader-grafana grafana-cli admin reset-admin-password newpassword
```

## License

Part of VelocityPTrader - Physics-Based Algorithmic Trading System
