# Claude Enterprise Trading

**Agentic Trading Infrastructure — Turn trading ideas into executable strategies using Claude Enterprise + OpenClaw**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![OpenClaw Compatible](https://img.shields.io/badge/OpenClaw-compatible-orange.svg)](https://github.com/openclaw/openclaw)

> 🔒 **AI never touches your money.** It generates deterministic strategy scripts — you control the switch.

---

## What is this?

A framework that connects Claude Enterprise's MCP data connectors with an agent-first trading pipeline. Describe your trading idea in plain language, and the system:

1. **Generates** a deterministic strategy tree (not a black box — every rule is visible)
2. **Backtests** against historical data with real metrics
3. **Iterates** automatically until performance targets are met
4. **Outputs** executable signals with full audit trail

```
You: "Buy BTC when Fear & Greed index drops below 20, 
      accumulate in 3 batches, stop loss at 8%"

AI:  → Generates if-else strategy tree
     → Backtests: Sharpe 1.4, MaxDD 6.2%, Win 58%
     → Outputs: BUY signal, 5% position, reason logged
```

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                   ORCHESTRATOR                       │
│          (Multi-agent parallel execution)            │
├─────────────────────────────────────────────────────┤
│                                                      │
│  ┌──────────┐  ┌───────────┐  ┌──────────────────┐  │
│  │  MCP     │  │ Strategy  │  │    Backtest      │  │
│  │  Bridge  │→ │ Generator │→ │    Engine         │  │
│  │          │  │ (Claude)  │  │                   │  │
│  │ FactSet  │  │           │  │ Sharpe/DD/WinRate │  │
│  │ MSCI     │  │ NL→Tree   │  │                   │  │
│  │ CoinGecko│  │ Validator │  └────────┬─────────┘  │
│  │ yfinance │  │ VibeCoder │           │             │
│  └──────────┘  └───────────┘           ▼             │
│                               ┌──────────────────┐  │
│                               │   Execution      │  │
│                               │   Runner         │  │
│                               │                  │  │
│                               │ Signals only     │  │
│                               │ (no real trades) │  │
│                               │ Circuit breakers │  │
│                               └──────────────────┘  │
│                                                      │
│  ╔══════════════════════════════════════════════════╗ │
│  ║  OpenTelemetry — Every step traced & auditable  ║ │
│  ╚══════════════════════════════════════════════════╝ │
└─────────────────────────────────────────────────────┘
```

## Features

- **Natural Language → Strategy Tree**: Describe trading ideas in plain English/Chinese, Claude generates deterministic if-else rule trees
- **Vibe Coder**: Iterative refinement — generate, backtest, evaluate, improve automatically
- **MCP Bridge**: Claude Enterprise MCP connectors for FactSet, MSCI (falls back to yfinance/CoinGecko for free tier)
- **Backtest Engine**: Full backtesting with Sharpe ratio, max drawdown, win rate, trade log
- **Multi-Agent Orchestrator**: Run multiple strategies in parallel, each as an independent agent
- **Circuit Breakers**: Max drawdown halt, position limits, rate limits, cooldown periods
- **OpenTelemetry Tracing**: Every pipeline step traced — data fetch, strategy eval, signal generation
- **White Box**: Every rule in the strategy tree is visible and auditable. No black box.

## Quickstart

### 1. Install

```bash
git clone https://github.com/jerrylearnscoding/claude-enterprise-trading.git
cd claude-enterprise-trading
pip install -r requirements.txt
```

### 2. Set API Key

```bash
export ANTHROPIC_API_KEY=your-key-here
```

### 3. Run

```bash
# Quick demo (works without API key — uses example strategy)
python examples/quickstart.py

# With your own trading idea
python examples/quickstart.py --idea "Buy ETH when RSI drops below 25, sell above 70"

# Multi-agent portfolio
python -m src.orchestrator --config examples/multi_agent_portfolio.yaml
```

### 4. Use as Library

```python
import asyncio
from src.orchestrator import Orchestrator, AgentConfig

async def main():
    orch = Orchestrator()
    result = await orch.run_single(
        "Buy BTC when Fear & Greed < 20, sell when > 75, stop loss 8%",
        symbol="BTC-USD"
    )
    print(f"Return: {result.backtest.total_return:.2%}")
    print(f"Sharpe: {result.backtest.sharpe_ratio:.2f}")

asyncio.run(main())
```

## How It Works

### Strategy Trees

Instead of letting AI make trading decisions in real-time, this system generates **deterministic strategy trees** — structured if-else rules that execute predictably:

```yaml
# Example: Fear & Greed Accumulation
nodes:
  - id: check_fear
    condition: fear_greed_index < 20
    true: buy_batch    # Extreme fear → accumulate
    false: check_greed

  - id: check_greed
    condition: fear_greed_index > 75
    true: sell_all     # Greed → take profit
    false: check_stop

  - id: check_stop
    condition: unrealized_pnl < -8%
    true: stop_loss    # Risk management
    false: hold
```

**AI generates the tree. The tree executes the trades. 100% your rules.**

### Vibe Coder Loop

```
     ┌──────────────────┐
     │ Your trading idea │
     └────────┬─────────┘
              ▼
     ┌──────────────────┐
     │ Claude generates  │
     │ strategy tree     │◄──────────────┐
     └────────┬─────────┘               │
              ▼                          │
     ┌──────────────────┐               │
     │ Backtest against  │               │
     │ historical data   │               │
     └────────┬─────────┘               │
              ▼                          │
     ┌──────────────────┐    No         │
     │ Meets targets?   │──────────────►│
     │ (Sharpe/DD/WR)   │    Feed back  │
     └────────┬─────────┘    results    │
              │ Yes                      
              ▼                          
     ┌──────────────────┐               
     │ Final strategy   │               
     │ ready to deploy  │               
     └──────────────────┘               
```

### MCP Bridge

Claude Enterprise provides MCP (Model Context Protocol) connectors to enterprise data sources. This project bridges them into the trading pipeline:

| Data Source | Enterprise | Free Fallback |
|-------------|-----------|---------------|
| Market Data | FactSet | yfinance |
| Index Data | MSCI | yfinance |
| Crypto | - | CoinGecko |
| Sentiment | - | Fear & Greed API |

## OpenClaw Integration

This project is built with an **agent-first** architecture, designed to work with [OpenClaw](https://github.com/openclaw/openclaw):

- Every interface is optimized for agent consumption first, human viewing second
- Strategy trees are structured data that agents can parse and execute efficiently
- The orchestrator can run as an OpenClaw skill
- OpenTelemetry traces integrate with OpenClaw's monitoring

See [docs/openclaw_setup.md](docs/openclaw_setup.md) for setup instructions.

## Project Structure

```
claude-enterprise-trading/
├── src/
│   ├── data/          # MCP bridge + market data connectors
│   ├── strategy/      # NL→tree, validation, vibe coder
│   ├── backtest/      # Backtest engine + reports
│   ├── execution/     # Signal runner + circuit breakers
│   ├── monitoring/    # OpenTelemetry + dashboard
│   └── orchestrator.py # Multi-agent pipeline
├── examples/          # Ready-to-run strategy configs
├── tests/             # Unit tests
├── config/            # Default configurations
└── docs/              # Architecture + setup guides
```

## Safety

This system is designed with safety as a core principle:

- **AI never executes trades directly** — it generates signals and scripts
- **Circuit breakers** halt trading on max drawdown, position limits, or rate limits
- **Full audit trail** via OpenTelemetry — every decision is logged
- **Strategy trees are white box** — every rule is visible, no hidden logic
- **Human approval required** — the final switch is always in your hands

## Contributing

PRs welcome! Areas we'd love help with:

- Additional data source connectors
- More technical indicators
- Strategy tree visualization
- Exchange API integrations (for signal forwarding)
- Documentation improvements

## License

MIT — see [LICENSE](LICENSE) for details.

---

**Built with [Claude](https://anthropic.com) + [OpenClaw](https://github.com/openclaw/openclaw)**
**Built by @jjjjeerryyyy — 10 years in AI, still learning to trade.**
