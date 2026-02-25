# Architecture

## Overview

Claude Enterprise Trading follows an **agent-first** design: every component is built for AI agents to consume efficiently, with human interfaces layered on top.

## Pipeline

```
Natural Language Input
        │
        ▼
┌─────────────────┐     ┌──────────────┐
│ NL → Tree       │────▶│ Tree         │
│ (Claude API)    │     │ Validator    │
└─────────────────┘     └──────┬───────┘
                               │
                    ┌──────────▼───────────┐
                    │    Vibe Coder Loop    │
                    │  generate → backtest  │
                    │  → evaluate → refine  │
                    └──────────┬───────────┘
                               │
                    ┌──────────▼───────────┐
                    │   Backtest Engine     │
                    │   OHLCV + strategy   │
                    │   → metrics           │
                    └──────────┬───────────┘
                               │
                    ┌──────────▼───────────┐
                    │  Execution Runner     │
                    │  strategy → signals   │
                    │  (no real trades)     │
                    └──────────┬───────────┘
                               │
                    ┌──────────▼───────────┐
                    │  Circuit Breakers     │
                    │  safety checks        │
                    └──────────────────────┘
```

## Data Layer

The MCP Bridge abstracts data sources behind a unified interface:

- **Claude Enterprise MCP**: FactSet, MSCI, Google Drive (enterprise users)
- **Free tier**: yfinance (stocks), CoinGecko (crypto), Fear & Greed API

Data connectors are hot-swappable — same interface regardless of source.

## Strategy Trees

Strategy trees are deterministic if-else rule structures stored as JSON/YAML:

- **Condition nodes**: evaluate market data against thresholds
- **Action nodes**: specify buy/sell/hold with position sizing
- **Risk nodes**: stop-loss, take-profit, circuit breaker triggers

Trees are validated before execution to ensure structural correctness.

## Orchestrator

The orchestrator manages multiple strategy agents in parallel:

- Each agent runs independently with its own config
- Asyncio-based parallelism
- Portfolio-level risk aggregation
- Unified signal output

## Tracing

OpenTelemetry spans cover every pipeline step:

```
orchestrator.run
  └── agent.{name}
       ├── strategy_generation
       │    └── claude_api_call
       ├── tree_validation
       ├── backtest
       │    ├── data_fetch
       │    └── simulation
       ├── signal_generation
       └── circuit_breaker_check
```

Export to console (default) or any OTLP-compatible backend (Jaeger, Grafana Tempo, etc.)
