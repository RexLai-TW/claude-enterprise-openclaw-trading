# OpenClaw Setup Guide

## What is OpenClaw?

[OpenClaw](https://github.com/openclaw/openclaw) is an agent-first open-source framework for building AI agents. This project is designed to run as an OpenClaw skill or standalone.

## Setup as OpenClaw Skill

### 1. Clone into your OpenClaw workspace

```bash
cd ~/.openclaw/workspace
git clone https://github.com/jerrylearnscoding/claude-enterprise-trading.git
```

### 2. Install dependencies

```bash
cd claude-enterprise-trading
pip install -r requirements.txt
```

### 3. Configure

```bash
cp .env.example .env
# Edit .env with your API keys
```

### 4. Create a skill wrapper (optional)

Create `skills/trading/SKILL.md` in your OpenClaw workspace:

```markdown
---
name: trading
description: Agentic trading - generate strategies from natural language, backtest, and get signals.
---

# Trading Skill

## Commands

### Generate Strategy
\`\`\`bash
cd ~/.openclaw/workspace/claude-enterprise-trading
python examples/quickstart.py --idea "YOUR TRADING IDEA"
\`\`\`

### Run Multi-Agent Portfolio
\`\`\`bash
python -m src.orchestrator --config examples/multi_agent_portfolio.yaml
\`\`\`
```

## Agent-First Design

This project follows OpenClaw's agent-first philosophy:

- **Structured outputs**: Strategy trees are JSON/YAML — easy for agents to parse
- **Programmatic interface**: All components have Python APIs, not just CLI
- **Traceable**: OpenTelemetry integration means agents can audit every step
- **Composable**: Each component (data, strategy, backtest, execution) works independently

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `ANTHROPIC_API_KEY` | Yes* | Claude API key for strategy generation |
| `COINGECKO_API_KEY` | No | CoinGecko Pro API key (free tier works without) |
| `FACTSET_API_KEY` | No | FactSet API key (enterprise users) |
| `OTEL_ENDPOINT` | No | OpenTelemetry collector endpoint |

*Not required if only running example strategies (backtest-only mode).
