"""
Quickstart Example — Claude Enterprise Trading

One-file demo: Natural Language → Strategy Tree → Backtest → Results

Requirements:
    pip install -r requirements.txt
    export ANTHROPIC_API_KEY=your-key-here

Usage:
    python examples/quickstart.py
    python examples/quickstart.py --idea "Buy ETH when RSI < 25, sell when RSI > 75"
    python examples/quickstart.py --symbol ETH-USD --days 180
"""

import asyncio
import argparse
import logging
import os
import sys

# Add parent dir to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.strategy.nl_to_tree import NaturalLanguageToTree
from src.strategy.tree_validator import StrategyTreeValidator
from src.backtest.engine import BacktestEngine, BacktestConfig
from src.backtest.report import generate_report, print_report
from src.data.market_data import create_market_data_connector
from src.monitoring.otel_tracer import setup_tracing
from datetime import datetime, timedelta


async def main():
    parser = argparse.ArgumentParser(description="Claude Enterprise Trading — Quickstart")
    parser.add_argument(
        "--idea", "-i", type=str,
        default="Buy BTC when the Fear & Greed index drops below 20, accumulate in 3 batches "
                "4 hours apart, max 5% of portfolio per batch. Sell when Fear & Greed goes above 75. "
                "Stop loss at 8% drawdown.",
    )
    parser.add_argument("--symbol", "-s", type=str, default="BTC-USD")
    parser.add_argument("--days", "-d", type=int, default=365)
    parser.add_argument("--capital", type=float, default=100000.0)
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s %(message)s")
    logger = logging.getLogger("quickstart")

    # Setup tracing
    from src.monitoring.otel_tracer import TraceConfig
    setup_tracing(TraceConfig(service_name="quickstart-demo"))

    print("\n" + "=" * 60)
    print("  Claude Enterprise Trading — Quickstart Demo")
    print("=" * 60)

    # Check API key
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("\n⚠️  ANTHROPIC_API_KEY not set.")
        print("   export ANTHROPIC_API_KEY=your-key-here")
        print("\n   Running with example strategy tree instead...\n")
        await run_example_backtest(args)
        return

    print(f"\n📝 Trading idea: {args.idea}")
    print(f"📈 Symbol: {args.symbol}")
    print(f"📅 Lookback: {args.days} days")
    print(f"💰 Capital: ${args.capital:,.0f}")

    # Step 1: Convert NL to strategy tree
    print("\n─── Step 1: Natural Language → Strategy Tree ───")
    converter = NaturalLanguageToTree()
    strategy = await converter.convert(args.idea)
    print(f"✅ Generated strategy: {strategy.name}")
    print(f"   Nodes: {len(strategy.nodes)}")
    print(f"   Description: {strategy.description[:100]}")

    # Step 2: Validate
    print("\n─── Step 2: Validate Strategy Tree ───")
    validator = StrategyTreeValidator()
    validation = validator.validate(strategy)
    if validation.is_valid:
        print("✅ Strategy tree is valid")
    else:
        print("⚠️  Validation issues:")
        for err in validation.errors:
            print(f"   ❌ {err}")
        for warn in validation.warnings:
            print(f"   ⚠️  {warn}")

    # Step 3: Backtest
    print("\n─── Step 3: Backtest ───")
    config = BacktestConfig(
        start_date=datetime.now() - timedelta(days=args.days),
        end_date=datetime.now(),
        initial_capital=args.capital,
    )
    data_connector = create_market_data_connector()
    engine = BacktestEngine(config=config, data_connector=data_connector)
    result = await engine.run(strategy, symbol=args.symbol)

    # Step 4: Report
    print("\n─── Step 4: Results ───")
    report = generate_report(result)
    print_report(report)

    # Save strategy tree
    out_path = f"output_strategy_{args.symbol.replace('-', '_').lower()}.json"
    strategy_dict = strategy.to_dict() if hasattr(strategy, 'to_dict') else {"name": strategy.name}
    import json
    with open(out_path, "w") as f:
        json.dump(strategy_dict, f, indent=2, default=str)
    print(f"\n💾 Strategy tree saved to: {out_path}")
    print("\n🔒 Remember: AI generates the strategy script.")
    print("   It does NOT execute trades. The switch is always in your hands.\n")


async def run_example_backtest(args):
    """Run backtest with a hardcoded example strategy (no API key needed)"""
    from src.strategy.tree_schema import create_simple_moving_average_strategy

    print("─── Running Example: SMA Crossover Strategy ───\n")
    strategy = create_simple_moving_average_strategy()
    print(f"📊 Strategy: {strategy.name}")
    print(f"   {strategy.description}")

    config = BacktestConfig(
        start_date=datetime.now() - timedelta(days=args.days),
        end_date=datetime.now(),
        initial_capital=args.capital,
    )
    data_connector = create_market_data_connector()
    engine = BacktestEngine(config=config, data_connector=data_connector)

    print(f"\n⏳ Backtesting {args.symbol} over {args.days} days...")
    result = await engine.run(strategy, symbol=args.symbol)

    report = generate_report(result)
    print_report(report)
    print("\n💡 Set ANTHROPIC_API_KEY to generate custom strategies from natural language!\n")


if __name__ == "__main__":
    asyncio.run(main())
