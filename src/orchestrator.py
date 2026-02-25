"""
Orchestrator — Multi-Strategy Agent Pipeline

Wires the full pipeline: data → strategy → backtest → execute
Supports running multiple strategies in parallel via asyncio.
Each strategy runs as an independent agent with its own config.
"""

import asyncio
import logging
import json
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from opentelemetry import trace

from .data.market_data import create_market_data_connector, MarketDataInterface
from .strategy.nl_to_tree import NaturalLanguageToTree
from .strategy.tree_schema import StrategyTree
from .strategy.vibe_coder import VibeCoder, RefinementTargets, RefinementResult
from .backtest.engine import BacktestEngine, BacktestConfig, BacktestResult
from .backtest.report import generate_report, print_report
from .execution.runner import ExecutionRunner, ExecutionConfig, SignalType
from .execution.circuit_breaker import CircuitBreakerConfig
from .monitoring.otel_tracer import setup_tracing


@dataclass
class AgentConfig:
    """Configuration for a single strategy agent"""
    name: str
    description: str  # Natural language strategy description
    symbol: str = "BTC-USD"
    enabled: bool = True
    # Backtest settings
    lookback_days: int = 365
    initial_capital: float = 100000.0
    # Refinement targets
    min_sharpe: float = 1.0
    max_drawdown: float = 0.20
    min_win_rate: float = 0.45
    # Execution
    max_position_pct: float = 0.10
    max_daily_loss_pct: float = 0.05


@dataclass
class AgentResult:
    """Result from running a single agent"""
    name: str
    strategy: Optional[StrategyTree]
    backtest: Optional[BacktestResult]
    signals: List[Dict[str, Any]]
    error: Optional[str] = None
    duration_seconds: float = 0.0


class Orchestrator:
    """
    Multi-strategy orchestrator.
    
    Run N strategies in parallel, each as an independent agent.
    Aggregates signals and applies portfolio-level risk rules.
    
    Usage:
        orch = Orchestrator(api_key="sk-...")
        agents = [
            AgentConfig(name="fear-greed", description="Buy BTC when fear index < 20"),
            AgentConfig(name="trend", description="MA crossover trend following on ETH"),
        ]
        results = await orch.run(agents)
    """
    
    def __init__(self, api_key: Optional[str] = None, enable_tracing: bool = True):
        self.logger = logging.getLogger('claude_trading.orchestrator')
        self.tracer = trace.get_tracer(__name__)
        self.api_key = api_key
        
        if enable_tracing:
            from .monitoring.otel_tracer import TraceConfig
            setup_tracing(TraceConfig(service_name="claude-enterprise-trading"))
    
    async def run(self, agents: List[AgentConfig]) -> List[AgentResult]:
        """Run multiple strategy agents in parallel"""
        with self.tracer.start_as_current_span("orchestrator.run") as span:
            span.set_attribute("agent_count", len(agents))
            
            enabled = [a for a in agents if a.enabled]
            self.logger.info(f"Running {len(enabled)} agents ({len(agents)} total, "
                           f"{len(agents) - len(enabled)} disabled)")
            
            tasks = [self._run_agent(agent) for agent in enabled]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Handle exceptions
            final_results = []
            for agent, result in zip(enabled, results):
                if isinstance(result, Exception):
                    self.logger.error(f"Agent {agent.name} failed: {result}")
                    final_results.append(AgentResult(
                        name=agent.name,
                        strategy=None,
                        backtest=None,
                        signals=[],
                        error=str(result),
                    ))
                else:
                    final_results.append(result)
            
            # Print summary
            self._print_summary(final_results)
            
            return final_results
    
    async def _run_agent(self, agent: AgentConfig) -> AgentResult:
        """Run a single strategy agent through the full pipeline"""
        start_time = datetime.now()
        
        with self.tracer.start_as_current_span(f"agent.{agent.name}") as span:
            span.set_attribute("agent.name", agent.name)
            span.set_attribute("agent.symbol", agent.symbol)
            
            self.logger.info(f"[{agent.name}] Starting pipeline for: {agent.description[:80]}")
            
            # 1. Generate & refine strategy
            span.add_event("strategy_generation")
            vibe = VibeCoder(api_key=self.api_key)
            
            backtest_config = BacktestConfig(
                start_date=datetime.now() - timedelta(days=agent.lookback_days),
                end_date=datetime.now(),
                initial_capital=agent.initial_capital,
            )
            
            targets = RefinementTargets(
                min_sharpe_ratio=agent.min_sharpe,
                max_drawdown=agent.max_drawdown,
                min_win_rate=agent.min_win_rate,
            )
            
            refinement = await vibe.refine(
                description=agent.description,
                targets=targets,
                backtest_config=backtest_config,
                symbol=agent.symbol,
            )
            
            self.logger.info(
                f"[{agent.name}] Strategy refined in {refinement.iterations} iterations. "
                f"Met targets: {refinement.met_targets}"
            )
            
            # 2. Print backtest report
            span.add_event("backtest_report")
            report = generate_report(refinement.backtest_result)
            self.logger.info(f"[{agent.name}] Backtest: "
                           f"return={report['total_return']:.2%}, "
                           f"sharpe={report['sharpe_ratio']:.2f}, "
                           f"drawdown={report['max_drawdown']:.2%}")
            
            # 3. Generate current signals
            span.add_event("signal_generation")
            data_connector = create_market_data_connector()
            
            exec_config = ExecutionConfig(
                max_position_pct=agent.max_position_pct,
                circuit_breaker=CircuitBreakerConfig(
                    max_daily_loss_pct=agent.max_daily_loss_pct,
                ),
            )
            
            runner = ExecutionRunner(
                config=exec_config,
                data_connector=data_connector,
            )
            
            execution_result = await runner.execute(
                strategy=refinement.strategy,
                symbol=agent.symbol,
                capital=agent.initial_capital,
            )
            
            signals = [
                {
                    "type": s.signal_type.value,
                    "symbol": s.symbol,
                    "amount": s.amount,
                    "reason": s.reason,
                    "timestamp": s.timestamp.isoformat() if hasattr(s, 'timestamp') else None,
                }
                for s in execution_result.signals
            ] if hasattr(execution_result, 'signals') else []
            
            duration = (datetime.now() - start_time).total_seconds()
            
            return AgentResult(
                name=agent.name,
                strategy=refinement.strategy,
                backtest=refinement.backtest_result,
                signals=signals,
                duration_seconds=duration,
            )
    
    def _print_summary(self, results: List[AgentResult]):
        """Print a summary of all agent results"""
        print("\n" + "=" * 60)
        print("  ORCHESTRATOR SUMMARY")
        print("=" * 60)
        
        for r in results:
            status = "✅" if not r.error else "❌"
            print(f"\n{status} Agent: {r.name}")
            
            if r.error:
                print(f"   Error: {r.error}")
                continue
            
            if r.backtest:
                print(f"   Return: {r.backtest.total_return:.2%}")
                print(f"   Sharpe: {r.backtest.sharpe_ratio:.2f}")
                print(f"   Max DD: {r.backtest.max_drawdown:.2%}")
                print(f"   Win Rate: {r.backtest.win_rate:.1%}")
            
            if r.signals:
                print(f"   Signals: {len(r.signals)}")
                for s in r.signals[:3]:
                    print(f"     → {s['type'].upper()} {s['symbol']}: {s['reason']}")
            
            print(f"   Duration: {r.duration_seconds:.1f}s")
        
        print("\n" + "=" * 60)
    
    async def run_single(self, description: str, symbol: str = "BTC-USD") -> AgentResult:
        """Convenience: run a single strategy from natural language"""
        agent = AgentConfig(name="default", description=description, symbol=symbol)
        results = await self.run([agent])
        return results[0]


async def main():
    """CLI entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Claude Enterprise Trading Orchestrator")
    parser.add_argument("--description", "-d", type=str, 
                       help="Natural language strategy description")
    parser.add_argument("--config", "-c", type=str,
                       help="Path to agent config YAML file")
    parser.add_argument("--symbol", "-s", type=str, default="BTC-USD",
                       help="Trading symbol (default: BTC-USD)")
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
    
    orch = Orchestrator()
    
    if args.description:
        result = await orch.run_single(args.description, symbol=args.symbol)
    elif args.config:
        import yaml
        with open(args.config) as f:
            config = yaml.safe_load(f)
        agents = [AgentConfig(**a) for a in config.get("agents", [])]
        await orch.run(agents)
    else:
        # Demo
        result = await orch.run_single(
            "Buy BTC when RSI drops below 30 and sell when it goes above 70. "
            "Use 5% position size with 3% stop loss.",
            symbol="BTC-USD",
        )


if __name__ == "__main__":
    asyncio.run(main())
