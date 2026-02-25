"""
Vibe Coder — Iterative Strategy Refinement

Loop: NL description → generate strategy tree → backtest → evaluate metrics
→ if below targets, feed results back to Claude → regenerate

This is "vibe coding" for trading strategies: describe what you want,
let AI iterate until it works.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from opentelemetry import trace

from .nl_to_tree import NaturalLanguageToTree
from .tree_schema import StrategyTree
from ..backtest.engine import BacktestEngine, BacktestConfig, BacktestResult
from ..data.market_data import create_market_data_connector


@dataclass
class RefinementTargets:
    """Performance targets for strategy refinement"""
    min_sharpe_ratio: float = 1.0
    max_drawdown: float = 0.20  # 20%
    min_win_rate: float = 0.45  # 45%
    min_total_return: float = 0.05  # 5%
    max_iterations: int = 5


@dataclass
class RefinementResult:
    """Result of a vibe coding session"""
    strategy: StrategyTree
    backtest_result: BacktestResult
    iterations: int
    met_targets: bool
    history: List[Dict[str, Any]] = field(default_factory=list)


class VibeCoder:
    """
    Iterative strategy refinement engine.
    
    Describe your trading idea in natural language, set performance targets,
    and let the vibe coder iterate until the strategy meets your criteria.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.logger = logging.getLogger('claude_trading.vibe_coder')
        self.tracer = trace.get_tracer(__name__)
        self.nl_converter = NaturalLanguageToTree(api_key=api_key)
    
    async def refine(
        self,
        description: str,
        targets: Optional[RefinementTargets] = None,
        backtest_config: Optional[BacktestConfig] = None,
        symbol: str = "BTC-USD",
    ) -> RefinementResult:
        """
        Iteratively refine a strategy until it meets targets.
        
        Args:
            description: Natural language trading idea
            targets: Performance targets to meet
            backtest_config: Backtest configuration
            symbol: Trading symbol for backtesting
        
        Returns:
            RefinementResult with best strategy found
        """
        targets = targets or RefinementTargets()
        backtest_config = backtest_config or BacktestConfig(
            start_date=datetime.now() - timedelta(days=365),
            end_date=datetime.now(),
        )
        
        with self.tracer.start_as_current_span("vibe_coder.refine") as span:
            span.set_attribute("description", description[:200])
            span.set_attribute("max_iterations", targets.max_iterations)
            
            data_connector = create_market_data_connector()
            engine = BacktestEngine(config=backtest_config, data_connector=data_connector)
            
            history: List[Dict[str, Any]] = []
            best_strategy = None
            best_result = None
            best_score = -999.0
            
            current_description = description
            
            for iteration in range(1, targets.max_iterations + 1):
                self.logger.info(f"Vibe coding iteration {iteration}/{targets.max_iterations}")
                span.add_event(f"iteration_{iteration}")
                
                # Generate strategy tree
                strategy = await self.nl_converter.convert(current_description)
                
                # Run backtest
                result = await engine.run(strategy, symbol=symbol)
                
                # Evaluate
                score = self._score(result, targets)
                met = self._check_targets(result, targets)
                
                history.append({
                    "iteration": iteration,
                    "sharpe": result.sharpe_ratio,
                    "max_drawdown": result.max_drawdown,
                    "win_rate": result.win_rate,
                    "total_return": result.total_return,
                    "score": score,
                    "met_targets": met,
                })
                
                if score > best_score:
                    best_score = score
                    best_strategy = strategy
                    best_result = result
                
                if met:
                    self.logger.info(f"Targets met on iteration {iteration}!")
                    break
                
                # Build feedback for next iteration
                current_description = self._build_feedback_prompt(
                    description, result, targets, iteration
                )
            
            span.set_attribute("iterations_used", len(history))
            span.set_attribute("targets_met", history[-1]["met_targets"])
            
            return RefinementResult(
                strategy=best_strategy,
                backtest_result=best_result,
                iterations=len(history),
                met_targets=history[-1]["met_targets"],
                history=history,
            )
    
    def _score(self, result: BacktestResult, targets: RefinementTargets) -> float:
        """Score a backtest result against targets (higher = better)"""
        score = 0.0
        score += min(result.sharpe_ratio / max(targets.min_sharpe_ratio, 0.01), 2.0) * 30
        score += max(0, 1 - result.max_drawdown / max(targets.max_drawdown, 0.01)) * 25
        score += min(result.win_rate / max(targets.min_win_rate, 0.01), 2.0) * 25
        score += min(result.total_return / max(targets.min_total_return, 0.01), 3.0) * 20
        return score
    
    def _check_targets(self, result: BacktestResult, targets: RefinementTargets) -> bool:
        """Check if backtest result meets all targets"""
        return (
            result.sharpe_ratio >= targets.min_sharpe_ratio
            and result.max_drawdown <= targets.max_drawdown
            and result.win_rate >= targets.min_win_rate
            and result.total_return >= targets.min_total_return
        )
    
    def _build_feedback_prompt(
        self,
        original: str,
        result: BacktestResult,
        targets: RefinementTargets,
        iteration: int,
    ) -> str:
        """Build feedback prompt for Claude to improve the strategy"""
        issues = []
        if result.sharpe_ratio < targets.min_sharpe_ratio:
            issues.append(
                f"Sharpe ratio is {result.sharpe_ratio:.2f}, need >= {targets.min_sharpe_ratio:.2f}. "
                "Consider tighter entry conditions or better risk management."
            )
        if result.max_drawdown > targets.max_drawdown:
            issues.append(
                f"Max drawdown is {result.max_drawdown:.1%}, need <= {targets.max_drawdown:.1%}. "
                "Add stop-loss or reduce position sizes."
            )
        if result.win_rate < targets.min_win_rate:
            issues.append(
                f"Win rate is {result.win_rate:.1%}, need >= {targets.min_win_rate:.1%}. "
                "Entry signals may be too loose. Add confirmation indicators."
            )
        if result.total_return < targets.min_total_return:
            issues.append(
                f"Total return is {result.total_return:.1%}, need >= {targets.min_total_return:.1%}. "
                "Consider more aggressive position sizing or longer hold periods."
            )
        
        feedback = "\n".join(f"- {issue}" for issue in issues)
        
        return (
            f"Original idea: {original}\n\n"
            f"Iteration {iteration} backtest results:\n"
            f"- Total return: {result.total_return:.2%}\n"
            f"- Sharpe ratio: {result.sharpe_ratio:.2f}\n"
            f"- Max drawdown: {result.max_drawdown:.2%}\n"
            f"- Win rate: {result.win_rate:.1%}\n"
            f"- Total trades: {result.total_trades}\n\n"
            f"Issues to fix:\n{feedback}\n\n"
            f"Please improve the strategy to address these issues while keeping "
            f"the core trading idea intact."
        )
