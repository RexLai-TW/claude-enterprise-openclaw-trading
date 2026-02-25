"""
Backtesting Engine for Strategy Trees

Provides comprehensive backtesting functionality for strategy trees against
historical market data with realistic execution simulation.
"""

from .engine import BacktestEngine, BacktestConfig, BacktestResult
from .report import BacktestReportGenerator, BacktestReport

__all__ = [
    'BacktestEngine',
    'BacktestConfig', 
    'BacktestResult',
    'BacktestReportGenerator',
    'BacktestReport'
]