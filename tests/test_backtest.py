"""Tests for backtest engine"""
import pytest
import asyncio
import sys
import os
from datetime import datetime, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.backtest.engine import BacktestEngine, BacktestConfig
from src.backtest.report import generate_report
from src.strategy.tree_schema import create_simple_moving_average_strategy
from src.data.market_data import create_market_data_connector


class TestBacktestConfig:
    def test_default_config(self):
        config = BacktestConfig(
            start_date=datetime(2025, 1, 1),
            end_date=datetime(2025, 12, 31),
        )
        assert config.initial_capital == 100000.0
        assert config.slippage == 0.001
        assert config.commission == 0.001

    def test_custom_config(self):
        config = BacktestConfig(
            start_date=datetime(2025, 1, 1),
            end_date=datetime(2025, 6, 30),
            initial_capital=50000.0,
            slippage=0.002,
            commission=0.0005,
        )
        assert config.initial_capital == 50000.0
        assert config.slippage == 0.002


class TestBacktestEngine:
    def test_engine_creation(self):
        config = BacktestConfig(
            start_date=datetime.now() - timedelta(days=30),
            end_date=datetime.now(),
        )
        connector = create_market_data_connector()
        engine = BacktestEngine(config=config, data_connector=connector)
        assert engine is not None

    def test_report_generation(self):
        """Test that report generation works with mock data"""
        from src.backtest.engine import BacktestResult
        result = BacktestResult(
            total_return=0.15,
            sharpe_ratio=1.5,
            max_drawdown=0.08,
            win_rate=0.55,
            total_trades=42,
            profitable_trades=23,
            losing_trades=19,
            avg_trade_return=0.003,
            start_date=datetime(2025, 1, 1),
            end_date=datetime(2025, 12, 31),
            initial_capital=100000.0,
            final_capital=115000.0,
            trade_log=[],
        )
        report = generate_report(result)
        assert report["total_return"] == 0.15
        assert report["sharpe_ratio"] == 1.5
        assert "total_trades" in report


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
