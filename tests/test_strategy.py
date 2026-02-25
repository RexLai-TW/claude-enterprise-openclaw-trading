"""Tests for strategy tree schema and validation"""
import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.strategy.tree_schema import (
    StrategyTree, StrategyNode, Condition, Action,
    ActionType, ConditionType, OperatorType, IndicatorType,
    PositionSizingMethod, create_simple_moving_average_strategy,
)
from src.strategy.tree_validator import StrategyTreeValidator, ValidationResult


class TestStrategyTree:
    def test_create_sma_strategy(self):
        strategy = create_simple_moving_average_strategy()
        assert strategy is not None
        assert strategy.name is not None
        assert len(strategy.nodes) > 0

    def test_strategy_has_root_node(self):
        strategy = create_simple_moving_average_strategy()
        root_ids = [n.id for n in strategy.nodes if n.id == strategy.root_node_id]
        assert len(root_ids) == 1

    def test_action_types(self):
        assert ActionType.BUY.value == "buy"
        assert ActionType.SELL.value == "sell"
        assert ActionType.HOLD.value == "hold"
        assert ActionType.STOP_LOSS.value == "stop_loss"

    def test_condition_types(self):
        assert ConditionType.PRICE_COMPARISON.value == "price_comparison"
        assert ConditionType.TECHNICAL_INDICATOR.value == "technical_indicator"
        assert ConditionType.SENTIMENT_CONDITION.value == "sentiment_condition"


class TestTreeValidator:
    def test_validate_sma_strategy(self):
        strategy = create_simple_moving_average_strategy()
        validator = StrategyTreeValidator()
        result = validator.validate(strategy)
        assert isinstance(result, ValidationResult)
        assert result.is_valid

    def test_empty_strategy_invalid(self):
        strategy = StrategyTree(
            name="empty",
            description="Empty strategy",
            nodes=[],
            root_node_id="nonexistent",
        )
        validator = StrategyTreeValidator()
        result = validator.validate(strategy)
        assert not result.is_valid


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
