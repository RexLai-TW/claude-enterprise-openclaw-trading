"""
Strategy Tree Validator

Validates strategy trees for safety, correctness, and risk management compliance.
Ensures strategies are safe to execute before deployment.
"""

from typing import List, Dict, Any, Optional, Tuple
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
import re

from .tree_schema import (
    StrategyTree, StrategyNode, Condition, Action,
    ConditionType, ActionType, IndicatorType, PositionSizingMethod
)
from opentelemetry import trace

@dataclass
class ValidationResult:
    """Result of strategy tree validation"""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    risk_score: float  # 0-100, higher = riskier
    recommendations: List[str]

@dataclass
class ValidationRule:
    """A single validation rule"""
    name: str
    description: str
    severity: str  # "error", "warning", "info"
    check_function: callable

class StrategyTreeValidator:
    """
    Comprehensive strategy tree validator
    
    Validates:
    - Tree structure integrity
    - Risk management compliance
    - Position sizing safety
    - Technical indicator parameters
    - Execution logic soundness
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.logger = logging.getLogger('claude_trading.validator')
        self.tracer = trace.get_tracer(__name__)
        
        # Load configuration
        self.config = config or self._default_config()
        
        # Initialize validation rules
        self.rules = self._initialize_rules()
    
    def _default_config(self) -> Dict[str, Any]:
        """Default validation configuration"""
        return {
            'max_position_size_pct': 25.0,  # Max 25% per position
            'max_portfolio_risk_pct': 10.0,  # Max 10% portfolio risk
            'max_leverage': 2.0,  # Max 2x leverage
            'min_stop_loss_pct': 1.0,  # Min 1% stop loss
            'max_stop_loss_pct': 20.0,  # Max 20% stop loss
            'max_daily_executions': 10,  # Max trades per day
            'required_risk_controls': True,  # Require stop losses
            'max_correlation': 0.8,  # Max correlation between positions
            'min_lookback_days': 30,  # Min historical data required
            'max_symbols_per_strategy': 20,  # Max symbols in one strategy
            'blacklisted_symbols': [],  # Symbols not allowed
            'allowed_timeframes': ['1m', '5m', '15m', '30m', '1h', '1d', '1w'],
        }
    
    def _initialize_rules(self) -> List[ValidationRule]:
        """Initialize all validation rules"""
        return [
            ValidationRule(
                "tree_structure",
                "Validate tree structure integrity",
                "error",
                self._validate_tree_structure
            ),
            ValidationRule(
                "position_sizing",
                "Validate position sizing parameters",
                "error", 
                self._validate_position_sizing
            ),
            ValidationRule(
                "risk_management",
                "Validate risk management controls",
                "error",
                self._validate_risk_management
            ),
            ValidationRule(
                "technical_indicators",
                "Validate technical indicator parameters",
                "warning",
                self._validate_technical_indicators
            ),
            ValidationRule(
                "execution_logic",
                "Validate execution logic and conditions",
                "warning",
                self._validate_execution_logic
            ),
            ValidationRule(
                "portfolio_limits",
                "Validate portfolio-level limits",
                "error",
                self._validate_portfolio_limits
            ),
            ValidationRule(
                "symbol_validation",
                "Validate symbols and timeframes",
                "warning",
                self._validate_symbols
            ),
            ValidationRule(
                "performance_targets",
                "Check for realistic performance expectations",
                "info",
                self._validate_performance_expectations
            ),
        ]
    
    def validate(self, strategy: StrategyTree) -> ValidationResult:
        """
        Comprehensive validation of strategy tree
        """
        with self.tracer.start_as_current_span("validate_strategy_tree") as span:
            span.set_attribute("strategy_id", strategy.id)
            span.set_attribute("strategy_name", strategy.name)
            
            errors = []
            warnings = []
            recommendations = []
            risk_factors = []
            
            self.logger.info(f"Validating strategy: {strategy.name} (ID: {strategy.id})")
            
            # Run all validation rules
            for rule in self.rules:
                try:
                    rule_result = rule.check_function(strategy)
                    
                    if rule.severity == "error":
                        errors.extend(rule_result.get('errors', []))
                    elif rule.severity == "warning": 
                        warnings.extend(rule_result.get('warnings', []))
                    
                    recommendations.extend(rule_result.get('recommendations', []))
                    risk_factors.extend(rule_result.get('risk_factors', []))
                    
                except Exception as e:
                    error_msg = f"Validation rule '{rule.name}' failed: {str(e)}"
                    self.logger.error(error_msg)
                    errors.append(error_msg)
            
            # Calculate risk score
            risk_score = self._calculate_risk_score(strategy, risk_factors)
            
            # Determine overall validity
            is_valid = len(errors) == 0
            
            result = ValidationResult(
                is_valid=is_valid,
                errors=errors,
                warnings=warnings,
                risk_score=risk_score,
                recommendations=recommendations
            )
            
            # Log validation summary
            status = "VALID" if is_valid else "INVALID"
            self.logger.info(f"Validation complete: {status} (Risk Score: {risk_score:.1f})")
            
            if errors:
                self.logger.error(f"Validation errors: {len(errors)}")
                for error in errors:
                    self.logger.error(f"  - {error}")
            
            if warnings:
                self.logger.warning(f"Validation warnings: {len(warnings)}")
                for warning in warnings:
                    self.logger.warning(f"  - {warning}")
            
            span.set_attribute("is_valid", is_valid)
            span.set_attribute("error_count", len(errors))
            span.set_attribute("warning_count", len(warnings))
            span.set_attribute("risk_score", risk_score)
            
            return result
    
    def _validate_tree_structure(self, strategy: StrategyTree) -> Dict[str, List[str]]:
        """Validate tree structure integrity"""
        errors = []
        warnings = []
        
        # Check for circular references
        tree_errors = strategy.validate_tree_structure()
        errors.extend(tree_errors)
        
        # Validate root node exists
        if not strategy.root_node:
            errors.append("Strategy tree must have a root node")
            return {'errors': errors, 'warnings': warnings}
        
        # Check node ID uniqueness
        all_nodes = strategy.get_all_nodes()
        node_ids = [node.id for node in all_nodes]
        
        if len(node_ids) != len(set(node_ids)):
            duplicates = [nid for nid in set(node_ids) if node_ids.count(nid) > 1]
            errors.append(f"Duplicate node IDs found: {duplicates}")
        
        # Validate node naming conventions
        for node in all_nodes:
            if not re.match(r'^[a-zA-Z0-9_]+$', node.id):
                warnings.append(f"Node ID '{node.id}' should only contain letters, numbers, and underscores")
            
            if not node.name:
                warnings.append(f"Node '{node.id}' should have a descriptive name")
        
        # Check for orphaned nodes (no conditions or actions)
        for node in all_nodes:
            if not node.conditions and not node.actions and not node.children:
                warnings.append(f"Node '{node.id}' has no conditions, actions, or children")
        
        return {
            'errors': errors,
            'warnings': warnings,
            'recommendations': []
        }
    
    def _validate_position_sizing(self, strategy: StrategyTree) -> Dict[str, List[str]]:
        """Validate position sizing parameters"""
        errors = []
        warnings = []
        recommendations = []
        risk_factors = []
        
        max_position_size = self.config['max_position_size_pct']
        total_exposure = 0.0
        
        for node in strategy.get_all_nodes():
            for action in node.actions:
                if action.type in [ActionType.BUY]:
                    # Validate position size
                    if action.sizing_method == PositionSizingMethod.PERCENTAGE:
                        if action.amount > max_position_size:
                            errors.append(
                                f"Position size {action.amount}% exceeds maximum {max_position_size}% "
                                f"in action for {action.symbol}"
                            )
                        
                        total_exposure += action.amount
                        
                        if action.amount > 10:
                            risk_factors.append(f"Large position size: {action.amount}%")
                    
                    elif action.sizing_method == PositionSizingMethod.FIXED_AMOUNT:
                        if action.amount <= 0:
                            errors.append(f"Fixed amount must be positive for {action.symbol}")
                    
                    # Check for position sizing parameters
                    if 'max_position_value' in action.parameters:
                        max_val = action.parameters['max_position_value']
                        if max_val <= 0:
                            errors.append(f"max_position_value must be positive for {action.symbol}")
        
        # Check total exposure
        if total_exposure > 100:
            errors.append(f"Total position exposure {total_exposure}% exceeds 100%")
        elif total_exposure > 80:
            warnings.append(f"High total exposure: {total_exposure}%")
            recommendations.append("Consider reducing position sizes to manage risk")
        
        return {
            'errors': errors,
            'warnings': warnings,
            'recommendations': recommendations,
            'risk_factors': risk_factors
        }
    
    def _validate_risk_management(self, strategy: StrategyTree) -> Dict[str, List[str]]:
        """Validate risk management controls"""
        errors = []
        warnings = []
        recommendations = []
        risk_factors = []
        
        min_stop_loss = self.config['min_stop_loss_pct']
        max_stop_loss = self.config['max_stop_loss_pct']
        require_risk_controls = self.config['required_risk_controls']
        
        buy_nodes_without_stops = []
        
        for node in strategy.get_all_nodes():
            # Check for stop losses on buy actions
            has_buy_action = any(action.type == ActionType.BUY for action in node.actions)
            
            if has_buy_action:
                if node.stop_loss_pct is None and require_risk_controls:
                    buy_nodes_without_stops.append(node.id)
                elif node.stop_loss_pct is not None:
                    if node.stop_loss_pct < min_stop_loss:
                        warnings.append(f"Stop loss {node.stop_loss_pct}% below minimum {min_stop_loss}% in node {node.id}")
                    elif node.stop_loss_pct > max_stop_loss:
                        warnings.append(f"Stop loss {node.stop_loss_pct}% above maximum {max_stop_loss}% in node {node.id}")
                
                # Check take profit levels
                if node.take_profit_pct is not None:
                    if node.stop_loss_pct is not None:
                        risk_reward_ratio = node.take_profit_pct / node.stop_loss_pct
                        if risk_reward_ratio < 1.5:
                            warnings.append(f"Poor risk/reward ratio {risk_reward_ratio:.1f} in node {node.id}")
                        elif risk_reward_ratio > 5:
                            warnings.append(f"Unrealistic take profit level in node {node.id}")
        
        if buy_nodes_without_stops:
            errors.append(f"Nodes without stop losses: {buy_nodes_without_stops}")
            recommendations.append("Add stop loss levels to all buy signals")
        
        # Check portfolio-level risk
        max_portfolio_risk = self.config['max_portfolio_risk_pct']
        if strategy.max_portfolio_risk > max_portfolio_risk:
            errors.append(f"Portfolio risk {strategy.max_portfolio_risk}% exceeds maximum {max_portfolio_risk}%")
        
        # Check execution limits
        max_daily_trades = self.config['max_daily_executions']
        for node in strategy.get_all_nodes():
            if node.max_daily_executions and node.max_daily_executions > max_daily_trades:
                warnings.append(f"High daily execution limit {node.max_daily_executions} in node {node.id}")
        
        return {
            'errors': errors,
            'warnings': warnings,
            'recommendations': recommendations,
            'risk_factors': risk_factors
        }
    
    def _validate_technical_indicators(self, strategy: StrategyTree) -> Dict[str, List[str]]:
        """Validate technical indicator parameters"""
        errors = []
        warnings = []
        recommendations = []
        
        for node in strategy.get_all_nodes():
            for condition in node.conditions:
                if condition.type == ConditionType.TECHNICAL_INDICATOR:
                    indicator = condition.indicator
                    params = condition.parameters
                    
                    # Validate specific indicators
                    if indicator in [IndicatorType.SMA, IndicatorType.EMA]:
                        period = params.get('period')
                        if not period or period < 1:
                            errors.append(f"Invalid period {period} for {indicator} in node {node.id}")
                        elif period > 200:
                            warnings.append(f"Very long period {period} for {indicator} in node {node.id}")
                    
                    elif indicator == IndicatorType.RSI:
                        period = params.get('period', 14)
                        if period < 2:
                            errors.append(f"RSI period {period} too small in node {node.id}")
                        elif period > 50:
                            warnings.append(f"RSI period {period} unusually large in node {node.id}")
                        
                        # Validate RSI levels
                        if condition.operator.value in ['<', '<=']:
                            if condition.value < 10 or condition.value > 40:
                                warnings.append(f"Unusual RSI oversold level {condition.value} in node {node.id}")
                        elif condition.operator.value in ['>', '>=']:
                            if condition.value < 60 or condition.value > 90:
                                warnings.append(f"Unusual RSI overbought level {condition.value} in node {node.id}")
                    
                    elif indicator == IndicatorType.MACD:
                        fast_period = params.get('fast_period', 12)
                        slow_period = params.get('slow_period', 26)
                        signal_period = params.get('signal_period', 9)
                        
                        if fast_period >= slow_period:
                            errors.append(f"MACD fast period {fast_period} must be less than slow period {slow_period}")
                    
                    elif indicator == IndicatorType.BOLLINGER_BANDS:
                        period = params.get('period', 20)
                        std_dev = params.get('std_dev', 2)
                        
                        if std_dev < 1 or std_dev > 3:
                            warnings.append(f"Unusual Bollinger Band std_dev {std_dev} in node {node.id}")
        
        return {
            'errors': errors,
            'warnings': warnings,
            'recommendations': recommendations
        }
    
    def _validate_execution_logic(self, strategy: StrategyTree) -> Dict[str, List[str]]:
        """Validate execution logic and conditions"""
        errors = []
        warnings = []
        recommendations = []
        
        # Check for conflicting conditions
        for node in strategy.get_all_nodes():
            if len(node.conditions) > 1:
                # Look for contradictory conditions
                for i, cond1 in enumerate(node.conditions):
                    for j, cond2 in enumerate(node.conditions[i+1:], i+1):
                        if self._conditions_conflict(cond1, cond2):
                            warnings.append(f"Potentially conflicting conditions in node {node.id}")
            
            # Check for missing exit conditions
            has_buy = any(action.type == ActionType.BUY for action in node.actions)
            has_sell = any(action.type == ActionType.SELL for action in node.actions)
            
            if has_buy and not has_sell and not node.stop_loss_pct and not node.take_profit_pct:
                warnings.append(f"Buy action in node {node.id} lacks clear exit strategy")
                recommendations.append(f"Add exit conditions or stop loss to node {node.id}")
        
        # Check for unreachable nodes
        reachable_nodes = set()
        self._mark_reachable_nodes(strategy.root_node, reachable_nodes)
        
        all_nodes = strategy.get_all_nodes()
        unreachable = [node.id for node in all_nodes if node.id not in reachable_nodes]
        
        if unreachable:
            warnings.append(f"Unreachable nodes detected: {unreachable}")
        
        return {
            'errors': errors,
            'warnings': warnings,
            'recommendations': recommendations
        }
    
    def _validate_portfolio_limits(self, strategy: StrategyTree) -> Dict[str, List[str]]:
        """Validate portfolio-level limits"""
        errors = []
        warnings = []
        risk_factors = []
        
        # Check symbol count
        symbols = strategy.get_symbols()
        max_symbols = self.config['max_symbols_per_strategy']
        
        if len(symbols) > max_symbols:
            errors.append(f"Too many symbols ({len(symbols)}) exceeds limit of {max_symbols}")
        
        # Check for blacklisted symbols
        blacklisted = self.config['blacklisted_symbols']
        forbidden_symbols = [s for s in symbols if s in blacklisted]
        
        if forbidden_symbols:
            errors.append(f"Blacklisted symbols detected: {forbidden_symbols}")
        
        # Check correlation limits
        if strategy.correlation_threshold > self.config['max_correlation']:
            warnings.append(f"High correlation threshold {strategy.correlation_threshold}")
        
        return {
            'errors': errors,
            'warnings': warnings,
            'risk_factors': risk_factors
        }
    
    def _validate_symbols(self, strategy: StrategyTree) -> Dict[str, List[str]]:
        """Validate symbols and timeframes"""
        warnings = []
        recommendations = []
        
        # Validate timeframe
        if strategy.timeframe not in self.config['allowed_timeframes']:
            warnings.append(f"Unusual timeframe: {strategy.timeframe}")
        
        # Check lookback period
        if strategy.lookback_period < self.config['min_lookback_days']:
            warnings.append(f"Short lookback period: {strategy.lookback_period} days")
            recommendations.append("Increase lookback period for more reliable backtesting")
        
        return {
            'warnings': warnings,
            'recommendations': recommendations
        }
    
    def _validate_performance_expectations(self, strategy: StrategyTree) -> Dict[str, List[str]]:
        """Check for realistic performance expectations"""
        warnings = []
        recommendations = []
        
        # Check for unrealistic take profit levels
        for node in strategy.get_all_nodes():
            if node.take_profit_pct and node.take_profit_pct > 50:
                warnings.append(f"Very high take profit {node.take_profit_pct}% in node {node.id}")
                recommendations.append("Consider more conservative profit targets")
        
        return {
            'warnings': warnings,
            'recommendations': recommendations
        }
    
    def _conditions_conflict(self, cond1: Condition, cond2: Condition) -> bool:
        """Check if two conditions conflict with each other"""
        # Simple conflict detection - can be expanded
        if (cond1.type == cond2.type and 
            cond1.indicator == cond2.indicator and
            cond1.parameters == cond2.parameters):
            
            # Same indicator, check for contradictory operators
            if (cond1.operator.value == '>' and cond2.operator.value == '<' and
                cond1.value >= cond2.value):
                return True
        
        return False
    
    def _mark_reachable_nodes(self, node: StrategyNode, reachable: set):
        """Mark nodes reachable from the given node"""
        reachable.add(node.id)
        for child in node.children:
            self._mark_reachable_nodes(child, reachable)
    
    def _calculate_risk_score(self, strategy: StrategyTree, risk_factors: List[str]) -> float:
        """Calculate overall risk score for the strategy"""
        risk_score = 0.0
        
        # Base risk from strategy risk level
        risk_level_scores = {'low': 10, 'medium': 30, 'high': 60}
        risk_score += risk_level_scores.get(strategy.risk_level, 30)
        
        # Position sizing risk
        symbols = strategy.get_symbols()
        if len(symbols) < 3:
            risk_score += 10  # Concentration risk
        
        # Risk management
        all_nodes = strategy.get_all_nodes()
        buy_nodes = [n for n in all_nodes if any(a.type == ActionType.BUY for a in n.actions)]
        nodes_without_stops = [n for n in buy_nodes if n.stop_loss_pct is None]
        
        if nodes_without_stops:
            risk_score += len(nodes_without_stops) * 15
        
        # Leverage and exposure
        total_exposure = 0
        for node in all_nodes:
            for action in node.actions:
                if action.type == ActionType.BUY and action.sizing_method == PositionSizingMethod.PERCENTAGE:
                    total_exposure += action.amount
        
        if total_exposure > 80:
            risk_score += (total_exposure - 80) * 0.5
        
        # Additional risk factors
        risk_score += len(risk_factors) * 3
        
        return min(risk_score, 100.0)  # Cap at 100

# Utility functions
def validate_strategy_tree(strategy: StrategyTree, config: Optional[Dict[str, Any]] = None) -> ValidationResult:
    """Convenience function to validate a strategy tree"""
    validator = StrategyTreeValidator(config)
    return validator.validate(strategy)

def is_strategy_safe(strategy: StrategyTree) -> bool:
    """Quick check if strategy passes basic safety validations"""
    result = validate_strategy_tree(strategy)
    return result.is_valid and result.risk_score < 70