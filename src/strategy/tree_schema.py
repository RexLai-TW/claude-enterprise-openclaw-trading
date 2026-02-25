"""
Strategy Tree Schema

Defines the JSON schema and data structures for deterministic trading strategies.
Strategy trees are if-else rule trees that can be executed deterministically.
"""

from typing import Dict, Any, List, Optional, Union, Literal
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
import json
import yaml
from pydantic import BaseModel, Field, validator

# Enums for strategy components
class ConditionType(str, Enum):
    """Types of conditions that can be evaluated"""
    PRICE_COMPARISON = "price_comparison"
    TECHNICAL_INDICATOR = "technical_indicator"
    VOLUME_COMPARISON = "volume_comparison"
    TIME_CONDITION = "time_condition"
    SENTIMENT_CONDITION = "sentiment_condition"
    PORTFOLIO_CONDITION = "portfolio_condition"

class ActionType(str, Enum):
    """Types of actions that can be executed"""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    REBALANCE = "rebalance"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"

class OperatorType(str, Enum):
    """Logical operators for conditions"""
    GREATER_THAN = ">"
    LESS_THAN = "<"
    GREATER_EQUAL = ">="
    LESS_EQUAL = "<="
    EQUALS = "=="
    NOT_EQUALS = "!="
    AND = "and"
    OR = "or"

class IndicatorType(str, Enum):
    """Technical indicators available"""
    SMA = "sma"  # Simple Moving Average
    EMA = "ema"  # Exponential Moving Average
    RSI = "rsi"  # Relative Strength Index
    MACD = "macd"  # MACD
    BOLLINGER_BANDS = "bollinger_bands"
    STOCHASTIC = "stochastic"
    ATR = "atr"  # Average True Range
    ADX = "adx"  # Average Directional Index
    VIX = "vix"  # Volatility Index
    VOLUME_SMA = "volume_sma"

class PositionSizingMethod(str, Enum):
    """Position sizing methods"""
    FIXED_AMOUNT = "fixed_amount"
    PERCENTAGE = "percentage"
    RISK_BASED = "risk_based"
    VOLATILITY_ADJUSTED = "volatility_adjusted"

# Pydantic models for validation
class Condition(BaseModel):
    """
    A single condition in the strategy tree
    """
    type: ConditionType
    indicator: Optional[IndicatorType] = None
    operator: OperatorType
    value: Union[float, int, str]
    parameters: Dict[str, Any] = Field(default_factory=dict)
    description: str = ""
    
    class Config:
        use_enum_values = True
    
    @validator('parameters')
    def validate_parameters(cls, v, values):
        """Validate parameters based on condition type"""
        condition_type = values.get('type')
        
        if condition_type == ConditionType.TECHNICAL_INDICATOR:
            indicator = values.get('indicator')
            if indicator in [IndicatorType.SMA, IndicatorType.EMA]:
                if 'period' not in v:
                    raise ValueError(f"{indicator} requires 'period' parameter")
            elif indicator == IndicatorType.RSI:
                if 'period' not in v:
                    v['period'] = 14  # Default RSI period
            elif indicator == IndicatorType.BOLLINGER_BANDS:
                if 'period' not in v:
                    v['period'] = 20
                if 'std_dev' not in v:
                    v['std_dev'] = 2
        
        return v

class Action(BaseModel):
    """
    An action to be executed when conditions are met
    """
    type: ActionType
    symbol: str
    sizing_method: PositionSizingMethod = PositionSizingMethod.PERCENTAGE
    amount: Union[float, int]  # Amount based on sizing method
    parameters: Dict[str, Any] = Field(default_factory=dict)
    description: str = ""
    priority: int = 1  # Execution priority (1 = highest)
    
    class Config:
        use_enum_values = True
    
    @validator('amount')
    def validate_amount(cls, v, values):
        """Validate amount based on sizing method"""
        sizing_method = values.get('sizing_method')
        
        if sizing_method == PositionSizingMethod.PERCENTAGE:
            if v <= 0 or v > 100:
                raise ValueError("Percentage amount must be between 0 and 100")
        elif sizing_method == PositionSizingMethod.FIXED_AMOUNT:
            if v <= 0:
                raise ValueError("Fixed amount must be positive")
        
        return v

class StrategyNode(BaseModel):
    """
    A node in the strategy tree - can contain conditions and actions
    """
    id: str
    name: str
    description: str = ""
    conditions: List[Condition] = Field(default_factory=list)
    condition_logic: str = "and"  # "and" or "or" for multiple conditions
    actions: List[Action] = Field(default_factory=list)
    children: List['StrategyNode'] = Field(default_factory=list)
    parent_id: Optional[str] = None
    
    # Risk management
    stop_loss_pct: Optional[float] = None
    take_profit_pct: Optional[float] = None
    max_position_size: Optional[float] = None
    
    # Execution controls
    enabled: bool = True
    cooldown_period: Optional[int] = None  # Minutes between executions
    max_daily_executions: Optional[int] = None
    
    class Config:
        use_enum_values = True
    
    @validator('condition_logic')
    def validate_condition_logic(cls, v):
        if v not in ['and', 'or']:
            raise ValueError("condition_logic must be 'and' or 'or'")
        return v

# Update forward reference
StrategyNode.update_forward_refs()

class StrategyTree(BaseModel):
    """
    Complete strategy tree with metadata
    """
    id: str
    name: str
    description: str
    version: str = "1.0"
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    
    # Strategy metadata
    author: str = "Claude Enterprise Trading"
    tags: List[str] = Field(default_factory=list)
    risk_level: Literal["low", "medium", "high"] = "medium"
    
    # Root node of the strategy tree
    root_node: StrategyNode
    
    # Strategy-wide settings
    symbols: List[str] = Field(default_factory=list)
    timeframe: str = "1d"  # Default timeframe for data
    lookback_period: int = 252  # Trading days for historical data
    
    # Portfolio constraints
    max_positions: int = 10
    max_portfolio_risk: float = 0.02  # 2% portfolio risk per trade
    correlation_threshold: float = 0.7  # Max correlation between positions
    
    # Execution settings
    execution_mode: Literal["paper", "live"] = "paper"
    slippage: float = 0.001  # 0.1% slippage assumption
    commission: float = 0.001  # 0.1% commission
    
    # Backtesting configuration
    backtest_start: Optional[datetime] = None
    backtest_end: Optional[datetime] = None
    benchmark: str = "SPY"
    
    class Config:
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return self.dict()
    
    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string"""
        return self.json(indent=indent)
    
    def to_yaml(self) -> str:
        """Convert to YAML string"""
        data = self.dict()
        return yaml.dump(data, default_flow_style=False, indent=2)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StrategyTree':
        """Create from dictionary"""
        return cls(**data)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'StrategyTree':
        """Create from JSON string"""
        data = json.loads(json_str)
        return cls.from_dict(data)
    
    @classmethod
    def from_yaml(cls, yaml_str: str) -> 'StrategyTree':
        """Create from YAML string"""
        data = yaml.safe_load(yaml_str)
        return cls.from_dict(data)
    
    def get_node_by_id(self, node_id: str) -> Optional[StrategyNode]:
        """Find a node by its ID"""
        def search_node(node: StrategyNode) -> Optional[StrategyNode]:
            if node.id == node_id:
                return node
            for child in node.children:
                result = search_node(child)
                if result:
                    return result
            return None
        
        return search_node(self.root_node)
    
    def get_all_nodes(self) -> List[StrategyNode]:
        """Get all nodes in the tree"""
        nodes = []
        
        def collect_nodes(node: StrategyNode):
            nodes.append(node)
            for child in node.children:
                collect_nodes(child)
        
        collect_nodes(self.root_node)
        return nodes
    
    def get_symbols(self) -> List[str]:
        """Get all symbols used in the strategy"""
        symbols = set(self.symbols)
        
        for node in self.get_all_nodes():
            for action in node.actions:
                symbols.add(action.symbol)
        
        return list(symbols)
    
    def validate_tree_structure(self) -> List[str]:
        """Validate the tree structure and return any errors"""
        errors = []
        
        # Check for circular references
        visited = set()
        
        def check_cycles(node: StrategyNode, path: set):
            if node.id in path:
                errors.append(f"Circular reference detected: {node.id}")
                return
            
            if node.id in visited:
                return
            
            visited.add(node.id)
            path.add(node.id)
            
            for child in node.children:
                check_cycles(child, path.copy())
        
        check_cycles(self.root_node, set())
        
        # Check for orphaned nodes (nodes with parent_id that don't exist)
        all_nodes = self.get_all_nodes()
        node_ids = {node.id for node in all_nodes}
        
        for node in all_nodes:
            if node.parent_id and node.parent_id not in node_ids:
                errors.append(f"Node {node.id} has invalid parent_id: {node.parent_id}")
        
        return errors

# Helper functions for creating common strategy patterns
def create_simple_moving_average_strategy(
    symbol: str,
    short_period: int = 10,
    long_period: int = 50,
    position_size: float = 25.0
) -> StrategyTree:
    """Create a simple moving average crossover strategy"""
    
    # Buy condition: short MA > long MA
    buy_condition = Condition(
        type=ConditionType.TECHNICAL_INDICATOR,
        indicator=IndicatorType.SMA,
        operator=OperatorType.GREATER_THAN,
        value=0,  # Will be compared to long MA
        parameters={
            "short_period": short_period,
            "long_period": long_period,
            "comparison_type": "ma_crossover"
        },
        description=f"SMA({short_period}) > SMA({long_period})"
    )
    
    buy_action = Action(
        type=ActionType.BUY,
        symbol=symbol,
        sizing_method=PositionSizingMethod.PERCENTAGE,
        amount=position_size,
        description=f"Buy {position_size}% of portfolio in {symbol}"
    )
    
    # Sell condition: short MA < long MA  
    sell_condition = Condition(
        type=ConditionType.TECHNICAL_INDICATOR,
        indicator=IndicatorType.SMA,
        operator=OperatorType.LESS_THAN,
        value=0,
        parameters={
            "short_period": short_period,
            "long_period": long_period,
            "comparison_type": "ma_crossover"
        },
        description=f"SMA({short_period}) < SMA({long_period})"
    )
    
    sell_action = Action(
        type=ActionType.SELL,
        symbol=symbol,
        sizing_method=PositionSizingMethod.PERCENTAGE,
        amount=100,  # Sell entire position
        description=f"Sell entire {symbol} position"
    )
    
    # Create strategy nodes
    buy_node = StrategyNode(
        id="buy_signal",
        name="Buy Signal",
        description="Buy when short MA crosses above long MA",
        conditions=[buy_condition],
        actions=[buy_action],
        stop_loss_pct=5.0,  # 5% stop loss
        take_profit_pct=15.0  # 15% take profit
    )
    
    sell_node = StrategyNode(
        id="sell_signal", 
        name="Sell Signal",
        description="Sell when short MA crosses below long MA",
        conditions=[sell_condition],
        actions=[sell_action]
    )
    
    root_node = StrategyNode(
        id="root",
        name="MA Crossover Strategy",
        description=f"Simple moving average crossover strategy for {symbol}",
        children=[buy_node, sell_node]
    )
    
    return StrategyTree(
        id=f"ma_crossover_{symbol}_{short_period}_{long_period}",
        name=f"MA Crossover - {symbol}",
        description=f"Moving average crossover strategy for {symbol} using {short_period}/{long_period} periods",
        root_node=root_node,
        symbols=[symbol],
        tags=["moving_average", "crossover", "trend_following"],
        risk_level="medium"
    )

def create_rsi_mean_reversion_strategy(
    symbol: str,
    rsi_period: int = 14,
    oversold_level: int = 30,
    overbought_level: int = 70,
    position_size: float = 20.0
) -> StrategyTree:
    """Create an RSI mean reversion strategy"""
    
    # Buy condition: RSI < oversold level
    buy_condition = Condition(
        type=ConditionType.TECHNICAL_INDICATOR,
        indicator=IndicatorType.RSI,
        operator=OperatorType.LESS_THAN,
        value=oversold_level,
        parameters={"period": rsi_period},
        description=f"RSI({rsi_period}) < {oversold_level}"
    )
    
    buy_action = Action(
        type=ActionType.BUY,
        symbol=symbol,
        sizing_method=PositionSizingMethod.PERCENTAGE,
        amount=position_size,
        description=f"Buy {position_size}% when oversold"
    )
    
    # Sell condition: RSI > overbought level
    sell_condition = Condition(
        type=ConditionType.TECHNICAL_INDICATOR,
        indicator=IndicatorType.RSI,
        operator=OperatorType.GREATER_THAN,
        value=overbought_level,
        parameters={"period": rsi_period},
        description=f"RSI({rsi_period}) > {overbought_level}"
    )
    
    sell_action = Action(
        type=ActionType.SELL,
        symbol=symbol,
        sizing_method=PositionSizingMethod.PERCENTAGE,
        amount=100,
        description=f"Sell entire position when overbought"
    )
    
    buy_node = StrategyNode(
        id="rsi_buy",
        name="RSI Oversold Buy",
        conditions=[buy_condition],
        actions=[buy_action],
        max_daily_executions=1
    )
    
    sell_node = StrategyNode(
        id="rsi_sell",
        name="RSI Overbought Sell", 
        conditions=[sell_condition],
        actions=[sell_action]
    )
    
    root_node = StrategyNode(
        id="root",
        name="RSI Mean Reversion",
        children=[buy_node, sell_node]
    )
    
    return StrategyTree(
        id=f"rsi_mean_reversion_{symbol}_{rsi_period}",
        name=f"RSI Mean Reversion - {symbol}",
        description=f"RSI-based mean reversion strategy for {symbol}",
        root_node=root_node,
        symbols=[symbol],
        tags=["rsi", "mean_reversion", "contrarian"],
        risk_level="medium"
    )