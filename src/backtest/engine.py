"""
Backtest Engine

Runs strategy trees against historical OHLCV data to evaluate performance.
Calculates key metrics: total return, Sharpe ratio, max drawdown, win rate.
Supports configurable slippage, fees, and realistic execution simulation.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
import pandas as pd
import numpy as np
from opentelemetry import trace

from ..strategy.tree_schema import StrategyTree, StrategyNode, Action, ActionType, PositionSizingMethod
from ..data.market_data import MarketDataInterface, create_market_data_connector

@dataclass
class BacktestConfig:
    """Configuration for backtesting"""
    start_date: datetime
    end_date: datetime
    initial_capital: float = 100000.0  # Starting capital
    slippage: float = 0.001  # 0.1% slippage per trade
    commission: float = 0.001  # 0.1% commission per trade
    interest_rate: float = 0.02  # Risk-free rate for Sharpe ratio
    max_positions: int = 10  # Maximum concurrent positions
    rebalance_threshold: float = 0.05  # 5% deviation to trigger rebalancing
    
    # Data configuration
    data_source: str = "yfinance"
    timeframe: str = "1d"
    
    # Risk management
    margin_requirement: float = 1.0  # 1.0 = no leverage, 0.5 = 2x leverage
    max_drawdown_halt: Optional[float] = None  # Halt trading if drawdown exceeds %
    
    # Execution settings
    execution_delay: int = 0  # Bars delay between signal and execution
    partial_fill_ratio: float = 1.0  # Ratio of order that gets filled (1.0 = 100%)

@dataclass 
class Position:
    """Represents a trading position"""
    symbol: str
    quantity: float
    entry_price: float
    entry_date: datetime
    entry_signal: str
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    unrealized_pnl: float = 0.0
    
    def update_unrealized_pnl(self, current_price: float) -> None:
        """Update unrealized P&L based on current price"""
        self.unrealized_pnl = (current_price - self.entry_price) * self.quantity

@dataclass
class Trade:
    """Represents a completed trade"""
    symbol: str
    entry_date: datetime
    exit_date: datetime
    entry_price: float
    exit_price: float
    quantity: float
    pnl: float
    pnl_pct: float
    entry_signal: str
    exit_signal: str
    commission_paid: float
    slippage_cost: float
    duration_days: int
    
    @property
    def is_winner(self) -> bool:
        return self.pnl > 0

@dataclass
class BacktestResult:
    """Results from a backtest run"""
    strategy_id: str
    start_date: datetime
    end_date: datetime
    initial_capital: float
    final_capital: float
    
    # Performance metrics
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    max_drawdown: float
    max_drawdown_duration_days: int
    win_rate: float
    profit_factor: float  # Gross profit / gross loss
    
    # Trade statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    avg_trade_pnl: float
    avg_winning_trade: float
    avg_losing_trade: float
    largest_win: float
    largest_loss: float
    
    # Time-based metrics
    daily_returns: List[float] = field(default_factory=list)
    monthly_returns: List[float] = field(default_factory=list)
    equity_curve: List[float] = field(default_factory=list)
    
    # Trading activity
    trades: List[Trade] = field(default_factory=list)
    trade_log: List[Dict[str, Any]] = field(default_factory=list)
    
    # Risk metrics
    volatility: float = 0.0
    var_95: float = 0.0  # Value at Risk (95%)
    calmar_ratio: float = 0.0  # Annual return / Max drawdown
    
    # Execution costs
    total_commission: float = 0.0
    total_slippage: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for serialization"""
        return {
            'strategy_id': self.strategy_id,
            'start_date': self.start_date.isoformat(),
            'end_date': self.end_date.isoformat(),
            'initial_capital': self.initial_capital,
            'final_capital': self.final_capital,
            'performance_metrics': {
                'total_return': self.total_return,
                'annualized_return': self.annualized_return,
                'sharpe_ratio': self.sharpe_ratio,
                'max_drawdown': self.max_drawdown,
                'max_drawdown_duration_days': self.max_drawdown_duration_days,
                'win_rate': self.win_rate,
                'profit_factor': self.profit_factor,
                'volatility': self.volatility,
                'var_95': self.var_95,
                'calmar_ratio': self.calmar_ratio,
            },
            'trade_statistics': {
                'total_trades': self.total_trades,
                'winning_trades': self.winning_trades, 
                'losing_trades': self.losing_trades,
                'avg_trade_pnl': self.avg_trade_pnl,
                'avg_winning_trade': self.avg_winning_trade,
                'avg_losing_trade': self.avg_losing_trade,
                'largest_win': self.largest_win,
                'largest_loss': self.largest_loss,
            },
            'execution_costs': {
                'total_commission': self.total_commission,
                'total_slippage': self.total_slippage,
            },
            'trade_count': len(self.trades)
        }

class BacktestEngine:
    """
    Comprehensive backtesting engine for strategy trees
    
    Features:
    - Realistic execution simulation with slippage and commissions
    - Position management with stop losses and take profits
    - Portfolio-level risk management
    - Comprehensive performance analytics
    - Support for multiple symbols and timeframes
    """
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.logger = logging.getLogger('claude_trading.backtest')
        self.tracer = trace.get_tracer(__name__)
        
        # Initialize data connector
        data_config = {'rate_limit': 2000, 'demo_mode': True}
        self.data_connector = create_market_data_connector(
            config.data_source, data_config
        )
        
        # Portfolio state
        self.portfolio_value = config.initial_capital
        self.cash = config.initial_capital
        self.positions: Dict[str, Position] = {}
        self.closed_trades: List[Trade] = []
        self.daily_portfolio_values: List[Tuple[datetime, float]] = []
        
        # Performance tracking
        self.peak_portfolio_value = config.initial_capital
        self.max_drawdown_amount = 0.0
        self.drawdown_start_date = None
        self.max_drawdown_duration = 0
        
        # Execution tracking
        self.trade_log: List[Dict[str, Any]] = []
        self.signal_log: List[Dict[str, Any]] = []
        
    async def run_backtest(self, strategy: StrategyTree) -> BacktestResult:
        """
        Run complete backtest for a strategy tree
        """
        with self.tracer.start_as_current_span("run_backtest") as span:
            span.set_attribute("strategy_id", strategy.id)
            span.set_attribute("start_date", self.config.start_date.isoformat())
            span.set_attribute("end_date", self.config.end_date.isoformat())
            
            self.logger.info(f"Starting backtest for strategy: {strategy.name}")
            self.logger.info(f"Period: {self.config.start_date} to {self.config.end_date}")
            self.logger.info(f"Initial capital: ${self.config.initial_capital:,.2f}")
            
            # Reset portfolio state
            self._reset_portfolio()
            
            # Get all required symbols
            symbols = strategy.get_symbols()
            self.logger.info(f"Trading symbols: {symbols}")
            
            # Fetch historical data for all symbols
            historical_data = await self._fetch_historical_data(symbols)
            
            # Calculate technical indicators for strategy evaluation
            indicator_data = self._calculate_indicators(historical_data, strategy)
            
            # Get common date range across all symbols
            date_range = self._get_common_date_range(historical_data)
            
            # Run day-by-day simulation
            for current_date in date_range:
                await self._simulate_trading_day(
                    strategy, current_date, historical_data, indicator_data
                )
                
                # Update portfolio value and tracking
                self._update_portfolio_metrics(current_date, historical_data)
                
                # Check halt conditions
                if self._should_halt_trading():
                    self.logger.warning(f"Trading halted due to risk limits on {current_date}")
                    break
            
            # Close any remaining positions
            final_date = date_range[-1] if date_range else self.config.end_date
            await self._close_all_positions(final_date, historical_data)
            
            # Calculate final results
            result = self._calculate_backtest_results(strategy)
            
            span.set_attribute("total_return", result.total_return)
            span.set_attribute("sharpe_ratio", result.sharpe_ratio)
            span.set_attribute("max_drawdown", result.max_drawdown)
            span.set_attribute("total_trades", result.total_trades)
            
            self.logger.info(f"Backtest completed - Total Return: {result.total_return:.2%}")
            self.logger.info(f"Sharpe Ratio: {result.sharpe_ratio:.2f}, Max Drawdown: {result.max_drawdown:.2%}")
            
            return result
    
    async def _fetch_historical_data(self, symbols: List[str]) -> Dict[str, pd.DataFrame]:
        """Fetch historical data for all symbols"""
        historical_data = {}
        
        for symbol in symbols:
            try:
                self.logger.debug(f"Fetching data for {symbol}")
                
                # Calculate period string for yfinance
                period_days = (self.config.end_date - self.config.start_date).days
                if period_days <= 5:
                    period = "5d"
                elif period_days <= 30:
                    period = "1mo"
                elif period_days <= 90:
                    period = "3mo"
                elif period_days <= 365:
                    period = "1y"
                else:
                    period = "2y"
                
                data = await self.data_connector.get_market_data(
                    symbol, period, self.config.timeframe
                )
                
                # Convert to pandas DataFrame
                df = pd.DataFrame(data['data'])
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
                df.sort_index(inplace=True)
                
                # Filter to backtest date range
                mask = (df.index >= self.config.start_date) & (df.index <= self.config.end_date)
                df = df.loc[mask]
                
                if len(df) == 0:
                    self.logger.warning(f"No data available for {symbol} in date range")
                    continue
                
                historical_data[symbol] = df
                self.logger.debug(f"Loaded {len(df)} data points for {symbol}")
                
            except Exception as e:
                self.logger.error(f"Failed to fetch data for {symbol}: {e}")
                
        return historical_data
    
    def _calculate_indicators(self, historical_data: Dict[str, pd.DataFrame], strategy: StrategyTree) -> Dict[str, Dict[str, pd.Series]]:
        """Calculate technical indicators needed for strategy evaluation"""
        indicator_data = {}
        
        for symbol, df in historical_data.items():
            indicator_data[symbol] = {}
            
            # Calculate common indicators
            # Simple Moving Averages
            for period in [10, 20, 50, 100, 200]:
                indicator_data[symbol][f'sma_{period}'] = df['close'].rolling(window=period).mean()
            
            # Exponential Moving Averages  
            for period in [10, 20, 50]:
                indicator_data[symbol][f'ema_{period}'] = df['close'].ewm(span=period).mean()
            
            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            indicator_data[symbol]['rsi'] = 100 - (100 / (1 + rs))
            
            # Volume indicators
            indicator_data[symbol]['volume_sma_10'] = df['volume'].rolling(window=10).mean()
            indicator_data[symbol]['volume_sma_20'] = df['volume'].rolling(window=20).mean()
            
            # Volatility (for position sizing)
            indicator_data[symbol]['volatility_20'] = df['close'].pct_change().rolling(window=20).std()
            
            # Price-based indicators
            indicator_data[symbol]['returns'] = df['close'].pct_change()
            
        return indicator_data
    
    def _get_common_date_range(self, historical_data: Dict[str, pd.DataFrame]) -> List[datetime]:
        """Get date range common to all symbols"""
        if not historical_data:
            return []
        
        # Find intersection of all date ranges
        common_dates = None
        for symbol, df in historical_data.items():
            dates = set(df.index)
            if common_dates is None:
                common_dates = dates
            else:
                common_dates = common_dates.intersection(dates)
        
        if common_dates is None:
            return []
        
        return sorted(list(common_dates))
    
    async def _simulate_trading_day(
        self, 
        strategy: StrategyTree,
        current_date: datetime, 
        historical_data: Dict[str, pd.DataFrame],
        indicator_data: Dict[str, Dict[str, pd.Series]]
    ) -> None:
        """Simulate trading for a single day"""
        
        # Update unrealized P&L for existing positions
        self._update_position_values(current_date, historical_data)
        
        # Check stop losses and take profits
        await self._check_exit_conditions(current_date, historical_data)
        
        # Evaluate strategy tree for new signals
        signals = self._evaluate_strategy_tree(
            strategy, current_date, historical_data, indicator_data
        )
        
        # Execute signals
        for signal in signals:
            await self._execute_signal(signal, current_date, historical_data)
    
    def _evaluate_strategy_tree(
        self,
        strategy: StrategyTree,
        current_date: datetime,
        historical_data: Dict[str, pd.DataFrame], 
        indicator_data: Dict[str, Dict[str, pd.Series]]
    ) -> List[Dict[str, Any]]:
        """Evaluate strategy tree and generate trading signals"""
        signals = []
        
        def evaluate_node(node: StrategyNode) -> bool:
            """Recursively evaluate strategy node conditions"""
            if not node.conditions:
                return True  # No conditions = always true
            
            condition_results = []
            
            for condition in node.conditions:
                result = self._evaluate_condition(
                    condition, current_date, historical_data, indicator_data
                )
                condition_results.append(result)
            
            # Apply condition logic
            if node.condition_logic == "and":
                return all(condition_results)
            elif node.condition_logic == "or":
                return any(condition_results)
            else:
                return all(condition_results)  # Default to AND
        
        def process_node(node: StrategyNode):
            """Process a strategy node and its children"""
            if not node.enabled:
                return
                
            # Evaluate conditions
            if evaluate_node(node):
                # Conditions met, execute actions
                for action in node.actions:
                    signal = {
                        'action': action,
                        'node_id': node.id,
                        'date': current_date,
                        'reason': f"Node {node.id} conditions met"
                    }
                    signals.append(signal)
            
            # Process children nodes
            for child in node.children:
                process_node(child)
        
        # Start evaluation from root
        process_node(strategy.root_node)
        
        return signals
    
    def _evaluate_condition(
        self,
        condition,
        current_date: datetime,
        historical_data: Dict[str, pd.DataFrame],
        indicator_data: Dict[str, Dict[str, pd.Series]]
    ) -> bool:
        """Evaluate a single condition"""
        try:
            from ..strategy.tree_schema import ConditionType, IndicatorType
            
            if condition.type == ConditionType.TECHNICAL_INDICATOR:
                # Get the main symbol (first one if multiple)
                symbol = list(historical_data.keys())[0]
                
                if symbol not in indicator_data:
                    return False
                
                indicators = indicator_data[symbol]
                
                if condition.indicator == IndicatorType.RSI:
                    if current_date not in indicators['rsi'].index:
                        return False
                    
                    rsi_value = indicators['rsi'].loc[current_date]
                    if pd.isna(rsi_value):
                        return False
                    
                    return self._compare_values(rsi_value, condition.operator.value, condition.value)
                
                elif condition.indicator in [IndicatorType.SMA, IndicatorType.EMA]:
                    period = condition.parameters.get('period', 20)
                    indicator_name = f"{condition.indicator.value}_{period}"
                    
                    if indicator_name not in indicators:
                        return False
                    
                    if current_date not in indicators[indicator_name].index:
                        return False
                    
                    indicator_value = indicators[indicator_name].loc[current_date]
                    if pd.isna(indicator_value):
                        return False
                    
                    # Handle MA crossover comparison
                    if condition.parameters.get('comparison_type') == 'ma_crossover':
                        short_period = condition.parameters.get('short_period', 10) 
                        long_period = condition.parameters.get('long_period', 50)
                        
                        short_ma_name = f"sma_{short_period}"
                        long_ma_name = f"sma_{long_period}"
                        
                        if (short_ma_name not in indicators or 
                            long_ma_name not in indicators or
                            current_date not in indicators[short_ma_name].index):
                            return False
                        
                        short_ma = indicators[short_ma_name].loc[current_date]
                        long_ma = indicators[long_ma_name].loc[current_date]
                        
                        if pd.isna(short_ma) or pd.isna(long_ma):
                            return False
                        
                        return self._compare_values(short_ma, condition.operator.value, long_ma)
                    
                    # Regular comparison
                    return self._compare_values(indicator_value, condition.operator.value, condition.value)
            
            elif condition.type == ConditionType.PRICE_COMPARISON:
                # Get price data for comparison
                symbol = list(historical_data.keys())[0]  # Use first symbol
                
                if symbol not in historical_data or current_date not in historical_data[symbol].index:
                    return False
                
                current_price = historical_data[symbol].loc[current_date, 'close']
                return self._compare_values(current_price, condition.operator.value, condition.value)
            
            elif condition.type == ConditionType.VOLUME_COMPARISON:
                symbol = list(historical_data.keys())[0]
                
                if symbol not in historical_data or current_date not in historical_data[symbol].index:
                    return False
                
                current_volume = historical_data[symbol].loc[current_date, 'volume'] 
                return self._compare_values(current_volume, condition.operator.value, condition.value)
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error evaluating condition: {e}")
            return False
    
    def _compare_values(self, left: float, operator: str, right: float) -> bool:
        """Compare two values using the given operator"""
        if operator == '>':
            return left > right
        elif operator == '<':
            return left < right
        elif operator == '>=':
            return left >= right
        elif operator == '<=':
            return left <= right
        elif operator == '==':
            return abs(left - right) < 1e-6  # Float equality with tolerance
        elif operator == '!=':
            return abs(left - right) >= 1e-6
        else:
            return False
    
    async def _execute_signal(
        self,
        signal: Dict[str, Any],
        current_date: datetime,
        historical_data: Dict[str, pd.DataFrame]
    ) -> None:
        """Execute a trading signal"""
        action = signal['action']
        
        # Get current price
        symbol = action.symbol
        if symbol not in historical_data or current_date not in historical_data[symbol].index:
            self.logger.warning(f"No price data for {symbol} on {current_date}")
            return
        
        row = historical_data[symbol].loc[current_date]
        current_price = row['close']
        
        # Apply slippage
        if action.type == ActionType.BUY:
            execution_price = current_price * (1 + self.config.slippage)
        else:
            execution_price = current_price * (1 - self.config.slippage)
        
        # Calculate position size
        position_value = self._calculate_position_size(action, current_price)
        
        if position_value <= 0:
            return
        
        quantity = position_value / execution_price
        commission = position_value * self.config.commission
        slippage_cost = abs(execution_price - current_price) * quantity
        
        if action.type == ActionType.BUY:
            # Check if we have enough cash
            total_cost = position_value + commission
            
            if self.cash < total_cost:
                self.logger.warning(f"Insufficient cash for {symbol} buy order: need ${total_cost:.2f}, have ${self.cash:.2f}")
                return
            
            # Execute buy order
            if symbol in self.positions:
                # Add to existing position (average in)
                existing = self.positions[symbol]
                total_quantity = existing.quantity + quantity
                avg_price = ((existing.entry_price * existing.quantity) + 
                           (execution_price * quantity)) / total_quantity
                
                existing.quantity = total_quantity
                existing.entry_price = avg_price
            else:
                # Create new position
                self.positions[symbol] = Position(
                    symbol=symbol,
                    quantity=quantity,
                    entry_price=execution_price,
                    entry_date=current_date,
                    entry_signal=signal['node_id']
                )
            
            self.cash -= total_cost
            
            self.trade_log.append({
                'date': current_date,
                'action': 'BUY',
                'symbol': symbol,
                'quantity': quantity,
                'price': execution_price,
                'commission': commission,
                'slippage_cost': slippage_cost,
                'cash_remaining': self.cash
            })
            
        elif action.type == ActionType.SELL:
            # Check if we have the position
            if symbol not in self.positions:
                self.logger.warning(f"Cannot sell {symbol}: no position held")
                return
            
            position = self.positions[symbol]
            
            # Determine quantity to sell
            if action.sizing_method == PositionSizingMethod.PERCENTAGE:
                sell_quantity = position.quantity * (action.amount / 100.0)
            else:
                sell_quantity = min(quantity, position.quantity)
            
            # Execute sell order
            proceeds = sell_quantity * execution_price
            commission_paid = proceeds * self.config.commission
            net_proceeds = proceeds - commission_paid
            
            # Calculate trade P&L
            trade_pnl = (execution_price - position.entry_price) * sell_quantity
            trade_pnl_pct = trade_pnl / (position.entry_price * sell_quantity)
            
            # Create trade record
            trade = Trade(
                symbol=symbol,
                entry_date=position.entry_date,
                exit_date=current_date,
                entry_price=position.entry_price,
                exit_price=execution_price,
                quantity=sell_quantity,
                pnl=trade_pnl,
                pnl_pct=trade_pnl_pct,
                entry_signal=position.entry_signal,
                exit_signal=signal['node_id'],
                commission_paid=commission + commission_paid,
                slippage_cost=slippage_cost,
                duration_days=(current_date - position.entry_date).days
            )
            
            self.closed_trades.append(trade)
            self.cash += net_proceeds
            
            # Update or close position
            if sell_quantity >= position.quantity:
                # Close entire position
                del self.positions[symbol]
            else:
                # Partial sell
                position.quantity -= sell_quantity
            
            self.trade_log.append({
                'date': current_date,
                'action': 'SELL',
                'symbol': symbol,
                'quantity': sell_quantity,
                'price': execution_price,
                'pnl': trade_pnl,
                'commission': commission_paid,
                'cash_remaining': self.cash
            })
    
    def _calculate_position_size(self, action: Action, current_price: float) -> float:
        """Calculate position size based on action parameters"""
        if action.sizing_method == PositionSizingMethod.PERCENTAGE:
            # Percentage of current portfolio value
            return self.portfolio_value * (action.amount / 100.0)
        
        elif action.sizing_method == PositionSizingMethod.FIXED_AMOUNT:
            return action.amount
        
        elif action.sizing_method == PositionSizingMethod.RISK_BASED:
            # Risk-based sizing (not implemented in detail)
            risk_amount = self.portfolio_value * 0.02  # 2% risk
            return risk_amount
        
        else:
            return 0.0
    
    def _update_position_values(self, current_date: datetime, historical_data: Dict[str, pd.DataFrame]) -> None:
        """Update unrealized P&L for all positions"""
        for symbol, position in self.positions.items():
            if symbol in historical_data and current_date in historical_data[symbol].index:
                current_price = historical_data[symbol].loc[current_date, 'close']
                position.update_unrealized_pnl(current_price)
    
    async def _check_exit_conditions(self, current_date: datetime, historical_data: Dict[str, pd.DataFrame]) -> None:
        """Check stop loss and take profit conditions"""
        positions_to_close = []
        
        for symbol, position in self.positions.items():
            if symbol not in historical_data or current_date not in historical_data[symbol].index:
                continue
            
            current_price = historical_data[symbol].loc[current_date, 'close']
            price_change_pct = (current_price - position.entry_price) / position.entry_price
            
            should_exit = False
            exit_reason = ""
            
            # Check stop loss
            if position.stop_loss and current_price <= position.stop_loss:
                should_exit = True
                exit_reason = "stop_loss"
            
            # Check take profit
            elif position.take_profit and current_price >= position.take_profit:
                should_exit = True
                exit_reason = "take_profit"
            
            if should_exit:
                positions_to_close.append((symbol, exit_reason))
        
        # Execute exit orders
        for symbol, exit_reason in positions_to_close:
            await self._close_position(symbol, current_date, historical_data, exit_reason)
    
    async def _close_position(
        self, 
        symbol: str, 
        exit_date: datetime, 
        historical_data: Dict[str, pd.DataFrame],
        exit_reason: str
    ) -> None:
        """Close a specific position"""
        if symbol not in self.positions:
            return
        
        position = self.positions[symbol]
        current_price = historical_data[symbol].loc[exit_date, 'close']
        
        # Apply slippage for sell order
        execution_price = current_price * (1 - self.config.slippage)
        
        proceeds = position.quantity * execution_price
        commission = proceeds * self.config.commission
        net_proceeds = proceeds - commission
        
        # Calculate trade metrics
        trade_pnl = (execution_price - position.entry_price) * position.quantity
        trade_pnl_pct = trade_pnl / (position.entry_price * position.quantity)
        
        trade = Trade(
            symbol=symbol,
            entry_date=position.entry_date,
            exit_date=exit_date,
            entry_price=position.entry_price,
            exit_price=execution_price,
            quantity=position.quantity,
            pnl=trade_pnl,
            pnl_pct=trade_pnl_pct,
            entry_signal=position.entry_signal,
            exit_signal=exit_reason,
            commission_paid=commission,
            slippage_cost=abs(execution_price - current_price) * position.quantity,
            duration_days=(exit_date - position.entry_date).days
        )
        
        self.closed_trades.append(trade)
        self.cash += net_proceeds
        del self.positions[symbol]
        
        self.trade_log.append({
            'date': exit_date,
            'action': f'SELL ({exit_reason})',
            'symbol': symbol,
            'quantity': position.quantity,
            'price': execution_price,
            'pnl': trade_pnl,
            'commission': commission
        })
    
    async def _close_all_positions(self, final_date: datetime, historical_data: Dict[str, pd.DataFrame]) -> None:
        """Close all remaining positions at the end of backtest"""
        symbols_to_close = list(self.positions.keys())
        
        for symbol in symbols_to_close:
            await self._close_position(symbol, final_date, historical_data, "backtest_end")
    
    def _update_portfolio_metrics(self, current_date: datetime, historical_data: Dict[str, pd.DataFrame]) -> None:
        """Update portfolio value and drawdown metrics"""
        # Calculate total portfolio value
        position_value = 0.0
        
        for symbol, position in self.positions.items():
            if symbol in historical_data and current_date in historical_data[symbol].index:
                current_price = historical_data[symbol].loc[current_date, 'close']
                position_value += position.quantity * current_price
        
        self.portfolio_value = self.cash + position_value
        self.daily_portfolio_values.append((current_date, self.portfolio_value))
        
        # Update drawdown tracking
        if self.portfolio_value > self.peak_portfolio_value:
            self.peak_portfolio_value = self.portfolio_value
            self.drawdown_start_date = None
        else:
            drawdown_amount = self.peak_portfolio_value - self.portfolio_value
            if drawdown_amount > self.max_drawdown_amount:
                self.max_drawdown_amount = drawdown_amount
                if self.drawdown_start_date is None:
                    self.drawdown_start_date = current_date
    
    def _should_halt_trading(self) -> bool:
        """Check if trading should be halted due to risk limits"""
        if self.config.max_drawdown_halt is None:
            return False
        
        current_drawdown_pct = self.max_drawdown_amount / self.peak_portfolio_value
        return current_drawdown_pct >= self.config.max_drawdown_halt
    
    def _calculate_backtest_results(self, strategy: StrategyTree) -> BacktestResult:
        """Calculate comprehensive backtest results"""
        
        # Basic performance metrics
        total_return = (self.portfolio_value - self.config.initial_capital) / self.config.initial_capital
        
        # Annualized return
        days = (self.config.end_date - self.config.start_date).days
        years = max(days / 365.25, 1/365.25)  # Minimum 1 day
        annualized_return = (1 + total_return) ** (1 / years) - 1
        
        # Daily returns for Sharpe ratio
        daily_values = [value for date, value in self.daily_portfolio_values]
        if len(daily_values) > 1:
            daily_returns = [(daily_values[i] - daily_values[i-1]) / daily_values[i-1] 
                           for i in range(1, len(daily_values))]
        else:
            daily_returns = [0.0]
        
        # Calculate Sharpe ratio
        if len(daily_returns) > 1:
            returns_std = np.std(daily_returns)
            if returns_std > 0:
                daily_excess_return = np.mean(daily_returns) - (self.config.interest_rate / 365)
                sharpe_ratio = (daily_excess_return * np.sqrt(365)) / returns_std
            else:
                sharpe_ratio = 0.0
        else:
            sharpe_ratio = 0.0
        
        # Max drawdown percentage
        max_drawdown_pct = self.max_drawdown_amount / self.peak_portfolio_value if self.peak_portfolio_value > 0 else 0.0
        
        # Trade statistics
        winning_trades = [t for t in self.closed_trades if t.is_winner]
        losing_trades = [t for t in self.closed_trades if not t.is_winner]
        
        total_trades = len(self.closed_trades)
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0.0
        
        # Profit factor (gross profit / gross loss)
        gross_profit = sum(t.pnl for t in winning_trades)
        gross_loss = abs(sum(t.pnl for t in losing_trades))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Trade P&L statistics
        avg_trade_pnl = np.mean([t.pnl for t in self.closed_trades]) if self.closed_trades else 0.0
        avg_winning_trade = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0.0
        avg_losing_trade = np.mean([t.pnl for t in losing_trades]) if losing_trades else 0.0
        largest_win = max([t.pnl for t in winning_trades], default=0.0)
        largest_loss = min([t.pnl for t in losing_trades], default=0.0)
        
        # Risk metrics
        volatility = np.std(daily_returns) * np.sqrt(365) if len(daily_returns) > 1 else 0.0
        var_95 = np.percentile(daily_returns, 5) if len(daily_returns) > 1 else 0.0
        calmar_ratio = annualized_return / max_drawdown_pct if max_drawdown_pct > 0 else 0.0
        
        # Execution costs
        total_commission = sum(t.commission_paid for t in self.closed_trades)
        total_slippage = sum(t.slippage_cost for t in self.closed_trades)
        
        # Monthly returns (simplified)
        monthly_returns = []  # Could be calculated from daily values
        
        return BacktestResult(
            strategy_id=strategy.id,
            start_date=self.config.start_date,
            end_date=self.config.end_date,
            initial_capital=self.config.initial_capital,
            final_capital=self.portfolio_value,
            total_return=total_return,
            annualized_return=annualized_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown_pct,
            max_drawdown_duration_days=self.max_drawdown_duration,
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_trades=total_trades,
            winning_trades=len(winning_trades),
            losing_trades=len(losing_trades),
            avg_trade_pnl=avg_trade_pnl,
            avg_winning_trade=avg_winning_trade,
            avg_losing_trade=avg_losing_trade,
            largest_win=largest_win,
            largest_loss=largest_loss,
            daily_returns=daily_returns,
            monthly_returns=monthly_returns,
            equity_curve=daily_values,
            trades=self.closed_trades,
            trade_log=self.trade_log,
            volatility=volatility,
            var_95=var_95,
            calmar_ratio=calmar_ratio,
            total_commission=total_commission,
            total_slippage=total_slippage
        )
    
    def _reset_portfolio(self) -> None:
        """Reset portfolio to initial state"""
        self.portfolio_value = self.config.initial_capital
        self.cash = self.config.initial_capital
        self.positions.clear()
        self.closed_trades.clear()
        self.daily_portfolio_values.clear()
        self.trade_log.clear()
        self.signal_log.clear()
        self.peak_portfolio_value = self.config.initial_capital
        self.max_drawdown_amount = 0.0
        self.drawdown_start_date = None
        self.max_drawdown_duration = 0

# Convenience function
async def backtest_strategy(
    strategy: StrategyTree,
    start_date: datetime,
    end_date: datetime,
    initial_capital: float = 100000.0,
    **config_kwargs
) -> BacktestResult:
    """Convenience function to run a backtest"""
    config = BacktestConfig(
        start_date=start_date,
        end_date=end_date,
        initial_capital=initial_capital,
        **config_kwargs
    )
    
    engine = BacktestEngine(config)
    return await engine.run_backtest(strategy)