"""
Natural Language to Strategy Tree Converter

Converts natural language trading descriptions into deterministic strategy trees
using Claude API with specialized prompts and validation.
"""

from typing import Dict, Any, List, Optional, Union
import asyncio
import logging
import json
import re
from datetime import datetime
import anthropic
import os
from opentelemetry import trace

from .tree_schema import StrategyTree, StrategyNode, create_simple_moving_average_strategy
from .tree_validator import StrategyTreeValidator, ValidationResult

class NaturalLanguageToTree:
    """
    Converts natural language trading ideas into executable strategy trees
    
    Features:
    - Claude API integration for NL understanding
    - Specialized trading prompts
    - Strategy tree generation and validation
    - Iterative refinement based on validation feedback
    """
    
    def __init__(self, api_key: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
        self.logger = logging.getLogger('claude_trading.nl_to_tree')
        self.tracer = trace.get_tracer(__name__)
        
        # Initialize Claude client
        self.api_key = api_key or os.getenv('ANTHROPIC_API_KEY')
        if not self.api_key:
            raise ValueError("Anthropic API key is required. Set ANTHROPIC_API_KEY environment variable.")
        
        self.client = anthropic.Anthropic(api_key=self.api_key)
        
        # Configuration
        self.config = config or self._default_config()
        
        # Initialize validator
        self.validator = StrategyTreeValidator()
        
        # Available indicators and data sources for prompt context
        self.available_indicators = [
            "SMA", "EMA", "RSI", "MACD", "Bollinger Bands", 
            "Stochastic", "ATR", "ADX", "VIX", "Volume SMA"
        ]
        
        self.available_data = [
            "OHLCV price data", "Volume", "Fear & Greed Index",
            "VIX (volatility)", "Market indices", "Sentiment indicators"
        ]
    
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration for NL to tree conversion"""
        return {
            'model': 'claude-3-sonnet-20240229',
            'max_tokens': 4000,
            'temperature': 0.1,  # Low temperature for deterministic output
            'max_refinement_iterations': 3,
            'require_validation': True,
            'auto_fix_common_errors': True,
            'include_risk_management': True,
        }
    
    async def convert(self, description: str, symbol: Optional[str] = None, **kwargs) -> StrategyTree:
        """
        Convert natural language description to strategy tree
        """
        with self.tracer.start_as_current_span("nl_to_tree_convert") as span:
            span.set_attribute("description_length", len(description))
            if symbol:
                span.set_attribute("symbol", symbol)
            
            self.logger.info(f"Converting strategy description: {description[:100]}...")
            
            # Extract or infer symbol if not provided
            if not symbol:
                symbol = self._extract_symbol_from_text(description)
            
            # Generate initial strategy tree
            strategy_tree = await self._generate_strategy_tree(description, symbol, **kwargs)
            
            # Validate and refine if enabled
            if self.config['require_validation']:
                strategy_tree = await self._validate_and_refine(strategy_tree, description)
            
            span.set_attribute("strategy_id", strategy_tree.id)
            span.set_attribute("strategy_name", strategy_tree.name)
            
            self.logger.info(f"Successfully converted strategy: {strategy_tree.name}")
            return strategy_tree
    
    async def _generate_strategy_tree(self, description: str, symbol: str, **kwargs) -> StrategyTree:
        """Generate strategy tree using Claude API"""
        
        # Create specialized prompt
        prompt = self._create_strategy_prompt(description, symbol, **kwargs)
        
        try:
            # Call Claude API
            response = self.client.messages.create(
                model=self.config['model'],
                max_tokens=self.config['max_tokens'],
                temperature=self.config['temperature'],
                messages=[{
                    "role": "user", 
                    "content": prompt
                }]
            )
            
            response_text = response.content[0].text
            
            # Parse JSON response
            strategy_json = self._extract_json_from_response(response_text)
            
            # Create strategy tree from JSON
            strategy_tree = StrategyTree.from_dict(strategy_json)
            
            self.logger.debug(f"Generated strategy tree with {len(strategy_tree.get_all_nodes())} nodes")
            
            return strategy_tree
            
        except Exception as e:
            self.logger.error(f"Failed to generate strategy tree: {e}")
            
            # Fallback: create a simple strategy if generation fails
            if symbol:
                self.logger.info("Falling back to simple moving average strategy")
                return create_simple_moving_average_strategy(symbol)
            else:
                raise ValueError(f"Strategy generation failed and no fallback symbol available: {e}")
    
    def _create_strategy_prompt(self, description: str, symbol: str, **kwargs) -> str:
        """Create specialized prompt for strategy generation"""
        
        base_prompt = f"""You are a professional quantitative trader and strategy architect. Convert the following natural language trading idea into a deterministic, executable strategy tree.

TRADING IDEA:
{description}

TARGET SYMBOL: {symbol}

REQUIREMENTS:
1. Create a deterministic if-else rule tree (NOT a black box)
2. Use only these technical indicators: {', '.join(self.available_indicators)}
3. Use only these data sources: {', '.join(self.available_data)}
4. Include specific entry and exit conditions
5. Add position sizing rules (percentage-based)
6. Include risk management (stop losses, take profits)
7. Return ONLY valid JSON following the schema below

AVAILABLE TECHNICAL INDICATORS:
- SMA/EMA: Specify period (e.g., SMA(20), EMA(50))
- RSI: Specify period and overbought/oversold levels (e.g., RSI(14) < 30)
- MACD: Specify fast/slow/signal periods
- Bollinger Bands: Specify period and standard deviation
- VIX: For market sentiment/volatility
- Volume indicators: Volume SMA, volume spikes

STRATEGY TREE SCHEMA:
{{
  "id": "unique_strategy_id",
  "name": "Strategy Name", 
  "description": "Clear description of strategy logic",
  "symbols": ["{symbol}"],
  "timeframe": "1d",
  "risk_level": "low|medium|high",
  "tags": ["tag1", "tag2"],
  "root_node": {{
    "id": "root",
    "name": "Root Node",
    "description": "Root strategy node",
    "children": [
      {{
        "id": "buy_signal",
        "name": "Buy Signal",
        "description": "When to buy",
        "conditions": [
          {{
            "type": "technical_indicator",
            "indicator": "sma|ema|rsi|macd|etc",
            "operator": ">|<|>=|<=|==",
            "value": 50.0,
            "parameters": {{"period": 20}},
            "description": "Clear condition description"
          }}
        ],
        "condition_logic": "and",
        "actions": [
          {{
            "type": "buy",
            "symbol": "{symbol}",
            "sizing_method": "percentage",
            "amount": 20.0,
            "description": "Buy 20% of portfolio"
          }}
        ],
        "stop_loss_pct": 5.0,
        "take_profit_pct": 15.0,
        "max_daily_executions": 1
      }},
      {{
        "id": "sell_signal",
        "name": "Sell Signal", 
        "description": "When to sell",
        "conditions": [...],
        "actions": [...]
      }}
    ]
  }}
}}

POSITION SIZING RULES:
- Use "percentage" sizing method
- Maximum 25% per position
- Total exposure should not exceed 80%
- Include stop losses (1-20%)
- Include take profits (5-50%)

RISK MANAGEMENT REQUIREMENTS:
- Every buy signal must have stop_loss_pct
- Consider take_profit_pct for profit taking
- Use max_daily_executions to prevent overtrading
- Position sizes should be reasonable (5-25%)

EXAMPLE CONDITIONS:
- RSI oversold: {{"type": "technical_indicator", "indicator": "rsi", "operator": "<", "value": 30, "parameters": {{"period": 14}}}}
- MA crossover: {{"type": "technical_indicator", "indicator": "sma", "operator": ">", "value": 0, "parameters": {{"short_period": 10, "long_period": 50, "comparison_type": "ma_crossover"}}}}
- Price above MA: {{"type": "price_comparison", "operator": ">", "value": "sma_20"}}
- High volume: {{"type": "volume_comparison", "operator": ">", "value": "volume_sma_10", "parameters": {{"multiplier": 1.5}}}}

Convert the trading idea above into a complete, valid JSON strategy tree. Respond with ONLY the JSON, no additional text."""

        # Add additional context if provided
        if kwargs.get('risk_level'):
            base_prompt += f"\n\nRISK LEVEL: {kwargs['risk_level']}"
        
        if kwargs.get('timeframe'):
            base_prompt += f"\nTIMEFRAME: {kwargs['timeframe']}"
        
        if kwargs.get('max_position_size'):
            base_prompt += f"\nMAX POSITION SIZE: {kwargs['max_position_size']}%"
        
        return base_prompt
    
    def _extract_symbol_from_text(self, description: str) -> str:
        """Extract symbol from natural language description"""
        
        # Common stock symbols pattern
        stock_pattern = r'\b[A-Z]{1,5}\b'
        
        # Crypto patterns
        crypto_patterns = [
            r'\bBTC\b', r'\bbitcoin\b',
            r'\bETH\b', r'\bethereum\b', 
            r'\bADA\b', r'\bcardano\b',
            r'\bDOGE\b', r'\bdogecoin\b'
        ]
        
        # Look for crypto first
        for pattern in crypto_patterns:
            match = re.search(pattern, description, re.IGNORECASE)
            if match:
                symbol = match.group().upper()
                if symbol in ['BTC', 'ETH', 'ADA', 'DOGE']:
                    return f"{symbol}-USD"
                else:
                    return symbol
        
        # Look for stock symbols
        stocks = re.findall(stock_pattern, description)
        if stocks:
            # Filter out common words that might match
            common_words = {'THE', 'AND', 'OR', 'BUT', 'IF', 'WHEN', 'THEN', 'WITH'}
            candidates = [s for s in stocks if s not in common_words and len(s) <= 4]
            if candidates:
                return candidates[0]
        
        # Default fallback
        return "SPY"
    
    def _extract_json_from_response(self, response_text: str) -> Dict[str, Any]:
        """Extract JSON from Claude's response"""
        
        # Find JSON block
        json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
        if json_match:
            json_text = json_match.group(1)
        else:
            # Look for JSON object without code blocks
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                json_text = json_match.group(0)
            else:
                raise ValueError("No valid JSON found in Claude's response")
        
        try:
            return json.loads(json_text)
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse JSON: {e}")
            self.logger.error(f"JSON text: {json_text[:500]}...")
            raise ValueError(f"Invalid JSON in response: {e}")
    
    async def _validate_and_refine(self, strategy_tree: StrategyTree, original_description: str) -> StrategyTree:
        """Validate strategy and refine if needed"""
        
        max_iterations = self.config['max_refinement_iterations']
        
        for iteration in range(max_iterations):
            self.logger.debug(f"Validation iteration {iteration + 1}")
            
            # Validate strategy
            validation_result = self.validator.validate(strategy_tree)
            
            if validation_result.is_valid:
                self.logger.info("Strategy validation passed")
                return strategy_tree
            
            # If invalid, try to refine
            if iteration < max_iterations - 1:
                self.logger.info(f"Strategy validation failed, attempting refinement (iteration {iteration + 1})")
                strategy_tree = await self._refine_strategy(
                    strategy_tree, validation_result, original_description
                )
            else:
                # Final iteration - log issues but proceed
                self.logger.warning("Strategy validation failed after maximum refinements")
                self.logger.warning(f"Errors: {validation_result.errors}")
                break
        
        return strategy_tree
    
    async def _refine_strategy(self, strategy_tree: StrategyTree, validation_result: ValidationResult, original_description: str) -> StrategyTree:
        """Refine strategy based on validation feedback"""
        
        # Create refinement prompt
        refinement_prompt = f"""The generated strategy has validation issues. Please fix them while maintaining the core trading logic.

ORIGINAL TRADING IDEA:
{original_description}

CURRENT STRATEGY:
{strategy_tree.to_json(indent=2)}

VALIDATION ERRORS:
{chr(10).join(validation_result.errors)}

VALIDATION WARNINGS:
{chr(10).join(validation_result.warnings)}

RECOMMENDATIONS:
{chr(10).join(validation_result.recommendations)}

Please return the CORRECTED strategy tree as JSON, fixing the validation issues while preserving the original trading logic. Focus on:

1. Fixing any structural errors
2. Ensuring proper risk management (stop losses, position sizing)
3. Correcting technical indicator parameters
4. Maintaining deterministic execution logic

Return ONLY the corrected JSON strategy tree."""

        try:
            # Call Claude for refinement
            response = self.client.messages.create(
                model=self.config['model'],
                max_tokens=self.config['max_tokens'],
                temperature=self.config['temperature'],
                messages=[{
                    "role": "user",
                    "content": refinement_prompt
                }]
            )
            
            response_text = response.content[0].text
            
            # Parse refined strategy
            refined_json = self._extract_json_from_response(response_text)
            refined_strategy = StrategyTree.from_dict(refined_json)
            
            self.logger.debug("Strategy refinement completed")
            return refined_strategy
            
        except Exception as e:
            self.logger.error(f"Strategy refinement failed: {e}")
            # Return original strategy if refinement fails
            return strategy_tree
    
    def convert_examples(self) -> List[StrategyTree]:
        """Generate example strategies for testing"""
        
        examples = [
            {
                "description": "Buy Bitcoin when RSI is below 30 and sell when RSI is above 70. Use 20% of portfolio.",
                "symbol": "BTC-USD"
            },
            {
                "description": "Moving average crossover strategy: buy AAPL when 10-day MA crosses above 50-day MA, sell when it crosses below. Risk 15% of portfolio with 5% stop loss.",
                "symbol": "AAPL"
            },
            {
                "description": "Mean reversion on SPY: buy when price is 2 standard deviations below 20-day moving average, sell when back to moving average. Use 10% position size.",
                "symbol": "SPY"
            },
            {
                "description": "VIX spike strategy: buy QQQ when VIX spikes above 25 and sell when VIX falls below 20. Use 25% of portfolio.",
                "symbol": "QQQ"
            }
        ]
        
        strategies = []
        
        for example in examples:
            try:
                strategy = asyncio.run(self.convert(
                    description=example["description"],
                    symbol=example["symbol"]
                ))
                strategies.append(strategy)
                self.logger.info(f"Generated example strategy: {strategy.name}")
            except Exception as e:
                self.logger.error(f"Failed to generate example strategy: {e}")
        
        return strategies

# Convenience function
async def convert_strategy_description(description: str, symbol: Optional[str] = None, api_key: Optional[str] = None) -> StrategyTree:
    """Convenience function to convert a strategy description"""
    converter = NaturalLanguageToTree(api_key=api_key)
    return await converter.convert(description, symbol)

# CLI support
def main():
    """CLI interface for natural language to strategy tree conversion"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert natural language to strategy tree")
    parser.add_argument("description", help="Natural language strategy description")
    parser.add_argument("--symbol", help="Target symbol (auto-detected if not provided)")
    parser.add_argument("--output", "-o", help="Output file for strategy JSON")
    parser.add_argument("--validate", action="store_true", help="Validate generated strategy")
    parser.add_argument("--examples", action="store_true", help="Generate example strategies")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    converter = NaturalLanguageToTree()
    
    if args.examples:
        print("Generating example strategies...")
        strategies = converter.convert_examples()
        for i, strategy in enumerate(strategies):
            filename = f"example_strategy_{i+1}.json"
            with open(filename, 'w') as f:
                f.write(strategy.to_json(indent=2))
            print(f"Saved example strategy to {filename}")
        return
    
    # Convert strategy description
    async def convert_and_save():
        strategy = await converter.convert(args.description, args.symbol)
        
        if args.validate:
            result = converter.validator.validate(strategy)
            print(f"Validation: {'PASSED' if result.is_valid else 'FAILED'}")
            if result.errors:
                print("Errors:")
                for error in result.errors:
                    print(f"  - {error}")
            if result.warnings:
                print("Warnings:")
                for warning in result.warnings:
                    print(f"  - {warning}")
        
        # Output strategy
        if args.output:
            with open(args.output, 'w') as f:
                f.write(strategy.to_json(indent=2))
            print(f"Strategy saved to {args.output}")
        else:
            print(strategy.to_json(indent=2))
    
    asyncio.run(convert_and_save())

if __name__ == "__main__":
    main()