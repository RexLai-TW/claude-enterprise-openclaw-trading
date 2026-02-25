"""
Claude Enterprise Trading - Agentic Trading Infrastructure

This package provides a complete framework for building AI-powered trading strategies
using Claude Enterprise MCP connectors and OpenClaw orchestration.

Key Components:
- Data connectors with MCP protocol support
- Natural language to strategy tree conversion
- Backtesting engine with performance metrics
- Execution framework with circuit breakers
- OpenTelemetry monitoring and tracing
"""

__version__ = "1.0.0"
__author__ = "OpenClaw Community"
__email__ = "community@openclaw.ai"

from typing import Dict, Any
import logging
import sys
from pathlib import Path

# Configure package-level logging
def setup_logging(level: str = "INFO", format_type: str = "standard") -> None:
    """Setup package-wide logging configuration."""
    
    if format_type == "json":
        import json
        class JSONFormatter(logging.Formatter):
            def format(self, record):
                log_data = {
                    'timestamp': self.formatTime(record),
                    'level': record.levelname,
                    'module': record.module,
                    'message': record.getMessage(),
                }
                if record.exc_info:
                    log_data['exception'] = self.formatException(record.exc_info)
                return json.dumps(log_data)
        
        formatter = JSONFormatter()
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    # Setup root logger
    root_logger = logging.getLogger('claude_trading')
    root_logger.setLevel(getattr(logging, level.upper()))
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (optional)
    log_dir = Path.home() / '.claude-trading' / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_dir / 'trading.log')
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

# Initialize logging with environment defaults
import os
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = os.getenv("LOG_FORMAT", "standard")
setup_logging(LOG_LEVEL, LOG_FORMAT)

# Create package logger
logger = logging.getLogger('claude_trading')
logger.info(f"Claude Enterprise Trading v{__version__} initialized")

# Export key components
from .orchestrator import Orchestrator
from .data.market_data import MarketDataInterface
from .strategy.tree_schema import StrategyTree
from .backtest.engine import BacktestEngine
from .execution.runner import ExecutionRunner
from .monitoring.otel_tracer import setup_tracing

__all__ = [
    'Orchestrator',
    'MarketDataInterface', 
    'StrategyTree',
    'BacktestEngine',
    'ExecutionRunner',
    'setup_tracing',
    'setup_logging',
]

# Package-level configuration
CONFIG = {
    'version': __version__,
    'data_dir': Path.home() / '.claude-trading' / 'data',
    'config_dir': Path.home() / '.claude-trading' / 'config',
    'cache_dir': Path.home() / '.claude-trading' / 'cache',
}

# Ensure directories exist
for dir_path in CONFIG.values():
    if isinstance(dir_path, Path):
        dir_path.mkdir(parents=True, exist_ok=True)