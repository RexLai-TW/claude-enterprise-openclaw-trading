"""
Strategy Execution Engine

Provides deterministic execution of strategy trees with circuit breaker safety controls.
Outputs trading signals without executing actual trades - AI never touches your money.
"""

from .runner import ExecutionRunner, ExecutionConfig, TradingSignal, SignalType
from .circuit_breaker import CircuitBreaker, CircuitBreakerConfig, SafetyViolation

__all__ = [
    'ExecutionRunner',
    'ExecutionConfig', 
    'TradingSignal',
    'SignalType',
    'CircuitBreaker',
    'CircuitBreakerConfig',
    'SafetyViolation'
]