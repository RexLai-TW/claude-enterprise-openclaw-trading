"""
Circuit Breaker Safety System

Provides safety controls to halt trading when risk limits are exceeded.
Includes drawdown limits, position limits, rate limits, and cooldown periods.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from enum import Enum
import json

class ViolationType(str, Enum):
    """Types of safety violations"""
    MAX_DRAWDOWN = "max_drawdown"
    POSITION_LIMIT = "position_limit"
    RATE_LIMIT = "rate_limit"
    CORRELATION_LIMIT = "correlation_limit"
    PORTFOLIO_EXPOSURE = "portfolio_exposure"
    DAILY_LOSS_LIMIT = "daily_loss_limit"
    VOLATILITY_SPIKE = "volatility_spike"

@dataclass
class SafetyViolation:
    """Record of a safety violation"""
    violation_type: ViolationType
    timestamp: datetime
    description: str
    current_value: float
    limit_value: float
    symbol: Optional[str] = None
    action_taken: str = ""
    severity: str = "warning"  # "warning", "error", "critical"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'violation_type': self.violation_type.value,
            'timestamp': self.timestamp.isoformat(),
            'description': self.description,
            'current_value': self.current_value,
            'limit_value': self.limit_value,
            'symbol': self.symbol,
            'action_taken': self.action_taken,
            'severity': self.severity
        }

@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker safety system"""
    # Drawdown limits
    max_drawdown_pct: float = 0.15  # 15% max drawdown before halt
    daily_loss_limit_pct: float = 0.05  # 5% max daily loss
    
    # Position limits
    max_position_size_pct: float = 10.0  # 10% max per position
    max_portfolio_exposure_pct: float = 80.0  # 80% max total exposure
    max_correlation: float = 0.7  # Max correlation between positions
    
    # Rate limits
    max_trades_per_hour: int = 10  # Max trades per hour
    max_trades_per_day: int = 50  # Max trades per day
    max_daily_turnover_pct: float = 50.0  # Max daily portfolio turnover
    
    # Volatility controls
    volatility_spike_threshold: float = 3.0  # Multiple of normal volatility
    vix_spike_threshold: float = 40.0  # VIX level to halt trading
    
    # Cooldown settings
    violation_cooldown_minutes: int = 60  # Cooldown after violation
    critical_violation_cooldown_hours: int = 24  # Cooldown after critical violation
    
    # Recovery settings
    recovery_mode_enabled: bool = True
    recovery_position_size_pct: float = 2.0  # Reduced position size during recovery
    recovery_period_hours: int = 12  # Hours in recovery mode after violation

@dataclass
class CircuitBreakerState:
    """Current state of the circuit breaker"""
    is_trading_halted: bool = False
    halt_reason: str = ""
    halt_timestamp: Optional[datetime] = None
    
    # Tracking metrics
    current_drawdown_pct: float = 0.0
    daily_loss_pct: float = 0.0
    daily_trade_count: int = 0
    hourly_trade_count: int = 0
    daily_turnover_pct: float = 0.0
    
    # Violation history
    violations: List[SafetyViolation] = field(default_factory=list)
    last_violation_time: Optional[datetime] = None
    
    # Recovery mode
    in_recovery_mode: bool = False
    recovery_start_time: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'is_trading_halted': self.is_trading_halted,
            'halt_reason': self.halt_reason,
            'halt_timestamp': self.halt_timestamp.isoformat() if self.halt_timestamp else None,
            'current_drawdown_pct': self.current_drawdown_pct,
            'daily_loss_pct': self.daily_loss_pct,
            'daily_trade_count': self.daily_trade_count,
            'hourly_trade_count': self.hourly_trade_count,
            'daily_turnover_pct': self.daily_turnover_pct,
            'violations_today': len([v for v in self.violations 
                                   if v.timestamp.date() == datetime.now().date()]),
            'in_recovery_mode': self.in_recovery_mode,
            'recovery_start_time': self.recovery_start_time.isoformat() if self.recovery_start_time else None
        }

class CircuitBreaker:
    """
    Trading safety circuit breaker system
    
    Monitors trading activity and market conditions to prevent excessive losses.
    Automatically halts trading when risk limits are exceeded and manages
    recovery procedures.
    """
    
    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.state = CircuitBreakerState()
        self.logger = logging.getLogger('claude_trading.circuit_breaker')
        
        # Tracking data
        self.portfolio_values: List[Tuple[datetime, float]] = []
        self.peak_value = 0.0
        self.daily_start_value = 0.0
        
        # Rate limiting tracking
        self.trade_timestamps: List[datetime] = []
        self.last_reset_date = datetime.now().date()
        
        self.logger.info("Circuit breaker initialized with safety limits")
    
    def is_safe_to_trade(self) -> bool:
        """
        Check if it's safe to trade based on all circuit breaker conditions
        """
        # Check if trading is explicitly halted
        if self.state.is_trading_halted:
            return self._check_recovery_conditions()
        
        # Update daily counters if new day
        self._update_daily_counters()
        
        # Check all safety conditions
        violations = self._check_all_safety_conditions()
        
        if violations:
            # Process violations
            for violation in violations:
                self._handle_violation(violation)
            
            # Return False if any critical violations
            critical_violations = [v for v in violations if v.severity == "critical"]
            if critical_violations:
                return False
        
        return not self.state.is_trading_halted
    
    def is_signal_safe(self, signal) -> bool:
        """
        Check if a specific trading signal is safe to execute
        """
        from .runner import TradingSignal, SignalType
        
        if not self.is_safe_to_trade():
            return False
        
        violations = []
        
        # Check position size limits
        max_position_value = self.portfolio_values[-1][1] * (self.config.max_position_size_pct / 100) if self.portfolio_values else 100000
        
        if signal.amount > max_position_value:
            violations.append(SafetyViolation(
                violation_type=ViolationType.POSITION_LIMIT,
                timestamp=datetime.now(),
                description=f"Position size ${signal.amount:.2f} exceeds limit ${max_position_value:.2f}",
                current_value=signal.amount,
                limit_value=max_position_value,
                symbol=signal.symbol,
                severity="warning"
            ))
        
        # Check rate limits
        if self._is_rate_limited():
            violations.append(SafetyViolation(
                violation_type=ViolationType.RATE_LIMIT,
                timestamp=datetime.now(),
                description="Trading rate limit exceeded",
                current_value=self.state.hourly_trade_count,
                limit_value=self.config.max_trades_per_hour,
                severity="warning"
            ))
        
        # Apply recovery mode restrictions
        if self.state.in_recovery_mode:
            recovery_max = self.portfolio_values[-1][1] * (self.config.recovery_position_size_pct / 100) if self.portfolio_values else 2000
            if signal.amount > recovery_max:
                violations.append(SafetyViolation(
                    violation_type=ViolationType.POSITION_LIMIT,
                    timestamp=datetime.now(),
                    description=f"Recovery mode: position size limited to ${recovery_max:.2f}",
                    current_value=signal.amount,
                    limit_value=recovery_max,
                    symbol=signal.symbol,
                    severity="warning"
                ))
        
        # Log violations but allow trading unless critical
        for violation in violations:
            self.state.violations.append(violation)
            self.logger.warning(f"Signal safety violation: {violation.description}")
        
        # Block signal if any critical violations
        critical_violations = [v for v in violations if v.severity == "critical"]
        return len(critical_violations) == 0
    
    def record_trade(self, symbol: str, amount: float, price: float, trade_type: str) -> None:
        """Record a trade for monitoring purposes"""
        
        self.trade_timestamps.append(datetime.now())
        self.state.daily_trade_count += 1
        self.state.hourly_trade_count += 1
        
        # Calculate turnover
        if self.portfolio_values:
            current_portfolio_value = self.portfolio_values[-1][1]
            turnover_contribution = (amount / current_portfolio_value) * 100
            self.state.daily_turnover_pct += turnover_contribution
        
        self.logger.debug(f"Recorded trade: {trade_type} {symbol} ${amount:.2f} at ${price:.2f}")
    
    def record_signal(self, signal) -> None:
        """Record a trading signal"""
        from .runner import TradingSignal
        
        # For now, just log the signal
        self.logger.debug(f"Recorded signal: {signal.signal_type.value} {signal.symbol} ${signal.amount:.2f}")
    
    def update_portfolio_value(self, current_value: float) -> None:
        """Update portfolio value and recalculate drawdown metrics"""
        
        now = datetime.now()
        self.portfolio_values.append((now, current_value))
        
        # Update peak value
        if current_value > self.peak_value:
            self.peak_value = current_value
        
        # Calculate current drawdown
        if self.peak_value > 0:
            self.state.current_drawdown_pct = (self.peak_value - current_value) / self.peak_value
        
        # Calculate daily loss if we have today's starting value
        if self.daily_start_value > 0:
            self.state.daily_loss_pct = (self.daily_start_value - current_value) / self.daily_start_value
        
        # Keep only recent portfolio values (last 30 days)
        cutoff_date = now - timedelta(days=30)
        self.portfolio_values = [(dt, val) for dt, val in self.portfolio_values if dt >= cutoff_date]
        
        self.logger.debug(f"Updated portfolio value: ${current_value:.2f}, Drawdown: {self.state.current_drawdown_pct:.2%}")
    
    def force_halt(self, reason: str) -> None:
        """Force halt trading with a custom reason"""
        self.state.is_trading_halted = True
        self.state.halt_reason = reason
        self.state.halt_timestamp = datetime.now()
        
        self.logger.critical(f"Trading forcibly halted: {reason}")
    
    def resume_trading(self) -> None:
        """Manually resume trading (admin function)"""
        self.state.is_trading_halted = False
        self.state.halt_reason = ""
        self.state.halt_timestamp = None
        
        self.logger.info("Trading manually resumed")
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive circuit breaker status"""
        
        status = {
            'timestamp': datetime.now().isoformat(),
            'is_safe_to_trade': self.is_safe_to_trade(),
            'state': self.state.to_dict(),
            'config_limits': {
                'max_drawdown_pct': self.config.max_drawdown_pct,
                'daily_loss_limit_pct': self.config.daily_loss_limit_pct,
                'max_position_size_pct': self.config.max_position_size_pct,
                'max_trades_per_day': self.config.max_trades_per_day,
            },
            'recent_violations': [
                v.to_dict() for v in self.state.violations[-10:]  # Last 10 violations
            ],
            'portfolio_metrics': {
                'current_value': self.portfolio_values[-1][1] if self.portfolio_values else 0,
                'peak_value': self.peak_value,
                'current_drawdown_pct': self.state.current_drawdown_pct,
                'daily_loss_pct': self.state.daily_loss_pct
            }
        }
        
        return status
    
    def _check_all_safety_conditions(self) -> List[SafetyViolation]:
        """Check all safety conditions and return violations"""
        violations = []
        
        # Drawdown checks
        if self.state.current_drawdown_pct > self.config.max_drawdown_pct:
            violations.append(SafetyViolation(
                violation_type=ViolationType.MAX_DRAWDOWN,
                timestamp=datetime.now(),
                description=f"Maximum drawdown exceeded: {self.state.current_drawdown_pct:.2%} > {self.config.max_drawdown_pct:.2%}",
                current_value=self.state.current_drawdown_pct,
                limit_value=self.config.max_drawdown_pct,
                severity="critical"
            ))
        
        # Daily loss check
        if self.state.daily_loss_pct > self.config.daily_loss_limit_pct:
            violations.append(SafetyViolation(
                violation_type=ViolationType.DAILY_LOSS_LIMIT,
                timestamp=datetime.now(),
                description=f"Daily loss limit exceeded: {self.state.daily_loss_pct:.2%} > {self.config.daily_loss_limit_pct:.2%}",
                current_value=self.state.daily_loss_pct,
                limit_value=self.config.daily_loss_limit_pct,
                severity="critical"
            ))
        
        # Rate limit checks
        if self.state.daily_trade_count > self.config.max_trades_per_day:
            violations.append(SafetyViolation(
                violation_type=ViolationType.RATE_LIMIT,
                timestamp=datetime.now(),
                description=f"Daily trade limit exceeded: {self.state.daily_trade_count} > {self.config.max_trades_per_day}",
                current_value=self.state.daily_trade_count,
                limit_value=self.config.max_trades_per_day,
                severity="warning"
            ))
        
        # Turnover check
        if self.state.daily_turnover_pct > self.config.max_daily_turnover_pct:
            violations.append(SafetyViolation(
                violation_type=ViolationType.RATE_LIMIT,
                timestamp=datetime.now(),
                description=f"Daily turnover exceeded: {self.state.daily_turnover_pct:.1%} > {self.config.max_daily_turnover_pct:.1%}",
                current_value=self.state.daily_turnover_pct,
                limit_value=self.config.max_daily_turnover_pct,
                severity="warning"
            ))
        
        return violations
    
    def _handle_violation(self, violation: SafetyViolation) -> None:
        """Handle a safety violation"""
        
        self.state.violations.append(violation)
        self.state.last_violation_time = violation.timestamp
        
        if violation.severity == "critical":
            # Halt trading for critical violations
            self.state.is_trading_halted = True
            self.state.halt_reason = violation.description
            self.state.halt_timestamp = violation.timestamp
            
            violation.action_taken = "Trading halted"
            
            self.logger.critical(f"CRITICAL VIOLATION - Trading halted: {violation.description}")
            
            # Enter recovery mode if enabled
            if self.config.recovery_mode_enabled:
                self._enter_recovery_mode()
        
        elif violation.severity == "warning":
            # Log warnings but continue trading
            self.logger.warning(f"Safety violation: {violation.description}")
            violation.action_taken = "Warning logged"
        
        # Clean up old violations (keep last 100)
        if len(self.state.violations) > 100:
            self.state.violations = self.state.violations[-100:]
    
    def _enter_recovery_mode(self) -> None:
        """Enter recovery mode with reduced risk parameters"""
        self.state.in_recovery_mode = True
        self.state.recovery_start_time = datetime.now()
        
        self.logger.info("Entered recovery mode - reduced position sizing active")
    
    def _check_recovery_conditions(self) -> bool:
        """Check if trading can resume from halted state"""
        
        if not self.state.is_trading_halted:
            return True
        
        if not self.state.halt_timestamp:
            return False
        
        # Check cooldown period
        now = datetime.now()
        halt_duration = now - self.state.halt_timestamp
        
        # Determine required cooldown based on violation severity
        required_cooldown = timedelta(minutes=self.config.violation_cooldown_minutes)
        
        # Check for recent critical violations
        recent_critical = any(
            v.severity == "critical" and 
            (now - v.timestamp) < timedelta(hours=self.config.critical_violation_cooldown_hours)
            for v in self.state.violations
        )
        
        if recent_critical:
            required_cooldown = timedelta(hours=self.config.critical_violation_cooldown_hours)
        
        if halt_duration < required_cooldown:
            remaining_time = required_cooldown - halt_duration
            self.logger.debug(f"Still in cooldown period: {remaining_time} remaining")
            return False
        
        # Check if original violation conditions have improved
        if self._have_conditions_improved():
            self.state.is_trading_halted = False
            self.state.halt_reason = ""
            self.state.halt_timestamp = None
            
            if self.config.recovery_mode_enabled:
                self._enter_recovery_mode()
            
            self.logger.info("Trading resumed - conditions improved")
            return True
        
        return False
    
    def _have_conditions_improved(self) -> bool:
        """Check if the conditions that caused the halt have improved"""
        
        # Check if drawdown has reduced
        if self.state.current_drawdown_pct > self.config.max_drawdown_pct * 0.8:  # 80% of limit
            return False
        
        # Check if we're in a new trading day (resets daily limits)
        if self.last_reset_date < datetime.now().date():
            return True
        
        return True
    
    def _exit_recovery_mode(self) -> None:
        """Exit recovery mode and return to normal trading"""
        self.state.in_recovery_mode = False
        self.state.recovery_start_time = None
        
        self.logger.info("Exited recovery mode - normal position sizing resumed")
    
    def _is_rate_limited(self) -> bool:
        """Check if rate limits are exceeded"""
        
        now = datetime.now()
        
        # Clean up old trade timestamps
        one_hour_ago = now - timedelta(hours=1)
        self.trade_timestamps = [ts for ts in self.trade_timestamps if ts >= one_hour_ago]
        
        # Update hourly count
        self.state.hourly_trade_count = len(self.trade_timestamps)
        
        # Check limits
        return (self.state.hourly_trade_count >= self.config.max_trades_per_hour or
                self.state.daily_trade_count >= self.config.max_trades_per_day)
    
    def _update_daily_counters(self) -> None:
        """Update daily counters and reset if new day"""
        
        today = datetime.now().date()
        
        if today != self.last_reset_date:
            # New day - reset daily counters
            self.state.daily_trade_count = 0
            self.state.daily_loss_pct = 0.0
            self.state.daily_turnover_pct = 0.0
            self.last_reset_date = today
            
            # Set daily start value for loss calculation
            if self.portfolio_values:
                self.daily_start_value = self.portfolio_values[-1][1]
            
            # Check if we can exit recovery mode
            if (self.state.in_recovery_mode and self.state.recovery_start_time and
                (datetime.now() - self.state.recovery_start_time) >= timedelta(hours=self.config.recovery_period_hours)):
                self._exit_recovery_mode()
            
            self.logger.debug("Daily counters reset for new trading day")
    
    def get_violation_history(self, days: int = 7) -> List[Dict[str, Any]]:
        """Get violation history for the last N days"""
        
        cutoff_date = datetime.now() - timedelta(days=days)
        
        recent_violations = [
            v.to_dict() for v in self.state.violations
            if v.timestamp >= cutoff_date
        ]
        
        return recent_violations
    
    def reset_state(self) -> None:
        """Reset circuit breaker state (admin function - use with caution)"""
        self.state = CircuitBreakerState()
        self.portfolio_values.clear()
        self.trade_timestamps.clear()
        self.peak_value = 0.0
        self.daily_start_value = 0.0
        
        self.logger.warning("Circuit breaker state has been reset")

# Convenience functions
def create_circuit_breaker(
    max_drawdown_pct: float = 0.15,
    daily_loss_limit_pct: float = 0.05,
    max_position_size_pct: float = 10.0,
    max_trades_per_day: int = 50
) -> CircuitBreaker:
    """Create circuit breaker with custom limits"""
    
    config = CircuitBreakerConfig(
        max_drawdown_pct=max_drawdown_pct,
        daily_loss_limit_pct=daily_loss_limit_pct,
        max_position_size_pct=max_position_size_pct,
        max_trades_per_day=max_trades_per_day
    )
    
    return CircuitBreaker(config)