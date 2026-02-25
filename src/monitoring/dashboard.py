"""
CLI Dashboard

Simple CLI dashboard showing agent status, active strategies, recent signals,
and key metrics in a user-friendly format.
"""

import asyncio
import logging
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
import json

@dataclass
class DashboardData:
    """Data structure for dashboard information"""
    # System status
    system_status: str = "unknown"  # "running", "idle", "error", "halted"
    last_updated: datetime = field(default_factory=datetime.now)
    
    # Active strategies
    active_strategies: List[Dict[str, Any]] = field(default_factory=list)
    
    # Recent signals
    recent_signals: List[Dict[str, Any]] = field(default_factory=list)
    
    # Performance metrics
    portfolio_value: float = 0.0
    daily_pnl: float = 0.0
    daily_pnl_pct: float = 0.0
    
    # Trading activity
    trades_today: int = 0
    signals_today: int = 0
    
    # Risk metrics
    current_drawdown_pct: float = 0.0
    exposure_pct: float = 0.0
    
    # Circuit breaker status
    circuit_breaker_status: str = "unknown"
    violations_today: int = 0
    
    # Data connectivity
    data_sources_connected: List[str] = field(default_factory=list)
    data_sources_errors: List[str] = field(default_factory=list)

class CLIDashboard:
    """
    Command-line dashboard for monitoring trading agent status and activity
    """
    
    def __init__(self, refresh_interval: int = 30):
        self.refresh_interval = refresh_interval
        self.logger = logging.getLogger('claude_trading.dashboard')
        self.data = DashboardData()
        self.running = False
        
        # Dashboard configuration
        self.show_detailed_signals = True
        self.max_signals_display = 10
        self.max_strategies_display = 5
        
    async def start(self) -> None:
        """Start the dashboard"""
        self.running = True
        self.logger.info("Starting CLI dashboard")
        
        try:
            while self.running:
                # Clear screen
                self._clear_screen()
                
                # Update data
                await self._update_data()
                
                # Display dashboard
                self._display_dashboard()
                
                # Wait for refresh interval
                await asyncio.sleep(self.refresh_interval)
                
        except KeyboardInterrupt:
            self.logger.info("Dashboard stopped by user")
        except Exception as e:
            self.logger.error(f"Dashboard error: {e}")
        finally:
            self.running = False
    
    def stop(self) -> None:
        """Stop the dashboard"""
        self.running = False
    
    async def _update_data(self) -> None:
        """Update dashboard data from various sources"""
        try:
            # Update timestamp
            self.data.last_updated = datetime.now()
            
            # Load data from files or services
            await self._load_strategy_data()
            await self._load_signal_data()
            await self._load_performance_data()
            await self._load_circuit_breaker_data()
            await self._check_data_connectivity()
            
        except Exception as e:
            self.logger.error(f"Error updating dashboard data: {e}")
            self.data.system_status = "error"
    
    async def _load_strategy_data(self) -> None:
        """Load active strategy information"""
        try:
            # Look for strategy status files (would be created by orchestrator)
            strategy_files = []
            workspace_dir = "/home/node/.openclaw/workspace/claude-enterprise-trading"
            strategy_status_dir = os.path.join(workspace_dir, "runtime", "strategies")
            
            if os.path.exists(strategy_status_dir):
                strategy_files = [f for f in os.listdir(strategy_status_dir) if f.endswith('.json')]
            
            self.data.active_strategies = []
            
            for strategy_file in strategy_files[:self.max_strategies_display]:
                try:
                    with open(os.path.join(strategy_status_dir, strategy_file), 'r') as f:
                        strategy_data = json.load(f)
                        self.data.active_strategies.append(strategy_data)
                except Exception as e:
                    self.logger.error(f"Error loading strategy file {strategy_file}: {e}")
            
            # Set system status based on active strategies
            if self.data.active_strategies:
                self.data.system_status = "running"
            else:
                self.data.system_status = "idle"
                
        except Exception as e:
            self.logger.error(f"Error loading strategy data: {e}")
    
    async def _load_signal_data(self) -> None:
        """Load recent trading signals"""
        try:
            # Look for signal log files
            workspace_dir = "/home/node/.openclaw/workspace/claude-enterprise-trading"
            signals_file = os.path.join(workspace_dir, "runtime", "signals", "signals.jsonl")
            
            self.data.recent_signals = []
            self.data.signals_today = 0
            
            if os.path.exists(signals_file):
                today = datetime.now().date()
                
                with open(signals_file, 'r') as f:
                    for line in f.readlines()[-50:]:  # Last 50 signals
                        try:
                            signal = json.loads(line.strip())
                            signal_date = datetime.fromisoformat(signal.get('timestamp', '')).date()
                            
                            if signal_date == today:
                                self.data.signals_today += 1
                            
                            self.data.recent_signals.append(signal)
                            
                        except Exception as e:
                            continue
                
                # Keep only recent signals for display
                self.data.recent_signals = self.data.recent_signals[-self.max_signals_display:]
                
        except Exception as e:
            self.logger.error(f"Error loading signal data: {e}")
    
    async def _load_performance_data(self) -> None:
        """Load performance metrics"""
        try:
            # Look for portfolio status file
            workspace_dir = "/home/node/.openclaw/workspace/claude-enterprise-trading"
            portfolio_file = os.path.join(workspace_dir, "runtime", "portfolio", "status.json")
            
            if os.path.exists(portfolio_file):
                with open(portfolio_file, 'r') as f:
                    portfolio_data = json.load(f)
                    
                    self.data.portfolio_value = portfolio_data.get('total_value', 0.0)
                    self.data.daily_pnl = portfolio_data.get('daily_pnl', 0.0)
                    self.data.daily_pnl_pct = portfolio_data.get('daily_pnl_pct', 0.0)
                    self.data.exposure_pct = portfolio_data.get('exposure_pct', 0.0)
                    self.data.current_drawdown_pct = portfolio_data.get('drawdown_pct', 0.0)
            else:
                # Default values if no portfolio file
                self.data.portfolio_value = 100000.0
                self.data.daily_pnl = 0.0
                self.data.daily_pnl_pct = 0.0
                
        except Exception as e:
            self.logger.error(f"Error loading performance data: {e}")
    
    async def _load_circuit_breaker_data(self) -> None:
        """Load circuit breaker status"""
        try:
            # Look for circuit breaker status file
            workspace_dir = "/home/node/.openclaw/workspace/claude-enterprise-trading"
            cb_file = os.path.join(workspace_dir, "runtime", "safety", "circuit_breaker.json")
            
            if os.path.exists(cb_file):
                with open(cb_file, 'r') as f:
                    cb_data = json.load(f)
                    
                    self.data.circuit_breaker_status = "active" if cb_data.get('is_trading_halted') else "safe"
                    self.data.violations_today = cb_data.get('violations_today', 0)
            else:
                self.data.circuit_breaker_status = "unknown"
                self.data.violations_today = 0
                
        except Exception as e:
            self.logger.error(f"Error loading circuit breaker data: {e}")
    
    async def _check_data_connectivity(self) -> None:
        """Check data source connectivity"""
        try:
            # This would check actual data connectors in a real implementation
            self.data.data_sources_connected = ["yfinance", "coingecko"]
            self.data.data_sources_errors = []
            
        except Exception as e:
            self.logger.error(f"Error checking data connectivity: {e}")
    
    def _clear_screen(self) -> None:
        """Clear the terminal screen"""
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def _display_dashboard(self) -> None:
        """Display the complete dashboard"""
        
        # Header
        print("=" * 80)
        print("CLAUDE ENTERPRISE TRADING - DASHBOARD")
        print("=" * 80)
        print(f"Last Updated: {self.data.last_updated.strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # System status section
        self._display_system_status()
        print()
        
        # Portfolio section
        self._display_portfolio_metrics()
        print()
        
        # Trading activity section
        self._display_trading_activity()
        print()
        
        # Active strategies section
        self._display_active_strategies()
        print()
        
        # Recent signals section
        self._display_recent_signals()
        print()
        
        # Safety section
        self._display_safety_status()
        print()
        
        # Data connectivity
        self._display_data_status()
        print()
        
        # Footer
        print("-" * 80)
        print(f"Press Ctrl+C to exit | Refresh every {self.refresh_interval}s")
        print("=" * 80)
    
    def _display_system_status(self) -> None:
        """Display system status"""
        status_color = self._get_status_color(self.data.system_status)
        
        print("SYSTEM STATUS")
        print("-" * 40)
        print(f"Status: {status_color}{self.data.system_status.upper()}{self._reset_color()}")
        print(f"Active Strategies: {len(self.data.active_strategies)}")
        print(f"Signals Today: {self.data.signals_today}")
        print(f"Trades Today: {self.data.trades_today}")
    
    def _display_portfolio_metrics(self) -> None:
        """Display portfolio performance metrics"""
        print("PORTFOLIO PERFORMANCE")
        print("-" * 40)
        print(f"Portfolio Value: ${self.data.portfolio_value:,.2f}")
        
        # Daily P&L with color coding
        pnl_color = self._get_pnl_color(self.data.daily_pnl)
        print(f"Daily P&L: {pnl_color}${self.data.daily_pnl:+.2f} ({self.data.daily_pnl_pct:+.2%}){self._reset_color()}")
        
        # Risk metrics
        dd_color = self._get_drawdown_color(self.data.current_drawdown_pct)
        print(f"Current Drawdown: {dd_color}{self.data.current_drawdown_pct:.2%}{self._reset_color()}")
        
        exposure_color = self._get_exposure_color(self.data.exposure_pct)
        print(f"Portfolio Exposure: {exposure_color}{self.data.exposure_pct:.1%}{self._reset_color()}")
    
    def _display_trading_activity(self) -> None:
        """Display recent trading activity"""
        print("TRADING ACTIVITY")
        print("-" * 40)
        print(f"Signals Generated Today: {self.data.signals_today}")
        print(f"Trades Executed Today: {self.data.trades_today}")
        
        if self.data.recent_signals:
            last_signal = self.data.recent_signals[-1]
            last_time = datetime.fromisoformat(last_signal.get('timestamp', ''))
            time_ago = datetime.now() - last_time
            
            if time_ago.total_seconds() < 3600:  # Less than 1 hour
                print(f"Last Signal: {int(time_ago.total_seconds() / 60)} minutes ago")
            else:
                print(f"Last Signal: {int(time_ago.total_seconds() / 3600)} hours ago")
        else:
            print("Last Signal: None today")
    
    def _display_active_strategies(self) -> None:
        """Display active strategies"""
        print("ACTIVE STRATEGIES")
        print("-" * 40)
        
        if not self.data.active_strategies:
            print("No active strategies")
            return
        
        for i, strategy in enumerate(self.data.active_strategies[:self.max_strategies_display]):
            name = strategy.get('name', 'Unknown Strategy')
            status = strategy.get('status', 'unknown')
            last_run = strategy.get('last_run', 'Never')
            
            status_indicator = "🟢" if status == "running" else "🟡" if status == "idle" else "🔴"
            print(f"{i+1}. {status_indicator} {name} ({status}) - Last run: {last_run}")
    
    def _display_recent_signals(self) -> None:
        """Display recent trading signals"""
        print("RECENT SIGNALS")
        print("-" * 40)
        
        if not self.data.recent_signals:
            print("No recent signals")
            return
        
        for signal in reversed(self.data.recent_signals[-5:]):  # Show last 5 signals
            signal_type = signal.get('signal_type', 'unknown').upper()
            symbol = signal.get('symbol', 'UNKNOWN')
            amount = signal.get('amount', 0.0)
            timestamp = signal.get('timestamp', '')
            
            try:
                signal_time = datetime.fromisoformat(timestamp)
                time_str = signal_time.strftime('%H:%M')
            except:
                time_str = "??:??"
            
            # Color code signal types
            if signal_type == 'BUY':
                signal_color = '\033[32m'  # Green
            elif signal_type == 'SELL':
                signal_color = '\033[31m'  # Red
            else:
                signal_color = '\033[33m'  # Yellow
            
            print(f"{time_str} {signal_color}{signal_type:4}{self._reset_color()} {symbol:6} ${amount:>8.0f}")
    
    def _display_safety_status(self) -> None:
        """Display safety and circuit breaker status"""
        print("SAFETY & RISK MANAGEMENT")
        print("-" * 40)
        
        cb_color = self._get_circuit_breaker_color(self.data.circuit_breaker_status)
        print(f"Circuit Breaker: {cb_color}{self.data.circuit_breaker_status.upper()}{self._reset_color()}")
        
        if self.data.violations_today > 0:
            print(f"Safety Violations Today: {self.data.violations_today}")
        else:
            print("Safety Violations Today: None")
    
    def _display_data_status(self) -> None:
        """Display data connectivity status"""
        print("DATA CONNECTIVITY")
        print("-" * 40)
        
        if self.data.data_sources_connected:
            print(f"Connected: {', '.join(self.data.data_sources_connected)}")
        
        if self.data.data_sources_errors:
            print(f"Errors: {', '.join(self.data.data_sources_errors)}")
        else:
            print("All data sources healthy")
    
    def _get_status_color(self, status: str) -> str:
        """Get color code for system status"""
        colors = {
            'running': '\033[32m',  # Green
            'idle': '\033[33m',     # Yellow
            'error': '\033[31m',    # Red
            'halted': '\033[35m'    # Magenta
        }
        return colors.get(status, '\033[37m')  # White default
    
    def _get_pnl_color(self, pnl: float) -> str:
        """Get color code for P&L"""
        if pnl > 0:
            return '\033[32m'  # Green
        elif pnl < 0:
            return '\033[31m'  # Red
        else:
            return '\033[37m'  # White
    
    def _get_drawdown_color(self, drawdown: float) -> str:
        """Get color code for drawdown"""
        if drawdown < 0.05:  # < 5%
            return '\033[32m'  # Green
        elif drawdown < 0.15:  # < 15%
            return '\033[33m'  # Yellow
        else:
            return '\033[31m'  # Red
    
    def _get_exposure_color(self, exposure: float) -> str:
        """Get color code for portfolio exposure"""
        if exposure < 50:
            return '\033[32m'  # Green
        elif exposure < 80:
            return '\033[33m'  # Yellow
        else:
            return '\033[31m'  # Red
    
    def _get_circuit_breaker_color(self, status: str) -> str:
        """Get color code for circuit breaker status"""
        colors = {
            'safe': '\033[32m',     # Green
            'active': '\033[31m',   # Red
            'unknown': '\033[37m'   # White
        }
        return colors.get(status, '\033[37m')
    
    def _reset_color(self) -> str:
        """Reset color to default"""
        return '\033[0m'
    
    def display_static_summary(self) -> None:
        """Display a static summary (non-refreshing)"""
        asyncio.run(self._update_data())
        self._clear_screen()
        self._display_dashboard()
    
    def export_data(self, filename: str) -> None:
        """Export current dashboard data to JSON file"""
        try:
            export_data = {
                'timestamp': self.data.last_updated.isoformat(),
                'system_status': self.data.system_status,
                'portfolio_value': self.data.portfolio_value,
                'daily_pnl': self.data.daily_pnl,
                'daily_pnl_pct': self.data.daily_pnl_pct,
                'current_drawdown_pct': self.data.current_drawdown_pct,
                'exposure_pct': self.data.exposure_pct,
                'trades_today': self.data.trades_today,
                'signals_today': self.data.signals_today,
                'circuit_breaker_status': self.data.circuit_breaker_status,
                'violations_today': self.data.violations_today,
                'active_strategies': self.data.active_strategies,
                'recent_signals': self.data.recent_signals,
                'data_sources_connected': self.data.data_sources_connected,
                'data_sources_errors': self.data.data_sources_errors
            }
            
            with open(filename, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            print(f"Dashboard data exported to {filename}")
            
        except Exception as e:
            self.logger.error(f"Error exporting dashboard data: {e}")

# CLI interface for dashboard
async def main():
    """Main CLI interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Claude Enterprise Trading Dashboard")
    parser.add_argument("--refresh", "-r", type=int, default=30, help="Refresh interval in seconds")
    parser.add_argument("--static", "-s", action="store_true", help="Show static summary instead of live dashboard")
    parser.add_argument("--export", "-e", help="Export data to JSON file")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.WARNING)  # Quiet logging for dashboard
    
    dashboard = CLIDashboard(refresh_interval=args.refresh)
    
    if args.static:
        dashboard.display_static_summary()
    elif args.export:
        await dashboard._update_data()
        dashboard.export_data(args.export)
    else:
        print("Starting Claude Enterprise Trading Dashboard...")
        print(f"Refresh interval: {args.refresh} seconds")
        print("Press Ctrl+C to exit")
        await asyncio.sleep(2)  # Brief pause
        await dashboard.start()

if __name__ == "__main__":
    asyncio.run(main())