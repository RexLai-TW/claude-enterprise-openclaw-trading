"""
Backtest Report Generator

Generates comprehensive, human-readable backtest reports from BacktestResult data.
Supports both dict/JSON output and formatted text summaries.
"""

import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import json

from .engine import BacktestResult, Trade

@dataclass
class BacktestReport:
    """
    Comprehensive backtest report with formatted output
    """
    result: BacktestResult
    summary: str
    performance_analysis: str
    risk_analysis: str
    trade_analysis: str
    recommendations: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary"""
        return {
            'strategy_id': self.result.strategy_id,
            'backtest_period': {
                'start_date': self.result.start_date.isoformat(),
                'end_date': self.result.end_date.isoformat(),
                'duration_days': (self.result.end_date - self.result.start_date).days
            },
            'summary': self.summary,
            'performance_analysis': self.performance_analysis,
            'risk_analysis': self.risk_analysis,
            'trade_analysis': self.trade_analysis,
            'recommendations': self.recommendations,
            'raw_metrics': self.result.to_dict(),
            'generated_at': datetime.now().isoformat()
        }
    
    def to_json(self, indent: int = 2) -> str:
        """Convert report to JSON string"""
        return json.dumps(self.to_dict(), indent=indent, default=str)
    
    def __str__(self) -> str:
        """Return formatted text report"""
        return self.get_formatted_report()
    
    def get_formatted_report(self) -> str:
        """Generate formatted text report"""
        lines = [
            "=" * 80,
            f"BACKTEST REPORT - {self.result.strategy_id}",
            "=" * 80,
            "",
            self.summary,
            "",
            "PERFORMANCE ANALYSIS",
            "-" * 40,
            self.performance_analysis,
            "",
            "RISK ANALYSIS", 
            "-" * 40,
            self.risk_analysis,
            "",
            "TRADE ANALYSIS",
            "-" * 40,
            self.trade_analysis,
            "",
            "RECOMMENDATIONS",
            "-" * 40
        ]
        
        for i, rec in enumerate(self.recommendations, 1):
            lines.append(f"{i}. {rec}")
        
        lines.extend([
            "",
            "=" * 80,
            f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "=" * 80
        ])
        
        return "\n".join(lines)

class BacktestReportGenerator:
    """
    Generates comprehensive backtest reports with analysis and recommendations
    """
    
    def __init__(self):
        self.logger = logging.getLogger('claude_trading.backtest_report')
    
    def generate_report(self, result: BacktestResult) -> BacktestReport:
        """
        Generate comprehensive backtest report from results
        """
        self.logger.info(f"Generating backtest report for strategy {result.strategy_id}")
        
        # Generate each section
        summary = self._generate_summary(result)
        performance_analysis = self._generate_performance_analysis(result)
        risk_analysis = self._generate_risk_analysis(result)
        trade_analysis = self._generate_trade_analysis(result)
        recommendations = self._generate_recommendations(result)
        
        return BacktestReport(
            result=result,
            summary=summary,
            performance_analysis=performance_analysis,
            risk_analysis=risk_analysis,
            trade_analysis=trade_analysis,
            recommendations=recommendations
        )
    
    def _generate_summary(self, result: BacktestResult) -> str:
        """Generate executive summary"""
        duration_days = (result.end_date - result.start_date).days
        duration_years = duration_days / 365.25
        
        # Performance grade
        if result.sharpe_ratio >= 2.0:
            performance_grade = "EXCELLENT"
        elif result.sharpe_ratio >= 1.5:
            performance_grade = "GOOD"
        elif result.sharpe_ratio >= 1.0:
            performance_grade = "FAIR"
        elif result.sharpe_ratio >= 0.5:
            performance_grade = "POOR"
        else:
            performance_grade = "VERY POOR"
        
        # Risk grade
        if result.max_drawdown <= 0.05:  # <= 5%
            risk_grade = "LOW"
        elif result.max_drawdown <= 0.15:  # <= 15%
            risk_grade = "MODERATE"
        elif result.max_drawdown <= 0.25:  # <= 25%
            risk_grade = "HIGH"
        else:
            risk_grade = "VERY HIGH"
        
        return f"""EXECUTIVE SUMMARY

Strategy: {result.strategy_id}
Period: {result.start_date.strftime('%Y-%m-%d')} to {result.end_date.strftime('%Y-%m-%d')} ({duration_days} days)
Initial Capital: ${result.initial_capital:,.2f}
Final Capital: ${result.final_capital:,.2f}

PERFORMANCE: {performance_grade}
• Total Return: {result.total_return:.1%}
• Annualized Return: {result.annualized_return:.1%}
• Sharpe Ratio: {result.sharpe_ratio:.2f}

RISK: {risk_grade}
• Maximum Drawdown: {result.max_drawdown:.1%}
• Volatility: {result.volatility:.1%}
• Win Rate: {result.win_rate:.1%}

TRADING ACTIVITY
• Total Trades: {result.total_trades}
• Average Trade P&L: ${result.avg_trade_pnl:.2f}
• Profit Factor: {result.profit_factor:.2f}"""
    
    def _generate_performance_analysis(self, result: BacktestResult) -> str:
        """Generate detailed performance analysis"""
        
        # Benchmark comparison (assuming SPY as benchmark)
        benchmark_annual_return = 0.10  # Assume 10% SPY annual return
        excess_return = result.annualized_return - benchmark_annual_return
        
        # Information ratio (simplified)
        info_ratio = excess_return / result.volatility if result.volatility > 0 else 0
        
        # Return consistency
        positive_days = len([r for r in result.daily_returns if r > 0])
        total_days = len(result.daily_returns)
        positive_day_ratio = positive_days / total_days if total_days > 0 else 0
        
        lines = [
            f"Total Return: {result.total_return:.2%}",
            f"Annualized Return: {result.annualized_return:.2%}",
            f"Excess Return vs Benchmark: {excess_return:.2%}",
            f"Information Ratio: {info_ratio:.2f}",
            "",
            f"Sharpe Ratio: {result.sharpe_ratio:.2f}",
            self._interpret_sharpe_ratio(result.sharpe_ratio),
            "",
            f"Calmar Ratio: {result.calmar_ratio:.2f}",
            self._interpret_calmar_ratio(result.calmar_ratio),
            "",
            f"Return Consistency: {positive_day_ratio:.1%} of days were positive",
        ]
        
        # Performance periods analysis
        if len(result.equity_curve) > 1:
            best_period, worst_period = self._analyze_performance_periods(result)
            lines.extend([
                "",
                "PERFORMANCE PERIODS",
                f"Best Period: {best_period}",
                f"Worst Period: {worst_period}"
            ])
        
        return "\n".join(lines)
    
    def _generate_risk_analysis(self, result: BacktestResult) -> str:
        """Generate detailed risk analysis"""
        lines = [
            f"Maximum Drawdown: {result.max_drawdown:.2%}",
            self._interpret_max_drawdown(result.max_drawdown),
            f"Drawdown Duration: {result.max_drawdown_duration_days} days",
            "",
            f"Volatility (Annualized): {result.volatility:.2%}",
            self._interpret_volatility(result.volatility),
            "",
            f"Value at Risk (95%): {result.var_95:.2%}",
            "This represents the daily loss expected 5% of the time",
        ]
        
        # Risk-adjusted returns
        if result.max_drawdown > 0:
            risk_adjusted_return = result.annualized_return / result.max_drawdown
            lines.extend([
                "",
                f"Risk-Adjusted Return: {risk_adjusted_return:.2f}",
                "Higher values indicate better risk-adjusted performance"
            ])
        
        # Downside analysis
        negative_returns = [r for r in result.daily_returns if r < 0]
        if negative_returns:
            avg_loss = sum(negative_returns) / len(negative_returns)
            worst_day = min(negative_returns)
            
            lines.extend([
                "",
                "DOWNSIDE ANALYSIS",
                f"Average Down Day: {avg_loss:.2%}",
                f"Worst Single Day: {worst_day:.2%}",
                f"Down Days: {len(negative_returns)} out of {len(result.daily_returns)} ({len(negative_returns)/len(result.daily_returns):.1%})"
            ])
        
        return "\n".join(lines)
    
    def _generate_trade_analysis(self, result: BacktestResult) -> str:
        """Generate detailed trade analysis"""
        
        if result.total_trades == 0:
            return "No trades were executed during the backtest period."
        
        lines = [
            f"Total Trades: {result.total_trades}",
            f"Winning Trades: {result.winning_trades} ({result.win_rate:.1%})",
            f"Losing Trades: {result.losing_trades} ({(1-result.win_rate):.1%})",
            "",
            f"Average Trade P&L: ${result.avg_trade_pnl:.2f}",
            f"Average Winning Trade: ${result.avg_winning_trade:.2f}",
            f"Average Losing Trade: ${result.avg_losing_trade:.2f}",
            "",
            f"Largest Win: ${result.largest_win:.2f}",
            f"Largest Loss: ${result.largest_loss:.2f}",
            "",
            f"Profit Factor: {result.profit_factor:.2f}",
            self._interpret_profit_factor(result.profit_factor),
        ]
        
        # Trade duration analysis
        if result.trades:
            durations = [trade.duration_days for trade in result.trades]
            avg_duration = sum(durations) / len(durations)
            max_duration = max(durations)
            min_duration = min(durations)
            
            lines.extend([
                "",
                "TRADE DURATION",
                f"Average: {avg_duration:.1f} days",
                f"Range: {min_duration} to {max_duration} days"
            ])
        
        # Monthly trade distribution
        monthly_trades = self._analyze_monthly_trading(result.trades)
        if monthly_trades:
            lines.extend([
                "",
                "MONTHLY TRADING ACTIVITY",
                monthly_trades
            ])
        
        # Consecutive wins/losses
        consecutive_analysis = self._analyze_consecutive_trades(result.trades)
        if consecutive_analysis:
            lines.extend([
                "",
                "CONSECUTIVE TRADES",
                consecutive_analysis
            ])
        
        return "\n".join(lines)
    
    def _generate_recommendations(self, result: BacktestResult) -> List[str]:
        """Generate actionable recommendations based on backtest results"""
        recommendations = []
        
        # Performance recommendations
        if result.sharpe_ratio < 1.0:
            recommendations.append(
                "Consider improving risk-adjusted returns. Sharpe ratio below 1.0 indicates poor risk-adjusted performance."
            )
        
        # Risk recommendations
        if result.max_drawdown > 0.20:  # > 20%
            recommendations.append(
                "Implement stronger risk management. Maximum drawdown exceeds 20%, which may be unacceptable for many investors."
            )
        
        if result.max_drawdown > 0.15 and result.max_drawdown_duration_days > 90:
            recommendations.append(
                "Consider position sizing adjustments. Extended drawdown periods can test investor patience."
            )
        
        # Trading frequency recommendations
        days_in_period = (result.end_date - result.start_date).days
        trades_per_month = result.total_trades / (days_in_period / 30)
        
        if trades_per_month > 10:
            recommendations.append(
                "High trading frequency detected. Consider transaction cost analysis and potential overtrading."
            )
        elif trades_per_month < 1:
            recommendations.append(
                "Low trading frequency. Consider whether strategy captures enough opportunities or if entry criteria are too restrictive."
            )
        
        # Win rate recommendations
        if result.win_rate < 0.40:  # < 40%
            recommendations.append(
                "Low win rate detected. Consider refining entry conditions or implementing better trend following logic."
            )
        elif result.win_rate > 0.80:  # > 80%
            recommendations.append(
                "Very high win rate may indicate overfitting. Verify strategy robustness with out-of-sample testing."
            )
        
        # Profit factor recommendations
        if result.profit_factor < 1.5:
            recommendations.append(
                "Profit factor below 1.5 suggests marginal profitability. Focus on reducing losses or improving exits."
            )
        
        # Volatility recommendations
        if result.volatility > 0.30:  # > 30% annual volatility
            recommendations.append(
                "High strategy volatility detected. Consider position sizing adjustments or volatility targeting."
            )
        
        # Return consistency
        if len(result.daily_returns) > 0:
            positive_days = len([r for r in result.daily_returns if r > 0])
            positive_ratio = positive_days / len(result.daily_returns)
            
            if positive_ratio < 0.45:
                recommendations.append(
                    "Low percentage of positive days. Consider market timing improvements or trend identification."
                )
        
        # Default recommendation if performance is good
        if not recommendations and result.sharpe_ratio > 1.5 and result.max_drawdown < 0.15:
            recommendations.append(
                "Strategy shows strong performance metrics. Consider forward testing with live data before deployment."
            )
        
        # Always include general recommendations
        recommendations.extend([
            "Conduct out-of-sample testing on different time periods to validate robustness.",
            "Consider transaction costs, slippage, and market impact for live trading.",
            "Monitor strategy performance closely during initial live trading phases."
        ])
        
        return recommendations
    
    def _interpret_sharpe_ratio(self, sharpe: float) -> str:
        """Interpret Sharpe ratio value"""
        if sharpe >= 3.0:
            return "Exceptional risk-adjusted performance"
        elif sharpe >= 2.0:
            return "Excellent risk-adjusted performance"
        elif sharpe >= 1.5:
            return "Good risk-adjusted performance"
        elif sharpe >= 1.0:
            return "Acceptable risk-adjusted performance"
        elif sharpe >= 0.5:
            return "Poor risk-adjusted performance"
        else:
            return "Very poor risk-adjusted performance"
    
    def _interpret_calmar_ratio(self, calmar: float) -> str:
        """Interpret Calmar ratio value"""
        if calmar >= 1.0:
            return "Strong return relative to maximum drawdown"
        elif calmar >= 0.5:
            return "Moderate return relative to maximum drawdown"
        else:
            return "Low return relative to maximum drawdown"
    
    def _interpret_max_drawdown(self, drawdown: float) -> str:
        """Interpret maximum drawdown"""
        if drawdown <= 0.05:
            return "Very low drawdown - excellent capital preservation"
        elif drawdown <= 0.10:
            return "Low drawdown - good capital preservation"
        elif drawdown <= 0.20:
            return "Moderate drawdown - acceptable for most strategies"
        elif drawdown <= 0.30:
            return "High drawdown - may require strong risk management"
        else:
            return "Very high drawdown - significant risk to capital"
    
    def _interpret_volatility(self, vol: float) -> str:
        """Interpret volatility level"""
        if vol <= 0.10:
            return "Low volatility - conservative strategy"
        elif vol <= 0.20:
            return "Moderate volatility - typical for equity strategies"
        elif vol <= 0.30:
            return "High volatility - aggressive strategy"
        else:
            return "Very high volatility - speculative strategy"
    
    def _interpret_profit_factor(self, pf: float) -> str:
        """Interpret profit factor"""
        if pf >= 2.0:
            return "Excellent profit factor - strong edge"
        elif pf >= 1.5:
            return "Good profit factor - solid edge"
        elif pf >= 1.2:
            return "Acceptable profit factor - modest edge"
        elif pf >= 1.0:
            return "Marginal profit factor - weak edge"
        else:
            return "Poor profit factor - strategy loses money"
    
    def _analyze_performance_periods(self, result: BacktestResult) -> tuple:
        """Analyze best and worst performance periods"""
        if len(result.equity_curve) < 30:  # Need at least 30 days
            return "Insufficient data", "Insufficient data"
        
        # Simple 30-day rolling returns
        returns_30d = []
        for i in range(30, len(result.equity_curve)):
            start_val = result.equity_curve[i-30]
            end_val = result.equity_curve[i]
            period_return = (end_val - start_val) / start_val
            returns_30d.append(period_return)
        
        if not returns_30d:
            return "Insufficient data", "Insufficient data"
        
        best_return = max(returns_30d)
        worst_return = min(returns_30d)
        
        return f"{best_return:.1%} (30-day)", f"{worst_return:.1%} (30-day)"
    
    def _analyze_monthly_trading(self, trades: List[Trade]) -> str:
        """Analyze monthly trading distribution"""
        if not trades:
            return ""
        
        monthly_counts = {}
        for trade in trades:
            month_key = trade.entry_date.strftime('%Y-%m')
            monthly_counts[month_key] = monthly_counts.get(month_key, 0) + 1
        
        if len(monthly_counts) <= 1:
            return ""
        
        avg_monthly_trades = sum(monthly_counts.values()) / len(monthly_counts)
        max_month = max(monthly_counts.values())
        min_month = min(monthly_counts.values())
        
        return f"Average: {avg_monthly_trades:.1f} trades/month, Range: {min_month}-{max_month}"
    
    def _analyze_consecutive_trades(self, trades: List[Trade]) -> str:
        """Analyze consecutive winning/losing streaks"""
        if not trades:
            return ""
        
        # Sort trades by exit date
        sorted_trades = sorted(trades, key=lambda t: t.exit_date)
        
        current_streak = 0
        max_win_streak = 0
        max_loss_streak = 0
        streak_type = None  # 'win' or 'loss'
        
        for trade in sorted_trades:
            if trade.is_winner:
                if streak_type == 'win':
                    current_streak += 1
                else:
                    max_loss_streak = max(max_loss_streak, current_streak)
                    current_streak = 1
                    streak_type = 'win'
            else:
                if streak_type == 'loss':
                    current_streak += 1
                else:
                    max_win_streak = max(max_win_streak, current_streak)
                    current_streak = 1
                    streak_type = 'loss'
        
        # Update final streak
        if streak_type == 'win':
            max_win_streak = max(max_win_streak, current_streak)
        else:
            max_loss_streak = max(max_loss_streak, current_streak)
        
        return f"Max winning streak: {max_win_streak}, Max losing streak: {max_loss_streak}"

# Convenience functions
def generate_backtest_report(result: BacktestResult) -> BacktestReport:
    """Convenience function to generate a backtest report"""
    generator = BacktestReportGenerator()
    return generator.generate_report(result)


def generate_report(result: BacktestResult) -> Dict[str, Any]:
    """Generate a simple dict report from backtest result"""
    return {
        "total_return": result.total_return,
        "sharpe_ratio": result.sharpe_ratio,
        "max_drawdown": result.max_drawdown,
        "win_rate": result.win_rate,
        "total_trades": result.total_trades,
        "profitable_trades": result.profitable_trades,
        "losing_trades": result.losing_trades,
        "avg_trade_return": result.avg_trade_return,
        "initial_capital": result.initial_capital,
        "final_capital": result.final_capital,
        "start_date": str(result.start_date),
        "end_date": str(result.end_date),
    }


def print_report(report: Dict[str, Any]):
    """Print a formatted report to console"""
    print("\n" + "=" * 50)
    print("  BACKTEST RESULTS")
    print("=" * 50)
    print(f"  Total Return:    {report['total_return']:>10.2%}")
    print(f"  Sharpe Ratio:    {report['sharpe_ratio']:>10.2f}")
    print(f"  Max Drawdown:    {report['max_drawdown']:>10.2%}")
    print(f"  Win Rate:        {report['win_rate']:>10.1%}")
    print(f"  Total Trades:    {report['total_trades']:>10d}")
    print(f"  Initial Capital: ${report['initial_capital']:>10,.0f}")
    print(f"  Final Capital:   ${report['final_capital']:>10,.0f}")
    print("=" * 50)