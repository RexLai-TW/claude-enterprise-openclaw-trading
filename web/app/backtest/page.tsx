import MetricCard from '@/components/MetricCard';
import EquityCurve from '@/components/EquityCurve';
import TradeLog from '@/components/TradeLog';
import { mockMetrics } from '@/lib/mockData';

export default function BacktestPage() {
  const m = mockMetrics;
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-white">Backtest Results</h1>
        <p className="text-gray-500 text-sm mt-1">
          BTC Fear & Greed Accumulation • 365 days • $100,000 initial capital
        </p>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
        <MetricCard label="Total Return" value={`${(m.totalReturn * 100).toFixed(1)}%`} sub={`$${m.initialCapital.toLocaleString()} → $${m.finalCapital.toLocaleString()}`} color="accent" />
        <MetricCard label="Sharpe Ratio" value={m.sharpeRatio.toFixed(2)} sub="Risk-adjusted return" color="info" />
        <MetricCard label="Max Drawdown" value={`${(m.maxDrawdown * 100).toFixed(1)}%`} sub="Worst peak-to-trough" color="sell" />
        <MetricCard label="Win Rate" value={`${(m.winRate * 100).toFixed(0)}%`} sub={`${m.profitableTrades}W / ${m.losingTrades}L`} color="accent" />
        <MetricCard label="Total Trades" value={m.totalTrades.toString()} sub="Over 365 days" color="white" />
      </div>

      <EquityCurve />
      <TradeLog />
    </div>
  );
}
