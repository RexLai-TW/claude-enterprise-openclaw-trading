import PipelineView from '@/components/PipelineView';

export default function MonitorPage() {
  return (
    <div className="space-y-6 max-w-3xl mx-auto">
      <div>
        <h1 className="text-2xl font-bold text-white">Pipeline Monitor</h1>
        <p className="text-gray-500 text-sm mt-1">
          Full pipeline tracing via OpenTelemetry. Every step auditable.
        </p>
      </div>

      <div className="grid grid-cols-2 gap-3">
        <div className="card">
          <p className="text-xs text-gray-500 uppercase">Active Agents</p>
          <p className="text-2xl font-bold font-mono text-accent mt-1">3</p>
          <p className="text-xs text-gray-600 mt-1">BTC Fear Greed • ETH Trend • BTC RSI</p>
        </div>
        <div className="card">
          <p className="text-xs text-gray-500 uppercase">Pipeline Status</p>
          <div className="flex items-center gap-2 mt-2">
            <div className="w-2 h-2 rounded-full bg-info pulse-dot" />
            <span className="text-info font-semibold">Running</span>
          </div>
          <p className="text-xs text-gray-600 mt-1">Trace ID: trace-7f3a2b</p>
        </div>
      </div>

      <PipelineView />

      <div className="card bg-white/5">
        <p className="text-sm text-gray-400 mb-2">OpenTelemetry Export</p>
        <div className="bg-bg rounded-lg p-3 font-mono text-xs text-gray-500 overflow-x-auto">
          <p>span: orchestrator.run → agent.btc-fear-greed</p>
          <p>  ├── data_fetch: 1.2s [FactSet MCP]</p>
          <p>  ├── strategy_eval: 3.8s [Claude API → 7 nodes]</p>
          <p>  ├── backtest: 2.1s [47 trades, Sharpe 1.68]</p>
          <p>  ├── signal_gen: 0.4s <span className="text-info">[running]</span></p>
          <p>  └── output: - <span className="text-gray-700">[pending]</span></p>
        </div>
      </div>
    </div>
  );
}
