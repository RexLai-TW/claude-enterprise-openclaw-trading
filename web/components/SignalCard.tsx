import { mockSignals } from '@/lib/mockData';

export default function SignalCard() {
  return (
    <div className="space-y-3">
      {mockSignals.map((signal, i) => (
        <div key={signal.id} className={`card fade-in`} style={{ animationDelay: `${i * 0.1}s` }}>
          <div className="flex items-start justify-between">
            <div className="flex items-center gap-3">
              <span className={`text-xs font-bold px-3 py-1 rounded-lg ${
                signal.type === 'BUY' ? 'bg-accent/10 text-accent border border-accent/20' :
                signal.type === 'SELL' ? 'bg-sell/10 text-sell border border-sell/20' :
                'bg-white/5 text-gray-400 border border-white/10'
              }`}>
                {signal.type}
              </span>
              <div>
                <span className="text-white font-mono text-sm">{signal.symbol}</span>
                <span className="text-gray-600 mx-2">•</span>
                <span className="text-gray-400 text-sm">{signal.amount}</span>
              </div>
            </div>
            <div className="text-right">
              <div className="flex items-center gap-1">
                <div className={`w-1.5 h-1.5 rounded-full ${
                  signal.confidence > 0.9 ? 'bg-accent' :
                  signal.confidence > 0.8 ? 'bg-info' : 'bg-warn'
                }`} />
                <span className="text-xs text-gray-500 font-mono">{(signal.confidence * 100).toFixed(0)}%</span>
              </div>
            </div>
          </div>
          <p className="text-gray-400 text-sm mt-2">{signal.reason}</p>
          <p className="text-gray-600 text-xs mt-2 font-mono">{signal.timestamp}</p>
        </div>
      ))}
    </div>
  );
}
