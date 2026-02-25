import { mockTrades } from '@/lib/mockData';

export default function TradeLog() {
  return (
    <div className="card overflow-x-auto">
      <h3 className="text-sm text-gray-400 mb-4">Trade History</h3>
      <table className="w-full text-sm">
        <thead>
          <tr className="text-gray-600 text-xs uppercase border-b border-white/5">
            <th className="text-left py-2">Date</th>
            <th className="text-left py-2">Type</th>
            <th className="text-left py-2">Symbol</th>
            <th className="text-right py-2">Amount</th>
            <th className="text-right py-2">Price</th>
            <th className="text-right py-2">P&L</th>
            <th className="text-left py-2">Reason</th>
          </tr>
        </thead>
        <tbody>
          {mockTrades.map((trade, i) => (
            <tr key={i} className="border-b border-white/5 hover:bg-white/5 transition-colors">
              <td className="py-2.5 font-mono text-gray-400 text-xs">{trade.date}</td>
              <td className="py-2.5">
                <span className={`text-xs font-bold px-2 py-0.5 rounded ${
                  trade.type === 'BUY' ? 'bg-accent/10 text-accent' :
                  trade.type === 'SELL' ? 'bg-sell/10 text-sell' :
                  'bg-warn/10 text-warn'
                }`}>
                  {trade.type}
                </span>
              </td>
              <td className="py-2.5 text-white font-mono text-xs">{trade.symbol}</td>
              <td className="py-2.5 text-right font-mono text-xs text-gray-300">{trade.amount}</td>
              <td className="py-2.5 text-right font-mono text-xs text-gray-300">${trade.price.toLocaleString()}</td>
              <td className={`py-2.5 text-right font-mono text-xs ${
                trade.pnl === null ? 'text-gray-600' :
                trade.pnl >= 0 ? 'text-accent' : 'text-sell'
              }`}>
                {trade.pnl === null ? '-' : trade.pnl >= 0 ? `+$${trade.pnl.toLocaleString()}` : `-$${Math.abs(trade.pnl).toLocaleString()}`}
              </td>
              <td className="py-2.5 text-gray-500 text-xs max-w-[200px] truncate">{trade.reason}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
