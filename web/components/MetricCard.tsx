interface MetricCardProps {
  label: string;
  value: string;
  sub?: string;
  color?: 'accent' | 'sell' | 'info' | 'warn' | 'white';
}

const colorMap = {
  accent: 'text-accent',
  sell: 'text-sell',
  info: 'text-info',
  warn: 'text-warn',
  white: 'text-white',
};

export default function MetricCard({ label, value, sub, color = 'white' }: MetricCardProps) {
  return (
    <div className="card">
      <p className="text-gray-500 text-xs uppercase tracking-wider mb-1">{label}</p>
      <p className={`text-2xl font-bold font-mono ${colorMap[color]}`}>{value}</p>
      {sub && <p className="text-gray-500 text-xs mt-1">{sub}</p>}
    </div>
  );
}
