'use client';
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, Area, AreaChart } from 'recharts';
import { mockEquityCurve } from '@/lib/mockData';

export default function EquityCurve() {
  return (
    <div className="card">
      <h3 className="text-sm text-gray-400 mb-4">Portfolio Value</h3>
      <div className="h-64">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={mockEquityCurve}>
            <defs>
              <linearGradient id="colorValue" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#00ff88" stopOpacity={0.2} />
                <stop offset="100%" stopColor="#00ff88" stopOpacity={0} />
              </linearGradient>
            </defs>
            <XAxis
              dataKey="date"
              tick={{ fill: '#666', fontSize: 10 }}
              tickFormatter={(v) => v.slice(5)}
              interval={60}
              axisLine={{ stroke: '#222' }}
            />
            <YAxis
              tick={{ fill: '#666', fontSize: 10 }}
              tickFormatter={(v) => `$${(v / 1000).toFixed(0)}k`}
              axisLine={{ stroke: '#222' }}
              domain={['auto', 'auto']}
            />
            <Tooltip
              contentStyle={{ background: '#1a1a2e', border: '1px solid #333', borderRadius: 8 }}
              labelStyle={{ color: '#888' }}
              formatter={(value: number) => [`$${value.toLocaleString()}`, 'Portfolio']}
            />
            <Area
              type="monotone"
              dataKey="value"
              stroke="#00ff88"
              strokeWidth={2}
              fill="url(#colorValue)"
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
