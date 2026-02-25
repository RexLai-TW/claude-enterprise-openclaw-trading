'use client';
import { useState } from 'react';
import { examplePrompts } from '@/lib/mockData';

export default function StrategyInput() {
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [generated, setGenerated] = useState(false);

  const handleGenerate = () => {
    if (!input.trim()) return;
    setLoading(true);
    setTimeout(() => {
      setLoading(false);
      setGenerated(true);
    }, 2000);
  };

  return (
    <div className="space-y-6">
      <div className="card">
        <label className="block text-sm text-gray-400 mb-3">
          Describe your trading idea in natural language
        </label>
        <textarea
          value={input}
          onChange={(e) => { setInput(e.target.value); setGenerated(false); }}
          placeholder="e.g. Buy BTC when Fear & Greed index drops below 20, accumulate in 3 batches 4 hours apart, stop loss at 8%..."
          className="w-full bg-bg border border-white/10 rounded-lg p-4 text-white placeholder-gray-600 focus:outline-none focus:border-accent/50 resize-none h-32 font-mono text-sm"
        />
        <div className="flex flex-wrap gap-2 mt-3">
          {examplePrompts.map((prompt, i) => (
            <button
              key={i}
              onClick={() => { setInput(prompt); setGenerated(false); }}
              className="text-xs px-3 py-1.5 rounded-full bg-white/5 text-gray-400 hover:bg-accent/10 hover:text-accent transition-all border border-white/5"
            >
              {prompt.slice(0, 50)}{prompt.length > 50 ? '...' : ''}
            </button>
          ))}
        </div>
      </div>
      
      <button
        onClick={handleGenerate}
        disabled={!input.trim() || loading}
        className={`w-full py-4 rounded-xl font-bold text-lg transition-all ${
          loading
            ? 'bg-accent/20 text-accent/50 cursor-wait'
            : input.trim()
              ? 'bg-accent text-bg hover:bg-accent/90 hover:shadow-lg hover:shadow-accent/20'
              : 'bg-white/5 text-gray-600 cursor-not-allowed'
        }`}
      >
        {loading ? (
          <span className="flex items-center justify-center gap-2">
            <span className="pulse-dot">⚡</span> Generating Strategy Tree...
          </span>
        ) : generated ? (
          '✅ Strategy Generated — View Tree →'
        ) : (
          '⚡ Generate Strategy'
        )}
      </button>
      
      {generated && (
        <div className="card fade-in border-accent/20">
          <div className="flex items-center gap-2 mb-3">
            <span className="text-accent">✓</span>
            <span className="text-white font-semibold">Strategy tree generated successfully</span>
          </div>
          <div className="grid grid-cols-3 gap-4 text-center">
            <div>
              <p className="text-2xl font-mono text-accent">7</p>
              <p className="text-xs text-gray-500">Nodes</p>
            </div>
            <div>
              <p className="text-2xl font-mono text-info">3</p>
              <p className="text-xs text-gray-500">Conditions</p>
            </div>
            <div>
              <p className="text-2xl font-mono text-warn">4</p>
              <p className="text-xs text-gray-500">Actions</p>
            </div>
          </div>
          <div className="mt-4 flex gap-2">
            <a href="/strategy" className="flex-1 text-center py-2 rounded-lg bg-accent/10 text-accent text-sm hover:bg-accent/20 transition-all">
              View Strategy Tree →
            </a>
            <a href="/backtest" className="flex-1 text-center py-2 rounded-lg bg-info/10 text-info text-sm hover:bg-info/20 transition-all">
              Run Backtest →
            </a>
          </div>
        </div>
      )}
    </div>
  );
}
