import StrategyInput from '@/components/StrategyInput';
import Banner from '@/components/Banner';

export default function Home() {
  return (
    <div className="space-y-8 max-w-3xl mx-auto">
      <div className="text-center space-y-3 fade-in">
        <h1 className="text-4xl font-bold text-white glow-green">
          ⚡ Claude Enterprise Trading
        </h1>
        <p className="text-gray-400 text-lg">
          Describe your trading idea → AI generates a deterministic strategy tree → Backtest → Execute
        </p>
        <p className="text-gray-600 text-sm">
          Agent-first architecture powered by Claude Enterprise + OpenClaw
        </p>
      </div>
      
      <Banner />
      <StrategyInput />

      <div className="grid grid-cols-3 gap-4 text-center fade-in fade-in-delay-3">
        <div className="card">
          <p className="text-3xl font-bold font-mono text-accent">8K+</p>
          <p className="text-xs text-gray-500 mt-1">Lines of Python</p>
        </div>
        <div className="card">
          <p className="text-3xl font-bold font-mono text-info">33</p>
          <p className="text-xs text-gray-500 mt-1">Files</p>
        </div>
        <div className="card">
          <p className="text-3xl font-bold font-mono text-warn">MIT</p>
          <p className="text-xs text-gray-500 mt-1">Licensed</p>
        </div>
      </div>
    </div>
  );
}
