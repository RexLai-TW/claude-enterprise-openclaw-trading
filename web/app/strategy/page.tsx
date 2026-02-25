import StrategyTree from '@/components/StrategyTree';
import Banner from '@/components/Banner';

export default function StrategyPage() {
  return (
    <div className="space-y-6 max-w-3xl mx-auto">
      <div>
        <h1 className="text-2xl font-bold text-white">Strategy Tree</h1>
        <p className="text-gray-500 text-sm mt-1">
          Every rule is visible. White box, not black box. Click nodes to inspect.
        </p>
      </div>
      <Banner />
      <StrategyTree />
      <div className="card bg-info/5 border-info/20">
        <p className="text-info text-sm font-semibold">How to read this tree</p>
        <ul className="text-gray-400 text-xs mt-2 space-y-1">
          <li>• <span className="text-info">Blue</span> nodes are conditions — they evaluate market data</li>
          <li>• <span className="text-accent">Green</span> nodes are buy actions</li>
          <li>• <span className="text-sell">Red</span> nodes are sell/stop-loss actions</li>
          <li>• <span className="text-gray-400">Gray</span> nodes are hold (no action)</li>
          <li>• The tree executes top-down, following TRUE/FALSE branches deterministically</li>
        </ul>
      </div>
    </div>
  );
}
