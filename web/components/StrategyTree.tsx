'use client';
import { mockStrategyTree } from '@/lib/mockData';

const colorClasses: Record<string, string> = {
  accent: 'border-accent/40 bg-accent/5',
  sell: 'border-sell/40 bg-sell/5',
  info: 'border-info/40 bg-info/5',
  warn: 'border-warn/40 bg-warn/5',
  gray: 'border-white/10 bg-white/5',
};

const badgeClasses: Record<string, string> = {
  condition: 'bg-info/20 text-info',
  action: 'bg-accent/20 text-accent',
};

export default function StrategyTree() {
  const tree = mockStrategyTree;
  const nodeMap = Object.fromEntries(tree.nodes.map(n => [n.id, n]));
  
  function renderNode(id: string, depth: number = 0): JSX.Element | null {
    const node = nodeMap[id];
    if (!node) return null;
    
    return (
      <div className={`fade-in`} style={{ animationDelay: `${depth * 0.1}s` }}>
        <div className={`card border ${colorClasses[node.color]} mb-3`}>
          <div className="flex items-center justify-between mb-2">
            <span className={`text-xs px-2 py-0.5 rounded-full font-mono ${badgeClasses[node.type]}`}>
              {node.type.toUpperCase()}
            </span>
            <span className="text-xs text-gray-600 font-mono">{node.id}</span>
          </div>
          <p className="text-white font-semibold text-sm">{node.label}</p>
          {node.type === 'condition' && (
            <p className="text-gray-500 text-xs mt-1 font-mono">{node.condition}</p>
          )}
          {node.type === 'action' && node.size && (
            <p className="text-gray-500 text-xs mt-1">Size: {node.size}</p>
          )}
        </div>
        
        {node.type === 'condition' && (
          <div className="ml-6 pl-4 border-l border-white/10 space-y-2">
            {node.trueBranch && (
              <div>
                <span className="text-xs text-accent mb-1 block">✓ TRUE</span>
                {renderNode(node.trueBranch, depth + 1)}
              </div>
            )}
            {node.falseBranch && (
              <div>
                <span className="text-xs text-sell mb-1 block">✗ FALSE</span>
                {renderNode(node.falseBranch, depth + 1)}
              </div>
            )}
          </div>
        )}
      </div>
    );
  }
  
  return (
    <div>
      <div className="mb-6">
        <h2 className="text-xl font-bold text-white">{tree.name}</h2>
        <p className="text-gray-500 text-sm mt-1">{tree.description}</p>
      </div>
      {renderNode('root')}
    </div>
  );
}
