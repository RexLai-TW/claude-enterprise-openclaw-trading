import { mockPipeline } from '@/lib/mockData';

const statusStyles = {
  done: { dot: 'bg-accent', text: 'text-accent', label: 'Done' },
  running: { dot: 'bg-info pulse-dot', text: 'text-info', label: 'Running' },
  pending: { dot: 'bg-gray-600', text: 'text-gray-600', label: 'Pending' },
};

export default function PipelineView() {
  return (
    <div className="space-y-0">
      {mockPipeline.map((step, i) => {
        const style = statusStyles[step.status];
        const isLast = i === mockPipeline.length - 1;
        return (
          <div key={step.id} className="flex gap-4">
            {/* Timeline */}
            <div className="flex flex-col items-center">
              <div className={`w-3 h-3 rounded-full mt-6 ${style.dot}`} />
              {!isLast && <div className="w-px flex-1 bg-white/10" />}
            </div>
            
            {/* Content */}
            <div className={`card flex-1 mb-3 ${step.status === 'running' ? 'border-info/30' : ''}`}>
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center gap-2">
                  <span className="text-white font-semibold text-sm">{step.name}</span>
                  <span className={`text-xs ${style.text}`}>{style.label}</span>
                </div>
                <span className="text-gray-600 font-mono text-xs">{step.duration}</span>
              </div>
              <p className="text-gray-500 text-xs">{step.detail}</p>
              <p className="text-gray-700 text-xs font-mono mt-1">trace: {step.traceId}</p>
            </div>
          </div>
        );
      })}
    </div>
  );
}
