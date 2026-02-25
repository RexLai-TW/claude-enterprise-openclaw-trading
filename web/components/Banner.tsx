export default function Banner() {
  return (
    <div className="bg-accent/5 border border-accent/20 rounded-xl px-6 py-4 flex items-center gap-3">
      <span className="text-2xl">🔒</span>
      <div>
        <p className="text-accent font-semibold text-sm">AI Never Touches Your Money</p>
        <p className="text-gray-400 text-xs">Generates deterministic strategy scripts — you control the switch. Every step auditable via OpenTelemetry.</p>
      </div>
    </div>
  );
}
