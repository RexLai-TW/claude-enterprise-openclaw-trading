'use client';
import Link from 'next/link';
import { usePathname } from 'next/navigation';

const navItems = [
  { href: '/', label: 'Strategy Builder' },
  { href: '/strategy', label: 'Tree View' },
  { href: '/backtest', label: 'Backtest' },
  { href: '/signals', label: 'Signals' },
  { href: '/monitor', label: 'Pipeline' },
];

export default function Navbar() {
  const pathname = usePathname();
  return (
    <nav className="border-b border-white/5 bg-bg/80 backdrop-blur-md sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          <Link href="/" className="flex items-center gap-3">
            <span className="text-accent text-xl font-bold glow-green">⚡</span>
            <span className="text-white font-bold text-lg hidden sm:block">Claude Enterprise Trading</span>
            <span className="text-white font-bold text-lg sm:hidden">CET</span>
          </Link>
          <div className="flex items-center gap-1">
            {navItems.map(item => (
              <Link
                key={item.href}
                href={item.href}
                className={`px-3 py-2 rounded-lg text-sm transition-all ${
                  pathname === item.href
                    ? 'bg-accent/10 text-accent'
                    : 'text-gray-400 hover:text-white hover:bg-white/5'
                }`}
              >
                {item.label}
              </Link>
            ))}
          </div>
        </div>
      </div>
    </nav>
  );
}
