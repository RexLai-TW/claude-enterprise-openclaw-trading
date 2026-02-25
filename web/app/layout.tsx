import type { Metadata } from 'next';
import './globals.css';
import Navbar from '@/components/Navbar';

export const metadata: Metadata = {
  title: 'Claude Enterprise Trading',
  description: 'Turn trading ideas into executable strategy trees using Claude Enterprise + OpenClaw',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-bg text-gray-200 antialiased">
        <Navbar />
        <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          {children}
        </main>
        <footer className="border-t border-white/5 mt-16 py-8 text-center text-gray-600 text-sm">
          Built with{' '}
          <a href="https://anthropic.com" className="text-gray-400 hover:text-white transition-colors">Claude Enterprise</a>
          {' + '}
          <a href="https://github.com/openclaw/openclaw" className="text-gray-400 hover:text-white transition-colors">OpenClaw</a>
          {' • '}
          <a href="https://github.com/jerrylearnscoding/claude-enterprise-openclaw-trading" className="text-accent/60 hover:text-accent transition-colors">
            GitHub ↗
          </a>
        </footer>
      </body>
    </html>
  );
}
