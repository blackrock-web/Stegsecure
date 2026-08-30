import React from 'react';
import { Lock, ShieldCheck, Sliders, BookOpen, KeyRound, BarChart3, GitCompare, Layers } from 'lucide-react';

export type AppTab = 'encode' | 'decode' | 'info' | 'benchmark' | 'compare' | 'batch';

interface HeaderProps {
  activeTab: AppTab;
  setActiveTab: (tab: AppTab) => void;
  onOpenConfig: () => void;
}

export const Header: React.FC<HeaderProps> = ({ activeTab, setActiveTab, onOpenConfig }) => {
  const tabBtn = (
    id: AppTab,
    label: string,
    icon: React.ReactNode,
    activeColor: string
  ) => (
    <button
      onClick={() => setActiveTab(id)}
      className={`flex items-center gap-1.5 px-3 py-2 text-sm font-semibold rounded-xl transition-all ${
        activeTab === id
          ? `bg-white ${activeColor} shadow-xs border border-pink-200`
          : 'text-purple-700 hover:text-purple-950 hover:bg-pink-100/50'
      }`}
    >
      {icon}
      <span className="hidden lg:inline">{label}</span>
    </button>
  );

  return (
    <header className="sticky top-0 z-30 border-b border-pink-200/80 bg-white/80 backdrop-blur-md shadow-xs">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-18 flex items-center justify-between gap-3">
        {/* Brand Logo */}
        <div className="flex items-center gap-3 flex-shrink-0">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-pink-500 to-purple-600 flex items-center justify-center text-white shadow-md shadow-pink-200">
            <ShieldCheck className="w-6 h-6" />
          </div>
          <div className="hidden sm:block">
            <div className="flex items-center gap-2">
              <h1 className="text-xl font-bold tracking-tight text-purple-950">SecureStegVault</h1>
              <span className="px-2 py-0.5 text-xs font-semibold rounded-full bg-pink-100 text-pink-800 border border-pink-200">
                v3.2 · AES-256 + CNN
              </span>
            </div>
            <p className="text-xs text-purple-700 font-medium">
              Classical EMD/OPAP Steganography guided by CNN Feature Cost Mapping
            </p>
          </div>
        </div>

        {/* Navigation Tabs */}
        <nav className="flex items-center gap-0.5 bg-pink-50/80 p-1 rounded-2xl border border-pink-200/60 overflow-x-auto">
          {tabBtn('encode', 'Hide Payload', <Lock className="w-4 h-4 text-pink-600" />, 'text-pink-950')}
          {tabBtn('decode', 'Extract', <KeyRound className="w-4 h-4 text-purple-600" />, 'text-purple-950')}
          {tabBtn('batch', 'Batch Lab', <Layers className="w-4 h-4 text-fuchsia-600" />, 'text-fuchsia-950')}
          {tabBtn('benchmark', 'Benchmark', <BarChart3 className="w-4 h-4 text-pink-500" />, 'text-purple-950')}
          {tabBtn('compare', 'Compare', <GitCompare className="w-4 h-4 text-purple-500" />, 'text-purple-950')}
          {tabBtn('info', 'Docs', <BookOpen className="w-4 h-4 text-pink-500" />, 'text-purple-950')}
        </nav>

        {/* Config Settings Button */}
        <div className="flex items-center gap-2 flex-shrink-0">
          <button
            onClick={onOpenConfig}
            className="flex items-center gap-2 px-3.5 py-2 text-sm font-medium text-purple-900 bg-purple-50 hover:bg-purple-100/80 border border-purple-200 rounded-xl transition-colors shadow-2xs"
            title="Configure Algorithm Hyperparameters"
          >
            <Sliders className="w-4 h-4 text-purple-600" />
            <span className="hidden sm:inline font-semibold">Config</span>
          </button>
        </div>
      </div>
    </header>
  );
};
