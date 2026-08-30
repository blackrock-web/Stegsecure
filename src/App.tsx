import React, { useState, useEffect } from 'react';
import { Navbar } from './components/Navbar';
import { EncodeStudio } from './components/EncodeStudio';
import { DecodeStudio } from './components/DecodeStudio';
import { SteganalysisScanner } from './components/SteganalysisScanner';
import { BenchmarkLab } from './components/BenchmarkLab';
import { ComparisonSuite } from './components/ComparisonSuite';
import { BatchLab } from './components/BatchLab';
import { QuickstartModal } from './components/QuickstartModal';
import { SystemHealth } from './types';
import { fetchHealth } from './lib/api';

export function App() {
  const [activeTab, setActiveTab] = useState<string>('encode');
  const [systemHealth, setSystemHealth] = useState<SystemHealth | null>(null);
  const [isDocsOpen, setIsDocsOpen] = useState<boolean>(false);

  useEffect(() => {
    fetchHealth().then((health) => setSystemHealth(health));
  }, []);

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col font-sans selection:bg-indigo-500 selection:text-white">
      {/* Navigation Bar */}
      <Navbar
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        systemHealth={systemHealth}
        onOpenDocs={() => setIsDocsOpen(true)}
      />

      {/* Main View Router */}
      <main className="flex-1 pb-12">
        {activeTab === 'encode' && <EncodeStudio />}
        {activeTab === 'decode' && <DecodeStudio />}
        {activeTab === 'steganalysis' && <SteganalysisScanner />}
        {activeTab === 'benchmark' && <BenchmarkLab />}
        {activeTab === 'compare' && <ComparisonSuite />}
        {activeTab === 'batch' && <BatchLab />}
      </main>

      {/* Research & Documentation Modal */}
      <QuickstartModal isOpen={isDocsOpen} onClose={() => setIsDocsOpen(false)} />

      {/* Persistent Footer */}
      <footer className="border-t border-slate-900 bg-slate-950 py-6 text-center text-xs text-slate-500">
        <div className="max-w-7xl mx-auto px-4 flex flex-col sm:flex-row items-center justify-between gap-2">
          <span>SecureStegVault v3.2 — CNN Adaptive EMD-OPAP Image Steganography Research Platform</span>
          <span className="font-mono text-slate-600">AES-256-GCM • EMD Base-5/7 • Optimal Pixel Adjustment Process</span>
        </div>
      </footer>
    </div>
  );
}

export default App;
