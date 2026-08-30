import React, { useState } from 'react';
import {
  Layers,
  Flame,
  Eye,
  Sliders,
  Maximize2,
  ZoomIn,
  Activity,
  Sparkles,
} from 'lucide-react';
import { VisualArtifacts } from '../types';

interface VisualInspectorProps {
  visuals: VisualArtifacts;
  dimensions?: { width: number; height: number };
}

export const VisualInspector: React.FC<VisualInspectorProps> = ({ visuals, dimensions }) => {
  const [activeLayer, setActiveLayer] = useState<'stego' | 'cover' | 'cost' | 'zones' | 'residual'>('stego');
  const [zoom, setZoom] = useState<number>(1);
  const [showSideBySide, setShowSideBySide] = useState<boolean>(false);

  const layers = [
    { id: 'stego', label: 'Stego Output', icon: Eye, desc: 'Watermark-free perceptual identical image' },
    { id: 'cover', label: 'Original Cover', icon: Eye, desc: 'Original unmodified source pixels' },
    { id: 'cost', label: 'CNN Cost Map', icon: Flame, desc: 'Texture & edge complexity heatmap' },
    { id: 'zones', label: 'Adaptive Zones (A/B/C)', icon: Layers, desc: 'Zone A (EMD Green), Zone B (OPAP Indigo), Zone C (Amber)' },
    { id: 'residual', label: 'Amplified Residuals (x25)', icon: Activity, desc: 'Pixel-level modifications amplified for analysis' },
  ];

  const getCurrentImageUrl = () => {
    switch (activeLayer) {
      case 'cover':
        return visuals.coverDataUrl;
      case 'cost':
        return visuals.costMapDataUrl;
      case 'zones':
        return visuals.zoneMapDataUrl;
      case 'residual':
        return visuals.residualDataUrl;
      case 'stego':
      default:
        return visuals.stegoDataUrl;
    }
  };

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-xl overflow-hidden shadow-lg">
      {/* Header with Layer Switcher */}
      <div className="bg-slate-950 p-3.5 border-b border-slate-800 flex flex-wrap items-center justify-between gap-2">
        <div className="flex items-center space-x-1.5 overflow-x-auto">
          {layers.map((layer) => {
            const Icon = layer.icon;
            const isSelected = activeLayer === layer.id;
            return (
              <button
                key={layer.id}
                onClick={() => setActiveLayer(layer.id as any)}
                className={`flex items-center space-x-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-all ${
                  isSelected
                    ? 'bg-indigo-600 text-white shadow-sm'
                    : 'bg-slate-800/80 text-slate-300 hover:bg-slate-700 hover:text-white'
                }`}
              >
                <Icon className="w-3.5 h-3.5" />
                <span>{layer.label}</span>
              </button>
            );
          })}
        </div>

        {/* View Controls */}
        <div className="flex items-center space-x-2 text-xs">
          <button
            onClick={() => setShowSideBySide(!showSideBySide)}
            className={`px-2.5 py-1.5 rounded-lg border font-medium transition-colors ${
              showSideBySide
                ? 'bg-indigo-500/20 text-indigo-300 border-indigo-500/40'
                : 'bg-slate-800 text-slate-300 border-slate-700 hover:bg-slate-700'
            }`}
          >
            {showSideBySide ? 'Single View' : 'Side-by-Side Split'}
          </button>

          <div className="flex items-center space-x-1 bg-slate-800 rounded-lg p-0.5 border border-slate-700">
            <button
              onClick={() => setZoom(Math.max(0.5, zoom - 0.25))}
              className="px-2 py-1 hover:bg-slate-700 rounded text-slate-300"
              title="Zoom out"
            >
              -
            </button>
            <span className="px-1 font-mono text-slate-400">{Math.round(zoom * 100)}%</span>
            <button
              onClick={() => setZoom(Math.min(3, zoom + 0.25))}
              className="px-2 py-1 hover:bg-slate-700 rounded text-slate-300"
              title="Zoom in"
            >
              +
            </button>
          </div>
        </div>
      </div>

      {/* Layer Description Banner */}
      <div className="bg-slate-900/90 px-4 py-2 border-b border-slate-800/60 flex items-center justify-between text-xs text-slate-400">
        <div className="flex items-center space-x-2">
          <span className="font-semibold text-slate-200">Active Map:</span>
          <span>{layers.find((l) => l.id === activeLayer)?.desc}</span>
        </div>
        {dimensions && (
          <div className="font-mono text-slate-400">
            {dimensions.width} × {dimensions.height} px
          </div>
        )}
      </div>

      {/* Main Canvas / Image Area */}
      <div className="p-4 bg-slate-950/80 flex items-center justify-center min-h-[380px] overflow-auto">
        {showSideBySide ? (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 w-full">
            <div className="flex flex-col items-center">
              <div className="text-xs font-medium text-slate-400 mb-1.5 flex items-center space-x-1">
                <span>Original Cover</span>
              </div>
              <div className="border border-slate-800 rounded-lg overflow-hidden bg-slate-900">
                <img
                  src={visuals.coverDataUrl}
                  alt="Cover"
                  className="max-h-[340px] object-contain rounded"
                  style={{ transform: `scale(${zoom})`, transformOrigin: 'top center' }}
                />
              </div>
            </div>
            <div className="flex flex-col items-center">
              <div className="text-xs font-medium text-slate-400 mb-1.5 flex items-center space-x-1">
                <span>Stego / Active Map</span>
              </div>
              <div className="border border-slate-800 rounded-lg overflow-hidden bg-slate-900">
                <img
                  src={getCurrentImageUrl()}
                  alt="Stego layer"
                  className="max-h-[340px] object-contain rounded"
                  style={{ transform: `scale(${zoom})`, transformOrigin: 'top center' }}
                />
              </div>
            </div>
          </div>
        ) : (
          <div className="flex flex-col items-center justify-center">
            <div className="border border-slate-800 rounded-lg overflow-hidden bg-slate-900 shadow-inner">
              <img
                src={getCurrentImageUrl()}
                alt={activeLayer}
                className="max-h-[420px] max-w-full object-contain rounded"
                style={{ transform: `scale(${zoom})`, transformOrigin: 'center center' }}
              />
            </div>
          </div>
        )}
      </div>

      {/* Color Legend (for Cost and Zones) */}
      {activeLayer === 'zones' && (
        <div className="bg-slate-900 p-3 border-t border-slate-800 flex items-center justify-around text-xs">
          <div className="flex items-center space-x-2">
            <span className="w-3.5 h-3.5 rounded bg-emerald-500 inline-block shadow-sm"></span>
            <span className="text-slate-300 font-medium">Zone A (High Texture / EMD n=2,3)</span>
          </div>
          <div className="flex items-center space-x-2">
            <span className="w-3.5 h-3.5 rounded bg-indigo-500 inline-block shadow-sm"></span>
            <span className="text-slate-300 font-medium">Zone B (Medium / OPAP k_b)</span>
          </div>
          <div className="flex items-center space-x-2">
            <span className="w-3.5 h-3.5 rounded bg-amber-500 inline-block shadow-sm"></span>
            <span className="text-slate-300 font-medium">Zone C (Smooth / OPAP k_c)</span>
          </div>
        </div>
      )}

      {activeLayer === 'cost' && (
        <div className="bg-slate-900 p-3 border-t border-slate-800 flex items-center justify-between text-xs text-slate-300">
          <span>Low Distortion Risk (Edges / Texture)</span>
          <div className="w-48 h-3 rounded-full bg-gradient-to-r from-purple-900 via-red-600 to-yellow-400 shadow-inner"></div>
          <span>High Distortion Risk (Smooth Regions)</span>
        </div>
      )}
    </div>
  );
};
