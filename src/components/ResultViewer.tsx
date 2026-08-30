import React, { useState } from 'react';
import { EncodeResponse } from '../types';
import { ZoomCanvas } from './ZoomCanvas';
import { Download, Eye, Layers, ShieldCheck, Sparkles, Image as ImageIcon } from 'lucide-react';

interface ResultViewerProps {
  coverUrl: string;
  resultData: EncodeResponse;
}

export const ResultViewer: React.FC<ResultViewerProps> = ({ coverUrl, resultData }) => {
  const [activeView, setActiveView] = useState<'sideBySide' | 'highlights' | 'rgbBits' | 'heatmap' | 'mask' | 'zoneMap' | 'gradOverlay'>('sideBySide');
  const [showHeatmapOverlay, setShowHeatmapOverlay] = useState(true);

  const visuals = resultData?.visuals || ({} as any);
  const metrics = resultData?.metrics;
  const maskUrl = visuals.mask_b64 || visuals.binary_mask_b64 || '';
  const rgbBitsUrl = visuals.rgb_bits_b64 || '';
  const stegoUrl = visuals.stego_b64 || '';

  const downloadImage = (dataUrl: string, filename: string) => {
    const link = document.createElement('a');
    link.href = dataUrl;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <div className="space-y-6 animate-fadeIn">
      {/* View Switcher Header */}
      <div className="flex flex-wrap items-center justify-between gap-3 bg-white/90 p-3 rounded-2xl border border-pink-200 shadow-xs">
        <div className="flex items-center gap-2">
          <Sparkles className="w-5 h-5 text-pink-600" />
          <h2 className="text-base font-bold text-purple-950">Steganography Results &amp; Visual Diff</h2>
        </div>

        {/* View Mode Buttons */}
        <div className="flex flex-wrap items-center gap-1.5 bg-pink-50 p-1 rounded-xl border border-pink-200">
          <button
            onClick={() => setActiveView('sideBySide')}
            className={`px-3 py-1.5 text-xs font-bold rounded-lg transition-all ${
              activeView === 'sideBySide'
                ? 'bg-white text-pink-950 shadow-2xs border border-pink-200'
                : 'text-purple-700 hover:text-purple-950'
            }`}
          >
            Side-by-Side
          </button>
          {visuals.highlight_overlay_b64 && (
            <button
              onClick={() => setActiveView('highlights')}
              className={`px-3 py-1.5 text-xs font-bold rounded-lg transition-all ${
                activeView === 'highlights'
                  ? 'bg-white text-pink-950 shadow-2xs border border-pink-200'
                  : 'text-purple-700 hover:text-purple-950'
              }`}
            >
              ✨ Embedded Highlights
            </button>
          )}
          {rgbBitsUrl && (
            <button
              onClick={() => setActiveView('rgbBits')}
              className={`px-3 py-1.5 text-xs font-bold rounded-lg transition-all ${
                activeView === 'rgbBits'
                  ? 'bg-white text-pink-950 shadow-2xs border border-pink-200'
                  : 'text-purple-700 hover:text-purple-950'
              }`}
            >
              RGB Bits Mode
            </button>
          )}
          <button
            onClick={() => setActiveView('heatmap')}
            className={`px-3 py-1.5 text-xs font-bold rounded-lg transition-all ${
              activeView === 'heatmap'
                ? 'bg-white text-pink-950 shadow-2xs border border-pink-200'
                : 'text-purple-700 hover:text-purple-950'
            }`}
          >
            Diff Heatmap
          </button>
          <button
            onClick={() => setActiveView('mask')}
            className={`px-3 py-1.5 text-xs font-bold rounded-lg transition-all ${
              activeView === 'mask'
                ? 'bg-white text-pink-950 shadow-2xs border border-pink-200'
                : 'text-purple-700 hover:text-purple-950'
            }`}
          >
            Binary Mask
          </button>
          <button
            onClick={() => setActiveView('zoneMap')}
            className={`px-3 py-1.5 text-xs font-bold rounded-lg transition-all ${
              activeView === 'zoneMap'
                ? 'bg-white text-pink-950 shadow-2xs border border-pink-200'
                : 'text-purple-700 hover:text-purple-950'
            }`}
          >
            CNN Zone Map
          </button>
          {visuals.gradient_overlay_b64 && (
            <button
              onClick={() => setActiveView('gradOverlay')}
              className={`px-3 py-1.5 text-xs font-bold rounded-lg transition-all ${
                activeView === 'gradOverlay'
                  ? 'bg-white text-pink-950 shadow-2xs border border-pink-200'
                  : 'text-purple-700 hover:text-purple-950'
              }`}
            >
              Adversarial Gradient
            </button>
          )}
        </div>

        {/* Download Stego Button */}
        <button
          onClick={() => downloadImage(stegoUrl, 'stego_image.png')}
          className="flex items-center gap-2 px-4 py-2 text-xs font-bold text-white bg-pink-600 hover:bg-pink-700 rounded-xl transition-all shadow-md shadow-pink-200 active:scale-95"
        >
          <Download className="w-4 h-4" />
          <span>Download Stego Image (PNG)</span>
        </button>
      </div>

      {/* Primary Display View */}
      {activeView === 'sideBySide' && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="p-4 rounded-2xl border border-pink-200 bg-white shadow-xs space-y-2">
            <div className="flex items-center justify-between text-xs font-bold text-purple-950">
              <span>Original Cover Image</span>
              <span className="px-2 py-0.5 rounded-full bg-purple-100 text-purple-800">Lossless</span>
            </div>
            <div className="rounded-xl border border-pink-100 overflow-hidden bg-pink-50/30 aspect-4/3 flex items-center justify-center">
              <img src={coverUrl} alt="Cover Image" className="max-h-full max-w-full object-contain" />
            </div>
          </div>

          <div className="p-4 rounded-2xl border border-pink-200 bg-white shadow-xs space-y-2">
            <div className="flex items-center justify-between text-xs font-bold text-purple-950">
              <span>Output Stego Image (Payload Hidden)</span>
              <span className="px-2 py-0.5 rounded-full bg-pink-100 text-pink-800 border border-pink-200">
                AES-256-GCM Embedded
              </span>
            </div>
            <div className="rounded-xl border border-pink-100 overflow-hidden bg-pink-50/30 aspect-4/3 flex items-center justify-center">
              <img src={stegoUrl} alt="Stego Image" className="max-h-full max-w-full object-contain" />
            </div>
          </div>
        </div>
      )}

      {activeView === 'highlights' && visuals.highlight_overlay_b64 && (
        <div className="p-4 rounded-2xl border border-pink-200 bg-white shadow-xs space-y-3">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-bold text-purple-950 flex items-center gap-2">
              <Sparkles className="w-4 h-4 text-amber-500" />
              <span>Embedded Output Image with Bright Pixel Highlights</span>
            </h3>
            <button
              onClick={() => downloadImage(visuals.highlight_overlay_b64!, 'stego_embedded_highlights.png')}
              className="text-xs text-pink-700 font-semibold hover:underline flex items-center gap-1"
            >
              <Download className="w-3.5 h-3.5" />
              Download Highlights Map
            </button>
          </div>
          <p className="text-xs text-purple-600">
            Electric gold/yellow glowing points highlight the exact pixels where payload bits are embedded in the stego output image.
          </p>
          <div className="rounded-xl border border-pink-100 overflow-hidden bg-purple-950 flex items-center justify-center max-h-112 p-2">
            <img src={visuals.highlight_overlay_b64} alt="Embedded Bright Highlights" className="max-h-112 object-contain rounded-lg" />
          </div>
        </div>
      )}

      {activeView === 'rgbBits' && rgbBitsUrl && (
        <div className="p-4 rounded-2xl border border-pink-200 bg-white shadow-xs space-y-3">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-bold text-purple-950 flex items-center gap-2">
              <Layers className="w-4 h-4 text-pink-600" />
              <span>RGB Bits Mode — Bright Pixels Where Data Bits Are Stored</span>
            </h3>
            <button
              onClick={() => downloadImage(rgbBitsUrl, 'stego_rgb_bits_map.png')}
              className="text-xs text-pink-700 font-semibold hover:underline flex items-center gap-1"
            >
              <Download className="w-3.5 h-3.5" />
              Download RGB Bits Map
            </button>
          </div>
          <p className="text-xs text-purple-600">
            Pure channel colours reveal exactly which colour plane holds the secret bits:
            <span className="font-bold text-red-600"> Red</span> = R-channel modified,
            <span className="font-bold text-green-600"> Green</span> = G-channel modified,
            <span className="font-bold text-blue-600"> Blue</span> = B-channel modified.
            Combinations (yellow / cyan / magenta / white) mean multiple channels were changed at that pixel.
          </p>
          <div className="rounded-xl border border-pink-100 overflow-hidden bg-black flex items-center justify-center max-h-112 p-2">
            <img src={rgbBitsUrl} alt="RGB Bits Map" className="max-h-112 object-contain rounded-lg" />
          </div>
        </div>
      )}

      {activeView === 'heatmap' && (
        <div className="p-4 rounded-2xl border border-pink-200 bg-white shadow-xs space-y-3">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-bold text-purple-950">Amplified Diff Heatmap Overlay</h3>
            <button
              onClick={() => downloadImage(visuals.heatmap_b64, 'stego_diff_heatmap.png')}
              className="text-xs text-pink-700 font-semibold hover:underline flex items-center gap-1"
            >
              <Download className="w-3.5 h-3.5" />
              Download Heatmap
            </button>
          </div>
          <p className="text-xs text-purple-600">
            Red/Magenta highlights indicate pixels modified during EMD and OPAP payload embedding.
          </p>
          <div className="rounded-xl border border-pink-100 overflow-hidden bg-purple-950/5 flex items-center justify-center max-h-112">
            <img src={visuals.heatmap_b64} alt="Diff Heatmap" className="max-h-112 object-contain" />
          </div>
        </div>
      )}

      {activeView === 'mask' && maskUrl && (
        <div className="p-4 rounded-2xl border border-pink-200 bg-white shadow-xs space-y-3">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-bold text-purple-950">Binary Modification Mask</h3>
            <button
              onClick={() => downloadImage(maskUrl, 'stego_binary_mask.png')}
              className="text-xs text-pink-700 font-semibold hover:underline flex items-center gap-1"
            >
              <Download className="w-3.5 h-3.5" />
              Download Mask
            </button>
          </div>
          <p className="text-xs text-purple-600">
            Pure binary map (White = modified pixel, Black = unchanged cover pixel).
          </p>
          <div className="rounded-xl border border-pink-100 overflow-hidden bg-black flex items-center justify-center max-h-112">
            <img src={maskUrl} alt="Binary Mask" className="max-h-112 object-contain" />
          </div>
        </div>
      )}

      {activeView === 'zoneMap' && (
        <div className="p-4 rounded-2xl border border-pink-200 bg-white shadow-xs space-y-3">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-bold text-purple-950">Pretrained CNN Cost Map &amp; Zoning Overlay</h3>
            <button
              onClick={() => downloadImage(visuals.zone_map_b64, 'cnn_zone_map.png')}
              className="text-xs text-pink-700 font-semibold hover:underline flex items-center gap-1"
            >
              <Download className="w-3.5 h-3.5" />
              Download Zone Map
            </button>
          </div>
          <p className="text-xs text-purple-600">
            VGG16 feature cost map classifies pixels into Zone A (Lilac = Smooth), Zone B (Pink = Texture), and Zone C (Fuchsia = Edges).
          </p>
          <div className="rounded-xl border border-pink-100 overflow-hidden bg-purple-950/5 flex items-center justify-center max-h-112">
            <img src={visuals.zone_map_b64} alt="Zone Map" className="max-h-112 object-contain" />
          </div>
        </div>
      )}

      {activeView === 'gradOverlay' && visuals.gradient_overlay_b64 && (
        <div className="p-4 rounded-2xl border border-pink-200 bg-white shadow-xs space-y-3">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-bold text-purple-950">Surrogate Steganalyzer Gradient Overlay</h3>
            <button
              onClick={() => downloadImage(visuals.gradient_overlay_b64!, 'adversarial_gradient_overlay.png')}
              className="text-xs text-pink-700 font-semibold hover:underline flex items-center gap-1"
            >
              <Download className="w-3.5 h-3.5" />
              Download Gradient Map
            </button>
          </div>
          <p className="text-xs text-purple-600">
            Cyan highlights show gradient sensitivity magnitudes computed by autograd on the surrogate steganalyzer model.
          </p>
          <div className="rounded-xl border border-pink-100 overflow-hidden bg-purple-950/5 flex items-center justify-center max-h-112">
            <img src={visuals.gradient_overlay_b64} alt="Gradient Overlay" className="max-h-112 object-contain" />
          </div>
        </div>
      )}

      {/* Interactive Zoomable/Pannable Canvas Inspection Tool */}
      <ZoomCanvas
        coverUrl={coverUrl}
        stegoUrl={stegoUrl}
        heatmapUrl={visuals.heatmap_b64 || undefined}
        showHeatmap={showHeatmapOverlay}
      />
    </div>
  );
};

