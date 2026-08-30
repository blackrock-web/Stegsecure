import React from 'react';
import { TuningConfig } from '../types';
import { Sliders, RotateCcw, X, ShieldAlert, Cpu } from 'lucide-react';

interface ConfigModalProps {
  isOpen: boolean;
  onClose: () => void;
  config: TuningConfig;
  onChange: (cfg: TuningConfig) => void;
}

export const DEFAULT_CONFIG: TuningConfig = {
  gamma: 0.7,
  threshA: 0.35,
  threshB: 0.65,
  kbBits: 2,
  kcBits: 3,
  costMapMode: 'cnn',
  adversarialStrength: 0.0,
  emdN: 2,
};

export const ConfigModal: React.FC<ConfigModalProps> = ({
  isOpen,
  onClose,
  config,
  onChange,
}) => {
  if (!isOpen) return null;

  const handleReset = () => {
    onChange(DEFAULT_CONFIG);
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-purple-950/40 backdrop-blur-xs animate-fadeIn">
      <div className="w-full max-w-lg rounded-3xl bg-white border border-pink-200 shadow-xl p-6 space-y-5 max-h-[90vh] overflow-y-auto">
        <div className="flex items-center justify-between border-b border-pink-100 pb-3">
          <div className="flex items-center gap-2">
            <Sliders className="w-5 h-5 text-pink-600" />
            <h2 className="text-lg font-bold text-purple-950">Algorithm Config Parameters</h2>
          </div>
          <button
            onClick={onClose}
            className="p-1 rounded-xl text-purple-400 hover:text-pink-600 hover:bg-pink-50 transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        <div className="space-y-4 text-xs text-purple-950">
          {/* Cost Map Mode */}
          <div className="p-3 rounded-2xl bg-pink-50/50 border border-pink-100 space-y-2">
            <div className="flex justify-between font-bold items-center">
              <span className="flex items-center gap-1.5">
                <Cpu className="w-4 h-4 text-pink-600" />
                Cost Map Mode
              </span>
              <span className="text-pink-600 font-mono text-[11px]">
                {config.costMapMode === 'fast'
                  ? 'Fast (classical residual)'
                  : config.costMapMode === 'cnn'
                  ? 'CNN (trained CostMapCNN)'
                  : 'Advanced (CNN + HILL + Adv)'}
              </span>
            </div>
            <p className="text-[10px] text-purple-600 leading-snug">
              <strong>cnn</strong> uses the newly trained CostMapCNN. <strong>advanced</strong> adds
              HILL residual + SteganalyzerNet gradient sensitivity. Requires Python/Torch.
            </p>
            <div className="grid grid-cols-3 gap-2">
              <button
                type="button"
                onClick={() => onChange({ ...config, costMapMode: 'fast' })}
                className={`py-2 px-2 rounded-xl font-bold border transition-all text-xs ${
                  config.costMapMode === 'fast'
                    ? 'bg-pink-600 text-white border-pink-600 shadow-2xs'
                    : 'bg-white text-purple-800 border-pink-200 hover:bg-pink-50'
                }`}
              >
                Fast
              </button>
              <button
                type="button"
                onClick={() => onChange({ ...config, costMapMode: 'cnn' })}
                className={`py-2 px-2 rounded-xl font-bold border transition-all text-xs ${
                  config.costMapMode === 'cnn'
                    ? 'bg-pink-600 text-white border-pink-600 shadow-2xs'
                    : 'bg-white text-purple-800 border-pink-200 hover:bg-pink-50'
                }`}
              >
                CNN (trained)
              </button>
              <button
                type="button"
                onClick={() => onChange({ ...config, costMapMode: 'advanced' })}
                className={`py-2 px-2 rounded-xl font-bold border transition-all text-xs ${
                  config.costMapMode === 'advanced'
                    ? 'bg-pink-600 text-white border-pink-600 shadow-2xs'
                    : 'bg-white text-purple-800 border-pink-200 hover:bg-pink-50'
                }`}
              >
                Advanced
              </button>
            </div>
          </div>

          {/* EMD Group Size n */}
          <div className="p-3 rounded-2xl bg-purple-50/50 border border-purple-100 space-y-2">
            <div className="flex justify-between font-bold items-center">
              <span>EMD Group Size (n) &amp; Base Digits</span>
              <span className="text-purple-600 font-mono text-[11px]">
                n = {config.emdN} ({config.emdN === 2 ? 'Base-5, 2.32 bpp' : 'Base-7, 2.81 bpp'})
              </span>
            </div>
            <div className="grid grid-cols-2 gap-2">
              <button
                type="button"
                onClick={() => onChange({ ...config, emdN: 2 })}
                className={`py-2 px-3 rounded-xl font-bold border transition-all text-xs ${
                  config.emdN === 2
                    ? 'bg-purple-600 text-white border-purple-600 shadow-2xs'
                    : 'bg-white text-purple-800 border-purple-200 hover:bg-purple-50'
                }`}
              >
                n = 2 (Base-5, 2 pixels)
              </button>
              <button
                type="button"
                onClick={() => onChange({ ...config, emdN: 3 })}
                className={`py-2 px-3 rounded-xl font-bold border transition-all text-xs ${
                  config.emdN === 3
                    ? 'bg-purple-600 text-white border-purple-600 shadow-2xs'
                    : 'bg-white text-purple-800 border-purple-200 hover:bg-purple-50'
                }`}
              >
                n = 3 (Base-7, 3 pixels)
              </button>
            </div>
            <p className="text-[11px] text-purple-600">
              {config.emdN === 2
                ? 'n=2 embeds 1 base-5 digit in 2 pixels (at most ±1 change in 1 pixel).'
                : 'n=3 embeds 1 base-7 digit in 3 pixels (at most ±1 change in 1 pixel, higher capacity).'}
            </p>
          </div>

          {/* Adversarial Embedding Strength */}
          <div className="space-y-1.5 p-3 rounded-2xl bg-fuchsia-50/50 border border-fuchsia-100">
            <div className="flex justify-between font-bold items-center">
              <span className="flex items-center gap-1.5">
                <ShieldAlert className="w-4 h-4 text-fuchsia-600" />
                Adversarial Embedding Guidance
              </span>
              <span className="text-fuchsia-600 font-mono">{config.adversarialStrength.toFixed(2)}</span>
            </div>
            <input
              type="range"
              min="0.0"
              max="1.0"
              step="0.05"
              value={config.adversarialStrength}
              onChange={(e) => onChange({ ...config, adversarialStrength: parseFloat(e.target.value) })}
              className="w-full accent-fuchsia-600 cursor-pointer"
            />
            <p className="text-[11px] text-purple-600">
              Applies sign(-∇_cover Loss) from surrogate steganalyzer to steer EMD/OPAP pixel adjustments away from detection features.
            </p>
          </div>

          {/* Gamma Slider */}
          <div className="space-y-1.5 p-3 rounded-2xl bg-pink-50/50 border border-pink-100">
            <div className="flex justify-between font-bold">
              <span>CNN vs Edge-Fusion Blend Weight (γ)</span>
              <span className="text-pink-600 font-mono">{config.gamma.toFixed(2)}</span>
            </div>
            <input
              type="range"
              min="0.1"
              max="1.0"
              step="0.05"
              value={config.gamma}
              onChange={(e) => onChange({ ...config, gamma: parseFloat(e.target.value) })}
              className="w-full accent-pink-600 cursor-pointer"
            />
            <p className="text-[11px] text-purple-600">
              γ = 0.7 balances VGG16 deep feature activations with Canny/Sobel edge responses.
            </p>
          </div>

          {/* Zone A Threshold Slider */}
          <div className="space-y-1.5 p-3 rounded-2xl bg-purple-50/50 border border-purple-100">
            <div className="flex justify-between font-bold">
              <span>Zone A Threshold (Smooth / EMD)</span>
              <span className="text-purple-600 font-mono">{config.threshA.toFixed(2)}</span>
            </div>
            <input
              type="range"
              min="0.1"
              max="0.5"
              step="0.05"
              value={config.threshA}
              onChange={(e) => onChange({ ...config, threshA: parseFloat(e.target.value) })}
              className="w-full accent-purple-600 cursor-pointer"
            />
            <p className="text-[11px] text-purple-600">
              Cost &lt; {config.threshA} classified as Zone A (EMD algorithm, highest security).
            </p>
          </div>

          {/* Zone B Threshold Slider */}
          <div className="space-y-1.5 p-3 rounded-2xl bg-fuchsia-50/50 border border-fuchsia-100">
            <div className="flex justify-between font-bold">
              <span>Zone B Threshold (Medium Texture)</span>
              <span className="text-fuchsia-600 font-mono">{config.threshB.toFixed(2)}</span>
            </div>
            <input
              type="range"
              min="0.5"
              max="0.9"
              step="0.05"
              value={config.threshB}
              onChange={(e) => onChange({ ...config, threshB: parseFloat(e.target.value) })}
              className="w-full accent-fuchsia-600 cursor-pointer"
            />
            <p className="text-[11px] text-purple-600">
              {config.threshA} ≤ Cost &lt; {config.threshB} classified as Zone B (OPAP k={config.kbBits}).
            </p>
          </div>

          {/* Zone B & C OPAP k-bit selections */}
          <div className="grid grid-cols-2 gap-3">
            <div className="p-3 rounded-2xl bg-pink-50/50 border border-pink-100 space-y-1">
              <label className="font-bold block">Zone B OPAP Bits (k_B)</label>
              <select
                value={config.kbBits}
                onChange={(e) => onChange({ ...config, kbBits: parseInt(e.target.value) })}
                className="w-full p-2 text-xs rounded-xl border border-pink-200 bg-white font-semibold outline-none"
              >
                <option value={1}>1 bit (1.0 bpp)</option>
                <option value={2}>2 bits (2.0 bpp)</option>
              </select>
            </div>

            <div className="p-3 rounded-2xl bg-fuchsia-50/50 border border-fuchsia-100 space-y-1">
              <label className="font-bold block">Zone C OPAP Bits (k_C)</label>
              <select
                value={config.kcBits}
                onChange={(e) => onChange({ ...config, kcBits: parseInt(e.target.value) })}
                className="w-full p-2 text-xs rounded-xl border border-fuchsia-200 bg-white font-semibold outline-none"
              >
                <option value={3}>3 bits (3.0 bpp)</option>
                <option value={4}>4 bits (4.0 bpp)</option>
              </select>
            </div>
          </div>
        </div>

        {/* Footer Actions */}
        <div className="flex items-center justify-between border-t border-pink-100 pt-3">
          <button
            onClick={handleReset}
            className="flex items-center gap-1.5 px-3 py-2 rounded-xl text-xs font-semibold text-purple-700 hover:bg-purple-50 transition-colors"
          >
            <RotateCcw className="w-3.5 h-3.5" />
            <span>Reset to Defaults</span>
          </button>

          <button
            onClick={onClose}
            className="px-5 py-2 rounded-xl text-xs font-bold text-white bg-pink-600 hover:bg-pink-700 transition-colors shadow-xs"
          >
            Apply &amp; Close
          </button>
        </div>
      </div>
    </div>
  );
};

