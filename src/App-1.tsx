import React, { useState, useEffect } from 'react';
import { Header, AppTab } from './components/Header';
import { ImageUploader } from './components/ImageUploader';
import { TextInput } from './components/TextInput';
import { PassphraseInput } from './components/PassphraseInput';
import { CapacityMeter } from './components/CapacityMeter';
import { ResultViewer } from './components/ResultViewer';
import { MetricsPanel } from './components/MetricsPanel';
import { DecodePage } from './components/DecodePage';
import { ConfigModal } from './components/ConfigModal';
import { AlgorithmInfo } from './components/AlgorithmInfo';
import { BenchmarkPage } from './components/BenchmarkPage';
import { ComparisonPage } from './components/ComparisonPage';
import { BatchLab } from './components/batch/BatchLab';
import { CapacityResponse, EncodeResponse, TuningConfig } from './types';
import { ShieldCheck, Lock, Loader2, AlertCircle, Sparkles, Sliders } from 'lucide-react';

const DEFAULT_CONFIG: TuningConfig = {
  gamma: 0.7,
  threshA: 0.35,
  threshB: 0.65,
  kbBits: 2,
  kcBits: 3,
  costMapMode: 'cnn',
  adversarialStrength: 0.0,
  emdN: 2,
};

export default function App() {
  const [activeTab, setActiveTab] = useState<AppTab>('encode');
  const [isConfigOpen, setIsConfigOpen] = useState(false);
  const [tuningConfig, setTuningConfig] = useState<TuningConfig>(DEFAULT_CONFIG);

  // Encode state
  const [coverFile, setCoverFile] = useState<File | null>(null);
  const [coverUrl, setCoverUrl] = useState<string | null>(null);
  const [secretText, setSecretText] = useState('');
  const [passphrase, setPassphrase] = useState('');

  const [capacityData, setCapacityData] = useState<CapacityResponse | null>(null);
  const [capacityLoading, setCapacityLoading] = useState(false);

  const [encoding, setEncoding] = useState(false);
  const [encodeResult, setEncodeResult] = useState<EncodeResponse | null>(null);
  const [encodeError, setEncodeError] = useState<string | null>(null);

  // Update cover preview URL and fetch capacity
  useEffect(() => {
    if (!coverFile) {
      setCoverUrl(null);
      setCapacityData(null);
      setEncodeResult(null);
      return;
    }

    const url = URL.createObjectURL(coverFile);
    setCoverUrl(url);
    fetchCapacity(coverFile);
  }, [coverFile, tuningConfig]);

  const fetchCapacity = async (file: File) => {
    setCapacityLoading(true);
    const formData = new FormData();
    formData.append('file', file);
    formData.append('thresh_a', tuningConfig.threshA.toString());
    formData.append('thresh_b', tuningConfig.threshB.toString());
    formData.append('gamma', tuningConfig.gamma.toString());
    formData.append('cost_map_mode', tuningConfig.costMapMode || 'cnn');
    formData.append('emd_n', (tuningConfig.emdN || 2).toString());

    try {
      const res = await fetch('/api/capacity', {
        method: 'POST',
        body: formData,
      });
      if (res.ok) {
        const data = await res.json();
        setCapacityData(data);
      }
    } catch (err) {
      console.error('Failed to fetch image capacity', err);
    } finally {
      setCapacityLoading(false);
    }
  };

  const handleEncodeSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!coverFile || !secretText.trim() || !passphrase) return;

    setEncoding(true);
    setEncodeError(null);
    setEncodeResult(null);

    const formData = new FormData();
    formData.append('file', coverFile);
    formData.append('secret_text', secretText);
    formData.append('passphrase', passphrase);
    formData.append('thresh_a', tuningConfig.threshA.toString());
    formData.append('thresh_b', tuningConfig.threshB.toString());
    formData.append('gamma', tuningConfig.gamma.toString());
    formData.append('kb_bits', tuningConfig.kbBits.toString());
    formData.append('kc_bits', tuningConfig.kcBits.toString());
    formData.append('cost_map_mode', tuningConfig.costMapMode || 'cnn');
    formData.append('adversarial_strength', (tuningConfig.adversarialStrength || 0.0).toString());
    formData.append('emd_n', (tuningConfig.emdN || 2).toString());

    try {
      const res = await fetch('/api/encode', {
        method: 'POST',
        body: formData,
      });

      const raw = await res.text();
      let data: any = {};
      try {
        data = raw ? JSON.parse(raw) : {};
      } catch {
        throw new Error(res.ok ? 'Invalid server response.' : `Server error (${res.status}).`);
      }
      if (!res.ok) {
        const detail = typeof data.detail === 'string' ? data.detail : (data.detail?.msg || data.message || 'Failed to encode steganographic payload.');
        throw new Error(detail);
      }
      if (!data.visuals || !data.visuals.stego_b64) {
        throw new Error('Encode succeeded but no stego image was returned. Try Fast mode or a smaller PNG.');
      }
      setEncodeResult(data);
    } catch (err: any) {
      setEncodeError(err.message || 'An error occurred during steganography encoding.');
    } finally {
      setEncoding(false);
    }
  };

  const plainTextBytes = new TextEncoder().encode(secretText).length;

  return (
    <div className="min-h-screen bg-gradient-to-br from-pink-50 via-purple-50/40 to-fuchsia-50/30 text-purple-950 font-sans pb-16 selection:bg-pink-200 selection:text-pink-900">
      {/* App Header */}
      <Header
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        onOpenConfig={() => setIsConfigOpen(true)}
      />

      {/* Main Container */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 pt-6">
        {activeTab === 'encode' && (
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
            {/* Left Control Panel (Form Inputs) */}
            <div className="lg:col-span-5 space-y-5">
              <div className="p-6 rounded-3xl bg-white/90 border border-pink-200 shadow-sm space-y-5">
                <div className="flex items-center justify-between border-b border-pink-100 pb-3">
                  <div className="flex items-center gap-2">
                    <Lock className="w-5 h-5 text-pink-600" />
                    <h2 className="text-base font-bold text-purple-950">Encode Payload Pipeline</h2>
                  </div>
                  <span className="text-xs font-semibold px-2.5 py-0.5 rounded-full bg-pink-100 text-pink-800 border border-pink-200">
                    EMD + OPAP
                  </span>
                </div>

                <form onSubmit={handleEncodeSubmit} className="space-y-4">
                  {/* Step 1: Cover Image Upload */}
                  <ImageUploader
                    label="Cover Image (PNG/BMP)"
                    selectedFile={coverFile}
                    onFileSelect={setCoverFile}
                  />

                  {/* Step 2: Secret Text Input */}
                  <TextInput
                    value={secretText}
                    onChange={setSecretText}
                    maxBytes={capacityData?.capacity.max_bytes}
                  />

                  {/* Step 3: Passphrase */}
                  <PassphraseInput
                    value={passphrase}
                    onChange={setPassphrase}
                    label="Encryption Passphrase"
                  />

                  {/* Capacity Meter */}
                  <CapacityMeter
                    capacityData={capacityData}
                    loading={capacityLoading}
                    usedBytes={plainTextBytes}
                  />

                  {/* Error display */}
                  {encodeError && (
                    <div className="p-4 rounded-2xl bg-pink-50 border border-pink-300 text-pink-950 text-xs flex items-start gap-2.5 animate-fadeIn">
                      <AlertCircle className="w-5 h-5 text-pink-600 flex-shrink-0 mt-0.5" />
                      <div>
                        <p className="font-bold">Encoding Error</p>
                        <p className="mt-0.5 font-medium text-pink-800">{encodeError}</p>
                      </div>
                    </div>
                  )}

                  {/* Submit Encode Button */}
                  <button
                    type="submit"
                    disabled={!coverFile || !secretText.trim() || !passphrase || encoding}
                    className="w-full py-3.5 px-6 rounded-2xl text-sm font-bold text-white bg-gradient-to-r from-pink-600 to-purple-600 hover:from-pink-700 hover:to-purple-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all shadow-md shadow-pink-200 flex items-center justify-center gap-2 active:scale-98"
                  >
                    {encoding ? (
                      <>
                        <Loader2 className="w-5 h-5 animate-spin" />
                        <span>Running VGG16 CNN Cost Map &amp; Embedding...</span>
                      </>
                    ) : (
                      <>
                        <ShieldCheck className="w-5 h-5" />
                        <span>Encrypt &amp; Hide Payload</span>
                      </>
                    )}
                  </button>
                </form>
              </div>
            </div>

            {/* Right Display Area (Results & Canvas Inspection) */}
            <div className="lg:col-span-7 space-y-6">
              {encodeResult && coverUrl && encodeResult.visuals ? (
                <>
                  <MetricsPanel metrics={encodeResult.metrics} securityReport={encodeResult.security_report} />
                  <ResultViewer coverUrl={coverUrl} resultData={encodeResult} />
                </>
              ) : (
                <div className="p-12 rounded-3xl border-2 border-dashed border-pink-200 bg-white/60 text-center space-y-4">
                  <div className="w-16 h-16 mx-auto rounded-3xl bg-pink-100 text-pink-600 flex items-center justify-center shadow-2xs">
                    <Sparkles className="w-8 h-8" />
                  </div>
                  <div className="max-w-md mx-auto space-y-1">
                    <h3 className="text-base font-bold text-purple-950">Awaiting Payload Input</h3>
                    <p className="text-xs text-purple-700 font-medium leading-relaxed">
                      Upload a cover image, enter your secret message and passphrase, and click &ldquo;Encrypt &amp; Hide Payload&rdquo; to execute the CNN feature cost mapping, EMD, and OPAP pipeline.
                    </p>
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        {activeTab === 'decode' && <DecodePage config={tuningConfig} />}

        {activeTab === 'batch' && <BatchLab tuningConfig={tuningConfig} />}

        {activeTab === 'benchmark' && <BenchmarkPage />}

        {activeTab === 'compare' && <ComparisonPage />}

        {activeTab === 'info' && <AlgorithmInfo />}
      </main>

      {/* Config Modal */}
      <ConfigModal
        isOpen={isConfigOpen}
        onClose={() => setIsConfigOpen(false)}
        config={tuningConfig}
        onChange={setTuningConfig}
      />
    </div>
  );
}
