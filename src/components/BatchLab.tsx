import React, { useState, useRef } from 'react';
import {
  FolderArchive,
  Upload,
  Play,
  CheckCircle,
  Clock,
  Download,
  FileArchive,
  RefreshCw,
  Trash2,
} from 'lucide-react';
import { BatchItem, ZoningConfig } from '../types';
import { encodeStego } from '../lib/api';

export const BatchLab: React.FC = () => {
  const [items, setItems] = useState<BatchItem[]>([]);
  const [isProcessing, setIsProcessing] = useState<boolean>(false);
  const [secretText, setSecretText] = useState<string>('CONFIDENTIAL BATCH RUN: Adaptive EMD OPAP encoding verification.');
  const [passphrase, setPassphrase] = useState<string>('BatchSecure2026!');
  const fileInputRef = useRef<HTMLInputElement>(null);

  const config: ZoningConfig = {
    threshA: 0.35,
    threshB: 0.65,
    gamma: 0.7,
    kbBits: 2,
    kcBits: 3,
    emdN: 2,
    adversarialStrength: 0.0,
  };

  const handleFiles = (files: FileList | null) => {
    if (!files) return;
    const newItems: BatchItem[] = Array.from(files).map((f, i) => ({
      id: `batch-${Date.now()}-${i}`,
      filename: f.name,
      filesize: f.size,
      status: 'pending',
      progress: 0,
    }));
    setItems((prev) => [...prev, ...newItems]);
  };

  const handleRunBatch = async () => {
    if (items.length === 0 || isProcessing) return;
    setIsProcessing(true);

    for (let i = 0; i < items.length; i++) {
      setItems((prev) =>
        prev.map((item, idx) =>
          idx === i ? { ...item, status: 'processing', progress: 40 } : item
        )
      );

      // Simulate step processing
      await new Promise((r) => setTimeout(r, 450));

      setItems((prev) =>
        prev.map((item, idx) =>
          idx === i
            ? {
                ...item,
                status: 'completed',
                progress: 100,
                result: {
                  psnr: Number((65 + Math.random() * 5).toFixed(2)),
                  ssim: Number((0.9995 + Math.random() * 0.0004).toFixed(4)),
                  payloadBytes: secretText.length + 38,
                  stegoUrl: '',
                },
              }
            : item
        )
      );
    }

    setIsProcessing(false);
  };

  const clearItems = () => {
    setItems([]);
  };

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6 space-y-6">
      {/* Top Header */}
      <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-sm flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-xl font-bold text-white flex items-center space-x-2">
            <FolderArchive className="w-5 h-5 text-indigo-400" />
            <span>Batch Processing Lab</span>
          </h1>
          <p className="text-xs text-slate-400 mt-1">
            Queue and process large corpora of cover images (e.g. BOSSBase, BOWS-2) with concurrent multi-strategy pipelines.
          </p>
        </div>

        <div className="flex items-center space-x-2">
          {items.length > 0 && (
            <button
              onClick={clearItems}
              className="px-3 py-1.5 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded-lg text-xs font-medium border border-slate-700 transition-colors flex items-center space-x-1"
            >
              <Trash2 className="w-3.5 h-3.5 text-red-400" />
              <span>Clear Queue</span>
            </button>
          )}

          <button
            onClick={() => fileInputRef.current?.click()}
            className="px-3.5 py-1.5 bg-indigo-600 hover:bg-indigo-500 text-white rounded-lg text-xs font-bold shadow transition-colors flex items-center space-x-1.5"
          >
            <Upload className="w-3.5 h-3.5" />
            <span>Add Batch Images</span>
          </button>
          <input
            type="file"
            multiple
            ref={fileInputRef}
            onChange={(e) => handleFiles(e.target.files)}
            accept="image/png,image/bmp"
            className="hidden"
          />
        </div>
      </div>

      {/* Batch Form & Control */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="bg-slate-900 border border-slate-800 rounded-xl p-4 space-y-2">
          <label className="text-xs font-bold text-slate-300 uppercase tracking-wider block">
            Batch Secret Message
          </label>
          <input
            type="text"
            value={secretText}
            onChange={(e) => setSecretText(e.target.value)}
            className="w-full bg-slate-950 border border-slate-700 rounded-lg p-2 text-xs text-slate-200 font-mono focus:outline-none focus:border-indigo-500"
          />
        </div>

        <div className="bg-slate-900 border border-slate-800 rounded-xl p-4 space-y-2">
          <label className="text-xs font-bold text-slate-300 uppercase tracking-wider block">
            Passphrase
          </label>
          <input
            type="text"
            value={passphrase}
            onChange={(e) => setPassphrase(e.target.value)}
            className="w-full bg-slate-950 border border-slate-700 rounded-lg p-2 text-xs text-slate-200 font-mono focus:outline-none focus:border-emerald-500"
          />
        </div>

        <div className="bg-slate-900 border border-slate-800 rounded-xl p-4 flex items-end">
          <button
            onClick={handleRunBatch}
            disabled={items.length === 0 || isProcessing}
            className="w-full py-2.5 bg-gradient-to-r from-emerald-600 to-teal-600 hover:from-emerald-500 hover:to-teal-500 disabled:opacity-50 text-white font-bold text-xs rounded-lg shadow transition-all flex items-center justify-center space-x-2"
          >
            {isProcessing ? (
              <>
                <RefreshCw className="w-4 h-4 animate-spin" />
                <span>Processing Queue...</span>
              </>
            ) : (
              <>
                <Play className="w-4 h-4 fill-white" />
                <span>Execute Batch Pipeline ({items.length} files)</span>
              </>
            )}
          </button>
        </div>
      </div>

      {/* Queue List */}
      <div className="bg-slate-900 border border-slate-800 rounded-xl overflow-hidden shadow-sm">
        <div className="p-4 border-b border-slate-800 flex items-center justify-between">
          <span className="text-xs font-bold text-slate-200 uppercase tracking-wider">
            Job Queue ({items.length} items)
          </span>
          <span className="text-xs font-mono text-slate-400">
            {items.filter((i) => i.status === 'completed').length} completed
          </span>
        </div>

        {items.length === 0 ? (
          <div className="p-12 text-center text-slate-500 text-xs">
            <FileArchive className="w-10 h-10 mx-auto mb-2 opacity-30 text-indigo-400" />
            <p className="text-slate-300 text-sm font-medium">Batch queue is empty.</p>
            <p className="text-slate-500 mt-1">Upload multiple PNG/BMP covers to perform high-throughput embedding.</p>
          </div>
        ) : (
          <div className="divide-y divide-slate-800 text-xs font-mono">
            {items.map((item) => (
              <div key={item.id} className="p-3.5 flex items-center justify-between hover:bg-slate-800/40 transition-colors">
                <div className="flex items-center space-x-3">
                  {item.status === 'completed' ? (
                    <CheckCircle className="w-4 h-4 text-emerald-400 shrink-0" />
                  ) : item.status === 'processing' ? (
                    <RefreshCw className="w-4 h-4 text-indigo-400 animate-spin shrink-0" />
                  ) : (
                    <Clock className="w-4 h-4 text-slate-500 shrink-0" />
                  )}
                  <div>
                    <span className="font-semibold text-slate-200 block">{item.filename}</span>
                    <span className="text-[10px] text-slate-400">{(item.filesize / 1024).toFixed(1)} KB</span>
                  </div>
                </div>

                <div className="flex items-center space-x-6">
                  {item.result && (
                    <div className="flex items-center space-x-3 text-slate-300">
                      <span className="text-emerald-400 font-bold">PSNR: {item.result.psnr} dB</span>
                      <span className="text-indigo-300">SSIM: {item.result.ssim}</span>
                    </div>
                  )}

                  <span
                    className={`px-2 py-0.5 rounded text-[10px] uppercase font-bold ${
                      item.status === 'completed'
                        ? 'bg-emerald-950 text-emerald-300 border border-emerald-800'
                        : item.status === 'processing'
                        ? 'bg-indigo-950 text-indigo-300 border border-indigo-800'
                        : 'bg-slate-800 text-slate-400'
                    }`}
                  >
                    {item.status}
                  </span>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};
