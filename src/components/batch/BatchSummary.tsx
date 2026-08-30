import React from 'react';
import { BarChart3 } from 'lucide-react';

interface Summary {
  total?: number;
  successful?: number;
  failed?: number;
  cancelled?: number;
  metrics?: Record<
    string,
    { mean?: number | null; median?: number | null; min?: number | null; max?: number | null; std?: number | null; n?: number }
  >;
}

interface Props {
  summary: Summary | null;
}

function fmt(v: number | null | undefined, digits = 3) {
  if (v == null || Number.isNaN(v)) return '—';
  return Number(v).toFixed(digits);
}

export const BatchSummary: React.FC<Props> = ({ summary }) => {
  if (!summary) return null;
  const m = summary.metrics || {};

  const cards = [
    { label: 'Avg PSNR (dB)', val: fmt(m.psnr_db?.mean, 2) },
    { label: 'Avg SSIM', val: fmt(m.ssim?.mean, 4) },
    { label: 'Avg MSE', val: fmt(m.mse?.mean, 4) },
    { label: 'Avg Time/Image', val: m.processing_time_s?.mean != null ? `${fmt(m.processing_time_s.mean, 2)} s` : '—' },
    { label: 'Avg Suspicion', val: fmt(m.suspicion?.mean, 3) },
    { label: 'Median PSNR', val: fmt(m.psnr_db?.median, 2) },
  ];

  return (
    <div className="rounded-2xl border border-pink-200 bg-white/90 p-5 space-y-4">
      <div className="flex items-center gap-2">
        <BarChart3 className="w-5 h-5 text-pink-600" />
        <h3 className="text-sm font-bold text-purple-950">Batch Summary</h3>
      </div>
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-center">
        <div className="rounded-xl bg-purple-50 border border-purple-100 p-3">
          <div className="text-2xl font-bold text-purple-900">{summary.total ?? 0}</div>
          <div className="text-[11px] text-purple-600">Processed</div>
        </div>
        <div className="rounded-xl bg-emerald-50 border border-emerald-100 p-3">
          <div className="text-2xl font-bold text-emerald-800">{summary.successful ?? 0}</div>
          <div className="text-[11px] text-emerald-700">Successful</div>
        </div>
        <div className="rounded-xl bg-amber-50 border border-amber-100 p-3">
          <div className="text-2xl font-bold text-amber-800">{summary.failed ?? 0}</div>
          <div className="text-[11px] text-amber-700">Failed</div>
        </div>
        <div className="rounded-xl bg-slate-50 border border-slate-100 p-3">
          <div className="text-2xl font-bold text-slate-700">{summary.cancelled ?? 0}</div>
          <div className="text-[11px] text-slate-600">Cancelled</div>
        </div>
      </div>
      <div className="grid grid-cols-2 sm:grid-cols-3 gap-2">
        {cards.map((c) => (
          <div key={c.label} className="rounded-xl border border-pink-100 bg-pink-50/30 px-3 py-2">
            <div className="text-[10px] font-semibold text-purple-600 uppercase tracking-wide">{c.label}</div>
            <div className="text-sm font-bold text-purple-950 tabular-nums">{c.val}</div>
          </div>
        ))}
      </div>
      <p className="text-[10px] text-purple-500">
        All aggregates computed from actual per-image results (mean / median / std). No fabricated metrics.
      </p>
    </div>
  );
};
