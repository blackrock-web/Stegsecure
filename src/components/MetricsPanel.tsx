import React from 'react';
import { MetricsData, SecurityReport } from '../types';
import { Activity, ShieldCheck, BarChart2, CheckCircle2, ShieldAlert } from 'lucide-react';

interface MetricsPanelProps {
  metrics: MetricsData;
  securityReport?: SecurityReport;
}

export const MetricsPanel: React.FC<MetricsPanelProps> = ({ metrics, securityReport }) => {
  if (!metrics) {
    return null;
  }
  const mse = metrics.mse ?? 0;
  const psnr_db = metrics.psnr_db ?? 0;
  const ssim = metrics.ssim ?? 0;
  const total_bits_embedded = metrics.total_bits_embedded ?? 0;
  const total_bytes_embedded = metrics.total_bytes_embedded ?? 0;
  const achieved_bpp = metrics.achieved_bpp ?? (metrics as any).bpp ?? 0;
  const modified_pixel_count = metrics.modified_pixel_count ?? (metrics as any).modified_pixels_count ?? 0;
  const modified_pixel_percentage = metrics.modified_pixel_percentage ?? 0;
  const zone_breakdown = metrics.zone_breakdown ?? { zone_a_bits: 0, zone_b_bits: 0, zone_c_bits: 0 };

  return (
    <div className="p-5 rounded-2xl border border-pink-200 bg-white/95 shadow-xs space-y-4">
      <div className="flex items-center justify-between border-b border-pink-100 pb-3">
        <div className="flex items-center gap-2">
          <Activity className="w-5 h-5 text-pink-600" />
          <h3 className="text-base font-bold text-purple-950">Steganographic Quality Metrics</h3>
        </div>
        <span className="text-xs font-semibold px-2.5 py-1 rounded-full bg-emerald-100 text-emerald-800 border border-emerald-200 flex items-center gap-1">
          <CheckCircle2 className="w-3.5 h-3.5 text-emerald-600" />
          High Imperceptibility
        </span>
      </div>

      {/* Primary 4 Metric Cards */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
        {/* PSNR Card */}
        <div className="p-3.5 rounded-2xl bg-gradient-to-br from-pink-50/80 to-purple-50/50 border border-pink-200/80 space-y-1">
          <p className="text-xs font-bold text-purple-700">PSNR (Peak Signal-to-Noise)</p>
          <div className="flex items-baseline gap-1">
            <span className="text-2xl font-black text-purple-950">{psnr_db}</span>
            <span className="text-xs font-semibold text-purple-600">dB</span>
          </div>
          <p className="text-[11px] text-purple-600 font-medium">
            {psnr_db > 40 ? '✨ Exceptional (>40 dB standard)' : 'Good fidelity'}
          </p>
        </div>

        {/* SSIM Card */}
        <div className="p-3.5 rounded-2xl bg-gradient-to-br from-purple-50/80 to-pink-50/50 border border-purple-200/80 space-y-1">
          <p className="text-xs font-bold text-purple-700">SSIM (Structural Similarity)</p>
          <div className="flex items-baseline gap-1">
            <span className="text-2xl font-black text-purple-950">{ssim}</span>
            <span className="text-xs font-semibold text-purple-600">/ 1.0</span>
          </div>
          <p className="text-[11px] text-purple-600 font-medium">
            {ssim > 0.98 ? '✨ Near-perfect structural match' : 'High similarity'}
          </p>
        </div>

        {/* MSE Card */}
        <div className="p-3.5 rounded-2xl bg-gradient-to-br from-pink-50/80 to-fuchsia-50/50 border border-pink-200/80 space-y-1">
          <p className="text-xs font-bold text-purple-700">MSE (Mean Squared Error)</p>
          <div className="flex items-baseline gap-1">
            <span className="text-2xl font-black text-purple-950">{mse}</span>
          </div>
          <p className="text-[11px] text-purple-600 font-medium">Lower distortion is safer</p>
        </div>

        {/* Achieved bpp Card */}
        <div className="p-3.5 rounded-2xl bg-gradient-to-br from-fuchsia-50/80 to-purple-50/50 border border-fuchsia-200/80 space-y-1">
          <p className="text-xs font-bold text-purple-700">Achieved Rate</p>
          <div className="flex items-baseline gap-1">
            <span className="text-2xl font-black text-purple-950">{achieved_bpp}</span>
            <span className="text-xs font-semibold text-purple-600">bpp</span>
          </div>
          <p className="text-[11px] text-purple-600 font-medium">
            {total_bits_embedded.toLocaleString()} bits ({total_bytes_embedded} B)
          </p>
        </div>
      </div>

      {/* Security Report Section (if provided) */}
      {securityReport && (
        <div className="p-4 rounded-2xl bg-gradient-to-r from-purple-50 to-fuchsia-50 border border-purple-200/80 space-y-2">
          <div className="flex items-center justify-between">
            <span className="flex items-center gap-1.5 text-xs font-bold text-purple-950">
              <ShieldAlert className="w-4 h-4 text-purple-600" />
              Surrogate Steganalyzer Security Evaluation
            </span>
            <span className="text-[11px] font-semibold text-purple-700">
              Δ Detection: +{(securityReport.detection_confidence_delta * 100).toFixed(2)}%
            </span>
          </div>
          <div className="grid grid-cols-3 gap-2 text-center text-xs">
            <div className="p-2 rounded-xl bg-white border border-purple-100">
              <div className="text-[10px] font-bold text-purple-700">Cover Confidence</div>
              <div className="text-sm font-extrabold text-purple-950">
                {(securityReport.cover_detection_confidence * 100).toFixed(1)}%
              </div>
            </div>
            <div className="p-2 rounded-xl bg-white border border-fuchsia-100">
              <div className="text-[10px] font-bold text-fuchsia-700">Stego Confidence</div>
              <div className="text-sm font-extrabold text-purple-950">
                {(securityReport.stego_detection_confidence * 100).toFixed(1)}%
              </div>
            </div>
            <div className="p-2 rounded-xl bg-white border border-pink-100">
              <div className="text-[10px] font-bold text-pink-700">Stego Delta</div>
              <div className="text-sm font-extrabold text-purple-950">
                +{(securityReport.detection_confidence_delta * 100).toFixed(1)}%
              </div>
            </div>
          </div>
          <p className="text-[10px] text-purple-600 italic leading-tight">
            {securityReport.note}
          </p>
        </div>
      )}

      {/* Per-Zone Payload Distribution Breakdown */}
      <div className="p-4 rounded-2xl bg-pink-50/40 border border-pink-200/60 space-y-2">
        <div className="flex items-center justify-between text-xs font-bold text-purple-950">
          <span className="flex items-center gap-1.5">
            <BarChart2 className="w-4 h-4 text-pink-600" />
            Per-Zone Embedded Payload Breakdown
          </span>
          <span className="text-purple-700">
            {modified_pixel_count.toLocaleString()} pixels modified ({modified_pixel_percentage}%)
          </span>
        </div>

        <div className="grid grid-cols-3 gap-2 text-center text-xs">
          <div className="p-2 rounded-xl bg-white border border-purple-200 shadow-2xs">
            <div className="text-[11px] font-bold text-purple-900">Zone A (EMD)</div>
            <div className="text-sm font-extrabold text-purple-950 mt-0.5">
              {zone_breakdown.zone_a_bits} bits
            </div>
            <div className="text-[10px] text-purple-600">Base-5 digit pairs</div>
          </div>

          <div className="p-2 rounded-xl bg-white border border-pink-200 shadow-2xs">
            <div className="text-[11px] font-bold text-pink-900">Zone B (OPAP)</div>
            <div className="text-sm font-extrabold text-pink-950 mt-0.5">
              {zone_breakdown.zone_b_bits} bits
            </div>
            <div className="text-[10px] text-pink-600">LSB + OPAP</div>
          </div>

          <div className="p-2 rounded-xl bg-white border border-fuchsia-200 shadow-2xs">
            <div className="text-[11px] font-bold text-fuchsia-900">Zone C (OPAP)</div>
            <div className="text-sm font-extrabold text-fuchsia-950 mt-0.5">
              {zone_breakdown.zone_c_bits} bits
            </div>
            <div className="text-[10px] text-fuchsia-600">LSB + OPAP</div>
          </div>
        </div>
      </div>
    </div>
  );
};

