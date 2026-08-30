import React from 'react';
import { CapacityResponse } from '../types';
import { Database, Cpu, BarChart3, Info } from 'lucide-react';

interface CapacityMeterProps {
  capacityData: CapacityResponse | null;
  loading: boolean;
  usedBytes: number;
}

export const CapacityMeter: React.FC<CapacityMeterProps> = ({
  capacityData,
  loading,
  usedBytes,
}) => {
  if (loading) {
    return (
      <div className="p-4 rounded-2xl border border-pink-200 bg-white/80 animate-pulse space-y-3">
        <div className="h-4 bg-pink-100 rounded-md w-1/3"></div>
        <div className="h-3 bg-pink-100 rounded-md w-2/3"></div>
        <div className="h-6 bg-pink-100 rounded-xl w-full"></div>
      </div>
    );
  }

  if (!capacityData) {
    return (
      <div className="p-4 rounded-2xl border border-pink-200/80 bg-white/60 text-center text-xs text-purple-600">
        Upload a cover image to compute live CNN cost map and zone capacity.
      </div>
    );
  }

  const { capacity, width, height, cost_map_mode } = capacityData;
  const maxBytes = capacity?.max_bytes ?? 0;
  const maxPlaintext = capacity?.max_plaintext_bytes ?? Math.max(0, maxBytes - 48);
  const estPayloadBytes = usedBytes + 48; // Plaintext + crypto overhead
  const percentUsed = maxBytes > 0 ? Math.min(100, Math.round((estPayloadBytes / maxBytes) * 100)) : 0;

  const countA = capacity?.count_zone_a ?? (capacity as any)?.zone_breakdown?.zone_a_count ?? 0;
  const countB = capacity?.count_zone_b ?? (capacity as any)?.zone_breakdown?.zone_b_count ?? 0;
  const countC = capacity?.count_zone_c ?? (capacity as any)?.zone_breakdown?.zone_c_count ?? 0;
  const overallBpp = capacity?.overall_bpp ?? 0;

  return (
    <div className="p-4 rounded-2xl border border-pink-200 bg-white/90 shadow-xs space-y-3">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Database className="w-4 h-4 text-pink-600" />
          <h3 className="text-sm font-bold text-purple-950">Image Embedding Capacity</h3>
        </div>
        <span className="text-xs font-semibold px-2.5 py-0.5 rounded-full bg-purple-100 text-purple-900 border border-purple-200">
          {width} × {height} px • {typeof overallBpp === 'number' ? overallBpp.toFixed(2) : '0.00'} bpp avg
        </span>
      </div>

      {/* Main progress bar */}
      <div className="space-y-1">
        <div className="flex justify-between text-xs font-semibold text-purple-900">
          <span>Payload Allocation</span>
          <span>
            {estPayloadBytes} / {maxBytes} Bytes ({percentUsed}%)
          </span>
        </div>
        <div className="w-full h-3 rounded-full bg-pink-100 overflow-hidden p-0.5 border border-pink-200">
          <div
            className={`h-full rounded-full transition-all duration-300 ${
              percentUsed > 90 ? 'bg-pink-600' : 'bg-gradient-to-r from-purple-500 to-pink-500'
            }`}
            style={{ width: `${percentUsed}%` }}
          />
        </div>
        <div className="text-[11px] text-purple-600 text-right">
          Max Plaintext: ~{maxPlaintext} characters
        </div>
      </div>

      {/* Per-zone breakdown cards */}
      <div className="grid grid-cols-3 gap-2 pt-1">
        <div className="p-2.5 rounded-xl bg-purple-50/70 border border-purple-200/80 text-center">
          <div className="text-[11px] font-bold text-purple-950 flex items-center justify-center gap-1">
            <span className="w-2 h-2 rounded-full bg-purple-500"></span>
            Zone A (Smooth)
          </div>
          <p className="text-xs font-bold text-purple-900 mt-1">
            {countA.toLocaleString()} px
          </p>
          <p className="text-[10px] text-purple-600">EMD (&lt; 0.5 bpp)</p>
        </div>

        <div className="p-2.5 rounded-xl bg-pink-50/70 border border-pink-200/80 text-center">
          <div className="text-[11px] font-bold text-pink-950 flex items-center justify-center gap-1">
            <span className="w-2 h-2 rounded-full bg-pink-500"></span>
            Zone B (Medium)
          </div>
          <p className="text-xs font-bold text-pink-900 mt-1">
            {countB.toLocaleString()} px
          </p>
          <p className="text-[10px] text-pink-600">OPAP (2.0 bpp)</p>
        </div>

        <div className="p-2.5 rounded-xl bg-fuchsia-50/70 border border-fuchsia-200/80 text-center">
          <div className="text-[11px] font-bold text-fuchsia-950 flex items-center justify-center gap-1">
            <span className="w-2 h-2 rounded-full bg-fuchsia-500"></span>
            Zone C (Edges)
          </div>
          <p className="text-xs font-bold text-fuchsia-900 mt-1">
            {countC.toLocaleString()} px
          </p>
          <p className="text-[10px] text-fuchsia-600">OPAP (3.0 bpp)</p>
        </div>
      </div>

      {/* Costmap badge indicator */}
      <div className="flex items-center gap-1.5 text-[11px] text-purple-700 bg-pink-50/50 p-2 rounded-xl border border-pink-100">
        <Cpu className="w-3.5 h-3.5 text-pink-600 flex-shrink-0" />
        <span className="truncate">{cost_map_mode}</span>
      </div>
    </div>
  );
};
