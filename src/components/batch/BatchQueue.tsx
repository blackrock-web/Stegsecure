import React, { useState } from 'react';
import { CheckCircle2, XCircle, Loader2, Circle, Ban } from 'lucide-react';

export interface QueueItem {
  id: string;
  filename: string;
  status: string;
  processing_time_s?: number | null;
  error?: string | null;
}

interface Props {
  items: QueueItem[];
}

const FILTERS = ['all', 'completed', 'processing', 'queued', 'failed', 'cancelled'] as const;

export const BatchQueue: React.FC<Props> = ({ items }) => {
  const [filter, setFilter] = useState<(typeof FILTERS)[number]>('all');
  const filtered = filter === 'all' ? items : items.filter((i) => i.status === filter);

  const icon = (s: string) => {
    if (s === 'completed') return <CheckCircle2 className="w-4 h-4 text-emerald-600" />;
    if (s === 'failed') return <XCircle className="w-4 h-4 text-amber-600" />;
    if (s === 'processing') return <Loader2 className="w-4 h-4 text-blue-600 animate-spin" />;
    if (s === 'cancelled') return <Ban className="w-4 h-4 text-slate-400" />;
    return <Circle className="w-4 h-4 text-slate-300" />;
  };

  return (
    <div className="rounded-2xl border border-pink-200 bg-white/90 overflow-hidden">
      <div className="flex flex-wrap gap-1 px-3 py-2 border-b border-pink-100 bg-pink-50/40">
        {FILTERS.map((f) => (
          <button
            key={f}
            type="button"
            onClick={() => setFilter(f)}
            className={`px-2.5 py-1 text-[11px] font-semibold rounded-lg capitalize ${
              filter === f ? 'bg-white text-pink-800 shadow-xs border border-pink-200' : 'text-purple-600 hover:bg-pink-50'
            }`}
          >
            {f}
          </button>
        ))}
      </div>
      <div className="max-h-64 overflow-y-auto divide-y divide-pink-50">
        {filtered.length === 0 && (
          <p className="text-xs text-purple-500 p-4 text-center">No items</p>
        )}
        {filtered.map((it) => (
          <div key={it.id} className="flex items-center gap-3 px-4 py-2 text-sm">
            {icon(it.status)}
            <div className="flex-1 min-w-0">
              <div className="font-medium text-purple-950 truncate">{it.filename}</div>
              {it.error && <div className="text-[11px] text-amber-700 truncate">{it.error}</div>}
            </div>
            <div className="text-xs text-purple-600 capitalize w-20 text-right">{it.status}</div>
            <div className="text-xs text-purple-500 w-14 text-right tabular-nums">
              {it.processing_time_s != null ? `${it.processing_time_s.toFixed(2)}s` : '—'}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};
