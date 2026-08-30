import React from 'react';
import { Loader2, CheckCircle2, XCircle, Clock, Ban } from 'lucide-react';

interface Props {
  total: number;
  completed: number;
  failed: number;
  cancelled: number;
  status: string;
  currentFilename?: string;
}

export const BatchProgress: React.FC<Props> = ({
  total,
  completed,
  failed,
  cancelled,
  status,
  currentFilename,
}) => {
  const done = completed + failed + cancelled;
  const processing = Math.max(0, total - done);
  const pct = total > 0 ? Math.round((done / total) * 100) : 0;
  const queued = Math.max(0, total - completed - failed - cancelled - (status === 'running' ? 1 : 0));

  return (
    <div className="rounded-2xl border border-pink-200 bg-white/90 p-5 space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-bold text-purple-950">Batch Processing</h3>
        <span className="text-xs font-semibold px-2 py-1 rounded-lg bg-pink-50 text-pink-800 border border-pink-200 uppercase">
          {status.replace(/_/g, ' ')}
        </span>
      </div>

      <div>
        <div className="flex justify-between text-xs font-medium text-purple-700 mb-1">
          <span>
            {done} / {total} completed
          </span>
          <span>{pct}%</span>
        </div>
        <div className="h-3 rounded-full bg-pink-100 overflow-hidden">
          <div
            className="h-full rounded-full bg-gradient-to-r from-pink-500 to-purple-500 transition-all duration-300"
            style={{ width: `${pct}%` }}
          />
        </div>
      </div>

      <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-center">
        <div className="rounded-xl bg-emerald-50 border border-emerald-100 p-2">
          <CheckCircle2 className="w-4 h-4 mx-auto text-emerald-600" />
          <div className="text-lg font-bold text-emerald-800">{completed}</div>
          <div className="text-[10px] text-emerald-700">Completed</div>
        </div>
        <div className="rounded-xl bg-blue-50 border border-blue-100 p-2">
          <Loader2 className={`w-4 h-4 mx-auto text-blue-600 ${status === 'running' ? 'animate-spin' : ''}`} />
          <div className="text-lg font-bold text-blue-800">{status === 'running' ? Math.max(processing, 0) : 0}</div>
          <div className="text-[10px] text-blue-700">Processing</div>
        </div>
        <div className="rounded-xl bg-slate-50 border border-slate-100 p-2">
          <Clock className="w-4 h-4 mx-auto text-slate-500" />
          <div className="text-lg font-bold text-slate-700">{Math.max(queued, 0)}</div>
          <div className="text-[10px] text-slate-600">Queued</div>
        </div>
        <div className="rounded-xl bg-amber-50 border border-amber-100 p-2">
          <XCircle className="w-4 h-4 mx-auto text-amber-600" />
          <div className="text-lg font-bold text-amber-800">{failed}</div>
          <div className="text-[10px] text-amber-700">Failed</div>
        </div>
      </div>

      {cancelled > 0 && (
        <div className="flex items-center gap-2 text-xs text-purple-600">
          <Ban className="w-3.5 h-3.5" /> {cancelled} cancelled
        </div>
      )}

      {currentFilename && status === 'running' && (
        <p className="text-xs text-purple-700 truncate">
          Current: <span className="font-semibold">{currentFilename}</span>
        </p>
      )}
    </div>
  );
};
