import React, { useMemo, useState } from 'react';

interface Item {
  id: string;
  filename: string;
  status: string;
  processing_time_s?: number | null;
  error?: string | null;
  strategy?: string | null;
  bpp_target?: number | null;
  result?: {
    metrics?: Record<string, number>;
    security?: Record<string, number>;
    success?: boolean;
    decrypted_text?: string;
  } | null;
}

interface Props {
  items: Item[];
  jobType: string;
}

type SortKey = 'filename' | 'psnr' | 'ssim' | 'time' | 'status';

export const BatchResults: React.FC<Props> = ({ items, jobType }) => {
  const [sort, setSort] = useState<SortKey>('filename');
  const [filter, setFilter] = useState<'all' | 'completed' | 'failed'>('all');

  const rows = useMemo(() => {
    let list = items.filter((i) => filter === 'all' || i.status === filter);
    list = [...list].sort((a, b) => {
      if (sort === 'filename') return a.filename.localeCompare(b.filename);
      if (sort === 'status') return a.status.localeCompare(b.status);
      if (sort === 'time') return (b.processing_time_s || 0) - (a.processing_time_s || 0);
      if (sort === 'psnr')
        return (b.result?.metrics?.psnr_db || 0) - (a.result?.metrics?.psnr_db || 0);
      if (sort === 'ssim')
        return (b.result?.metrics?.ssim || 0) - (a.result?.metrics?.ssim || 0);
      return 0;
    });
    return list;
  }, [items, sort, filter]);

  return (
    <div className="rounded-2xl border border-pink-200 bg-white/90 overflow-hidden">
      <div className="flex flex-wrap items-center gap-2 px-4 py-2 border-b border-pink-100 bg-pink-50/40">
        <span className="text-sm font-bold text-purple-950 mr-2">Results</span>
        {(['all', 'completed', 'failed'] as const).map((f) => (
          <button
            key={f}
            type="button"
            onClick={() => setFilter(f)}
            className={`px-2 py-1 text-[11px] font-semibold rounded-lg capitalize ${
              filter === f ? 'bg-white border border-pink-200 text-pink-800' : 'text-purple-600'
            }`}
          >
            {f}
          </button>
        ))}
        <select
          value={sort}
          onChange={(e) => setSort(e.target.value as SortKey)}
          className="ml-auto text-xs border border-pink-200 rounded-lg px-2 py-1 bg-white"
        >
          <option value="filename">Sort: Filename</option>
          <option value="status">Sort: Status</option>
          <option value="psnr">Sort: PSNR</option>
          <option value="ssim">Sort: SSIM</option>
          <option value="time">Sort: Time</option>
        </select>
      </div>
      <div className="overflow-x-auto max-h-80">
        <table className="w-full text-xs">
          <thead className="bg-pink-50/60 sticky top-0">
            <tr className="text-left text-purple-700">
              <th className="px-3 py-2 font-semibold">Image</th>
              <th className="px-3 py-2 font-semibold">Status</th>
              {jobType !== 'decode' && (
                <>
                  <th className="px-3 py-2 font-semibold">PSNR</th>
                  <th className="px-3 py-2 font-semibold">SSIM</th>
                  <th className="px-3 py-2 font-semibold">bpp</th>
                  <th className="px-3 py-2 font-semibold">% Mod</th>
                  <th className="px-3 py-2 font-semibold">Suspicion</th>
                </>
              )}
              {jobType === 'decode' && <th className="px-3 py-2 font-semibold">Payload</th>}
              <th className="px-3 py-2 font-semibold">Time</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-pink-50">
            {rows.map((r) => {
              const m = r.result?.metrics || {};
              const s = r.result?.security || {};
              return (
                <tr key={r.id} className="hover:bg-pink-50/30">
                  <td className="px-3 py-2 font-medium text-purple-950 max-w-[160px] truncate">
                    {r.filename}
                  </td>
                  <td className="px-3 py-2 capitalize">
                    <span
                      className={
                        r.status === 'completed'
                          ? 'text-emerald-700'
                          : r.status === 'failed'
                          ? 'text-amber-700'
                          : 'text-purple-600'
                      }
                    >
                      {r.status}
                    </span>
                    {r.error && (
                      <div className="text-[10px] text-amber-600 max-w-[140px] truncate" title={r.error}>
                        {r.error}
                      </div>
                    )}
                  </td>
                  {jobType !== 'decode' && (
                    <>
                      <td className="px-3 py-2 tabular-nums">
                        {m.psnr_db != null ? Number(m.psnr_db).toFixed(2) : '—'}
                      </td>
                      <td className="px-3 py-2 tabular-nums">
                        {m.ssim != null ? Number(m.ssim).toFixed(4) : '—'}
                      </td>
                      <td className="px-3 py-2 tabular-nums">
                        {m.achieved_bpp != null ? Number(m.achieved_bpp).toFixed(3) : '—'}
                      </td>
                      <td className="px-3 py-2 tabular-nums">
                        {m.modified_pixel_percentage != null
                          ? Number(m.modified_pixel_percentage).toFixed(2)
                          : '—'}
                      </td>
                      <td className="px-3 py-2 tabular-nums">
                        {s.composite_suspicion != null
                          ? Number(s.composite_suspicion).toFixed(3)
                          : '—'}
                      </td>
                    </>
                  )}
                  {jobType === 'decode' && (
                    <td className="px-3 py-2 max-w-[200px] truncate text-purple-800">
                      {r.result?.decrypted_text
                        ? r.result.decrypted_text.slice(0, 40) +
                          (r.result.decrypted_text.length > 40 ? '…' : '')
                        : r.status === 'completed'
                        ? '(empty)'
                        : '—'}
                    </td>
                  )}
                  <td className="px-3 py-2 tabular-nums text-purple-600">
                    {r.processing_time_s != null ? `${r.processing_time_s.toFixed(2)}s` : '—'}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
};
