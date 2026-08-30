import React, { useCallback, useEffect, useRef, useState } from 'react';
import {
  Layers,
  Play,
  Square,
  RotateCcw,
  Download,
  Loader2,
  FlaskConical,
} from 'lucide-react';
import { BatchUploader, BatchFile } from './BatchUploader';
import { BatchProgress } from './BatchProgress';
import { BatchQueue } from './BatchQueue';
import { BatchResults } from './BatchResults';
import { BatchSummary } from './BatchSummary';
import { TuningConfig } from '../../types';

const STRATEGIES = [
  { id: 'emd_opap', label: 'EMD + OPAP (Classical)' },
  { id: 'cnn_emd_opap', label: 'CNN CostMap + EMD/OPAP' },
  { id: 'cnn_emd_opap_adv', label: 'CNN + EMD/OPAP + Adversarial' },
  { id: 'cnn_stc_emd_opap', label: 'CNN + STC Approx + EMD/OPAP' },
  { id: 'cnn_stc_emd_opap_adv', label: 'Full Pipeline (CNN+STC+Adv)' },
];

interface Props {
  tuningConfig: TuningConfig;
}

export const BatchLab: React.FC<Props> = ({ tuningConfig }) => {
  const [jobType, setJobType] = useState<'encode' | 'decode' | 'experiment'>('encode');
  const [files, setFiles] = useState<BatchFile[]>([]);
  const [secretText, setSecretText] = useState('SecureStegVault batch payload.');
  const [passphrase, setPassphrase] = useState('batch-passphrase');
  const [strategy, setStrategy] = useState('cnn_emd_opap');
  const [workers, setWorkers] = useState(2);
  const [messageMode, setMessageMode] = useState<'same' | 'per_image'>('same');
  const [expStrategies, setExpStrategies] = useState<string[]>(['emd_opap', 'cnn_emd_opap']);
  const [expBpps, setExpBpps] = useState<number[]>([0.1, 0.2]);

  const [jobId, setJobId] = useState<string | null>(null);
  const [job, setJob] = useState<any>(null);
  const [summary, setSummary] = useState<any>(null);
  const [starting, setStarting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const pollRef = useRef<number | null>(null);

  const stopPoll = () => {
    if (pollRef.current) {
      window.clearInterval(pollRef.current);
      pollRef.current = null;
    }
  };

  const pollJob = useCallback(async (id: string) => {
    try {
      const res = await fetch(`/api/batch/jobs/${id}`);
      if (!res.ok) return;
      const data = await res.json();
      setJob(data);
      if (['completed', 'completed_with_errors', 'failed', 'cancelled'].includes(data.status)) {
        stopPoll();
        const s = await fetch(`/api/batch/jobs/${id}/summary`);
        if (s.ok) setSummary(await s.json());
      }
    } catch {
      /* ignore transient */
    }
  }, []);

  useEffect(() => () => stopPoll(), []);

  const startBatch = async () => {
    setError(null);
    setSummary(null);
    const ready = files.filter((f) => f.status === 'ready');
    if (ready.length === 0) {
      setError('Add at least one valid image.');
      return;
    }
    if ((jobType === 'encode' || jobType === 'experiment') && messageMode === 'same' && !secretText.trim()) {
      setError('Secret text is required.');
      return;
    }
    if (!passphrase) {
      setError('Passphrase is required.');
      return;
    }

    setStarting(true);
    try {
      const form = new FormData();
      form.append('type', jobType);
      form.append('secret_text', secretText);
      form.append('passphrase', passphrase);
      form.append('strategy', strategy);
      form.append('workers', String(workers));
      form.append('message_mode', messageMode);
      form.append('cost_map_mode', tuningConfig.costMapMode || 'cnn');
      form.append('thresh_a', String(tuningConfig.threshA));
      form.append('thresh_b', String(tuningConfig.threshB));
      form.append('gamma', String(tuningConfig.gamma));
      form.append('kb_bits', String(tuningConfig.kbBits));
      form.append('kc_bits', String(tuningConfig.kcBits));
      form.append('emd_n', String(tuningConfig.emdN || 2));
      form.append('adversarial_strength', String(tuningConfig.adversarialStrength || 0));
      form.append('engine', 'python');
      if (jobType === 'experiment') {
        form.append('strategies', JSON.stringify(expStrategies));
        form.append('bpp_list', JSON.stringify(expBpps));
      }
      for (const f of ready) {
        form.append('files', f.file, f.name);
      }

      const res = await fetch('/api/batch/jobs', { method: 'POST', body: form });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || `HTTP ${res.status}`);
      setJobId(data.job_id);
      setJob({ job_id: data.job_id, status: data.status, total: data.total, completed: 0, failed: 0, items: [] });
      stopPoll();
      pollRef.current = window.setInterval(() => pollJob(data.job_id), 800);
      pollJob(data.job_id);
    } catch (e: any) {
      setError(e.message || 'Failed to start batch');
    } finally {
      setStarting(false);
    }
  };

  const cancelBatch = async () => {
    if (!jobId) return;
    await fetch(`/api/batch/jobs/${jobId}/cancel`, { method: 'POST' });
    pollJob(jobId);
  };

  const retryFailed = async () => {
    if (!jobId) return;
    await fetch(`/api/batch/jobs/${jobId}/retry`, { method: 'POST' });
    stopPoll();
    pollRef.current = window.setInterval(() => pollJob(jobId), 800);
    pollJob(jobId);
  };

  const download = (fmt: 'json' | 'csv' | 'zip') => {
    if (!jobId) return;
    window.open(`/api/batch/jobs/${jobId}/export?format=${fmt}`, '_blank');
  };

  const current =
    job?.items?.find((i: any) => i.status === 'processing')?.filename ||
    job?.items?.find((i: any) => i.status === 'queued')?.filename;

  return (
    <div className="max-w-6xl mx-auto space-y-6 py-4 animate-fadeIn">
      <div className="p-6 rounded-3xl bg-white/90 border border-pink-200 shadow-xs space-y-2">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-2xl bg-pink-100 text-pink-600 flex items-center justify-center">
            <Layers className="w-6 h-6" />
          </div>
          <div>
            <h2 className="text-xl font-bold text-purple-950">Batch Lab</h2>
            <p className="text-xs text-purple-700 font-medium">
              Multi-image encode / decode / experiment queue. Uses the existing single-image pipeline per item.
            </p>
          </div>
        </div>

        {/* Job type */}
        <div className="flex flex-wrap gap-2 pt-2">
          {(['encode', 'decode', 'experiment'] as const).map((t) => (
            <button
              key={t}
              type="button"
              onClick={() => setJobType(t)}
              className={`px-3 py-1.5 text-xs font-semibold rounded-xl capitalize border ${
                jobType === t
                  ? 'bg-pink-500 text-white border-pink-500'
                  : 'bg-white text-purple-800 border-pink-200 hover:bg-pink-50'
              }`}
            >
              {t === 'experiment' ? (
                <span className="inline-flex items-center gap-1">
                  <FlaskConical className="w-3.5 h-3.5" /> Experiment
                </span>
              ) : (
                t
              )}
            </button>
          ))}
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Left: upload + config */}
        <div className="space-y-4">
          <BatchUploader files={files} onChange={setFiles} />

          <div className="rounded-2xl border border-pink-200 bg-white/90 p-4 space-y-3">
            <h3 className="text-sm font-bold text-purple-950">Configuration</h3>

            {jobType !== 'decode' && (
              <>
                <div>
                  <label className="text-[11px] font-semibold text-purple-800">Secret message</label>
                  <textarea
                    value={secretText}
                    onChange={(e) => setSecretText(e.target.value)}
                    rows={2}
                    className="w-full mt-1 text-sm rounded-xl border border-pink-200 px-3 py-2 focus:ring-2 focus:ring-pink-300 outline-none"
                  />
                </div>
                <div className="flex gap-3">
                  <label className="flex items-center gap-1.5 text-xs text-purple-800">
                    <input
                      type="radio"
                      checked={messageMode === 'same'}
                      onChange={() => setMessageMode('same')}
                    />
                    Same message → all images
                  </label>
                  <label className="flex items-center gap-1.5 text-xs text-purple-800">
                    <input
                      type="radio"
                      checked={messageMode === 'per_image'}
                      onChange={() => setMessageMode('per_image')}
                    />
                    Per-image messages
                  </label>
                </div>
              </>
            )}

            <div>
              <label className="text-[11px] font-semibold text-purple-800">Passphrase (AES-256-GCM)</label>
              <input
                type="password"
                value={passphrase}
                onChange={(e) => setPassphrase(e.target.value)}
                className="w-full mt-1 text-sm rounded-xl border border-pink-200 px-3 py-2 focus:ring-2 focus:ring-pink-300 outline-none"
              />
            </div>

            {jobType !== 'experiment' && (
              <div>
                <label className="text-[11px] font-semibold text-purple-800">Strategy</label>
                <select
                  value={strategy}
                  onChange={(e) => setStrategy(e.target.value)}
                  className="w-full mt-1 text-sm rounded-xl border border-pink-200 px-3 py-2 bg-white"
                >
                  {STRATEGIES.map((s) => (
                    <option key={s.id} value={s.id}>
                      {s.label}
                    </option>
                  ))}
                </select>
              </div>
            )}

            {jobType === 'experiment' && (
              <div className="space-y-2">
                <div className="text-[11px] font-semibold text-purple-800">Strategies</div>
                <div className="flex flex-wrap gap-2">
                  {STRATEGIES.map((s) => (
                    <label key={s.id} className="text-[11px] flex items-center gap-1 text-purple-800">
                      <input
                        type="checkbox"
                        checked={expStrategies.includes(s.id)}
                        onChange={(e) =>
                          setExpStrategies((prev) =>
                            e.target.checked ? [...prev, s.id] : prev.filter((x) => x !== s.id)
                          )
                        }
                      />
                      {s.label.split('(')[0].trim()}
                    </label>
                  ))}
                </div>
                <div className="text-[11px] font-semibold text-purple-800">Payload rates (bpp)</div>
                <div className="flex flex-wrap gap-2">
                  {[0.05, 0.1, 0.2, 0.3, 0.4, 0.5].map((b) => (
                    <label key={b} className="text-[11px] flex items-center gap-1 text-purple-800">
                      <input
                        type="checkbox"
                        checked={expBpps.includes(b)}
                        onChange={(e) =>
                          setExpBpps((prev) =>
                            e.target.checked ? [...prev, b] : prev.filter((x) => x !== b)
                          )
                        }
                      />
                      {b.toFixed(2)}
                    </label>
                  ))}
                </div>
              </div>
            )}

            <div>
              <label className="text-[11px] font-semibold text-purple-800">Workers</label>
              <input
                type="number"
                min={1}
                max={8}
                value={workers}
                onChange={(e) => setWorkers(Number(e.target.value))}
                className="w-24 mt-1 text-sm rounded-xl border border-pink-200 px-3 py-2"
              />
            </div>

            {error && (
              <div className="text-sm text-amber-800 bg-amber-50 border border-amber-200 rounded-xl px-3 py-2">
                {error}
              </div>
            )}

            <div className="flex flex-wrap gap-2 pt-1">
              <button
                type="button"
                onClick={startBatch}
                disabled={starting}
                className="inline-flex items-center gap-2 px-4 py-2 rounded-xl bg-pink-600 hover:bg-pink-700 text-white text-sm font-semibold disabled:opacity-50"
              >
                {starting ? <Loader2 className="w-4 h-4 animate-spin" /> : <Play className="w-4 h-4" />}
                Start Batch
              </button>
              {jobId && job && !['completed', 'completed_with_errors', 'failed', 'cancelled'].includes(job.status) && (
                <button
                  type="button"
                  onClick={cancelBatch}
                  className="inline-flex items-center gap-2 px-4 py-2 rounded-xl border border-pink-300 text-pink-800 text-sm font-semibold hover:bg-pink-50"
                >
                  <Square className="w-4 h-4" /> Cancel
                </button>
              )}
              {job?.failed > 0 && (
                <button
                  type="button"
                  onClick={retryFailed}
                  className="inline-flex items-center gap-2 px-4 py-2 rounded-xl border border-purple-300 text-purple-800 text-sm font-semibold hover:bg-purple-50"
                >
                  <RotateCcw className="w-4 h-4" /> Retry Failed
                </button>
              )}
            </div>
          </div>
        </div>

        {/* Right: progress + queue */}
        <div className="space-y-4">
          {job && (
            <BatchProgress
              total={job.total || 0}
              completed={job.completed || 0}
              failed={job.failed || 0}
              cancelled={job.cancelled || 0}
              status={job.status || 'queued'}
              currentFilename={current}
            />
          )}
          {job?.items && <BatchQueue items={job.items} />}
          {jobId && ['completed', 'completed_with_errors', 'failed', 'cancelled'].includes(job?.status) && (
            <div className="flex flex-wrap gap-2">
              <button
                type="button"
                onClick={() => download('zip')}
                className="inline-flex items-center gap-1.5 px-3 py-2 text-xs font-semibold rounded-xl bg-purple-600 text-white hover:bg-purple-700"
              >
                <Download className="w-3.5 h-3.5" /> ZIP
              </button>
              <button
                type="button"
                onClick={() => download('csv')}
                className="inline-flex items-center gap-1.5 px-3 py-2 text-xs font-semibold rounded-xl border border-pink-200 text-purple-800 hover:bg-pink-50"
              >
                CSV
              </button>
              <button
                type="button"
                onClick={() => download('json')}
                className="inline-flex items-center gap-1.5 px-3 py-2 text-xs font-semibold rounded-xl border border-pink-200 text-purple-800 hover:bg-pink-50"
              >
                JSON
              </button>
            </div>
          )}
        </div>
      </div>

      {job?.items && <BatchResults items={job.items} jobType={job.type || jobType} />}
      <BatchSummary summary={summary} />
    </div>
  );
};
