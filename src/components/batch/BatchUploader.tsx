import React, { useCallback, useRef, useState } from 'react';
import { Upload, X, Image as ImageIcon, AlertTriangle, Trash2 } from 'lucide-react';

export interface BatchFile {
  id: string;
  file: File;
  name: string;
  size: number;
  status: 'ready' | 'invalid';
  error?: string;
  width?: number;
  height?: number;
}

interface Props {
  files: BatchFile[];
  onChange: (files: BatchFile[]) => void;
  maxFiles?: number;
}

const ALLOWED = ['image/png', 'image/bmp', 'image/x-ms-bmp'];
const MAX_BYTES = 50 * 1024 * 1024;

function uid() {
  return Math.random().toString(36).slice(2, 10);
}

async function probeImage(file: File): Promise<{ ok: boolean; error?: string; w?: number; h?: number }> {
  if (!ALLOWED.includes(file.type) && !/\.(png|bmp)$/i.test(file.name)) {
    return { ok: false, error: 'PNG or BMP only' };
  }
  if (file.size > MAX_BYTES) return { ok: false, error: 'File too large (>50MB)' };
  return new Promise((resolve) => {
    const url = URL.createObjectURL(file);
    const img = new Image();
    img.onload = () => {
      URL.revokeObjectURL(url);
      if (img.width < 8 || img.height < 8) resolve({ ok: false, error: 'Too small' });
      else resolve({ ok: true, w: img.width, h: img.height });
    };
    img.onerror = () => {
      URL.revokeObjectURL(url);
      resolve({ ok: false, error: 'Unreadable image' });
    };
    img.src = url;
  });
}

export const BatchUploader: React.FC<Props> = ({ files, onChange, maxFiles = 100 }) => {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragOver, setDragOver] = useState(false);

  const addFiles = useCallback(
    async (list: FileList | File[]) => {
      const arr = Array.from(list);
      const existing = new Set(files.map((f) => f.name.toLowerCase()));
      const next = [...files];
      for (const file of arr) {
        if (next.length >= maxFiles) break;
        if (existing.has(file.name.toLowerCase())) continue;
        const probe = await probeImage(file);
        next.push({
          id: uid(),
          file,
          name: file.name,
          size: file.size,
          status: probe.ok ? 'ready' : 'invalid',
          error: probe.error,
          width: probe.w,
          height: probe.h,
        });
        existing.add(file.name.toLowerCase());
      }
      onChange(next);
    },
    [files, maxFiles, onChange]
  );

  const onDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    if (e.dataTransfer.files?.length) addFiles(e.dataTransfer.files);
  };

  const remove = (id: string) => onChange(files.filter((f) => f.id !== id));
  const clear = () => onChange([]);

  const ready = files.filter((f) => f.status === 'ready').length;

  return (
    <div className="space-y-3">
      <div
        onDragOver={(e) => {
          e.preventDefault();
          setDragOver(true);
        }}
        onDragLeave={() => setDragOver(false)}
        onDrop={onDrop}
        onClick={() => inputRef.current?.click()}
        className={`rounded-2xl border-2 border-dashed p-8 text-center cursor-pointer transition-colors ${
          dragOver
            ? 'border-pink-400 bg-pink-50'
            : 'border-pink-200 bg-white/80 hover:border-pink-300 hover:bg-pink-50/40'
        }`}
      >
        <Upload className="w-8 h-8 mx-auto text-pink-500 mb-2" />
        <p className="text-sm font-semibold text-purple-900">Drop images here or click to browse</p>
        <p className="text-xs text-purple-600 mt-1">PNG / BMP · up to {maxFiles} files · max 50 MB each</p>
        <input
          ref={inputRef}
          type="file"
          accept=".png,.bmp,image/png,image/bmp"
          multiple
          className="hidden"
          onChange={(e) => e.target.files && addFiles(e.target.files)}
        />
      </div>

      {files.length > 0 && (
        <div className="rounded-2xl border border-pink-200 bg-white/90 overflow-hidden">
          <div className="flex items-center justify-between px-4 py-2 border-b border-pink-100 bg-pink-50/50">
            <span className="text-sm font-semibold text-purple-900">
              {ready} / {files.length} images ready
            </span>
            <button
              type="button"
              onClick={clear}
              className="text-xs font-medium text-pink-700 hover:text-pink-900 flex items-center gap-1"
            >
              <Trash2 className="w-3.5 h-3.5" /> Clear all
            </button>
          </div>
          <div className="max-h-56 overflow-y-auto divide-y divide-pink-50">
            {files.map((f) => (
              <div key={f.id} className="flex items-center gap-3 px-4 py-2 text-sm">
                <ImageIcon className={`w-4 h-4 flex-shrink-0 ${f.status === 'ready' ? 'text-pink-500' : 'text-amber-500'}`} />
                <div className="flex-1 min-w-0">
                  <div className="font-medium text-purple-950 truncate">{f.name}</div>
                  <div className="text-xs text-purple-600">
                    {(f.size / 1024).toFixed(1)} KB
                    {f.width && f.height ? ` · ${f.width}×${f.height}` : ''}
                    {f.error && (
                      <span className="text-amber-700 ml-2 inline-flex items-center gap-0.5">
                        <AlertTriangle className="w-3 h-3" /> {f.error}
                      </span>
                    )}
                  </div>
                </div>
                <button type="button" onClick={() => remove(f.id)} className="text-purple-400 hover:text-pink-600">
                  <X className="w-4 h-4" />
                </button>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};
