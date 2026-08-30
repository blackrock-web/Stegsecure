import React, { useRef, useState } from 'react';
import { Upload, Image as ImageIcon, AlertCircle, FileCheck, X } from 'lucide-react';

interface ImageUploaderProps {
  label: string;
  sublabel?: string;
  selectedFile: File | null;
  onFileSelect: (file: File | null) => void;
  acceptFormats?: string;
}

export const ImageUploader: React.FC<ImageUploaderProps> = ({
  label,
  sublabel = "Lossless PNG or BMP cover image required",
  selectedFile,
  onFileSelect,
}) => {
  const [isDragging, setIsDragging] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileChange = (file: File | null) => {
    setErrorMsg(null);
    if (!file) {
      onFileSelect(null);
      setPreviewUrl(null);
      return;
    }

    const ext = file.name.split('.').pop()?.toLowerCase();
    if (ext !== 'png' && ext !== 'bmp') {
      setErrorMsg("Invalid file format. Steganography requires lossless PNG or BMP images only (JPEG compression destroys payload data).");
      onFileSelect(null);
      setPreviewUrl(null);
      return;
    }

    onFileSelect(file);
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileChange(e.dataTransfer.files[0]);
    }
  };

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <label className="text-sm font-bold text-purple-950 flex items-center gap-2">
          <ImageIcon className="w-4 h-4 text-pink-600" />
          <span>{label}</span>
        </label>
        <span className="text-xs font-semibold px-2 py-0.5 rounded-full bg-pink-100 text-pink-800 border border-pink-200">
          PNG / BMP Only
        </span>
      </div>

      {!selectedFile ? (
        <div
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current?.click()}
          className={`relative border-2 border-dashed rounded-2xl p-6 text-center cursor-pointer transition-all ${
            isDragging
              ? 'border-pink-500 bg-pink-100/60 scale-[1.01]'
              : 'border-pink-200 hover:border-pink-400 bg-white/70 hover:bg-pink-50/40'
          }`}
        >
          <input
            type="file"
            ref={fileInputRef}
            onChange={(e) => handleFileChange(e.target.files?.[0] || null)}
            accept=".png,.bmp,image/png,image/bmp"
            className="hidden"
          />
          <div className="w-12 h-12 mx-auto rounded-2xl bg-pink-100 text-pink-600 flex items-center justify-center mb-3 shadow-2xs">
            <Upload className="w-6 h-6" />
          </div>
          <p className="text-sm font-semibold text-purple-950">
            Click to upload or drag and drop cover image
          </p>
          <p className="text-xs text-purple-600 mt-1">{sublabel}</p>
        </div>
      ) : (
        <div className="relative rounded-2xl border border-pink-200 bg-white p-3 shadow-xs flex items-center gap-4">
          {previewUrl && (
            <img
              src={previewUrl}
              alt="Cover Preview"
              className="w-16 h-16 object-cover rounded-xl border border-pink-200 bg-pink-50/50"
            />
          )}
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2">
              <FileCheck className="w-4 h-4 text-emerald-600 flex-shrink-0" />
              <p className="text-sm font-bold text-purple-950 truncate">{selectedFile.name}</p>
            </div>
            <p className="text-xs text-purple-600 mt-0.5">
              {(selectedFile.size / 1024).toFixed(1)} KB • {selectedFile.type || 'PNG/BMP'}
            </p>
          </div>
          <button
            onClick={() => handleFileChange(null)}
            className="p-1.5 rounded-lg text-purple-400 hover:text-pink-600 hover:bg-pink-50 transition-colors"
            title="Remove file"
          >
            <X className="w-5 h-5" />
          </button>
        </div>
      )}

      {errorMsg && (
        <div className="p-3 rounded-xl bg-pink-50 border border-pink-300 text-pink-900 text-xs flex items-start gap-2 animate-fadeIn">
          <AlertCircle className="w-4 h-4 text-pink-600 flex-shrink-0 mt-0.5" />
          <span>{errorMsg}</span>
        </div>
      )}
    </div>
  );
};
