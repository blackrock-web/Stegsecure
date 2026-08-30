import React, { useState } from 'react';
import { ImageUploader } from './ImageUploader';
import { PassphraseInput } from './PassphraseInput';
import { KeyRound, ShieldCheck, AlertCircle, Copy, Check, Lock, Loader2, Sparkles } from 'lucide-react';
import { TuningConfig } from '../types';

interface DecodePageProps {
  config: TuningConfig;
}

export const DecodePage: React.FC<DecodePageProps> = ({ config }) => {
  const [stegoFile, setStegoFile] = useState<File | null>(null);
  const [passphrase, setPassphrase] = useState('');
  const [loading, setLoading] = useState(false);

  const [decryptedText, setDecryptedText] = useState<string | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);

  const handleDecode = async () => {
    if (!stegoFile || !passphrase) return;

    setLoading(true);
    setErrorMsg(null);
    setDecryptedText(null);

    const formData = new FormData();
    formData.append('file', stegoFile);
    formData.append('passphrase', passphrase);
    formData.append('thresh_a', config.threshA.toString());
    formData.append('thresh_b', config.threshB.toString());
    formData.append('gamma', config.gamma.toString());
    formData.append('kb_bits', config.kbBits.toString());
    formData.append('kc_bits', config.kcBits.toString());
    formData.append('cost_map_mode', config.costMapMode || 'fast');
    formData.append('emd_n', (config.emdN || 2).toString());


    try {
      const res = await fetch('/api/decode', {
        method: 'POST',
        body: formData,
      });

      const raw = await res.text();
      let data: any = {};
      try {
        data = raw ? JSON.parse(raw) : {};
      } catch {
        throw new Error(`Server error (${res.status}). Try Fast mode and the same settings used for encoding.`);
      }
      if (!res.ok) {
        const detail = typeof data.detail === 'string'
          ? data.detail
          : (data.message || 'Message could not be decrypted — wrong passphrase or corrupted image.');
        throw new Error(detail);
      }
      if (!data.decrypted_text) {
        throw new Error('No message recovered. Use the same passphrase and config as encoding.');
      }
      setDecryptedText(data.decrypted_text);
    } catch (err: any) {
      setErrorMsg(err.message || 'Message could not be decrypted — wrong passphrase or corrupted image.');
    } finally {
      setLoading(false);
    }
  };

  const handleCopy = () => {
    if (decryptedText) {
      navigator.clipboard.writeText(decryptedText);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };

  return (
    <div className="max-w-3xl mx-auto space-y-6 py-4 animate-fadeIn">
      {/* Title Header Card */}
      <div className="p-6 rounded-3xl bg-white/90 border border-pink-200 shadow-sm space-y-2">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-2xl bg-purple-100 text-purple-800 flex items-center justify-center shadow-2xs">
            <KeyRound className="w-6 h-6" />
          </div>
          <div>
            <h2 className="text-xl font-bold text-purple-950">Extract &amp; Decrypt Steganographic Payload</h2>
            <p className="text-xs text-purple-700">
              Recomputes identical CNN feature cost map, reverses EMD/OPAP extraction, and verifies AES-256-GCM auth tag.
            </p>
          </div>
        </div>
      </div>

      {/* Stego File & Passphrase Form */}
      <div className="p-6 rounded-3xl bg-white/90 border border-pink-200 shadow-sm space-y-5">
        <ImageUploader
          label="Upload Stego Image"
          sublabel="Select the PNG or BMP stego image containing encrypted hidden text"
          selectedFile={stegoFile}
          onFileSelect={setStegoFile}
        />

        <PassphraseInput
          value={passphrase}
          onChange={setPassphrase}
          label="Passphrase"
          placeholder="Enter the passphrase used during encoding..."
        />

        <button
          onClick={handleDecode}
          disabled={!stegoFile || !passphrase || loading}
          className="w-full py-3.5 px-6 rounded-2xl text-sm font-bold text-white bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-700 hover:to-pink-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all shadow-md shadow-pink-200 flex items-center justify-center gap-2"
        >
          {loading ? (
            <>
              <Loader2 className="w-5 h-5 animate-spin" />
              <span>Running CNN Cost Map &amp; Reversing EMD/OPAP...</span>
            </>
          ) : (
            <>
              <ShieldCheck className="w-5 h-5" />
              <span>Extract &amp; Decrypt Payload</span>
            </>
          )}
        </button>
      </div>

      {/* Decrypted Output Card */}
      {decryptedText && (
        <div className="p-6 rounded-3xl bg-white border-2 border-emerald-300 shadow-md space-y-4 animate-fadeIn">
          <div className="flex items-center justify-between border-b border-emerald-100 pb-3">
            <div className="flex items-center gap-2">
              <Sparkles className="w-5 h-5 text-emerald-600" />
              <h3 className="text-base font-bold text-emerald-950">Payload Decrypted Successfully</h3>
            </div>

            <button
              onClick={handleCopy}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-emerald-50 hover:bg-emerald-100 border border-emerald-200 text-xs font-bold text-emerald-800 transition-colors"
            >
              {copied ? (
                <>
                  <Check className="w-4 h-4 text-emerald-600" />
                  <span>Copied!</span>
                </>
              ) : (
                <>
                  <Copy className="w-4 h-4 text-emerald-600" />
                  <span>Copy Text</span>
                </>
              )}
            </button>
          </div>

          <div className="p-4 rounded-2xl bg-emerald-50/40 border border-emerald-200 text-sm font-sans text-emerald-950 whitespace-pre-wrap break-words min-h-24">
            {decryptedText}
          </div>
        </div>
      )}

      {/* Clean Error Message Box */}
      {errorMsg && (
        <div className="p-5 rounded-3xl bg-pink-50 border-2 border-pink-300 text-pink-950 shadow-xs flex items-start gap-3 animate-fadeIn">
          <AlertCircle className="w-6 h-6 text-pink-600 flex-shrink-0 mt-0.5" />
          <div className="space-y-1">
            <h4 className="text-sm font-bold text-pink-900">Decryption Failed</h4>
            <p className="text-xs font-medium text-pink-800">{errorMsg}</p>
          </div>
        </div>
      )}
    </div>
  );
};
