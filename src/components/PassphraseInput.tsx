import React, { useState } from 'react';
import { Key, Eye, EyeOff, ShieldAlert } from 'lucide-react';

interface PassphraseInputProps {
  value: string;
  onChange: (val: string) => void;
  label?: string;
  placeholder?: string;
}

export const PassphraseInput: React.FC<PassphraseInputProps> = ({
  value,
  onChange,
  label = "Passphrase",
  placeholder = "Enter passphrase for PBKDF2-HMAC-SHA256 key derivation...",
}) => {
  const [showPass, setShowPass] = useState(false);

  // Simple passphrase strength indicator
  const getStrength = (pass: string) => {
    if (!pass) return { score: 0, text: 'Empty', color: 'bg-purple-200' };
    if (pass.length < 6) return { score: 1, text: 'Weak', color: 'bg-pink-300' };
    if (pass.length < 10) return { score: 2, text: 'Medium', color: 'bg-purple-400' };
    return { score: 3, text: 'Strong', color: 'bg-pink-600' };
  };

  const strength = getStrength(value);

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <label className="text-sm font-bold text-purple-950 flex items-center gap-2">
          <Key className="w-4 h-4 text-purple-600" />
          <span>{label}</span>
        </label>
        {value && (
          <span className="text-xs text-purple-700 font-semibold">
            Strength: <span className="text-purple-950">{strength.text}</span>
          </span>
        )}
      </div>

      <div className="relative">
        <input
          type={showPass ? 'text' : 'password'}
          value={value}
          onChange={(e) => onChange(e.target.value)}
          placeholder={placeholder}
          className="w-full pl-3.5 pr-10 py-2.5 text-sm rounded-xl border border-pink-200 bg-white text-purple-950 focus:border-pink-400 focus:ring-2 focus:ring-pink-200 outline-none transition-all font-mono"
        />
        <button
          type="button"
          onClick={() => setShowPass(!showPass)}
          className="absolute right-3 top-1/2 -translate-y-1/2 text-purple-400 hover:text-purple-700 transition-colors p-1"
          title={showPass ? 'Hide passphrase' : 'Show passphrase'}
        >
          {showPass ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
        </button>
      </div>

      <div className="flex items-center justify-between gap-2 px-1 text-xs text-purple-600">
        <div className="flex items-center gap-1.5">
          <ShieldAlert className="w-3.5 h-3.5 text-purple-500" />
          <span>AES-256-GCM + PBKDF2 (200k iterations). Never stored or logged.</span>
        </div>

        {value && (
          <div className="flex items-center gap-1 w-16">
            <div className={`h-1 flex-1 rounded-full ${strength.score >= 1 ? strength.color : 'bg-pink-100'}`} />
            <div className={`h-1 flex-1 rounded-full ${strength.score >= 2 ? strength.color : 'bg-pink-100'}`} />
            <div className={`h-1 flex-1 rounded-full ${strength.score >= 3 ? strength.color : 'bg-pink-100'}`} />
          </div>
        )}
      </div>
    </div>
  );
};
