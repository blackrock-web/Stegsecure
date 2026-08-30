import React from 'react';
import { MessageSquareText, FileText, Trash2 } from 'lucide-react';

interface TextInputProps {
  value: string;
  onChange: (val: string) => void;
  maxBytes?: number;
}

export const TextInput: React.FC<TextInputProps> = ({ value, onChange, maxBytes }) => {
  const currentBytes = new TextEncoder().encode(value).length;
  // Crypto payload overhead is ~48 bytes (4 header + 16 salt + 12 nonce + 16 auth tag)
  const estEncryptedBytes = currentBytes + 48;
  const isOverCapacity = maxBytes ? estEncryptedBytes > maxBytes : false;

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <label className="text-sm font-bold text-purple-950 flex items-center gap-2">
          <MessageSquareText className="w-4 h-4 text-pink-600" />
          <span>Secret Text Message</span>
        </label>
        {value && (
          <button
            onClick={() => onChange('')}
            className="text-xs text-purple-500 hover:text-pink-600 flex items-center gap-1 transition-colors"
          >
            <Trash2 className="w-3.5 h-3.5" />
            Clear
          </button>
        )}
      </div>

      <div className="relative">
        <textarea
          value={value}
          onChange={(e) => onChange(e.target.value)}
          placeholder="Enter confidential message to AES-256 encrypt and hide inside cover image..."
          rows={4}
          className={`w-full p-3.5 text-sm rounded-2xl border transition-all resize-y outline-none font-sans ${
            isOverCapacity
              ? 'border-pink-500 bg-pink-50/30 text-pink-950 focus:ring-2 focus:ring-pink-400'
              : 'border-pink-200 bg-white text-purple-950 focus:border-pink-400 focus:ring-2 focus:ring-pink-200'
          }`}
        />
      </div>

      <div className="flex items-center justify-between text-xs px-1">
        <div className="flex items-center gap-2 text-purple-700">
          <FileText className="w-3.5 h-3.5 text-purple-500" />
          <span>
            {value.length} characters • {currentBytes} bytes plaintext (est. {estEncryptedBytes}B AES payload)
          </span>
        </div>

        {maxBytes && (
          <div
            className={`font-semibold px-2 py-0.5 rounded-md text-[11px] ${
              isOverCapacity
                ? 'bg-pink-100 text-pink-800 border border-pink-300'
                : 'bg-purple-100 text-purple-800 border border-purple-200'
            }`}
          >
            Cap: {estEncryptedBytes} / {maxBytes} B
          </div>
        )}
      </div>

      {isOverCapacity && (
        <p className="text-xs text-pink-700 font-medium px-1">
          ⚠️ Warning: Secret payload size exceeds maximum embeddable image capacity! Shorten your message or upload a larger image.
        </p>
      )}
    </div>
  );
};
