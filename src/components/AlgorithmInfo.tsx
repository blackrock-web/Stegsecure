import React from 'react';
import { BookOpen, ShieldCheck, Cpu, Database, Eye, CheckCircle2 } from 'lucide-react';

export const AlgorithmInfo: React.FC = () => {
  return (
    <div className="max-w-4xl mx-auto space-y-6 py-4 animate-fadeIn">
      {/* Title Header */}
      <div className="p-6 rounded-3xl bg-white/90 border border-pink-200 shadow-xs space-y-2">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-2xl bg-pink-100 text-pink-600 flex items-center justify-center shadow-2xs">
            <BookOpen className="w-6 h-6" />
          </div>
          <div>
            <h2 className="text-xl font-bold text-purple-950">
              Steganography Architecture &amp; Mathematical Specifications
            </h2>
            <p className="text-xs text-purple-700 font-medium">
              A 5-phase hybrid pipeline combining AES-256-GCM, CNN feature cost mapping, EMD (Zhang &amp; Wang 2006), and OPAP (Chan &amp; Cheng 2004).
            </p>
          </div>
        </div>
      </div>

      {/* 5-Phase Pipeline Cards */}
      <div className="space-y-4">
        {/* Phase 1 */}
        <div className="p-5 rounded-3xl bg-white border border-pink-200 shadow-xs space-y-2">
          <div className="flex items-center gap-2 text-sm font-bold text-purple-950">
            <span className="w-6 h-6 rounded-full bg-pink-100 text-pink-800 text-xs flex items-center justify-center font-black">
              1
            </span>
            <span>Phase 1: Cryptographic Pre-Processing (AES-256-GCM)</span>
          </div>
          <p className="text-xs text-purple-800 leading-relaxed pl-8">
            The secret text message is encrypted using AES-256-GCM. The key is derived via 200,000 iterations of PBKDF2-HMAC-SHA256 from the user passphrase and a random 16-byte salt. The output payload byte array contains a 32-bit big-endian length header, 16-byte salt, 12-byte nonce, ciphertext, and 16-byte GCM authentication tag.
          </p>
          <div className="pl-8 text-[11px] font-mono text-purple-900 bg-pink-50/50 p-2.5 rounded-xl border border-pink-100">
            payload_to_embed = [4-byte length] + [16-byte salt] + [12-byte nonce] + [ciphertext + 16-byte tag]
          </div>
        </div>

        {/* Phase 2 */}
        <div className="p-5 rounded-3xl bg-white border border-pink-200 shadow-xs space-y-2">
          <div className="flex items-center gap-2 text-sm font-bold text-purple-950">
            <span className="w-6 h-6 rounded-full bg-purple-100 text-purple-800 text-xs flex items-center justify-center font-black">
              2
            </span>
            <span>Phase 2: CNN-Driven Image Profiling &amp; Cost Map Generation</span>
          </div>
          <p className="text-xs text-purple-800 leading-relaxed pl-8">
            The cover image is processed through a pretrained VGG16 CNN (ImageNet weights, inference-only) to extract activation features from block2_conv2. Channel dimensions are averaged and upsampled via bilinear interpolation to form cost map <code className="font-mono font-bold">ρ</code>. This is blended with classical Canny and Sobel edge responses:
          </p>
          <div className="pl-8 text-[11px] font-mono text-purple-900 bg-purple-50/50 p-2.5 rounded-xl border border-purple-100">
            H(x, y) = 0.5 * Canny(x, y) + 0.5 * Sobel(x, y)<br />
            final_map = γ * ρ + (1 - γ) * H  (default γ = 0.7)
          </div>
        </div>

        {/* Phase 3 */}
        <div className="p-5 rounded-3xl bg-white border border-pink-200 shadow-xs space-y-2">
          <div className="flex items-center gap-2 text-sm font-bold text-purple-950">
            <span className="w-6 h-6 rounded-full bg-fuchsia-100 text-fuchsia-800 text-xs flex items-center justify-center font-black">
              3
            </span>
            <span>Phase 3: Adaptive Regional Payload Allocation</span>
          </div>
          <p className="text-xs text-purple-800 leading-relaxed pl-8">
            Pixels are thresholded into 3 zones based on cost map values:
          </p>
          <ul className="pl-12 text-xs text-purple-800 list-disc space-y-1">
            <li><strong>Zone A (Smooth, Cost &lt; 0.35):</strong> Low texture, high detection risk. Allocated to EMD algorithm (&lt; 0.5 - 1.16 bpp).</li>
            <li><strong>Zone B (Medium Texture, 0.35 ≤ Cost &lt; 0.65):</strong> Moderate detail. Embedded via k=2 bit LSB + OPAP (2.0 bpp).</li>
            <li><strong>Zone C (High Complexity, Cost ≥ 0.65):</strong> Complex edges. Embedded via k=3 bit LSB + OPAP (3.0 bpp).</li>
          </ul>
        </div>

        {/* Phase 4 */}
        <div className="p-5 rounded-3xl bg-white border border-pink-200 shadow-xs space-y-2">
          <div className="flex items-center gap-2 text-sm font-bold text-purple-950">
            <span className="w-6 h-6 rounded-full bg-pink-100 text-pink-800 text-xs flex items-center justify-center font-black">
              4
            </span>
            <span>Phase 4: Hybrid EMD-OPAP Embedding Engine</span>
          </div>
          <div className="pl-8 space-y-2 text-xs text-purple-800">
            <p><strong>EMD (Zhang &amp; Wang 2006) for Zone A:</strong></p>
            <div className="font-mono text-[11px] bg-pink-50/50 p-2.5 rounded-xl border border-pink-100">
              f(g1, g2) = (g1*1 + g2*2) mod 5<br />
              s = (d - f) mod 5<br />
              if s == 1: g1 += 1 | if s == 2: g2 += 1 | if s == 3: g2 -= 1 | if s == 4: g1 -= 1
            </div>
            <p>Guarantees at most ±1 modification on at most 1 pixel per pair!</p>

            <p className="pt-2"><strong>OPAP (Chan &amp; Cheng 2004) for Zone B/C:</strong></p>
            <div className="font-mono text-[11px] bg-pink-50/50 p-2.5 rounded-xl border border-pink-100">
              p&apos;_i = (p_i &amp; ~mask) | secret_k_bits<br />
              Candidates: [p&apos;_i, p&apos;_i + 2^k, p&apos;_i - 2^k]<br />
              Select candidate p&apos;&apos;_i in [0, 255] minimizing |p&apos;&apos;_i - p_i|.
            </div>
          </div>
        </div>

        {/* Phase 5 */}
        <div className="p-5 rounded-3xl bg-white border border-pink-200 shadow-xs space-y-2">
          <div className="flex items-center gap-2 text-sm font-bold text-purple-950">
            <span className="w-6 h-6 rounded-full bg-emerald-100 text-emerald-800 text-xs flex items-center justify-center font-black">
              5
            </span>
            <span>Phase 5: Output Generation &amp; Pixel Inspection</span>
          </div>
          <p className="text-xs text-purple-800 leading-relaxed pl-8">
            Generates stego PNG image, amplified red/magenta heatmap overlay, pure binary modification mask, and calculates MSE, PSNR, SSIM, and achieved bpp.
          </p>
        </div>
      </div>
    </div>
  );
};
