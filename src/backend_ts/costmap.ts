import { ImageRGB } from './imageUtils';

/**
 * Computes dense per-pixel embedding cost map C(x, y) in [0.0, 1.0].
 * Higher values indicate complex/textured/edge regions safer for payload embedding.
 *
 * Stability Rationale:
 * 1. quantizeForCostStability() masks off the lower 3 bits of grayscale values.
 *    Embedding modifications (which introduce +/- 1 to 2 pixel alterations) do not
 *    shift the quantized input, ensuring identical features before and after embedding.
 * 2. Normalization uses fixed analytical kernel maximums rather than dynamic per-image
 *    min/max bounds. This completely prevents whole-image normalization drift when local
 *    pixel modifications occur.
 */
export function computeCostMap(
  image: ImageRGB,
  gamma: number = 0.7,
  costMapMode: string = 'fast'
): Float32Array {
  const { width, height, data } = image;
  const N = width * height;
  const rawGray = new Float32Array(N);

  // 1. Grayscale conversion with channel stability masking (ITU-R BT.601)
  const mask = ~((1 << 3) - 1);
  for (let i = 0; i < N; i++) {
    const r = data[i * 3 + 0] & mask;
    const g = data[i * 3 + 1] & mask;
    const b = data[i * 3 + 2] & mask;
    rawGray[i] = 0.299 * r + 0.587 * g + 0.114 * b;
  }

  // Stability quantization: mask off low 3 bits before computing features
  const gray = quantizeForCostStability(rawGray, 3);

  // 2. Sobel edge magnitude
  const sobel = new Float32Array(N);
  for (let y = 1; y < height - 1; y++) {
    for (let x = 1; x < width - 1; x++) {
      const idx = y * width + x;

      const gx =
        -1 * gray[(y - 1) * width + (x - 1)] + 1 * gray[(y - 1) * width + (x + 1)] +
        -2 * gray[y * width + (x - 1)]       + 2 * gray[y * width + (x + 1)] +
        -1 * gray[(y + 1) * width + (x - 1)] + 1 * gray[(y + 1) * width + (x + 1)];

      const gy =
        -1 * gray[(y - 1) * width + (x - 1)] - 2 * gray[(y - 1) * width + x] - 1 * gray[(y - 1) * width + (x + 1)] +
         1 * gray[(y + 1) * width + (x - 1)] + 2 * gray[(y + 1) * width + x] + 1 * gray[(y + 1) * width + (x + 1)];

      sobel[idx] = Math.sqrt(gx * gx + gy * gy);
    }
  }

  // Stable fixed normalization for Sobel (theoretical max for 3x3 sobel: 4 * 255 * sqrt(2) ≈ 1442.5)
  const MAX_SOBEL = 4 * 255 * Math.SQRT2;
  const sobelNorm = new Float32Array(N);
  for (let i = 0; i < N; i++) {
    sobelNorm[i] = Math.min(1.0, Math.max(0.0, sobel[i] / MAX_SOBEL));
  }

  // 3. Texture response (sum of 4-neighborhood absolute differences)
  const texture = new Float32Array(N);
  for (let y = 1; y < height - 1; y++) {
    for (let x = 1; x < width - 1; x++) {
      const idx = y * width + x;
      const center = gray[idx];
      const diff =
        Math.abs(gray[(y - 1) * width + x] - center) +
        Math.abs(gray[(y + 1) * width + x] - center) +
        Math.abs(gray[y * width + (x - 1)] - center) +
        Math.abs(gray[y * width + (x + 1)] - center);
      texture[idx] = diff;
    }
  }

  // Stable fixed normalization for texture (theoretical max: 4 * 255 = 1020)
  const MAX_TEXTURE = 4 * 255;
  const textureNorm = new Float32Array(N);
  for (let i = 0; i < N; i++) {
    textureNorm[i] = Math.min(1.0, Math.max(0.0, texture[i] / MAX_TEXTURE));
  }

  let hEdgeNorm = new Float32Array(N);
  for (let i = 0; i < N; i++) {
    hEdgeNorm[i] = 0.5 * sobelNorm[i] + 0.5 * textureNorm[i];
  }

  if (costMapMode === 'advanced') {
    // HILL-style high-pass residual filter: [-1, 2, -1; 2, -4, 2; -1, 2, -1] / 12
    const hillRes = new Float32Array(N);
    for (let y = 1; y < height - 1; y++) {
      for (let x = 1; x < width - 1; x++) {
        const idx = y * width + x;
        const hp =
          -1 * gray[(y - 1) * width + (x - 1)] + 2 * gray[(y - 1) * width + x] - 1 * gray[(y - 1) * width + (x + 1)] +
           2 * gray[y * width + (x - 1)]       - 4 * gray[y * width + x]       + 2 * gray[y * width + (x + 1)] +
          -1 * gray[(y + 1) * width + (x - 1)] + 2 * gray[(y + 1) * width + x] - 1 * gray[(y + 1) * width + (x + 1)];
        hillRes[idx] = Math.abs(hp) / 12.0;
      }
    }

    // Stable fixed normalization for HILL filter (theoretical max: 8 * 255 / 12 = 170.0)
    const MAX_HILL = (8 * 255) / 12.0;
    for (let i = 0; i < N; i++) {
      const hillNorm = Math.min(1.0, Math.max(0.0, hillRes[i] / MAX_HILL));
      hEdgeNorm[i] = 0.5 * hEdgeNorm[i] + 0.5 * hillNorm;
    }
  }

  // Fusion: texture + edge weighted by gamma
  const finalMap = new Float32Array(N);
  for (let i = 0; i < N; i++) {
    const cnnLike = textureNorm[i];
    const cost = gamma * cnnLike + (1 - gamma) * hEdgeNorm[i];
    finalMap[i] = cost < 0 ? 0 : cost > 1 ? 1 : cost;
  }

  return finalMap;
}

/**
 * Stability quantization: clears the lower `stabilizeBits` so that cover-time
 * and stego-time cost map inputs remain identical despite embedding modifications (+/- 1-2 pixel changes).
 */
export function quantizeForCostStability(
  gray: Float32Array,
  stabilizeBits: number = 3
): Float32Array {
  const mask = ~((1 << stabilizeBits) - 1);
  const out = new Float32Array(gray.length);
  for (let i = 0; i < gray.length; i++) {
    const val = Math.round(gray[i]);
    out[i] = Math.min(255, Math.max(0, val & mask));
  }
  return out;
}

