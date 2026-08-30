import { ImageRGB } from './imageUtils';

/**
 * Computes dense per-pixel embedding cost map C(x, y) in [0.0, 1.0].
 * Higher values indicate complex/textured/edge regions safer for payload embedding.
 */
export function computeCostMap(
  image: ImageRGB,
  gamma: number = 0.7,
  costMapMode: string = 'fast'
): Float32Array {
  const { width, height, data } = image;
  const N = width * height;
  const gray = new Float32Array(N);

  // 1. Grayscale conversion
  for (let i = 0; i < N; i++) {
    const r = data[i * 3 + 0];
    const g = data[i * 3 + 1];
    const b = data[i * 3 + 2];
    gray[i] = 0.299 * r + 0.587 * g + 0.114 * b;
  }

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

  // Normalize Sobel
  let sMin = Infinity, sMax = -Infinity;
  for (let i = 0; i < N; i++) {
    if (sobel[i] < sMin) sMin = sobel[i];
    if (sobel[i] > sMax) sMax = sobel[i];
  }
  const sRange = sMax - sMin + 1e-8;
  const sobelNorm = new Float32Array(N);
  for (let i = 0; i < N; i++) {
    sobelNorm[i] = (sobel[i] - sMin) / sRange;
  }

  // 3. Early feature / texture response
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

  let tMin = Infinity, tMax = -Infinity;
  for (let i = 0; i < N; i++) {
    if (texture[i] < tMin) tMin = texture[i];
    if (texture[i] > tMax) tMax = texture[i];
  }
  const tRange = tMax - tMin + 1e-8;
  const textureNorm = new Float32Array(N);
  for (let i = 0; i < N; i++) {
    textureNorm[i] = (texture[i] - tMin) / tRange;
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

    let hMin = Infinity, hMax = -Infinity;
    for (let i = 0; i < N; i++) {
      if (hillRes[i] < hMin) hMin = hillRes[i];
      if (hillRes[i] > hMax) hMax = hillRes[i];
    }
    const hRange = hMax - hMin + 1e-8;
    for (let i = 0; i < N; i++) {
      const hillNorm = (hillRes[i] - hMin) / hRange;
      hEdgeNorm[i] = 0.5 * hEdgeNorm[i] + 0.5 * hillNorm;
    }
  }

  // Fast fusion: texture + edge (skip expensive 5x5 multi-scale for speed)
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

