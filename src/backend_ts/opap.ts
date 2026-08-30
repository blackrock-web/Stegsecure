/**
 * Optimal Pixel Adjustment Process (OPAP) (Chan & Cheng 2004).
 */

export function embedOPAPZone(
  imageFlat: Uint8Array,
  indices: number[],
  bitsStream: number[],
  k: number
): number {
  let embeddedCount = 0;
  const totalBits = bitsStream.length;
  const numPixels = Math.floor(totalBits / k);

  const mask = (1 << k) - 1;
  const twoK = 1 << k;

  for (let pIdx = 0; pIdx < numPixels && pIdx < indices.length; pIdx++) {
    const idx = indices[pIdx];
    const origP = imageFlat[idx];

    // Extract k bits from bitsStream
    let val = 0;
    for (let b = 0; b < k; b++) {
      val = (val << 1) | bitsStream[pIdx * k + b];
    }

    // Direct LSB replacement
    const pPrime = (origP & ~mask) | val;

    // OPAP candidate search in [pPrime, pPrime + 2^k, pPrime - 2^k]
    // Bound candidates within the same 8-pixel bucket (origP & ~7) for cost map stability
    const bucketBase = origP & ~7;
    const bucketTop = Math.min(255, bucketBase + 7);
    const candidates = [pPrime, pPrime + twoK, pPrime - twoK];
    let bestP = pPrime;
    let minDiff = Math.abs(pPrime - origP);

    for (const cand of candidates) {
      if (cand >= bucketBase && cand <= bucketTop) {
        const diff = Math.abs(cand - origP);
        if (diff < minDiff) {
          minDiff = diff;
          bestP = cand;
        }
      }
    }

    imageFlat[idx] = bestP;
    embeddedCount += k;
  }

  return embeddedCount;
}

export function extractOPAPZone(
  imageFlat: Uint8Array,
  indices: number[],
  maxBits: number,
  k: number
): number[] {
  const bits: number[] = [];
  const mask = (1 << k) - 1;
  const numPixels = Math.floor(maxBits / k);

  for (let pIdx = 0; pIdx < numPixels && pIdx < indices.length; pIdx++) {
    const idx = indices[pIdx];
    const val = imageFlat[idx] & mask;

    for (let b = k - 1; b >= 0; b--) {
      bits.push((val >> b) & 1);
    }
  }

  return bits;
}
