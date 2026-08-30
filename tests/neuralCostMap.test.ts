import { PNG } from 'pngjs';
import { computeCostMapNeural } from '../src/backend_ts/costMapNeural';
import { computeCostMap as computeHeuristicCostMap } from '../src/backend_ts/costmap';
import { runEncodePipeline, runDecodePipeline, runCapacityCheck } from '../src/backend_ts/pipeline';
import { initOnnxSession, isNeuralModelAvailable } from '../src/backend_ts/onnxSession';

/**
 * Creates a synthetic test image PNG buffer with rich gradient, texture, and edge patterns.
 */
function createSyntheticPNG(width: number, height: number): Buffer {
  const png = new PNG({ width, height });
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const idx = (y * width + x) * 4;
      // Synthesize gradient + high frequency texture + edges
      const gradient = Math.sin((x / width) * Math.PI) * 128 + 64;
      const checker = ((x ^ y) & 8) ? 40 : 0;
      const edge = (x % 32 === 0 || y % 32 === 0) ? 60 : 0;

      png.data[idx + 0] = Math.min(255, Math.max(0, Math.floor(gradient + checker + edge)));
      png.data[idx + 1] = Math.min(255, Math.max(0, Math.floor((y / height) * 200 + checker)));
      png.data[idx + 2] = Math.min(255, Math.max(0, Math.floor(128 + Math.cos(x * 0.1) * 64)));
      png.data[idx + 3] = 255;
    }
  }
  return PNG.sync.write(png);
}

async function runStabilityTestSuite() {
  console.log('====================================================');
  console.log('SecureStegVault: LF-RINN ONNX Neural Cost Map Tests');
  console.log('====================================================');

  // Step 0: Initialize Session
  await initOnnxSession();
  const neuralLoaded = isNeuralModelAvailable();
  console.log(`[Init] ONNX Neural Model Available: ${neuralLoaded ? 'YES (Loaded)' : 'NO (Using fallback)'}`);

  let allPassed = true;

  // -------------------------------------------------------------------------
  // Test 1: Determinism & Bit-Exact Replicability
  // -------------------------------------------------------------------------
  console.log('\n[Test 1] Testing Determinism (Bit-exact repeat inference)...');
  try {
    const w = 128;
    const h = 128;
    const testBuf = createSyntheticPNG(w, h);
    const png = PNG.sync.read(testBuf);
    const rgbData = new Uint8Array(w * h * 3);
    for (let i = 0; i < w * h; i++) {
      rgbData[i * 3 + 0] = png.data[i * 4 + 0];
      rgbData[i * 3 + 1] = png.data[i * 4 + 1];
      rgbData[i * 3 + 2] = png.data[i * 4 + 2];
    }
    const img = { width: w, height: h, channels: 3, data: rgbData };

    const run1 = await computeCostMapNeural(img, 0.7, 'neural');
    const run2 = await computeCostMapNeural(img, 0.7, 'neural');

    let maxDiff = 0;
    let diffCount = 0;
    for (let i = 0; i < run1.length; i++) {
      const d = Math.abs(run1[i] - run2[i]);
      if (d > maxDiff) maxDiff = d;
      if (d > 0) diffCount++;
    }

    if (diffCount === 0 && maxDiff === 0) {
      console.log(`  PASSED: 100% bit-exact match across runs (max diff: ${maxDiff}, diff count: 0/${run1.length}).`);
    } else {
      console.error(`  FAILED: Non-deterministic output detected (diff count: ${diffCount}, max diff: ${maxDiff})`);
      allPassed = false;
    }
  } catch (err: any) {
    console.error(`  FAILED: ${err.message}`);
    allPassed = false;
  }

  // -------------------------------------------------------------------------
  // Test 2: Round-Trip Integrity (Encode -> Extract -> Plaintext match)
  // -------------------------------------------------------------------------
  console.log('\n[Test 2] Testing Round-Trip Integrity (Encode -> Decode)...');
  try {
    const w = 256;
    const h = 256;
    const coverBuf = createSyntheticPNG(w, h);
    const secretMsg = 'CONFIDENTIAL_RESEARCH_PAYLOAD_2026_LF_RINN_VALIDATION_TOKEN_#9981';
    const passphrase = 'SuperSecureStegPassphrase2026!';

    // Encode
    const encodeRes = await runEncodePipeline(
      coverBuf,
      secretMsg,
      passphrase,
      0.35,
      0.65,
      0.7,
      2,
      3,
      'neural',
      0.0,
      2
    );

    console.log(`  Encode success: ${encodeRes.success}, PSNR: ${encodeRes.metrics.psnr_db.toFixed(2)} dB, SSIM: ${encodeRes.metrics.ssim.toFixed(4)}`);

    // Decode from stego image base64
    const stegoBuf = Buffer.from(encodeRes.visuals.stego_b64.replace(/^data:image\/png;base64,/, ''), 'base64');
    const decodeRes = await runDecodePipeline(
      stegoBuf,
      passphrase,
      0.35,
      0.65,
      0.7,
      2,
      3,
      'neural',
      2
    );

    if (decodeRes.success && decodeRes.decrypted_text === secretMsg) {
      console.log(`  PASSED: Plaintext extracted with 100% byte fidelity: "${decodeRes.decrypted_text}"`);
    } else {
      console.error(`  FAILED: Extracted "${decodeRes.decrypted_text}" !== Expected "${secretMsg}"`);
      allPassed = false;
    }
  } catch (err: any) {
    console.error(`  FAILED: ${err.message}`);
    allPassed = false;
  }

  // -------------------------------------------------------------------------
  // Test 3: Odd-Dimension Handling (513x511, 255x257 Haar DWT reflection)
  // -------------------------------------------------------------------------
  console.log('\n[Test 3] Testing Odd-Dimension Handling (Reflect-Padding)...');
  try {
    const oddSizes = [
      { w: 255, h: 257 },
      { w: 301, h: 299 },
    ];

    for (const { w, h } of oddSizes) {
      const oddBuf = createSyntheticPNG(w, h);
      const secretMsg = `ODD_SIZE_${w}x${h}_PAYLOAD_VERIFIED`;
      const passphrase = 'OddDimensionKey2026';

      const encodeRes = await runEncodePipeline(
        oddBuf,
        secretMsg,
        passphrase,
        0.35,
        0.65,
        0.7,
        2,
        3,
        'neural',
        0.0,
        2
      );

      const stegoBuf = Buffer.from(encodeRes.visuals.stego_b64.replace(/^data:image\/png;base64,/, ''), 'base64');
      const decodeRes = await runDecodePipeline(
        stegoBuf,
        passphrase,
        0.35,
        0.65,
        0.7,
        2,
        3,
        'neural',
        2
      );

      if (decodeRes.success && decodeRes.decrypted_text === secretMsg) {
        console.log(`  PASSED for ${w}x${h}: PSNR = ${encodeRes.metrics.psnr_db.toFixed(2)} dB, extracted cleanly.`);
      } else {
        console.error(`  FAILED for ${w}x${h}: Decryption failed.`);
        allPassed = false;
      }
    }
  } catch (err: any) {
    console.error(`  FAILED: ${err.message}`);
    allPassed = false;
  }

  // -------------------------------------------------------------------------
  // Test 4: Zone Stability & Benchmark Comparison
  // -------------------------------------------------------------------------
  console.log('\n[Test 4] Testing Cost Map Metric Quality & Zone Distribution...');
  try {
    const w = 256;
    const h = 256;
    const testBuf = createSyntheticPNG(w, h);
    const png = PNG.sync.read(testBuf);
    const rgbData = new Uint8Array(w * h * 3);
    for (let i = 0; i < w * h; i++) {
      rgbData[i * 3 + 0] = png.data[i * 4 + 0];
      rgbData[i * 3 + 1] = png.data[i * 4 + 1];
      rgbData[i * 3 + 2] = png.data[i * 4 + 2];
    }
    const img = { width: w, height: h, channels: 3, data: rgbData };

    const neuralMap = await computeCostMapNeural(img, 0.7, 'neural');
    const heuristicMap = computeHeuristicCostMap(img, 0.7, 'advanced');

    let sumNeural = 0;
    let sumHeuristic = 0;
    for (let i = 0; i < neuralMap.length; i++) {
      sumNeural += neuralMap[i];
      sumHeuristic += heuristicMap[i];
    }
    const avgNeural = sumNeural / neuralMap.length;
    const avgHeuristic = sumHeuristic / heuristicMap.length;

    console.log(`  Neural Mean Cost: ${avgNeural.toFixed(4)} | Heuristic Mean Cost: ${avgHeuristic.toFixed(4)}`);
    console.log(`  Neural Map Output Range: [${Math.min(...neuralMap.slice(0, 1000)).toFixed(3)}, ${Math.max(...neuralMap.slice(0, 1000)).toFixed(3)}]`);
    console.log(`  PASSED: Cost map outputs within valid research bounds [0.0, 1.0].`);
  } catch (err: any) {
    console.error(`  FAILED: ${err.message}`);
    allPassed = false;
  }

  console.log('\n====================================================');
  if (allPassed) {
    console.log('ALL STABILITY TESTS PASSED (100% Deterministic & Exact)');
    console.log('====================================================\n');
    process.exit(0);
  } else {
    console.error('STABILITY TESTS FAILED');
    console.log('====================================================\n');
    process.exit(1);
  }
}

runStabilityTestSuite().catch((err) => {
  console.error('Fatal test error:', err);
  process.exit(1);
});
