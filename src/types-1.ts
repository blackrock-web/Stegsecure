export interface CapacityResponse {
  width: number;
  height: number;
  channels: number;
  capacity: {
    total_pixels: number;
    count_zone_a: number;
    count_zone_b: number;
    count_zone_c: number;
    max_bits: number;
    max_bytes: number;
    max_plaintext_bytes: number;
    overall_bpp: number;
    zone_a_bpp: number;
    zone_b_bpp: number;
    zone_c_bpp: number;
  };
  cost_map_mode: string;
  emd_n?: number;
}

export interface MetricsData {
  mse: number;
  psnr_db: number;
  ssim: number;
  total_bits_embedded?: number;
  total_bytes_embedded?: number;
  achieved_bpp?: number;
  bpp?: number;
  modified_pixel_count?: number;
  modified_pixels_count?: number;
  modified_pixel_percentage?: number;
  zone_breakdown?: {
    zone_a_bits: number;
    zone_b_bits: number;
    zone_c_bits: number;
  };
}

export interface SecurityReport {
  cover_detection_confidence: number;
  stego_detection_confidence: number;
  detection_confidence_delta: number;
  note: string;
}

export interface VisualsData {
  stego_b64: string;
  heatmap_b64: string;
  mask_b64: string;
  binary_mask_b64?: string;
  zone_map_b64: string;
  gradient_overlay_b64?: string;
  highlight_overlay_b64?: string;
  /** RGB mode: pure R/G/B bright pixels show exactly which colour channel stores data bits */
  rgb_bits_b64?: string;
}

export interface EncodeResponse {
  success: boolean;
  metrics: MetricsData;
  security_report?: SecurityReport;
  visuals: VisualsData;
  cost_map_mode: string;
  adversarial_strength?: number;
  emd_n?: number;
}

export interface DecodeResponse {
  success: boolean;
  decrypted_text?: string;
  error?: string;
}

export interface PixelHoverInfo {
  x: number;
  y: number;
  coverRgb: [number, number, number] | null;
  stegoRgb: [number, number, number] | null;
  isModified: boolean;
  delta: number;
}

export interface TuningConfig {
  gamma: number;
  threshA: number;
  threshB: number;
  kbBits: number;
  kcBits: number;
  costMapMode: 'fast' | 'cnn' | 'advanced';
  adversarialStrength: number;
  emdN: 2 | 3;
}

