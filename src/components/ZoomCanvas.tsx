import React, { useRef, useEffect, useState } from 'react';
import { ZoomIn, ZoomOut, RotateCcw, Crosshair, Eye, EyeOff } from 'lucide-react';
import { PixelHoverInfo } from '../types';

interface ZoomCanvasProps {
  coverUrl: string;
  stegoUrl: string;
  heatmapUrl?: string;
  showHeatmap: boolean;
}

export const ZoomCanvas: React.FC<ZoomCanvasProps> = ({
  coverUrl,
  stegoUrl,
  heatmapUrl,
  showHeatmap,
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  const [zoom, setZoom] = useState<number>(1);
  const [pan, setPan] = useState<{ x: number; y: number }>({ x: 0, y: 0 });
  const [isPanning, setIsPanning] = useState(false);
  const [startPan, setStartPan] = useState<{ x: number; y: number }>({ x: 0, y: 0 });

  const [hoverInfo, setHoverInfo] = useState<PixelHoverInfo | null>(null);

  const coverImgRef = useRef<HTMLImageElement | null>(null);
  const stegoImgRef = useRef<HTMLImageElement | null>(null);
  const heatImgRef = useRef<HTMLImageElement | null>(null);

  const coverCtxRef = useRef<CanvasRenderingContext2D | null>(null);
  const stegoCtxRef = useRef<CanvasRenderingContext2D | null>(null);

  // Load images
  useEffect(() => {
    let isMounted = true;

    const imgC = new Image();
    const imgS = new Image();
    const imgH = new Image();

    imgC.crossOrigin = "anonymous";
    imgS.crossOrigin = "anonymous";
    imgH.crossOrigin = "anonymous";

    imgC.src = coverUrl;
    imgS.src = stegoUrl;
    if (heatmapUrl) imgH.src = heatmapUrl;

    Promise.all([
      new Promise((res) => { imgC.onload = res; }),
      new Promise((res) => { imgS.onload = res; }),
      heatmapUrl ? new Promise((res) => { imgH.onload = res; }) : Promise.resolve(),
    ]).then(() => {
      if (!isMounted) return;
      coverImgRef.current = imgC;
      stegoImgRef.current = imgS;
      if (heatmapUrl) heatImgRef.current = imgH;

      // Prepare hidden canvas contexts for pixel RGB reading
      const offC = document.createElement('canvas');
      offC.width = imgC.width;
      offC.height = imgC.height;
      const ctxC = offC.getContext('2d', { willReadFrequently: true });
      if (ctxC) {
        ctxC.drawImage(imgC, 0, 0);
        coverCtxRef.current = ctxC;
      }

      const offS = document.createElement('canvas');
      offS.width = imgS.width;
      offS.height = imgS.height;
      const ctxS = offS.getContext('2d', { willReadFrequently: true });
      if (ctxS) {
        ctxS.drawImage(imgS, 0, 0);
        stegoCtxRef.current = ctxS;
      }

      drawCanvas();
    });

    return () => { isMounted = false; };
  }, [coverUrl, stegoUrl, heatmapUrl]);

  // Re-draw canvas on state changes
  const drawCanvas = () => {
    const canvas = canvasRef.current;
    if (!canvas || !stegoImgRef.current) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const targetImg = showHeatmap && heatImgRef.current ? heatImgRef.current : stegoImgRef.current;

    canvas.width = canvas.parentElement?.clientWidth || 800;
    canvas.height = 450;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    ctx.save();
    // Center image and apply zoom and pan transform
    ctx.translate(canvas.width / 2 + pan.x, canvas.height / 2 + pan.y);
    ctx.scale(zoom, zoom);

    // Disable image smoothing when zoomed in above 3x for crisp pixel grid
    ctx.imageSmoothingEnabled = zoom < 3;

    ctx.drawImage(
      targetImg,
      -targetImg.width / 2,
      -targetImg.height / 2,
      targetImg.width,
      targetImg.height
    );

    // Draw pixel grid overlay when zoomed in deep (>= 8x)
    if (zoom >= 8) {
      ctx.strokeStyle = "rgba(216, 180, 254, 0.4)";
      ctx.lineWidth = 0.5 / zoom;
      const startX = -targetImg.width / 2;
      const startY = -targetImg.height / 2;

      for (let x = 0; x <= targetImg.width; x++) {
        ctx.beginPath();
        ctx.moveTo(startX + x, startY);
        ctx.lineTo(startX + x, startY + targetImg.height);
        ctx.stroke();
      }
      for (let y = 0; y <= targetImg.height; y++) {
        ctx.beginPath();
        ctx.moveTo(startX, startY + y);
        ctx.lineTo(startX + targetImg.width, startY + y);
        ctx.stroke();
      }
    }

    ctx.restore();
  };

  useEffect(() => {
    drawCanvas();
  }, [zoom, pan, showHeatmap]);

  // Pan controls
  const handleMouseDown = (e: React.MouseEvent) => {
    setIsPanning(true);
    setStartPan({ x: e.clientX - pan.x, y: e.clientY - pan.y });
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    const canvas = canvasRef.current;
    const img = stegoImgRef.current;

    if (isPanning) {
      setPan({ x: e.clientX - startPan.x, y: e.clientY - startPan.y });
    }

    if (!canvas || !img || !coverCtxRef.current || !stegoCtxRef.current) return;

    const rect = canvas.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;

    // Inverse transform to calculate original image (x, y) coordinates
    const centerX = canvas.width / 2 + pan.x;
    const centerY = canvas.height / 2 + pan.y;

    const imgX = Math.floor((mouseX - centerX) / zoom + img.width / 2);
    const imgY = Math.floor((mouseY - centerY) / zoom + img.height / 2);

    if (imgX >= 0 && imgX < img.width && imgY >= 0 && imgY < img.height) {
      const pC = coverCtxRef.current.getImageData(imgX, imgY, 1, 1).data;
      const pS = stegoCtxRef.current.getImageData(imgX, imgY, 1, 1).data;

      const coverRgb: [number, number, number] = [pC[0], pC[1], pC[2]];
      const stegoRgb: [number, number, number] = [pS[0], pS[1], pS[2]];

      const delta = Math.abs(pC[0] - pS[0]) + Math.abs(pC[1] - pS[1]) + Math.abs(pC[2] - pS[2]);
      const isModified = delta > 0;

      setHoverInfo({
        x: imgX,
        y: imgY,
        coverRgb,
        stegoRgb,
        isModified,
        delta,
      });
    } else {
      setHoverInfo(null);
    }
  };

  const handleMouseUp = () => {
    setIsPanning(false);
  };

  const handleWheel = (e: React.WheelEvent) => {
    e.preventDefault();
    const factor = e.deltaY < 0 ? 1.2 : 0.8;
    setZoom((prev) => Math.min(32, Math.max(0.5, prev * factor)));
  };

  const handleReset = () => {
    setZoom(1);
    setPan({ x: 0, y: 0 });
  };

  return (
    <div className="space-y-3">
      {/* Controls toolbar */}
      <div className="flex items-center justify-between bg-purple-50/80 p-2.5 rounded-2xl border border-purple-200/70">
        <div className="flex items-center gap-2">
          <Crosshair className="w-4 h-4 text-pink-600" />
          <span className="text-xs font-bold text-purple-950">
            Interactive Pixel Inspection Canvas
          </span>
          <span className="text-[11px] text-purple-600 font-medium hidden sm:inline">
            (Hover pixel to inspect RGB &amp; modification status)
          </span>
        </div>

        <div className="flex items-center gap-1.5">
          <button
            onClick={() => setZoom((z) => Math.min(32, z * 1.5))}
            className="p-1.5 rounded-xl bg-white border border-purple-200 text-purple-900 hover:bg-pink-100/50 text-xs font-semibold flex items-center gap-1 shadow-2xs"
            title="Zoom In"
          >
            <ZoomIn className="w-3.5 h-3.5" />
          </button>
          <button
            onClick={() => setZoom((z) => Math.max(0.5, z / 1.5))}
            className="p-1.5 rounded-xl bg-white border border-purple-200 text-purple-900 hover:bg-pink-100/50 text-xs font-semibold flex items-center gap-1 shadow-2xs"
            title="Zoom Out"
          >
            <ZoomOut className="w-3.5 h-3.5" />
          </button>
          <button
            onClick={handleReset}
            className="p-1.5 rounded-xl bg-white border border-purple-200 text-purple-900 hover:bg-pink-100/50 text-xs font-semibold flex items-center gap-1 shadow-2xs"
            title="Reset Zoom"
          >
            <RotateCcw className="w-3.5 h-3.5" />
            <span className="hidden sm:inline">Reset</span>
          </button>

          <span className="text-xs font-bold text-purple-900 bg-white px-2.5 py-1 rounded-xl border border-purple-200">
            {Math.round(zoom * 100)}%
          </span>
        </div>
      </div>

      {/* Canvas view container */}
      <div
        ref={containerRef}
        className="relative w-full rounded-2xl border border-pink-200 bg-purple-950/5 overflow-hidden shadow-inner cursor-grab active:cursor-grabbing"
      >
        <canvas
          ref={canvasRef}
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onMouseLeave={() => setHoverInfo(null)}
          onWheel={handleWheel}
          className="w-full block"
        />

        {/* Floating Pixel Hover Inspector Tooltip */}
        {hoverInfo && (
          <div className="absolute top-3 left-3 z-10 p-3 rounded-2xl bg-white/95 backdrop-blur-md border border-pink-300 shadow-lg text-xs space-y-2 min-w-56 animate-fadeIn">
            <div className="flex items-center justify-between border-b border-pink-100 pb-1.5">
              <span className="font-bold text-purple-950 flex items-center gap-1">
                📍 Pixel ({hoverInfo.x}, {hoverInfo.y})
              </span>
              <span
                className={`px-2 py-0.5 rounded-full text-[10px] font-extrabold ${
                  hoverInfo.isModified
                    ? 'bg-pink-100 text-pink-800 border border-pink-300'
                    : 'bg-emerald-100 text-emerald-800 border border-emerald-300'
                }`}
              >
                {hoverInfo.isModified ? 'MODIFIED' : 'UNMODIFIED'}
              </span>
            </div>

            <div className="space-y-1 font-mono text-[11px]">
              <div className="flex justify-between items-center text-purple-900">
                <span className="font-sans text-purple-600">Cover RGB:</span>
                {hoverInfo.coverRgb && (
                  <span className="flex items-center gap-1.5 font-bold">
                    <span
                      className="w-3 h-3 rounded-md border border-purple-300 inline-block"
                      style={{
                        backgroundColor: `rgb(${hoverInfo.coverRgb.join(',')})`,
                      }}
                    />
                    ({hoverInfo.coverRgb.join(', ')})
                  </span>
                )}
              </div>

              <div className="flex justify-between items-center text-purple-900">
                <span className="font-sans text-purple-600">Stego RGB:</span>
                {hoverInfo.stegoRgb && (
                  <span className="flex items-center gap-1.5 font-bold">
                    <span
                      className="w-3 h-3 rounded-md border border-purple-300 inline-block"
                      style={{
                        backgroundColor: `rgb(${hoverInfo.stegoRgb.join(',')})`,
                      }}
                    />
                    ({hoverInfo.stegoRgb.join(', ')})
                  </span>
                )}
              </div>

              {hoverInfo.isModified && (
                <div className="flex justify-between items-center text-pink-700 font-bold border-t border-pink-100 pt-1">
                  <span className="font-sans text-pink-600">Delta |C - S|:</span>
                  <span>+{hoverInfo.delta} LSB</span>
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
