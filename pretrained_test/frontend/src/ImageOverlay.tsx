import { useEffect, useRef, useState } from "react";
import type { OcrLine } from "./types";

interface Props {
  imageUrl: string;
  beforeUrl?: string;   // original image — when set, enables split-view slider
  lines: OcrLine[];
  highlightIdx: number | null;
  onHover: (idx: number | null) => void;
}

function easeInOut(t: number): number {
  return t < 0.5 ? 2 * t * t : -1 + (4 - 2 * t) * t;
}

export function ImageOverlay({ imageUrl, beforeUrl, lines, highlightIdx, onHover }: Props) {
  const canvasRef      = useRef<HTMLCanvasElement>(null);
  const afterImgRef    = useRef<HTMLImageElement | null>(null);
  const beforeImgRef   = useRef<HTMLImageElement | null>(null);
  const splitXRef      = useRef(-1);          // canvas px; -1 = uninitialised
  const isDraggingRef  = useRef(false);
  // Keep latest props accessible inside RAF callbacks without stale closures
  const linesRef       = useRef(lines);
  const highlightRef   = useRef(highlightIdx);
  linesRef.current     = lines;
  highlightRef.current = highlightIdx;

  const [bothLoaded, setBothLoaded] = useState(false);

  // Reset when URLs change (new file uploaded)
  useEffect(() => {
    setBothLoaded(false);
    splitXRef.current = -1;
  }, [imageUrl, beforeUrl]);

  // Load "after" (enhanced / main) image
  useEffect(() => {
    const img = new Image();
    img.onload = () => {
      afterImgRef.current = img;
      const canvas = canvasRef.current;
      if (canvas) { canvas.width = img.naturalWidth; canvas.height = img.naturalHeight; }
      if (!beforeUrl || beforeImgRef.current) setBothLoaded(true);
      draw();
    };
    img.src = imageUrl;
    return () => { afterImgRef.current = null; };
  }, [imageUrl]); // eslint-disable-line react-hooks/exhaustive-deps

  // Load "before" (original) image
  useEffect(() => {
    if (!beforeUrl) { beforeImgRef.current = null; return; }
    const img = new Image();
    img.onload = () => {
      beforeImgRef.current = img;
      if (afterImgRef.current) setBothLoaded(true);
      draw();
    };
    img.src = beforeUrl;
    return () => { beforeImgRef.current = null; };
  }, [beforeUrl]); // eslint-disable-line react-hooks/exhaustive-deps

  // Intro sweep animation: 50 % → 75 % → 25 % → 50 %
  useEffect(() => {
    if (!bothLoaded || !beforeUrl) return;
    const canvas = canvasRef.current;
    const after  = afterImgRef.current;
    if (!canvas || !after) return;

    if (splitXRef.current < 0) splitXRef.current = canvas.width * 0.5;

    const keys = [0.5, 0.75, 0.25, 0.5];
    const segMs = 480;
    let seg = 0, segStart = -1, rafId = 0;

    function step(now: number) {
      if (segStart < 0) segStart = now;
      const t = Math.min(1, (now - segStart) / segMs);
      const c = canvasRef.current;
      if (!c) return;
      splitXRef.current = (keys[seg] + (keys[seg + 1] - keys[seg]) * easeInOut(t)) * c.width;
      draw();
      if (t < 1) {
        rafId = requestAnimationFrame(step);
      } else if (seg + 2 < keys.length) {
        seg++; segStart = -1;
        rafId = requestAnimationFrame(step);
      }
    }

    const tid = setTimeout(() => { rafId = requestAnimationFrame(step); }, 120);
    return () => { clearTimeout(tid); cancelAnimationFrame(rafId); };
  }, [bothLoaded, beforeUrl]); // eslint-disable-line react-hooks/exhaustive-deps

  // Redraw when boxes or highlight change
  useEffect(() => { draw(); }, [lines, highlightIdx]);

  // ─── drawing ────────────────────────────────────────────────────────────────

  function draw() {
    const canvas = canvasRef.current;
    const after  = afterImgRef.current;
    if (!canvas || !after || !canvas.width) return;

    const W = canvas.width, H = canvas.height;
    if (splitXRef.current < 0) splitXRef.current = W / 2;

    const ctx    = canvas.getContext("2d")!;
    const before = beforeImgRef.current;
    const sx     = splitXRef.current;

    ctx.clearRect(0, 0, W, H);

    if (before) {
      ctx.drawImage(after, 0, 0);           // "after" (enhanced) — full width
      ctx.save();
      ctx.beginPath();
      ctx.rect(0, 0, sx, H);
      ctx.clip();
      ctx.drawImage(before, 0, 0);          // "before" (original) — left portion
      ctx.restore();
      drawBoxes(ctx);
      drawDivider(ctx, sx, W, H);
    } else {
      ctx.drawImage(after, 0, 0);
      drawBoxes(ctx);
    }
  }

  function drawBoxes(ctx: CanvasRenderingContext2D) {
    ctx.textBaseline = "alphabetic";
    linesRef.current.forEach((line, i) => {
      const hot = i === highlightRef.current;

      ctx.save();
      ctx.shadowColor = hot ? "#fb7185" : "#6ee7b7";
      ctx.shadowBlur  = hot ? 18 : 10;
      ctx.beginPath();
      line.box.forEach(([x, y], k) => { k === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y); });
      ctx.closePath();
      ctx.fillStyle = hot ? "rgba(251,113,133,0.18)" : "rgba(110,231,183,0.10)";
      ctx.fill();
      ctx.restore();

      ctx.lineWidth   = hot ? 3 : 1.8;
      ctx.strokeStyle = hot ? "#fb7185" : "#6ee7b7";
      ctx.beginPath();
      line.box.forEach(([x, y], k) => { k === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y); });
      ctx.closePath();
      ctx.stroke();

      const [x0, y0] = line.box[0];
      const label = `${i + 1}`;
      ctx.font = `bold 13px 'Inter', system-ui`;
      const tW = ctx.measureText(label).width;
      const bW = tW + 10, bH = 18;
      const bY = Math.max(y0 - bH - 2, 2);
      ctx.fillStyle = hot ? "#fb7185" : "#6ee7b7";
      roundRect(ctx, x0, bY, bW, bH, 4);
      ctx.fill();
      ctx.fillStyle = hot ? "#1a0a0e" : "#0a1a12";
      ctx.fillText(label, x0 + 5, bY + bH - 4);
    });
  }

  function drawDivider(ctx: CanvasRenderingContext2D, sx: number, W: number, H: number) {
    ctx.save();
    ctx.font = "600 13px 'Inter', system-ui";
    ctx.textBaseline = "middle";

    // Label — left side
    const bTxt = "ORYGINAŁ";
    const bTW  = ctx.measureText(bTxt).width;
    if (sx > bTW + 20) {
      const lx = Math.max(8, sx - bTW - 20);
      ctx.fillStyle = "rgba(0,0,0,0.58)";
      roundRect(ctx, lx - 6, 8, bTW + 12, 24, 5);
      ctx.fill();
      ctx.fillStyle = "#e8ecf4";
      ctx.fillText(bTxt, lx, 20);
    }

    // Label — right side
    const aTxt = "PRZETWORZONY";
    const aTW  = ctx.measureText(aTxt).width;
    if (W - sx > aTW + 20) {
      const lx = Math.min(W - aTW - 14, sx + 14);
      ctx.fillStyle = "rgba(0,0,0,0.58)";
      roundRect(ctx, lx - 6, 8, aTW + 12, 24, 5);
      ctx.fill();
      ctx.fillStyle = "#6ee7b7";
      ctx.fillText(aTxt, lx, 20);
    }
    ctx.restore();

    // Vertical line
    ctx.save();
    ctx.shadowColor = "rgba(255,255,255,0.7)";
    ctx.shadowBlur  = 8;
    ctx.strokeStyle = "#fff";
    ctx.lineWidth   = 2;
    ctx.beginPath();
    ctx.moveTo(sx, 0);
    ctx.lineTo(sx, H);
    ctx.stroke();
    ctx.restore();

    // Handle circle
    const hY = H / 2, r = 22;
    ctx.save();
    ctx.shadowColor = "rgba(0,0,0,0.45)";
    ctx.shadowBlur  = 12;
    ctx.fillStyle   = "#fff";
    ctx.beginPath();
    ctx.arc(sx, hY, r, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();

    // Arrow triangles ◀ ▶
    ctx.fillStyle = "#555";
    const aS = 7, gap = 5;
    ctx.beginPath();
    ctx.moveTo(sx - gap, hY);
    ctx.lineTo(sx - gap - aS, hY - aS * 0.65);
    ctx.lineTo(sx - gap - aS, hY + aS * 0.65);
    ctx.closePath();
    ctx.fill();
    ctx.beginPath();
    ctx.moveTo(sx + gap, hY);
    ctx.lineTo(sx + gap + aS, hY - aS * 0.65);
    ctx.lineTo(sx + gap + aS, hY + aS * 0.65);
    ctx.closePath();
    ctx.fill();
  }

  // ─── interaction ─────────────────────────────────────────────────────────────

  function canvasCoords(e: React.MouseEvent<HTMLCanvasElement>): [number, number] {
    const c = canvasRef.current!;
    const r = c.getBoundingClientRect();
    const sx = c.width / r.width, sy = c.height / r.height;
    return [(e.clientX - r.left) * sx, (e.clientY - r.top) * sy];
  }

  function handleMouseDown(e: React.MouseEvent<HTMLCanvasElement>) {
    if (!beforeImgRef.current) return;
    const c = canvasRef.current!;
    const r = c.getBoundingClientRect();
    const scale = c.width / r.width;
    const [cx] = canvasCoords(e);
    if (Math.abs(cx - splitXRef.current) < 24 * scale) isDraggingRef.current = true;
  }

  function handleMouseUp() { isDraggingRef.current = false; }

  function handleMove(e: React.MouseEvent<HTMLCanvasElement>) {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const [cx, cy] = canvasCoords(e);

    if (isDraggingRef.current) {
      splitXRef.current = Math.max(0, Math.min(canvas.width, cx));
      draw();
      return;
    }

    if (beforeImgRef.current) {
      const r     = canvas.getBoundingClientRect();
      const scale = canvas.width / r.width;
      if (Math.abs(cx - splitXRef.current) < 24 * scale) {
        canvas.style.cursor = "ew-resize";
        onHover(null);
        return;
      }
    }

    canvas.style.cursor = "crosshair";
    for (let i = 0; i < lines.length; i++) {
      if (pointInPoly(cx, cy, lines[i].box)) { onHover(i); return; }
    }
    onHover(null);
  }

  return (
    <canvas
      ref={canvasRef}
      className="viewer"
      onMouseMove={handleMove}
      onMouseDown={handleMouseDown}
      onMouseUp={handleMouseUp}
      onMouseLeave={() => { isDraggingRef.current = false; onHover(null); }}
      id="ocr-canvas"
    />
  );
}

// ─── helpers ─────────────────────────────────────────────────────────────────

function roundRect(
  ctx: CanvasRenderingContext2D,
  x: number, y: number, w: number, h: number, r: number,
) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.lineTo(x + w - r, y);
  ctx.quadraticCurveTo(x + w, y, x + w, y + r);
  ctx.lineTo(x + w, y + h - r);
  ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
  ctx.lineTo(x + r, y + h);
  ctx.quadraticCurveTo(x, y + h, x, y + h - r);
  ctx.lineTo(x, y + r);
  ctx.quadraticCurveTo(x, y, x + r, y);
  ctx.closePath();
}

function pointInPoly(x: number, y: number, poly: [number, number][]): boolean {
  let inside = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const [xi, yi] = poly[i];
    const [xj, yj] = poly[j];
    const intersect =
      yi > y !== yj > y && x < ((xj - xi) * (y - yi)) / (yj - yi + 1e-9) + xi;
    if (intersect) inside = !inside;
  }
  return inside;
}
