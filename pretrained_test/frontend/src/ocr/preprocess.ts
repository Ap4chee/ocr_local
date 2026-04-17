// Preprocessing dla PP-OCRv5 (det + rec). Wszystko CHW float32, batch=1.

// PaddleOCR default: limit_type='max', limit_side_len=960 — u nas 1280 zgodnie z CLAUDE.md.
// Małe obrazy NIE są upscale'owane (upscale halucynuje DB na nierealnych krawędziach).
// Duże (zdjęcia z telefonu 4000×3000) są downscalowane, inaczej tensor liczy >100 MB
// i DB kompiluje shadery WebGPU dla nierealistycznego rozmiaru.
const DET_MAX_SIDE = 1280;
const DET_MEAN = [0.485, 0.456, 0.406];
const DET_STD = [0.229, 0.224, 0.225];

const REC_HEIGHT = 48;
const REC_MAX_WIDTH = 320;

export interface DetInput {
  tensor: Float32Array;
  shape: [number, number, number, number]; // [1,3,H,W]
  scaleX: number; // resized → original
  scaleY: number;
  origW: number;
  origH: number;
}

export interface RecInput {
  tensor: Float32Array;
  shape: [number, number, number, number]; // [1,3,48,W]
}

export interface RecBatchInput {
  tensor: Float32Array;
  shape: [number, number, number, number]; // [B,3,48,maxW]
  widths: number[];                         // real width per item (before pad)
}

function makeCanvas(w: number, h: number): OffscreenCanvas | HTMLCanvasElement {
  if (typeof OffscreenCanvas !== "undefined") return new OffscreenCanvas(w, h);
  const c = document.createElement("canvas");
  c.width = w;
  c.height = h;
  return c;
}

export function preprocessDet(bitmap: ImageBitmap): DetInput {
  const origW = bitmap.width;
  const origH = bitmap.height;

  // Downscale jeśli dłuższy bok > DET_MAX_SIDE (tylko w dół, nigdy w górę).
  const maxSide = Math.max(origW, origH);
  const scale = maxSide > DET_MAX_SIDE ? DET_MAX_SIDE / maxSide : 1;
  let newW = Math.round(origW * scale);
  let newH = Math.round(origH * scale);

  // multi-32 (wymóg DB head po stride'ach). Używamy ceil, żeby nie ucinać obrazu w dół.
  newW = Math.max(32, Math.ceil(newW / 32) * 32);
  newH = Math.max(32, Math.ceil(newH / 32) * 32);

  const canvas = makeCanvas(newW, newH);
  const ctx = canvas.getContext("2d") as
    | OffscreenCanvasRenderingContext2D
    | CanvasRenderingContext2D;
  ctx.drawImage(bitmap, 0, 0, newW, newH);
  const img = ctx.getImageData(0, 0, newW, newH);
  const data = img.data;

  const plane = newW * newH;
  const out = new Float32Array(3 * plane);
  for (let i = 0, p = 0; i < data.length; i += 4, p++) {
    out[p] = (data[i] / 255 - DET_MEAN[0]) / DET_STD[0];
    out[plane + p] = (data[i + 1] / 255 - DET_MEAN[1]) / DET_STD[1];
    out[2 * plane + p] = (data[i + 2] / 255 - DET_MEAN[2]) / DET_STD[2];
  }

  return {
    tensor: out,
    shape: [1, 3, newH, newW],
    scaleX: origW / newW,
    scaleY: origH / newH,
    origW,
    origH,
  };
}

// Wycina poly z bitmapy (perspective-warped do prostokąta H×W) i zwraca tensor rec.
// Box: 4 punkty w oryginalnych współrzędnych obrazu, kolejność TL,TR,BR,BL.
export function preprocessRec(
  bitmap: ImageBitmap,
  box: [number, number][],
): RecInput {
  const [tl, tr, br, bl] = box;
  const wTop = Math.hypot(tr[0] - tl[0], tr[1] - tl[1]);
  const wBot = Math.hypot(br[0] - bl[0], br[1] - bl[1]);
  const hLeft = Math.hypot(bl[0] - tl[0], bl[1] - tl[1]);
  const hRight = Math.hypot(br[0] - tr[0], br[1] - tr[1]);
  const cropW = Math.max(1, Math.round(Math.max(wTop, wBot)));
  const cropH = Math.max(1, Math.round(Math.max(hLeft, hRight)));

  // resize do H=48 z zachowaniem aspect ratio, potem pad białym do max W=320
  let targetW = Math.round((cropW * REC_HEIGHT) / cropH);
  if (targetW > REC_MAX_WIDTH) targetW = REC_MAX_WIDTH;
  if (targetW < 1) targetW = 1;

  // 1) wyciągnij prostokąt — affine z 3 punktów (TL, TR, BL) do (0,0),(W,0),(0,H)
  const src = makeCanvas(targetW, REC_HEIGHT);
  const sctx = src.getContext("2d") as
    | OffscreenCanvasRenderingContext2D
    | CanvasRenderingContext2D;
  sctx.fillStyle = "#fff";
  sctx.fillRect(0, 0, targetW, REC_HEIGHT);

  // affine matrix mapping (TL → 0,0), (TR → W,0), (BL → 0,H)
  // canvas setTransform(a,b,c,d,e,f) odwzorowuje: dst = M * src + t
  // chcemy mapować punkty obrazu źródłowego do dst, więc liczymy macierz odwrotną
  const dx = tr[0] - tl[0], dy = tr[1] - tl[1];
  const ex = bl[0] - tl[0], ey = bl[1] - tl[1];
  // src-affine (image coords → unit box scaled): [u,v] = A^-1 * (p - tl), gdzie A = [[dx,ex],[dy,ey]]
  // canvas drawImage z obrazem oryginalnym + transform: dst = S * (A^-1 * (img - tl))
  // gdzie S = diag(targetW, REC_HEIGHT). Ustaw transform(a,c,b,d,e,f) tak by:
  // dst.x = a*img.x + c*img.y + e
  // dst.y = b*img.x + d*img.y + f
  const det = dx * ey - dy * ex;
  if (Math.abs(det) < 1e-6) {
    // degenerate — wystarczy axis-aligned bbox
    const xs = box.map((p) => p[0]);
    const ys = box.map((p) => p[1]);
    const x0 = Math.max(0, Math.min(...xs));
    const y0 = Math.max(0, Math.min(...ys));
    const x1 = Math.min(bitmap.width, Math.max(...xs));
    const y1 = Math.min(bitmap.height, Math.max(...ys));
    sctx.drawImage(bitmap, x0, y0, x1 - x0, y1 - y0, 0, 0, targetW, REC_HEIGHT);
  } else {
    const sx = targetW;
    const sy = REC_HEIGHT;
    // A^-1 = (1/det) * [[ey,-ex],[-dy,dx]]
    const a = (sx * ey) / det;
    const c = (-sx * ex) / det;
    const b = (-sy * dy) / det;
    const d = (sy * dx) / det;
    const e = -a * tl[0] - c * tl[1];
    const f = -b * tl[0] - d * tl[1];
    sctx.save();
    sctx.setTransform(a, b, c, d, e, f);
    sctx.drawImage(bitmap, 0, 0);
    sctx.restore();
  }

  const img = sctx.getImageData(0, 0, targetW, REC_HEIGHT);
  const data = img.data;
  const plane = targetW * REC_HEIGHT;
  const out = new Float32Array(3 * plane);
  for (let i = 0, p = 0; i < data.length; i += 4, p++) {
    out[p] = (data[i] / 255 - 0.5) / 0.5;
    out[plane + p] = (data[i + 1] / 255 - 0.5) / 0.5;
    out[2 * plane + p] = (data[i + 2] / 255 - 0.5) / 0.5;
  }

  return { tensor: out, shape: [1, 3, REC_HEIGHT, targetW] };
}

// Policz docelową szerokość crop-a (po resize do H=48), clampowaną do REC_MAX_WIDTH.
export function recTargetWidth(box: [number, number][]): number {
  const [tl, tr, br, bl] = box;
  const wTop = Math.hypot(tr[0] - tl[0], tr[1] - tl[1]);
  const wBot = Math.hypot(br[0] - bl[0], br[1] - bl[1]);
  const hLeft = Math.hypot(bl[0] - tl[0], bl[1] - tl[1]);
  const hRight = Math.hypot(br[0] - tr[0], br[1] - tr[1]);
  const cropW = Math.max(1, Math.max(wTop, wBot));
  const cropH = Math.max(1, Math.max(hLeft, hRight));
  let w = Math.round((cropW * REC_HEIGHT) / cropH);
  if (w > REC_MAX_WIDTH) w = REC_MAX_WIDTH;
  if (w < 1) w = 1;
  return w;
}

// Zbiera kilka crop-ów do jednego tensora [B,3,48,maxW] z białym paddingiem.
// Wewnątrz używa pojedynczego canvasa (maxW×48) — jeden alloc na batch zamiast B.
export function preprocessRecBatch(
  bitmap: ImageBitmap,
  boxes: [number, number][][],
): RecBatchInput {
  const widths = boxes.map(recTargetWidth);
  const maxW = Math.max(1, ...widths);
  const B = boxes.length;

  const canvas = makeCanvas(maxW, REC_HEIGHT);
  const ctx = canvas.getContext("2d") as
    | OffscreenCanvasRenderingContext2D
    | CanvasRenderingContext2D;

  const plane = REC_HEIGHT * maxW;
  const out = new Float32Array(B * 3 * plane);
  // pad = biały = (1 - 0.5)/0.5 = 1.0
  out.fill(1.0);

  for (let bi = 0; bi < B; bi++) {
    const box = boxes[bi];
    const targetW = widths[bi];

    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.fillStyle = "#fff";
    ctx.fillRect(0, 0, maxW, REC_HEIGHT);

    const [tl, tr, , bl] = box;
    const dx = tr[0] - tl[0], dy = tr[1] - tl[1];
    const ex = bl[0] - tl[0], ey = bl[1] - tl[1];
    const det = dx * ey - dy * ex;

    if (Math.abs(det) < 1e-6) {
      const xs = box.map((p) => p[0]);
      const ys = box.map((p) => p[1]);
      const x0 = Math.max(0, Math.min(...xs));
      const y0 = Math.max(0, Math.min(...ys));
      const x1 = Math.min(bitmap.width, Math.max(...xs));
      const y1 = Math.min(bitmap.height, Math.max(...ys));
      ctx.drawImage(bitmap, x0, y0, x1 - x0, y1 - y0, 0, 0, targetW, REC_HEIGHT);
    } else {
      const sx = targetW;
      const sy = REC_HEIGHT;
      const a = (sx * ey) / det;
      const c = (-sx * ex) / det;
      const b = (-sy * dy) / det;
      const d = (sy * dx) / det;
      const e = -a * tl[0] - c * tl[1];
      const f = -b * tl[0] - d * tl[1];
      ctx.setTransform(a, b, c, d, e, f);
      ctx.drawImage(bitmap, 0, 0);
    }

    // Czytamy tylko realną część (targetW), reszta tensora zostaje biała z fill.
    const img = ctx.getImageData(0, 0, targetW, REC_HEIGHT);
    const data = img.data;
    const batchOff = bi * 3 * plane;

    // Piksele są RGBA w rzędach szerokości targetW; wpisujemy do odpowiednich pozycji
    // w rzędach szerokości maxW w tensorze.
    for (let y = 0; y < REC_HEIGHT; y++) {
      const rowSrc = y * targetW * 4;
      const rowDstR = batchOff + 0 * plane + y * maxW;
      const rowDstG = batchOff + 1 * plane + y * maxW;
      const rowDstB = batchOff + 2 * plane + y * maxW;
      for (let x = 0; x < targetW; x++) {
        const s = rowSrc + x * 4;
        out[rowDstR + x] = (data[s] / 255 - 0.5) / 0.5;
        out[rowDstG + x] = (data[s + 1] / 255 - 0.5) / 0.5;
        out[rowDstB + x] = (data[s + 2] / 255 - 0.5) / 0.5;
      }
    }
  }

  return { tensor: out, shape: [B, 3, REC_HEIGHT, maxW], widths };
}
