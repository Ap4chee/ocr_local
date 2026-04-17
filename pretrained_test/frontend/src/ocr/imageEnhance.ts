// Image enhancement before OCR: CLAHE (adaptive contrast) + unsharp mask (sharpening).
// Runs in main thread on the original-resolution bitmap before transfer to worker.

function makeOffscreen(w: number, h: number): OffscreenCanvas | HTMLCanvasElement {
  if (typeof OffscreenCanvas !== "undefined") return new OffscreenCanvas(w, h);
  const c = document.createElement("canvas");
  c.width = w;
  c.height = h;
  return c;
}

// Contrast Limited Adaptive Histogram Equalization on luminance channel.
// Scales RGB proportionally so colour tint is preserved.
// tilesX/tilesY: grid resolution (8×8 = standard).
// clipLimit: noise amplification cap per tile (2–4 typical for documents).
function applyClahe(
  data: Uint8ClampedArray,
  w: number,
  h: number,
  tilesX = 8,
  tilesY = 8,
  clipLimit = 3.0,
): void {
  const nx = Math.max(1, Math.min(tilesX, w));
  const ny = Math.max(1, Math.min(tilesY, h));

  // Per-tile: histogram → clip → redistribute → CDF → LUT
  const luts: Uint8Array[][] = [];
  for (let ty = 0; ty < ny; ty++) {
    luts[ty] = [];
    for (let tx = 0; tx < nx; tx++) {
      const x0 = Math.round((tx / nx) * w);
      const y0 = Math.round((ty / ny) * h);
      const x1 = Math.round(((tx + 1) / nx) * w);
      const y1 = Math.round(((ty + 1) / ny) * h);

      const hist = new Float64Array(256);
      let count = 0;
      for (let y = y0; y < y1; y++) {
        for (let x = x0; x < x1; x++) {
          const i = (y * w + x) * 4;
          const lum = ((0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2]) + 0.5) | 0;
          hist[lum]++;
          count++;
        }
      }

      if (count === 0) { luts[ty][tx] = new Uint8Array(256); continue; }

      const clip = clipLimit * (count / 256);
      let excess = 0;
      for (let v = 0; v < 256; v++) {
        if (hist[v] > clip) { excess += hist[v] - clip; hist[v] = clip; }
      }
      const add = excess / 256;
      for (let v = 0; v < 256; v++) hist[v] += add;

      const lut = new Uint8Array(256);
      let cdf = 0;
      const scale = 255 / count;
      for (let v = 0; v < 256; v++) {
        cdf += hist[v];
        lut[v] = Math.min(255, (cdf * scale + 0.5) | 0);
      }
      luts[ty][tx] = lut;
    }
  }

  // Apply: bilinear interpolation between 4 surrounding tile LUTs
  for (let y = 0; y < h; y++) {
    const fyRaw = ((y + 0.5) / h) * ny - 0.5;
    const ty0 = Math.max(0, Math.min(ny - 1, Math.floor(fyRaw)));
    const ty1 = Math.min(ny - 1, ty0 + 1);
    const fy = Math.max(0, Math.min(1, fyRaw - ty0));

    for (let x = 0; x < w; x++) {
      const fxRaw = ((x + 0.5) / w) * nx - 0.5;
      const tx0 = Math.max(0, Math.min(nx - 1, Math.floor(fxRaw)));
      const tx1 = Math.min(nx - 1, tx0 + 1);
      const fx = Math.max(0, Math.min(1, fxRaw - tx0));

      const i = (y * w + x) * 4;
      const lum = ((0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2]) + 0.5) | 0;

      const v00 = luts[ty0][tx0][lum];
      const v01 = luts[ty0][tx1][lum];
      const v10 = luts[ty1][tx0][lum];
      const v11 = luts[ty1][tx1][lum];
      const newLum =
        v00 * (1 - fx) * (1 - fy) +
        v01 * fx       * (1 - fy) +
        v10 * (1 - fx) * fy       +
        v11 * fx       * fy;

      if (lum === 0) {
        data[i] = data[i + 1] = data[i + 2] = 0;
      } else {
        const s = newLum / lum;
        data[i]     = Math.min(255, (data[i]     * s + 0.5) | 0);
        data[i + 1] = Math.min(255, (data[i + 1] * s + 0.5) | 0);
        data[i + 2] = Math.min(255, (data[i + 2] * s + 0.5) | 0);
      }
    }
  }
}

// Unsharp mask via separable 3-tap Gaussian blur [1,2,1]/4.
// amount: 0 = no effect, 1 = 100% sharpening boost.
function applyUnsharpMask(
  data: Uint8ClampedArray,
  w: number,
  h: number,
  amount = 0.8,
): void {
  const tmp = new Uint8ClampedArray(data.length);
  const row = w * 4;

  // Horizontal blur pass → tmp
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = (y * w + x) * 4;
      const l = x > 0 ? i - 4 : i;
      const r = x < w - 1 ? i + 4 : i;
      tmp[i]     = (data[l]     + 2 * data[i]     + data[r])     >> 2;
      tmp[i + 1] = (data[l + 1] + 2 * data[i + 1] + data[r + 1]) >> 2;
      tmp[i + 2] = (data[l + 2] + 2 * data[i + 2] + data[r + 2]) >> 2;
      tmp[i + 3] = 255;
    }
  }

  // Vertical blur pass + apply unsharp delta back to data
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = (y * w + x) * 4;
      const u = y > 0 ? i - row : i;
      const d = y < h - 1 ? i + row : i;
      for (let c = 0; c < 3; c++) {
        const blurred = (tmp[u + c] + 2 * tmp[i + c] + tmp[d + c]) >> 2;
        const v = data[i + c] + amount * (data[i + c] - blurred);
        data[i + c] = v < 0 ? 0 : v > 255 ? 255 : v | 0;
      }
    }
  }
}

async function canvasToObjectUrl(canvas: OffscreenCanvas | HTMLCanvasElement): Promise<string> {
  if (canvas instanceof OffscreenCanvas) {
    const blob = await canvas.convertToBlob({ type: "image/jpeg", quality: 0.88 });
    return URL.createObjectURL(blob);
  }
  return new Promise((resolve) => {
    (canvas as HTMLCanvasElement).toBlob(
      (blob) => resolve(blob ? URL.createObjectURL(blob) : ""),
      "image/jpeg",
      0.88,
    );
  });
}

export interface EnhanceResult {
  enhanced: ImageBitmap;
  previewUrl: string;
  ms: number;
}

export async function enhanceForOcr(bitmap: ImageBitmap): Promise<EnhanceResult> {
  const t0 = performance.now();
  const { width: w, height: h } = bitmap;

  const canvas = makeOffscreen(w, h);
  const ctx = canvas.getContext("2d") as
    | OffscreenCanvasRenderingContext2D
    | CanvasRenderingContext2D;

  ctx.drawImage(bitmap, 0, 0);
  const imgData = ctx.getImageData(0, 0, w, h);

  applyClahe(imgData.data, w, h, 8, 8, 3.0);
  applyUnsharpMask(imgData.data, w, h, 0.8);

  ctx.putImageData(imgData, 0, 0);

  const [enhanced, previewUrl] = await Promise.all([
    createImageBitmap(canvas),
    canvasToObjectUrl(canvas),
  ]);

  return { enhanced, previewUrl, ms: performance.now() - t0 };
}
