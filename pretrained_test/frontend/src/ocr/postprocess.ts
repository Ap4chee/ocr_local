// Postprocessing: DB (detekcja) + CTC (rozpoznawanie).

const DB_THRESH = 0.3;
// PaddleOCR default box confidence threshold is 0.5; using a slightly lower value improves detection on borderline images.
const DB_BOX_THRESH = 0.5;
// PaddleOCR uses an unclip ratio of 2.0 for expanding boxes; matching it reduces false‑negative boxes.
const DB_UNCLIP_RATIO = 2.0;
const DB_MIN_SIZE = 3;

type Pt = [number, number];

export interface DbStats {
  pixelsAboveThresh: number;
  rawComponents: number;
  survivedMinSize: number;
  survivedScore: number;
  survivedFinalSize: number;
  scoreMaxOfRejected: number;
}

// === DB postprocess ===

export function dbPostprocess(
  probMap: Float32Array,
  mapH: number,
  mapW: number,
  scaleX: number,
  scaleY: number,
  origW: number,
  origH: number,
): { boxes: Pt[][]; stats: DbStats } {
  const stats: DbStats = {
    pixelsAboveThresh: 0,
    rawComponents: 0,
    survivedMinSize: 0,
    survivedScore: 0,
    survivedFinalSize: 0,
    scoreMaxOfRejected: 0,
  };

  // 1) binarize
  const bin = new Uint8Array(mapH * mapW);
  for (let i = 0; i < bin.length; i++) {
    const b = probMap[i] > DB_THRESH ? 1 : 0;
    bin[i] = b;
    if (b) stats.pixelsAboveThresh++;
  }

  // 2) connected components (4-conn flood fill, iterative)
  const labels = new Int32Array(mapH * mapW);
  let nextLabel = 0;
  const stack: number[] = [];
  const components: number[][] = []; // każda lista pikseli flat-idx

  for (let y = 0; y < mapH; y++) {
    for (let x = 0; x < mapW; x++) {
      const idx = y * mapW + x;
      if (!bin[idx] || labels[idx]) continue;
      nextLabel++;
      const comp: number[] = [];
      stack.length = 0;
      stack.push(idx);
      labels[idx] = nextLabel;
      while (stack.length) {
        const p = stack.pop()!;
        comp.push(p);
        const py = (p / mapW) | 0;
        const px = p - py * mapW;
        const neigh = [
          px > 0 ? p - 1 : -1,
          px < mapW - 1 ? p + 1 : -1,
          py > 0 ? p - mapW : -1,
          py < mapH - 1 ? p + mapW : -1,
        ];
        for (const n of neigh) {
          if (n >= 0 && bin[n] && !labels[n]) {
            labels[n] = nextLabel;
            stack.push(n);
          }
        }
      }
      components.push(comp);
    }
  }

  stats.rawComponents = components.length;

  const boxes: Pt[][] = [];
  for (const comp of components) {
    if (comp.length < DB_MIN_SIZE * DB_MIN_SIZE) continue;
    stats.survivedMinSize++;

    // contour points (border pixels of component) — wystarczy convex hull z całości
    const pts: Pt[] = comp.map((p) => {
      const py = (p / mapW) | 0;
      const px = p - py * mapW;
      return [px, py];
    });

    const hull = convexHull(pts);
    if (hull.length < 3) continue;

    const rect = minAreaRect(hull); // 4 punkty rotated
    if (!rect) continue;

    // mean prob na pikselach komponentu
    let sum = 0;
    for (const p of comp) sum += probMap[p];
    const score = sum / comp.length;
    if (score < DB_BOX_THRESH) {
      if (score > stats.scoreMaxOfRejected) stats.scoreMaxOfRejected = score;
      continue;
    }
    stats.survivedScore++;

    // unclip
    const expanded = unclip(rect, DB_UNCLIP_RATIO);
    const finalRect = minAreaRect(expanded);
    if (!finalRect) continue;

    // sortuj clockwise od TL
    const ordered = orderPoints(finalRect);

    // skala do oryginalnego obrazu + clipping
    const scaled: Pt[] = ordered.map(([x, y]) => [
      Math.max(0, Math.min(origW, x * scaleX)),
      Math.max(0, Math.min(origH, y * scaleY)),
    ]);

    // odrzuć zbyt małe
    const w = Math.hypot(scaled[1][0] - scaled[0][0], scaled[1][1] - scaled[0][1]);
    const h = Math.hypot(scaled[3][0] - scaled[0][0], scaled[3][1] - scaled[0][1]);
    if (w < DB_MIN_SIZE + 1 || h < DB_MIN_SIZE + 1) continue;
    stats.survivedFinalSize++;

    boxes.push(scaled);
  }

  return { boxes, stats };
}

function convexHull(points: Pt[]): Pt[] {
  if (points.length < 3) return points.slice();
  const pts = points.slice().sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  const n = pts.length;
  const cross = (o: Pt, a: Pt, b: Pt) =>
    (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
  const lower: Pt[] = [];
  for (const p of pts) {
    while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], p) <= 0)
      lower.pop();
    lower.push(p);
  }
  const upper: Pt[] = [];
  for (let i = n - 1; i >= 0; i--) {
    const p = pts[i];
    while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], p) <= 0)
      upper.pop();
    upper.push(p);
  }
  lower.pop();
  upper.pop();
  return lower.concat(upper);
}

// Rotating calipers — min area rect z convex hull.
function minAreaRect(hull: Pt[]): Pt[] | null {
  if (hull.length < 3) {
    if (hull.length === 0) return null;
    // expand do prostokąta
    const xs = hull.map((p) => p[0]);
    const ys = hull.map((p) => p[1]);
    const x0 = Math.min(...xs), y0 = Math.min(...ys);
    const x1 = Math.max(...xs), y1 = Math.max(...ys);
    return [
      [x0, y0],
      [x1, y0],
      [x1, y1],
      [x0, y1],
    ];
  }

  let bestArea = Infinity;
  let bestRect: Pt[] | null = null;
  const n = hull.length;
  for (let i = 0; i < n; i++) {
    const j = (i + 1) % n;
    const ex = hull[j][0] - hull[i][0];
    const ey = hull[j][1] - hull[i][1];
    const len = Math.hypot(ex, ey) || 1;
    const ux = ex / len, uy = ey / len;
    // wektor prostopadły
    const vx = -uy, vy = ux;
    let minU = Infinity, maxU = -Infinity, minV = Infinity, maxV = -Infinity;
    for (const p of hull) {
      const u = p[0] * ux + p[1] * uy;
      const v = p[0] * vx + p[1] * vy;
      if (u < minU) minU = u;
      if (u > maxU) maxU = u;
      if (v < minV) minV = v;
      if (v > maxV) maxV = v;
    }
    const area = (maxU - minU) * (maxV - minV);
    if (area < bestArea) {
      bestArea = area;
      bestRect = [
        [minU * ux + minV * vx, minU * uy + minV * vy],
        [maxU * ux + minV * vx, maxU * uy + minV * vy],
        [maxU * ux + maxV * vx, maxU * uy + maxV * vy],
        [minU * ux + maxV * vx, minU * uy + maxV * vy],
      ];
    }
  }
  return bestRect;
}

// Unclip: rozszerz wielokąt o offset = area * ratio / perimeter (tak robi PaddleOCR via pyclipper).
function unclip(rect: Pt[], ratio: number): Pt[] {
  const area = polyArea(rect);
  const perim = polyPerimeter(rect);
  if (perim < 1e-6) return rect;
  const distance = (area * ratio) / perim;
  // offset każdej krawędzi na zewnątrz o `distance`
  const n = rect.length;
  const out: Pt[] = [];
  for (let i = 0; i < n; i++) {
    const prev = rect[(i - 1 + n) % n];
    const cur = rect[i];
    const next = rect[(i + 1) % n];

    const e1x = cur[0] - prev[0], e1y = cur[1] - prev[1];
    const e2x = next[0] - cur[0], e2y = next[1] - cur[1];
    const l1 = Math.hypot(e1x, e1y) || 1;
    const l2 = Math.hypot(e2x, e2y) || 1;
    // normalne (na zewnątrz dla CCW; dla CW odwrócić)
    let n1x = e1y / l1, n1y = -e1x / l1;
    let n2x = e2y / l2, n2y = -e2x / l2;

    // suma normalnych wskazuje kierunek przesunięcia rogu
    let bx = n1x + n2x, by = n1y + n2y;
    const blen = Math.hypot(bx, by) || 1;
    bx /= blen;
    by /= blen;

    // długość przesunięcia tak by krawędź odsunęła się o distance
    const dot = bx * n1x + by * n1y;
    const k = dot !== 0 ? distance / dot : distance;

    // sprawdź orientację — jeśli wielokąt CW, neguj
    out.push([cur[0] + bx * k, cur[1] + by * k]);
  }

  // Jeśli wynikowy wielokąt jest mniejszy → znaczy że sortowanie było CW, zrób z drugiej strony
  if (polyArea(out) < area) {
    return rect.map((cur, i) => {
      const prev = rect[(i - 1 + n) % n];
      const next = rect[(i + 1) % n];
      const e1x = cur[0] - prev[0], e1y = cur[1] - prev[1];
      const e2x = next[0] - cur[0], e2y = next[1] - cur[1];
      const l1 = Math.hypot(e1x, e1y) || 1;
      const l2 = Math.hypot(e2x, e2y) || 1;
      const n1x = -e1y / l1, n1y = e1x / l1;
      const n2x = -e2y / l2, n2y = e2x / l2;
      let bx = n1x + n2x, by = n1y + n2y;
      const blen = Math.hypot(bx, by) || 1;
      bx /= blen;
      by /= blen;
      const dot = bx * n1x + by * n1y;
      const k = dot !== 0 ? distance / dot : distance;
      return [cur[0] + bx * k, cur[1] + by * k] as Pt;
    });
  }
  return out;
}

function polyArea(poly: Pt[]): number {
  let s = 0;
  const n = poly.length;
  for (let i = 0; i < n; i++) {
    const [x1, y1] = poly[i];
    const [x2, y2] = poly[(i + 1) % n];
    s += x1 * y2 - x2 * y1;
  }
  return Math.abs(s) / 2;
}

function polyPerimeter(poly: Pt[]): number {
  let s = 0;
  const n = poly.length;
  for (let i = 0; i < n; i++) {
    const [x1, y1] = poly[i];
    const [x2, y2] = poly[(i + 1) % n];
    s += Math.hypot(x2 - x1, y2 - y1);
  }
  return s;
}

// Sortuj 4 punkty: TL, TR, BR, BL (zgodnie z konwencją PaddleOCR).
function orderPoints(rect: Pt[]): Pt[] {
  const sorted = rect.slice().sort((a, b) => a[0] - b[0]);
  const left = sorted.slice(0, 2).sort((a, b) => a[1] - b[1]); // TL, BL
  const right = sorted.slice(2, 4).sort((a, b) => a[1] - b[1]); // TR, BR
  return [left[0], right[0], right[1], left[1]];
}

// Posortuj boxy top-to-bottom, left-to-right (jak w PaddleOCR).
export function sortBoxes(boxes: Pt[][]): Pt[][] {
  const withCenters = boxes.map((b) => ({
    box: b,
    cy: (b[0][1] + b[2][1]) / 2,
    cx: (b[0][0] + b[2][0]) / 2,
  }));
  withCenters.sort((a, b) => {
    if (Math.abs(a.cy - b.cy) < 10) return a.cx - b.cx;
    return a.cy - b.cy;
  });
  return withCenters.map((w) => w.box);
}

// === CTC decode ===

// logits: [T, V] (lub [1, T, V]); blank=0 dla PaddleOCR.
// Zwraca tekst + średnią konfidencję (max-prob na non-blank tokenach).
export function ctcDecode(
  logits: Float32Array,
  T: number,
  V: number,
  dict: string[], // index 0 = blank, znaki od 1
): { text: string; conf: number } {
  let prev = -1;
  let chars = "";
  let confSum = 0;
  let confCount = 0;
  for (let t = 0; t < T; t++) {
    const off = t * V;
    let best = 0;
    let bestVal = logits[off];
    for (let v = 1; v < V; v++) {
      const x = logits[off + v];
      if (x > bestVal) {
        bestVal = x;
        best = v;
      }
    }
    if (best !== 0 && best !== prev) {
      // wartości w rec PaddleOCR są już po Softmax → traktujemy bestVal jako prob.
      // Jeśli to logit, użytkownik i tak dostanie monotoniczną miarę.
      const c = dict[best - 1];
      if (c !== undefined) {
        chars += c;
        confSum += bestVal;
        confCount++;
      }
    }
    prev = best;
  }
  return { text: chars, conf: confCount ? confSum / confCount : 0 };
}
