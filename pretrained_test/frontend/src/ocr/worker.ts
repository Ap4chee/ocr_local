/// <reference lib="webworker" />
import * as ort from "onnxruntime-web/webgpu";
import type { OcrLine, OcrResponse } from "../types";
import type {
  DiagnosticReport,
  EpReport,
  WorkerRequest,
  WorkerResponse,
} from "./protocol";
import { preprocessDet, preprocessRecBatch, recTargetWidth } from "./preprocess";
import { ctcDecode, dbPostprocess, sortBoxes } from "./postprocess";
import { fetchModelCached } from "./modelCache";

const REC_BATCH_SIZE = 8;
const POLISH_DIACRITICS = "ąćęłńóśźżĄĆĘŁŃÓŚŹŻ";

declare const self: DedicatedWorkerGlobalScope & typeof globalThis;

ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.24.3/dist/";
ort.env.wasm.numThreads = Math.min(navigator.hardwareConcurrency ?? 4, 8);

let detSession: ort.InferenceSession | null = null;
let recSession: ort.InferenceSession | null = null;
let dict: string[] = [];
let backend: "webgpu" | "wasm" = "wasm";
let vocabCheckDone = false;

function post(msg: WorkerResponse, transfer: Transferable[] = []) {
  self.postMessage(msg, { transfer });
}

async function loadDict(): Promise<string[]> {
  const res = await fetch("/models/latin_dict.txt");
  if (!res.ok) throw new Error(`dict fetch failed: ${res.status}`);
  const txt = await res.text();
  const chars = txt
    .split(/\r?\n/)
    .filter((line, i, arr) => i < arr.length - 1 || line.length > 0);
  chars.push(" ");
  return chars;
}

async function init() {
  const hasWebGPU = "gpu" in navigator;
  if (!hasWebGPU) {
    post({
      type: "warning",
      msg: "Brak WebGPU — używam CPU (WASM). Będzie 5–10× wolniej.",
    });
  }
  const eps: any[] = hasWebGPU ? ["webgpu", "wasm"] : ["wasm"];

  let detBytes: Uint8Array;
  let recBytes: Uint8Array;
  try {
    [detBytes, recBytes] = await Promise.all([
      fetchModelCached("/models/det.onnx"),
      fetchModelCached("/models/rec.onnx"),
    ]);
  } catch (e) {
    throw new Error(`Pobranie modeli padło: ${e instanceof Error ? e.message : e}`);
  }

  try {
    detSession = await ort.InferenceSession.create(detBytes, {
      executionProviders: eps,
    });
    recSession = await ort.InferenceSession.create(recBytes, {
      executionProviders: eps,
    });
  } catch (e) {
    throw new Error(`Inicjalizacja sesji ORT padła: ${e instanceof Error ? e.message : e}`);
  }

  backend = hasWebGPU ? "webgpu" : "wasm";

  dict = await loadDict();

  try {
    const detH = 736, detW = 960;
    const detDummy = new Float32Array(3 * detH * detW);
    await detSession!.run({
      [detSession!.inputNames[0]]: new ort.Tensor("float32", detDummy, [1, 3, detH, detW]),
    });
  } catch { /* niekrytyczne */ }
  try {
    const recDummy = new Float32Array(3 * 48 * 160);
    await recSession!.run({
      [recSession!.inputNames[0]]: new ort.Tensor("float32", recDummy, [1, 3, 48, 160]),
    });
  } catch { /* niekrytyczne */ }

  const missing: string[] = [];
  for (const ch of POLISH_DIACRITICS) {
    if (!dict.includes(ch)) missing.push(ch);
  }
  if (missing.length > 0) {
    post({
      type: "warning",
      msg:
        `Słownik rec nie zawiera ${missing.length}/${POLISH_DIACRITICS.length} ` +
        `polskich diakrytyków (${missing.join("")}). ` +
        `Te znaki będą rozpoznawane jako najbliższy odpowiednik ASCII lub gubione. ` +
        `Rozważ model z pełnym pokryciem PL (własny CRNN, Etap 2).`,
    });
  }

  post({ type: "ready", backend });
}

async function runOcr(id: number, bitmap: ImageBitmap) {
  if (!detSession || !recSession) throw new Error("Sesje nie zainicjalizowane");
  const t0 = performance.now();

  const det = preprocessDet(bitmap);
  const detTensor = new ort.Tensor("float32", det.tensor, det.shape);
  const detOut = await detSession.run({ [detSession.inputNames[0]]: detTensor });
  const probTensor = detOut[detSession.outputNames[0]];
  const probData = probTensor.data as Float32Array;
  const [, , mapH, mapW] = probTensor.dims as number[];

  const dbResult = dbPostprocess(
    probData,
    mapH,
    mapW,
    det.scaleX * (det.shape[3] / mapW),
    det.scaleY * (det.shape[2] / mapH),
    det.origW,
    det.origH,
  );
  const boxes = sortBoxes(dbResult.boxes);

  const indexed = boxes.map((box, origIdx) => ({
    box,
    origIdx,
    targetW: recTargetWidth(box),
  }));
  indexed.sort((a, b) => a.targetW - b.targetW);

  const results: (OcrLine | null)[] = new Array(boxes.length).fill(null);

  for (let start = 0; start < indexed.length; start += REC_BATCH_SIZE) {
    const group = indexed.slice(start, start + REC_BATCH_SIZE);
    const batchIn = preprocessRecBatch(bitmap, group.map((g) => g.box));
    const recTensor = new ort.Tensor("float32", batchIn.tensor, batchIn.shape);
    const recOut = await recSession.run({ [recSession.inputNames[0]]: recTensor });
    const logits = recOut[recSession.outputNames[0]];
    const dims = logits.dims as number[];
    const B = dims[0];
    const T = dims[1];
    const V = dims[2];
    const data = logits.data as Float32Array;
    const stride = T * V;

    if (!vocabCheckDone) {
      vocabCheckDone = true;
      const expected = dict.length + 1;
      if (V !== expected) {
        post({
          type: "warning",
          msg:
            `Vocab mismatch: model rec zwraca V=${V}, a dict daje V=${expected} ` +
            `(blank + ${dict.length} znaków). Sprawdź czy latin_dict.txt pasuje do rec.onnx — ` +
            `inaczej niektóre znaki będą błędnie mapowane.`,
        });
      }
    }

    for (let j = 0; j < B; j++) {
      const slice = data.subarray(j * stride, (j + 1) * stride);
      const { text, conf } = ctcDecode(slice, T, V, dict);
      if (text.length > 0) {
        results[group[j].origIdx] = {
          text,
          conf: Math.round(conf * 10000) / 10000,
          box: group[j].box,
        };
      }
    }
  }

  const lines: OcrLine[] = results.filter((l): l is OcrLine => l !== null);
  bitmap.close();

  const dt = (performance.now() - t0) / 1000;
  const meanConf = lines.length ? lines.reduce((s, l) => s + l.conf, 0) / lines.length : 0;

  const payload: OcrResponse = {
    n_lines: lines.length,
    mean_conf: Math.round(meanConf * 10000) / 10000,
    time_s: Math.round(dt * 1000) / 1000,
    lines,
  };
  post({ type: "result", id, payload });
}

async function benchEp(ep: "webgpu" | "wasm"): Promise<EpReport> {
  const H = 736, W = 736;
  const dummy = new Float32Array(3 * H * W);
  for (let i = 0; i < dummy.length; i++) dummy[i] = Math.random() * 2 - 1;
  const shape: [number, number, number, number] = [1, 3, H, W];

  const t0 = performance.now();
  let sess: ort.InferenceSession;
  try {
    sess = await ort.InferenceSession.create("/models/det.onnx", {
      executionProviders: [ep],
    });
  } catch (e) {
    return { ok: false, error: e instanceof Error ? e.message : String(e) };
  }
  const sessionMs = performance.now() - t0;
  const inputName = sess.inputNames[0];

  try {
    await sess.run({ [inputName]: new ort.Tensor("float32", dummy.slice(), shape) });
  } catch (e) {
    return { ok: false, sessionMs, error: `run failed: ${e instanceof Error ? e.message : e}` };
  }

  const times: number[] = [];
  for (let i = 0; i < 3; i++) {
    const t1 = performance.now();
    await sess.run({ [inputName]: new ort.Tensor("float32", dummy.slice(), shape) });
    times.push(performance.now() - t1);
  }
  const meanRunMs = times.reduce((a, b) => a + b, 0) / times.length;
  try { await sess.release(); } catch { /* niekrytyczne */ }
  return { ok: true, sessionMs, meanRunMs };
}

function makeVerdict(r: DiagnosticReport): string {
  if (!r.webgpuAvailable) return "Brak WebGPU w przeglądarce — zawsze lecisz WASM.";
  if (!r.webgpu.ok) return `WebGPU padł przy ładowaniu modelu (${r.webgpu.error}). W praktyce używasz WASM.`;
  if (!r.wasm.ok) return `WASM padł (${r.wasm.error}) — coś bardzo nietypowego.`;
  const s = r.speedup ?? 0;
  if (s >= 2) return `WebGPU działa realnie: ${s.toFixed(1)}× szybszy od WASM. Label w UI jest prawdziwy.`;
  if (s >= 1.2) return `WebGPU ładuje i liczy, ale tylko ${s.toFixed(1)}× szybciej — prawdopodobny częściowy fallback niektórych ops na CPU.`;
  return `WebGPU ładuje model ale praktycznie nie przyspiesza (${s.toFixed(1)}×) — prawie na pewno silent fallback większości grafu na WASM.`;
}

async function diagnose(id: number) {
  const report: DiagnosticReport = {
    webgpuAvailable: "gpu" in navigator,
    adapter: null,
    webgpu: { ok: false },
    wasm: { ok: false },
    speedup: null,
    verdict: "",
  };

  if (report.webgpuAvailable) {
    try {
      const gpu = (navigator as any).gpu;
      const adapter = await gpu.requestAdapter();
      if (adapter) {
        const info =
          adapter.info ??
          (adapter.requestAdapterInfo ? await adapter.requestAdapterInfo() : null);
        report.adapter = info
          ? {
              vendor: info.vendor,
              architecture: info.architecture,
              description: info.description,
            }
          : { description: "adapter OK (info niewystawione przez przeglądarkę)" };
      } else {
        report.adapter = { description: "requestAdapter() zwrócił null" };
      }
    } catch (e) {
      report.adapter = { description: `adapter error: ${e instanceof Error ? e.message : e}` };
    }
    report.webgpu = await benchEp("webgpu");
  }
  report.wasm = await benchEp("wasm");

  if (
    report.webgpu.ok &&
    report.wasm.ok &&
    report.webgpu.meanRunMs &&
    report.wasm.meanRunMs
  ) {
    report.speedup = report.wasm.meanRunMs / report.webgpu.meanRunMs;
  }
  report.verdict = makeVerdict(report);

  post({ type: "diagnostic", id, report });
}

self.onmessage = async (ev: MessageEvent<WorkerRequest>) => {
  const msg = ev.data;
  try {
    if (msg.type === "init") {
      await init();
    } else if (msg.type === "ocr") {
      await runOcr(msg.id, msg.bitmap);
    } else if (msg.type === "diagnose") {
      await diagnose(msg.id);
    }
  } catch (e) {
    const errMsg = e instanceof Error ? e.message : String(e);
    post({
      type: "error",
      id: msg.type === "init" ? undefined : msg.id,
      msg: errMsg,
    });
  }
};
