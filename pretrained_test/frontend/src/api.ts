import type { OcrResponse } from "./types";
import type { DiagnosticReport, WorkerRequest, WorkerResponse } from "./ocr/protocol";
import { enhanceForOcr } from "./ocr/imageEnhance";

export type BackendStatus =
  | { state: "loading" }
  | { state: "ready"; backend: "webgpu" | "wasm" }
  | { state: "error"; msg: string };

let worker: Worker | null = null;
let nextId = 1;
const pending = new Map<
  number,
  { resolve: (r: OcrResponse) => void; reject: (e: Error) => void }
>();
let readyPromise: Promise<{ backend: "webgpu" | "wasm" }> | null = null;
const warningListeners = new Set<(msg: string) => void>();

function ensureWorker(): Worker {
  if (worker) return worker;
  worker = new Worker(new URL("./ocr/worker.ts", import.meta.url), { type: "module" });
  worker.onmessage = (ev: MessageEvent<WorkerResponse>) => {
    const msg = ev.data;
    if (msg.type === "result") {
      pending.get(msg.id)?.resolve(msg.payload);
      pending.delete(msg.id);
    } else if (msg.type === "error") {
      const err = new Error(msg.msg);
      if (msg.id !== undefined) {
        pending.get(msg.id)?.reject(err);
        pending.delete(msg.id);
      }
    } else if (msg.type === "warning") {
      warningListeners.forEach((cb) => cb(msg.msg));
    }
  };
  worker.onerror = (e) => {
    const err = new Error(`Worker error: ${e.message}`);
    pending.forEach(({ reject }) => reject(err));
    pending.clear();
  };
  return worker;
}

export function initOcr(): Promise<{ backend: "webgpu" | "wasm" }> {
  if (readyPromise) return readyPromise;
  const w = ensureWorker();
  readyPromise = new Promise((resolve, reject) => {
    const onMsg = (ev: MessageEvent<WorkerResponse>) => {
      const msg = ev.data;
      if (msg.type === "ready") {
        w.removeEventListener("message", onMsg);
        resolve({ backend: msg.backend });
      } else if (msg.type === "error" && msg.id === undefined) {
        w.removeEventListener("message", onMsg);
        reject(new Error(msg.msg));
      }
    };
    w.addEventListener("message", onMsg);
    const req: WorkerRequest = { type: "init" };
    w.postMessage(req);
  });
  return readyPromise;
}

export function onOcrWarning(cb: (msg: string) => void): () => void {
  warningListeners.add(cb);
  return () => warningListeners.delete(cb);
}

export async function postOcrRaw(file: File): Promise<OcrResponse> {
  await initOcr();
  const w = ensureWorker();
  const bitmap = await createImageBitmap(file);
  const id = nextId++;
  const promise = new Promise<OcrResponse>((resolve, reject) => {
    pending.set(id, { resolve, reject });
  });
  w.postMessage({ type: "ocr", id, bitmap } as WorkerRequest, [bitmap]);
  return promise;
}

export interface OcrResult {
  result: OcrResponse;
  previewUrl: string;
  enhanceMs: number;
}

export async function postOcr(file: File): Promise<OcrResult> {
  await initOcr();
  const w = ensureWorker();
  const original = await createImageBitmap(file);

  let detBitmap: ImageBitmap = original;
  let recBitmap: ImageBitmap | undefined;   // undefined = worker reuses detBitmap
  let previewUrl = "";
  let enhanceMs = 0;

  try {
    const enh = await enhanceForOcr(original);
    detBitmap = enh.enhanced;   // enhanced → detection only
    recBitmap = original;        // original → recognition crops
    previewUrl = enh.previewUrl;
    enhanceMs = enh.ms;
  } catch {
    // Enhancement failed — detBitmap === original, no separate recBitmap
  }

  const id = nextId++;
  const promise = new Promise<OcrResponse>((resolve, reject) => {
    pending.set(id, { resolve, reject });
  });
  const transferables: Transferable[] = [detBitmap];
  if (recBitmap) transferables.push(recBitmap);
  const req: WorkerRequest = { type: "ocr", id, bitmap: detBitmap, originalBitmap: recBitmap };
  w.postMessage(req, transferables);
  const result = await promise;
  return { result, previewUrl, enhanceMs };
}

export async function runDiagnostics(): Promise<DiagnosticReport> {
  await initOcr();
  const w = ensureWorker();
  const id = nextId++;
  return new Promise<DiagnosticReport>((resolve, reject) => {
    const onMsg = (ev: MessageEvent<WorkerResponse>) => {
      const msg = ev.data;
      if (msg.type === "diagnostic" && msg.id === id) {
        w.removeEventListener("message", onMsg);
        resolve(msg.report);
      } else if (msg.type === "error" && msg.id === id) {
        w.removeEventListener("message", onMsg);
        reject(new Error(msg.msg));
      }
    };
    w.addEventListener("message", onMsg);
    const req: WorkerRequest = { type: "diagnose", id };
    w.postMessage(req);
  });
}
