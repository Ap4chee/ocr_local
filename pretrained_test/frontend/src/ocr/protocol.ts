import type { OcrResponse } from "../types";

export type WorkerRequest =
  | { type: "init" }
  | { type: "ocr"; id: number; bitmap: ImageBitmap }
  | { type: "diagnose"; id: number };

export interface EpReport {
  ok: boolean;
  sessionMs?: number;
  meanRunMs?: number;
  error?: string;
}

export interface DiagnosticReport {
  webgpuAvailable: boolean;
  adapter: { vendor?: string; architecture?: string; description?: string } | null;
  webgpu: EpReport;
  wasm: EpReport;
  speedup: number | null;
  verdict: string;
}

export type WorkerResponse =
  | { type: "ready"; backend: "webgpu" | "wasm" }
  | { type: "warning"; msg: string }
  | { type: "error"; id?: number; msg: string }
  | { type: "result"; id: number; payload: OcrResponse }
  | { type: "diagnostic"; id: number; report: DiagnosticReport };
