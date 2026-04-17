import { useEffect, useRef, useState } from "react";
import { initOcr, onOcrWarning, postOcr, runDiagnostics } from "./api";
import { ImageOverlay } from "./ImageOverlay";
import { LinesList } from "./LinesList";
import type { DiagnosticReport } from "./ocr/protocol";
import type { OcrResponse } from "./types";

type EngineStatus =
  | { state: "loading" }
  | { state: "ready"; backend: "webgpu" | "wasm" }
  | { state: "error"; msg: string };

export function App() {
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [result, setResult] = useState<OcrResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [engine, setEngine] = useState<EngineStatus>({ state: "loading" });
  const [warning, setWarning] = useState<string | null>(null);
  const [highlight, setHighlight] = useState<number | null>(null);
  const [minConf, setMinConf] = useState(0);
  const [diag, setDiag] = useState<DiagnosticReport | null>(null);
  const [diagRunning, setDiagRunning] = useState(false);
  const [diagError, setDiagError] = useState<string | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const [imageDims, setImageDims] = useState<{ w: number; h: number } | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [enhanceMs, setEnhanceMs] = useState(0);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const visibleLines = result ? result.lines.filter((l) => l.conf >= minConf) : [];

  useEffect(() => {
    const off = onOcrWarning(setWarning);
    initOcr()
      .then(({ backend }) => setEngine({ state: "ready", backend }))
      .catch((e) =>
        setEngine({ state: "error", msg: e instanceof Error ? e.message : String(e) }),
      );
    return off;
  }, []);

  useEffect(() => {
    return () => { if (imageUrl) URL.revokeObjectURL(imageUrl); };
  }, [imageUrl]);

  useEffect(() => {
    return () => { if (previewUrl) URL.revokeObjectURL(previewUrl); };
  }, [previewUrl]);

  useEffect(() => {
    function onPaste(e: ClipboardEvent) {
      const items = e.clipboardData?.items;
      if (!items) return;
      for (const it of items) {
        if (it.kind === "file" && it.type.startsWith("image/")) {
          const f = it.getAsFile();
          if (f) {
            e.preventDefault();
            handleFile(f);
            return;
          }
        }
      }
    }
    window.addEventListener("paste", onPaste);
    return () => window.removeEventListener("paste", onPaste);
  }, []);

  async function handleFile(file: File) {
    setError(null);
    setResult(null);
    setHighlight(null);
    setImageDims(null);
    setPreviewUrl(null);
    setEnhanceMs(0);
    if (imageUrl) URL.revokeObjectURL(imageUrl);
    const url = URL.createObjectURL(file);
    setImageUrl(url);

    const img = new Image();
    img.onload = () => setImageDims({ w: img.naturalWidth, h: img.naturalHeight });
    img.src = url;

    setLoading(true);
    try {
      const { result, previewUrl: pUrl, enhanceMs: ms } = await postOcr(file);
      setResult(result);
      if (pUrl) { setPreviewUrl(pUrl); setEnhanceMs(ms); }
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }

  function onDrop(e: React.DragEvent) {
    e.preventDefault();
    setDragOver(false);
    const f = e.dataTransfer.files[0];
    if (f) handleFile(f);
  }

  async function handleDiagnose() {
    setDiag(null);
    setDiagError(null);
    setDiagRunning(true);
    try {
      setDiag(await runDiagnostics());
    } catch (e) {
      setDiagError(e instanceof Error ? e.message : String(e));
    } finally {
      setDiagRunning(false);
    }
  }

  const engineStatus =
    engine.state === "loading" ? "loading"
    : engine.state === "error" ? "bad"
    : engine.backend === "webgpu" ? "ok"
    : ""; // wasm = no special class

  const engineLabel =
    engine.state === "loading" ? "Ładowanie modeli…"
    : engine.state === "error" ? "Błąd silnika"
    : engine.backend === "webgpu" ? "WebGPU"
    : "WASM (CPU)";

  return (
    <div className="app">
      {/* ── HEADER ── */}
      <header className="header">
        <div className="header-brand">
          <div className="logo">
            <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
              <path d="M9 3H5a2 2 0 0 0-2 2v4m6-6h10a2 2 0 0 1 2 2v4M9 3v18m0 0h10a2 2 0 0 0 2-2v-4M9 21H5a2 2 0 0 1-2-2v-4m0 0h18" stroke="#fff" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
          </div>
          <div>
            <div className="brand-title">OCR Studio</div>
            <div className="brand-sub">PP-OCRv5 · in-browser inference · no server</div>
          </div>
        </div>

        <span className={`status-badge ${engineStatus}`} title={engine.state === "error" ? engine.msg : undefined} id="engine-status">
          <span className="dot" />
          {engineLabel}
        </span>

        <button
          id="btn-diagnose"
          className="btn"
          onClick={handleDiagnose}
          disabled={diagRunning || engine.state !== "ready"}
        >
          {diagRunning ? (
            <>
              <span className="spinner" style={{ width: 13, height: 13, borderWidth: 2 }} />
              Diagnozuję…
            </>
          ) : (
            <>
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M9 3H5a2 2 0 0 0-2 2v4m6-6h10a2 2 0 0 1 2 2v4M9 3v18m0 0h10a2 2 0 0 0 2-2v-4M9 21H5a2 2 0 0 1-2-2v-4m0 0h18"/>
              </svg>
              Diagnostyka
            </>
          )}
        </button>
      </header>

      {/* ── ALERTS ── */}
      {warning && (
        <div className="alert alert-warning" role="alert">
          <svg className="alert-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/>
          </svg>
          <span>{warning}</span>
        </div>
      )}

      {diagError && (
        <div className="alert alert-error" role="alert">
          <svg className="alert-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/>
          </svg>
          <span>Diagnostyka padła: {diagError}</span>
        </div>
      )}

      {engine.state === "error" && (
        <div className="alert alert-error" role="alert">
          <svg className="alert-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/>
          </svg>
          <span>Silnik nie wystartował: {engine.msg}. Spróbuj Chrome/Edge 113+ z włączonym WebGPU lub odśwież stronę.</span>
        </div>
      )}

      {/* ── DIAGNOSTICS PANEL ── */}
      {diag && (
        <div className="diag-panel">
          <div className="diag-verdict">{diag.verdict}</div>
          <div className="diag-row">
            <span className="diag-label">Adapter&nbsp;</span>
            <span className="diag-val">
              {diag.adapter?.vendor ?? "?"}
              {diag.adapter?.architecture ? ` / ${diag.adapter.architecture}` : ""}
              {diag.adapter?.description ? ` (${diag.adapter.description})` : ""}
            </span>
          </div>
          <div className="diag-row">
            <span className="diag-label">WebGPU&nbsp;</span>
            {diag.webgpu.ok
              ? <span className="diag-val">load {diag.webgpu.sessionMs?.toFixed(0)} ms · run {diag.webgpu.meanRunMs?.toFixed(1)} ms avg</span>
              : <span className="diag-val err">ERROR — {diag.webgpu.error}</span>
            }
          </div>
          <div className="diag-row">
            <span className="diag-label">WASM&nbsp;</span>
            {diag.wasm.ok
              ? <span className="diag-val">load {diag.wasm.sessionMs?.toFixed(0)} ms · run {diag.wasm.meanRunMs?.toFixed(1)} ms avg</span>
              : <span className="diag-val err">ERROR — {diag.wasm.error}</span>
            }
          </div>
          {diag.speedup != null && (
            <div className="diag-speedup">⚡ Speedup WebGPU vs WASM: <strong>{diag.speedup.toFixed(2)}×</strong></div>
          )}
        </div>
      )}

      {/* ── DROPZONE ── */}
      <div
        id="dropzone"
        className={`dropzone${dragOver ? " drag-over" : ""}`}
        onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
        onDragLeave={() => setDragOver(false)}
        onDrop={onDrop}
        onClick={() => fileInputRef.current?.click()}
        role="button"
        aria-label="Upuść obraz lub kliknij, aby wybrać plik"
        tabIndex={0}
        onKeyDown={(e) => e.key === "Enter" && fileInputRef.current?.click()}
      >
        <input
          ref={fileInputRef}
          id="file-input"
          type="file"
          accept="image/*"
          hidden
          onChange={(e) => e.target.files?.[0] && handleFile(e.target.files[0])}
        />
        <div className="dropzone-inner">
          <div className="dropzone-icon">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
              <rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="8.5" cy="8.5" r="1.5"/><polyline points="21 15 16 10 5 21"/>
            </svg>
          </div>
          <div className="dropzone-title">
            {dragOver ? "Upuść obraz tutaj" : "Upuść obraz, kliknij lub wklej"}
          </div>
          <div className="dropzone-sub">PNG, JPG, WEBP — dokumenty, faktury, paragony</div>
          <div className="dropzone-kbd">
            <span className="kbd">Ctrl</span> + <span className="kbd">V</span>
            &ensp;·&ensp; przeciągnij i upuść
          </div>
        </div>
      </div>

      {/* ── ERROR ── */}
      {error && (
        <div className="alert alert-error" role="alert">
          <svg className="alert-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/>
          </svg>
          <span>Błąd OCR: {error}</span>
        </div>
      )}

      {/* ── PROCESSING ── */}
      {loading && (
        <div className="processing-bar">
          <div className="spinner" />
          <span className="processing-text">Przetwarzanie obrazu…</span>
          <div className="processing-shimmer" />
        </div>
      )}

      {/* ── STATS ── */}
      {result && (
        <div className="stats-bar">
          <div className="stat-chip">
            <strong>{visibleLines.length}</strong>/{result.n_lines} linii
          </div>
          <div className="stat-divider" />
          <div className="stat-chip">
            avg conf <strong>{(result.mean_conf * 100).toFixed(1)}%</strong>
          </div>
          <div className="stat-divider" />
          <div className="stat-chip">
            czas <strong>{result.time_s.toFixed(2)} s</strong>
          </div>
          {enhanceMs > 0 && (
            <>
              <div className="stat-divider" />
              <div className="stat-chip">
                preprocessing <strong>{enhanceMs.toFixed(0)} ms</strong>
              </div>
            </>
          )}
          {imageDims && (
            <>
              <div className="stat-divider" />
              <div className="stat-chip">
                {imageDims.w}×{imageDims.h} px
              </div>
            </>
          )}
          <label className="conf-filter" htmlFor="conf-checkbox">
            <input
              type="checkbox"
              id="conf-checkbox"
              checked={minConf >= 0.9}
              onChange={(e) => {
                setMinConf(e.target.checked ? 0.9 : 0);
                setHighlight(null);
              }}
            />
            tylko ≥ 90%
          </label>
        </div>
      )}

      {/* ── DEBUG ── */}
      {result?.debug && (
        <details className="debug-details">
          <summary className="debug-summary">
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
              <polyline points="9 18 15 12 9 6"/>
            </svg>
            DB debug
          </summary>
          <div className="debug-body">
            <div className="debug-row">
              <span className="debug-key">det shape:</span>
              <span className="debug-val">[{result.debug.detShape.join(", ")}]</span>
              <span className="debug-key">out:</span>
              <span className="debug-val">[{result.debug.detDims.join(", ")}]</span>
            </div>
            <div className="debug-row">
              <span className="debug-key">prob map:</span>
              <span className="debug-val">min={result.debug.probMin.toFixed(4)}</span>
              <span className="debug-val">max={result.debug.probMax.toFixed(4)}</span>
              <span className="debug-val">mean={result.debug.probMean.toFixed(4)}</span>
            </div>
            <div className="debug-row">
              <span className="debug-key">px &gt; 0.3:</span>
              <span className="debug-val">{result.debug.pixelsAboveThresh}/{result.debug.detDims.slice(-2).reduce((a,b) => a*b, 1)}</span>
            </div>
            <hr className="debug-sep" />
            <div className="debug-row">
              <span className="debug-key">components:</span>
              <span className="debug-val">raw={result.debug.rawComponents}</span>
              <span className="debug-val">minSize={result.debug.survivedMinSize}</span>
              <span className="debug-val">score={result.debug.survivedScore}</span>
              <span className="debug-val">final={result.debug.survivedFinalSize}</span>
            </div>
            <div className="debug-row">
              <span className="debug-key">scoreMaxRej:</span>
              <span className="debug-val">{result.debug.scoreMaxOfRejected.toFixed(4)}</span>
            </div>
            <hr className="debug-sep" />
            <div className="debug-row">
              <span className="debug-key">boxes:</span>
              <span className="debug-val">{result.debug.boxesDetected} → lines: {result.debug.linesRecognized}</span>
            </div>
            <div className="debug-row">
              <span className="debug-key">rec dims:</span>
              <span className="debug-val">[{result.debug.recOutputDims?.join(", ") ?? "—"}]</span>
              <span className="debug-key">vocab:</span>
              <span className="debug-val">{result.debug.recVocabSize ?? "—"}</span>
              <span className="debug-key">dict:</span>
              <span className="debug-val">{result.debug.dictSize}</span>
            </div>
            {result.debug.firstBoxWH && (
              <div className="debug-row">
                <span className="debug-key">box[0]:</span>
                <span className="debug-val">{result.debug.firstBoxWH[0]}×{result.debug.firstBoxWH[1]}px</span>
                {result.debug.firstBoxWH[1] > 80 && <span className="debug-warn">⚠ bardzo wysoki</span>}
                {result.debug.firstBoxWH[0] < 20 && <span className="debug-warn">⚠ bardzo wąski</span>}
              </div>
            )}
            {result.debug.firstBoxArgmax && (
              <div className="debug-row" style={{ flexWrap: "nowrap", overflow: "hidden" }}>
                <span className="debug-key">argmax:</span>
                <span className="debug-val" style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                  [{result.debug.firstBoxArgmax.join(", ")}]
                </span>
              </div>
            )}
          </div>
        </details>
      )}

      {/* ── RESULTS ── */}
      {imageUrl && result && (
        <div className="results">
          <div className="viewer-wrap">
            <ImageOverlay
              imageUrl={previewUrl ?? imageUrl}
              beforeUrl={previewUrl ? imageUrl : undefined}
              lines={visibleLines}
              highlightIdx={highlight}
              onHover={setHighlight}
            />
            {imageDims && (
              <div className="viewer-label">{imageDims.w}×{imageDims.h}</div>
            )}
          </div>
          <LinesList lines={visibleLines} highlightIdx={highlight} onHover={setHighlight} />
        </div>
      )}
    </div>
  );
}
