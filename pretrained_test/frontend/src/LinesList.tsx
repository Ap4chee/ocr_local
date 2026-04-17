import { useState } from "react";
import type { OcrLine } from "./types";

interface Props {
  lines: OcrLine[];
  highlightIdx: number | null;
  onHover: (idx: number | null) => void;
}

export function LinesList({ lines, highlightIdx, onHover }: Props) {
  const [copied, setCopied] = useState(false);

  if (lines.length === 0) {
    return (
      <div className="lines-panel">
        <div className="lines-header">
          <span className="lines-title">Wyniki OCR</span>
        </div>
        <div className="lines-empty">Brak rozpoznanych linii.</div>
      </div>
    );
  }

  const fullText = lines.map((l) => l.text).join("\n");

  async function handleCopy() {
    try {
      await navigator.clipboard.writeText(fullText);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (_) {
      // fallback — textarea trick
      const ta = document.createElement("textarea");
      ta.value = fullText;
      document.body.appendChild(ta);
      ta.select();
      document.execCommand("copy");
      document.body.removeChild(ta);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  }

  return (
    <>
      <div className="lines-panel" id="lines-panel">
        <div className="lines-header">
          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <span className="lines-title">Wyniki OCR</span>
            <span className="lines-count">{lines.length}</span>
          </div>
          <button
            id="btn-copy"
            className={`btn${copied ? " btn-accent" : ""}`}
            onClick={handleCopy}
            title="Kopiuj cały rozpoznany tekst"
          >
            {copied ? (
              <>
                <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                  <polyline points="20 6 9 17 4 12"/>
                </svg>
                Skopiowano!
              </>
            ) : (
              <>
                <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/>
                </svg>
                Kopiuj tekst
              </>
            )}
          </button>
        </div>

        <div className="lines-scroll">
          <ol>
            {lines.map((l, i) => (
              <li
                key={i}
                className={`line-item${i === highlightIdx ? " active" : ""}`}
                onMouseEnter={() => onHover(i)}
                onMouseLeave={() => onHover(null)}
                id={`line-${i}`}
              >
                <span className="line-num">{i + 1}</span>
                <span className={`conf-badge conf-${confBucket(l.conf)}`}>
                  {(l.conf * 100).toFixed(0)}%
                </span>
                <span className="line-text">{l.text}</span>
              </li>
            ))}
          </ol>
        </div>
      </div>

      {copied && (
        <div className="toast" role="status" aria-live="polite">
          ✓ Tekst skopiowany do schowka
        </div>
      )}
    </>
  );
}

function confBucket(c: number): "hi" | "mid" | "lo" {
  if (c >= 0.9) return "hi";
  if (c >= 0.7) return "mid";
  return "lo";
}
