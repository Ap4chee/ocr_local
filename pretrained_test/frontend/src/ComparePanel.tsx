import type { OcrLine, OcrResponse } from "./types";

interface Props {
  enhanced: OcrResponse;
  raw: OcrResponse;
  enhanceMs: number;
  onClose: () => void;
}

interface Diff {
  gained: string[];   // linie tylko w enhanced
  lost: string[];     // linie tylko w raw
}

function diffLines(enhanced: OcrLine[], raw: OcrLine[]): Diff {
  const eMap = new Map<string, number>();
  const rMap = new Map<string, number>();
  for (const l of enhanced) { const t = l.text.trim(); eMap.set(t, (eMap.get(t) ?? 0) + 1); }
  for (const l of raw)      { const t = l.text.trim(); rMap.set(t, (rMap.get(t) ?? 0) + 1); }

  const gained: string[] = [];
  const lost: string[] = [];
  for (const key of new Set([...eMap.keys(), ...rMap.keys()])) {
    const d = (eMap.get(key) ?? 0) - (rMap.get(key) ?? 0);
    for (let i = 0; i < Math.abs(d); i++) (d > 0 ? gained : lost).push(key);
  }
  return { gained, lost };
}

function Delta({ v, unit = "", invert = false }: { v: number; unit?: string; invert?: boolean }) {
  if (Math.abs(v) < 0.001) return <span className="cmp-neutral">–</span>;
  const positive = invert ? v < 0 : v > 0;
  const cls = positive ? "cmp-pos" : "cmp-neg";
  const sign = v > 0 ? "+" : "";
  return <span className={cls}>{sign}{v.toFixed(unit === "%" ? 1 : unit === "s" ? 2 : 0)}{unit}</span>;
}

export function ComparePanel({ enhanced, raw, enhanceMs, onClose }: Props) {
  const diff = diffLines(enhanced.lines, raw.lines);

  const dLines = enhanced.n_lines - raw.n_lines;
  const dConf  = (enhanced.mean_conf - raw.mean_conf) * 100;
  const dTime  = enhanced.time_s - raw.time_s;

  const verdict = (() => {
    const moreLines = dLines > 0;
    const betterConf = dConf > 0.5;
    const worseConf  = dConf < -0.5;
    if (moreLines && betterConf)
      return `Preprocessing pomógł: +${dLines} linii, pewność +${dConf.toFixed(1)}%.`;
    if (moreLines && worseConf)
      return `Preprocessing wykrył ${dLines > 0 ? "+" : ""}${dLines} dodatkowych linii, ale pewność rozpoznawania spadła o ${Math.abs(dConf).toFixed(1)}%. Detekcja lepsza, rekognicja gorsza — prawdopodobnie model rec nie był trenowany na CLAHE.`;
    if (moreLines)
      return `Preprocessing wykrył ${dLines > 0 ? "+" : ""}${dLines} dodatkowych linii bez istotnej zmiany pewności.`;
    if (!moreLines && worseConf)
      return `Preprocessing pogorszył oba wskaźniki (${dLines} linii, ${dConf.toFixed(1)}% pewności). Może obraz był już dobrej jakości.`;
    if (!moreLines && betterConf)
      return `Preprocessing poprawił pewność (+${dConf.toFixed(1)}%), choć liczba linii nie wzrosła.`;
    return `Preprocessing nie zmienił istotnie wyników (Δlinie=${dLines}, Δpewność=${dConf.toFixed(1)}%).`;
  })();

  return (
    <div className="cmp-panel" role="region" aria-label="Porównanie wyników OCR">
      <div className="cmp-header">
        <span className="cmp-title">Porównanie: oryginał vs. preprocessed</span>
        <button className="cmp-close" onClick={onClose} aria-label="Zamknij">✕</button>
      </div>

      <div className="cmp-verdict">{verdict}</div>

      {/* Stats table */}
      <div className="cmp-table">
        <div className="cmp-th" />
        <div className="cmp-th">Bez preproc.</div>
        <div className="cmp-th">Z preproc.</div>
        <div className="cmp-th">Zmiana</div>

        <div className="cmp-label">Wykryte linie</div>
        <div className="cmp-val">{raw.n_lines}</div>
        <div className="cmp-val accent">{enhanced.n_lines}</div>
        <div className="cmp-val"><Delta v={dLines} /></div>

        <div className="cmp-label">Średnia pewność</div>
        <div className="cmp-val">{(raw.mean_conf * 100).toFixed(1)}%</div>
        <div className="cmp-val accent">{(enhanced.mean_conf * 100).toFixed(1)}%</div>
        <div className="cmp-val"><Delta v={dConf} unit="%" /></div>

        <div className="cmp-label">Czas OCR</div>
        <div className="cmp-val">{raw.time_s.toFixed(2)} s</div>
        <div className="cmp-val accent">{enhanced.time_s.toFixed(2)} s</div>
        <div className="cmp-val"><Delta v={dTime} unit="s" invert /></div>

        {enhanceMs > 0 && <>
          <div className="cmp-label">Preprocessing</div>
          <div className="cmp-val">—</div>
          <div className="cmp-val accent">{enhanceMs.toFixed(0)} ms</div>
          <div className="cmp-val" />
        </>}
      </div>

      {/* Gained / lost lines */}
      {(diff.gained.length > 0 || diff.lost.length > 0) && (
        <div className="cmp-diff">
          {diff.gained.length > 0 && (
            <div className="cmp-diff-col">
              <div className="cmp-diff-hdr cmp-pos">
                +{diff.gained.length} nowych linii (tylko z preproc.)
              </div>
              <ul className="cmp-diff-list">
                {diff.gained.map((t, i) => (
                  <li key={i} className="cmp-diff-item gained">
                    <span className="cmp-diff-badge gained">+</span>
                    <span className="cmp-diff-text">{t || <em className="cmp-empty">(pusta linia)</em>}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}
          {diff.lost.length > 0 && (
            <div className="cmp-diff-col">
              <div className="cmp-diff-hdr cmp-neg">
                −{diff.lost.length} utraconych linii (tylko bez preproc.)
              </div>
              <ul className="cmp-diff-list">
                {diff.lost.map((t, i) => (
                  <li key={i} className="cmp-diff-item lost">
                    <span className="cmp-diff-badge lost">−</span>
                    <span className="cmp-diff-text">{t || <em className="cmp-empty">(pusta linia)</em>}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}

      {diff.gained.length === 0 && diff.lost.length === 0 && (
        <div className="cmp-nodiff">Tekst identyczny w obu przebiegach.</div>
      )}
    </div>
  );
}
