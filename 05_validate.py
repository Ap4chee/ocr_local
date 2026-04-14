"""
05_validate.py – walidacja wygenerowanego datasetu.

Sprawdza:
- liczbę linii w labels.tsv,
- istnienie plików PNG,
- poprawność alfabetu (każdy znak w labelu ∈ charset),
- wymiary H=32±2, W<=MAX_W,
- brak duplikatów nazw,
- rozkład długości labelek,
- zapisuje preview_grid.png (8x8) losowych próbek.
"""

from __future__ import annotations
import argparse
import random
from collections import Counter
from pathlib import Path
from typing import List, Tuple

from PIL import Image


def load_charset(path: Path) -> set[str]:
    with open(path, "r", encoding="utf-8") as f:
        return {line.rstrip("\n") for line in f if line.rstrip("\n")}


def validate(out_dir: Path, preview_n: int = 64, h_tol: int = 2, w_max: int = 512) -> int:
    img_dir = out_dir / "images"
    labels_path = out_dir / "labels.tsv"
    charset_path = out_dir / "charset.txt"

    if not labels_path.exists():
        print("BŁĄD: brak labels.tsv")
        return 1
    if not charset_path.exists():
        print("BŁĄD: brak charset.txt")
        return 1

    charset = load_charset(charset_path)
    print(f"Alfabet: {len(charset)} znaków")

    # Parse labels
    entries: List[Tuple[str, str]] = []
    with open(labels_path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 2:
                print(f"  [warn] linia {ln}: zły format ({len(parts)} kolumn)")
                continue
            entries.append((parts[0], parts[1]))
    print(f"Linie w labels.tsv: {len(entries)}")

    # Duplikaty nazw
    names = [e[0] for e in entries]
    dup = [n for n, c in Counter(names).items() if c > 1]
    if dup:
        print(f"  [warn] duplikaty nazw: {len(dup)} (pierwsze 5: {dup[:5]})")

    # Sprawdzenie alfabetu
    bad_char = 0
    unknown_chars: Counter[str] = Counter()
    for _, txt in entries:
        for ch in txt:
            if ch not in charset:
                unknown_chars[ch] += 1
        if any(ch not in charset for ch in txt):
            bad_char += 1
    if bad_char:
        print(f"  [err ] labels poza alfabetem: {bad_char}")
        print(f"         top nieznane znaki: {unknown_chars.most_common(10)}")
    else:
        print("  OK: wszystkie labele w alfabecie")

    # Wymiary + istnienie plików (sampluj 2000 dla szybkości; jeśli <2000 to wszystkie)
    sample_size = min(2000, len(entries))
    sampled = random.Random(0).sample(entries, sample_size)
    missing = 0
    bad_h = 0
    bad_w = 0
    heights = Counter()
    widths: List[int] = []
    for fname, _ in sampled:
        p = img_dir / fname
        if not p.exists():
            missing += 1
            continue
        with Image.open(p) as im:
            w, h = im.size
        heights[h] += 1
        widths.append(w)
        if abs(h - 32) > h_tol:
            bad_h += 1
        if w > w_max or w < 1:
            bad_w += 1
    print(f"Sprawdzone pliki (próbka {sample_size}): missing={missing}, bad_H={bad_h}, bad_W={bad_w}")
    print(f"  rozkład H: {dict(heights)}")
    if widths:
        print(f"  W: min={min(widths)}, max={max(widths)}, avg={sum(widths)/len(widths):.1f}")

    # Histogram długości labelek
    length_hist = Counter(len(t) for _, t in entries)
    print("Rozkład długości labelek:")
    for L in sorted(length_hist):
        bar = "#" * min(60, length_hist[L] // max(1, len(entries) // 600))
        print(f"  {L:>2}: {length_hist[L]:>6} {bar}")

    # Preview grid
    preview_path = out_dir / "preview_grid.png"
    if len(entries) >= preview_n:
        picks = random.Random(123).sample(entries, preview_n)
        thumbs: List[Image.Image] = []
        for fname, _ in picks:
            p = img_dir / fname
            if not p.exists():
                continue
            im = Image.open(p).convert("L")
            # Znormalizuj W do 256
            w, h = im.size
            if w > 256:
                im = im.crop((0, 0, 256, h))
            else:
                pad = Image.new("L", (256, h), 255)
                pad.paste(im, (0, 0))
                im = pad
            thumbs.append(im)
        if thumbs:
            cols = 8
            rows = (len(thumbs) + cols - 1) // cols
            grid = Image.new("L", (256 * cols, 32 * rows), 255)
            for i, t in enumerate(thumbs):
                r, c = divmod(i, cols)
                grid.paste(t, (c * 256, r * 32))
            grid.save(preview_path)
            print(f"Zapisano preview: {preview_path}")

    # Kryteria OK/FAIL
    errors = 0
    if bad_char: errors += 1
    if missing: errors += 1
    if bad_h: errors += 1
    if bad_w: errors += 1
    if dup: errors += 1
    print("=" * 60)
    print("WYNIK:", "OK" if errors == 0 else f"BŁĘDY: {errors}")
    return 0 if errors == 0 else 2


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, required=True, help="Katalog datasetu")
    ap.add_argument("--preview-n", type=int, default=64)
    args = ap.parse_args()
    raise SystemExit(validate(Path(args.out), preview_n=args.preview_n))


if __name__ == "__main__":
    main()
