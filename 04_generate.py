"""
04_generate.py – generator datasetu 100k linii z trdg + randomizowane augmentacje.

Użycie w Colabie:
    !python 04_generate.py --n 100000 --out /content/drive/MyDrive/ocr/dataset_v1

Zawiera:
- Filtr fontów które nie renderują PL diakrytyków (przez porównanie metryk).
- Per-próbka randomizację: font, tło, skew, distortion, blur.
- Post-resize do H=32 zachowujący aspect ratio, grayscale.
- Resume-on-crash: skip próbek które już istnieją; labels.tsv append.
"""

from __future__ import annotations
import argparse
import json
import os
import random
import sys
import hashlib
from pathlib import Path
from typing import List, Tuple

from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

# trdg API (pip install trdg)
try:
    from trdg.generators import GeneratorFromStrings
except ImportError as e:
    print("Zainstaluj trdg:  !pip install trdg", file=sys.stderr)
    raise

# Lokalne moduły – w Colabie trzymaj je w tym samym katalogu.
# Importy przez exec, bo nazwy plików zaczynają się od cyfr:
_here = Path(__file__).parent
exec((_here / "02_charset.py").read_text(encoding="utf-8"), globals())   # ALPHABET, save_charset
exec((_here / "03_text_sampler.py").read_text(encoding="utf-8"), globals())  # sample_text

TARGET_H = 32
MAX_W = 512
PL_TEST_STRING = "ąćęłńóśźżĄĆĘŁŃÓŚŹŻ"


# ----------------------------------------------------------------------------
# Filtr fontów PL
# ----------------------------------------------------------------------------

def font_supports_polish(font_path: str) -> bool:
    """
    Renderuje PL diakrytyki i porównuje bounding box z placeholderem .notdef.
    Jeśli szerokości są podejrzanie równe (wszystkie glify = .notdef), font odrzucamy.
    """
    try:
        font = ImageFont.truetype(font_path, size=32)
    except Exception:
        return False
    img = Image.new("L", (10, 10), 255)
    draw = ImageDraw.Draw(img)
    widths = []
    for ch in PL_TEST_STRING:
        bbox = draw.textbbox((0, 0), ch, font=font)
        widths.append(bbox[2] - bbox[0])
    # .notdef zwykle daje identyczną szerokość dla wszystkich znaków
    return len(set(widths)) > 3 and all(w > 0 for w in widths)


def discover_fonts(search_dirs: List[Path]) -> List[str]:
    """Znajdź wszystkie .ttf/.otf i przefiltruj pod kątem PL."""
    candidates: List[str] = []
    for d in search_dirs:
        if not d.exists():
            continue
        for ext in ("*.ttf", "*.otf", "*.TTF", "*.OTF"):
            candidates.extend(str(p) for p in d.rglob(ext))
    candidates = sorted(set(candidates))
    good = [f for f in candidates if font_supports_polish(f)]
    return good


# ----------------------------------------------------------------------------
# Postprocessing obrazu
# ----------------------------------------------------------------------------

def to_gray_h32(img: Image.Image) -> Image.Image:
    """Konwersja do grayscale + resize do H=32 zachowując aspect ratio, clamp W<=MAX_W."""
    if img.mode != "L":
        img = img.convert("L")
    w, h = img.size
    if h == 0:
        raise ValueError("Pusty obraz")
    new_w = max(1, int(round(w * TARGET_H / h)))
    new_w = min(new_w, MAX_W)
    return img.resize((new_w, TARGET_H), Image.BILINEAR)


# ----------------------------------------------------------------------------
# Główna pętla generacji
# ----------------------------------------------------------------------------

def generate(n: int, out_dir: Path, seed: int = 42,
             font_dirs: List[Path] | None = None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    img_dir = out_dir / "images"
    img_dir.mkdir(exist_ok=True)
    labels_path = out_dir / "labels.tsv"
    charset_path = out_dir / "charset.txt"
    meta_path = out_dir / "meta.json"

    # Zapis alfabetu
    save_charset(charset_path)  # type: ignore[name-defined]

    # Wyszukanie fontów
    if font_dirs is None:
        font_dirs = [
            Path("/usr/share/fonts"),
            Path("/content/fonts"),
            Path.home() / ".fonts",
        ]
    fonts = discover_fonts(font_dirs)
    if not fonts:
        raise RuntimeError(
            "Nie znaleziono fontów z obsługą PL. "
            "Zainstaluj: apt-get install -y fonts-dejavu fonts-liberation "
            "fonts-noto fonts-roboto fonts-open-sans"
        )
    print(f"Fontów z obsługą PL: {len(fonts)}")

    # Resume: sprawdź które pliki już istnieją
    existing = set(p.name for p in img_dir.iterdir()) if any(img_dir.iterdir()) else set()
    print(f"Istniejących próbek: {len(existing)}")

    # Pre-generuj teksty deterministycznie
    rng = random.Random(seed)
    texts: List[str] = [sample_text(rng) for _ in range(n)]  # type: ignore[name-defined]

    # trdg GeneratorFromStrings: tworzymy go jednorazowo na całą listę, z losowymi
    # parametrami dla tła/skew/blur. Niestety trdg nie mikształci "per próbka":
    # stąd rozbijamy na małe paczki po ~500 z różnymi ustawieniami.
    CHUNK = 500
    written = 0
    skipped = 0

    with open(labels_path, "a", encoding="utf-8", newline="\n") as lf:
        pbar = tqdm(total=n, desc="generate", unit="img")
        for start in range(0, n, CHUNK):
            chunk_texts = texts[start:start + CHUNK]
            # Losuj parametry dla tej paczki
            chunk_rng = random.Random(seed + start)
            params = dict(
                count=len(chunk_texts),
                fonts=[chunk_rng.choice(fonts)],
                size=48,                                  # trdg size = wysokość tekstu
                background_type=chunk_rng.choice([0, 1, 2, 3]),  # 0=gaussian,1=plain,2=quasi,3=picture
                skewing_angle=chunk_rng.randint(0, 3),
                random_skew=True,
                blur=chunk_rng.randint(0, 2),
                random_blur=True,
                distorsion_type=chunk_rng.choice([0, 0, 1, 2]),  # częściej brak
                distorsion_orientation=chunk_rng.randint(0, 2),
                text_color=chunk_rng.choice(["#000000", "#222222", "#111133"]),
                margins=(2, 4, 2, 4),
                fit=True,
                character_spacing=chunk_rng.randint(0, 2),
            )
            try:
                gen = GeneratorFromStrings(strings=chunk_texts, **params)
            except TypeError:
                # Starsze wersje trdg mają nieco inne argumenty – fallback minimalny
                gen = GeneratorFromStrings(strings=chunk_texts, fonts=params["fonts"],
                                           size=params["size"], fit=True)

            for local_i, (img, txt) in enumerate(gen):
                global_i = start + local_i
                fname = f"{global_i:07d}.png"
                if fname in existing:
                    skipped += 1
                    pbar.update(1)
                    continue
                if img is None or not txt:
                    pbar.update(1)
                    continue
                try:
                    img32 = to_gray_h32(img)
                    img32.save(img_dir / fname, optimize=True)
                    # Zabezpieczenie: labels ze spacjami na skraju trim
                    clean = txt.strip()
                    if not clean:
                        (img_dir / fname).unlink(missing_ok=True)
                        pbar.update(1)
                        continue
                    lf.write(f"{fname}\t{clean}\n")
                    if (written + 1) % 1000 == 0:
                        lf.flush()
                        os.fsync(lf.fileno())
                    written += 1
                except Exception as e:
                    print(f"\n[warn] {fname}: {e}", file=sys.stderr)
                pbar.update(1)
        pbar.close()

    # meta.json
    meta = {
        "n_requested": n,
        "n_written": written,
        "n_skipped_existing": skipped,
        "target_h": TARGET_H,
        "max_w": MAX_W,
        "alphabet_len": len(ALPHABET),  # type: ignore[name-defined]
        "alphabet_hash": hashlib.sha1(ALPHABET.encode("utf-8")).hexdigest(),  # type: ignore[name-defined]
        "seed": seed,
        "n_fonts_used": len(fonts),
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nGotowe. Zapisano {written} nowych, pominięto {skipped}. Meta: {meta_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100_000)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--font-dir", action="append", default=None,
                    help="Dodatkowe katalogi z fontami (można podać wielokrotnie)")
    args = ap.parse_args()

    font_dirs = [Path(p) for p in args.font_dir] if args.font_dir else None
    generate(n=args.n, out_dir=Path(args.out), seed=args.seed, font_dirs=font_dirs)


if __name__ == "__main__":
    main()
