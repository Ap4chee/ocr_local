# Projekt: OCR w przeglądarce (CPU-only)

## Cel
System OCR drukowanego tekstu (dokumenty, faktury) działający w przeglądarce bez backendu.
Głównie język polski; docelowo też dokumenty dwujęzyczne PL/EN.

## Pipeline
Zdjęcie → **CRAFT** (gotowy ONNX, detekcja linii) → kadrowanie linii → **CRNN** (CNN + BiGRU + CTC) → tekst.

## Stack
- **Trening**: Python, PyTorch, Google Colab, `trdg` (syntetyczne dane po polsku).
- **Eksport**: `torch.onnx` → `onnxsim` → kwantyzacja INT8 (`onnxruntime.quantization`).
- **Inference**: Vite + TypeScript (bez React/Vue), `onnxruntime-web` (WASM + SIMD), Canvas API, Web Workers.

## Profil użytkownika
- Średnie doświadczenie ML, trenował już modele w PyTorch.
- Zna TypeScript i podstawy Canvas API.

## Zasady współpracy
- **Kompletny, działający kod** – nie szkielety z TODO, `pass`, `...`.
- Python piszesz **pod Google Colab** (mount Drive, `!pip install`, `tqdm`, zapisy pośrednie).
- Frontend w **TypeScript, bez frameworków**. DOM API, Web Workers, onnxruntime-web.
- Optymalizuj pod CPU i przeglądarkę: **WASM + SIMD + Web Workers + INT8**. Preferuj operatory wspierane przez onnxruntime-web i kwantyzację (Conv/GRU, unikaj custom ops).
- Gdy są alternatywy: krótkie porównanie (2–3 zdania) + rekomendacja.
- Zawsze wskazuj pułapki (dynamiczne osie ONNX, CTC dekodowanie, kwantyzacja, fonty bez diakrytyków itp.).

## Kluczowe parametry techniczne

### CRNN – wejście
- **Grayscale, H=32 px stałe, W zmienne**; pad W do wielokrotności 4 w batchu.
- ONNX eksport z **dynamic axis** na batch i width.

### Alfabet (~130)
- PL + EN litery: `a-zA-Z` + `ąćęłńóśźżĄĆĘŁŃÓŚŹŻ`
- Cyfry `0-9`
- Interpunkcja i symbole: `.,;:!?"'()[]{}<>/\|@#$%^&*-+=_~` + spacja + `€§°`
- **CTC blank = index 0**, znaki od indeksu 1.

### Dataset v1 (Etap 1)
- **100 000 próbek** (95k train / 5k val).
- Generowane przez `trdg.GeneratorFromStrings` z **losowych ciągów znaków** (nie korpus słownikowy – lepsza generalizacja na NIP/kody/kwoty).
- Długość linii 1–25 znaków, rozkład **LogNormal(μ=1.8, σ=0.6)**.
- Lokalizacja na Drive:
  ```
  /content/drive/MyDrive/ocr/dataset_v1/
    images/        # H≈32, W zmienne, grayscale PNG
    labels.tsv     # "filename\tTEXT\n" UTF-8
    charset.txt    # 1 znak / linia
    meta.json      # parametry generatora
  ```

## Etapy projektu
1. **Etap 1 – generowanie danych** ← aktualny. Plan: `~/.claude/plans/expressive-discovering-fox.md`.
2. Etap 2 – Dataset/DataLoader PyTorch, architektura CRNN, pętla treningowa z CTCLoss.
3. Etap 3 – eksport ONNX + onnxsim + kwantyzacja INT8.
4. Etap 4 – frontend Vite/TS: pipeline CRAFT→CRNN w Web Worker, Canvas API.

## Pułapki (do pamiętania w całym projekcie)
- **Fonty bez PL diakrytyków**: przed generacją przefiltruj fonty renderując testowy string `ąćęłńóśźż`.
- **trdg `size` ≠ wysokość obrazu** – to wysokość tekstu, margines dochodzi ekstra → post-resize do H=32.
- **TSV a Unicode**: zapis `encoding='utf-8'`, separator **tab** (przecinek jest w labelach).
- **CTC dekodowanie**: blank=0, potem collapse powtórzeń, potem usuwanie blanków. Łatwo pomylić kolejność.
- **Dynamic axes w ONNX**: onnxruntime-web wymaga jawnych nazw osi dynamicznych; kwantyzacja INT8 może nie pokryć wszystkich operatorów BiGRU – sprawdzić `per_channel=False, reduce_range=True` dla WASM.
