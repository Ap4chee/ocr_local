# Projekt: OCR w przeglądarce (CPU/GPU-only, bez backendu)

## Cel
System OCR drukowanego tekstu (dokumenty, faktury, paragony) działający w przeglądarce bez serwera.
Głównie język polski; docelowo też dokumenty dwujęzyczne PL/EN.

## Stan obecny
Faza **pretrained baseline / POC**: gotowy PaddleOCR **PP-OCRv5** (detekcja + rozpoznawanie) uruchomiony w Web Workerze przez `onnxruntime-web` z WebGPU (+ fallback WASM). Celem tej fazy jest zmierzyć jakość i latencję „out of the box" na polskich dokumentach, zanim zaczniemy trenować własny model.

Kod żywy: [pretrained_test/frontend/](pretrained_test/frontend/) — Vite + TypeScript + React, modele w `public/models/` (`det.onnx`, `rec.onnx`, `latin_dict.txt`).

## Pipeline (bieżący — PP-OCRv5)
Obraz → **DB detekcja** (`det.onnx`) → poly-boxy → perspektywiczne kadrowanie (affine przez `ctx.setTransform`) → **rec CRNN+CTC** (`rec.onnx`) → greedy CTC decode → linie tekstu + konfidencja.

Cała inferencja siedzi w [src/ocr/worker.ts](pretrained_test/frontend/src/ocr/worker.ts); pre/post w [preprocess.ts](pretrained_test/frontend/src/ocr/preprocess.ts) i [postprocess.ts](pretrained_test/frontend/src/ocr/postprocess.ts); komunikacja UI↔worker przez typowany [protocol.ts](pretrained_test/frontend/src/ocr/protocol.ts).

## Pipeline (docelowy — własny model, odłożone)
Zdjęcie → **CRAFT** (gotowy ONNX, detekcja linii) → kadrowanie linii → **własny CRNN** (CNN + BiGRU + CTC) trenowany na syntetycznych danych `trdg` → tekst. Szczegóły poniżej w sekcji „Plan docelowy".

## Stack
- **Frontend (bieżący)**: Vite + **TypeScript + React 18**, `onnxruntime-web@1.24.3` (WebGPU EP → fallback WASM), Web Workers, Canvas / OffscreenCanvas.
- **Cross-origin isolation**: `COOP: same-origin` + `COEP: require-corp` (wymagane przez threaded WASM / SharedArrayBuffer) — plugin w [vite.config.ts](pretrained_test/frontend/vite.config.ts).
- **WASM paths**: `ort.env.wasm.wasmPaths` ustawione na CDN (`cdn.jsdelivr.net/npm/onnxruntime-web@1.24.3/dist/`) — unikamy kopiowania `.mjs` loadera do `public/`.
- **Trening (gdy wrócimy do własnego CRNN)**: Python, PyTorch, Google Colab, `trdg` (syntetyki PL).
- **Eksport**: `torch.onnx` → `onnxsim` → kwantyzacja INT8 (`onnxruntime.quantization`).

## Profil użytkownika
- Średnie doświadczenie ML, trenował już modele w PyTorch.
- Zna TypeScript i podstawy Canvas API.

## Zasady współpracy
- **Kompletny, działający kod** — nie szkielety z TODO, `pass`, `...`.
- Frontend w TypeScript; React jest OK w tej fazie POC (jest już w [package.json](pretrained_test/frontend/package.json)). Nie dodawaj kolejnych frameworków (stan app-level tylko `useState`/`useRef`).
- Logika inferencji ma zostać **poza React** — w workerze i czystych modułach `src/ocr/*`, bez zależności od DOM/UI.
- Python (jeśli wrócimy do treningu) piszesz **pod Google Colab** (mount Drive, `!pip install`, `tqdm`, zapisy pośrednie).
- Optymalizuj pod CPU i przeglądarkę: **WebGPU (gdy dostępne) + WASM/SIMD + Web Workers + INT8**. Preferuj operatory wspierane przez `onnxruntime-web` i kwantyzację (Conv/GRU, unikaj custom ops).
- Gdy są alternatywy: krótkie porównanie (2–3 zdania) + rekomendacja.
- Zawsze wskazuj pułapki (dynamiczne osie ONNX, CTC dekodowanie, kwantyzacja, fonty bez diakrytyków, słownik PP-OCR vs polskie znaki).

## Kluczowe parametry techniczne (PP-OCRv5 w browserze)

### Detekcja (DB)
- Wejście `[1,3,H,W]`, RGB, normalizacja ImageNet (mean `0.485/0.456/0.406`, std `0.229/0.224/0.225`).
- Resize: short-side = 736, max side = 1280, H/W zaokrąglane do wielokrotności 32.
- Post: binarize (`thresh=0.3`) → 4-connected flood-fill → convex hull (Andrew) → `minAreaRect` (rotating calipers) → `unclip` (`ratio=1.5`, PaddleOCR-style) → `orderPoints` TL/TR/BR/BL → sort wierszami (`Δcy < 10 px` traktowane jako ten sam wiersz).

### Rozpoznawanie (rec, CTC)
- Wejście `[1,3,48,W]`, W ≤ 320, normalizacja `(x/255 - 0.5)/0.5`.
- Perspektywa: affine z 3 punktów (TL, TR, BL) przez `setTransform`; degenerate (det≈0) → axis-aligned bbox.
- **Blank=0** (PaddleOCR), znaki od indeksu 1; słownik z `public/models/latin_dict.txt`.
- Greedy CTC: argmax per-t, collapse powtórzeń, drop blank. Konfidencja = średnia prob. na non-blank tokenach (zakładamy softmax na wyjściu `rec.onnx`).

## Plan docelowy (własny CRNN, poza POC-iem)
Zachowane jako referencja — wraca do gry po ocenie PP-OCRv5.

### CRNN — wejście
- **Grayscale, H=32 px stałe, W zmienne**; pad W do wielokrotności 4 w batchu.
- ONNX eksport z **dynamic axis** na batch i width.

### Alfabet (~130)
- PL + EN litery: `a-zA-Z` + `ąćęłńóśźżĄĆĘŁŃÓŚŹŻ`
- Cyfry `0-9`
- Interpunkcja i symbole: `.,;:!?"'()[]{}<>/\|@#$%^&*-+=_~` + spacja + `€§°`
- **CTC blank = index 0**, znaki od indeksu 1.

### Dataset v1 (gdy wrócimy do treningu)
- **100 000 próbek** (95k train / 5k val).
- Generowane przez `trdg.GeneratorFromStrings` z **losowych ciągów znaków** (nie korpus słownikowy — lepsza generalizacja na NIP/kody/kwoty).
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
0. **Etap 0 — POC z PP-OCRv5** ← aktualny. Cel: zmierzyć jakość/latencję pretrained OCR w przeglądarce na polskich dokumentach i ustalić, czy trening własnego modelu jest w ogóle potrzebny.
1. Etap 1 — generowanie syntetycznych danych (`trdg`, 100k linii PL). Plan: `~/.claude/plans/expressive-discovering-fox.md`.
2. Etap 2 — Dataset/DataLoader PyTorch, architektura CRNN, pętla treningowa z CTCLoss.
3. Etap 3 — eksport ONNX + onnxsim + kwantyzacja INT8.
4. Etap 4 — frontend Vite/TS: pipeline CRAFT→CRNN w Web Worker (docelowo zastępuje PP-OCRv5 w tym samym workerze).

## Pułapki (do pamiętania w całym projekcie)

### PP-OCRv5 w browserze (bieżące)
- **`latin_dict.txt` a polskie diakrytyki**: słownik „latin" PP-OCRv5 może nie zawierać pełnego `ąćęłńóśźż` — przed oceną jakości sprawdź pokrycie, inaczej te znaki będą wycinane na etapie dekodowania.
- **Backend label vs realny EP**: [worker.ts:53](pretrained_test/frontend/src/ocr/worker.ts#L53) ustawia `backend` heurystycznie (`hasWebGPU ? "webgpu" : "wasm"`); jeśli WebGPU padnie wewnątrz `InferenceSession.create` i ORT po cichu spadnie na WASM, UI pokaże błędnie „WebGPU".
- **Softmax vs logits w `rec.onnx`**: `ctcDecode` uśrednia `bestVal` jako prob; jeśli model wypluwa logits, wartość „conf" nadal jest monotoniczna, ale nie jest prawdziwym prawdopodobieństwem.
- **Affine degenerate**: bardzo cienkie/poziome boxy → `det≈0` → fallback do axis-aligned bbox (już obsłużone, ale pamiętać przy debugowaniu artefaktów).
- **COOP/COEP wymagane** do threaded WASM — bez nich ORT cichutko spadnie do 1 wątku (a WebGPU może odmówić współpracy na niektórych przeglądarkach).
- **`ort.env.wasm.wasmPaths`**: musi wskazywać na dokładnie tę samą wersję co `onnxruntime-web` z `package.json`, inaczej runtime/loader się rozjadą.

### CRNN/własny trening (docelowe)
- **Fonty bez PL diakrytyków**: przed generacją przefiltruj fonty renderując testowy string `ąćęłńóśźż`.
- **trdg `size` ≠ wysokość obrazu** — to wysokość tekstu, margines dochodzi ekstra → post-resize do H=32.
- **TSV a Unicode**: zapis `encoding='utf-8'`, separator **tab** (przecinek jest w labelach).
- **CTC dekodowanie**: blank=0, potem collapse powtórzeń, potem usuwanie blanków. Łatwo pomylić kolejność.
- **Dynamic axes w ONNX**: onnxruntime-web wymaga jawnych nazw osi dynamicznych; kwantyzacja INT8 może nie pokryć wszystkich operatorów BiGRU — sprawdzić `per_channel=False, reduce_range=True` dla WASM.
