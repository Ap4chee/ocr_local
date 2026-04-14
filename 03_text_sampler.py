"""
03_text_sampler.py – generator losowych stringów do treningu CRNN.

Strategia: losowe ciągi z alfabetu z ważonym próbkowaniem po kategoriach
(litery 70%, cyfry 15%, interpunkcja 12%, symbole+spacja 3%) zamiast korpusu
słownikowego. Dzięki temu model lepiej generalizuje na OOV (NIP, kody, kwoty).

Długość linii: LogNormal(μ=1.8, σ=0.6), clamp [1, 25].
"""

from __future__ import annotations
import random
from typing import Iterator, List

# Import alfabetu z sąsiedniego pliku – w Colabie wklejasz oba do tego samego katalogu.
from importlib import import_module
_charset = import_module("02_charset") if False else None  # placeholder; patrz niżej

# W Pythonie nie można importować z nazwy zaczynającej się od cyfry przez `import`.
# W Colabie użyj: `exec(open("02_charset.py").read())` albo zmień nazwę pliku.
# Poniżej duplikuję definicje kategorii, żeby plik działał samodzielnie.

_DIGITS = "0123456789"
_LATIN_LOWER = "abcdefghijklmnopqrstuvwxyz"
_LATIN_UPPER = _LATIN_LOWER.upper()
_PL_LOWER = "ąćęłńóśźż"
_PL_UPPER = _PL_LOWER.upper()
_PUNCT = ".,;:!?\"'()[]{}<>/\\|@#$%^&*-+=_~"
_EXTRA = "€§°"
_SPACE = " "

LETTERS = _LATIN_LOWER + _LATIN_UPPER + _PL_LOWER + _PL_UPPER
DIGITS = _DIGITS
PUNCT = _PUNCT
SYMBOLS = _EXTRA + _SPACE

CATEGORY_WEIGHTS = {
    "letters": 0.70,
    "digits": 0.15,
    "punct": 0.12,
    "symbols": 0.03,
}
_CATS = list(CATEGORY_WEIGHTS.keys())
_WEIGHTS = list(CATEGORY_WEIGHTS.values())
_POOLS = {
    "letters": LETTERS,
    "digits": DIGITS,
    "punct": PUNCT,
    "symbols": SYMBOLS,
}


def sample_length(rng: random.Random, mu: float = 1.8, sigma: float = 0.6,
                  lo: int = 1, hi: int = 25) -> int:
    """Długość linii ~ LogNormal, clamp do [lo, hi]."""
    # random.lognormvariate ma parametry (mu, sigma) loga naturalnego rozkładu
    n = int(round(rng.lognormvariate(mu, sigma)))
    return max(lo, min(hi, n))


def sample_text(rng: random.Random, length: int | None = None) -> str:
    """Wygeneruj jeden losowy string zgodnie z rozkładami."""
    if length is None:
        length = sample_length(rng)
    out: List[str] = []
    for _ in range(length):
        cat = rng.choices(_CATS, weights=_WEIGHTS, k=1)[0]
        pool = _POOLS[cat]
        out.append(rng.choice(pool))
    # Usuń wiodące/końcowe spacje żeby trdg ich nie zjadł
    s = "".join(out).strip(" ")
    # Jeżeli po stripie puste – dorzuć literę
    if not s:
        s = rng.choice(LETTERS)
    return s


def text_stream(n: int, seed: int = 42) -> Iterator[str]:
    """Deterministyczny strumień n próbek."""
    rng = random.Random(seed)
    for _ in range(n):
        yield sample_text(rng)


if __name__ == "__main__":
    rng = random.Random(0)
    print("Przykładowe próbki:")
    for _ in range(20):
        s = sample_text(rng)
        print(f"  [{len(s):>2}] {s!r}")

    # Histogram długości
    from collections import Counter
    rng = random.Random(1)
    lens = Counter(len(sample_text(rng)) for _ in range(10_000))
    print("\nHistogram długości (10k próbek):")
    for L in sorted(lens):
        bar = "#" * (lens[L] // 50)
        print(f"  {L:>2}: {lens[L]:>5} {bar}")
