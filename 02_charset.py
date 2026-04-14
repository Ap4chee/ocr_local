"""
02_charset.py – definicja alfabetu dla CRNN+CTC.

Zasady:
- blank CTC = indeks 0 (poza alfabetem, nie zapisywany w charset.txt).
- Każdy znak alfabetu ma indeks od 1 (= numer linii w charset.txt).
- Encode/decode: str <-> List[int].
"""

from __future__ import annotations
from pathlib import Path
from typing import List

# -- Definicja alfabetu --------------------------------------------------------

_DIGITS = "0123456789"
_LATIN_LOWER = "abcdefghijklmnopqrstuvwxyz"
_LATIN_UPPER = _LATIN_LOWER.upper()
_PL_LOWER = "ąćęłńóśźż"
_PL_UPPER = _PL_LOWER.upper()
# Pełna interpunkcja + symbole (uwaga: backslash zachowany jako pojedynczy znak w stringu)
_PUNCT = ".,;:!?\"'()[]{}<>/\\|@#$%^&*-+=_~"
_EXTRA = "€§°"
_SPACE = " "

ALPHABET: str = (
    _LATIN_LOWER + _LATIN_UPPER
    + _PL_LOWER + _PL_UPPER
    + _DIGITS
    + _PUNCT
    + _EXTRA
    + _SPACE
)

# Walidacja: bez duplikatów
assert len(ALPHABET) == len(set(ALPHABET)), "Duplikat w ALPHABET"

CHAR_TO_IDX = {ch: i + 1 for i, ch in enumerate(ALPHABET)}  # 0 zarezerwowane na blank
IDX_TO_CHAR = {i + 1: ch for i, ch in enumerate(ALPHABET)}
BLANK_IDX = 0
NUM_CLASSES = len(ALPHABET) + 1  # +1 dla blank


def encode(text: str) -> List[int]:
    """Tekst -> lista indeksów CTC (bez blank). Rzuca KeyError dla znaku spoza alfabetu."""
    return [CHAR_TO_IDX[ch] for ch in text]


def decode(ids: List[int]) -> str:
    """Lista indeksów -> tekst. Pomija blank i nieznane."""
    return "".join(IDX_TO_CHAR[i] for i in ids if i in IDX_TO_CHAR)


def save_charset(path: str | Path) -> None:
    """Zapis alfabetu: jeden znak per linia, indeks = nr linii + 1."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for ch in ALPHABET:
            f.write(ch + "\n")


def load_charset(path: str | Path) -> str:
    """Wczytaj alfabet z pliku (odwrotność save_charset)."""
    with open(path, "r", encoding="utf-8") as f:
        return "".join(line.rstrip("\n") for line in f if line.rstrip("\n"))


if __name__ == "__main__":
    print(f"Liczba znaków: {len(ALPHABET)}")
    print(f"NUM_CLASSES (z blank): {NUM_CLASSES}")
    print(f"Alfabet: {ALPHABET!r}")
    sample = "Faktura nr 123/2026, kwota 1 234,56 zł – pozdro Żółć!"
    try:
        ids = encode(sample)
        back = decode(ids)
        print(f"Test encode/decode OK: {back!r}")
    except KeyError as e:
        print(f"Znak poza alfabetem: {e}")
