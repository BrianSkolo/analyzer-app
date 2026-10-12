"""Read artist, title, genre, tempo and key from a song's metadata tags.

Store downloads (Beatport, Traxsource, Bandcamp) tag the genre, and DJ tools
tag the key in Open Key ("11m"), Camelot ("6A") or plain ("Gm") notation.
"""

from __future__ import annotations

import re
from pathlib import Path

NOTES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
FLATS = {"Db": "C#", "Eb": "D#", "Gb": "F#", "Ab": "G#", "Bb": "A#"}

# Camelot 1A = A-flat minor, each number a fifth up. Open Key 1m = A minor
# and 1d = C major, also a fifth per step.
_CAMELOT_MINOR_START = NOTES.index("G#")
_CAMELOT_MAJOR_START = NOTES.index("B")
_OPENKEY_MINOR_START = NOTES.index("A")
_OPENKEY_MAJOR_START = NOTES.index("C")


def _fifths(start: int, number: int) -> str:
    return NOTES[(start + 7 * (number - 1)) % 12]


def parse_key(text: str | None) -> str | None:
    """'11m', '6A', 'Gm', 'G minor', 'Bbmaj' -> 'G minor' style, else None."""
    if not text:
        return None
    t = text.strip()
    m = re.fullmatch(r"(\d{1,2})\s*([mdAB])", t, re.I)
    if m:
        n, letter = int(m.group(1)), m.group(2)
        if not 1 <= n <= 12:
            return None
        if letter in "mM":
            return f"{_fifths(_OPENKEY_MINOR_START, n)} minor"
        if letter in "dD":
            return f"{_fifths(_OPENKEY_MAJOR_START, n)} major"
        if letter in "aA":
            return f"{_fifths(_CAMELOT_MINOR_START, n)} minor"
        return f"{_fifths(_CAMELOT_MAJOR_START, n)} major"
    m = re.fullmatch(r"([A-Ga-g])([#b♯♭]?)\s*(m|min|minor|maj|major)?", t)
    if not m:
        return None
    note = m.group(1).upper() + m.group(2).replace("♯", "#").replace("♭", "b")
    note = FLATS.get(note, note)
    if note not in NOTES:  # e.g. Cb, E#
        return None
    mode = (m.group(3) or "").lower()
    return f"{note} {'minor' if mode in ('m', 'min', 'minor') else 'major'}"


def _first(value) -> str | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        value = value[0] if value else None
    if value is None:
        return None
    if hasattr(value, "text"):  # ID3 frame
        return _first(value.text)
    if isinstance(value, bytes):
        value = value.decode("utf-8", "ignore")
    text = str(value).strip()
    return text or None


def read_tags(path: Path) -> dict:
    """Tags as {artist, title, genre, bpm, key}; missing ones are None."""
    out = {"artist": None, "title": None, "genre": None, "bpm": None, "key": None}
    try:
        import mutagen
        audio = mutagen.File(str(path))
    except Exception:
        return out
    if audio is None or audio.tags is None:
        return out
    tags = audio.tags

    def get(*names):
        for n in names:
            try:
                v = _first(tags.get(n))
            except (KeyError, ValueError, TypeError):
                v = None
            if v:
                return v
        return None

    out["artist"] = get("TPE1", "artist", "ARTIST", "\xa9ART")
    out["title"] = get("TIT2", "title", "TITLE", "\xa9nam")
    out["genre"] = get("TCON", "genre", "GENRE", "\xa9gen")
    bpm = get("TBPM", "bpm", "BPM", "tmpo")
    key = get("TKEY", "initialkey", "INITIALKEY", "KEY", "key",
              "----:com.apple.iTunes:initialkey")
    try:
        out["bpm"] = round(float(bpm), 1) if bpm else None
    except ValueError:
        pass
    out["key"] = parse_key(key)
    return out
