"""Measure one song and write the analysis JSON from the Mix Analysis Spec.

Every later step (genre profiles, scoring, the app) reads only this JSON, so
this is the one place that touches audio.

    python -m analyzer_app.analyze my_song.wav -o my_song.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pyloudnorm
import soundfile as sf
from scipy import signal

from analyzer_app.tags import read_tags

SCHEMA_VERSION = 1
ANALYZER_VERSION = "0.1.0"
SAMPLE_RATE = 44100
NORMALIZE_TO_LUFS = -14.0

BANDS = {"sub": (20, 60), "bass": (60, 120), "low_mid": (120, 250),
         "mid_low": (250, 500), "mid": (500, 1000), "upper_mid": (1000, 2000),
         "presence": (2000, 4000), "high": (4000, 8000), "air": (8000, 12000),
         "top": (12000, 20000)}
THIRD_OCTAVE_HZ = [20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250,
                   315, 400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500,
                   3150, 4000, 5000, 6300, 8000, 10000, 12500, 16000, 20000]
LOSSY_EXTENSIONS = {".mp3", ".m4a", ".aac", ".ogg", ".opus", ".wma"}
AUDIO_EXTENSIONS = LOSSY_EXTENSIONS | {".wav", ".aif", ".aiff", ".flac"}

KEY_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
# Krumhansl-Schmuckler key profiles.
MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19,
                          2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75,
                          3.98, 2.69, 3.34, 3.17])

SILENCE_LUFS = -70.0


class AnalysisError(Exception):
    pass


def file_sha1(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_audio(path: Path) -> np.ndarray:
    """Audio as float64 (samples, channels) at 44.1 kHz. At most 2 channels."""
    try:
        data, sr = sf.read(str(path), always_2d=True, dtype="float64")
    except Exception:
        try:
            import librosa
            y, sr = librosa.load(str(path), sr=None, mono=False)
        except Exception as e:
            raise AnalysisError(f"{path.name}: can't read this audio file ({e})") from e
        data = np.atleast_2d(y).T.astype(np.float64)
    if data.shape[1] > 2:
        data = data[:, :2]
    if sr != SAMPLE_RATE:
        from math import gcd
        g = gcd(int(sr), SAMPLE_RATE)
        data = signal.resample_poly(data, SAMPLE_RATE // g, int(sr) // g, axis=0)
    if data.shape[0] < SAMPLE_RATE * 5:
        raise AnalysisError(f"{path.name}: shorter than 5 seconds")
    return data


def _db(x: float, floor: float = 1e-12) -> float:
    return float(10 * np.log10(max(x, floor)))


def k_weighted(data: np.ndarray) -> np.ndarray:
    meter = pyloudnorm.Meter(SAMPLE_RATE)
    out = data.copy()
    for stage in meter._filters.values():
        for ch in range(out.shape[1]):
            out[:, ch] = stage.apply_filter(out[:, ch])
    return out


def short_term_curve(weighted: np.ndarray, hop_s: float = 1.0,
                     window_s: float = 3.0) -> np.ndarray:
    """Short-term loudness (3 s window) every hop_s seconds, in LUFS."""
    win, hop = int(window_s * SAMPLE_RATE), int(hop_s * SAMPLE_RATE)
    power = np.sum(weighted ** 2, axis=1)  # channel gains are 1.0 for L/R
    csum = np.concatenate([[0.0], np.cumsum(power)])
    starts = np.arange(0, len(power) - win + 1, hop)
    ms = (csum[starts + win] - csum[starts]) / win
    return -0.691 + 10 * np.log10(np.maximum(ms, 1e-12))


def loudness_range(short_term: np.ndarray) -> float:
    """EBU Tech 3342 loudness range from short-term values."""
    st = short_term[short_term > -70]
    if len(st) < 2:
        return 0.0
    rel_gate = 10 * np.log10(np.mean(10 ** (st / 10))) - 20
    st = st[st > rel_gate]
    return float(np.percentile(st, 95) - np.percentile(st, 10))


def true_peak_db(data: np.ndarray) -> float:
    over = signal.resample_poly(data, 4, 1, axis=0)
    return 20 * np.log10(max(np.max(np.abs(over)), 1e-12))


def _psd(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return signal.welch(x, fs=SAMPLE_RATE, nperseg=16384, noverlap=8192,
                        window="hann", detrend=False)


def _band_power(freqs, psd, lo, hi) -> float:
    sel = (freqs >= lo) & (freqs < hi)
    return float(np.sum(psd[sel]))


def lossy_cutoff_hz(freqs: np.ndarray, psd: np.ndarray) -> float | None:
    """Where a lossy encoder cut the top end, or None if it reaches 19.5 kHz.

    Encoders remove everything above a fixed frequency, which shows as a
    cliff: the level falls more than 25 dB within 1 kHz and stays down.
    """
    db = 10 * np.log10(np.maximum(psd, 1e-20))
    step = freqs[1] - freqs[0]
    w = max(1, int(500 / step))
    for i in np.where((freqs >= 10000) & (freqs <= 19500))[0]:
        below = db[max(0, i - w):i].mean()
        above = db[i + int(200 / step):i + int(1200 / step)]
        if len(above) and below - above.max() > 25:
            return float(round(freqs[i], -2))
    return None


def detect_key(mono: np.ndarray) -> str | None:
    import librosa
    y = librosa.resample(mono.astype(np.float32), orig_sr=SAMPLE_RATE,
                         target_sr=22050)
    chroma = librosa.feature.chroma_cqt(y=y, sr=22050).mean(axis=1)
    if not np.any(chroma):
        return None
    best, name = -2.0, None
    for i in range(12):
        for profile, mode in ((MAJOR_PROFILE, "major"), (MINOR_PROFILE, "minor")):
            r = np.corrcoef(chroma, np.roll(profile, i))[0, 1]
            if r > best:
                best, name = r, f"{KEY_NAMES[i]} {mode}"
    return name


def detect_bpm(mono: np.ndarray) -> float | None:
    """Tempo from the autocorrelation of the onset envelope.

    Each candidate from 80 to 180 BPM is scored by the autocorrelation at
    1 to 8 beat lengths, which favours the steady pulse of dance music.
    """
    import librosa
    y = librosa.resample(mono.astype(np.float32), orig_sr=SAMPLE_RATE,
                         target_sr=22050)
    hop = 256
    rate = 22050 / hop
    onset = librosa.onset.onset_strength(y=y, sr=22050, hop_length=hop)
    onset = onset - onset.mean()
    ac = signal.correlate(onset, onset, mode="full", method="fft")[len(onset) - 1:]
    if ac[0] <= 0:
        return None
    ac /= ac[0]
    lags = np.arange(len(ac))
    candidates = np.arange(80.0, 180.0, 0.05)
    scores = np.zeros_like(candidates)
    for k in range(1, 9):
        scores += np.interp(k * 60 * rate / candidates, lags, ac) / np.sqrt(k)
    return round(float(candidates[np.argmax(scores)]), 1)


def _corr(a: np.ndarray, b: np.ndarray) -> float | None:
    if np.std(a) < 1e-9 or np.std(b) < 1e-9:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def analyze_audio(data: np.ndarray, *, file_name: str = "",
                  lossy: bool = False, musical: bool = True) -> dict:
    channels = data.shape[1]
    stereo = channels == 2 and not np.allclose(data[:, 0], data[:, 1])

    meter = pyloudnorm.Meter(SAMPLE_RATE)
    integrated = float(meter.integrated_loudness(data))
    if not np.isfinite(integrated) or integrated < SILENCE_LUFS:
        raise AnalysisError(f"{file_name}: the file is silent")

    weighted = k_weighted(data)
    curve = short_term_curve(weighted)
    tp = true_peak_db(data)
    peak = 20 * np.log10(max(np.max(np.abs(data)), 1e-12))
    rms = 10 * np.log10(max(np.mean(data ** 2), 1e-24))

    # Tonal and stereo measurements at matched loudness.
    norm = data * 10 ** ((NORMALIZE_TO_LUFS - integrated) / 20)
    left, right = norm[:, 0], norm[:, -1]
    mid, side = (left + right) / 2, (left - right) / 2
    freqs, p_mid = _psd(mid)
    total = _band_power(freqs, p_mid, 20, 20000)

    bands_db = {n: round(_db(_band_power(freqs, p_mid, lo, hi) / total), 2)
                for n, (lo, hi) in BANDS.items()}

    third, density = [], []
    for fc in THIRD_OCTAVE_HZ:
        lo, hi = fc * 2 ** (-1 / 6), min(fc * 2 ** (1 / 6), SAMPLE_RATE / 2)
        third.append(round(_db(_band_power(freqs, p_mid, lo, hi) / total), 2))
        sel = (freqs >= lo) & (freqs < hi)
        density.append(_db(float(np.mean(p_mid[sel]))) if sel.any() else None)
    # Tilt: slope of the spectral density per octave, 31.5 Hz to 16 kHz.
    pts = [(np.log2(f), d) for f, d in zip(THIRD_OCTAVE_HZ, density)
           if 31.5 <= f <= 16000 and d is not None]
    tilt = float(np.polyfit([p[0] for p in pts], [p[1] for p in pts], 1)[0])
    sel = (freqs >= 20) & (freqs <= 20000)
    centroid = float(np.sum(freqs[sel] * p_mid[sel]) / np.sum(p_mid[sel]))

    cutoff = lossy_cutoff_hz(freqs, p_mid) if lossy else None

    analysis = {
        "schema_version": SCHEMA_VERSION,
        "analyzer_version": ANALYZER_VERSION,
        "track": {"file": file_name,
                  "duration_s": round(data.shape[0] / SAMPLE_RATE, 2),
                  "sample_rate": SAMPLE_RATE,
                  "channels": 2 if stereo else 1},
        "loudness": {"integrated_lufs": round(integrated, 2),
                     "max_short_term_lufs": round(float(np.max(curve)), 2),
                     "true_peak_dbtp": round(float(tp), 2)},
        "dynamics": {"lra_lu": round(loudness_range(curve), 2),
                     "crest_db": round(float(peak - rms), 2),
                     "plr_db": round(float(tp - integrated), 2),
                     "short_term_curve_lufs": [round(float(v), 2) for v in curve]},
        "tonal": {"bands_db": bands_db, "third_octave_db": third,
                  "third_octave_hz": THIRD_OCTAVE_HZ,
                  "tilt_db_per_oct": round(tilt, 2),
                  "centroid_hz": round(centroid)},
        "stereo": None,
        "low_end": {"sub_to_bass_db": round(bands_db["sub"] - bands_db["bass"], 2),
                    "low_width": None},
        "musical": {"bpm": None, "key": None},
    }
    if cutoff is not None:
        analysis["track"]["lossy_cutoff_hz"] = cutoff

    if stereo:
        _, p_side = _psd(side)
        by_band = {}
        for n, (lo, hi) in BANDS.items():
            m = _band_power(freqs, p_mid, lo, hi)
            by_band[n] = round(_band_power(freqs, p_side, lo, hi) / m, 3) if m > 0 else None
        sos = signal.butter(4, 120, "lowpass", fs=SAMPLE_RATE, output="sos")
        low = signal.sosfilt(sos, norm, axis=0)
        e_l, e_r = np.mean(left ** 2), np.mean(right ** 2)
        corr, low_corr = _corr(left, right), _corr(low[:, 0], low[:, 1])
        analysis["stereo"] = {
            "width": round(float(np.mean(side ** 2) / np.mean(mid ** 2)), 3),
            "width_by_band": by_band,
            "correlation": round(corr, 3) if corr is not None else None,
            "low_correlation": round(low_corr, 3) if low_corr is not None else None,
            "balance_db": round(_db(e_l) - _db(e_r), 2),
        }
        m_low = _band_power(freqs, p_mid, 20, 120)
        analysis["low_end"]["low_width"] = round(
            _band_power(freqs, p_side, 20, 120) / m_low, 3) if m_low > 0 else None

    if musical:
        mono_raw = data.mean(axis=1)
        analysis["musical"] = {"bpm": detect_bpm(mono_raw),
                               "key": detect_key(mono_raw)}
    return analysis


def analyze_file(path: Path, *, musical: bool = True) -> dict:
    path = Path(path)
    if path.suffix.lower() not in AUDIO_EXTENSIONS:
        raise AnalysisError(f"{path.name}: not a supported audio file")
    data = load_audio(path)
    analysis = analyze_audio(data, file_name=path.name,
                             lossy=path.suffix.lower() in LOSSY_EXTENSIONS,
                             musical=musical)
    analysis["track"]["sha1"] = file_sha1(path)
    tags = read_tags(path)
    analysis["track"]["tags"] = tags
    # DJ software keys from the tags beat our chroma estimate, which often
    # gets the tonic right but the mode wrong.
    if tags["key"]:
        analysis["musical"]["key_detected"] = analysis["musical"]["key"]
        analysis["musical"]["key"] = tags["key"]
    return analysis


def analyze_cached(path: Path, cache_dir: Path, sha: str | None = None) -> dict:
    """Analyze a file once; later calls read the cached JSON by file hash.

    A cache entry from an older analyzer version is measured again.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    sha = sha or file_sha1(path)
    cached = cache_dir / f"{sha}.json"
    if cached.exists():
        try:
            data = json.loads(cached.read_text())
            if (data.get("schema_version") == SCHEMA_VERSION
                    and data.get("analyzer_version") == ANALYZER_VERSION):
                data["track"]["file"] = Path(path).name
                return data
        except (OSError, json.JSONDecodeError):
            pass
    analysis = analyze_file(path)
    tmp = cached.with_suffix(".tmp")
    tmp.write_text(json.dumps(analysis))
    tmp.replace(cached)
    return analysis


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("audio", type=Path)
    ap.add_argument("-o", "--out", type=Path)
    args = ap.parse_args(argv)
    try:
        analysis = analyze_file(args.audio)
    except AnalysisError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    text = json.dumps(analysis, indent=2)
    if args.out:
        args.out.write_text(text + "\n")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
