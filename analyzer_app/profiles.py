"""Build genre reference profiles from per-track analysis JSON files.

A profile is the median and spread of every measurement in the Mix Analysis
Spec, taken across a set of reference songs in one genre. The scoring step
compares a song's analysis against these ranges. This module only reads
analysis JSON (schema_version 1); it never touches audio.

The app ships built-in profiles, the way DropTrack benchmarks a song without
asking for reference tracks, and lets the user build custom ones from their
own songs. `match` ranks every profile against one song, which suggests a
genre and the closest-sounding reference tracks.

Usage:
    python -m analyzer_app.profiles build "Tech House" path/to/analyses/ -o profiles/
    python -m analyzer_app.profiles show profiles/tech-house.json
    python -m analyzer_app.profiles match my_song.json profiles/
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
from pathlib import Path

import numpy as np

PROFILE_SCHEMA_VERSION = 1
SUPPORTED_ANALYSIS_VERSIONS = {1}
BUILDER_VERSION = "0.1.0"

BANDS = ["sub", "bass", "low_mid", "mid_low", "mid", "upper_mid",
         "presence", "high", "air", "top"]

# Upper edge of each band. A lossy file's band is dropped when this edge is
# above the encoder's cutoff.
BAND_HIGH_EDGE_HZ = {"sub": 60, "bass": 120, "low_mid": 250, "mid_low": 500,
                     "mid": 1000, "upper_mid": 2000, "presence": 4000,
                     "high": 8000, "air": 12000, "top": 20000}

# ISO 1/3-octave centres, 20 Hz to 20 kHz. Used when an analysis doesn't
# carry its own frequency list.
THIRD_OCTAVE_HZ = [20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250,
                   315, 400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500,
                   3150, 4000, 5000, 6300, 8000, 10000, 12500, 16000, 20000]

# Smallest half-width (median to p10/p90) a range may have. Small or very
# uniform reference sets otherwise produce ranges so tight that any song
# fails them. Values are in each metric's own unit.
MIN_HALF_WIDTH = {
    "loudness.integrated_lufs": 1.0,
    "loudness.max_short_term_lufs": 1.0,
    "loudness.true_peak_dbtp": 0.5,
    "dynamics.lra_lu": 1.0,
    "dynamics.crest_db": 1.0,
    "dynamics.plr_db": 1.0,
    "dynamics.section_contrast_lu": 1.5,
    "tonal.tilt_db_per_oct": 0.4,
    "tonal.centroid_hz": 250.0,
    "stereo.width": 0.05,
    "stereo.correlation": 0.05,
    "stereo.low_correlation": 0.03,
    "stereo.balance_db": 0.5,
    "low_end.sub_to_bass_db": 1.5,
    "low_end.low_width": 0.03,
    "musical.bpm": 2.0,
}
MIN_HALF_WIDTH_BAND_DB = 1.5        # tonal.bands_db.* and the 1/3-octave curve
MIN_HALF_WIDTH_BAND_WIDTH = 0.05    # stereo.width_by_band.*

# Metrics used to spot a reference track that doesn't belong in the set.
FINGERPRINT = ["loudness.integrated_lufs", "tonal.tilt_db_per_oct",
               "low_end.sub_to_bass_db", "stereo.width", "dynamics.crest_db"]
OUTLIER_Z = 3.5          # robust z-score (MAD based)
OUTLIER_MIN_HITS = 2     # flag a track only when this many metrics are off
OUTLIER_MIN_TRACKS = 8   # below this the median is too unstable to judge

# Numbers stored per reference track so a song can be matched to the
# closest-sounding references ("sounds most like ...").
MATCH_METRICS = FINGERPRINT + [f"tonal.bands_db.{b}" for b in BANDS]

PROFILE_KINDS = {"builtin", "custom"}

# One song works (compare against a single reference); the profile is marked
# low confidence and its ranges fall back to the minimum widths.
MIN_TRACKS = 1
GOOD_TRACKS = 10

# A loudness-curve frame below this is silence (intro, fade, gap) and is
# left out of the section-contrast figure.
SILENCE_LUFS = -40.0


class ProfileError(Exception):
    pass


def slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    if not slug:
        raise ProfileError(f"genre name {name!r} has no letters or digits")
    return slug


def load_analysis(path: Path) -> dict:
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as e:
        raise ProfileError(f"{path.name}: can't read analysis ({e})") from e
    version = data.get("schema_version")
    if version not in SUPPORTED_ANALYSIS_VERSIONS:
        raise ProfileError(f"{path.name}: analysis schema_version {version!r} "
                           f"is not supported (expected one of "
                           f"{sorted(SUPPORTED_ANALYSIS_VERSIONS)})")
    return data


def _num(value) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if np.isfinite(f) else None


def _get(data: dict, dotted: str):
    node = data
    for key in dotted.split("."):
        if not isinstance(node, dict):
            return None
        node = node.get(key)
    return node


def _lossy_cutoff(analysis: dict) -> float | None:
    """Frequency above which a lossy file has no real content, if known."""
    return _num(_get(analysis, "track.lossy_cutoff_hz"))


def section_contrast(curve) -> float | None:
    """Loud-section minus quiet-section level, from the 1 s loudness curve.

    95th minus 10th percentile of the non-silent frames, so one loud hit or
    a long fade doesn't decide it.
    """
    if not isinstance(curve, list):
        return None
    values = np.array([v for v in (_num(x) for x in curve)
                       if v is not None and v > SILENCE_LUFS])
    if len(values) < 10:
        return None
    return float(np.percentile(values, 95) - np.percentile(values, 10))


def flatten(analysis: dict) -> dict[str, float]:
    """One analysis -> {metric_name: value} for every scalar we profile.

    Mono tracks have no stereo values, and lossy files drop the bands above
    their encoder cutoff. Missing metrics are simply absent.
    """
    out: dict[str, float] = {}
    channels = _get(analysis, "track.channels")
    is_mono = channels == 1

    for key in ["loudness.integrated_lufs", "loudness.max_short_term_lufs",
                "loudness.true_peak_dbtp", "dynamics.lra_lu",
                "dynamics.crest_db", "dynamics.plr_db",
                "tonal.tilt_db_per_oct", "tonal.centroid_hz",
                "low_end.sub_to_bass_db", "musical.bpm"]:
        v = _num(_get(analysis, key))
        if v is not None:
            out[key] = v

    if not is_mono:
        for key in ["stereo.width", "stereo.correlation",
                    "stereo.low_correlation", "stereo.balance_db",
                    "low_end.low_width"]:
            v = _num(_get(analysis, key))
            if v is not None:
                out[key] = v
        by_band = _get(analysis, "stereo.width_by_band") or {}
        if isinstance(by_band, dict):
            for band, value in by_band.items():
                v = _num(value)
                if v is not None and band in BANDS:
                    out[f"stereo.width_by_band.{band}"] = v

    contrast = section_contrast(_get(analysis, "dynamics.short_term_curve_lufs"))
    if contrast is not None:
        out["dynamics.section_contrast_lu"] = contrast

    cutoff = _lossy_cutoff(analysis)
    bands = _get(analysis, "tonal.bands_db") or {}
    if isinstance(bands, dict):
        for band in BANDS:
            v = _num(bands.get(band))
            if v is None:
                continue
            if cutoff is not None and BAND_HIGH_EDGE_HZ[band] > cutoff:
                continue
            out[f"tonal.bands_db.{band}"] = v

    # A lossy file's centroid and tilt are pulled down by the missing top end.
    if cutoff is not None and cutoff < 19000:
        out.pop("tonal.centroid_hz", None)
        out.pop("tonal.tilt_db_per_oct", None)
    return out


def _third_octave(analysis: dict) -> tuple[list[float], list[float | None]] | None:
    values = _get(analysis, "tonal.third_octave_db")
    if not isinstance(values, list) or not values:
        return None
    freqs = _get(analysis, "tonal.third_octave_hz") or THIRD_OCTAVE_HZ
    if len(freqs) != len(values):
        return None
    cutoff = _lossy_cutoff(analysis)
    cleaned = []
    for f, v in zip(freqs, values):
        n = _num(v)
        if cutoff is not None and f > cutoff:
            n = None
        cleaned.append(n)
    return [float(f) for f in freqs], cleaned


def fold_bpm(values: list[float]) -> list[float]:
    """Fix half- and double-time tempo guesses against the set's median.

    Tempo detection often reports 63 or 252 for a 126 BPM track. Each value
    is halved or doubled when that brings it clearly closer to the median.
    Run twice, since the first median can itself sit on a wrong octave.
    """
    if len(values) < 3:
        return values
    folded = list(values)
    for _ in range(2):
        centre = float(np.median(folded))
        folded = [min((v, v / 2, v * 2),
                      key=lambda c: abs(np.log2(c / centre)))
                  for v in values]
    return [round(v, 2) for v in folded]


def stats(values, min_half_width: float = 0.0) -> dict:
    arr = np.asarray(values, dtype=float)
    median = float(np.median(arr))
    p10, p25, p75, p90 = (float(x) for x in np.percentile(arr, [10, 25, 75, 90]))
    widened = False
    if median - p10 < min_half_width:
        p10, widened = median - min_half_width, True
    if p90 - median < min_half_width:
        p90, widened = median + min_half_width, True
    return {
        "n": int(arr.size),
        "median": round(median, 3),
        "p10": round(p10, 3),
        "p25": round(p25, 3),
        "p75": round(p75, 3),
        "p90": round(p90, 3),
        "mean": round(float(arr.mean()), 3),
        "std": round(float(arr.std(ddof=1)) if arr.size > 1 else 0.0, 3),
        "min": round(float(arr.min()), 3),
        "max": round(float(arr.max()), 3),
        "widened": widened,
    }


def _min_half_width(metric: str) -> float:
    if metric.startswith("tonal.bands_db."):
        return MIN_HALF_WIDTH_BAND_DB
    if metric.startswith("stereo.width_by_band."):
        return MIN_HALF_WIDTH_BAND_WIDTH
    return MIN_HALF_WIDTH.get(metric, 0.0)


def find_outliers(rows: list[dict[str, float]]) -> dict[int, list[str]]:
    """Index of each track that is far from the set on several metrics.

    Uses a MAD-based z-score, which a single odd track can't drag around the
    way it can a mean and standard deviation.
    """
    if len(rows) < OUTLIER_MIN_TRACKS:
        return {}
    hits: dict[int, list[str]] = {}
    for metric in FINGERPRINT:
        idx = [i for i, r in enumerate(rows) if metric in r]
        if len(idx) < OUTLIER_MIN_TRACKS:
            continue
        vals = np.array([rows[i][metric] for i in idx])
        med = np.median(vals)
        mad = np.median(np.abs(vals - med))
        floor = _min_half_width(metric) / 2  # don't flag tiny absolute gaps
        scale = max(mad / 0.6745, floor, 1e-9)
        for i, v in zip(idx, vals):
            z = (v - med) / scale
            if abs(z) > OUTLIER_Z:
                hits.setdefault(i, []).append(
                    f"{metric} {v:g} vs genre median {med:g}")
    return {i: why for i, why in hits.items() if len(why) >= OUTLIER_MIN_HITS}


def build_profile(name: str, analyses: list[tuple[str, dict]], *,
                  exclude_outliers: bool = True,
                  keep: set[str] | None = None,
                  kind: str = "custom",
                  now: dt.datetime | None = None) -> dict:
    """Aggregate analyses into a profile dict.

    analyses: (label, analysis_json) pairs, label usually the file name.
    keep: labels never excluded as outliers (the user said they belong).
    kind: "builtin" for profiles shipped with the app, "custom" for ones the
    user builds from their own songs.
    """
    if kind not in PROFILE_KINDS:
        raise ProfileError(f"kind must be one of {sorted(PROFILE_KINDS)}")
    if len(analyses) < MIN_TRACKS:
        raise ProfileError(f"{name}: no reference songs yet")
    keep = keep or set()
    rows = [flatten(a) for _, a in analyses]
    outliers = find_outliers(rows) if exclude_outliers else {}
    outliers = {i: w for i, w in outliers.items() if analyses[i][0] not in keep}

    used = [i for i in range(len(analyses)) if i not in outliers]
    if len(used) < MIN_TRACKS:
        used, outliers = list(range(len(analyses))), {}

    used_rows = [rows[i] for i in used]
    bpm_idx = [j for j, r in enumerate(used_rows) if "musical.bpm" in r]
    folded = fold_bpm([used_rows[j]["musical.bpm"] for j in bpm_idx])
    for j, v in zip(bpm_idx, folded):
        used_rows[j] = {**used_rows[j], "musical.bpm": v}

    metric_names = sorted({k for r in used_rows for k in r})
    metrics = {}
    for metric in metric_names:
        vals = [r[metric] for r in used_rows if metric in r]
        if len(vals) < MIN_TRACKS:
            continue
        metrics[metric] = stats(vals, _min_half_width(metric))

    curves = {}
    thirds = [_third_octave(analyses[i][1]) for i in used]
    thirds = [t for t in thirds if t is not None]
    if thirds:
        freqs = thirds[0][0]
        same = [vals for f, vals in thirds if f == freqs]
        med, p10, p90, n = [], [], [], []
        for k in range(len(freqs)):
            col = [v[k] for v in same if v[k] is not None]
            if len(col) < MIN_TRACKS:
                med.append(None); p10.append(None); p90.append(None)
                n.append(len(col))
                continue
            s = stats(col, MIN_HALF_WIDTH_BAND_DB)
            med.append(s["median"]); p10.append(s["p10"]); p90.append(s["p90"])
            n.append(s["n"])
        curves["third_octave_db"] = {"freqs_hz": freqs, "median": med,
                                     "p10": p10, "p90": p90, "n": n}

    keys: dict[str, int] = {}
    for i in used:
        key = _get(analyses[i][1], "musical.key")
        if isinstance(key, str) and key:
            keys[key] = keys.get(key, 0) + 1

    n_used = len(used)
    notes = []
    if n_used < GOOD_TRACKS:
        notes.append(f"Only {n_used} reference tracks. Ranges are rough until "
                     f"there are {GOOD_TRACKS} or more.")
    widened = [m for m, s in metrics.items() if s["widened"]]
    if widened:
        notes.append(f"{len(widened)} ranges were set to a minimum width so "
                     f"small differences aren't marked as problems.")
    lossy = [analyses[i][0] for i in used if _lossy_cutoff(analyses[i][1])]
    if lossy:
        notes.append(f"{len(lossy)} lossy files: their top bands were left "
                     f"out. Lossless files give a better top end.")
    mono = [analyses[i][0] for i in used
            if _get(analyses[i][1], "track.channels") == 1]
    if mono:
        notes.append(f"{len(mono)} mono files were left out of stereo ranges.")

    confidence = ("good" if n_used >= GOOD_TRACKS * 1.5 else
                  "ok" if n_used >= GOOD_TRACKS else "low")
    now = now or dt.datetime.now(dt.timezone.utc)

    return {
        "schema_version": PROFILE_SCHEMA_VERSION,
        "profile": {
            "id": slugify(name),
            "name": name,
            "kind": kind,
            "built_at": now.replace(microsecond=0).isoformat(),
            "builder_version": BUILDER_VERSION,
            "analysis_schema_version": 1,
            "normalized_to_lufs": -14.0,
        },
        "source": {
            "track_count": len(analyses),
            "used_count": n_used,
            "tracks": [{**_track_ref(analyses[i]),
                        "match": {k: round(row[k], 3) for k in MATCH_METRICS
                                  if k in row}}
                       for i, row in zip(used, used_rows)],
            "excluded": [{**_track_ref(analyses[i]), "reasons": why}
                         for i, why in sorted(outliers.items())],
        },
        "metrics": metrics,
        "curves": curves,
        "context": {"keys": dict(sorted(keys.items(), key=lambda kv: -kv[1]))},
        "quality": {"confidence": confidence, "notes": notes},
    }


def _track_ref(item: tuple[str, dict]) -> dict:
    label, analysis = item
    ref = {"label": label}
    for k in ("file", "duration_s", "sha1"):
        v = _get(analysis, f"track.{k}")
        if v is not None:
            ref[k] = v
    return ref


def _half_range(s: dict) -> float:
    return max((s["p90"] - s["p10"]) / 2, 1e-6)


def distance(row: dict[str, float], profile: dict) -> tuple[float, int]:
    """Root-mean-square distance of a song from a profile, in half-ranges.

    0 means every number sits on the genre median, 1 means numbers sit at the
    edge of the genre's 10th to 90th percentile range on average. Returns
    (distance, number of metrics compared).
    """
    parts = []
    for metric in MATCH_METRICS:
        s = profile["metrics"].get(metric)
        if s and metric in row:
            parts.append(((row[metric] - s["median"]) / _half_range(s)) ** 2)
    if not parts:
        return float("inf"), 0
    return float(np.sqrt(np.mean(parts))), len(parts)


def nearest_tracks(row: dict[str, float], profile: dict, k: int = 3) -> list:
    """The reference tracks in a profile that sound closest to the song."""
    scored = []
    for t in profile["source"]["tracks"]:
        parts = []
        for metric, v in t.get("match", {}).items():
            s = profile["metrics"].get(metric)
            if s and metric in row:
                parts.append(((row[metric] - v) / _half_range(s)) ** 2)
        if parts:
            scored.append((float(np.sqrt(np.mean(parts))), t["label"]))
    return [{"label": label, "distance": round(d, 2)}
            for d, label in sorted(scored)[:k]]


def match(analysis: dict, profiles: list[dict]) -> list[dict]:
    """Rank profiles by how close the song sits to each, closest first."""
    row = flatten(analysis)
    ranked = []
    for p in profiles:
        d, n = distance(row, p)
        if n == 0:
            continue
        ranked.append({"id": p["profile"]["id"], "name": p["profile"]["name"],
                       "kind": p["profile"].get("kind", "custom"),
                       "distance": round(d, 2), "compared": n,
                       "nearest_tracks": nearest_tracks(row, p)})
    return sorted(ranked, key=lambda r: r["distance"])


def load_profiles(*folders: Path) -> list[dict]:
    """All profiles in the given folders. A later folder's profile replaces
    an earlier one with the same id, so user profiles override built-ins."""
    found: dict[str, dict] = {}
    for folder in folders:
        for path in sorted(Path(folder).glob("*.json")):
            try:
                p = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            if p.get("schema_version") == PROFILE_SCHEMA_VERSION:
                found[p["profile"]["id"]] = p
    return list(found.values())


def build_from_dir(name: str, folder: Path, **kw) -> dict:
    paths = sorted(p for p in folder.glob("*.json"))
    if not paths:
        raise ProfileError(f"no analysis .json files in {folder}")
    analyses, skipped = [], []
    for p in paths:
        try:
            analyses.append((p.stem, load_analysis(p)))
        except ProfileError as e:
            skipped.append(str(e))
    profile = build_profile(name, analyses, **kw)
    if skipped:
        profile["quality"]["notes"].append(
            f"{len(skipped)} files skipped: " + "; ".join(skipped))
    return profile


def save_profile(profile: dict, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{profile['profile']['id']}.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(profile, indent=2) + "\n")
    tmp.replace(path)
    return path


def describe(profile: dict) -> str:
    p, src, m = profile["profile"], profile["source"], profile["metrics"]
    lines = [f"{p['name']}: {src['used_count']} of {src['track_count']} "
             f"tracks used, confidence {profile['quality']['confidence']}"]

    def row(label, key, unit, fmt="{:.1f}"):
        s = m.get(key)
        if s:
            lines.append(f"  {label:<22}{fmt.format(s['median'])} {unit}  "
                         f"(range {fmt.format(s['p10'])} to "
                         f"{fmt.format(s['p90'])})")

    row("Loudness", "loudness.integrated_lufs", "LUFS")
    row("True peak", "loudness.true_peak_dbtp", "dBTP")
    row("Loudness range", "dynamics.lra_lu", "LU")
    row("Crest factor", "dynamics.crest_db", "dB")
    row("Section contrast", "dynamics.section_contrast_lu", "LU")
    row("Tilt", "tonal.tilt_db_per_oct", "dB/oct", "{:.2f}")
    row("Stereo width", "stereo.width", "", "{:.2f}")
    row("Sub vs bass", "low_end.sub_to_bass_db", "dB")
    row("Tempo", "musical.bpm", "BPM", "{:.0f}")
    lines.append("  Bands (dB vs total):")
    for band in BANDS:
        s = m.get(f"tonal.bands_db.{band}")
        if s:
            lines.append(f"    {band:<10}{s['median']:7.1f}  "
                         f"({s['p10']:.1f} to {s['p90']:.1f})")
    for e in src["excluded"]:
        lines.append(f"  Left out {e['label']}: " + "; ".join(e["reasons"]))
    for note in profile["quality"]["notes"]:
        lines.append(f"  Note: {note}")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="build a profile from analysis files")
    b.add_argument("genre")
    b.add_argument("analyses", type=Path, help="folder of analysis .json")
    b.add_argument("-o", "--out", type=Path, default=Path("profiles"))
    b.add_argument("--keep-outliers", action="store_true",
                   help="use every track, even ones that look out of place")
    b.add_argument("--keep", action="append", default=[],
                   help="label of a track never to leave out (repeatable)")
    b.add_argument("--builtin", action="store_true",
                   help="mark as a built-in profile shipped with the app")
    s = sub.add_parser("show", help="print a profile summary")
    s.add_argument("profile", type=Path)
    m = sub.add_parser("match", help="rank genres for one song's analysis")
    m.add_argument("analysis", type=Path)
    m.add_argument("profiles", type=Path, nargs="+",
                   help="profile folders, built-in first, custom last")
    args = ap.parse_args(argv)

    try:
        if args.cmd == "build":
            profile = build_from_dir(args.genre, args.analyses,
                                     exclude_outliers=not args.keep_outliers,
                                     keep=set(args.keep),
                                     kind="builtin" if args.builtin else "custom")
            path = save_profile(profile, args.out)
            print(describe(profile))
            print(f"Saved {path}")
        elif args.cmd == "show":
            print(describe(json.loads(args.profile.read_text())))
        else:
            ranked = match(load_analysis(args.analysis),
                           load_profiles(*args.profiles))
            if not ranked:
                raise ProfileError("no profiles to match against")
            for r in ranked:
                near = ", ".join(t["label"] for t in r["nearest_tracks"])
                print(f"{r['name']:<24}distance {r['distance']:.2f}"
                      f"  closest: {near}")
    except ProfileError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
