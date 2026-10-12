"""Compare one song against a genre profile: what's missing, what to adjust,
and a 0 to 100 readiness score with a sub-score per area.

Reads only analysis JSON and profile JSON, never audio. Every number is
judged against the genre's 10th to 90th percentile range: inside the range
scores 90 to 100, and the score falls off the further outside it goes. A few
technical checks (clipping, phase, bass width, balance) apply to every genre.

    python -m analyzer_app.compare my_song.json profiles/indie-dance.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

from analyzer_app import profiles as P

REPORT_SCHEMA_VERSION = 1

# Share of the overall score per area. An area with nothing to compare (a
# mono song has no stereo) is left out and the others are scaled up.
AREA_WEIGHTS = {"tone": 0.30, "loudness": 0.20, "dynamics": 0.20,
                "low_end": 0.15, "stereo": 0.15}
AREA_LABELS = {"loudness": "Loudness", "tone": "Tonal balance",
               "dynamics": "Dynamics", "stereo": "Stereo image",
               "low_end": "Low end"}

# How quickly a score falls once a number leaves the genre range, in
# half-ranges. One half-range outside scores about 72, two about 37.
FALLOFF = 1.25

# Findings that cost less than this many points are left off the lists (they
# stay in "metrics"), so the report leads with what matters.
MIN_IMPACT = 0.5

READINESS = [(85, "Release ready"), (70, "Almost there"),
             (50, "Needs work"), (0, "Early mix")]

BAND_HZ = {"sub": "20-60 Hz", "bass": "60-120 Hz", "low_mid": "120-250 Hz",
           "mid_low": "250-500 Hz", "mid": "500 Hz-1 kHz",
           "upper_mid": "1-2 kHz", "presence": "2-4 kHz", "high": "4-8 kHz",
           "air": "8-12 kHz", "top": "12-20 kHz"}
BAND_NAMES = {"sub": "sub", "bass": "bass", "low_mid": "low mids",
              "mid_low": "lower mids", "mid": "mids", "upper_mid": "upper mids",
              "presence": "presence", "high": "highs", "air": "air",
              "top": "top end"}


@dataclass(frozen=True)
class Metric:
    key: str
    area: str
    label: str
    unit: str
    weight: float
    # What it means when the song is below / above the genre range, and
    # whether that's something missing or something to pull back.
    low: tuple[str, str]
    high: tuple[str, str]
    fmt: str = "{:.1f}"


def _m(key, area, label, unit, weight, low, high, fmt="{:.1f}"):
    return Metric(key, area, label, unit, weight, low, high, fmt)


METRICS = [
    _m("loudness.integrated_lufs", "loudness", "Loudness", "LUFS", 1.0,
       ("adjust", "Quieter than the genre. Raise the limiter's gain by about {gap}."),
       ("adjust", "Louder than the genre. Back the limiter off by about {gap} to get punch back.")),
    _m("loudness.max_short_term_lufs", "loudness", "Loudest section", "LUFS", 0.4,
       ("missing", "The loudest part doesn't get as loud as the genre's drops."),
       ("adjust", "The loudest part is pushed harder than the genre's drops.")),
    _m("dynamics.crest_db", "dynamics", "Punch (crest factor)", "dB", 1.0,
       ("adjust", "Squashed compared with the genre. Use less limiting or bus compression so the kick and snare punch through."),
       ("adjust", "Peaks stick out more than in the genre. A little more bus compression or limiting will glue it.")),
    _m("dynamics.plr_db", "dynamics", "Peak to loudness", "dB", 0.5,
       ("adjust", "Little headroom between peaks and loudness, which points to heavy limiting."),
       ("adjust", "Lots of room between peaks and loudness. The master could take more limiting.")),
    _m("dynamics.lra_lu", "dynamics", "Loudness range", "LU", 0.5,
       ("missing", "The track stays at one level more than the genre does. Breakdowns could drop further."),
       ("adjust", "Level swings more than the genre. Even out quiet and loud parts.")),
    _m("dynamics.section_contrast_lu", "dynamics", "Drop impact", "LU", 0.8,
       ("missing", "The drop doesn't lift above the breakdown as much as in the genre. Thin out the breakdown or add energy to the drop."),
       ("adjust", "The gap between breakdown and drop is bigger than the genre's. The quiet parts may lose the floor.")),
    _m("tonal.tilt_db_per_oct", "tone", "Brightness (tilt)", "dB/oct", 1.0,
       ("adjust", "Darker overall than the genre. A gentle high shelf from about 3 kHz brings it in line."),
       ("adjust", "Brighter overall than the genre. Ease the top end with a gentle high shelf."),
       "{:.2f}"),
    _m("tonal.centroid_hz", "tone", "Brightness (centroid)", "Hz", 0.4,
       ("adjust", "The balance sits lower than the genre's."),
       ("adjust", "The balance sits higher than the genre's."), "{:.0f}"),
    _m("stereo.width", "stereo", "Stereo width", "", 1.0,
       ("missing", "Narrower than the genre. Widen pads, synths and effects, not the low end."),
       ("adjust", "Wider than the genre. Check how it sounds in mono."), "{:.2f}"),
    _m("stereo.correlation", "stereo", "Mono compatibility", "", 0.6,
       ("adjust", "Left and right are less alike than in the genre. Check wideners and chorus in mono."),
       ("adjust", "Left and right are more alike than in the genre, so it may sound flat."), "{:.2f}"),
    _m("low_end.sub_to_bass_db", "low_end", "Sub vs bass", "dB", 1.0,
       ("missing", "Less sub (20-60 Hz) under the bass than the genre. Bring up the sub or the kick's tail."),
       ("adjust", "The sub outweighs the 60-120 Hz punch. Lift the kick and bass body around 80-100 Hz.")),
    _m("low_end.low_width", "low_end", "Bass width", "", 0.6,
       ("adjust", "The low end is more mono than the genre."),
       ("adjust", "The low end is wider than the genre. Keep everything below 120 Hz in mono."),
       "{:.3f}"),
    _m("stereo.low_correlation", "low_end", "Bass phase", "", 0.6,
       ("adjust", "Left and right disagree more below 120 Hz than in the genre. Mono the bass."),
       ("adjust", "The bass is more mono than the genre."), "{:.2f}"),
]
for _band in P.BANDS:
    METRICS.append(_m(
        f"tonal.bands_db.{_band}", "tone",
        f"{BAND_HZ[_band]} ({BAND_NAMES[_band]})", "dB", 0.6,
        ("missing", f"Less {BAND_NAMES[_band]} than the genre. Boost {BAND_HZ[_band]} by about {{gap}}."),
        ("adjust", f"Too much {BAND_NAMES[_band]} for the genre. Cut {BAND_HZ[_band]} by about {{gap}}.")))
for _band in ("mid", "upper_mid", "presence", "high", "air"):
    METRICS.append(_m(
        f"stereo.width_by_band.{_band}", "stereo",
        f"Width at {BAND_HZ[_band]}", "", 0.25,
        ("missing", f"Narrower than the genre at {BAND_HZ[_band]}."),
        ("adjust", f"Wider than the genre at {BAND_HZ[_band]}."), "{:.2f}"))
METRICS_BY_KEY = {m.key: m for m in METRICS}


class CompareError(Exception):
    pass


def metric_score(value: float, s: dict) -> tuple[float, float]:
    """(score 0-100, signed distance outside the range in half-ranges)."""
    half = P._half_range(s)
    d = (value - s["median"]) / half
    if s["p10"] <= value <= s["p90"]:
        return 100 - 10 * min(abs(d), 1.0) ** 2, 0.0
    outside = (value - s["p90"] if value > s["p90"] else value - s["p10"]) / half
    return 90 * math.exp(-0.5 * (outside / FALLOFF) ** 2), outside


def _fold_bpm(bpm: float, median: float) -> float:
    return min((bpm, bpm / 2, bpm * 2), key=lambda c: abs(math.log2(c / median)))


def _gap_text(gap: float, unit: str) -> str:
    gap = round(abs(gap) * 2) / 2
    return f"{gap:g} {unit}".strip()


def technical_checks(analysis: dict) -> list[dict]:
    """Problems in any genre, judged on fixed limits rather than the profile."""
    found = []

    def add(key, title, text, penalty):
        found.append({"kind": "problem", "metric": key, "label": title,
                      "text": text, "penalty": penalty})

    tp = P._num(P._get(analysis, "loudness.true_peak_dbtp"))
    lossy = P._get(analysis, "track.lossy_cutoff_hz") is not None
    if tp is not None and tp > 0.0:
        add("loudness.true_peak_dbtp", "Clipping",
            f"True peak is {tp:+.1f} dBTP, so it clips. Set the limiter's ceiling to -1 dBTP."
            + (" (MP3 encoding adds some overs, so check the WAV too.)" if lossy else ""), 8)
    elif tp is not None and tp > -1.0:
        add("loudness.true_peak_dbtp", "True peak",
            f"True peak is {tp:.1f} dBTP. Streaming services ask for -1 dBTP or lower, "
            f"or the encode can distort. Lower the limiter's ceiling.", 3)
    if P._get(analysis, "track.channels") != 1:
        corr = P._num(P._get(analysis, "stereo.correlation"))
        low = P._num(P._get(analysis, "stereo.low_correlation"))
        bal = P._num(P._get(analysis, "stereo.balance_db"))
        if corr is not None and corr < 0:
            add("stereo.correlation", "Phase",
                "Left and right are out of phase overall. Parts will vanish in mono. "
                "Look for a polarity flip or an extreme widener.", 10)
        if low is not None and low < 0.5:
            add("stereo.low_correlation", "Bass phase",
                f"The bass is out of phase between left and right (correlation {low:.2f}). "
                f"It will thin out on club systems. Mono everything below 120 Hz.", 6)
        if bal is not None and abs(bal) > 1.5:
            side = "left" if bal > 0 else "right"
            add("stereo.balance_db", "Balance",
                f"The mix leans {abs(bal):.1f} dB to the {side}. Check panning.", 3)
    return found


def compare(analysis: dict, profile: dict) -> dict:
    """The comparison report for one song against one genre profile."""
    if profile.get("schema_version") != P.PROFILE_SCHEMA_VERSION:
        raise CompareError("profile was made by a different version of the app")
    row = P.flatten(analysis)
    pm = profile["metrics"]
    if "musical.bpm" in row and "musical.bpm" in pm:
        row["musical.bpm"] = _fold_bpm(row["musical.bpm"], pm["musical.bpm"]["median"])

    metrics, findings = [], []
    area_parts: dict[str, list[tuple[float, float]]] = {a: [] for a in AREA_WEIGHTS}
    for m in METRICS:
        s = pm.get(m.key)
        if not s or m.key not in row:
            continue
        v = row[m.key]
        score, outside = metric_score(v, s)
        status = "ok" if outside == 0 else ("low" if outside < 0 else "high")
        metrics.append({"metric": m.key, "area": m.area, "label": m.label,
                        "unit": m.unit, "value": round(v, 3),
                        "target": s["median"], "low": s["p10"], "high": s["p90"],
                        "status": status, "score": round(score, 1)})
        area_parts[m.area].append((score, m.weight))
        if status != "ok":
            kind, text = m.low if status == "low" else m.high
            findings.append({"kind": kind, "metric": m.key, "area": m.area,
                             "label": m.label, "status": status,
                             "text": text.format(gap=_gap_text(v - s["median"], m.unit)),
                             "value": round(v, 3), "target": s["median"],
                             "_score": score, "_weight": m.weight})

    areas = {}
    for a, parts in area_parts.items():
        if parts:
            total = sum(w for _, w in parts)
            areas[a] = sum(sc * w for sc, w in parts) / total
    if not areas:
        raise CompareError("nothing in this song could be compared with the profile")

    problems = technical_checks(analysis)
    for p in problems:
        area = METRICS_BY_KEY[p["metric"]].area if p["metric"] in METRICS_BY_KEY else "loudness"
        areas[area] = max(0.0, areas.get(area, 100.0) - p["penalty"] * 2)

    weight_sum = sum(AREA_WEIGHTS[a] for a in areas)
    overall = sum(areas[a] * AREA_WEIGHTS[a] for a in areas) / weight_sum

    # Points each finding costs the overall score, used to order next steps.
    area_w = {a: sum(w for _, w in area_parts[a]) for a in areas}
    for f in findings:
        f["impact"] = round((100 - f.pop("_score")) * f.pop("_weight")
                            / area_w[f["area"]] * AREA_WEIGHTS[f["area"]]
                            / weight_sum, 1)
    findings = [f for f in _merge_band_width(findings) if f["impact"] >= MIN_IMPACT]
    for p in problems:
        p["impact"] = p.pop("penalty")
    findings.sort(key=lambda f: -f["impact"])
    problems.sort(key=lambda f: -f["impact"])
    # Technical problems come first: they hurt in any genre.
    next_steps = [f["text"] for f in (problems + findings)[:3]]

    context = {}
    if "musical.bpm" in row and "musical.bpm" in pm:
        s = pm["musical.bpm"]
        context["bpm"] = {"value": round(row["musical.bpm"], 1),
                          "genre_low": s["p10"], "genre_high": s["p90"],
                          "in_range": s["p10"] - 1 <= row["musical.bpm"] <= s["p90"] + 1}
    key = P._get(analysis, "musical.key")
    if key:
        context["key"] = key
        context["genre_keys"] = list(profile.get("context", {}).get("keys", {}))[:5]

    overall_r = round(overall)
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "song": {"file": P._get(analysis, "track.file"),
                 "duration_s": P._get(analysis, "track.duration_s"),
                 "lossy": P._get(analysis, "track.lossy_cutoff_hz") is not None,
                 "channels": P._get(analysis, "track.channels")},
        "genre": {"id": profile["profile"]["id"], "name": profile["profile"]["name"],
                  "reference_count": profile["source"]["used_count"],
                  "confidence": profile["quality"]["confidence"]},
        "score": {"overall": overall_r, "verdict": verdict(overall_r),
                  "areas": {a: round(v) for a, v in areas.items()}},
        "missing": [f for f in findings if f["kind"] == "missing"],
        "adjust": [f for f in findings if f["kind"] == "adjust"],
        "problems": problems,
        "next_steps": next_steps,
        "sounds_like": P.nearest_tracks(row, profile),
        "context": context,
        "metrics": metrics,
        "curve": tonal_curve(analysis, profile),
    }


def _merge_band_width(findings: list[dict]) -> list[dict]:
    """One finding for all bands that are too wide, one for too narrow."""
    out, groups = [], {}
    for f in findings:
        if f["metric"].startswith("stereo.width_by_band."):
            groups.setdefault(f["status"], []).append(f)
        else:
            out.append(f)
    for status, fs in groups.items():
        if len(fs) == 1:
            out.append(fs[0])
            continue
        ranges = [BAND_HZ[f["metric"].rsplit(".", 1)[1]] for f in fs]
        where = ", ".join(ranges[:-1]) + " and " + ranges[-1]
        word = "Narrower" if status == "low" else "Wider"
        out.append({**fs[0], "metric": "stereo.width_by_band", "label": "Width by band",
                    "text": f"{word} than the genre at {where}.",
                    "value": None, "target": None,
                    "impact": round(sum(f["impact"] for f in fs), 1)})
    return sorted(out, key=lambda f: -f["impact"])


def verdict(score: float) -> str:
    return next(label for floor, label in READINESS if score >= floor)


def tonal_curve(analysis: dict, profile: dict) -> list[dict]:
    """Song vs genre on the 1/3-octave curve, for drawing the EQ chart."""
    mine = P._third_octave(analysis)
    ref = profile.get("curves", {}).get("third_octave_db")
    if not mine or not ref or mine[0] != ref["freqs_hz"]:
        return []
    out = []
    for hz, v, med, lo, hi in zip(mine[0], mine[1], ref["median"],
                                  ref["p10"], ref["p90"]):
        if v is None or med is None:
            continue
        out.append({"hz": hz, "yours": v, "target": med, "low": lo, "high": hi})
    return out


def check(analysis: dict, all_profiles: list[dict], genre: str | None = None) -> dict:
    """Pick the genre (the closest one when none is given) and compare.

    The report also lists every genre ranked by closeness, so the app can
    suggest a better fit than the one chosen.
    """
    if not all_profiles:
        raise CompareError("No genre profiles yet. Add reference songs first.")
    ranked = P.match(analysis, all_profiles)
    if genre:
        want = genre.strip().lower()
        chosen = [p for p in all_profiles
                  if want in (p["profile"]["name"].lower(), p["profile"]["id"])]
        if not chosen:
            names = ", ".join(p["profile"]["name"] for p in all_profiles)
            raise CompareError(f"No genre called {genre!r}. Genres: {names}")
        profile = chosen[0]
    else:
        if not ranked:
            raise CompareError("nothing in this song could be compared")
        profile = next(p for p in all_profiles if p["profile"]["id"] == ranked[0]["id"])
    report = compare(analysis, profile)
    report["genre"]["chosen_by"] = "you" if genre else "closest match"
    report["closest_genres"] = [{"name": r["name"], "distance": r["distance"]}
                                for r in ranked]
    return report


def _songs(n: int) -> str:
    return f"{n} reference song" + ("" if n == 1 else "s")


def describe(report: dict) -> str:
    """The report as plain text for the terminal."""
    s, g = report["score"], report["genre"]
    lines = [f"{report['song']['file']}  vs  {g['name']} "
             f"({_songs(g['reference_count'])}, chosen by {g.get('chosen_by', 'you')})",
             "",
             f"Score: {s['overall']}/100  {s['verdict']}"]
    for a in AREA_WEIGHTS:
        if a in s["areas"]:
            lines.append(f"  {AREA_LABELS[a]:<16}{s['areas'][a]:>4}")
        else:
            lines.append(f"  {AREA_LABELS[a]:<16}   -  (mono, not scored)")
    if g["confidence"] == "low":
        lines.append(f"  (Rough: this genre has only {_songs(g['reference_count'])}. "
                     f"10 or more make the score reliable.)")

    def block(title, items):
        if items:
            lines.extend(["", title])
            lines.extend(f"  - {f['text']}" for f in items)

    block("Problems", report["problems"])
    block("Missing compared with the genre", report["missing"])
    block("Adjust", report["adjust"])
    if not (report["problems"] or report["missing"] or report["adjust"]):
        lines.extend(["", "Everything sits inside the genre's range."])
    if report["next_steps"]:
        lines.extend(["", "Do these first:"])
        lines.extend(f"  {i}. {t}" for i, t in enumerate(report["next_steps"], 1))

    ctx = report["context"]
    extra = []
    if "bpm" in ctx:
        b = ctx["bpm"]
        note = "" if b["in_range"] else f" (genre: {b['genre_low']:.0f}-{b['genre_high']:.0f})"
        extra.append(f"Tempo {b['value']:.0f} BPM{note}")
    if "key" in ctx:
        extra.append(f"Key {ctx['key']}")
    if report["sounds_like"]:
        extra.append("Sounds closest to: "
                     + ", ".join(t["label"] for t in report["sounds_like"]))
    closest = report.get("closest_genres") or []
    if closest and closest[0]["name"] != g["name"]:
        extra.append(f"Closest genre by sound: {closest[0]['name']}")
    if extra:
        lines.append("")
        lines.extend(extra)
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("analysis", type=Path)
    ap.add_argument("profile", type=Path)
    ap.add_argument("--json", action="store_true", help="print the full report JSON")
    args = ap.parse_args(argv)
    try:
        report = compare(P.load_analysis(args.analysis),
                         json.loads(args.profile.read_text()))
    except (P.ProfileError, CompareError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2) if args.json else describe(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
