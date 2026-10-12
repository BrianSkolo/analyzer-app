"""Write made-up analysis JSON files for trying the profile builder.

The numbers imitate a modern tech house reference set, plus three awkward
cases: an ambient track that doesn't belong, a 128 kbps MP3 and a mono file.
They are not measurements of real songs.

    python -m tests.make_samples samples/tech-house
"""

import json
import sys
from pathlib import Path

import numpy as np

from analyzer_app.profiles import BANDS, THIRD_OCTAVE_HZ

BASE_BANDS = {"sub": -9.0, "bass": -7.2, "low_mid": -10.5, "mid_low": -13.2,
              "mid": -15.4, "upper_mid": -18.0, "presence": -20.3,
              "high": -23.1, "air": -27.6, "top": -33.5}


def track(rng, i, **over):
    lufs = rng.normal(-7.8, 0.9)
    tp = min(rng.normal(-0.6, 0.3), 0.3)
    bands = {b: round(v + rng.normal(0, 0.9), 2) for b, v in BASE_BANDS.items()}
    curve = list(np.round(np.concatenate([
        rng.normal(-14, 1.0, 30),            # intro
        rng.normal(lufs - 0.5, 0.6, 60),     # groove
        rng.normal(-13, 1.2, 25),            # break
        rng.normal(lufs + 0.5, 0.5, 60),     # drop
        np.linspace(-12, -45, 20),           # outro fade
    ]), 2))
    third = list(np.round(np.interp(np.log2(THIRD_OCTAVE_HZ),
                                    np.log2([40, 90, 185, 375, 750, 1500,
                                             3000, 6000, 10000, 16000]),
                                    [bands[b] for b in BANDS]) - 4.7, 2))
    a = {
        "schema_version": 1,
        "track": {"file": f"ref_{i:02d}.wav", "duration_s": 360.0,
                  "sample_rate": 44100, "channels": 2},
        "loudness": {"integrated_lufs": round(lufs, 2),
                     "max_short_term_lufs": round(lufs + rng.normal(1.6, 0.3), 2),
                     "true_peak_dbtp": round(tp, 2)},
        "dynamics": {"lra_lu": round(abs(rng.normal(5.0, 1.0)), 2),
                     "crest_db": round(rng.normal(9.5, 0.8), 2),
                     "plr_db": round(tp - lufs, 2),
                     "short_term_curve_lufs": curve},
        "tonal": {"bands_db": bands, "third_octave_db": third,
                  "tilt_db_per_oct": round(rng.normal(-4.5, 0.3), 2),
                  "centroid_hz": round(rng.normal(2100, 180))},
        "stereo": {"width": round(rng.normal(0.30, 0.05), 3),
                   "width_by_band": {b: round(max(0.0, rng.normal(w, 0.04)), 3)
                                     for b, w in zip(BANDS, [0.01, 0.02, 0.1, 0.2,
                                                             0.3, 0.35, 0.4, 0.45,
                                                             0.5, 0.5])},
                   "correlation": round(rng.normal(0.72, 0.06), 3),
                   "low_correlation": round(min(1.0, rng.normal(0.97, 0.015)), 3),
                   "balance_db": round(rng.normal(0, 0.2), 2)},
        "low_end": {"sub_to_bass_db": round(bands["sub"] - bands["bass"], 2),
                    "low_width": round(abs(rng.normal(0.03, 0.01)), 3)},
        "musical": {"bpm": round(rng.normal(126, 1.5), 1),
                    "key": rng.choice(["F minor", "G minor", "A minor",
                                       "D minor", "C minor"])},
    }
    for path, value in over.items():
        node = a
        *parents, leaf = path.split(".")
        for p in parents:
            node = node[p]
        node[leaf] = value
    return a


def main(out: Path):
    rng = np.random.default_rng(7)
    out.mkdir(parents=True, exist_ok=True)
    tracks = [track(rng, i) for i in range(1, 13)]
    tracks[3]["musical"]["bpm"] = 63.2                     # half-time guess
    tracks.append(track(rng, 13, **{                        # ambient, wrong genre
        "loudness.integrated_lufs": -16.5, "dynamics.crest_db": 16.0,
        "tonal.tilt_db_per_oct": -6.2, "low_end.sub_to_bass_db": -9.0,
        "stereo.width": 0.62, "musical.bpm": 84.0}))
    tracks.append(track(rng, 14, **{"track.lossy_cutoff_hz": 16000,
                                    "track.file": "ref_14.mp3"}))
    tracks.append(track(rng, 15, **{"track.channels": 1}))
    for a in tracks:
        name = Path(a["track"]["file"]).stem + ".json"
        (out / name).write_text(json.dumps(a, indent=2))
    print(f"Wrote {len(tracks)} sample analyses to {out}")


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "samples/tech-house"))
