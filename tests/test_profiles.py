import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tests import make_samples
from analyzer_app import profiles as P


def sample_set():
    rng = np.random.default_rng(1)
    return [(f"t{i}", make_samples.track(rng, i)) for i in range(12)]


class FlattenTest(unittest.TestCase):
    def test_reads_every_scalar(self):
        a = make_samples.track(np.random.default_rng(0), 1)
        row = P.flatten(a)
        for key in ["loudness.integrated_lufs", "dynamics.section_contrast_lu",
                    "tonal.bands_db.top", "stereo.width_by_band.air",
                    "low_end.low_width", "musical.bpm"]:
            self.assertIn(key, row)

    def test_mono_drops_stereo(self):
        a = make_samples.track(np.random.default_rng(0), 1)
        a["track"]["channels"] = 1
        row = P.flatten(a)
        self.assertFalse(any(k.startswith("stereo.") for k in row))
        self.assertNotIn("low_end.low_width", row)
        self.assertIn("low_end.sub_to_bass_db", row)

    def test_lossy_drops_bands_above_cutoff(self):
        a = make_samples.track(np.random.default_rng(0), 1)
        a["track"]["lossy_cutoff_hz"] = 16000
        row = P.flatten(a)
        self.assertNotIn("tonal.bands_db.top", row)
        self.assertIn("tonal.bands_db.air", row)
        self.assertNotIn("tonal.centroid_hz", row)

    def test_bad_values_skipped(self):
        a = make_samples.track(np.random.default_rng(0), 1)
        a["loudness"]["true_peak_dbtp"] = None
        a["dynamics"]["crest_db"] = "n/a"
        a["tonal"]["centroid_hz"] = float("nan")
        row = P.flatten(a)
        for key in ["loudness.true_peak_dbtp", "dynamics.crest_db",
                    "tonal.centroid_hz"]:
            self.assertNotIn(key, row)

    def test_section_contrast_ignores_silence(self):
        curve = [-60.0] * 50 + [-12.0] * 50 + [-6.0] * 50
        self.assertAlmostEqual(P.section_contrast(curve), 6.0, places=3)
        self.assertIsNone(P.section_contrast([-60.0] * 100))


class HelpersTest(unittest.TestCase):
    def test_fold_bpm(self):
        self.assertEqual(P.fold_bpm([126, 125, 63, 252, 127]),
                         [126, 125, 126, 126, 127])

    def test_fold_bpm_when_median_is_wrong_octave(self):
        out = P.fold_bpm([87, 174, 175, 173, 86, 88])
        self.assertTrue(all(abs(v - out[1]) < 3 for v in out), out)

    def test_stats_widens_tight_ranges(self):
        s = P.stats([-8.0] * 10, min_half_width=1.0)
        self.assertEqual((s["p10"], s["median"], s["p90"]), (-9.0, -8.0, -7.0))
        self.assertTrue(s["widened"])
        s = P.stats(list(range(20)), min_half_width=1.0)
        self.assertFalse(s["widened"])

    def test_slugify(self):
        self.assertEqual(P.slugify("Drum & Bass"), "drum-bass")
        with self.assertRaises(P.ProfileError):
            P.slugify("!!!")


class BuildTest(unittest.TestCase):
    def test_basic_profile(self):
        prof = P.build_profile("Tech House", sample_set())
        self.assertEqual(prof["profile"]["id"], "tech-house")
        self.assertEqual(prof["source"]["used_count"], 12)
        m = prof["metrics"]["loudness.integrated_lufs"]
        self.assertLess(m["p10"], m["median"])
        self.assertLess(m["median"], m["p90"])
        self.assertEqual(len(prof["curves"]["third_octave_db"]["median"]), 31)
        self.assertEqual(prof["quality"]["confidence"], "ok")

    def test_outlier_left_out_and_reported(self):
        items = sample_set()
        odd = copy.deepcopy(items[0][1])
        odd["loudness"]["integrated_lufs"] = -17.0
        odd["tonal"]["tilt_db_per_oct"] = -7.0
        odd["stereo"]["width"] = 0.7
        items.append(("ambient", odd))
        prof = P.build_profile("Tech House", items)
        self.assertEqual([e["label"] for e in prof["source"]["excluded"]],
                         ["ambient"])
        self.assertEqual(prof["source"]["used_count"], 12)
        kept = P.build_profile("Tech House", items, keep={"ambient"})
        self.assertEqual(kept["source"]["used_count"], 13)

    def test_one_odd_metric_is_not_an_outlier(self):
        items = sample_set()
        items[0][1]["loudness"]["integrated_lufs"] = -16.0
        prof = P.build_profile("Tech House", items)
        self.assertEqual(prof["source"]["excluded"], [])

    def test_too_few_tracks(self):
        with self.assertRaises(P.ProfileError):
            P.build_profile("X", [])

    def test_single_song_profile(self):
        prof = P.build_profile("X", sample_set()[:1])
        m = prof["metrics"]["loudness.integrated_lufs"]
        self.assertEqual(m["n"], 1)
        self.assertAlmostEqual(m["p90"] - m["p10"], 2.0, places=3)
        self.assertEqual(prof["quality"]["confidence"], "low")

    def test_small_set_is_low_confidence(self):
        prof = P.build_profile("X", sample_set()[:4])
        self.assertEqual(prof["quality"]["confidence"], "low")
        self.assertEqual(prof["source"]["excluded"], [])

    def test_mono_tracks_only_affect_stereo_counts(self):
        items = sample_set()
        for _, a in items[:3]:
            a["track"]["channels"] = 1
        prof = P.build_profile("X", items)
        self.assertEqual(prof["metrics"]["stereo.width"]["n"], 9)
        self.assertEqual(prof["metrics"]["loudness.integrated_lufs"]["n"], 12)


def ambient_set(n=12, seed=2):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        a = make_samples.track(rng, 100 + i)
        a["loudness"]["integrated_lufs"] = round(rng.normal(-15, 0.8), 2)
        a["tonal"]["tilt_db_per_oct"] = round(rng.normal(-6.0, 0.3), 2)
        a["stereo"]["width"] = round(rng.normal(0.6, 0.05), 3)
        a["dynamics"]["crest_db"] = round(rng.normal(15, 0.8), 2)
        for b in ("sub", "bass"):
            a["tonal"]["bands_db"][b] -= 5
        a["low_end"]["sub_to_bass_db"] = round(rng.normal(-6, 1), 2)
        out.append((f"amb{i}", a))
    return out


class MatchTest(unittest.TestCase):
    def setUp(self):
        self.house = P.build_profile("Tech House", sample_set(), kind="builtin")
        self.ambient = P.build_profile("Ambient", ambient_set(), kind="builtin")

    def test_suggests_the_right_genre(self):
        song = make_samples.track(np.random.default_rng(99), 1)
        ranked = P.match(song, [self.ambient, self.house])
        self.assertEqual([r["id"] for r in ranked], ["tech-house", "ambient"])
        self.assertLess(ranked[0]["distance"], 1.5)
        self.assertEqual(len(ranked[0]["nearest_tracks"]), 3)

        song = ambient_set(1, seed=50)[0][1]
        self.assertEqual(P.match(song, [self.house, self.ambient])[0]["id"],
                         "ambient")

    def test_nearest_track_is_itself(self):
        label, a = sample_set()[5]
        near = P.nearest_tracks(P.flatten(a), self.house, k=1)
        self.assertEqual(near[0]["label"], label)
        self.assertEqual(near[0]["distance"], 0.0)

    def test_kind_validated(self):
        self.assertEqual(self.house["profile"]["kind"], "builtin")
        with self.assertRaises(P.ProfileError):
            P.build_profile("X", sample_set(), kind="other")

    def test_custom_profile_overrides_builtin(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            P.save_profile(self.house, d / "builtin")
            custom = P.build_profile("Tech House", sample_set()[:5])
            P.save_profile(custom, d / "mine")
            loaded = P.load_profiles(d / "builtin", d / "mine")
            self.assertEqual(len(loaded), 1)
            self.assertEqual(loaded[0]["profile"]["kind"], "custom")


class FilesTest(unittest.TestCase):
    def test_end_to_end(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            make_samples.main(d / "in")
            (d / "in" / "broken.json").write_text("{not json")
            (d / "in" / "future.json").write_text('{"schema_version": 9}')
            prof = P.build_from_dir("Tech House", d / "in")
            path = P.save_profile(prof, d / "out")
            self.assertEqual(path.name, "tech-house.json")
            self.assertEqual([e["label"] for e in prof["source"]["excluded"]],
                             ["ref_13"])
            self.assertTrue(any("2 files skipped" in n
                                for n in prof["quality"]["notes"]))
            self.assertAlmostEqual(prof["metrics"]["musical.bpm"]["median"],
                                   126, delta=2)
            self.assertEqual(P.main(["show", str(path)]), 0)


if __name__ == "__main__":
    unittest.main()
