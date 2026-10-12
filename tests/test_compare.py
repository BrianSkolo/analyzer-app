import copy
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from tests import make_samples
from analyzer_app import compare as C
from analyzer_app import feedback as F
from analyzer_app import profiles as P


def genre(name="Tech House", seed=1, **over):
    rng = np.random.default_rng(seed)
    tracks = [(f"t{i}", make_samples.track(rng, i, **over)) for i in range(12)]
    return P.build_profile(name, tracks)


def song(**over):
    return make_samples.track(np.random.default_rng(99), 50, **over)


class CompareTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.profile = genre()

    def test_reference_like_song_scores_high(self):
        r = C.compare(song(), self.profile)
        self.assertGreaterEqual(r["score"]["overall"], 85)
        self.assertEqual(set(r["score"]["areas"]), set(C.AREA_WEIGHTS))
        self.assertEqual(r["score"]["verdict"], "Release ready")

    def test_dull_quiet_song_gets_fixes(self):
        s = song(**{"loudness.integrated_lufs": -13.0,
                    "tonal.bands_db.air": -34.0, "tonal.bands_db.top": -41.0,
                    "tonal.tilt_db_per_oct": -6.0})
        r = C.compare(s, self.profile)
        good = C.compare(song(), self.profile)
        self.assertLess(r["score"]["overall"], good["score"]["overall"] - 10)
        self.assertLess(r["score"]["areas"]["tone"], good["score"]["areas"]["tone"])
        missing = {f["metric"] for f in r["missing"]}
        self.assertIn("tonal.bands_db.air", missing)
        texts = " ".join(f["text"] for f in r["missing"] + r["adjust"])
        self.assertIn("Boost 8-12 kHz by about", texts)
        self.assertIn("Raise the limiter", texts)
        self.assertEqual(len(r["next_steps"]), 3)

    def test_technical_problems_come_first(self):
        s = song(**{"loudness.true_peak_dbtp": 0.6, "stereo.low_correlation": 0.2})
        r = C.compare(s, self.profile)
        labels = [p["label"] for p in r["problems"]]
        self.assertIn("Clipping", labels)
        self.assertIn("Bass phase", labels)
        self.assertIn("clips", r["next_steps"][0] + r["next_steps"][1])

    def test_mono_song_skips_stereo(self):
        r = C.compare(song(**{"track.channels": 1}), self.profile)
        self.assertNotIn("stereo", r["score"]["areas"])
        self.assertFalse(any(m["area"] == "stereo" for m in r["metrics"]))

    def test_half_time_tempo_is_folded(self):
        r = C.compare(song(**{"musical.bpm": 63.0}), self.profile)
        self.assertAlmostEqual(r["context"]["bpm"]["value"], 126.0)

    def test_metric_score_shape(self):
        s = {"median": 0.0, "p10": -1.0, "p90": 1.0}
        self.assertEqual(C.metric_score(0.0, s), (100.0, 0.0))
        inside, _ = C.metric_score(0.9, s)
        just_out, d1 = C.metric_score(1.5, s)
        far, d2 = C.metric_score(4.0, s)
        self.assertTrue(90 <= inside < 100)
        self.assertTrue(far < just_out < 90)
        self.assertTrue(0 < d1 < d2)

    def test_curve_for_chart(self):
        r = C.compare(song(), self.profile)
        self.assertTrue(r["curve"])
        self.assertEqual(set(r["curve"][0]), {"hz", "yours", "target", "low", "high"})

    def test_check_picks_closest_or_named_genre(self):
        dark = genre("Melodic Techno", seed=2, **{"tonal.tilt_db_per_oct": -6.5,
                                                   "stereo.width": 0.55,
                                                   "loudness.integrated_lufs": -10.5})
        r = C.check(song(), [dark, self.profile])
        self.assertEqual(r["genre"]["name"], "Tech House")
        self.assertEqual(r["genre"]["chosen_by"], "closest match")
        r = C.check(song(), [dark, self.profile], genre="melodic techno")
        self.assertEqual(r["genre"]["name"], "Melodic Techno")
        self.assertEqual(r["closest_genres"][0]["name"], "Tech House")
        self.assertIn("Closest genre by sound: Tech House", C.describe(r))
        with self.assertRaises(C.CompareError):
            C.check(song(), [self.profile], genre="Polka")
        with self.assertRaises(C.CompareError):
            C.check(song(), [])


class FeedbackTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.home = Path(self.tmp.name)
        self.report = C.compare(song(**{"loudness.true_peak_dbtp": 0.6,
                                        "tonal.bands_db.air": -34.0}), genre())

    def tearDown(self):
        self.tmp.cleanup()

    def test_template_without_key(self):
        with mock.patch.dict("os.environ", {}, clear=True):
            fb = F.feedback(self.report, self.home)
        self.assertEqual(fb["source"], "template")
        self.assertIn("Next steps", fb["text"])
        self.assertIn("clipping", fb["text"])

    def test_key_saved_privately(self):
        with self.assertRaises(F.FeedbackError):
            F.save_api_key(self.home, "not-a-key")
        path = F.save_api_key(self.home, " sk-ant-test ")
        self.assertEqual(path.stat().st_mode & 0o777, 0o600)
        with mock.patch.dict("os.environ", {}, clear=True):
            self.assertEqual(F.api_key(self.home), "sk-ant-test")

    def test_claude_used_and_falls_back(self):
        F.save_api_key(self.home, "sk-ant-test")
        with mock.patch.dict("os.environ", {}, clear=True):
            with mock.patch.object(F, "claude_feedback", return_value="Great groove."):
                fb = F.feedback(self.report, self.home)
            self.assertEqual(fb, {"text": "Great groove.", "source": "claude", "note": None})
            with mock.patch.object(F, "claude_feedback",
                                   side_effect=F.FeedbackError("offline")):
                fb = F.feedback(self.report, self.home)
        self.assertEqual((fb["source"], fb["note"]), ("template", "offline"))

    def test_only_numbers_are_sent(self):
        sent = F._for_claude(self.report)
        self.assertNotIn('"curve"', sent)
        self.assertIn('"next_steps"', sent)


if __name__ == "__main__":
    unittest.main()
