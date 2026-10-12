import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

from analyzer_app import analyze as A
from analyzer_app import profiles as P

SR = A.SAMPLE_RATE


def sine(freq=1000.0, db=-20.0, seconds=10.0):
    t = np.arange(int(SR * seconds)) / SR
    return 10 ** (db / 20) * np.sin(2 * np.pi * freq * t)


def pink(seconds=20.0, seed=0):
    rng = np.random.default_rng(seed)
    n = int(SR * seconds)
    spec = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n, 1 / SR)
    spec[1:] /= np.sqrt(f[1:])
    spec[0] = 0
    x = np.fft.irfft(spec, n)
    return 0.1 * x / np.std(x)


def stereo(left, right=None):
    return np.stack([left, left if right is None else right], axis=1)


def kick_track(bpm=126.0, seconds=30.0):
    """Kick drum on every beat over quiet pink noise."""
    n = int(SR * seconds)
    out = 0.05 * pink(seconds, seed=3)
    t = np.arange(int(0.25 * SR)) / SR
    kick = np.sin(2 * np.pi * (50 + 100 * np.exp(-t * 30)) * t) * np.exp(-t * 12)
    for start in np.arange(0, seconds, 60 / bpm):
        i = int(start * SR)
        seg = kick[: max(0, min(len(kick), n - i))]
        out[i:i + len(seg)] += 0.8 * seg
    return out


class MeasurementTest(unittest.TestCase):
    def test_short_term_matches_integrated_on_steady_signal(self):
        a = A.analyze_audio(stereo(sine()), musical=False)
        self.assertAlmostEqual(a["loudness"]["max_short_term_lufs"],
                               a["loudness"]["integrated_lufs"], delta=0.2)
        self.assertLess(a["dynamics"]["lra_lu"], 0.5)

    def test_true_peak_and_crest(self):
        a = A.analyze_audio(stereo(sine(997, -1.0)), musical=False)
        self.assertAlmostEqual(a["loudness"]["true_peak_dbtp"], -1.0, delta=0.1)
        self.assertAlmostEqual(a["dynamics"]["crest_db"], 3.01, delta=0.1)

    def test_identical_channels_count_as_mono(self):
        a = A.analyze_audio(stereo(pink()), musical=False)
        self.assertEqual(a["track"]["channels"], 1)
        self.assertIsNone(a["stereo"])

    def test_independent_channels_are_wide(self):
        a = A.analyze_audio(stereo(pink(seed=1), pink(seed=2)), musical=False)
        self.assertEqual(a["track"]["channels"], 2)
        self.assertAlmostEqual(a["stereo"]["correlation"], 0.0, delta=0.05)
        self.assertAlmostEqual(a["stereo"]["width"], 1.0, delta=0.1)
        self.assertAlmostEqual(a["stereo"]["balance_db"], 0.0, delta=0.3)

    def test_pink_noise_tilt_and_bands(self):
        a = A.analyze_audio(stereo(pink(seed=1), pink(seed=2)), musical=False)
        self.assertAlmostEqual(a["tonal"]["tilt_db_per_oct"], -3.0, delta=0.4)
        bands = a["tonal"]["bands_db"]
        # Pink noise has equal energy per octave: the 1-octave bands match.
        self.assertAlmostEqual(bands["mid"], bands["upper_mid"], delta=0.5)
        self.assertAlmostEqual(bands["bass"], bands["presence"], delta=0.7)
        self.assertEqual(len(a["tonal"]["third_octave_db"]), 31)

    def test_sine_lands_in_its_band(self):
        a = A.analyze_audio(stereo(sine(700)), musical=False)
        self.assertGreater(a["tonal"]["bands_db"]["mid"], -0.5)
        self.assertAlmostEqual(a["tonal"]["centroid_hz"], 700, delta=30)

    def test_lossy_cutoff_found(self):
        # An encoder's brick-wall cut at 16 kHz.
        n = SR * 20
        spec = np.fft.rfft(pink(seed=4))
        spec[np.fft.rfftfreq(n, 1 / SR) > 16000] = 0
        x = np.fft.irfft(spec, n)
        a = A.analyze_audio(stereo(x, x * 0.9 + 0.001 * pink(seed=6)),
                            lossy=True, musical=False)
        self.assertAlmostEqual(a["track"]["lossy_cutoff_hz"], 16000, delta=800)
        full = A.analyze_audio(stereo(pink(seed=1), pink(seed=2)),
                               lossy=True, musical=False)
        self.assertNotIn("lossy_cutoff_hz", full["track"])

    def test_silence_rejected(self):
        with self.assertRaises(A.AnalysisError):
            A.analyze_audio(np.zeros((SR * 10, 2)), musical=False)

    def test_tempo(self):
        a = A.analyze_audio(stereo(kick_track(126)), musical=True)
        bpm = a["musical"]["bpm"]
        self.assertTrue(min(abs(bpm - 126), abs(bpm - 63), abs(bpm - 252)) < 3, bpm)

    def test_output_feeds_profiles(self):
        a = A.analyze_audio(stereo(pink(seed=1), pink(seed=2)), musical=False)
        row = P.flatten(a)
        self.assertIn("stereo.width_by_band.air", row)
        self.assertIn("dynamics.section_contrast_lu", row)


class FileTest(unittest.TestCase):
    def test_wav_and_mp3(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            wav = d / "song.wav"
            sf.write(wav, stereo(kick_track(124, 20), kick_track(124, 20) * 0.8
                                 + 0.01 * pink(20, seed=9)), SR)
            a = A.analyze_file(wav)
            self.assertEqual(a["track"]["file"], "song.wav")
            self.assertEqual(len(a["track"]["sha1"]), 40)
            cached = A.analyze_cached(wav, d / "cache")
            again = A.analyze_cached(wav, d / "cache")
            self.assertEqual(cached["loudness"], again["loudness"])
            if shutil.which("ffmpeg"):
                mp3 = d / "song.mp3"
                subprocess.run(["ffmpeg", "-loglevel", "error", "-i", str(wav),
                                "-b:a", "128k", str(mp3)], check=True)
                m = A.analyze_file(mp3)
                self.assertIn("lossy_cutoff_hz", m["track"])
                self.assertAlmostEqual(m["loudness"]["integrated_lufs"],
                                       a["loudness"]["integrated_lufs"], delta=0.5)

    def test_rejects_other_files(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "notes.txt"
            p.write_text("hi")
            with self.assertRaises(A.AnalysisError):
                A.analyze_file(p)


if __name__ == "__main__":
    unittest.main()
