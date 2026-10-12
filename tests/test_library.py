import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from analyzer_app import cli
from analyzer_app.library import UNSORTED, Library
from analyzer_app.tags import parse_key, read_tags
from tests.test_analyze import SR, kick_track, pink, stereo


class KeyTagTest(unittest.TestCase):
    def test_notations(self):
        cases = {"11m": "G minor", "1m": "A minor", "1d": "C major",
                 "12m": "D minor", "8A": "A minor", "8B": "C major",
                 "6A": "G minor", "Gm": "G minor", "Bbmaj": "A# major",
                 "F# minor": "F# minor", "Ebm": "D# minor", "C": "C major"}
        for text, want in cases.items():
            self.assertEqual(parse_key(text), want, text)
        for bad in ["", None, "13m", "H", "Cb", "unknown"]:
            self.assertIsNone(parse_key(bad), bad)


def make_mp3(folder: Path, name: str, bpm: float, **tags) -> Path:
    wav = folder / f"{name}.wav"
    x = kick_track(bpm, 8)
    sf.write(wav, stereo(x, 0.9 * x + 0.02 * pink(8, seed=len(name))), SR)
    mp3 = folder / f"{name}.mp3"
    meta = []
    for k, v in tags.items():
        meta += ["-metadata", f"{k}={v}"]
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(wav),
                    "-b:a", "320k", *meta, str(mp3)], check=True)
    wav.unlink()
    return mp3


@unittest.skipUnless(shutil.which("ffmpeg"), "needs ffmpeg to make test MP3s")
class LibraryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        src = Path(cls.tmp.name) / "downloads"
        src.mkdir()
        make_mp3(src, "a", 124, artist="Artist A", title="One",
                 genre="Indie Dance", TBPM="124", TKEY="11m")
        make_mp3(src, "b", 122, artist="Artist B", title="Two",
                 genre="Indie Dance")
        make_mp3(src, "c", 128, artist="Artist C", title="Three",
                 genre="Melodic House & Techno")
        make_mp3(src, "d", 126, title="No genre")
        cls.src = src

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def setUp(self):
        self.home = Path(tempfile.mkdtemp(dir=self.tmp.name))
        self.lib = Library(self.home)

    def test_reads_tags(self):
        t = read_tags(self.src / "a.mp3")
        self.assertEqual((t["artist"], t["title"], t["genre"], t["bpm"], t["key"]),
                         ("Artist A", "One", "Indie Dance", 124.0, "G minor"))

    def test_add_files_by_genre_tag(self):
        r = self.lib.add([self.src])
        self.assertEqual(len(r.added), 4)
        self.assertEqual(self.lib.genres(),
                         ["Indie Dance", "Melodic House & Techno", UNSORTED])
        self.assertEqual([p.name for p in self.lib.songs("Indie Dance")],
                         ["Artist A - One.mp3", "Artist B - Two.mp3"])
        self.assertEqual(sorted(r.rebuilt), ["Indie Dance", "Melodic House & Techno"])
        prof = self.lib.profile("Indie Dance")
        self.assertEqual(prof["source"]["used_count"], 2)
        self.assertIsNone(self.lib.profile(UNSORTED))
        # The original downloads are left where they were.
        self.assertTrue((self.src / "a.mp3").exists())

    def test_duplicates_skipped(self):
        self.lib.add([self.src / "a.mp3"])
        r = self.lib.add([self.src / "a.mp3"])
        self.assertEqual(r.duplicates, ["a.mp3"])
        self.assertEqual(len(self.lib.songs("Indie Dance")), 1)

    def test_genre_override_move_and_remove(self):
        self.lib.add([self.src / "d.mp3"], genre="Nu Disco")
        self.assertEqual(self.lib.genres(), ["Nu Disco"])
        self.lib.add([self.src / "a.mp3"])
        self.lib.move("No genre", "Indie Dance")
        self.assertEqual(self.lib.genres(), ["Indie Dance"])
        self.assertIsNone(self.lib.profile("Nu Disco"))
        self.assertEqual(self.lib.profile("Indie Dance")["source"]["used_count"], 2)
        self.lib.remove("One")
        self.assertEqual(self.lib.profile("Indie Dance")["source"]["used_count"], 1)
        with self.assertRaises(LookupError):
            self.lib.find("nothing like this")

    def test_cli(self):
        home = ["--home", str(self.home)]
        self.assertEqual(cli.main(home + ["add", str(self.src / "a.mp3"),
                                          str(self.src / "c.mp3")]), 0)
        self.assertEqual(cli.main(home + ["genres"]), 0)
        self.assertEqual(cli.main(home + ["profile", "Indie Dance"]), 0)
        self.assertEqual(cli.main(home + ["match", str(self.src / "b.mp3")]), 0)
        self.assertEqual(cli.main(home + ["move", "nope", "X"]), 1)


if __name__ == "__main__":
    unittest.main()
