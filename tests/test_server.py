import json
import tempfile
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest import mock

import numpy as np

from tests import make_samples
from analyzer_app import profiles as P
from analyzer_app import server as S


class ServerTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        home = Path(self.tmp.name)
        rng = np.random.default_rng(1)
        tracks = [(f"t{i}", make_samples.track(rng, i)) for i in range(12)]
        P.save_profile(P.build_profile("Tech House", tracks), home / "profiles")
        self.server, self.app = S.serve(home)
        self.base = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.app.close()
        self.tmp.cleanup()

    def call(self, path, body=None, token=True, raw=None):
        headers = {"X-Token": self.app.token} if token else {}
        data = raw
        if body is not None:
            data = json.dumps(body).encode()
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(self.base + path, data=data, headers=headers)
        try:
            with urllib.request.urlopen(req) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    def wait(self, job):
        for _ in range(100):
            if job["status"] != "running":
                return job
            time.sleep(0.05)
            job = self.call(f"/api/job/{job['id']}")[1]
        self.fail("job didn't finish")

    def test_page_carries_the_token(self):
        with urllib.request.urlopen(self.base + "/") as r:
            self.assertIn(self.app.token, r.read().decode())

    def test_api_needs_the_token(self):
        self.assertEqual(self.call("/api/library", token=False)[0], 403)
        code, lib = self.call("/api/library")
        self.assertEqual(code, 200)
        self.assertEqual(lib["genres"], [])
        self.assertFalse(lib["has_key"])

    def test_check_scores_an_uploaded_mix(self):
        code, _ = self.call("/api/upload?batch=b1&name=..%2Fmy%20mix.wav", raw=b"audio")
        self.assertEqual(code, 200)
        self.assertTrue((self.app.uploads / "b1" / "my mix.wav").exists())
        song = make_samples.track(np.random.default_rng(99), 50)
        song["track"]["sha1"] = "abc"
        with mock.patch.object(S.analyze, "analyze_file", return_value=song):
            job = self.wait(self.call("/api/check", {"batch": "b1"})[1])
        self.assertEqual(job["status"], "done", job["error"])
        r = job["result"]
        self.assertEqual(r["genre"]["name"], "Tech House")
        self.assertEqual(r["song"]["file"], "my mix.wav")
        self.assertIn(r["feedback"]["source"], ("template", "claude"))
        self.assertTrue(Path(r["saved_to"]).exists())
        self.assertFalse((self.app.uploads / "b1").exists())

        job = self.wait(self.call("/api/recheck", {"sha": "abc", "genre": "Nope"})[1])
        self.assertEqual(job["status"], "error")
        self.assertIn("No genre called", job["error"])

    def test_recheck_unknown_mix(self):
        job = self.wait(self.call("/api/recheck", {"sha": "zzz"})[1])
        self.assertEqual(job["status"], "error")

    def test_bad_batch_and_key(self):
        self.assertEqual(self.call("/api/upload?batch=../x&name=a.wav", raw=b"a")[0], 400)
        code, body = self.call("/api/setkey", {"key": "nope"})
        self.assertEqual(code, 400)
        self.assertIn("sk-ant-", body["error"])


if __name__ == "__main__":
    unittest.main()
