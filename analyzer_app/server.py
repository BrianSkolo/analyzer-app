"""The web server behind the app's page (ui/index.html).

By default it only answers this computer, at http://127.0.0.1:8765/. Songs
are sent to it as file uploads, saved to a temporary folder, and handed to
the same functions the command line uses. Slow work (measuring a song,
adding references, A&R notes) runs as a background job the page polls.

    POST /api/upload?batch=B&name=N    save one file into upload batch B
    POST /api/check     {batch, genre}  measure the mix in B and score it
    POST /api/recheck   {sha, genre}    score the same mix against another genre
    POST /api/add       {batch, genre}  add the songs in B to the library
    POST /api/move      {song, genre}
    POST /api/remove    {song}
    POST /api/setkey    {key}
    GET  /api/library                   genres, songs, profiles, key saved
    GET  /api/job/<id>                  a job's progress and result

Every /api request must carry the token printed into the page, so other
web pages open in the same browser can't use the server.
"""

from __future__ import annotations

import json
import secrets
import shutil
import tempfile
import threading
import traceback
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from analyzer_app import analyze, compare, feedback, profiles
from analyzer_app.library import UNSORTED, Library

UI_DIR = Path(__file__).parent / "ui"
MAX_UPLOAD = 2 * 1024 ** 3
EXPECTED = (LookupError, ValueError, analyze.AnalysisError, profiles.ProfileError,
            compare.CompareError, feedback.FeedbackError)


class Job:
    def __init__(self):
        self.id = secrets.token_hex(8)
        self.status = "running"   # running, done, error
        self.step = ""
        self.result = None
        self.error = None

    def as_dict(self) -> dict:
        return {"id": self.id, "status": self.status, "step": self.step,
                "result": self.result, "error": self.error}


class App:
    """State shared by every request: the library, uploads, jobs."""

    def __init__(self, home: Path | None = None):
        self.lib = Library(home)
        self.token = secrets.token_urlsafe(24)
        self.uploads = Path(tempfile.mkdtemp(prefix="analyzer-uploads-"))
        self.jobs: dict[str, Job] = {}
        self.analyses: dict[str, dict] = {}   # sha1 -> analysis of a checked mix
        self.lock = threading.Lock()

    def close(self) -> None:
        shutil.rmtree(self.uploads, ignore_errors=True)

    # Uploads

    def batch_dir(self, batch: str) -> Path:
        if not batch or not batch.isalnum() or len(batch) > 40:
            raise ValueError("bad upload batch")
        return self.uploads / batch

    def save_upload(self, batch: str, name: str, stream, length: int) -> None:
        name = Path(name).name
        if not name or name.startswith("."):
            raise ValueError("bad file name")
        folder = self.batch_dir(batch)
        folder.mkdir(parents=True, exist_ok=True)
        dest = folder / name
        n = 2
        while dest.exists():
            dest = folder / f"{Path(name).stem} ({n}){Path(name).suffix}"
            n += 1
        left = length
        with open(dest, "wb") as f:
            while left > 0:
                chunk = stream.read(min(left, 1 << 20))
                if not chunk:
                    raise ValueError("upload ended early")
                f.write(chunk)
                left -= len(chunk)

    def take_batch(self, batch: str) -> list[Path]:
        folder = self.batch_dir(batch)
        if not folder.is_dir():
            raise ValueError("no files were received")
        return sorted(folder.iterdir())

    # Jobs

    def start(self, work) -> Job:
        job = Job()
        with self.lock:
            self.jobs[job.id] = job

        def run():
            try:
                job.result = work(job)
                job.status = "done"
            except EXPECTED as e:
                job.error, job.status = str(e), "error"
            except Exception as e:   # shown in the window, and logged
                traceback.print_exc()
                job.error, job.status = f"Something went wrong: {e}", "error"

        threading.Thread(target=run, daemon=True).start()
        return job

    # Work

    def check(self, job: Job, batch: str, genre: str | None) -> dict:
        files = self.take_batch(batch)
        mix = files[0]
        try:
            job.step = "Measuring your mix"
            a = analyze.analyze_file(mix)
        finally:
            shutil.rmtree(mix.parent, ignore_errors=True)
        sha = a["track"].get("sha1") or secrets.token_hex(20)
        a["track"]["file"] = mix.name
        with self.lock:
            self.analyses[sha] = a
        return self._report(job, sha, genre)

    def recheck(self, job: Job, sha: str, genre: str | None) -> dict:
        if sha not in self.analyses:
            raise LookupError("Drop the mix in again to check it.")
        return self._report(job, sha, genre)

    def _report(self, job: Job, sha: str, genre: str | None) -> dict:
        a = self.analyses[sha]
        job.step = "Comparing with the genre"
        report = compare.check(a, self.lib.all_profiles(), genre or None)
        job.step = "Writing A&R notes"
        report["feedback"] = feedback.feedback(report, self.lib.home)
        saved = self.lib.save_report(report, Path(a["track"]["file"]).stem)
        report["sha"] = sha
        report["saved_to"] = str(saved)
        return report

    def add(self, job: Job, batch: str, genre: str | None) -> dict:
        files = self.take_batch(batch)

        def progress(i, n, name):
            job.step = f"Measuring song {i} of {n}: {name}"

        try:
            r = self.lib.add(files, genre=genre or None, progress=progress)
        finally:
            shutil.rmtree(self.batch_dir(batch), ignore_errors=True)
        return {"added": r.added, "duplicates": r.duplicates,
                "failed": r.failed, "rebuilt": r.rebuilt}

    def library(self) -> dict:
        genres = []
        for g in self.lib.genres():
            p = self.lib.profile(g)
            genres.append({"name": g, "songs": [s.name for s in self.lib.songs(g)],
                           "confidence": p["quality"]["confidence"] if p else None,
                           "unsorted": g == UNSORTED})
        return {"genres": genres, "home": str(self.lib.home),
                "has_key": bool(feedback.api_key(self.lib.home))}

    def move(self, song: str, genre: str) -> dict:
        dest = self.lib.move(song, genre)
        return {"moved_to": dest.parent.name}

    def remove(self, song: str) -> dict:
        self.lib.remove(song)
        return {"removed": song}


def make_handler(app: App):
    class Handler(BaseHTTPRequestHandler):
        server_version = "AnalyzerApp"

        def log_message(self, fmt, *args):   # keep the log quiet
            pass

        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, data, code: int = 200) -> None:
            self._send(code, json.dumps(data).encode(), "application/json")

        def _error(self, message: str, code: int = 400) -> None:
            self._json({"error": message}, code)

        def _allowed(self) -> bool:
            if self.headers.get("X-Token") != app.token:
                self._error("forbidden", HTTPStatus.FORBIDDEN)
                return False
            return True

        def _body(self) -> dict:
            n = int(self.headers.get("Content-Length") or 0)
            return json.loads(self.rfile.read(n) or b"{}") if n else {}

        def do_GET(self):
            url = urlparse(self.path)
            if url.path in ("/", "/index.html"):
                page = (UI_DIR / "index.html").read_text()
                page = page.replace("__TOKEN__", app.token)
                self._send(200, page.encode(), "text/html; charset=utf-8")
                return
            if not url.path.startswith("/api/"):
                self._error("not found", 404)
                return
            if not self._allowed():
                return
            if url.path == "/api/library":
                self._json(app.library())
            elif url.path.startswith("/api/job/"):
                job = app.jobs.get(url.path.rsplit("/", 1)[1])
                if job:
                    self._json(job.as_dict())
                else:
                    self._error("no such job", 404)
            else:
                self._error("not found", 404)

        def do_POST(self):
            url = urlparse(self.path)
            if not self._allowed():
                return
            try:
                if url.path == "/api/upload":
                    q = parse_qs(url.query)
                    n = int(self.headers.get("Content-Length") or 0)
                    if n > MAX_UPLOAD:
                        raise ValueError("That file is too big.")
                    app.save_upload(q.get("batch", [""])[0], q.get("name", [""])[0],
                                    self.rfile, n)
                    self._json({"ok": True})
                    return
                body = self._body()
                if url.path == "/api/check":
                    job = app.start(lambda j: app.check(j, body.get("batch", ""),
                                                        body.get("genre")))
                elif url.path == "/api/recheck":
                    job = app.start(lambda j: app.recheck(j, body.get("sha", ""),
                                                          body.get("genre")))
                elif url.path == "/api/add":
                    job = app.start(lambda j: app.add(j, body.get("batch", ""),
                                                      body.get("genre")))
                elif url.path == "/api/move":
                    self._json(app.move(body["song"], body["genre"]))
                    return
                elif url.path == "/api/remove":
                    self._json(app.remove(body["song"]))
                    return
                elif url.path == "/api/setkey":
                    feedback.save_api_key(app.lib.home, body.get("key", ""))
                    self._json({"ok": True})
                    return
                else:
                    self._error("not found", 404)
                    return
                self._json(job.as_dict())
            except (KeyError, json.JSONDecodeError):
                self._error("bad request")
            except EXPECTED as e:
                self._error(str(e))

    return Handler


def serve(home: Path | None = None, host: str = "127.0.0.1",
          port: int = 0) -> tuple[ThreadingHTTPServer, App]:
    """Start the server in a background thread. Returns (server, app); the
    page is at http://<host>:<server.server_port>/."""
    server = ThreadingHTTPServer((host, port), BaseHTTPRequestHandler)
    app = App(home)
    server.RequestHandlerClass = make_handler(app)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, app
