"""Run Analyzer App as a web page.

    analyzer app                      start it and open it in your browser
    analyzer app --no-browser         start it without opening the browser
    analyzer app --host 0.0.0.0 --port 8000
                                      let other computers reach it

"Start Analyzer App.command" double-clicks into this.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import webbrowser
from pathlib import Path

from analyzer_app.library import default_home


def run(home: Path | None = None, host: str = "127.0.0.1", port: int = 8765,
        browser: bool = True) -> int:
    home = Path(home) if home else default_home()
    home.mkdir(parents=True, exist_ok=True)
    # librosa's compiled helpers are cached in the library folder.
    os.environ.setdefault("NUMBA_CACHE_DIR", str(home / ".cache" / "numba"))
    from analyzer_app.server import serve

    try:
        server, app = serve(home, host, port)
    except OSError:
        if port == 0:
            raise
        server, app = serve(home, host, 0)   # that port is taken, use any free one
    shown = "127.0.0.1" if host in ("0.0.0.0", "") else host
    url = f"http://{shown}:{server.server_port}/"
    print(f"Analyzer App is running at {url}\n"
          "Keep this window open while you use it. Close it (or press Ctrl+C) to stop.",
          flush=True)
    if browser:
        webbrowser.open(url)
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        return 0
    finally:
        server.shutdown()
        app.close()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="analyzer app", description=__doc__.splitlines()[0])
    ap.add_argument("--home", type=Path, help="library folder (default ~/AnalyzerApp)")
    ap.add_argument("--host", default="127.0.0.1",
                    help="address to listen on (default: this computer only)")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true", help="don't open the browser")
    args = ap.parse_args(argv)
    return run(args.home, args.host, args.port, not args.no_browser)


if __name__ == "__main__":
    sys.exit(main())
