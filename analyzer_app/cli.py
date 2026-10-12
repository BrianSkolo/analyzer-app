"""Command line for Analyzer App.

    analyzer add ~/Downloads/new-tracks          file songs by their genre tag
    analyzer add song.mp3 --genre "Nu Disco"     or choose the genre yourself
    analyzer genres                              genres, song counts, profiles
    analyzer songs "Indie Dance"                 songs in one genre
    analyzer move "Grandline" "Melodic House & Techno"
    analyzer remove "Grandline"
    analyzer match my_mix.wav                    which genre your song is closest to
    analyzer check my_mix.wav                    score your mix against the closest genre
    analyzer check my_mix.wav --genre "Indie Dance"
    analyzer setkey sk-ant-...                   turn on written A&R feedback from Claude
    analyzer app                                 open the app in your browser
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from analyzer_app import analyze, compare, feedback, profiles
from analyzer_app.library import UNSORTED, Library


def _progress(i: int, n: int, name: str) -> None:
    print(f"  [{i}/{n}] {name}", flush=True)


def cmd_add(lib: Library, args) -> int:
    print(f"Adding to {lib.references}")
    r = lib.add(args.paths, genre=args.genre, progress=_progress)
    by_genre: dict[str, int] = {}
    for _, g in r.added:
        by_genre[g] = by_genre.get(g, 0) + 1
    for g, n in sorted(by_genre.items()):
        print(f"Added {n} to {g}")
    if r.duplicates:
        print(f"Skipped {len(r.duplicates)} already in the library")
    for name, why in r.failed:
        print(f"Couldn't add {name}: {why}")
    if UNSORTED in by_genre:
        print(f'Songs with no genre tag are in "{UNSORTED}". File them with: '
              f'analyzer move "<song>" "<genre>"')
    return 0 if r.added or r.duplicates else 1


def cmd_genres(lib: Library, args) -> int:
    genres = lib.genres()
    if not genres:
        print("The library is empty. Add songs with: analyzer add <files or folder>")
        return 0
    for g in genres:
        n = len(lib.songs(g))
        p = lib.profile(g)
        status = (f"profile: {p['quality']['confidence']} confidence"
                  if p else "no profile")
        print(f"{g:<32}{n:>3} songs   {status}")
    return 0


def cmd_songs(lib: Library, args) -> int:
    songs = lib.songs(args.genre)
    if not songs:
        print(f"No songs in {args.genre!r}. Genres: {', '.join(lib.genres())}")
        return 1
    for s in songs:
        print(s.name)
    return 0


def cmd_move(lib: Library, args) -> int:
    dest = lib.move(args.song, args.genre)
    print(f"Moved to {dest.parent.name}: {dest.name}")
    return 0


def cmd_remove(lib: Library, args) -> int:
    path = lib.remove(args.song)
    print(f"Removed {path.name} from {path.parent.name}")
    return 0


def cmd_rebuild(lib: Library, args) -> int:
    if args.genre:
        lib.rebuild(args.genre)
        built = [args.genre]
    else:
        built = lib.rebuild_all()
    print("Rebuilt: " + (", ".join(built) or "nothing"))
    return 0


def cmd_profile(lib: Library, args) -> int:
    p = lib.profile(args.genre)
    if not p:
        print(f"No profile for {args.genre!r}")
        return 1
    print(profiles.describe(p))
    return 0


def cmd_match(lib: Library, args) -> int:
    a = analyze.analyze_file(args.song)
    ranked = profiles.match(a, lib.all_profiles())
    if not ranked:
        print("No genre profiles yet. Add reference songs first.")
        return 1
    for r in ranked:
        near = ", ".join(t["label"] for t in r["nearest_tracks"])
        print(f"{r['name']:<32}distance {r['distance']:.2f}   sounds like: {near}")
    return 0


def cmd_check(lib: Library, args) -> int:
    print(f"Measuring {args.song.name}...", flush=True)
    a = analyze.analyze_file(args.song)
    report = compare.check(a, lib.all_profiles(), args.genre)
    print()
    print(compare.describe(report))
    if not args.no_feedback:
        fb = report["feedback"] = feedback.feedback(report, lib.home)
        print()
        if fb["source"] == "claude":
            print("A&R notes")
        elif fb["note"] == "no Claude API key saved":
            print("A&R notes (basic version; for notes written by Claude: analyzer setkey <key>)")
        else:
            print(f"A&R notes (basic version, because {fb['note']})")
        print(fb["text"])
    out = lib.save_report(report, args.song.stem)
    print(f"\nFull report saved to {out}")
    return 0


def cmd_setkey(lib: Library, args) -> int:
    path = feedback.save_api_key(lib.home, args.key)
    print(f"Saved. A&R notes will now be written by Claude. (Key stored in {path})")
    return 0


def cmd_app(lib: Library, args) -> int:
    from analyzer_app import app
    return app.run(lib.home, args.host, args.port, not args.no_browser)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="analyzer", description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog="\n".join(__doc__.splitlines()[2:]))
    ap.add_argument("--home", type=Path, help="library folder (default ~/AnalyzerApp)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("add", help="add reference songs (files or folders)")
    p.add_argument("paths", type=Path, nargs="+")
    p.add_argument("--genre", help="file every song under this genre")
    p.set_defaults(func=cmd_add)

    sub.add_parser("genres", help="list genres").set_defaults(func=cmd_genres)

    p = sub.add_parser("songs", help="list songs in a genre")
    p.add_argument("genre")
    p.set_defaults(func=cmd_songs)

    p = sub.add_parser("move", help="move a song to another genre")
    p.add_argument("song", help="part of the song's file name")
    p.add_argument("genre")
    p.set_defaults(func=cmd_move)

    p = sub.add_parser("remove", help="remove a song from the library")
    p.add_argument("song", help="part of the song's file name")
    p.set_defaults(func=cmd_remove)

    p = sub.add_parser("rebuild", help="rebuild genre profiles")
    p.add_argument("genre", nargs="?")
    p.set_defaults(func=cmd_rebuild)

    p = sub.add_parser("profile", help="show a genre's profile")
    p.add_argument("genre")
    p.set_defaults(func=cmd_profile)

    p = sub.add_parser("match", help="find the closest genre for a song")
    p.add_argument("song", type=Path)
    p.set_defaults(func=cmd_match)

    p = sub.add_parser("check", help="score a mix against a genre")
    p.add_argument("song", type=Path)
    p.add_argument("--genre", help="genre to compare with (default: the closest)")
    p.add_argument("--no-feedback", action="store_true",
                   help="skip the written A&R notes")
    p.set_defaults(func=cmd_check)

    p = sub.add_parser("setkey", help="save your Claude API key for A&R notes")
    p.add_argument("key")
    p.set_defaults(func=cmd_setkey)

    p = sub.add_parser("app", help="open the app in your browser")
    p.add_argument("--host", default="127.0.0.1",
                   help="address to listen on (default: this computer only)")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--no-browser", action="store_true", help="don't open the browser")
    p.set_defaults(func=cmd_app)

    args = ap.parse_args(argv)
    lib = Library(args.home)
    try:
        return args.func(lib, args)
    except (LookupError, analyze.AnalysisError, profiles.ProfileError,
            compare.CompareError, feedback.FeedbackError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
