"""The reference library: songs filed by genre, and a profile per genre.

Layout (the home folder defaults to ~/AnalyzerApp, or $ANALYZER_HOME):

    references/<Genre>/<Artist - Title>.mp3   songs, filed by their genre tag
    cache/<sha1>.json                          one analysis per song
    profiles/<genre-id>.json                   rebuilt whenever a genre changes

Adding songs copies them in (the originals are left alone), reads each
song's genre tag, analyzes it once, and rebuilds the affected profiles.
"""

from __future__ import annotations

import json
import os
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from analyzer_app import analyze, profiles
from analyzer_app.tags import read_tags

UNSORTED = "Unsorted"


def default_home() -> Path:
    return Path(os.environ.get("ANALYZER_HOME", Path.home() / "AnalyzerApp"))


def _safe_name(text: str, limit: int = 120) -> str:
    text = re.sub(r'[\\/:*?"<>|\x00-\x1f]', " ", text)
    text = re.sub(r"\s+", " ", text).strip(" .")
    return text[:limit] or "Untitled"


@dataclass
class AddResult:
    added: list[tuple[str, str]] = field(default_factory=list)   # (file, genre)
    duplicates: list[str] = field(default_factory=list)
    failed: list[tuple[str, str]] = field(default_factory=list)  # (file, reason)
    rebuilt: list[str] = field(default_factory=list)


class Library:
    def __init__(self, home: Path | None = None):
        self.home = Path(home) if home else default_home()
        self.references = self.home / "references"
        self.cache = self.home / "cache"
        self.profiles_dir = self.home / "profiles"

    # Songs and genres

    def genres(self) -> list[str]:
        if not self.references.exists():
            return []
        return sorted(p.name for p in self.references.iterdir()
                      if p.is_dir() and self.songs(p.name))

    def songs(self, genre: str) -> list[Path]:
        folder = self.references / genre
        if not folder.is_dir():
            return []
        return sorted(p for p in folder.iterdir()
                      if p.suffix.lower() in analyze.AUDIO_EXTENSIONS)

    def _known_hashes(self) -> dict[str, Path]:
        known = {}
        for genre in self.genres():
            for song in self.songs(genre):
                known[analyze.file_sha1(song)] = song
        return known

    def add(self, sources: list[Path], genre: str | None = None,
            progress=None) -> AddResult:
        """Copy songs (files or folders) into the library, filed by genre.

        genre overrides the tag for every song in this call. Songs with no
        genre tag go to "Unsorted", where they can be moved later.
        """
        result = AddResult()
        files: list[Path] = []
        for src in sources:
            src = Path(src)
            if src.is_dir():
                files += sorted(p for p in src.rglob("*")
                                if p.suffix.lower() in analyze.AUDIO_EXTENSIONS)
            elif src.suffix.lower() in analyze.AUDIO_EXTENSIONS:
                files.append(src)
            else:
                result.failed.append((src.name, "not an audio file"))

        known = self._known_hashes()
        touched: set[str] = set()
        for i, src in enumerate(files, 1):
            if progress:
                progress(i, len(files), src.name)
            try:
                sha = analyze.file_sha1(src)
            except OSError as e:
                result.failed.append((src.name, str(e)))
                continue
            if sha in known:
                result.duplicates.append(src.name)
                continue
            tags = read_tags(src)
            target_genre = _safe_name(genre or tags["genre"] or UNSORTED, 60)
            if tags["artist"] and tags["title"]:
                stem = _safe_name(f"{tags['artist']} - {tags['title']}")
            else:
                stem = _safe_name(tags["title"] or src.stem)
            dest = self._free_path(self.references / target_genre,
                                   stem, src.suffix.lower())
            try:
                analysis = self._analysis_for(src, sha)
            except analyze.AnalysisError as e:
                result.failed.append((src.name, str(e)))
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)
            analysis["track"]["file"] = dest.name
            known[sha] = dest
            touched.add(target_genre)
            result.added.append((src.name, target_genre))

        for g in sorted(touched):
            if g != UNSORTED:
                self.rebuild(g)
                result.rebuilt.append(g)
        return result

    def move(self, song: str, genre: str) -> Path:
        src = self.find(song)
        old_genre = src.parent.name
        genre = _safe_name(genre, 60)
        dest = self._free_path(self.references / genre, src.stem, src.suffix)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), dest)
        for g in (old_genre, genre):
            self._rebuild_or_drop(g)
        return dest

    def remove(self, song: str) -> Path:
        src = self.find(song)
        genre = src.parent.name
        src.unlink()
        self._rebuild_or_drop(genre)
        return src

    def find(self, song: str) -> Path:
        """A song by part of its file name. Must match exactly one song."""
        needle = song.lower()
        hits = [p for g in self.genres() for p in self.songs(g)
                if needle in p.name.lower()]
        if not hits:
            raise LookupError(f"no song matching {song!r}")
        if len(hits) > 1:
            names = ", ".join(p.name for p in hits[:5])
            raise LookupError(f"{len(hits)} songs match {song!r}: {names}")
        return hits[0]

    # Profiles

    def rebuild(self, genre: str) -> dict:
        songs = self.songs(genre)
        if not songs:
            raise profiles.ProfileError(f"no songs in {genre}")
        items = []
        for song in songs:
            a = self._analysis_for(song)
            items.append((song.stem, a))
        profile = profiles.build_profile(genre, items, kind="custom")
        profiles.save_profile(profile, self.profiles_dir)
        return profile

    def rebuild_all(self) -> list[str]:
        built = []
        for g in self.genres():
            if g != UNSORTED:
                self.rebuild(g)
                built.append(g)
        return built

    def profile(self, genre: str) -> dict | None:
        path = self.profiles_dir / f"{profiles.slugify(genre)}.json"
        if not path.exists():
            return None
        return json.loads(path.read_text())

    def all_profiles(self) -> list[dict]:
        return profiles.load_profiles(self.profiles_dir)

    # Reports

    def save_report(self, report: dict, song_stem: str) -> Path:
        name = re.sub(r"[^\w.-]+", "_", f"{song_stem}--{report['genre']['id']}")
        out = self.home / "reports" / f"{name}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2) + "\n")
        return out

    # Helpers

    def _analysis_for(self, path: Path, sha: str | None = None) -> dict:
        return analyze.analyze_cached(path, self.cache, sha=sha)

    def _rebuild_or_drop(self, genre: str) -> None:
        if genre == UNSORTED:
            return
        if self.songs(genre):
            self.rebuild(genre)
            return
        path = self.profiles_dir / f"{profiles.slugify(genre)}.json"
        path.unlink(missing_ok=True)
        folder = self.references / genre
        if folder.is_dir() and not any(folder.iterdir()):
            folder.rmdir()

    @staticmethod
    def _free_path(folder: Path, stem: str, suffix: str) -> Path:
        dest = folder / f"{stem}{suffix}"
        n = 2
        while dest.exists():
            dest = folder / f"{stem} ({n}){suffix}"
            n += 1
        return dest
