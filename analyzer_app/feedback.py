"""Written A&R-style feedback on a comparison report.

With a Claude API key saved (`analyzer setkey ...`) or ANTHROPIC_API_KEY set,
Claude writes the notes from the report's numbers. Only the numbers are sent,
never the audio. Without a key, or offline, the notes are built from
templates instead, so the app always gives feedback.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

MODEL = "claude-opus-5-5"

SYSTEM = """You are a senior A&R at an electronic dance music label, giving a \
producer feedback on their mixdown. You get a JSON report comparing their song \
with reference tracks in a genre: a 0-100 readiness score with sub-scores, \
things missing compared with the genre, things to adjust, technical problems, \
tempo, key and the reference songs it sounds closest to.

Write plain-language notes the producer can act on in their DAW:
1. One short paragraph on what already works, from the high sub-scores.
2. One or two paragraphs on what holds it back, most important first.
3. "Next steps": the three fixes to do first, as a numbered list, each with a \
concrete move (which frequency range, how many dB, which processor).

Use only facts in the report; don't invent details about the arrangement, \
instruments or vocals that the numbers can't show. If the genre has few \
reference songs, say the comparison is rough. Keep it under 300 words. \
No headings other than "Next steps"."""


class FeedbackError(Exception):
    pass


def settings_path(home: Path) -> Path:
    return Path(home) / "settings.json"


def save_api_key(home: Path, key: str) -> Path:
    key = key.strip()
    if not key.startswith("sk-ant-"):
        raise FeedbackError("That doesn't look like a Claude API key "
                            "(they start with sk-ant-).")
    path = settings_path(home)
    path.parent.mkdir(parents=True, exist_ok=True)
    settings = _read_settings(home)
    settings["anthropic_api_key"] = key
    path.write_text(json.dumps(settings, indent=2) + "\n")
    path.chmod(0o600)
    return path


def api_key(home: Path) -> str | None:
    return (os.environ.get("ANTHROPIC_API_KEY")
            or _read_settings(home).get("anthropic_api_key"))


def _read_settings(home: Path) -> dict:
    try:
        return json.loads(settings_path(home).read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def _for_claude(report: dict) -> str:
    """The report without the long curve and per-metric list."""
    slim = {k: v for k, v in report.items() if k not in ("curve", "metrics")}
    return json.dumps(slim, indent=1)


def claude_feedback(report: dict, key: str) -> str:
    try:
        import anthropic
    except ImportError as e:
        raise FeedbackError("the anthropic package isn't installed "
                            "(pip install -e '.[feedback]')") from e
    client = anthropic.Anthropic(api_key=key)
    try:
        response = client.beta.messages.create(
            model=MODEL,
            max_tokens=16000,
            system=SYSTEM,
            messages=[{"role": "user", "content": _for_claude(report)}],
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
    except anthropic.AuthenticationError as e:
        raise FeedbackError("the saved Claude API key was rejected") from e
    except anthropic.APIConnectionError as e:
        raise FeedbackError("couldn't reach Claude (no internet?)") from e
    except anthropic.APIStatusError as e:
        raise FeedbackError(f"Claude returned an error ({e.status_code})") from e
    if response.stop_reason == "refusal":
        raise FeedbackError("Claude declined to write feedback")
    text = "".join(b.text for b in response.content if b.type == "text").strip()
    if not text:
        raise FeedbackError("Claude returned no text")
    return text


def template_feedback(report: dict) -> str:
    """Offline notes built from the report alone."""
    from analyzer_app.compare import AREA_LABELS

    s, g = report["score"], report["genre"]
    areas = sorted(s["areas"].items(), key=lambda kv: -kv[1])
    strong = [AREA_LABELS[a].lower() for a, v in areas if v >= 85]
    weak = [AREA_LABELS[a].lower() for a, v in reversed(areas) if v < 70]

    parts = [f"Against {g['name']} this mix scores {s['overall']}/100 "
             f"({s['verdict'].lower()})."]
    if strong:
        parts.append(f"Already in line with the genre: {_join(strong)}.")
    if weak:
        parts.append(f"The biggest gap is in {_join(weak)}.")
    if report["problems"]:
        parts.append("Fix the technical problems first: "
                     + _join([p["label"].lower() for p in report["problems"]]) + ".")
    if report["missing"]:
        parts.append("Compared with the reference tracks it's short on "
                     + _join([f["label"].split(" (")[-1].rstrip(")").lower()
                              for f in report["missing"][:3]]) + ".")
    if report["sounds_like"]:
        parts.append("It sounds closest to "
                     + _join([t["label"] for t in report["sounds_like"]]) + ".")
    if g["confidence"] == "low":
        parts.append(f"The genre has only {g['reference_count']} reference "
                     f"song{'s' if g['reference_count'] != 1 else ''}, so treat "
                     f"this as a rough comparison.")
    text = " ".join(parts)
    if report["next_steps"]:
        text += "\n\nNext steps\n" + "\n".join(
            f"{i}. {t}" for i, t in enumerate(report["next_steps"], 1))
    return text


def feedback(report: dict, home: Path, use_claude: bool = True) -> dict:
    """{"text", "source": "claude" or "template", "note": why Claude wasn't
    used, or None}."""
    key = api_key(home) if use_claude else None
    if not key:
        return {"text": template_feedback(report), "source": "template",
                "note": "no Claude API key saved"}
    try:
        return {"text": claude_feedback(report, key), "source": "claude", "note": None}
    except FeedbackError as e:
        return {"text": template_feedback(report), "source": "template", "note": str(e)}


def _join(items: list[str]) -> str:
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]
