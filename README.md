# Analyzer App

Compare your mixdown against reference songs in a genre, like DropTrack, but
running on your own computer. You feed it reference songs whenever you like, it
files them by their genre tag and builds a profile for each genre. Then you
check a mix: it gets a score out of 100, a list of what's missing and what to
adjust, and written A&R notes.

## Open the app on a Mac

The app is a web page that runs on your own computer. Your songs never leave it.

**The first time:**

1. On the GitHub page for this project, click the green **Code** button, then
   **Download ZIP**. Double-click the downloaded zip to unpack it, and move the
   `analyzer-app` folder somewhere you'll keep it, such as Documents.
2. Open **Terminal** (press Cmd+Space, type Terminal, press Return).
3. Drag the file **Start Analyzer App.command** from that folder into the
   Terminal window, then press Return.

It installs what it needs (a few minutes, only the first time), then opens the
app in your browser. Keep the Terminal window open while you use the app.

**After that:** double-click **Start Analyzer App.command**. The app opens in
your browser at http://127.0.0.1:8765. Close the Terminal window to stop it.

**Using it:**

- **Reference songs:** drop songs or whole folders. Each one is filed under the
  genre in its tag (the genre Beatport and other stores write into the file).
  Songs with no genre tag go into Unsorted, where you can move them.
- **Check a mix:** drop your mix. You get the score, what to fix first, what's
  missing, what to adjust, an EQ chart against the genre, and A&R notes. It
  compares with the closest genre; pick another from the list to compare again.
- **Settings:** paste a Claude API key to have the A&R notes written by Claude.

## Command line (optional)

Everything the app does also works from Terminal. Set it up once with:

```
brew install python@3.12
cd analyzer-app
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Each time you open a new Terminal window later, run
`cd analyzer-app && source .venv/bin/activate` first. `analyzer app` opens the
app; add `--host 0.0.0.0` to reach it from other computers on your network.

### Feed it reference songs

```
analyzer add ~/Downloads/new-tracks
```

Point it at songs or a whole folder. Each song is copied into
`~/AnalyzerApp/references/<Genre>/`, using the genre tag Beatport and other
stores write into the file. Your originals aren't touched, and songs already in
the library are skipped. Each new song takes about 30 seconds to measure.

To choose the genre yourself: `analyzer add song.mp3 --genre "Nu Disco"`.
Songs with no genre tag go into `Unsorted`.

### See and organize the library

```
analyzer genres                                  every genre, song count, profile
analyzer songs "Indie Dance"                     the songs in one genre
analyzer profile "Indie Dance"                   the genre's target numbers
analyzer move "Grandline" "Nu Disco"             refile a song (part of its name is enough)
analyzer remove "Grandline"                      take a song out
analyzer match my_mix.wav                        which genre your mix is closest to
```

Profiles rebuild by themselves whenever a genre's songs change. A genre works
from one song, but 10 to 20 per genre gives reliable ranges.

### Check a mix

```
analyzer check ~/Desktop/my_mix.wav
```

Drag the file from Finder into Terminal to type its path for you. It compares
the mix with the genre it sounds closest to. To pick the genre yourself, add
`--genre "Indie Dance"`. You get:

- a score out of 100 (85 or more is release ready), plus a score for
  tonal balance, loudness, dynamics, stereo image and low end
- technical problems such as clipping or out-of-phase bass
- what's missing compared with the genre, and what to adjust, with EQ moves
  like "boost 8-12 kHz by about 2.5 dB"
- the three fixes to do first, the reference songs it sounds closest to, and
  its tempo and key
- A&R notes in plain language

The full report is saved in `~/AnalyzerApp/reports/`. Export a WAV of your
mix for the most accurate result.

#### A&R notes written by Claude (optional)

Without a key you get basic notes built from the numbers. For notes written by
Claude, get an API key at [console.anthropic.com](https://console.anthropic.com),
then run once:

```
pip install -e ".[feedback]"
analyzer setkey sk-ant-...
```

Only the measurements are sent to Claude, never your audio. Each check costs a
few cents. If you're offline, you get the basic notes instead.

## What's inside

| File | What it does |
| --- | --- |
| `analyzer_app/analyze.py` | Measures a song: loudness, dynamics, tonal balance, stereo, low end, tempo, key |
| `analyzer_app/tags.py` | Reads artist, title, genre, BPM and key tags |
| `analyzer_app/library.py` | Files songs by genre and keeps profiles up to date |
| `analyzer_app/profiles.py` | Turns a genre's songs into target ranges, and matches a song to genres |
| `analyzer_app/compare.py` | Compares a song with a genre profile: score, what's missing, what to adjust |
| `analyzer_app/feedback.py` | The A&R notes, from Claude or built-in templates |
| `analyzer_app/cli.py` | The `analyzer` command |
| `analyzer_app/server.py`, `analyzer_app/ui/index.html` | The app: a local web server and its page |
| `analyzer_app/app.py` | Starts the server and opens the browser |
| `Start Analyzer App.command` | Double-click to set up and open the app |

Design notes: [Mix Analysis Spec](https://claude.ai/code/artifact/b0fbe5f6-522c-4e81-80df-b4f937f90842)
and [Genre Profiles](https://claude.ai/code/artifact/f254a74e-4762-46d8-92bf-8a2c00f9fb3b).

## Tests

```
python -m unittest discover -s tests -t .
```
