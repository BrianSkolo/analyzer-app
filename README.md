# Analyzer App

Compare your mixdown against reference songs in a genre, like DropTrack, but
running on your own computer. This first part handles the reference library:
you feed it songs whenever you like, it files them by their genre tag, and it
builds a profile for each genre that your mixes are compared against.

## Set up on a Mac (once)

1. Install Homebrew if you don't have it: open **Terminal** and paste the line
   from [brew.sh](https://brew.sh).
2. Install Python and ffmpeg:

   ```
   brew install python@3.12 ffmpeg
   ```

3. Download this project and install it:

   ```
   git clone https://github.com/BrianSkolo/analyzer-app.git
   cd analyzer-app
   python3.12 -m venv .venv
   source .venv/bin/activate
   pip install -e .
   ```

Each time you open a new Terminal window later, run
`cd analyzer-app && source .venv/bin/activate` first.

## Feed it reference songs

```
analyzer add ~/Downloads/new-tracks
```

Point it at songs or a whole folder. Each song is copied into
`~/AnalyzerApp/references/<Genre>/`, using the genre tag Beatport and other
stores write into the file. Your originals aren't touched, and songs already in
the library are skipped. Each new song takes about 30 seconds to measure.

To choose the genre yourself: `analyzer add song.mp3 --genre "Nu Disco"`.
Songs with no genre tag go into `Unsorted`.

## See and organize the library

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

## What's inside

| File | What it does |
| --- | --- |
| `analyzer_app/analyze.py` | Measures a song: loudness, dynamics, tonal balance, stereo, low end, tempo, key |
| `analyzer_app/tags.py` | Reads artist, title, genre, BPM and key tags |
| `analyzer_app/library.py` | Files songs by genre and keeps profiles up to date |
| `analyzer_app/profiles.py` | Turns a genre's songs into target ranges, and matches a song to genres |
| `analyzer_app/cli.py` | The `analyzer` command |

Design notes: [Mix Analysis Spec](https://claude.ai/code/artifact/b0fbe5f6-522c-4e81-80df-b4f937f90842)
and [Genre Profiles](https://claude.ai/code/artifact/f254a74e-4762-46d8-92bf-8a2c00f9fb3b).

## Tests

```
python -m unittest discover -s tests -t .
```
