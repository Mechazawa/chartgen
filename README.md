# chartgen

Turn any song into a playable Clone Hero chart. Paste a YouTube link or drop
an audio file; chartgen listens to the recording and writes a complete song
folder — Expert through Easy, synced lyrics, star power, solos, taps, album
art — ready to drop into your Songs directory.

Everything runs on your own PC. No audio is ever uploaded.

**Website:** https://chartgen.org

## What it does

- **Any input.** YouTube links, whole playlists, or local MP3/FLAC/WAV/OGG/
  Opus/M4A files. Batches queue up and chart one after another; songs you
  have already charted are skipped.
- **Real transcription, not a guess.** Notes come from
  [Basic Pitch](https://github.com/spotify/basic-pitch) (exact pitches, real
  polyphony, real note durations) over a six-stem
  [BS-RoFormer](https://github.com/openmirlab/bs-roformer-infer) separation,
  so the chart follows the actual guitar, synth or piano part instead of
  whatever is loudest.
- **Full difficulty ladder.** Expert, Hard, Medium, Easy — each reduced the way
  human charters do it, calibrated against an 800-chart library of hand-made
  charts. A 0–6 `diff_guitar` rating is written for the song list.
- **The details that make a chart feel charted.** Sustains from real note
  lengths, HOPOs, star power phrases, numbered practice sections, tap notes
  on synth and piano parts, solo markers where a lead break is found, and
  riffs that repeat consistently every time the song does.
- **Lyrics that scroll in time.** Looked up on [LRCLIB](https://lrclib.net)
  first and word-aligned to the vocal stem; when nobody has synced the song
  yet, the vocals are transcribed with
  [faster-whisper](https://github.com/SYSTRAN/faster-whisper) instead.
- **Three ways to use it.** A desktop app, a command line, and a
  phone-friendly web page that hands the finished song back to whatever
  device you're on — from the couch, or from anywhere with a free Cloudflare
  Tunnel in front.

Guitar (5-fret) only. Songs must be at least 30 seconds.

## Install (Linux)

1. Clone the repo and `cd` into it.
2. **Run `scripts/install.sh`.** It creates `.venv`, installs the CUDA
   PyTorch build when `nvidia-smi` works (else the CPU build) and every
   dependency, then runs the self-checks. Expect several minutes and roughly
   3 GB of downloads.
3. **Run `scripts/chartgen.sh`.**

The first songs you chart download the models automatically — about 2.5 GB
in total: the stem separator (0.7 GB), the lyric aligner (1.2 GB) and the
speech recogniser (0.5 GB, 0.15 GB CPU-only). After that everything is local.

**Requirements:** Linux, an internet connection for setup and for YouTube
links. Python 3.10–3.14 with tkinter (Arch: `tk`, Debian/Ubuntu:
`python3-tk`; set `PYTHON` to pick another interpreter than `python3`). An
NVIDIA GPU is strongly recommended — a 4-minute song takes about 90 seconds
on an RTX 3060 Ti, or about 20 minutes CPU-only.

Installer flags: `--cpu` forces the CPU build, `--cuda-tag cu128` if your
driver prefers a newer CUDA build (the default is cu126), `--training` adds
the training stack.

## Using it

**Desktop app** (`scripts/chartgen.sh`): paste a link or browse for files,
point *Save to* at your Clone Hero songs folder once (`~/.clonehero/Songs`;
it's remembered), hit
*Generate chart*. Progress and the finished chart's stats show in the window;
*Open song folder* takes you straight to the result. Then rescan songs in
Clone Hero.

Options worth knowing: *Max difficulty* caps how hard Expert gets; *Note
grid* switches between 16ths (default), 8ths (sparser) and triplets; the
toggles turn sustains, star power, sections, HOPOs, lyrics, solos, open notes
and taps on or off. Everything defaults to on.

**Command line**, same pipeline:

```
.venv/bin/python -m chartgen "https://www.youtube.com/watch?v=..." -o ~/.clonehero/Songs
.venv/bin/python -m chartgen song.mp3 other.flac "https://youtu.be/..." -o out
.venv/bin/python -m chartgen --help
```

Artist and title come from the video title or the file's tags. Override them
for a single song with `--artist` and `--name`.

**Web UI** (`scripts/chartgen-web.sh`): a page where any device can paste links or
upload files, watch its own queue, and get each finished song back as a zip —
downloaded to that device automatically. Put it behind a free Cloudflare
Tunnel + Access login and it works from anywhere (see
[docs/remote-access.md](docs/remote-access.md)): people you list as owners
get songs saved to the PC's library too; anyone else is a guest whose song
is charted, handed over, and deleted from the PC. Without Access in front it
has **no login**, so on a LAN or Tailscale (`scripts/chartgen-web.sh
--host 0.0.0.0`) treat it as a shared queue, and never port-forward it to the open
internet.

## Output

```
<Save to>/Artist - Title [chartgen]/
    notes.chart     all four difficulties, star power, sections, solos, lyrics
    song.ini        name, artist, charter, length, diff_guitar, preview time
    song.opus       the audio (mp3/ogg/opus inputs are copied as-is)
    album.png       when available: the file's embedded art or the YouTube thumbnail
```

Every generated song is titled "… [chartgen]" so it is always distinguishable
from a human chart in the song list.

## Privacy and network use

- Audio never leaves your machine. Separation, transcription and lyric
  alignment all run locally.
- With lyrics on (the default), the **artist and title** of each song are
  sent to lrclib.net to look up synced lyrics. Use `--no-lyrics` or the
  "Listen to the vocals" lyric source to avoid that.
- YouTube links are fetched with yt-dlp. Downloads stay in
  `<Save to>/_downloads/`. Artist and title are parsed from the video title,
  so a channel's suffix (e.g. "| House | NCS") can end up in the song name —
  the CLI's `--name`/`--artist` override it.
- Model weights download once on first use (BS-RoFormer from the UVR public
  model mirror, faster-whisper from Hugging Face, the forced aligner from
  PyTorch's CDN). Stems and transcriptions are cached under
  `~/.cache/chartgen/cache` (up to 6 GB, oldest evicted; `CHARTGEN_CACHE_DIR`
  and `CHARTGEN_STEM_CACHE_GB` override).

## Licensing

chartgen is MIT licensed. It builds on Basic Pitch (Apache-2.0),
bs-roformer-infer and Demucs (MIT), faster-whisper (MIT), yt-dlp (Unlicense),
librosa (ISC) and PyTorch (BSD); the full pinned list is in
`requirements.txt`. Model weights are downloaded from their own sources at
first run and are not redistributed here.

Generated charts are derivative of copyrighted recordings. This is a tool for
charting music you have the right to use, for your own play — keep it local.

## For developers

```
scripts/        install.sh, chartgen.sh (desktop app), chartgen-web.sh (web UI)
tests/          python -m tests.test_chartgen   (also run by the installer)
docs/history.md the full engineering log: every measurement and design decision
docs/index.html the website (GitHub Pages serves docs/)
tools/          calibration studies against the human chart library
```

Manual setup, if you would rather not use the installer: create a venv,
install `torch==2.9.1 torchaudio==2.9.1` from the right PyTorch index, `pip
install -r requirements.txt`, then `pip install basic-pitch==0.4.0 --no-deps`
(its wheel asks for a TensorFlow that does not exist on modern Python; the
ONNX backend is what runs).
