# chartgen — engineering history

This is the development log that used to be the README: every measurement,
dead end and design decision, kept verbatim as the record of why the pipeline
looks the way it does. Some sections describe code that has since been
removed (the neural charter) and are headed "Historical". For how to install
and use the app, see the [README](../README.md).

---

Generate Clone Hero charts from an audio file, at all four difficulty tiers.

Status: works end to end. Expert notes come from Basic Pitch transcription
(exact pitches, real polyphony, real note durations); a followed-instrument
timeline over BS-RoFormer stems drives taps, solos and the rescue passes;
texture, playability, motif and reduction stages are calibrated against an
800-chart human library. Guitar (5-fret) only.

**The neural charter (audio2chart) was removed on 2026-09-07.** A 10-song
comparison against human charts (`out/enginecmp/report.md` when regenerated)
found it tied on note timing, far less human-like in texture (1-5% chords vs
24% human, 12-16 same-fret 16ths in a row vs 3), twice the wall time and
three times the VRAM, unable to use any evidence-based feature shipped since
August, and unlicensed. Its code lives on the `legacy-audio2chart` branch.
The sections below headed "Historical" describe that engine and are kept as
the record of why the pipeline looks the way it does.

## Installing on a new machine

Run `scripts/install.sh`, then `scripts/chartgen.sh`.

The installer checks for Python 3.10-3.14 *with tkinter* (the UI needs it),
uses the CUDA torch build when `nvidia-smi` works, then runs the self-checks so
a broken setup fails loudly instead of at first use. Needs internet: roughly
3 GB of packages, plus ~1 GB of model on the first chart.

Useful flags: `--cpu` to force the CPU build, `--training` to add
lightning/hydra/wandb, `--cuda-tag cu128` if your driver is new enough to
prefer it.

Torch pins worth knowing: 2.9.1 has no `cu124` build. `cu126` (the default),
`cu128` and `cu129` all have cp314 wheels.

## Running it

**Double-click `chartgen.bat`** for the desktop UI. It runs through
`pythonw.exe`, so there is no console window, and errors surface in the UI rather
than vanishing. Pin it to the taskbar or make a desktop shortcut if you want.

The UI remembers your output folder and settings between runs — point it at your
Clone Hero songs folder once and it stays there. Generation runs on a worker
thread so the window stays responsive, with per-stage progress, a cancel button,
and an "Open song folder" button when it finishes.

Tkinter is used on purpose: it ships with Python, so the UI costs no dependency
and nothing extra to bundle.

**There is deliberately no PyInstaller `.exe`.** Bundling torch, transformers,
librosa, numba and llvmlite lands somewhere around 1.5-2.5 GB before the model
checkpoints, one-file mode re-extracts all of that to temp on every launch, and
numba/llvmlite are a known source of PyInstaller grief. Since this is not being
redistributed and runs on the machine that already has the venv, the `.bat` gives
the same double-click convenience for none of that cost. A real bundle is only
worth building if you want to run it on a machine without Python set up.

Command line, same pipeline:

```
python -m chartgen path/to/song.mp3 --artist "Band" --name "Song Title"
```

**YouTube links work directly** — paste one into the GUI's audio field or the
CLI and it downloads the audio (yt-dlp + a pip-bundled static ffmpeg, no
system install), guesses artist/title from the video title, and charts it:

```
python -m chartgen "https://www.youtube.com/watch?v=..." -o out
```

Downloads land in `<outdir>/_downloads/` and stay local, like everything
else. Same rule as the rest of the project: personal use — a downloaded
recording is still a copyrighted recording.

**Batch charting**: multi-select files in the GUI's Browse dialog, paste
several YouTube links into the audio field (space-separated), or pass any mix
to the CLI — each song gets its own folder with metadata from its own
tags/video title:

```
python -m chartgen song1.mp3 song2.flac "https://youtu.be/..." -o out
```

**Playlists**: a pure playlist URL (`youtube.com/playlist?list=…`) expands to
every video in it, with the count announced before work starts — no silent
caps, so know that a 200-video playlist is ~200 x 5 minutes of generation. A
normal watch URL that merely carries a `&list=` parameter deliberately stays
a single video.

**Remote use** (`chartgen-web.bat`): serves a phone-friendly web page on port
8471 — paste links/playlists or upload audio from any device, watch the
queue, download finished songs as zips for another Clone Hero install.
Charts land in the same output folder as the desktop app, whose saved
settings it reuses. Meant to be reached over Tailscale (or another private
network) — it deliberately has **no login**, so never port-forward it to the
open internet. Combined with remote power-on, this is the
wake-the-PC-from-anywhere workflow. Every generated song is titled
"… [chartgen]" so model output is always distinguishable from human charts
in the song list.

**Lyrics** (on by default, `--no-lyrics` to skip): vocals are transcribed
with faster-whisper (small model on GPU, base on CPU; downloads once) into
CH `phrase_start`/`lyric` events, so words scroll at the top during play.
Measured on Faded: 192 words, full-song coverage; a true instrumental
produces zero events rather than hallucinated text (the VAD is deliberately
off — it ate produced choruses — and a no-speech-probability filter at 0.9
handles instrumental stretches instead, since sung vocals legitimately score
~0.66). Auto-detects language. Expect imperfect words on heavily processed
vocals; they read fine in motion.

Output is a Clone Hero song folder: `notes.chart`, `song.ini`, and playable audio.
Drop it in your CH songs directory. Input must be at least 30 seconds.

Useful flags: `--subdiv 2` for an 8th-note grid (sparser, easier), `--subdiv 3`
for triplet feel, `--no-prominence` to skip the stem timeline (faster, fewer
features), `--no-taps` to strum everything.

Timing on the RTX 3060 Ti: about 85 s for a 4-minute song (the first song
of a session also loads the separator), and about 30 s when the song has
been charted before. The stem separator stays resident in the process, the
CPU-only analyses (mix transcription, sections, riff profiles, solo pitch
analysis, the lyrics lookup) run on threads while the GPU separates, and
separated stems plus Basic Pitch transcriptions are cached under
`%APPDATA%\chartgen\cache` (override with `CHARTGEN_CACHE_DIR`; stems are
kept as exact float32, ~230 MB per song, least-recently-used eviction at
`CHARTGEN_STEM_CACHE_GB`, default 6; `CHARTGEN_STEM_CACHE=0` disables it).
None of this changes a chart: the eight comparison songs come out
byte-identical with everything on, cold and cached. On a CPU-only machine the
same song still takes ~20 minutes the first time, almost all of it
BS-RoFormer separation.

Verified against the format spec in `vendor/ChartFormats`
([GuitarGame_ChartFormats](https://github.com/TheNathannator/GuitarGame_ChartFormats)):
only `Resolution` is a required `[Song]` field, `song.ini` metadata takes
priority over `[Song]`, `S 2 <length>` is the star power phrase, and lanes are
0-4 frets / 7 open / 5 forced / 6 tap. Reading it caught two bugs: the natural
HOPO window is `(65/192) × resolution` = 162 ticks at our resolution, not the
160 originally used, and `.wav` is not among the formats the docs list as being
in wide use, so it is now transcoded rather than passed through.

## Testing it in Clone Hero

You need a real song — the synthetic fixture in `tools/` validates timing and
grid maths, but tells you nothing about whether a chart is any good.

1. `python -m chartgen "C:\path\to\song.mp3" --artist "Band" --name "Song Title"`
2. Copy the generated folder (`out/Band - Song Title`) into a folder Clone Hero
   scans for songs. CH does not have a fixed path — you pick song folders in its
   settings, so use whichever you already have, or add one there.
3. In Clone Hero, rescan songs (it also scans on startup).
4. Play it, and check Practice mode too — that is where section markers show up.

Audio handling: `.mp3`/`.ogg`/`.opus` are copied through unchanged as `song.*`
(the reserved name CH looks for). Anything else, including `.flac` and `.wav`, is
transcoded to `song.opus` at 48 kHz. Input must be at least 30 seconds.

**Priority is Expert first, then Hard.** Medium and Easy are nice-to-haves, so
judge the chart on the top two tiers.

What to look at first, since these are the things measurement cannot settle:
whether notes land *with* the music rather than near it, and whether Expert feels
like the actual guitar part. If everything is uniformly late or early that is the
tempo/offset path; if notes sit on the beat but on the wrong frets, that is the
model and the answer is fine-tuning.

On Hard specifically, expect the known defect described under Known gaps: it is
currently all strums with no HOPOs, and only ~13% lighter than Expert. That is a
reduction-rule problem, not a detection problem, and it is fixable — worth
confirming it feels as wrong to play as it measures.

## Format compliance (audited against the spec)

`vendor/ChartFormats` is the authoritative format reference, and auditing the
writer against it caught three defects that were silently wrong in shipped
charts — all confirmed by inspecting a real generated file, not just reading
code:

- **Lyric markup.** `-` `=` `_` and `#^*%$/<>` are *markup* in `.chart`, not
  text: a hyphen joins a syllable to the next one. Any transcribed hyphenated
  word silently swallowed the following word, and the old cleaner actively
  created markup by mapping `=` to `-`.
- **Same-tick event ordering.** Sorting events alphabetically put
  `lyric <word>` ahead of its own `phrase_start`, so the first word of every
  phrase attached to the previous phrase. Events are now ranked: sections,
  `phrase_end`, `phrase_start`, `lyric`.
- **Phrase lead-in.** Phrases open half a beat before their first syllable
  (never before the previous phrase closed) so the line is readable when sung.

Also verified correct and left alone: note/modifier values (0-4 frets, 7 open,
5 forced, 6 tap), `S 2 <length>` star power, unquoted local `E solo` /
`E soloend` inside instrument tracks vs quoted global `E "section …"`, and the
natural HOPO threshold of `(65/192) × resolution`.

`song.ini` improvements from the same pass: `preview_start_time` points at the
first star power phrase (the densest window) rather than 0:00 silence, and
`diff_band` accompanies `diff_guitar`. **No `end` event is written** on
purpose — Clone Hero honours them (`end_events`), and a misplaced one would
truncate an outro.

## Charting standards the rules come from

There is no single "Clone Hero charting standard" — CH ships no authoring
guidelines. The de facto rulebook is three sources that largely agree, and
where they conflict the scanner wins because it is automated:

- **C3/RBN** authoring docs (`docs.c3universe.com` — note its TLS cert has
  expired, fetch accordingly), inherited from Rock Band Network.
- **YARN/YARG** submission standards (`wiki.yarg.in`), the modern equivalent.
- **`scan-chart`** (`github.com/Geomitron/scan-chart`), the scanner behind
  Chorus Encore, whose hard-coded thresholds decide what gets publicly
  flagged.

Numbers these gave us, now encoded in the pipeline rather than guessed:

| rule | source | what we do |
| :-- | :-- | :-- |
| 1 star power phrase per 40 beats, 1 measure long | RBN | `star_power_phrases`, counted in beats not seconds |
| No star power in the last ~8 measures | RBN | end guard — CH needs 2 phrases to activate, so late meter is unspendable |
| Sustains ≥200ms, gaps 50–150ms; scanner errors under 100ms | YARN + scanner | tempo-aware floor in `add_sustains` |
| No Green+Orange 2-note chord on Hard; no green-to-orange 3-note chord on Expert; no Green+Blue on Medium | scanner + RBN | `reduce.enforce_chord_rules` |
| Medium = GRYB, Easy = GRY, Easy has no chords | YARN + scanner | lane remaps in `reduce.py` (already matched) |
| Chords count as ONE note for density | scanner's `maxNps` | our Hard notes/sec cap counts positions (already matched) |
| Never drag a note out to fill a gap left by reduction | RBN | sustains propagate down from Expert (already matched, with a test) |
| A lone 120 BPM marker is *the* autochart tell | YARN rejects on it | real tempo map — the first thing this project fixed |

No source anywhere states a maximum notes-per-second for Expert; treat any
such number as invented. That is why our density is calibrated against a real
chart library instead. `diff_guitar` is a real 0–6 community scale
(Custom Songs Central) benchmarked against on-disc songs: 0 sightreadable,
3 keeps Expert players honest, 5 as hard as the hardest official charts.

One guideline we do **not** yet satisfy: *"only author one guitar part at
once — it's fine to jump back and forth between guitarists, but you can only
do one at a time."* Following whatever is loudest moment-to-moment is named
in the docs as the classic autocharter failure. Our Basic Pitch engine picks
the loudest pitches per tick, so instrument switching happens mid-phrase
rather than at section boundaries. Open problem.

## ⚠️ Licensing — read before sharing anything

The app no longer depends on audio2chart (removed 2026-09-07; it has no
license file and no data statement for the ~10k community charts it was
trained on). The shipping stack is Basic Pitch (Apache-2.0), Demucs (MIT),
bs-roformer-infer (MIT), faster-whisper (MIT). If you still have
`vendor/audio2chart` or `runs/pitch-*` on disk from before, they are not
needed and must not be redistributed.

`vendor/EasyChartGenerator` is MIT, so that part is unencumbered.

Separately: generated charts are derivative of copyrighted recordings. Keeping
this a local tool avoids hosting other people's audio — see "Why local" below.

## How it works

```
audio ─► beat/tempo detection + octave pick (librosa)
      ─► Basic Pitch transcription ─► (start, end, pitch, amplitude) events
      ─► BS-RoFormer stems ─► followed-instrument timeline (per 4 bars)
      ─► Expert selection: register/amplitude filters, quantize to the grid,
         chords from real polyphony, keyed/rhythm rescue where starved
      ─► texture, playability, motifs, riff unification, section reuse
      ─► chartgen reducer ─► Hard / Medium / Easy (HOPO-preserving)
      ─► real sustains + star power + sections + taps + solos + lyrics
      ─► notes.chart + song.ini + audio ─► Clone Hero song folder
```

`--reducer easygen` restores the vendored reducer for A/B comparison.

### Historical: the bit that actually mattered with the neural charter

audio2chart alone does not give you a usable multi-difficulty chart, for a
reason that is easy to miss. It writes every chart at a hardcoded 200 BPM and
places notes at whatever tick the raw onset time happens to land on. Measured on
a test clip: **2 of 180 notes fell on a beat.**

Difficulty reducers decide what to keep by asking "is this note on a beat?" With
an off-grid chart that test never fires, so every tier falls through to the
reducer's "don't leave a gap" fallback and the tiers collapse into each other —
Easy came out at 60 notes and Medium at 62, essentially the same track. Which is
exactly the problem you were trying to solve.

So `chartgen/tempo.py` detects the real beat grid and quantizes onto it. After:

| | before | after |
|---|---|---|
| Notes on the 16th grid | — | 118/118 |
| Notes on a beat | 2/180 | 56/118 |
| Detected tempo (true 120.0) | 117.5 | 120.0 |
| Tempo events (steady track) | 74 | 2 |
| Tier ladder | 180/120/62/**60** | 118/95/56/**28** |

Two subtleties worth knowing, both covered by tests:

- **Jitter is not drift.** librosa snaps beats to ~23ms analysis frames. Taking
  the median of beat gaps read a 120.0 BPM click track as 117.5, and treating
  each beat's gap as its own tempo emitted 74 tempo events for a metronome.
  `sync_track()` greedily fits one constant-tempo run for as long as every beat
  stays within tolerance, which absorbs jitter while still following real drift.
- **Pickup bars.** The first detected beat is rarely at t=0. Rather than guess at
  `.chart`'s ambiguous `Offset` semantics, the lead-in gets its own tempo event
  so beat N lands exactly on tick N×resolution.

## Expression: sustains, HOPOs, taps, star power (partly historical)

None of this can come from the model. audio2chart's token vocabulary is *only*
the set of lanes held at each 40ms frame (32 tokens). Note duration and the
`is5`/`is6`/`isS` flags exist in its tokenizer's tuples but are dropped by
training discretization, so the model emits bare onsets and nothing else.

**HOPOs need no markup and already work.** In `.chart`, HOPO-ness is implicit:
a single note within a 1/12 step of a different-fret single note is a HOPO
automatically. At resolution 480 that window is 160 ticks, a 16th is 120 ticks
and an 8th is 240 — so 16th runs HOPO and 8ths strum, which is the behaviour you
want. A generated test chart already contained 27 natural HOPOs. `N 5` is not
"a HOPO", it is the *forced* flag that inverts the natural state.

**Sustains** are derived from note spacing: a note sustains when the next one is
at least a beat away, released a 16th early and capped at 4 beats. Lengths are
computed from **Expert** spacing and propagated down, not recomputed per tier.
Expert spacing is the best available proxy for what the music is doing — if the
song plays 16ths, nothing should sustain even though reduction left the lower
tiers wide gaps. Computing per tier was the first attempt and made 39 of Easy's
40 notes sustains, turning a busy song into a slow one. Propagation is also
always safe: a reduced tier is a subset of Expert's ticks, so an inherited
sustain can never collide with its next note (asserted in tests).

An audio-envelope test would be the obvious alternative and is worse than
useless here: on a mixed track, drums, bass and vocals hold the RMS up whether
or not the guitar is still ringing, so it would sustain everything. Knowing a
note is *actually* held needs an isolated guitar stem. Tune `--min-sustain-gap`
during playtesting.

**Star power** phrases are placed on the densest bar-aligned windows (2 bars
each), spaced at least 6 bars apart, roughly one per 25s. Phrases repeat in every
difficulty track since `.chart` scopes them per track, and a phrase whose notes
were all reduced away is dropped rather than left unactivatable.

**Sections** come from chroma segmentation, snapped to bar boundaries. They are
numbered, not named: segmentation finds *where* the song changes but not whether
a part is a chorus, and wrong labels are worse than none for practice-mode
navigation.

**Taps are deliberately not generated — and the data says that is right.**
No authoring guideline anywhere states when a tap is appropriate: YARN says
only "use them sensibly based on the instrument and texture", the Chorus
Encore scanner has no tap rule at all (it tracks `hasTapNotes` purely as
metadata), and RBN/C3 never mentions them because Rock Band has no such note.
So `tools/study_taps.py` measured the local library instead — 129 human
charts, 76% of which use taps:

| tier | tap share of positions | spacing before a tap vs everything else |
| :-- | --: | :-- |
| Expert | 30% | 46% vs 43% within a 16th (**1.06x**) |
| Hard | 27% | 19% vs 26% (**0.72x**) |
| Medium | 29% | 10% vs 15% (**0.68x**) |
| Easy | 29% | 10% vs 19% (**0.54x**) |

Medians are identical at every tier, and on the lower tiers tapped notes are
*slower* than average. **Speed is not the trigger** — a "tap the fast runs"
rule would invent a pattern real charters do not follow. Taps also appear at
~29% even on Easy, so they are not an advanced-technique marker either. They
track instrument texture, which is an interpretive judgement, so generating
them would be guessing. Forced flags are skipped for the same reason. Both are written as
`N 6` / `N 5` lines, and the vendored reducer treats them as ordinary notes: for
Medium it keeps `notes[:2]`, so a single note plus a flag keeps the flag while a
chord plus a flag silently drops it. Tiers would disagree about which notes are
HOPOs. Beyond that, forcing is a phrasing judgement, and arbitrary taps make
charts worse rather than better.

## Historical: fret variety and the quality gate

Sampled generation makes lane choice the least stable part of the pipeline. Four
runs at identical settings on one clip produced 84% red, an even spread, and 34%
open notes. Two things were checked and cleared: the token→lane mapping is
correct (token 4 really is orange, 31 is open), and the model genuinely uses all
five frets — a direct token dump showed orange at 10–15%.

The cause was the *fixture*. The original test signal was four sine-stack tones
in a narrow bass range, nothing like guitar timbre, so the model had little to
condition on and sampled erratically. Widening it to span E2–E5 with chords fixed
it outright: variety 0.95 with G9/R16/Y24/B33/O15 on the first attempt.

But the variance is real even on good input (0.84 / 0.94 / 0.98 across three runs,
one of them missing blue entirely), so charts are now scored and regenerated
rather than trusted on one roll. `quality.variety_score` is normalized entropy
over the five frets, penalized for open-note spam — entropy rather than
"share of the most-used lane" because it also catches a chart that only ping-pongs
two frets. `--attempts` (default 3) stops as soon as a candidate clears
`--min-variety` (default 0.80), so a good first roll costs nothing, and the best
candidate is kept with a warning if none clear the bar. Thresholds are pinned in
tests against the four real distributions above.

`tools/diagnose.py` reports the raw model token histogram next to the final chart
lanes, which is how the above was separated; it caches tokens so the analysis can
be rerun without paying for generation.

**How human charters actually use pitch** (`tools/study_charters.py`, run over
the library): against the full mix, fret choice barely correlates with the
dominant pitch (corr +0.05, contour agreement 54%) — the mix's loudest pitch
is usually the vocal. Against isolated guitar stems the signal doubles:
corr +0.20, contour 59%, and frets repeat 39% of the time when the pitch
repeats vs 25% when it moves. Read: charters follow the lead instrument's
*direction and repetition* moderately (roughly 60/40 pitch vs phrasing
discretion), and absolute fret choice is largely stylistic — which is also
why exact-fret agreement between any two charts of the same song is near
chance. The pitch-based fret rule imitates the right signal, slightly more
rigidly than humans do; softening it is a playtest question, not a
measurement one. (Diagnostic only — the pipeline itself never needs stems.)

**The gate is now calibrated against real charts** (`tools/calibrate_quality.py`
over the 116-chart local library). The original walk_score penalised high
adjacency OR low repeats independently and would have failed **41%** of real
community charts — real charts span 18–76% adjacency and 2–63% repeats. What
they never do is both at once: heavy walking with almost no repeats, which is
exactly the degenerate staircase. The recalibrated score penalises the
*product* (knee at 60% adjacency, repeats discounted up to 15%), cutting real-
chart false failures to 13% — the stragglers are genuinely walky synth-arp
charts that share the staircase's shape. Real-chart landscape for reference:
adjacency median 50%, repeats median 20%, chord share median 18%.

## Historical: first playtest of the neural charter

First human playthrough of a real song (3.4 min): **Expert plays as a walk along
the fretboard** — one note at a time stepping left and right, rather than
anything resembling the guitar part. Measured on that exact chart:

| | Expert | what a guitar part does |
| :-- | --: | :-- |
| steps of +/-1 fret | **74%** | far fewer |
| repeats of the same fret | **6%** | constantly — chugs, tremolo, riffs |
| chords | 13% | more |

The 6% repeat rate is the tell. Real parts hammer the same note over and over;
this chart almost never does, because nothing maps audio pitch to fret. The model
is producing plausible *timing* (85% of real audio onsets are covered) attached to
essentially arbitrary *frets*.

**The quality gate did not catch this, by construction.** `variety_score` is
entropy over lane usage, and a perfect G-R-Y-B-O staircase uses all five frets
evenly — it scored the unfun chart **0.82**. Entropy measures distribution, not
pattern. `walk_score` was added afterwards to look at transitions instead, and
rates the same chart **0.40** against a 0.80 bar. Both now gate generation, scored
on the weaker of the two.

Note the reduced tiers score 1.00 on walk: thinning the notes breaks up the
staircase, so this is specifically an Expert problem.

HOPOs are disabled by default as of this finding (`hopo_frequency = 1` in
song.ini) — with 16th-heavy output nearly every Expert note became a HOPO, which
made the walking worse. `--hopos` re-enables them.

## Historical: the neural model is pitch-blind (`tools/pitch_probe.py`, removed)

Before investing in fine-tuning, this asks whether fret choice responds to pitch
at all. It feeds the model a rising two-octave scale, its exact reverse, and
octave leaps, then correlates fret against log pitch. Two runs:

| fixture | corr(log pitch, fret) | what the frets did |
| :-- | :-- | :-- |
| rising scale | +0.44, +0.58 | +/-1 walk |
| **falling scale** | **-0.04, -0.05** | fixed repeating zigzag |
| octave leaps | +0.09, +0.40 | `0,1,0,1,0,1…` |

**The falling case settles it.** A descending two-octave scale should produce a
strongly negative correlation under any real pitch mapping. It produces zero,
twice. The octave-leap fixture jumps between 82 Hz and 659 Hz every single note
and the model ping-pongs between green and red.

The rising case's positive number is an artifact, not evidence: a +/-1 walk
bounded at fret 0 drifts upward on its own, and the fixture's pitch rises
monotonically, so any upward drift correlates. A genuine pitch mapping would be
symmetric between rising and falling. This one is not.

**Consequence: fine-tuning alone will not fix fret assignment.** The model was
already trained on community charts, and its frets do not track pitch. Better
training data can teach it better rhythms and pattern vocabulary, but nothing
downstream of a pitch-blind encoder can decide *which* fret a note belongs on.
Making the ML path work needs pitch reaching the model — chroma/CQT features
alongside the Encodec embeddings, or transcription (Basic Pitch/MT3) or a Demucs
guitar stem used as conditioning. All of those require training, not just
fine-tuning.

Caveat: the correlation assumes the i-th chart note lines up with the i-th audio
onset, which is only approximate (note counts varied 67-156 across runs). The
correlations are indicative; the printed fret sequences are the real evidence.

### The signal is there — it just never reaches the model

`chartgen/pitch.py` extracts semitone-resolution CQT features on the model's own
frame grid. Run against the *same three fixtures*
(`tools/pitch_feature_check.py`):

| fixture | model | pitch features |
| :-- | --: | --: |
| rising scale | +0.44 / +0.58 | **+1.000** |
| falling scale | -0.04 / -0.05 | **+1.000** |
| octave leaps | +0.09 / +0.40 | **+1.000** |

Frets derived straight from those features track pitch at r = 0.98 and produce
`0,4,0,4,…` on the octave-leap fixture — the case where the model ping-ponged
green/red regardless of a three-octave jump.

So the audio carries pitch at essentially perfect fidelity and it is cheap to
extract. The gap is purely that Encodec codes do not expose it to the model.
That is what makes pitch conditioning worth training, and it is why fine-tuning
on better charts alone would not have worked.

Two bugs this experiment caught, both of the silent kind:

- `f0_norm` encoded silence as `0.0`, which collides with *lowest pitch* — the
  open low E, the most-played region on the instrument. Every low note was
  filtered out as silence and the whole song collapsed onto one fret. `voiced` is
  now a separate mask. Pinned by a regression test.
- Sampling a note's feature by truncating its onset to the frame below lands on
  the previous note's ring-out, which with a 400ms decay is still the louder of
  the two. Frame alignment needs rounding and a frame of lead-in.

## Historical: pitch conditioning (removed)

`chartgen/conditioning.py` + `chartgen/model.py`. Inference works now; training
integration is not wired yet (see below).

**How it attaches.** `audio_emb` is `[B, T', d_model]`, and at compression 3 a 30s
window gives 750 frames — the same 40 ms grid the pitch features use, so they line
up 1:1 with no resampling. A `Linear(50 -> d_model)` projection of the pitch
features is *added* to that audio memory.

**Why add, and why zero-init.** The projection starts at exactly zero, so an
untrained conditioner is the identity: pretrained checkpoints load unchanged, the
model starts at its current quality instead of from a damaged input layer, and if
pitch turns out not to help the projection just stays near zero. Concatenating
instead would have changed tensor shapes and forced reinitialisation. Verified
both directions with `tools/check_conditioning.py`:

- zero-init produces **byte-identical tokens** to the unmodified model (1500/1500
  the same, given the same seed)
- non-zero weights change **1500 of 1500** positions, proving pitch actually
  reaches the decoder rather than being silently dropped

A test that only checked the first would pass equally well if pitch were ignored,
which is why both are asserted.

**vendor/ is untouched** on the inference path: `Charter.generate` builds
`audio_emb` once and reuses it every decode step, so `chartgen/model.py` wraps
`transformer.forward` for the duration of a call instead of reimplementing the
~50-line loop.

### Training integration (done)

The three vendor edits are in place:

- `modules/training_transformer.py` `forward(..., audio_bias=None)` adds the
  bias right after `audio_compression`, with a shape assertion so a
  misaligned window fails loudly instead of training on garbage
- `dataloader/audio_loader.py` loads the cached `pitch_path` windows per
  chunk (edge-padded, never zero-padded) and the discrete collator stacks
  them into `batch["pitch"]`
- `modules/trainer.py` instantiates the conditioner when `model.use_pitch`
  is set and passes `batch["pitch"]` through; being a registered submodule,
  its params land in the optimizer automatically

`tools/finetune.py` drives it: loads the pretrained HF checkpoint into the
training transformer (strict, and verifies the tokenizer's special-token ids
against the checkpoint), applies a freeze policy, trains bf16 with early
stopping, and exports an inference-ready folder plus `conditioner.pt`:

```
python tools/build_dataset.py --root "C:/Users/ernes/Documents/Clone Hero/Songs" --out data
python tools/finetune.py --data data --out runs/pitch-m --smoke   # wiring check
python tools/finetune.py --data data --out runs/pitch-m           # real run
python -m chartgen song.mp3 --model runs/pitch-m/export --conditioner runs/pitch-m/export/conditioner.pt
```

Facts from the first real run on the 3060 Ti: the manifest is one row per
(song, difficulty) — the vendor splitter requires a `difficulty` field and
groups by `raw_path` so a song never straddles train/val. 99 of 116 songs pass
validation (the rest have events closer than the 40 ms grid). VRAM: the
collator flattens `batch x pieces` windows into one step — 4x4 OOMed instantly
(13.7 GB wanted); batch 4 x pieces 1 runs at ~7.9 GB. ~4.6 s/step with
workers=0, 21 steps/epoch.

Evaluating a model (or the pitch/model fret modes) against the human charts on
the held-out songs:

```
python tools/evaluate.py --val data/val.json --outdir out/eval-pitch
python tools/evaluate.py --val data/val.json --fret-mode model --outdir out/eval-model
python tools/pitch_probe.py --model runs/pitch-m/export --conditioner runs/pitch-m/export/conditioner.pt
```

The probe is the pass/fail: a fine-tune that worked turns the falling-scale
correlation strongly POSITIVE — pitch and fret descend together under any true
mapping; the direct features score +1.0 there and the unconditioned model ~0.0.
(An earlier version of the probe expected negative, which no correct mapping
can produce.) `tools/calibrate_quality.py` prints the real-chart score
landscape the gate thresholds came from.

**First fine-tune result** (18 min on the 3060 Ti, 86 songs, xattn policy):
falling-scale correlation went 0.0 → **+0.42** with visibly descending fret
runs — the model is now partially pitch-aware. Octave leaps still don't track
(+0.03), and +0.42 is far from the direct mapping's +1.0, so `--fret-mode
pitch` remains the right default.

**Second fine-tune cautionary tale** (177 songs incl. converted .mid, stem-mix
audio): higher best-val-accuracy (0.767 vs 0.744) and much better human-chart
recall on held-out songs (0.83 vs 0.59) — but the noisy val curve peaked at
**epoch 1** and early stopping kept that checkpoint, before the conditioner
had learned anything: the probe came back pitch-blind (falling **-0.03**).
The champion conditioned model is therefore still `runs/pitch-m` (the 86-song
run whose best epoch was 7). Val accuracy and pitch-awareness are different
objectives; the checkpoint selector only sees the former. Next-run recipe:
`--lr 5e-5 --patience 10`, and consider monitoring the probe correlation as
the selection metric rather than val token accuracy.

**The real risk is not the plumbing, it is whether a frozen decoder cooperates.**
The conditioner is tiny (~26k params for S, ~52k for M). Injecting pitch into the
cross-attention memory only helps if the decoder learns to read it, and a frozen
decoder was trained to interpret Encodec features, not this new channel. Expect to
fine-tune the decoder (or at least the cross-attention layers and output
projection) alongside the conditioner, not just train the projection. The additive
zero-init design supports that — start from the pretrained weights and unfreeze
whatever is needed — but do not expect conditioner-only training to fix frets.

## Historical: training throughput

Measured here (Ryzen 7 7735U, 8 torch threads, ~400 GFLOP/s FP32 achieved), one
forward+backward+AdamW step at batch 4, seq 750:

| model | params | this laptop CPU | i5-10600K (est.) | RTX 3060 Ti (est.) |
| :-- | --: | --: | --: | --: |
| S | 28M | 4.66 s | ~3.5-4.5 s | ~0.10-0.20 s |
| M | 228M | 25.0 s | ~19-25 s | ~0.50-1.0 s |

So 10k steps on M: ~69 h on this CPU, ~55-70 h on a 10600K, **~1.5-3 h on a
3060 Ti**. CPU-only training is not viable for M on either machine; the GPU turns
a multi-day run into an afternoon. GPU figures are scaled estimates, not measured
— replace them with a real step time on the first run there.

The 10600K is 6c/12t Comet Lake with AVX2 and no AVX-512. Fewer cores than this
laptop but a 125 W desktop part holding ~4.4 GHz all-core against a 15-28 W chip
throttling to ~3.2, so compute roughly cancels out; DDR4 (~51 GB/s) vs DDR5
(~77 GB/s) loses on the memory-bound parts.

**8 GB VRAM is the binding constraint on a 3060 Ti, not compute.** For M at batch
4 in FP32: weights 0.91 GB + gradients 0.91 GB + AdamW states 1.82 GB = 3.6 GB
before activations, then ~3.5 GB of self-attention across 16 layers and ~2.3 GB of
cross-attention to the 1500-frame audio embedding. That is ~9-10 GB. Expect to
need AMP/bf16 (which also uses the tensor cores), batch 2, or gradient
checkpointing. Reach for AMP first — it helps both problems at once.

**Pitch features must be precomputed, or the GPU starves.** Measured: CQT costs
~1.0 s per 30 s window, so a batch of 4 is ~4 s of CPU work against a sub-second
GPU step — on-the-fly extraction would idle a 3060 Ti roughly 75% of the time, and
6 cores cannot hide that. `tools/build_dataset.py` therefore caches them:
~7 s per song once, 127 KB per song (float16), so a 300-song corpus is ~39 MB.
Frame alignment is asserted against the audio length.

## Historical: training requirements

- **This machine has an RTX 3060 Ti (8 GB, bf16).** The old laptop's "no usable
  GPU" no longer applies: torch reports `cuda: True`, and the config defaults
  to bf16-mixed precision with the Encodec encoder frozen, which fits in 8 GB
  at batch 4. A fine-tune epoch over ~116 songs is minutes, not hours.
- **Raw PCM must be 24 kHz for the Encodec checkpoints.**
  `tools/build_dataset.py` writes 24 kHz (`RAW_RATE`); `load_raw_audio()`
  trusts whatever rate the config claims, so a 16 kHz cache would silently
  train on audio playing 1.5x fast. If you regenerate an old 16 kHz cache,
  delete `data/raw_audio` first.
- **Corpus format is already what you have.** `dataloader/convert_to_raw.py`
  discovers any directory tree where each folder holds `notes.chart` plus an
  audio file — exactly a Clone Hero songs folder, and exactly what Chorus Encore
  downloads give you. No API or scraping needed; the official Bridge client
  downloads into that shape.
- **ffmpeg is missing**, so `tools/build_dataset.py` replaces
  `convert_to_raw.py` — it writes the same headerless 16 kHz mono s16le PCM via
  librosa/soundfile, which are already dependencies. Point it at a folder of
  Clone Hero songs:

  ```
  python tools/build_dataset.py --root "D:/CH Songs" --scan-only   # report first
  python tools/build_dataset.py --root "D:/CH Songs" --out data/   # then decode
  ```

  It reports tier coverage and why songs were skipped, and emits
  `data/audio_dataset_with_raw.json` in the schema `main.py` reads.
- Charts must be `.chart`; many community charts ship as `.mid`. The scan counts
  them and tells you how many are being ignored.
- audio2chart still has **no license**, which matters more once you are training
  a derived model rather than just running one locally.

## Historical: known gaps of the neural charter

- **The sustain ratio is uncalibrated.** On the synthetic fixture it is verifiably
  correct — the riff rests on 2 of every 6 eighths, and Expert sustains exactly
  33% of notes — but whether a beat is the right threshold for real music is
  unknown until someone plays the result. Hence the flag.
- **Not fine-tuned yet.** The training path (pitch conditioning, freeze
  policies, bf16, export) is wired and smoke-tested; the first real run on the
  3060 Ti is the next step. Pitch-based fret assignment already fixes lane
  choice without it — fine-tuning is about better rhythm/pattern vocabulary.
- **No playtesting yet.** All improvements above are measured, not felt. The
  quality-gate thresholds (`walk_score` in particular) are reasoned, not
  calibrated against real community charts — calibrating them against the
  116-song corpus is the natural follow-up.
- **~41% of Expert notes sit more than 60ms from any detected audio onset**
  — the model fills the grid rather than tracking the part, and the held-out
  eval measured 1.4-2x the human chart's note count. `chartgen/density.py`
  now gates positions on onset evidence (`--density onset`, the default):
  strength is judged against a local rolling ceiling so quiet sections thin
  by their own loudness — genre-neutral by construction, an EDM drop keeps
  its wall, a breakdown empties — and a drop is refused if it would open a
  gap over two beats. The threshold needs playtest calibration.
- Guitar only — no bass/drums/vocals tracks.
- 61 of the library's charts are `.mid` and are skipped by the dataset builder;
  converting them would grow the corpus ~50%.

### Fixed since the playtest (measured on "Faded", 216s)

| gap | before | after |
| :-- | :-- | :-- |
| Expert walks the fretboard | 66% adjacent steps, 21% repeats | **21% / 45%** (`--fret-mode pitch`) |
| Hard all strums | 6 natural HOPOs | **59** (chartgen reducer thins walls, keeps figures) |
| Medium/Easy lane collapse | remap piles onto green | order-preserving maps; variety normalized per tier's lane budget |
| Aliased chords | 38 model chords became 210 chart chords | nearest-frame rule; fake chords gone |
| Tempo octave ambiguity | no logic | `auto` halves/doubles into 70–165 BPM, accent-aware parity |
| Non-deterministic runs | no seed hook | `--seed N` (attempts advance it) |
| HOPOs off by default | walking made every 16th HOPO | on by default; repeats never HOPO |

## Why local

CPU-only is viable: the generation loop batches 30s chunks, so it is ~16s for the
S checkpoint on a 40s clip and scales sub-linearly with song length. Running
locally also means users' audio files never get uploaded anywhere, which sidesteps
both the copyright exposure of hosting other people's music and any GPU bill.

## Layout

```
chartgen/tempo.py        beat detection, octave pick, tempo map, time->tick quantization
chartgen/transcribe.py   Basic Pitch events -> Expert notes; rescue, triples, ladders
chartgen/prominence.py   followed-instrument timeline, keyed/rhythm rescue, solo windows
chartgen/stems.py        BS-RoFormer / Demucs separation shared by every consumer
chartgen/texture.py, motifs.py, structure.py, playability.py   chart-shaping passes
chartgen/frets.py        brightness re-laning, fret-jump smoothing
chartgen/reduce.py       chartgen difficulty reducer (+ vendored one for A/B)
chartgen/expression.py   star power, sections, sustain top-up; taps.py, solo.py, lyrics.py
chartgen/chart.py        .chart / song.ini writers
chartgen/__main__.py     CLI pipeline; chartgen/app.py  Tkinter GUI; chartgen/web.py  web UI
tests/                   python tests/test_chartgen.py (or -m tests.test_chartgen)
tools/                   calibration studies against the human chart library
vendor/                  EasyChartGenerator (MIT), ChartFormats (spec)
```

Setup: `scripts/install.sh` (see above), or `python -m venv .venv`, install
`torch torchaudio` from the right index, then `pip install -r requirements.txt`.
Works on Python 3.14.
