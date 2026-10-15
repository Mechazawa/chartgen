"""Guitar-stem evidence via BS-RoFormer SW, and the chord-texture pass.

The signal that unblocked this file: htdemucs has no guitar stem and its
6-stem variant is unusable (guitar 5.2 SDR, poisons "other"), so "is a
strummed guitar playing chords here" was undecidable - Mary Jane and Faded
measured identically on every chart-level signal (33%/33% raw chords,
79%/82% same-shape runs), which killed the first chord-riff gate. BS-RoFormer
SW (guitar 9.05 SDR) separates them 33% vs 0.0% guitar-stem energy with
94-98% vs 0% active windows on the local library test (2026-08-31), at
~34-48s per song on the RTX 3060 Ti.

The chord-texture pass this enables attacks the largest measured fidelity
gap in the project: chord-heavy songs where humans chart 50-100% chord
positions and transcription supplies 8-30% (24-song max-chord study).
Demotion is not the thief - SUPPLY is: the mix transcription hears one
voice where the guitar plays two. So where the isolated guitar stem's own
transcription shows a second comparable voice at a tick the chart left as
a single note, the second lane is added - evidence-backed, never invented.

Separation itself lives in chartgen.stems (one pass per song, shared with
lyrics/solos/taps/rescue). Everything degrades gracefully: no SW backend,
or no guitar in the song, means no changes.
"""
import numpy as np

OPEN = 7
SW_SR = 44100

# Run the (expensive) separation only on chord-textured candidates; a song
# whose transcription is nearly chord-free has nothing to chordify.
TRIGGER_CHORD_SHARE = 0.20
# Below this guitar-stem energy share the song has no real guitar (library
# test: guitar songs 32-33%, synth songs 0.0% - the gap is enormous).
MIN_GUITAR_SHARE = 0.10
# A second voice must be comparable to the first, same ratio as the mix path.
COMPARABLE = 0.65
# Human punk tops out around 67% chord positions; metal chug charts sit at
# 52-59% (Slipknot Metabolic, Haste the Day Gnaw); never chordify past this.
MAX_CHORD_SHARE = 0.60
# Run-uniform promotion (the shape-coherence fix): a run of singles gains
# the second voice as a unit or not at all, so the riff keeps one shape.
# The support bar is LOW for the same reason the triple rule's is: the stem
# transcribes the second voice on roughly one strum in five, so a 50% bar
# promoted 12 runs where the evidence really supports ~100 (measured on
# Mary Jane). Two evidenced ticks in a run is the floor that keeps a single
# stray from voicing a whole phrase.
CHORDIFY_RUN_SUPPORT = 0.20
CHORDIFY_MIN_EVIDENCED = 2
CHORDIFY_MIN_RUN = 3
CHORDIFY_RUN_GAP_BEATS = 1.0
# Interval -> added-lane offset, per RBN chord feel: small intervals are
# "small" adjacent chords, fourths/fifths are the 1-3 power-chord default.
POWER_CHORD_SEMITONES = 5


def separate_guitar(audio_path: str, progress=lambda m: None):
    """The song's mono guitar stem, or None when no guitar-capable backend
    ran. Delegates to chartgen.stems so a song is separated ONCE and every
    consumer (lyrics, solos, taps, rescue, this) shares the result - the
    dual-separation cost of the first version is gone."""
    from . import stems as stemsmod

    # Explicitly SW: it is the only backend with a guitar stem. SW is also
    # the default since the 2026-09-02 recalibration, so this hits the
    # same cache entry every other consumer uses - one separation per song.
    mono = stemsmod.separate(str(audio_path), progress, backend="sw")
    if not mono:
        return None
    return mono.get("guitar")


def guitar_share(guitar, mix_energy_stems=None) -> float:
    """Share proxy: guitar RMS energy vs a reference. Without other stems
    on hand, the ABSOLUTE activity test from the library study is used:
    share of one-second windows with real energy, scaled by mean level.
    The library separation was stark (94-98% active vs 0%), so any
    reasonable statistic clears the gap; this one needs no second model."""
    if guitar is None or not len(guitar):
        return 0.0
    n = len(guitar) // SW_SR
    if n < 8:
        return 0.0
    wins = guitar[:n * SW_SR].reshape(n, SW_SR)
    rms = np.sqrt((wins ** 2).mean(axis=1))
    peak = float(np.percentile(rms, 95))
    if peak < 1e-4:
        return 0.0
    return float((rms > 0.1 * peak).mean() * min(1.0, peak / 0.02))


def second_voices(guitar_events, tempo,
                  min_pitch: int = 40, min_amplitude: float = 0.20):
    """tick -> (root_pitch, second_pitch) where the guitar stem carries two
    comparable voices. Loudest two only - triples remain the separate,
    evidence-gated feature."""
    out = {}
    by_tick: dict[int, list] = {}
    for s, _, p, a in guitar_events:
        if p >= min_pitch and a >= min_amplitude:
            by_tick.setdefault(tempo.quantize(s, subdiv=4), []).append((a, p))
    for t, group in by_tick.items():
        group = sorted(group, reverse=True)
        comparable = [g for g in group if g[0] >= group[0][0] * COMPARABLE]
        if len(comparable) >= 2:
            pitches = sorted(p for _, p in comparable[:2])
            out[t] = (pitches[0], pitches[1])
    return out


def _runs(ticks, resolution):
    """Group positions into runs separated by more than a beat of air."""
    runs, cur = [], []
    for t in ticks:
        if cur and t - cur[-1] > resolution * CHORDIFY_RUN_GAP_BEATS:
            runs.append(cur)
            cur = []
        cur.append(t)
    if cur:
        runs.append(cur)
    return runs


def chordify(notes, voices, resolution: int, allowed_spans=None):
    """Give whole single-note RUNS the guitar's second voice, uniformly.

    `allowed_spans`: optional [(tick0, tick1)] - with a followed-instrument
    timeline, only runs starting inside a guitar-followed span are voiced,
    so a synth song never inherits guitar-stem bleed as chords.

    Per-tick addition was the first version and it repeated the mistake
    raw MAX_CHORD=3 made: evidence arrives at scattered ticks (the stem
    transcribes the second voice on some strums and not others), so the
    chart gained chords in a flicker pattern and same-shape adjacency fell
    80% -> 73% on a playtested song. Humans voice a RIFF, not a tick.

    So a run of singles (positions within a beat of each other, at least
    CHORDIFY_MIN_RUN long) is promoted as a unit when enough of its
    positions carry stem evidence: one interval offset for the whole run,
    decided by majority vote across the evidenced ticks, and one direction,
    chosen as whichever fits more of the run on the neck. Positions the
    shape does not fit stay single - a passing note, not a flipped shape.
    Existing chords and opens are untouched, and the song-wide chord share
    is still capped at the punk-band ceiling.
    """
    if not notes or not voices:
        return notes, 0
    by_tick: dict[int, list] = {}
    for n in notes:
        by_tick.setdefault(n[0], []).append(n)
    ticks = sorted(by_tick)
    chords_now = sum(1 for t in ticks
                     if len([x for x in by_tick[t] if x[1] != OPEN]) >= 2)
    budget = int(MAX_CHORD_SHARE * len(ticks)) - chords_now
    if budget <= 0:
        return notes, 0

    def lone_lane(t):
        group = by_tick[t]
        fretted = [x for x in group if x[1] != OPEN]
        if len(fretted) != 1 or any(x[1] == OPEN for x in group):
            return None
        return fretted[0][1]

    singles = [t for t in ticks if lone_lane(t) is not None]
    added = []
    for run in _runs(singles, resolution):
        if budget <= 0:
            break
        if len(run) < CHORDIFY_MIN_RUN:
            continue
        if allowed_spans is not None and not any(a <= run[0] < b for a, b in allowed_spans):
            continue
        evidenced = [t for t in run if t in voices]
        if (len(evidenced) < CHORDIFY_MIN_EVIDENCED
                or len(evidenced) < len(run) * CHORDIFY_RUN_SUPPORT):
            continue
        # one offset for the run: majority interval among evidenced ticks
        votes = [2 if voices[t][1] - voices[t][0] >= POWER_CHORD_SEMITONES
                 else 1 for t in evidenced]
        offset = 2 if votes.count(2) >= votes.count(1) else 1
        # one direction: whichever keeps more of the run on the neck
        up = sum(1 for t in run if lone_lane(t) + offset <= 4)
        down = sum(1 for t in run if lone_lane(t) - offset >= 0)
        step = offset if up >= down else -offset
        for t in run:
            if budget <= 0:
                break
            lane = lone_lane(t)
            new_lane = lane + step
            if not 0 <= new_lane <= 4:
                continue  # passing note: stays single, shape stays uniform
            added.append((t, new_lane, by_tick[t][0][2]))
            budget -= 1
    if not added:
        return notes, 0
    return sorted(set(notes + added)), len(added)
