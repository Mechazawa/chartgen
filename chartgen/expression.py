"""Star power, section markers and the sustain top-up.

Expert sustains themselves come from the transcribed note durations
(chartgen.transcribe); this module holds the expression that is derived from
the chart and the audio structure.

Tap notes (chartgen.taps) and forced HOPO flags (chartgen.motifs) are computed
elsewhere and applied at WRITE time, after reduction — the vendored difficulty
reducer treats `N 6` / `N 5` lines as ordinary notes and would keep or drop
them inconsistently per tier, so they never pass through it. Natural HOPOs
need no markup at all; they are implicit in note spacing.
"""

from itertools import pairwise


def hopo_threshold(resolution: int) -> int:
    """Tick gap under which a different-fret single note is already a HOPO.

    Per the .chart spec: (65/192) * resolution, rounded down — 65 ticks at the
    classic 192 resolution, 162 at ours. Notes on the same lane as the previous
    one, and chords, never become natural HOPOs regardless of spacing.
    """
    return (65 * resolution) // 192


def star_power_phrases(
    notes: list[tuple[int, int, int]],
    resolution: int,
    duration_s: float,
    beats_per_bar: int = 4,
    phrase_bars: int = 2,
    min_gap_bars: int = 8,
    min_notes: int = 4,
    beats_per_phrase: int = 50,
    end_guard_bars: int = 8,
) -> list[tuple[int, int]]:
    """Pick bar-aligned [(start_tick, length_ticks)] star power phrases.

    Frequency and length track what charters measurably do, cross-checked
    against the RBN/C3 spec. Measured on 800 full-ladder community charts:
    phrase length median 8 beats (two bars — twice the RBN spec's one),
    starts 44 beats apart (RBN: 40), 0.79 phrases per 40 beats (RBN: 1.0),
    and a median 48 beats left clear after the last phrase. The constants
    sit on the community medians; RBN and YARG arriving at ~40-beat spacing
    independently makes the neighbourhood trustworthy. Counting beats
    rather than seconds matters — the old one-per-25s rule under-filled
    slow songs and over-filled fast ones.

    Nothing is placed in the last 8 measures (human p10 is ~7): Clone Hero
    needs half a meter (two phrases) to activate at all, so late star power
    is meter the player can never spend. Within those rules phrases go on
    bars of the song's TYPICAL density, spaced out so activations spread
    across the song. They used to go on the densest bars: measured against
    the library (2026-09-05) human phrases sit at 1.04x the song's density
    and hold a median 12 notes, ours sat at 1.0-1.5x with 16-23 notes, and
    a 20-note phrase completes about 12% of the time at 90% accuracy where
    a 12-note one completes 28% - the player banked half the meter and
    "couldn't fill a bar". Same count, spacing and length; ordinary bars.
    """
    bar = beats_per_bar * resolution
    ticks = sorted(t for t, _, _ in notes)
    if not ticks:
        return []

    length = phrase_bars * bar
    last_usable = ticks[-1] - end_guard_bars * bar
    candidates = []
    for start in range(0, ticks[-1] + 1, bar):
        if start > last_usable:
            break
        count = sum(1 for t in ticks if start <= t < start + length)
        if count >= min_notes:
            candidates.append((count, start))

    played_beats = (ticks[-1] - ticks[0]) / resolution
    target = max(2, int(played_beats / beats_per_phrase))
    chosen: list[tuple[int, int]] = []
    # Closest to the song's typical phrase count first; tie-break on
    # position so the result is deterministic.
    typical = sorted(c[0] for c in candidates)[len(candidates) // 2] if candidates else 0
    for _, start in sorted(candidates, key=lambda c: (abs(c[0] - typical), c[1])):
        if all(abs(start - s) >= min_gap_bars * bar for s, _ in chosen):
            chosen.append((start, length))
            if len(chosen) >= target:
                break
    return sorted(chosen)


def sections(y, sr, tempo, beats_per_bar: int = 4) -> list[tuple[int, str]]:
    """Bar-aligned practice-mode section markers from audio structure.

    Numbered, not named: chroma segmentation finds *where* the song changes but
    says nothing about whether a part is a chorus or a bridge. Guessing labels
    would be fiction, and wrong names are worse than none for navigation.
    """
    import librosa
    import numpy as np

    duration = librosa.get_duration(y=y, sr=sr)
    k = max(2, min(12, int(duration / 25)))
    try:
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        bounds = librosa.segment.agglomerative(chroma, k)
        times = librosa.frames_to_time(bounds, sr=sr)
    except Exception:
        return []  # segmentation is a nice-to-have; never fail the whole chart

    bar = beats_per_bar * tempo.resolution
    out: list[tuple[int, str]] = []
    for t in np.sort(times):
        tick = max(0, int(round(tempo.time_to_beat(float(t)) * tempo.resolution / bar)) * bar)
        if not out or tick > out[-1][0]:
            out.append((tick, f"Section {len(out) + 1}"))
    return out


# Median sustain share per tier across 49 human charts with a full ladder.
# Note it RISES as tiers get easier: fewer notes, so each is held longer.
# Propagating Expert's sustains downward unchanged left every tier flat at
# Expert's rate, which is why our Easy felt clipped next to a real one.
# Median share of positions carrying a sustain, per tier, across 800
# genre-balanced full-ladder charts (2026-08-24). Scale raised every tier
# from the 49-chart numbers (8.7/11.1/14.9/15.3%), Easy most of all -
# sparse tiers lean on held notes to stay musical.
TIER_SUSTAIN_SHARE = {
    "ExpertSingle": 0.116, "HardSingle": 0.146,
    "MediumSingle": 0.170, "EasySingle": 0.222,
}


def top_up_sustains(
    notes: list[tuple[int, int, int]],
    resolution: int,
    end_tick: int,
    target_share: float,
    release_beats: float = 0.25,
    max_beats: float = 4.0,
) -> list[tuple[int, int, int]]:
    """Sustain the roomiest notes until the tier reaches its measured share.

    Deriving sustains from a reduced tier's own gaps was the original mistake
    — it made 39 of Easy's 40 notes sustains, turning a busy song slow. The
    bound is what makes it safe now: only the notes with the most space get
    one, and only until the tier matches what human charts do, so the failure
    mode is capped by construction rather than by hoping the gaps behave.
    """
    if not notes:
        return notes

    release = int(release_beats * resolution)
    cap = int(max_beats * resolution)
    floor = resolution // 2

    ticks = sorted({t for t, _, _ in notes})
    next_tick = dict(pairwise(ticks))
    held = {t for t, _, sus in notes if sus > 0}
    target = int(round(target_share * len(ticks)))
    if len(held) >= target:
        return notes

    room = []
    for tick in ticks:
        if tick in held:
            continue
        gap = next_tick.get(tick, end_tick) - tick
        length = min(gap - release, cap)
        if length >= floor:
            room.append((length, tick))
    room.sort(reverse=True)

    added = {tick: length for length, tick in room[:max(0, target - len(held))]}
    return [(t, lane, added.get(t, sus) if sus == 0 else sus)
            for t, lane, sus in notes]
