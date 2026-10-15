"""Thin over-generated notes down to what the audio supports.

The model is near grid-saturated on real songs (~5.3 notes/sec against a
~6.35/sec ceiling) and the held-out eval measured 1.4-2x the human chart's
note count. The README's onset analysis showed why: ~41% of Expert notes sit
more than 60ms from any detected audio onset — the model fills the grid
instead of tracking the part.

This gates chart positions on onset evidence from the audio itself, which
scales with the song instead of imposing a genre prior: an EDM drop keeps its
wall of notes because every one has onset energy behind it; a breakdown thins
out because nothing does. Two safeguards keep it musical:

- strength is judged against a LOCAL rolling ceiling, so a quiet bridge is
  gated by bridge loudness, not by the drop three sections earlier;
- a note is never dropped if removing it would open a gap of more than two
  beats — sparse passages are already sparse, and silence reads as a bug.
"""
import numpy as np

from .tempo import BEAT_LAG_S

GAP_BEATS = 2.0  # never open a hole bigger than this


def onset_scores(notes, y, sr, tempo):
    """Per-position onset evidence in [0, 1], judged against local loudness."""
    import librosa

    env = librosa.onset.onset_strength(y=y, sr=sr)
    times = librosa.times_like(env, sr=sr) - BEAT_LAG_S  # on the beat grid's time base
    if env.max() <= 0:
        return {tick: 1.0 for tick, _, _ in notes}

    # Rolling 95th percentile over ~4s windows: the local "loud" reference.
    hop = times[1] - times[0] if len(times) > 1 else 0.023
    half = max(1, int(2.0 / hop))
    ceiling = np.array([
        np.percentile(env[max(0, i - half):i + half + 1], 95)
        for i in range(len(env))
    ])
    ceiling = np.maximum(ceiling, env.max() * 0.05)  # silence floor

    res = tempo.resolution
    scores = {}
    for tick in {t for t, _, _ in notes}:
        t = tempo.beat_to_time(tick / res)
        # Strongest onset within one frame either side: quantization moved the
        # note up to half a grid step from the audio event.
        i = int(np.searchsorted(times, t))
        lo, hi = max(0, i - 2), min(len(env), i + 3)
        peak = float(env[lo:hi].max()) if hi > lo else 0.0
        scores[tick] = peak / float(ceiling[min(i, len(ceiling) - 1)])
    return scores


def gate_by_onsets(notes, y, sr, tempo, threshold: float = 0.25):
    """Drop positions with weak onset evidence, never opening big gaps.

    Weakest positions go first, and each drop re-checks the gap constraint
    against what is left, so a run of weak notes cannot all vanish at once.
    """
    if not notes:
        return notes
    scores = onset_scores(notes, y, sr, tempo)
    max_gap = GAP_BEATS * tempo.resolution

    ticks = sorted({t for t, _, _ in notes})
    keep = set(ticks)
    for tick in sorted((t for t in ticks if scores[t] < threshold),
                       key=lambda t: scores[t]):
        kept = sorted(keep - {tick})
        if not kept:
            continue
        # Playtest: fade-out tails vanished entirely — the gap check below
        # has no "after" neighbour at the edges, so trailing quiet notes fell
        # one by one and the chart ended early. The first and last notes may
        # retreat from the model's coverage by at most the same gap budget.
        if ticks[-1] - kept[-1] > max_gap or kept[0] - ticks[0] > max_gap:
            continue
        import bisect
        i = bisect.bisect_left(kept, tick)
        before = kept[i - 1] if i > 0 else None
        after = kept[i] if i < len(kept) else None
        if before is not None and after is not None and after - before > max_gap:
            continue  # dropping this would tear a hole in the chart
        keep.discard(tick)

    return [n for n in notes if n[0] in keep]


def onset_fidelity(notes, y, sr, tempo, tolerance_s: float = 0.07) -> float:
    """How well the chart's notes line up with what the audio actually plays.

    Playtest of the two engines: the neural model "does some things better,
    but also worse — bigger wins and bigger losses", while transcription is
    more consistent. That variance is a SELECTION problem, since the neural
    engine samples and can be rolled several times. What the existing gates
    could not measure is whether a roll actually caught the part, and this
    can: F1 between the chart's note positions and the audio's detected
    onsets. Precision falls when a roll invents notes, recall when it misses
    the part — the two failure modes of a bad roll.
    """
    import librosa

    if not notes:
        return 0.0
    res = tempo.resolution
    chart = sorted({tempo.beat_to_time(t / res) for t, _, _ in notes})
    peaks = librosa.onset.onset_detect(y=y, sr=sr, units="time")
    if len(peaks) == 0:
        return 0.0

    peaks = np.asarray(peaks, dtype=float) - BEAT_LAG_S
    chart_arr = np.asarray(chart, dtype=float)

    def covered(a, b):
        """Share of `a` with a partner in `b` inside the tolerance."""
        idx = np.searchsorted(b, a)
        best = np.full(len(a), np.inf)
        for shift in (-1, 0):
            probe = np.clip(idx + shift, 0, len(b) - 1)
            best = np.minimum(best, np.abs(b[probe] - a))
        return float(np.mean(best <= tolerance_s))

    precision = covered(chart_arr, peaks)
    recall = covered(peaks, chart_arr)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)
