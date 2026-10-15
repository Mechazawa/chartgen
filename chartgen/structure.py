"""Reuse note patterns across repeated sections of a song.

Measured on the local library: human charts show a median 55% best-match
pattern overlap between similar-length sections; ours showed 13%. We re-chart
every section from scratch, so a returning chorus arrives with different frets
than the first one — the specific thing that reads as machine-made even when
every note lands on a real onset.

Charters reuse only partially, though: same-named sections overlap 28% at the
median against an 8% control, and 44% of pairs share at least half their notes.
So this does not paste a section wholesale. It keeps the later section's own
RHYTHM — which follows what the audio actually does there — and takes the
earlier section's LANE choices wherever the two line up. Where the sections
genuinely differ, the later one keeps its own material.
"""

from itertools import pairwise

import numpy as np

OPEN = 7


def bar_profiles(y, sr, tempo, n_bars: int, beats_per_bar: int = 4):
    """Per-bar (chroma, onset-pattern) fingerprints, or None where a bar
    is too short/quiet to characterise.

    Chroma says WHAT the bar plays (harmony, timbre-blind); the onset
    envelope resampled to 16 slots says WHEN it plays. A riff is the same
    answer to both questions, which is what lets bar unification stamp
    rhythm as well as lanes - section_signatures alone could never make
    two bars identical because it kept each bar's own rhythm.
    """
    import librosa

    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    onset = librosa.onset.onset_strength(y=y, sr=sr)
    times_c = librosa.times_like(chroma, sr=sr)
    times_o = librosa.times_like(onset, sr=sr)
    res = tempo.resolution
    bar = beats_per_bar * res

    profiles = []
    for b in range(n_bars):
        t0 = tempo.beat_to_time(b * bar / res)
        t1 = tempo.beat_to_time((b + 1) * bar / res)
        cw = chroma[:, (times_c >= t0) & (times_c < t1)]
        ow = onset[(times_o >= t0) & (times_o < t1)]
        if cw.shape[1] < 4 or ow.size < 8:
            profiles.append(None)
            continue
        cvec = cw.mean(axis=1)
        cn = float(np.linalg.norm(cvec))
        # 16 slots per bar: enough to distinguish rhythms down to 16ths.
        slots = np.interp(np.linspace(0, 1, 16, endpoint=False),
                          np.linspace(0, 1, ow.size, endpoint=False), ow)
        on = float(np.linalg.norm(slots))
        if cn == 0 or on == 0:
            profiles.append(None)
            continue
        profiles.append((cvec / cn, slots / on))
    return profiles


def unify_riff_bars(notes, profiles, resolution: int,
                    min_chroma: float = 0.97, min_onset: float = 0.93,
                    min_cluster: int = 3, min_cluster_sim: float = 0.25,
                    min_member_sim: float = 0.30, beats_per_bar: int = 4):
    """Stamp each riff's bars with one consensus note pattern.

    The single strongest machine-tell measured on the 800-chart library:
    the median human chart's most-repeated bar occurs 8x and 48% of its
    bars belong to a >=3x-repeated pattern; ours repeated nothing, ever
    (97 bars, 97 patterns on a song whose human chart strums one riff
    bar 12 times). Players learn a riff physically; a chart that never
    repeats can't be learned, only read.

    Validated against a 19-song calibration sample (2026-08-29): pairwise
    leader-stamping topped out at 39% agreement with the human chart even
    at brutal audio thresholds, because it propagates the leader bar's own
    transcription noise. What works is what charters do - ONE canonical
    pattern per riff: cluster bars by sound alone (chroma AND onset-shape
    cosine against the earliest cluster leader, so clusters cannot drift),
    then majority-vote the pattern across the cluster. On human charts the
    consensus matches the charter's bars at 0.90 median similarity; on
    generated charts it lifts repeated-bar coverage from 0% to ~15-44% per
    song, all of it audio-evidenced denoising.

    Guards, each earning its place in validation:
    - clusters under min_cluster members are not riffs and stay untouched;
    - a cluster whose members do not already agree (mean similarity to
      consensus below min_cluster_sim) is incoherent - skipped whole. The
      0.25 default is deliberately below the 0.40 first guess: freshly
      generated charts carry more cross-repeat transcription noise than
      the library charts the gate was first tuned on (live clusters
      measured 0.27-0.34 mean), and the member guard below is what
      actually protects deliberate variation. At 0.25, human-chart churn
      stays at 5% median / 10% p90 on the calibration sample;
    - a member bar below min_member_sim to the consensus is a fill or a
      variation, exactly what human charters keep distinct - left alone;
    - empty bars stay empty (matching audio is not a licence to invent
      notes where generation heard none), and stamped sustains are
      clamped so they can never overlap whatever follows the bar.
    """
    if not notes or not profiles:
        return notes, 0
    bar = beats_per_bar * resolution
    grid = resolution // 12  # offsets keyed to 48ths: exact for 16ths+triplets
    by_bar: dict[int, list] = {}
    for note in notes:
        by_bar.setdefault(note[0] // bar, []).append(note)

    def fingerprint(b):
        out: dict[int, list] = {}
        for tick, lane, sus in by_bar.get(b, ()):
            out.setdefault(round((tick % bar) / grid), []).append((lane, sus))
        return {off: (tuple(sorted(l for l, _ in group)),
                      max(s for _, s in group))
                for off, group in out.items()}

    leaders: list[int] = []
    cluster_of: dict[int, int] = {}
    for b in range(len(profiles)):
        p = profiles[b]
        if p is None or not by_bar.get(b):
            continue
        match = None
        for lead in leaders:
            q = profiles[lead]
            if (float(np.dot(p[0], q[0])) >= min_chroma
                    and float(np.dot(p[1], q[1])) >= min_onset):
                match = lead
                break
        if match is None:
            leaders.append(b)
            cluster_of[b] = b
        else:
            cluster_of[b] = match

    clusters: dict[int, list[int]] = {}
    for b, lead in cluster_of.items():
        clusters.setdefault(lead, []).append(b)

    def similarity(fp, cons):
        a = {(off, lanes) for off, (lanes, _) in fp.items()}
        c = {(off, lanes) for off, (lanes, _) in cons.items()}
        return len(a & c) / len(a | c) if a and c else 0.0

    all_ticks = sorted({n[0] for n in notes})
    stamped_bars = 0
    out_by_bar = dict(by_bar)
    for members in clusters.values():
        if len(members) < min_cluster:
            continue
        fps = {b: fingerprint(b) for b in members}
        votes: dict[int, dict] = {}
        for fp in fps.values():
            for off, (lanes, sus) in fp.items():
                slot = votes.setdefault(off, {"lanes": {}, "sus": []})
                slot["lanes"][lanes] = slot["lanes"].get(lanes, 0) + 1
                slot["sus"].append(sus)
        consensus = {}
        for off, slot in votes.items():
            if sum(slot["lanes"].values()) < len(members) / 2:
                continue  # an offset most bars do not play is not the riff
            lanes = max(slot["lanes"], key=slot["lanes"].get)
            sus = int(np.median(slot["sus"]))
            consensus[off] = (lanes, sus)
        if not consensus:
            continue
        sims = {b: similarity(fps[b], consensus) for b in members}
        if float(np.mean(list(sims.values()))) < min_cluster_sim:
            continue
        for b in members:
            if sims[b] < min_member_sim:
                continue  # a fill or variation; charters keep those distinct
            start = b * bar
            limit = next((t for t in all_ticks if t >= start + bar), None)
            stamped = []
            for off, (lanes, sus) in sorted(consensus.items()):
                tick = start + off * grid
                if limit is not None and sus > 0:
                    sus = min(sus, max(0, limit - tick - resolution // 8))
                stamped.extend((tick, lane, sus) for lane in lanes)
            out_by_bar[b] = stamped
            stamped_bars += 1

    if not stamped_bars:
        return notes, 0
    out = []
    for b in sorted(out_by_bar):
        out.extend(out_by_bar[b])
    return sorted(set(out)), stamped_bars


def section_signatures(y, sr, tempo, section_ticks: list[int]):
    """One normalised mean-chroma vector per section, or None where a section
    is too short to characterise. Chroma because it captures harmony while
    ignoring timbre and loudness — a chorus is the same chords each time."""
    import librosa

    if not section_ticks:
        return []
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    times = librosa.times_like(chroma, sr=sr)
    res = tempo.resolution

    signatures = []
    edges = list(section_ticks) + [None]
    for start, end in pairwise(edges):
        t0 = tempo.beat_to_time(start / res)
        t1 = tempo.beat_to_time(end / res) if end is not None else float(times[-1])
        window = chroma[:, (times >= t0) & (times < t1)]
        if window.shape[1] < 4:
            signatures.append(None)
            continue
        mean = window.mean(axis=1)
        norm = float(np.linalg.norm(mean))
        signatures.append(mean / norm if norm else None)
    return signatures


def find_repeats(signatures, lengths, min_similarity: float = 0.95,
                 length_tolerance: float = 0.15) -> dict[int, int]:
    """{later section index: the earlier section it repeats}.

    Requires near-identical harmony AND comparable length: a chorus that runs
    twice as long is a different arrangement, and copying into it would drift
    out of alignment.
    """
    repeats: dict[int, int] = {}
    for i, sig_i in enumerate(signatures):
        if sig_i is None or lengths[i] <= 0:
            continue
        for j in range(i):
            sig_j = signatures[j]
            if sig_j is None or lengths[j] <= 0:
                continue
            ratio = lengths[i] / lengths[j]
            if not (1 - length_tolerance) <= ratio <= (1 + length_tolerance):
                continue
            if float(np.dot(sig_i, sig_j)) >= min_similarity:
                # The EARLIEST acceptable match, deliberately, not the
                # closest one: every instance of a part should trace back to
                # the same canonical section, so all the choruses inherit from
                # the first chorus rather than each from its predecessor.
                # (Whether "closest" would do better is untested — the run
                # that appeared to show it losing used an inconsistent
                # metric.)
                repeats[i] = j
                break
    return repeats


def reuse_patterns(notes, section_ticks: list[int], repeats: dict[int, int]):
    """Copy lane choices from a section onto later repeats of it.

    Reads every source pattern from the ORIGINAL notes, so a chain of repeats
    all inherit from the first instance rather than from each other.
    """
    if not notes or not repeats:
        return notes

    by_tick: dict[int, list[tuple[int, int]]] = {}
    for tick, lane, sus in notes:
        by_tick.setdefault(tick, []).append((lane, sus))

    last = max(by_tick) + 1
    edges = list(section_ticks) + [last]
    bounds = list(pairwise(edges))
    updated = {tick: list(group) for tick, group in by_tick.items()}

    for target, source in sorted(repeats.items()):
        if target >= len(bounds) or source >= len(bounds):
            continue
        (t_start, t_end), (s_start, _) = bounds[target], bounds[source]
        offset = t_start - s_start
        for tick in [t for t in by_tick if t_start <= t < t_end]:
            origin = by_tick.get(tick - offset)
            if not origin:
                continue  # the repeat plays something the first pass did not
            # The target keeps its own sustain: spacing there is its own.
            sus = max(s for _, s in by_tick[tick])
            updated[tick] = [(lane, sus) for lane, _ in origin]

    out = {(tick, lane, sus)
           for tick, group in updated.items() for lane, sus in group}
    return sorted(out)
