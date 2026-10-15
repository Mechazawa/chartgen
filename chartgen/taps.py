"""Mark keys/synth-led sections as tap notes. On by default since 2026-09-05.

No charting standard documents when to tap (YARG: "use them sensibly", and
nothing else anywhere), and measurement settled that the community has no
rule at all: on 232 genre-balanced charts (76k tapped positions, 34% base
rate) the best acoustic predictor of human taps reaches 39-42% precision -
barely above chance. Human taps are charter style, not acoustics. So this
implements a chosen philosophy rather than a mimicked one: taps mark SOFT
sounds - the piano lines, plucks and gentle synth runs where a strum is
the wrong physical gesture.

The section rule is the measured part (v4 instrument keying, then the
followed-instrument timeline); the constants below are still swept against the human tap
positions (tools/sweep_taps.py) so "soft" means something calibrated
rather than guessed.

Taps are applied to PHRASES, never lone notes: a single tapped note in a
strummed line is imperceptible at best and confusing at worst, and human
taps arrive in runs. In .chart a tap is `N 6 0` at the note's tick; in CH
it plays without strumming and overrides HOPO state.
"""

from itertools import pairwise

import numpy as np

from .tempo import ONSET_LAG_S

# Softness blend. The 24-song library sweep favoured brightness; the
# 232-song genre-balanced re-sweep flipped it - attack edges out
# brightness at scale (42% vs 40% precision over a 34% base rate) and the
# two libraries disagreeing is itself the finding: no acoustic axis
# predicts human taps reliably, so the weights follow the PHILOSOPHY.
# Attack leads because "soft sounds" means soft attacks; darkness stays as
# a half-weight tiebreak; loudness earned nothing anywhere.
WEIGHTS = {"attack": 1.0, "bright": 0.5, "loud": 0.0}
# How soft, in per-song standard deviations, a note must be to join a
# tapped phrase. Mild on purpose: the phrase requirement below does the
# filtering that a harsher note-level bar would do worse.
SOFT_Z = 0.25
# A phrase: at least this many qualifying notes, no gap wider than this.
MIN_RUN = 4
MAX_GAP_BEATS = 1.0
# Never tap more of the chart than this - within MIXED sections only. A
# uniformly soft song (a piano piece) taps whole, because playtest proved
# the alternative: per-song z-scores on an all-piano song split the notes
# around the song's own average and the 40% cap truncated by rank, so
# most notes were not tapped and the boundaries fell at statistically
# arbitrary places while the piano never changed character.
MAX_SHARE = 0.40

# Section texture follows the FOREGROUND, not the backing. Drums-share
# classification failed its first playtest for a reason the player named
# exactly: "the piano was pretty much always main and there was like a
# snare to it" - a backing beat vetoed taps on a piano-led song. The
# chart follows the loudest melodic content, so the tap decision judges
# THAT instrument: the Demucs other-stem share of section energy (where
# piano/synth/guitar live). Measured on the three-song test matrix:
# Luv Letter 0.62-0.99 in every section, In the End piano intro
# 0.83-0.86 vs band 0.14-0.22, Clocks verses 0.31-0.44 (genuinely
# shared foreground). The centroid guard keeps a DISTORTED lead - other-
# dominant but harsh - from reading as soft: soft sections measured
# 700-1600 Hz, noise/harsh 4400+.
FOREGROUND_SOFT_SHARE = 0.60   # other-stem dominance: section taps whole
FOREGROUND_HARD_SHARE = 0.35   # below: band/vocal foreground, no taps
SOFT_MAX_CENTROID_HZ = 2500.0

# Taps v4: INSTRUMENT keying on BS-RoFormer SW's true stems. The v3 rule
# above reads the shim's "other", which sums guitar, piano and the synth
# residual - and that sum CANCELS the signal. Six-stem study (2026-09-03,
# tools/dump_taps_stems.py + sweep_taps_stems.py; 300 songs, 1822 scored
# sections, 710 tapped-whole): charters tap where the guitar stem is
# silent and piano/synth carry the section. Median tapped vs untapped:
# guitar share 0.00 vs 0.21, piano+synth 0.30 vs 0.05. The keyed score
# share(piano) + share(synth) - share(guitar) separates at AUC 0.86
# (shim 0.58); the shim's best rule reaches 51% precision at 47% recall,
# this one 75%/77%, and split-half by song holds 69-79% precision with
# the threshold stable at 0.06-0.12. Below the lower bar the section is
# a band/guitar foreground: 6% of those are tapped by humans, so none
# tap; the middle band taps 33% of the time and falls through to the
# phrase logic. Centroid gates added nothing and are not used here.
KEYS_TAP_SHARE = 0.09      # keyed score at/above: section taps whole
KEYS_NO_TAP_SHARE = -0.09  # (v4 no longer uses a middle band; kept for the sweep)
TRUE_STEMS = ("drums", "bass", "vocals", "guitar", "piano", "sw_other")
# A keyed section taps WHOLE, chords included. The library statistic that
# argued otherwise (11.9% of human-tapped positions are chords vs 28.5%
# untapped; Faded's charter taps 400 notes and zero chords) describes how
# charters VOICE tapped passages - as single-note lines - not a habit of
# strumming the chords inside them. Leaving chords strummed inside a
# tapped section was playtested on Clocks (2026-09-03) and rejected
# outright: "definitely no mixing taps and strums together, it's awful",
# and the piano intro, charted with chords, lost most of its taps.
TAP_CHORDS_IN_KEYED_SECTIONS = True


def _samples(feature, times, when):
    """Strongest value within one frame either side, like density.py."""
    i = int(np.searchsorted(times, when))
    lo, hi = max(0, i - 2), min(len(feature), i + 3)
    return float(feature[lo:hi].max()) if hi > lo else 0.0


def softness_by_tick(notes, y, sr, tempo) -> dict[int, float]:
    """Per-position softness in per-song z-units; higher = softer."""
    import librosa

    res = tempo.resolution
    ticks = sorted({t for t, _, _ in notes})
    if len(ticks) < MIN_RUN:
        return {}

    onset = librosa.onset.onset_strength(y=y, sr=sr)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    rms = librosa.feature.rms(y=y)[0]
    times = librosa.times_like(onset, sr=sr)

    raw = {}
    for tick in ticks:
        when = tempo.beat_to_time(tick / res)
        raw[tick] = {
            "attack": _samples(onset, times - ONSET_LAG_S, when),
            "bright": _samples(centroid, times, when),
            "loud": _samples(rms, times, when),
        }
    z = {}
    stats = {}
    for key in WEIGHTS:
        values = np.array([raw[t][key] for t in ticks], dtype=float)
        stats[key] = (float(values.mean()), float(values.std()) or 1e-9)
    total = sum(WEIGHTS.values())
    for tick in ticks:
        # Negated: high attack/brightness/level is HARD, so softness is the
        # weighted distance below the song's own norm.
        z[tick] = -sum(
            WEIGHTS[k] * (raw[tick][k] - stats[k][0]) / stats[k][1]
            for k in WEIGHTS
        ) / total
    return z


def phrases(notes, softness: dict[int, float], resolution: int,
            soft_z: float = None, min_run: int = None,
            max_gap_beats: float = None, max_share: float = None) -> set[int]:
    """Ticks to tap: maximal soft runs of MIN_RUN+ notes, share-capped.

    The parameters exist for tools/sweep_taps.py; production callers use
    the module constants.
    """
    soft_z = SOFT_Z if soft_z is None else soft_z
    min_run = MIN_RUN if min_run is None else min_run
    max_gap_beats = MAX_GAP_BEATS if max_gap_beats is None else max_gap_beats
    max_share = MAX_SHARE if max_share is None else max_share
    ticks = sorted({t for t, _, _ in notes})
    if not ticks:
        return set()
    max_gap = max_gap_beats * resolution

    runs, current = [], []
    for tick in ticks:
        soft = softness.get(tick, 0.0) >= soft_z
        if soft and (not current or tick - current[-1] <= max_gap):
            current.append(tick)
            continue
        if len(current) >= min_run:
            runs.append(current)
        current = [tick] if soft else []
    if len(current) >= min_run:
        runs.append(current)

    # Softest phrases first, then stop at the share cap: if the whole song
    # reads "soft", tapping all of it would just change instruments.
    runs.sort(key=lambda run: -float(np.median([softness[t] for t in run])))
    chosen: set[int] = set()
    budget = int(max_share * len(ticks))
    for run in runs:
        if len(chosen) + len(run) > budget:
            continue
        chosen.update(run)
    return chosen


def foreground_by_span(spans_s, mono, sr, centroid=None, cen_times=None):
    """[(other_share, other_centroid_hz)] per (t0, t1) span.

    other_share = the melodic-instrument stem's share of total stem
    energy; the centroid of that stem says whether the foreground is
    soft (piano/pluck) or harsh (distorted lead).
    """
    hop = int(0.05 * sr)
    envs = {}
    # Only the four canonical stems count toward the total: the SW shim in
    # chartgen.stems also returns guitar/piano/sw_other, whose energy is
    # ALREADY inside its reconstructed "other" - summing every key would
    # double-count it and silently shrink every share.
    canonical = ("drums", "bass", "other", "vocals")
    for name, stem in mono.items():
        if name not in canonical:
            continue
        n = len(stem) // hop
        envs[name] = np.sqrt((stem[:n * hop].reshape(n, hop) ** 2).mean(axis=1))
    frames = len(next(iter(envs.values())))
    out = []
    for t0, t1 in spans_s:
        a = max(0, min(frames - 1, int(t0 / 0.05)))
        b = max(a + 1, min(frames, int(t1 / 0.05)))
        total = sum(float(env[a:b].mean()) for env in envs.values())
        share = float(envs["other"][a:b].mean()) / total if total > 0 else 0.0
        bright = 0.0
        if centroid is not None:
            w = (cen_times >= t0) & (cen_times < t1)
            bright = float(centroid[w].mean()) if w.any() else 0.0
        out.append((share, bright))
    return out


def section_tap_ticks(inside, tap_chords: bool = None):
    """Ticks a whole-section tap marks: every position, or only the
    single-note ones when chords are to stay strummed."""
    tap_chords = (TAP_CHORDS_IN_KEYED_SECTIONS if tap_chords is None
                  else tap_chords)
    lanes: dict[int, set] = {}
    for t, lane, _ in inside:
        lanes.setdefault(t, set()).add(lane)
    if tap_chords:
        return set(lanes)
    return {t for t, ls in lanes.items() if len(ls - {7}) <= 1}


def keys_by_span(spans_s, mono, sr):
    """Keyed foreground score per (t0, t1) span, or None without the six
    true SW stems: share(piano) + share(synth residual) - share(guitar),
    each a share of the six true stems' summed RMS energy (no shim
    "other", so nothing is double counted)."""
    if any(k not in mono for k in TRUE_STEMS):
        return None
    hop = int(0.05 * sr)
    envs = {}
    for name in TRUE_STEMS:
        stem = mono[name]
        n = len(stem) // hop
        if n < 1:
            return None
        envs[name] = np.sqrt((stem[:n * hop].reshape(n, hop) ** 2).mean(axis=1))
    frames = min(len(e) for e in envs.values())
    out = []
    for t0, t1 in spans_s:
        a = max(0, min(frames - 1, int(t0 / 0.05)))
        b = max(a + 1, min(frames, int(t1 / 0.05)))
        means = {k: float(e[a:b].mean()) for k, e in envs.items()}
        total = sum(means.values())
        if total <= 0:
            out.append(0.0)
            continue
        out.append((means["piano"] + means["sw_other"] - means["guitar"]) / total)
    return out


# Followed-instrument runs (chartgen.prominence): a run followed on keys or
# synth taps whole, one followed on guitar, bass or vocals stays strummed,
# and a run the model had no opinion on falls back to the keyed-share
# rule on its own span. The runs are 4-bar-granular and smoothed, so the
# Sexualizer case - a 20 s guitar passage inside a synth song, which one
# 57 s chroma section averaged into "tap it all" - gets its own run.
TAP_STEMS = ("piano", "sw_other")
STRUM_STEMS = ("guitar", "vocals")
# Bass is NEUTRAL: a bass-followed run adopts the decision of the runs
# around it and never flips taps on its own. Playtest (Sexualizer,
# 2026-09-05): two-window bass runs inside a synth song strummed, "strums
# mixed into a mostly-taps section that didn't have much audio change" -
# the bass taking over is real but is not an instrument change a player
# hears, unlike In the End's piano-to-guitar switches, which are and
# stay. In the 299-song dump 14% of bass runs are short and sandwiched
# in synth. Bass decides for itself (strum) only where it is the song's
# main followed stem.


def detect(notes, y, sr, tempo, progress=lambda m: None,
           audio_path=None, section_marks=None, followed=None) -> set[int]:
    """Tap ticks for this chart, or an empty set when nothing reads soft.

    Two-level decision. Sections classify by their FOREGROUND. With the
    six true SW stems (v4): a keys/synth-led section (piano + synth share
    minus guitar share >= KEYS_TAP_SHARE) taps whole, every other section
    not at all - never a mix inside one section. With only the four Demucs-style
    stems (v3): the melodic stem's dominance plus a centroid guard decide
    the same three ways. Without stems or sections the old full-mix
    behaviour stands.
    """
    res = tempo.resolution
    ticks = sorted({t for t, _, _ in notes})
    if not ticks:
        return set()

    mono = None
    if audio_path and section_marks:
        try:
            from . import stems as stemsmod

            mono = stemsmod.separate(str(audio_path), progress)
        except Exception:
            mono = None

    if mono is None or not section_marks:
        try:
            soft = softness_by_tick(notes, y, sr, tempo)
        except Exception as error:  # decoration, never worth failing a chart
            progress(f"      tap detection skipped: {type(error).__name__}: {error}")
            return set()
        chosen = phrases(notes, soft, res)
        if chosen:
            share = len(chosen) / max(1, len(ticks))
            progress(f"      taps: {len(chosen)} notes ({share:.0%}) in soft phrases")
        return chosen

    from . import stems as stemsmod

    stems_of = None
    if followed:
        # Sections ARE the followed-instrument runs.
        spans = [(int(r["beat0"] * res), int(r["beat1"] * res)) for r in followed]
        spans_s = [(r["t0"], r["t1"]) for r in followed]
        stems_of = [r["stem"] for r in followed]
    else:
        bounds = [t for t, _ in sorted(section_marks)] + [ticks[-1] + 1]
        spans = list(pairwise(bounds))
        spans_s = [(tempo.beat_to_time(a / res), tempo.beat_to_time(b / res))
                   for a, b in spans]
    import librosa

    chosen: set[int] = set()
    mixed_notes = []
    soft_sections = band_sections = 0
    keyed = keys_by_span(spans_s, mono, stemsmod.SR)
    if keyed is not None:
        # v4: instrument keying on the true stems (see KEYS_TAP_SHARE), or
        # the followed-instrument run's stem when a timeline is given.
        rule = "followed-instrument" if stems_of else "keys/synth"
        fallback = 0
        decisions = []
        for i, score in enumerate(keyed):
            stem = stems_of[i] if stems_of else None
            if stem in TAP_STEMS:
                decisions.append(True)
            elif stem in STRUM_STEMS:
                decisions.append(False)
            elif stem == "bass":
                decisions.append(None)  # resolved from the neighbours below
            else:
                decisions.append(score >= KEYS_TAP_SHARE)
                fallback += bool(stems_of)
        if stems_of:
            weights = [b - a for a, b in spans]
            bass_main = (sum(w for w, st in zip(weights, stems_of, strict=True) if st == "bass")
                         > 0.5 * sum(weights))
            for i, d in enumerate(decisions):
                if d is not None:
                    continue
                if bass_main:
                    decisions[i] = False
                    continue
                prev = next((decisions[j] for j in range(i - 1, -1, -1) if decisions[j] is not None), None)
                nxt = next((decisions[j] for j in range(i + 1, len(decisions)) if decisions[j] is not None), None)
                if prev is None and nxt is None:
                    decisions[i] = False
                elif prev is None or nxt is None:
                    decisions[i] = prev if nxt is None else nxt
                elif prev == nxt:
                    decisions[i] = prev
                else:
                    # neighbours disagree: follow the longer one
                    lp = next(weights[j] for j in range(i - 1, -1, -1) if decisions[j] is not None)
                    ln = next(weights[j] for j in range(i + 1, len(decisions)) if decisions[j] is not None)
                    decisions[i] = prev if lp >= ln else nxt
        for i, (a, b) in enumerate(spans):
            inside = [n for n in notes if a <= n[0] < b]
            if not inside:
                continue
            tap = decisions[i]
            # Whole sections only: the note-level phrase fallback for the
            # middle band was playtested on In the End (2026-09-03) and
            # rejected - it mixed taps and strums inside guitar-forward
            # verses. A section either taps whole or not at all.
            if tap:
                chosen.update(section_tap_ticks(inside))
                soft_sections += 1
            else:
                band_sections += 1
        if stems_of and fallback:
            progress(f"      taps: {fallback} run(s) without a followed instrument "
                     f"decided by the keyed-share rule")
    else:
        # v3: shim/Demucs foreground dominance with a centroid guard.
        rule = "soft"
        other_ds = librosa.resample(mono["other"], orig_sr=stemsmod.SR,
                                    target_sr=22050)
        centroid = librosa.feature.spectral_centroid(y=other_ds, sr=22050)[0]
        cen_times = librosa.times_like(centroid, sr=22050)
        fg = foreground_by_span(spans_s, mono, stemsmod.SR, centroid, cen_times)
        for (a, b), (share, bright) in zip(spans, fg, strict=True):
            inside = [n for n in notes if a <= n[0] < b]
            if not inside:
                continue
            if share >= FOREGROUND_SOFT_SHARE and bright <= SOFT_MAX_CENTROID_HZ:
                chosen.update(t for t, _, _ in inside)
                soft_sections += 1
            elif share > FOREGROUND_HARD_SHARE:
                mixed_notes.extend(inside)
            else:
                band_sections += 1

    if mixed_notes:
        # De-drummed audio: the melodic content without the kit, resampled
        # to the analysis rate the full-mix path used.
        try:
            import librosa

            dedrummed = (mono["vocals"] + mono["other"] + mono["bass"])
            dedrummed = librosa.resample(dedrummed, orig_sr=stemsmod.SR,
                                         target_sr=sr)
            soft = softness_by_tick(mixed_notes, dedrummed, sr, tempo)
            chosen.update(phrases(mixed_notes, soft, res))
        except Exception:
            pass

    share = len(chosen) / max(1, len(ticks))
    progress(f"      taps: {len(chosen)} notes ({share:.0%}) - "
             f"{soft_sections} {rule} section(s) tapped whole, "
             f"{band_sections} band/guitar section(s) untouched, "
             f"{len(mixed_notes)} notes in shared sections went to phrase logic")
    return chosen
