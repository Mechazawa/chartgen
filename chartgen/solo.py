"""Find the guitar solo by listening for one, not by waiting for a lyric gap.

The old rule marked any chroma section with no transcribed words that played
busier than average. Measured against 141 human charts, every assumption in
it is wrong:

- only 30% of songs carry a solo at all, so marking one is the exception;
- a lyric-free gap of two bars or more contains a solo 18% of the time —
  intros, bridges, breakdowns and outros all look identical under "nobody is
  singing";
- solos are not reliably busier. Median density is 1.20x the song average but
  the lower quartile is 0.89x, so a >=1.2x gate throws away half of them;
- they are long: median 64 beats, about 16 bars.

A sweep over 508 lyric-free gaps found no combination of chart-only features
above ~20% precision, which settles where the missing signal has to come
from. Pati & Lerch (AES 2017, "A Dataset and Method for Electric Guitar Solo
Detection in Rock Music") define a solo as a lead voice in the FOREGROUND,
playing phrases that DO NOT REPEAT, for longer than two measures — and their
classifier reaches 63% precision from three feature groups: spectral
baseline, predominant pitch with its confidence, and structural repetition.

That is the shape copied here, minus the SVM (no training data) and minus
Melodia (Essentia). librosa's pyin gives the pitch
track and its voiced probability; the chroma sections the pipeline already
computes give the repetition feature. Since even a purpose-built classifier
tops out near 63% precision, this is deliberately biased toward silence: a
section has to look like a solo on several independent counts before it is
marked, and songs with nothing convincing get no marker at all — which is
what 70% of human charts do anyway.
"""

from itertools import pairwise

import numpy as np

# Every constant below came out of tools/sweep_solos.py, hits IoU-gated at
# >= 0.25 against the human marker. Recalibrated 2026-08-24 against 359
# songs / 492 human solos from the genre-balanced Chorus Encore pull —
# and scale changed the answer: brightness, the winning feature on the
# 60-song local library (100% precision there), collapsed on diverse data
# and earns NO weight. What survives across genres is melodic movement
# (spread) and non-repetition (novelty), with the lead-line share as a
# half-weight assist. The recalibrated rule scores ~75% precision at ~5%
# recall across ten half-splits (58-100% band) with 20/20 solo-less songs
# left unmarked — beating the library-tuned rule on BOTH axes at scale
# (64%/3%). Low recall is still the accepted price, and the binding
# constraint is candidate generation: chroma-section boundaries reach the
# broader corpus's solos at a median IoU of only 0.34, so better
# segmentation, not scoring, is where recall lives.

# Length. The human library's lower quartile is 47 beats; the sweep pushed
# the floor there and no lower — every shorter floor it was offered let
# grazing false positives in.
MIN_BEATS = 48.0
MAX_BEATS = 128.0
# Position. Library p25/p75 is 0.49-0.76 of the way through; the sweep chose
# nearly that exact band over wider ones.
MIN_POS, MAX_POS = 0.45, 0.80
# A stray misheard word should not veto a real solo (32% of human solos have
# a word inside), but singing through it should.
MAX_WORDS = 2
# Lead register: above the bass and the body of a male vocal, low enough to
# keep a guitar's middle octaves.
FMIN_HZ, FMAX_HZ = 180.0, 1400.0
HOP = 1024

# A solo can straddle two chroma segments — the segmenter splits on harmonic
# change, and a solo that moves through two key centres becomes two of them.
# Allowing short runs of sections as candidates raised the best achievable
# overlap with real human solos from 0.50 to 0.64 without loosening any
# boundary.
MAX_RUN = 3
# What a solo is, per 359 genre-diverse songs: a passage whose melody
# MOVES (spread) and appears NOWHERE ELSE in the song (novelty), with a
# clear lead line (voiced) as supporting evidence. Brightness is zeroed on
# purpose — dominant on the local library, it inverted into noise on the
# genre-balanced corpus, the clearest case yet of a threshold that would
# have silently encoded one library's mixing style. confidence stays at
# zero — whatever it knows, voiced already knows.
FEATURES = ("voiced", "confidence", "spread", "bright", "novelty")
WEIGHTS = {"voiced": 0.5, "confidence": 0.0, "spread": 1.0,
           "bright": 0.0, "novelty": 1.0}
MIN_Z = 1.25  # 0.5 standard deviations x the total weight of 2.5
# Demucs "other"-stem energy, as a per-song z-score: the lead instrument
# surging even when the full mix stays flat. Swept on 299 songs with stem
# features (2026-08-25): adding it doubled true positives (22 -> 45) at
# 66% precision / 10% recall, stable across ten half-splits (53-76%).
# The stem singing-share feature earned nothing - the melodic features
# already avoid sung sections - and stays out. With stems unavailable the
# rule falls back to the full-mix score and threshold.
LEAD_WEIGHT = 2.0
MIN_Z_STEMS = 1.5
MAX_SOLOS = 2           # human median is 1, max 3
# Guitar-stem rule (BS-RoFormer SW, 2026-09-02). The evidence class every
# earlier sweep lacked: the guitar stem's surge against the song's own
# guitar baseline, plus a guitar PRESENCE gate and a vocal-silence gate.
# Controlled recalibration on 299 solo songs + 100 solo-less controls:
# zero-quiet-fire rules plateau at 83% precision / 7% recall, and the
# rule below was picked by 7 of 10 half-splits, scoring 82% / 6% held-out
# with ~1 quiet fire per 50 controls. The Demucs-era family cannot reach
# zero quiet fires above 2% recall, and its usable rules fire on 14-22 of
# the same 100 controls. lead is ZERO here: guitar surge replaces the
# "other"-stem term. Without a guitar stem the lead rule above still runs.
GUITAR_WEIGHT = 2.0
MIN_GUITAR_SHARE = 0.10   # a solo needs a real guitar in the section
MAX_SINGING = 0.40        # vocal stem active in <= 40% of the span
MIN_Z_GUITAR = 1.25
MIN_BEATS_GUITAR = 32.0   # the split-halves chose 32 over 48 every time


def _section_spans(section_marks, last_tick, resolution):
    bounds = [t for t, _ in section_marks] + [last_tick + 1]
    return list(pairwise(bounds))


def section_evidence(y, sr, tempo, spans, resolution, analysis=None):
    """[(dict | None)] of lead-voice evidence per section, plus novelty.

    Split out from detect() so the threshold sweep in tools/ can cache these
    once per song — the audio analysis costs ~30s and the thresholds needed
    dozens of passes over the same songs to settle. `analysis` is the
    output of analyze(y, sr) when the caller computed it earlier (the
    pipeline runs it on a CPU thread while the GPU separates stems).
    """
    rows = _features(y, sr, tempo, spans, resolution, analysis)
    for row, fresh in zip(rows, _novelty(rows), strict=True):
        if row is not None:
            row["novelty"] = fresh
            row.pop("chroma", None)
    return rows


def analyze(y, sr) -> dict:
    """The song-wide, chart-independent half of the solo evidence: pitch
    track, centroid and chroma of the harmonic part, with their time axes.
    ~30 s of CPU on a 4-minute song; nothing here depends on the chart,
    the sections or the tempo map, so it can run while the GPU separates.
    """
    import librosa

    # Percussive energy carries no melody; the lead line lives in the
    # harmonic part, and separating them keeps a busy drum fill from reading
    # as a bright, active section.
    harmonic = librosa.effects.harmonic(y, margin=2.0)
    # hop 1024 rather than the default 512: pyin dominates the cost of the
    # whole detector (20.5s of a 33s pass on a 3.5-minute song, halved to
    # 10.1s here) and every feature below is a section-level average, so 46ms
    # of pitch resolution is already far finer than anything that is read
    # off it. Measured on the same song, the voiced share moved 0.559 ->
    # 0.571, which is noise against a per-section z-score.
    f0, voiced, voiced_prob = librosa.pyin(
        harmonic, fmin=FMIN_HZ, fmax=FMAX_HZ, sr=sr, fill_na=np.nan,
        hop_length=HOP,
    )
    times = librosa.times_like(f0, sr=sr, hop_length=HOP)
    centroid = librosa.feature.spectral_centroid(y=harmonic, sr=sr)[0]
    centroid_times = librosa.times_like(centroid, sr=sr)
    chroma = librosa.feature.chroma_cqt(y=harmonic, sr=sr)
    chroma_times = librosa.times_like(chroma, sr=sr, hop_length=512)
    return {"f0": f0, "voiced_prob": voiced_prob, "times": times,
            "centroid": centroid, "centroid_times": centroid_times,
            "chroma": chroma, "chroma_times": chroma_times}


def _features(y, sr, tempo, spans, resolution, analysis=None):
    """Per-section lead-voice evidence, computed once over the whole song."""
    a = analysis if analysis is not None else analyze(y, sr)
    f0, voiced_prob, times = a["f0"], a["voiced_prob"], a["times"]
    centroid, centroid_times = a["centroid"], a["centroid_times"]
    chroma, chroma_times = a["chroma"], a["chroma_times"]

    rows = []
    for start, end in spans:
        t0 = tempo.beat_to_time(start / resolution)
        t1 = tempo.beat_to_time(end / resolution)
        window = (times >= t0) & (times < t1)
        cwindow = (centroid_times >= t0) & (centroid_times < t1)
        hwindow = (chroma_times >= t0) & (chroma_times < t1)
        if window.sum() < 8 or hwindow.sum() < 4:
            rows.append(None)
            continue

        pitches = f0[window]
        confident = voiced_prob[window]
        sung = pitches[np.isfinite(pitches)]
        # Semitone spread of the line: a solo moves, a held pad does not.
        spread = (float(np.std(12 * np.log2(np.maximum(sung, 1e-6) / 440.0)))
                  if len(sung) > 4 else 0.0)
        rows.append({
            "voiced": float(np.mean(np.isfinite(pitches))),
            "confidence": float(np.mean(confident)),
            "spread": spread,
            "bright": float(np.mean(centroid[cwindow])) if cwindow.sum() else 0.0,
            "chroma": chroma[:, hwindow].mean(axis=1),
        })
    return rows


def _novelty(rows):
    """1 - the best chroma match to any other section.

    The formal thing that separates a solo from a riff is that it does not
    repeat. Sections here are already the pipeline's chroma segmentation, so
    comparing their average chroma to every other section says directly
    whether this passage happens anywhere else in the song.
    """
    out = []
    for i, row in enumerate(rows):
        if row is None:
            out.append(0.0)
            continue
        mine = row["chroma"]
        mine = mine / (np.linalg.norm(mine) or 1.0)
        best = 0.0
        for j, other in enumerate(rows):
            if j == i or other is None:
                continue
            theirs = other["chroma"]
            theirs = theirs / (np.linalg.norm(theirs) or 1.0)
            best = max(best, float(np.dot(mine, theirs)))
        out.append(1.0 - best)
    return out


def _zscores(rows):
    """Each feature restated as standard deviations from this song's own mean.

    Absolute thresholds cannot survive the jump from a dense metal mix to a
    sparse acoustic one; "unlike anything else in THIS song" can. It is also
    what makes a single number comparable across the feature set.
    """
    usable = [r for r in rows if r]
    if len(usable) < 3:
        return None
    stats = {}
    for key in FEATURES:
        values = np.array([r[key] for r in usable], dtype=float)
        stats[key] = (float(values.mean()), float(values.std()) or 1e-6)
    for row in rows:
        if row is not None:
            row["z"] = {k: (row[k] - stats[k][0]) / stats[k][1] for k in FEATURES}
    return rows


def _candidates(spans, rows, resolution):
    """Runs of 1..MAX_RUN consecutive sections, with their evidence pooled."""
    usable = [(span, row) for span, row in zip(spans, rows, strict=True) if row is not None]
    out = []
    for i in range(len(usable)):
        for run in range(1, MAX_RUN + 1):
            group = usable[i:i + run]
            if len(group) < run:
                break
            weights = [end - start for (start, end), _ in group]
            total = sum(weights) or 1
            out.append((
                group[0][0][0], group[-1][0][1],
                {k: sum(row["z"][k] * w for (_, row), w in zip(group, weights, strict=True)) / total
                 for k in FEATURES},
            ))
    return out


def _lead_evidence(audio_path, progress):
    """(times, rms, mean, std) from the Demucs other stem, or None."""
    if not audio_path:
        return None
    try:
        import numpy as np

        from . import stems as stemsmod

        mono = stemsmod.separate(str(audio_path), progress)
        if mono is None:
            return None
        _, rms = stemsmod.activity(mono["other"])
        times = (np.arange(len(rms)) + 0.5) * 0.05
        mean, std = float(rms.mean()), float(rms.std()) or 1e-9
        return times, rms, mean, std
    except Exception:
        return None


def _guitar_evidence(audio_path, progress):
    """Per-frame guitar surge stats, canonical envelopes and the vocal stem,
    from the SW backend - or None when it did not run (then the Demucs-era
    lead rule applies unchanged)."""
    if not audio_path:
        return None
    try:
        import numpy as np

        from . import stems as stemsmod

        mono = stemsmod.separate(str(audio_path), progress, backend="sw")
        if not mono or "guitar" not in mono:
            return None
        _, g_rms = stemsmod.activity(mono["guitar"])
        canon = {k: stemsmod.activity(mono[k])[1] for k in stemsmod.DEMUCS_STEMS}
        times = (np.arange(len(g_rms)) + 0.5) * 0.05
        return {"times": times, "guitar": g_rms,
                "mean": float(g_rms.mean()), "std": float(g_rms.std()) or 1e-9,
                "canon": canon, "vocal": mono["vocals"]}
    except Exception:
        return None


def detect(expert, section_marks, lyric_events, y, sr, tempo,
           progress=lambda m: None, audio_path=None,
           analysis=None) -> list[tuple[int, int]]:
    """[(start_tick, end_tick)] where the audio carries a lead break.

    Returns nothing far more often than not, on purpose: 70% of human charts
    mark no solo at all, and a marker over a verse reads as plainly wrong
    where a missing one merely costs a bonus.
    """
    resolution = tempo.resolution
    ticks = sorted({t for t, _, _ in expert})
    if len(ticks) < 64 or not section_marks:
        return []
    words = sorted(t for t, e in lyric_events if e.startswith("lyric"))

    last = ticks[-1]
    spans = _section_spans(section_marks, last, resolution)
    try:
        rows = _zscores(section_evidence(y, sr, tempo, spans, resolution, analysis))
    except Exception as error:  # a solo marker is never worth failing a chart
        progress(f"      solo detection skipped: {type(error).__name__}: {error}")
        return []
    if rows is None:
        return []

    guitar = _guitar_evidence(audio_path, progress)
    lead = None if guitar else _lead_evidence(audio_path, progress)
    if guitar:
        threshold, min_beats = MIN_Z_GUITAR, MIN_BEATS_GUITAR
    else:
        threshold, min_beats = (MIN_Z_STEMS if lead else MIN_Z), MIN_BEATS

    scored = []
    for start, end, z in _candidates(spans, rows, resolution):
        beats = (end - start) / resolution
        if not (min_beats <= beats <= MAX_BEATS):
            continue
        if not (MIN_POS <= start / last <= MAX_POS):
            continue
        if sum(1 for w in words if start <= w < end) > MAX_WORDS:
            continue
        inside = [t for t in ticks if start <= t < end]
        # scan-chart counts a solo containing no notes as a chart defect, and
        # a nearly-empty one scores nothing anyway.
        if len(inside) < 16:
            continue
        score = sum(WEIGHTS[k] * z[k] for k in FEATURES)
        t0 = tempo.beat_to_time(start / resolution)
        t1 = tempo.beat_to_time(end / resolution)
        if guitar:
            from . import stems as stemsmod

            window = (guitar["times"] >= t0) & (guitar["times"] < t1)
            if not window.any():
                continue
            if stemsmod.singing_share(guitar["vocal"], stemsmod.SR, t0, t1) > MAX_SINGING:
                continue
            total = sum(float(v[window].mean()) for v in guitar["canon"].values())
            g_mean = float(guitar["guitar"][window].mean())
            if total <= 0 or g_mean / total < MIN_GUITAR_SHARE:
                continue
            score += GUITAR_WEIGHT * (g_mean - guitar["mean"]) / guitar["std"]
        elif lead:
            times, rms, mean, std = lead
            window = (times >= t0) & (times < t1)
            if window.any():
                score += LEAD_WEIGHT * (float(rms[window].mean()) - mean) / std
        if score >= threshold:
            scored.append((score, inside[0], inside[-1]))

    if not scored:
        return []
    # Overlapping runs describe the same passage; keep the best-scoring one.
    scored.sort(reverse=True)
    chosen: list[tuple[int, int]] = []
    for _, start, end in scored:
        if all(end < a or start > b for a, b in chosen):
            chosen.append((start, end))
        if len(chosen) >= MAX_SOLOS:
            break
    progress(f"      {len(chosen)} solo(s) from "
             f"{'guitar-stem' if guitar else 'lead-line'} evidence")
    return sorted(chosen)
