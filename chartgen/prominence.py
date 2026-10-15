"""Followed-instrument timeline: which stem a charter would follow, per
4-bar window (the Route B prominence prior).

Why a model and not a rule: the 95-song ownership study measured stem
ENERGY and found it predicts nothing about which line humans chart, and
the 299-song Route B dump (2026-09-05) confirmed it - the loudest stem by
energy names the followed instrument in 32% of contested windows, by
K-weighted loudness 35%. But a small gradient-boosting model on the same
level shares reaches 81%, and 91% once per-stem onset density joins in
(92.4% over all labeled windows, song-level 5-fold, precision 96% while
committing on 90% of windows). Labels came from onset matching between
each SW stem's transcription and the human chart, per-song offset
corrected, coincidence split (tools/prominence_pilot.py); the model is
trained by tools/prominence_train.py --save and stored next to this file.

Runtime cost on top of the SW separation every song already pays: four
extra stem transcriptions (~7 s each on CPU; the guitar stem's is reused
when the chord-texture pass already made it) and the features below.

Consumers read the timeline as merged RUNS of consecutive same-stem
windows: taps decide per run, note selection may commit per run. The
window is a 4-bar unit on the chart's own beat grid, the granularity the
charters' switches happen at; single-window blips between two agreeing
neighbours are smoothed away first.
"""
from pathlib import Path

import numpy as np

from .tempo import ONSET_LAG_S

STEMS = ("vocals", "bass", "guitar", "piano", "sw_other")
FEATS = ("energy", "loudness", "density", "pitch", "mono")
LETTER = {"vocals": "V", "bass": "B", "guitar": "G", "piano": "P", "sw_other": "S"}
BARS_PER_WINDOW = 4
MODEL_PATH = Path(__file__).resolve().parent / "prominence_model.joblib"
# Windows below this confidence carry no commitment: consumers treat them
# as "no opinion" (taps fall back to the keyed-share rule, selection stays
# loudness-following). 0.6 sits where held-out precision is ~96%.
MIN_CONFIDENCE = 0.60
MIN_AUDIBLE = 0.05

_MODEL = None


def window_features(mono, sr, meter, stem_events, t0, t1):
    """Per-stem features of one (t0, t1) span. IDENTICAL to the training
    extractor (tools/prominence_pilot.py imports this): energy share and
    K-weighted loudness share over the six true stems, onset density,
    median pitch and monophonic share from the stem's transcription."""
    a, b = int(t0 * sr), int(t1 * sr)
    feats = {}
    energy, loud = {}, {}
    for name in STEMS + ("drums",):
        seg = mono[name][a:b]
        if len(seg) < sr // 2:
            return None
        energy[name] = float(np.sqrt((seg ** 2).mean()))
        try:
            lufs = meter.integrated_loudness(seg.astype(np.float64))
            loud[name] = 10 ** (lufs / 10) if np.isfinite(lufs) else 0.0
        except Exception:
            loud[name] = 0.0
    et, lt = sum(energy.values()), sum(loud.values())
    for name in STEMS:
        ev = [(s, p) for s, e, p, amp in stem_events[name] if t0 <= s < t1 and amp >= 0.20]
        by = {}
        for s, p in ev:
            by.setdefault(round(s, 2), []).append(p)
        feats[name] = {
            "energy": energy[name] / et if et else 0.0,
            "loudness": loud[name] / lt if lt else 0.0,
            "density": len(by) / (t1 - t0),
            "pitch": float(np.median([p for _, p in ev])) if ev else 0.0,
            "mono": (sum(1 for v in by.values() if len(v) == 1) / len(by)) if by else 0.0,
        }
    feats["drums_energy"] = energy["drums"] / et if et else 0.0
    return feats


def feature_vector(feats, prev_feats, next_feats):
    """Model input for one window: its own features plus the energy and
    loudness shares of the neighbouring windows (zeros at the edges)."""
    x = []
    for s in STEMS:
        x += [feats[s][k] for k in FEATS]
    x.append(feats["drums_energy"])
    for ctx in (prev_feats, next_feats):
        if ctx:
            x += [ctx[s][k] for s in STEMS for k in ("energy", "loudness")]
        else:
            x += [0.0] * (2 * len(STEMS))
    return x


def load_model():
    global _MODEL
    if _MODEL is None and MODEL_PATH.is_file():
        import joblib

        _MODEL = joblib.load(MODEL_PATH)
    return _MODEL


def windows_for(tempo, duration_s: float):
    """4-bar windows on the chart's beat grid, as (t0, t1, beat0, beat1)."""
    step = BARS_PER_WINDOW * 4
    out = []
    beat = 0
    while True:
        t0 = tempo.beat_to_time(beat)
        if t0 >= duration_s:
            break
        t1 = min(duration_s, tempo.beat_to_time(beat + step))
        if t1 - t0 >= 1.0:
            out.append((t0, t1, beat, beat + step))
        beat += step
    return out


def _transcribe_stem(samples, sr, key):
    import tempfile

    import soundfile as sf

    from . import transcribe

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fh:
        path = fh.name
    try:
        sf.write(path, samples, sr)
        return transcribe.transcribe(path, cache_key=key)
    finally:
        Path(path).unlink(missing_ok=True)


def transcribe_stems(mono, sr, progress=lambda m: None, known=None):
    """Basic Pitch on each true melodic stem; `known` supplies already
    transcribed stems (the chord-texture pass makes the guitar's). Stems
    without a cached transcription run in parallel worker processes, since
    Basic Pitch's note creation is pure Python and holds the GIL."""
    from concurrent.futures import ProcessPoolExecutor

    from . import transcribe

    out = dict(known or {})
    stem_key = mono.get("_key")
    keys = {name: f"{stem_key}:{name}:bp" if stem_key else None
            for name in STEMS if name not in out}
    events = {name: transcribe.cached_transcription(key) for name, key in keys.items()}
    todo = [name for name in keys if events[name] is None]
    if todo:
        with ProcessPoolExecutor(len(todo)) as pool:
            jobs = {name: pool.submit(_transcribe_stem, mono[name], sr, keys[name]) for name in todo}
            events |= {name: job.result() for name, job in jobs.items()}
    return out | {name: events[name] for name in keys}


def smooth(labels):
    """A single window disagreeing with two agreeing neighbours takes
    their label: charters switch instruments at phrase boundaries, not
    for one bar."""
    out = list(labels)
    for i in range(1, len(out) - 1):
        if out[i - 1] is not None and out[i - 1] == out[i + 1] != out[i]:
            out[i] = out[i - 1]
    return out


def runs(windows):
    """Merge consecutive same-stem windows: [(t0, t1, beat0, beat1, stem,
    mean confidence)]. Windows without an opinion (stem None) form their
    own runs so consumers can fall back there."""
    out = []
    for w in windows:
        if out and out[-1]["stem"] == w["stem"]:
            last = out[-1]
            last["t1"], last["beat1"] = w["t1"], w["beat1"]
            last["conf"].append(w["conf"])
        else:
            out.append({"t0": w["t0"], "t1": w["t1"], "beat0": w["beat0"], "beat1": w["beat1"],
                        "stem": w["stem"], "conf": [w["conf"]]})
    for r in out:
        r["conf"] = float(np.mean(r["conf"]))
    return out


def letters(windows) -> str:
    return "".join(LETTER.get(w["stem"], "?") if w["stem"] else "?" for w in windows)


def timeline(audio_path, tempo, duration_s, progress=lambda m: None, known_events=None):
    """Per-window followed stem with confidence, or None when the model or
    the SW stems are unavailable. Each entry: {t0, t1, beat0, beat1, stem,
    conf, probs}; stem is None below MIN_CONFIDENCE."""
    model = load_model()
    if model is None:
        return None
    from . import stems as stemsmod

    mono = stemsmod.separate(str(audio_path), progress, backend="sw")
    if not mono or any(k not in mono for k in STEMS + ("drums",)):
        return None
    import pyloudnorm as pyln

    meter = pyln.Meter(stemsmod.SR)
    stem_events = transcribe_stems(mono, stemsmod.SR, progress, known_events)
    if known_events is not None:
        # in/out: the caller keeps every stem's transcription (the
        # chord-texture pass reuses the guitar's; keyed rescue all five)
        known_events.update(stem_events)
    spans = windows_for(tempo, duration_s)
    feats = [window_features(mono, stemsmod.SR, meter, stem_events, t0, t1) for t0, t1, _, _ in spans]
    rows, keep = [], []
    for i, f in enumerate(feats):
        if f is None:
            continue
        prev = feats[i - 1] if i > 0 else None
        nxt = feats[i + 1] if i + 1 < len(feats) else None
        rows.append(feature_vector(f, prev, nxt))
        keep.append(i)
    if not rows:
        return None
    clf = model["model"]
    proba = np.zeros((len(rows), len(STEMS)))
    p = clf.predict_proba(np.array(rows))
    proba[:, list(clf.classes_)] = p
    # Training labels required the followed stem to hold >= MIN_AUDIBLE of
    # the window's energy; enforce the same at inference, or a silent
    # stem with a few hallucinated onsets can win a window (Dubstep Is
    # Dead drop: "guitar 1.00" at 1% guitar energy).
    for j, i in enumerate(keep):
        f = feats[i]
        for n, stem in enumerate(STEMS):
            if f[stem]["energy"] < MIN_AUDIBLE:
                proba[j][n] = 0.0
        total = proba[j].sum()
        if total > 0:
            proba[j] /= total
    out = []
    labels = []
    for j, i in enumerate(keep):
        t0, t1, b0, b1 = spans[i]
        k = int(proba[j].argmax())
        conf = float(proba[j][k])
        stem = STEMS[k] if conf >= MIN_CONFIDENCE else None
        labels.append(stem)
        out.append({"t0": t0, "t1": t1, "beat0": b0, "beat1": b1, "stem": stem, "conf": conf,
                    "probs": {s: float(proba[j][n]) for n, s in enumerate(STEMS)}})
    for w, lab, j in zip(out, smooth(labels), range(len(out)), strict=True):
        w["stem"] = lab
        w["features"] = feats[keep[j]]
    return out


# Solos from the timeline. The shipped guitar-stem rule (82% precision,
# 6% recall, ~2 quiet fires per 100 solo-less songs) loses its solos to
# section width: chroma sections merge a solo with its neighbours (38%
# of misses) and then trip the singing gate on the sung bars they
# swallowed (38%). The 4-bar windows carry the same signals at solo
# granularity - guitar density, energy, register, monophony, vocal
# share, followed-stem probabilities - and a song-fold model on them,
# turned into hysteresis runs, measured 86% precision / 18% recall /
# 2.1% quiet fires on the 299-song dump (tools/solo_windows.py), three
# times the shipped recall at higher precision. Boundaries land within
# a window or so (median ~10 s), against the old section-wide markers.
SOLO_MODEL_PATH = Path(__file__).resolve().parent / "solo_model.joblib"
SOLO_ENTER = 0.70        # a run needs SOLO_MIN_WINDOWS windows at/above this
SOLO_EXTEND = 0.40       # then grows over neighbours at/above this
SOLO_MIN_WINDOWS = 4     # 16 bars: the shipped rule's 32-beat floor, doubled
SOLO_BRIDGE = 0          # windows below EXTEND a run may jump (0 = none)
_SOLO_MODEL = None


def solo_vector(feats, probs, i, end_s):
    """Window i's solo features: the followed-instrument vector, the
    stem probabilities, guitar statistics relative to the song, position,
    window length and the guitar density of the four neighbours. `feats`
    is the per-window feature list (None where absent), `probs` an
    (n, len(STEMS)) array. Shared by tools/solo_windows.py (training)
    and solo_runs (inference)."""
    f = feats[i]
    gd = [x["guitar"]["density"] for x in feats if x]
    ge = [x["guitar"]["energy"] for x in feats if x]
    gp = [x["guitar"]["pitch"] for x in feats if x and x["guitar"]["pitch"] > 0]
    md = float(np.median(gd)) if gd else 0.0
    me = float(np.median(ge)) if ge else 0.0
    mp = float(np.median(gp)) if gp else 0.0
    vec = feature_vector(f, feats[i - 1] if i > 0 else None,
                         feats[i + 1] if i + 1 < len(feats) else None)
    ctx = [feats[j]["guitar"]["density"] if 0 <= j < len(feats) and feats[j] else 0.0
           for j in (i - 2, i - 1, i + 1, i + 2)]
    t0, t1 = feats[i]["_t0"], feats[i]["_t1"]
    return (vec + list(probs[i])
            + [f["guitar"]["density"] / (md + 1e-6), f["guitar"]["energy"] / (me + 1e-6),
               (f["guitar"]["pitch"] - mp) if f["guitar"]["pitch"] > 0 else 0.0,
               t0 / max(1.0, end_s), t1 - t0] + ctx)


def load_solo_model():
    global _SOLO_MODEL
    if _SOLO_MODEL is None and SOLO_MODEL_PATH.is_file():
        import joblib

        _SOLO_MODEL = joblib.load(SOLO_MODEL_PATH)
    return _SOLO_MODEL


def solo_hysteresis(probs, enter=SOLO_ENTER, extend=SOLO_EXTEND,
                    min_windows=SOLO_MIN_WINDOWS, bridge=SOLO_BRIDGE):
    """[(i0, i1)] inclusive window index runs."""
    n = len(probs)
    core = [p >= enter for p in probs]
    soft = [p >= extend for p in probs]
    runs, i = [], 0
    while i < n:
        if not core[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and core[j + 1]:
            j += 1
        if j - i + 1 < min_windows:
            i = j + 1
            continue
        a, b = i, j
        while a - 1 >= 0 and soft[a - 1]:
            a -= 1
        while b + 1 < n and soft[b + 1]:
            b += 1
        k = b + 1
        while k < n and not soft[k] and k - b <= bridge:
            k += 1
        if k < n and soft[k] and k - b <= bridge + 1 and k > b + 1:
            b = k
            while b + 1 < n and soft[b + 1]:
                b += 1
        runs.append((a, b))
        i = b + 1
    return runs


def solo_runs(windows, model, tempo):
    """[(start_tick, end_tick)] lead breaks from a timeline (windows with
    features and probs, as timeline() returns them)."""
    if not windows or model is None:
        return []
    feats = []
    for w in windows:
        f = dict(w["features"]) if w.get("features") else None
        if f is not None:
            f["_t0"], f["_t1"] = w["t0"], w["t1"]
        feats.append(f)
    probs = np.array([[w["probs"][s] for s in STEMS] for w in windows])
    end_s = windows[-1]["t1"]
    idx = [i for i, f in enumerate(feats) if f is not None]
    if not idx:
        return []
    X = np.array([solo_vector(feats, probs, i, end_s) for i in idx])
    p = model["model"].predict_proba(X)[:, 1]
    per_window = np.zeros(len(windows))
    per_window[idx] = p
    res = tempo.resolution
    return [(int(windows[a]["beat0"] * res), int(windows[b]["beat1"] * res))
            for a, b in solo_hysteresis(list(per_window))]


# Rhythm rescue. In EDM drops the followed stem is synth or bass but its
# PITCHED transcription starves: the wub is a filter sweep on one note,
# and Basic Pitch on the mix recovers 9-12% of the human onsets there
# (Dubstep Is Dead drops, Go Beyond drop; 2026-09-05 go/no-go test),
# while onset detection on the followed stem recovers 45-82% at 93-100%
# precision. Charters chart the wub's RHYTHM, laned by its filter
# contour - the same thing brightness lanes do for stuck stretches, but
# here the notes have to be supplied first.
RESCUE_STEMS = ("sw_other", "bass", "guitar")
RESCUE_MIN_ONSETS_PER_S = 3.0     # a real rhythm, not a pad
RESCUE_STARVED_RATIO = 0.5        # chart holds < half the stem's onsets
RESCUE_ONSET_DELTA = 0.07         # librosa peak-pick threshold (tested)
# Guitar joined on 2026-09-09 after its own go/no-go (tools/
# guitar_onset_gonogo.py): in guitar windows still starved after keyed
# rescue, new stem onsets at delta 0.05 land on human notes at a median
# 0.96 precision (min 0.79) and lift coverage of the charter's notes
# there from ~25% to ~65% (Metabolic 15 -> 60%). Down-tuned palm-muted
# chugs are where pitched transcription gives up; the onsets do not.
# Guitar onsets are voiced with the run's chord shape (the chug), not the
# filter contour that lanes a wub.
RESCUE_ONSET_DELTA_GUITAR = 0.05
RESCUE_NOTE_WINDOW_S = 0.12
RESCUE_MIN_SPAN = 1.3             # centroid p90/p10 below this: static
RESCUE_MIN_SHARE = 0.10           # stem must carry this much of the window


def _centroids(signal, sr, times):
    out = []
    for t in times:
        a = int(t * sr)
        seg = signal[a:min(len(signal), a + int(RESCUE_NOTE_WINDOW_S * sr))]
        if len(seg) < 256:
            out.append(0.0)
            continue
        spec = np.abs(np.fft.rfft(seg * np.hanning(len(seg))))
        freqs = np.fft.rfftfreq(len(seg), 1.0 / sr)
        power = spec.sum()
        out.append(float((spec * freqs).sum() / power) if power > 0 else 0.0)
    return np.asarray(out)


def _lanes_from_contour(values, strengths):
    """Lanes 0-4 from the log-centroid contour between its p10 and p90,
    median-filtered; a spectrally static run lanes by hit strength
    instead (accents climb), so a monotone drop is not one lane."""
    v = np.asarray(values, dtype=float)
    voiced = v[v > 0]
    if len(voiced) >= 4:
        p10, p90 = np.percentile(voiced, 10), np.percentile(voiced, 90)
        if p10 > 0 and p90 / p10 >= RESCUE_MIN_SPAN:
            logc = np.log(np.maximum(v, p10 * 0.5))
            lanes = np.clip(((logc - np.log(p10)) / (np.log(p90) - np.log(p10)) * 5).astype(int), 0, 4)
            sm = lanes.copy()
            for k in range(1, len(lanes) - 1):
                sm[k] = sorted(lanes[k - 1:k + 2])[1]
            return [int(x) for x in sm]
    st = np.asarray(strengths, dtype=float)
    if len(st) and st.max() > st.min():
        ranks = (st - st.min()) / (st.max() - st.min())
        return [int(min(4, r * 5)) for r in ranks]
    return [1] * len(v)


def rhythm_rescue(expert, followed, mono, sr, tempo):
    """Add notes from the followed stem's onsets in starved synth/bass
    runs. Returns (notes, added, runs_touched)."""
    import librosa

    res = tempo.resolution
    by_tick = {t for t, _, _ in expert}
    open_ticks = sorted({t for t, l, _ in expert if l == 7})
    added = []
    relaned_opens = {}
    touched = 0
    # Onsets are picked on the WHOLE stem once: peak-picking normalises
    # within the signal it is given, so a run segment of a near-silent
    # stem would manufacture dozens of "onsets" out of noise.
    peaks = {}
    wanted = {r["stem"] for r in followed if r["stem"] in RESCUE_STEMS}
    for stem in wanted:
        env = librosa.onset.onset_strength(y=mono[stem], sr=sr)
        delta = RESCUE_ONSET_DELTA_GUITAR if stem == "guitar" else RESCUE_ONSET_DELTA
        frames = librosa.onset.onset_detect(onset_envelope=env, sr=sr, units="frames",
                                            delta=delta, backtrack=False)
        peaks[stem] = (librosa.frames_to_time(frames, sr=sr) - ONSET_LAG_S, env[frames])
    # chord shapes already on the chart, for voicing guitar onsets
    chord_at = {}
    for t, l, _ in expert:
        chord_at.setdefault(t, set()).add(l)
    chord_at = {t: tuple(sorted(x for x in ls if x <= 4)) for t, ls in chord_at.items()
                if len([x for x in ls if x <= 4]) == 2}
    hop = int(0.05 * sr)
    envs = {}
    for stem in STEMS + ("drums",):
        n = len(mono[stem]) // hop
        envs[stem] = np.sqrt((mono[stem][:n * hop].reshape(n, hop) ** 2).mean(axis=1))
    for r in followed:
        if r["stem"] not in RESCUE_STEMS:
            continue
        t0, t1 = r["t0"], r["t1"]
        sig = mono[r["stem"]]
        all_times, all_strengths = peaks[r["stem"]]
        sel = (all_times >= t0) & (all_times < t1)
        if not sel.any():
            continue
        times, strengths = all_times[sel], all_strengths[sel]
        ticks = [tempo.quantize(t, subdiv=4) for t in times]
        # Starvation is judged per 4-bar WINDOW inside the run: a song-long
        # synth run is well charted on average while its drop is empty
        # (Go Beyond: 405 chart positions vs 527 human, drop 51 vs 211).
        step = BARS_PER_WINDOW * 4 * res
        keep = []
        seen = set()
        fired = []
        for lo in range(int(r["beat0"] * res), int(r["beat1"] * res), step):
            hi = min(lo + step, int(r["beat1"] * res))
            idx = [i for i, tk in enumerate(ticks) if lo <= tk < hi]
            secs = tempo.beat_to_time(hi / res) - tempo.beat_to_time(lo / res)
            if secs <= 0 or len(idx) / secs < RESCUE_MIN_ONSETS_PER_S:
                continue
            # the rescued stem must be audible in this window
            fa, fb = int(tempo.beat_to_time(lo / res) / 0.05), int(tempo.beat_to_time(hi / res) / 0.05)
            fb = max(fa + 1, min(fb, min(len(e) for e in envs.values())))
            means = {st: float(e[fa:fb].mean()) for st, e in envs.items()}
            tot = sum(means.values())
            if tot <= 0 or means[r["stem"]] / tot < RESCUE_MIN_SHARE:
                continue
            present = sum(1 for t in by_tick if lo <= t < hi)
            if present >= RESCUE_STARVED_RATIO * len(idx):
                continue
            fired.append((lo, hi))
            for i in idx:
                tk = ticks[i]
                if tk in seen:
                    continue
                if any(abs(tk - e) <= res // 4 for e in (by_tick & {tk - res // 4, tk, tk + res // 4})):
                    continue
                seen.add(tk)
                keep.append(i)
        if not keep:
            continue
        # The mix transcription's own notes in a fired window are the
        # wub heard as a sub-register pitch, charted as opens; they were
        # fretted before only because brightness relaning fired on the
        # sparse, stuck stretch. Lane them from the same contour as the
        # rescued notes (humans chart 0-5% opens on these songs).
        opens_here = [t for t in open_ticks if any(lo <= t < hi for lo, hi in fired)
                      and t not in relaned_opens]
        open_times = np.array([tempo.beat_to_time(t / res) for t in opens_here])
        all_times = np.concatenate([times[keep], open_times]) if len(open_times) else times[keep]
        fill = float(np.median(strengths[keep])) if len(keep) else 0.0
        all_str = np.concatenate([strengths[keep], np.full(len(open_times), fill)]) \
            if len(open_times) else strengths[keep]
        shape = None
        if r["stem"] == "guitar":
            # the run's chug shape: the commonest two-lane chord inside the
            # run (chord texture has voiced it by now); no chord -> contour
            lo_r, hi_r = int(r["beat0"] * res), int(r["beat1"] * res)
            counts = {}
            for t, sh in chord_at.items():
                if lo_r <= t < hi_r:
                    counts[sh] = counts.get(sh, 0) + 1
            if counts:
                shape = max(counts, key=counts.get)
        if shape is not None:
            voicing = [shape] * (len(keep) + len(opens_here))
        else:
            lanes = _lanes_from_contour(_centroids(sig, sr, all_times), all_str)
            voicing = [(lane,) for lane in lanes]
        for i, v in zip(keep, voicing[:len(keep)], strict=True):
            for lane in v:
                added.append((ticks[i], lane, 0))
            by_tick.add(ticks[i])
        for t, v in zip(opens_here, voicing[len(keep):], strict=True):
            relaned_opens[t] = v
        touched += 1
    if not added and not relaned_opens:
        return expert, 0, 0
    out = []
    for t, l, sus in expert:
        if l == 7 and t in relaned_opens:
            out += [(t, lane, sus) for lane in relaned_opens[t]]
        else:
            out.append((t, l, sus))
    return sorted(set(out + added)), len(added), touched


# Keyed rescue: the pitched half of section commitment, applied only where
# the chart is empty. The shipped stem rescue fills a starved stretch from
# the shim "other" and bass stems - a blend with no idea which instrument
# the charter follows, which is where bass notes in a quiet guitar
# passage (and the open-note deserts) come from. With the timeline every
# stem is already transcribed, so a starved 4-bar window is filled from
# the FOLLOWED stem alone, through the same admission rules (amplitude
# and pitch floors, octave lift, empty-tick-only dedupe). Rhythm rescue
# stays the second pass for synth/bass drops where pitch fails outright.
KEYED_STEMS = ("guitar", "piano", "sw_other")
KEYED_STARVED_RATIO = 0.5      # chart holds < half the stem's positions
KEYED_MIN_STEM_PER_S = 2.0     # the stem must be playing a line, not a pad
KEYED_MIN_SHARE = 0.10         # and be audible in the window
# Admitted notes need a stronger posterior than the mix path's 0.20: a
# separated stem's weak notes are bleed and artifacts far more often than
# a mix's. Sweep on 16 songs (work/keyed_rescue_sweep.txt), precision
# against human note times within 80 ms: 0.69 at 0.20, 0.71 at 0.40 (best
# on 8 of 13 songs, 71 notes/song), 0.74 at 0.50 but with fewer notes
# (58) than the shipped blended rescue (0.70, 69 notes). 0.40 keeps the
# coverage; the starvation test itself keeps the standard floors so a
# stricter admission cannot widen what counts as starved. That metric is
# timing-only - it cannot see the instrument prior, which is the point.
KEYED_ADMIT_AMPLITUDE = 0.40
# A busy synth stem transcribes more positions than a charter writes: in
# Sexualizer's rescued windows keyed rescue reached 1.6 notes per human
# note (A/B report 2026-09-06). A window may not end up denser than
# 8th-note density; the strongest stem notes are kept up to the budget.
KEYED_MAX_PER_BEAT = 2.0


# Bass is deliberately NOT committed: on Hands Up the bass stem's events
# un-starved the drop windows, rhythm rescue then supplied 105 notes
# instead of 430 and the chart lost 150 positions (coverage 0.68 -> 0.54);
# elsewhere bass added <= 13 notes per song. Rhythm rescue is the better
# supplier for bass-followed windows (onset envelope, not pitch events).
COMMIT_STEMS = KEYED_STEMS


def commit_section_events(events, windows, stem_events, tempo,
                          min_pitch: int = 40, min_amplitude: float = 0.20,
                          admit_amplitude: float = None):
    """Section commitment, supply side (on by default since 2026-09-10,
    --no-commit-sections to A/B): in every followed window where the stem is audible
    and playing a line, admit the followed stem's events on the empty 16th
    ticks - not only where the window is starved (that half is keyed
    rescue, which runs first and stays as is). The mix's own events are
    never culled: measured offline on 16 songs (tools/section_commit_eval.py,
    2026-09-10) the cull trades 12 points of human coverage for 6 of
    precision, while supply alone lifts coverage inside committed windows
    0.53 -> 0.61 at unchanged precision, density 0.88 -> 0.93 of human.
    Same density budget as keyed rescue. Returns (events_to_add,
    windows_touched, stem_counts)."""
    from . import transcribe

    admit_amplitude = KEYED_ADMIT_AMPLITUDE if admit_amplitude is None else admit_amplitude
    known = list(events)
    added, touched, counts = [], 0, {}
    for w in windows:
        stem = w.get("stem")
        if stem not in COMMIT_STEMS or stem not in stem_events:
            continue
        f = w.get("features")
        if not f or f[stem]["energy"] < KEYED_MIN_SHARE:
            continue
        t0, t1 = w["t0"], w["t1"]
        lo, hi = tempo.quantize(t0, subdiv=4), tempo.quantize(t1, subdiv=4)
        stem_pos = {tempo.quantize(s0, subdiv=4) for s0, _, pch, amp in stem_events[stem]
                    if t0 <= s0 < t1 and pch >= min_pitch and amp >= min_amplitude}
        if len(stem_pos) / max(1e-6, t1 - t0) < KEYED_MIN_STEM_PER_S:
            continue
        extra = transcribe.admit_stem_events(stem_events[stem], [(t0, t1)], known, tempo,
                                             min_pitch, admit_amplitude)
        if not extra:
            continue
        present = len({tempo.quantize(s0, subdiv=4) for s0, _, pch, amp in known
                       if lo <= tempo.quantize(s0, subdiv=4) < hi
                       and pch >= min_pitch and amp >= min_amplitude})
        allowed = max(0, int(KEYED_MAX_PER_BEAT * (hi - lo) / tempo.resolution) - present)
        ticks_of = {}
        for e in extra:
            ticks_of.setdefault(tempo.quantize(e[0], subdiv=4), []).append(e)
        if len(ticks_of) > allowed:
            keep_ticks = sorted(ticks_of, key=lambda tk: -max(e[3] for e in ticks_of[tk]))[:allowed]
            extra = [e for tk in keep_ticks for e in ticks_of[tk]]
            if not extra:
                continue
        added += extra
        known += extra
        touched += 1
        counts[stem] = counts.get(stem, 0) + len(extra)
    return added, touched, counts


def keyed_rescue_events(events, windows, stem_events, tempo,
                        min_pitch: int = 40, min_amplitude: float = 0.20,
                        admit_amplitude: float = None):
    """Extra transcription events from each starved window's followed
    stem. Returns (events_to_add, windows_touched, stem_counts)."""
    from . import transcribe

    admit_amplitude = KEYED_ADMIT_AMPLITUDE if admit_amplitude is None else admit_amplitude

    kept: dict = {}
    for s0, _, pch, amp in events:
        if pch >= min_pitch and amp >= min_amplitude:
            kept.setdefault(tempo.quantize(s0, subdiv=4), True)
    known = list(events)
    added = []
    touched = 0
    counts = {}
    for w in windows:
        stem = w.get("stem")
        if stem not in KEYED_STEMS or stem not in stem_events:
            continue
        f = w.get("features")
        if not f or f[stem]["energy"] < KEYED_MIN_SHARE:
            continue
        t0, t1 = w["t0"], w["t1"]
        lo, hi = tempo.quantize(t0, subdiv=4), tempo.quantize(t1, subdiv=4)
        stem_pos = {tempo.quantize(s0, subdiv=4) for s0, _, pch, amp in stem_events[stem]
                    if t0 <= s0 < t1 and pch >= min_pitch and amp >= min_amplitude}
        if len(stem_pos) / max(1e-6, t1 - t0) < KEYED_MIN_STEM_PER_S:
            continue
        present = sum(1 for tk in kept if lo <= tk < hi)
        if present >= KEYED_STARVED_RATIO * len(stem_pos):
            continue
        extra = transcribe.admit_stem_events(stem_events[stem], [(t0, t1)], known, tempo,
                                             min_pitch, admit_amplitude)
        if not extra:
            continue
        # density budget: positions (chords count once), strongest first
        beats = (hi - lo) / tempo.resolution
        allowed = max(0, int(KEYED_MAX_PER_BEAT * beats) - present)
        ticks_of = {}
        for e in extra:
            ticks_of.setdefault(tempo.quantize(e[0], subdiv=4), []).append(e)
        if len(ticks_of) > allowed:
            keep_ticks = sorted(ticks_of, key=lambda tk: -max(e[3] for e in ticks_of[tk]))[:allowed]
            extra = [e for tk in keep_ticks for e in ticks_of[tk]]
            if not extra:
                continue
        added += extra
        known += extra
        touched += 1
        counts[stem] = counts.get(stem, 0) + len(extra)
    return added, touched, counts
