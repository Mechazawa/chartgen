"""Route B pilot: label every 4-bar window of a human chart with the SW stem
the charter FOLLOWED, from per-stem transcription onsets.

The section-commitment research (2026-09-02) concluded that stem energy
never predicted which line a charter charts, because the 95-song study
only ever measured density. This is the missing outcome measure: for
each 4-bar window, which separated stem's note onsets best explain the
human chart's onsets. The label is instrument-FOLLOWING (a musical fact
charters largely agree on), not tapping (charter style).

Recipe, with the three fixes the research required:
  * per-song global timing offset: +-200 ms at 10 ms steps, chosen on the
    full-mix transcription against the human onsets (STRUM's method);
  * coincidence-split F1 scoring: a human onset matched by k stems
    credits 1/k to each stem's recall, and the stem's own unmatched
    onsets cost it precision, so density alone cannot win a window;
  * label only with margin (best F1 >= LABEL_MIN and a gap
    of LABEL_MARGIN over the runner-up), else 'ambiguous'.

Also records per-stem features per window (RMS energy share, BS.1770
K-weighted loudness share via pyloudnorm, onset density, median pitch,
monophonic share) so the Route A question - does loudness beat energy
at predicting the label - is answered on the same windows.

    python tools/prominence_pilot.py -o work/prominence_pilot.json FOLDER [FOLDER ...]
"""
import argparse
import json
import re
import sys
import tempfile
import time
from itertools import pairwise
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from validate_solos import find_audio  # noqa: E402
from chartgen.prominence import STEMS, window_features as features  # noqa: E402

LETTER = {"vocals": "V", "bass": "B", "guitar": "G", "piano": "P", "sw_other": "S",
          "ambiguous": "?", "none": "."}
TOL = 0.10          # STRUM's greedy 1:1 window
OFFSET_RANGE = 0.20
OFFSET_STEP = 0.01
BARS_PER_WINDOW = 4
MIN_HUMAN_ONSETS = 8
LABEL_MIN = 0.35
LABEL_MARGIN = 0.15
NOTE_RE = re.compile(r"(\d+)\s*=\s*N\s*(\d+)\s+\d+")
BPM_RE = re.compile(r"(\d+)\s*=\s*B\s*(\d+)")


def parse_chart(chart: Path):
    """(human onset times, window bounds in seconds) via the chart's own tempo map."""
    txt = chart.read_text(encoding="utf-8-sig", errors="replace")
    m = re.search(r"Resolution\s*=\s*(\d+)", txt)
    sync = re.search(r"\[SyncTrack\]\s*\{(.*?)\}", txt, re.S)
    track = re.search(r"\[ExpertSingle\]\s*\{(.*?)\}", txt, re.S)
    if not (m and sync and track):
        return None
    res = int(m.group(1))
    bpms = [(int(a), int(b) / 1000.0) for a, b in BPM_RE.findall(sync.group(1))]
    if not bpms:
        return None

    def t2s(tick):
        sec, lt, lb = 0.0, 0, bpms[0][1]
        for t, b in bpms:
            if t >= tick:
                break
            sec += (t - lt) / res * 60.0 / lb
            lt, lb = t, b
        return sec + (tick - lt) / res * 60.0 / lb

    ticks = sorted({int(t) for t, lane in NOTE_RE.findall(track.group(1))
                    if int(lane) <= 4 or int(lane) == 7})
    if len(ticks) < 50:
        return None
    onsets = np.array([t2s(t) for t in ticks])
    last = ticks[-1]
    step = BARS_PER_WINDOW * 4 * res
    bounds = [t2s(t) for t in range(0, last + step, step)] + [t2s(last + step)]
    windows = [(a, b) for a, b in pairwise(bounds) if b > a]
    return onsets, windows


def onsets_of(events, min_amp=0.20):
    times = sorted(s for s, e, p, a in events if a >= min_amp)
    out = []
    for t in times:
        if not out or t - out[-1] > 0.03:
            out.append(t)
    return np.array(out)


def greedy_match(a, b, tol=TOL):
    """Greedy 1:1 matching of sorted arrays a, b within tol -> list of (i, j)."""
    pairs = []
    used = np.zeros(len(b), dtype=bool)
    for i, t in enumerate(a):
        lo = np.searchsorted(b, t - tol)
        hi = np.searchsorted(b, t + tol, side="right")
        best, bestd = -1, tol + 1
        for k in range(lo, hi):
            if not used[k] and abs(b[k] - t) < bestd:
                best, bestd = k, abs(b[k] - t)
        if best >= 0:
            used[best] = True
            pairs.append((i, best))
    return pairs


def best_offset(mix_onsets, human):
    best, bestn = 0.0, -1
    for d in np.arange(-OFFSET_RANGE, OFFSET_RANGE + 1e-9, OFFSET_STEP):
        n = len(greedy_match(mix_onsets + d, human))
        if n > bestn:
            best, bestn = float(d), n
    return best, bestn


def window_scores(human, stem_onsets, t0, t1):
    """Per-stem chance-corrected, coincidence-split recall of human onsets."""
    H = human[(human >= t0) & (human < t1)]
    if len(H) < MIN_HUMAN_ONSETS:
        return None
    W = t1 - t0
    matched = {}
    hits = np.zeros(len(H), dtype=int)
    per_stem_hit = {}
    for name, ons in stem_onsets.items():
        S = ons[(ons >= t0 - TOL) & (ons < t1 + TOL)]
        pairs = greedy_match(S, H)
        idx = [j for _, j in pairs]
        per_stem_hit[name] = idx
        for j in idx:
            hits[j] += 1
        matched[name] = (len(S), len(pairs))
    out = {}
    for name, idx in per_stem_hit.items():
        n_s, n_m = matched[name]
        credit = sum(1.0 / hits[j] for j in idx)
        recall = credit / len(H)
        precision = (n_m / n_s) if n_s else 0.0
        # Score = F1 of stem onsets vs human onsets. A dense stem that
        # fires everywhere matches many human onsets by sheer density but
        # leaves most of its own onsets unmatched, so precision pays for
        # it; a Poisson null (recall minus n*2*TOL/W) was tried first and
        # drove a synth-only song's only real stem negative.
        f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
        null = min(1.0, n_s * 2 * TOL / W)
        out[name] = {"recall": recall, "precision": precision, "f1": f1,
                     "corrected": recall - null, "n": int(n_s)}
    return out


def label_of(scores):
    ranked = sorted(scores.items(), key=lambda kv: -kv[1]["f1"])
    best, second = ranked[0], ranked[1]
    if best[1]["f1"] >= LABEL_MIN and best[1]["f1"] - second[1]["f1"] >= LABEL_MARGIN:
        return best[0]
    return "ambiguous"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("folders", nargs="*", type=Path)
    ap.add_argument("-o", "--out", type=Path, required=True)
    ap.add_argument("--list", type=Path, help="UTF-8 file of folder paths, one per line")
    args = ap.parse_args(argv)
    if args.list:
        args.folders += [Path(line.strip()) for line in
                         args.list.read_text(encoding="utf-8").splitlines() if line.strip()]

    import pyloudnorm as pyln
    import soundfile as sf

    from chartgen import stems as stemsmod, transcribe

    meter = pyln.Meter(stemsmod.SR)
    done = {}
    if args.out.exists():
        for row in json.loads(args.out.read_text(encoding="utf-8")):
            done[row["song"]] = row
    rows = list(done.values())
    started = time.time()
    for folder in args.folders:
        if folder.name in done:
            continue
        if not folder.is_dir():
            print(f"skip {folder.name}: folder not found", flush=True)
            continue
        chart, audio = folder / "notes.chart", find_audio(folder)
        if not chart.is_file() or audio is None:
            print(f"skip {folder.name}: no chart/audio", flush=True)
            continue
        parsed = parse_chart(chart)
        if parsed is None:
            print(f"skip {folder.name}: unparseable chart", flush=True)
            continue
        human, windows = parsed
        stemsmod._CACHE.clear()
        mono = stemsmod.separate(str(audio), backend="sw")
        if mono is None or any(k not in mono for k in STEMS):
            print(f"skip {folder.name}: stems missing", flush=True)
            continue
        mix_events = transcribe.transcribe(str(audio))
        stem_events = {}
        for name in STEMS:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fh:
                path = fh.name
            try:
                sf.write(path, mono[name], stemsmod.SR)
                stem_events[name] = transcribe.transcribe(path)
            finally:
                Path(path).unlink(missing_ok=True)
        offset, nmatch = best_offset(onsets_of(mix_events), human)
        stem_onsets = {name: onsets_of(ev) + offset for name, ev in stem_events.items()}
        wins = []
        for t0, t1 in windows:
            scores = window_scores(human, stem_onsets, t0, t1)
            if scores is None:
                wins.append({"t0": t0, "t1": t1, "label": "none"})
                continue
            wins.append({"t0": t0, "t1": t1, "label": label_of(scores), "scores": scores,
                         "features": features(mono, stemsmod.SR, meter, stem_events, t0, t1)})
        row = {"song": folder.name, "offset": offset, "mix_matches": nmatch,
               "human_onsets": int(len(human)), "windows": wins}
        rows.append(row)
        done[folder.name] = row
        args.out.write_text(json.dumps(rows), encoding="utf-8")
        timeline = "".join(LETTER[w["label"]] for w in wins)
        el = time.time() - started
        print(f"[{len(rows)}] {folder.name[:44]:46s} offset {offset:+.2f}s  {timeline}  ({el / 60:.0f} min)",
              flush=True)
    print(f"done: {len(rows)} songs -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
