"""Sliding-window solo sweep over the frame-level cache.

Chroma-section candidates capped recall at ~10%: sections merge solos
with their neighbours, diluting every signal (MJ's harmonica solo:
lead z -0.15 inside a merged 46s span, +3.26 in a too-short sliver at
the true boundary). Windows of 32-96 beats at 8-beat steps frame the
evidence at solo scale regardless of where the segmenter drew lines.

Truth is the label-noise-corrected set (markers + named-solo sections)
and quiet controls report the false-fire rate separately.

    python tools/sweep_windows.py
"""
import itertools
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "work/framecache"
LENGTHS = (32, 48, 64, 96)  # beats
STEP = 8


def iou(a, b):
    lo, hi = max(a[0], b[0]), min(a[1], b[1])
    if hi <= lo:
        return 0.0
    return (hi - lo) / ((a[1] - a[0]) + (b[1] - b[0]) - (hi - lo))


def window_features(npz):
    """Per-window raw features for one song -> list of dicts."""
    res, last_tick, quiet, n_truth = npz["meta"]
    grid_beats, grid_times = npz["grid_beats"], npz["grid_times"]
    f0, f0_t = npz["f0"], npz["f0_times"]
    cen, cen_t = npz["centroid"], npz["cen_times"]
    chroma, ch_t = npz["chroma"], npz["chroma_times"]
    other = npz["other_rms"]
    other_t = (np.arange(len(other)) + 0.5) * 0.05
    ticks = npz["note_ticks"]

    voiced = np.isfinite(f0)
    semis = np.where(voiced, 12 * np.log2(np.maximum(f0, 1e-6) / 440.0), np.nan)

    rows = []
    for length in LENGTHS:
        span_steps = length // STEP
        for i in range(len(grid_beats) - span_steps):
            t0, t1 = grid_times[i], grid_times[i + span_steps]
            start_tick = int(grid_beats[i] * res)
            end_tick = int(grid_beats[i + span_steps] * res)
            w = (f0_t >= t0) & (f0_t < t1)
            if w.sum() < 8:
                continue
            cw = (cen_t >= t0) & (cen_t < t1)
            hw = (ch_t >= t0) & (ch_t < t1)
            ow = (other_t >= t0) & (other_t < t1)
            sung = semis[w]
            sung = sung[np.isfinite(sung)]
            rows.append({
                "start": start_tick, "end": end_tick, "length": length,
                "t0": float(t0), "t1": float(t1),
                "voiced": float(voiced[w].mean()),
                "spread": float(np.std(sung)) if len(sung) > 4 else 0.0,
                "bright": float(cen[cw].mean()) if cw.any() else 0.0,
                "chroma": chroma[:, hw].mean(axis=1) if hw.any() else np.zeros(12),
                "lead_raw": float(other[ow].mean()) if ow.any() else 0.0,
                "notes": int(((ticks >= start_tick) & (ticks < end_tick)).sum()),
            })
    if not rows:
        return None, None, None, None

    # novelty: best chroma match to any NON-OVERLAPPING window of the same
    # length; z-scores per song across all windows.
    for length in LENGTHS:
        group = [r for r in rows if r["length"] == length]
        mats = np.array([r["chroma"] / (np.linalg.norm(r["chroma"]) or 1) for r in group])
        sims = mats @ mats.T
        for gi, r in enumerate(group):
            mask = np.array([abs(o["start"] - r["start"]) >= length * res // 2
                             for o in group])
            r["novelty"] = float(1 - sims[gi][mask].max()) if mask.any() else 0.0

    for key in ("voiced", "spread", "bright", "novelty", "lead_raw"):
        vals = np.array([r[key] for r in rows])
        mean, std = float(vals.mean()), float(vals.std()) or 1e-9
        for r in rows:
            r["z_" + key] = (r[key] - mean) / std
    truth = [tuple(t) for t in npz["truth"] if t[1] > t[0]]
    return rows, truth, int(last_tick), bool(quiet)


def load_all(limit=None):
    songs = []
    files = sorted(CACHE.glob("*.npz"))[:limit]
    for path in files:
        npz = np.load(path, allow_pickle=False)
        rows, truth, last, quiet = window_features(npz)
        if rows:
            songs.append({"name": path.stem, "rows": rows, "truth": truth,
                          "last": last, "quiet": quiet,
                          "res": int(npz["meta"][0])})
    return songs


def evaluate(songs, w_spread, w_novelty, w_voiced, w_lead, pos, threshold,
             top_k=2, min_iou=0.25):
    tp = fp = fn = 0
    quiet_fired = quiet_total = 0
    for song in songs:
        picks = []
        for r in song["rows"]:
            if r["notes"] < 16:
                continue
            p = r["start"] / song["last"]
            if not (pos[0] <= p <= pos[1]):
                continue
            score = (w_spread * r["z_spread"] + w_novelty * r["z_novelty"]
                     + w_voiced * r["z_voiced"] + w_lead * r["z_lead_raw"])
            if score >= threshold:
                picks.append((score, r["start"], r["end"]))
        picks.sort(reverse=True)
        chosen = []
        for _, a, b in picks:
            if all(iou((a, b), c) < 0.2 and (b < c[0] or a > c[1] or iou((a, b), c) < 0.2)
                   for c in chosen):
                chosen.append((a, b))
            if len(chosen) >= top_k:
                break
        if song["quiet"]:
            quiet_total += 1
            quiet_fired += 1 if chosen else 0
            continue
        truth = song["truth"]
        hits = [g for g in chosen if any(iou(g, t) >= min_iou for t in truth)]
        tp += len(hits)
        fp += len(chosen) - len(hits)
        fn += sum(1 for t in truth if not any(iou(g, t) >= min_iou for g in chosen))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f_half = (1.25 * precision * recall / (0.25 * precision + recall)) \
        if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f_half": f_half,
            "tp": tp, "fp": fp,
            "quiet_fired": quiet_fired, "quiet_total": quiet_total}


def main(argv=None):
    print("loading frame cache (window aggregation)...", flush=True)
    songs = load_all()
    n_truth = sum(1 for s in songs if not s["quiet"])
    print(f"{len(songs)} songs ({n_truth} with truth, "
          f"{len(songs) - n_truth} quiet), "
          f"{sum(len(s['truth']) for s in songs)} solos\n", flush=True)

    rows = []
    for ws, wn, wv, wl in itertools.product((0.5, 1.0), (0.5, 1.0), (0, 0.5), (1.0, 2.0)):
        for pos in ((0.10, 0.95), (0.25, 0.92), (0.45, 0.80)):
            for z in (1.5, 2.0, 2.5, 3.0):
                r = evaluate(songs, ws, wn, wv, wl, pos, z)
                r["rule"] = (f"sp{ws} nov{wn} v{wv} lead{wl} "
                             f"pos{pos[0]}-{pos[1]} z>={z}")
                rows.append(r)

    rows.sort(key=lambda r: -r["f_half"])
    print("top by F0.5:")
    for r in rows[:8]:
        print(f"  {r['rule']:<48}{r['precision']:>5.0%}{r['recall']:>6.0%}  "
              f"(tp{r['tp']} fp{r['fp']})  quiet fired {r['quiet_fired']}/{r['quiet_total']}")
    print("\nprecision-first (recall >= 15%):")
    good = [r for r in rows if r["recall"] >= 0.15]
    for r in sorted(good, key=lambda r: (-r["precision"], -r["recall"]))[:8]:
        print(f"  {r['rule']:<48}{r['precision']:>5.0%}{r['recall']:>6.0%}  "
              f"(tp{r['tp']} fp{r['fp']})  quiet fired {r['quiet_fired']}/{r['quiet_total']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
