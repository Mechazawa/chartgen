"""Dump taps-v3 foreground features under BOTH separators, with ground truth.

Taps v3 classifies each chroma section by the melodic stem's share of stem
energy (FOREGROUND_SOFT_SHARE / FOREGROUND_HARD_SHARE) and the other-stem
centroid. Those are ABSOLUTE thresholds, so a different separator moves
what they measure. This dumps, per song and per section, exactly what
chartgen.taps.foreground_by_span would see under each backend, joined with
what the human charter tapped there - the input for a controlled
recalibration (tools/sweep_foreground.py).

Ground truth needs no cache: tap notes are the `N 6` lines of every chart.
Songs are visited in a seeded random order and written incrementally, so an
interrupted run leaves a valid random sample. Re-running resumes.

    python tools/dump_foreground.py -o work/foreground_ab.json --limit 800
"""
import argparse
import json
import random
import re
import sys
import time
from itertools import pairwise
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from validate_solos import find_audio  # noqa: E402

NOTE_RE = re.compile(r"(\d+)\s*=\s*N\s*(\d+)\s+\d+")
BPM_RE = re.compile(r"(\d+)\s*=\s*B\s*(\d+)")


def human_truth(chart: Path):
    """(note_times, tap_times) in seconds via the chart's own tempo map."""
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

    notes, taps = set(), set()
    for tick, lane in NOTE_RE.findall(track.group(1)):
        tick, lane = int(tick), int(lane)
        if lane <= 4 or lane == 7:
            notes.add(tick)
        elif lane == 6:
            taps.add(tick)
    if len(notes) < 50:
        return None
    return sorted(t2s(t) for t in notes), sorted(t2s(t) for t in taps)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-o", "--out", type=Path, required=True)
    ap.add_argument("--library", type=Path,
                    default=Path(__file__).resolve().parent.parent / "data/calibration")
    ap.add_argument("--limit", type=int, default=800)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--backends", default="demucs,sw")
    args = ap.parse_args(argv)

    import librosa
    import numpy as np

    from chartgen import expression, stems as stemsmod, tempo as tempomod
    from chartgen.taps import foreground_by_span

    backends = args.backends.split(",")
    folders = sorted(p for p in args.library.iterdir() if p.is_dir())
    random.Random(args.seed).shuffle(folders)
    done = {}
    if args.out.exists():
        for row in json.loads(args.out.read_text(encoding="utf-8")):
            done[row["song"]] = row
    rows = list(done.values())
    started = time.time()
    processed = 0
    for folder in folders:
        if processed >= args.limit:
            break
        if folder.name in done:
            processed += 1
            continue
        chart, audio = folder / "notes.chart", find_audio(folder)
        if not chart.is_file() or audio is None:
            continue
        truth = human_truth(chart)
        if truth is None:
            continue
        note_times, tap_times = truth
        try:
            y, sr = tempomod.load(str(audio))
            tm = tempomod.detect(y, sr)
            marks = expression.sections(y, sr, tm)
        except Exception as error:
            print(f"  skip {folder.name[:44]}: {type(error).__name__}", flush=True)
            continue
        if not marks:
            continue
        res = tm.resolution
        last_s = float(note_times[-1]) + 1.0
        bounds = [tm.beat_to_time(t / res) for t, _ in marks] + [last_s]
        spans_s = [(a, b) for a, b in pairwise(bounds) if b > a]
        nt = np.array(note_times)
        tt = np.array(tap_times) if tap_times else np.zeros(0)
        sections = []
        for a, b in spans_s:
            n = int(((nt >= a) & (nt < b)).sum())
            k = int(((tt >= a) & (tt < b)).sum()) if len(tt) else 0
            sections.append({"t0": a, "t1": b, "notes": n, "taps": k})

        row = {"song": folder.name, "sections": sections, "fg": {}}
        for backend in backends:
            stemsmod._CACHE.clear()
            mono = stemsmod.separate(str(audio), backend=backend)
            if mono is None:
                continue
            four = {k: mono[k] for k in stemsmod.DEMUCS_STEMS if k in mono}
            other_ds = librosa.resample(four["other"], orig_sr=stemsmod.SR,
                                        target_sr=22050)
            cen = librosa.feature.spectral_centroid(y=other_ds, sr=22050)[0]
            cen_t = librosa.times_like(cen, sr=22050)
            fg = foreground_by_span(spans_s, four, stemsmod.SR, cen, cen_t)
            row["fg"][backend] = [(float(s), float(b)) for s, b in fg]
        rows.append(row)
        done[folder.name] = row
        processed += 1
        args.out.write_text(json.dumps(rows), encoding="utf-8")
        el = time.time() - started
        print(f"  [{processed}/{args.limit}] {folder.name[:40]:42s} "
              f"{len(sections)} sections, {len(tap_times)} taps  "
              f"({el / 60:.0f} min)", flush=True)
    print(f"done: {len(rows)} songs -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
