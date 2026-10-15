"""Calibrate the soft-phrase tap rule against human tap positions.

The rule is a chosen philosophy (taps mark soft sounds), not a mimicked
consensus - the community itself only half-agrees. So the target here is
NOT to reproduce every human tap; it is to pick the weights and thresholds
that agree with human taps as often as this philosophy can, and - just as
important - to report the honest number so nobody mistakes an aesthetic
for a measurement.

    python tools/sweep_taps.py work/taps_feat_F1.json work/taps_feat_F2.json
"""
import argparse
import itertools
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chartgen.taps import phrases

FEATURES = ("attack", "bright", "loud")


def softness(song, weights):
    notes = song["notes"]
    stats = {}
    for key in FEATURES:
        vals = [n[key] for n in notes]
        stats[key] = (statistics.mean(vals), statistics.pstdev(vals) or 1e-9)
    total = sum(weights.values()) or 1.0
    return {
        n["tick"]: -sum(weights[k] * (n[k] - stats[k][0]) / stats[k][1]
                        for k in FEATURES) / total
        for n in notes
    }


def evaluate(songs, weights, soft_z, min_run, max_gap, max_share):
    tp = fp = fn = 0
    per_song = []
    for song in songs:
        soft = softness(song, weights)
        notes = [(n["tick"], 0, 0) for n in song["notes"]]
        got = phrases(notes, soft, song["resolution"], soft_z=soft_z,
                      min_run=min_run, max_gap_beats=max_gap, max_share=max_share)
        truth = {n["tick"] for n in song["notes"] if n["tap"]}
        hit = len(got & truth)
        tp += hit
        fp += len(got) - hit
        fn += len(truth) - hit
        if got:
            per_song.append(hit / len(got))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1,
            "songs_marked": len(per_song),
            "median_song_precision": statistics.median(per_song) if per_song else 0.0}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("caches", type=Path, nargs="+")
    ap.add_argument("--top", type=int, default=10)
    args = ap.parse_args(argv)

    songs = [s for p in args.caches
             for s in json.loads(p.read_text(encoding="utf-8"))]
    songs = list({s["name"]: s for s in songs}.values())
    total_taps = sum(sum(1 for n in s["notes"] if n["tap"]) for s in songs)
    total = sum(len(s["notes"]) for s in songs)
    print(f"{len(songs)} songs, {total_taps}/{total} positions tapped "
          f"({100 * total_taps / total:.0f}% base rate)\n")

    results = []
    for wa, wb, wl in ((1, 1, 0.5), (1, 0, 0), (0, 1, 0), (1, 1, 0),
                       (0.5, 1, 0.5), (1, 0.5, 0.5), (0, 1, 0.5), (0.5, 1, 0)):
        weights = dict(zip(FEATURES, (wa, wb, wl), strict=True))
        for soft_z, min_run, max_gap, share in itertools.product(
            (0.25, 0.4, 0.5, 0.65, 0.8), (4, 6, 8), (1.0, 2.0), (0.3, 0.4, 0.5),
        ):
            row = evaluate(songs, weights, soft_z, min_run, max_gap, share)
            row["rule"] = (f"a{wa} b{wb} l{wl} z>={soft_z} run>={min_run} "
                           f"gap<={max_gap} share<={share}")
            results.append(row)

    results.sort(key=lambda r: -r["f1"])
    print(f"{'rule':<50}{'prec':>6}{'rec':>6}{'F1':>6}   med-song-prec")
    for row in results[:args.top]:
        print(f"{row['rule']:<50}{row['precision']:>5.0%}{row['recall']:>6.0%}"
              f"{row['f1']:>6.2f}   {row['median_song_precision']:.0%} "
              f"({row['songs_marked']} songs marked)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
