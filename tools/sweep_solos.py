"""Choose the solo rule by search over cached evidence, not by intuition.

The first audio-based attempt scored 20% precision / 25% recall - no better
than the lyric-gap rule it replaced - because every test was "above this
song's median", which roughly half of all sections pass by construction.
This searches the actual threshold space against human solo markers.

Scored with F0.5, weighting precision over recall on purpose: a missing solo
marker costs a scoring bonus, while one over a verse reads as plainly wrong.

    python tools/sweep_solos.py work/solo_features.json
"""
import argparse
import itertools
import json
import statistics
import sys
from pathlib import Path

FEATURES = ("voiced", "confidence", "spread", "bright", "novelty")


def zscores(sections):
    """Per-song z-scores, so thresholds mean the same thing in every mix."""
    usable = [s for s in sections if s["evidence"]]
    stats = {}
    for key in FEATURES:
        vals = [s["evidence"][key] for s in usable]
        if len(vals) < 3:
            return None
        mean = statistics.mean(vals)
        spread = statistics.pstdev(vals) or 1e-6
        stats[key] = (mean, spread)
    for s in sections:
        if not s["evidence"]:
            s["z"] = None
            continue
        s["z"] = {k: (s["evidence"][k] - stats[k][0]) / stats[k][1] for k in FEATURES}
    return sections


def overlaps(a, b):
    return not (a[1] < b[0] or a[0] > b[1])


def iou(a, b) -> float:
    lo, hi = max(a[0], b[0]), min(a[1], b[1])
    if hi <= lo:
        return 0.0
    return (hi - lo) / ((a[1] - a[0]) + (b[1] - b[0]) - (hi - lo))


def candidates(song, max_run=3):
    """Spans of 1..max_run consecutive sections.

    A solo does not have to occupy exactly one chroma segment — the
    segmenter splits on harmonic change, and a solo that moves through two
    key centres becomes two segments. Allowing short runs lets a candidate
    match the real thing without loosening the boundaries themselves.
    """
    sections = [s for s in song["sections"] if s.get("z")]
    out = []
    for i in range(len(sections)):
        for run in range(1, max_run + 1):
            group = sections[i:i + run]
            if len(group) < run:
                break
            weights = [g["end"] - g["start"] for g in group]
            total = sum(weights) or 1
            out.append({
                "start": group[0]["start"], "end": group[-1]["end"],
                "notes": sum(g["notes"] for g in group),
                "z": {k: sum(g["z"][k] * w for g, w in zip(group, weights, strict=True)) / total
                      for k in FEATURES},
            })
    return out


def ceiling(songs, min_iou=0.5):
    """The best any rule over these sections could possibly score.

    If our chroma boundaries do not line up with where solos actually start
    and stop, no amount of threshold tuning reaches them — that would be a
    segmentation problem wearing a scoring problem's clothes, and worth
    knowing before tuning anything.
    """
    total = hit = 0
    best_ious = []
    for song in songs:
        for truth in song["truth"]:
            total += 1
            best = max((iou(tuple(truth), (s["start"], s["end"]))
                        for s in candidates(song)), default=0.0)
            best_ious.append(best)
            hit += best >= min_iou
    best_ious.sort()
    median = best_ious[len(best_ious) // 2] if best_ious else 0.0
    print(f"segmentation ceiling: {hit}/{total} human solos have a candidate "
          f"overlapping them at IoU >= {min_iou} (median best IoU "
          f"{median:.2f})\n")


def evaluate(songs, weights, min_beats, pos_lo, pos_hi, threshold, top_k,
             max_words, min_notes=16, min_iou=0.25):
    """A pick only counts as a hit if it substantially covers the real solo.

    The first pass scored ANY overlap as a hit, and that flattered the rule:
    of eight 'hits' it reported, three touched the human solo by one section
    boundary or a few grazing ticks (IoU 0.00-0.01). A marker that covers
    the section BEFORE the solo is a miss in every way that matters to the
    player, so hits are IoU-gated now.
    """
    tp = fp = fn = 0
    clean_songs = clean_ok = 0
    for song in songs:
        res, last = song["resolution"], song["last_tick"]
        picks = []
        for s in candidates(song):
            if s["notes"] < min_notes:
                continue
            beats = (s["end"] - s["start"]) / res
            if beats < min_beats or beats > 128:
                continue
            if not (pos_lo <= s["start"] / last <= pos_hi):
                continue
            if sum(1 for w in song["words"] if s["start"] <= w < s["end"]) > max_words:
                continue
            score = sum(weights[k] * s["z"][k] for k in FEATURES)
            if score >= threshold:
                picks.append((score, s["start"], s["end"]))
        picks.sort(reverse=True)
        got = [(a, b) for _, a, b in picks[:top_k]]
        truth = [tuple(t) for t in song["truth"]]

        hits = [g for g in got if any(iou(g, t) >= min_iou for t in truth)]
        tp += len(hits)
        fp += len(got) - len(hits)
        fn += sum(1 for t in truth if not any(iou(g, t) >= min_iou for g in got))
        if not truth:
            clean_songs += 1
            clean_ok += not got
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f_half = ((1.25 * precision * recall / (0.25 * precision + recall))
              if precision + recall else 0.0)
    return {"precision": precision, "recall": recall, "f_half": f_half,
            "tp": tp, "fp": fp, "fn": fn,
            "quiet_songs_kept_quiet": (clean_ok, clean_songs)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cache", type=Path, nargs="+",
                    help="one or more caches; the dump is split across "
                         "processes, so they merge here")
    ap.add_argument("--top", type=int, default=12, help="rules to print")
    args = ap.parse_args(argv)

    songs = [song for path in args.cache
             for song in json.loads(path.read_text(encoding="utf-8"))]
    songs = list({s["name"]: s for s in songs}.values())
    songs = [s for s in (zscores(s["sections"]) and s for s in songs) if s]
    have = sum(1 for s in songs if s["truth"])
    print(f"{len(songs)} songs ({have} with solos, "
          f"{sum(len(s['truth']) for s in songs)} solos)\n")
    ceiling(songs)

    weight_sets = {
        "lead only": {"voiced": 1, "confidence": 0, "spread": 0, "bright": 0, "novelty": 0},
        "lead+novelty": {"voiced": 1, "confidence": 0, "spread": 0, "bright": 0, "novelty": 1},
        "lead+bright": {"voiced": 1, "confidence": 0, "spread": 0, "bright": 1, "novelty": 0},
        "pitchy": {"voiced": 1, "confidence": 1, "spread": 1, "bright": 0, "novelty": 0},
        "all equal": {k: 1 for k in FEATURES},
        "novelty only": {"voiced": 0, "confidence": 0, "spread": 0, "bright": 0, "novelty": 1},
        "bright only": {"voiced": 0, "confidence": 0, "spread": 0, "bright": 1, "novelty": 0},
        "spread only": {"voiced": 0, "confidence": 0, "spread": 1, "bright": 0, "novelty": 0},
        "lead+spread": {"voiced": 1, "confidence": 0, "spread": 1, "bright": 0, "novelty": 0},
    }

    results = []
    for label, weights in weight_sets.items():
        n = sum(1 for v in weights.values() if v) or 1
        for min_beats, (lo, hi), threshold, top_k, max_words in itertools.product(
            (16, 24, 32, 40), ((0.2, 0.95), (0.35, 0.85), (0.45, 0.8)),
            (0.5, 0.75, 1.0, 1.25, 1.5, 2.0), (1, 2), (0, 2),
        ):
            row = evaluate(songs, weights, min_beats, lo, hi, threshold * n,
                           top_k, max_words)
            row["rule"] = (f"{label:<13} beats>={min_beats:<3} pos {lo}-{hi} "
                           f"z>={threshold} top{top_k} words<={max_words}")
            results.append(row)

    results.sort(key=lambda r: (-r["f_half"], -r["precision"]))
    print(f"{'rule':<62}{'prec':>6}{'rec':>6}{'F0.5':>7}   quiet songs kept quiet")
    for row in results[:args.top]:
        kept, total = row["quiet_songs_kept_quiet"]
        print(f"{row['rule']:<62}{row['precision']:>5.0%}{row['recall']:>6.0%}"
              f"{row['f_half']:>7.2f}   {kept}/{total}")

    print("\nbaseline for reference (lyric-gap rule measured earlier): "
          "18% precision")
    return 0


if __name__ == "__main__":
    sys.exit(main())
