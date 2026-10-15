"""Measure ladder structure and star power conventions in human charts.

Three questions the cross-game research raised, answered on the 800-chart
calibration set before any constant changes:

1. ADJACENT-TIER RATIOS: how steep is each step of the ladder song by song?
   (Tier-to-Expert medians are known; the per-song adjacent-step spread is
   what an anti-spike bound needs.)
2. INCLUSION: is each tier a subset of the tier above? Professional charts
   measure 97-99% elsewhere (GenéLive); what does this community do?
3. STAR POWER: phrases per song, spacing in beats, phrase length, and the
   dead zone before the end. RBN's spec says one phrase per 40 beats,
   ~1 measure long, none in the last ~8 measures; YARG independently
   converged on the same. Measure what charters actually do.

    python tools/study_ladder.py data/calibration
"""
import argparse
import re
import statistics
import sys
from itertools import pairwise
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from study_solos import blocks, is_generated, RES_RE

NOTE_RE = re.compile(r"(\d+)\s*=\s*N\s+(\d+)\s+(\d+)")
SP_RE = re.compile(r"(\d+)\s*=\s*S\s+2\s+(\d+)")
TIERS = ("ExpertSingle", "HardSingle", "MediumSingle", "EasySingle")
OPEN = 7


def dist(vals, fmt="{:.2f}"):
    vals = sorted(vals)
    if len(vals) < 4:
        return "n/a"
    q = statistics.quantiles(vals, n=10)
    return (f"median {fmt.format(statistics.median(vals))}  "
            f"p10 {fmt.format(q[0])}  p90 {fmt.format(q[8])}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("library", type=Path)
    args = ap.parse_args(argv)

    adjacent = {pair: [] for pair in pairwise(TIERS)}
    inclusion = {pair: [] for pair in pairwise(TIERS)}
    sp_per_song, sp_gap_beats, sp_len_beats, sp_tail_beats = [], [], [], []

    n = 0
    for chart in sorted(args.library.rglob("*.chart")):
        b = blocks(chart.read_text(encoding="utf-8", errors="ignore"))
        if is_generated(b):
            continue
        found = RES_RE.search(b.get("Song", ""))
        res = int(found.group(1)) if found else 192
        ticks = {}
        for tier in TIERS:
            if tier in b:
                ticks[tier] = {int(t) for t, l, _ in NOTE_RE.findall(b[tier])
                               if int(l) < 5 or int(l) == OPEN}
        if any(len(ticks.get(t, ())) < 30 for t in TIERS):
            continue
        n += 1
        for hi, lo in pairwise(TIERS):
            adjacent[(hi, lo)].append(len(ticks[lo]) / len(ticks[hi]))
            inclusion[(hi, lo)].append(len(ticks[lo] & ticks[hi]) / len(ticks[lo]))

        expert = b["ExpertSingle"]
        phrases = sorted((int(t), int(l)) for t, l in SP_RE.findall(expert))
        last = max(ticks["ExpertSingle"])
        if phrases:
            sp_per_song.append(len(phrases) / max(1.0, last / res / 40))
            sp_len_beats.extend((l / res) for _, l in phrases)
            sp_gap_beats.extend((b2 - a) / res for (a, _), (b2, _) in pairwise(phrases))
            sp_tail_beats.append((last - (phrases[-1][0] + phrases[-1][1])) / res)

    print(f"{n} full-ladder human charts\n")
    print("ADJACENT-TIER POSITION RATIOS (lower / upper, per song):")
    for (hi, lo), vals in adjacent.items():
        print(f"  {lo:<13} / {hi:<13} {dist(vals)}")
    print("\nINCLUSION (share of lower tier's positions present in upper):")
    for (hi, lo), vals in inclusion.items():
        print(f"  {lo:<13} in {hi:<13} {dist(vals, '{:.1%}')}")
    print("\nSTAR POWER (Expert):")
    print(f"  phrases per 40 beats:      {dist(sp_per_song)}   (RBN spec: 1.0)")
    print(f"  phrase length, beats:      {dist(sp_len_beats, '{:.1f}')}   (RBN: ~4)")
    print(f"  gap between starts, beats: {dist(sp_gap_beats, '{:.0f}')}   (RBN: ~40)")
    print(f"  beats after last phrase:   {dist(sp_tail_beats, '{:.0f}')}   (RBN: >=32)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
