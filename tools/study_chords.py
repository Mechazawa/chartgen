"""How human charters actually use chords and sustain texture.

The chord simplification that fixed unplayable shape-storms over-corrected
(some charts at 0% chords vs a human 13-22%). Before reintroducing chords
and "fun combos", measure what the 800-chart set does with them:

- chord width: 2-note vs 3-note vs wider, and the span between lanes
- chord runs: same shape repeated (a strummed riff) vs shape changes,
  and how fast shapes actually change
- extended sustains: notes held under subsequent notes, and disjoint
  chord sustains (chord members with different lengths)
- gallops and trills: the rhythmic textures players name as fun

    python tools/study_chords.py data/calibration
"""
import argparse
import re
import statistics
import sys
from itertools import pairwise
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from study_solos import blocks, is_generated, RES_RE
from study_tap_timbre import SYNC_RE

NOTE_RE = re.compile(r"(\d+)\s*=\s*N\s+(\d+)\s+(\d+)")
OPEN = 7


def study(chart: Path):
    b = blocks(chart.read_text(encoding="utf-8", errors="ignore"))
    found = RES_RE.search(b.get("Song", ""))
    res = int(found.group(1)) if found else 192
    sync = [(int(t), int(v)) for t, v in SYNC_RE.findall(b.get("SyncTrack", ""))]
    if "ExpertSingle" not in b or not sync:
        return None
    by_tick: dict[int, dict[int, int]] = {}
    for t, lane, sus in NOTE_RE.findall(b["ExpertSingle"]):
        lane = int(lane)
        if lane < 5 or lane == OPEN:
            by_tick.setdefault(int(t), {})[lane] = int(sus)
    ticks = sorted(by_tick)
    if len(ticks) < 100:
        return None
    chords = [t for t in ticks if len(by_tick[t]) > 1]
    if not ticks:
        return None
    widths = {2: 0, 3: 0, 4: 0}
    spans = []
    for t in chords:
        lanes = [l for l in by_tick[t] if l != OPEN]
        widths[min(4, len(by_tick[t]))] = widths.get(min(4, len(by_tick[t])), 0) + 1
        if len(lanes) > 1:
            spans.append(max(lanes) - min(lanes))

    # Chord transitions: consecutive chord positions within 1 beat.
    same_shape = changed = 0
    change_gaps = []
    for a, c in pairwise(ticks):
        if len(by_tick[a]) > 1 and len(by_tick[c]) > 1 and c - a <= res:
            if set(by_tick[a]) == set(by_tick[c]):
                same_shape += 1
            else:
                changed += 1
                change_gaps.append((c - a) / res)

    # Extended sustains: a note still ringing when the next position starts.
    extended = 0
    sustained = 0
    disjoint = 0
    for i, t in enumerate(ticks):
        lens = list(by_tick[t].values())
        if any(v > res // 4 for v in lens):
            sustained += 1
            if i + 1 < len(ticks) and t + max(lens) > ticks[i + 1]:
                extended += 1
        if len(lens) > 1 and len(set(lens)) > 1:
            disjoint += 1

    # Gallops: gap pattern short-short-long at 8th/16th scale on singles.
    gallops = 0
    gaps = [(c - a) / res for a, c in pairwise(ticks)]
    for g1, g2, g3 in zip(gaps, gaps[1:], gaps[2:], strict=False):
        if 0.2 <= g1 <= 0.3 and 0.2 <= g2 <= 0.3 and g3 >= 0.45:
            gallops += 1

    n = len(ticks)
    total_chords = max(1, len(chords))
    return {
        "name": chart.parent.name,
        "chord_share": len(chords) / n,
        "three_plus": (widths.get(3, 0) + widths.get(4, 0)) / total_chords,
        "wide_span": sum(1 for s in spans if s >= 3) / max(1, len(spans)),
        "same_shape_share": same_shape / max(1, same_shape + changed),
        "fast_changes_per_100": 100 * sum(1 for g in change_gaps if g <= 0.5) / n,
        "extended_share": extended / max(1, sustained),
        "disjoint_per_song": disjoint,
        "gallops_per_100": 100 * gallops / n,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("library", type=Path)
    args = ap.parse_args(argv)

    groups = {"HUMAN": [], "chartgen": []}
    roots = [args.library, Path.home() / ".clonehero/Songs"]
    seen = set()
    for root in roots:
        for chart in sorted(root.rglob("*.chart")):
            if chart in seen:
                continue
            seen.add(chart)
            b = blocks(chart.read_text(encoding="utf-8", errors="ignore"))
            row = study(chart)
            if row:
                groups["chartgen" if is_generated(b) else "HUMAN"].append(row)

    keys = ("chord_share", "three_plus", "wide_span", "same_shape_share",
            "fast_changes_per_100", "extended_share", "disjoint_per_song",
            "gallops_per_100")
    labels = {
        "chord_share": "chord share of positions",
        "three_plus": "3+ note chords (of chords)",
        "wide_span": "wide chords, span>=3 (of chords)",
        "same_shape_share": "same-shape repeats (of chord pairs)",
        "fast_changes_per_100": "shape changes <=1/2 beat apart /100 pos",
        "extended_share": "extended sustains (of sustains)",
        "disjoint_per_song": "disjoint chord sustains per song",
        "gallops_per_100": "gallop patterns /100 positions",
    }
    for label, rows in groups.items():
        if not rows:
            continue
        print(f"\n{label} ({len(rows)} charts)")
        for k in keys:
            vals = sorted(r[k] for r in rows)
            q = statistics.quantiles(vals, n=4) if len(vals) > 3 else [vals[0]] * 3
            fmt = "{:.0f}" if k == "disjoint_per_song" else "{:.1%}" if "per_100" not in k else "{:.2f}"
            print(f"  {labels[k]:<42} median {fmt.format(statistics.median(vals)):>7}"
                  f"   p25 {fmt.format(q[0]):>7}  p75 {fmt.format(q[2]):>7}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
