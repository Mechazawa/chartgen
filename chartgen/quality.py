"""Reject degenerate charts before they reach the player.

Generation is sampled, and lane choice turned out to be the least stable thing
about it: four runs at identical settings on one clip produced 84% red, an even
spread, and 34% open notes. The mapping is fine and the model does use all five
frets, but nothing downstream noticed when a run collapsed onto one lane.

So score a candidate and regenerate if it is bad, rather than trusting one roll.
"""
import math
from collections import Counter
from itertools import pairwise

FRETS = (0, 1, 2, 3, 4)
OPEN = 7
OPEN_BUDGET = 0.10  # share of open notes tolerated before it counts against us


def lane_counts(notes) -> Counter:
    return Counter(lane for _, lane, _ in notes)


def variety_score(notes, n_lanes: int = 5) -> float:
    """0-1, higher is better: fret notes spread evenly, open notes used sparingly.

    Normalized Shannon entropy over the frets, scaled down when open notes
    take over. Entropy rather than "share of the most-used lane" because it also
    catches a chart that only ever alternates two frets.

    n_lanes is the number of lanes the tier is *allowed* to use: Easy charts
    are G/R/Y by convention, so judging them against five lanes caps their
    score at log(3)/log(5) = 0.68 and flags every Easy chart as degenerate.
    """
    counts = lane_counts(notes)
    total = sum(counts.values())
    if not total:
        return 0.0

    fret_total = sum(counts.get(f, 0) for f in FRETS)
    if not fret_total:
        return 0.0

    shares = [counts.get(f, 0) / fret_total for f in FRETS if counts.get(f, 0)]
    entropy = -sum(s * math.log(s) for s in shares) / math.log(max(2, n_lanes))

    open_share = counts.get(OPEN, 0) / total
    penalty = max(0.0, open_share - OPEN_BUDGET) * 3
    return max(0.0, min(1.0, entropy) * (1 - penalty))


def step_profile(notes) -> dict:
    """How the chart moves across the fretboard, not just which frets it uses.

    Playtesting a real song produced a chart that scored 0.82 on variety_score
    and was still no fun: it walked the fretboard one fret at a time. Measured,
    74% of its fret-to-fret steps were +/-1 and only 6% repeated a fret. Entropy
    cannot see this — a perfect G-R-Y-B-O staircase uses all five frets evenly and
    therefore scores near 1.0. This looks at transitions instead.
    """
    by_tick: dict[int, list[int]] = {}
    for tick, lane, _ in notes:
        by_tick.setdefault(tick, []).append(lane)

    singles = [v[0] for _, v in sorted(by_tick.items()) if len(v) == 1 and v[0] != OPEN]
    steps = [b - a for a, b in pairwise(singles)]
    if not steps:
        return {"steps": 0, "adjacent": 0.0, "repeat": 0.0, "run_share": 0.0}

    runs, current = [], 1
    for s in steps:
        if s == 1:
            current += 1
        else:
            runs.append(current)
            current = 1
    runs.append(current)

    return {
        "steps": len(steps),
        "adjacent": sum(1 for s in steps if abs(s) == 1) / len(steps),
        "repeat": sum(1 for s in steps if s == 0) / len(steps),
        "run_share": sum(r for r in runs if r >= 4) / max(1, len(singles)),
        "chord_share": sum(1 for v in by_tick.values() if len(v) > 1) / len(by_tick),
    }


def walk_score(notes) -> float:
    """0-1, higher is better: penalises fretboard-walking.

    Calibrated against the 116 real community Expert charts in the local
    library (tools/calibrate_quality.py). The earlier version penalised high
    adjacency OR low repeats independently and failed 41% of those charts —
    real charts run the gamut on both (adjacency p95 = 72%, repeat p25 = 4%).
    What real charts never do is BOTH at once: high adjacency with almost no
    repeats, which is exactly the generated staircase (74% / 6%). So the
    penalty is the product, and a chart only fails when it walks *and* never
    lands twice on a fret.
    """
    p = step_profile(notes)
    if not p["steps"]:
        return 0.0
    # Real-chart landscape: adjacency median 0.50 / p95 0.76, repeats median
    # 0.20 / p25 0.08. The 0.60 knee sits above the median but below the
    # degenerate staircase's 0.74.
    walkiness = max(0.0, p["adjacent"] - 0.60)
    repeat_deficit = max(0.0, 1.0 - p["repeat"] / 0.15)
    return max(0.0, 1.0 - 3.5 * walkiness * repeat_deficit)


def describe(notes) -> str:
    """One-line summary for progress output."""
    counts = lane_counts(notes)
    total = sum(counts.values()) or 1
    names = {0: "G", 1: "R", 2: "Y", 3: "B", 4: "O", OPEN: "open"}
    spread = " ".join(
        f"{names.get(k, k)}:{counts[k] * 100 // total}%" for k in sorted(counts)
    )
    p = step_profile(notes)
    return (f"variety {variety_score(notes):.2f}  walk {walk_score(notes):.2f}  "
            f"[{spread}]  adjacent-steps {p['adjacent'] * 100:.0f}% "
            f"repeats {p['repeat'] * 100:.0f}%")
