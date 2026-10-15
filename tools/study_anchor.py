"""Do human charts respect a rolling hand anchor?

The cross-game research converged on "playability is a property of note
sequences, not single transitions", and the deep-dive found the literature
has the CONCEPT (Radicioni's phrase-level hand position) but no calibrated
formula — every accessible cost model is one-step-back, same as our
playability.cost(). Before adding a rolling-anchor term, measure whether
the behaviour it would enforce actually exists in human charts:

1. OUTSIDE-WINDOW RATE: with an exponential moving average of the anchor
   (lowest lane) as the hand's resting position, how often does a note land
   outside a hand-span window around it? If humans are rarely outside,
   the term describes something real; if they are outside constantly, no
   rolling anchor governs human charting and the term would only enforce
   monotony.
2. ANCHOR SHIFTS PER BAR: how often the resting position relocates.

Timing comes from each chart's own SyncTrack. Our generated charts are
measured alongside for the comparison that matters.

    python tools/study_anchor.py data/calibration
"""
import argparse
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from study_solos import blocks, is_generated, RES_RE
from study_tap_timbre import NOTE_RE, SYNC_RE, tick_to_time

OPEN = 7
HALF_LIFE_S = 0.4   # EMA half-life for the resting position
WINDOW = 1.5        # lanes either side of the anchor = a 4-lane hand span
RESET_GAP_S = 1.0   # a rest this long frees the hand (mirrors free_gap_beats)
SHIFT_LANES = 2.0   # a resting-position move this big counts as a relocation


def study(chart: Path):
    b = blocks(chart.read_text(encoding="utf-8", errors="ignore"))
    found = RES_RE.search(b.get("Song", ""))
    res = int(found.group(1)) if found else 192
    sync = [(int(t), int(v)) for t, v in SYNC_RE.findall(b.get("SyncTrack", ""))]
    if not sync or "ExpertSingle" not in b:
        return None
    by_tick: dict[int, list[int]] = {}
    for t, lane, _ in NOTE_RE.findall(b["ExpertSingle"]):
        lane = int(lane)
        if lane < 5:  # opens have no lane position; skip them for hand state
            by_tick.setdefault(int(t), []).append(lane)
    ticks = sorted(by_tick)
    if len(ticks) < 100:
        return None
    at = tick_to_time(sync, res)

    outside = total = shifts = 0
    rolling = None
    settled = None
    prev_time = None
    for tick in ticks:
        anchor = min(by_tick[tick])
        now = at(tick)
        if rolling is None or (prev_time is not None and now - prev_time >= RESET_GAP_S):
            rolling = float(anchor)
            settled = float(anchor)
        else:
            total += 1
            if abs(anchor - rolling) > WINDOW:
                outside += 1
            alpha = 1 - math.exp(-math.log(2) * (now - prev_time) / HALF_LIFE_S)
            rolling += alpha * (anchor - rolling)
            if settled is None or abs(rolling - settled) >= SHIFT_LANES:
                shifts += 1
                settled = rolling
        prev_time = now

    duration_bars = max(1.0, (ticks[-1] - ticks[0]) / res / 4)
    return {
        "outside": outside / max(1, total),
        "shifts_per_bar": shifts / duration_bars,
        "name": chart.parent.name,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("library", type=Path)
    args = ap.parse_args(argv)

    groups = {"HUMAN": [], "chartgen": []}
    for chart in sorted(args.library.rglob("*.chart")):
        b = blocks(chart.read_text(encoding="utf-8", errors="ignore"))
        row = study(chart)
        if row:
            groups["chartgen" if is_generated(b) else "HUMAN"].append(row)
    extra = Path.home() / ".clonehero/Songs"
    for chart in sorted(extra.rglob("*.chart")):
        b = blocks(chart.read_text(encoding="utf-8", errors="ignore"))
        if is_generated(b):
            row = study(chart)
            if row:
                groups["chartgen"].append(row)

    for label, rows in groups.items():
        if not rows:
            continue
        out = sorted(r["outside"] for r in rows)
        sh = sorted(r["shifts_per_bar"] for r in rows)
        q = statistics.quantiles(out, n=10) if len(out) > 9 else [out[0]] * 9
        print(f"{label:<9} {len(rows)} charts")
        print(f"  notes outside the rolling hand window: "
              f"median {statistics.median(out):.1%}  p10 {q[0]:.1%}  p90 {q[8]:.1%}")
        print(f"  anchor relocations per bar:            "
              f"median {statistics.median(sh):.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
