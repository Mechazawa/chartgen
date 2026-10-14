"""Measure how human charters actually use tap notes.

No authoring guideline anywhere states when taps are appropriate — the YARN
standard says only "use them sensibly", the Chorus Encore scanner has no
tap rule at all, and RBN/C3 never mentions them because Rock Band has no
such note. So rather than invent a threshold, measure one: this reports how
often taps appear in a real library, at which difficulties, on chords or
singles, and — the number that would actually drive a rule — how fast the
passages carrying them are compared with the rest of the chart.

    python tools/study_taps.py ~/.clonehero/Songs
"""
import argparse
import re
import statistics
import sys
from pathlib import Path

TIERS = ("ExpertSingle", "HardSingle", "MediumSingle", "EasySingle")
NOTE_RE = re.compile(r"(\d+)\s*=\s*N\s+(\d+)\s+(\d+)")
TAP, FORCED, OPEN = 6, 5, 7


def tier_stats(body: str, resolution: int) -> dict | None:
    """Per-tier tap usage and the note spacing around tapped positions."""
    by_tick: dict[int, set[int]] = {}
    for tick, lane, _ in NOTE_RE.findall(body):
        by_tick.setdefault(int(tick), set()).add(int(lane))
    if not by_tick:
        return None

    ticks = sorted(by_tick)
    # Gap to the previous position, in beats — tempo-independent, and what
    # "fast passage" means in practice.
    gaps = {t: (t - p) / resolution for p, t in zip(ticks, ticks[1:])}

    tapped, tap_chords, tap_gaps, other_gaps = [], 0, [], []
    for tick in ticks:
        lanes = by_tick[tick]
        frets = {l for l in lanes if l < 5 or l == OPEN}
        gap = gaps.get(tick)
        if TAP in lanes:
            tapped.append(tick)
            if len(frets) > 1:
                tap_chords += 1
            if gap is not None:
                tap_gaps.append(gap)
        elif gap is not None:
            other_gaps.append(gap)

    positions = [t for t in ticks if any(l < 5 or l == OPEN for l in by_tick[t])]
    return {
        "positions": len(positions),
        "taps": len(tapped),
        "tap_chords": tap_chords,
        "tap_gaps": tap_gaps,
        "other_gaps": other_gaps,
        "forced": sum(1 for t in ticks if FORCED in by_tick[t]),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root", type=Path)
    ap.add_argument("--min-positions", type=int, default=100)
    args = ap.parse_args()

    per_tier: dict[str, list[dict]] = {t: [] for t in TIERS}
    charts_with_taps = songs = 0
    for chart in sorted(args.root.rglob("notes.chart")):
        if "chartgen" in chart.parent.name:
            continue  # measure humans, not ourselves
        try:
            text = chart.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        match = re.search(r"Resolution\s*=\s*(\d+)", text)
        if not match:
            continue
        resolution = int(match.group(1))
        songs += 1
        used_taps = False
        for tier in TIERS:
            block = re.search(rf"\[{tier}\]\s*\{{(.*?)\}}", text, re.S)
            if not block:
                continue
            stats = tier_stats(block.group(1), resolution)
            if stats and stats["positions"] >= args.min_positions:
                per_tier[tier].append(stats)
                used_taps |= stats["taps"] > 0
        charts_with_taps += used_taps

    if not songs:
        print("no charts found")
        return 1

    print(f"{songs} human charts scanned; "
          f"{charts_with_taps} ({charts_with_taps / songs:.0%}) use taps at all\n")
    print(f"{'tier':<14}{'charts':>7}{'w/taps':>8}{'tap share':>11}"
          f"{'chord taps':>12}{'forced':>8}")
    for tier in TIERS:
        rows = per_tier[tier]
        if not rows:
            continue
        with_taps = [r for r in rows if r["taps"]]
        total_pos = sum(r["positions"] for r in rows)
        total_taps = sum(r["taps"] for r in rows)
        chord_share = (sum(r["tap_chords"] for r in with_taps)
                       / max(1, sum(r["taps"] for r in with_taps)))
        print(f"{tier:<14}{len(rows):>7}{len(with_taps):>8}"
              f"{total_taps / max(1, total_pos):>10.1%}"
              f"{chord_share:>11.0%}"
              f"{sum(r['forced'] for r in rows) / max(1, total_pos):>8.1%}")

    print("\nspacing before a note, in beats (lower = faster passage):")
    print(f"{'tier':<14}{'tapped':>18}{'everything else':>20}")
    for tier in TIERS:
        tap_gaps = [g for r in per_tier[tier] for g in r["tap_gaps"]]
        other = [g for r in per_tier[tier] for g in r["other_gaps"]]
        if len(tap_gaps) < 20:
            continue
        print(f"{tier:<14}"
              f"{statistics.median(tap_gaps):>13.3f} beats"
              f"{statistics.median(other):>15.3f} beats")
        # Medians can match while the tails differ, so compare the share of
        # genuinely fast spacing on both sides — that is what a speed-based
        # rule would key on.
        fast = sum(1 for g in tap_gaps if g <= 0.25) / len(tap_gaps)
        fast_other = sum(1 for g in other if g <= 0.25) / max(1, len(other))
        print(f"{'':14}  a 16th or closer: {fast:.0%} of taps vs "
              f"{fast_other:.0%} of everything else"
              f"   (ratio {fast / max(fast_other, 1e-9):.2f}x)")

    print("\nReading: if tapped notes cluster at much tighter spacing than the "
          "rest, speed is the trigger and a threshold can be derived. If the "
          "spacings look alike, taps are a stylistic choice we should not "
          "auto-generate.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
