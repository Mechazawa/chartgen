"""Measure the quality scores against real community charts.

walk_score's thresholds were reasoned, not calibrated (its docstring says so).
This runs variety/walk/step profiles over every ExpertSingle in a Clone Hero
library, so the gate can be judged against charts humans actually enjoy:

    python tools/calibrate_quality.py --root ~/.clonehero/Songs

If real charts routinely fail the gate, the gate is wrong, not the charts.
"""
import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chartgen.quality import step_profile, variety_score, walk_score  # noqa: E402

NOTE_RE = re.compile(r"(\d+) = N (\d) (\d+)")


def expert_notes(chart_path: Path):
    try:
        text = chart_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    block = re.search(r"\[ExpertSingle\]\s*\{(.*?)\}", text, re.S)
    if not block:
        return None
    # 5 and 6 are the forced/tap FLAGS, not notes; counting them makes every
    # flagged tick look like a chord and empties the single-note step profile.
    notes = [(int(t), int(l), int(s)) for t, l, s in NOTE_RE.findall(block.group(1))
             if int(l) in (0, 1, 2, 3, 4, 7)]
    return notes or None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--min-notes", type=int, default=100)
    args = ap.parse_args()

    rows = []
    for chart in sorted(args.root.rglob("notes.chart")):
        notes = expert_notes(chart)
        if not notes or len(notes) < args.min_notes:
            continue
        p = step_profile(notes)
        rows.append({
            "name": chart.parent.name[:44],
            "notes": len(notes),
            "variety": variety_score(notes),
            "walk": walk_score(notes),
            "adjacent": p["adjacent"],
            "repeat": p["repeat"],
            "chords": p.get("chord_share", 0.0),
        })

    if not rows:
        print("no usable charts found")
        return 1

    rows.sort(key=lambda r: r["walk"])
    print(f"{'chart':<46}{'notes':>6}{'variety':>9}{'walk':>7}{'adj':>6}{'rep':>6}{'chrd':>6}")
    for r in rows:
        flag = "  <- would fail gate" if min(r["variety"], r["walk"]) < 0.80 else ""
        print(f"{r['name']:<46}{r['notes']:>6}{r['variety']:>9.2f}{r['walk']:>7.2f}"
              f"{r['adjacent']:>6.0%}{r['repeat']:>6.0%}{r['chords']:>6.0%}{flag}")

    def pct(key, q):
        vals = sorted(r[key] for r in rows)
        return vals[int(q * (len(vals) - 1))]

    n_fail = sum(1 for r in rows if min(r["variety"], r["walk"]) < 0.80)
    print(f"\n{len(rows)} charts;  {n_fail} ({n_fail / len(rows):.0%}) would fail the 0.80 gate")
    for key in ("variety", "walk", "adjacent", "repeat", "chords"):
        print(f"{key:>9}: p5 {pct(key, .05):.2f}  p25 {pct(key, .25):.2f}  "
              f"median {pct(key, .5):.2f}  p75 {pct(key, .75):.2f}  p95 {pct(key, .95):.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
