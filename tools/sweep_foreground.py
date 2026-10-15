"""Recalibrate the taps-v3 section rule per separator, on identical truth.

For each backend in the dump (tools/dump_foreground.py): a section whose
human chart taps at least half its notes is a POSITIVE ("tap the section
whole" is what a charter did), one with essentially no taps is a NEGATIVE;
the ambiguous middle is left out of scoring. The production rule fires
when other-share >= SOFT and other-centroid <= CEN. Both are swept per
backend, and the production pair is scored under each backend too - the
number that says whether a separator swap silently moved the decision.

    python tools/sweep_foreground.py work/foreground_ab.json
"""
import argparse
import itertools
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chartgen.taps import (  # noqa: E402
    FOREGROUND_SOFT_SHARE,
    SOFT_MAX_CENTROID_HZ,
)

POSITIVE_TAP_SHARE = 0.50
NEGATIVE_TAP_SHARE = 0.05
MIN_NOTES = 4


def score(sections, soft, cen):
    tp = fp = fn = 0
    for share, bright, label in sections:
        fired = share >= soft and bright <= cen
        if label and fired:
            tp += 1
        elif fired:
            fp += 1
        elif label:
            fn += 1
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = (1.25 * p * r / (0.25 * p + r)) if p + r else 0.0
    return {"soft": soft, "cen": cen, "precision": p, "recall": r,
            "f_half": f, "tp": tp, "fp": fp, "fn": fn}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dump", type=Path)
    args = ap.parse_args(argv)
    rows = json.loads(args.dump.read_text(encoding="utf-8"))
    backends = sorted({b for r in rows for b in r["fg"]})
    print(f"{len(rows)} songs, backends {backends}")

    for backend in backends:
        sections = []
        songs = 0
        for r in rows:
            fg = r["fg"].get(backend)
            if not fg:
                continue
            songs += 1
            for (share, bright), sec in zip(fg, r["sections"], strict=True):
                if sec["notes"] < MIN_NOTES:
                    continue
                ts = sec["taps"] / sec["notes"]
                if ts >= POSITIVE_TAP_SHARE:
                    sections.append((share, bright, True))
                elif ts <= NEGATIVE_TAP_SHARE:
                    sections.append((share, bright, False))
        pos = [s for s, _, l in sections if l]
        neg = [s for s, _, l in sections if not l]
        print(f"\n=== {backend}: {songs} songs, {len(sections)} scored sections "
              f"({len(pos)} tapped-whole, {len(neg)} untapped)")
        if not pos or not neg:
            print("   not enough of both classes to sweep")
            continue
        print(f"   other-share median: tapped {statistics.median(pos):.3f}  "
              f"untapped {statistics.median(neg):.3f}   "
              f"(this gap is the signal; its LOCATION is what a separator moves)")
        prod = score(sections, FOREGROUND_SOFT_SHARE, SOFT_MAX_CENTROID_HZ)
        print(f"   production rule (soft>={FOREGROUND_SOFT_SHARE}, "
              f"cen<={SOFT_MAX_CENTROID_HZ:.0f}): "
              f"P {prod['precision']:.0%} R {prod['recall']:.0%} "
              f"(tp{prod['tp']} fp{prod['fp']} fn{prod['fn']})")
        grid = []
        for soft, cen in itertools.product(
                [x / 100 for x in range(35, 86, 5)],
                (2000.0, 2500.0, 3000.0, 3500.0, 4000.0, 1e9)):
            grid.append(score(sections, soft, cen))
        grid.sort(key=lambda g: -g["f_half"])
        print("   best by F0.5:")
        for g in grid[:5]:
            cen = "any" if g["cen"] >= 1e8 else f"{g['cen']:.0f}"
            print(f"     soft>={g['soft']:.2f} cen<={cen:>5s}  "
                  f"P {g['precision']:.0%} R {g['recall']:.0%}  "
                  f"(tp{g['tp']} fp{g['fp']})")
        strict = [g for g in grid if g["precision"] >= 0.70]
        if strict:
            best = max(strict, key=lambda g: g["recall"])
            cen = "any" if best["cen"] >= 1e8 else f"{best['cen']:.0f}"
            print(f"   precision-first (>=70%): soft>={best['soft']:.2f} "
                  f"cen<={cen}  P {best['precision']:.0%} R {best['recall']:.0%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
