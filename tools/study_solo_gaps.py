"""Which lyric-free gaps are actually solos?

study_solos.py established that a >=2-bar gap in the singing predicts a solo
only ~18% of the time. That kills the premise our detector rests on, but it
does not say what to replace it with. This compares the gaps that DO carry a
solo against the ones that do not, on every feature a chart alone can see, so
the replacement rule is chosen from evidence rather than intuition.

    python tools/study_solo_gaps.py ~/.clonehero/Songs
"""
import argparse
import statistics
import sys
from pathlib import Path

from study_solos import blocks, lyric_ticks, note_ticks, solo_spans, RES_RE, NOTE_RE, OPEN


def gap_features(path: Path):
    text = path.read_text(encoding="utf-8", errors="ignore")
    b = blocks(text)
    found = RES_RE.search(b.get("Song", ""))
    res = int(found.group(1)) if found else 192
    expert = b.get("ExpertSingle", "")
    ticks = note_ticks(expert)
    words = lyric_ticks(b.get("Events", ""))
    spans = solo_spans(expert)
    if len(ticks) < 50 or len(words) < 20:
        return []

    # Sustain lengths, per position: solos are usually picked, not held.
    sustain = {}
    for t, lane, length in NOTE_RE.findall(expert):
        if int(lane) < 5 or int(lane) == OPEN:
            sustain[int(t)] = max(sustain.get(int(t), 0), int(length))

    last = ticks[-1]
    overall = len(ticks) / max(1.0, last / res)
    held = statistics.mean([1 if sustain.get(t, 0) > res // 4 else 0 for t in ticks])

    rows = []
    edges = [0] + words + [last]
    for a, b2 in zip(edges, edges[1:]):
        beats = (b2 - a) / res
        if beats < 8:
            continue
        inside = [t for t in ticks if a <= t <= b2]
        if len(inside) < 8:
            continue
        is_solo = any(not (e < a or s > b2) for s, e in spans)
        rows.append({
            "beats": beats,
            "pos": a / last,
            "density": (len(inside) / beats) / overall,
            "notes": len(inside),
            "held": statistics.mean([1 if sustain.get(t, 0) > res // 4 else 0
                                     for t in inside]) - held,
            "is_solo": is_solo,
        })
    return rows


def summarize(label, rows, keys):
    print(f"\n{label}  (n={len(rows)})")
    for k in keys:
        vals = sorted(r[k] for r in rows)
        if not vals:
            continue
        q = statistics.quantiles(vals, n=4) if len(vals) > 3 else [vals[0]] * 3
        print(f"  {k:<10} median {statistics.median(vals):7.2f}   "
              f"p25 {q[0]:7.2f}   p75 {q[2]:7.2f}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("library", type=Path)
    args = ap.parse_args(argv)

    rows = []
    for p in sorted(args.library.rglob("*.chart")):
        try:
            rows.extend(gap_features(p))
        except (OSError, ValueError):
            continue

    solo = [r for r in rows if r["is_solo"]]
    other = [r for r in rows if not r["is_solo"]]
    keys = ("beats", "pos", "density", "notes", "held")
    print(f"{len(rows)} lyric-free gaps (>=2 bars, >=8 notes) in songs that have lyrics")
    summarize("GAPS THAT CARRY A SOLO", solo, keys)
    summarize("GAPS THAT DO NOT", other, keys)

    # What would simple rules buy us?
    print("\nrule sweep (precision / recall over these gaps):")
    for name, fn in (
        ("baseline: any gap", lambda r: True),
        ("length >= 12 beats", lambda r: r["beats"] >= 12),
        ("length >= 24 beats", lambda r: r["beats"] >= 24),
        ("density >= 1.2x", lambda r: r["density"] >= 1.2),
        ("density >= 1.0x", lambda r: r["density"] >= 1.0),
        ("pos 0.4-0.85", lambda r: 0.4 <= r["pos"] <= 0.85),
        ("len>=24 & pos.4-.85", lambda r: r["beats"] >= 24 and 0.4 <= r["pos"] <= 0.85),
        ("len>=24 & dens>=1.0", lambda r: r["beats"] >= 24 and r["density"] >= 1.0),
        ("all three", lambda r: r["beats"] >= 24 and r["density"] >= 1.0
                                and 0.4 <= r["pos"] <= 0.85),
    ):
        picked = [r for r in rows if fn(r)]
        hit = sum(1 for r in picked if r["is_solo"])
        prec = 100.0 * hit / len(picked) if picked else 0
        rec = 100.0 * hit / len(solo) if solo else 0
        print(f"  {name:<22} picks {len(picked):>4}   precision {prec:5.1f}%   "
              f"recall {rec:5.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
