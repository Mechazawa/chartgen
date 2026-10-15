"""Paired per-song significance testing for calibration decisions.

STRUM's ablation found that 4 of its 7 hand-tuned heuristics had NO
statistically significant effect (|dF1| <= 0.001, p >= 0.29) despite each
having been justified by real failure cases during development — aggregate
deltas hide that a change helps three songs, hurts three others, and does
nothing elsewhere. The remedy is the test they used: Wilcoxon signed-rank
over per-song paired scores.

This provides the helper every future sweep should call before a constant
changes, and demonstrates it on a live question: do the tap softness axes
(attack-led vs brightness-led) actually differ, per song, on the 232-chart
calibration cache?

    python tools/paired_test.py work/cal_taps_A.json work/cal_taps_B.json
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def paired_verdict(a: list, b: list, label_a: str, label_b: str) -> str:
    """One-line verdict on whether two per-song score lists really differ."""
    from scipy.stats import wilcoxon

    deltas = [x - y for x, y in zip(a, b, strict=True)]
    if all(abs(d) < 1e-12 for d in deltas):
        return f"{label_a} vs {label_b}: identical on every song"
    stat, p = wilcoxon(a, b)
    med = statistics.median(deltas)
    better = sum(1 for d in deltas if d > 0)
    worse = sum(1 for d in deltas if d < 0)
    verdict = "SIGNIFICANT" if p < 0.05 else "no real difference"
    return (f"{label_a} vs {label_b}: median delta {med:+.3f}, better on "
            f"{better} songs / worse on {worse}, p={p:.3f} -> {verdict}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("caches", type=Path, nargs="+")
    args = ap.parse_args(argv)

    from sweep_taps import softness
    from chartgen.taps import phrases

    songs = [s for p in args.caches
             for s in json.loads(p.read_text(encoding="utf-8"))]
    songs = list({s["name"]: s for s in songs}.values())

    def per_song_precision(weights):
        out = []
        for song in songs:
            soft = softness(song, weights)
            notes = [(n["tick"], 0, 0) for n in song["notes"]]
            got = phrases(notes, soft, song["resolution"])
            truth = {n["tick"] for n in song["notes"] if n["tap"]}
            out.append(len(got & truth) / len(got) if got else 0.0)
        return out

    configs = {
        "attack-led (shipped)": {"attack": 1.0, "bright": 0.5, "loud": 0.0},
        "bright-led": {"attack": 0.5, "bright": 1.0, "loud": 0.0},
        "attack-only": {"attack": 1.0, "bright": 0.0, "loud": 0.0},
    }
    scores = {label: per_song_precision(w) for label, w in configs.items()}
    print(f"{len(songs)} songs, per-song tap precision:\n")
    labels = list(configs)
    for i, a in enumerate(labels):
        for b in labels[i + 1:]:
            print(" ", paired_verdict(scores[a], scores[b], a, b))
    return 0


if __name__ == "__main__":
    sys.exit(main())
