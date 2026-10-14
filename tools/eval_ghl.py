"""Score the 6-fret part chartgen writes against the official chart, on the
songs the GHL model held out from training.

Each song's audio is mixed as in tools/build_ghl_dataset.py and charted by
the full pipeline. Positions match within MATCH_S after the song.ini delay;
shapes are then compared on the matched positions.

    python -m tools.eval_ghl --charts data/ghl-official -n 10
"""
import argparse
import re
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from chartgen import ghl, stems
from chartgen.__main__ import build_args
from chartgen import pipeline
from tools.build_ghl_dataset import mix
from tools.ghl_charts import DIFF_BASE, load
from tools.train_ghl import ini_delay_s, song_key

MATCH_S = 0.05
DIFFS = tuple(DIFF_BASE)


def read_ghl(chart_text: str) -> dict[str, list[tuple[float, frozenset]]]:
    """{difficulty: [(seconds, shape)]} from a written notes.chart."""
    res = int(re.search(r"Resolution = (\d+)", chart_text).group(1))
    sync = re.search(r"\[SyncTrack\]\s*\{(.*?)\}", chart_text, re.S).group(1)
    bpms = [(int(t), int(b) / 1000) for t, b in re.findall(r"(\d+) = B (\d+)", sync)]

    def seconds(tick):
        elapsed, last_tick, bpm = 0.0, 0, bpms[0][1]
        for change, value in bpms:
            if change >= tick:
                break
            elapsed += (change - last_tick) / res * 60 / bpm
            last_tick, bpm = change, value
        return elapsed + (tick - last_tick) / res * 60 / bpm

    lanes = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 8: 5}
    out = {}
    for diff in DIFFS:
        block = re.search(rf"\[{diff}GHLGuitar\]\s*\{{(.*?)\}}", chart_text, re.S)
        shapes: dict[int, set] = {}
        for tick, number in re.findall(r"(\d+) = N (\d+) \d+", block.group(1) if block else ""):
            if int(number) in lanes or int(number) == 7:
                shapes.setdefault(int(tick), set())
                if int(number) in lanes:
                    shapes[int(tick)].add(lanes[int(number)])
        out[diff] = [(seconds(t), frozenset(s)) for t, s in sorted(shapes.items())]
    return out


def column(shape) -> float:
    return np.mean([b % 3 for b in shape]) if shape else -1.0


def score(ours, theirs) -> dict:
    """Greedy one-to-one matching in time order, then shape agreement."""
    their_s = np.array([s for s, _ in theirs])
    used = np.zeros(len(theirs), bool)
    pairs = []
    for s, shape in ours:
        j = int(np.searchsorted(their_s, s))
        best = min((k for k in (j - 1, j) if 0 <= k < len(theirs) and not used[k]),
                   key=lambda k: abs(their_s[k] - s), default=None)
        if best is not None and abs(their_s[best] - s) <= MATCH_S:
            used[best] = True
            pairs.append((shape, theirs[best][1]))
    moves = [(np.sign(column(a2) - column(a1)), np.sign(column(b2) - column(b1)))
             for (a1, b1), (a2, b2) in zip(pairs, pairs[1:], strict=False)]
    return {"ours": len(ours), "theirs": len(theirs), "matched": len(pairs),
            "exact": sum(a == b for a, b in pairs),
            "size": sum(len(a) == len(b) for a, b in pairs),
            "row": sum((min(a, default=-1) >= 3) == (min(b, default=-1) >= 3) for a, b in pairs),
            "direction": sum(a == b for a, b in moves), "moves": len(moves)}


def violations(tiers) -> list[str]:
    out = []
    if any(b >= 3 for _, s in tiers["Easy"] for b in s):
        out.append("Easy uses the black row")
    if any(len(s) > 2 for _, s in tiers["Medium"]):
        out.append("Medium has 3-button chords")
    expert = {round(s, 3) for s, _ in tiers["Expert"]}
    if any(round(s, 3) not in expert for d in DIFFS[1:] for s, _ in tiers[d]):
        out.append("a lower tier has a position Expert lacks")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--charts", type=Path, default=Path("data/ghl-official"))
    ap.add_argument("-n", type=int, default=10, help="holdout songs to chart")
    ap.add_argument("--out", type=Path, default=Path("data/ghl-eval"))
    args = ap.parse_args(argv)

    holdout = set(torch.load(ghl.MODEL_PATH, map_location="cpu")["holdout"])
    folders = sorted(p.parent for p in args.charts.rglob("notes.mid")
                     if song_key(p.parent.name) in holdout)[:args.n]
    totals = {d: {} for d in DIFFS}
    for folder in folders:
        official = load(folder / "notes.mid")
        delay = ini_delay_s(folder)
        with tempfile.TemporaryDirectory() as tmp:
            audio = Path(tmp) / f"{folder.name}.wav"
            sf.write(audio, mix(folder), stems.SR)
            opts, _ = build_args([str(audio), "-o", str(args.out), "--no-lyrics",
                                  "--name", folder.name, "--artist", "eval"])
            opts.audio = str(audio)
            result = pipeline.run(opts, progress=lambda m: None)
        ours = read_ghl((result["song_dir"] / "notes.chart").read_text())
        theirs = {d: [(official.seconds(t) + delay, s) for t, s, _ in official.tiers[d]]
                  for d in DIFFS}
        problems = violations(ours)
        line = []
        for d in DIFFS:
            stats = score(ours[d], theirs[d])
            for key, value in stats.items():
                totals[d][key] = totals[d].get(key, 0) + value
            line.append(f"{d[0]} {stats['ours']}/{stats['theirs']}")
        print(f"{folder.name[:44]:44s} {' '.join(line)} {'; '.join(problems) or 'rules ok'}",
              flush=True)

    print()
    for d in DIFFS:
        t = totals[d]
        m = max(t.get("matched", 0), 1)
        print(f"{d:6s} notes {t['ours'] / max(t['theirs'], 1):.0%} of official, "
              f"precision {t['matched'] / max(t['ours'], 1):.0%} recall {t['matched'] / max(t['theirs'], 1):.0%}; "
              f"on matches: exact {t['exact'] / m:.0%}, size {t['size'] / m:.0%}, "
              f"row {t['row'] / m:.0%}, direction {t['direction'] / max(t['moves'], 1):.0%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
