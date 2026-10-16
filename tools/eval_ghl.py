"""Score the 6-fret part chartgen writes against the official chart, on the
songs the GHL model held out from training.

Each song's audio is mixed as in tools/build_ghl_dataset.py and charted by
the full pipeline. Positions match within MATCH_S after the song.ini delay;
shapes and sustains are then compared on the matched positions, next to the
official shapes shuffled over the official positions (chance). The Expert
tier is also scored in seconds against the population of official training
charts (z of each style statistic), and each chart's repeated sections are
scored against each other.

    python -m tools.eval_ghl --charts data/ghl-official -n 10
"""
import argparse
import itertools
import json
import pickle
import random
import re
import sys
import tempfile
from collections import Counter, defaultdict
from itertools import groupby, pairwise
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from chartgen import ghl, stems
from chartgen.__main__ import build_args
from chartgen import pipeline
from tools.build_ghl_dataset import mix
from tools.ghl_charts import DIFF_BASE, GhlChart, load
from tools.train_ghl import ini_delay_s, song_key

MATCH_S = 0.05
DIFFS = tuple(DIFF_BASE)
ROW_DIFFS = ("Expert", "Hard")
BAR_SLOTS, REPEAT_BARS = 48, 8
SHUFFLES = 20
FLAG_Z = 2.0
SHAPE_KEYS = ("exact", "size", "row", "direction")


def read_ghl(chart_text: str) -> dict[str, list[tuple[float, frozenset, float]]]:
    """{difficulty: [(seconds, shape, sustain seconds)]} from a written notes.chart."""
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
        notes: dict[int, tuple[set, int]] = {}
        for tick, number, length in re.findall(r"(\d+) = N (\d+) (\d+)", block.group(1) if block else ""):
            tick, number, length = int(tick), int(number), int(length)
            if number in lanes or number == 7:
                buttons, sustain = notes.setdefault(tick, (set(), 0))
                if number in lanes:
                    buttons.add(lanes[number])
                notes[tick] = (buttons, max(sustain, length))
        out[diff] = [(seconds(t), frozenset(b), seconds(t + n) - seconds(t))
                     for t, (b, n) in sorted(notes.items())]
    return out


def timed(chart: GhlChart, diff: str, delay: float = 0.0) -> list[tuple[float, frozenset, float]]:
    return [(chart.seconds(t) + delay, s, chart.seconds(t + n) - chart.seconds(t))
            for t, s, n in chart.tiers[diff]]


def column(shape) -> float:
    return np.mean([b % 3 for b in shape]) if shape else -1.0


def score(ours, theirs) -> dict:
    """Greedy one-to-one matching in time order, then shape and sustain agreement."""
    their_s = np.array([s for s, *_ in theirs])
    used = np.zeros(len(theirs), bool)
    pairs = []
    for s, shape, sustain in ours:
        j = int(np.searchsorted(their_s, s))
        best = min((k for k in (j - 1, j) if 0 <= k < len(theirs) and not used[k]),
                   key=lambda k: abs(their_s[k] - s), default=None)
        if best is not None and abs(their_s[best] - s) <= MATCH_S:
            used[best] = True
            pairs.append((shape, theirs[best][1], sustain > 0, theirs[best][2] > 0))
    moves = [(np.sign(column(a2) - column(a1)), np.sign(column(b2) - column(b1)))
             for (a1, b1, *_), (a2, b2, *_) in zip(pairs, pairs[1:], strict=False)]
    return {"ours": len(ours), "theirs": len(theirs), "matched": len(pairs),
            "exact": sum(a == b for a, b, *_ in pairs),
            "size": sum(len(a) == len(b) for a, b, *_ in pairs),
            "row": sum((min(a, default=-1) >= 3) == (min(b, default=-1) >= 3) for a, b, *_ in pairs),
            "row_rest": sum(min(b, default=-1) < 3 for _, b, *_ in pairs),
            "direction": sum(a == b for a, b in moves), "moves": len(moves),
            "sustained": sum(x and y for *_, x, y in pairs),
            "ours_sustained": sum(x for *_, x, _ in pairs),
            "theirs_sustained": sum(y for *_, y in pairs)}


def chance(theirs) -> dict:
    """Mean score of the official shapes shuffled over the official positions."""
    rng = np.random.default_rng(0)
    runs = [score([(s, theirs[k][1], n) for (s, _, n), k in zip(theirs, rng.permutation(len(theirs)), strict=True)],
                  theirs) for _ in range(SHUFFLES)]
    return {key: float(np.mean([r[key] for r in runs])) for key in runs[0]}


def panel(notes) -> dict[str, float]:
    """Style statistics of one tier from seconds only, so a different tempo grid cannot move them."""
    shapes = [shape for _, shape, _ in notes]
    share = np.array(list(Counter(shapes).values())) / len(shapes)
    return {"density": (len(notes) - 1) / max(notes[-1][0] - notes[0][0], 1e-9),
            "single": float(np.mean([len(s) == 1 for s in shapes])),
            "two": float(np.mean([len(s) == 2 for s in shapes])),
            "white": float(np.mean([bool(s) and max(s) < 3 for s in shapes])),
            "mixed": float(np.mean([bool(s) and min(s) < 3 <= max(s) for s in shapes])),
            "repeat": float(np.mean([a == b for a, b in zip(shapes, shapes[1:], strict=False)])),
            "sustain": float(np.mean([n > 0 for *_, n in notes])),
            "entropy": float(-(share * np.log2(share)).sum())}


def row_style(notes) -> dict[str, float]:
    """Counts behind the row style of one tier: switches between the white
    row, the black row and mixed shapes from one fretted note to the next,
    the row runs, and how many singles sit on the black row."""
    rows = [0 if max(s) < 3 else 1 if min(s) >= 3 else 2 for _, s, _ in notes if s]
    runs = [len(list(group)) for _, group in groupby(rows)]
    singles = [s for _, s, _ in notes if len(s) == 1]
    return {"switches": sum(a != b for a, b in pairwise(rows)), "pairs": max(len(rows) - 1, 0),
            "runs": runs, "singles": len(singles), "black_singles": sum(min(s) >= 3 for s in singles),
            "notes": len(notes)}


def row_line(t: dict) -> str:
    return (f"switch {t['switches'] / max(t['pairs'], 1):.0%}, run median {np.median(t['runs']) if t['runs'] else 0:g}, "
            f"black singles {t['black_singles'] / max(t['singles'], 1):.0%}, singles {t['singles'] / max(t['notes'], 1):.0%}")


def population(features: Path, stats_path: Path, holdout: set[str]) -> dict:
    """Mean and sd of the Expert panel over the official training charts, one chart per song key."""
    if stats_path.exists():
        return json.loads(stats_path.read_text())
    seen, rows = set(), []
    for path in sorted(features.glob("*.pkl")):
        key = song_key(path.stem)
        if key not in holdout and key not in seen:
            seen.add(key)
            rows.append(panel(timed(pickle.loads(path.read_bytes())["chart"], "Expert")))
    stats = {"songs": len(rows),
             "mean": {k: float(np.mean([r[k] for r in rows])) for k in rows[0]},
             "sd": {k: float(np.std([r[k] for r in rows])) for k in rows[0]}}
    stats_path.write_text(json.dumps(stats, indent=1))
    return stats


def zscores(notes, stats) -> dict[str, float]:
    return {k: (v - stats["mean"][k]) / max(stats["sd"][k], 1e-9) for k, v in panel(notes).items()}


def repeats(chart: GhlChart, delay: float) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """Pairs of same-named sections (numbers ignored) of equal tick length, as spans in seconds."""
    groups = defaultdict(list)
    for (start, name), (end, _) in zip(chart.sections, chart.sections[1:], strict=False):
        key = (" ".join(re.sub(r"\d+", "", name).split()).lower(), end - start)
        groups[key].append((chart.seconds(start) + delay, chart.seconds(end) + delay))
    return [pair for spans in groups.values() for pair in itertools.combinations(spans, 2)]


def bar_patterns(notes, bars: list[float]) -> list[tuple]:
    """Per official bar (start seconds), its notes as (48th of the bar, shape)."""
    out = []
    for start, stop in pairwise(bars):
        out.append(tuple((round((s - start) / (stop - start) * BAR_SLOTS), shape)
                         for s, shape, _ in notes if start <= s < stop))
    return out


def same_bar(a: tuple, b: tuple) -> bool:
    """Same notes and shapes, each within one BAR_SLOTS slot: the tracked
    beat grid jitters by more than half a slot."""
    return len(a) == len(b) and all(abs(x - y) <= 1 and s == t for (x, s), (y, t) in zip(a, b, strict=True))


def bar_repeats(ours, theirs, bars: list[float]) -> dict[str, int]:
    """How often a bar repeats one of the REPEAT_BARS bars before it note for
    note, in each chart, and how often chartgen repeats where the official
    chart does. A riff the charter plays four times reads as one figure."""
    mine, official = bar_patterns(ours, bars), bar_patterns(theirs, bars)
    out = {"bars": 0, "official": 0, "chartgen": 0, "kept": 0}
    for j in range(1, len(bars) - 1):
        earlier = range(max(0, j - REPEAT_BARS), j)
        if len(official[j]) < 2:
            continue
        out["bars"] += 1
        source = next((i for i in reversed(earlier) if same_bar(official[i], official[j])), None)
        out["official"] += source is not None
        out["chartgen"] += any(same_bar(mine[i], mine[j]) for i in earlier) and len(mine[j]) > 1
        out["kept"] += source is not None and same_bar(mine[source], mine[j]) and len(mine[j]) > 1
    return out


def self_precision(notes, pairs) -> list[float]:
    """Per section pair, the share of the later repeat's notes that the earlier repeat also has."""
    def window(span):
        return [(s - span[0], shape, n) for s, shape, n in notes if span[0] <= s < span[1]]
    out = []
    for first, later in pairs:
        stats = score(window(later), window(first))
        if stats["ours"]:
            out.append(stats["matched"] / stats["ours"])
    return out


def violations(tiers) -> list[str]:
    out = []
    if any(b >= 3 for _, s, _ in tiers["Easy"] for b in s):
        out.append("Easy uses the black row")
    if any(len(s) > 2 for _, s, _ in tiers["Medium"]):
        out.append("Medium has 3-button chords")
    expert = {round(s, 3) for s, *_ in tiers["Expert"]}
    if any(round(s, 3) not in expert for d in DIFFS[1:] for s, *_ in tiers[d]):
        out.append("a lower tier has a position Expert lacks")
    return out


def rates(t: dict) -> dict[str, float]:
    m = max(t.get("matched", 0), 1)
    return {"exact": t["exact"] / m, "size": t["size"] / m, "row": t["row"] / m,
            "direction": t["direction"] / max(t["moves"], 1)}


def versus(got: dict, base: dict) -> str:
    ours, luck = rates(got), rates(base)
    return ", ".join(f"{k} {ours[k]:.0%} chance {luck[k]:.0%} ({(ours[k] - luck[k]) * 100:+.0f})"
                     for k in SHAPE_KEYS)


def sustain_pr(t: dict) -> str:
    return (f"sustain precision {t['sustained'] / max(t['ours_sustained'], 1):.0%} "
            f"recall {t['sustained'] / max(t['theirs_sustained'], 1):.0%}")


def median(values) -> str:
    return f"{np.median(values):.0%}" if values else "n/a"


def zline(z: dict[str, dict[str, float]]) -> str:
    def cell(value):
        return f"{value:+.1f}{'*' if abs(value) > FLAG_Z else ''}"
    return " ".join(f"{k} {cell(z['chartgen'][k])}/{cell(z['official'][k])}" for k in z["chartgen"])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--charts", type=Path, default=Path("data/ghl-official"))
    ap.add_argument("-n", type=int, default=10, help="holdout songs to chart, 0 for every held-out song")
    ap.add_argument("--seed", type=int, help="pick the songs at random with this seed instead of by name")
    ap.add_argument("--out", type=Path, default=Path("data/ghl-eval"))
    ap.add_argument("--features", type=Path, default=Path("data/ghl-features"),
                    help="official records the style population is built from")
    ap.add_argument("--stats", type=Path, default=Path("data/ghl-eval-stats.json"),
                    help="cached style population; built from --features when missing")
    args = ap.parse_args(argv)

    holdout = set(torch.load(ghl.MODEL_PATH, map_location="cpu")["holdout"])
    population_stats = population(args.features, args.stats, holdout)
    by_key = {}
    for folder in sorted(p.parent for p in args.charts.rglob("notes.mid") if song_key(p.parent.name) in holdout):
        by_key.setdefault(song_key(folder.name), folder)
    folders = list(by_key.values())
    if args.seed is not None:
        random.Random(args.seed).shuffle(folders)
    folders = folders[:args.n or None]
    style = {who: {d: {} for d in ROW_DIFFS} for who in ("chartgen", "official")}
    bar_totals = {d: {} for d in ROW_DIFFS}
    totals = {d: {} for d in DIFFS}
    flags = {"chartgen": [], "official": []}
    consistency = {"chartgen": [], "official": []}
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
        theirs = {d: timed(official, d, delay) for d in DIFFS}
        problems = violations(ours)
        line, details = [], []
        for d in DIFFS:
            got, base = score(ours[d], theirs[d]), chance(theirs[d])
            for key, value in got.items():
                totals[d][key] = totals[d].get(key, 0) + value
                totals[d]["chance " + key] = totals[d].get("chance " + key, 0) + base[key]
            line.append(f"{d[0]} {got['ours']}/{got['theirs']}")
            details.append(f"  {d:6s} {versus(got, base)}; {sustain_pr(got)}")
        print(f"{folder.name[:44]:44s} {' '.join(line)} {'; '.join(problems) or 'rules ok'}")
        print("\n".join(details))
        z = {"chartgen": zscores(ours["Expert"], population_stats),
             "official": zscores(theirs["Expert"], population_stats)}
        for who in flags:
            flags[who].append(sum(abs(v) > FLAG_Z for v in z[who].values()))
        print(f"  style z chartgen/official: {zline(z)}")
        bars = [official.seconds(t) + delay for t in official.bar_lines(max(t for t, *_ in official.tiers["Expert"]))]
        for d in ROW_DIFFS:
            for key, value in bar_repeats(ours[d], theirs[d], bars).items():
                bar_totals[d][key] = bar_totals[d].get(key, 0) + value
        for who, tiers in (("chartgen", ours), ("official", theirs)):
            for d in ROW_DIFFS:
                for key, value in row_style(tiers[d]).items():
                    style[who][d][key] = style[who][d].get(key, 0 if key != "runs" else []) + value
        pairs = repeats(official, delay)
        own = {"chartgen": self_precision(ours["Expert"], pairs),
               "official": self_precision(theirs["Expert"], pairs)}
        for who in consistency:
            consistency[who] += own[who]
        print(f"  repeats ({len(pairs)} section pairs) median precision "
              f"chartgen {median(own['chartgen'])} official {median(own['official'])}", flush=True)

    print()
    for d in DIFFS:
        t = totals[d]
        m = max(t.get("matched", 0), 1)
        print(f"{d:6s} notes {t['ours'] / max(t['theirs'], 1):.0%} of official, "
              f"precision {t['matched'] / max(t['ours'], 1):.0%} recall {t['matched'] / max(t['theirs'], 1):.0%}; "
              f"on matches: exact {t['exact'] / m:.0%}, size {t['size'] / m:.0%}, "
              f"row {t['row'] / m:.0%}, direction {t['direction'] / max(t['moves'], 1):.0%}")
        base = {k.removeprefix("chance "): v for k, v in t.items() if k.startswith("chance ")}
        print(f"       {versus(t, base)}; {sustain_pr(t)}")
    for d in ROW_DIFFS:
        t = totals[d]
        print(f"{d:6s} row on matches {t['row'] / max(t['matched'], 1):.0%}, all non-black {t['row_rest'] / max(t['matched'], 1):.0%}; "
              f"style chartgen {row_line(style['chartgen'][d])} | official {row_line(style['official'][d])}")
    for d in ROW_DIFFS:
        t = bar_totals[d]
        print(f"{d:6s} bars repeating one of the {REPEAT_BARS} before: official {t['official'] / max(t['bars'], 1):.0%}, "
              f"chartgen {t['chartgen'] / max(t['bars'], 1):.0%}; where official repeats, chartgen repeats the same bar "
              f"{t['kept'] / max(t['official'], 1):.0%}")
    songs = max(len(flags["chartgen"]), 1)
    print(f"style flags (Expert, |z| > {FLAG_Z:g} against {population_stats['songs']} official training songs): "
          + "; ".join(f"{who} >= 1 {sum(n >= 1 for n in counts) / songs:.0%}, "
                      f">= 2 {sum(n >= 2 for n in counts) / songs:.0%}" for who, counts in flags.items()))
    print(f"repeats ({len(consistency['official'])} section pairs) median precision "
          f"chartgen {median(consistency['chartgen'])} official {median(consistency['official'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
