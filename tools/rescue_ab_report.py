"""Keyed vs old rescue, judged inside the windows keyed rescue fired in.

Playtesting cannot tell the two apart when the rescued passages are a
minority of the chart and unlabeled. This scores the PROM (keyed) and
OLDRESCUE charts against the HUMAN chart of the same song, restricted to
the 4-bar windows where keyed rescue fired, on what a player feels:

  * density: our notes per human note in the window
  * onset F1 at 80 ms: right moments
  * contour: Spearman correlation between our lowest lane and the human's
    lowest lane over matched onsets - the wrong instrument (bass under a
    guitar line) gives the right moments with the wrong shape
  * chord share and open share vs the human's
  * for the OLD rule only: which stem its rescued notes really came from,
    against the timeline's followed stem for that window

Reads the per-song cache written by tools/keyed_rescue_eval.py and the
three chart folders under out/prom.

    python tools/rescue_ab_report.py
"""
import pickle
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from chartgen import prominence  # noqa: E402
from chartgen.tempo import TempoMap  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "work" / "keyed_cache"
PROM = ROOT / "out" / "prom"
CH = Path.home() / ".clonehero/Songs"
SONGS = [
    ("Perturbator - Sexualizer", "Perturbator - Sexualizer PROM [chartgen]",
     "Perturbator - Sexualizer OLDRESCUE [chartgen]", "Perturbator - Sexualizer HUMAN"),
    ("Voyager - Reconnected", "Voyager - Reconnected PROM [chartgen]",
     "Voyager - Reconnected OLDRESCUE [chartgen]", "Voyager - Reconnected HUMAN"),
    ("Haste the Day - Gnaw", "Haste the Day - Gnaw PROM [chartgen]",
     "Haste the Day - Gnaw OLDRESCUE [chartgen]", "Haste the Day - Gnaw HUMAN"),
]
TOL = 0.08
BPM_RE = re.compile(r"(\d+)\s*=\s*B\s*(\d+)")


def parse_chart(path: Path):
    """[(time_s, lanes_set)] per position, via the chart's own tempo map."""
    txt = path.read_text(encoding="utf-8-sig", errors="replace")
    res = int(re.search(r"Resolution\s*=\s*(\d+)", txt).group(1))
    bpms = [(int(a), int(b) / 1000.0) for a, b in
            BPM_RE.findall(re.search(r"\[SyncTrack\]\s*\{(.*?)\}", txt, re.S).group(1))] or [(0, 120.0)]

    def t2s(tick):
        sec, lt, lb = 0.0, 0, bpms[0][1]
        for t, b in bpms:
            if t >= tick:
                break
            sec += (t - lt) / res * 60.0 / lb
            lt, lb = t, b
        return sec + (tick - lt) / res * 60.0 / lb

    body = re.search(r"\[ExpertSingle\]\s*\{(.*?)\}", txt, re.S).group(1)
    by = {}
    for t, l in re.findall(r"(\d+)\s*=\s*N\s*(\d+)\s+\d+", body):
        if int(l) <= 4 or int(l) == 7:
            by.setdefault(int(t), set()).add(int(l))
    return [(t2s(t), by[t]) for t in sorted(by)]


def in_span(notes, t0, t1):
    return [(t, ls) for t, ls in notes if t0 <= t < t1]


def match(ours, human):
    """greedy 1:1 within TOL -> list of (i, j)"""
    h = [t for t, _ in human]
    used = set()
    pairs = []
    for i, (t, _) in enumerate(ours):
        best, bd = None, TOL + 1
        for j, ht in enumerate(h):
            if j in used:
                continue
            d = abs(ht - t)
            if d < bd:
                best, bd = j, d
        if best is not None and bd <= TOL:
            used.add(best)
            pairs.append((i, best))
    return pairs


def lowest(ls):
    fr = [l for l in ls if l <= 4]
    return min(fr) if fr else -1


def spearman(a, b):
    if len(a) < 4:
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def window_stats(ours, human):
    if not human or not ours:
        return None
    pairs = match(ours, human)
    p = len(pairs) / len(ours)
    r = len(pairs) / len(human)
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    rho = spearman([lowest(ours[i][1]) for i, _ in pairs], [lowest(human[j][1]) for _, j in pairs])
    chord = lambda ns: sum(1 for _, ls in ns if len([l for l in ls if l <= 4]) >= 2) / len(ns)
    opens = lambda ns: sum(1 for _, ls in ns if ls == {7}) / len(ns)
    return {"density": len(ours) / len(human), "f1": f1, "rho": rho,
            "chord": chord(ours), "hchord": chord(human), "open": opens(ours), "hopen": opens(human)}


def main():
    for label, prom, old, human in SONGS:
        cache = next((p for p in CACHE.glob("*.pkl") if p.name.startswith(label.split(" - ")[0][:12])), None)
        if cache is None:
            print(f"{label}: no cache")
            continue
        c = pickle.loads(cache.read_bytes())
        tm = TempoMap(beat_times=c["beat_times"], pickup_beats=c["pickup"])
        # the windows keyed rescue fires in, under the shipped rule
        fired = []
        for w in c["windows"] or []:
            one = [w]
            extra, touched, _ = prominence.keyed_rescue_events(c["events"], one, c["stem_events"], tm)
            if extra:
                fired.append((w["t0"], w["t1"], w["stem"], len(extra)))
        charts = {}
        for key, folder in (("keyed", prom), ("old", old), ("human", human)):
            path = (PROM / folder / "notes.chart") if key != "human" else (CH / folder / "notes.chart")
            if not path.is_file():
                path = CH / folder / "notes.chart"
            charts[key] = parse_chart(path)
        print(f"\n=== {label}: keyed rescue fired in {len(fired)} window(s) ===")
        print("  windows: " + ", ".join(f"{t0:.0f}-{t1:.0f}s {st or '?'}({n})" for t0, t1, st, n in fired))
        agg = {"keyed": [], "old": []}
        for t0, t1, st, n in fired:
            hn = in_span(charts["human"], t0, t1)
            for key in ("keyed", "old"):
                s = window_stats(in_span(charts[key], t0, t1), hn)
                if s:
                    agg[key].append(s)
        print(f"  {'':8s} {'notes/human':>11s} {'onset F1':>9s} {'contour rho':>11s} {'chords':>13s} {'opens':>13s}")
        for key in ("keyed", "old"):
            a = agg[key]
            if not a:
                print(f"  {key:8s} (no overlapping notes)")
                continue
            med = lambda k: np.nanmedian([x[k] for x in a])
            print(f"  {key:8s} {med('density'):11.2f} {med('f1'):9.2f} {med('rho'):11.2f} "
                  f"{med('chord'):5.0%} vs {med('hchord'):4.0%} {med('open'):5.0%} vs {med('hopen'):4.0%}")
        # where did the OLD rule's rescued notes come from?
        if c["old"]:
            attribution = {}
            for t0, t1, st, n in fired:
                for s0, _, _, _ in c["old"]:
                    if not t0 <= s0 < t1:
                        continue
                    src = None
                    for stem, evs in c["stem_events"].items():
                        if any(abs(e[0] - s0) <= 0.03 for e in evs):
                            src = stem if src is None else src
                    attribution.setdefault(st, {}).setdefault(src or "none", 0)
                    attribution[st][src or "none"] += 1
            for st, d in attribution.items():
                total = sum(d.values())
                print(f"  old rule's notes in {st}-followed windows came from: "
                      + ", ".join(f"{k} {v / total:.0%}" for k, v in sorted(d.items(), key=lambda kv: -kv[1])))
    return 0


if __name__ == "__main__":
    sys.exit(main())
