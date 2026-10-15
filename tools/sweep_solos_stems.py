"""Re-sweep the solo rule with Demucs stem evidence joined in.

The full-mix rule plateaued at ~75% precision / ~5% recall, and the
missed solos were invisible to full-mix features by any threshold. Stems
add the two signals that were missing:

- singing: true share of the section where the vocal stem is active
  (replaces lyric-word counting, which calibration could never exercise
  because human charts rarely embed lyrics);
- lead: the "other" stem's energy as a per-song z-score - the lead
  instrument surging even when the full mix stays flat.

The position gate is deliberately re-opened in this sweep: it existed
because full-mix features could not tell an intro riff from a solo, and
Mary Jane's Last Dance - whose human chart marks solos at ~31% and ~85%
of the song - is the named acceptance case.

    python tools/sweep_solos_stems.py
"""
import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sweep_solos import zscores, iou

WORK = Path(__file__).resolve().parent.parent / "work"
FEATURES = ("voiced", "confidence", "spread", "bright", "novelty")


def load_songs(stems_tag: str = ""):
    """stems_tag selects which stem-feature dump joins in: "" for the
    Demucs-era cal_stems_A/B.json, "sw" for cal_stems_A_sw/B_sw.json -
    the same sections and ground truth, only the separator differs, which
    is what makes a recalibration under a new backend a controlled test."""
    # A and B are the solo-having sets; Q is the 100-song solo-less control
    # set (built from the frame cache's quiet flags), present only once
    # tools/dump_solo_features.py has run on it. Without Q every rule
    # trivially shows zero quiet fires - the bar cannot be evaluated.
    songs = [s for name in ("cal_solo_A.json", "cal_solo_B.json", "cal_solo_Q.json")
             if (WORK / name).exists()
             for s in json.loads((WORK / name).read_text(encoding="utf-8"))]
    songs = list({s["name"]: s for s in songs}.values())
    stems = {}
    suffix = f"_{stems_tag}" if stems_tag else ""
    for name in (f"cal_stems_A{suffix}.json", f"cal_stems_B{suffix}.json",
                 f"cal_stems_Q{suffix}.json"):
        path = WORK / name
        if path.exists():
            for row in json.loads(path.read_text(encoding="utf-8")):
                stems[row["name"]] = row["stem_sections"]
    joined = []
    for song in songs:
        st = stems.get(song["name"])
        if not st or len(st) != len(song["sections"]):
            continue
        for section, extra in zip(song["sections"], st, strict=True):
            section["stem"] = extra
        if zscores(song["sections"]):
            joined.append(song)
    return joined


def candidates(song, max_run=3):
    sections = [s for s in song["sections"] if s.get("z") and s.get("stem")]
    out = []
    for i in range(len(sections)):
        for run in range(1, max_run + 1):
            group = sections[i:i + run]
            if len(group) < run:
                break
            weights = [g["end"] - g["start"] for g in group]
            total = sum(weights) or 1
            out.append({
                "start": group[0]["start"], "end": group[-1]["end"],
                "notes": sum(g["notes"] for g in group),
                "z": {k: sum(g["z"][k] * w for g, w in zip(group, weights, strict=True)) / total
                      for k in FEATURES},
                "singing": sum(g["stem"]["singing"] * w
                               for g, w in zip(group, weights, strict=True)) / total,
                "lead": sum(g["stem"]["lead"] * w
                            for g, w in zip(group, weights, strict=True)) / total,
                # Guitar-stem evidence (SW dumps only; 0 under Demucs):
                # the featured-line-vs-accompaniment signal earlier sweeps
                # lacked. z = surge against the song's own guitar baseline.
                "guitar": sum(g["stem"].get("guitar", 0.0) * w
                              for g, w in zip(group, weights, strict=True)) / total,
                "guitar_share": sum(g["stem"].get("guitar_share", 0.0) * w
                                    for g, w in zip(group, weights, strict=True)) / total,
            })
    return out


def evaluate(songs, weights, lead_w, max_sing, min_beats, pos_lo, pos_hi,
             threshold, top_k, min_iou=0.25, guitar_w=0.0, min_guitar=0.0):
    tp = fp = fn = 0
    quiet = 0  # solo-less songs that still got a solo: the bar that killed
    quiet_total = 0  # every earlier rule above 2% recall
    for song in songs:
        res, last = song["resolution"], song["last_tick"]
        picks = []
        for s in candidates(song):
            beats = (s["end"] - s["start"]) / res
            if not (min_beats <= beats <= 128):
                continue
            if not (pos_lo <= s["start"] / last <= pos_hi):
                continue
            if s["singing"] > max_sing or s["notes"] < 16:
                continue
            if s["guitar_share"] < min_guitar:
                continue
            score = sum(weights[k] * s["z"][k] for k in FEATURES) \
                + lead_w * s["lead"] + guitar_w * s["guitar"]
            if score >= threshold:
                picks.append((score, s["start"], s["end"]))
        picks.sort(reverse=True)
        chosen = []
        for _, a, b in picks:
            if all(b < c or a > d for c, d in chosen):
                chosen.append((a, b))
            if len(chosen) >= top_k:
                break
        truth = [tuple(t) for t in song["truth"]]
        if not truth:
            quiet_total += 1
            if chosen:
                quiet += 1
        hits = [g for g in chosen if any(iou(g, t) >= min_iou for t in truth)]
        tp += len(hits)
        fp += len(chosen) - len(hits)
        fn += sum(1 for t in truth if not any(iou(g, t) >= min_iou for g in chosen))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f_half = (1.25 * precision * recall / (0.25 * precision + recall)) \
        if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f_half": f_half,
            "tp": tp, "fp": fp, "quiet": quiet, "quiet_total": quiet_total}


def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stems-tag", default="",
                    help='stem dump variant to join ("" = Demucs, "sw")')
    args = ap.parse_args(argv)
    songs = load_songs(args.stems_tag)
    print(f"stems: {args.stems_tag or 'demucs'}")
    print(f"{len(songs)} songs with stem features, "
          f"{sum(len(s['truth']) for s in songs)} solos\n")

    base = {"voiced": 0.5, "confidence": 0, "spread": 1.0, "bright": 0.0,
            "novelty": 1.0}
    has_guitar = any("guitar" in sec.get("stem", {})
                     for song in songs for sec in song["sections"])
    guitar_grid = (list(itertools.product((0, 1.0, 1.5, 2.0, 3.0, 4.0), (0.0, 0.10)))
                   if has_guitar else [(0.0, 0.0)])
    print(f"guitar features: {'yes' if has_guitar else 'no'} "
          f"({len(guitar_grid)} guitar settings)")
    rows = []
    for lead_w, max_sing in itertools.product((0, 0.5, 1.0, 1.5), (0.15, 0.25, 0.4, 1.0)):
        for pos in ((0.45, 0.80), (0.25, 0.92), (0.10, 0.95)):
            for z, k, mb in itertools.product((1.0, 1.25, 1.5, 2.0), (1, 2), (32, 48)):
                for gw, gsh in guitar_grid:
                    r = evaluate(songs, base, lead_w, max_sing, mb, pos[0],
                                 pos[1], z, k, guitar_w=gw, min_guitar=gsh)
                    r["rule"] = (f"lead{lead_w} sing<={max_sing} pos{pos[0]}-{pos[1]} "
                                 f"beats>={mb} z>={z} top{k}"
                                 + (f" g{gw} gsh>={gsh}" if has_guitar else ""))
                    rows.append(r)

    def show(r):
        print(f"  {r['rule']:<64}{r['precision']:>5.0%}{r['recall']:>6.0%}  "
              f"(tp{r['tp']} fp{r['fp']} quiet{r['quiet']}/{r['quiet_total']})")

    rows.sort(key=lambda r: -r["f_half"])
    print("top by F0.5:")
    for r in rows[:8]:
        show(r)
    print()
    print("precision-first (recall >= 8%):")
    good = [r for r in rows if r["recall"] >= 0.08]
    for r in sorted(good, key=lambda r: (-r["precision"], -r["recall"]))[:8]:
        show(r)
    print()
    print("zero-quiet-fire rules by recall (the shipping bar):")
    clean = [r for r in rows if r["quiet"] == 0 and r["tp"] > 0]
    for r in sorted(clean, key=lambda r: (-r["recall"], -r["precision"]))[:8]:
        show(r)
    if not clean:
        print("  (none)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
