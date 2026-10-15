"""Score generated charts against the human charts on held-out songs.

    python tools/evaluate.py --val data/val.json --outdir out/eval
    python tools/evaluate.py --val data/val.json --outdir out/eval --skip-generate

Parses both .chart files itself (tick -> seconds through each chart's own
SyncTrack), so human and generated notes go through identical conversion.
Metrics per song, then medians:

    onset F1   how well our note times match the human chart's (tolerance ms)
    fret agree of the onsets that matched, how often the lane is the same
               (singles only; chords compared as root lane)
    contour    sign agreement of fret motion on matched onsets: "did we move
               the same direction the charter did", which is what following
               the music feels like even when absolute lanes differ

The human chart is a stylistic reference, not ground truth: two humans chart
the same song differently. Trends across songs matter; single numbers don't.

Caveat: Harmonix-style multitrack folders ship a near-silent residual
`song.opus` next to instrument stems; sum the stems into a real mix before
pointing this at them.
"""
import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def chart_note_times(chart_path: str, section: str = "ExpertSingle"):
    """[(seconds, lanes tuple)] for one difficulty of a .chart file."""
    text = Path(chart_path).read_text(encoding="utf-8", errors="replace")
    blocks = {m.group(1): m.group(2)
              for m in re.finditer(r"\[(\w+)\]\s*\{(.*?)\n\}", text, re.S)}
    res = 192
    for line in blocks.get("Song", "").splitlines():
        if line.strip().startswith("Resolution"):
            res = int(line.split("=", 1)[1].strip())
    sync = sorted(
        (int(m.group(1)), int(m.group(2)) / 1000.0)
        for m in (re.match(r"\s*(\d+)\s*=\s*B\s+(\d+)", l)
                  for l in blocks.get("SyncTrack", "").splitlines())
        if m)
    if not sync or sync[0][0] != 0:
        sync.insert(0, (0, sync[0][1] if sync else 120.0))
    anchors, t = [], 0.0
    for i, (tick, bpm) in enumerate(sync):
        if i:
            ptick, pbpm = sync[i - 1]
            t += (tick - ptick) / res * 60.0 / pbpm
        anchors.append((tick, t, bpm))

    def seconds(tick: int) -> float:
        lo, hi = 0, len(anchors) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if anchors[mid][0] <= tick:
                lo = mid
            else:
                hi = mid - 1
        a_tick, a_t, a_bpm = anchors[lo]
        return a_t + (tick - a_tick) / res * 60.0 / a_bpm

    by_tick: dict[int, set] = {}
    for line in blocks.get(section, "").splitlines():
        m = re.match(r"\s*(\d+)\s*=\s*N\s+(\d+)\s+\d+", line)
        if m and int(m.group(2)) in (0, 1, 2, 3, 4, 7):
            by_tick.setdefault(int(m.group(1)), set()).add(int(m.group(2)))
    return [(seconds(t), tuple(sorted(by_tick[t]))) for t in sorted(by_tick)]


def match_onsets(ours, theirs, tol_s: float):
    """Greedy 1:1 nearest-neighbour matching inside the tolerance."""
    matches, used = [], set()
    j = 0
    for i, (t, _) in enumerate(ours):
        while j < len(theirs) and theirs[j][0] < t - tol_s:
            j += 1
        best, best_d = None, tol_s
        k = j
        while k < len(theirs) and theirs[k][0] <= t + tol_s:
            d = abs(theirs[k][0] - t)
            if k not in used and d <= best_d:
                best, best_d = k, d
            k += 1
        if best is not None:
            used.add(best)
            matches.append((i, best))
    return matches


def root(lanes):
    fretted = [l for l in lanes if l != 7]
    return min(fretted) if fretted else 7


def evaluate_song(gen_chart: Path, human_chart: Path, tol_s: float):
    ours = chart_note_times(str(gen_chart))
    theirs = chart_note_times(str(human_chart))
    if not ours or not theirs:
        return None
    matches = match_onsets(ours, theirs, tol_s)
    precision = len(matches) / len(ours)
    recall = len(matches) / len(theirs)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    agree = moves = same_dir = 0
    prev = None
    for i, k in matches:
        a, b = root(ours[i][1]), root(theirs[k][1])
        if a == b:
            agree += 1
        if prev is not None:
            pa, pb = prev
            da, db = a - pa, b - pb
            if da or db:
                moves += 1
                if (da > 0) == (db > 0) and (da < 0) == (db < 0):
                    same_dir += 1
        prev = (a, b)
    return {
        "gen_notes": len(ours), "human_notes": len(theirs),
        "precision": precision, "recall": recall, "f1": f1,
        "fret_agree": agree / len(matches) if matches else 0.0,
        "contour": same_dir / moves if moves else 0.0,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--val", type=Path, default=Path("data/val.json"))
    ap.add_argument("--outdir", type=Path, default=Path("out/eval"))
    ap.add_argument("--tolerance-ms", type=float, default=70.0)
    ap.add_argument("--limit", type=int, default=0, help="cap song count (0 = all)")
    ap.add_argument("--skip-generate", action="store_true",
                    help="re-score existing charts in --outdir")
    args = ap.parse_args()

    entries = json.loads(args.val.read_text(encoding="utf-8"))
    # One row per song (val.json has one row per tier).
    songs = {}
    for e in entries:
        songs.setdefault(e["chart_path"], e)
    rows = list(songs.values())
    if args.limit:
        rows = rows[:args.limit]

    from chartgen import pipeline
    from chartgen.__main__ import build_args

    results = []
    for e in rows:
        folder = Path(e["chart_path"]).parent
        name = folder.name
        gen_dir = args.outdir / f"eval - {name}"[:80]
        gen_chart = gen_dir / "notes.chart"
        if not args.skip_generate or not gen_chart.is_file():
            # The CLI defaults, with the decorations that do not affect Expert
            # note placement switched off so every run measures the same thing.
            opts, _ = build_args([str(e["audio_path"]), "-o", str(args.outdir),
                                  "--name", f"eval - {name}"[:76], "--artist", "",
                                  "--no-star-power", "--no-sections",
                                  "--no-lyrics", "--density", "model"])
            opts.audio = opts.audio[0]
            try:
                result = pipeline.run(opts, progress=lambda m: None)
            except Exception as error:
                print(f"SKIP {name}: {type(error).__name__}: {error}")
                continue
            gen_chart = Path(result["song_dir"]) / "notes.chart"
        try:
            row = evaluate_song(gen_chart, Path(e["chart_path"]),
                                args.tolerance_ms / 1000.0)
        except Exception as error:
            print(f"SKIP {name} (parse): {type(error).__name__}: {error}")
            continue
        if row:
            row["name"] = name[:40]
            results.append(row)
            print(f"{row['name']:<42} F1 {row['f1']:.2f}  "
                  f"fret {row['fret_agree']:.2f}  contour {row['contour']:.2f}  "
                  f"({row['gen_notes']}/{row['human_notes']} notes)")

    if not results:
        print("no results")
        return 1

    def med(key):
        vals = sorted(r[key] for r in results)
        return vals[len(vals) // 2]

    print(f"\n{len(results)} songs — medians: "
          f"F1 {med('f1'):.2f}  precision {med('precision'):.2f}  "
          f"recall {med('recall'):.2f}  fret-agree {med('fret_agree'):.2f}  "
          f"contour {med('contour'):.2f}")
    (args.outdir / "results.json").parent.mkdir(parents=True, exist_ok=True)
    (args.outdir / "results.json").write_text(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
