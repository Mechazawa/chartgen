"""Cache per-section audio evidence for the human library, once.

Feature extraction costs ~30s a song, and picking thresholds needs dozens of
passes over the same songs. So this does the slow half once and writes it to
JSON; sweep_solos.py then tunes rules against the cache in a second, which is
the only way to choose thresholds on evidence rather than on the first
plausible guess.

    python tools/dump_solo_features.py ~/.clonehero/Songs -o work/solo_features.json
"""
import argparse
import json
import random
import re
import sys
import time
from itertools import pairwise
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from study_solos import is_generated, blocks, solo_spans, RES_RE, NOTE_RE, OPEN
from validate_solos import find_audio


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("library", type=Path)
    ap.add_argument("-o", "--out", type=Path, default=Path("work/solo_features.json"))
    ap.add_argument("--limit", type=int, default=90)
    ap.add_argument("--start", type=int, default=0,
                    help="index into the sample to begin at; with --end this "
                         "splits the work across several processes, which is "
                         "the difference between 110 minutes and 35")
    ap.add_argument("--end", type=int, default=None)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args(argv)

    from chartgen import expression, solo, tempo as tempomod

    candidates = []
    for chart in sorted(args.library.rglob("*.chart")):
        if "chartgen" in chart.parent.name.lower():
            continue
        audio = find_audio(chart.parent)
        if not audio:
            continue
        b = blocks(chart.read_text(encoding="utf-8", errors="ignore"))
        if "ExpertSingle" not in b or is_generated(b):
            continue
        candidates.append((chart, audio, solo_spans(b["ExpertSingle"])))

    truth = [c for c in candidates if c[2]]
    rest = [c for c in candidates if not c[2]]
    random.Random(args.seed).shuffle(rest)
    sample = truth + rest[:max(0, args.limit - len(truth))]
    sample = sample[args.start:args.end if args.end is not None else len(sample)]
    print(f"{len(candidates)} charts with audio; dumping {len(sample)} "
          f"({len(truth)} with solos)", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out = []
    started = time.time()
    for i, (chart, audio, spans) in enumerate(sample, 1):
        b = blocks(chart.read_text(encoding="utf-8", errors="ignore"))
        found = RES_RE.search(b.get("Song", ""))
        res = int(found.group(1)) if found else 192
        ticks = sorted({int(t) for t, lane, _ in NOTE_RE.findall(b["ExpertSingle"])
                        if int(lane) < 5 or int(lane) == OPEN})
        if len(ticks) < 64:
            continue
        words = []
        for tick, ev in re.findall(r"(\d+)\s*=\s*E\s+(.+)", b.get("Events", "")):
            if ev.strip().strip('"').lower().startswith("lyric "):
                words.append(int(tick))
        try:
            y, sr = tempomod.load(str(audio))
            tmap = tempomod.detect(y, sr, resolution=res)
            marks = expression.sections(y, sr, tmap)
            bounds = [t for t, _ in marks] + [ticks[-1] + 1]
            sections = list(pairwise(bounds))
            rows = solo.section_evidence(y, sr, tmap, sections, res)
        except Exception as error:
            print(f"  [{i}] {chart.parent.name[:40]}: {type(error).__name__}: {error}",
                  flush=True)
            continue

        out.append({
            "name": chart.parent.name,
            "resolution": res,
            "last_tick": ticks[-1],
            "truth": spans,
            "words": words,
            "sections": [
                {
                    "start": start, "end": end,
                    "notes": sum(1 for t in ticks if start <= t < end),
                    "evidence": row,
                }
                for (start, end), row in zip(sections, rows, strict=True)
            ],
        })
        args.out.write_text(json.dumps(out), encoding="utf-8")
        print(f"  [{i}/{len(sample)}] {chart.parent.name[:50]:<52} "
              f"{len(sections)} sections, {len(spans)} solo(s), "
              f"{time.time() - started:.0f}s", flush=True)

    print(f"\nwrote {len(out)} songs to {args.out} in {time.time() - started:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
