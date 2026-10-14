"""Score the audio-based solo detector against human solo markers.

The old lyric-gap rule scored 18% precision on this library, so any
replacement has to be measured the same way rather than eyeballed. Human
charts supply the ground truth: where a charter wrote `E solo`, there is a
solo. This feeds the detector the human chart's own notes and our own
audio-derived sections, so what is being measured is the audio evidence and
nothing else.

    python tools/validate_solos.py ~/.clonehero/Songs --limit 40
"""
import argparse
import random
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from study_solos import is_generated, blocks, solo_spans, RES_RE, NOTE_RE, OPEN

AUDIO = ("song.ogg", "song.opus", "song.mp3", "guitar.ogg", "song.wav")


def find_audio(folder: Path):
    for name in AUDIO:
        if (folder / name).is_file():
            return folder / name
    for p in folder.iterdir():
        if p.suffix.lower() in (".ogg", ".mp3", ".opus", ".wav", ".flac"):
            return p
    return None


def overlaps(a, b, min_iou=0.25) -> bool:
    """Substantial overlap, not a grazing touch: the first version counted
    a marker sharing one boundary tick with the real solo as a hit."""
    lo, hi = max(a[0], b[0]), min(a[1], b[1])
    if hi <= lo:
        return False
    return (hi - lo) / ((a[1] - a[0]) + (b[1] - b[0]) - (hi - lo)) >= min_iou


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("library", type=Path)
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--match", default=None,
                    help="only songs whose folder contains one of these "
                         "comma-separated substrings")
    args = ap.parse_args(argv)

    from chartgen import expression, solo, tempo as tempomod

    charts = [p for p in sorted(args.library.rglob("*.chart"))
              if "chartgen" not in p.parent.name.lower()]
    rows = []
    for chart in charts:
        text = chart.read_text(encoding="utf-8", errors="ignore")
        b = blocks(text)
        expert = b.get("ExpertSingle", "")
        spans = solo_spans(expert)
        audio = find_audio(chart.parent)
        if is_generated(b):
            continue
        if audio and expert:
            rows.append((chart, audio, spans))

    if args.match:
        needles = [n.strip().lower() for n in args.match.split(",")]
        rows = [r for r in rows
                if any(n in r[0].parent.name.lower() for n in needles)]
    truth = [r for r in rows if r[2]]
    empty = [r for r in rows if not r[2]]
    rng = random.Random(args.seed)
    rng.shuffle(empty)
    half = max(1, args.limit // 2)
    sample = truth[:half] + empty[:args.limit - half]
    print(f"{len(rows)} charts with audio; testing {len(sample)} "
          f"({sum(1 for s in sample if s[2])} with solos)\n")

    tp = fp = fn = 0
    songs_hit = songs_missed = songs_false = 0
    started = time.time()
    for i, (chart, audio, spans) in enumerate(sample, 1):
        text = chart.read_text(encoding="utf-8", errors="ignore")
        b = blocks(text)
        found_res = RES_RE.search(b.get("Song", ""))
        res = int(found_res.group(1)) if found_res else 192
        notes = [(int(t), int(lane), int(length))
                 for t, lane, length in NOTE_RE.findall(b["ExpertSingle"])
                 if int(lane) < 5 or int(lane) == OPEN]
        if len(notes) < 64:
            continue
        try:
            y, sr = tempomod.load(str(audio))
            tmap = tempomod.detect(y, sr, resolution=res)
            sections = expression.sections(y, sr, tmap)
            lyrics = []
            for tick, ev in re.findall(r"(\d+)\s*=\s*E\s+(.+)", b.get("Events", "")):
                clean = ev.strip().strip('"')
                if clean.lower().startswith("lyric "):
                    lyrics.append((int(tick), clean))
            got = solo.detect(notes, sections, lyrics, y, sr, tmap)
        except Exception as error:
            print(f"  [{i}] {chart.parent.name[:44]}: {type(error).__name__}: {error}")
            continue

        hits = [g for g in got if any(overlaps(g, s) for s in spans)]
        missed = [s for s in spans if not any(overlaps(g, s) for g in got)]
        tp += len(hits)
        fp += len(got) - len(hits)
        fn += len(missed)
        if spans and hits:
            songs_hit += 1
        elif spans:
            songs_missed += 1
        if got and not hits:
            songs_false += 1
        mark = "OK " if (hits or (not spans and not got)) else "-- "
        print(f"  [{i}/{len(sample)}] {mark}{chart.parent.name[:44]:<46} "
              f"truth {len(spans)}  found {len(got)}  hit {len(hits)}")

    print(f"\n{time.time() - started:.0f}s")
    prec = 100.0 * tp / (tp + fp) if tp + fp else 0.0
    rec = 100.0 * tp / (tp + fn) if tp + fn else 0.0
    print(f"per-solo   precision {prec:.0f}%  recall {rec:.0f}%  "
          f"(tp {tp} fp {fp} fn {fn})")
    print(f"per-song   found a real solo in {songs_hit}, missed all in "
          f"{songs_missed}, invented one in {songs_false}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
