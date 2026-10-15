"""Look at the tail of every generated chart's lyrics.

Two playtest complaints: lyrics cut off before the vocals do, and phantom
words appear near the end. Both are end-of-song problems, so this reports
what each chart's lyrics do in its final stretch - where they stop relative
to the last note, how big the last gap is, and whether the closing phrases
repeat themselves (the signature of a Whisper hallucination loop).

    python tools/audit_lyrics.py ~/.clonehero/Songs
"""
import argparse
import re
import sys
from itertools import pairwise
from pathlib import Path

from study_solos import blocks, note_ticks, RES_RE

EVENT_RE = re.compile(r"(\d+)\s*=\s*E\s+(.+)")


def audit(path: Path):
    b = blocks(path.read_text(encoding="utf-8", errors="ignore"))
    found = RES_RE.search(b.get("Song", ""))
    res = int(found.group(1)) if found else 192
    ticks = note_ticks(b.get("ExpertSingle", ""))
    if not ticks:
        return None

    words = []
    for tick, text in sorted(((int(t), e.strip().strip('"'))
                              for t, e in EVENT_RE.findall(b.get("Events", "")))):
        if text.lower().startswith("lyric "):
            words.append((tick, text[6:].strip()))
    if not words:
        return None

    last_note = ticks[-1]
    # Phrases: consecutive words split on gaps of a bar or more.
    phrases, current = [], [words[0]]
    for prev, cur in pairwise(words):
        if cur[0] - prev[0] > 4 * res:
            phrases.append(current)
            current = []
        current.append(cur)
    phrases.append(current)

    texts = [" ".join(w for _, w in p).lower() for p in phrases]
    tail = texts[-4:]
    repeated = len(tail) != len(set(tail))

    return {
        "name": path.parent.name,
        "words": len(words),
        "phrases": len(phrases),
        "last_word_beats_before_end": (last_note - words[-1][0]) / res,
        "biggest_tail_gap": max(
            ((cur[0] - prev[0]) / res for prev, cur in pairwise(words)
             if cur[0] > last_note * 0.6), default=0.0),
        "repeated_tail": repeated,
        "tail_text": " | ".join(t[:48] for t in texts[-3:]),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("library", type=Path)
    ap.add_argument("--filter", default="chartgen", help="only folders matching this")
    args = ap.parse_args(argv)

    for p in sorted(args.library.rglob("*.chart")):
        if args.filter.lower() not in p.parent.name.lower():
            continue
        try:
            row = audit(p)
        except (OSError, ValueError):
            continue
        if not row:
            continue
        print(f"\n{row['name']}")
        print(f"  {row['words']} words in {row['phrases']} phrases; last word "
              f"{row['last_word_beats_before_end']:.0f} beats before the last note")
        print(f"  biggest gap in the last 40%: {row['biggest_tail_gap']:.0f} beats"
              f"{'   REPEATED TAIL PHRASES' if row['repeated_tail'] else ''}")
        print(f"  ends: {row['tail_text']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
