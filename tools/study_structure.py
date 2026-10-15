"""Do human charters reuse note patterns across repeated song sections?

The premise behind section pattern reuse: verse 2 should be charted like
verse 1. Before building that, check it. Human charts name their practice
sections, so instances of the same name (chorus, chorus 2, ...) can be
compared note-for-note, with differently-named sections as the control.

    python tools/study_structure.py ~/.clonehero/Songs
"""
import argparse
import re
import statistics
import sys
from itertools import pairwise
from pathlib import Path

SECTION_RE = re.compile(r'(\d+)\s*=\s*E\s+"?(?:section|prc)[ _]([^"]+)"?')
NOTE_RE = re.compile(r"(\d+)\s*=\s*N\s+([0-4])\s+\d+")


def normalise(name: str) -> str:
    """'Chorus 2b' -> 'chorus': instances of one part share a key."""
    name = name.lower().strip()
    name = re.sub(r"[\s_\-]*\d+[a-z]?$", "", name)
    return re.sub(r"[^a-z ]", "", name).strip()


def similarity(a: set, b: set) -> float:
    """Jaccard overlap; 1.0 means the two spans carry the same pattern."""
    if not a and not b:
        return 1.0
    return len(a & b) / max(1, len(a | b))


def chart_sections(text: str):
    block = re.search(r"\[Events\]\s*\{(.*?)\}", text, re.S)
    if not block:
        return []
    marks = [(int(t), normalise(n)) for t, n in SECTION_RE.findall(block.group(1))]
    return sorted(set(marks))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root", type=Path)
    args = ap.parse_args()

    same, different, charts = [], [], 0
    for chart in sorted(args.root.rglob("notes.chart")):
        if "chartgen" in chart.parent.name:
            continue
        text = chart.read_text(encoding="utf-8", errors="replace")
        marks = chart_sections(text)
        expert = re.search(r"\[ExpertSingle\]\s*\{(.*?)\}", text, re.S)
        if len(marks) < 3 or not expert:
            continue
        notes = [(int(t), int(l)) for t, l in NOTE_RE.findall(expert.group(1))]
        if len(notes) < 100:
            continue
        charts += 1

        spans = []
        for (start, name), (end, _) in pairwise(marks):
            pattern = {(t - start, l) for t, l in notes if start <= t < end}
            if len(pattern) >= 8:
                spans.append((name, end - start, pattern))

        for i, (name_a, len_a, pat_a) in enumerate(spans):
            for name_b, len_b, pat_b in spans[i + 1:]:
                # Only compare spans of comparable length, or the Jaccard
                # score just measures the length mismatch.
                if not 0.8 <= len_a / max(1, len_b) <= 1.25:
                    continue
                (same if name_a == name_b else different).append(
                    similarity(pat_a, pat_b))

    if not same:
        print("no repeated named sections found")
        return 1
    print(f"{charts} human charts with named sections\n")
    print(f"same-named section pairs      : {len(same):>5}, "
          f"median pattern overlap {statistics.median(same):.0%}")
    print(f"differently-named pairs (ctrl): {len(different):>5}, "
          f"median pattern overlap {statistics.median(different):.0%}")
    strong = sum(1 for s in same if s >= 0.5) / len(same)
    print(f"\n{strong:.0%} of same-named pairs share at least half their notes")
    print("\nReading: if same-named sections overlap far more than the control, "
          "charters really do reuse patterns and reuse is worth building. If "
          "the two are close, they re-chart each section to the audio.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
