"""Calibrate the playability simulator against real charts, then judge ours.

A gate is only meaningful if human charts pass it. This replays every chart's
own SyncTrack so the simulator sees real seconds, then reports what fraction
of note-to-note transitions demand more of the hands than the chart allows.

    python tools/study_playability.py ~/.clonehero/Songs
"""
import argparse
import re
import statistics
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chartgen.playability import analyse  # noqa: E402
from chartgen.tempo import TempoMap  # noqa: E402

NOTE = re.compile(r"(\d+)\s*=\s*N\s+([0-7])\s+(\d+)")


def tempo_from_chart(text: str):
    """Beat times replayed from the chart's own SyncTrack."""
    res = int(re.search(r"Resolution\s*=\s*(\d+)", text).group(1))
    events = sorted((int(t), int(b) / 1000.0)
                    for t, b in re.findall(r"(\d+)\s*=\s*B\s+(\d+)", text))
    if not events:
        return None
    last_tick = max(int(t) for t in re.findall(r"(\d+)\s*=\s*N\s+\d+\s+\d+", text))
    beats, now, tick_at, bpm, i = [], 0.0, 0, events[0][1], 0
    for beat in range(last_tick // res + 4):
        tick = beat * res
        while i < len(events) and events[i][0] <= tick:
            now += (events[i][0] - tick_at) / res * (60.0 / bpm)
            tick_at, bpm = events[i][0], events[i][1]
            i += 1
        beats.append(now + (tick - tick_at) / res * (60.0 / bpm))
    return TempoMap(beat_times=np.array(beats), resolution=res, pickup_beats=0)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root", type=Path)
    ap.add_argument("--tier", default="ExpertSingle")
    args = ap.parse_args()

    groups: dict[str, list] = {"HUMAN": [], "ours (chartgen)": []}
    detail = []
    for chart in sorted(args.root.rglob("notes.chart")):
        text = chart.read_text(encoding="utf-8", errors="replace")
        block = re.search(rf"\[{args.tier}\]\s*\{{(.*?)\}}", text, re.S)
        if not block:
            continue
        notes = [(int(t), int(l), int(s)) for t, l, s in NOTE.findall(block.group(1))
                 if int(l) <= 4 or int(l) == 7]
        if len(notes) < 100:
            continue
        tempo = tempo_from_chart(text)
        if tempo is None:
            continue
        result = analyse(notes, tempo)
        if not result["transitions"]:
            continue
        key = "ours (chartgen)" if "chartgen" in chart.parent.name else "HUMAN"
        groups[key].append(result["impossible"])
        if key == "ours (chartgen)":
            detail.append((result["impossible"], chart.parent.name, result["worst"]))

    for key, values in groups.items():
        if not values:
            continue
        print(f"{key:<18}{len(values):>4} charts   "
              f"median impossible {statistics.median(values):.2%}   "
              f"p90 {sorted(values)[int(0.9 * (len(values) - 1))]:.2%}   "
              f"worst {max(values):.2%}")

    if detail:
        print("\nour charts, worst first (ratio > 1 means the hands run out of time):")
        for share, name, worst in sorted(detail, reverse=True)[:8]:
            spots = ", ".join(f"{r}x @{s}s" for r, s in worst[:3])
            print(f"  {share:>6.2%}  {name[:44]:<46} {spots}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
