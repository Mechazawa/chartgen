"""Patch lyric events into existing generated charts, no regeneration.

Transcribes each song folder's audio and splices phrase/lyric events into the
[Events] block of notes.chart, using the chart's own SyncTrack for the
seconds->tick mapping (via a TempoMap replayed from the B events). Skips
folders that already contain lyric events.

    python tools/add_lyrics.py ~/.clonehero/Songs --match chartgen
"""
import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import chartgen  # noqa: F401,E402

AUDIO_EXTS = (".opus", ".ogg", ".mp3", ".wav")


def tempo_from_chart(text: str):
    """Replay the SyncTrack into beat times so time_to_beat works exactly."""
    import numpy as np

    from chartgen.tempo import TempoMap

    resolution = int(re.search(r"Resolution = (\d+)", text).group(1))
    events = [(int(t), int(b) / 1000.0)
              for t, b in re.findall(r"(\d+) = B (\d+)", text)]
    events.sort()
    last_note = max(int(t) for t in re.findall(r"(\d+) = N \d+ \d+", text))

    beat_times, t_now, tick_now = [], 0.0, 0
    bpm = events[0][1] if events else 120.0
    idx = 0
    for beat in range(last_note // resolution + 8):
        tick = beat * resolution
        while idx < len(events) and events[idx][0] <= tick:
            t_now += (events[idx][0] - tick_now) / resolution * 60.0 / bpm
            tick_now, bpm = events[idx][0], events[idx][1]
            idx += 1
        beat_times.append(t_now + (tick - tick_now) / resolution * 60.0 / bpm)
    return TempoMap(beat_times=np.array(beat_times), resolution=resolution,
                    pickup_beats=0)


def patch(folder: Path) -> str:
    from chartgen.lyrics import transcribe

    chart = folder / "notes.chart"
    text = chart.read_text(encoding="utf-8")
    if 'E "lyric ' in text or 'E "phrase_start"' in text:
        return "already has lyrics"
    audio = next((folder / f"song{e}" for e in AUDIO_EXTS
                  if (folder / f"song{e}").is_file()), None)
    if audio is None:
        return "no audio"

    tempo = tempo_from_chart(text)
    events = transcribe(str(audio), tempo, lambda m: None)
    if not events:
        return "instrumental (no lyrics)"

    block = re.search(r"(\[Events\]\s*\{\n)(.*?)(\})", text, re.S)
    existing = [ln for ln in block.group(2).splitlines() if "=" in ln]
    merged = existing + [f'  {tick} = E "{txt}"' for tick, txt in events]
    merged.sort(key=lambda ln: int(ln.split(" = ")[0]))
    text = text[:block.start(2)] + "\n".join(merged) + "\n" + text[block.end(2):]
    chart.write_text(text, encoding="utf-8")
    words = sum(1 for _, t in events if t.startswith("lyric"))
    return f"{words} words added"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root", type=Path)
    ap.add_argument("--match", default="chartgen",
                    help="only folders whose name contains this")
    args = ap.parse_args()

    for folder in sorted(args.root.iterdir()):
        if not folder.is_dir() or args.match not in folder.name:
            continue
        if not (folder / "notes.chart").is_file():
            continue
        try:
            outcome = patch(folder)
        except Exception as error:
            outcome = f"FAILED: {type(error).__name__}: {error}"
        print(f"{folder.name[:52]:<54} {outcome}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
