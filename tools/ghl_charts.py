"""Read the 6-fret part (PART GUITAR GHL) of an official notes.mid.

Buttons are 0-5 = W1 W2 W3 B1 B2 B3; an open note is the empty shape.
A shape is every button struck on one tick, so a chord is one entry.

    chart = load(Path(".../notes.mid"))
    chart.tiers["Expert"]  # [(tick, frozenset({0, 3}), sustain), ...]
"""
import re
from dataclasses import dataclass
from pathlib import Path

import mido

TRACK = "PART GUITAR GHL"
# Per difficulty the base note is open, then W1 W2 W3 B1 B2 B3 above it.
DIFF_BASE = {"Expert": 94, "Hard": 82, "Medium": 70, "Easy": 58}
BUTTONS = ("W1", "W2", "W3", "B1", "B2", "B3")

Shape = frozenset[int]


def shape_name(shape: Shape) -> str:
    return "+".join(BUTTONS[b] for b in sorted(shape)) or "OPEN"


@dataclass(frozen=True)
class GhlChart:
    resolution: int
    tempo: list[tuple[int, int]]  # (tick, microseconds per beat)
    tiers: dict[str, list[tuple[int, Shape, int]]]
    sections: tuple[tuple[int, str], ...] = ()  # (tick, name) from the EVENTS track

    def seconds(self, tick: int) -> float:
        elapsed, last_tick, last_tempo = 0.0, 0, 500_000
        for change_tick, tempo in self.tempo:
            if change_tick >= tick:
                break
            elapsed += (change_tick - last_tick) * last_tempo / self.resolution / 1e6
            last_tick, last_tempo = change_tick, tempo
        return elapsed + (tick - last_tick) * last_tempo / self.resolution / 1e6


def load(mid_path: Path) -> GhlChart | None:
    mid = mido.MidiFile(str(mid_path), clip=True)
    track = next((t for t in mid.tracks if (t.name or "").strip().upper() == TRACK), None)
    if track is None:
        return None

    tempo = sorted({tick: msg.tempo for tick, msg in _absolute(mid.tracks[0])
                    if msg.type == "set_tempo"}.items())
    pressed: dict[int, int] = {}
    hits: dict[str, dict[int, tuple[set[int], int]]] = {d: {} for d in DIFF_BASE}
    for tick, msg in _absolute(track):
        if msg.type == "note_on" and msg.velocity > 0:
            pressed[msg.note] = tick
            continue
        if msg.type not in ("note_off", "note_on") or msg.note not in pressed:
            continue
        start = pressed.pop(msg.note)
        for diff, base in DIFF_BASE.items():
            offset = msg.note - base
            if 0 <= offset <= 6:
                buttons, sustain = hits[diff].setdefault(start, (set(), 0))
                if offset:
                    buttons.add(offset - 1)
                hits[diff][start] = (buttons, max(sustain, tick - start))

    cutoff = mid.ticks_per_beat // 3  # shorter holds are struck notes
    tiers = {diff: [(tick, frozenset(buttons), sustain if sustain >= cutoff else 0)
                    for tick, (buttons, sustain) in sorted(by_tick.items())]
             for diff, by_tick in hits.items()}
    sections = tuple((tick, found.group(1).strip()) for track in mid.tracks if track.name == "EVENTS"
                     for tick, msg in _absolute(track)
                     if msg.type == "text" and (found := re.fullmatch(r"\[section (.+)\]", msg.text.strip())))
    return GhlChart(mid.ticks_per_beat, tempo or [(0, 500_000)], tiers, sections)


def _absolute(track):
    tick = 0
    for msg in track:
        tick += msg.time
        yield tick, msg
