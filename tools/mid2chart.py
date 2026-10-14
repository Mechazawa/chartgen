"""Convert Rock-Band-style notes.mid charts to .chart text for training.

61 of the library's songs ship only notes.mid, which audio2chart cannot read;
converting them grows the training corpus ~50%. Only what training uses needs
to be faithful: note tick positions per difficulty and the tempo map (which
converts ticks to seconds). Durations survive too (cheap), but tap/forced
flags, opens-via-sysex, star power and events are deliberately dropped —
training discretization discards them anyway.

Library use: `chart_text = convert(Path("notes.mid"))`.
CLI (writes <out>/<song folder name>.chart, never touches the song folders):

    python tools/mid2chart.py --root ~/.clonehero/Songs --out data/converted
"""
import argparse
import sys
from pathlib import Path

GUITAR_TRACK = "PART GUITAR"
# RB pitch layout for 5-fret guitar: base pitch per difficulty, 5 lanes up.
DIFF_BASE = {"Easy": 60, "Medium": 72, "Hard": 84, "Expert": 96}


def _tempo_events(mid) -> list[tuple[int, int]]:
    """[(tick, bpm*1000)] from set_tempo metas (they live in the first track,
    but scan all — some exporters scatter them)."""
    out = []
    for track in mid.tracks:
        tick = 0
        for msg in track:
            tick += msg.time
            if msg.type == "set_tempo":
                bpm_milli = round(60_000_000_000 / msg.tempo)
                out.append((tick, bpm_milli))
    out.sort()
    # Collapse duplicates at the same tick (last one wins, like the game does).
    dedup = {}
    for tick, bpm in out:
        dedup[tick] = bpm
    events = sorted(dedup.items())
    return events or [(0, 120_000)]


def _time_signatures(mid) -> list[tuple[int, int]]:
    out = {}
    for track in mid.tracks:
        tick = 0
        for msg in track:
            tick += msg.time
            if msg.type == "time_signature":
                out[tick] = msg.numerator
    return sorted(out.items()) or [(0, 4)]


def _guitar_notes(mid, resolution: int) -> dict[str, list[tuple[int, int, int]]]:
    """{difficulty: [(tick, lane, sustain), ...]} from PART GUITAR."""
    track = next((t for t in mid.tracks if (t.name or "").strip().upper() == GUITAR_TRACK), None)
    if track is None:
        return {}

    # Sustains shorter than 1/12 beat are struck notes, per the common cutoff.
    cutoff = resolution // 3
    pressed: dict[int, int] = {}
    notes: dict[str, list[tuple[int, int, int]]] = {d: [] for d in DIFF_BASE}
    tick = 0
    for msg in track:
        tick += msg.time
        if msg.type == "note_on" and msg.velocity > 0:
            pressed[msg.note] = tick
        elif msg.type in ("note_off", "note_on"):  # note_on vel 0 == off
            start = pressed.pop(msg.note, None)
            if start is None:
                continue
            for diff, base in DIFF_BASE.items():
                lane = msg.note - base
                if 0 <= lane <= 4:
                    length = tick - start
                    notes[diff].append((start, lane, length if length >= cutoff else 0))
    return {d: sorted(v) for d, v in notes.items() if v}


def convert(mid_path: Path, name: str = "") -> str | None:
    """notes.mid -> .chart text, or None if there is no guitar track."""
    import mido

    # clip=True: several community mids carry out-of-range data bytes in
    # sysex/venue junk we don't read anyway; strict parsing rejects the file.
    mid = mido.MidiFile(str(mid_path), clip=True)
    resolution = mid.ticks_per_beat
    tiers = _guitar_notes(mid, resolution)
    if not tiers.get("Expert"):
        return None

    sync_lines = [f"  {tick} = TS {num}" for tick, num in _time_signatures(mid)]
    sync_lines += [f"  {tick} = B {bpm}" for tick, bpm in _tempo_events(mid)]
    sync_lines.sort(key=lambda line: int(line.split(" = ")[0]))

    blocks = []
    for diff, noteset in tiers.items():
        body = "\n".join(f"  {t} = N {lane} {sus}" for t, lane, sus in noteset)
        blocks.append(f"[{diff}Single]\n{{\n{body}\n}}")

    return (
        "[Song]\n{\n"
        f'  Name = "{name or mid_path.parent.name}"\n'
        "  Offset = 0\n"
        f"  Resolution = {resolution}\n"
        '  MediaType = "cd"\n'
        "}\n"
        "[SyncTrack]\n{\n" + "\n".join(sync_lines) + "\n}\n"
        "[Events]\n{\n}\n" + "\n".join(blocks) + "\n"
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("data/converted"))
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    done = skipped = 0
    for mid_path in sorted(args.root.rglob("notes.mid")):
        if (mid_path.parent / "notes.chart").is_file():
            continue  # song already has a chart; nothing to gain
        try:
            text = convert(mid_path)
        except Exception as error:
            print(f"FAIL {mid_path.parent.name}: {type(error).__name__}: {error}")
            skipped += 1
            continue
        if text is None:
            print(f"skip {mid_path.parent.name}: no PART GUITAR / Expert notes")
            skipped += 1
            continue
        dest = args.out / f"{mid_path.parent.name}.chart"
        dest.write_text(text, encoding="utf-8")
        done += 1
    print(f"\nconverted {done}, skipped {skipped} -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
