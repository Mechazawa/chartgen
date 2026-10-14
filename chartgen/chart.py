"""Write Clone Hero song folders: notes.chart + song.ini."""
import re
from pathlib import Path

from .tempo import TempoMap

TIERS = ("ExpertSingle", "HardSingle", "MediumSingle", "EasySingle")
GHL_TIERS = ("ExpertGHLGuitar", "HardGHLGuitar", "MediumGHLGuitar", "EasyGHLGuitar")
# 6-fret lanes are W1 W2 W3 B1 B2 B3 = 0-5, but .chart numbers B3 as 8
# because 5 and 6 were already the forced and tap flags.
GHL_NOTE = {5: 8}

Note = tuple[int, int, int]  # (tick, lane, sustain_ticks); lane 7 is an open note


def _tier_block(name: str, notes: list[Note], star_power: list[tuple[int, int]],
                solos: list[tuple[int, int]] = (), taps: set[int] = frozenset(),
                forced: set[int] = frozenset(),
                note_numbers: dict[int, int] | None = None) -> str:
    """One difficulty track. Star power, solos and tap markers are per-track
    in .chart, so they repeat in every block.

    A phrase covering no notes in this tier is dropped: after reduction a
    phrase that survives into Easy may have had all its notes removed, and an
    empty phrase is unactivatable dead weight. Solo markers are the local
    (unquoted) track-event form CH scores its solo bonus from. A tap is note
    value 6 at the same tick as the note it modifies — emitted only where
    this tier still has a note, since a bare modifier is dead weight too.
    Forced flags (N 5) TOGGLE the natural HOPO state, and that state depends
    on note spacing — which reduction changes — so they are only ever passed
    for the Expert tier they were computed against.
    """
    ticks = {t for t, _, _ in notes}
    numbers = note_numbers or {}
    lines = [(t, 0, f"  {t} = N {numbers.get(lane, lane)} {sus}") for t, lane, sus in notes]
    lines += [(t, 0, f"  {t} = N 6 0") for t in taps if t in ticks]
    lines += [(t, 0, f"  {t} = N 5 0") for t in forced if t in ticks]
    lines += [
        (start, 1, f"  {start} = S 2 {length}")
        for start, length in star_power
        if any(start <= t < start + length for t in ticks)
    ]
    for start, end in solos:
        if any(start <= t < end for t in ticks):
            lines.append((start, 2, f"  {start} = E solo"))
            lines.append((end, 2, f"  {end} = E soloend"))
    body = "\n".join(text for _, _, text in sorted(lines))
    return f"[{name}]\n{{\n{body}\n}}"


def _event_rank(text: str) -> int:
    """Ordering for events sharing a tick.

    Sections lead; everything in the lyric stream keeps the order it was
    generated in, which is already correct — a phrase opens before its own
    first syllable and closes before the next one opens. Sorting these lines
    by text instead put "lyric You" ahead of "phrase_start" and lost the
    first word of every phrase, and when a whole phrase quantized onto one
    tick it reordered the words alphabetically: a real chart's outro read
    "thank i'll and for in next one. see the watching".
    """
    return 0 if text.startswith("section ") else 1


def write_chart(
    tiers: dict[str, list[Note]],
    tempo: TempoMap,
    meta: dict,
    audio_filename: str,
    star_power: list[tuple[int, int]] = (),
    events: list[tuple[int, str]] = (),
    lyrics: list[tuple[int, str]] = (),
    solos: list[tuple[int, int]] = (),
    taps: set[int] = frozenset(),
    forced: set[int] = frozenset(),
) -> str:
    """Render a full .chart: real SyncTrack, sections, lyrics, every difficulty.

    `events` are section names (prefixed "section" for practice mode);
    `lyrics` are raw event texts (phrase_start / lyric <word> / phrase_end)
    written verbatim. Both live in [Events], merged in tick order.
    """
    sync = "\n".join(f"  {tick} = B {bpm}" for tick, bpm in tempo.sync_track())
    # The third field is emission order, and it is the only tie-break: lyric
    # events must come out in the order they were written, not sorted.
    merged = [(tick, f'section {name}', i) for i, (tick, name) in enumerate(events)]
    merged += [(tick, text, i) for i, (tick, text) in enumerate(lyrics)]
    event_lines = "\n".join(
        f'  {tick} = E "{text}"'
        for tick, text, _ in sorted(merged, key=lambda e: (e[0], _event_rank(e[1]), e[2]))
    )
    blocks = "\n".join(
        _tier_block(name, tiers[name], list(star_power), list(solos), set(taps),
                    set(forced) if name == "ExpertSingle" else frozenset(),
                    GHL_NOTE if name in GHL_TIERS else None)
        for name in TIERS + GHL_TIERS if tiers.get(name)
    )

    return f"""[Song]
{{
  Name = "{meta['name']}"
  Artist = "{meta['artist']}"
  Charter = "{meta['charter']}"
  Album = "{meta.get('album', '')}"
  Year = "{meta.get('year', '')}"
  Offset = 0
  Resolution = {tempo.resolution}
  Genre = "{meta.get('genre', '')}"
  MediaType = "cd"
  MusicStream = "{audio_filename}"
}}
[SyncTrack]
{{
  0 = TS 4
{sync}
}}
[Events]
{{
{event_lines}
}}
{blocks}
"""


def write_song_ini(meta: dict, length_ms: int, hopos: bool = False,
                   diff_guitar: int = -1, preview_start_ms: int = 0,
                   diff_guitarghl: int = -1) -> str:
    """song.ini so Clone Hero shows real metadata instead of the filename.

    diff_guitar is the 0-6 tier badge CH shows in the song list (Rock Band's
    tier system by lineage); chartgen.rating predicts it from the chart,
    calibrated against the local library. -1 means "unrated".

    hopos=False writes `hopo_frequency = 1`, a 1-tick threshold no two distinct
    positions can satisfy, so every note strums. This is the clean way to switch
    HOPOs off: they are implicit in note spacing, and the alternative — stamping a
    forced-strum `N 5` on every affected note — is exactly the markup the vendored
    reducer mishandles, which would leave the tiers disagreeing.
    """
    hopo_line = "" if hopos else "hopo_frequency = 1\n"
    return f"""[song]
name = {meta['name']}
artist = {meta['artist']}
album = {meta.get('album', '')}
genre = {meta.get('genre', '')}
year = {meta.get('year', '')}
charter = {meta['charter']}
song_length = {length_ms}
diff_guitar = {diff_guitar}
diff_guitarghl = {diff_guitarghl}
diff_band = {diff_guitar}
delay = 0
preview_start_time = {preview_start_ms}
{hopo_line}"""


def summarize(chart_path: Path) -> dict[str, tuple[int, int, int]]:
    """Per tier: (notes, sustains, star power phrases), for reporting and tests."""
    text = Path(chart_path).read_text(encoding="utf-8")
    out = {}
    for tier in TIERS:
        block = re.search(rf"\[{tier}\]\s*\{{(.*?)\}}", text, re.S)
        if not block:
            continue
        body = block.group(1)
        out[tier] = (
            len(re.findall(r"= N \d+ \d+", body)),
            len(re.findall(r"= N \d+ [1-9]\d*", body)),
            len(re.findall(r"= S 2 ", body)),
        )
    return out
