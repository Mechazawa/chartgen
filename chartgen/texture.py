"""Chord texture, rhythm consolidation, and climax accents.

Measured on 916 human charts (tools/study_chords.py), the fun gap in
generated charts is not chord COUNT (ours is in-band) but chord TEXTURE:

- humans repeat the SAME chord shape on 59.7% of adjacent chord pairs
  (a strummed riff); ours repeated 17%, because transcription jitter
  flickers the shape and every flicker then read as a "rapid chord
  change" and got demoted to its root. Smoothing flicker into honest
  repeats is what unlocks riffs - same-shape repeats cost the hand
  nothing and were never the playability problem;
- 29-33% of our chords spanned 3+ lanes against a human 5.4%. RBN named
  this failure in 2009 and prescribed the fix: narrow the wide chord
  inward (Orange to Blue), don't delete it;
- our gallop-shaped figures are structurally sound (more grid-aligned
  than humans') but 4.5x too frequent, because the density gate keeps
  every 16th fragment the audio supports while human charters follow
  the RBN reduction doctrine: find the intended rhythm. A real gallop
  REPEATS; a one-off fragment consolidates to even eighths.

Everything here runs on chart notes only - no audio - so it applies to
both engines.
"""

OPEN = 7

# Shape flicker vs a real chord change: a real progression moves the
# anchor by 2+ lanes or changes the chord size; jitter wobbles within one.
FLICKER_MAX_ANCHOR_MOVE = 1
# Wide chords: humans keep spans >= 3 to 5.4% of chords, and never in
# quick succession. Isolated wide chords stay; ones adjacent to another
# chord shape get pulled inward.
WIDE_SPAN = 3
# One-off gallop fragments consolidate; a figure repeating within this
# window is a deliberate rhythm and stays.
GALLOP_REPEAT_WINDOW_BEATS = 8
# Climax accents: 3-note chords are 1.6% of human chords, reserved for
# "big" harmony. A few per song at section entries, never more.
MAX_CLIMAX_CHORDS = 6


def _by_tick(notes):
    groups: dict[int, list] = {}
    for note in notes:
        groups.setdefault(note[0], []).append(note)
    return groups


def _shape(group):
    return tuple(sorted(n[1] for n in group if n[1] != OPEN))


def smooth_chord_shapes(notes, resolution: int):
    """Turn transcription flicker into the same-shape runs humans strum.

    Within a run of chords a beat or less apart, a chord whose shape
    differs from its predecessor only by jitter (same note count, anchor
    moved at most one lane) adopts the predecessor's shape. A real
    progression - anchor moving two lanes, chord growing or shrinking -
    passes through untouched, so pitch direction survives.
    """
    groups = _by_tick(notes)
    ticks = sorted(groups)
    out = []
    prev_tick = None
    prev_shape = ()
    for tick in ticks:
        group = sorted(groups[tick], key=lambda n: n[1])
        shape = _shape(group)
        if (len(shape) > 1 and len(prev_shape) == len(shape)
                and shape != prev_shape
                and prev_tick is not None
                and tick - prev_tick <= resolution
                and abs(min(shape) - min(prev_shape)) <= FLICKER_MAX_ANCHOR_MOVE):
            fretted = [n for n in group if n[1] != OPEN]
            others = [n for n in group if n[1] == OPEN]
            group = [(tick, lane, note[2])
                     for lane, note in zip(prev_shape, fretted, strict=True)] + others
            shape = prev_shape
        out.extend(group)
        if len(shape) > 1:
            prev_tick, prev_shape = tick, shape
        elif prev_tick is not None and tick - prev_tick > resolution:
            prev_shape = ()
    return sorted(out)


def narrow_wide_chords(notes, resolution: int):
    """Pull wide chords inward when they sit next to another chord shape.

    RBN's remedy for the one failure it names by colour: "wrap 8th-note
    Green/Orange jumps to Blue gems". An isolated wide chord is a real
    accent and stays; one within a beat of a DIFFERENT chord shape asks
    the hand to re-form across the whole neck at speed, so its top lane
    comes down to anchor + 2.
    """
    groups = _by_tick(notes)
    ticks = sorted(groups)
    shapes = {t: _shape(groups[t]) for t in ticks}
    chord_ticks = [t for t in ticks if len(shapes[t]) > 1]

    out = []
    for i, tick in enumerate(chord_ticks):
        shape = shapes[tick]
        span = max(shape) - min(shape)
        if span < WIDE_SPAN:
            continue
        near_change = any(
            0 < abs(tick - other) <= resolution and shapes[other] != shape
            for other in chord_ticks[max(0, i - 1):i + 2]
        )
        if not near_change:
            continue
        anchor = min(shape)
        narrowed = []
        used = {anchor}
        for lane in shape[1:]:
            new = min(lane, anchor + 2)
            while new in used and new < 4:
                new += 1
            while new in used and new > 0:
                new -= 1
            used.add(new)
            narrowed.append(new)
        remap = dict(zip(shape[1:], narrowed, strict=True))
        groups[tick] = [(t, remap.get(lane, lane), sus)
                        for t, lane, sus in groups[tick]]
    for tick in ticks:
        out.extend(groups[tick])
    return sorted(out)


def consolidate_gallops(notes, resolution: int):
    """Simplify ONE-OFF 16th-note fragments to even eighths; keep the
    figures that repeat, because a repeating figure is the rhythm.

    Measured: our gallop-shaped figures are as grid-aligned and clustered
    as human ones but 4.5x as frequent (21.5 vs 4.7 per 100 positions) -
    the audio really has those onsets, but charters consolidate fragments
    that do not recur. The middle note of an isolated
    16th-16th-then-space figure is removed, leaving the eighth pulse.
    """
    groups = _by_tick(notes)
    ticks = sorted(groups)
    if len(ticks) < 4:
        return sorted(notes)
    sixteenth = resolution // 4
    tolerance = max(1, sixteenth // 3)

    def is16(gap):
        return abs(gap - sixteenth) <= tolerance

    # Figure = three positions with two 16th gaps, isolated by >= 8th gaps.
    figures = []
    for i in range(1, len(ticks) - 2):
        g_before = ticks[i] - ticks[i - 1]
        g1 = ticks[i + 1] - ticks[i]
        g2 = ticks[i + 2] - ticks[i + 1]
        if is16(g1) and is16(g2) and g_before >= 2 * sixteenth - tolerance:
            after = (ticks[i + 3] - ticks[i + 2]) if i + 3 < len(ticks) else resolution
            if after >= 2 * sixteenth - tolerance:
                figures.append(i)

    window = GALLOP_REPEAT_WINDOW_BEATS * resolution
    doomed = set()
    for i in figures:
        repeats = any(other != i and abs(ticks[other] - ticks[i]) <= window
                      for other in figures)
        middle = ticks[i + 1]
        # Only a lone, sustainless single note consolidates; chords and
        # held notes are deliberate whatever their spacing.
        group = groups[middle]
        if not repeats and len(group) == 1 and group[0][2] == 0 \
                and middle % (resolution // 2) != 0:
            doomed.add(middle)
    if not doomed:
        return sorted(notes)
    return sorted(n for n in notes if n[0] not in doomed)


def climax_chords(notes, section_marks, resolution: int,
                  max_add: int = MAX_CLIMAX_CHORDS):
    """Grow a 2-note chord to 3 at section entries - the documented role
    of 3-note chords: "big" harmony at arrival points, never a density
    lever. Only isolated chords (a beat of air both sides) qualify, and
    the added lane fills the shape inward, so no wide monsters appear.
    """
    if not section_marks:
        return sorted(notes)
    groups = _by_tick(notes)
    ticks = sorted(groups)
    added = 0
    for start, _ in sorted(section_marks):
        if added >= max_add:
            break
        # The first chord within a bar of the section entry.
        candidates = [t for t in ticks
                      if start <= t <= start + 4 * resolution
                      and len(_shape(groups[t])) == 2]
        for tick in candidates:
            i = ticks.index(tick)
            before = tick - ticks[i - 1] if i > 0 else resolution * 2
            after = ticks[i + 1] - tick if i + 1 < len(ticks) else resolution * 2
            if before < resolution or after < resolution:
                continue
            shape = _shape(groups[tick])
            span = max(shape) - min(shape)
            if span == 2:
                new_lane = min(shape) + 1      # fill the middle
            elif span == 1 and max(shape) < 4:
                new_lane = max(shape) + 1      # extend upward
            elif span == 1:
                new_lane = min(shape) - 1
            else:
                continue
            if new_lane in shape or not (0 <= new_lane <= 4):
                continue
            sustain = min(n[2] for n in groups[tick])
            groups[tick].append((tick, new_lane, sustain))
            added += 1
            break
    out = []
    for tick in ticks:
        out.extend(groups[tick])
    return sorted(out)
