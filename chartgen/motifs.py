"""Flow motifs: the pattern layer between "correct" and "feels human".

Measured against the 800-chart genre-balanced library (2026-08-29 motif
study), the biggest kinesthetic gaps in generated charts were:

- sustain stairs (stepped legato holds): 78% of human charts use them,
  median 5.7 per 100 bars in every genre family; ours had ~0. Humans hold
  each note of a slow stepped phrase until ~80ms before the next one.
- structured fast runs: our fast-run density matches humans but the runs
  wander (shapeless zigzags) where humans chunk them into 3-4 note
  monotonic wraps (rolls) - the RBN wrapping doctrine.
- machine-gun spam: same-lane 16th runs in 77% of our charts vs a human
  31% (median 0 per 100 bars) - pitch-collapse artifacts, not chugs.
- stray pushes: 2.4x the human share of off-8th 16ths, because jitter
  reads as syncopation. Human pushes RECUR at the same bar position.
- forced HOPOs: 86% of human charts force, median 3.9 flags per 100
  notes, mostly turning on-beat stepwise 8ths into legato runs.
- chord ladders: a minority accent (42% of charts) with one canonical
  template - RBN's walk where one finger moves at a time
  (BO YO YB RB RY GY GR), pickups charted as the next chord's root.

Everything here runs on chart notes only - no audio - so it applies to
both engines. Transforms are deterministic and note-count-preserving
except where the doctrine is explicitly reductive (machine-gun thinning,
stray-push settling), mirroring texture.consolidate_gallops: a repeated
figure is the rhythm, a one-off consolidates.
"""
import collections
from itertools import pairwise

OPEN = 7

# Sustain stairs: legato phrases move stepwise at these gaps. Below the
# window is a fast run (HOPO territory, no holds); above it the ordinary
# gap-sustain rule already fires.
STAIR_MIN_GAP_BEATS = 0.45
STAIR_MAX_GAP_BEATS = 1.55
STAIR_MIN_RUN = 3
# CSC/YARN sustain-gap standard: 65-90ms of air before the next note.
STAIR_GAP_SECONDS = 0.080

# Fast-run restructuring: only runs this long with this much directional
# noise are considered shapeless. Reversal rate is direction changes per
# opportunity; humans' wrapped rolls sit far below the threshold. The
# budget is the real safety: corpus stress testing showed every chart-only
# gate still catches some deliberate human runs (irregular solo contours
# look exactly like mapping noise), so at most this share of a chart's
# notes may be re-laned, worst runs first, and a run whose lane sequence
# recurs is a riff and is never touched.
RUN_MIN_LEN = 6
RUN_REVERSAL_RATE = 0.5
RUN_BUDGET_SHARE = 0.04
CHUNK_SIZES = (4, 3)  # RBN wrapping: 3-4 note chunks, mixed

# Machine-gun: a same-lane 16th run this long that never recurs is a
# transcription artifact; a recurring one is a deliberate chug.
GUN_MIN_LEN = 8
GUN_REPEAT_WINDOW_BEATS = 16

# Forcing: human median 3.9 flags per 100 notes; cap a little above.
FORCE_BUDGET_PER100 = 6.0
FORCE_MIN_RUN = 4

# RBN chord rungs in pitch order; one finger moves per step.
CHORD_RUNGS = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (2, 4), (3, 4)]


def _by_tick(notes):
    out: dict[int, list] = {}
    for note in notes:
        out.setdefault(note[0], []).append(note)
    return out


def _positions(notes):
    """[(tick, [lanes], max_sus)] sorted; the working view for every motif."""
    groups = _by_tick(notes)
    return [(t, sorted(n[1] for n in groups[t]), max(n[2] for n in groups[t]))
            for t in sorted(groups)]


def _single_lane(pos):
    """The lane if this position is one fretted single, else None."""
    tick, lanes, _ = pos
    return lanes[0] if len(lanes) == 1 and lanes[0] != OPEN else None


def legato_stairs(notes, resolution: int, bpm: float | None = None):
    """Hold each note of a stepped phrase until just before the next.

    A phrase is >=STAIR_MIN_RUN consecutive fretted singles, gaps within
    the stair window, every lane step 1-2 (direction free - real phrases
    arch). Each phrase note that is not already sustained gets held to
    its successor minus the community-standard air gap. The phrase's
    last note keeps whatever the ordinary sustain rule gave it.

    Runs BEFORE top_up_sustains so the added holds count toward the
    tier's calibrated sustain share instead of stacking on top of it.
    """
    if not notes:
        return notes
    pos = _positions(notes)
    gap_ticks = int(STAIR_GAP_SECONDS * (bpm or 120.0) / 60.0 * resolution)
    gap_ticks = max(gap_ticks, resolution // 8)
    # The 200ms community floor ONLY - not the transcription path's stricter
    # max(res//2, 200ms). Copying that floor here was a real shipped bug:
    # at 84 BPM half a beat is 357ms, so stepped 8th phrases (the bread
    # and butter of stairs - Mary Jane's human chart holds them for ~1/3
    # beat, 236ms) could never produce a stair at all.
    floor = int(0.200 * bpm * resolution / 60.0) if bpm else resolution // 3
    lo = STAIR_MIN_GAP_BEATS * resolution
    hi = STAIR_MAX_GAP_BEATS * resolution

    stretch: dict[int, int] = {}  # tick -> new sustain
    i = 0
    while i < len(pos) - 1:
        run = [i]
        j = i
        while j + 1 < len(pos):
            a, b = pos[j], pos[j + 1]
            la, lb = _single_lane(a), _single_lane(b)
            gap = b[0] - a[0]
            if (la is not None and lb is not None
                    and 1 <= abs(lb - la) <= 2 and lo <= gap <= hi):
                run.append(j + 1)
                j += 1
            else:
                break
        if len(run) >= STAIR_MIN_RUN:
            for k in run[:-1]:
                tick, _, sus = pos[k]
                length = pos[k + 1][0] - tick - gap_ticks
                if length >= floor and length > sus:
                    stretch[tick] = length
            i = j
        else:
            i += 1
    if not stretch:
        return notes
    return sorted((t, lane, stretch.get(t, sus)) for t, lane, sus in notes)


def _runs_of_fast_singles(pos, max_gap: int):
    """Index runs of consecutive fretted singles closer than max_gap."""
    runs, cur = [], []
    for i, p in enumerate(pos):
        if _single_lane(p) is None:
            if len(cur) > 1:
                runs.append(cur)
            cur = []
            continue
        if cur and p[0] - pos[cur[-1]][0] > max_gap:
            if len(cur) > 1:
                runs.append(cur)
            cur = []
        cur.append(i)
    if len(cur) > 1:
        runs.append(cur)
    return runs


def chunk_rolls(notes, resolution: int):
    """Re-lane shapeless fast runs into 3-4 note monotonic wrap chunks.

    The RBN wrapping doctrine: long lines travel in overlapping 3-4 note
    chunks (O-B-Y, B-Y-R...), mixing chunk sizes. Our fast runs have the
    right density but wander - the pitch mapping's jitter reads as a
    zigzag no hand enjoys. Only runs that are already shapeless are
    touched (high reversal rate, wide spread, not a trill/pedal), so a
    deliberate alternation survives; the coarse contour survives because
    each chunk takes its direction from the original lanes' local trend.

    Lanes change; ticks, counts and sustains do not.
    """
    if not notes:
        return notes
    pos = _positions(notes)
    candidates = []  # (reversal_rate, run, lanes)
    seen_sequences = collections.Counter()
    all_runs = _runs_of_fast_singles(pos, resolution // 2)
    for run in all_runs:
        seen_sequences[tuple(_single_lane(pos[i]) for i in run)] += 1
    for run in all_runs:
        if len(run) < RUN_MIN_LEN:
            continue
        lanes = [_single_lane(pos[i]) for i in run]
        if seen_sequences[tuple(lanes)] > 1:
            continue  # a run that recurs note-for-note is a riff
        if any(pos[i][2] > 0 for i in run[:-1]):
            continue  # holds mid-run mean this is not a plain run
        distinct = set(lanes)
        if len(distinct) <= 2:
            continue  # trill or machine-gun; other motifs own those
        deltas = [b - a for a, b in pairwise(lanes)]
        nz = [d for d in deltas if d != 0]
        if len(nz) < 3:
            continue
        reversals = sum(1 for a, b in pairwise(nz) if a * b < 0)
        rate = reversals / max(1, len(nz) - 1)
        if rate < RUN_REVERSAL_RATE:
            continue  # already directional enough to read as a line
        common = collections.Counter(lanes).most_common(1)[0]
        if common[1] >= len(lanes) * 0.4:
            continue  # pedal bounce - a real motif, keep it
        # Corpus stress test: reversal rate alone flagged 15% of HUMAN runs,
        # because deliberate zigzags/chimneys reverse constantly too. What
        # separates them from mapping noise is REGULARITY: a human zigzag
        # cycles a short period. Any decent period fit means hands off.
        periodic = False
        for period in (2, 3, 4):
            if len(lanes) <= period:
                break
            hits = sum(1 for a, b in zip(lanes, lanes[period:], strict=False) if a == b)
            if hits / (len(lanes) - period) >= 0.75:
                periodic = True
                break
        if periodic:
            continue
        candidates.append((rate, run, lanes))

    relane: dict[int, int] = {}
    budget = max(RUN_MIN_LEN * 2, int(len(pos) * RUN_BUDGET_SHARE))
    touched = 0
    for _, run, lanes in sorted(candidates, key=lambda c: -c[0]):
        if touched + len(run) > budget:
            continue
        touched += len(run)
        # Rebuild: chunks of 4,3,4,3... each stepping one lane in its
        # segment's original trend direction. Chunks continuing the same
        # direction slide their window one lane over (the RBN wrap:
        # O-B-Y then B-Y-R); a reversal turns from the current lane.
        new = [lanes[0]]
        k = 1
        chunk_i = 0
        prev_start = prev_dir = None
        while k < len(lanes):
            size = CHUNK_SIZES[chunk_i % len(CHUNK_SIZES)]
            end = min(k + size, len(lanes))
            length = end - k
            seg = lanes[max(0, k - 1):end]
            trend = seg[-1] - seg[0]
            dirn = (1 if trend > 0 else -1 if trend < 0
                    else (-prev_dir if prev_dir else 1))

            def fit(start, d, length=length):
                """Clamp a chunk start so the whole chunk stays on the neck."""
                if d > 0:
                    return max(0, min(start, 5 - length))
                return min(4, max(start, length - 1))

            if dirn == prev_dir and prev_start is not None:
                start = fit(prev_start + dirn, dirn)
            else:
                start = fit(new[-1] + dirn, dirn)
            if start == new[-1]:
                bumped = fit(start + dirn, dirn)
                if bumped == new[-1]:  # cornered against the neck edge: turn
                    dirn = -dirn
                    bumped = fit(new[-1] + dirn, dirn)
                    if bumped == new[-1]:
                        bumped = fit(bumped + dirn, dirn)
                start = bumped
            for n in range(length):
                new.append(start + dirn * n)
            k = end
            chunk_i += 1
            prev_start, prev_dir = start, dirn
        for i, lane in zip(run, new, strict=True):
            if lane != _single_lane(pos[i]):
                relane[pos[i][0]] = lane
    if not relane:
        return notes
    return sorted((t, relane.get(t, lane) if lane != OPEN else lane, sus)
                  for t, lane, sus in notes)


def break_machine_gun(notes, resolution: int):
    """Thin one-off same-lane 16th walls to 8ths; keep recurring chugs.

    Same doctrine as texture.consolidate_gallops: the audio really has
    those onsets, but a same-lane 16th wall that appears once is the
    pitch mapping collapsing, not a riff. A run whose twin (same lane,
    comparable length) sits within the repeat window is deliberate and
    stays. Thinning keeps the on-8th members so the pulse survives.
    """
    if not notes:
        return notes
    pos = _positions(notes)
    sixteenth = resolution // 4
    guns = []  # (start_tick, lane, [indices])
    for run in _runs_of_fast_singles(pos, sixteenth + sixteenth // 3):
        if len(run) < GUN_MIN_LEN:
            continue
        lanes = {_single_lane(pos[i]) for i in run}
        if len(lanes) == 1:
            guns.append((pos[run[0]][0], lanes.pop(), run))
    window = GUN_REPEAT_WINDOW_BEATS * resolution
    doomed = set()
    for start, lane, run in guns:
        recurs = any(other is not run and o_lane == lane
                     and abs(o_start - start) <= window
                     and abs(len(other) - len(run)) <= 2
                     for o_start, o_lane, other in guns)
        if recurs:
            continue
        for i in run:
            tick, _, sus = pos[i]
            if sus == 0 and tick % (resolution // 2) != 0:
                doomed.add(tick)
    if not doomed:
        return notes
    return sorted(n for n in notes if n[0] not in doomed)


def settle_pushes(notes, resolution: int):
    """Snap non-recurring stray off-8th 16ths onto the 8th grid.

    A real push (Mary Jane's chorus riff) anticipates the beat at the
    SAME bar position again and again; jitter lands anywhere once. An
    isolated off-8th single whose bar-offset never recurs in the
    neighbouring four bars moves to the nearest free 8th position -
    consolidating to the intended rhythm rather than deleting the note.
    Notes inside fast runs are left alone (they are runs, not pushes).
    """
    if not notes:
        return notes
    pos = _positions(notes)
    eighth = resolution // 2
    bar = resolution * 4
    off_grid = set()  # (bar_index, offset) of every off-8th position
    strays = []
    ticks = [p[0] for p in pos]
    for i, (tick, lanes, sus) in enumerate(pos):
        if tick % eighth == 0:
            continue
        off_grid.add((tick // bar, tick % bar))
        before = tick - ticks[i - 1] if i else bar
        after = ticks[i + 1] - tick if i + 1 < len(ticks) else bar
        if (len(lanes) == 1 and sus == 0 and lanes[0] != OPEN
                and before >= eighth and after >= eighth):
            strays.append(i)
    taken = set(ticks)
    move: dict[int, int] = {}
    for i in strays:
        tick = pos[i][0]
        # the same off-grid bar offset in a NEIGHBOURING bar means
        # deliberate syncopation - the push is part of a groove
        recurs = any((tick // bar + d, tick % bar) in off_grid
                     for d in (-2, -1, 1, 2))
        if recurs:
            continue
        snapped = int(round(tick / eighth)) * eighth
        if snapped in taken or snapped <= 0:
            continue
        # Snapping earlier must not land inside the previous note's
        # sustain, and either direction must keep clear of neighbours.
        if i > 0:
            prev_tick, _, prev_sus = pos[i - 1]
            if snapped <= prev_tick + prev_sus:
                continue
        if i + 1 < len(pos) and snapped >= pos[i + 1][0]:
            continue
        move[tick] = snapped
        taken.add(snapped)
    if not move:
        return notes
    out = []
    for t, lane, sus in notes:
        nt = move.get(t, t)
        out.append((nt, lane, sus))
    return sorted(out)


def legato_forcing(notes, resolution: int) -> set[int]:
    """Ticks to flag N 5: on-beat stepwise 8th phrases become HOPO runs.

    What the human flags measurably do (Mary Jane: 77 of 128 forces sit
    on the 8th grid inside stepwise lines): a moving melody at 8th
    spacing naturally strums every note; forcing all but the phrase's
    first note makes it legato, the way the guitarist actually played
    it. RBN's warning bounds the budget - forcing is seasoning, "used
    sparingly", so the cap sits just above the human median rate.

    Only notes whose natural state is a strum are flagged (N 5 TOGGLES:
    flagging a natural HOPO would un-HOPO it), never chords, opens,
    same-lane repeats, or phrase-initial notes.
    """
    if not notes:
        return set()
    pos = _positions(notes)
    thr = (65 * resolution) // 192
    lo, hi = thr + 1, int(resolution * 0.75)
    phrases = []
    i = 0
    while i < len(pos) - 1:
        run = [i]
        j = i
        while j + 1 < len(pos):
            a, b = pos[j], pos[j + 1]
            la, lb = _single_lane(a), _single_lane(b)
            gap = b[0] - a[0]
            if (la is not None and lb is not None and la != lb
                    and abs(lb - la) <= 2 and lo <= gap <= hi):
                run.append(j + 1)
                j += 1
            else:
                break
        if len(run) >= FORCE_MIN_RUN:
            phrases.append(run)
            i = j
        else:
            i += 1
    budget = max(8, int(len(pos) * FORCE_BUDGET_PER100 / 100.0))
    forced: set[int] = set()
    # Longest phrases first: one long legato line reads better than
    # scattered two-note forces, and the budget goes further. A phrase
    # bigger than what remains gets its prefix - legato that trails off
    # into strums is normal charting; an unforced phrase is a lost one.
    for run in sorted(phrases, key=len, reverse=True):
        room = budget - len(forced)
        if room < FORCE_MIN_RUN - 1:
            break
        flags = run[1:][:room]
        forced.update(pos[k][0] for k in flags)
    return forced


def shape_chord_walks(notes, resolution: int):
    """Re-shape monotonic 2-note chord walks onto the RBN rung ladder.

    "If we have a descending pattern that is 7 chords long, we would use
    BO, YO, YB, RB, RY, GY, GR" - each step moves one finger. A run of
    >=3 two-note chords whose anchors step in one direction becomes
    consecutive rungs in that direction, starting from the rung nearest
    the first chord's actual shape. Same-shape runs (anchors static) are
    riffs and are never touched.
    """
    if not notes:
        return notes
    groups = _by_tick(notes)
    ticks = sorted(groups)
    chord_ticks = [t for t in ticks
                   if len([n for n in groups[t] if n[1] != OPEN]) == 2
                   and not any(n[1] == OPEN for n in groups[t])]
    if len(chord_ticks) < 3:
        return notes

    def shape(t):
        return tuple(sorted(n[1] for n in groups[t]))

    def nearest_rung(s):
        return min(range(len(CHORD_RUNGS)),
                   key=lambda r: abs(CHORD_RUNGS[r][0] - s[0]) + abs(CHORD_RUNGS[r][1] - s[1]))

    remap: dict[int, tuple[int, int]] = {}
    i = 0
    while i < len(chord_ticks) - 2:
        run = [chord_ticks[i]]
        j = i
        dirn = 0
        while j + 1 < len(chord_ticks):
            a, b = chord_ticks[j], chord_ticks[j + 1]
            d = shape(b)[0] - shape(a)[0]
            if (b - a <= resolution * 2 and d != 0 and abs(d) <= 2
                    and (dirn == 0 or (d > 0) == (dirn > 0))):
                dirn = d
                run.append(b)
                j += 1
            else:
                break
        if len(run) >= 3:
            step = 1 if dirn > 0 else -1
            rung = nearest_rung(shape(run[0]))
            for t in run:
                remap[t] = CHORD_RUNGS[max(0, min(len(CHORD_RUNGS) - 1, rung))]
                rung += step
            i = j
        else:
            i += 1
    if not remap:
        return notes
    out = []
    for t in ticks:
        if t in remap:
            sus = max(n[2] for n in groups[t])
            out.extend((t, lane, sus) for lane in remap[t])
        else:
            out.extend(groups[t])
    return sorted(out)


def pickup_roots(notes, resolution: int):
    """A lone pickup note right before a chord takes the chord's root.

    RBN: chart the 8th/16th pickup into a chord change as "the single
    note that corresponds to the root note of the chord you are moving
    to" - the hand is already travelling there. Applies only to an
    isolated sustainless single (air before it, chord within half a
    beat after it) so runs and riffs are untouched.
    """
    if not notes:
        return notes
    pos = _positions(notes)
    relane: dict[int, int] = {}
    for i in range(1, len(pos) - 1):
        tick, lanes, sus = pos[i]
        lane = _single_lane(pos[i])
        if lane is None or sus > 0:
            continue
        nxt_tick, nxt_lanes, _ = pos[i + 1]
        if len(nxt_lanes) < 2 or OPEN in nxt_lanes:
            continue
        if not (0 < nxt_tick - tick <= resolution // 2):
            continue
        if tick - pos[i - 1][0] < resolution:
            continue  # a real pickup is isolated; this is inside a line
        root = min(nxt_lanes)
        if root != lane:
            relane[tick] = root
    if not relane:
        return notes
    return sorted((t, relane.get(t, lane) if lane != OPEN else lane, sus)
                  for t, lane, sus in notes)
