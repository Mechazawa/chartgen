"""Simulate a human playing the chart, and flag what cannot be played.

Every other quality gate here measures STATISTICS — density, chord share,
fret spread — and a chart can satisfy all of them and still contain a passage
no hand can execute. This models the hands instead: between two positions a
player has to move/re-shape the fretting hand and (unless the note hammers on)
strum. Each of those costs time, and the chart hands you a fixed amount.

The costs below are calibrated so that real human charts come out almost
entirely feasible — the point of reference is what charters actually ask of
players, not a theory of finger speed.
"""

from itertools import pairwise

OPEN = 7

# Seconds. A sustained alternate-strum tops out near 12-13 notes/sec for
# strong players, so ~0.075s is the floor between strums.
STRUM = 0.075
# Moving between frets is cheap on a 5-fret neck — four fingers cover it — so
# this is finger reassignment, not hand travel.
PER_LANE = 0.015
# Re-shaping the hand for a different chord costs far more than moving one
# finger, which is why rapid chord changes were the unplayable part.
SHAPE_CHANGE = 0.045


def _anchor(lanes):
    fretted = [l for l in lanes if l != OPEN]
    return min(fretted) if fretted else 0


def cost(prev_lanes, lanes, is_hopo: bool) -> float:
    """Seconds a player needs to get from one position to the next."""
    needed = 0.0 if is_hopo else STRUM
    prev_fret = {l for l in prev_lanes if l != OPEN}
    cur_fret = {l for l in lanes if l != OPEN}
    if prev_fret and cur_fret and (cur_fret <= prev_fret
                                   or prev_fret <= cur_fret):
        # One position is a subset of the other: fingers stay planted and
        # one lifts or lands - not a re-shape, and the hand does not move
        # even though min() says the "anchor" changed. Charging the full
        # shape cost here flagged 29% of an EDM chart's transitions
        # impossible (chord-stab riffs alternate G+R and single at 16ths,
        # which humans play by releasing a finger) and enforcement then
        # demoted the stabs to roots - gutting exactly the intense
        # sections. Chord-stab mixes are a documented human staple (5 per
        # 100 bars median, 27 on punk).
        return needed + PER_LANE * len(cur_fret ^ prev_fret)
    needed += abs(_anchor(lanes) - _anchor(prev_lanes)) * PER_LANE
    prev_shape = tuple(sorted(l - _anchor(prev_lanes) for l in prev_lanes))
    shape = tuple(sorted(l - _anchor(lanes) for l in lanes))
    if shape != prev_shape:
        needed += SHAPE_CHANGE * max(len(shape), len(prev_shape))
    return needed


def analyse(notes, tempo, hopo_ticks: int | None = None) -> dict:
    """Fraction of transitions a human cannot make, plus the worst moments.

    hopo_ticks defaults to the .chart natural-HOPO window, since a hammer-on
    costs no strum and that materially changes what is possible.
    """
    res = tempo.resolution
    if hopo_ticks is None:
        hopo_ticks = (65 * res) // 192

    by_tick: dict[int, set] = {}
    for tick, lane, _ in notes:
        by_tick.setdefault(tick, set()).add(lane)
    ticks = sorted(by_tick)
    if len(ticks) < 2:
        return {"transitions": 0, "impossible": 0.0, "tight": 0.0, "worst": []}

    times = {t: tempo.beat_to_time(t / res) for t in ticks}
    impossible, tight, worst = 0, 0, []
    for prev, cur in pairwise(ticks):
        available = times[cur] - times[prev]
        if available <= 0:
            continue
        prev_lanes, lanes = by_tick[prev], by_tick[cur]
        is_hopo = (len(lanes) == 1 and len(prev_lanes) == 1
                   and lanes != prev_lanes and OPEN not in lanes
                   and (cur - prev) < hopo_ticks)
        needed = cost(prev_lanes, lanes, is_hopo)
        ratio = needed / available
        if ratio > 1.0:
            impossible += 1
            worst.append((ratio, times[cur]))
        elif ratio > 0.8:
            tight += 1

    total = len(ticks) - 1
    worst.sort(reverse=True)
    return {
        "transitions": total,
        "impossible": impossible / total,
        "tight": tight / total,
        "worst": [(round(r, 2), round(s, 1)) for r, s in worst[:5]],
    }


def enforce(notes, tempo, max_passes: int = 4, hopo_ticks: int | None = None):
    """Simplify the chart until the hands can keep up.

    Two remedies, gentlest first. A chord whose shape the player has no time
    to form collapses to its root — that removes the shape-change cost and
    keeps the melody. Only if a position is STILL unreachable, and sits off
    the beat/eighth grid, is it dropped: an off-grid note is ornament, while
    dropping downbeats would gut the rhythm.

    Deliberately conservative about single notes. A fast run of repeats on one
    lane genuinely needs a strum each time and cannot hammer on, so it reads
    as "impossible" here — but human charts do contain tremolo picking, and
    deleting it would be charting for the simulator instead of the player.
    """
    res = tempo.resolution
    if hopo_ticks is None:
        hopo_ticks = (65 * res) // 192

    current = list(notes)
    for _ in range(max_passes):
        by_tick: dict[int, list] = {}
        for tick, lane, sus in current:
            by_tick.setdefault(tick, []).append((lane, sus))
        ticks = sorted(by_tick)
        if len(ticks) < 2:
            return current

        times = {t: tempo.beat_to_time(t / res) for t in ticks}
        demote, drop = set(), set()
        for prev, cur in pairwise(ticks):
            available = times[cur] - times[prev]
            if available <= 0:
                continue
            prev_lanes = {l for l, _ in by_tick[prev]}
            lanes = {l for l, _ in by_tick[cur]}
            is_hopo = (len(lanes) == 1 and len(prev_lanes) == 1
                       and lanes != prev_lanes and OPEN not in lanes
                       and (cur - prev) < hopo_ticks)
            if cost(prev_lanes, lanes, is_hopo) <= available:
                continue
            # Simplify EITHER side: the shape-change cost belongs to the
            # pair, and only ever demoting the later position left the very
            # first note of a chart unfixable, since it is never a `cur`.
            if len(lanes) > 1 or len(prev_lanes) > 1:
                if len(lanes) > 1:
                    demote.add(cur)
                if len(prev_lanes) > 1:
                    demote.add(prev)
            elif cur % (res // 2) != 0:
                drop.add(cur)

        if not demote and not drop:
            return current

        rebuilt = []
        for tick in ticks:
            if tick in drop:
                continue
            group = by_tick[tick]
            if tick in demote:
                fretted = sorted((g for g in group if g[0] != OPEN))
                group = fretted[:1] or group[:1]
            rebuilt.extend((tick, lane, sus) for lane, sus in group)
        current = sorted(rebuilt)
    return current
