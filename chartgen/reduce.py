"""Derive Hard/Medium/Easy from Expert.

This replaces driving the vendored EasyChartGenerator (still available as
`derive_lower_tiers_vendored`), which had two measured defects on real songs:

- Its Hard rule is a pure grid filter — keep beats and 8ths, delete every 16th.
  Natural HOPOs only occur under a 162-tick gap, i.e. between 16ths, so Hard
  came out all strums: 3 HOPOs where Expert had 650. Real Hard charts lean on
  HOPOs; an all-strum Hard is needlessly tiring and feels wrong.
- Its lane remap collapses orange AND open onto green (4->0, 7->0, 3->1),
  piling notes onto one lane: Medium measured 0.72 variety and Easy 0.41
  against the 0.80 gate.

The rules here, per tier:

Hard    keep beats and 8ths; keep short 16th figures whole; thin long 16th
        runs by dropping only the "e" (first 16th after the beat), so every
        beat keeps a HOPO pair and density drops instead of rhythm dying.
        Chords capped at 2 notes.
Medium  quarter notes, with 8th-position notes kept only where a bar would
        otherwise go quiet. Chords capped at 2. No orange (order-preserving
        remap to G/R/Y/B, not a collapse).
Easy    on-beat singles at least 2 beats apart (1 beat where the song would
        otherwise go silent for a bar). G/R/Y only, order-preserving.
"""
from itertools import pairwise
from types import SimpleNamespace

import re

NOTE_RE = re.compile(r"(\d+)\s*=\s*N\s+(\d+)\s+(\d+)")
OPEN = 7

# Order-preserving compressions: relative pitch motion survives, one adjacent
# pair merges (which reads as a repeat — what real parts do anyway). Opens map
# to green below Expert; sub-Expert opens are unusual in real charts.
MEDIUM_LANES = {0: 0, 1: 1, 2: 2, 3: 2, 4: 3, OPEN: 0}
EASY_LANES = {0: 0, 1: 1, 2: 1, 3: 2, 4: 2, OPEN: 0}

Note = tuple[int, int, int]


def _by_tick(notes: list[Note]) -> dict[int, list[Note]]:
    out: dict[int, list[Note]] = {}
    for note in notes:
        out.setdefault(note[0], []).append(note)
    return out


def _sixteenth_chains(ticks: list[int], resolution: int) -> dict[int, int]:
    """tick -> length of the dense run it belongs to.

    A "dense run" is consecutive positions closer than an 8th — 16ths on the
    straight grid, triplet 8ths on the triplet grid. Run length decides whether
    a figure is kept whole (short lick) or thinned (wall of 16ths).
    """
    chain_len: dict[int, int] = {}
    run = [ticks[0]] if ticks else []
    for prev, cur in pairwise(ticks):
        if cur - prev < resolution // 2:
            run.append(cur)
        else:
            for t in run:
                chain_len[t] = len(run)
            run = [cur]
    for t in run:
        chain_len[t] = len(run)
    return chain_len


def _cap_chord(group: list[Note], max_notes: int) -> list[Note]:
    """Keep the lowest lanes: the root carries the motion of a chord line."""
    fretted = sorted((n for n in group if n[1] != OPEN), key=lambda n: n[1])
    opens = [n for n in group if n[1] == OPEN]
    kept = (opens[:1] + fretted)[:max_notes] if not fretted else fretted[:max_notes]
    return kept or group[:max_notes]


def _remap(notes: list[Note], table: dict[int, int]) -> list[Note]:
    return [(t, table[lane], s) for t, lane, s in notes]


def _dedupe(notes: list[Note]) -> list[Note]:
    """Remapping can land two chord notes on one lane; keep the longer."""
    best: dict[tuple[int, int], Note] = {}
    for note in notes:
        key = (note[0], note[1])
        if key not in best or note[2] > best[key][2]:
            best[key] = note
    return sorted(best.values())


HARD_MAX_NPS = 4.5  # real Hard charts cap around here regardless of genre


def _cap_bar_density(ticks: list[int], resolution: int, per_bar_cap: int) -> list[int]:
    """Enforce a per-bar position budget: fine positions go first, then every
    other off-8th. Beats are never dropped."""
    bar = 4 * resolution
    capped: list[int] = []
    i = 0
    while i < len(ticks):
        bar_start = (ticks[i] // bar) * bar
        bar_ticks = [t for t in ticks[i:] if t < bar_start + bar]
        i += len(bar_ticks)
        over = len(bar_ticks) - per_bar_cap
        if over > 0:
            fine = [t for t in bar_ticks if t % resolution not in (0, resolution // 2)]
            drop = set(fine[:over])
            over -= len(drop)
            if over > 0:
                off8 = [t for t in bar_ticks
                        if t % resolution == resolution // 2 and t not in drop]
                drop.update(off8[1::2][:over])
            bar_ticks = [t for t in bar_ticks if t not in drop]
        capped.extend(bar_ticks)
    return capped


def thin_to_rating(expert: list[Note], resolution: int, tempo,
                   target: int) -> tuple[list[Note], int]:
    """Thin an Expert track until its predicted tier drops to `target`.

    Tightens a per-bar position budget one notch at a time, re-rating after
    each cut, so the chart loses exactly as much as the target demands and no
    more. Returns (notes, achieved_rating); achieved may sit above target
    when even a beats-only chart still rates higher (short fast songs).
    """
    from . import rating

    groups = _by_tick(expert)
    current = expert
    achieved = rating.rate_expert(current, tempo)
    per_bar = 16
    while achieved > target and per_bar > 2:
        per_bar -= 1
        kept = _cap_bar_density(sorted(groups), resolution, per_bar)
        current = [n for n in expert if n[0] in set(kept)]
        achieved = rating.rate_expert(current, tempo)
    return current, achieved


def reduce_hard(expert: list[Note], resolution: int,
                bpm: float | None = None) -> list[Note]:
    groups = _by_tick(expert)
    ticks = sorted(groups)
    chains = _sixteenth_chains(ticks, resolution)

    kept: list[int] = []
    for tick in ticks:
        pos = tick % resolution
        fine = pos not in (0, resolution // 2)  # not a beat, not an 8th
        # Long saturated runs: drop the fine positions in the first half of
        # the beat. Straight grid: the "e" goes, the "a" stays -> the beat
        # keeps one 120-tick HOPO pair. Short figures are kept whole.
        if fine and chains.get(tick, 1) > 4 and 0 < pos < resolution // 2:
            continue
        kept.append(tick)

    if bpm:
        # Playtest: on dense EDM the rule above barely reduces (92% of Expert
        # survived) because everything is beats and 8ths. Enforce a
        # notes-per-second ceiling per bar.
        per_bar_cap = max(4, int(round(HARD_MAX_NPS * 4 * 60.0 / bpm)))
        kept = _cap_bar_density(kept, resolution, per_bar_cap)

    out: list[Note] = []
    for tick in kept:
        out.extend(_cap_chord(groups[tick], 2))
    return sorted(out)


def reduce_medium(expert: list[Note], resolution: int) -> list[Note]:
    groups = _by_tick(expert)
    out: list[Note] = []
    for tick in sorted(groups):
        pos = tick % resolution
        # Beats and 8ths, deliberately generous: the ratio cap in derive_tiers
        # trims whatever this produces down to the measured 55% of Expert. A
        # strict rule cannot be trimmed UP, and on songs with sparse downbeats
        # it left Medium sitting on top of Easy (33% vs 32%) — the scanner's
        # difficultyNotReduced error.
        if pos not in (0, resolution // 2):
            continue
        out.extend(_cap_chord(groups[tick], 2))
    return _dedupe(_remap(out, MEDIUM_LANES))


def reduce_easy(expert: list[Note], resolution: int) -> list[Note]:
    groups = _by_tick(expert)
    out: list[Note] = []
    last_kept = -resolution * 8
    for tick in sorted(groups):
        pos = tick % resolution
        on_beat = pos == 0
        gap = tick - last_kept
        # Downbeats carry Easy; an 8th position is allowed only to break a
        # silence of two beats or more. Like Medium this is deliberately
        # generous and trimmed to the measured 39% of Expert afterwards.
        #
        # Half-note pacing came from the RBN guideline ("leave half note
        # spaces between strums") and measured far too sparse against real
        # Clone Hero charts — 0.5 notes/sec where they average ~1.8.
        keep = on_beat or (gap >= 2 * resolution and pos == resolution // 2)
        if not keep:
            continue
        out.extend(_cap_chord(groups[tick], 1))
        last_kept = tick
    return _dedupe(_remap(out, EASY_LANES))


def enforce_chord_rules(tiers: dict[str, list[Note]]) -> dict[str, list[Note]]:
    """Remove chord shapes the community standards call unplayable.

    The Chorus Encore scanner errors on a Green+Orange two-note chord on Hard,
    and the RBN guidelines keep three-note chords spanning green to orange off
    Expert as well ("as sparingly as possible" even as a pair). Medium adds no
    Green+Blue. These are hand contortions rather than musical choices, so the
    outer note moves inward one lane and keeps its sustain.
    """
    out: dict[str, list[Note]] = {}
    for name, notes in tiers.items():
        groups = _by_tick(notes)
        fixed: list[Note] = []
        for tick in sorted(groups):
            held = {lane: sus for _, lane, sus in groups[tick]}
            fretted = sorted(l for l in held if l != OPEN)

            spans_neck = 0 in held and 4 in held
            if spans_neck and (
                (name == "HardSingle" and len(fretted) == 2)
                or (name == "ExpertSingle" and len(fretted) >= 3)
            ):
                sus = held.pop(4)
                held.setdefault(3, sus)
            if name == "MediumSingle" and 0 in held and 3 in held:
                sus = held.pop(3)
                held.setdefault(2, sus)

            fixed.extend((tick, lane, held[lane]) for lane in sorted(held))
        out[name] = fixed
    return out



# Median position counts relative to Expert across 49 human charts in the
# local library that carry a full four-tier ladder. Grid rules alone drifted
# well off these (Hard landed at 96% of Expert on one song), so the ratios are
# enforced as a cap rather than hoped for.
# Median position counts vs Expert across 800 genre-balanced full-ladder
# charts (2026-08-24 calibration set). The original 49-chart numbers
# (.78/.55/.39) were within a few points - one of the few thresholds the
# bigger data validated rather than overturned.
TIER_RATIO = {"HardSingle": 0.81, "MediumSingle": 0.58, "EasySingle": 0.41}
# Anti-spike floor: no tier may fall below this share of the tier above it.
# Measured on the same 800 charts, the steepest adjacent step humans take is
# p10 = 0.53 (Easy/Medium); three unrelated ecosystems (Rocksmith's DDC
# tool, osu!mania's per-tier caps, BandFuse's reviews) independently name
# tier-to-tier difficulty cliffs as the failure players actually notice,
# separately from each tier's absolute density.
MIN_ADJACENT_RATIO = 0.53


def _thin_to_ratio(ticks: list[int], resolution: int, target: int) -> list[int]:
    """Drop the least structural positions until at most `target` remain.

    Fine positions go before 8ths and 8ths before beats, so the pulse of the
    song survives even a heavy cut, and within each class the drops are spread
    evenly rather than taken from one stretch.
    """
    if len(ticks) <= target:
        return ticks

    def rank(tick: int) -> int:
        pos = tick % resolution
        if pos == 0:
            return 2                     # downbeats are the last to go
        return 1 if pos == resolution // 2 else 0

    keep = set(range(len(ticks)))
    need = len(ticks) - target
    for level in (0, 1, 2):
        if need <= 0:
            break
        idx = [i for i in sorted(keep) if rank(ticks[i]) == level]
        if not idx:
            continue
        take = min(need, len(idx))
        step = len(idx) / take
        for j in range(take):
            keep.discard(idx[int(j * step)])
        need -= take
    return [t for i, t in enumerate(ticks) if i in keep]


def derive_tiers(expert: list[Note], resolution: int,
                 bpm: float | None = None) -> dict[str, list[Note]]:
    """{'HardSingle': ..., 'MediumSingle': ..., 'EasySingle': ...}"""
    tiers = {
        "HardSingle": reduce_hard(expert, resolution, bpm=bpm),
        "MediumSingle": reduce_medium(expert, resolution),
        "EasySingle": reduce_easy(expert, resolution),
    }
    expert_positions = len({t for t, _, _ in expert})
    out: dict[str, list[Note]] = {}
    # Walk down the ladder: each tier is capped at its measured share of
    # Expert AND forced strictly below the tier above it. Without the second
    # clause a uniformly dense song collapses Medium and Easy onto the same
    # "every beat" grid, which is the scanner's `difficultyNotReduced` error.
    previous = expert_positions
    for name in ("HardSingle", "MediumSingle", "EasySingle"):
        notes = tiers[name]
        target = max(1, min(int(expert_positions * TIER_RATIO[name]), previous - 1))
        # The ratio targets are Expert-relative; when something upstream has
        # shrunk a middle tier (a --target-diff pass, NPS caps), the next
        # tier's Expert-relative target can open a cliff. Clamp the step.
        target = min(max(target, int(MIN_ADJACENT_RATIO * previous)), max(1, previous - 1))
        kept = set(_thin_to_ratio(sorted(_by_tick(notes)), resolution, target))
        out[name] = [n for n in notes if n[0] in kept]
        previous = len(kept)
    return out


def derive_lower_tiers_vendored(expert_chart_text: str) -> dict[str, list[Note]]:
    """The original EasyChartGenerator path, kept for A/B comparison."""
    import easygen

    options = SimpleNamespace(
        force=True, bpm_multiplier=1, doublekick=0,
        easy=False, medium=False, hard=False,
    )
    parser = easygen.Parser(options)
    # It expects pre-stripped lines with any BOM already removed.
    parser.parse_file([ln.strip().replace("﻿", "") for ln in expert_chart_text.splitlines()])

    tiers = {}
    for part_name, lines in parser.new_parts.items():
        notes = [
            (int(m.group(1)), int(m.group(2)), int(m.group(3)))
            for m in (NOTE_RE.match(ln) for ln in lines) if m
        ]
        tiers[part_name.strip("[]")] = sorted(notes)
    return tiers


def simplify_rapid_chords(notes: list[Note], resolution: int,
                          max_chord: int = 3, tempo=None,
                          chord_spans=None) -> list[Note]:
    """Demote chords that arrive too fast to re-shape the hand for.

    `chord_spans`: optional [(tick0, tick1)] where the followed-instrument
    timeline says a chord instrument (guitar) leads. Inside them the
    16th-single-beside-a-chord case resolves the other way round - the
    single takes the neighbouring chord's shape - and a same-run off-grid
    chord is not a stray. Chord audit over 16 songs (2026-09-09): this
    pass culled a median 90 chords per song, mostly by the flicker and
    stray rules, taking guitar songs from 35-55% chords to 12-22%
    against human 40-60%, while on keys/synth songs the same cull was
    right (Millington 34% -> 10%, human 9%). The instrument decides.

    Playtest: three charts were unplayable inside 30 seconds — "combos very
    complex and rapid, some seemed impossible". Against the human charts of
    the same songs ours carried 41-54% chords where they used 5-13%, and one
    had 192 chord-to-chord shape changes inside an eighth note where the human
    chart had none.

    Two eras of this rule. The original (tempo=None, kept as the fallback)
    was a blunt clamp: any chord within a half-beat of a DIFFERENT chord, or
    off the eighth grid, collapsed to its root — installed when the
    playability simulator could not be trusted to catch the contortions
    itself (the finger-lift bug meant it flagged legitimate stabs and missed
    nothing either way). With tempo, the rule defers to the FIXED simulator:
    a chord is demoted only when forming it from the previous position
    genuinely costs more time than the chart provides. Off-grid push chords
    that the hand can reach (Mary Jane's chorus lives on them) survive;
    every real contortion still dies, just surgically.
    """
    from . import playability

    groups = _by_tick(notes)
    # Off-grid chords: the old blanket demotion of every chord off the 8th
    # grid turned out to be the main EDM chord suppressor (removing it took
    # a test chart from 18% to 29% chords against a 5-13% human band) - it
    # enforces the real "chords are accents on beats and 8ths" norm. But it
    # also killed deliberate push chords (Mary Jane's chorus anticipates the
    # beat bar after bar). Same doctrine as motifs.settle_pushes: recurrence
    # is the evidence of intent. An off-grid chord whose bar-offset recurs
    # in a neighbouring bar is a push and lives; a one-off pad hit demotes.
    bar = 4 * resolution
    off_grid_chords = {t for t, g in groups.items()
                       if sum(1 for n in g if n[1] != OPEN) > 1
                       and t % (resolution // 2) != 0}
    push_ok = {t for t in off_grid_chords
               if any((t + d * bar) in off_grid_chords
                      and (t + d * bar) % bar == t % bar
                      for d in (-2, -1, 1, 2))}
    # A chord with a SINGLE within a 16th on either side demotes to its
    # root. Bisected on Clocks (2026-09-03): the playability fix c5ce11b
    # (a subset transition is a finger lift, not a re-shape) was right
    # for chord stabs but let chord-single-chord flicker at 16th spacing
    # survive - 3 -> 23 flickers, 0% -> 6% of positions - which the
    # player called "notes right beside chords that felt random and hard
    # to hit". Humans do not write it: across 335 chord-heavy library
    # charts the median share of singles within a 16th of a chord is
    # 0.2% (p90 5.5%) and chord-single-chord flickers median 0 per 1000
    # positions (p90 3.6). The chord goes, not the single: Basic Pitch
    # hears an arpeggio as on-beat chords with 16th singles between, and
    # a charter writes the arpeggio as the single-note run it is.
    ordered = sorted(groups)
    beside = resolution // 4
    single_ticks = {t for t in ordered
                    if sum(1 for n in groups[t] if n[1] != OPEN) <= 1}
    in_span = (lambda t: any(a <= t < b for a, b in chord_spans)) if chord_spans else (lambda t: False)
    # Inside a chord-instrument span a lone single within a 16th of a
    # chord is the missed second voice: it takes that chord's shape.
    # Any-neighbour, not only sandwiched: over the nine guitar-led songs
    # of the chord audit (2026-09-09) the summed chord-share error vs the
    # human charts is 166 points for any-neighbour, 183 for sandwiched-
    # only, 204 with no promotion. Seven of nine sit BELOW their charter,
    # so under-chording is the systematic direction; the two that
    # overshoot (Thrice, Dream Chaser) over-chord at the transcription
    # stage already, which no culling rule can distinguish from a
    # chord-heavy charter.
    promote_to: dict[int, tuple] = {}
    for i, t in enumerate(ordered):
        if t not in single_ticks or not in_span(t):
            continue
        for j in (i - 1, i + 1):
            if 0 <= j < len(ordered) and ordered[j] not in single_ticks \
                    and abs(ordered[j] - t) <= beside:
                shape = tuple(sorted(n[1] for n in groups[ordered[j]] if n[1] != OPEN))
                if 1 < len(shape) <= max_chord:
                    promote_to[t] = shape
                    break
    shape_of = {t: frozenset(n[1] for n in groups[t] if n[1] != OPEN)
                for t in ordered}
    # A demoted chord is itself a single afterwards, so its own 16th
    # chord neighbours must be judged against that - but only when they
    # are a DIFFERENT shape. An arpeggio heard as varying chord pairs
    # unravels into the single-note run it is; a same-shape 16th chug
    # touching one stray single keeps its chords.
    flicker_chords = set()
    frontier = set(single_ticks) - set(promote_to)
    while frontier:
        new_frontier = set()
        for i, t in enumerate(ordered):
            if t in single_ticks or t in flicker_chords or in_span(t):
                continue
            for j in (i - 1, i + 1):
                if not 0 <= j < len(ordered) or ordered[j] not in frontier:
                    continue
                if abs(ordered[j] - t) > beside:
                    continue
                if (ordered[j] in flicker_chords
                        and shape_of[ordered[j]] == shape_of[t]):
                    continue  # same-shape chug: do not cascade
                flicker_chords.add(t)
                new_frontier.add(t)
                break
        frontier = new_frontier

    out: list[Note] = []
    prev_tick, prev_shape = None, None
    for tick in ordered:
        group = groups[tick]
        fretted = sorted((n for n in group if n[1] != OPEN), key=lambda n: n[1])
        opens = [n for n in group if n[1] == OPEN]
        kept = fretted[:max_chord] if fretted else group[:max_chord]
        shape = tuple(n[1] for n in kept)

        if tick in promote_to and fretted:
            sus = fretted[0][2]
            kept = [(tick, lane, sus) for lane in promote_to[tick]]
            shape = promote_to[tick]

        if len(shape) > 1 and tick in flicker_chords:
            kept = kept[:1]
            shape = tuple(n[1] for n in kept)

        if len(shape) > 1 and prev_shape is not None:
            # Style rule in BOTH eras: a different chord shape within a
            # half-beat of the last one. Humans strum same-shape runs
            # (59.7% of adjacent chord pairs) and avoid rapid re-shapes;
            # relaxing this to pure pairwise reachability flooded an EDM
            # test chart with 42 fast shape changes the simulator approved
            # individually - each reachable, the stream exhausting (the
            # cost model has no fatigue).
            rapid_change = (len(prev_shape) > 1 and shape != prev_shape
                            and tick - prev_tick < resolution // 2)
            if tempo is not None:
                available = (tempo.beat_to_time(tick / resolution)
                             - tempo.beat_to_time(prev_tick / resolution))
                # A same-shape hit within a 16th of the previous chord is
                # a strum continuation, not a stray pad hit: demoting it
                # manufactures the chord-single-chord flicker this pass
                # exists to remove (Clocks: the last 8 flickers were all
                # this rule's own doing inside same-shape 16th runs).
                stray = (tick in off_grid_chords and tick not in push_ok
                         and not in_span(tick)
                         and not (shape == prev_shape
                                  and tick - prev_tick <= beside))
                too_fast = rapid_change or stray or playability.cost(
                    set(prev_shape), set(shape), is_hopo=False) > available
            else:
                too_fast = rapid_change or (tick % (resolution // 2) != 0
                                            and not in_span(tick))
            if too_fast:
                kept = kept[:1]       # keep the root; the melody survives
                shape = tuple(n[1] for n in kept)

        out.extend(kept)
        out.extend(opens[:1] if not kept else [])
        prev_tick, prev_shape = tick, shape
    return sorted(out)
