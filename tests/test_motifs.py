"""Self-check for the flow-motif layer and riff unification.

Run: python -m tests.test_motifs
"""
import sys
from itertools import pairwise
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chartgen.chart import write_chart  # noqa: E402
from chartgen.motifs import (  # noqa: E402
    OPEN,
    break_machine_gun,
    chunk_rolls,
    legato_forcing,
    legato_stairs,
    pickup_roots,
    settle_pushes,
    shape_chord_walks,
    CHORD_RUNGS,
)
from chartgen.structure import unify_riff_bars  # noqa: E402
from chartgen.tempo import RESOLUTION, TempoMap  # noqa: E402
from chartgen.transcribe import (  # noqa: E402
    LEGAL_TRIPLES,
    admit_stem_events,
    bass_fallback_events,
    expert_from_notes,
    extend_ladders,
    merge_triple_evidence,
    promote_triple_runs,
    starved_runs,
    swung_beats,
    triple_song_evidence,
)

RES = RESOLUTION  # 480


def lanes_at(notes, tick):
    return sorted(lane for t, lane, _ in notes if t == tick)


def sus_at(notes, tick):
    return max((s for t, _, s in notes if t == tick), default=None)


def invariants(notes, res=RES):
    """Properties every transform must preserve."""
    seen = set()
    by_tick = {}
    for t, lane, sus in notes:
        assert t >= 0 and sus >= 0, (t, sus)
        assert 0 <= lane <= 4 or lane == OPEN, lane
        assert (t, lane) not in seen, f"duplicate note {(t, lane)}"
        seen.add((t, lane))
        by_tick.setdefault(t, []).append(sus)
    ticks = sorted(by_tick)
    for a, b in pairwise(ticks):
        assert max(by_tick[a]) <= b - a, \
            f"sustain at {a} overlaps next note at {b}"


# ---------------------------------------------------------------- stairs

def test_stairs_hold_stepped_phrase():
    beat = RES
    notes = [(0, 0, 0), (beat, 1, 0), (2 * beat, 2, 0), (3 * beat, 3, 0)]
    out = legato_stairs(notes, RES, bpm=120.0)
    # every phrase note but the last is held to just before its successor
    for tick in (0, beat, 2 * beat):
        s = sus_at(out, tick)
        assert s and s >= RES // 2, (tick, s)
        assert s < beat, s
    assert sus_at(out, 3 * beat) == 0
    invariants(out)


def test_stairs_fire_on_stepped_eighths_at_moderate_tempo():
    # The shipped-bug regression: at 84 BPM an 8th is 357ms and holding it
    # to ~80ms before the next note leaves ~270ms - over the 200ms floor,
    # so stepped 8th phrases MUST produce stairs (the human chart's own
    # stair sustains there are ~236ms).
    eighth = RES // 2
    notes = [(i * eighth, [0, 1, 2, 1, 0][i], 0) for i in range(5)]
    out = legato_stairs(notes, RES, bpm=84.0)
    held = [t for t, _, s in out if s > 0]
    assert len(held) >= 3, out


def test_stairs_leave_short_runs_and_wide_leaps():
    beat = RES
    # only two stepped notes: below STAIR_MIN_RUN
    out = legato_stairs([(0, 0, 0), (beat, 1, 0)], RES, bpm=120.0)
    assert all(s == 0 for _, _, s in out)
    # a 3-lane leap is not a stair
    out = legato_stairs([(0, 0, 0), (beat, 3, 0), (2 * beat, 4, 0),
                         (3 * beat, 3, 0)], RES, bpm=120.0)
    assert sus_at(out, 0) == 0


def test_stairs_respect_existing_and_chords():
    beat = RES
    long_sus = 2 * beat
    notes = [(0, 0, long_sus), (beat, 1, 0), (2 * beat, 2, 0), (3 * beat, 3, 0)]
    out = legato_stairs(notes, RES, bpm=120.0)
    assert sus_at(out, 0) == long_sus  # never shrink a real sustain
    # a chord breaks the phrase
    notes = [(0, 0, 0), (beat, 1, 0), (beat, 2, 0), (2 * beat, 2, 0),
             (3 * beat, 3, 0)]
    out = legato_stairs(notes, RES, bpm=120.0)
    assert sus_at(out, 0) == 0


def test_stairs_too_fast_tempo_no_baby_sustains():
    # At 240 BPM a beat is 250ms; holding to ~80ms before the next note
    # leaves ~170ms - under the 200ms floor, so no stair at all.
    beat = RES
    notes = [(0, 0, 0), (beat, 1, 0), (2 * beat, 2, 0), (3 * beat, 3, 0)]
    out = legato_stairs(notes, RES, bpm=240.0)
    assert all(s == 0 for _, _, s in out)


# ---------------------------------------------------------------- rolls

def test_chunk_rolls_structures_a_shapeless_run():
    step = RES // 4
    lanes = [0, 2, 0, 3, 1, 4, 0, 3, 1, 2]
    notes = [(i * step, lane, 0) for i, lane in enumerate(lanes)]
    out = chunk_rolls(notes, RES)
    new = [lanes_at(out, i * step)[0] for i in range(len(lanes))]
    assert new != lanes, "run should have been restructured"
    deltas = [b - a for a, b in pairwise(new)]
    assert all(d != 0 for d in deltas), f"no same-lane repeats inside a run: {new}"
    nz = [d for d in deltas if d != 0]
    reversals = sum(1 for a, b in pairwise(nz) if a * b < 0)
    assert reversals <= 3, f"still zigzagging: {new}"
    assert all(abs(d) <= 2 for d in deltas), f"jump introduced: {new}"
    invariants(out)


def test_chunk_rolls_preserves_deliberate_patterns():
    step = RES // 4
    trill = [(i * step, [0, 1][i % 2], 0) for i in range(8)]
    assert chunk_rolls(trill, RES) == sorted(trill)
    pedal = [(i * step, [0, 2, 0, 3, 0, 4][i % 6], 0) for i in range(12)]
    assert chunk_rolls(pedal, RES) == sorted(pedal)
    gun = [(i * step, 2, 0) for i in range(8)]
    assert chunk_rolls(gun, RES) == sorted(gun)
    # an already-directional line stays
    roll = [(i * step, i % 5, 0) for i in range(10)]
    assert chunk_rolls(roll, RES) == sorted(roll)


def test_chunk_rolls_leaves_slow_and_short_material():
    beat = RES
    slow = [(i * beat * 2, [0, 3, 1, 4, 2, 0][i], 0) for i in range(6)]
    assert chunk_rolls(slow, RES) == sorted(slow)
    short = [(i * (RES // 4), [0, 3, 1, 4][i], 0) for i in range(4)]
    assert chunk_rolls(short, RES) == sorted(short)


# ---------------------------------------------------------------- machine gun

def test_machine_gun_thins_one_off_wall():
    step = RES // 4
    wall = [(i * step, 2, 0) for i in range(10)]
    out = break_machine_gun(wall, RES)
    kept = {t for t, _, _ in out}
    assert len(kept) < 10
    assert all(t % (RES // 2) == 0 for t in kept), "on-8th pulse must survive"
    invariants(out)


def test_machine_gun_keeps_recurring_chug():
    step = RES // 4
    bar = 4 * RES
    riff = [(i * step, 2, 0) for i in range(10)]
    again = [(bar + i * step, 2, 0) for i in range(10)]
    out = break_machine_gun(riff + again, RES)
    assert len(out) == 20, "a chug that recurs is deliberate"


def test_machine_gun_ignores_short_and_varied_runs():
    step = RES // 4
    short = [(i * step, 2, 0) for i in range(6)]
    assert break_machine_gun(short, RES) == sorted(short)
    varied = [(i * step, [2, 2, 2, 2, 3, 2, 2, 2][i % 8], 0) for i in range(10)]
    assert break_machine_gun(varied, RES) == sorted(varied)


# ---------------------------------------------------------------- pushes

def test_settle_pushes_snaps_lone_stray():
    eighth = RES // 2
    push = 2 * RES + RES // 4  # off-8th 16th, air on both sides
    notes = [(0, 0, 0), (RES, 1, 0), (push, 2, 0), (3 * RES, 3, 0)]
    out = settle_pushes(notes, RES)
    assert lanes_at(out, push) == [], "stray push should have moved"
    snapped = int(round(push / eighth)) * eighth
    assert lanes_at(out, snapped) == [2]
    invariants(out)


def test_settle_pushes_keeps_recurring_syncopation():
    bar = 4 * RES
    offset = 2 * RES + RES // 4
    notes = []
    for b in range(3):
        notes += [(b * bar, 0, 0), (b * bar + offset, 2, 0)]
    out = settle_pushes(notes, RES)
    for b in range(3):
        assert lanes_at(out, b * bar + offset) == [2], \
            "recurring push is deliberate syncopation"


def test_settle_pushes_ignores_notes_inside_runs():
    step = RES // 4
    run = [(RES + i * step, [0, 1, 2, 1][i % 4], 0) for i in range(8)]
    assert settle_pushes(run, RES) == sorted(run)


# ---------------------------------------------------------------- forcing

def test_forcing_flags_stepwise_eighth_phrase():
    eighth = RES // 2
    lanes = [0, 1, 2, 3, 2, 1]
    notes = [(i * eighth, lane, 0) for i, lane in enumerate(lanes)]
    forced = legato_forcing(notes, RES)
    assert 0 not in forced, "phrase-initial note keeps its strum"
    assert forced == {i * eighth for i in range(1, 6)}, forced


def test_forcing_never_touches_natural_hopos_or_chords():
    sixteenth = RES // 4  # under the HOPO threshold: already natural HOPOs
    fast = [(i * sixteenth, i % 3, 0) for i in range(8)]
    assert legato_forcing(fast, RES) == set()
    eighth = RES // 2
    chords = [(i * eighth, lane, 0) for i in range(6) for lane in (0, 1)]
    assert legato_forcing(chords, RES) == set()
    same = [(i * eighth, 2, 0) for i in range(6)]
    assert legato_forcing(same, RES) == set()


def test_forcing_respects_budget():
    eighth = RES // 2
    # one long phrase: 40 stepwise notes - the budget (max of 8 and 6% of
    # notes) truncates it to a legato prefix rather than skipping it.
    lanes = [0, 1, 2, 1, 0, 1, 2, 3, 2, 1] * 4
    notes = [(i * eighth, lane, 0) for i, lane in enumerate(lanes)]
    forced = legato_forcing(notes, RES)
    assert 0 < len(forced) <= 8, len(forced)
    assert 0 not in forced


# ---------------------------------------------------------------- chords

def test_chord_walk_takes_the_rung_ladder():
    beat = RES
    walk = []
    for i, anchor in enumerate((3, 2, 1, 0)):  # descending anchors
        walk += [(i * beat, anchor, 0), (i * beat, anchor + 1, 0)]
    out = shape_chord_walks(walk, RES)
    shapes = [tuple(lanes_at(out, i * beat)) for i in range(4)]
    for a, b in pairwise(shapes):
        assert b in [tuple(r) for r in CHORD_RUNGS], b
        moved = len(set(a) ^ set(b)) // 2
        assert moved == 1, f"{a}->{b} should move exactly one finger"
    anchors = [s[0] for s in shapes]
    assert anchors == sorted(anchors, reverse=True), "descent must survive"
    invariants(out)


def test_chord_walk_leaves_riffs_and_short_runs():
    beat = RES
    riff = [(i * beat, lane, 0) for i in range(4) for lane in (0, 1)]
    assert shape_chord_walks(riff, RES) == sorted(riff)
    two = [(0, 0, 0), (0, 1, 0), (beat, 1, 0), (beat, 2, 0)]
    assert shape_chord_walks(two, RES) == sorted(two)


def test_pickup_takes_next_chord_root():
    eighth = RES // 2
    notes = [(0, 3, 0), (4 * RES - eighth, 4, 0),
             (4 * RES, 1, 0), (4 * RES, 2, 0)]
    out = pickup_roots(notes, RES)
    assert lanes_at(out, 4 * RES - eighth) == [1], "pickup takes the root"
    # inside a run it is not a pickup
    run = [(4 * RES - 4 * eighth + i * eighth, 4, 0) for i in range(4)]
    notes = run + [(4 * RES, 1, 0), (4 * RES, 2, 0)]
    out = pickup_roots(notes, RES)
    assert lanes_at(out, 4 * RES - eighth) == [4]


# ---------------------------------------------------------------- riffs

def _profiles(kinds):
    """Synthetic bar profiles: same letter = identical audio."""
    out = []
    rng = {}
    for k in kinds:
        if k is None:
            out.append(None)
            continue
        if k not in rng:
            c = np.zeros(12); c[len(rng) % 12] = 1.0
            o = np.zeros(16); o[(2 * len(rng)) % 16] = 1.0
            rng[k] = (c, o)
        out.append(rng[k])
    return out


def test_riff_unify_stamps_consensus():
    bar = 4 * RES
    beat = RES
    canonical = [(0, 0, 0), (beat, 1, 0), (2 * beat, 0, 0), (3 * beat, 1, 0)]
    notes = []
    for b in range(4):  # four bars of the same riff, one with jitter
        for t, lane, s in canonical:
            jitter = 1 if (b == 2 and t == beat) else 0
            notes.append((b * bar + t, lane + jitter, s))
    out, stamped = unify_riff_bars(notes, _profiles("AAAA"), RES)
    assert stamped >= 3, stamped
    fps = [tuple(sorted((t % bar, l) for t, l, _ in out if t // bar == b))
           for b in range(4)]
    assert len(set(fps)) == 1, "all four bars must be identical after stamping"
    invariants(out)


def test_riff_unify_needs_a_real_cluster():
    bar = 4 * RES
    notes = [(b * bar, b % 5, 0) for b in range(4)]
    # all-different audio: nothing to unify
    out, stamped = unify_riff_bars(notes, _profiles("ABCD"), RES)
    assert stamped == 0 and out == sorted(notes)
    # a pair is not a riff
    out, stamped = unify_riff_bars(notes, _profiles("AABB"), RES)
    assert stamped == 0


def test_riff_unify_spares_outliers_and_empty_bars():
    bar = 4 * RES
    beat = RES
    notes = []
    for b in range(3):
        notes += [(b * bar, 0, 0), (b * bar + beat, 1, 0),
                  (b * bar + 2 * beat, 0, 0)]
    # bar 3 sounds the same but is charted completely differently (a fill)
    notes += [(3 * bar + i * (RES // 4), 4, 0) for i in range(6)]
    out, stamped = unify_riff_bars(notes, _profiles("AAAA"), RES)
    fill = [(t, l) for t, l, _ in out if t // bar == 3]
    assert fill == [(3 * bar + i * (RES // 4), 4) for i in range(6)], \
        "an outlier bar is a variation and must survive"
    # bar 4 with no notes stays empty even if audio matches
    out2, _ = unify_riff_bars(notes, _profiles("AAAAA"), RES)
    assert not any(t // bar == 4 for t, _, _ in out2)


def test_riff_unify_clamps_sustains_at_the_boundary():
    bar = 4 * RES
    beat = RES
    riff = [(0, 0, 3 * beat), (3 * beat, 1, 0)]
    notes = []
    for b in range(3):
        notes += [(b * bar + t, l, s) for t, l, s in riff]
    # the note right after bar 2's copy sits early in bar 3
    notes += [(3 * bar + RES // 4, 2, 0)]
    out, stamped = unify_riff_bars(notes, _profiles("AAAB"), RES)
    invariants(out)


# ---------------------------------------------------------------- swing

def steady(bpm=120.0, n=200, first=0.5):
    return TempoMap(beat_times=first + np.arange(n) * (60.0 / bpm),
                    pickup_beats=1)


def test_swing_detected_on_triplet_onsets():
    t = steady()
    events = []
    for b in range(2, 60):
        for frac in (0.0, 1 / 3, 2 / 3):
            events.append((t.beat_to_time(b + frac), 0.0, 60, 0.6))
    swing = swung_beats(events, t)
    assert len(swing) >= 40, len(swing)


def test_swing_not_detected_on_straight_or_sparse():
    t = steady()
    straight = [(t.beat_to_time(b + f), 0.0, 60, 0.6)
                for b in range(2, 60) for f in (0.0, 0.25, 0.5, 0.75)]
    assert swung_beats(straight, t) == set()
    sparse = [(t.beat_to_time(b + 1 / 3), 0.0, 60, 0.6) for b in range(2, 8)]
    assert swung_beats(sparse, t) == set()


def test_swing_beats_quantize_to_triplets():
    t = steady()
    events = []
    for b in range(2, 40):
        for frac in (0.0, 1 / 3, 2 / 3):
            events.append((t.beat_to_time(b + frac),
                           t.beat_to_time(b + frac) + 0.05, 60 + b % 12, 0.6))
    swing = swung_beats(events, t)
    notes = expert_from_notes(events, t, swing_beats=swing, ornaments=False)
    offs = {tick % RES for tick, _, _ in notes}
    assert offs <= {0, RES // 3, 2 * RES // 3}, offs


# ---------------------------------------------------------------- ornaments

def test_ornament_recovered_only_with_repeat_evidence():
    t = steady(bpm=120.0)
    bar_beats = 4
    events = []
    # main notes on beats, plus a strong 32nd flam before beat 2 of bars 1-3
    for b in range(1, 13):
        events.append((t.beat_to_time(b), t.beat_to_time(b) + 0.1, 60, 0.6))
    # a grace note is quieter than its main note (here: under the chord
    # amplitude ratio, so it merges away at the 16th grid) but still well
    # above the ornament evidence floor
    for barn in (1, 2, 3):
        beat = 1 + barn * bar_beats
        events.append((t.beat_to_time(beat - 1 / 8),
                       t.beat_to_time(beat - 1 / 8) + 0.05, 64, 0.35))
    notes = expert_from_notes(events, t, ornaments=True)
    orn = [tick for tick, _, _ in notes if tick % (RES // 8) == 0
           and tick % (RES // 4) != 0]
    assert orn, "recurring, loud, well-placed flams should be recovered"
    # without repetition: nothing
    events2 = [e for e in events][:13]
    events2.append((t.beat_to_time(5 - 1 / 8), t.beat_to_time(5 - 1 / 8) + 0.05,
                    64, 0.35))
    notes2 = expert_from_notes(events2, t, ornaments=True)
    orn2 = [tick for tick, _, _ in notes2 if tick % (RES // 8) == 0
            and tick % (RES // 4) != 0]
    assert not orn2, "a one-off flam is jitter, not an ornament"


def test_ornament_quiet_onsets_ignored():
    t = steady(bpm=120.0)
    events = []
    for b in range(1, 13):
        events.append((t.beat_to_time(b), t.beat_to_time(b) + 0.1, 60, 0.6))
    for barn in (1, 2, 3):
        beat = 1 + barn * 4
        events.append((t.beat_to_time(beat - 1 / 8),
                       t.beat_to_time(beat - 1 / 8) + 0.05, 64, 0.22))
    notes = expert_from_notes(events, t, ornaments=True)
    orn = [tick for tick, _, _ in notes if tick % (RES // 8) == 0
           and tick % (RES // 4) != 0]
    assert not orn, "a quiet 32nd is a ghost, not evidence"


# ---------------------------------------------------------------- fallback

def _song(t, melody_bars, bass_bars):
    """Synthetic events: 4 melody notes/bar in melody_bars, 4 bass
    notes/bar (pitch 30) in bass_bars."""
    ev = []
    for b in melody_bars:
        for k in range(4):
            s = t.beat_to_time(b * 4 + k)
            ev.append((s, s + 0.2, 60 + k, 0.6))
    for b in bass_bars:
        for k in range(4):
            s = t.beat_to_time(b * 4 + k)
            ev.append((s, s + 0.2, 30, 0.6))
    return ev


def test_fallback_admits_bass_only_in_starved_runs():
    t = steady()
    # melody plays bars 0-3, vanishes bars 4-7 while the bass keeps going
    ev = _song(t, melody_bars=range(0, 4), bass_bars=range(0, 8))
    extra = bass_fallback_events(ev, t)
    assert len(extra) == 16, len(extra)  # bars 4-7 only, 4 notes each
    bars = {int(t.time_to_beat(s) / 4) for s, _, _, _ in extra}
    assert bars == {4, 5, 6, 7}, bars
    assert all(p >= 40 for _, _, p, _ in extra), "must be octave-lifted"
    assert all((p - 30) % 12 == 0 for _, _, p, _ in extra), \
        "lift preserves pitch class"


def test_fallback_never_fires_on_short_gaps_or_beside_a_lead():
    t = steady()
    # a single starved bar (a breath) does not flip the chart to the bass
    ev = _song(t, melody_bars=[0, 1, 2, 4, 5], bass_bars=range(0, 6))
    assert bass_fallback_events(ev, t) == []
    # bass under a playing lead never gets admitted
    ev = _song(t, melody_bars=range(0, 8), bass_bars=range(0, 8))
    assert bass_fallback_events(ev, t) == []
    # starved bars with no bass either: nothing to admit
    ev = _song(t, melody_bars=range(0, 4), bass_bars=range(0, 4))
    assert bass_fallback_events(ev, t) == []


def test_starved_runs_found_and_bounded():
    t = steady()
    ev = _song(t, melody_bars=[0, 1, 2, 3, 8, 9], bass_bars=[])
    runs = starved_runs(ev, t)
    assert len(runs) == 1, runs
    t0, t1 = runs[0]
    assert abs(t0 - t.beat_to_time(16)) < 0.05, runs  # bars 4-7
    assert abs(t1 - t.beat_to_time(32)) < 0.05, runs
    # continuous melody: no starvation
    assert starved_runs(_song(t, melody_bars=range(8), bass_bars=[]), t) == []


def test_admit_stem_events_gates():
    t = steady()
    runs = [(t.beat_to_time(16), t.beat_to_time(32))]
    inside = t.beat_to_time(20)
    outside = t.beat_to_time(4)
    stem_ev = [
        (inside, inside + 0.3, 45, 0.5),        # good: admitted as-is
        (inside + 1.0, inside + 1.2, 30, 0.5),  # low: admitted octave-lifted
        (outside, outside + 0.3, 45, 0.5),      # outside starved run: no
        (inside + 2.0, inside + 2.1, 45, 0.05), # ghost: no
        (inside + 3.0, inside + 3.1, 20, 0.5),  # rumble: no
    ]
    out = admit_stem_events(stem_ev, runs, [], t)
    assert len(out) == 2, out
    assert {p for _, _, p, _ in out} == {45, 42}, out
    # a stem event whose 16th TICK the mix already charted is a twin, not a
    # rescue - even ~90ms away (the fake-chord bug: time-based dedupe let it
    # through and it quantized onto the same tick)
    kept = [(inside + 0.09, inside + 0.3, 60, 0.5)]
    out = admit_stem_events(stem_ev, runs, kept, t)
    assert all(t.quantize(s, subdiv=4) != t.quantize(inside + 0.09, subdiv=4)
               for s, _, _, _ in out), out
    # within ONE stem, two comparable notes of different pitch on an empty
    # tick are a chord (charters write chords where rescue used to write
    # singles); a same-pitch twin is still one note
    pair = [(inside, inside + 0.3, 45, 0.5),
            (inside + 0.01, inside + 0.2, 50, 0.5)]
    out = admit_stem_events(pair, runs, [], t)
    assert len(out) == 2, out
    twin = [(inside, inside + 0.3, 45, 0.5),
            (inside + 0.01, inside + 0.2, 45, 0.5)]
    assert len(admit_stem_events(twin, runs, [], t)) == 1
    # two STEMS cannot double-admit the same tick: the second stem sees the
    # first's admissions among the kept events and is refused there
    first = admit_stem_events([(inside, inside + 0.3, 45, 0.5)], runs, [], t)
    second = admit_stem_events([(inside + 0.01, inside + 0.2, 50, 0.5)], runs, first, t)
    assert second == [], second


def test_fallback_ignores_rumble_and_ghosts():
    t = steady()
    ev = _song(t, melody_bars=range(0, 2), bass_bars=[])
    for b in (2, 3, 4):
        for k in range(4):
            s = t.beat_to_time(b * 4 + k)
            ev.append((s, s + 0.2, 20, 0.6))   # sub-bass rumble: too low
            ev.append((s, s + 0.2, 30, 0.10))  # ghost: too quiet
    assert bass_fallback_events(ev, t) == []


# ---------------------------------------------------------------- ladders

def _ladder_setup(t, host_ring_beats):
    """Host G sustained at beat 4, R joins a 16th later, next G at beat 8.
    The transcribed event for the host rings for host_ring_beats."""
    beat = RES
    notes = [(4 * beat, 0, beat // 2), (4 * beat + beat // 4, 1, beat // 2),
             (6 * beat, 2, 0), (8 * beat, 0, 0)]
    host_t = t.beat_to_time(4)
    events = [(host_t, t.beat_to_time(4 + host_ring_beats), 50, 0.6),
              (t.beat_to_time(4.25), t.beat_to_time(5), 55, 0.6)]
    return notes, events


def test_ladder_extends_evidenced_hold_through_join():
    t = steady()
    notes, events = _ladder_setup(t, host_ring_beats=1.5)
    out, made = extend_ladders(notes, events, t)
    assert made == 1
    host = sus_at(out, 4 * RES)
    join_tick = 4 * RES + RES // 4
    assert host > join_tick - 4 * RES, "host must ring past the join"
    assert sus_at(out, join_tick) == RES // 2, "the join keeps its own hold"
    # tail stops before the next note on the host's own lane at beat 8
    assert 4 * RES + host < 8 * RES, out


def test_ladder_needs_duration_evidence_and_a_higher_join():
    t = steady()
    # host sound stops before the join: no licence to overlap
    notes, events = _ladder_setup(t, host_ring_beats=0.2)
    out, made = extend_ladders(notes, events, t)
    assert made == 0 and out == sorted(notes)
    # join BELOW the held lane: not a build-up
    beat = RES
    notes = [(4 * beat, 2, beat // 2), (4 * beat + beat // 4, 0, 0),
             (8 * beat, 2, 0)]
    events = [(t.beat_to_time(4), t.beat_to_time(6), 60, 0.6)]
    out, made = extend_ladders(notes, events, t)
    assert made == 0


def test_ladder_keeps_gyb_but_pulls_a_leap_inward():
    t = steady()
    beat = RES
    host_ev = [(t.beat_to_time(4), t.beat_to_time(5.5), 50, 0.6)]
    # G host, Y then B joins: every step <=2, the legitimate span-3 stack
    notes = [(4 * beat, 0, beat // 2), (4 * beat + beat // 4, 2, 0),
             (4 * beat + beat // 2, 3, 0), (8 * beat, 0, 0)]
    out, made = extend_ladders(notes, host_ev, t)
    assert made == 1
    assert lanes_at(out, 4 * beat + beat // 4) == [2], "G-Y-B must survive"
    assert lanes_at(out, 4 * beat + beat // 2) == [3]
    # G host, isolated B join (+3 leap): pulled inward to Y
    notes = [(4 * beat, 0, beat // 2), (4 * beat + beat // 4, 3, 0),
             (8 * beat, 0, 0)]
    out, made = extend_ladders(notes, host_ev, t)
    assert made == 1
    assert lanes_at(out, 4 * beat + beat // 4) == [2], out
    # same leap but the join is a chord: the ladder is vetoed, lanes stay
    notes = [(4 * beat, 0, beat // 2), (4 * beat + beat // 4, 3, 0),
             (4 * beat + beat // 4, 4, 0), (8 * beat, 0, 0)]
    out, made = extend_ladders(notes, host_ev, t)
    assert made == 0 and out == sorted(notes)


def test_ladder_respects_budget_and_spacing():
    t = steady()
    beat = RES
    notes = []
    events = []
    for k in range(6):  # six candidate ladders two beats apart
        base = (4 + 2 * k) * beat
        notes += [(base, 0, beat // 2), (base + beat // 4, 1, 0)]
        events.append((t.beat_to_time(base / RES),
                       t.beat_to_time(base / RES + 1.5), 50, 0.6))
    out, made = extend_ladders(notes, events, t)
    # 4-beat minimum spacing: of six candidates 2 beats apart, every
    # other one ladders
    assert made == 3, made


# ---------------------------------------------------------------- triples

def _triple_song_events(t, bars=6, third=True):
    """A chord riff: G+Y power chords on every 8th, with (optionally) a
    comparable third pitch, recurring bar after bar."""
    ev = []
    for b in range(bars):
        for k in range(8):
            s = t.beat_to_time(b * 4 + k * 0.5)
            ev.append((s, s + 0.2, 50, 0.6))
            ev.append((s, s + 0.2, 57, 0.55))
            if third:
                ev.append((s, s + 0.2, 62, 0.5))
    return ev


def test_triple_evidence_gates_bimodally():
    t = steady()
    yes = triple_song_evidence(_triple_song_events(t, third=True), t)
    assert yes["qualifies"], yes
    no = triple_song_evidence(_triple_song_events(t, third=False), t)
    assert not no["qualifies"], no
    # a scatter of one-off third pitches (no recurrence) must not qualify
    ev = _triple_song_events(t, third=False)
    for i, b in enumerate(range(0, 24, 3)):  # varying bar offsets, once each
        s = t.beat_to_time(b + (i % 7) * 0.5)
        ev.append((s, s + 0.2, 62, 0.5))
    scatter = triple_song_evidence(ev, t)
    assert scatter["recur"] < 0.5 or not scatter["qualifies"], scatter


def test_promote_triple_runs_uniform_and_legal():
    t = steady()
    ev = triple_song_evidence(_triple_song_events(t), t)
    assert ev["qualifies"]
    eighth = RES // 2
    notes = [(i * eighth, lane, 0) for i in range(12) for lane in (0, 2)]
    out, promoted = promote_triple_runs(notes, ev, t)
    assert promoted >= 1
    for i in range(12):
        lanes = tuple(lanes_at(out, i * eighth))
        assert len(lanes) == 3, f"run must promote uniformly: {lanes}"
        assert lanes in LEGAL_TRIPLES, lanes
    # non-qualifying song: untouched
    no = triple_song_evidence(_triple_song_events(t, third=False), t)
    out2, promoted2 = promote_triple_runs(notes, no, t)
    assert promoted2 == 0 and out2 == sorted(set(notes))


def test_promote_skips_offgrid_and_short_runs():
    t = steady()
    ev = triple_song_evidence(_triple_song_events(t), t)
    sixteenth = RES // 4
    # a 16th-offset chord run (off the 8th grid) stays 2-note
    notes = [(i * (RES // 2) + sixteenth, lane, 0)
             for i in range(12) for lane in (0, 2)]
    out, promoted = promote_triple_runs(notes, ev, t)
    assert promoted == 0
    # two chords only: below the run floor
    short = [(i * (RES // 2), lane, 0) for i in range(2) for lane in (0, 2)]
    out, promoted = promote_triple_runs(short, ev, t)
    assert promoted == 0


# ---------------------------------------------------------------- brightness

from chartgen.frets import (  # noqa: E402
    degenerate_stretches,
    relane_by_brightness,
)

SR_TEST = 22050


def _tone_track(t, note_ticks, freqs, dur=0.12):
    """Synthetic melodic audio: a sine burst at each note's time."""
    total = t.beat_to_time(note_ticks[-1] / RES) + 1.0
    buf = np.zeros(int(total * SR_TEST), dtype=np.float32)
    for tick, f in zip(note_ticks, freqs, strict=True):
        a = int(t.beat_to_time(tick / RES) * SR_TEST)
        n = int(dur * SR_TEST)
        x = np.arange(n)
        buf[a:a + n] += 0.5 * np.sin(2 * np.pi * f * x / SR_TEST).astype(np.float32)
    return buf


def _stuck_notes(n=20, lane=0, gap=RES // 2, sus_at_idx=None):
    return [(i * gap, lane, RES // 4 if i == sus_at_idx else 0) for i in range(n)]


def test_brightness_relanes_a_swept_stuck_stretch():
    t = steady()
    notes = _stuck_notes()
    ticks = [n[0] for n in notes]
    freqs = [200 * (2 ** (i / 4)) for i in range(len(ticks))]  # rising sweep
    audio = _tone_track(t, ticks, freqs)
    out, changed = relane_by_brightness(notes, t, audio, SR_TEST)
    assert changed == 1
    lanes = [l for _, l, _ in out]
    assert len(set(lanes)) >= 3, lanes
    assert lanes[0] < lanes[-1], f"rising sweep must rise: {lanes}"
    assert [n[0] for n in out] == ticks, "ticks never move"
    # determinism
    again, _ = relane_by_brightness(notes, t, audio, SR_TEST)
    assert again == out


def test_brightness_leaves_chugs_and_varied_lines():
    t = steady()
    notes = _stuck_notes()
    ticks = [n[0] for n in notes]
    drone = _tone_track(t, ticks, [400.0] * len(ticks))  # static: a chug
    out, changed = relane_by_brightness(notes, t, drone, SR_TEST)
    assert changed == 0 and out == sorted(notes)
    varied = [(i * RES // 2, i % 4, 0) for i in range(12)]  # already moving
    stretches, _ = degenerate_stretches(varied, RES)
    assert stretches == []
    silence = np.zeros(SR_TEST * 10, dtype=np.float32)
    out, changed = relane_by_brightness(notes, t, silence, SR_TEST)
    assert changed == 0, "silence has no contour to follow"
    out, changed = relane_by_brightness(notes, t, None, SR_TEST)
    assert changed == 0


def test_brightness_respects_chords_sustains_and_opens():
    t = steady()
    # chords break stretches: two 5-note stuck halves around a chord
    notes = _stuck_notes(5) + [(5 * RES // 2, 1, 0), (5 * RES // 2, 2, 0)]
    notes += [((6 + i) * RES // 2, 0, 0) for i in range(5)]
    stretches, _ = degenerate_stretches(sorted(notes), RES)
    assert stretches == [], "5-note fragments are below the floor"
    # sustains survive a relane; opens become fretted
    notes = _stuck_notes(sus_at_idx=3)
    notes[5] = (notes[5][0], 7, 0)  # an open inside the stretch
    ticks = [n[0] for n in notes]
    freqs = [200 * (2 ** (i / 4)) for i in range(len(ticks))]
    out, changed = relane_by_brightness(notes, t, _tone_track(t, ticks, freqs),
                                        SR_TEST)
    assert changed == 1
    assert sus_at(out, notes[3][0]) == RES // 4, "sustain length preserved"
    assert lanes_at(out, notes[5][0])[0] != 7, "open inside sweep gets a fret"


# ---------------------------------------------------------------- guitar

from chartgen.guitar import (  # noqa: E402
    chordify,
    guitar_share,
    second_voices,
)


def test_second_voices_needs_comparable_pair():
    t = steady()
    beat_s = t.beat_to_time(4)
    ev = [(beat_s, beat_s + 0.3, 50, 0.6), (beat_s, beat_s + 0.3, 57, 0.5),
          (t.beat_to_time(5), t.beat_to_time(5) + 0.3, 50, 0.6),
          (t.beat_to_time(5), t.beat_to_time(5) + 0.3, 62, 0.1)]  # ghost
    v = second_voices(ev, t)
    assert len(v) == 1
    (tick, (lo, hi)), = v.items()
    assert (lo, hi) == (50, 57)


def test_chordify_promotes_a_run_uniformly():
    eighth = RES // 2
    run = [(i * eighth, 1, 0) for i in range(6)]
    filler = [((20 + i) * RES, 0, 0) for i in range(20)]  # keeps the cap open
    # evidence on 4 of 6 positions, all fifths -> +2 for the whole run
    voices = {i * eighth: (50, 57) for i in (0, 1, 3, 5)}
    out, added = chordify(run + filler, voices, RES)
    assert added == 6, "the whole run promotes, not just evidenced ticks"
    for i in range(6):
        assert lanes_at(out, i * eighth) == [1, 3], i
    # a third-interval majority gives the adjacent shape instead
    voices = {i * eighth: (50, 53) for i in (0, 1, 3, 5)}
    out, _ = chordify(run + filler, voices, RES)
    assert lanes_at(out, 0) == [1, 2]


def test_chordify_skips_thin_evidence_and_short_runs():
    eighth = RES // 2
    run = [(i * eighth, 1, 0) for i in range(6)]
    filler = [((20 + i) * RES, 0, 0) for i in range(20)]
    # one evidenced tick out of six: below the run-support bar
    out, added = chordify(run + filler, {0: (50, 57)}, RES)
    assert added == 0
    # a two-position run is not a riff
    short = [(0, 1, 0), (eighth, 1, 0)]
    out, added = chordify(short + filler, {0: (50, 57), eighth: (50, 57)}, RES)
    assert added == 0


def test_chordify_keeps_one_direction_per_run():
    eighth = RES // 2
    # lanes 1,1,1,4: +2 fits three of four; the outlier must stay single
    lanes = [1, 1, 1, 4]
    run = [(i * eighth, lane, 0) for i, lane in enumerate(lanes)]
    filler = [((20 + i) * RES, 0, 0) for i in range(20)]
    voices = {i * eighth: (50, 57) for i in range(4)}
    out, added = chordify(run + filler, voices, RES)
    assert added == 3
    for i in range(3):
        assert lanes_at(out, i * eighth) == [1, 3]
    assert lanes_at(out, 3 * eighth) == [4], "no flipped shape on the outlier"


def test_chordify_leaves_chords_opens_and_respects_cap():
    eighth = RES // 2
    voices = {i * eighth: (50, 57) for i in range(6)}
    # existing chords and opens are never touched
    notes = [(i * eighth, lane, 0) for i in range(3) for lane in (1, 3)]
    notes += [((3 + i) * eighth, OPEN, 0) for i in range(3)]
    out, added = chordify(notes, voices, RES)
    assert added == 0
    # a chart already at the chord ceiling gains nothing
    dense = [(i * eighth, lane, 0) for i in range(10) for lane in (0, 2)]
    out, added = chordify(dense, {i * eighth: (50, 57) for i in range(10)}, RES)
    assert added == 0


def test_guitar_share_separates_silence():
    sr = 44100
    loud = np.random.RandomState(0).randn(sr * 20).astype(np.float32) * 0.1
    silent = np.zeros(sr * 20, dtype=np.float32)
    assert guitar_share(loud) > 0.5
    assert guitar_share(silent) == 0.0
    assert guitar_share(None) == 0.0


# ---------------------------------------------------------------- writing

def test_forced_flags_written_expert_only():
    t = steady()
    tiers = {
        "ExpertSingle": [(0, 0, 0), (RES, 1, 0)],
        "HardSingle": [(0, 0, 0), (RES, 1, 0)],
    }
    text = write_chart(tiers, t, {"name": "x", "artist": "y", "charter": "z"},
                       "song.opus", forced={RES})
    expert = text.split("[ExpertSingle]")[1].split("[HardSingle]")[0]
    hard = text.split("[HardSingle]")[1]
    assert f"{RES} = N 5 0" in expert
    assert "N 5" not in hard, "forcing is spacing-relative: Expert only"


def test_merge_triple_evidence_pools_sources():
    t = steady()
    mix = triple_song_evidence(_triple_song_events(t, third=True), t)
    # guitar stem hears the third voice on OTHER strums: the same riff on
    # the off-16ths, so its ticks are disjoint from the mix's 8ths
    shift = t.beat_to_time(0.25) - t.beat_to_time(0.0)
    shifted = [(s + shift, e + shift, p, a)
               for s, e, p, a in _triple_song_events(t, third=True)]
    gtr = triple_song_evidence(shifted, t)
    merged = merge_triple_evidence(mix, gtr, t)
    assert merged["qualifies"]
    assert len(merged["third_pitch"]) == len(mix["third_pitch"]) + len(gtr["third_pitch"])
    assert merged["guitar_ticks"] == len(gtr["third_pitch"])
    assert set(mix["third_pitch"]) <= set(merged["third_pitch"])
    # no guitar evidence: the mix evidence passes through untouched
    assert merge_triple_evidence(mix, None, t) is mix
    empty = triple_song_evidence([], t)
    assert merge_triple_evidence(mix, empty, t) is mix
    # a non-qualifying mix still qualifies when the stem certifies it
    weak = triple_song_evidence(_triple_song_events(t, third=False), t)
    assert not weak["qualifies"]
    assert merge_triple_evidence(weak, gtr, t)["qualifies"]


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
    print("\nall checks passed")
