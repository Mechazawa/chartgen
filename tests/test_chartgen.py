"""Self-check for the parts that would silently produce bad charts.

Run: python -m tests.test_chartgen
"""
import sys
from itertools import pairwise
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chartgen.chart import write_chart  # noqa: E402
from chartgen.expression import hopo_threshold, star_power_phrases  # noqa: E402
from chartgen.quality import step_profile, variety_score, walk_score  # noqa: E402
from chartgen.tempo import RESOLUTION, TempoMap  # noqa: E402


def steady(bpm=120.0, n=64, first=0.5):
    """Perfectly steady beat grid, the case we can reason about exactly."""
    return TempoMap(
        beat_times=first + np.arange(n) * (60.0 / bpm),
        pickup_beats=max(1, round(first / (60.0 / bpm))),
    )


def test_quantize_lands_on_grid():
    t = steady()
    beat = 60.0 / 120.0
    # A note 12ms late off beat 4 must snap to exactly beat 4.
    tick = t.quantize(t.beat_times[4] + 0.012, subdiv=4)
    assert tick % RESOLUTION == 0, f"expected an integer beat, got {tick / RESOLUTION}"
    assert tick == (4 + t.pickup_beats) * RESOLUTION, tick

    # Anything on the grid must stay on the grid: this is the property that was
    # broken before (2 of 180 notes on a beat) and that difficulty reduction needs.
    step = RESOLUTION // 4
    for i in range(len(t.beat_times) - 1):
        for frac in (0.0, 0.25, 0.5, 0.75):
            noisy = t.beat_times[i] + frac * beat + 0.008
            assert t.quantize(noisy, subdiv=4) % step == 0


def test_sync_track_collapses_steady_tempo():
    # Pickup is exactly one beat, so it needs no separate tempo event.
    events = steady(bpm=120.0, first=0.5).sync_track()
    assert events == [(0, 120000)], events

    # Pickup shorter than a beat needs its own event to stay in sync, but the
    # rest of the song still collapses to a single steady event.
    events = steady(bpm=120.0, first=0.3).sync_track()
    assert len(events) == 2, events
    assert events[0] == (0, 200000), events[0]
    assert events[1] == (RESOLUTION, 120000), events[1]


def test_sync_track_follows_drift():
    # Tempo ramps 120 -> 150; a single BPM event cannot represent this.
    times, t_cur = [], 0.0
    for i in range(48):
        times.append(t_cur)
        t_cur += 60.0 / (120.0 + i * 0.625)
    t = TempoMap(beat_times=np.array(times), pickup_beats=0)
    assert len(t.sync_track()) > 1, "drift should produce multiple tempo events"


def test_sync_track_absorbs_frame_jitter():
    """librosa snaps beats to ~23ms frames; that jitter is not tempo drift.

    This is the regression that read a 120.0 BPM click track as 117.5 BPM with
    74 tempo events.
    """
    hop = 512 / 22050.0  # librosa's default analysis frame, ~23.2ms
    true_times = np.arange(80) * 0.5  # exactly 120 BPM
    detected = np.round(true_times / hop) * hop

    t = TempoMap(beat_times=detected, pickup_beats=0)
    assert abs(t.bpm - 120.0) < 0.5, f"fitted tempo {t.bpm:.2f} should be ~120"

    events = t.sync_track()
    assert len(events) <= 2, f"jitter must not spawn tempo events: {len(events)}"

    # And the emitted map must track the TRUE grid, not the jittered detections.
    worst = max(abs(_replay(events, i) - true_times[i]) for i in range(len(true_times)))
    assert worst < 0.020, f"drifts {worst * 1000:.1f}ms from the true grid"


def _replay(events, beat_idx):
    """Playback time of a beat, as a game reading the SyncTrack would compute it."""
    tick = beat_idx * RESOLUTION
    played, prev_tick, prev_bpm = 0.0, 0, events[0][1] / 1000.0
    for ev_tick, ev_bpm in events[1:]:
        if ev_tick >= tick:
            break
        played += (ev_tick - prev_tick) / RESOLUTION * (60.0 / prev_bpm)
        prev_tick, prev_bpm = ev_tick, ev_bpm / 1000.0
    return played + (tick - prev_tick) / RESOLUTION * (60.0 / prev_bpm)


def test_sync_track_honours_drift_tolerance():
    """A wobbling tempo must be tracked to whatever tolerance we ask for."""
    times, t_cur = [], 0.0
    for i in range(64):
        times.append(t_cur)
        t_cur += 60.0 / (128.0 + 6.0 * np.sin(i / 5.0))
    t = TempoMap(beat_times=np.array(times), pickup_beats=0)

    tol = 0.005
    events = t.sync_track(max_drift_s=tol)
    worst = max(abs(_replay(events, i) - times[i]) for i in range(len(times)))
    assert worst <= tol + 0.001, f"tempo map desyncs by {worst * 1000:.1f}ms"


def test_write_chart_is_parseable_and_has_real_tempo():
    t = steady()
    text = write_chart(
        {"ExpertSingle": [(480, 0, 0), (600, 2, 0)]}, t,
        {"name": "T", "artist": "A", "charter": "c"}, "song.ogg",
    )
    assert "[ExpertSingle]" in text and "[SyncTrack]" in text
    assert f"Resolution = {RESOLUTION}" in text
    assert "B 200000" not in text, "must not fall back to audio2chart's fake 200 BPM"
    assert "B 120000" in text, "real detected tempo missing from SyncTrack"
    assert "480 = N 0 0" in text


def test_write_chart_emits_all_tiers_star_power_and_sections():
    t = steady()
    tiers = {
        "ExpertSingle": [(0, 0, 0), (480, 1, 0)],
        "HardSingle": [(0, 0, 0)],
        "MediumSingle": [(0, 0, 0)],
        "EasySingle": [(0, 0, 0)],
    }
    text = write_chart(
        tiers, t, {"name": "T", "artist": "A", "charter": "c"}, "song.ogg",
        star_power=[(0, 4 * RESOLUTION)], events=[(0, "Section 1")],
    )
    for tier in tiers:
        assert f"[{tier}]" in text, tier
        # Star power is per-track in .chart, so it must repeat in each block.
        block = text.split(f"[{tier}]")[1].split("}")[0]
        assert "= S 2 " in block, f"{tier} lost its star power phrase"
    assert '= E "section Section 1"' in text


def test_write_chart_numbers_ghl_lanes_and_keeps_forcing_off_them():
    """B3 is lane 5 in memory but N 8 in .chart, where 5 means forced."""
    tiers = {"ExpertSingle": [(480, 0, 0)],
             "ExpertGHLGuitar": [(480, 5, 0), (960, 7, 0), (1440, 3, 0)]}
    text = write_chart(tiers, steady(), {"name": "T", "artist": "A", "charter": "c"},
                       "song.ogg", forced={480})
    block = text.split("[ExpertGHLGuitar]")[1].split("}")[0]
    assert "480 = N 8 0" in block and "960 = N 7 0" in block and "1440 = N 3 0" in block
    assert "= N 5 " not in block, "6-fret block must not carry 5-fret forcing"
    assert "480 = N 5 0" in text.split("[ExpertSingle]")[1].split("}")[0]


def test_write_chart_keeps_taps_and_solos_off_ghl_tiers():
    tiers = {"ExpertSingle": [(480, 0, 0), (960, 1, 0)],
             "ExpertGHLGuitar": [(480, 0, 0), (960, 3, 0)],
             "HardGHLGuitar": [(480, 0, 0)]}
    text = write_chart(tiers, steady(), {"name": "T", "artist": "A", "charter": "c"},
                       "song.ogg", solos=[(0, 1920)], taps={480, 960})
    for tier in ("ExpertGHLGuitar", "HardGHLGuitar"):
        block = text.split(f"[{tier}]")[1].split("}")[0]
        assert "= N 6 " not in block and "E solo" not in block, tier
    five = text.split("[ExpertSingle]")[1].split("}")[0]
    assert "480 = N 6 0" in five and "0 = E solo" in five and "1920 = E soloend" in five


def test_ghl_tiers_follow_the_official_ladder_and_sustain_rules():
    """Expert notes sit on grid slots at least GAP_S apart; each lower tier
    is drawn from the one above it, up to the official share and never
    closer than its minimum gap; no tier emits a shape the official charts
    never use there (Easy is white-row only); and an Expert note with room
    after it sustains up to the next note unless it is an open note."""
    import tempfile

    import torch

    from chartgen import ghl

    torch.manual_seed(0)
    white = [1, 2, 4, 3, 6]
    allowed = {"Expert": list(range(64)), "Hard": list(range(64)),
               "Medium": [c for c in range(64) if c.bit_count() <= 2], "Easy": white}
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "ghl_model.pt"
        torch.save({"timing": ghl.TimingNet().state_dict(), "timing_threshold": -1e9,
                    "shape": ghl.ShapeNet().state_dict(), "reduce": ghl.ReduceNet().state_dict(),
                    "allowed": allowed, "quota_power": 1.25,
                    "switch_weights": {"Hard": 1.0, "Medium": 0.5, "Easy": 0.0},
                    "keep_share": {"Hard": 0.75, "Medium": 0.5, "Easy": 0.35},
                    "config": {"timing": {}, "shape": {}, "reduce": {}}}, path)
        saved_path, ghl.MODEL_PATH = ghl.MODEL_PATH, path
        ghl._MODELS.clear()
        try:
            events = [(0.5 + i * 0.5, 0.7 + i * 0.5, 40 + i % 24, 0.8) for i in range(150)]
            tempo = steady(n=200)
            out = ghl.chart_tiers(tempo, np.ones(4000, dtype=np.float32), {}, (), events, {"guitar": events})
        finally:
            ghl.MODEL_PATH = saved_path
            ghl._MODELS.clear()
    expert_ticks = {t for t, _, _ in out["ExpertGHLGuitar"]}
    assert all(t % (RESOLUTION // ghl.SLOTS_PER_BEAT) == 0 for t in expert_ticks)
    times = [tempo.beat_to_time(t / RESOLUTION) for t in sorted(expert_ticks)]
    assert min(b - a for a, b in pairwise(times)) >= ghl.GAP_S - 1e-9
    above = expert_ticks
    for tier, share in (("Hard", 0.75), ("Medium", 0.5), ("Easy", 0.35)):
        kept = {t for t, _, _ in out[f"{tier}GHLGuitar"]}
        assert kept <= above, tier
        assert 0 < len(kept) <= round(share * len(expert_ticks)), (tier, len(kept))
        assert min(b - a for a, b in pairwise(sorted(kept))) >= ghl.TIER_MIN_GAP_BEATS[tier] * RESOLUTION
        above = kept
    expert = sorted({(t, sus) for t, _, sus in out["ExpertGHLGuitar"]})
    opens = {t for t, lane, _ in out["ExpertGHLGuitar"] if lane == ghl.OPEN}
    for (tick, sus), (nxt, _) in pairwise(expert):
        room = tempo.beat_to_time(nxt / RESOLUTION) - tempo.beat_to_time(tick / RESOLUTION)
        sustained = room >= ghl.SUSTAIN_GAP_S and tick not in opens
        assert sus == (nxt - tick if sustained else 0), (tick, sus, nxt)
    assert {lane for _, lane, _ in out["EasyGHLGuitar"]} <= {0, 1, 2}
    medium_sizes = {}
    for tick, _, _ in out["MediumGHLGuitar"]:
        medium_sizes[tick] = medium_sizes.get(tick, 0) + 1
    assert max(medium_sizes.values(), default=0) <= 2


def test_ghl_expert_figures_take_one_shape_each_within_the_songs_quota():
    """Notes the relation head joins to the previous one (P(SAME) > 0.5)
    share its shape, and no shape takes much more than its expected share
    even where it is every note's best."""
    from chartgen import ghl

    a, b = 1, 8
    shapes = np.full((20, 64), -30.0)
    shapes[:, a], shapes[:, b] = np.log(0.6), np.log(0.4)
    relations = np.zeros((20, ghl.RELATIONS))
    out = ghl.figure_shapes(np.hstack([shapes, relations]), [a, b], 1.0)
    # expected counts 12 and 8: shape a may take at most ceil(1.1 x 12) + 2 notes
    assert set(out) == {a, b} and 7 <= (out == a).sum() <= 16, out

    rng = np.random.default_rng(0)
    relations[:, ghl.SAME] = np.where(np.arange(20) % 4, 5.0, -5.0)
    out = ghl.figure_shapes(np.hstack([rng.normal(size=(20, 64)), relations]), list(range(64)), 1.25)
    assert all(len(set(out[i:i + 4])) == 1 for i in range(0, 20, 4)), out


def test_ghl_expert_notes_of_one_pitch_in_a_passage_share_a_shape():
    """With pitch keys, every note of a key takes one shape even where the
    relation head splits them into separate figures; unheard notes (-1)
    keep their figures."""
    from chartgen import ghl

    rng = np.random.default_rng(1)
    relations = np.zeros((24, ghl.RELATIONS))
    relations[:, ghl.SAME] = -5.0  # every note its own figure
    keys = np.array([60, 67, 60, 72] * 5 + [-1] * 4)
    shapes = rng.normal(size=(24, 64)) - 30.0
    shapes[:, [1, 2, 4, 8]] = rng.normal(size=(24, 4))  # the song uses four shapes
    out = ghl.figure_shapes(np.hstack([shapes, relations]), list(range(64)), 1.0, keys)
    for key in (60, 67, 72):
        assert len(set(out[keys == key])) == 1, (key, out)


def test_ghl_expert_switch_weight_keeps_neighbouring_groups_on_one_row():
    from chartgen import ghl

    rng = np.random.default_rng(2)
    relations = np.zeros((40, ghl.RELATIONS))
    relations[:, ghl.SAME] = -5.0
    keys = rng.integers(0, 10, 40)
    shapes = rng.normal(size=(40, 64)) - 30.0
    shapes[:, [1, 2, 8, 16]] = rng.normal(size=(40, 4))  # W1 W2 B1 B2

    def switches(weight):
        rows = ghl.FACTOR_OF[ghl.figure_shapes(np.hstack([shapes, relations]), list(range(64)), 1.0, keys, weight), 1]
        return int((rows[1:] != rows[:-1]).sum())

    assert switches(10.0) < switches(0.0), (switches(10.0), switches(0.0))


def test_ghl_tier_switch_weight_keeps_the_row():
    from chartgen import ghl

    logits = np.full((6, 64), -9.0)
    logits[[0, 2, 4], 1] = 0.0  # W1 best
    logits[[1, 3, 5], 8] = 0.0  # B1 best
    assert ghl.viterbi(logits, [1, 8], 0.0).tolist() == [1, 8, 1, 8, 1, 8]
    assert len(set(ghl.viterbi(logits, [1, 8], 20.0))) == 1


def test_ghl_tier_sustains_run_to_the_tiers_next_note_when_it_is_far_enough():
    from chartgen import ghl

    ticks = np.array([0, 480, 960, 2880, 3360])
    keep = np.array([True, False, True, True, True])
    sustained = np.array([True, True, True, False, True])
    out = ghl.sustains(ticks, keep, sustained, 2.0, RESOLUTION)
    # 0 -> 960 is two beats; 960 -> 2880 four; 2880 is not sustained on
    # Expert; the last kept note has no next note to run to.
    assert out.tolist() == [960, 0, 1920, 0, 0]



def test_ghl_bars_whose_audio_repeats_take_the_earlier_bars_notes():
    from chartgen import ghl

    rng = np.random.default_rng(0)
    tempo = steady(n=80)
    bar = ghl.BAR_BEATS * RESOLUTION
    frames = int(tempo.beat_to_time(80) * ghl.ENVELOPE_SR / ghl.ENVELOPE_HOP) + 10
    envelope = rng.random(frames).astype(np.float32)
    riff = rng.random(int(tempo.beat_to_time(4) * ghl.ENVELOPE_SR / ghl.ENVELOPE_HOP)).astype(np.float32)
    start = int((tempo.beat_to_time(0) - ghl.ONSET_LAG_S) * ghl.ENVELOPE_SR / ghl.ENVELOPE_HOP)
    for k in range(4):  # bars 0-3 carry the same audio, the rest is noise
        begin = int(tempo.beat_to_time(k * 4) * ghl.ENVELOPE_SR / ghl.ENVELOPE_HOP)
        envelope[begin - start:begin - start + len(riff)] = riff
    ticks = np.sort(rng.choice(np.arange(0, 16 * bar, RESOLUTION // 4), 120, replace=False))
    shapes = rng.integers(1, 64, len(ticks))
    source = ghl.repeat_sources(ghl.slots(tempo, tempo.beat_to_time(64))[1], 0, envelope, {}, [])
    out_ticks, out_shapes, origin = ghl.copy_repeated_bars(ticks, shapes, RESOLUTION, 0, source)
    first = [(t, s) for t, s in zip(out_ticks, out_shapes, strict=True) if t < bar]
    for k in (1, 2, 3):
        later = [(t - k * bar, s) for t, s in zip(out_ticks, out_shapes, strict=True) if k * bar <= t < (k + 1) * bar]
        assert later == first, k
    assert (origin[out_ticks >= bar][:len(first)] >= 0).all()
    assert (origin[out_ticks >= 4 * bar] == -1).mean() > 0.5


def test_ghl_timing_logits_of_repeated_bars_pool_over_the_repeat_chain():
    from chartgen import ghl

    bar_slots = ghl.BAR_BEATS * ghl.SLOTS_PER_BEAT
    logits = np.arange(2 + 4 * bar_slots, dtype=float)
    out = ghl.pool_repeats(logits, 0, np.array([-1, 0, -1, 1]))  # bars 0, 1 and 3 form one chain
    bars, before = out[:4 * bar_slots].reshape(4, bar_slots), logits[:4 * bar_slots].reshape(4, bar_slots)
    for bar in (0, 1, 3):
        assert np.allclose(bars[bar], before[[0, 1, 3]].mean(0)), bar
    assert np.allclose(bars[2], before[2]) and (out[-2:] == logits[-2:]).all()

def test_write_chart_merges_lyrics_into_events_in_tick_order():
    t = steady()
    text = write_chart(
        {"ExpertSingle": [(0, 0, 0)]}, t,
        {"name": "T", "artist": "A", "charter": "c"}, "song.ogg",
        events=[(960, "Section 1")],
        lyrics=[(480, "phrase_start"), (480, "lyric Hello"),
                (720, "lyric world"), (900, "phrase_end")],
    )
    block = text.split("[Events]")[1].split("}")[0]
    lines = [ln.strip() for ln in block.splitlines() if "= E" in ln]
    assert any('lyric Hello' in ln for ln in lines)
    assert any('section Section 1' in ln for ln in lines)
    ticks = [int(ln.split(" = ")[0]) for ln in lines]
    assert ticks == sorted(ticks), "events must stay tick-ordered"
    # A phrase sharing a tick with its own first syllable must still open
    # first, or that word belongs to the previous phrase.
    same_tick = [ln for ln in lines if ln.startswith("480 = ")]
    assert "phrase_start" in same_tick[0], same_tick


def test_empty_star_power_phrases_are_dropped():
    """A phrase whose notes were all reduced away is unactivatable dead weight."""
    text = write_chart(
        {"ExpertSingle": [(0, 0, 0)], "EasySingle": [(0, 0, 0)]},
        steady(), {"name": "T", "artist": "A", "charter": "c"}, "song.ogg",
        # Second phrase sits far past every note.
        star_power=[(0, RESOLUTION), (100 * RESOLUTION, RESOLUTION)],
    )
    assert text.count("= S 2 ") == 2, "one phrase per tier should survive"
    assert f"{100 * RESOLUTION} = S 2" not in text


def test_star_power_is_bar_aligned_spaced_and_non_empty():
    res = RESOLUTION
    bar = 4 * res
    # Dense 8th notes across 32 bars.
    notes = [(i * (res // 2), i % 5, 0) for i in range(32 * 8)]
    phrases = star_power_phrases(notes, res, duration_s=120.0)

    assert phrases, "a dense 32-bar song should get star power"
    ticks = {t for t, _, _ in notes}
    for start, length in phrases:
        assert start % bar == 0, f"phrase at {start} is not bar-aligned"
        assert any(start <= t < start + length for t in ticks), "empty phrase"
    starts = [s for s, _ in phrases]
    assert starts == sorted(starts)
    gaps = [b - a for a, b in pairwise(starts)]
    assert all(g >= 6 * bar for g in gaps), f"phrases too bunched: {gaps}"


def test_star_power_handles_sparse_and_empty_input():
    assert star_power_phrases([], RESOLUTION, 120.0) == []
    # Too few notes to be worth an activation.
    assert star_power_phrases([(0, 0, 0)], RESOLUTION, 120.0) == []


def _notes(lane_counts):
    """Build a note list with a given lane distribution."""
    out, tick = [], 0
    for lane, count in lane_counts.items():
        for _ in range(count):
            out.append((tick, lane, 0))
            tick += RESOLUTION // 4
    return out


def test_variety_score_separates_real_degenerate_charts():
    """Thresholded against distributions actually produced by the model.

    All four came off the same clip at identical settings, which is why the gate
    exists at all.
    """
    balanced = variety_score(_notes({0: 39, 1: 40, 2: 25, 3: 15}))
    spread = variety_score(_notes({0: 2, 1: 12, 2: 40, 3: 38, 4: 27}))
    one_lane = variety_score(_notes({0: 1, 1: 100, 2: 13, 3: 5}))
    open_heavy = variety_score(_notes({0: 10, 1: 39, 2: 30, 7: 40}))

    assert balanced > 0.80, balanced
    assert spread > 0.80, spread
    assert one_lane < 0.40, f"84%-on-one-lane must fail the gate, got {one_lane}"
    assert open_heavy < 0.40, f"34% open notes must fail the gate, got {open_heavy}"
    assert spread > one_lane and balanced > open_heavy


def test_variety_score_flags_two_fret_alternation():
    """Ping-ponging two frets is boring even though no single lane dominates."""
    assert variety_score(_notes({0: 60, 1: 60})) < 0.80


def test_variety_score_tolerates_realistic_low_fret_bias():
    """Real charts favour low frets; that must not trigger pointless retries."""
    assert variety_score(_notes({0: 40, 1: 30, 2: 20, 3: 8, 4: 2})) >= 0.80


def test_variety_score_handles_degenerate_input():
    assert variety_score([]) == 0.0
    assert variety_score(_notes({7: 20})) == 0.0  # open notes only, no frets


def _seq(lanes):
    return [(i * (RESOLUTION // 4), lane, 0) for i, lane in enumerate(lanes)]


def test_walk_score_catches_the_staircase_variety_missed():
    """The failure that got through playtesting: a chart nobody enjoyed.

    A G-R-Y-B-O staircase uses all five frets perfectly evenly, so entropy rates
    it near-perfect. Only a transition-based measure can see it.
    """
    staircase = _seq([0, 1, 2, 3, 4] * 20)
    assert variety_score(staircase) > 0.95, "entropy really is blind to this"
    assert walk_score(staircase) < 0.30, walk_score(staircase)


def test_walk_score_on_the_real_measured_chart():
    """The shipped chart's signature — heavy walking with almost no repeats —
    must fail the 0.80 gate. Built deterministically: a fretboard zigzag with
    one repeat per 16 notes (~6%, what the playtested chart measured)."""
    lanes, cur, direction = [], 0, 1
    for i in range(400):
        if i % 16 == 15:
            lanes.append(cur)  # the rare repeat
            continue
        cur += direction
        if cur in (0, 4):
            direction *= -1
        lanes.append(cur)
    chart = _seq(lanes)
    p = step_profile(chart)
    assert p["adjacent"] > 0.70 and p["repeat"] < 0.10, p
    assert walk_score(chart) < 0.80, f"should be penalised, got {walk_score(chart)}"
    assert walk_score(chart) < walk_score(_seq([0, 0, 3, 1, 1, 4, 2, 2] * 50))


def test_walk_score_accepts_a_riff_like_pattern():
    """Repeated frets and jumps — what real guitar parts do — must pass."""
    riff = _seq([0, 0, 3, 0, 0, 2, 4, 4, 1, 1, 0, 0, 2, 2, 4, 0] * 12)
    p = step_profile(riff)
    assert p["repeat"] > 0.12, p
    assert walk_score(riff) > 0.80, f"riff wrongly penalised: {walk_score(riff)}"


def test_step_profile_ignores_chords_and_opens():
    notes = [(0, 0, 0), (0, 2, 0), (120, 7, 0), (240, 1, 0), (360, 1, 0)]
    p = step_profile(notes)
    # Only the two single non-open notes form one step, and it is a repeat.
    assert p["steps"] == 1 and p["repeat"] == 1.0, p
    assert p["chord_share"] > 0


def test_rating_scales_with_density_and_stays_in_range():
    from chartgen.rating import rate_expert

    t = steady(n=256)
    res = RESOLUTION
    sparse = [(i * 2 * res, i % 3, 0) for i in range(60)]       # half notes
    busy = [(i * (res // 4), i % 5, 0) for i in range(600)]      # wall of 16ths
    lo, hi = rate_expert(sparse, t), rate_expert(busy, t)
    assert 0 <= lo <= 6 and 0 <= hi <= 6
    assert hi > lo, f"denser chart must rate higher: {lo} vs {hi}"
    assert rate_expert([(0, 0, 0)], t) == 0 or rate_expert([(0, 0, 0)], t) <= 6


def test_thin_to_rating_hits_the_target_and_keeps_beats():
    from chartgen.rating import rate_expert
    from chartgen.reduce import thin_to_rating

    t = steady(n=512)
    res = RESOLUTION
    # A wall of 16ths: rates high naturally.
    expert = [(i * (res // 4), (i * 3) % 5, 0) for i in range(800)]
    natural = rate_expert(expert, t)
    assert natural >= 4, f"fixture should rate hard, got {natural}"

    target = 2
    thinned, achieved = thin_to_rating(expert, res, t, target)
    assert achieved <= target, (natural, achieved)
    assert rate_expert(thinned, t) == achieved, "reported rating must be real"
    assert len(thinned) < len(expert)
    kept = {tick for tick, _, _ in thinned}
    beats = {tick for tick, _, _ in expert if tick % res == 0}
    assert beats <= kept, "thinning must never drop beats"

    # Already at or below target: untouched.
    sparse = [(i * 2 * res, i % 3, 0) for i in range(60)]
    out, achieved = thin_to_rating(sparse, res, t, 6)
    assert out == sparse


def test_song_ini_carries_rating():
    from chartgen.chart import write_song_ini

    meta = {"name": "n", "artist": "a", "charter": "c"}
    assert "diff_guitar = 4" in write_song_ini(meta, 1000, diff_guitar=4)
    assert "diff_guitar = -1" in write_song_ini(meta, 1000)


def test_song_ini_disables_hopos_by_default():
    from chartgen.chart import write_song_ini

    meta = {"name": "n", "artist": "a", "charter": "c"}
    assert "hopo_frequency = 1" in write_song_ini(meta, 1000)
    assert "hopo_frequency" not in write_song_ini(meta, 1000, hopos=True)


def _natural_hopos(notes, thr=162):
    by = {}
    for tick, lane, _ in notes:
        by.setdefault(tick, []).append(lane)
    ticks = sorted(by)
    return sum(
        1 for a, b in pairwise(ticks)
        if len(by[b]) == 1 and by[b][0] != OPEN_LANE
        and b - a < thr and by[b] != by[a]
    )


OPEN_LANE = 7


def test_reduce_hard_keeps_hopos_and_thins_walls():
    """The old reducer's Hard was all strums: 3 HOPOs where Expert had 650."""
    from chartgen.reduce import reduce_hard

    res = RESOLUTION
    # 4 bars of saturated 16ths — a wall — then a short 3-note lick.
    wall = [(i * (res // 4), i % 5, 0) for i in range(64)]
    lick_start = 32 * res
    lick = [(lick_start + i * (res // 4), i, 0) for i in range(3)]
    hard = reduce_hard(wall + lick, res)

    assert len(hard) < len(wall) + len(lick), "walls must thin"
    assert _natural_hopos(hard) >= 16, f"Hard lost its HOPOs: {_natural_hopos(hard)}"
    kept_lick = [n for n in hard if n[0] >= lick_start]
    assert len(kept_lick) == 3, "short figures must survive whole"


def test_reduce_tiers_ladder_and_lane_ranges():
    from chartgen.reduce import derive_tiers

    res = RESOLUTION
    expert = [(i * (res // 4), (i * 3) % 5, 0) for i in range(128)]
    # Sprinkle chords so the caps are exercised.
    expert += [(i * res * 2, ((i * 3) % 4) + 1, 0) for i in range(16)]
    tiers = derive_tiers(sorted(expert), res)

    n_e = len({t for t, _, _ in expert})
    n_h = len({t for t, _, _ in tiers["HardSingle"]})
    n_m = len({t for t, _, _ in tiers["MediumSingle"]})
    n_y = len({t for t, _, _ in tiers["EasySingle"]})
    assert n_e > n_h > n_m > n_y, (n_e, n_h, n_m, n_y)

    assert max(l for _, l, _ in tiers["MediumSingle"]) <= 3, "no orange on Medium"
    assert max(l for _, l, _ in tiers["EasySingle"]) <= 2, "Easy is G/R/Y"
    from chartgen.quality import variety_score
    assert variety_score(tiers["MediumSingle"]) > 0.60, "remap must not collapse lanes"

    # Chord caps: Hard <= 2 notes per tick, Easy strictly single.
    for name, cap in (("HardSingle", 2), ("MediumSingle", 2), ("EasySingle", 1)):
        by_tick = {}
        for t, l, _ in tiers[name]:
            by_tick.setdefault(t, []).append(l)
        assert max(map(len, by_tick.values())) <= cap, name


def test_reduce_hard_respects_nps_cap_on_dense_charts():
    """Playtest: Hard kept 92% of a dense EDM Expert. With a BPM the reducer
    must enforce ~4.5 notes/sec, dropping fine positions before off-8ths and
    never touching beats."""
    from chartgen.reduce import HARD_MAX_NPS, reduce_hard

    res = RESOLUTION
    bpm = 128.0
    # Saturated 16ths for 16 bars: 16 positions per bar.
    expert = [(i * (res // 4), i % 5, 0) for i in range(16 * 16)]
    hard = reduce_hard(expert, res, bpm=bpm)

    seconds = 16 * 4 * 60.0 / bpm
    nps = len({t for t, _, _ in hard}) / seconds
    assert nps <= HARD_MAX_NPS + 0.3, f"Hard too dense: {nps:.1f} nps"
    kept = {t for t, _, _ in hard}
    beats = {t for t, _, _ in expert if t % res == 0}
    assert beats <= kept, "beats must never be dropped"
    # Without a BPM the cap is off and the old behaviour stands.
    assert len(reduce_hard(expert, res)) > len(hard)


def test_chord_rules_remove_green_orange_stretches():
    """Chorus Encore errors on a Green+Orange chord on Hard; RBN keeps
    green-to-orange three-note chords off Expert too."""
    from chartgen.reduce import enforce_chord_rules

    tiers = enforce_chord_rules({
        "ExpertSingle": [(0, 0, 0), (0, 2, 0), (0, 4, 90)],   # 3-note G..O
        "HardSingle": [(0, 0, 0), (0, 4, 0)],                  # 2-note G+O
        "MediumSingle": [(0, 0, 0), (0, 3, 0)],                # G+B
        "EasySingle": [(0, 0, 0)],
    })
    for name, notes in tiers.items():
        lanes = {lane for _, lane, _ in notes}
        assert not (0 in lanes and 4 in lanes), f"{name} kept a G+O stretch"
    assert {l for _, l, _ in tiers["MediumSingle"]} == {0, 2}, tiers["MediumSingle"]
    # The moved note keeps its sustain rather than silently losing it.
    assert (0, 3, 90) in tiers["ExpertSingle"], tiers["ExpertSingle"]
    # A two-note G+O on EXPERT is allowed (sparingly) and must survive.
    kept = enforce_chord_rules({"ExpertSingle": [(0, 0, 0), (0, 4, 0)]})
    assert {l for _, l, _ in kept["ExpertSingle"]} == {0, 4}


def test_star_power_avoids_the_unspendable_ending():
    """CH needs two phrases to activate, so meter awarded in the last measures
    can never be spent. RBN: no phrase in roughly the last 8 measures."""
    res = RESOLUTION
    bar = 4 * res
    notes = [(i * (res // 2), i % 5, 0) for i in range(32 * 8)]
    phrases = star_power_phrases(notes, res, duration_s=120.0)
    last_note = max(t for t, _, _ in notes)
    for start, length in phrases:
        assert start <= last_note - 8 * bar, f"phrase at {start} is too late"
        # Two bars: the measured human median is 8 beats, twice the RBN
        # spec's one measure.
        assert length == 2 * bar, "phrases should be two measures"


def test_reduce_keeps_syncopation():
    """An isolated off-beat note is the rhythm; deleting it kills the song."""
    from chartgen.reduce import reduce_hard

    res = RESOLUTION
    # Sparse syncopated figure: beat, then a lone 16th pickup before beat 3.
    notes = [(0, 0, 0), (2 * res + 3 * res // 4, 2, 0), (3 * res, 1, 0)]
    hard = reduce_hard(notes, res)
    assert (2 * res + 3 * res // 4, 2, 0) in hard, "syncopation must survive on Hard"


def test_density_gate_drops_weak_notes_but_never_tears_holes():
    """Notes without onset evidence go; gaps larger than 2 beats never open."""
    from chartgen.density import gate_by_onsets

    sr = 22050
    t = steady(bpm=120.0, n=32)
    res = RESOLUTION
    # Audio: clicks on beats 1-8, then silence for beats 9-16.
    y = np.zeros(sr * 10, dtype=np.float32)
    for beat in range(8):
        s = int(t.beat_to_time(beat + 1) * sr)
        y[s:s + 200] = 1.0
    # Chart: a note on every 8th across beats 1-16.
    notes = [(res + i * (res // 2), i % 5, 0) for i in range(30)]

    out = gate_by_onsets(notes, y, sr, t)
    kept = sorted({tick for tick, _, _ in out})
    assert len(kept) < 30, "silent-section notes must thin out"
    # Louder half survives more densely than the silent half.
    mid = res + 15 * (res // 2)
    loud = sum(1 for k in kept if k < mid)
    quiet = sum(1 for k in kept if k >= mid)
    assert loud > quiet, (loud, quiet)
    # The gap constraint held everywhere.
    for a, b in pairwise(kept):
        assert b - a <= 2 * res + res // 2, f"gate tore a hole: {a}->{b}"


def test_density_gate_never_eats_the_fade_out_tail():
    """Playtest: charts ended early on fade-outs — edge notes had no 'after'
    neighbour so the gap rule never fired. Coverage may retreat at most the
    gap budget from either end."""
    from chartgen.density import gate_by_onsets

    sr = 22050
    t = steady(bpm=120.0, n=64)
    res = RESOLUTION
    y = np.zeros(sr * 20, dtype=np.float32)
    # Loud clicks for beats 1-16, then a fading tail: beats 17-28 get quieter
    # and quieter but the song is still going.
    for beat in range(16):
        s = int(t.beat_to_time(beat + 1) * sr)
        y[s:s + 300] = 1.0
    for i, beat in enumerate(range(16, 28)):
        s = int(t.beat_to_time(beat + 1) * sr)
        y[s:s + 300] = max(0.02, 0.25 * (0.7 ** i))
    notes = [(res * (b + 1), b % 5, 0) for b in range(28)]

    out = gate_by_onsets(notes, y, sr, t)
    last_kept = max(tick for tick, _, _ in out)
    last_orig = max(tick for tick, _, _ in notes)
    assert last_orig - last_kept <= 2 * res, \
        f"tail retreated {(last_orig - last_kept) / res:.1f} beats"


def test_tier_block_writes_solo_markers_only_where_notes_exist():
    from chartgen.chart import _tier_block

    notes = [(0, 0, 0), (480, 1, 0), (960, 2, 0)]
    block = _tier_block("ExpertSingle", notes, [], solos=[(0, 960), (5000, 6000)])
    assert "0 = E solo" in block and "960 = E soloend" in block
    assert "5000 = E solo" not in block, "empty solo phrase must be dropped"


def test_density_gate_keeps_everything_on_strong_audio():
    from chartgen.density import gate_by_onsets

    sr = 22050
    t = steady(bpm=120.0, n=32)
    res = RESOLUTION
    y = np.zeros(sr * 10, dtype=np.float32)
    notes = []
    for i in range(16):
        s = int(t.beat_to_time(i + 1) * sr)
        y[s:s + 300] = 1.0
        notes.append((res * (i + 1), i % 5, 0))
    out = gate_by_onsets(notes, y, sr, t)
    assert len(out) == len(notes), "well-evidenced notes must all survive"


def test_youtube_url_detection_and_title_parsing():
    from chartgen.youtube import is_youtube_url, parse_title

    for url in ("https://www.youtube.com/watch?v=abc123",
                "https://youtu.be/abc123",
                "https://music.youtube.com/watch?v=abc123",
                "http://m.youtube.com/watch?v=abc123"):
        assert is_youtube_url(url), url
    for not_url in ("/music/track.mp3", "song.opus",
                    "https://vimeo.com/123", "youtube song name"):
        assert not is_youtube_url(not_url), not_url

    assert parse_title("Alan Walker - Faded (Official Video)") == ("Alan Walker", "Faded")
    assert parse_title("Daft Punk - Get Lucky [Official Audio] (HD)") == ("Daft Punk", "Get Lucky")
    assert parse_title("Bohemian Rhapsody (Remastered 2011)") == ("", "Bohemian Rhapsody")
    # Auto-generated music channels and label channels carry suffixes that are
    # not part of the artist's name.
    from chartgen.youtube import channel_artist
    assert channel_artist("Astron - Topic") == "Astron"
    assert channel_artist("QueenVEVO") == "Queen"
    assert channel_artist("NoCopyrightSounds") == "NoCopyrightSounds"
    assert channel_artist(None) == ""
    artist, name = parse_title("YOASOBI「アイドル」 Official Music Video")
    assert name, (artist, name)


def test_guess_metadata_prefers_tags_then_filename():
    import tempfile

    import mutagen.flac
    import soundfile as sf

    from chartgen.tags import guess_metadata

    with tempfile.TemporaryDirectory() as tmp:
        # Tagged file: embedded metadata wins over the filename.
        tagged = Path(tmp) / "random_download_x93.flac"
        sf.write(tagged, np.zeros(2048, dtype=np.float32), 22050)
        flac = mutagen.flac.FLAC(tagged)
        flac["artist"], flac["title"] = "Daft Punk", "Around the World"
        flac.save()
        assert guess_metadata(tagged) == ("Daft Punk", "Around the World")

        # Untagged file: the "Artist - Title" filename convention.
        plain = Path(tmp) / "Queen - Bohemian Rhapsody (Remastered).flac"
        sf.write(plain, np.zeros(2048, dtype=np.float32), 22050)
        assert guess_metadata(plain) == ("Queen", "Bohemian Rhapsody")

        # No tags, no dash: title falls back to the stem, artist stays empty.
        bare = Path(tmp) / "jam.flac"
        sf.write(bare, np.zeros(2048, dtype=np.float32), 22050)
        artist, name = guess_metadata(bare)
        assert (artist, name) == ("", "jam")


def test_folder_name_helper_and_skip_check_agree():
    """skip_existing predicts the folder before generating; the prediction
    must match what the writer actually creates, or skips silently break."""
    from chartgen.pipeline import _folder_name

    assert _folder_name("A", 'B - "C"') == "A - B - C"
    assert _folder_name("", "Song") == "Song"
    assert _folder_name("", '"??"') == "song"
    assert _folder_name("Alan Walker", "Faded [chartgen]") == "Alan Walker - Faded [chartgen]"


def test_song_folder_names_survive_reserved_characters():
    """A real YouTube title ('Baldur's Gate 3 - "I Want To Live"') crashed
    folder creation: quotes and friends are illegal on FAT and exFAT drives. The
    folder is sanitised; metadata keeps the original text."""
    import re

    for title in ('BG3 - OST - "I Want To Live" (Acoustic)',
                  "What?! <Live/Mix> C:\\stuff|here*"):
        cleaned = re.sub(r'[<>:"/\\|?*]', "", f"Artist - {title}").strip(" -.")
        assert not re.search(r'[<>:"/\\|?*]', cleaned), cleaned
        assert cleaned, "sanitised name must not be empty"
    # A name that is nothing but illegal chars falls back to non-empty.
    assert (re.sub(r'[<>:"/\\|?*]', "", '"??"').strip(" -.") or "song") == "song"


def test_playlist_url_detection_is_strict():
    """Only pure playlist URLs expand; a video that happens to carry a
    &list= parameter must stay a single video."""
    from chartgen.youtube import expand_inputs, is_playlist_url

    assert is_playlist_url("https://www.youtube.com/playlist?list=PLx123")
    assert is_playlist_url("https://music.youtube.com/playlist?list=PLx123")
    assert not is_playlist_url("https://www.youtube.com/watch?v=abc&list=PLx123")
    assert not is_playlist_url("https://youtu.be/abc123")
    assert not is_playlist_url("/music/track.mp3")

    # Non-playlist inputs pass through expand_inputs untouched, in order.
    items = ["/music/a.mp3", "https://youtu.be/abc", "/music/b.flac"]
    assert expand_inputs(items) == items


def test_tempo_octave_helpers():
    """The octave label decides HOPO vs strum feel; helpers must move only the
    label, never the physical grid."""
    from chartgen.tempo import TempoMap, _double

    slow = np.arange(32) * 1.0  # 60 BPM
    doubled = _double(slow)
    assert abs(TempoMap(beat_times=doubled, pickup_beats=0).bpm - 120.0) < 0.5
    # Every original beat survives doubling: the grid only gains midpoints.
    assert all(any(abs(doubled - b) < 1e-9) for b in slow)


def _click_track(bpm, seconds, sr=22050, start=0.5):
    clicks = np.arange(start, seconds - 0.1, 60.0 / bpm)
    y = np.zeros(int(seconds * sr), dtype=np.float32)
    burst = (np.random.default_rng(0).standard_normal(int(0.02 * sr))
             * np.exp(-np.arange(int(0.02 * sr)) / (0.003 * sr))).astype(np.float32)
    for click in clicks:
        at = int(click * sr)
        y[at:at + len(burst)] += burst
    return y, sr, clicks


def test_click_tracks_come_out_at_their_tempo_and_on_the_clicks():
    """The grid sits on the transient with no lag shift, and the tempo keeps
    its octave: 75 and 180 BPM both lie outside any 'conventional' range a
    fold would push them into."""
    from chartgen.tempo import detect

    for bpm in (75.0, 120.0, 180.0):
        y, sr, clicks = _click_track(bpm, 20.0)
        found = detect(y, sr)
        assert abs(found.bpm - bpm) < 0.5, (bpm, found.bpm)
        inner = clicks[2:-2]
        error = np.abs(inner[:, None] - found.beat_times[None, :]).min(axis=1)
        assert np.median(error) < 0.008, (bpm, np.median(error))
        assert error.max() < 0.02, (bpm, error.max())


def test_stacked_networks_match_madmom():
    """_logits reimplements madmom's network loop; the beat and downbeat
    activations must be the ones madmom's own processors give."""
    import librosa
    from madmom.features.beats import RNNBeatProcessor
    from madmom.features.downbeats import RNNDownBeatProcessor
    from scipy.special import expit, softmax

    from chartgen.tempo import TRACKER_SR, _logits, _madmom

    y, sr, _ = _click_track(97.0, 6.0)
    y += 0.05 * np.sin(2 * np.pi * 220 * np.arange(len(y)) / sr).astype(np.float32)
    y44 = librosa.resample(y, orig_sr=sr, target_sr=TRACKER_SR)
    beat_net, downbeat_net, _ = _madmom()
    for ours, theirs in ((expit(_logits(beat_net, y44)).mean(axis=0)[:, 0], RNNBeatProcessor()(y44)),
                         (softmax(_logits(downbeat_net, y44), axis=-1).mean(axis=0)[:, 1:],
                          RNNDownBeatProcessor()(y44))):
        assert ours.shape == theirs.shape
        assert np.abs(ours - theirs).max() < 1e-3, np.abs(ours - theirs).max()


def test_downbeat_phase_follows_the_kick_on_beat_one():
    from chartgen.tempo import detect

    y, sr, clicks = _click_track(100.0, 20.0)
    kick_t = np.arange(int(0.25 * sr)) / sr
    kick = (np.sin(2 * np.pi * (50 * kick_t + 18 * (1 - np.exp(-kick_t / 0.03))))
            * np.exp(-kick_t / 0.12)).astype(np.float32)
    for first in (1, 2):
        accented = y.copy()
        for click in clicks[first::4]:
            at = int(click * sr)
            accented[at:at + len(kick)] += kick[:len(accented) - at]
        found = detect(accented, sr)
        assert found.downbeat_phase == int(np.argmin(np.abs(found.beat_times - clicks[first]))) % 4, first


def test_tier_features_place_a_tick_in_its_bar():
    from chartgen import ghl

    ticks = np.array([0, 5 * RESOLUTION, 6 * RESOLUTION + RESOLUTION // 2])
    flat = np.zeros(3, dtype=np.int64)
    bar = ghl.tier_features(ticks, flat, np.array([1, 2, 4]), RESOLUTION, downbeat=5)[:, -5:]
    assert bar[:, :4].argmax(1).tolist() == [3, 0, 1]
    assert np.allclose(bar[:, 4], [0.75, 0.0, 0.375])
    unknown = ghl.tier_features(ticks, flat, np.array([1, 2, 4]), RESOLUTION, downbeat=None)
    assert unknown.shape == (3, ghl.TIER_FEATURES) and not unknown[:, -5:].any()


def test_sixteenth_grid_stays_inside_the_natural_hopo_window():
    """HOPOs are implicit in .chart, so the grid choice is what creates them.

    At resolution 480 a 16th is 120 ticks, inside the 160-tick window, so 16th
    runs HOPO and 8ths strum. If either the resolution or the default subdivision
    changes, HOPOs silently disappear — hence this guard.
    """
    thr = hopo_threshold(RESOLUTION)
    assert thr == 162, thr
    assert hopo_threshold(192) == 65, "must match the spec's reference value"
    assert RESOLUTION // 4 < thr, "16ths must fall inside the HOPO window"
    assert RESOLUTION // 2 > thr, "8ths must strum, not HOPO"


def test_smooth_fret_jumps_matches_human_hand_movement():
    """Human charts of the same songs jump 3+ lanes on 0-1% of consecutive
    single notes; ours measured 9-30%. Clamping must kill the teleports while
    keeping pitch direction, and leave long rests free to relocate."""
    from chartgen.frets import smooth_fret_jumps, step_share

    t = steady(n=64)
    res = RESOLUTION
    teleporting = [(i * (res // 2), 4 if i % 2 == 0 else 0, 0) for i in range(32)]
    assert step_share(teleporting, 3) > 0.9, "fixture should be all teleports"

    out = smooth_fret_jumps(teleporting, t, max_step=2)
    assert step_share(out, 3) == 0.0, out[:6]
    assert [n[0] for n in out] == [n[0] for n in teleporting], "timing moved"
    assert all(0 <= lane <= 4 for _, lane, _ in out)

    # A rising line still rises — contour is what the guidelines ask for.
    rising = [(i * (res // 2), min(4, i), 0) for i in range(10)]
    lanes = [lane for _, lane, _ in smooth_fret_jumps(rising, t, max_step=2)]
    assert lanes == sorted(lanes), lanes

    # Two beats of rest is enough time to move the hand anywhere.
    far = [(0, 0, 0), (8 * res, 4, 0)]
    assert smooth_fret_jumps(far, t, max_step=2) == far

    # Chords move as a unit, keeping their shape.
    chord = [(0, 0, 0), (0, 2, 0), (res // 2, 3, 0), (res // 2, 4, 0)]
    out = smooth_fret_jumps(chord, t, max_step=1)
    second = sorted(lane for tick, lane, _ in out if tick == res // 2)
    assert second[1] - second[0] == 1, f"chord shape distorted: {second}"


def test_top_up_sustains_is_bounded_by_the_measured_share():
    """Deriving sustains from a reduced tier's own gaps once made 39 of Easy's
    40 notes sustains. The share bound is what makes it safe to do at all."""
    from chartgen.expression import TIER_SUSTAIN_SHARE, top_up_sustains

    res = RESOLUTION
    # Sparse tier with room everywhere: every note COULD sustain.
    notes = [(i * 4 * res, i % 3, 0) for i in range(40)]
    out = top_up_sustains(notes, res, end_tick=200 * res,
                          target_share=TIER_SUSTAIN_SHARE["EasySingle"])
    share = sum(1 for _, _, sus in out if sus > 0) / len(out)
    # The bound tracks the measured Easy-tier median (22.2% across the
    # 800-chart calibration set), not a hardcoded band that goes stale
    # every time the calibration data grows.
    target = TIER_SUSTAIN_SHARE["EasySingle"]
    assert target - 0.08 <= share <= target + 0.05, \
        f"share {share:.0%} strays from the {target:.0%} target"

    # Sustains still never reach the next note.
    for (tick, _, sus), (nxt, _, _) in pairwise(out):
        if nxt != tick:
            assert tick + sus < nxt, f"sustain at {tick} runs into {nxt}"

    # Notes that already sustain are left alone, and a tier already at target
    # gains nothing.
    already = [(i * 4 * res, 0, res) for i in range(10)]
    assert top_up_sustains(already, res, 200 * res, 0.05) == already


def test_section_reuse_copies_lanes_but_keeps_the_repeat_rhythm():
    """Human charts overlap 55% between repeated sections; ours managed 13%.
    Reuse must raise that without pasting a section wholesale — the later
    section keeps whatever rhythm the audio gave it."""
    from chartgen.structure import find_repeats, reuse_patterns

    res = RESOLUTION
    bar = 4 * res
    # Two 4-bar sections. The second has the same rhythm on beats but adds an
    # extra note the first does not have, and different lanes throughout.
    first = [(i * res, i % 5, 0) for i in range(16)]
    second = [(4 * bar + i * res, (i + 2) % 5, 0) for i in range(16)]
    second.append((4 * bar + res // 2, 3, 0))   # unique to the repeat
    notes = sorted(first + second)

    out = reuse_patterns(notes, [0, 4 * bar], {1: 0})
    out_by_tick = {}
    for tick, lane, _ in out:
        out_by_tick.setdefault(tick, []).append(lane)

    # Lanes now match the first section wherever the two line up.
    for i in range(16):
        assert out_by_tick[4 * bar + i * res] == [i % 5], i
    # The repeat's own extra note survives untouched.
    assert out_by_tick[4 * bar + res // 2] == [3]
    # Timing is never invented or dropped.
    assert {t for t, _, _ in out} == {t for t, _, _ in notes}

    # find_repeats needs both similar harmony and comparable length.
    import numpy as np
    a = np.ones(12) / np.sqrt(12)
    b = np.zeros(12); b[0] = 1.0
    assert find_repeats([a, a], [bar, bar]) == {1: 0}
    assert find_repeats([a, b], [bar, bar]) == {}, "different harmony"
    assert find_repeats([a, a], [bar, 4 * bar]) == {}, "different length"


def test_tempo_map_extrapolates_past_the_last_detected_beat():
    """librosa stops tracking beats when a song fades out. np.interp clamps
    past its last beat, so every outro note quantized onto one tick and charts
    ended up to 25s early — on both engines, since the tempo map is shared."""
    t = steady(bpm=120.0, n=32, first=0.5)   # beats end at ~16s
    last_beat = float(t.beat_times[-1])

    # Ten seconds past the last beat must be ten seconds of ticks further on,
    # not pinned to the final beat.
    a = t.quantize(last_beat + 1.0)
    b = t.quantize(last_beat + 10.0)
    assert b > a, "outro notes collapsed onto one tick"
    beats_apart = (b - a) / RESOLUTION
    assert 17 < beats_apart < 19, f"9s at 120bpm should be ~18 beats, got {beats_apart}"

    # And the mapping round-trips, so sustains and section times stay honest.
    for probe in (last_beat + 2.0, last_beat + 12.0):
        tick = t.quantize(probe)
        assert abs(t.beat_to_time(tick / RESOLUTION) - probe) < 0.3, probe


def test_playability_simulator_flags_and_fixes_the_unplayable():
    """Statistical gates passed charts that no hand could execute. This models
    the hands: strumming and re-shaping cost time, and the chart supplies a
    fixed amount. Human charts sit at ~1.3% impossible transitions."""
    from chartgen.playability import analyse, enforce

    t = steady(bpm=180.0, n=200)
    res = RESOLUTION
    # Alternating two-note chords on every 16th at 180 BPM: the hand would
    # have to re-form a shape every 83ms.
    brutal = []
    for i in range(80):
        tick = i * (res // 4)
        lanes = (0, 2) if i % 2 == 0 else (2, 4)
        brutal += [(tick, l, 0) for l in lanes]
    before = analyse(brutal, t)
    assert before["impossible"] > 0.5, before
    assert before["worst"], "should report where it breaks down"

    fixed = enforce(brutal, t)
    assert analyse(fixed, t)["impossible"] == 0.0
    assert {n[0] for n in fixed} <= {n[0] for n in brutal}, "no notes invented"

    # A comfortable chart is left completely alone.
    easy = [(i * res, i % 3, 0) for i in range(32)]
    assert analyse(easy, t)["impossible"] == 0.0
    assert enforce(easy, t) == sorted(easy)


def test_lrc_lines_become_ordered_phrases():
    from chartgen.lyrics import from_lines

    t = steady(bpm=120.0, n=400)
    lines = [(10.0, "You were the shadow to my light"),
             (14.0, "Did you feel us"),
             (60.0, "So lost, I'm faded")]
    events = from_lines(lines, t, duration_s=200.0)

    words = [e for _, e in events if e.startswith("lyric ")]
    assert len(words) == 15, words
    ticks = [tick for tick, e in events if e.startswith("lyric ")]
    assert ticks == sorted(ticks), "words must never move backwards"
    # Every phrase opens before its own first word and closes after its last.
    depth = 0
    for _, event in events:
        if event == "phrase_start":
            assert depth == 0, "phrases must not nest"
            depth = 1
        elif event == "phrase_end":
            assert depth == 1, "phrase closed without opening"
            depth = 0
        else:
            assert depth == 1, "a word landed outside any phrase"
    assert depth == 0

    # A line before a long instrumental break must not crawl across it: the
    # third line's words all sit near 60s, not spread to the end of the song.
    tail = [tick for tick, e in events if e.startswith("lyric ")][-3:]
    assert max(tail) / t.resolution < 70 * 2, tail


def test_lrc_parsing_skips_instrumental_markers():
    from chartgen.lrclib import parse_lrc

    # Blank timestamps mark instrumental breaks and a lone note glyph marks
    # a vocalise; both are real information, neither belongs on the highway.
    body = "\n".join((
        "[00:11.14]You were the shadow",
        "[00:14.00]",
        "[00:20.05] Another star",
        "[00:17.77]♪",
    ))
    lines = parse_lrc(body)
    assert lines == [(11.14, "You were the shadow"), (20.05, "Another star")], lines


def test_invented_outro_is_dropped_but_a_real_ending_survives():
    from chartgen.lyrics import _drop_invented_tail

    def seg(start, words):
        step = 0.3
        return (start, start + step * len(words),
                [(start + i * step, w) for i, w in enumerate(words)])

    verse = [seg(10.0, ["I", "walk", "alone"]), seg(14.0, ["through", "the", "night"])]

    # Measured in a real chart: eleven words of YouTube caption boilerplate,
    # every one stamped with the same timestamp, stranded after the outro.
    hallucinated = (95.0, 96.0, [(95.0, w) for w in
                                 "Thank you for watching I'll see you in the next one".split()])
    assert _drop_invented_tail(verse + [hallucinated]) == verse

    # A bare word marooned after a long instrumental break: also invented.
    assert _drop_invented_tail(verse + [seg(95.0, ["you"])]) == verse

    # But a genuine closing line, sung right after the last verse, stays.
    ending = seg(17.0, ["and", "then", "I", "was", "alone"])
    assert _drop_invented_tail(verse + [ending]) == verse + [ending]


def test_same_tick_lyrics_keep_their_written_order():
    from chartgen.chart import write_chart

    t = steady(bpm=120.0, n=64)
    # Whisper stamps a hallucinated phrase with one timestamp for every word;
    # sorting those lines by text read "thank i'll and for in next one".
    words = ["Thank", "you", "for", "watching"]
    lyrics = [(1920, "phrase_start")] + [(2400, f"lyric {w}") for w in words]         + [(2400, "phrase_end")]
    chart = write_chart({"ExpertSingle": [(0, 0, 0), (480, 1, 0)]}, t,
                        {"name": "x", "artist": "y", "charter": "z"},
                        "song.ogg", lyrics=lyrics)
    written = [line.split('"')[1] for line in chart.splitlines()
               if "lyric " in line or "phrase_" in line]
    assert written == ["phrase_start"] + [f"lyric {w}" for w in words] + ["phrase_end"],         written


def test_synced_lyrics_are_rejected_when_they_miss_the_audio():
    from chartgen.lyrics import _lands_on_sound

    sr = 22050
    # A song that is silent for its first half: a lyric sheet timed against a
    # different edit (a YouTube rip with an added intro matches on duration
    # but not on where the words land) puts every line in the silence.
    quiet = np.zeros(30 * sr, dtype=np.float32)
    tone = 0.5 * np.sin(2 * np.pi * 440 * np.arange(30 * sr) / sr)
    y = np.concatenate([quiet, tone]).astype(np.float32)

    assert not _lands_on_sound([(t, "x") for t in range(2, 28, 2)], y, sr)
    assert _lands_on_sound([(t, "x") for t in range(32, 58, 2)], y, sr)
    # With no audio to check against, the sheet is taken at its word.
    assert _lands_on_sound([(2.0, "x")], None, None)


def test_solo_scoring_rewards_the_standout_and_stays_quiet_otherwise():
    """The audio detector's scoring, on synthetic evidence.

    The full detector needs audio; the decision logic does not. A section
    that is brighter, more voiced AND more novel than the rest of its song
    must clear the swept threshold, and a song whose sections all look alike
    must produce nothing - 70% of human charts mark no solo, so silence is
    the default being protected here.
    """
    from chartgen import solo

    def rows(standout=None):
        out = []
        for i in range(8):
            hot = i == standout
            out.append({
                "voiced": 0.55 if hot else 0.25,
                "confidence": 0.4,
                "spread": 1.2,
                "bright": 1400.0 if hot else 950.0,
                "novelty": 0.55 if hot else 0.18,
            })
        return out

    spans = [(i * 16 * 480, (i + 1) * 16 * 480) for i in range(8)]

    z = solo._zscores(rows(standout=5))
    cands = solo._candidates(spans, z, 480)
    scores = {(a, b): sum(solo.WEIGHTS[k] * zz[k] for k in solo.FEATURES)
              for a, b, zz in cands}
    hot_span = spans[5]
    assert scores[hot_span] >= solo.MIN_Z, scores[hot_span]
    assert scores[hot_span] == max(scores.values())

    # A flat song: nothing may clear the bar.
    z = solo._zscores(rows(standout=None))
    cands = solo._candidates(spans, z, 480)
    assert all(sum(solo.WEIGHTS[k] * zz[k] for k in solo.FEATURES) < solo.MIN_Z
               for _, _, zz in cands)


def test_tap_phrases_need_soft_runs_and_respect_the_share_cap():
    from chartgen import taps

    res = RESOLUTION
    notes = [(i * res // 2, i % 5, 0) for i in range(60)]  # 8ths, 60 notes
    ticks = [t for t, _, _ in notes]

    # A soft island shorter than MIN_RUN never taps; a long one does, whole.
    soft = {t: -1.0 for t in ticks}
    for t in ticks[10:10 + taps.MIN_RUN - 1]:
        soft[t] = 1.5
    assert taps.phrases(notes, soft, res) == set()

    for t in ticks[20:36]:
        soft[t] = 1.5
    got = taps.phrases(notes, soft, res)
    assert got == set(ticks[20:36]), "the full soft run should tap"

    # A gap wider than MAX_GAP_BEATS splits a run; the halves stand alone.
    notes2 = [(i * res * 2, 0, 0) for i in range(20)]  # half notes: 2-beat gaps
    soft2 = {t: 1.5 for t, _, _ in notes2}
    assert taps.phrases(notes2, soft2, res) == set(),         "notes too far apart never form a tapped phrase"

    # Share cap: when everything reads soft, most of the chart must stay
    # strummed - tapping it all would just change instruments.
    soft3 = {t: 1.5 for t in ticks}
    got = taps.phrases(notes, soft3, res)
    assert len(got) <= taps.MAX_SHARE * len(ticks) + 1, len(got)


def test_tap_markers_written_only_where_the_tier_keeps_the_note():
    from chartgen.chart import _tier_block

    notes = [(0, 0, 0), (480, 1, 0), (960, 2, 0)]
    block = _tier_block("HardSingle", notes, [], taps={480, 5000})
    assert "480 = N 6 0" in block, "tap marker missing at a kept note"
    assert "5000 = N 6 0" not in block, "bare tap marker with no note is dead weight"
    # The modifier and its note share a tick and both survive sorting.
    lines = [l.strip() for l in block.splitlines() if "480" in l]
    assert "480 = N 1 0" in lines and "480 = N 6 0" in lines


def test_lrc_phrases_stay_on_screen_until_the_next_line():
    """Playtest: lines vanished while their last word was still sung. A
    phrase must close when the NEXT line begins (LRC line timing means
    'displayed until then'), not a quarter-beat after its last word."""
    from chartgen.lyrics import from_lines

    t = steady(bpm=120.0, n=400)
    lines = [(10.0, "first line of words here"), (18.0, "second line"),
             (26.0, "final line")]
    events = from_lines(lines, t, duration_s=120.0)

    ends = [tick for tick, e in events if e == "phrase_end"]
    starts = [tick for tick, e in events if e == "phrase_start"]
    res = t.resolution
    # First phrase holds until just before the second line starts (18s at
    # 120bpm = beat 36), not until shortly after its last word (~11s).
    assert ends[0] >= int(17.0 * 2 * res), (ends[0], "closed too early")
    assert ends[0] <= starts[1], "phrases must not overlap"
    # The last line rings out rather than dying instantly.
    assert ends[-1] >= int(27.0 * 2 * res)


def test_smooth_chord_shapes_unlocks_riffs_but_keeps_progressions():
    from chartgen.texture import smooth_chord_shapes

    res = RESOLUTION
    # Jittering shapes half a beat apart: (0,2) (1,2) (0,2) (1,3) - all the
    # same 2-note chord wobbling by one lane. Humans strum ONE shape.
    notes = []
    for i, shape in enumerate([(0, 2), (1, 2), (0, 2), (1, 3)]):
        for lane in shape:
            notes.append((i * res // 2, lane, 0))
    out = smooth_chord_shapes(notes, res)
    shapes = {}
    for t, lane, _ in out:
        shapes.setdefault(t, []).append(lane)
    kinds = {tuple(sorted(v)) for v in shapes.values()}
    assert kinds == {(0, 2)}, f"flicker should collapse to one shape: {kinds}"

    # A real progression - anchor moves two lanes - is untouched.
    notes = []
    for i, shape in enumerate([(0, 2), (2, 4), (0, 2), (2, 4)]):
        for lane in shape:
            notes.append((i * res, lane, 0))
    out = smooth_chord_shapes(notes, res)
    shapes = {}
    for t, lane, _ in out:
        shapes.setdefault(t, []).append(lane)
    assert {tuple(sorted(v)) for v in shapes.values()} == {(0, 2), (2, 4)}


def test_narrow_wide_chords_only_when_rushed():
    from chartgen.texture import narrow_wide_chords

    res = RESOLUTION
    # A wide chord right next to a different shape narrows to span 2.
    rushed = [(0, 0, 0), (0, 4, 0), (res // 2, 1, 0), (res // 2, 2, 0)]
    out = narrow_wide_chords(rushed, res)
    first = sorted(l for t, l, _ in out if t == 0)
    assert max(first) - min(first) <= 2, first

    # The same wide chord isolated stays wide - an accent, not a problem.
    lonely = [(0, 0, 0), (0, 4, 0), (4 * res, 2, 0)]
    assert sorted(narrow_wide_chords(lonely, res)) == sorted(lonely)


def test_consolidate_gallops_keeps_repeats_kills_oneoffs():
    from chartgen.texture import consolidate_gallops

    res = RESOLUTION
    s16 = res // 4
    # An isolated 16th-16th fragment in an otherwise-8th passage: the
    # middle (off-eighth) note goes, restoring the pulse.
    ticks = [0, res // 2, res, res + s16, res + 2 * s16, 2 * res, 3 * res]
    notes = [(t, 0, 0) for t in ticks]
    out = consolidate_gallops(notes, res)
    assert (res + s16) not in {t for t, _, _ in out}, "one-off middle should go"

    # The same figure repeating a bar later is a rhythm and stays whole.
    notes = [(t, 0, 0) for t in ticks] +         [(t + 4 * res, 0, 0) for t in (res, res + s16, res + 2 * s16)]
    out = consolidate_gallops(sorted(notes), res)
    kept = {t for t, _, _ in out}
    assert (res + s16) in kept and (5 * res + s16) in kept, "repeats must stay"


def test_climax_chords_grow_sparingly_at_sections():
    from chartgen.texture import climax_chords

    res = RESOLUTION
    # A 2-note chord sitting isolated right at a section entry grows to 3.
    notes = [(0, 0, 0), (4 * res, 0, 240), (4 * res, 2, 240), (8 * res, 1, 0)]
    out = climax_chords(notes, [(4 * res, "Chorus")], res)
    grown = sorted(l for t, l, _ in out if t == 4 * res)
    assert grown == [0, 1, 2], grown
    # The new note inherits the chord's (uniform) sustain.
    assert {sus for t, _, sus in out if t == 4 * res} == {240}

    # No sections, no growth; and never more than the cap.
    assert sorted(climax_chords(notes, [], res)) == sorted(notes)


def test_low_lone_pitches_become_open_notes():
    from chartgen.transcribe import expert_from_notes

    t = steady(bpm=120.0, n=200)
    # Melody around MIDI 60-72, plus lone low hits at 41 - an octave-plus
    # below the melodic body. The low hits should come out as opens
    # (lane 7), the melody as fretted lanes, and a low note that COINCIDES
    # with a melody note must not form an open chord.
    events = []
    for i in range(24):
        events.append((1.0 + i * 0.5, 1.2 + i * 0.5, 60 + (i % 12), 0.8))
    for k in range(6):
        events.append((1.25 + k * 2.0, 1.4 + k * 2.0, 41, 0.8))
    notes = expert_from_notes(events, t, subdiv=4)
    lanes_by_tick = {}
    for tick, lane, _ in notes:
        lanes_by_tick.setdefault(tick, set()).add(lane)
    opens = [t2 for t2, ls in lanes_by_tick.items() if 7 in ls]
    assert opens, "low lone notes should become opens"
    assert all(ls == {7} for t2, ls in lanes_by_tick.items() if 7 in ls),         "opens must never be part of a chord"
    fretted = [ls for ls in lanes_by_tick.values() if 7 not in ls]
    assert fretted, "melody must stay fretted"


def test_chord_sustains_are_uniform():
    from chartgen.transcribe import expert_from_notes

    t = steady(bpm=120.0, n=64)
    # Two simultaneous pitches with different real durations: the chart
    # chord must carry ONE sustain value, not a disjoint pair.
    events = [(2.0, 4.0, 60, 0.9), (2.0, 2.4, 64, 0.9), (6.0, 6.1, 62, 0.9)]
    notes = expert_from_notes(events, t, subdiv=4)
    by_tick = {}
    for tick, _, sus in notes:
        by_tick.setdefault(tick, []).append(sus)
    chords = [v for v in by_tick.values() if len(v) > 1]
    assert chords and all(len(set(v)) == 1 for v in chords), by_tick


def test_a_handful_of_hallucinated_words_is_not_lyrics():
    """Playtest: an entirely-piano song carried a single 'you'. One lone
    segment slips the tail guards (no gap precedes it, not boilerplate);
    the whole-song word floor is what catches it."""
    from chartgen.lyrics import MIN_SONG_WORDS, _drop_invented_tail

    lone = [(95.0, 95.4, [(95.0, "you")])]
    survived = _drop_invented_tail(lone)
    words = sum(len(w) for _, _, w in survived)
    assert words < MIN_SONG_WORDS,         "a lone hallucinated word must fall below the song floor"


def test_tap_sections_classify_by_foreground_dominance():
    """Playtest verdict on the drums-share classifier: a backing beat
    vetoed taps on a piano-led song ("the piano was pretty much always
    main and there was like a snare to it"). Sections now classify by
    the melodic stem's FOREGROUND share: piano-over-beat taps, a
    band/vocal foreground does not."""
    from chartgen.taps import (foreground_by_span, FOREGROUND_SOFT_SHARE,
                               FOREGROUND_HARD_SHARE)

    sr = 44100
    quiet = np.full(30 * sr, 0.01, dtype=np.float32)
    loud = np.full(30 * sr, 0.3, dtype=np.float32)
    # First 30s: piano over a modest beat (other dominates regardless).
    # Last 30s: full band, melodic stem just one voice among four.
    mono = {
        "drums": np.concatenate([quiet * 8, loud]),
        "bass": np.concatenate([quiet, loud]),
        "other": np.concatenate([loud, loud * 0.8]),
        "vocals": np.concatenate([quiet * 0, loud]),
    }
    fg = foreground_by_span([(0.0, 30.0), (30.0, 60.0)], mono, sr)
    assert fg[0][0] >= FOREGROUND_SOFT_SHARE, fg
    assert fg[1][0] <= FOREGROUND_HARD_SHARE, fg


def test_tap_sections_key_on_piano_synth_minus_guitar():
    """Taps v4 (six-stem study, 2026-09-03): charters tap where the guitar
    stem is silent and piano/synth carry the section. A piano-led span
    scores high, a guitar-led span negative, a shared span in between;
    without the true stems the helper declines so v3 takes over."""
    from chartgen.taps import (keys_by_span, KEYS_TAP_SHARE,
                               KEYS_NO_TAP_SHARE, TRUE_STEMS)

    sr = 44100
    q = np.full(20 * sr, 0.01, dtype=np.float32)
    L = np.full(20 * sr, 0.3, dtype=np.float32)
    # 0-20s piano over a beat, 20-40s guitar band, 40-60s piano and guitar
    # sharing the foreground equally.
    mono = {
        "drums": np.concatenate([q * 8, L, L]),
        "bass": np.concatenate([q, L, L]),
        "vocals": np.concatenate([q * 0, L, L]),
        "guitar": np.concatenate([q * 0, L * 1.5, L]),
        "piano": np.concatenate([L, q * 0, L]),
        "sw_other": np.concatenate([q, q, q]),
    }
    mono["other"] = mono["guitar"] + mono["piano"] + mono["sw_other"]  # shim
    spans = [(0.0, 20.0), (20.0, 40.0), (40.0, 60.0)]
    keyed = keys_by_span(spans, mono, sr)
    assert keyed[0] >= KEYS_TAP_SHARE, keyed
    assert keyed[1] <= KEYS_NO_TAP_SHARE, keyed
    assert KEYS_NO_TAP_SHARE < keyed[2] < KEYS_TAP_SHARE, keyed
    four = {k: mono[k] for k in ("drums", "bass", "other", "vocals")}
    assert keys_by_span(spans, four, sr) is None
    assert set(TRUE_STEMS) <= set(mono)
    # A keyed section taps whole, chords included: mixing taps and strums
    # inside one section was playtested on Clocks and rejected.
    from chartgen.taps import section_tap_ticks
    inside = [(0, 0, 0), (100, 1, 0), (100, 2, 0), (200, 7, 0), (300, 3, 0)]
    assert section_tap_ticks(inside) == {0, 100, 200, 300}
    assert section_tap_ticks(inside, tap_chords=False) == {0, 200, 300}


def test_rapid_chords_demote_a_chord_beside_a_16th_single():
    """Bisected Clocks regression (playability finger-lift fix let it
    through): a chord with a single within a 16th on either side demotes
    to its root; chords a full 8th from any single keep their shape."""
    from chartgen import reduce

    res = 192
    six = res // 4
    # chord, single a 16th later, chord: both chords demote to roots
    notes = [(0, 0, 0), (0, 1, 0), (six, 2, 0), (2 * six, 0, 0), (2 * six, 1, 0)]
    out = reduce.simplify_rapid_chords(notes, res)
    lanes = {}
    for t, l, _ in out:
        lanes.setdefault(t, set()).add(l)
    assert lanes[0] == {0} and lanes[2 * six] == {0} and lanes[six] == {2}, lanes
    # same chords an 8th apart from a single on the grid: chord survives
    notes = [(0, 0, 0), (0, 1, 0), (res // 2, 2, 0), (res, 0, 0), (res, 1, 0)]
    out = reduce.simplify_rapid_chords(notes, res)
    lanes = {}
    for t, l, _ in out:
        lanes.setdefault(t, set()).add(l)
    assert lanes[0] == {0, 1} and lanes[res] == {0, 1}, lanes
    # cascade: single, then three different-shape chords at 16ths - all
    # unravel; a same-shape 16th chug after one single keeps its chords
    arp = [(0, 2, 0), (six, 0, 0), (six, 1, 0), (2 * six, 1, 0), (2 * six, 2, 0),
           (3 * six, 2, 0), (3 * six, 3, 0)]
    out = reduce.simplify_rapid_chords(arp, res)
    assert all(len({l for t2, l, _ in out if t2 == t}) == 1
               for t in (six, 2 * six, 3 * six)), out
    chug = [(0, 2, 0)] + [(k * six, l, 0) for k in (1, 2, 3, 4) for l in (0, 1)]
    out = reduce.simplify_rapid_chords(chug, res)
    kept = {t for t, l, _ in out if l == 1}
    # (off-8th chords 48/144 fall to the no-tempo era's grid rule; the
    # on-grid chords prove the cascade stopped at the same-shape boundary)
    assert kept >= {2 * six, 4 * six}, out


def test_prominence_windows_smoothing_and_runs():
    """Route B runtime pieces: 4-bar windows on the beat grid, one-window
    blips smoothed to their neighbours, consecutive same-stem windows
    merged into runs, and the feature vector shaped for the model."""
    from chartgen import prominence
    from chartgen.tempo import TempoMap

    tm = TempoMap(beat_times=np.arange(0, 120, 0.5), pickup_beats=0)  # 120 bpm
    spans = prominence.windows_for(tm, 40.0)
    assert spans[0][:2] == (0.0, 8.0) and spans[0][2:] == (0, 16), spans[0]
    assert all(b1 - b0 == 16 for _, _, b0, b1 in spans[:-1])
    assert spans[-1][1] == 40.0
    assert prominence.smooth(["guitar", "sw_other", "guitar", "piano", "piano"]) == \
        ["guitar", "guitar", "guitar", "piano", "piano"]
    # "no opinion" never spreads over a confident neighbour; a lone
    # no-opinion window between two agreeing neighbours takes their label
    assert prominence.smooth([None, "bass", None, "bass", None]) == [None, "bass", "bass", "bass", None]
    assert prominence.smooth([None, "bass", None, None, "bass"]) == [None, "bass", None, None, "bass"]
    wins = [{"t0": i * 8.0, "t1": (i + 1) * 8.0, "beat0": i * 16, "beat1": (i + 1) * 16,
             "stem": st, "conf": 0.9} for i, st in enumerate(["guitar", "guitar", None, "piano"])]
    rs = prominence.runs(wins)
    assert [r["stem"] for r in rs] == ["guitar", None, "piano"]
    assert rs[0]["t1"] == 16.0 and rs[0]["beat1"] == 32
    assert prominence.letters(wins) == "GG?P"
    feats = {s: {k: 0.1 for k in prominence.FEATS} for s in prominence.STEMS}
    feats["drums_energy"] = 0.2
    vec = prominence.feature_vector(feats, None, feats)
    assert len(vec) == 5 * 5 + 1 + 2 * 10, len(vec)


def test_taps_follow_the_instrument_runs():
    """With a followed-instrument timeline, tap sections ARE the runs: a
    synth/piano run taps whole, a guitar run stays strummed, a run with
    no opinion falls back to the keyed-share rule on its own span."""
    from chartgen import taps
    from chartgen.tempo import TempoMap

    tm = TempoMap(beat_times=np.arange(0, 120, 0.5), pickup_beats=0)  # 120 bpm, 8 s per run
    res = tm.resolution
    sr = 44100
    q = np.full(24 * sr, 0.01, dtype=np.float32)
    L = np.full(24 * sr, 0.3, dtype=np.float32)
    # audio: synth-led throughout, so the keyed rule alone would tap everything
    mono = {"drums": q, "bass": q, "vocals": q * 0, "guitar": q * 0, "piano": q, "sw_other": L}
    mono["other"] = mono["guitar"] + mono["piano"] + mono["sw_other"]
    notes = [(t, 0, 0) for t in range(0, 48 * res, res // 2)]  # 8ths for 24 s
    runs = [{"t0": 0.0, "t1": 8.0, "beat0": 0, "beat1": 16, "stem": "sw_other", "conf": 0.9},
            {"t0": 8.0, "t1": 16.0, "beat0": 16, "beat1": 32, "stem": "guitar", "conf": 0.9},
            {"t0": 16.0, "t1": 24.0, "beat0": 32, "beat1": 48, "stem": None, "conf": 0.4}]
    # bass is neutral: a bass run between two synth runs taps with them
    bass_runs = [{"t0": 0.0, "t1": 8.0, "beat0": 0, "beat1": 16, "stem": "sw_other", "conf": 0.9},
                 {"t0": 8.0, "t1": 16.0, "beat0": 16, "beat1": 32, "stem": "bass", "conf": 0.9},
                 {"t0": 16.0, "t1": 24.0, "beat0": 32, "beat1": 48, "stem": "sw_other", "conf": 0.9}]
    from chartgen import stems as stemsmod
    stemsmod._CACHE.clear()
    stemsmod._CACHE["fake"] = mono
    real = stemsmod.separate
    stemsmod.separate = lambda path, progress=None, backend=None: mono
    try:
        chosen = taps.detect(notes, np.zeros(sr), sr, tm, audio_path="fake",
                             section_marks=[(0, "a")], followed=runs)
        chosen_bass = taps.detect(notes, np.zeros(sr), sr, tm, audio_path="fake",
                                  section_marks=[(0, "a")], followed=bass_runs)
    finally:
        stemsmod.separate = real
        stemsmod._CACHE.clear()
    first = {t for t, _, _ in notes if t < 16 * res}
    second = {t for t, _, _ in notes if 16 * res <= t < 32 * res}
    third = {t for t, _, _ in notes if t >= 32 * res}
    assert first <= chosen, "synth run taps whole"
    assert not (second & chosen), "guitar run stays strummed even on synth-led audio"
    assert third <= chosen, "no-opinion run falls back to the keyed rule (synth-led -> tap)"
    assert second <= chosen_bass, "a bass run between synth runs adopts their taps"


def test_rhythm_rescue_supplies_a_starved_synth_run():
    """A synth-followed run whose chart is nearly empty gains notes at the
    stem's onsets (16th-quantized, laned 0-4, none on top of existing
    notes); a guitar-followed run and a well-charted run are untouched."""
    from chartgen import prominence
    from chartgen.tempo import TempoMap

    sr = 44100
    tm = TempoMap(beat_times=np.arange(0, 60, 0.5), pickup_beats=0)  # 120 bpm
    res = tm.resolution
    # 8 s of 8th-note clicks (4 Hz) with a slow brightness sweep
    t = np.arange(0, 8.0, 1 / sr)
    sig = np.zeros_like(t, dtype=np.float32)
    for k in range(32):
        a = int(k * 0.25 * sr)
        n = np.arange(0, int(0.05 * sr))
        f = 200 + 40 * k
        sig[a:a + len(n)] = (np.sin(2 * np.pi * f * n / sr) * np.exp(-n / (0.01 * sr))).astype(np.float32)
    silent = np.zeros(8 * sr, dtype=np.float32)
    mono = {"sw_other": np.concatenate([sig, sig]), "bass": np.concatenate([silent, silent]),
            "guitar": np.concatenate([silent, silent]), "vocals": np.concatenate([silent, silent]),
            "piano": np.concatenate([silent, silent]), "drums": np.concatenate([silent, silent])}
    expert = [(0, 0, 0), (4 * res, 1, 0), (6 * res, 7, 0)]  # three notes in 16 s: starved; one open
    runs = [{"t0": 0.0, "t1": 8.0, "beat0": 0, "beat1": 16, "stem": "sw_other", "conf": 0.9},
            {"t0": 8.0, "t1": 16.0, "beat0": 16, "beat1": 32, "stem": "guitar", "conf": 0.9}]
    out, added, touched = prominence.rhythm_rescue(expert, runs, mono, sr, tm)
    assert touched == 1 and added >= 20, (added, touched)
    new = [n for n in out if n not in expert]
    assert all(0 <= l <= 4 and s == 0 for _, l, s in new)
    assert not any(l == 7 for _, l, _ in out), "the pre-existing open is re-laned from the contour"
    assert any(t == 6 * res for t, _, _ in out), "and kept at its tick"
    assert all(tk % (res // 4) == 0 for tk, _, _ in new), "16th grid"
    assert all(tk < 16 * res for tk, _, _ in new), "guitar run untouched"
    assert len({l for _, l, _ in new}) >= 3, "brightness sweep lanes the run"
    # a well-charted run is left alone
    full = [(k * res // 2, 2, 0) for k in range(32)]
    out2, added2, _ = prominence.rhythm_rescue(full, runs[:1], mono, sr, tm)
    assert added2 == 0 and out2 == full


def test_star_power_prefers_typical_bars_over_the_densest():
    """Human phrases sit at the song's typical density (1.04x, 12 notes);
    ours went on the densest bars (16-23 notes) and the player banked
    half the meter. With one 2-bar burst of 16ths in an 8th-note song,
    no phrase lands on the burst."""
    res = RESOLUTION
    bar = 4 * res
    notes = [(i * (res // 2), i % 5, 0) for i in range(40 * 8)]          # 8ths, 40 bars
    burst = [(10 * bar + k * (res // 4), k % 5, 0) for k in range(32)]   # bars 10-11 in 16ths
    notes = sorted(set(notes + burst))
    phrases = star_power_phrases(notes, res, duration_s=120.0)
    assert phrases, phrases
    assert all(not (start <= 10 * bar < start + length) for start, length in phrases), phrases


def test_solo_hysteresis_runs():
    """A run needs SOLO_MIN_WINDOWS core windows, grows over soft
    neighbours, and ignores short cores; the solo vector has the
    followed-instrument vector, probabilities, song-relative stats,
    position, length and four neighbours' guitar density."""
    from chartgen import prominence
    probs = [0.1, 0.5, 0.9, 0.9, 0.9, 0.9, 0.5, 0.1, 0.9, 0.9, 0.1, 0.2]
    assert prominence.solo_hysteresis(probs, 0.7, 0.4, 4, 0) == [(1, 6)]
    assert prominence.solo_hysteresis(probs, 0.7, 0.4, 2, 0) == [(1, 6), (8, 9)]
    assert prominence.solo_hysteresis([0.9] * 3, 0.7, 0.4, 4, 0) == []
    feats = []
    for i in range(3):
        f = {s: {k: 0.1 for k in prominence.FEATS} for s in prominence.STEMS}
        f["drums_energy"] = 0.2
        f["_t0"], f["_t1"] = i * 8.0, (i + 1) * 8.0
        feats.append(f)
    vec = prominence.solo_vector(feats, np.full((3, 5), 0.2), 1, 24.0)
    assert len(vec) == 46 + 5 + 5 + 4, len(vec)


def test_keyed_rescue_fills_a_starved_window_from_the_followed_stem():
    """A starved piano-followed window gains the piano stem's events (not
    the guitar's), on empty ticks only; a well-charted window and a
    window followed on an inaudible stem stay untouched."""
    from chartgen import prominence
    from chartgen.tempo import TempoMap

    tm = TempoMap(beat_times=np.arange(0, 60, 0.5), pickup_beats=0)  # 120 bpm: a window = 8 s
    beat = 0.5
    mix = [(0.0, 0.3, 60, 0.8), (4 * beat, 0.3, 62, 0.8)]            # two mix notes in window 0
    piano = [(k * beat, 0.3, 64 + (k % 5), 0.6) for k in range(32)]  # 8ths for 16 s
    guitar = [(k * beat + 0.25 * beat, 0.3, 50, 0.6) for k in range(32)]
    feats = {s: {k: 0.1 for k in prominence.FEATS} for s in prominence.STEMS}
    feats["drums_energy"] = 0.1
    quiet = {s: {k: 0.0 for k in prominence.FEATS} for s in prominence.STEMS}
    quiet["drums_energy"] = 0.1
    windows = [{"t0": 0.0, "t1": 8.0, "beat0": 0, "beat1": 16, "stem": "piano", "conf": 0.9, "features": feats},
               {"t0": 8.0, "t1": 16.0, "beat0": 16, "beat1": 32, "stem": "piano", "conf": 0.9, "features": quiet}]
    extra, touched, counts = prominence.keyed_rescue_events(
        mix, windows, {"piano": piano, "guitar": guitar}, tm)
    assert touched == 1 and counts == {"piano": len(extra)}, (touched, counts)
    assert all(s0 < 8.0 for s0, _, _, _ in extra), "the inaudible window is untouched"
    assert all(p >= 64 for _, _, p, _ in extra), "piano events only"
    taken = {tm.quantize(s0, subdiv=4) for s0, _, _, _ in mix}
    assert not any(tm.quantize(s0, subdiv=4) in taken for s0, _, _, _ in extra), "empty ticks only"
    # well charted: the mix already holds most of the stem's positions
    full = [(k * beat, 0.3, 60, 0.8) for k in range(16)]
    extra2, touched2, _ = prominence.keyed_rescue_events(full, windows[:1], {"piano": piano}, tm)
    assert touched2 == 0 and not extra2


def test_stem_rescue_admits_chords_on_empty_ticks():
    """Two comparable stem notes on an empty tick come through as a chord;
    a weak partner does not; a tick the mix already holds admits nothing."""
    from chartgen import transcribe
    from chartgen.tempo import TempoMap

    tm = TempoMap(beat_times=np.arange(0, 60, 0.5), pickup_beats=0)
    beat = 0.5
    kept = [(0.0, 0.3, 60, 0.8)]
    stem = [(beat, 0.3, 64, 0.6), (beat, 0.3, 67, 0.5),          # comparable pair -> chord
            (2 * beat, 0.3, 62, 0.6), (2 * beat, 0.3, 69, 0.2),  # weak partner -> single
            (0.0, 0.3, 72, 0.9)]                                  # taken tick -> nothing
    out = transcribe.admit_stem_events(stem, [(0.0, 4.0)], kept, tm, 40, 0.2)
    ticks = {}
    for s0, _, p, _ in out:
        ticks.setdefault(tm.quantize(s0, subdiv=4), []).append(p)
    assert sorted(ticks[tm.quantize(beat, subdiv=4)]) == [64, 67]
    assert ticks[tm.quantize(2 * beat, subdiv=4)] == [62]
    assert tm.quantize(0.0, subdiv=4) not in ticks


def test_chordify_respects_allowed_spans():
    """With a timeline, chord texture voices only runs inside guitar-
    followed spans; the same evidence outside them is left single."""
    from chartgen import guitar as G

    res = 480
    run_a = [(k * (res // 2), 0, 0) for k in range(8)]            # ticks 0-1680
    run_b = [(8 * res + k * (res // 2), 0, 0) for k in range(8)]  # ticks 3840-5520
    voices = {t: (40, 47) for t, _, _ in run_a + run_b}          # a fifth everywhere
    out, added = G.chordify(run_a + run_b, voices, res, allowed_spans=[(0, 4 * res)])
    lanes = {}
    for t, l, _ in out:
        lanes.setdefault(t, set()).add(l)
    assert all(len(lanes[t]) == 2 for t, _, _ in run_a), "run inside the span is voiced"
    assert all(len(lanes[t]) == 1 for t, _, _ in run_b), "run outside stays single"
    out_all, added_all = G.chordify(run_a + run_b, voices, res)
    assert added_all > added



def test_disk_cache_round_trips_exactly():
    """Cached stems and transcriptions must come back bit-identical: the
    cache exists to skip recomputation, never to change a chart. Stems stay
    float32 (a float16 or FLAC copy would quantise), transcriptions ride
    json's exact float repr."""
    import os
    import tempfile

    from chartgen import cache

    with tempfile.TemporaryDirectory() as tmp:
        old = os.environ.get("CHARTGEN_CACHE_DIR")
        os.environ["CHARTGEN_CACHE_DIR"] = tmp
        try:
            rng = np.random.default_rng(3)
            raw = {name: rng.standard_normal(4410).astype(np.float32) * 0.3
                   for name in ("drums", "bass", "other", "vocals", "guitar", "piano")}
            cache.save_stems("sw:test:abc", raw)
            back = cache.load_stems("sw:test:abc")
            assert back is not None and set(back) == set(raw)
            for name in raw:
                assert back[name].dtype == np.float32
                assert np.array_equal(back[name], raw[name])
            assert cache.load_stems("sw:test:missing") is None

            events = [(0.123456789012, 0.98765432101, 64, 0.5000000001),
                      (1.0 / 3.0, 2.0 / 7.0, 40, 0.1234567890123)]
            cache.save_transcription("mix:abc:bp", events)
            assert cache.load_transcription("mix:abc:bp") == events
            assert cache.load_transcription("mix:zzz:bp") is None

            # the size cap evicts the least recently used entry, never the
            # one just written
            os.environ["CHARTGEN_STEM_CACHE_GB"] = str(120_000 / 2**30)
            cache.save_stems("sw:test:second", raw)
            assert cache.load_stems("sw:test:second") is not None
            assert cache.load_stems("sw:test:abc") is None
        finally:
            os.environ.pop("CHARTGEN_STEM_CACHE_GB", None)
            if old is None:
                os.environ.pop("CHARTGEN_CACHE_DIR", None)
            else:
                os.environ["CHARTGEN_CACHE_DIR"] = old


def test_solo_evidence_is_the_same_with_prefetched_analysis():
    """section_evidence(analysis=analyze(y, sr)) must equal the inline
    computation: the pipeline runs analyze() on a thread while the GPU
    separates, and the chart must not know the difference."""
    from chartgen import solo

    sr = 22050
    rng = np.random.default_rng(7)
    t = np.arange(sr * 12) / sr
    y = (0.3 * np.sin(2 * np.pi * 330 * t) * (t < 6)
         + 0.3 * np.sin(2 * np.pi * 660 * t) * (t >= 6)
         + 0.02 * rng.standard_normal(len(t))).astype(np.float32)
    tm = steady(bpm=120.0, n=32, first=0.0)
    res = tm.resolution
    spans = [(0, 8 * res), (8 * res, 16 * res), (16 * res, 24 * res)]
    inline = solo.section_evidence(y, sr, tm, spans, res)
    ahead = solo.section_evidence(y, sr, tm, spans, res, analysis=solo.analyze(y, sr))
    assert len(inline) == len(ahead)
    for a, b in zip(inline, ahead, strict=True):
        assert (a is None) == (b is None)
        if a is not None:
            assert a == b, (a, b)


def test_rapid_chords_promote_singles_inside_guitar_spans():
    """Inside a guitar-followed span the chord-single-chord flicker resolves
    by promotion (the single takes the chord's shape); outside it the
    chords demote as before."""
    from chartgen import reduce

    res = 192
    six = res // 4
    notes = [(0, 0, 0), (0, 1, 0), (six, 2, 0), (2 * six, 0, 0), (2 * six, 1, 0)]

    def lanes(out):
        d = {}
        for t, l, _ in out:
            d.setdefault(t, set()).add(l)
        return d

    inside = lanes(reduce.simplify_rapid_chords(notes, res, chord_spans=[(0, 4 * res)]))
    assert inside[0] == {0, 1} and inside[six] == {0, 1} and inside[2 * six] == {0, 1}, inside
    outside = lanes(reduce.simplify_rapid_chords(notes, res, chord_spans=[(8 * res, 12 * res)]))
    assert outside[0] == {0} and outside[2 * six] == {0}, outside


def test_rhythm_rescue_voices_guitar_onsets_with_the_run_chord():
    """A starved guitar-followed run gains chord chugs at the stem's onsets,
    shaped like the chords already in the run; nothing lands on existing
    positions."""
    from chartgen import prominence
    from chartgen.tempo import TempoMap

    sr = 44100
    tm = TempoMap(beat_times=np.arange(0, 60, 0.5), pickup_beats=0)  # 120 bpm
    res = tm.resolution
    t = np.arange(0, 8.0, 1 / sr)
    sig = np.zeros_like(t, dtype=np.float32)
    for k in range(32):                                        # 8th-note chugs
        a = int(k * 0.25 * sr)
        n = np.arange(0, int(0.05 * sr))
        sig[a:a + len(n)] = (np.sin(2 * np.pi * 110 * n / sr) * np.exp(-n / (0.01 * sr))).astype(np.float32)
    silent = np.zeros(8 * sr, dtype=np.float32)
    mono = {"guitar": sig, "sw_other": silent, "bass": silent, "vocals": silent, "piano": silent, "drums": silent}
    expert = [(0, 0, 0), (0, 1, 0), (4 * res, 0, 0), (4 * res, 1, 0)]   # two G+R chords in 8 s: starved
    runs = [{"t0": 0.0, "t1": 8.0, "beat0": 0, "beat1": 16, "stem": "guitar", "conf": 0.9}]
    out, added, touched = prominence.rhythm_rescue(expert, runs, mono, sr, tm)
    assert touched == 1 and added >= 40, (added, touched)
    new_ticks = {tk for tk, _, _ in out} - {0, 4 * res}
    lanes = {tk: {l for t2, l, _ in out if t2 == tk} for tk in new_ticks}
    assert new_ticks and all(v == {0, 1} for v in lanes.values()), "voiced as the run's G+R chug"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
    print("\nall checks passed")
