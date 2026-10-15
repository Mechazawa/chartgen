"""Basic Pitch engine: transcribed notes -> Expert chart notes.

Basic Pitch (Apache-2.0 weights, ONNX runtime) gives real (start, end,
midi_pitch, amplitude) note events: exact pitch, actual polyphony, and
sustains from real note durations. It replaced the audio2chart neural charter
(retired 2026-09-07 after the engine comparison; the code lives on the
legacy-audio2chart branch).

What stands between a raw transcription and a playable chart — and what this
module does — is selection: a full mix transcribes EVERYTHING (bass, pads,
vocal melody, bells). A chart wants the lead line plus supporting chords.
"""

from itertools import pairwise

import numpy as np

# Below guitar low E: bass lines and kick-drum fundamentals. A guitar chart
# following the bass feels wrong even when the transcription is correct.
MIN_PITCH = 40
# Basic Pitch amplitudes run 0-1; quiet events are usually harmonic ghosts.
MIN_AMPLITUDE = 0.20
# Two, not three - re-verified 2026-08-30 after the playability fix made a
# relaxation experiment worth running. Raising to 3 flooded charts far past
# the human bands (a punk chart hit 11.8% three-note positions vs its human
# chart's 0%; EDM chord share 18%->29% with 39 rapid shape changes),
# because a third comparable-amplitude pitch also mints NEW two-note chords
# whenever two of the three map to distinct frets. The cap is doing real
# stylistic work the playability simulator cannot (reachable != charted by
# humans). Three-note shapes stay on the curated climax-chord route.
MAX_CHORD = 2
# A chord is several strings struck TOGETHER, so its notes have comparable
# energy. A lead note over a quiet pad is a single note with accompaniment —
# charting the pad as a second button is what buried the EDM songs in chords
# (41-54% of positions against a human 5-13%).
CHORD_AMPLITUDE_RATIO = 0.65
# Open notes (lane 7, the no-fret purple strum): 60% of human charts use
# them, as PUNCTUATION - median 4.5% of positions, median run length 1,
# never inside chords. Musically they are the register BELOW the melody:
# bass drops, chugs, pedal tones. A lone note this many semitones under
# the melodic body (p25 of kept pitches) becomes an open.
OPEN = 7
OPEN_GAP_SEMITONES = 5


def transcribe(audio_path: str, progress=lambda m: None, cache_key: str | None = None):
    """[(start_s, end_s, midi_pitch, amplitude)] via Basic Pitch (ONNX).

    cache_key names the audio content (chartgen.cache); when given, a
    previous transcription of the same content is returned as is, and a
    fresh one is stored. Basic Pitch on CPU is deterministic, so the cached
    events are the events it would produce again. onnxruntime's worker
    threads sleep instead of spinning between kernels, so several
    transcriptions at once (prominence.transcribe_stems) do not burn each
    other's cores; the events are bit-identical either way.
    """
    import logging

    from . import cache as diskcache

    if cache_key:
        hit = diskcache.load_transcription(cache_key)
        if hit is not None:
            progress("      transcription: cached notes reused")
            return hit
    logging.getLogger().setLevel(logging.ERROR)  # its TF warnings are noise here
    import onnxruntime as ort
    from basic_pitch import ICASSP_2022_MODEL_PATH
    from basic_pitch.inference import Model, predict

    model = Model(ICASSP_2022_MODEL_PATH)
    options = ort.SessionOptions()
    options.add_session_config_entry("session.intra_op.allow_spinning", "0")
    model.model = ort.InferenceSession(str(ICASSP_2022_MODEL_PATH), options,
                                       providers=["CPUExecutionProvider"])
    progress("      transcribing notes (Basic Pitch)")
    _, _, note_events = predict(str(audio_path), model)
    events = [(float(s), float(e), int(p), float(a))
              for s, e, p, a, *_ in note_events]
    if cache_key:
        diskcache.save_transcription(cache_key, events)
    return events


def cached_transcription(cache_key: str | None):
    """A previously stored transcription for this key, or None."""
    if not cache_key:
        return None
    from . import cache as diskcache

    return diskcache.load_transcription(cache_key)


def _fret_map(pitches: np.ndarray, n_frets: int = 5):
    """midi pitch -> fret via quantile bins over the song's own kept pitches.

    Quantile bins over the song's own kept pitches, on exact transcribed
    pitches: a repeated pitch is a repeated
    fret, rising lines rise, and both verse-register and solo-register songs
    use the whole neck.
    """
    lo = float(np.percentile(pitches, 5))
    hi = float(np.percentile(pitches, 95))
    if hi <= lo:
        return lambda p: 0
    def to_fret(p):
        return int(np.clip((p - lo) / (hi - lo) * n_frets, 0, n_frets - 1))
    return to_fret


def swung_beats(events, tempo,
                min_pitch: int = MIN_PITCH,
                min_amplitude: float = MIN_AMPLITUDE) -> set[int]:
    """Beat indices whose onsets sit on the triplet grid, not the 16th grid.

    43% of the sub-16th content in the 800-chart library is 16th-triplets
    (24ths) - shuffle and swing feel, not "faster notes" - and a swung song
    hard-quantized to straight 16ths is charted wrong by the community's own
    "chart the intent" rule. Per beat, onsets vote: a position near 1/6, 1/3,
    2/3 or 5/6 of a beat fits only the triplet grid; near 1/4 or 3/4 fits
    only the 16th grid (0 and 1/2 are shared and say nothing). A beat swings
    when triplet-only onsets outnumber 16th-only ones there AND the song as
    a whole shows a real triplet population - the song gate stops one noisy
    beat from swinging an otherwise straight chart.

    Per-beat (not per-onset) grid choice is what keeps the result playable:
    two grids inside one beat can put notes 1/12 of a beat apart, which is
    brokenNote territory; whole beats on either grid keep every gap >= 1/6.

    OPT-IN (--swing), not a default. Measured on cached transcriptions of a
    straight EDM song vs a shuffle rock song: the detected beat grid's local
    phase error (librosa frame jitter plus interpolation, ~±40ms) exceeds
    the 55ms that separates a 16th slot from a triplet slot, and Basic
    Pitch's own onset timing smears the inter-onset histogram so badly that
    the straight song produced MORE triplet-only votes than straight votes
    (235 vs 133). No gate on top of votes that noisy can hold. ponytail:
    revisit if beat tracking ever gets tick-accurate.
    """
    TOL = 0.045  # beats; ~30ms at 120 BPM, tighter than half a 24th
    trip_votes: dict[int, int] = {}
    straight_votes: dict[int, int] = {}
    for start, _, pitch, amp in events:
        if pitch < min_pitch or amp < min_amplitude:
            continue
        beat = tempo.time_to_beat(start)
        b, frac = int(beat), beat % 1.0
        trip = min(abs(frac - g) for g in (1/6, 1/3, 2/3, 5/6))
        straight = min(abs(frac - g) for g in (0.25, 0.75))
        if trip <= TOL < straight:
            trip_votes[b] = trip_votes.get(b, 0) + 1
        elif straight <= TOL < trip:
            straight_votes[b] = straight_votes.get(b, 0) + 1
    total_trip = sum(trip_votes.values())
    # Song gate: a genuine shuffle produces dozens of triplet-only onsets
    # spread across many beats; jitter produces a scatter of single votes.
    if total_trip < 12 or len(trip_votes) < 8:
        return set()
    if total_trip < 0.5 * sum(straight_votes.values()):
        return set()
    return {b for b, v in trip_votes.items()
            if v >= 2 and v > straight_votes.get(b, 0)}


def expert_from_notes(
    events,
    tempo,
    subdiv: int = 4,
    min_pitch: int = MIN_PITCH,
    min_amplitude: float = MIN_AMPLITUDE,
    min_sustain_beats: float = 0.5,
    allow_opens: bool = True,
    swing_beats: set[int] = frozenset(),
    ornaments: bool = True,
    max_chord: int = MAX_CHORD,
) -> list[tuple[int, int, int]]:
    """Chart-ready Expert notes [(tick, lane, sustain_ticks)].

    Selection rules, in order:
    - register + amplitude filters drop bass fundamentals and harmonic ghosts
    - events quantize onto the same grid the rest of the pipeline uses
    - per tick, the loudest MAX_CHORD pitches survive; distinct frets after
      mapping form a chord (real voicings, capped for playability)
    - sustain when the transcribed note itself rings for >= 3/4 beat, released
      a 16th early, capped at 4 beats — real durations, not the gap heuristic
    """
    res = tempo.resolution
    kept = [(s, e, p, a) for s, e, p, a in events
            if p >= min_pitch and a >= min_amplitude]
    if not kept:
        return []

    pitches = np.array([p for _, _, p, _ in kept], dtype=float)
    fret_of = _fret_map(pitches)
    # Below this, a lone note is an open: clearly under the melodic body,
    # not just at the bottom of it.
    open_cut = min(float(np.percentile(pitches, 10)),
                   float(np.percentile(pitches, 25)) - OPEN_GAP_SEMITONES)

    by_tick: dict[int, list[tuple[float, int, float]]] = {}
    for start, end, pitch, amp in kept:
        # Swung beats quantize to the triplet grid (subdiv 6); everything
        # else keeps the straight grid. Chosen per whole beat, so no two
        # grids ever mix inside one (see swung_beats).
        beat_subdiv = 6 if int(tempo.time_to_beat(start)) in swing_beats else subdiv
        tick = tempo.quantize(start, subdiv=beat_subdiv)
        if tick < 0:
            continue
        by_tick.setdefault(tick, []).append((amp, pitch, end - start))

    ticks = sorted(by_tick)
    next_tick = dict(pairwise(ticks))
    floor = res // 2          # shorter reads as a tap, not a sustain
    release = res // 4
    cap = 4 * res

    notes: list[tuple[int, int, int]] = []
    for tick in ticks:
        group = sorted(by_tick[tick], reverse=True)[:max_chord]
        loudest = group[0][0]
        group = [g for g in group if g[0] >= loudest * CHORD_AMPLITUDE_RATIO]
        lanes: dict[int, float] = {}
        if allow_opens and len(group) == 1 and group[0][1] <= open_cut:
            # Lone low note: the purple strum. Never inside a chord - the
            # human corpus has 0.0% open+fret chords.
            lanes[OPEN] = group[0][2]
        else:
            for _, pitch, duration in group:
                lane = fret_of(pitch)
                lanes[lane] = max(lanes.get(lane, 0.0), duration)

        # One sustain for the whole position: chord members with different
        # lengths are "disjoint chords", which human charts never write
        # (median 0 per song; we accidentally wrote ~6) and the community
        # scanner flags.
        duration = max(lanes.values())
        beats_held = tempo.time_to_beat(
            tempo.beat_to_time(tick / res) + duration) - tick / res
        sustain = 0
        if beats_held >= min_sustain_beats:
            sustain = min(int(beats_held * res) - release, cap)
            limit = next_tick.get(tick)
            if limit is not None:
                sustain = min(sustain, limit - tick - release)
            if sustain < floor:
                sustain = 0
        for lane in sorted(lanes):
            notes.append((tick, lane, max(0, sustain)))
    if ornaments:
        notes = _add_ornaments(notes, kept, tempo, fret_of, swing_beats,
                               min_amplitude)
    return notes


# 32nd ornaments are the riskiest motif in the study: a flam the audio does
# not have feels broken instantly, so every gate here is deliberately tight.
ORNAMENT_AMP_RATIO = 1.25     # louder than the ghost-note floor by a margin
ORNAMENT_TOL_BEATS = 0.03     # ~20ms at 120 BPM; the onset must SIT there
ORNAMENT_CAP_PER_SONG = 12
ORNAMENT_REPEAT_BARS = 2      # the figure must recur nearby, like gallops


def _add_ornaments(notes, kept, tempo, fret_of, swing_beats, min_amplitude):
    """Recover a few 32nd grace notes the 16th grid swallowed.

    37% of human sub-16th content is 32nds, and their shape is telling:
    median burst run 3 notes - short ornamental flicks beside a main note,
    not streams. Quantization currently merges such an onset into its
    neighbour. One is re-emitted only when everything lines up: the onset
    truly sits on an odd 32nd slot (tight tolerance), lands a 32nd from an
    existing note with clear air on the other side, is louder than the
    ghost floor by a margin, maps to a DIFFERENT lane (a same-lane 32nd
    pair needs a 60ms double strum - and an invented contour is worse than
    no ornament), and the same bar-position figure recurs within two bars.
    At most one per bar and a dozen per song.
    """
    import bisect

    res = tempo.resolution
    step32 = res // 8
    bar = res * 4
    occupied = {t for t, _, _ in notes}
    if not occupied:
        return notes
    ticks = sorted(occupied)

    candidates = []
    for start, _, pitch, amp in kept:
        if amp < min_amplitude * ORNAMENT_AMP_RATIO:
            continue
        beat = tempo.time_to_beat(start)
        if int(beat) in swing_beats:
            continue  # triplet beats have their own grid
        tick32 = int(round(beat * res / step32)) * step32
        if tick32 <= 0 or tick32 % (res // 4) == 0 or tick32 in occupied:
            continue  # not a true 32nd-only slot, or already a note
        if abs(beat - tick32 / res) > ORNAMENT_TOL_BEATS:
            continue
        i = bisect.bisect_left(ticks, tick32)
        before = tick32 - ticks[i - 1] if i > 0 else bar
        after = ticks[i] - tick32 if i < len(ticks) else bar
        # a flam: exactly one 32nd from a real note, air on the other side
        if not ((before == step32 and after >= res // 4)
                or (after == step32 and before >= res // 4)):
            continue
        parent = ticks[i - 1] if before == step32 else ticks[i]
        parent_lanes = {l for t, l, _ in notes if t == parent}
        lane = fret_of(pitch)
        if lane in parent_lanes or OPEN in parent_lanes:
            continue
        candidates.append((tick32, lane))

    if not candidates:
        return notes
    # Repeat gate: the same bar-offset must appear in another bar nearby.
    by_offset: dict[int, list[int]] = {}
    for t, _ in candidates:
        by_offset.setdefault(t % bar, []).append(t // bar)
    picked = []
    used_bars = set()
    for tick32, lane in sorted(candidates):
        bars_here = by_offset[tick32 % bar]
        if not any(0 < abs(b - tick32 // bar) <= ORNAMENT_REPEAT_BARS
                   for b in bars_here):
            continue
        if tick32 // bar in used_bars or len(picked) >= ORNAMENT_CAP_PER_SONG:
            continue
        used_bars.add(tick32 // bar)
        picked.append((tick32, lane, 0))
    return sorted(notes + picked) if picked else notes


# Triple riffs: the polished max-chord-3, built from the 800-chart triple
# study + community doctrine (2026-08-30). Usage is BIMODAL - 42% of human
# charts use triples heavily (user-median 7% of positions, punk 13%), the
# rest barely at all - so this is a per-SONG decision, then a per-RUN
# promotion: 56% of human triple-containing chord runs are pure-3, and the
# rejected raw MAX_CHORD=3 failed precisely by per-tick coincidence mixing.
# Gate thresholds from the 24-song calibration study (2026-08-30), set
# EVIDENCE-led per the user's call, not chart-mimicry-led: human labels
# are style (a charter may voice real triads as 2-note - vampire's piano
# triads carry 96% recurrence and its chart has zero triples), so the
# gate opens where the evidence itself is trustworthy. Users median 23%
# evidence share vs nonusers 13%; above share>=0.15 AND recurrence>=0.75
# every sampled song carries a genuine recurring third voice. Misses are
# conservative (dense metalcore whose third voice transcription cannot
# hear stays 2-note).
TRIPLE_EV_SHARE = 0.15
TRIPLE_EV_RECUR = 0.75
# Evidence granularity is per SHAPE, not per tick: on Mary Jane the mix
# transcription hears the chorus riff's third voice on only ~1 hit in 5
# (masking varies strum to strum) even though 95% recurrence proves it is
# the same voicing every time - a per-run 60% support bar silently
# promoted nothing. RBN's own chord-consistency rule is shape-level: one
# voicing per real chord, all song. A shape heard with its third voice
# SHAPE_EVIDENCE_MIN times song-wide voices as a triple wherever it runs,
# provided the run itself shows the voice at least once.
TRIPLE_SHAPE_EVIDENCE_MIN = 3
TRIPLE_RUN_SUPPORT = 0.20
TRIPLE_RUN_MIN_CHORDS = 3
TRIPLE_RUN_ON8_SHARE = 0.80
# RBN's legal pool: adjacent + one-gap shapes; never Green AND Orange.
LEGAL_TRIPLES = {(0, 1, 2), (1, 2, 3), (2, 3, 4),
                 (0, 1, 3), (0, 2, 3), (1, 2, 4), (1, 3, 4)}


def triple_song_evidence(events, tempo,
                         min_pitch: int = MIN_PITCH,
                         min_amplitude: float = MIN_AMPLITUDE):
    """Is this a triple song? Evidence shares from the mix transcription."""
    res = tempo.resolution
    bar = res * 4
    by_tick: dict[int, list] = {}
    for s, _, p, a in events:
        if p >= min_pitch and a >= min_amplitude:
            by_tick.setdefault(tempo.quantize(s, subdiv=4), []).append((a, p))
    chords = 0
    three_pos = []
    third_pitch: dict[int, float] = {}
    for t, group in by_tick.items():
        group = sorted(group, reverse=True)
        comparable = [g for g in group
                      if g[0] >= group[0][0] * CHORD_AMPLITUDE_RATIO]
        if len(comparable) >= 2:
            chords += 1
            if len(comparable) >= 3:
                three_pos.append(t)
                third_pitch[t] = comparable[2][1]
    offs: dict[int, set] = {}
    for t in three_pos:
        offs.setdefault(t % bar, set()).add(t // bar)
    recur = sum(1 for t in three_pos if len(offs[t % bar]) >= 2)
    share = len(three_pos) / max(1, chords)
    recur_share = recur / max(1, len(three_pos))
    return {
        "share": share, "recur": recur_share, "third_pitch": third_pitch,
        "qualifies": (share >= TRIPLE_EV_SHARE
                      and recur_share >= TRIPLE_EV_RECUR
                      and len(three_pos) >= 8),
    }


def merge_triple_evidence(mix_ev, guitar_ev, tempo):
    """Pool the mix's and the guitar stem's third-voice evidence.

    Both transcriptions hear the third voice on roughly one strum in five,
    but on DIFFERENT strums: Mary Jane's mix evidence and guitar-stem
    evidence overlapped on a handful of ticks out of 74 + 59 (2026-09-02),
    so the union carries ~1.6x the ticks of either alone and promotes
    2-4x the runs under the unchanged shape-level grammar. Nothing new is
    invented - every tick still comes from a transcription that heard a
    third comparable voice there; the guitar stem's pitch wins where both
    heard one, since it is the isolated instrument. The song gate passes
    when EITHER source certifies a recurring third voice; recurrence is
    recomputed on the pooled ticks for the log.
    """
    if not guitar_ev or not guitar_ev["third_pitch"]:
        return mix_ev
    bar = tempo.resolution * 4
    third = {**mix_ev["third_pitch"], **guitar_ev["third_pitch"]}
    offs: dict[int, set] = {}
    for t in third:
        offs.setdefault(t % bar, set()).add(t // bar)
    recur = sum(1 for t in third if len(offs[t % bar]) >= 2)
    return {
        "share": max(mix_ev["share"], guitar_ev["share"]),
        "recur": recur / max(1, len(third)),
        "third_pitch": third,
        "guitar_ticks": len(set(guitar_ev["third_pitch"]) - set(mix_ev["third_pitch"])),
        "qualifies": mix_ev["qualifies"] or guitar_ev["qualifies"],
    }


def promote_triple_runs(notes, evidence, tempo):
    """Voice whole chord runs as triples where a third pitch supports them.

    Grammar from the study + doctrine: promotion is per RUN (>=3 chords
    within a beat of each other, >=80% on the 8th grid - human triples sit
    median 100% on-8th), a run promotes only when >=60% of its chords carry
    third-pitch evidence, every instance of a 2-note shape in the run takes
    the SAME third lane (chord consistency: one voicing per real chord),
    the added lane follows the third pitch's register (below/between/above
    the charted pair), and the result must be one of the seven legal
    shapes - adjacent or one-gap, never Green and Orange together. Runs
    stay size-consistent by construction, which is what keeps the
    same-shape adjacency that "cleaner" playtests ride on.
    """
    if not notes or not evidence["qualifies"]:
        return notes, 0
    res = tempo.resolution
    third = evidence["third_pitch"]
    by_tick: dict[int, list] = {}
    for n in notes:
        by_tick.setdefault(n[0], []).append(n)
    ticks = sorted(by_tick)
    fretted = {t: sorted(l for _, l, _ in by_tick[t] if l != OPEN)
               for t in ticks}
    chord_ticks = [t for t in ticks if len(fretted[t]) == 2]

    runs = []
    cur: list[int] = []
    for t in chord_ticks:
        if cur and t - cur[-1] > res:
            if len(cur) >= TRIPLE_RUN_MIN_CHORDS:
                runs.append(cur)
            cur = []
        cur.append(t)
    if len(cur) >= TRIPLE_RUN_MIN_CHORDS:
        runs.append(cur)

    def third_lane(shape, direction):
        lo, hi = shape
        options = {"below": [lo - 1], "above": [hi + 1],
                   "between": list(range(lo + 1, hi))}
        order = {"below": ["below", "between", "above"],
                 "above": ["above", "between", "below"],
                 "between": ["between", "above", "below"]}[direction]
        for kind in order:
            for lane in options[kind]:
                cand = tuple(sorted({lo, hi, lane}))
                if len(cand) == 3 and cand in LEGAL_TRIPLES:
                    return lane
        return None

    # song-wide evidence per 2-note shape (the chord-consistency unit)
    shape_evidence: dict[tuple, int] = {}
    for t in chord_ticks:
        if t in third:
            s = tuple(fretted[t])
            shape_evidence[s] = shape_evidence.get(s, 0) + 1

    added: dict[int, int] = {}
    promoted_runs = 0
    for run in runs:
        on8 = sum(1 for t in run if t % (res // 2) == 0)
        if on8 < len(run) * TRIPLE_RUN_ON8_SHARE:
            continue
        support = sum(1 for t in run if t in third)
        if support < max(1, len(run) * TRIPLE_RUN_SUPPORT):
            continue
        if not all(shape_evidence.get(tuple(fretted[t]), 0)
                   >= TRIPLE_SHAPE_EVIDENCE_MIN for t in run):
            continue
        # one third-lane decision per 2-note shape in this run
        by_shape: dict[tuple, list] = {}
        for t in run:
            by_shape.setdefault(tuple(fretted[t]), []).append(t)
        run_added = {}
        for shape, members in by_shape.items():
            # Doctrine default for the added voice: fill the gap when the
            # pair has one (a split shape gains its middle), else stack
            # upward - voicings are pitch-ordered and the layered note
            # "plays a higher or dissonant note over it" per RBN.
            direction = "between" if shape[1] - shape[0] >= 2 else "above"
            lane = third_lane(shape, direction)
            if lane is None:
                run_added = {}
                break
            for t in members:
                run_added[t] = lane
        if not run_added:
            continue
        added.update(run_added)
        promoted_runs += 1

    if not added:
        return notes, 0
    out = list(notes)
    for t, lane in added.items():
        sus = max(s for _, _, s in by_tick[t])
        out.append((t, lane, sus))
    return sorted(set(out)), promoted_runs


# Register fallback: below this is kick fundamentals and rumble, never a
# melodic bassline worth charting.
BASS_MIN_PITCH = 24
# A bar is starved when the normal selection leaves at most this many
# positions in it; only runs of starved bars trigger the fallback, so a
# one-bar breath between phrases never flips the chart to the bass.
FALLBACK_MAX_KEPT_PER_BAR = 1
FALLBACK_MIN_RUN_BARS = 2
FALLBACK_MIN_LOW_PER_BAR = 2


def starved_runs(events, tempo,
                 min_pitch: int = MIN_PITCH,
                 min_amplitude: float = MIN_AMPLITUDE):
    """[(start_s, end_s)] stretches where the melodic selection starves.

    Runs of >=FALLBACK_MIN_RUN_BARS bars keeping <=FALLBACK_MAX_KEPT_PER_BAR
    positions - the trigger validated by the register-fallback sweep (49
    songs: 97% of triggered bars carry >=2 notes in the human chart).
    """
    bar_beats = 4.0
    kept: dict[int, set] = {}
    last_bar = 0
    for start, _, pitch, amp in events:
        b = int(tempo.time_to_beat(start) / bar_beats)
        last_bar = max(last_bar, b)
        if pitch >= min_pitch and amp >= min_amplitude:
            kept.setdefault(b, set()).add(tempo.quantize(start, subdiv=4))
    starved = [b for b in range(last_bar + 1)
               if len(kept.get(b, ())) <= FALLBACK_MAX_KEPT_PER_BAR]
    out = []
    run: list[int] = []
    for b in starved + [None]:
        if run and (b is None or b != run[-1] + 1):
            if len(run) >= FALLBACK_MIN_RUN_BARS:
                out.append((tempo.beat_to_time(run[0] * bar_beats),
                            tempo.beat_to_time((run[-1] + 1) * bar_beats)))
            run = []
        if b is not None:
            run.append(b)
    return out


def admit_stem_events(stem_events, runs_sec, kept_events, tempo,
                      min_pitch: int = MIN_PITCH,
                      min_amplitude: float = MIN_AMPLITUDE):
    """Filter stem-transcribed events down to what starved stretches may add.

    Only events inside a starved run survive; ghosts and sub-bass rumble are
    dropped by the same floors as the mix path; sub-guitar pitches
    octave-lift exactly like the register fallback.

    Dedupe is at the QUANTIZED-TICK level, learned the hard way: a 60ms
    time-proximity dedupe let stem events 70-110ms from a kept mix note
    through, and at 129 BPM those quantize onto the SAME 16th tick - the
    same musical note heard twice became a fake two-note chord. The rescued
    sections filled with accidental chords, playability flagged 28.7% of
    transitions impossible, and enforcement collapsed everything back to
    roots: we manufactured intensity and then correctly deleted it. A
    rescued note may only land on an EMPTY tick - filling holes is the
    rescue's entire purpose.
    """
    taken = {tempo.quantize(s, subdiv=4) for s, _, p, a in kept_events
             if p >= min_pitch and a >= min_amplitude}
    by_tick: dict[int, list] = {}
    for start, end, pitch, amp in stem_events:
        if amp < min_amplitude or pitch < BASS_MIN_PITCH:
            continue
        if not any(t0 <= start < t1 for t0, t1 in runs_sec):
            continue
        tick = tempo.quantize(start, subdiv=4)
        if tick in taken:
            continue  # the moment is already charted; a twin is not a rescue
        by_tick.setdefault(tick, []).append((amp, start, end, pitch))
    # An empty tick may take a CHORD from the stem: the loudest note plus
    # one comparable partner of a different pitch, the mix path's own
    # chord rule. Rescue used to keep one note per tick, so every rescued
    # passage came out as a single-note line where charters write chords
    # (A/B report 2026-09-06: human 100% chords in Gnaw's rescued windows,
    # 55% in Voyager's, ours 0% under either rescue rule).
    out = []
    for tick in sorted(by_tick):
        group = sorted(by_tick[tick], reverse=True)
        chosen = [group[0]]
        for cand in group[1:]:
            if cand[0] >= CHORD_AMPLITUDE_RATIO * group[0][0] and cand[3] != group[0][3]:
                chosen.append(cand)
                break  # MAX_CHORD is 2
        for amp, start, end, pitch in chosen:
            lifted = pitch
            while lifted < min_pitch:
                lifted += 12
            out.append((start, end, lifted, amp))
        taken.add(tick)  # stems must not duplicate each other either
    return sorted(out)


def stem_rescue_events(events, tempo, stems: dict, progress=lambda m: None,
                       min_pitch: int = MIN_PITCH,
                       min_amplitude: float = MIN_AMPLITUDE):
    """Transcribe isolated stems inside starved stretches and admit the finds.

    The register fallback re-admits low notes the MIX transcription heard;
    this goes further for the sections where the mix transcription itself
    collapses (Toccata's growl breakdowns: 13-18 raw events per 10s).
    Separation removes the masking, so Basic Pitch on the isolated bass and
    "other" stems recovers lines the mix hides - the same UVR+Basic Pitch
    scaffolding workflow the YARG charting wiki recommends to human charters,
    and the separate-then-transcribe result the Cerberus/Jointist literature
    reports. Only starved stretches are touched: elsewhere the mix
    transcription is already the better witness (stem artifacts would only
    add ghosts). "other" is admitted before bass so a recovered lead
    outranks a duplicate bassline at the same moments.
    """
    runs = starved_runs(events, tempo, min_pitch, min_amplitude)
    if not runs:
        return []
    import soundfile as sf
    import tempfile
    from pathlib import Path

    from basic_pitch.inference import predict

    from . import stems as stemsmod

    known = list(events)
    admitted = []
    for name in ("other", "bass"):
        wav = stems.get(name)
        if wav is None or not len(wav):
            continue
        tmp = Path(tempfile.gettempdir()) / f"chartgen_stem_{name}.wav"
        try:
            sf.write(str(tmp), wav, stemsmod.SR)
            _, _, note_events = predict(str(tmp))
        except Exception as error:  # rescue is a bonus, never fatal
            progress(f"      stem rescue skipped for {name}: "
                     f"{type(error).__name__}: {error}")
            continue
        finally:
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass
        stem_events = [(float(s), float(e), int(p), float(a))
                       for s, e, p, a, *_ in note_events]
        new = admit_stem_events(stem_events, runs, known, tempo,
                                min_pitch, min_amplitude)
        admitted.extend(new)
        known.extend(new)
    return admitted


def bass_fallback_events(events, tempo,
                         min_pitch: int = MIN_PITCH,
                         min_amplitude: float = MIN_AMPLITUDE):
    """Low-register events to admit where the melodic selection starves.

    The MIN_PITCH floor exists because a guitar chart following the bass
    feels wrong WHEN A LEAD IS PLAYING. In an EDM breakdown the growl
    bassline IS the foreground - measured on SIKS' Toccata, 40s stretches
    kept 4-5 notes per 10s while dropping 13-14 bass events as
    "too low-pitched", charting near-silence through the loudest part of
    the song. Human charters follow the most interesting voice; when the
    only voice is the bass, so should we.

    This is the lowest-risk slice of the parked section-committed
    instrument plan: it only acts where there is essentially nothing to
    override (kept positions collapse for >=2 consecutive bars while low
    events keep playing), so the failure mode of overriding a present
    lead cannot occur by construction. Admitted notes are octave-lifted
    into the guitar register so the pitch->fret map places the bassline's
    CONTOUR at the bottom of the neck - and a lone lifted low note can
    still become an open note downstream, which is exactly the human use
    of opens (bass drops, chugs, pedal tones).
    """
    bar_beats = 4.0

    def bar_of(t):
        return int(tempo.time_to_beat(t) / bar_beats)

    kept_positions: dict[int, set] = {}
    low_by_bar: dict[int, list] = {}
    for start, end, pitch, amp in events:
        if amp < min_amplitude:
            continue
        b = bar_of(start)
        if pitch >= min_pitch:
            kept_positions.setdefault(b, set()).add(
                tempo.quantize(start, subdiv=4))
        elif pitch >= BASS_MIN_PITCH:
            low_by_bar.setdefault(b, []).append((start, end, pitch, amp))

    if not low_by_bar:
        return []
    last_bar = max(max(kept_positions, default=0), max(low_by_bar))
    starved = [b for b in range(last_bar + 1)
               if len(kept_positions.get(b, ())) <= FALLBACK_MAX_KEPT_PER_BAR
               and len(low_by_bar.get(b, ())) >= FALLBACK_MIN_LOW_PER_BAR]

    admitted = []
    run: list[int] = []
    for b in starved + [None]:
        if run and (b is None or b != run[-1] + 1):
            if len(run) >= FALLBACK_MIN_RUN_BARS:
                for rb in run:
                    for start, end, pitch, amp in low_by_bar.get(rb, ()):
                        lifted = pitch
                        while lifted < min_pitch:
                            lifted += 12
                        admitted.append((start, end, lifted, amp))
            run = []
        if b is not None:
            run.append(b)
    return admitted


# Extended-sustain ladders: a held note keeps RINGING while higher lanes
# join on top (G held, R joins, Y stacks - the community's "building
# towards" idiom, the one sanctioned route to big chords). Library: 39% of
# the 800 charts use them, 25% meaningfully; 75% of joins sit ABOVE the
# held lane and 54% of joining notes are themselves sustained. Rate among
# charts that use them: ~1.2 joins per 100 notes.
LADDER_RATE_PER100 = 1.5      # cap just above the user-median rate
LADDER_MIN_SPACING_BEATS = 4  # human ladder accents sit ~2 bars apart
LADDER_MAX_HOLD_BEATS = 2.0
LADDER_JOIN_WINDOW = (0.2, 1.05)  # joins land a 16th to a beat after the host


def extend_ladders(notes, events, tempo,
                   min_pitch: int = MIN_PITCH,
                   min_amplitude: float = MIN_AMPLITUDE):
    """Let evidence-backed holds ring through the notes that join them.

    Everywhere else the pipeline CLAMPS a sustain at the next note. That
    made the staircase the user kept asking for - green ringing while red
    joins it - structurally impossible. The licence to overlap comes from
    the transcription itself: Basic Pitch reports real durations, so a
    note whose transcribed sound genuinely outlives the next onsets may
    keep its tail. The host qualifies by its TRANSCRIBED duration, not its
    charted sustain - the charted one was clamped at the next note and
    floored to zero, which is precisely what happens to every real ladder
    host (it rings 1.5 beats, something joins 0.25 later). Gates: the
    transcribed duration must reach past the first
    join, joins must sit on HIGHER lanes (the 75% norm - it also reads as
    a build-up, not a smear), at most three lanes ring at once, the tail
    stops a breath before the next note on the host's own lane, and
    ladders stay rare and spread out like the accents they are. Expert
    only - callers apply this after lower tiers are derived, because
    Rock-Band-lineage reducers and reduced spacing both dislike overlaps.
    """
    if not notes or not events:
        return notes, 0
    res = tempo.resolution
    by_tick: dict[int, list] = {}
    for note in notes:
        by_tick.setdefault(note[0], []).append(note)
    ticks = sorted(by_tick)
    n = len(ticks)

    # Longest true (transcribed) ring-out per charted position, in ticks.
    true_len: dict[int, int] = {}
    for start, end, pitch, amp in events:
        if pitch < min_pitch or amp < min_amplitude:
            continue
        tick = tempo.quantize(start, subdiv=4)
        if tick not in by_tick:
            continue
        length = int((tempo.time_to_beat(end) - tick / res) * res)
        if length > true_len.get(tick, 0):
            true_len[tick] = length

    next_on_lane: dict[int, dict[int, int]] = {}
    lane_last: dict[int, int] = {}
    for tick in reversed(ticks):
        next_on_lane[tick] = dict(lane_last)
        for _, lane, _ in by_tick[tick]:
            lane_last[lane] = tick

    budget = max(4, int(n * LADDER_RATE_PER100 / 100.0))
    spacing = int(LADDER_MIN_SPACING_BEATS * res)
    cap = int(LADDER_MAX_HOLD_BEATS * res)
    gap = res // 8
    lo, hi = (int(w * res) for w in LADDER_JOIN_WINDOW)

    stretch: dict[tuple[int, int], int] = {}
    moved: dict[tuple[int, int], int] = {}
    last_ladder = -spacing
    made = 0
    for i, tick in enumerate(ticks):
        if made >= budget or tick - last_ladder < spacing:
            continue
        group = by_tick[tick]
        fretted = sorted((lane, sus) for _, lane, sus in group if lane != OPEN)
        if not fretted:
            continue
        host_lane, host_sus = fretted[0]
        truth = true_len.get(tick, 0)
        if truth < (res * 3) // 4:
            continue  # the sound itself is not hold-worthy
        # joins: later positions inside the join window, strictly above the
        # host lane, no opens, that the host's real sound rings through
        joins = []          # (tick, new_lane or None to keep)
        relane: dict[int, int] = {}
        top = host_lane     # highest lane in the stack so far
        feasible = True
        for j in range(i + 1, n):
            jt = ticks[j]
            if jt - tick > hi:
                break
            lanes = [l for _, l, _ in by_tick[jt]]
            if jt - tick < lo or OPEN in lanes or min(lanes) <= host_lane:
                continue
            if truth < (jt - tick) + res // 4:
                continue
            # Library norm: 76% of joins land 1-2 lanes above the stack's
            # top - the awkward shape is one finger leaping a 3-4 lane gap
            # mid-ladder, not total width (span-3 stacks like G-Y-B are a
            # legitimate 25%). A wide SINGLE join with air after it pulls
            # inward to the highest free lane within reach, mirroring the
            # narrow-wide-chords doctrine; a wide join that cannot move (a
            # chord, or mid-line) vetoes the ladder rather than shipping
            # the leap.
            jl = min(lanes)
            if jl - top > 2:
                used = {host_lane} | {l for _, l in joins}
                target = next((c for c in (top + 2, top + 1)
                               if 0 <= c <= 4 and c not in used), None)
                after = ticks[j + 1] - jt if j + 1 < n else res
                if (target is None or len(lanes) > 1 or after < res // 2):
                    # An immovable leap as the FIRST join means the ladder
                    # itself would be the gappy shape - veto. Later, it just
                    # ends the stack: a one-join ladder is still a ladder.
                    if not joins:
                        feasible = False
                    break
                relane[jt] = target
                jl = target
            joins.append((jt, jl))
            top = max(top, jl)
            if len(joins) >= 2:  # host + two joins = three lanes ringing
                break
        if not joins or not feasible:
            continue
        limit = next_on_lane[tick].get(host_lane)
        new_sus = min(truth, cap, (limit - tick - gap) if limit else cap)
        if new_sus < (joins[0][0] - tick) + res // 4:
            continue  # the overlap would be too short to read as a ladder
        if new_sus <= host_sus:
            continue
        stretch[(tick, host_lane)] = new_sus
        for jt, target in relane.items():
            old = min(l for _, l, _ in by_tick[jt])
            moved[(jt, old)] = target
        last_ladder = tick
        made += 1

    if not stretch:
        return notes, 0
    out = [(t, moved.get((t, l), l), stretch.get((t, l), s))
           for t, l, s in notes]
    return sorted(out), made


def propagate_sustains(tiers: dict, expert_key: str = "ExpertSingle") -> dict:
    """Copy Expert's real sustains onto reduced tiers by tick."""
    by_tick = {}
    for tick, _, sus in tiers[expert_key]:
        by_tick[tick] = max(by_tick.get(tick, 0), sus)
    return {
        name: notes if name == expert_key
        else [(t, lane, by_tick.get(t, 0)) for t, lane, _ in notes]
        for name, notes in tiers.items()
    }
