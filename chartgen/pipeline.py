"""The audio -> Clone Hero song folder pipeline, driven by CLI or GUI.

Expert notes come from Basic Pitch transcription (chartgen.transcribe); the
followed-instrument timeline, rescue passes, texture, reduction and expression
stages build the song folder from there. Progress is pushed through a callback
rather than printed, so the GUI can show it live. Runs take minutes, so there
is a cancel hook checked between stages.
"""
import shutil
from itertools import pairwise
from pathlib import Path

from . import chart as chartio
from . import density, expression, frets, quality, reduce
from . import tempo as tempomod

# The only formats the .chart docs list as being in wide use. wav is deliberately
# absent: it is not in that table, so it gets transcoded rather than passed through.
PLAYABLE_AUDIO = {".ogg", ".mp3", ".opus"}
OPUS_RATE = 48000  # Opus accepts only 8/12/16/24/48 kHz


class Cancelled(Exception):
    """Raised when the caller asked to stop between stages."""


class _Prefetch:
    """Chart-independent analyses started on CPU threads as soon as the
    tempo map exists, while the GPU separates stems.

    Every job is a pure function of (audio, tempo) that the pipeline used
    to compute later, in sequence, on an idle CPU: the mix transcription,
    the section segmentation and its chroma signatures, the per-bar riff
    profiles, the solo detector's pitch/chroma analysis, and the LRCLIB
    lookup. Each result is consumed at exactly the point the inline call
    used to happen, and a job's exception surfaces there too, so the chart
    - and every failure message - is the same as before. onnxruntime and
    librosa's numba kernels release the GIL for their heavy work, which is
    what makes the overlap real.
    """

    def __init__(self):
        from concurrent.futures import ThreadPoolExecutor

        self.pool = ThreadPoolExecutor(max_workers=4,
                                       thread_name_prefix="chartgen-prefetch")
        self.jobs = {}

    def submit(self, name, fn, *args, **kwargs):
        self.jobs[name] = self.pool.submit(fn, *args, **kwargs)

    def has(self, name) -> bool:
        return name in self.jobs

    def result(self, name):
        return self.jobs[name].result()

    def close(self):
        self.pool.shutdown(wait=False)


def _folder_name(artist: str, name: str) -> str:
    """Song folder name: characters FAT and exFAT drives reject are stripped, never empty."""
    import re

    return re.sub(r'[<>:"/\\|?*]', "", f"{artist} - {name}").strip(" -.") or "song"


def _display_name(opts, audio: Path) -> str:
    """Song title with the [chartgen] tag, so generated charts are obvious in
    the CH song list next to human charts of the same track."""
    name = opts.name or audio.stem
    return name if "[chartgen]" in name else f"{name} [chartgen]"


def stage_audio(src: Path, dest_dir: Path) -> str:
    """Put playable audio in the song folder under the reserved name `song.*`.

    Transcodes to Opus, which the .chart docs recommend.
    """
    if src.suffix.lower() in PLAYABLE_AUDIO:
        shutil.copy2(src, dest_dir / f"song{src.suffix.lower()}")
        return f"song{src.suffix.lower()}"

    import librosa
    import numpy as np
    import soundfile as sf

    data, sr = sf.read(src, dtype="float32", always_2d=True)
    data = data.T  # librosa wants (channels, samples)
    if sr != OPUS_RATE:
        # Resampling is mandatory, not a nicety: writing 44.1k samples while
        # claiming 48k plays the whole song ~9% fast and desyncs every note.
        data = librosa.resample(data, orig_sr=sr, target_sr=OPUS_RATE)
    sf.write(dest_dir / "song.opus", np.ascontiguousarray(data.T), OPUS_RATE,
             format="OGG", subtype="OPUS")
    return "song.opus"


def run(opts, progress=print, should_cancel=lambda: False) -> dict:
    """Generate a song folder. `opts` matches the CLI namespace.

    Returns {'song_dir', 'summary', 'bpm', 'tempo_events', 'star_power',
    'sections', 'variety'}. Raises Cancelled, or ValueError for bad input.
    """
    def check():
        if should_cancel():
            raise Cancelled()

    import librosa

    from . import youtube

    if youtube.is_youtube_url(str(opts.audio)):
        progress("[0/6] downloading from YouTube")
        fetched = youtube.download(str(opts.audio), Path(opts.outdir) / "_downloads",
                                   progress,
                                   cookies_from=getattr(opts, "cookies_from", None))
        progress(f"      {fetched['title']}")
        opts.audio = fetched["audio"]
        opts.thumbnail_url = fetched.get("thumbnail")
        # Autofill metadata the user did not supply; a pasted link rarely
        # comes with artist/title typed in.
        if not getattr(opts, "name", None):
            opts.name = fetched["name"]
        if getattr(opts, "artist", "Unknown") in ("", "Unknown"):
            opts.artist = fetched["artist"]
        check()

    audio = Path(opts.audio)
    if not getattr(opts, "name", None) or getattr(opts, "artist", "Unknown") in ("", "Unknown"):
        from . import tags

        artist, name = tags.guess_metadata(audio)
        if not getattr(opts, "name", None) and name:
            opts.name = name
        if getattr(opts, "artist", "Unknown") in ("", "Unknown") and artist:
            opts.artist = artist

    if getattr(opts, "skip_existing", False):
        # Batch reruns (re-pasted playlists after 403s) must not re-chart
        # finished songs. Single-song runs skip this: regenerating in place
        # is how you reroll a chart you did not like.
        existing = Path(opts.outdir) / _folder_name(opts.artist, _display_name(opts, audio))
        if (existing / "notes.chart").is_file():
            progress(f"      already charted, skipping: {existing.name}")
            return {"skipped": True, "song_dir": existing}

    progress(f"[1/6] reading {audio.name}")
    y, sr = tempomod.load(str(audio))
    duration_s = librosa.get_duration(y=y, sr=sr)
    if duration_s < 30:
        raise ValueError(f"audio must be at least 30s, got {duration_s:.0f}s")
    tempo = tempomod.detect(y, sr, bpm_mult=getattr(opts, "bpm_mult", "auto"))
    res = tempo.resolution
    progress(f"      {duration_s:.0f}s, {tempo.bpm:.1f} BPM, "
             f"{len(tempo.sync_track())} tempo event(s)")
    check()

    from . import transcribe

    progress("[2/6] transcribing (Basic Pitch engine)")
    from . import cache as diskcache

    try:
        mix_key = f"mix:{diskcache.audio_key(str(audio))}:bp"
    except OSError:
        mix_key = None
    # Everything below the transcription that needs only the audio and the
    # tempo map starts now, on threads, so it overlaps the GPU separation
    # instead of running afterwards on an idle CPU. Results are collected
    # exactly where the sequential code used them.
    pre = _Prefetch()
    pre.submit("transcribe", transcribe.transcribe, str(audio), progress,
               cache_key=mix_key)
    if not getattr(opts, "no_solos", False):
        from . import solo as solomod

        pre.submit("solo_analysis", solomod.analyze, y, sr)
    if not opts.no_sections:
        from . import structure

        def _structure():
            marks = expression.sections(y, sr, tempo)
            signatures = None
            if marks and not getattr(opts, "no_section_reuse", False):
                signatures = structure.section_signatures(
                    y, sr, tempo, [tick for tick, _ in marks])
            return marks, signatures

        pre.submit("structure", _structure)
    if not getattr(opts, "no_riff_unify", False):
        from . import structure

        # Profiles are per bar and independent of one another, so computing
        # them out to the end of the audio and slicing to the chart's bar
        # count later gives the same list bar_profiles(n_bars) would.
        audio_bars = int(tempo.time_to_beat(duration_s) // 4) + 2
        pre.submit("profiles", structure.bar_profiles, y, sr, tempo, audio_bars)
    if (getattr(opts, "lyrics", True)
            and getattr(opts, "lyric_source", "auto") in ("auto", "online")):
        from . import lrclib

        pre.submit("lyrics_lookup", lrclib.find, opts.artist,
                   opts.name or audio.stem, duration_s, progress=progress)
    check()
    progress("[3/6] building Expert from transcription")
    # The followed-instrument timeline comes first now: its per-stem
    # transcriptions feed keyed rescue (below), chord texture and the
    # rhythm/solo consumers further down.
    windows = None
    stem_events = {}
    if getattr(opts, "prominence", True):
        from . import prominence

        try:
            windows = prominence.timeline(str(audio), tempo, duration_s, progress,
                                          known_events=stem_events)
        except Exception as error:  # a prior, never worth failing a chart
            windows = None
            progress(f"      followed instrument skipped: {type(error).__name__}: {error}")
        if windows:
            share = {}
            for w in windows:
                share[w["stem"]] = share.get(w["stem"], 0) + 1
            mix = ", ".join(f"{(k or 'unsure')} {v / len(windows):.0%}"
                            for k, v in sorted(share.items(), key=lambda kv: -kv[1]))
            progress(f"      followed instrument: {prominence.letters(windows)} "
                     f"({len(prominence.runs(windows))} run(s); {mix})")
        else:
            progress("      followed instrument: no timeline (model or SW stems unavailable)")
    events = pre.result("transcribe")
    check()
    if not getattr(opts, "no_bass_fallback", False):
        extra = []
        if windows and not getattr(opts, "no_keyed_rescue", False):
            extra, touched, counts = prominence.keyed_rescue_events(
                events, windows, stem_events, tempo)
            if extra:
                detail = ", ".join(f"{k} {v}" for k, v in counts.items())
                progress(f"      keyed rescue: {len(extra)} note(s) from the followed "
                         f"stem in {touched} starved window(s) ({detail})")
        if windows and not getattr(opts, "no_commit_sections", False):
            more, touched, counts = prominence.commit_section_events(
                events + extra, windows, stem_events, tempo)
            if more:
                detail = ", ".join(f"{k} {v}" for k, v in counts.items())
                progress(f"      section commitment: {len(more)} note(s) from the "
                         f"followed stem in {touched} window(s) ({detail})")
                extra = extra + more
        # Song-level fallback: where keyed rescue found nothing anywhere
        # (no timeline, or no followed stem playing in the starved
        # windows), the blended stem rescue still fills starved runs.
        # Measured on 16 songs: keyed 0.71 precision vs blended 0.70,
        # and three songs fire only the blended one (Emmure: 225 notes).
        if not extra and transcribe.starved_runs(events, tempo):
            # Starved stretches first get the strong medicine: transcribe
            # the isolated stems there (separation removes the masking
            # that collapsed the mix transcription). Demucs results are
            # cached, so lyrics/solos reuse this same separation later.
            from . import stems as stemsmod

            st = stemsmod.separate(str(audio), progress)
            if st is not None:
                extra = transcribe.stem_rescue_events(events, tempo, st,
                                                      progress)
                if extra:
                    progress(f"      stem rescue: {len(extra)} note(s) "
                             f"transcribed from isolated stems where the "
                             f"mix transcription starves")
        if not extra:
            extra = transcribe.bass_fallback_events(events, tempo)
            if extra:
                progress(f"      register fallback: {len(extra)} bass "
                         f"note(s) admitted where the melodic selection "
                         f"starves")
        if extra:
            events = sorted(events + extra)
    swing = set()
    if getattr(opts, "swing", False):
        # Opt-in: on real songs the detected beat grid's local phase
        # error (±40ms) exceeds the 55ms separating the two grids, so
        # per-onset classification misfires on straight songs (Faded
        # measured 61 wrongly-swung beats). Until beat tracking is that
        # precise, straight 16ths are the safe default.
        swing = transcribe.swung_beats(events, tempo)
        if swing:
            progress(f"      swing: {len(swing)} beat(s) quantized to "
                     f"the triplet grid")
    expert = transcribe.expert_from_notes(
        events, tempo, subdiv=opts.subdiv,
        min_sustain_beats=getattr(opts, 'min_sustain_beats', 0.5),
        allow_opens=getattr(opts, "opens", True),
        swing_beats=swing,
        ornaments=not getattr(opts, "no_ornaments", False),
        max_chord=int(getattr(opts, "max_chord", 2)))
    orn = sum(1 for t, _, _ in expert
              if t % (res // 8) == 0 and t % (res // 4) != 0)
    if orn:
        progress(f"      {orn} 32nd grace note(s) recovered from the "
                 f"transcription")
    if getattr(opts, "density", "onset") == "onset":
        before = len({t for t, _, _ in expert})
        expert = density.gate_by_onsets(expert, y, sr, tempo)
        after = len({t for t, _, _ in expert})
        if before > after:
            progress(f"      density: {before} -> {after} positions")
    gevents = stem_events.get("guitar")  # from the timeline when it ran
    if not getattr(opts, "no_guitar_texture", False):
        from . import guitar as guitarmod

        positions = {}
        for t, l, _ in expert:
            positions.setdefault(t, set()).add(l)
        share = (sum(1 for v in positions.values()
                     if len(v - {7}) >= 2) / max(1, len(positions)))
        # The chart-share trigger alone missed every chug song: Gnaw
        # and Metabolic came out at 6% and 1% chords against human 59%
        # and 52%, because a chart that starts as singles never
        # reaches 20%. With a timeline, guitar-followed windows open
        # the gate instead, and the pass only voices runs inside them.
        guitar_spans = None
        if windows:
            guitar_spans = [(int(w["beat0"] * res), int(w["beat1"] * res))
                            for w in windows if w.get("stem") == "guitar"]
        guitar_led = bool(guitar_spans) and len(guitar_spans) >= 0.25 * len(windows)
        if share >= guitarmod.TRIGGER_CHORD_SHARE or guitar_led:
            g = guitarmod.separate_guitar(str(audio), progress)
            gshare = guitarmod.guitar_share(g)
            if gshare >= guitarmod.MIN_GUITAR_SHARE:
                progress(f"      guitar stem active ({gshare:.0%}); "
                         + ("reusing the timeline's transcription for chord texture"
                            if gevents is not None else "transcribing it for chord texture"))
                if gevents is None:
                    import soundfile as sf
                    import tempfile as tf

                    with tf.NamedTemporaryFile(suffix=".wav", delete=False) as fh:
                        gpath = fh.name
                    try:
                        sf.write(gpath, g, guitarmod.SW_SR)
                        gkey = None
                        gst = stemsmod_key(str(audio))
                        if gst:
                            gkey = f"{gst}:guitar:bp"
                        gevents = transcribe.transcribe(gpath, cache_key=gkey)
                    finally:
                        Path(gpath).unlink(missing_ok=True)
                voices = guitarmod.second_voices(gevents, tempo)
                expert, added = guitarmod.chordify(
                    expert, voices, res,
                    allowed_spans=guitar_spans if guitar_led else None)
                if added:
                    progress(f"      guitar texture: {added} single(s) "
                             f"gained the guitar's second voice (humans "
                             f"chart 50-100% chords on strummed songs)")
            elif g is not None:
                progress(f"      guitar stem quiet ({gshare:.0%}); "
                         f"chord texture unchanged")
    if not getattr(opts, "no_triple_riffs", False):
        ev = transcribe.triple_song_evidence(events, tempo)
        if gevents is not None:
            # The guitar stem hears the third voice on different strums
            # than the mix does; pooling them is free evidence.
            ev = transcribe.merge_triple_evidence(
                ev, transcribe.triple_song_evidence(gevents, tempo), tempo)
        if ev["qualifies"]:
            expert, promoted = transcribe.promote_triple_runs(
                expert, ev, tempo)
            if promoted:
                extra = ev.get("guitar_ticks", 0)
                progress(f"      triple riffs: {promoted} chord run(s) "
                         f"voiced as three-note (evidence share "
                         f"{ev['share']:.0%}"
                         + (f", +{extra} ticks from the guitar stem"
                            if extra else "")
                         + "; 42% of human charts are triple songs)")
            else:
                # A qualifying song promoting nothing must say so - the
                # stairs no-op bug hid behind exactly this silence.
                progress(f"      triple song (evidence {ev['share']:.0%})"
                         f" but no chord run met the promotion grammar")
    followed = None
    timeline_solos = None
    if windows:
        from . import prominence

        if True:
            followed = prominence.runs(windows)
            from . import stems as stemsmod

            st = stemsmod.separate(str(audio), progress, backend="sw")
            if st is not None:
                expert, added, touched = prominence.rhythm_rescue(
                    expert, followed, st, stemsmod.SR, tempo)
                if added:
                    progress(f"      rhythm rescue: {added} note(s) from followed-stem "
                             f"onsets in {touched} starved run(s) (synth/bass laned by "
                             f"the filter contour, guitar voiced as the run's chug)")
            solo_model = prominence.load_solo_model()
            if solo_model is not None:
                timeline_solos = prominence.solo_runs(windows, solo_model, tempo)
                spans = ", ".join(f"{tempo.beat_to_time(a / res):.0f}-{tempo.beat_to_time(b / res):.0f}s"
                                  for a, b in timeline_solos)
                progress(f"      solos: {len(timeline_solos)} lead break(s) from the "
                         f"followed-instrument timeline" + (f" ({spans})" if spans else ""))
    if not expert:
        raise ValueError(
            "transcription found no chartable notes — is this audio "
            "mostly percussion or noise?")
    best = min(quality.variety_score(expert), quality.walk_score(expert))
    progress(f"      {quality.describe(expert)}")
    if best < opts.min_variety:
        progress(f"      warning: score {best:.2f} below "
                 f"{opts.min_variety:.2f} (transcription is "
                 f"deterministic; retries would not change it)")
    return _finish_chart(opts, progress, check, y, sr, tempo, expert, best,
                         audio, duration_s, bp_events=events,
                         followed=followed, timeline_solos=timeline_solos,
                         prefetch=pre, stem_events=stem_events)


def stemsmod_key(audio_path: str):
    """The disk-cache key of the SW stems already separated for this song,
    if that separation happened in this process."""
    from . import stems as stemsmod

    mono = stemsmod._CACHE.get((audio_path, "sw"))
    return mono.get("_key") if mono else None


def _group(notes):
    out: dict[int, list] = {}
    for tick, lane, _ in notes:
        out.setdefault(tick, []).append(lane)
    return out


def _finish_chart(opts, progress, check, y, sr, tempo, expert, best,
                  audio, duration_s, bp_events=None, followed=None,
                  timeline_solos=None, prefetch=None, stem_events=None) -> dict:
    """Everything downstream of Expert-note production: difficulty target,
    reduction, expression, lyrics, art, rating, writing."""
    res = tempo.resolution
    from . import rating

    # Human charts of the same songs jump 3+ lanes on 0-1% of consecutive
    # single notes; ours did it 9-30% of the time. Smooth before reduction so
    # every tier inherits playable hand movement.
    # Playtest: charts were unplayable inside 30 seconds because harmony was
    # charted as chords. Do this before the jump smoothing so it sees the
    # shapes that actually survive.
    if not getattr(opts, "keep_dense_chords", False):
        from . import texture

        before = sum(1 for _, g in _group(expert).items() if len(g) > 1)
        # Texture first, backstop second: smoothing shape flicker into
        # honest same-shape runs is what lets chord riffs SURVIVE the
        # rapid-change demotion (same-shape repeats were never the
        # playability problem), and narrowing wide chords keeps the chord
        # where the old rule deleted it.
        expert = texture.smooth_chord_shapes(expert, res)
        expert = texture.narrow_wide_chords(expert, res)
        n_before = len({t for t, _, _ in expert})
        expert = texture.consolidate_gallops(expert, res)
        dropped = n_before - len({t for t, _, _ in expert})
        if dropped:
            progress(f"      rhythm: {dropped} one-off 16th fragment(s) "
                     f"consolidated to eighths")
        # ponytail: a "chord-riff song" gate (keep strummed chords on
        # punk/strum-rock, library median 49% chords vs EDM 7%) was shipped
        # here and removed by audit: no chart-level signal separates the two.
        # Raw chord share measured 33% on BOTH Mary Jane (strummy rock) and
        # Faded (EDM), and same-shape-run share 79% vs 82% - full-mix
        # transcription makes strummed guitar and pad stabs look identical.
        # Needs the Demucs guitar stem (already computed later for solos)
        # moved earlier; until then, unconditional demotion is the version
        # that survived playtesting.
        chord_spans = None
        if followed:
            gspans = [(int(r["beat0"] * res), int(r["beat1"] * res))
                      for r in followed if r["stem"] == "guitar"]
            total = sum(r["beat1"] - r["beat0"] for r in followed)
            if gspans and sum(b - a for a, b in gspans) >= 0.25 * total * res:
                chord_spans = gspans
        expert = reduce.simplify_rapid_chords(expert, res, tempo=tempo,
                                              chord_spans=chord_spans)
        after = sum(1 for _, g in _group(expert).items() if len(g) > 1)
        if before > after:
            progress(f"      chords: {before} -> {after} positions "
                     f"(human charts use 5-13% on EDM)")

    if not getattr(opts, "no_motifs", False):
        from . import motifs

        # settle_pushes MOVES notes without changing position counts, so a
        # dropped-positions delta alone would hide it (the stairs no-op bug
        # taught that every feature needs a visible count).
        ticks_before = {t for t, _, _ in expert}
        expert = motifs.settle_pushes(expert, res)
        moved = len({t for t, _, _ in expert} - ticks_before)
        n_before = len({t for t, _, _ in expert})
        expert = motifs.break_machine_gun(expert, res)
        thinned = n_before - len({t for t, _, _ in expert})
        if moved or thinned:
            progress(f"      flow: {moved} stray push(es) settled onto the "
                     f"8th grid, {thinned} machine-gun note(s) thinned")

    # Nothing may be charted past the end of the audio: a note there is
    # simply unhittable. Cheap insurance against any tempo-map drift in an
    # outro, which is exactly where beat detection is least reliable.
    end_of_audio = int(tempo.time_to_beat(duration_s) * res)
    trimmed = [n for n in expert if n[0] <= end_of_audio]
    if len(trimmed) < len(expert):
        progress(f"      trimmed {len(expert) - len(trimmed)} note(s) past the "
                 f"end of the audio")
        expert = trimmed

    if not getattr(opts, "no_playability", False):
        from . import playability

        before = playability.analyse(expert, tempo)["impossible"]
        expert = playability.enforce(expert, tempo)
        after = playability.analyse(expert, tempo)["impossible"]
        if before > after:
            progress(f"      playability: {before:.1%} -> {after:.1%} of "
                     f"transitions too fast to play (human median 1.3%)")

    if not getattr(opts, "no_brightness_lanes", False):
        stretches, _ = frets.degenerate_stretches(expert, res)
        if stretches:
            from . import stems as stemsmod

            st = stemsmod.separate(str(audio), progress)
            if st is not None:
                import numpy as np

                melodic = (st.get("bass", 0) + st.get("other", 0))
                if isinstance(melodic, np.ndarray):
                    expert, relaned = frets.relane_by_brightness(
                        expert, tempo, melodic, stemsmod.SR)
                    if relaned:
                        progress(f"      brightness lanes: {relaned} stuck "
                                 f"stretch(es) re-laned from the filter/"
                                 f"timbre contour (humans chart 4 lanes "
                                 f"even in bass-led sections)")

    max_jump = int(getattr(opts, "max_fret_jump", 2))
    if max_jump < 4:
        before = frets.step_share(expert, 3)
        expert = frets.smooth_fret_jumps(expert, tempo, max_step=max_jump)
        after = frets.step_share(expert, 3)
        if before > after:
            progress(f"      fret jumps >=3 lanes: {before:.0%} -> {after:.0%}")

    if not getattr(opts, "no_motifs", False):
        from . import motifs

        # Lane-only reshapes, after jump smoothing so they see (and keep)
        # playable hand travel: shapeless fast runs become 3-4 note wrap
        # chunks, monotonic chord walks take the RBN rung ladder, and lone
        # pickups into a chord take the chord's root.
        lanes_before = {(t, l) for t, l, _ in expert}
        expert = motifs.chunk_rolls(expert, res)
        expert = motifs.shape_chord_walks(expert, res)
        expert = motifs.pickup_roots(expert, res)
        relaned = len({(t, l) for t, l, _ in expert} - lanes_before)
        if relaned:
            progress(f"      lanes: {relaned} note(s) reshaped into wrap "
                     f"chunks, chord-ladder rungs, or pickup roots")

    if not getattr(opts, "keep_dense_chords", False):
        from . import texture

        # Second, LATE shape-smoothing pass. The early one runs before jump
        # smoothing and playability enforcement, which re-anchor and knock
        # out chords contextually - the same pad chord ends up shifted
        # differently at different moments, and a final chart measured 69%
        # same-shape adjacency where the demotion stage had produced 84%
        # (playtest verdict: "messier"). Re-unifying here restores the
        # strummed-run coherence the human norm (59.7%) is built on; the
        # pass is chord-count-preserving and riff unification downstream
        # then stamps coherent shapes instead of flickered ones.
        expert = texture.smooth_chord_shapes(expert, res)
        expert = texture.smooth_chord_shapes(expert, res)

    target_diff = getattr(opts, "target_diff", None)
    if target_diff is not None:
        natural = rating.rate_expert(expert, tempo)
        if natural > int(target_diff):
            expert, achieved = reduce.thin_to_rating(expert, res, tempo,
                                                     int(target_diff))
            progress(f"      target difficulty {target_diff}: thinned "
                     f"{natural}/6 -> {achieved}/6")
        elif natural < int(target_diff):
            progress(f"      target difficulty {target_diff}: chart rates "
                     f"{natural}/6 naturally; notes are never invented, so "
                     f"the target only thins (try --density model to keep "
                     f"every generated note)")

    name = _display_name(opts, audio)
    meta = {
        "name": name, "artist": opts.artist, "album": opts.album,
        "genre": opts.genre, "year": opts.year,
        "charter": "chartgen (basic-pitch)",
    }

    signatures = None
    if prefetch is not None and prefetch.has("structure"):
        events, signatures = prefetch.result("structure")
    else:
        events = () if opts.no_sections else expression.sections(y, sr, tempo)
    if events and not getattr(opts, "no_section_reuse", False):
        from . import structure

        ticks = [tick for tick, _ in events]
        lengths = [b - a for a, b in pairwise(ticks)] + [0]
        if signatures is None:
            signatures = structure.section_signatures(y, sr, tempo, ticks)
        repeats = structure.find_repeats(signatures, lengths)
        if repeats:
            expert = structure.reuse_patterns(expert, ticks, repeats)
            progress(f"      reused patterns across {len(repeats)} repeated "
                     f"section(s)")

    if not getattr(opts, "no_riff_unify", False):
        from . import structure

        n_bars = max((t for t, _, _ in expert), default=0) // (4 * res) + 1
        profiles = None
        if prefetch is not None and prefetch.has("profiles"):
            ahead = prefetch.result("profiles")
            if len(ahead) >= n_bars:
                profiles = ahead[:n_bars]
        if profiles is None:
            profiles = structure.bar_profiles(y, sr, tempo, n_bars)
        expert, stamped = structure.unify_riff_bars(expert, profiles, res)
        if stamped:
            progress(f"      riffs: {stamped} bar(s) unified onto their "
                     f"cluster's consensus pattern (human median: 48% of "
                     f"bars repeat)")

    progress("[4/6] deriving Hard/Medium/Easy")
    check()
    if getattr(opts, "reducer", "chartgen") == "easygen":
        # The vendored reducer needs a chart skeleton to read beat positions.
        skeleton = chartio.write_chart({"ExpertSingle": expert}, tempo, meta, "song.ogg")
        tiers = {"ExpertSingle": expert, **reduce.derive_lower_tiers_vendored(skeleton)}
    else:
        tiers = {"ExpertSingle": expert,
                 **reduce.derive_tiers(expert, res, bpm=tempo.bpm)}
    tiers = reduce.enforce_chord_rules(tiers)

    progress("[5/6] adding sustains, star power, sections")
    end_tick = int(tempo.time_to_beat(duration_s) * res)
    if opts.no_sustains:
        tiers = {n: [(t, l, 0) for t, l, _ in notes] for n, notes in tiers.items()}
    else:
        # Expert already carries REAL sustains (transcribed note durations);
        # only propagate them down to the reduced tiers.
        from . import transcribe

        tiers = transcribe.propagate_sustains(tiers)
    if not opts.no_sustains and not getattr(opts, "no_motifs", False):
        from . import motifs
        from . import transcribe as transcribemod

        # Sustain stairs: hold each note of a stepped phrase to just before
        # its successor (78% of human charts, ~6 phrases per 100 bars; we had
        # none). Before top-up, so the holds count toward the calibrated
        # sustain share; re-propagated so lower tiers inherit the same holds.
        stepped = motifs.legato_stairs(tiers["ExpertSingle"], res,
                                       bpm=tempo.bpm)
        if stepped != tiers["ExpertSingle"]:
            grown = sum(1 for a, b in zip(sorted(tiers["ExpertSingle"]),
                                          sorted(stepped), strict=True) if a[2] != b[2])
            progress(f"      {grown} note(s) held as sustain stairs "
                     f"(78% of human charts, ~6 phrases per 100 bars)")
            tiers["ExpertSingle"] = stepped
            tiers = transcribemod.propagate_sustains(tiers)
    if not opts.no_sustains:
        # Human charts sustain MORE as tiers get easier (8.7% of Expert notes
        # up to 15.3% on Easy); propagating Expert's lengths down left every
        # tier flat at Expert's rate.
        tiers = {
            name: expression.top_up_sustains(
                notes, res, end_tick,
                expression.TIER_SUSTAIN_SHARE.get(name, 0.09))
            for name, notes in tiers.items()
        }
    star_power = () if opts.no_star_power else expression.star_power_phrases(
        expert, res, duration_s
    )
    lyric_events = ()
    if getattr(opts, "lyrics", True):
        progress("      looking up lyrics")
        try:
            from . import lyrics as lyricsmod

            lookup = None
            if prefetch is not None and prefetch.has("lyrics_lookup"):
                lookup = lambda: prefetch.result("lyrics_lookup")  # noqa: E731
            lyric_events = lyricsmod.collect(
                str(audio), tempo, opts.artist,
                # The [chartgen] tag is ours; a lyrics database has never
                # heard of it.
                opts.name or audio.stem, duration_s,
                getattr(opts, "lyric_source", "auto"), progress, y, sr,
                lookup=lookup)
        except Exception as error:  # lyrics are a nice-to-have, never fatal
            progress(f"      lyrics skipped: {type(error).__name__}: {error}")
    tap_ticks = set()
    if getattr(opts, "taps", True):
        from . import taps as tapsmod

        tap_ticks = tapsmod.detect(expert, y, sr, tempo, progress,
                                   audio_path=str(audio),
                                   section_marks=list(events),
                                   followed=followed)
    if not getattr(opts, "keep_dense_chords", False) and events:
        from . import texture

        grown = texture.climax_chords(tiers["ExpertSingle"], list(events), res)
        if len(grown) > len(tiers["ExpertSingle"]):
            progress(f"      {len(grown) - len(tiers['ExpertSingle'])} section-entry "
                     f"chord(s) grown to three notes")
        tiers["ExpertSingle"] = grown
    if (bp_events and not opts.no_sustains
            and not getattr(opts, "no_motifs", False)):
        from . import transcribe as transcribemod

        # Extended-sustain ladders (a held lane rings while higher lanes
        # join) go on Expert ONLY, after every tier is derived: reduced
        # spacing dislikes overlaps, and the joins are Expert expression.
        laddered, made = transcribemod.extend_ladders(
            tiers["ExpertSingle"], bp_events, tempo)
        if made:
            tiers["ExpertSingle"] = laddered
            progress(f"      {made} sustain ladder(s): a held note rings "
                     f"while higher lanes join (39% of human charts)")
    solo_phrases = ()
    if not getattr(opts, "no_solos", False):
        from . import solo as solomod

        analysis = None
        if prefetch is not None and prefetch.has("solo_analysis"):
            try:
                analysis = prefetch.result("solo_analysis")
            except Exception:
                analysis = None  # detect() recomputes and reports as before
        solo_phrases = solomod.detect(expert, list(events), list(lyric_events),
                                      y, sr, tempo, progress,
                                      audio_path=str(audio), analysis=analysis)
        if timeline_solos:
            # UNION with the timeline detector (86%p/18%r/2.1% quiet on the
            # 299-song dump): the two miss different solos - the section
            # rule caught Mary Jane's outro, the window model did not, and
            # the window model caught both of Sexualizer's within 4 s -
            # and each spends about a 2% false-fire budget. Overlapping
            # spans merge into one marker.
            merged = sorted(list(solo_phrases) + list(timeline_solos))
            out = []
            for a, b in merged:
                if out and a <= out[-1][1]:
                    out[-1] = (out[-1][0], max(out[-1][1], b))
                else:
                    out.append((a, b))
            added = len(out) - len(solo_phrases)
            solo_phrases = out
            if added > 0:
                progress(f"      solos: {added} added by the followed-instrument "
                         f"timeline, {len(out)} total")
    forced_ticks: set[int] = set()
    if opts.hopos and not getattr(opts, "no_motifs", False):
        from . import motifs

        # Forcing only means anything when natural HOPOs are on: with them
        # off, song.ini's hopo_frequency=1 already makes every note a strum.
        # Taps already never strum; a force flag on one is dead markup.
        forced_ticks = motifs.legato_forcing(tiers["ExpertSingle"], res) - tap_ticks
        if forced_ticks:
            progress(f"      {len(forced_ticks)} note(s) forced to legato "
                     f"HOPO (human median 3.9 per 100 notes)")
    progress(f"      {len(star_power)} star power phrase(s), {len(events)} section(s), "
             f"{len(solo_phrases)} solo(s), HOPOs {'on' if opts.hopos else 'off'}")
    check()

    if getattr(opts, "ghl", True) and bp_events:
        from . import ghl

        if "guitar" not in (stem_events or {}):
            from . import prominence
            from . import stems as stemsmod

            mono = stemsmod.separate(str(audio), progress, backend="sw")
            stem_events = prominence.transcribe_stems(mono, stemsmod.SR) if mono else {}
        ghl_tiers = ghl.chart_tiers(tiers["ExpertSingle"], tempo, bp_events, stem_events)
        if ghl_tiers:
            tiers |= ghl_tiers
            progress(f"      6-fret: {len(_group(ghl_tiers['ExpertGHLGuitar']))} Expert "
                     f"position(s) shaped by the GHL model")
        else:
            progress("      6-fret skipped: no trained GHL model in this install")
    check()

    progress("[6/6] writing song folder")
    # YouTube titles carry characters FAT and exFAT forbid in folder names
    # (measured in the wild: Baldur's Gate 3 - "I Want To Live"). Only the
    # folder is sanitised; song.ini keeps the original text.
    song_dir = Path(opts.outdir) / _folder_name(opts.artist, name)
    song_dir.mkdir(parents=True, exist_ok=True)
    audio_filename = stage_audio(audio, song_dir)
    from . import art

    art.add_album_art(song_dir, audio, getattr(opts, "thumbnail_url", None), progress)
    chart_path = song_dir / "notes.chart"
    chart_path.write_text(
        chartio.write_chart(tiers, tempo, meta, audio_filename, star_power, events,
                            lyrics=lyric_events, solos=solo_phrases,
                            taps=tap_ticks, forced=forced_ticks),
        encoding="utf-8",
    )
    from . import rating

    tier = rating.rate_expert(expert, tempo)
    progress(f"      difficulty rating: {tier}/6")
    # The menu preview should land on an interesting part, not the intro.
    # Star power already marks the densest bar-aligned windows, so its first
    # phrase is the best "hook" estimate we have; otherwise a quarter in.
    preview_ms = int(duration_s * 250)
    if star_power:
        preview_ms = int(tempo.beat_to_time(star_power[0][0] / res) * 1000)
    (song_dir / "song.ini").write_text(
        chartio.write_song_ini(meta, int(duration_s * 1000), hopos=opts.hopos,
                               diff_guitar=tier, preview_start_ms=preview_ms,
                               diff_guitarghl=tier if "ExpertGHLGuitar" in tiers else -1),
        encoding="utf-8",
    )

    summary = {}
    tier_lanes = {"MediumSingle": 4, "EasySingle": 3}  # RBN-style lane budgets
    for tier, (notes, sustains, sp) in chartio.summarize(chart_path).items():
        summary[tier] = {
            "notes": notes, "sustains": sustains, "star_power": sp,
            "variety": quality.variety_score(tiers[tier], tier_lanes.get(tier, 5)),
            "walk": quality.walk_score(tiers[tier]),
        }
    if prefetch is not None:
        prefetch.close()
    progress(f"done -> {song_dir}")
    return {
        "song_dir": song_dir, "summary": summary, "bpm": tempo.bpm,
        "tempo_events": len(tempo.sync_track()), "star_power": len(star_power),
        "sections": len(events), "score": best,
    }
