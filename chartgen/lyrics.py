"""Lyric events for Clone Hero: real synced lyrics first, transcription second.

CH renders lyrics from `[Events]` lines: `phrase_start` opens a line at the
top of the highway, `lyric <word>` fills it word by word, `phrase_end` closes
it. Timing comes from the tempo map — lyrics are display text, so they keep
their raw ticks instead of being snapped to the note grid.

Two sources, tried in order:

1. LRCLIB (see lrclib.py). Hand-synced lines, real words, no model. Measured
   on the charts we had generated, it matched every song whose metadata was
   clean, which makes it the primary source rather than a nicety.
2. faster-whisper. The fallback for anything LRCLIB does not carry —
   unreleased tracks, remixes, anything the community has not synced.

Instrumentals need no special-casing: LRCLIB has no entry and the no-speech
filter yields nothing, so the chart is written without an empty lyric line.

The Whisper model (base, ~150MB) downloads from Hugging Face on first use and
is cached. GPU is tried first; any CUDA/cuDNN mismatch falls back to CPU int8,
which transcribes a 4-minute song in a couple of minutes.
"""
import re

# Reference-decoder thresholds (openai/whisper transcribe.py, inherited by
# faster-whisper). The gate that matters is a CONJUNCTION: a segment is only
# silence when it both scores high on no-speech AND decoded poorly. We used
# to test no_speech_prob > 0.9 alone, which is neither half of that rule —
# too lax at the fade-out where hallucinations live, and too strict on quiet
# singing (measured 0.658 on a clearly-sung chorus). Against LRCLIB as a
# reference, that filter left us at 80-88% of the real word count on three
# songs and 107-125% on three others.
NO_SPEECH = 0.6
LOG_PROB = -1.0
# gzip ratio of the decoded text; a repetition loop compresses far better
# than language does. The reference default is 2.4.
COMPRESSION = 2.4
# faster-whisper skips silent stretches longer than this while assigning word
# timestamps — the named remedy for text invented over a fade-out.
SILENCE_SKIP = 2.0

# Whisper was trained on YouTube captions, so on non-speech audio it falls
# back on caption boilerplate. This is not hypothetical here: a generated
# chart ended with "Thank you for watching, I'll see you in the next one"
# over its outro, and three others closed on a bare "you".
_BOILERPLATE = (
    "thank you for watching", "thanks for watching", "thank you so much for watching",
    "see you in the next", "see you next time", "please subscribe", "like and subscribe",
    "subscribe to", "subtitles by", "subtitles created by", "amara.org",
    "transcription by", "translated by", "captions by", "all rights reserved",
)
# A trailing scrap this short, stranded after a long instrumental gap, is
# never a real closing line.
ORPHAN_WORDS = 5
ORPHAN_GAP_S = 5.0
# A whole-song transcription this small is not singing at all: playtest
# found an entirely-piano piece carrying a single hallucinated "you" -
# one lone segment slips the tail guards (no gap before it, not
# boilerplate), so the floor is the last line of defence.
MIN_SONG_WORDS = 8

_MODEL = None


def _load_model(progress):
    global _MODEL
    if _MODEL is not None:
        return _MODEL

    from faster_whisper import WhisperModel

    try:
        import torch

        if torch.cuda.is_available():
            try:
                # small: the GPU affords it, and sung vocals need the accuracy.
                _MODEL = WhisperModel("small", device="cuda", compute_type="float16")
                return _MODEL
            except Exception:
                progress("      (GPU transcription unavailable, using CPU)")
    except ImportError:
        pass
    _MODEL = WhisperModel("base", device="cpu", compute_type="int8")
    return _MODEL


# In .chart, a handful of ASCII characters are lyric MARKUP rather than text
# (per the format spec): '-' and '=' join this syllable to the next one, '_'
# stands in for a space, and '#^*%$/<>' are pitch/display directives. Whisper
# writes ordinary punctuation — a transcribed "well-known" would silently
# swallow the following word on the highway. Double quotes would end the
# E "..." line outright.
_AS_SPACE = "-=_"
_DROP = '+#^*%$/<>"§'


def _clean(word: str) -> str:
    """Strip .chart lyric markup so transcribed words display as written."""
    text = word.strip()
    for char in _AS_SPACE:
        text = text.replace(char, " ")
    for char in _DROP:
        text = text.replace(char, "")
    return " ".join(text.split())


def _phrases_to_events(phrases, tempo) -> list[tuple[int, str]]:
    """[(tick, event)] for [(words, end_time), ...] groups, in reading order.

    Each phrase is ([(word_time, word), ...], end_seconds). The end matters
    as much as the start: CH clears the line at phrase_end, and the first
    playtest of the LRC path closed phrases a quarter-beat after the LAST
    WORD STARTED — lines vanished while their final word was still being
    sung. A line stays up until its stated end (for LRC, the next line's
    start; for Whisper, the segment's real end).

    Emission order is the output order: write_chart keeps it for events that
    land on the same tick, which is what stops a phrase from being scrambled
    when several words quantize together.
    """
    res = tempo.resolution
    events: list[tuple[int, str]] = []
    last_end = 0
    for words, end_s in phrases:
        if not words:
            continue
        first = max(0, int(round(tempo.time_to_beat(words[0][0]) * res)))
        # Open the phrase half a beat early — never before the previous one
        # closed — so the line is on screen by the time it is sung.
        events.append((max(last_end, first - res // 2, 0), "phrase_start"))
        tick = first
        for when, text in words:
            # Non-decreasing: a source with sloppy timing must not reorder
            # the line, and equal ticks stay in the order written here.
            tick = max(tick, int(round(tempo.time_to_beat(when) * res)))
            events.append((tick, f"lyric {text}"))
        end_tick = int(round(tempo.time_to_beat(float(end_s)) * res)) if end_s else 0
        last_end = max(end_tick, tick + res // 4, first + 1)
        events.append((last_end, "phrase_end"))
    return events


_ALIGNER = None


def _load_aligner():
    """torchaudio's MMS forced aligner: known text -> per-word times."""
    global _ALIGNER
    if _ALIGNER is None:
        import torch
        import torchaudio

        bundle = torchaudio.pipelines.MMS_FA
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _ALIGNER = (bundle.get_model(with_star=False).to(device).eval(),
                    bundle.get_tokenizer(), bundle.get_aligner(),
                    bundle.sample_rate, device)
    return _ALIGNER


def align_words(lines, vocal, sr: int, duration_s: float,
                progress=lambda m: None):
    """Real per-word times for LRC lines, from the isolated vocal stem.

    LRC syncs LINES; the word-by-word scroll inside each line was
    synthetic and visibly wrong in play. We know the words (LRCLIB) and
    where the singing physically is (the Demucs vocal stem); forced
    alignment joins them. Per line: slice the stem over the line's
    window, align that line's text, keep the emitted word starts.
    Returns {line_index: [(seconds, word), ...]} for lines that aligned;
    callers fall back to the spread for the rest.
    """
    import re as _re

    import torch
    import torchaudio

    model, tokenizer, aligner, model_sr, device = _load_aligner()
    wave = torch.from_numpy(vocal).to(device)
    if sr != model_sr:
        wave = torchaudio.functional.resample(wave, sr, model_sr)

    aligned: dict[int, list] = {}
    with torch.inference_mode():
        for i, (start, text) in enumerate(lines):
            words = [w for w in (_clean(part) for part in text.split()) if w]
            tokens = [_re.sub(r"[^a-z']", "", w.lower()) for w in words]
            if not words or any(not t for t in tokens):
                continue
            end = lines[i + 1][0] if i + 1 < len(lines) else                 min(duration_s, start + 12.0)
            a = max(0, int((start - 0.25) * model_sr))
            b = min(len(wave), int(end * model_sr))
            if b - a < model_sr // 4:
                continue
            try:
                emission, _ = model(wave[a:b].unsqueeze(0))
                spans = aligner(emission[0], tokenizer(tokens))
            except Exception:
                continue
            ratio = (b - a) / emission.shape[1] / model_sr
            aligned[i] = [
                (a / model_sr + span[0].start * ratio, word)
                for span, word in zip(spans, words, strict=True) if span
            ]
    progress(f"      {len(aligned)}/{len(lines)} lines word-aligned to the "
             f"vocal stem")
    return aligned


def from_lines(lines, tempo, duration_s: float,
               word_times: dict | None = None) -> list[tuple[int, str]]:
    """CH events from LRC lines: [(seconds, text)] with line-level timing.

    When forced alignment supplied real word times for a line
    (word_times[i]), they are used as-is. Otherwise LRC is synced per
    line, not per word, so words are spread across the line at a
    plausible singing rate — weighted by length, because "the" is not
    held as long as "shadow". The spread is capped so a line before a
    long instrumental break does not crawl across it.
    """
    phrases = []
    for i, (start, text) in enumerate(lines):
        words = [w for w in (_clean(part) for part in text.split()) if w]
        if not words:
            continue
        following = lines[i + 1][0] if i + 1 < len(lines) else duration_s
        real = (word_times or {}).get(i)
        if real and len(real) == len(words):
            span = max(0.3, real[-1][0] - start)
            spread = [(max(start, when), word) for when, word in real]
        else:
            room = max(0.3, following - start - 0.1)
            # ~2.6 words/sec is an ordinary sung line; never overrun the gap.
            span = min(room, 0.38 * len(words))
            weights = [len(w) + 1 for w in words]
            total = sum(weights)
            at, spread = start, []
            for word, weight in zip(words, weights, strict=True):
                spread.append((min(at, start + span), word))
                at += span * weight / total
        # The line is displayed until the next one starts (that is what LRC
        # line timing MEANS); the last line rings out for its sung length.
        end = (following - 0.1) if i + 1 < len(lines) else             min(duration_s, start + span + 1.5)
        phrases.append((spread, end))
    return _phrases_to_events(phrases, tempo)


def _boilerplate(text: str) -> bool:
    flat = re.sub(r"[^a-z0-9 ]", "", text.lower()).strip()
    return any(phrase in flat for phrase in _BOILERPLATE)


def _drop_invented_tail(segments):
    """Remove the caption boilerplate Whisper strands in instrumental outros.

    Two structural signals, no phrase list needed for most of it: text that
    collapses onto a single timestamp is not something anyone sang, and a
    handful of words marooned after a long gap at the end of a song is the
    fade-out hallucination rather than a closing line. The phrase list only
    catches the longer, well-formed hallucinations that pass both.
    """
    kept = list(segments)
    while kept:
        start, end, words = kept[-1]
        gap = start - kept[-2][1] if len(kept) > 1 else 0.0
        stranded = gap >= ORPHAN_GAP_S and len(words) <= ORPHAN_WORDS
        collapsed = len(words) > 2 and len({round(w[0], 2) for w in words}) <= 1
        if stranded or collapsed or _boilerplate(" ".join(w for _, w in words)):
            kept.pop()
            continue
        break
    # Boilerplate anywhere, not just at the end — a mid-song instrumental
    # break invites the same invention.
    return [s for s in kept if not _boilerplate(" ".join(w for _, w in s[2]))]


def transcribe(audio_path: str, tempo, progress=lambda m: None) -> list[tuple[int, str]]:
    """[(tick, event_text), ...] lyric events from the audio itself."""
    model = _load_model(progress)
    # No VAD: it eats heavily-produced choruses (Faded lost everything past
    # 48s with it on). condition_on_previous_text off stops one misheard
    # phrase from propagating through the whole song. The thresholds below
    # are the reference decoder's, which handle instrumental stretches.
    segments, info = model.transcribe(
        str(audio_path), word_timestamps=True, vad_filter=False,
        condition_on_previous_text=False,
        no_speech_threshold=NO_SPEECH, log_prob_threshold=LOG_PROB,
        compression_ratio_threshold=COMPRESSION,
        hallucination_silence_threshold=SILENCE_SKIP,
    )

    collected = []
    previous = None
    for segment in segments:
        if not segment.words:
            continue
        words = [(float(w.start), _clean(w.word)) for w in segment.words]
        words = [(t, text) for t, text in words if text]
        if not words:
            continue
        flat = " ".join(text for _, text in words).lower()
        # A segment repeating the one before it is a decoder loop, not a
        # repeated chorus line — a real repeat is separated by other words.
        if flat == previous:
            continue
        previous = flat
        collected.append((float(segment.start), float(segment.end), words))

    collected = _drop_invented_tail(collected)
    events = _phrases_to_events([(w, end) for _, end, w in collected], tempo)

    count = sum(1 for _, e in events if e.startswith("lyric "))
    if count < MIN_SONG_WORDS:
        progress("      no vocals detected; skipping lyrics")
        return []
    progress(f"      {count} words in "
             f"{sum(1 for _, e in events if e == 'phrase_start')} phrases"
             f" (language: {info.language})")
    return events


# A sheet can match on duration and still sit seconds off — Faded's LRCLIB
# sync ran 3.1s early against our rip (216s audio vs the 213s cut it was
# timed to), which played as lines vanishing before the singing started.
# Grid-search the global shift that puts line starts on the audio's onsets;
# apply it only when it clearly beats no shift, so a correct sheet is never
# nudged by noise.
ALIGN_RANGE_S = 6.0
ALIGN_STEP_S = 0.05
ALIGN_GAIN = 1.15


def align_offset(lines, y, sr, duration_s: float) -> float:
    """Global offset (seconds) that best lands line starts on onsets."""
    import librosa
    import numpy as np

    starts = np.array([t for t, _ in lines], dtype=float)
    if len(starts) < 8 or y is None or not len(y):
        return 0.0
    env = librosa.onset.onset_strength(y=y, sr=sr)
    if env.max() <= 0:
        return 0.0
    env = env / env.max()
    times = librosa.times_like(env, sr=sr)

    def score(offset: float) -> float:
        shifted = starts + offset
        ok = (shifted >= 0) & (shifted <= duration_s)
        if ok.sum() < 0.8 * len(starts):
            return -1.0
        idx = np.clip(np.searchsorted(times, shifted[ok]), 0, len(env) - 1)
        return float(np.mean([env[max(0, i - 2):i + 3].max() for i in idx]))

    base = score(0.0)
    offsets = np.arange(-ALIGN_RANGE_S, ALIGN_RANGE_S + 1e-9, ALIGN_STEP_S)
    scores = [score(o) for o in offsets]
    best = float(offsets[int(np.argmax(scores))])
    if base <= 0 or max(scores) < ALIGN_GAIN * base or abs(best) < 0.2:
        return 0.0
    return best


# Share of synced lines that must land on audible audio. A sheet timed
# against a different edit — a YouTube rip with an added intro, a radio cut —
# can still match on duration, and would then sit wrong for the whole song.
# Checking that the words land where there is sound catches that; it is a
# gross-misalignment test, not a precision one, hence the generous bar.
MIN_LINES_ON_SOUND = 0.7


def _lands_on_sound(lines, y, sr) -> bool:
    """Do the synced lines fall where the recording actually plays?"""
    import numpy as np

    if y is None or not len(y):
        return True
    frame = max(1, int(0.25 * sr))
    loud = float(np.percentile(np.abs(y), 95)) or 1.0
    hits = 0
    for when, _ in lines:
        at = int(when * sr)
        chunk = y[max(0, at - frame):at + frame]
        if len(chunk) and float(np.abs(chunk).mean()) > 0.05 * loud:
            hits += 1
    return hits >= MIN_LINES_ON_SOUND * len(lines)


def collect(audio_path, tempo, artist: str, title: str, duration_s: float,
            source: str = "auto", progress=lambda m: None,
            y=None, sr=None, lookup=None) -> list[tuple[int, str]]:
    """Lyric events from the best source available.

    'auto' looks the song up on LRCLIB and only transcribes if that misses;
    'online' and 'transcribe' pin one source. `lookup`, when given, is a
    zero-argument callable standing in for the LRCLIB query (the pipeline
    starts the network round trip early and hands its result here).
    """
    if source in ("auto", "online"):
        from . import lrclib

        if lookup is not None:
            lines = lookup()
        else:
            lines = lrclib.find(artist, title, duration_s, progress=progress)
        if lines and y is not None:
            shift = align_offset(lines, y, sr, duration_s)
            if shift:
                progress(f"      synced lyrics shifted {shift:+.1f}s to match "
                         f"this recording")
                lines = [(max(0.0, t + shift), text) for t, text in lines]
        if lines and not _lands_on_sound(lines, y, sr):
            progress("      the synced lyrics do not line up with this audio; "
                     "transcribing instead")
            lines = None
        if lines:
            word_times = None
            try:
                from . import stems

                mono = stems.separate(str(audio_path), progress)
                if mono is not None:
                    word_times = align_words(lines, mono["vocals"], stems.SR,
                                             duration_s, progress)
            except Exception as error:  # alignment is polish, never fatal
                progress(f"      word alignment skipped: "
                         f"{type(error).__name__}: {error}")
            events = from_lines(lines, tempo, duration_s, word_times)
            progress(f"      {sum(1 for _, e in events if e.startswith('lyric '))}"
                     f" words in {len(lines)} synced lines")
            return events
        if source == "online":
            progress("      no synced lyrics found; skipping lyrics")
            return []
    return transcribe(audio_path, tempo, progress)
