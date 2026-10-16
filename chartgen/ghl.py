"""6-fret (Guitar Hero Live) shapes from the transcription.

Lanes 0-5 are W1 W2 W3 B1 B2 B3 and 7 is open, in the same (tick, lane,
sustain) tuples the 5-fret chart uses, so every pass that only reads ticks
and lane identity works on either part.

Three models trained on the official FreeStyleGames charts
(tools/train_ghl.py): TimingNet picks the Expert note times from the onsets
of every stream, ShapeNet picks the Expert shape at each of them from the
pitches heard around it, and ReduceNet derives Hard, Medium and Easy from
the Expert shapes.
"""
import bisect
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix
from torch import nn

from .tempo import ONSET_LAG_S

OPEN = 7
BUTTONS = 6
TIERS = ("Hard", "Medium", "Easy")
MODEL_PATH = Path(__file__).with_name("ghl_model.pt")

LOW_MIDI, PITCHES = 21, 88
STREAMS = ("mix", "guitar", "bass", "piano", "sw_other", "vocals")
ONSET_BEFORE_S, ONSET_AFTER_S = 0.05, 0.08
HEARD = 0.2  # amplitude above which a pitch counts as played
ROWS = len(STREAMS) + 2  # onsets per stream, then held mix and guitar pitches
STATS = 5
SCALARS = 6
CONTEXT_NOTES = (4, 16)
N_FEATURES = (3 + len(CONTEXT_NOTES) + 2) * ROWS * STATS + 2 + SCALARS
WINDOW = 256
MERGE_S = 0.03  # onsets this close to a slot count as heard there
GAP_S = 0.06  # the closest two official Expert notes ever sit
GRID = 4  # 16ths, the grid the tier features describe
SLOTS_PER_BEAT = 12  # 16ths plus 8th and 16th triplets: every official note but 32nds
ENVELOPE_SR, ENVELOPE_HOP = 22050, 512
ENVELOPE_FRAMES = 2  # envelope samples either side of a slot
N_TIMING = N_FEATURES - 2 + 7 + 2 * ENVELOPE_FRAMES + 1
# Official Expert notes with at least this long a gap to the next note are
# sustained, to that next note (F1 0.80 on held-out songs; opens never are).
SUSTAIN_GAP_S = 0.3
# A lower tier keeps an Expert sustain when its own next note is at least
# this many beats away, and never places two notes closer than its gap.
TIER_SUSTAIN_BEATS = {"Hard": 1.0, "Medium": 2.0, "Easy": 2.25}
TIER_MIN_GAP_BEATS = {"Hard": 1 / 12, "Medium": 0.2, "Easy": 1 / 3}
TIER_FEATURES = 25
FIGURE_SAME = 0.5  # P(SAME) at or below which a note starts a new figure
DIFFERENT_SAME = 0.35  # P(SAME) at or below which a new figure must change shape
QUOTA_BOUNDS = (0.7, 1.1)  # share of its quota each Expert shape may use
MILP_SECONDS = 20


def shape_class(lanes) -> int:
    """Bitmask over the six buttons; 0 is an open note."""
    return sum(1 << lane for lane in lanes if lane != OPEN)


def class_lanes(shape: int) -> list[int]:
    return [b for b in range(BUTTONS) if shape >> b & 1] or [OPEN]


# How a shape relates to the one before it: the same shape, the same
# column (another row or size), a column further up or down the neck, or
# a change to or from an open note.
SAME, SAME_COLUMN, UP, DOWN, TO_OPEN, FROM_OPEN = range(6)
RELATIONS = 6


def _relation(prev: int, cur: int) -> int:
    if prev == cur:
        return SAME
    if not cur:
        return TO_OPEN
    if not prev:
        return FROM_OPEN
    column = [np.mean([b % 3 for b in class_lanes(shape)]) for shape in (prev, cur)]
    return SAME_COLUMN if column[0] == column[1] else UP if column[1] > column[0] else DOWN


RELATION_OF = np.array([[_relation(a, b) for b in range(1 << BUTTONS)]
                        for a in range(1 << BUTTONS)])


def _factors(shape: int) -> tuple[int, int, int]:
    """Buttons held (0 open, 3 for any triple), row (0 white, 1 black, 2
    both) and the set of columns used as a 3-bit mask less one. An open
    note has no row or columns (-1)."""
    lanes = [b for b in range(BUTTONS) if shape >> b & 1]
    if not lanes:
        return 0, -1, -1
    row = 0 if max(lanes) < 3 else 1 if min(lanes) >= 3 else 2
    return min(len(lanes), 3), row, sum(1 << column for column in {b % 3 for b in lanes}) - 1


FACTOR_OF = np.array([_factors(shape) for shape in range(1 << BUTTONS)])
FACTOR_SIZES = (4, 3, 7)
# 1 where two shapes are both fretted and on different rows (white, black, both).
ROW_SWITCH = ((FACTOR_OF[:, 1][:, None] != FACTOR_OF[:, 1][None, :])
              & (FACTOR_OF[:, 1][:, None] >= 0) & (FACTOR_OF[:, 1][None, :] >= 0)).astype(np.float32)


def _pitch_rows(events, times: np.ndarray, sounding: bool) -> np.ndarray:
    """[N, 88] max amplitude per pitch of the events starting near (or, with
    sounding, held across) each time."""
    out = np.zeros((len(times), PITCHES), dtype=np.float32)
    if not events:
        return out
    ev = np.asarray(events, dtype=np.float64)
    start, end = ev[:, 0][None, :], ev[:, 1][None, :]
    pitch = np.clip(ev[:, 2].astype(int) - LOW_MIDI, 0, PITCHES - 1)
    t = times[:, None]
    hit = ((start <= t) & (end > t + ONSET_AFTER_S)) if sounding else \
        ((start >= t - ONSET_BEFORE_S) & (start <= t + ONSET_AFTER_S))
    rows, cols = np.nonzero(hit)
    np.maximum.at(out, (rows, pitch[cols]), ev[cols, 3].astype(np.float32))
    return out


def _stats(rows: np.ndarray, centre: int) -> np.ndarray:
    """[N, STATS]: how many pitches sound, the loudest amplitude, the top
    and bottom pitch in octaves from the song's centre, and their span."""
    heard = rows > HEARD
    has = heard.any(1)
    index = np.arange(PITCHES)[None, :]
    top = np.where(heard, index, -1).max(1)
    low = np.where(heard, index, PITCHES).min(1)
    return np.stack([heard.sum(1), rows.max(1), np.where(has, top - centre, 0) / 12,
                     np.where(has, low - centre, 0) / 12, np.where(has, top - low, 0) / 12], 1)


def _context(stats: np.ndarray, times: np.ndarray) -> np.ndarray:
    """The stats averaged over CONTEXT_NOTES positions either side, their
    mean and spread over the song, and the log count of positions within
    2 s and per second of the song."""
    total = np.vstack([np.zeros_like(stats[:1]), np.cumsum(stats, 0)])
    index = np.arange(len(stats))
    rolling = []
    for k in CONTEXT_NOTES:
        low, high = np.maximum(index - k, 0), np.minimum(index + k + 1, len(stats))
        rolling.append((total[high] - total[low]) / (high - low)[:, None])
    song = np.hstack([stats.mean(0), stats.std(0)]) if len(stats) else np.zeros(2 * stats.shape[1])
    near = np.searchsorted(times, times + 2) - np.searchsorted(times, times - 2)
    rate = len(times) / max(np.ptp(times), 1.0) if len(times) else 0.0
    return np.hstack([*rolling, np.broadcast_to(song, (len(stats), len(song))),
                      np.log1p(near)[:, None], np.full((len(stats), 1), np.log1p(rate))])


def features(times: np.ndarray, beat_phase: np.ndarray, sustain_s: np.ndarray,
             mix_events, stem_events: dict) -> np.ndarray:
    """[N, N_FEATURES] per chart position: pitch statistics of each stream's
    onsets and of the held mix and guitar notes, their change from the
    previous position and into the next, their context in the song
    (_context) and timing scalars. Pitches are relative to the song's
    centre, so the key does not matter."""
    streams = {"mix": mix_events} | {k: v for k, v in stem_events.items() if k in STREAMS}
    centre_events = streams.get("guitar") or mix_events or [(0, 0, 60, 0)]
    centre = int(np.median([e[2] for e in centre_events]))
    rows = [_pitch_rows(streams.get(name) or [], times, False) for name in STREAMS]
    rows += [_pitch_rows(streams.get(name) or [], times, True) for name in ("mix", "guitar")]
    stats = np.hstack([_stats(r, centre - LOW_MIDI) for r in rows])
    before = np.vstack([np.zeros_like(stats[:1]), stats[:-1]])
    after = np.vstack([stats[1:], np.zeros_like(stats[:1])])
    shift = centre - 60

    gap_prev = np.diff(times, prepend=times[0] - 4.0) if len(times) else times
    gap_next = np.diff(times, append=times[-1] + 4.0) if len(times) else times
    scalars = np.stack([np.log1p(gap_prev), np.log1p(gap_next), np.log1p(sustain_s),
                        np.sin(2 * np.pi * beat_phase), np.cos(2 * np.pi * beat_phase),
                        np.full_like(times, shift / 12.0)], axis=1).astype(np.float32)
    return np.hstack([stats, stats - before, after - stats, _context(stats, times), scalars]).astype(np.float32)


def onset_envelope(y: np.ndarray, sr: int) -> np.ndarray:
    """librosa onset strength of the mix at ENVELOPE_SR, over its 95th
    percentile."""
    import librosa

    if sr != ENVELOPE_SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=ENVELOPE_SR)
    env = librosa.onset.onset_strength(y=y, sr=ENVELOPE_SR, hop_length=ENVELOPE_HOP)
    return (env / max(float(np.percentile(env, 95)), 1e-6)).astype(np.float32)


def slots(tempo, end_s: float) -> tuple[np.ndarray, np.ndarray]:
    """Note-time candidates: (ticks, seconds) of every SLOTS_PER_BEAT
    position of the beat grid up to end_s."""
    step = tempo.resolution // SLOTS_PER_BEAT
    ticks = np.arange(0, round(tempo.time_to_beat(end_s) * tempo.resolution) + 1, step)
    return ticks, np.array([tempo.beat_to_time(t / tempo.resolution) for t in ticks])


def timing_features(ticks: np.ndarray, times: np.ndarray, res: int, envelope: np.ndarray, mix_events,
                    stem_events: dict) -> np.ndarray:
    """[N, N_TIMING] per slot: the pitch statistics of features(), how many
    onsets of any stream fall within MERGE_S and how far the nearest is,
    the onset envelope around it, and which grids it belongs to."""
    beat = ticks / res
    x = features(times, beat % 1, np.zeros_like(times), mix_events, stem_events)
    x = np.delete(x, [N_FEATURES - 4, N_FEATURES - 1], axis=1)  # sustain and key shift
    onsets = np.sort([e[0] for events in [mix_events, *stem_events.values()] for e in events])
    low, high = np.searchsorted(onsets, times - MERGE_S), np.searchsorted(onsets, times + MERGE_S)
    if len(onsets):
        k = np.clip(np.searchsorted(onsets, times), 1, len(onsets) - 1)
        nearest = np.minimum(np.abs(onsets[k] - times), np.abs(onsets[k - 1] - times))
    else:
        nearest = np.full(len(times), 1.0)
    frame = np.round((times + ONSET_LAG_S) * ENVELOPE_SR / ENVELOPE_HOP).astype(int)
    around = np.stack([envelope[np.clip(frame + d, 0, len(envelope) - 1)] if len(envelope) else np.zeros(len(times))
                       for d in range(-ENVELOPE_FRAMES, ENVELOPE_FRAMES + 1)], 1)
    slot = ticks % res * SLOTS_PER_BEAT // res
    grid = np.stack([np.log1p(high - low), np.log1p(nearest * 100), slot % 3 == 0, slot % 4 == 0,
                     slot % 2 == 0, np.sin(np.pi * beat), np.cos(np.pi * beat)], 1)
    return np.hstack([x, grid, around]).astype(np.float32)


def pick_slots(logits: np.ndarray, times: np.ndarray, threshold: float) -> np.ndarray:
    """Indices of the slots above threshold, most likely first, skipping any
    within GAP_S of one already picked."""
    taken = np.zeros(len(times), bool)
    picked = []
    for i in np.argsort(-logits):
        if logits[i] <= threshold:
            break
        if not taken[i]:
            picked.append(i)
            low, high = np.searchsorted(times, [times[i] - GAP_S, times[i] + GAP_S])
            taken[low:high] = True
    return np.sort(np.array(picked, dtype=np.int64))


class _Encoder(nn.Module):
    def __init__(self, n_in: int, width: int, layers: int, dropout: float):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(n_in, width), nn.GELU(), nn.LayerNorm(width),
                                 nn.Dropout(dropout))
        self.pos = nn.Embedding(WINDOW, width)
        layer = nn.TransformerEncoderLayer(width, 4, 4 * width, dropout=dropout,
                                           batch_first=True, norm_first=True)
        self.body = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)

    def forward(self, x, pad):
        index = torch.arange(x.shape[1], device=x.device)
        return self.body(self.inp(x) + self.pos(index), src_key_padding_mask=pad)


class ShapeNet(nn.Module):
    """heads(): per position, logits for each shape factor (FACTOR_SIZES)
    and for the RELATIONS to the previous position's shape. forward()
    composes the factors' log probabilities, scaled by factor_weights, into
    scores over the 64 button bitmasks, adding the official charts' log
    P(shape | its factors). The scores are then divided by temperature and
    shifted by shape_bias, both fitted by likelihood on the validation
    songs, since the network alone is far too sure of its best shape. The
    relation logits follow the 64 scores."""

    def __init__(self, width: int = 128, layers: int = 4, dropout: float = 0.2):
        super().__init__()
        self.encoder = _Encoder(N_FEATURES, width, layers, dropout)
        self.head = nn.Linear(width, sum(FACTOR_SIZES) + RELATIONS)
        self.register_buffer("given_factors", torch.zeros(1 << BUTTONS))
        self.register_buffer("factor_weights", torch.ones(len(FACTOR_SIZES)))
        self.register_buffer("temperature", torch.ones(()))
        self.register_buffer("shape_bias", torch.zeros(1 << BUTTONS))
        self.register_buffer("factor_of", torch.from_numpy(FACTOR_OF), persistent=False)

    def heads(self, x, pad):
        return self.head(self.encoder(x, pad)).split([*FACTOR_SIZES, RELATIONS], -1)

    def forward(self, x, pad):
        *factors, relations = self.heads(x, pad)
        score = self.given_factors.expand(*relations.shape[:-1], -1)
        for k, logits in enumerate(factors):
            index = self.factor_of[:, k]
            score = score + self.factor_weights[k] * torch.where(
                index >= 0, logits.log_softmax(-1)[..., index.clamp(min=0)], 0)
        return torch.cat([score / self.temperature + self.shape_bias, relations], -1)


class ReduceNet(nn.Module):
    """Per position and lower tier: keep logit plus 64 shape logits, from
    the Expert shape and tier_features()."""

    def __init__(self, width: int = 128, layers: int = 3, dropout: float = 0.2):
        super().__init__()
        self.encoder = _Encoder((1 << BUTTONS) + TIER_FEATURES, width, layers, dropout)
        self.head = nn.Linear(width, len(TIERS) * (1 + (1 << BUTTONS)))

    def forward(self, x, pad):
        out = self.head(self.encoder(x, pad))
        return out.view(*out.shape[:2], len(TIERS), 1 + (1 << BUTTONS))


class TimingNet(nn.Module):
    """Per note-time candidate: the logit that the official chart has an
    Expert note there."""

    def __init__(self, width: int = 128, layers: int = 4, dropout: float = 0.2):
        super().__init__()
        self.encoder = _Encoder(N_TIMING, width, layers, dropout)
        self.head = nn.Linear(width, 1)

    def forward(self, x, pad):
        return self.head(self.encoder(x, pad))


def tier_features(ticks: np.ndarray, sustain: np.ndarray, shapes: np.ndarray, res: int,
                  downbeat: int | None) -> np.ndarray:
    """[N, TIER_FEATURES] per Expert position, from the chart alone: its
    16th within the beat, the gaps to the two notes either side in beats,
    its sustain, chord size, row and repeat, its neighbours' size and
    whether they sit on the beat, and its beat in a 4/4 bar counted from
    downbeat (the beat number of any downbeat). Without a downbeat the bar
    columns are zero: bars counted from tick 0 place notes worse than none."""
    beat = ticks / res
    sixteenth = np.round(beat * 4)
    phase = np.eye(4, dtype=np.float32)[(sixteenth % 4).astype(int)]
    off_grid = np.abs(beat * 4 - sixteenth) > 0.05
    gap = np.diff(beat)

    def shifted(values, by, fill):
        out = np.full(len(values), fill, dtype=np.float32)
        if by > 0:
            out[by:] = values[:-by]
        else:
            out[:by] = values[-by:]
        return out

    before = np.concatenate([[4.0], gap])
    after = np.concatenate([gap, [4.0]])
    size = np.minimum(np.bitwise_count(shapes), 3).astype(np.float32)
    row = np.array([3 if not s else 0 if s < 8 else 1 if not s & 7 else 2 for s in shapes])
    on_beat = (sixteenth % 4 == 0).astype(np.float32)
    in_bar = (beat - (downbeat or 0)) % 4
    bar = np.column_stack([np.eye(4)[in_bar.astype(int)], in_bar / 4]) * (downbeat is not None)
    columns = [*phase.T, off_grid, np.log1p(before), np.log1p(after), np.log1p(shifted(before, 1, 4.0)),
               np.log1p(shifted(after, -1, 4.0)), np.log1p(sustain / res), size / 3,
               *np.eye(4, dtype=np.float32)[row].T, shifted(shapes, 1, -1) == shapes,
               shifted(size, 1, 0) / 3, shifted(size, -1, 0) / 3, shifted(on_beat, 1, 0), shifted(on_beat, -1, 0),
               *bar.T]
    return np.stack(columns, 1).astype(np.float32)


def reduce_inputs(shapes: np.ndarray, tier_x: np.ndarray) -> np.ndarray:
    return np.concatenate([np.eye(1 << BUTTONS, dtype=np.float32)[shapes], tier_x], axis=1)


def nested_keep(scores: np.ndarray, ticks: np.ndarray, keep_share: dict, res: int) -> dict[str, np.ndarray]:
    """Per lower tier: the official share of Expert positions, best score
    first, each tier drawn from the one above it and never closer than its
    minimum gap."""
    out, pool = {}, np.ones(len(ticks), bool)
    for k, tier in enumerate(TIERS):
        keep = np.zeros(len(ticks), bool)
        kept_ticks: list[int] = []
        gap = TIER_MIN_GAP_BEATS[tier] * res
        for i in np.argsort(-scores[:, k]):
            if len(kept_ticks) >= round(keep_share[tier] * len(ticks)):
                break
            at = bisect.bisect_left(kept_ticks, ticks[i])
            if pool[i] and all(abs(ticks[i] - t) >= gap for t in kept_ticks[max(0, at - 1):at + 1]):
                keep[i] = True
                kept_ticks.insert(at, int(ticks[i]))
        out[tier] = pool = keep
    return out


def sustains(ticks: np.ndarray, keep: np.ndarray, expert_sustained: np.ndarray, min_beats: float,
             res: int) -> np.ndarray:
    """Per position: the sustain a tier gives it, which runs to the tier's
    next note when the Expert note is sustained and that note is at least
    min_beats away."""
    kept = ticks[keep]
    following = np.concatenate([kept[1:], [kept[-1] if len(kept) else 0]])
    out = np.zeros(len(ticks), dtype=np.int64)
    room = following - kept
    out[keep] = np.where(expert_sustained[keep] & (room >= min_beats * res), room, 0)
    return out


def windowed(model: nn.Module, x: np.ndarray, device: str) -> np.ndarray:
    """Run a sequence model over a whole song in overlapping windows and
    keep each window's middle, where both sides have context."""
    step = WINDOW // 2
    out = None
    with torch.no_grad():
        for begin in range(0, max(1, len(x) - step), step):
            chunk = torch.from_numpy(x[begin:begin + WINDOW]).to(device)[None]
            pad = torch.zeros(chunk.shape[:2], dtype=torch.bool, device=device)
            logits = model(chunk, pad)[0].float().cpu().numpy()
            if out is None:
                out = np.zeros((len(x),) + logits.shape[1:], dtype=np.float32)
            keep_from = 0 if begin == 0 else WINDOW // 4
            out[begin + keep_from:begin + len(logits)] = logits[keep_from:]
    return out if out is not None else np.zeros((0, 1 << BUTTONS), dtype=np.float32)


_MODELS: dict = {}


def _models(device: str):
    if device not in _MODELS:
        saved = torch.load(MODEL_PATH, map_location=device)
        nets = []
        for name, net in (("timing", TimingNet), ("shape", ShapeNet), ("reduce", ReduceNet)):
            model = net(**saved["config"][name])
            model.load_state_dict({k: v.float() for k, v in saved[name].items()})
            nets.append(model.to(device).eval())
        _MODELS[device] = (*nets, saved)
    return _MODELS[device]


def chart_tiers(tempo, envelope, mix_events, stem_events) -> dict[str, list]:
    """{"ExpertGHLGuitar": notes, "HardGHLGuitar": ...}, or {} when no
    trained model ships. envelope is onset_envelope() of the mix."""
    onsets = [e[0] for events in [mix_events, *stem_events.values()] for e in events]
    if not MODEL_PATH.is_file() or not onsets:
        return {}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    timing_net, shape_net, reduce_net, saved = _models(device)
    allowed = saved["allowed"]
    res = tempo.resolution
    slot_ticks, slot_times = slots(tempo, max(onsets) + 1.0)
    x = timing_features(slot_ticks, slot_times, res, envelope, mix_events, stem_events)
    ticks = slot_ticks[pick_slots(windowed(timing_net, x, device)[:, 0], slot_times, saved["timing_threshold"])]
    if not len(ticks):
        return {}
    times = np.array([tempo.beat_to_time(t / res) for t in ticks])
    everywhere = np.ones(len(ticks), bool)
    sustained = np.diff(times, append=times[-1]) >= SUSTAIN_GAP_S
    expert_sustain = sustains(ticks, everywhere, sustained, 0, res)
    sustain_s = np.array([tempo.beat_to_time((t + s) / res) for t, s in zip(ticks, expert_sustain, strict=True)]) - times
    x = features(times, (ticks % res) / res, sustain_s, mix_events, stem_events)
    shapes = figure_shapes(windowed(shape_net, x, device), allowed["Expert"], saved["quota_power"])
    sustained &= shapes != 0
    expert_sustain[~sustained] = 0
    downbeat = None if tempo.downbeat_phase is None else tempo.pickup_beats + tempo.downbeat_phase
    tier_x = tier_features(ticks, expert_sustain, shapes, res, downbeat)
    reduced = windowed(reduce_net, reduce_inputs(shapes, tier_x), device)

    def notes(keep, tier_shapes, tier_sustain):
        return [(int(t), lane, int(sus)) for t, k, s, sus in zip(ticks, keep, tier_shapes, tier_sustain, strict=True)
                if k for lane in class_lanes(int(s))]

    out = {"ExpertGHLGuitar": notes(everywhere, shapes, expert_sustain)}
    kept = nested_keep(reduced[..., 0], ticks, saved["keep_share"], res)
    for k, tier in enumerate(TIERS):
        tier_shapes = np.zeros(len(ticks), dtype=np.int64)
        tier_shapes[kept[tier]] = viterbi(reduced[kept[tier], k, 1:], allowed[tier], saved["switch_weights"][tier])
        tier_sustain = sustains(ticks, kept[tier], sustained & (tier_shapes != 0), TIER_SUSTAIN_BEATS[tier], res)
        out[f"{tier}GHLGuitar"] = notes(kept[tier], tier_shapes, tier_sustain)
    return out


def _log_softmax(logits: np.ndarray) -> np.ndarray:
    return logits - np.logaddexp.reduce(logits, axis=-1, keepdims=True)


def figure_shapes(logits: np.ndarray, allowed, power: float) -> np.ndarray:
    """Expert shapes from ShapeNet's output. A figure (a run of one shape)
    starts wherever the relation head gives P(SAME) <= FIGURE_SAME. Each
    figure gets one shape, chosen by a MILP that maximises the total log
    probability while every shape's note count stays within QUOTA_BOUNDS of
    its quota, and two neighbouring figures differ where P(SAME) <=
    DIFFERENT_SAME. The quota is the song's expected count of the shape
    ** power, rescaled so each chord size keeps its expected note count:
    the power concentrates a song on fewer shapes without changing how
    many notes are single. Per-note picks put the same few shapes
    everywhere. When the MILP finds no solution in MILP_SECONDS, each
    figure takes its best shape."""
    mask = np.full(1 << BUTTONS, -np.inf)
    mask[list(allowed)] = 0
    logp = _log_softmax(logits[:, :1 << BUTTONS] + mask)
    same = np.exp(_log_softmax(logits[:, 1 << BUTTONS:]))[:, SAME]
    starts = np.concatenate([[True], same[1:] <= FIGURE_SAME])
    figure = np.cumsum(starts) - 1
    expected = np.exp(logp).sum(0)
    size = FACTOR_OF[:, 0]
    sharp = expected ** power
    quota = (sharp * np.bincount(size, weights=expected)[size]
             / np.maximum(np.bincount(size, weights=sharp)[size], 1e-12))[list(allowed)]
    gain = np.zeros((figure[-1] + 1, len(logp[0])))
    np.add.at(gain, figure, logp)
    gain = gain[:, list(allowed)]
    n_figures, n_shapes = gain.shape
    choices = np.arange(gain.size)
    one_shape = coo_matrix((np.ones(gain.size), (choices // n_shapes, choices)), shape=(n_figures, gain.size))
    notes = coo_matrix((np.bincount(figure)[choices // n_shapes].astype(float), (choices % n_shapes, choices)),
                       shape=(n_shapes, gain.size))
    # One row per shape and figure f whose next figure starts with P(SAME) <= DIFFERENT_SAME.
    sure = np.flatnonzero(same[starts][1:] <= DIFFERENT_SAME)
    pair = (sure[:, None] * n_shapes + np.arange(n_shapes)).ravel()
    different = coo_matrix((np.ones(2 * len(pair)), (np.tile(np.arange(len(pair)), 2),
                                                      np.concatenate([pair, pair + n_shapes]))),
                           shape=(len(pair), gain.size))
    low, high = QUOTA_BOUNDS
    found = milp(-gain.ravel(), integrality=np.ones(gain.size), bounds=Bounds(0, 1),
                 constraints=[LinearConstraint(one_shape, 1, 1),
                              LinearConstraint(notes, np.maximum(np.floor(quota * low) - 1, 0),
                                               np.ceil(quota * high) + 2),
                              LinearConstraint(different, 0, 1)],
                 options={"time_limit": MILP_SECONDS})
    best = gain.argmax(1) if found.x is None else found.x.reshape(gain.shape).argmax(1)
    return np.array(allowed)[best][figure]


def viterbi(logits: np.ndarray, allowed, switch_weight: float) -> np.ndarray:
    """The allowed shape sequence with the best total of each position's
    shape log probability, less switch_weight for every change of row
    between fretted notes."""
    if not len(logits):
        return np.zeros(0, dtype=np.int64)
    shapes = _log_softmax(logits)
    banned = np.ones(shapes.shape[1], bool)
    banned[list(allowed)] = False
    shapes[:, banned] = -np.inf
    score = shapes[0].copy()
    back = np.zeros(shapes.shape, dtype=np.int64)
    for i in range(1, len(shapes)):
        candidates = score[:, None] - switch_weight * ROW_SWITCH
        back[i] = candidates.argmax(0)
        score = candidates.max(0) + shapes[i]
    path = np.zeros(len(shapes), dtype=np.int64)
    path[-1] = score.argmax()
    for i in range(len(shapes) - 1, 0, -1):
        path[i - 1] = back[i, path[i]]
    return path
