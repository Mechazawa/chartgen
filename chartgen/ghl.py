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
from pathlib import Path

import numpy as np
import torch
from torch import nn

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
MERGE_S = 0.03  # onsets closer than this are one note-time candidate
GAP_S = 0.06  # the closest two official Expert notes ever sit
GRID = 4  # notes snap to 16ths
N_TIMING = N_FEATURES - 2 + 5


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


def candidates(mix_events, stem_events: dict) -> tuple[np.ndarray, np.ndarray]:
    """Note-time candidates: the onsets of the mix and every stem merged
    into MERGE_S clusters, as (first onset, events in the cluster)."""
    onsets = np.sort([e[0] for events in [mix_events, *stem_events.values()] for e in events])
    if not len(onsets):
        return onsets, onsets
    first = [0]
    for i in range(1, len(onsets)):
        if onsets[i] - onsets[first[-1]] > MERGE_S:
            first.append(i)
    first = np.array(first)
    return onsets[first], np.diff(first, append=len(onsets)).astype(np.float32)


def timing_features(times: np.ndarray, counts: np.ndarray, tempo, mix_events,
                    stem_events: dict) -> np.ndarray:
    """[N, N_TIMING] per candidate: the pitch statistics of features(), the
    gaps to the neighbouring candidates, how many onsets it merges, and
    where it sits on the beat grid."""
    beat = np.array([tempo.time_to_beat(t) for t in times])
    x = features(times, beat % 1, np.zeros_like(times), mix_events, stem_events)
    x = np.delete(x, [N_FEATURES - 4, N_FEATURES - 1], axis=1)  # sustain and key shift
    grid = np.stack([np.log1p(counts), np.abs(beat * 4 - np.round(beat * 4)),
                     np.abs(beat * 3 - np.round(beat * 3)), np.sin(np.pi * beat), np.cos(np.pi * beat)], 1)
    return np.hstack([x, grid]).astype(np.float32)


def pick_times(logits: np.ndarray, times: np.ndarray, threshold: float) -> np.ndarray:
    """The candidates above threshold, most likely first, skipping any
    within GAP_S of one already picked."""
    taken = np.zeros(len(times), bool)
    picked = []
    for i in np.argsort(-logits):
        if logits[i] <= threshold:
            break
        if not taken[i]:
            picked.append(times[i])
            low, high = np.searchsorted(times, [times[i] - GAP_S, times[i] + GAP_S])
            taken[low:high] = True
    return np.sort(picked)


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
    P(shape | its factors), then the relation logits."""

    def __init__(self, width: int = 128, layers: int = 4, dropout: float = 0.2):
        super().__init__()
        self.encoder = _Encoder(N_FEATURES, width, layers, dropout)
        self.head = nn.Linear(width, sum(FACTOR_SIZES) + RELATIONS)
        self.register_buffer("given_factors", torch.zeros(1 << BUTTONS))
        self.register_buffer("factor_weights", torch.ones(len(FACTOR_SIZES)))
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
        return torch.cat([score, relations], -1)


class ReduceNet(nn.Module):
    """Per position and lower tier: keep logit plus 64 shape logits, from
    the Expert shape and the timing scalars."""

    def __init__(self, width: int = 128, layers: int = 3, dropout: float = 0.2):
        super().__init__()
        self.encoder = _Encoder((1 << BUTTONS) + SCALARS, width, layers, dropout)
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


def reduce_inputs(shapes: np.ndarray, scalars: np.ndarray) -> np.ndarray:
    return np.concatenate([np.eye(1 << BUTTONS, dtype=np.float32)[shapes], scalars], axis=1)


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


def _masked(logits: np.ndarray, allowed) -> np.ndarray:
    """Argmax over the shapes the official charts use on this tier."""
    mask = np.full(logits.shape[-1], -np.inf, dtype=np.float32)
    mask[list(allowed)] = 0
    return (logits + mask).argmax(-1)


def chart_tiers(expert, tempo, mix_events, stem_events) -> dict[str, list]:
    """{"ExpertGHLGuitar": notes, "HardGHLGuitar": ...}, or {} when no
    trained model ships. Sustains come from the 5-fret Expert note on the
    same 16th, cut short of the next 6-fret note."""
    candidate_times, counts = candidates(mix_events, stem_events)
    if not MODEL_PATH.is_file() or not len(candidate_times):
        return {}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    timing_net, shape_net, reduce_net, saved = _models(device)
    allowed = saved["allowed"]
    res = tempo.resolution
    x = timing_features(candidate_times, counts, tempo, mix_events, stem_events)
    picked = pick_times(windowed(timing_net, x, device)[:, 0], candidate_times, saved["timing_threshold"])
    ticks = np.array(sorted({tempo.quantize(t, GRID) for t in picked}))
    if not len(ticks):
        return {}
    five_fret: dict[int, int] = {}
    for tick, _, sus in expert:
        snapped = round(tick * GRID / res) * res // GRID
        five_fret[snapped] = max(five_fret.get(snapped, 0), sus)
    room = np.diff(ticks, append=ticks[-1] + 4 * res) - res // GRID
    sustain_of = {int(t): sus if (sus := min(five_fret.get(int(t), 0), int(r))) >= res // GRID else 0
                  for t, r in zip(ticks, room, strict=True)}
    times = np.array([tempo.beat_to_time(t / res) for t in ticks])
    sustain_s = np.array([tempo.beat_to_time((t + sustain_of[t]) / res) for t in ticks]) - times
    x = features(times, (ticks % res) / res, sustain_s, mix_events, stem_events)
    shapes = viterbi(windowed(shape_net, x, device), saved["transitions"].cpu().numpy(),
                     allowed["Expert"], *saved["decode_weights"])
    reduced = windowed(reduce_net, reduce_inputs(shapes, x[:, -SCALARS:]), device)

    def notes(keep, tier_shapes):
        return [(int(t), lane, sustain_of[t]) for t, k, s in zip(ticks, keep, tier_shapes, strict=True)
                if k for lane in class_lanes(int(s))]

    out = {"ExpertGHLGuitar": notes(np.ones(len(ticks), bool), shapes)}
    for k, tier in enumerate(TIERS):
        # Each tier keeps the share of Expert positions the official charts
        # keep, taking the positions the model most wants to keep.
        keep = np.zeros(len(ticks), bool)
        keep[np.argsort(-reduced[:, k, 0])[:round(saved["keep_share"][tier] * len(ticks))]] = True
        out[f"{tier}GHLGuitar"] = notes(keep, _masked(reduced[:, k, 1:], allowed[tier]))
    return out


def _log_softmax(logits: np.ndarray) -> np.ndarray:
    return logits - np.logaddexp.reduce(logits, axis=-1, keepdims=True)


def viterbi(logits: np.ndarray, transitions: np.ndarray, allowed,
            transition_weight: float, relation_weight: float) -> np.ndarray:
    """The allowed shape sequence with the best total of: each position's
    shape log probability, transition_weight x the official charts'
    shape-to-shape log probability, and relation_weight x the model's log
    probability of how each shape relates to the one before it."""
    shapes = _log_softmax(logits[:, :1 << BUTTONS])
    relations = _log_softmax(logits[:, 1 << BUTTONS:])
    banned = np.ones(shapes.shape[1], bool)
    banned[list(allowed)] = False
    shapes[:, banned] = -np.inf
    score = shapes[0].copy()
    back = np.zeros(shapes.shape, dtype=np.int64)
    for i in range(1, len(shapes)):
        candidates = (score[:, None] + transition_weight * transitions
                      + relation_weight * relations[i][RELATION_OF])
        back[i] = candidates.argmax(0)
        score = candidates.max(0) + shapes[i]
    path = np.zeros(len(shapes), dtype=np.int64)
    path[-1] = score.argmax()
    for i in range(len(shapes) - 1, 0, -1):
        path[i - 1] = back[i, path[i]]
    return path
