"""6-fret (Guitar Hero Live) shapes from the transcription.

Lanes 0-5 are W1 W2 W3 B1 B2 B3 and 7 is open, in the same (tick, lane,
sustain) tuples the 5-fret chart uses, so every pass that only reads ticks
and lane identity works on either part.

Two models trained on the official FreeStyleGames charts (tools/train_ghl.py):
ShapeNet picks the Expert shape at each position from the pitches heard
around it, ReduceNet derives Hard, Medium and Easy from the Expert shapes.
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
N_FEATURES = 3 * ROWS * STATS + SCALARS
WINDOW = 256


def shape_class(lanes) -> int:
    """Bitmask over the six buttons; 0 is an open note."""
    return sum(1 << lane for lane in lanes if lane != OPEN)


def class_lanes(shape: int) -> list[int]:
    return [b for b in range(BUTTONS) if shape >> b & 1] or [OPEN]


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


def features(times: np.ndarray, beat_phase: np.ndarray, sustain_s: np.ndarray,
             mix_events, stem_events: dict) -> np.ndarray:
    """[N, N_FEATURES] per chart position: pitch statistics of each stream's
    onsets and of the held mix and guitar notes, their change from the
    previous position and into the next, and timing scalars. Pitches are
    relative to the song's centre, so the key does not matter."""
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
    return np.hstack([stats, stats - before, after - stats, scalars]).astype(np.float32)


class _Encoder(nn.Module):
    def __init__(self, n_in: int, width: int, layers: int, dropout: float):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(n_in, width), nn.GELU(), nn.LayerNorm(width),
                                 nn.Dropout(dropout))
        self.pos = nn.Embedding(WINDOW, width)
        layer = nn.TransformerEncoderLayer(width, 4, 4 * width, dropout=dropout,
                                           batch_first=True, norm_first=True)
        self.body = nn.TransformerEncoder(layer, layers)

    def forward(self, x, pad):
        index = torch.arange(x.shape[1], device=x.device)
        return self.body(self.inp(x) + self.pos(index), src_key_padding_mask=pad)


class ShapeNet(nn.Module):
    """Per position: logits over the 64 button bitmasks."""

    def __init__(self, width: int = 128, layers: int = 4, dropout: float = 0.2):
        super().__init__()
        self.encoder = _Encoder(N_FEATURES, width, layers, dropout)
        self.head = nn.Linear(width, 1 << BUTTONS)

    def forward(self, x, pad):
        return self.head(self.encoder(x, pad))


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
        shape_net = ShapeNet(**saved["config"]["shape"])
        reduce_net = ReduceNet(**saved["config"]["reduce"])
        shape_net.load_state_dict({k: v.float() for k, v in saved["shape"].items()})
        reduce_net.load_state_dict({k: v.float() for k, v in saved["reduce"].items()})
        _MODELS[device] = (shape_net.to(device).eval(), reduce_net.to(device).eval(), saved)
    return _MODELS[device]


def _masked(logits: np.ndarray, allowed) -> np.ndarray:
    """Argmax over the shapes the official charts use on this tier."""
    mask = np.full(logits.shape[-1], -np.inf, dtype=np.float32)
    mask[list(allowed)] = 0
    return (logits + mask).argmax(-1)


def chart_tiers(expert, tempo, mix_events, stem_events) -> dict[str, list]:
    """{"ExpertGHLGuitar": notes, "HardGHLGuitar": ...} on the 5-fret Expert's
    positions and sustains, or {} when no trained model ships."""
    if not MODEL_PATH.is_file() or not expert:
        return {}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    shape_net, reduce_net, saved = _models(device)
    allowed = saved["allowed"]
    res = tempo.resolution
    sustain_of: dict[int, int] = {}
    for tick, _, sus in expert:
        sustain_of[tick] = max(sustain_of.get(tick, 0), sus)
    ticks = np.array(sorted(sustain_of))
    times = np.array([tempo.beat_to_time(t / res) for t in ticks])
    sustain_s = np.array([tempo.beat_to_time((t + sustain_of[t]) / res) for t in ticks]) - times
    x = features(times, (ticks % res) / res, sustain_s, mix_events, stem_events)
    shapes = viterbi(windowed(shape_net, x, device), saved["transitions"].cpu().numpy(), allowed["Expert"],
                     saved["transition_weight"])
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


def viterbi(logits: np.ndarray, transitions: np.ndarray, allowed, weight: float) -> np.ndarray:
    """The shape sequence that maximises the model's per-position log
    probabilities plus weight x the official charts' shape-to-shape log
    probabilities, over the allowed shapes only."""
    logp = logits - np.logaddexp.reduce(logits, axis=1, keepdims=True)
    banned = np.ones(logits.shape[1], bool)
    banned[list(allowed)] = False
    logp[:, banned] = -np.inf
    step = weight * transitions
    score = logp[0].copy()
    back = np.zeros(logp.shape, dtype=np.int64)
    for i in range(1, len(logp)):
        candidates = score[:, None] + step
        back[i] = candidates.argmax(0)
        score = candidates.max(0) + logp[i]
    path = np.zeros(len(logp), dtype=np.int64)
    path[-1] = score.argmax()
    for i in range(len(logp) - 1, 0, -1):
        path[i - 1] = back[i, path[i]]
    return path
