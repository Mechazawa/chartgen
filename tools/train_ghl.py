"""Train the GHL timing, shape and reduce models on the official charts.

Input is the per-song records from tools/build_ghl_dataset.py, plus the
song folders for the audio delay in each song.ini. Songs are
split by artist and title, so a song that exists both on disc and on GHTV
never sits on both sides of the split. Writes chartgen/ghl_model.pt.

Losses and validation metrics stream to TensorBoard under --runs; watch
them with `tensorboard --logdir data/ghl-runs --host 127.0.0.1`.

--sweep N trains N random configurations of each network (size, depth,
dropout, learning rate) and keeps the one with the lowest validation loss.

    python -m tools.train_ghl data/ghl-features --charts data/ghl-official
"""
import argparse
import json
import pickle
import random
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from chartgen import ghl, tempo

# The song.ini delay is where the chart meets the audio; transcribed onsets
# only refine it, since on periodic music their mode can sit a beat away.
OFFSET_REFINE_S, OFFSET_BIN_S = 0.05, 0.005
MATCH_S = 0.05  # a picked note this close to an official one is a hit
HOLDOUT_SHARE = 0.1
VALIDATION_SHARE = 0.05  # of the training songs, for early stopping
VALIDATE_EVERY = 250
# Larger weights push the share of repeated shapes far past the official
# charts' (about 38%) without improving agreement.
DECODE_WEIGHTS = [(transition, relation) for transition in (0.0, 0.05, 0.1, 0.2)
                  for relation in (0.0, 0.1, 0.2, 0.3)]
SWEEP = {"width": (64, 128, 256), "layers": (2, 4, 6), "dropout": (0.1, 0.2, 0.3),
         "lr": (1e-4, 3e-4, 1e-3)}
DEFAULT = {"width": 128, "layers": 4, "dropout": 0.2, "lr": 3e-4}
BUTTON_COUNT = torch.tensor([shape.bit_count() for shape in range(1 << ghl.BUTTONS)])
RELATION_OF = torch.from_numpy(ghl.RELATION_OF)
# A shape the official charts use fewer times than this on a tier is
# never emitted there (Easy, for one, is white-row only).
MIN_SHAPE_USES = 20


def song_key(name: str) -> str:
    base = re.sub(r"\s*\(FreeStyleGames.*\)$", "", name)
    return re.sub(r"[^a-z0-9]", "", base.lower())


def ini_delay_s(folder: Path) -> float:
    found = re.search(r"^delay\s*=\s*(-?\d+)", (folder / "song.ini").read_text(errors="ignore"), re.M)
    return int(found.group(1)) / 1000 if found else 0.0


def audio_offset(note_s: np.ndarray, events, delay_s: float) -> float:
    """Seconds to add to chart time to land on the audio: the mode of the
    note-to-onset differences within OFFSET_REFINE_S of the delay."""
    onsets = np.array([e[0] for e in events])
    diffs = (onsets[None, :] - note_s[:, None] - delay_s).ravel()
    diffs = diffs[np.abs(diffs) <= OFFSET_REFINE_S]
    if not len(diffs):
        return delay_s
    values, counts = np.unique(np.round(diffs / OFFSET_BIN_S).astype(int), return_counts=True)
    return delay_s + float(values[counts.argmax()] * OFFSET_BIN_S)


def song_arrays(record: dict, delay_s: float) -> dict:
    chart = record["chart"]
    expert = chart.tiers["Expert"]
    ticks = np.array([t for t, _, _ in expert])
    note_s = np.array([chart.seconds(t) for t in ticks])
    offset = audio_offset(note_s, record["stems"]["guitar"] + record["mix"], delay_s)
    times = note_s + offset
    sustain_s = np.array([chart.seconds(t + sus) - chart.seconds(t) for t, _, sus in expert])
    phase = (ticks % chart.resolution) / chart.resolution
    x = ghl.features(times, phase, sustain_s, record["mix"], record["stems"])
    shapes = np.array([ghl.shape_class(s) for _, s, _ in expert])
    keep = np.zeros((len(ticks), len(ghl.TIERS)), dtype=np.float32)
    tier_shapes = np.zeros((len(ticks), len(ghl.TIERS)), dtype=np.int64)
    index = {t: i for i, t in enumerate(ticks)}
    for k, tier in enumerate(ghl.TIERS):
        for t, s, _ in chart.tiers[tier]:
            if t in index:
                keep[index[t], k] = 1
                tier_shapes[index[t], k] = ghl.shape_class(s)
    return {"x": x.astype(np.float16), "shapes": shapes, "keep": keep,
            "tier_shapes": tier_shapes, "offset": offset}


def timing_arrays(record: dict, delay_s: float) -> tuple[dict, tuple]:
    """({"x", "label"} per note-time candidate, (candidate times, official
    note times, beat grid)). The label marks the candidate nearest each
    official Expert note, when within MATCH_S."""
    beats, pickup = record["tempo"]
    grid = tempo.TempoMap(beat_times=np.asarray(beats), resolution=tempo.RESOLUTION, pickup_beats=int(pickup))
    times, counts = ghl.candidates(record["mix"], record["stems"])
    chart = record["chart"]
    official = np.array([chart.seconds(t) for t in sorted({t for t, _, _ in chart.tiers["Expert"]})]) + delay_s
    k = np.clip(np.searchsorted(times, official), 1, len(times) - 1)
    nearest = np.where(np.abs(times[k] - official) < np.abs(times[k - 1] - official), k, k - 1)
    label = np.zeros(len(times), dtype=np.float32)
    label[nearest[np.abs(times[nearest] - official) <= MATCH_S]] = 1
    x = ghl.timing_features(times, counts, grid, record["mix"], record["stems"])
    return {"x": x.astype(np.float16), "label": label}, (times, official, grid)


def crops(songs, rng, batch):
    picks = [songs[rng.randrange(len(songs))] for _ in range(batch)]
    out = []
    for song in picks:
        begin = rng.randrange(max(1, len(song["x"]) - ghl.WINDOW + 1))
        out.append({k: v[begin:begin + ghl.WINDOW] for k, v in song.items() if k != "offset"})
    return out


def stack(items, key, dtype):
    width = max(len(item[key]) for item in items)
    shape = (len(items), width) + items[0][key].shape[1:]
    out = np.zeros(shape, dtype=dtype)
    pad = np.ones((len(items), width), dtype=bool)
    for i, item in enumerate(items):
        out[i, :len(item[key])] = item[key]
        pad[i, :len(item[key])] = False
    return torch.from_numpy(out), torch.from_numpy(pad)


def reduce_x(item, shapes):
    return ghl.reduce_inputs(shapes, item["x"][:, -ghl.SCALARS:].astype(np.float32))


def train(model, songs, val_songs, steps, batch, lr, device, loss_of, board, tag):
    """Train, then keep the weights with the lowest validation loss. loss_of
    returns (loss, {metric: value}); both go to the board under tag."""
    rng = random.Random(0)
    val_batches = [crops(val_songs, random.Random(i), batch) for i in range(8)]
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps, pct_start=0.05)
    best, best_state = float("inf"), None
    for step in range(1, steps + 1):
        model.train()
        with torch.autocast(device, dtype=torch.bfloat16):
            loss, _ = loss_of(model, crops(songs, rng, batch))
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 25 == 0:
            board.add_scalar(f"{tag}/loss/train", loss.item(), step)
            board.add_scalar(f"{tag}/learning_rate", sched.get_last_lr()[0], step)
        if step % VALIDATE_EVERY == 0 or step == steps:
            model.eval()
            with torch.no_grad(), torch.autocast(device, dtype=torch.bfloat16):
                results = [loss_of(model, items) for items in val_batches]
            val = float(np.mean([loss.item() for loss, _ in results]))
            board.add_scalar(f"{tag}/loss/validation", val, step)
            for name in results[0][1]:
                board.add_scalar(f"{tag}/validation/{name}",
                                 float(np.mean([metrics[name] for _, metrics in results])), step)
            board.flush()
            if val < best:
                best, best_state = val, {k: v.detach().clone() for k, v in model.state_dict().items()}
            print(f"  step {step}/{steps} train {loss.item():.3f} val {val:.3f}", flush=True)
    model.load_state_dict(best_state)
    print(f"  kept the weights at val loss {best:.3f}", flush=True)
    return best


def fit(net, configs, train_songs, val_songs, steps, batch, device, loss_of, runs):
    """Train one net per configuration; the lowest validation loss wins.
    Returns (model, its configuration without the learning rate)."""
    best = None
    for config in configs:
        name = "-".join(f"{k}{v}" for k, v in config.items())
        print(f"training {net.__name__} {name}", flush=True)
        model = net(**{k: v for k, v in config.items() if k != "lr"})
        board = SummaryWriter(runs / name)
        loss = train(model, train_songs, val_songs, steps, batch, config["lr"], device,
                     loss_of, board, net.__name__)
        board.add_hparams(config, {"best_validation_loss": loss})
        board.close()
        if best is None or loss < best[0]:
            best = (loss, model, config)
    print(f"{net.__name__}: kept {best[2]} at validation loss {best[0]:.3f}", flush=True)
    return best[1], {k: v for k, v in best[2].items() if k != "lr"}


def timing_loss(device):
    bce = nn.BCEWithLogitsLoss(reduction="none")

    def loss_of(model, items):
        x, pad = stack(items, "x", np.float32)
        y, _ = stack(items, "label", np.float32)
        out = model(x.to(device), pad.to(device)).float()[..., 0]
        real, y = ~pad.to(device), y.to(device)
        guess, truth = out[real] > 0, y[real] > 0
        metrics = {"precision": (guess & truth).sum().item() / max(1, guess.sum().item()),
                   "recall": (guess & truth).sum().item() / max(1, truth.sum().item())}
        return bce(out, y)[real].mean(), metrics
    return loss_of


def snapped_notes(logits, times, grid, threshold) -> np.ndarray:
    """Seconds of the picked notes after the 16th snap chart_tiers applies."""
    ticks = sorted({grid.quantize(t, ghl.GRID) for t in ghl.pick_times(logits, times, threshold)})
    return np.array([grid.beat_to_time(t / grid.resolution) for t in ticks])


def timing_threshold(timing_net, songs, extras, device) -> float:
    """The keep threshold at which the validation songs get as many picked
    notes as the official charts have."""
    logits = [ghl.windowed(timing_net, s["x"].astype(np.float32), device)[:, 0] for s in songs]
    target = sum(len(official) for _, official, _ in extras)
    low, high = -8.0, 8.0
    for _ in range(20):
        middle = (low + high) / 2
        count = sum(len(snapped_notes(lg, times, grid, middle))
                    for lg, (times, _, grid) in zip(logits, extras, strict=True))
        low, high = (middle, high) if count > target else (low, middle)
    return (low + high) / 2


def near(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per time in a: whether b has a time within MATCH_S."""
    if not len(b):
        return np.zeros(len(a), bool)
    k = np.clip(np.searchsorted(b, a), 1, max(1, len(b) - 1))
    return np.minimum(np.abs(b[k] - a), np.abs(b[k - 1] - a)) <= MATCH_S


def evaluate_timing(timing_net, songs, extras, device, threshold, board) -> None:
    hits = np.zeros(4)  # picked, picked near official, official, official near picked
    for song, (times, official, grid) in zip(songs, extras, strict=True):
        notes = snapped_notes(ghl.windowed(timing_net, song["x"].astype(np.float32), device)[:, 0],
                              times, grid, threshold)
        hits += [len(notes), near(notes, official).sum(), len(official), near(official, notes).sum()]
    scores = {"timing_count": hits[0] / hits[2], "timing_precision": hits[1] / max(hits[0], 1),
              "timing_recall": hits[3] / max(hits[2], 1)}
    for name, value in scores.items():
        print(f"holdout {name}: {value:.1%}")
        board.add_scalar(f"holdout/{name}", value)


def shape_loss(device):
    ce = nn.CrossEntropyLoss(ignore_index=-1)

    def loss_of(model, items):
        x, pad = stack(items, "x", np.float32)
        y, _ = stack(items, "shapes", np.int64)
        y[pad] = -1
        out = model(x.to(device), pad.to(device)).float()
        logits, relation_logits = out[..., :1 << ghl.BUTTONS], out[..., 1 << ghl.BUTTONS:]
        y = y.to(device)
        real = y >= 0
        relation = torch.full_like(y, -1)
        both = real[:, 1:] & real[:, :-1]
        relation[:, 1:][both] = RELATION_OF.to(device)[y[:, :-1][both], y[:, 1:][both]]
        guess = logits.argmax(-1)
        buttons = BUTTON_COUNT.to(device)
        metrics = {"exact_shape": (guess == y)[real].float().mean().item(),
                   "chord_size": (buttons[guess] == buttons[y.clamp(min=0)])[real].float().mean().item(),
                   "relation": (relation_logits.argmax(-1) == relation)[relation >= 0].float().mean().item()}
        loss = ce(logits.flatten(0, 1), y.flatten()) + ce(relation_logits.flatten(0, 1), relation.flatten())
        return loss, metrics
    return loss_of


def reduce_loss(device):
    ce = nn.CrossEntropyLoss(ignore_index=-1)
    bce = nn.BCEWithLogitsLoss(reduction="none")

    def loss_of(model, items):
        x, pad = stack([{"x": reduce_x(i, i["shapes"])} for i in items], "x", np.float32)
        keep, _ = stack(items, "keep", np.float32)
        target, _ = stack(items, "tier_shapes", np.int64)
        target[(keep == 0) | pad[..., None]] = -1
        out = model(x.to(device), pad.to(device)).float()
        keep, target, real = keep.to(device), target.to(device), ~pad.to(device)
        keep_loss = bce(out[..., 0], keep)[real].mean()
        metrics = {}
        for k, tier in enumerate(ghl.TIERS):
            guess, truth = out[..., k, 0][real] > 0, keep[..., k][real] > 0
            metrics[f"{tier}_keep_precision"] = (guess & truth).sum().item() / max(1, guess.sum().item())
            metrics[f"{tier}_keep_recall"] = (guess & truth).sum().item() / max(1, truth.sum().item())
            kept = target[..., k] >= 0
            metrics[f"{tier}_shape"] = (out[..., k, 1:].argmax(-1) == target[..., k])[kept].float().mean().item()
        return keep_loss + ce(out[..., 1:].flatten(0, 2), target.flatten()), metrics
    return loss_of


def transitions(songs) -> np.ndarray:
    """[64, 64] log P(next shape | shape) over the songs' Expert charts,
    add-one smoothed."""
    counts = np.ones((1 << ghl.BUTTONS, 1 << ghl.BUTTONS))
    for song in songs:
        np.add.at(counts, (song["shapes"][:-1], song["shapes"][1:]), 1)
    return np.log(counts / counts.sum(1, keepdims=True))


def structure_score(pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """[exact, size, row, move hits, fretted pairs] of a predicted sequence."""
    ours, theirs = structure(pred), structure(truth)
    both = ~np.isnan(ours[2]) & ~np.isnan(theirs[2])
    return np.array([(pred == truth).sum(), (ours[0] == theirs[0]).sum(), (ours[1] == theirs[1]).sum(),
                     (ours[2] == theirs[2])[both].sum(), both.sum()])


def repeat_rate(sequences) -> float:
    return float(np.mean(np.concatenate([s[1:] == s[:-1] for s in sequences])))


def pick_decoding(shape_net, songs, device, allowed, moves) -> tuple[float, float]:
    """The (transition, relation) weights with the best mean of size, row
    and move agreement on the validation songs, less how far the share of
    repeated shapes strays from the official charts'. Agreement alone
    rewards a chart that never moves."""
    logits = [ghl.windowed(shape_net, s["x"].astype(np.float32), device) for s in songs]
    total = sum(len(s["shapes"]) for s in songs)
    official = repeat_rate([s["shapes"] for s in songs])
    best = (-np.inf, DECODE_WEIGHTS[0])
    for weights in DECODE_WEIGHTS:
        preds = [ghl.viterbi(lg, moves, allowed, *weights) for lg in logits]
        hits = sum(structure_score(pred, s["shapes"]) for pred, s in zip(preds, songs, strict=True))
        agreement = (hits[1] / total + hits[2] / total + hits[3] / max(hits[4], 1)) / 3
        repeats = repeat_rate(preds)
        score = agreement - abs(repeats - official)
        print(f"  decode weights {weights}: agreement {agreement:.1%}, repeats {repeats:.0%} "
              f"(official {official:.0%}), score {score:.3f}", flush=True)
        best = max(best, (score, weights))
    return best[1]


def keep_share(songs) -> dict[str, float]:
    """Per lower tier: the share of Expert positions the official charts keep."""
    keep = np.concatenate([s["keep"] for s in songs])
    return {tier: float(keep[:, k].mean()) for k, tier in enumerate(ghl.TIERS)}


def allowed_shapes(songs) -> dict[str, list[int]]:
    expert = np.concatenate([s["shapes"] for s in songs])
    out = {"Expert": expert}
    for k, tier in enumerate(ghl.TIERS):
        out[tier] = np.concatenate([s["tier_shapes"][s["keep"][:, k] > 0, k] for s in songs])
    return {tier: [int(v) for v, n in zip(*np.unique(shapes, return_counts=True), strict=True)
                   if n >= MIN_SHAPE_USES] for tier, shapes in out.items()}


def structure(shapes: np.ndarray):
    """Per position: buttons held (3 for any triple), row (0 white, 1 black,
    2 both, 3 open) and the column move into the next note (-1, 0, 1, or
    nan next to an open)."""
    lanes = [[b for b in range(ghl.BUTTONS) if s >> b & 1] for s in shapes]
    row = np.array([3 if not ls else 0 if max(ls) < 3 else 1 if min(ls) >= 3 else 2 for ls in lanes])
    column = np.array([np.mean([b % 3 for b in ls]) if ls else np.nan for ls in lanes])
    return np.minimum(np.bitwise_count(shapes), 3), row, np.sign(np.diff(column, append=np.nan))


def evaluate(shape_net, reduce_net, songs, device, allowed, moves, weights, board) -> None:
    shape_net.eval()
    reduce_net.eval()
    hits = np.zeros(5)  # exact, size, row, move, fretted pairs
    total = 0
    tier_stats = np.zeros((len(ghl.TIERS), 4))  # tp, fp, fn, same shape among tp
    preds = []
    for song in songs:
        x = song["x"].astype(np.float32)
        pred = ghl.viterbi(ghl.windowed(shape_net, x, device), moves, allowed["Expert"], *weights)
        preds.append(pred)
        hits += structure_score(pred, song["shapes"])
        total += len(pred)
        out = ghl.windowed(reduce_net, reduce_x(song, song["shapes"]), device)
        kept = out[..., 0] > 0
        tier_pred = out[..., 1:].argmax(-1)
        truth = song["keep"] > 0
        tier_stats[:, 0] += (kept & truth).sum(0)
        tier_stats[:, 1] += (kept & ~truth).sum(0)
        tier_stats[:, 2] += (~kept & truth).sum(0)
        tier_stats[:, 3] += ((tier_pred == song["tier_shapes"]) & kept & truth).sum(0)
    scores = {"expert_exact": hits[0] / total, "expert_size": hits[1] / total,
              "expert_row": hits[2] / total, "expert_move": hits[3] / max(hits[4], 1),
              "expert_repeats": repeat_rate(preds),
              "official_repeats": repeat_rate([s["shapes"] for s in songs])}
    for k, tier in enumerate(ghl.TIERS):
        tp, fp, fn, same = tier_stats[k]
        scores |= {f"{tier}_keep_precision": tp / max(tp + fp, 1),
                   f"{tier}_keep_recall": tp / max(tp + fn, 1), f"{tier}_shape": same / max(tp, 1)}
    for name, value in scores.items():
        print(f"holdout {name}: {value:.1%}")
        board.add_scalar(f"holdout/{name}", value)
    board.flush()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("features", type=Path)
    ap.add_argument("--charts", type=Path, default=Path("data/ghl-official"))
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--sweep", type=int, default=0, metavar="N",
                    help="try N random configurations per network instead of the default")
    ap.add_argument("--config", type=json.loads, help="JSON configuration to train instead of the default")
    ap.add_argument("--train-share", type=float, default=1.0,
                    help="train on this share of the training songs (for a learning curve)")
    ap.add_argument("--seed", type=int, default=3, help="picks the songs --train-share keeps")
    ap.add_argument("-o", "--out", type=Path, default=ghl.MODEL_PATH)
    ap.add_argument("--runs", type=Path, default=Path("data/ghl-runs"))
    args = ap.parse_args(argv)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    records = {p.stem: pickle.loads(p.read_bytes()) for p in sorted(args.features.glob("*.pkl"))}
    keys = sorted({song_key(name) for name in records})
    holdout = set(random.Random(0).sample(keys, max(1, int(len(keys) * HOLDOUT_SHARE))))
    folders = {p.parent.name: p.parent for p in args.charts.rglob("notes.mid")}
    songs = {name: song_arrays(record, ini_delay_s(folders[name])) for name, record in records.items()}
    timing = {name: timing_arrays(record, ini_delay_s(folders[name])) for name, record in records.items()}
    offsets = np.array([s["offset"] for s in songs.values()])
    print(f"{len(songs)} songs ({len(keys)} distinct), offsets median {np.median(offsets):.2f}s "
          f"range {offsets.min():.2f}..{offsets.max():.2f}s", flush=True)
    rest = sorted(set(keys) - holdout)
    validation = set(random.Random(1).sample(rest, max(1, int(len(rest) * VALIDATION_SHARE))))
    train_songs = [s for n, s in songs.items() if song_key(n) not in holdout | validation]
    val_songs = [s for n, s in songs.items() if song_key(n) in validation]
    test_songs = [s for n, s in songs.items() if song_key(n) in holdout]
    train_songs = random.Random(args.seed).sample(train_songs, round(len(train_songs) * args.train_share))
    train_timing = [timing[n][0] for n in songs if song_key(n) not in holdout | validation]
    train_timing = random.Random(args.seed).sample(train_timing, round(len(train_timing) * args.train_share))
    val_timing = [timing[n] for n in songs if song_key(n) in validation]
    test_timing = [timing[n] for n in songs if song_key(n) in holdout]
    print(f"train {len(train_songs)}, validation {len(val_songs)}, holdout {len(test_songs)}",
          flush=True)

    runs = args.runs / time.strftime("%Y-%m-%d_%H-%M-%S")
    rng = random.Random(2)
    configs = [{k: rng.choice(v) for k, v in SWEEP.items()} for _ in range(args.sweep)] or [args.config or DEFAULT]
    timing_net, timing_config = fit(ghl.TimingNet, configs, train_timing, [t for t, _ in val_timing],
                                    args.steps, args.batch, device, timing_loss(device), runs / "timing")
    shape_net, shape_config = fit(ghl.ShapeNet, configs, train_songs, val_songs, args.steps,
                                  args.batch, device, shape_loss(device), runs / "shape")
    reduce_net, reduce_config = fit(ghl.ReduceNet, configs, train_songs, val_songs, args.steps // 2,
                                    args.batch, device, reduce_loss(device), runs / "reduce")
    board = SummaryWriter(runs / "holdout")
    board.add_text("data", f"{len(train_songs)} train, {len(val_songs)} validation, "
                           f"{len(test_songs)} holdout songs; shape {shape_config}, "
                           f"reduce {reduce_config}")
    allowed = allowed_shapes(train_songs)
    moves = transitions(train_songs)
    weights = pick_decoding(shape_net, val_songs, device, allowed["Expert"], moves)
    board.add_text("decode_weights", str(weights))
    threshold = timing_threshold(timing_net, *zip(*val_timing, strict=True), device)
    evaluate_timing(timing_net, *zip(*test_timing, strict=True), device, threshold, board)
    evaluate(shape_net, reduce_net, test_songs, device, allowed, moves, weights, board)
    torch.save({"timing": {k: v.half() for k, v in timing_net.state_dict().items()},
                "timing_threshold": threshold,
                "shape": {k: v.half() for k, v in shape_net.state_dict().items()},
                "reduce": {k: v.half() for k, v in reduce_net.state_dict().items()},
                "allowed": allowed, "transitions": torch.from_numpy(moves), "decode_weights": weights,
                "keep_share": keep_share(train_songs),
                "config": {"timing": timing_config, "shape": shape_config, "reduce": reduce_config},
                "holdout": sorted(holdout)}, args.out)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
