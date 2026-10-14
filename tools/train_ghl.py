"""Train the GHL shape and reduce models on the official charts.

Input is the per-song records from tools/build_ghl_dataset.py, plus the
song folders for the audio delay in each song.ini. Songs are
split by artist and title, so a song that exists both on disc and on GHTV
never sits on both sides of the split. Writes chartgen/ghl_model.pt.

Losses and validation metrics stream to TensorBoard under --runs; watch
them with `tensorboard --logdir data/ghl-runs --host 127.0.0.1`.

    python -m tools.train_ghl data/ghl-features --charts data/ghl-official
"""
import argparse
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

from chartgen import ghl

# The song.ini delay is where the chart meets the audio; transcribed onsets
# only refine it, since on periodic music their mode can sit a beat away.
OFFSET_REFINE_S, OFFSET_BIN_S = 0.05, 0.005
HOLDOUT_SHARE = 0.1
VALIDATION_SHARE = 0.05  # of the training songs, for early stopping
VALIDATE_EVERY = 250
# Training crops are transposed by up to this many semitones. Only the
# absolute pitch rows move; the song-centred rows already ignore the key.
TRANSPOSE = 3
ABSOLUTE_ROWS = (len(ghl.STREAMS) + 2) * ghl.PITCHES
BUTTON_COUNT = torch.tensor([shape.bit_count() for shape in range(1 << ghl.BUTTONS)])
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


def crops(songs, rng, batch, transpose=0):
    picks = [songs[rng.randrange(len(songs))] for _ in range(batch)]
    out = []
    for song in picks:
        begin = rng.randrange(max(1, len(song["shapes"]) - ghl.WINDOW + 1))
        item = {k: v[begin:begin + ghl.WINDOW] for k, v in song.items() if k != "offset"}
        semitones = rng.randint(-transpose, transpose)
        if semitones:
            x = item["x"].astype(np.float32)
            rows = x[:, :ABSOLUTE_ROWS].reshape(len(x), -1, ghl.PITCHES)
            x[:, :ABSOLUTE_ROWS] = np.roll(rows, semitones, axis=2).reshape(len(x), -1)
            x[:, -1] += semitones / 12.0  # the song-centre scalar
            item["x"] = x
        out.append(item)
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
            loss, _ = loss_of(model, crops(songs, rng, batch, TRANSPOSE))
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 25 == 0:
            board.add_scalar(f"{tag}/loss/train", loss.item(), step)
            board.add_scalar(f"{tag}/learning_rate", sched.get_last_lr()[0], step)
        if step % VALIDATE_EVERY == 0:
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


def shape_loss(device):
    ce = nn.CrossEntropyLoss(ignore_index=-1)

    def loss_of(model, items):
        x, pad = stack(items, "x", np.float32)
        y, _ = stack(items, "shapes", np.int64)
        y[pad] = -1
        logits = model(x.to(device), pad.to(device)).float()
        y = y.to(device)
        real = y >= 0
        guess = logits.argmax(-1)
        buttons = BUTTON_COUNT.to(device)
        metrics = {"exact_shape": (guess == y)[real].float().mean().item(),
                   "chord_size": (buttons[guess] == buttons[y.clamp(min=0)])[real].float().mean().item()}
        return ce(logits.flatten(0, 1), y.flatten()), metrics
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


def allowed_shapes(songs) -> dict[str, list[int]]:
    expert = np.concatenate([s["shapes"] for s in songs])
    out = {"Expert": expert}
    for k, tier in enumerate(ghl.TIERS):
        out[tier] = np.concatenate([s["tier_shapes"][s["keep"][:, k] > 0, k] for s in songs])
    return {tier: [int(v) for v, n in zip(*np.unique(shapes, return_counts=True), strict=True)
                   if n >= MIN_SHAPE_USES] for tier, shapes in out.items()}


def evaluate(shape_net, reduce_net, songs, device):
    shape_net.eval()
    reduce_net.eval()
    exact = size = total = 0
    tier_stats = np.zeros((len(ghl.TIERS), 4))  # tp, fp, fn, same shape among tp
    for song in songs:
        x = song["x"].astype(np.float32)
        pred = ghl.windowed(shape_net, x, device).argmax(-1)
        exact += (pred == song["shapes"]).sum()
        size += (np.bitwise_count(pred) == np.bitwise_count(song["shapes"])).sum()
        total += len(pred)
        out = ghl.windowed(reduce_net, reduce_x(song, song["shapes"]), device)
        kept = out[..., 0] > 0
        tier_pred = out[..., 1:].argmax(-1)
        truth = song["keep"] > 0
        tier_stats[:, 0] += (kept & truth).sum(0)
        tier_stats[:, 1] += (kept & ~truth).sum(0)
        tier_stats[:, 2] += (~kept & truth).sum(0)
        tier_stats[:, 3] += ((tier_pred == song["tier_shapes"]) & kept & truth).sum(0)
    print(f"Expert shape: exact {exact / total:.1%}, chord size {size / total:.1%}")
    for k, tier in enumerate(ghl.TIERS):
        tp, fp, fn, same = tier_stats[k]
        print(f"{tier}: keep precision {tp / (tp + fp):.1%} recall {tp / (tp + fn):.1%}, "
              f"shape {same / tp:.1%} (from true Expert shapes)")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("features", type=Path)
    ap.add_argument("--charts", type=Path, default=Path("data/ghl-official"))
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("-o", "--out", type=Path, default=ghl.MODEL_PATH)
    ap.add_argument("--runs", type=Path, default=Path("data/ghl-runs"))
    args = ap.parse_args(argv)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    records = {p.stem: pickle.loads(p.read_bytes()) for p in sorted(args.features.glob("*.pkl"))}
    keys = sorted({song_key(name) for name in records})
    holdout = set(random.Random(0).sample(keys, max(1, int(len(keys) * HOLDOUT_SHARE))))
    folders = {p.parent.name: p.parent for p in args.charts.rglob("notes.mid")}
    songs = {name: song_arrays(record, ini_delay_s(folders[name])) for name, record in records.items()}
    offsets = np.array([s["offset"] for s in songs.values()])
    print(f"{len(songs)} songs ({len(keys)} distinct), offsets median {np.median(offsets):.2f}s "
          f"range {offsets.min():.2f}..{offsets.max():.2f}s", flush=True)
    rest = sorted(set(keys) - holdout)
    validation = set(random.Random(1).sample(rest, max(1, int(len(rest) * VALIDATION_SHARE))))
    train_songs = [s for n, s in songs.items() if song_key(n) not in holdout | validation]
    val_songs = [s for n, s in songs.items() if song_key(n) in validation]
    test_songs = [s for n, s in songs.items() if song_key(n) in holdout]
    print(f"train {len(train_songs)}, validation {len(val_songs)}, holdout {len(test_songs)}",
          flush=True)

    board = SummaryWriter(args.runs / time.strftime("%Y-%m-%d_%H-%M-%S"))
    board.add_text("data", f"{len(train_songs)} train, {len(val_songs)} validation, "
                           f"{len(test_songs)} holdout songs")
    shape_net, reduce_net = ghl.ShapeNet(), ghl.ReduceNet()
    print("training ShapeNet", flush=True)
    train(shape_net, train_songs, val_songs, args.steps, args.batch, args.lr, device,
          shape_loss(device), board, "shape")
    print("training ReduceNet", flush=True)
    train(reduce_net, train_songs, val_songs, args.steps // 2, args.batch, args.lr, device,
          reduce_loss(device), board, "reduce")
    evaluate(shape_net, reduce_net, test_songs, device)
    torch.save({"shape": {k: v.half() for k, v in shape_net.state_dict().items()},
                "reduce": {k: v.half() for k, v in reduce_net.state_dict().items()},
                "allowed": allowed_shapes(train_songs), "holdout": sorted(holdout)}, args.out)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
