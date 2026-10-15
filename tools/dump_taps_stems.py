"""Dump per-section shares of ALL SIX BS-RoFormer SW stems, with tap truth.

The overnight taps recalibration (tools/dump_foreground.py) ran taps v3
through the Demucs-compatible shim, where SW's guitar, piano and residual
stems are summed back into one "other" before taps sees them. So the
instrument the user switched separators FOR - a piano/synth stem keyed
to what charters tap - was never measured. This dump records each true
stem's share of section energy separately (drums, bass, vocals, guitar,
piano, sw_other - no double counting), plus the centroid of the three
melodic stems, on a seeded subset of the recal songs: songs that contain
a tapped-whole section, and songs with no taps at all as controls.

Section bounds and truth are reused from the recal dump, so the only work
per song is one SW separation (~37s on the RTX 3060 Ti). Incremental and
resumable.

    python tools/dump_taps_stems.py work/foreground_ab.json -o work/taps_stems6.json
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from validate_solos import find_audio  # noqa: E402

TRUE_STEMS = ("drums", "bass", "vocals", "guitar", "piano", "sw_other")
MELODIC = ("guitar", "piano", "sw_other")
POSITIVE_TAP_SHARE = 0.50
NEGATIVE_TAP_SHARE = 0.05
MIN_NOTES = 4


def classify(sections):
    """'pos' if any section is tapped-whole, 'neg' if no section has taps,
    else 'mixed' (partially tapped only - skipped as ambiguous)."""
    scored = [s for s in sections if s["notes"] >= MIN_NOTES]
    if any(s["taps"] / s["notes"] >= POSITIVE_TAP_SHARE for s in scored):
        return "pos"
    if all(s["taps"] / s["notes"] <= NEGATIVE_TAP_SHARE for s in scored):
        return "neg"
    return "mixed"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("recal", type=Path, help="work/foreground_ab.json")
    ap.add_argument("-o", "--out", type=Path, required=True)
    ap.add_argument("--library", type=Path,
                    default=Path(__file__).resolve().parent.parent / "data/calibration")
    ap.add_argument("--positives", type=int, default=200)
    ap.add_argument("--negatives", type=int, default=100)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args(argv)

    import librosa
    import numpy as np

    from chartgen import stems as stemsmod

    recal = json.loads(args.recal.read_text(encoding="utf-8"))
    rng = random.Random(args.seed)
    rng.shuffle(recal)
    pos = [r for r in recal if classify(r["sections"]) == "pos"][:args.positives]
    neg = [r for r in recal if classify(r["sections"]) == "neg"][:args.negatives]
    plan = pos + neg
    rng.shuffle(plan)
    print(f"plan: {len(pos)} tap songs + {len(neg)} no-tap controls", flush=True)

    done = {}
    if args.out.exists():
        for row in json.loads(args.out.read_text(encoding="utf-8")):
            done[row["song"]] = row
    rows = list(done.values())
    started = time.time()
    hop = int(0.05 * stemsmod.SR)
    for r in plan:
        if r["song"] in done:
            continue
        folder = args.library / r["song"]
        audio = find_audio(folder) if folder.is_dir() else None
        if audio is None:
            print(f"  skip {r['song'][:44]}: no audio", flush=True)
            continue
        stemsmod._CACHE.clear()
        try:
            mono = stemsmod.separate(str(audio), backend="sw")
        except Exception as error:
            print(f"  skip {r['song'][:44]}: {type(error).__name__}", flush=True)
            continue
        if mono is None or any(k not in mono for k in TRUE_STEMS):
            print(f"  skip {r['song'][:44]}: stems missing", flush=True)
            continue
        envs = {}
        for name in TRUE_STEMS:
            stem = mono[name]
            n = len(stem) // hop
            envs[name] = np.sqrt((stem[:n * hop].reshape(n, hop) ** 2).mean(axis=1))
        frames = min(len(e) for e in envs.values())
        cens = {}
        for name in MELODIC:
            ds = librosa.resample(mono[name], orig_sr=stemsmod.SR, target_sr=22050)
            c = librosa.feature.spectral_centroid(y=ds, sr=22050)[0]
            cens[name] = (c, librosa.times_like(c, sr=22050))
        sections = []
        for sec in r["sections"]:
            t0, t1 = sec["t0"], sec["t1"]
            a = max(0, min(frames - 1, int(t0 / 0.05)))
            b = max(a + 1, min(frames, int(t1 / 0.05)))
            means = {k: float(envs[k][a:b].mean()) for k in TRUE_STEMS}
            total = sum(means.values())
            shares = {k: (v / total if total > 0 else 0.0) for k, v in means.items()}
            bright = {}
            for name, (c, ct) in cens.items():
                w = (ct >= t0) & (ct < t1)
                bright[name] = float(c[w].mean()) if w.any() else 0.0
            sections.append({**sec, "share": shares, "cen": bright})
        row = {"song": r["song"], "cls": classify(r["sections"]), "sections": sections}
        rows.append(row)
        done[r["song"]] = row
        args.out.write_text(json.dumps(rows), encoding="utf-8")
        el = time.time() - started
        print(f"  [{len(rows)}/{len(plan)}] {r['song'][:40]:42s} {row['cls']}  "
              f"({el / 60:.0f} min, {el / max(1, len(rows)):.0f}s/song)", flush=True)
    print(f"done: {len(rows)} songs -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
