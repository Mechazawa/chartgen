"""Fit the diff_guitar (0-6 tier) predictor against the local library.

Every song with a rated song.ini votes on how chart features map to the tier
number, but votes are weighted by how much the rating can be trusted:

    x3  charter matches a reputable source (Harmonix/Neversoft/official rips)
    x2  chart has all four difficulties (a charter who did the whole job
        usually also rated honestly; expert-only drops often guess)

Community expert-only charts still participate — there are too many to
ignore — they just can't outvote the reliable ones. Writes the fitted
coefficients to chartgen/rating_coeffs.json (which chartgen.rating loads) and
prints the weighted fit error so a bad fit is visible immediately.

    python tools/calibrate_rating.py --manifest data/audio_dataset_with_raw.json
"""
import argparse
import configparser
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import chartgen  # noqa: F401,E402

REPUTABLE = re.compile(r"harmonix|neversoft|rock ?band|guitar ?hero|rb[0-9]", re.I)
TIER_RE = re.compile(r"\[(Expert|Hard|Medium|Easy)Single\]")


def read_song_ini(folder: Path) -> dict:
    ini = folder / "song.ini"
    if not ini.is_file():
        return {}
    parser = configparser.ConfigParser(strict=False, interpolation=None)
    try:
        parser.read_string(ini.read_text(encoding="utf-8", errors="replace"))
    except configparser.Error:
        return {}
    section = next((s for s in parser.sections() if s.lower() == "song"), None)
    return dict(parser[section]) if section else {}


def song_features(chart_path: str) -> dict | None:
    """Difficulty-relevant features from the Expert track, in seconds."""
    from tools.evaluate import chart_note_times

    notes = chart_note_times(chart_path)
    if len(notes) < 50:
        return None
    times = np.array([t for t, _ in notes])
    duration = float(times[-1] - times[0])
    if duration < 30:
        return None
    gaps = np.diff(times)
    gaps = gaps[gaps > 1e-4]

    window = 5.0
    starts = np.arange(times[0], times[-1] - window, 1.0)
    counts = np.array([np.count_nonzero((times >= s) & (times < s + window))
                       for s in starts]) / window if len(starts) else np.array([0.0])
    return {
        "nps": len(times) / duration,
        "nps_peak": float(np.percentile(counts, 95)),
        "chords": sum(1 for _, l in notes if len(l) > 1) / len(notes),
        "fast": float(np.mean(gaps < 0.14)) if len(gaps) else 0.0,
    }


FEATURE_ORDER = ("nps", "nps_peak", "chords", "fast")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, default=Path("data/audio_dataset_with_raw.json"))
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).resolve().parent.parent / "chartgen" / "rating_coeffs.json")
    args = ap.parse_args()

    entries = json.loads(args.manifest.read_text(encoding="utf-8"))
    songs = {}
    for e in entries:
        songs.setdefault(e["chart_path"], e)

    rows = []
    for chart_path, e in songs.items():
        folder = Path(e["audio_path"]).parent
        meta = read_song_ini(folder)
        try:
            rating = int(str(meta.get("diff_guitar", "-1")).strip())
        except ValueError:
            continue
        if not 0 <= rating <= 7:
            continue
        feats = song_features(chart_path)
        if feats is None:
            continue

        text = Path(chart_path).read_text(encoding="utf-8", errors="replace")
        tiers = set(TIER_RE.findall(text))
        weight = 1.0
        if len(tiers) == 4:
            weight *= 2.0
        charter = str(meta.get("charter", "")) + " " + folder.name
        if REPUTABLE.search(charter):
            weight *= 3.0
        rows.append((feats, min(rating, 6), weight, folder.name[:40]))

    if len(rows) < 20:
        print(f"only {len(rows)} rated songs — not enough to fit")
        return 1

    X = np.array([[f[k] for k in FEATURE_ORDER] for f, _, _, _ in rows])
    y = np.array([r for _, r, _, _ in rows], dtype=float)
    w = np.array([wt for _, _, wt, _ in rows])

    # Weighted least squares with intercept; features are few and monotone
    # by construction, so plain linear + clamp is enough and stays legible.
    Xb = np.hstack([X, np.ones((len(X), 1))])
    Ws = np.sqrt(w)[:, None]
    coef, *_ = np.linalg.lstsq(Xb * Ws, y * Ws.ravel(), rcond=None)

    pred = np.clip(np.round(Xb @ coef), 0, 6)
    mae = float(np.average(np.abs(pred - y), weights=w))
    within1 = float(np.average(np.abs(pred - y) <= 1, weights=w))
    print(f"{len(rows)} rated songs "
          f"(total weight {w.sum():.0f}, reputable/full-tier emphasised)")
    print(f"weighted MAE {mae:.2f} tiers; within +/-1 tier: {within1:.0%}")
    for name, value in zip(FEATURE_ORDER + ("intercept",), coef, strict=True):
        print(f"   {name:>10}: {value:+.3f}")

    worst = sorted(rows, key=lambda r: -abs(
        np.clip(round(np.dot([r[0][k] for k in FEATURE_ORDER] + [1.0], coef)), 0, 6) - r[1]))
    print("\nlargest disagreements (often the mislabeled charts):")
    for f, r, wt, name in worst[:5]:
        p = int(np.clip(round(np.dot([f[k] for k in FEATURE_ORDER] + [1.0], coef)), 0, 6))
        print(f"   labeled {r} predicted {p} (w={wt:.0f})  {name}")

    args.out.write_text(json.dumps(
        {"features": FEATURE_ORDER, "coef": coef.tolist(),
         "fit": {"songs": len(rows), "weighted_mae": mae, "within_1": within1}},
        indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
