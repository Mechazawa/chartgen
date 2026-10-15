"""Refit the 0-6 difficulty tier predictor against the calibration set.

The old fit used the local library's ratings, which the playtests flagged
as unreliable ("a chart labeled 1 that plays like a 3"). The Encore
manifest carries a charter-assigned diff_guitar for all 800 pulled charts,
and every one passed the full-ladder + no-issues filter - the same "did
the whole job" trust signal the old calibration weighted x2, but for the
entire sample.

Features are computed by chartgen.rating.features_from_notes on the human
chart's own SyncTrack, so the numbers are exactly what the pipeline
computes for generated charts. Writes chartgen/rating_coeffs.json.

    python tools/calibrate_rating_encore.py
"""
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chartgen.rating import FEATURE_ORDER, features_from_notes
from study_solos import blocks, RES_RE
from study_tap_timbre import NOTE_RE, SYNC_RE, tick_to_time

OPEN = 7


class ChartTempo:
    """Adapter: the human chart's SyncTrack, shaped like chartgen's TempoMap
    for the one method the rating features call."""

    def __init__(self, sync, resolution):
        self.resolution = resolution
        self._at = tick_to_time(sync, resolution)

    def beat_to_time(self, beat: float) -> float:
        return self._at(beat * self.resolution)


def main(argv=None):
    root = Path(__file__).resolve().parent.parent
    cal = root / "data/calibration"
    manifest = {}
    for line in (cal / "manifest.jsonl").read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        text = f"{row.get('artist') or 'Unknown'} - {row.get('name') or 'Untitled'} ({row.get('charter') or 'unknown'})"
        folder = re.sub(r'[<>:"/\\|?*]', "", text).strip(" -.")[:120]
        manifest[folder] = row

    rows, labels = [], []
    skipped = 0
    for folder in sorted(cal.iterdir()):
        meta = manifest.get(folder.name)
        chart = folder / "notes.chart"
        if not meta or not chart.is_file():
            continue
        tier = meta.get("diff_guitar")
        if tier is None or not (0 <= tier <= 6):
            skipped += 1
            continue
        b = blocks(chart.read_text(encoding="utf-8", errors="ignore"))
        found = RES_RE.search(b.get("Song", ""))
        res = int(found.group(1)) if found else 192
        sync = [(int(t), int(v)) for t, v in SYNC_RE.findall(b.get("SyncTrack", ""))]
        notes = [(int(t), int(l), int(s))
                 for t, l, s in NOTE_RE.findall(b.get("ExpertSingle", ""))
                 if int(l) < 5 or int(l) == OPEN]
        if not sync or len(notes) < 50:
            skipped += 1
            continue
        feats = features_from_notes(notes, ChartTempo(sync, res))
        rows.append([feats[k] for k in FEATURE_ORDER] + [1.0])
        labels.append(float(tier))

    X = np.asarray(rows)
    y = np.asarray(labels)
    print(f"{len(y)} rated charts ({skipped} skipped); "
          f"tier distribution: {[int((y == t).sum()) for t in range(7)]}")

    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = np.clip(np.round(X @ coef), 0, 6)
    mae = float(np.mean(np.abs(pred - y)))
    within1 = float(np.mean(np.abs(pred - y) <= 1))
    print(f"fit: MAE {mae:.2f}, {within1:.0%} within +/-1 tier")
    print("coef:", {k: round(float(c), 3)
                    for k, c in zip(FEATURE_ORDER + ("intercept",), coef, strict=True)})

    out = root / "chartgen/rating_coeffs.json"
    out.write_text(json.dumps({
        "features": list(FEATURE_ORDER), "coef": [float(c) for c in coef],
        "fitted_on": f"{len(y)} Encore full-ladder charts, 2026-08-24",
        "mae": mae, "within_1_tier": within1,
    }, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
