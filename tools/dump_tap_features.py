"""Cache per-note softness features for every human chart that uses taps.

study_tap_timbre.py established the aggregate: taps lean dark (2:1, -0.5z
brightness) but not soft-attacked. Building an actual tap rule needs the
per-note detail — which runs of notes the charter tapped and what each note
sounded like — so the phrase threshold can be swept in seconds instead of
re-decoding audio per guess.

    python tools/dump_tap_features.py ~/.clonehero/Songs -o work/taps_feat.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from study_solos import is_generated, blocks, RES_RE
from study_tap_timbre import NOTE_RE, SYNC_RE, TAP, OPEN, tick_to_time
from validate_solos import find_audio


def dump(chart: Path, audio: Path):
    import librosa
    import numpy as np

    b = blocks(chart.read_text(encoding="utf-8", errors="ignore"))
    found = RES_RE.search(b.get("Song", ""))
    res = int(found.group(1)) if found else 192
    expert = b.get("ExpertSingle", "")
    sync = [(int(t), int(v)) for t, v in SYNC_RE.findall(b.get("SyncTrack", ""))]
    if not expert or not sync:
        return None

    by_tick: dict[int, set[int]] = {}
    for tick, lane, _ in NOTE_RE.findall(expert):
        by_tick.setdefault(int(tick), set()).add(int(lane))
    positions = sorted(t for t, lanes in by_tick.items()
                       if any(l < 5 or l == OPEN for l in lanes))
    tapped = {t for t in positions if TAP in by_tick[t]}
    if len(tapped) < 24 or len(positions) - len(tapped) < 24:
        return None

    at = tick_to_time(sync, res)
    y, sr = librosa.load(str(audio), mono=True)
    onset = librosa.onset.onset_strength(y=y, sr=sr)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    rms = librosa.feature.rms(y=y)[0]
    times = librosa.times_like(onset, sr=sr)

    def sample(feature, when):
        i = int(np.searchsorted(times, when))
        lo, hi = max(0, i - 2), min(len(feature), i + 3)
        return float(feature[lo:hi].max()) if hi > lo else 0.0

    notes = []
    for t in positions:
        when = at(t)
        lanes = by_tick[t]
        notes.append({
            "tick": t,
            "tap": t in tapped,
            "chord": sum(1 for l in lanes if l < 5 or l == OPEN) > 1,
            "attack": sample(onset, when),
            "bright": sample(centroid, when),
            "loud": sample(rms, when),
        })
    return {"name": chart.parent.name, "resolution": res, "notes": notes}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("library", type=Path)
    ap.add_argument("-o", "--out", type=Path, default=Path("work/taps_feat.json"))
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=None)
    args = ap.parse_args(argv)

    charts = []
    for chart in sorted(args.library.rglob("*.chart")):
        if "chartgen" in chart.parent.name.lower():
            continue
        audio = find_audio(chart.parent)
        if not audio:
            continue
        b = blocks(chart.read_text(encoding="utf-8", errors="ignore"))
        if is_generated(b):
            continue
        lanes = [int(l) for _, l, _ in NOTE_RE.findall(b.get("ExpertSingle", ""))]
        if sum(1 for l in lanes if l == TAP) >= 24:
            charts.append((chart, audio))
    charts = charts[args.start:args.end if args.end is not None else len(charts)]

    out = []
    started = time.time()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for i, (chart, audio) in enumerate(charts, 1):
        try:
            row = dump(chart, audio)
        except Exception as error:
            print(f"  [{i}] {chart.parent.name[:44]}: {type(error).__name__}: {error}",
                  flush=True)
            continue
        if not row:
            continue
        out.append(row)
        args.out.write_text(json.dumps(out), encoding="utf-8")
        print(f"  [{i}/{len(charts)}] {row['name'][:48]:<50} "
              f"{sum(1 for n in row['notes'] if n['tap'])} taps / "
              f"{len(row['notes'])} notes  {time.time() - started:.0f}s", flush=True)
    print(f"\nwrote {len(out)} songs to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
