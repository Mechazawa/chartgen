"""Do human charters reserve tap notes for softer sounds?

study_taps.py already ruled out the obvious theory: spacing before tapped
notes is the same as or slower than everywhere else, so speed is not the
trigger, and no charting standard documents any rule at all. The remaining
credible hypothesis is timbre - a tap removes the strum, and a strum is a
percussive gesture that suits sounds with a hard attack and feels wrong on
piano, plucks and soft arps.

That is measurable. For every tapped position in a human chart, sample the
audio at that moment - attack (onset strength), brightness (spectral
centroid), loudness (RMS) - and compare against the same song's strummed
and HOPO positions as per-song z-scores, so a soft ballad and a loud metal
mix speak the same units. Tick-to-time comes from the chart's own
SyncTrack, not from our beat detector, so timing error does not blur the
comparison.

    python tools/study_tap_timbre.py ~/.clonehero/Songs
"""
import argparse
import re
import statistics
import sys
from itertools import pairwise
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from study_solos import is_generated, blocks, RES_RE
from validate_solos import find_audio

NOTE_RE = re.compile(r"(\d+)\s*=\s*N\s+(\d+)\s+(\d+)")
SYNC_RE = re.compile(r"(\d+)\s*=\s*B\s+(\d+)")
TAP, FORCED, OPEN = 6, 5, 7


def tick_to_time(sync, resolution):
    """Closure mapping tick -> seconds using the chart's own tempo events."""
    events = sorted(sync)  # [(tick, bpm_milli)]
    if not events or events[0][0] != 0:
        events = [(0, 120000)] + events
    starts = [0.0]
    for (t0, bpm), (t1, _) in pairwise(events):
        starts.append(starts[-1] + (t1 - t0) / resolution * 60.0 / (bpm / 1000.0))

    def at(tick):
        import bisect
        i = bisect.bisect_right([e[0] for e in events], tick) - 1
        t0, bpm = events[i]
        return starts[i] + (tick - t0) / resolution * 60.0 / (bpm / 1000.0)
    return at


def study(chart: Path, audio: Path):
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
    tapped = sorted(t for t, lanes in by_tick.items() if TAP in lanes)
    plain = sorted(t for t, lanes in by_tick.items()
                   if TAP not in lanes and any(l < 5 or l == OPEN for l in lanes))
    # Enough of both kinds to compare, and taps must not be the whole chart.
    if len(tapped) < 24 or len(plain) < 24:
        return None

    at = tick_to_time(sync, res)
    y, sr = librosa.load(str(audio), mono=True)
    onset = librosa.onset.onset_strength(y=y, sr=sr)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    rms = librosa.feature.rms(y=y)[0]
    times = librosa.times_like(onset, sr=sr)

    def sample(feature, when):
        """Strongest value within one frame either side, like density.py."""
        i = int(np.searchsorted(times, when))
        lo, hi = max(0, i - 2), min(len(feature), i + 3)
        return float(feature[lo:hi].max()) if hi > lo else 0.0

    rows = {"attack": onset, "bright": centroid, "loud": rms}
    out = {}
    for name, feature in rows.items():
        tap_vals = [sample(feature, at(t)) for t in tapped]
        plain_vals = [sample(feature, at(t)) for t in plain]
        spread = statistics.pstdev(plain_vals) or 1e-9
        # Median tap position, in z-units of this song's strummed notes.
        out[name] = (statistics.median(tap_vals) - statistics.median(plain_vals)) / spread
    out["taps"] = len(tapped)
    out["plain"] = len(plain)
    out["name"] = chart.parent.name
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("library", type=Path)
    ap.add_argument("--limit", type=int, default=40)
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
        # Cheap pre-scan: no point decoding audio for a chart with no taps.
        b = blocks(chart.read_text(encoding="utf-8", errors="ignore"))
        if is_generated(b):
            continue
        lanes = [int(l) for _, l, _ in NOTE_RE.findall(b.get("ExpertSingle", ""))]
        if sum(1 for l in lanes if l == TAP) >= 24:
            charts.append((chart, audio))
    charts = charts[args.start:args.end if args.end is not None else args.limit]

    rows = []
    for i, (chart, audio) in enumerate(charts, 1):
        try:
            row = study(chart, audio)
        except Exception as error:
            print(f"  [{i}] {chart.parent.name[:44]}: {type(error).__name__}: {error}",
                  flush=True)
            continue
        if not row:
            continue
        rows.append(row)
        print(f"  [{i}/{len(charts)}] {row['name'][:44]:<46} "
              f"taps {row['taps']:>4}  attack {row['attack']:>+6.2f}  "
              f"bright {row['bright']:>+6.2f}  loud {row['loud']:>+6.2f}",
              flush=True)

    if not rows:
        sys.exit("no charts with enough taps AND plain notes to compare")

    print(f"\n{len(rows)} songs with a real tap/plain split")
    for key, label in (("attack", "attack (onset strength)"),
                       ("bright", "brightness (centroid)"),
                       ("loud", "loudness (RMS)")):
        vals = sorted(r[key] for r in rows)
        neg = sum(1 for v in vals if v < -0.1)
        pos = sum(1 for v in vals if v > 0.1)
        print(f"  {label:<26} median {statistics.median(vals):>+6.2f} z   "
              f"softer-on-taps in {neg}/{len(vals)} songs, "
              f"harder in {pos}/{len(vals)}")
    print("\n(negative = tapped notes sit LOWER than the same song's "
          "strummed notes on that feature)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
