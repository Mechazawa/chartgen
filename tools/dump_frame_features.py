"""Frame-level feature cache for the sliding-window solo sweep.

The section-level caches capped recall: chroma sections merge solos with
their neighbours (MJ's harmonica solo diluted into a 46s span), so no
threshold could frame them. Sliding windows need the features at FRAME
level, cached once per song so the window sweep runs in seconds:

- pyin voiced flags + f0 (hop 1024) on the harmonic part
- spectral centroid + chroma (12 x frames)
- Demucs other-stem RMS envelope (the lead-instrument signal)

Ground truth is corrected for the measured label noise: 4+ of the current
rule's "false positives" sat in sections the charter NAMED Guitar Solo /
Organ Solo but never gave scoring markers. Truth = marker spans PLUS
named-solo section spans; quiet controls (no markers, no solo-named
sections) are sampled properly this time.

Each song -> work/framecache/<name>.npz + one manifest line.

    python tools/dump_frame_features.py --start 0 --end 200
"""
import argparse
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from study_solos import blocks, is_generated, note_ticks, section_ticks, solo_spans, RES_RE
from validate_solos import find_audio

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "work/framecache"
SOLO_NAME = re.compile(r"solo", re.I)
HOP = 1024


def named_solo_spans(events_block, ticks, resolution):
    """[(start, end)] for sections whose NAME contains 'solo'."""
    sections = section_ticks(events_block)
    bounds = [t for t, _ in sections] + [max(ticks) + 1 if ticks else 0]
    spans = []
    for (start, name), end in zip(sections, bounds[1:], strict=True):
        if SOLO_NAME.search(name):
            spans.append((start, end))
    # merge adjacent named parts (Solo A / Solo B / Solo C)
    merged = []
    for a, b in sorted(spans):
        if merged and a - merged[-1][1] <= resolution:
            merged[-1] = (merged[-1][0], b)
        else:
            merged.append((a, b))
    return merged


def pick_songs():
    """(chart, audio, truth, quiet) rows: all solo-evidence charts + controls."""
    rows = []
    quiet_pool = []
    for chart in sorted((ROOT / "data/calibration").rglob("*.chart")):
        b = blocks(chart.read_text(encoding="utf-8-sig", errors="ignore"))
        if is_generated(b) or "ExpertSingle" not in b:
            continue
        audio = find_audio(chart.parent)
        if not audio:
            continue
        found = RES_RE.search(b.get("Song", ""))
        res = int(found.group(1)) if found else 192
        ticks = note_ticks(b["ExpertSingle"])
        if len(ticks) < 100:
            continue
        marked = solo_spans(b["ExpertSingle"])
        named = named_solo_spans(b.get("Events", ""), ticks, res)
        # truth = markers plus named-solo sections not already covered
        truth = list(marked)
        for a, c in named:
            if not any(not (c < s or a > e) for s, e in marked):
                truth.append((a, c))
        if truth:
            rows.append((chart, audio, sorted(truth), False))
        else:
            quiet_pool.append((chart, audio, [], True))
    import random
    random.Random(11).shuffle(quiet_pool)
    return rows + quiet_pool[:100]


def dump_song(chart, audio, truth, quiet):
    import librosa
    import numpy as np

    from chartgen import solo as solomod
    from chartgen import stems as stemsmod
    from chartgen import tempo as tempomod

    b = blocks(chart.read_text(encoding="utf-8-sig", errors="ignore"))
    found = RES_RE.search(b.get("Song", ""))
    res = int(found.group(1)) if found else 192
    ticks = note_ticks(b["ExpertSingle"])

    y, sr = tempomod.load(str(audio))
    tmap = tempomod.detect(y, sr, resolution=res)
    harmonic = librosa.effects.harmonic(y, margin=2.0)
    f0, _, _ = librosa.pyin(harmonic, fmin=solomod.FMIN_HZ, fmax=solomod.FMAX_HZ,
                            sr=sr, fill_na=np.nan, hop_length=HOP)
    f0_times = librosa.times_like(f0, sr=sr, hop_length=HOP)
    centroid = librosa.feature.spectral_centroid(y=harmonic, sr=sr)[0]
    cen_times = librosa.times_like(centroid, sr=sr)
    chroma = librosa.feature.chroma_cqt(y=harmonic, sr=sr)
    chroma_times = librosa.times_like(chroma, sr=sr, hop_length=512)

    mono = stemsmod.separate(str(audio))
    other = stemsmod.activity(mono["other"])[1] if mono else np.zeros(1)

    # Beat times for every 8th-beat grid point, so windows are defined in
    # BEATS at sweep time without needing the tempo map again.
    last_beat = ticks[-1] / res
    grid_beats = np.arange(0, last_beat + 1e-6, 8.0)
    grid_times = np.array([tmap.beat_to_time(bt) for bt in grid_beats])

    return dict(
        f0=f0.astype(np.float32), f0_times=f0_times.astype(np.float32),
        centroid=centroid.astype(np.float32), cen_times=cen_times.astype(np.float32),
        chroma=chroma.astype(np.float32), chroma_times=chroma_times.astype(np.float32),
        other_rms=other.astype(np.float32),
        grid_beats=grid_beats.astype(np.float32), grid_times=grid_times.astype(np.float32),
        note_ticks=np.array(ticks, dtype=np.int64),
        truth=np.array(truth or [(0, 0)], dtype=np.int64),
        meta=np.array([res, ticks[-1], 1 if quiet else 0, len(truth)], dtype=np.int64),
    )


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=None)
    args = ap.parse_args(argv)

    import numpy as np

    CACHE.mkdir(parents=True, exist_ok=True)
    rows = pick_songs()
    print(f"{len(rows)} songs ({sum(1 for r in rows if not r[3])} with truth, "
          f"{sum(1 for r in rows if r[3])} quiet controls)", flush=True)
    rows = rows[args.start:args.end if args.end is not None else len(rows)]

    started = time.time()
    for i, (chart, audio, truth, quiet) in enumerate(rows, 1):
        out = CACHE / (re.sub(r'[<>:"/\\|?*]', "", chart.parent.name)[:100] + ".npz")
        if out.exists():
            continue
        try:
            data = dump_song(chart, audio, truth, quiet)
        except Exception as error:
            print(f"  [{i}] {chart.parent.name[:44]}: {type(error).__name__}: {error}",
                  flush=True)
            continue
        np.savez_compressed(out, **data)
        print(f"  [{i}/{len(rows)}] {chart.parent.name[:50]:<52} "
              f"truth {len(truth)}  {time.time() - started:.0f}s", flush=True)
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
