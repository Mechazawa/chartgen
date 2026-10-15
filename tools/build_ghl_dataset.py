"""Run chartgen's listening front end on every official 6-fret song, so the
GHL shape model trains on the same note events it sees at inference.

Each song's audio files (all but the preview and the crowd) are mixed back
into one recording, then transcribed the way the pipeline does it: Basic
Pitch on the mix, and Basic Pitch on each BS-RoFormer stem. One pickle per
song lands in --out: {"mix": events, "stems": {stem: events}, "chart":
GhlChart}, where events are [(start_s, end_s, midi, amplitude)].

Separation holds the GPU, so one process runs it song by song while a
thread loads the next song's audio and Basic Pitch transcribes the previous
songs' streams in CPU worker processes. Each record also holds the beat
grid the pipeline detects ("tempo": (beat_times, pickup_beats)); a record
built without one gets it added on the next run.

    python -m tools.build_ghl_dataset data/ghl-official -o data/ghl-features
"""
import argparse
import os
import pickle
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

os.environ.setdefault("CHARTGEN_STEM_CACHE", "0")  # stems are ~200 MB a song

import librosa  # noqa: E402
import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

from chartgen import prominence, stems, tempo, transcribe  # noqa: E402
from tools.ghl_charts import load  # noqa: E402

NOT_MUSIC = ("preview.opus", "crowd.opus")
TRANSCRIBE_WORKERS = 12


def mix(folder: Path) -> np.ndarray:
    parts = [librosa.load(str(path), sr=stems.SR, mono=True)[0]
             for path in sorted(folder.glob("*.opus")) if path.name not in NOT_MUSIC]
    out = np.zeros(max(len(p) for p in parts), dtype=np.float32)
    for part in parts:
        out[:len(part)] += part
    return out / max(1.0, float(np.abs(out).max()))


def beat_grid(audio: np.ndarray) -> tuple:
    found = tempo.detect(audio, stems.SR)
    return found.beat_times, found.pickup_beats


def add_beat_grid(path: Path, folder: Path) -> None:
    record = pickle.loads(path.read_bytes())
    record["tempo"] = beat_grid(mix(folder))
    path.write_bytes(pickle.dumps(record))


def transcribe_audio(audio: np.ndarray) -> list:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fh:
        path = fh.name
    try:
        sf.write(path, audio, stems.SR)
        return transcribe.transcribe(path)
    finally:
        Path(path).unlink(missing_ok=True)


def separate(audio: np.ndarray) -> dict | None:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fh:
        path = fh.name
    try:
        sf.write(path, audio, stems.SR)
        mono = stems.separate(path, lambda m: print(m, flush=True), backend="sw")
    finally:
        Path(path).unlink(missing_ok=True)
        stems._CACHE.clear()
    return mono if mono and "guitar" in mono else None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root", type=Path)
    ap.add_argument("-o", "--out", type=Path, default=Path("data/ghl-features"))
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    folders = sorted(p.parent for p in args.root.rglob("notes.mid"))
    todo = [f for f in folders if not (args.out / f"{f.name}.pkl").exists()]
    print(f"{len(folders)} songs, {len(todo)} to build", flush=True)
    pool = ProcessPoolExecutor(TRANSCRIBE_WORKERS)
    built = [(args.out / f"{f.name}.pkl", f) for f in folders if (args.out / f"{f.name}.pkl").exists()]
    no_grid = [(path, f) for path, f in built if "tempo" not in pickle.loads(path.read_bytes())]
    if no_grid:
        print(f"adding the beat grid to {len(no_grid)} records", flush=True)
        list(pool.map(add_beat_grid, *zip(*no_grid, strict=True)))
    loader = ThreadPoolExecutor(1)
    next_audio = loader.submit(mix, todo[0]) if todo else None
    pending = []

    def finish(entry):
        index, folder, chart, grid, started, mix_job, stem_jobs = entry
        record = {"mix": mix_job.result(),
                  "stems": {name: job.result() for name, job in stem_jobs.items()},
                  "chart": chart, "tempo": grid}
        (args.out / f"{folder.name}.pkl").write_bytes(pickle.dumps(record))
        print(f"[{index}/{len(todo)}] {folder.name} {time.time() - started:.0f}s", flush=True)

    for index, folder in enumerate(todo, 1):
        started = time.time()
        loaded = next_audio
        if index < len(todo):
            next_audio = loader.submit(mix, todo[index])
        try:
            audio = loaded.result()
        except (OSError, ValueError) as error:
            print(f"skip {folder.name}: audio unreadable ({error})", flush=True)
            continue
        chart = load(folder / "notes.mid")
        if chart is None or not chart.tiers["Expert"]:
            print(f"skip {folder.name}: no GHL Expert", flush=True)
            continue
        mono = separate(audio)
        if mono is None:
            print(f"skip {folder.name}: BS-RoFormer separation failed", flush=True)
            continue
        pending.append((index, folder, chart, beat_grid(audio), started, pool.submit(transcribe_audio, audio),
                        {name: pool.submit(transcribe_audio, mono[name])
                         for name in prominence.STEMS}))
        while len(pending) > 3:  # bounds the stems held in memory
            finish(pending.pop(0))
    for entry in pending:
        finish(entry)
    pool.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
