"""Research-only sweep: how much of each human Expert chart does Basic Pitch
*see* (any onset within tolerance), on the isolated guitar stem vs the mix,
under the default decode parameters vs relaxed ones?  Nothing in chartgen is
touched; this reads the song library and writes JSONL to the scratchpad.

Per song and source (guitar stem / mix), one model pass, then several decodes:
    D  onset 0.5 frame 0.3 min_len 127.7 ms   (chartgen today = library default)
    A  onset 0.5 frame 0.3 min_len  58 ms
    B  onset 0.5 frame 0.3 min_len  23 ms
    C  onset 0.3 frame 0.3 min_len  58 ms
    E  onset 0.3 frame 0.2 min_len  23 ms
    F  onset 0.2 frame 0.2 min_len  23 ms
plus two ceilings: raw onset-posteriorgram peaks (any pitch, >=0.1) and
librosa spectral-flux onsets on the same audio.
"""
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import soundfile as sf
import librosa

warnings.filterwarnings("ignore")
import logging
logging.getLogger().setLevel(logging.ERROR)

ROOT = Path(__file__).resolve().parents[2]
SONGS = Path.home() / ".clonehero/Songs"
CONVERTED = ROOT / "data" / "converted"
SCRATCH = Path(__file__).resolve().parent
OUT = ROOT / "out" / "bp_research" / "bp_sweep.jsonl"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from evaluate import chart_note_times  # noqa: E402

from basic_pitch import ICASSP_2022_MODEL_PATH  # noqa: E402
from basic_pitch.inference import Model, run_inference  # noqa: E402
from basic_pitch.note_creation import model_output_to_notes  # noqa: E402

SR = 22050
FPS = 22050 / 256
CONFIGS = {
    "D": (0.5, 0.3, 127.7),
    "A": (0.5, 0.3, 58.0),
    "B": (0.5, 0.3, 23.0),
    "C": (0.3, 0.3, 58.0),
    "E": (0.3, 0.2, 23.0),
    "F": (0.2, 0.2, 23.0),
}
STEM_NAMES = ("guitar", "rhythm", "bass", "drums", "drums_1", "drums_2",
              "drums_3", "drums_4", "vocals", "keys", "song")
AUDIO_EXT = (".opus", ".ogg", ".mp3", ".wav", ".flac")


def load_mono(path: Path) -> np.ndarray:
    y, sr = sf.read(str(path), dtype="float32", always_2d=True)
    y = y.mean(axis=1)
    if sr != SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=SR, res_type="soxr_hq")
    return y


def song_sources(folder: Path):
    """{'guitar': array|None, 'mix': array} at 22050 mono."""
    files = {p.stem.lower(): p for p in folder.iterdir()
             if p.suffix.lower() in AUDIO_EXT}
    stems = {k: v for k, v in files.items() if k in STEM_NAMES}
    guitar = load_mono(files["guitar"]) if "guitar" in files else None
    if len(stems) > 1:  # multitrack: song.* is a residual, sum everything
        parts = [load_mono(p) for k, p in stems.items()]
        n = max(len(p) for p in parts)
        mix = np.zeros(n, dtype=np.float32)
        for p in parts:
            mix[:len(p)] += p
    elif "song" in files:
        mix = load_mono(files["song"])
    else:
        return None
    peak = float(np.max(np.abs(mix))) or 1.0
    if peak > 1.0:
        mix = mix / peak
    return {"guitar": guitar, "mix": mix}


def human_chart(folder: Path):
    p = folder / "notes.chart"
    if not p.is_file():
        p = CONVERTED / f"{folder.name}.chart"
    if not p.is_file():
        return None
    notes = chart_note_times(str(p), "ExpertSingle")
    times = np.array(sorted({round(t, 4) for t, _ in notes}))
    return times


def onset_times_from_events(events):
    t = np.array(sorted(e[0] for e in events)) if events else np.zeros(0)
    if len(t) == 0:
        return t
    keep = np.concatenate([[True], np.diff(t) > 0.010])
    return t[keep]


def coverage(human: np.ndarray, det: np.ndarray, tol: float):
    if len(det) == 0 or len(human) == 0:
        return 0.0, np.zeros(0)
    idx = np.searchsorted(det, human)
    lo = det[np.clip(idx - 1, 0, len(det) - 1)]
    hi = det[np.clip(idx, 0, len(det) - 1)]
    d = np.where(np.abs(human - lo) < np.abs(human - hi), human - lo, human - hi)
    hit = np.abs(d) <= tol
    return float(hit.mean()), d[hit]


def score(human: np.ndarray, det: np.ndarray):
    r50, _ = coverage(human, det, 0.050)
    r80, err = coverage(human, det, 0.080)
    p80, _ = coverage(det, human, 0.080)
    return {
        "n_det": int(len(det)),
        "recall50": round(r50, 4), "recall80": round(r80, 4),
        "precision80": round(p80, 4),
        "abs_err_med_ms": round(float(np.median(np.abs(err))) * 1000, 1) if len(err) else None,
        "bias_med_ms": round(float(np.median(err)) * 1000, 1) if len(err) else None,
    }


def analyse(model: Model, y: np.ndarray, human: np.ndarray, tag: str):
    tmp = SCRATCH / f"_tmp_{tag}.wav"
    sf.write(str(tmp), y, SR)
    t0 = time.perf_counter()
    out = run_inference(str(tmp), model)
    t_model = time.perf_counter() - t0
    tmp.unlink(missing_ok=True)
    res = {"t_model_s": round(t_model, 1), "configs": {}}
    for name, (on, fr, mnl) in CONFIGS.items():
        t0 = time.perf_counter()
        _, events = model_output_to_notes(
            out, onset_thresh=on, frame_thresh=fr,
            min_note_len=int(np.round(mnl / 1000 * FPS)),
            include_pitch_bends=False, melodia_trick=True)
        det = onset_times_from_events(events)
        row = score(human, det)
        row["n_notes"] = len(events)
        durs = np.array([e[1] - e[0] for e in events]) if events else np.zeros(0)
        row["short_note_share"] = round(float((durs < 0.128).mean()), 3) if len(durs) else None
        row["t_decode_s"] = round(time.perf_counter() - t0, 1)
        res["configs"][name] = row
    # ceiling 1: raw onset posteriorgram peaks at any pitch
    on_any = out["onset"].max(axis=1)
    peaks = librosa.util.peak_pick(on_any, pre_max=2, post_max=2, pre_avg=2,
                                   post_avg=2, delta=0.0, wait=1)
    for thr in (0.1, 0.25):
        pk = peaks[on_any[peaks] >= thr]
        res[f"onsetpost_{thr}"] = score(human, pk / FPS)
    # ceiling 2: spectral flux onsets on the same audio
    env = librosa.onset.onset_strength(y=y, sr=SR)
    frames = librosa.onset.onset_detect(onset_envelope=env, sr=SR, units="frames")
    res["specflux"] = score(human, librosa.frames_to_time(frames, sr=SR))
    return res


def main():
    model = Model(ICASSP_2022_MODEL_PATH)
    done = set()
    if OUT.is_file():
        for line in OUT.read_text(encoding="utf-8").splitlines():
            try:
                done.add(json.loads(line)["song"])
            except Exception:
                pass
    folders = [f for f in sorted(SONGS.iterdir()) if f.is_dir() and "[chartgen]" not in f.name]
    # multitrack folders (real guitar stem) first, then the rest
    folders.sort(key=lambda f: not (f / "guitar.opus").exists() and not (f / "guitar.ogg").exists())
    for folder in folders:
        if folder.name in done:
            continue
        try:
            human = human_chart(folder)
            if human is None or len(human) < 50:
                continue
            src = song_sources(folder)
            if src is None:
                continue
            rec = {"song": folder.name, "n_human": int(len(human)),
                   "duration_s": round(len(src["mix"]) / SR, 1),
                   "multitrack": src["guitar"] is not None}
            t0 = time.perf_counter()
            rec["mix"] = analyse(model, src["mix"], human, "mix")
            if src["guitar"] is not None:
                rec["guitar"] = analyse(model, src["guitar"], human, "gtr")
            rec["t_total_s"] = round(time.perf_counter() - t0, 1)
            with open(OUT, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec) + "\n")
            d = rec["mix"]["configs"]
            g = rec.get("guitar", {}).get("configs", {})
            print(f"{folder.name[:48]:48s} human={len(human):5d} "
                  f"mix D/B r80={d['D']['recall80']:.2f}/{d['B']['recall80']:.2f} "
                  + (f"gtr D/B r80={g['D']['recall80']:.2f}/{g['B']['recall80']:.2f} " if g else "")
                  + f"({rec['t_total_s']}s)", flush=True)
        except Exception as exc:  # keep sweeping
            print(f"FAIL {folder.name}: {exc!r}", flush=True)


if __name__ == "__main__":
    main()
