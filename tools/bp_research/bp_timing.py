"""Research-only: onset TIMING precision on real guitar stems vs the official
chart, after removing each detector's constant bias.  Compares Basic Pitch
note onsets (relaxed decode B), Basic Pitch raw onset-posteriorgram peaks,
librosa SuperFlux-style onsets, and a fusion: each Basic Pitch onset snapped
to the nearest SuperFlux onset within 40 ms (else kept).
Reports p50 / p90 absolute error in ms among human notes matched within 80 ms.
"""
import sys
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
SCRATCH = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tools"))
from evaluate import chart_note_times  # noqa: E402
from basic_pitch import ICASSP_2022_MODEL_PATH  # noqa: E402
from basic_pitch.inference import Model, run_inference  # noqa: E402
from basic_pitch.note_creation import model_output_to_notes  # noqa: E402
from bp_stability import load  # noqa: E402
from bp_sweep import onset_times_from_events  # noqa: E402

SR = 22050; FPS = SR / 256
PICK = ["Metallica - Enter Sandman", "The Killers - Mr. Brightside (Harmonix)",
        "Creed - Higher", "Judas Priest - Halls of Valhalla (Harmonix)",
        "Foo Fighters - Learn to Fly (Harmonix)", "Van Halen - Panama (Harmonix)"]


def errors(human, det, tol=0.080):
    det = np.sort(det)
    idx = np.searchsorted(det, human)
    lo = det[np.clip(idx - 1, 0, len(det) - 1)]; hi = det[np.clip(idx, 0, len(det) - 1)]
    d = np.where(np.abs(human - lo) < np.abs(human - hi), human - lo, human - hi)
    d = d[np.abs(d) <= tol]
    return d


def stats(human, det):
    d = errors(human, det)
    if len(d) == 0:
        return "n/a"
    bias = np.median(d)
    a = np.abs(d - bias)
    return f"rec80 {len(d)/len(human):.2f} bias {bias*1000:+5.1f} ms  jitter p50 {np.median(a)*1000:4.1f} p90 {np.percentile(a,90)*1000:4.1f} ms  <=10ms {np.mean(a<=0.010):.2f}"


def superflux(y):
    S = librosa.feature.melspectrogram(y=y, sr=SR, n_fft=1024, hop_length=256, fmax=8000, n_mels=138)
    env = librosa.onset.onset_strength(S=librosa.power_to_db(S, ref=np.max), sr=SR, hop_length=256, lag=2, max_size=3)
    fr = librosa.onset.onset_detect(onset_envelope=env, sr=SR, hop_length=256, units="frames", backtrack=False)
    return librosa.frames_to_time(fr, sr=SR, hop_length=256)


def snap(bp, sfx, tol=0.040):
    sfx = np.sort(sfx)
    out = []
    for t in bp:
        i = np.searchsorted(sfx, t)
        c = [sfx[j] for j in (i - 1, i) if 0 <= j < len(sfx) and abs(sfx[j] - t) <= tol]
        out.append(min(c, key=lambda x: abs(x - t)) if c else t)
    return np.array(out)


def main():
    model = Model(ICASSP_2022_MODEL_PATH)
    for name in PICK:
        y = load(SONGS / name)
        if y is None:
            continue
        chart = ROOT / "data" / "converted" / f"{name}.chart"
        if not chart.is_file():
            chart = SONGS / name / "notes.chart"
        human = np.array(sorted({round(t, 4) for t, _ in chart_note_times(str(chart))}))
        tmp = SCRATCH / "_tim.wav"; sf.write(str(tmp), y, SR)
        out = run_inference(str(tmp), model); tmp.unlink(missing_ok=True)
        _, ev = model_output_to_notes(out, onset_thresh=0.5, frame_thresh=0.3, min_note_len=2, include_pitch_bends=False)
        bp = onset_times_from_events(ev)
        on_any = out["onset"].max(axis=1)
        pk = librosa.util.peak_pick(on_any, pre_max=2, post_max=2, pre_avg=2, post_avg=2, delta=0.0, wait=1)
        post = pk[on_any[pk] >= 0.25] / FPS
        sfx = superflux(y)
        # sub-frame refinement of BP onsets: parabolic interpolation on the onset posteriorgram
        refined = []
        for t in bp:
            f = int(round(t * FPS))
            if 1 <= f < len(on_any) - 1:
                a, b, c = on_any[f - 1], on_any[f], on_any[f + 1]
                den = a - 2 * b + c
                off = 0.5 * (a - c) / den if den != 0 else 0.0
                refined.append((f + np.clip(off, -0.5, 0.5)) / FPS)
            else:
                refined.append(t)
        print(f"\n{name}  (human {len(human)} notes)")
        print(f"  BP notes (B)        : {stats(human, bp)}")
        print(f"  BP notes + parabola : {stats(human, np.array(refined))}")
        print(f"  BP onset peaks>=.25 : {stats(human, post)}")
        print(f"  SuperFlux           : {stats(human, sfx)}")
        print(f"  BP snapped to SFlux : {stats(human, snap(bp, sfx))}", flush=True)


if __name__ == "__main__":
    main()
