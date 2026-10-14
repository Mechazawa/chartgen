"""Research-only: does averaging Basic Pitch posteriorgrams over a few input
shifts (test-time augmentation on window alignment) make the notes stable
and/or cover more of the human chart?  Shifts are whole frames (256 samples)
so the outputs align exactly after cropping.

Ensemble A = shifts {0, 43, 86} frames; ensemble B = {21, 64, 107}.
Single passes: shift 0 vs shift 21.  Agreement A-vs-B measures ensemble
stability the same way single-0 vs single-21 measures single-pass stability.
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
from bp_stability import compare, load  # noqa: E402
from bp_sweep import coverage, onset_times_from_events  # noqa: E402

SR = 22050; HOP = 256; FPS = SR / HOP
PICK = ["Metallica - Enter Sandman", "Van Halen - Panama (Harmonix)",
        "The Killers - Mr. Brightside (Harmonix)", "Creed - Higher"]


def posterior(model, y, shift_frames):
    pad = np.zeros(shift_frames * HOP, dtype=np.float32)
    tmp = SCRATCH / "_ens.wav"
    sf.write(str(tmp), np.concatenate([pad, y]), SR)
    out = run_inference(str(tmp), model)
    tmp.unlink(missing_ok=True)
    return {k: v[shift_frames:] for k, v in out.items()}


def average(posts):
    n = min(p["note"].shape[0] for p in posts)
    return {k: np.mean([p[k][:n] for p in posts], axis=0) for k in posts[0]}


def decode(out, on=0.5, fr=0.3, mnl=127.7):
    _, ev = model_output_to_notes(out, onset_thresh=on, frame_thresh=fr,
                                  min_note_len=int(np.round(mnl / 1000 * FPS)),
                                  include_pitch_bends=False)
    return ev


def main():
    model = Model(ICASSP_2022_MODEL_PATH)
    for name in PICK:
        y = load(SONGS / name)
        if y is None:
            continue
        y = y[: SR * 90]
        chart = ROOT / "data" / "converted" / f"{name}.chart"
        if not chart.is_file():
            chart = SONGS / name / "notes.chart"
        human = np.array(sorted({round(t, 4) for t, _ in chart_note_times(str(chart))}))
        human = human[human < 90]
        posts = {s: posterior(model, y, s) for s in (0, 21, 43, 64, 86, 107)}
        ensA = average([posts[0], posts[43], posts[86]])
        ensB = average([posts[21], posts[64], posts[107]])
        for cfg, kw in (("D", {}), ("B", {"mnl": 23.0})):
            s0 = decode(posts[0], **kw); s21 = decode(posts[21], **kw)
            eA = decode(ensA, **kw); eB = decode(ensB, **kw)
            ev = lambda e: sorted((x[0], x[2]) for x in e)
            single = compare(ev(s0), ev(s21))[0]
            ens = compare(ev(eA), ev(eB))[0]
            r = lambda e: coverage(human, onset_times_from_events(e), 0.080)[0]
            p = lambda e: coverage(onset_times_from_events(e), human, 0.080)[0]
            print(f"{name[:36]:36s} cfg {cfg}: stability single {single:.3f} -> ensemble3 {ens:.3f} | "
                  f"recall80 single {r(s0):.3f} ens {r(eA):.3f} | prec80 single {p(s0):.3f} ens {p(eA):.3f} | "
                  f"notes {len(s0)} -> {len(eA)}", flush=True)


if __name__ == "__main__":
    main()
