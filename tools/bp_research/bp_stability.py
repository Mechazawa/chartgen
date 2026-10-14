"""Research-only: is Basic Pitch's note output stable under a tiny shift of
the input (which changes only its internal 2 s window alignment)?  If the
same audio shifted by 50 ms yields different onsets, riff repeats will never
transcribe identically even when the audio does repeat.

For a few isolated guitar stems: transcribe as-is and with 50 / 130 ms of
leading silence, undo the offset, and compare onset sets at 20 ms tolerance.
Also compare pitch sets of matched onsets.
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

SONGS = Path.home() / ".clonehero/Songs"
SCRATCH = Path(__file__).resolve().parent
from basic_pitch import ICASSP_2022_MODEL_PATH  # noqa: E402
from basic_pitch.inference import Model, run_inference  # noqa: E402
from basic_pitch.note_creation import model_output_to_notes  # noqa: E402

SR = 22050
FPS = 22050 / 256
PICK = ["Metallica - Enter Sandman", "Van Halen - Panama (Harmonix)",
        "The Killers - Mr. Brightside (Harmonix)", "Foo Fighters - Learn to Fly (Harmonix)",
        "Weezer - Say It Ain't So (Harmonix)", "Creed - Higher"]


def load(folder):
    for name in ("guitar.opus", "guitar.ogg"):
        p = folder / name
        if p.is_file():
            y, sr = sf.read(str(p), dtype="float32", always_2d=True)
            y = y.mean(axis=1)
            return librosa.resample(y, orig_sr=sr, target_sr=SR, res_type="soxr_hq")
    return None


def events(model, y, tag, on=0.5, fr=0.3, mnl=127.7):
    tmp = SCRATCH / f"_stab_{tag}.wav"
    sf.write(str(tmp), y, SR)
    out = run_inference(str(tmp), model)
    tmp.unlink(missing_ok=True)
    _, ev = model_output_to_notes(out, onset_thresh=on, frame_thresh=fr,
                                  min_note_len=int(np.round(mnl / 1000 * FPS)),
                                  include_pitch_bends=False)
    return sorted((e[0], e[2]) for e in ev)


def compare(a, b, tol=0.020):
    """fraction of a's onsets that have an onset in b within tol; and of those,
    fraction with the same pitch present at that onset."""
    bt = np.array([t for t, _ in b])
    bp = {}
    for t, p in b:
        bp.setdefault(round(t, 3), set()).add(p)
    hit = same = 0
    for t, p in a:
        i = np.searchsorted(bt, t)
        cands = [j for j in (i - 1, i) if 0 <= j < len(bt) and abs(bt[j] - t) <= tol]
        if cands:
            hit += 1
            if any(p in bp[round(bt[j], 3)] for j in cands):
                same += 1
    return hit / max(1, len(a)), same / max(1, hit)


def main():
    model = Model(ICASSP_2022_MODEL_PATH)
    for name in PICK:
        folder = SONGS / name
        y = load(folder)
        if y is None:
            print("skip", name); continue
        y = y[: SR * 90]  # first 90 s is plenty
        base = events(model, y, "a")
        for shift_ms in (50, 130, 1000):
            pad = np.zeros(int(SR * shift_ms / 1000), dtype=np.float32)
            shifted = events(model, np.concatenate([pad, y]), "b")
            shifted = [(t - shift_ms / 1000, p) for t, p in shifted]
            r1, s1 = compare(base, shifted)
            r2, s2 = compare(shifted, base)
            print(f"{name[:40]:40s} shift {shift_ms:4d} ms: notes {len(base)} vs {len(shifted)}; "
                  f"onset agreement a->b {r1:.2f} b->a {r2:.2f}; pitch same {s1:.2f}", flush=True)


if __name__ == "__main__":
    main()
