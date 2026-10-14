"""Research-only: low distorted guitar sits at the bottom of Basic Pitch's
range where its CQT front end has the worst time resolution.  Play the stem
back at 2x (header trick: same samples, double the sample rate) so it lands an
octave higher and twice as fast, transcribe, then map times x2 and pitch -12.
Also try 0.5x (octave down, twice as slow) for completeness.
Compare human-chart coverage on the real guitar stem against the normal pass.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import soundfile as sf

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
from bp_sweep import coverage, onset_times_from_events  # noqa: E402

SR = 22050; FPS = SR / 256
PICK = ["Van Halen - Panama (Harmonix)", "Metallica - Enter Sandman",
        "Judas Priest - Halls of Valhalla (Harmonix)", "The Red Chord - Hour of Rats (Harmonix)",
        "Ozzy Osbourne - Miracle Man (Harmonix)", "Creed - Higher"]


def run(model, y, header_sr, factor):
    tmp = SCRATCH / "_spd.wav"
    sf.write(str(tmp), y, header_sr)
    out = run_inference(str(tmp), model)
    tmp.unlink(missing_ok=True)
    res = {}
    for cfg, mnl in (("D", 127.7), ("B", 23.0)):
        _, ev = model_output_to_notes(out, onset_thresh=0.5, frame_thresh=0.3,
                                      min_note_len=int(np.round(mnl / 1000 * FPS)),
                                      include_pitch_bends=False)
        ev = [(s * factor, e * factor, p, a) for s, e, p, a, *_ in ev]
        res[cfg] = ev
    return res


def main():
    model = Model(ICASSP_2022_MODEL_PATH)
    for name in PICK:
        y = load(SONGS / name)
        if y is None:
            print("skip", name); continue
        chart = ROOT / "data" / "converted" / f"{name}.chart"
        if not chart.is_file():
            chart = SONGS / name / "notes.chart"
        human = np.array(sorted({round(t, 4) for t, _ in chart_note_times(str(chart))}))
        variants = {"1x": run(model, y, SR, 1.0), "2x": run(model, y, SR * 2, 2.0), "0.5x": run(model, y, SR // 2, 0.5)}
        for cfg in ("D", "B"):
            line = f"{name[:34]:34s} {cfg}:"
            for v, res in variants.items():
                det = onset_times_from_events(res[cfg])
                r = coverage(human, det, 0.080)[0]; p = coverage(det, human, 0.080)[0]
                line += f"  {v}: rec80 {r:.3f} prec80 {p:.3f} n={len(det)}"
            print(line, flush=True)
        # union of 1x and 2x
        for cfg in ("D", "B"):
            det = onset_times_from_events(variants["1x"][cfg] + variants["2x"][cfg])
            print(f"{'':34s} {cfg}: union 1x+2x rec80 {coverage(human, det, 0.080)[0]:.3f} prec80 {coverage(det, human, 0.080)[0]:.3f} n={len(det)}", flush=True)


if __name__ == "__main__":
    main()
