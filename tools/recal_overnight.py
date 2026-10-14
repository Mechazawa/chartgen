"""Overnight controlled recalibration: Demucs vs BS-RoFormer SW.

Same songs, same ground truth, same sweep code; only the separator differs.
Phases, cheapest-certain first so an early stop still yields answers:

  1. solos: re-dump stem features under SW for the 300 cached solo songs
     (the Demucs features already exist), then sweep both.
  2. taps:  dump foreground features under BOTH backends over the 800-song
     library in seeded random order (any prefix is a random sample),
     then sweep both.

Everything is logged to work/recal_overnight.log and the final numbers to
work/recal_report.txt. To run it detached from the terminal:

    nohup .venv/bin/python -u tools/recal_overnight.py --taps-limit 800 \
        > work/recal_overnight.log 2> work/recal_overnight.err &
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WORK = ROOT / "work"
PY = sys.executable


def run(cmd, env=None, capture=False):
    print(f"\n$ {' '.join(str(c) for c in cmd)}", flush=True)
    full = dict(os.environ)
    full.update(env or {})
    t0 = time.time()
    res = subprocess.run([str(c) for c in cmd], env=full, cwd=str(ROOT),
                         capture_output=capture, text=True)
    print(f"  -> exit {res.returncode} after {(time.time() - t0) / 60:.1f} min",
          flush=True)
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--taps-limit", type=int, default=800)
    ap.add_argument("--skip-solos", action="store_true")
    ap.add_argument("--skip-taps", action="store_true")
    args = ap.parse_args(argv)
    WORK.mkdir(exist_ok=True)
    report = [f"recalibration started {time.strftime('%Y-%m-%d %H:%M')}"]

    if not args.skip_solos:
        for tag in ("A", "B"):
            out = WORK / f"cal_stems_{tag}_sw.json"
            # No exists-skip: both dump tools resume from their own output,
            # so a relaunch after a pause continues where it stopped.
            run([PY, ROOT / "tools/dump_stem_features.py",
                 WORK / f"cal_solo_{tag}.json", "-o", out],
                env={"CHARTGEN_STEMS": "sw"})
        for tag_arg, label in (([], "demucs"), (["--stems-tag", "sw"], "sw")):
            res = run([PY, ROOT / "tools/sweep_solos_stems.py", *tag_arg],
                      capture=True)
            report.append(f"\n===== SOLO SWEEP ({label}) =====\n{res.stdout}")
            print(res.stdout, flush=True)

    if not args.skip_taps:
        dump = WORK / "foreground_ab.json"
        run([PY, ROOT / "tools/dump_foreground.py", "-o", dump,
             "--limit", args.taps_limit])
        res = run([PY, ROOT / "tools/sweep_foreground.py", dump], capture=True)
        report.append(f"\n===== TAPS FOREGROUND SWEEP =====\n{res.stdout}")
        print(res.stdout, flush=True)

    report.append(f"\nfinished {time.strftime('%Y-%m-%d %H:%M')}")
    (WORK / "recal_report.txt").write_text("\n".join(report), encoding="utf-8")
    print("\nreport -> work/recal_report.txt", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
