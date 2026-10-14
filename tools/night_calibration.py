"""Overnight calibration run: fetch a quality-filtered chart set, then
extract every feature the threshold sweeps need.

Stages, sequential so the network hour finishes before the CPU hours:

1. fetch ~800 full-ladder, issue-free charts from Chorus Encore
   (~2h, ~3GB; audio is KEPT so future features never re-download)
2. chart-only studies on the new set (minutes: solo conventions, tap
   usage, lyric-gap sweep - these print to the log for the morning)
3. solo evidence dump, two workers (~3-4h)
4. tap feature dump, two workers (~1.5-2h)

Everything lands under work/cal_*; the morning job is just re-running
sweep_solos.py / sweep_taps.py against the new caches next to the old.

    python tools/night_calibration.py            # full run
    python tools/night_calibration.py --skip-fetch   # data already pulled
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PYTHON = ROOT / ".venv/bin/python"
TOOLS = ROOT / "tools"
CAL = ROOT / "data/calibration"
WORK = ROOT / "work"


def run(label: str, argv: list, log_name: str) -> int:
    log = WORK / log_name
    print(f"[{time.strftime('%H:%M:%S')}] {label} -> {log.name}", flush=True)
    with log.open("w", encoding="utf-8") as sink:
        code = subprocess.call([str(PYTHON), *map(str, argv)],
                               cwd=str(TOOLS), stdout=sink, stderr=subprocess.STDOUT)
    print(f"[{time.strftime('%H:%M:%S')}] {label} finished (exit {code})", flush=True)
    return code


def run_pair(label: str, script: str, out_stem: str, total: int, extra: list = ()) -> None:
    """Two worker processes over a --start/--end split of the same tool."""
    half = total // 2
    procs = []
    for tag, start, end in (("A", 0, half), ("B", half, total)):
        log = (WORK / f"{out_stem}_{tag}.log").open("w", encoding="utf-8")
        argv = [str(PYTHON), str(TOOLS / script), str(CAL),
                "-o", str(WORK / f"{out_stem}_{tag}.json"),
                "--start", str(start), "--end", str(end), *map(str, extra)]
        procs.append((subprocess.Popen(argv, cwd=str(TOOLS), stdout=log,
                                       stderr=subprocess.STDOUT), log))
    print(f"[{time.strftime('%H:%M:%S')}] {label}: 2 workers running", flush=True)
    for proc, log in procs:
        proc.wait()
        log.close()
    print(f"[{time.strftime('%H:%M:%S')}] {label} finished", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--count", type=int, default=800)
    ap.add_argument("--skip-fetch", action="store_true")
    args = ap.parse_args(argv)

    WORK.mkdir(exist_ok=True)
    started = time.time()

    if not args.skip_fetch:
        code = run("fetch charts", [TOOLS / "fetch_charts.py",
                                    "-o", CAL, "-n", args.count], "cal_fetch.log")
        if code != 0:
            print("fetch failed; continuing with whatever landed", flush=True)

    # Chart-only measurements: cheap, and the morning's first read.
    run("solo conventions (chart-only)",
        [TOOLS / "study_solos.py", CAL], "cal_study_solos.log")
    run("tap usage (chart-only)",
        [TOOLS / "study_taps.py", CAL], "cal_study_taps.log")

    # The audio hours. Solo evidence first: it feeds the higher-stakes rule.
    run_pair("solo evidence dump", "dump_solo_features.py", "cal_solo",
             total=300, extra=["--limit", "300"])
    run_pair("tap feature dump", "dump_tap_features.py", "cal_taps", total=250)

    hours = (time.time() - started) / 3600
    print(f"\nall stages done in {hours:.1f}h. Morning: re-run sweep_solos.py "
          f"and sweep_taps.py over work/cal_*.json plus the original caches.",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
