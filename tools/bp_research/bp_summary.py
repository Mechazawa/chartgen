"""Aggregate bp_sweep.jsonl: medians per config and source, paired deltas vs
the default decode (D), split multitrack (real guitar stem) vs mix-only."""
import json
from pathlib import Path

import numpy as np

rows = [json.loads(l) for l in (Path(__file__).resolve().parents[2] / "out" / "bp_research" / "bp_sweep.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
print(f"{len(rows)} songs; multitrack {sum(r['multitrack'] for r in rows)}")


def med(vals):
    vals = [v for v in vals if v is not None]
    return f"{np.median(vals):.3f}" if vals else "  -  "


def table(source, subset):
    subset = [r for r in subset if source in r]
    if not subset:
        return
    print(f"\n== source={source}  n={len(subset)} ==")
    print(f"{'cfg':>14s} {'rec50':>6s} {'rec80':>6s} {'prec80':>6s} {'det/human':>9s} {'|err|ms':>7s} {'bias':>6s} {'d_rec80 vs D (med, p25, p75)':>30s} {'wins':>5s}")
    keys = list(subset[0][source]["configs"].keys()) + ["onsetpost_0.1", "onsetpost_0.25", "specflux"]
    for k in keys:
        get = (lambda r, k=k: r[source]["configs"][k]) if k in subset[0][source]["configs"] else (lambda r, k=k: r[source][k])
        base = lambda r: r[source]["configs"]["D"]
        d = np.array([get(r)["recall80"] - base(r)["recall80"] for r in subset])
        wins = int((d > 0.01).sum())
        print(f"{k:>14s} {med(get(r)['recall50'] for r in subset):>6s} {med(get(r)['recall80'] for r in subset):>6s} "
              f"{med(get(r)['precision80'] for r in subset):>6s} {med(get(r)['n_det']/r['n_human'] for r in subset):>9s} "
              f"{med(get(r)['abs_err_med_ms'] for r in subset):>7s} {med(get(r)['bias_med_ms'] for r in subset):>6s} "
              f"{np.median(d):+.3f} {np.percentile(d,25):+.3f} {np.percentile(d,75):+.3f}{'':>9s} {wins:>3d}/{len(subset)}")


multi = [r for r in rows if r["multitrack"]]
single = [r for r in rows if not r["multitrack"]]
table("guitar", multi)
table("mix", multi)
table("mix", single)
table("mix", rows)

# guitar stem vs mix on the same songs (default decode and B)
if multi:
    print("\n== real guitar stem vs mix, same songs (recall80) ==")
    for k in ("D", "B"):
        d = np.array([r["guitar"]["configs"][k]["recall80"] - r["mix"]["configs"][k]["recall80"] for r in multi])
        print(f"  {k}: median {np.median(d):+.3f}, stem better in {(d>0.01).sum()}/{len(d)}, mix better in {(d<-0.01).sum()}")
    print("\n== per-song: human notes/s, D and B recall80 on guitar stem, and F (near-ceiling) ==")
    for r in sorted(multi, key=lambda r: r["guitar"]["configs"]["D"]["recall80"]):
        g = r["guitar"]["configs"]; m = r["mix"]["configs"]
        print(f"  {r['song'][:44]:44s} nps={r['n_human']/r['duration_s']:4.1f} gtr D {g['D']['recall80']:.2f} B {g['B']['recall80']:.2f} F {g['F']['recall80']:.2f} | mix D {m['D']['recall80']:.2f} B {m['B']['recall80']:.2f}")
if single:
    print("\n== mix-only songs (community charts), worst D recall80 first ==")
    for r in sorted(single, key=lambda r: r["mix"]["configs"]["D"]["recall80"])[:25]:
        m = r["mix"]["configs"]
        print(f"  {r['song'][:44]:44s} nps={r['n_human']/r['duration_s']:4.1f} D {m['D']['recall80']:.2f} B {m['B']['recall80']:.2f} C {m['C']['recall80']:.2f} F {m['F']['recall80']:.2f} post0.1 {r['mix']['onsetpost_0.1']['recall80']:.2f}")
