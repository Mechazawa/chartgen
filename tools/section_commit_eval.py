"""Section commitment, offline, against human note times (test env only:
nothing here is wired into the pipeline).

For every followed window (stem in STEMS, audible, playing a line, conf >=
gate) two moves are simulated on the transcription supply that feeds
Expert selection:
  supply - admit the followed stem's events on every empty 16th tick in
           the window, not only where the window is starved (keyed rescue
           already does the starved half; this is the rest);
  cull   - drop mix events whose 16th tick the followed stem does not play.
Scored on unique positions: precision (share within 80 ms of a human note),
human coverage, and density relative to the human chart inside committed
windows. Baseline = mix + shipped keyed rescue.

    python tools/section_commit_eval.py            # totals per variant
    python tools/section_commit_eval.py --songs    # per song, best variant
"""
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from chartgen import prominence, transcribe  # noqa: E402
from chartgen.tempo import TempoMap  # noqa: E402

TOL = 0.08
CACHE = Path(__file__).resolve().parent.parent / "work" / "keyed_cache"
MIN_PITCH, MIN_AMP = 40, 0.20


def near_mask(a, b):
    a = np.asarray(a, float)
    if not len(a):
        return np.zeros(0, bool)
    if not len(b):
        return np.zeros(len(a), bool)
    b = np.sort(np.asarray(b, float))
    i = np.searchsorted(b, a)
    lo = np.abs(a - b[np.clip(i - 1, 0, len(b) - 1)])
    hi = np.abs(a - b[np.clip(i, 0, len(b) - 1)])
    return np.minimum(lo, hi) <= TOL


def eligible(w, stem_events, tempo, stems, gate):
    stem = w.get("stem")
    if stem not in stems or stem not in stem_events or w.get("conf", 0) < gate:
        return None
    f = w.get("features")
    if not f or f[stem]["energy"] < prominence.KEYED_MIN_SHARE:
        return None
    t0, t1 = w["t0"], w["t1"]
    pos = {tempo.quantize(s0, subdiv=4) for s0, _, p, a in stem_events[stem]
           if t0 <= s0 < t1 and p >= MIN_PITCH and a >= MIN_AMP}
    if len(pos) / max(1e-6, t1 - t0) < prominence.KEYED_MIN_STEM_PER_S:
        return None
    return stem, pos


def commit(c, stems, gate, supply, cull, budget=prominence.KEYED_MAX_PER_BEAT):
    """Return (events, committed windows, tempo) after the chosen moves."""
    tempo = TempoMap(beat_times=c["beat_times"], pickup_beats=c["pickup"])
    events = list(c["events"])
    keyed = prominence.keyed_rescue_events(events, c["windows"], c["stem_events"], tempo)[0] \
        if c["windows"] else []
    events += keyed
    committed = []
    drop = set()
    add = []
    for w in c["windows"]:
        el = eligible(w, c["stem_events"], tempo, stems, gate)
        if el is None:
            continue
        stem, pos = el
        committed.append(w)
        t0, t1 = w["t0"], w["t1"]
        if cull:
            for i, (s0, _, p, a) in enumerate(events):
                if t0 <= s0 < t1 and p >= MIN_PITCH and a >= MIN_AMP \
                        and tempo.quantize(s0, subdiv=4) not in pos:
                    drop.add(i)
        if supply:
            kept = [e for i, e in enumerate(events) if i not in drop]
            extra = transcribe.admit_stem_events(c["stem_events"][stem], [(t0, t1)], kept + add,
                                                 tempo, MIN_PITCH, prominence.KEYED_ADMIT_AMPLITUDE)
            if extra:
                lo, hi = tempo.quantize(t0, subdiv=4), tempo.quantize(t1, subdiv=4)
                present = len({tempo.quantize(s0, subdiv=4) for s0, _, p, a in kept
                               if lo <= tempo.quantize(s0, subdiv=4) < hi
                               and p >= MIN_PITCH and a >= MIN_AMP})
                allowed = max(0, int(budget * (hi - lo) / tempo.resolution) - present)
                ticks = {}
                for e in extra:
                    ticks.setdefault(tempo.quantize(e[0], subdiv=4), []).append(e)
                if len(ticks) > allowed:
                    keep = sorted(ticks, key=lambda tk: -max(e[3] for e in ticks[tk]))[:allowed]
                    extra = [e for tk in keep for e in ticks[tk]]
                add += extra
    out = [e for i, e in enumerate(events) if i not in drop] + add
    return out, committed, tempo


def positions(events, tempo):
    seen = {}
    for s0, _, p, a in events:
        if p >= MIN_PITCH and a >= MIN_AMP:
            seen.setdefault(tempo.quantize(s0, subdiv=4), s0)
    return np.array(sorted(seen.values()))


def score(c, stems, gate, supply, cull):
    ev, committed, tempo = commit(c, stems, gate, supply, cull)
    t = positions(ev, tempo)
    h = np.asarray(c["human"])
    if committed:
        inw = np.zeros(len(t), bool)
        inh = np.zeros(len(h), bool)
        for w in committed:
            inw |= (t >= w["t0"]) & (t < w["t1"])
            inh |= (h >= w["t0"]) & (h < w["t1"])
        tw, hw = t[inw], h[inh]
    else:
        tw, hw = t[:0], h[:0]
    return {
        "song_P": near_mask(t, h).mean() if len(t) else 0.0,
        "song_cov": near_mask(h, t).mean() if len(h) else 0.0,
        "win_P": near_mask(tw, hw).mean() if len(tw) else float("nan"),
        "win_cov": near_mask(hw, tw).mean() if len(hw) else float("nan"),
        "win_density": len(tw) / max(1, len(hw)),
        "n_win": len(committed), "n_pos": len(t), "n_win_pos": len(tw), "n_win_h": len(hw),
    }


VARIANTS = [("baseline (mix + keyed rescue)", False, False),
            ("supply everywhere", True, False),
            ("cull only", False, True),
            ("supply + cull", True, True)]


def main(argv):
    caches = [pickle.loads(p.read_bytes()) for p in sorted(CACHE.glob("*.pkl"))]
    per_song = "--songs" in argv
    stem_sets = [("guitar+piano+synth", prominence.KEYED_STEMS),
                 ("+bass", prominence.KEYED_STEMS + ("bass",))]
    print(f"{len(caches)} cached songs. Columns: whole-song precision / coverage; inside "
          f"committed windows precision / coverage / our positions per human note; windows.")
    print()
    for sname, stems in stem_sets:
        for gate in (0.60, 0.80, 0.95):
            print(f"== stems {sname}, conf >= {gate:.2f}")
            for label, supply, cull in VARIANTS:
                rows = [score(c, stems, gate, supply, cull) for c in caches]
                agg = {k: np.nanmedian([r[k] for r in rows]) for k in rows[0]
                       if k.startswith(("song", "win"))}
                pos = sum(r["n_win_pos"] for r in rows)
                hh = sum(r["n_win_h"] for r in rows)
                print(f"  {label:32s} song P {agg['song_P']:.2f} cov {agg['song_cov']:.2f} | "
                      f"win P {agg['win_P']:.2f} cov {agg['win_cov']:.2f} dens {pos / max(1, hh):.2f} | "
                      f"win/song {np.median([r['n_win'] for r in rows]):.0f}")
                if per_song and label == "supply everywhere":
                    base = [score(c, stems, gate, False, False) for c in caches]
                    for c, r, b in zip(caches, rows, base, strict=True):
                        print(f"      {c['song'][:36]:36s} P {b['song_P']:.2f}->{r['song_P']:.2f} "
                              f"cov {b['song_cov']:.2f}->{r['song_cov']:.2f} | win P {b['win_P']:.2f}->{r['win_P']:.2f} "
                              f"cov {b['win_cov']:.2f}->{r['win_cov']:.2f} dens {b['win_density']:.2f}->{r['win_density']:.2f} "
                              f"({r['n_win']} win)")
            print()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
