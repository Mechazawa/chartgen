"""Solo detection from the followed-instrument timeline: offline evaluation.

The shipped guitar-stem solo rule holds 82% precision at 6% recall with
~2 false fires per 100 solo-less songs; its misses are granularity
(chroma sections merge solos with neighbours, 38%) and the singing gate
those wide sections trip (38%). The Route B dump has, per 4-bar window,
every solo signal at 4-bar granularity: guitar density, energy and
loudness share, register, monophony, vocal energy share, plus the
followed-stem probabilities. This labels each window by overlap with
the human charts' `E solo` spans, trains a song-fold model, turns window
probabilities into runs, and scores at the SOLO level against the bar
the shipped rule had to clear: precision >= 80%, quiet fires <= 2%.

    python tools/solo_windows.py work/routeb_dump.json
"""
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from chartgen.prominence import STEMS, load_model, solo_vector, feature_vector  # noqa: E402

CAL = Path(__file__).resolve().parent.parent / "data/calibration"
BPM_RE = re.compile(r"(\d+)\s*=\s*B\s*(\d+)")
SOLO_RE = re.compile(r"(\d+)\s*=\s*E\s*(soloend|solo)\b")


def solo_spans_s(chart: Path):
    txt = chart.read_text(encoding="utf-8-sig", errors="replace")
    res = int(re.search(r"Resolution\s*=\s*(\d+)", txt).group(1))
    sync = re.search(r"\[SyncTrack\]\s*\{(.*?)\}", txt, re.S)
    body = re.search(r"\[ExpertSingle\]\s*\{(.*?)\}", txt, re.S)
    if not (sync and body):
        return []
    bpms = [(int(a), int(b) / 1000.0) for a, b in BPM_RE.findall(sync.group(1))] or [(0, 120.0)]

    def t2s(tick):
        sec, lt, lb = 0.0, 0, bpms[0][1]
        for t, b in bpms:
            if t >= tick:
                break
            sec += (t - lt) / res * 60.0 / lb
            lt, lb = t, b
        return sec + (tick - lt) / res * 60.0 / lb

    spans, start = [], None
    for m in SOLO_RE.finditer(body.group(1)):
        tick, kind = int(m.group(1)), m.group(2)
        if kind == "solo":
            start = tick
        elif start is not None:
            if tick > start:
                spans.append((t2s(start), t2s(tick)))
            start = None
    return spans


def overlap(a0, a1, b0, b1):
    return max(0.0, min(a1, b1) - max(a0, b0))


def main(argv=None):
    import argparse

    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import GroupKFold

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dump", type=Path)
    ap.add_argument("--save", type=Path, help="fit on all windows and joblib-dump here")
    args = ap.parse_args(argv)
    path = args.dump
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    prom = load_model()["model"]
    X, y, groups, meta = [], [], [], []
    truth = {}
    for gi, r in enumerate(rows):
        chart = CAL / r["song"] / "notes.chart"
        if not chart.is_file():
            continue
        spans = solo_spans_s(chart)
        truth[gi] = spans
        ws = r["windows"]
        feats = []
        for w in ws:
            f = dict(w["features"]) if w.get("features") else None
            if f is not None:
                f["_t0"], f["_t1"] = w["t0"], w["t1"]
            feats.append(f)
        end = ws[-1]["t1"]
        vecs = []
        for i, f in enumerate(feats):
            if not f:
                vecs.append(None)
                continue
            vecs.append(feature_vector(f, feats[i - 1] if i > 0 else None,
                                       feats[i + 1] if i + 1 < len(feats) else None))
        probs = np.zeros((len(ws), len(STEMS)))
        idx = [i for i, v in enumerate(vecs) if v is not None]
        if idx:
            p = prom.predict_proba(np.array([vecs[i] for i in idx]))
            probs[np.ix_(idx, list(prom.classes_))] = p
        for i, w in enumerate(ws):
            if vecs[i] is None:
                continue
            X.append(solo_vector(feats, probs, i, end))
            ov = sum(overlap(w["t0"], w["t1"], a, b) for a, b in spans)
            y.append(int(ov >= 0.5 * (w["t1"] - w["t0"])))
            groups.append(gi)
            meta.append((gi, w["t0"], w["t1"]))
    X, y, groups = np.array(X), np.array(y), np.array(groups)
    n_songs = len(set(groups))
    n_solo_songs = sum(1 for g in set(groups) if truth[g])
    print(f"{n_songs} songs ({n_solo_songs} with solos, {sum(len(v) for v in truth.values())} solos), "
          f"{len(y)} windows, {int(y.sum())} solo windows ({y.mean():.1%})")

    gkf = GroupKFold(n_splits=5)
    prob = np.zeros(len(y))
    for tr, te in gkf.split(X, y, groups):
        clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_depth=5,
                                             l2_regularization=1.0, class_weight="balanced",
                                             random_state=0)
        clf.fit(X[tr], y[tr])
        prob[te] = clf.predict_proba(X[te])[:, 1]

    # window-level sanity
    order = np.argsort(-prob)
    top = order[: int(y.sum())]
    print(f"window-level: precision among the top-{len(top)} windows {y[top].mean():.0%}; "
          f"AUC-ish: mean rank of solo windows {np.mean([np.searchsorted(np.sort(-prob), -prob[i]) for i in np.where(y == 1)[0]]) / len(y):.2f} (0 = best)")

    # runs -> solos
    by_song = {}
    for (gi, t0, t1), p in zip(meta, prob, strict=True):
        by_song.setdefault(gi, []).append((t0, t1, p))
    print("\n=== solo-level (a predicted run hits a human solo when they overlap by >= half the shorter one) ===")
    print("shipped guitar-stem rule for reference: precision 82%  recall 6%  quiet fires ~2%  (held-out)")
    print(f"{'threshold':>9s} {'min run':>7s} {'precision':>9s} {'recall':>7s} {'quiet fires':>11s} {'start err':>9s} {'end err':>8s} {'pred/song':>9s}")
    best = None

    def make_runs(wins, tau, min_run, low, bridge):
        """Hysteresis runs: a run needs >= min_run windows at >= tau, then
        extends over neighbours at >= low, and bridges up to `bridge`
        windows below low if the run continues beyond them."""
        n = len(wins)
        core = [p >= tau for _, _, p in wins]
        soft = [p >= low for _, _, p in wins]
        runs, i = [], 0
        while i < n:
            if not core[i]:
                i += 1
                continue
            j = i
            while j + 1 < n and core[j + 1]:
                j += 1
            if j - i + 1 < min_run:
                i = j + 1
                continue
            a, b = i, j
            while a - 1 >= 0 and soft[a - 1]:
                a -= 1
            while b + 1 < n and soft[b + 1]:
                b += 1
            # bridge a short dip if a soft stretch resumes after it
            k = b + 1
            while k < n and not soft[k] and k - b <= bridge:
                k += 1
            if k < n and soft[k] and k - b <= bridge + 1 and k > b + 1:
                b = k
                while b + 1 < n and soft[b + 1]:
                    b += 1
            runs.append((wins[a][0], wins[b][1]))
            i = b + 1
        return runs

    for tau, low, bridge in ((0.6, 0.6, 0), (0.7, 0.7, 0), (0.8, 0.8, 0),
                             (0.7, 0.4, 0), (0.8, 0.4, 0), (0.9, 0.4, 0),
                             (0.7, 0.4, 1), (0.8, 0.4, 1), (0.9, 0.5, 1)):
        for min_run in (2, 3, 4):
            tp = fp = 0
            hit_solos = 0
            quiet_fires = 0
            start_err, end_err = [], []
            total_pred = 0
            for gi, wins in by_song.items():
                wins.sort()
                runs = make_runs(wins, tau, min_run, low, bridge)
                total_pred += len(runs)
                spans = truth[gi]
                if not spans and runs:
                    quiet_fires += 1
                matched = set()
                for r0, r1 in runs:
                    hit = None
                    for k, (a, b) in enumerate(spans):
                        if overlap(r0, r1, a, b) >= 0.5 * min(r1 - r0, b - a):
                            hit = k
                            break
                    if hit is None:
                        fp += 1
                    else:
                        tp += 1
                        if hit not in matched:
                            matched.add(hit)
                            start_err.append(abs(r0 - spans[hit][0]))
                            end_err.append(abs(r1 - spans[hit][1]))
                hit_solos += len(matched)
            n_solos = sum(len(v) for v in truth.values())
            n_quiet = sum(1 for g in by_song if not truth[g])
            prec = tp / max(1, tp + fp)
            rec = hit_solos / max(1, n_solos)
            qf = quiet_fires / max(1, n_quiet)
            print(f"{tau:4.2f}/{low:.2f}/{bridge} {min_run:7d} {prec:9.0%} {rec:7.0%} {qf:11.1%} "
                  f"{np.median(start_err) if start_err else 0:8.1f}s {np.median(end_err) if end_err else 0:7.1f}s "
                  f"{total_pred / len(by_song):9.2f}")
            if prec >= 0.80 and qf <= 0.02 and (best is None or rec > best[0]):
                best = (rec, (tau, low, bridge), min_run, prec, qf)
    if best:
        print(f"\nbest rule clearing the bar (P>=80%, quiet<=2%): enter/extend/bridge {best[1]}, min run {best[2]} -> "
              f"precision {best[3]:.0%}, recall {best[0]:.0%}, quiet fires {best[4]:.1%}")
    else:
        print("\nno rule clears the bar (P>=80%, quiet<=2%)")
    if args.save:
        import joblib

        clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_depth=5,
                                             l2_regularization=1.0, class_weight="balanced",
                                             random_state=0).fit(X, y)
        joblib.dump({"model": clf, "windows": int(len(y)), "songs": n_songs, "version": 1}, args.save)
        print(f"saved solo model -> {args.save} ({args.save.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
