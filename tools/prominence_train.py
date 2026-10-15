"""Train the Route B prominence prior: per-window audio features -> which
stem the charter followed (labels from tools/prominence_pilot.py).

Labels use rule v2 (pilot QA, 2026-09-03): coincidence-split recall times
a prior (bass 0.7, vocals 0.6), a stem eligible only when audible
(energy share >= 0.05) and not noise (precision >= 0.15); a window is
labeled when the best score >= 0.25 and beats the runner-up by >= 0.06.
F1 labels favoured sparse bass on coincidences.

Features are AUDIO ONLY (per stem: energy share, K-weighted loudness
share, onset density, median pitch, monophonic share; plus drums energy
share and neighbour-window context) - never the onset-match scores,
which are computed against the human chart and would leak the label.

Validation is by SONG (GroupKFold), never by window. The bar the
research set: beat the 0.60-energy-share heuristic's precision at the
heuristic's own commit rate. Also prints a learning curve by number of
training songs, so the 300-vs-800 question is answered by data.

    python tools/prominence_train.py work/routeb_dump.json
"""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from chartgen.prominence import FEATS, STEMS, feature_vector  # noqa: E402

PRIOR = {"bass": 0.7, "vocals": 0.6, "guitar": 1.0, "piano": 1.0, "sw_other": 1.0}


def label_v2(w, min_energy=0.05, min_prec=0.15, min_score=0.25, gap=0.06):
    if w["label"] == "none" or not w.get("features"):
        return None
    sc = {}
    for s in STEMS:
        v, e = w["scores"][s], w["features"][s]["energy"]
        if e < min_energy or v["precision"] < min_prec:
            continue
        sc[s] = v["recall"] * PRIOR[s]
    if not sc:
        return None
    ranked = sorted(sc.items(), key=lambda kv: -kv[1])
    best, second = ranked[0], (ranked[1][1] if len(ranked) > 1 else 0.0)
    return best[0] if best[1] >= min_score and best[1] - second >= gap else None


def vector(w, prev, nxt):
    """The runtime feature vector (chartgen.prominence), so training and
    inference can never drift apart."""
    return feature_vector(w["features"],
                          prev["features"] if prev and prev.get("features") else None,
                          nxt["features"] if nxt and nxt.get("features") else None)


def heuristic(w, key, thresh):
    """The 0.60-share energy heuristic (or loudness variant): commit to the
    stem holding >= thresh of the share, else abstain."""
    f = w["features"]
    best = max(STEMS, key=lambda s: f[s][key])
    return best if f[best][key] >= thresh else None


def main(argv=None):
    import argparse

    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import GroupKFold

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dump", type=Path)
    ap.add_argument("--save", type=Path, help="fit on ALL labeled windows and joblib-dump here")
    args = ap.parse_args(argv)
    path = args.dump
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    X, y, groups, wins = [], [], [], []
    for gi, r in enumerate(rows):
        ws = r["windows"]
        for i, w in enumerate(ws):
            lab = label_v2(w)
            if lab is None:
                continue
            X.append(vector(w, ws[i - 1] if i > 0 else None, ws[i + 1] if i + 1 < len(ws) else None))
            y.append(STEMS.index(lab))
            groups.append(gi)
            wins.append(w)
    X, y, groups = np.array(X), np.array(y), np.array(groups)
    print(f"{len(rows)} songs, {len(y)} labeled windows: "
          + ", ".join(f"{STEMS[k]} {v}" for k, v in sorted(Counter(y).items())))

    gkf = GroupKFold(n_splits=5)
    proba = np.zeros((len(y), len(STEMS)))
    for tr, te in gkf.split(X, y, groups):
        clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_depth=6,
                                             l2_regularization=1.0, random_state=0)
        clf.fit(X[tr], y[tr])
        p = clf.predict_proba(X[te])
        proba[np.ix_(te, clf.classes_)] = p
    pred = proba.argmax(axis=1)
    conf = proba.max(axis=1)
    acc = (pred == y).mean()
    print(f"\n=== held-out (song-level 5-fold) ===\n  accuracy on labeled windows: {acc:.1%}")
    for k, s in enumerate(STEMS):
        tp = ((pred == k) & (y == k)).sum()
        p = tp / max(1, (pred == k).sum())
        r = tp / max(1, (y == k).sum())
        print(f"  {s:9s} precision {p:.0%} recall {r:.0%}  (n={int((y == k).sum())})")

    print("\n=== the bar: precision at the energy heuristic's own commit rate ===")
    for key, thresh in (("energy", 0.60), ("loudness", 0.60), ("energy", 0.50)):
        h = np.array([STEMS.index(v) if (v := heuristic(w, key, thresh)) else -1 for w in wins])
        commit = (h >= 0).mean()
        hp = (h[h >= 0] == y[h >= 0]).mean() if commit else 0.0
        # model at the same commit rate: commit on its most confident windows
        tau = np.quantile(conf, 1 - commit) if commit else 1.0
        m = conf >= tau
        mp = (pred[m] == y[m]).mean() if m.any() else 0.0
        print(f"  {key} >= {thresh:.2f}: commits on {commit:.0%} of windows at {hp:.0%} precision  |  "
              f"model at the same {m.mean():.0%} commit rate: {mp:.0%} precision (confidence >= {tau:.2f})")
    print("\n=== model precision vs commit rate ===")
    for q in (0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3):
        tau = np.quantile(conf, 1 - q)
        m = conf >= tau
        print(f"  commit {q:.0%} (conf >= {tau:.2f}): precision {(pred[m] == y[m]).mean():.0%}")

    print("\n=== circularity check: contested windows (>=2 stems audible AND transcribed) ===")
    def contested(w):
        f = w["features"]
        return sum(1 for s in STEMS if f[s]["energy"] >= 0.05 and f[s]["density"] >= 1.0) >= 2
    c = np.array([contested(w) for w in wins])
    dens = np.array([STEMS.index(max(STEMS, key=lambda s: w["features"][s]["density"])) for w in wins])
    ener = np.array([STEMS.index(max(STEMS, key=lambda s: w["features"][s]["energy"])) for w in wins])
    loud = np.array([STEMS.index(max(STEMS, key=lambda s: w["features"][s]["loudness"])) for w in wins])
    print(f"  contested windows: {c.sum()} of {len(y)} ({c.mean():.0%})")
    for name, arr in (("model", pred), ("argmax density", dens), ("argmax energy", ener), ("argmax loudness", loud)):
        print(f"  {name:16s} accuracy all {(arr == y).mean():.1%}   contested {(arr[c] == y[c]).mean():.1%}   "
              f"uncontested {(arr[~c] == y[~c]).mean():.1%}")
    # model WITHOUT any transcription-derived feature (density, pitch, mono):
    # level and timbre only - the prominence signal proper
    all_names = ([f"{s}.{k}" for s in STEMS for k in FEATS] + ["drums.energy"]
                 + [f"{side}.{s}.{k}" for side in ("prev", "next") for s in STEMS for k in ("energy", "loudness")])
    keep_idx = [i for i, nm in enumerate(all_names)
                if not any(nm.endswith(t) for t in (".density", ".pitch", ".mono"))]
    Xl = X[:, keep_idx]
    pl = np.zeros(len(y), dtype=int)
    for tr, te in gkf.split(Xl, y, groups):
        clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_depth=6,
                                             l2_regularization=1.0, random_state=0).fit(Xl[tr], y[tr])
        pl[te] = clf.predict(Xl[te])
    print(f"  {'level-only model':16s} accuracy all {(pl == y).mean():.1%}   contested {(pl[c] == y[c]).mean():.1%}   "
          f"uncontested {(pl[~c] == y[~c]).mean():.1%}")
    for r in rows:
        if r["song"].startswith("Perturbator"):
            for w in r["windows"]:
                if 235 <= w["t0"] < 300 and label_v2(w) is not None:
                    j = wins.index(w)
                    print(f"  Sexualizer {w['t0']:.0f}-{w['t1']:.0f}s label {STEMS[y[j]]:8s} model {STEMS[pred[j]]:8s} "
                          f"conf {conf[j]:.2f}  level-only {STEMS[pl[j]]}")

    print("\n=== learning curve: held-out accuracy vs number of training songs ===")
    rng = np.random.RandomState(0)
    for n_train in (50, 100, 150, 200, 240):
        accs = []
        for tr, te in gkf.split(X, y, groups):
            tr_songs = np.unique(groups[tr])
            if n_train > len(tr_songs):
                continue
            keep = set(rng.choice(tr_songs, n_train, replace=False))
            sub = np.array([i for i in tr if groups[i] in keep])
            clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_depth=6,
                                                 l2_regularization=1.0, random_state=0)
            clf.fit(X[sub], y[sub])
            accs.append((clf.predict(X[te]) == y[te]).mean())
        if accs:
            print(f"  {n_train:3d} training songs: {np.mean(accs):.1%} (+-{np.std(accs):.1%})")

    print("\n=== feature importance (permutation, one fold) ===")
    from sklearn.inspection import permutation_importance
    tr, te = next(iter(gkf.split(X, y, groups)))
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_depth=6,
                                         l2_regularization=1.0, random_state=0).fit(X[tr], y[tr])
    names = [f"{s}.{k}" for s in STEMS for k in FEATS] + ["drums.energy"] + \
            [f"{side}.{s}.{k}" for side in ("prev", "next") for s in STEMS for k in ("energy", "loudness")]
    imp = permutation_importance(clf, X[te], y[te], n_repeats=5, random_state=0, n_jobs=1)
    order = np.argsort(-imp.importances_mean)[:12]
    for i in order:
        print(f"  {names[i]:22s} {imp.importances_mean[i]:+.3f}")
    if args.save:
        import joblib

        clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_depth=6,
                                             l2_regularization=1.0, random_state=0).fit(X, y)
        joblib.dump({"model": clf, "stems": list(STEMS), "feats": list(FEATS), "version": 1,
                     "songs": len(rows), "windows": int(len(y)), "held_out_accuracy": float(acc)},
                    args.save)
        print(f"\nsaved model ({len(y)} windows, {len(rows)} songs) -> {args.save} "
              f"({args.save.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
