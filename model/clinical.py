#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Kush Shah
# SPDX-License-Identifier: Apache-2.0
"""
Clinical-facing analysis: what the chip's behaviour means for a person.

The verification work answers "is the hardware correct?". This answers the
questions a clinician or a reviewer of biomedical work would actually ask:

  1. Does it work for the people it is aimed at?  Performance split by
     glycemic status (healthy / pre-diabetic / T2D), because a model that
     works only for healthy adults is useless for the population this project
     exists to serve.

  2. Does it rank meals correctly for one person?  R^2 measures variance
     explained across a population. The actual use -- "should I swap this
     ingredient?" -- is a WITHIN-PERSON RANKING question, and Spearman
     answers it directly. A model can have mediocre R^2 and still order a
     given person's meals well, or vice versa.

  3. What does the hardware cost in clinical terms?  Float reference versus
     the INT8 pipeline the silicon actually runs, in mg/dL.min.

  4. Is the R-4 truncation defect reachable with real data?  A bug that
     cannot occur in deployment is a different thing from one that can, and
     saying which requires measuring it rather than asserting it.

Subgroups follow the standard ADA A1c thresholds: < 5.7 % healthy,
5.7-6.4 % pre-diabetic, >= 6.5 % type 2 diabetes.
"""

import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'test'))
import golden_float as gf   # noqa: E402
import golden_quant as gq   # noqa: E402
from sklearn.metrics import mean_absolute_error, r2_score

from cgmacros_loader import FEATURES, load_all
from train import (CARB, CONSTRAINT_SETS, build, kfold_participants,
                   monotone_predict, standardise, train_mlp, train_monotone,
                   weights_of)

DEFAULT_ROOT = r"C:\Users\kushk\Downloads\Claude\cgmacros\csv"

A1C_PREDIABETES = 5.7
A1C_DIABETES = 6.5


def load_a1c(root):
    """participant id -> A1c, from bio.csv."""
    out = {}
    with open(os.path.join(root, "bio.csv"), newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            key = next((k for k in row if k.strip().lower() == "subject"), None)
            a1c = next((k for k in row if "a1c" in k.lower()), None)
            if key is None or a1c is None:
                continue
            try:
                out[int(float(row[key]))] = float(row[a1c])
            except (ValueError, TypeError):
                continue
    return out


def group_of(a1c):
    if a1c is None:
        return "unknown"
    if a1c < A1C_PREDIABETES:
        return "healthy"
    if a1c < A1C_DIABETES:
        return "pre-diabetic"
    return "T2D"


def spearman(a, b):
    """Rank correlation without scipy."""
    def rank(v):
        order = np.argsort(np.argsort(v))
        return order.astype(float)
    ra, rb = rank(np.asarray(a, float)), rank(np.asarray(b, float))
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def fit_oof(X, y, g, k=5, seeds=3, constraints=None):
    """Out-of-fold predictions: every participant held out exactly once.

    A single split leaves only a handful of participants per glycemic
    subgroup -- 4 healthy, 2 pre-diabetic, 5 T2D on this data -- which is far
    too few to report a per-group R^2 from. Cross-fitting gives a held-out
    prediction for all 45 participants and all 1,346 meals, so the subgroup
    and per-person numbers below rest on the whole cohort rather than on
    whoever happened to land in one test set.
    """
    folds = kfold_participants(g, k=k, seed=0)
    pids, true, pred = [], [], []

    for test_p in folds:
        m = np.array([p in test_p for p in g])
        Xtr, Xte, ytr, yte, (mu, sd, ym, ys) = standardise(
            X[~m], X[m], y[~m], y[m])

        if constraints is None:
            best, best_r2 = None, -9e9
            for s in range(seeds):
                mm = train_mlp(Xtr, ytr, seed=s)
                r2 = r2_score(yte, mm.predict(Xte))
                if r2 > best_r2:
                    best, best_r2 = mm, r2
            out = best.predict(Xte)
        else:
            _, _, W2u, _ = weights_of(train_mlp(Xtr, ytr, seed=0))
            dirs = [1.0 if v >= 0 else -1.0 for v in W2u[0]]
            best, best_r2 = None, -9e9
            for s in range(seeds):
                w = train_monotone(Xtr, ytr, seed=s,
                                   constraints=constraints, dirs=dirs)
                r2 = r2_score(yte, monotone_predict(Xte, *w))
                if r2 > best_r2:
                    best, best_r2 = w, r2
            out = monotone_predict(Xte, *best)

        # Back to mg/dL.min so every number below is in clinical units.
        pids.append(g[m])
        true.append(yte * ys + ym)
        pred.append(out * ys + ym)

    pids = np.concatenate(pids)
    true = np.concatenate(true)
    pred = np.concatenate(pred)
    return {"pids": pids, "y_true": true, "y_pred": pred,
            "r2": r2_score(true, pred)}


def by_subgroup(res, a1c):
    rows = []
    groups = {}
    for pid, t, p in zip(res["pids"], res["y_true"], res["y_pred"]):
        groups.setdefault(group_of(a1c.get(int(pid))), []).append((pid, t, p))
    for name in ("healthy", "pre-diabetic", "T2D", "unknown"):
        if name not in groups:
            continue
        items = groups[name]
        t = np.array([i[1] for i in items])
        p = np.array([i[2] for i in items])
        n_sub = len(set(i[0] for i in items))
        rows.append({
            "group": name,
            "participants": n_sub,
            "meals": len(items),
            "r2": r2_score(t, p) if len(t) > 2 else float("nan"),
            "mae": mean_absolute_error(t, p),
            "mean_iauc": t.mean(),
        })
    return rows


def within_participant_spearman(res, min_meals=5):
    """Does the model order ONE PERSON's meals correctly? The real use case."""
    out = []
    for pid in sorted(set(res["pids"].tolist())):
        m = res["pids"] == pid
        if m.sum() < min_meals:
            continue
        rho = spearman(res["y_true"][m], res["y_pred"][m])
        if not np.isnan(rho):
            out.append((int(pid), int(m.sum()), rho))
    return out


def quantise_cost_and_reach(X, y, g, k=5, seeds=3):
    """Sections 3 and 4: what quantisation costs, and whether R-4 is reachable.

    Both questions are asked with the SAME trained network and the SAME real
    meals, so the numbers are comparable and neither is a synthetic construct.
    """
    folds = kfold_participants(g, k=k, seed=0)
    test_p = folds[0]
    m = np.array([p in test_p for p in g])
    Xtr, Xte, ytr, yte, (mu, sd, ym, ys) = standardise(X[~m], X[m], y[~m], y[m])

    _, _, W2u, _ = weights_of(train_mlp(Xtr, ytr, seed=0))
    dirs = [1.0 if v >= 0 else -1.0 for v in W2u[0]]
    best, best_r2 = None, -9e9
    for s in range(seeds):
        w = train_monotone(Xtr, ytr, seed=s, constraints={CARB: +1}, dirs=dirs)
        r2 = r2_score(yte, monotone_predict(Xte, *w))
        if r2 > best_r2:
            best, best_r2 = w, r2

    W1n, b1n, W2n, b2n = best
    W1 = W1n.tolist()
    b1 = b1n.tolist()
    W2 = [W2n.tolist()]
    b2 = b2n.tolist()

    # --- float reference vs the INT8 pipeline the silicon runs -------------
    fl, qt = [], []
    for row in Xte:
        x = row.tolist()
        _, y_f = gf.mlp_float(W1, b1, W2, b2, x)
        l1, l2, meta = gf.to_chip_streams(W1, b1, W2, b2, x, **gf.BEST_SCALES)
        res = gq.mode_b(l1, l2, meta["s1"], meta["s2"])
        fl.append(y_f[0])
        qt.append(res["y"][0] / meta["y_scale"])

    fl = np.array(fl) * ys
    qt = np.array(qt) * ys
    quant = {
        "n": len(fl),
        "mae": float(np.abs(fl - qt).mean()),
        "p95": float(np.percentile(np.abs(fl - qt), 95)),
        "max": float(np.abs(fl - qt).max()),
        "iauc_scale": float(ys),
    }

    # --- is the R-4 truncation defect reachable on real meals? -------------
    # The host trades precision against range through s2. Smaller s2 keeps
    # more precision and pushes the value further up the field, so the
    # question is not "does it overflow" but "at which settings".
    reach = []
    for s2 in range(0, 7):
        over = 0
        for row in Xte:
            x = row.tolist()
            sc = dict(gf.BEST_SCALES, s2=s2)
            l1, l2, meta = gf.to_chip_streams(W1, b1, W2, b2, x, **sc)
            v = gq.mode_b(l1, l2, meta["s1"], meta["s2"])["y"][0]
            if v > 32767 or v < -32768:
                over += 1
        reach.append((s2, over, len(Xte)))
    return quant, reach



def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=DEFAULT_ROOT)
    args = ap.parse_args()

    a1c = load_a1c(args.root)
    X, y, g = build(args.root, None)

    print("=" * 72)
    print(" CLINICAL ANALYSIS -- CGMacros, all meal types")
    print("=" * 72)

    counts = {}
    for pid in sorted(set(g.tolist())):
        counts[group_of(a1c.get(int(pid)))] = counts.get(group_of(a1c.get(int(pid))), 0) + 1
    print("  cohort by A1c (ADA thresholds):")
    for k in ("healthy", "pre-diabetic", "T2D", "unknown"):
        if k in counts:
            print(f"    {k:14s} {counts[k]:3d} participants")
    print("    (dataset paper documents 15 healthy / 16 pre-diabetic / 14 T2D)")

    print()
    print("  [1] Held-out performance by glycemic status")
    res = fit_oof(X, y, g, constraints={CARB: +1})
    print(f"    overall out-of-fold R2 {res['r2']:+.3f}  ({len(res['y_true'])} meals, all 45 participants)")
    print(f"    {'group':14s} {'subj':>5s} {'meals':>6s} {'R2':>8s} "
          f"{'MAE':>9s} {'mean iAUC':>11s}")
    for r in by_subgroup(res, a1c):
        print(f"    {r['group']:14s} {r['participants']:5d} {r['meals']:6d} "
              f"{r['r2']:+8.3f} {r['mae']:9.0f} {r['mean_iauc']:11.0f}")

    print()
    print("  [2] Within-participant ranking (the actual use case)")
    print("      'for THIS person, does it order meals correctly?'")
    rhos = within_participant_spearman(res, min_meals=8)
    if rhos:
        vals = np.array([r[2] for r in rhos])
        print(f"    participants with >= 8 out-of-fold meals : {len(rhos)}")
        print(f"    Spearman rho  median {np.median(vals):+.3f}   "
              f"mean {vals.mean():+.3f}   IQR "
              f"{np.percentile(vals, 25):+.3f}..{np.percentile(vals, 75):+.3f}")
        print(f"    positive rho in {int((vals > 0).sum())}/{len(vals)} participants")
    else:
        print("    too few held-out meals per participant to rank")

    print()
    print("  [3] Quantisation cost: float reference vs the INT8 silicon path")
    quant, reach = quantise_cost_and_reach(X, y, g)
    print(f"    held-out meals compared : {quant['n']}")
    print(f"    |float - INT8|  mean {quant['mae']:7.1f}   "
          f"p95 {quant['p95']:7.1f}   max {quant['max']:7.1f}  mg/dL.min")
    print(f"    for context, mean iAUC across the cohort is "
          f"{float(np.mean(y)):.0f} mg/dL.min  (n={len(y)} meals)")

    print()
    print("  [4] Is the R-4 truncation defect reachable with real meals?")
    print("      the host trades precision against range via s2; smaller s2")
    print("      keeps more precision and pushes values further up the field")
    print(f"    {'s2':>4s} {'meals overflowing 16-bit field':>32s}")
    for s2, over, n in reach:
        pct = 100.0 * over / n
        note = "  <- truncation would invert these" if over else ""
        print(f"    {s2:4d} {over:14d} / {n:<6d} ({pct:5.1f} %){note}")

    print("=" * 72)


if __name__ == "__main__":
    main()
