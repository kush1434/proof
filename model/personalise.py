#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Kush Shah
# SPDX-License-Identifier: Apache-2.0
"""
How little data does personalisation need before it beats the population model?

WHY THIS QUESTION AND NOT ANOTHER
---------------------------------
The chip streams its weights from the host rather than storing them, which
began as an area decision -- 83 INT8 parameters do not fit beside the datapath
in one tile. The consequence is that the weights can be *per person*: same
silicon, different patient. That is the design's central claim, and until now
nothing measured whether it is worth anything.

This measures it. A population model is trained on other participants, then
adapted to a held-out person using only the first `k` of their meals, and
evaluated on their remaining meals. Sweeping k gives the curve: how many meals
must someone log before a personalised model beats the one-size-fits-all one?

WHAT "ADAPT" MEANS HERE, AND WHY IT IS THE MINIMAL HONEST CHOICE
----------------------------------------------------------------
Only the output bias is refitted. That is the smallest possible intervention,
it maps exactly onto something the chip already does -- the host streams the
layer-2 bias as two ordinary weight bytes -- and it needs no gradient descent
on the device. A full per-person retrain would show larger gains and would also
be a different claim; this one is defensible with a handful of meals, which is
the regime a real user is actually in.

Meals are taken in TIME ORDER, not at random: a person accumulates history
forwards, and sampling their future to predict their past would inflate every
number here.
"""

import argparse
import os

import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score

from cgmacros_loader import FEATURES, load_all
from train import (CARB, kfold_participants, monotone_predict, standardise,
                   train_mlp, train_monotone, weights_of)

DEFAULT_ROOT = r"C:\Users\kushk\Downloads\Claude\cgmacros\csv"
K_VALUES = (0, 1, 2, 3, 5, 8, 12, 16)
MIN_EVAL = 6  # a participant must retain this many meals to be scored


def build_ordered(root):
    """Meals with their timestamps, so 'first k' means first in time."""
    recs, _ = load_all(root, meal_type=None, drop_nonpositive=True)
    recs.sort(key=lambda r: (r["pid"], r["t"]))
    X = np.array([[r[f] for f in FEATURES] for r in recs], float)
    y = np.array([r["iauc"] for r in recs], float)
    g = np.array([r["pid"] for r in recs], int)
    return X, y, g


def population_model(Xtr, ytr, seeds=3):
    _, _, W2u, _ = weights_of(train_mlp(Xtr, ytr, seed=0))
    dirs = [1.0 if v >= 0 else -1.0 for v in W2u[0]]
    best, br = None, -9e9
    for s in range(seeds):
        w = train_monotone(Xtr, ytr, seed=s, constraints={CARB: +1}, dirs=dirs)
        r = r2_score(ytr, monotone_predict(Xtr, *w))
        if r > br:
            best, br = w, r
    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    X, y, g = build_ordered(args.root)
    folds = kfold_participants(g, k=args.folds, seed=0)

    # k -> list of (true, pred) over every held-out participant's later meals
    acc = {k: {"t": [], "p": [], "subjects": 0, "improved": 0} for k in K_VALUES}

    for test_p in folds:
        m = np.array([p in test_p for p in g])
        Xtr, Xte, ytr, yte, (mu, sd, ym, ys) = standardise(X[~m], X[m], y[~m], y[m])
        model = population_model(Xtr, ytr)
        pid_te = g[m]

        for pid in sorted(set(pid_te.tolist())):
            sel = pid_te == pid
            xs, ts = Xte[sel], yte[sel]           # already in time order
            base_pred = monotone_predict(xs, *model)

            for k in K_VALUES:
                if len(ts) - k < MIN_EVAL:
                    continue
                # --- affine calibration y' = a*y + b, with a >= 0 -----------
                # Two parameters instead of one, still closed form, and still
                # MONOTONICITY-PRESERVING: a non-negative scale cannot reorder
                # anything, so the guarantee survives per-person adaptation.
                # Clamping a at zero is what keeps that true; an unconstrained
                # least-squares fit could return a negative slope and silently
                # invert the model for that patient.
                if k >= 2:
                    A = np.vstack([base_pred[:k], np.ones(k)]).T
                    try:
                        sol, *_ = np.linalg.lstsq(A, ts[:k], rcond=None)
                        a_hat, b_hat = float(sol[0]), float(sol[1])
                    except np.linalg.LinAlgError:
                        a_hat, b_hat = 1.0, 0.0
                    a_hat = max(0.0, a_hat)
                    aff = a_hat * base_pred[k:] + b_hat
                    acc[k].setdefault("affine", [])
                    acc[k]["affine"].append(
                        (mean_absolute_error(ts[k:], base_pred[k:]) * ys,
                         mean_absolute_error(ts[k:], aff) * ys,
                         a_hat))
                acc[k].setdefault("per_subject", [])
                # Personalise on the first k meals; score on everything after.
                offset = 0.0 if k == 0 else float((ts[:k] - base_pred[:k]).mean())
                t_eval = ts[k:]
                p_eval = base_pred[k:] + offset

                acc[k]["t"].append(t_eval * ys + ym)
                acc[k]["p"].append(p_eval * ys + ym)
                acc[k]["subjects"] += 1
                # Per-subject MAE delta, so the gain can carry a confidence
                # interval instead of resting on one pooled number. Paired by
                # construction: same person, same meals, same population model.
                acc[k]["per_subject"].append(
                    (int(pid),
                     mean_absolute_error(t_eval, base_pred[k:]) * ys,
                     mean_absolute_error(t_eval, p_eval) * ys))
                if k > 0:
                    mae_pop = mean_absolute_error(t_eval, base_pred[k:])
                    mae_per = mean_absolute_error(t_eval, p_eval)
                    acc[k]["improved"] += int(mae_per < mae_pop)

    print("=" * 74)
    print(" PERSONALISATION CURVE -- meals logged before the model is worth it")
    print("=" * 74)
    print("  adaptation: refit the output bias only, from the first k meals in")
    print("  TIME ORDER; scored on that person's remaining meals")
    print()
    print(f"  {'k meals':>8s} {'subjects':>9s} {'R2':>8s} {'MAE':>9s} "
          f"{'better than population':>24s}")
    base_mae = None
    for k in K_VALUES:
        d = acc[k]
        if not d["t"]:
            continue
        t = np.concatenate(d["t"])
        p = np.concatenate(d["p"])
        mae = mean_absolute_error(t, p)
        if k == 0:
            base_mae = mae
        frac = "" if k == 0 else f"{d['improved']}/{d['subjects']}"
        delta = "" if base_mae is None or k == 0 else f"  ({mae - base_mae:+.0f})"
        print(f"  {k:8d} {d['subjects']:9d} {r2_score(t, p):+8.3f} "
              f"{mae:9.0f}{delta:>9s} {frac:>15s}")
    print()
    print("  paired per-subject MAE change (population -> personalised),")
    print("  95 % CI across participants:")
    for k in K_VALUES:
        d = acc[k]
        if k == 0 or not d.get("per_subject"):
            continue
        dd = np.array([per - pop for _, pop, per in d["per_subject"]])
        se = dd.std(ddof=1) / np.sqrt(len(dd))
        lo, hi = dd.mean() - 1.96 * se, dd.mean() + 1.96 * se
        verdict = ("helps" if hi < 0 else
                   "HARMS" if lo > 0 else "not established")
        print(f"    k={k:2d}  delta MAE {dd.mean():+7.0f}  "
              f"[95% CI {lo:+7.0f}, {hi:+7.0f}]   {verdict}")
    print()
    print("  affine calibration (2 params, a >= 0) vs bias only (1 param):")
    print(f"    {'k':>3s} {'bias MAE':>10s} {'affine MAE':>11s} {'delta':>8s} "
          f"{'median a':>9s} {'a clamped':>10s}")
    for k in K_VALUES:
        d = acc[k]
        if not d.get("affine") or not d.get("per_subject"):
            continue
        bias_mae = np.mean([per for _, _, per in
                            [(0, pop, per) for _, pop, per in d["per_subject"]]])
        aff = np.array([[p, a, s] for p, a, s in d["affine"]])
        n_clamp = int((aff[:, 2] == 0.0).sum())
        print(f"    {k:3d} {bias_mae:10.0f} {aff[:, 1].mean():11.0f} "
              f"{aff[:, 1].mean() - bias_mae:+8.0f} {np.median(aff[:, 2]):9.2f} "
              f"{n_clamp:6d}/{len(aff):<4d}")
    print()
    print("  k = 0 is the population model with no personalisation at all.")
    print("=" * 74)


if __name__ == "__main__":
    main()
