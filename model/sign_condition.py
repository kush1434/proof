#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Kush Shah
# SPDX-License-Identifier: Apache-2.0
"""Does a realistic per-person refit still admit the monotonicity guarantee?

WHY THIS SCRIPT EXISTS
----------------------
RESULTS.md 1.3 reports that an unconstrained per-person refit violates the sign
condition in 44 of 44 cases, and that is the single measurement the whole
argument for an on-chip guard rests on: if a per-patient weight set were
normally fine, the chip would be checking a precondition that never fails.

Every other headline in RESULTS.md names the script that produces it. This one
did not -- nothing in the repository computed it -- so the number could not be
re-derived after a pipeline change, which is exactly the failure mode that put
a stale monotonicity cost in the results sheet for days. This closes that.

WHAT "FINE-TUNE" MEANS HERE, SINCE IT DECIDES THE ANSWER
--------------------------------------------------------
A population network is trained UNCONSTRAINED on the other participants, then
warm-started and trained further on one held-out participant's own meals. That
is the most ordinary thing a host would do, and it is deliberately the setting
most favourable to trusting the host:

  * the refit sees ALL of that person's meals, not the 8 the personalisation
    curve argues for, so it is better resourced than a real deployment;
  * it starts from a population solution rather than from noise, so it has no
    reason to wander far;
  * the sign condition is checked on the WEIGHTS, so nothing about the data or
    the fit quality can excuse a violation.

If the condition fails even here, it fails in the cases a device would meet.

The budget sweep matters as much as the headline: a violation rate that only
appears at one arbitrary number of epochs would be an artifact of that choice.

    python sign_condition.py
"""

import argparse
import os

import numpy as np
from sklearn.metrics import r2_score

from cgmacros_loader import FEATURES, load_all
from train import CARB, N_HIDDEN, kfold_participants, standardise, train_mlp

DEFAULT_ROOT = r"C:\Users\kushk\Downloads\Claude\cgmacros\csv"
MIN_MEALS = 6      # matches personalise.py's MIN_EVAL: fewer is not a refit
BUDGETS = (20, 50, 100, 200, 400)


def build_ordered(root):
    recs, _ = load_all(root, meal_type=None, drop_nonpositive=True)
    recs.sort(key=lambda r: (r["pid"], r["t"]))
    X = np.array([[r[f] for f in FEATURES] for r in recs], float)
    y = np.array([r["iauc"] for r in recs], float)
    g = np.array([r["pid"] for r in recs], int)
    return X, y, g


def offending_units(mlp, c=CARB):
    """Hidden units whose two signs disagree, so no proof is available.

    Mirrors train.check_constraints and the chip's own guard: the product
    W1[j][c] * W2[j] is negative only when the signs differ and neither operand
    is zero, so a zero carbohydrate weight can never count as a violation.
    """
    W1 = mlp.coefs_[0]      # (n_features, n_hidden)
    W2 = mlp.coefs_[1]      # (n_hidden, 1)
    return [j for j in range(W1.shape[1]) if W1[CARB, j] * W2[j, 0] < 0]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    X, y, g = build_ordered(args.root)
    folds = kfold_participants(g, k=args.folds, seed=0)

    print("=" * 74)
    print(" SIGN CONDITION UNDER A PER-PERSON REFIT")
    print("=" * 74)
    print("  population model trained unconstrained on the other participants,")
    print("  then warm-started and trained further on one person's own meals")
    print()

    results = {b: [] for b in BUDGETS}          # budget -> [(pid, n_bad, n_meals)]
    pop_bad = []

    for test_p in folds:
        m = np.array([p in test_p for p in g])
        Xtr, Xte, ytr, yte, _ = standardise(X[~m], X[m], y[~m], y[m])
        pid_te = g[m]

        base = train_mlp(Xtr, ytr, seed=0)
        pop_bad.append(len(offending_units(base)))

        for pid in sorted(set(pid_te.tolist())):
            sel = pid_te == pid
            xs, ts = Xte[sel], yte[sel]
            if len(ts) < MIN_MEALS:
                continue
            for budget in BUDGETS:
                # Warm start from the population solution and keep training on
                # this person alone. A fresh copy per budget, so the budgets are
                # independent rather than cumulative.
                mm = train_mlp(Xtr, ytr, seed=0)
                mm.set_params(warm_start=True, max_iter=budget)
                import warnings
                with warnings.catch_warnings():
                    # Non-convergence is expected and is not the question here.
                    warnings.simplefilter("ignore")
                    mm.fit(xs, ts)
                results[budget].append((int(pid), len(offending_units(mm)),
                                        int(len(ts))))

    n_pop_bad = int(np.mean(pop_bad))
    print(f"  the POPULATION model itself fails on {n_pop_bad} of {N_HIDDEN} "
          f"hidden units (mean over folds)")
    print("  -- so an unconstrained fit does not satisfy the condition even "
          "before personalisation")
    print()
    print(f"  {'epochs':>7s} {'people':>7s} {'violating':>10s} {'rate':>7s} "
          f"{'offending units: median':>25s}")
    for b in BUDGETS:
        rows = results[b]
        bad = [n for _, n, _ in rows if n > 0]
        med = int(np.median([n for _, n, _ in rows]))
        print(f"  {b:7d} {len(rows):7d} {len(bad):10d} "
              f"{100.0 * len(bad) / max(len(rows), 1):6.1f}% "
              f"{med:>25d}")

    print()
    print("  RESULTS.md 1.3 records 44 of 44 violating, median 3 of 8 units.")
    print("=" * 74)


if __name__ == "__main__":
    main()
