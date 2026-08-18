#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Kush Shah
# SPDX-License-Identifier: Apache-2.0
"""Is 8 hidden units the right width? Paired sweep over the same folds.

WHY THIS SCRIPT EXISTS
----------------------
Hidden width is the one structural choice baked into silicon -- it is the depth
of the `h` shift register, and of the guard's `sgnreg`/`nzreg` beside it -- so
changing it is an RTL change, not a retrain. RESULTS.md 1.2 concludes that
widths 4 to 12 are indistinguishable from 8, and uses that to refuse a proposal
to shrink the hidden layer to buy area for a second guard.

Nothing in the repository computed it: `N_HIDDEN` is a module-level constant in
`train.py` and no code swept it. This does.

HOW IT WORKS, AND THE TRAP IN IT
---------------------------------
`train.N_HIDDEN` is read as a module global at call time by both `train_mlp`
and `train_monotone`, so a sweep can rebind it. That is genuinely how the width
has to be varied without editing the file, but it is process-global mutable
state: anything that caches a model across widths would silently compare two
different architectures. Each width therefore gets a full, independent pass and
the constant is restored afterwards.

Folds are the same partitions for every width (seed 0), so differences are
PAIRED and the confidence interval is on the difference, which is the quantity
with the smaller variance and the honest one to report.

Both the unconstrained network and the shipped carbohydrate-constrained one are
swept, because it is the constrained model that goes on the chip and there is
no reason to assume the two respond to width the same way.

    python width_sweep.py
"""

import argparse

import numpy as np
from sklearn.metrics import r2_score

import train
from train import (CARB, build, kfold_participants, monotone_predict,
                   standardise, train_mlp, train_monotone, weights_of)

DEFAULT_ROOT = r"C:\Users\kushk\Downloads\Claude\cgmacros\csv"
WIDTHS = (4, 6, 8, 12)
REFERENCE = 8


def sweep_width(X, y, g, width, k=5, seeds=3):
    """Per-fold R^2 at one hidden width, unconstrained and carbs-constrained."""
    folds = kfold_participants(g, k=k, seed=0)
    unc, con = [], []

    old = train.N_HIDDEN
    train.N_HIDDEN = width          # read at call time by both trainers
    try:
        for test_p in folds:
            m = np.array([p in test_p for p in g])
            Xtr, Xte, ytr, yte, _ = standardise(X[~m], X[m], y[~m], y[m])

            best, best_r2 = None, -9e9
            for s in range(seeds):
                mm = train_mlp(Xtr, ytr, seed=s)
                r2 = r2_score(yte, mm.predict(Xte))
                if r2 > best_r2:
                    best, best_r2 = mm, r2
            unc.append(best_r2)

            # Centre the constrained family on the unconstrained solution's own
            # sign structure, exactly as train.py and clinical.py do.
            _, _, W2u, _ = weights_of(best)
            dirs = [1.0 if v >= 0 else -1.0 for v in W2u[0]]
            b = -9e9
            for s in range(seeds):
                w = train_monotone(Xtr, ytr, seed=s, constraints={CARB: +1},
                                   dirs=dirs)
                b = max(b, r2_score(yte, monotone_predict(Xte, *w)))
            con.append(b)
    finally:
        train.N_HIDDEN = old        # never leave the global mutated

    return np.array(unc), np.array(con)


def paired_ci(a, b):
    """Mean difference a - b and its 95 % CI, paired across folds."""
    d = a - b
    se = d.std(ddof=1) / np.sqrt(len(d))
    return d.mean(), d.mean() - 1.96 * se, d.mean() + 1.96 * se


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    X, y, g = build(args.root, None)

    print("=" * 74)
    print(" HIDDEN WIDTH SWEEP -- paired over the same %d folds" % args.folds)
    print("=" * 74)
    print("  width is structural: it is the depth of the h shift register,")
    print("  so changing it is an RTL change rather than a retrain")
    print()

    res = {}
    for w in WIDTHS:
        res[w] = sweep_width(X, y, g, w, k=args.folds)
        print("    width %2d done" % w)

    for label, idx in (("unconstrained", 0), ("carbs up (as shipped)", 1)):
        print()
        print("  [%s]" % label)
        print("    %-7s %9s %8s   %s" % ("width", "mean R2", "sd", "per fold"))
        for w in WIDTHS:
            a = res[w][idx]
            per = " ".join("%+.2f" % v for v in a)
            print("    %-7d %+9.3f %8.3f   %s" % (w, a.mean(), a.std(), per))
        print()
        base = res[REFERENCE][idx]
        for w in WIDTHS:
            if w == REFERENCE:
                continue
            d, lo, hi = paired_ci(res[w][idx], base)
            verdict = ("no measurable difference" if lo <= 0 <= hi
                       else "DIFFERS from %d" % REFERENCE)
            print("    width %-3d vs %d: delta R2 %+.3f [95%% CI %+.3f, %+.3f]  %s"
                  % (w, REFERENCE, d, lo, hi, verdict))

    print()
    print("  RESULTS.md 1.2 records, against width 8:")
    print("    4  +0.016 [-0.023, +0.055] | 6  +0.029 [-0.010, +0.068] "
          "| 12  +0.001 [-0.029, +0.030]")
    print("=" * 74)


if __name__ == "__main__":
    main()
