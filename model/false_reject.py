#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Kush Shah
# SPDX-License-Identifier: Apache-2.0
"""When the guard rejects a weight set, was it right to?

WHY THIS SCRIPT EXISTS
----------------------
The sign condition is SUFFICIENT for monotonicity, not necessary. Liu et al.
(NeurIPS 2020) name the predicate `sign verification` and show MILP-certified
monotone networks failing it, so the guard is sound but not complete: it can
raise UNTRUSTED on a weight set that is in fact monotone. RESULTS.md 9.1 and
the paper both say so, and both say the rate has never been measured. This
measures a bound on it.

WHAT IS AND IS NOT MEASURABLE HERE
----------------------------------
Monotonicity in carbohydrate must hold for EVERY setting of the other five
inputs. Sweeping the carbohydrate byte at sampled settings of the others can
FIND a counterexample but can never establish that none exists. So:

  * a sweep that falls  ->  the weight set is genuinely non-monotone, and the
    guard's rejection was CORRECT. This direction is sound.
  * a sweep that never falls  ->  nothing is proved. The set may be monotone,
    or the counterexample may live at a setting we did not sample.

The reportable quantity is therefore a LOWER BOUND on how often the guard is
right, never an estimate of how often it is wrong. Reporting "the fraction that
are monotone" from a sampled sweep would be unsound and a reviewer would say
so.

WHICH WEIGHT SETS
-----------------
The 44 per-person unconstrained refits from sign_condition.py -- the same sets
behind the 44-of-44 headline. Random weight sets would be cheaper and would
answer a question nobody deploys; these are the sets the device would actually
be handed, and the bound directly qualifies the headline.

    python false_reject.py                  (needs .venv-legacy)
"""

import argparse
import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "test"))

import golden_float as gf                       # noqa: E402
import monotonicity as mono                     # noqa: E402

from cgmacros_loader import FEATURES            # noqa: E402,F401
from sign_condition import build_ordered, offending_units, MIN_MEALS  # noqa: E402
from train import CARB, kfold_participants, standardise, train_mlp    # noqa: E402

DEFAULT_ROOT = r"C:\Users\kushk\Downloads\Claude\cgmacros\csv"

# The budget sign_condition.py's headline is quoted at the middle of, so the
# weight sets here are the ones the 44-of-44 figure describes.
BUDGET = 200

# Settings of the other five inputs to sweep carbohydrate at. Monotonicity must
# hold at all of them; sampling more can only find more counterexamples, never
# fewer, so this number is a floor on the search and is reported alongside the
# result.
N_CONTEXTS = 24
SEED = 20260826


def chip_weights(mlp):
    """The trained network as the INT8 rows the sweep consumes.

    Biases are omitted deliberately: mono.sweep_mode_b builds its streams
    without bias terms, and a constant cannot affect whether the output is
    monotone in x_c. Scales are the ones that ship.
    """
    W1f = mlp.coefs_[0]          # (n_features, n_hidden)
    W2f = mlp.coefs_[1]          # (n_hidden, 1)
    kw1, kw2 = gf.BEST_SCALES["kw1"], gf.BEST_SCALES["kw2"]
    W1 = [gf.quantise(W1f[:, j].tolist(), kw1) for j in range(W1f.shape[1])]
    W2_row = gf.quantise([W2f[j, 0] for j in range(W2f.shape[0])], kw2)
    return W1, W2_row


def is_provably_non_monotone(W1, W2_row, contexts, s1, s2):
    """Did any swept context show the REPORTED value falling? Sound one way."""
    for x in contexts:
        trace = mono.sweep_mode_b(W1, x, W2_row, s1, s2, c=CARB)
        if mono.violations(trace, which=2, direction=+1):
            return True
    return False


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--contexts", type=int, default=N_CONTEXTS)
    args = ap.parse_args()

    rng = np.random.RandomState(SEED)
    s1, s2 = gf.BEST_SCALES["s1"], gf.BEST_SCALES["s2"]
    kx = gf.BEST_SCALES["kx"]

    X, y, g = build_ordered(args.root)
    folds = kfold_participants(g, k=args.folds, seed=0)

    print("=" * 74)
    print(" IS THE GUARD RIGHT WHEN IT REJECTS?")
    print("=" * 74)
    print("  the 44 per-person unconstrained refits behind RESULTS.md 1.3,")
    print("  swept over the full INT8 carbohydrate range at %d sampled"
          % args.contexts)
    print("  settings of the other five inputs")
    print()
    print("  a falling sweep PROVES non-monotone, so the rejection was right;")
    print("  a flat sweep proves nothing -- this is a lower bound, not a rate")
    print()

    rejected = 0
    proved_bad = 0
    control_bad = 0
    rows = []

    for test_p in folds:
        m = np.array([p in test_p for p in g])
        Xtr, Xte, ytr, yte, _ = standardise(X[~m], X[m], y[~m], y[m])
        pid_te = g[m]

        for pid in sorted(set(pid_te.tolist())):
            sel = pid_te == pid
            xs, ts = Xte[sel], yte[sel]
            if len(ts) < MIN_MEALS:
                continue

            mm = train_mlp(Xtr, ytr, seed=0)
            mm.set_params(warm_start=True, max_iter=BUDGET)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                mm.fit(xs, ts)

            bad_units = offending_units(mm)
            if not bad_units:
                continue                      # guard would not fire; not our set
            rejected += 1

            W1, W2_row = chip_weights(mm)
            # Contexts drawn from this person's own standardised meals, so the
            # sweep visits input combinations the device would really see,
            # padded with random draws to widen the search.
            ctx = []
            for row in xs[:min(len(xs), args.contexts // 2)]:
                ctx.append(gf.quantise(row.tolist(), kx))
            while len(ctx) < args.contexts:
                ctx.append([int(v) for v in rng.randint(-128, 128, size=X.shape[1])])

            found = is_provably_non_monotone(W1, W2_row, ctx, s1, s2)
            proved_bad += int(found)

            # CONTROL, and the result is worthless without it. A detector that
            # reported "falling" for everything would also score 44 of 44. Flip
            # the layer-2 sign of every offending unit: same magnitudes, same
            # contexts, same sweep, but now the sign condition HOLDS -- so the
            # composition argument applies and no fall may exist. If this fires
            # even once, the measurement above is measuring the detector.
            W2_fixed = list(W2_row)
            for j in bad_units:
                W2_fixed[j] = -W2_fixed[j]
            assert mono.sign_condition(W1, W2_fixed, c=CARB),                 "control set still violates the condition"
            ctrl = is_provably_non_monotone(W1, W2_fixed, ctx, s1, s2)
            control_bad += int(ctrl)

            rows.append((int(pid), len(bad_units), found, ctrl))

    print("  %-6s %-14s %-20s %s"
          % ("pid", "offending units", "falling sweep found", "control (signs fixed)"))
    for pid, nbad, found, ctrl in rows:
        print("  %-6d %-14d %-20s %s"
              % (pid, nbad, "YES" if found else "-", "FELL" if ctrl else "flat"))

    print()
    print("  weight sets the guard rejects        : %d" % rejected)
    print("  proved genuinely non-monotone        : %d" % proved_bad)
    if rejected:
        print("  guard provably correct in at least   : %d of %d (%.0f %%)"
              % (proved_bad, rejected, 100.0 * proved_bad / rejected))
    print()
    print("  control, same weights with offending signs flipped so the")
    print("  condition HOLDS: %d of %d showed a fall (must be 0)"
          % (control_bad, rejected))
    if control_bad:
        print("  ** CONTROL FAILED -- the number above is not trustworthy **")
    print()
    print("  The remainder are NOT shown to be monotone -- a sampled sweep")
    print("  cannot establish that. They are the sets for which this search")
    print("  found no counterexample at %d contexts." % args.contexts)
    print("=" * 74)


if __name__ == "__main__":
    main()
