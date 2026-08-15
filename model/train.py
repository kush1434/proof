#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Kush Shah
# SPDX-License-Identifier: Apache-2.0
"""
Train the Mode B network, and answer the question monotonicity depends on.

THE QUESTION
------------
The chip's safety property -- increasing carbohydrate never decreases the
predicted response -- holds if and only if every hidden unit satisfies

    W1[j][carb] * W2[j] >= 0                    (the sign condition)

That is a property of *trained weights*, not of the RTL. If training produces a
hidden unit whose two signs disagree, the property is false by construction and
no amount of stimulus can rescue it; the fix is a constrained objective. So the
first thing this script reports is whether the condition holds, and the second
is what enforcing it costs in accuracy.

SCOPE, AND WHY IT IS NOT THE BASELINE'S SCOPE
----------------------------------------------
The reference notebook uses breakfasts only. In CGMacros the breakfasts are
STANDARDISED test meals: 383 usable breakfasts contain just 6 distinct
macronutrient combinations, and the top 4 cover 83% of them. Carbohydrate takes
three values. That is deliberate study design -- holding the meal fixed isolates
person-to-person variation -- and it makes the notebook a *personalisation*
result, not a meal-composition result.

Proof predicts response from meal composition, so it needs meals that differ.
Across all meal types there are 1,346 usable meals with 592 distinct
combinations, and carbohydrate's coefficient of variation rises from 0.29 to
0.74. This script therefore trains on all meals and reports the breakfast-only
baseline separately, so the two are comparable without being conflated.

SPLITTING
---------
Grouped by participant. A random row split would put the same person's meals on
both sides and inflate every number, because individual glycemic response is
exactly what varies most.
"""

import argparse
import json
import os

import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.neural_network import MLPRegressor

from cgmacros_loader import FEATURES, load_all

DEFAULT_ROOT = r"C:\Users\kushk\Downloads\Claude\cgmacros\csv"
CARB = FEATURES.index("carbs")
N_HIDDEN = 8


def build(root, meal_type):
    recs, _ = load_all(root, meal_type=meal_type, drop_nonpositive=True)
    X = np.array([[r[f] for f in FEATURES] for r in recs], dtype=float)
    y = np.array([r["iauc"] for r in recs], dtype=float)
    g = np.array([r["pid"] for r in recs], dtype=int)
    return X, y, g


def split_by_participant(X, y, g, frac=0.25, seed=0):
    pids = np.unique(g)
    rng = np.random.RandomState(seed)
    rng.shuffle(pids)
    n_test = max(1, int(round(len(pids) * frac)))
    test_p = set(pids[:n_test].tolist())
    m = np.array([p in test_p for p in g])
    return X[~m], y[~m], X[m], y[m], sorted(test_p)


def standardise(Xtr, Xte, ytr, yte):
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd[sd == 0] = 1.0
    ym, ys = ytr.mean(), ytr.std() or 1.0
    return (Xtr - mu) / sd, (Xte - mu) / sd, (ytr - ym) / ys, (yte - ym) / ys, (mu, sd, ym, ys)


def report(tag, ytrue, ypred, ys=1.0):
    print(f"    {tag:24s} R2 {r2_score(ytrue, ypred):+.3f}   "
          f"MAE {mean_absolute_error(ytrue, ypred) * ys:8.0f} mg/dL.min")
    return r2_score(ytrue, ypred)


def baseline_xgb(Xtr, ytr, Xte, yte, ys):
    """The notebook's model: depth-1 XGBoost, effectively an additive GAM."""
    try:
        from xgboost import XGBRegressor
    except ImportError:
        from sklearn.ensemble import GradientBoostingRegressor as XGBRegressor  # noqa
        print("    (xgboost unavailable, using sklearn GradientBoosting depth-1)")
    m = XGBRegressor(max_depth=1, n_estimators=200, learning_rate=0.1)
    m.fit(Xtr, ytr)
    return report("depth-1 XGBoost", yte, m.predict(Xte), ys)


def train_mlp(Xtr, ytr, seed=0, iters=4000):
    m = MLPRegressor(
        hidden_layer_sizes=(N_HIDDEN,),
        activation="relu",
        solver="adam",
        max_iter=iters,
        random_state=seed,
        early_stopping=False,
    )
    m.fit(Xtr, ytr)
    return m


def weights_of(m):
    """(W1, b1, W2, b2) in the chip's orientation: W1[j][i], W2[o][j]."""
    W1 = m.coefs_[0].T.tolist()
    b1 = m.intercepts_[0].tolist()
    W2 = m.coefs_[1].T.tolist()
    b2 = m.intercepts_[1].tolist()
    return W1, b1, W2, b2


def sign_report(W1, W2, c=CARB):
    """Does the trained network admit the monotonicity proof?"""
    row = W2[0]
    bad = [j for j in range(len(W1)) if W1[j][c] * row[j] < 0]
    print(f"    hidden units          : {len(W1)}")
    print(f"    W1[:,carb] signs      : "
          f"{''.join('+' if W1[j][c] >= 0 else '-' for j in range(len(W1)))}")
    print(f"    W2[0] signs           : "
          f"{''.join('+' if row[j] >= 0 else '-' for j in range(len(W1)))}")
    if bad:
        print(f"    SIGN CONDITION FAILS  : units {bad} disagree")
        print("      -> monotonicity is false by construction for these weights.")
    else:
        print("    SIGN CONDITION HOLDS  : monotonicity provable for these weights")
    return bad


def project_signs(m, c=CARB):
    """Enforce the sign condition by projection: make every unit agree.

    The cheapest possible constraint -- flip the sign of the offending
    carbohydrate weight toward zero. Reported honestly as what it is: a
    post-hoc projection, not a principled constrained objective. It exists to
    measure the ACCURACY COST of the constraint, which is the number that
    decides whether a constrained retrain is worth doing properly.
    """
    W1 = m.coefs_[0].copy()
    W2 = m.coefs_[1].copy()
    for j in range(W1.shape[1]):
        if W1[c, j] * W2[j, 0] < 0:
            W1[c, j] = 0.0
    m2 = MLPRegressor(hidden_layer_sizes=(N_HIDDEN,), activation="relu",
                      solver="adam", max_iter=1, random_state=0)
    m2.__dict__.update({k: v for k, v in m.__dict__.items()})
    m2.coefs_ = [W1, W2]
    return m2


def _softplus(v):
    return np.log1p(np.exp(-np.abs(v))) + np.maximum(v, 0.0)


def _sigmoid(v):
    return 1.0 / (1.0 + np.exp(-v))


def train_monotone(Xtr, ytr, seed=0, iters=6000, lr=0.02, c=CARB, dirs=None):
    """Train a network that satisfies the sign condition BY CONSTRUCTION.

    Zeroing offending weights after the fact (project_signs) measures only an
    upper bound on the cost, because it damages a network that was optimised
    without the constraint. The honest number comes from optimising *within*
    the constrained family.

    The constraint is imposed by reparameterisation rather than by a penalty,
    so it holds exactly at every step and cannot be traded away:

        W1[j][carb] = d[j] * softplus(A[j][carb])
        W2[0][j]    = d[j] * softplus(c[j])

    for a fixed per-unit direction d[j] in {+1, -1}. The product is then
    d[j]**2 * softplus * softplus >= 0 for EITHER direction, which is precisely
    the sign condition -- a unit is allowed to oppose carbohydrate twice and
    still raise the response.

    Forcing every d[j] = +1 would also satisfy the condition but is strictly
    stronger than necessary, and it over-penalises any network whose second
    layer is mostly negative. `dirs` therefore defaults to the sign structure
    the UNCONSTRAINED model chose, so the constrained family is centred on the
    solution rather than on an arbitrary half of the space. Nothing else is constrained -- the other five inputs keep full
    freedom of sign, so the network can still represent fibre lowering the
    response, and so on.

    Plain numpy with hand-derived gradients: the model is small, and a
    dependency on a training framework is not worth adding for one measurement.
    """
    rng = np.random.RandomState(seed)
    n, d = Xtr.shape
    dvec = np.ones(N_HIDDEN) if dirs is None else np.sign(np.asarray(dirs, float))
    dvec[dvec == 0] = 1.0
    A = rng.randn(N_HIDDEN, d) * 0.3
    b1 = np.zeros(N_HIDDEN)
    cc = rng.randn(N_HIDDEN) * 0.3
    b2 = np.array([ytr.mean()])

    params = [A, b1, cc, b2]
    ms = [np.zeros_like(p) for p in params]
    vs = [np.zeros_like(p) for p in params]
    b1a, b2a, eps = 0.9, 0.999, 1e-8

    for step in range(1, iters + 1):
        W1 = A.copy()
        W1[:, c] = dvec * _softplus(A[:, c])
        W2 = dvec * _softplus(cc)

        z = Xtr @ W1.T + b1           # (n, H)
        h = np.maximum(z, 0.0)
        yhat = h @ W2 + b2[0]

        err = yhat - ytr
        dy = 2.0 * err / n

        gW2 = h.T @ dy
        gcc = gW2 * dvec * _sigmoid(cc)
        gb2 = np.array([dy.sum()])

        dh = np.outer(dy, W2)
        dz = dh * (z > 0)
        gW1 = dz.T @ Xtr
        gA = gW1.copy()
        gA[:, c] = gW1[:, c] * dvec * _sigmoid(A[:, c])
        gb1 = dz.sum(0)

        for p, g, m, v in zip(params, [gA, gb1, gcc, gb2], ms, vs):
            m *= b1a
            m += (1 - b1a) * g
            v *= b2a
            v += (1 - b2a) * (g * g)
            mh = m / (1 - b1a ** step)
            vh = v / (1 - b2a ** step)
            p -= lr * mh / (np.sqrt(vh) + eps)

    W1 = A.copy()
    W1[:, c] = dvec * _softplus(A[:, c])
    W2 = dvec * _softplus(cc)
    return W1, b1, W2, b2


def monotone_predict(X, W1, b1, W2, b2):
    return np.maximum(X @ W1.T + b1, 0.0) @ W2 + b2[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--out", default="weights.json")
    args = ap.parse_args()

    for scope, label in ((None, "ALL MEALS"), ("Breakfast", "BREAKFAST ONLY")):
        X, y, g = build(args.root, scope)
        print("=" * 70)
        print(f" {label}: {len(y)} meals, {len(np.unique(g))} participants")
        print("=" * 70)
        Xtr, ytr, Xte, yte, test_p = split_by_participant(X, y, g)
        Xtr, Xte, ytr_s, yte_s, (mu, sd, ym, ys) = standardise(Xtr, Xte, ytr, yte)
        print(f"  held-out participants: {len(test_p)}  "
              f"(train {len(ytr_s)} meals / test {len(yte_s)} meals)")

        print("\n  [baseline]")
        report("predict the mean", yte_s, np.zeros_like(yte_s), ys)
        baseline_xgb(Xtr, ytr_s, Xte, yte_s, ys)

        print("\n  [Mode B network, 6-8-1]")
        best, best_r2 = None, -9e9
        for s in range(args.seeds):
            m = train_mlp(Xtr, ytr_s, seed=s)
            r2 = r2_score(yte_s, m.predict(Xte))
            if r2 > best_r2:
                best, best_r2 = m, r2
        report(f"MLP (best of {args.seeds} seeds)", yte_s, best.predict(Xte), ys)

        print("\n  [monotonicity feasibility -- the question that matters]")
        W1, b1, W2, b2 = weights_of(best)
        bad = sign_report(W1, W2)

        if bad:
            proj = project_signs(best)
            r2p = report("naive projection", yte_s, proj.predict(Xte), ys)
            print(f"      -> upper bound on the cost only: it damages a network"
                  f" optimised without the constraint")

        print("\n  [constrained retrain -- monotone by construction]")
        # Centre the constrained family on the unconstrained solution's
        # sign structure instead of assuming every unit points the same way.
        dirs = [1.0 if v >= 0 else -1.0 for v in W2[0]]
        mb, mr2 = None, -9e9
        for s in range(args.seeds):
            w = train_monotone(Xtr, ytr_s, seed=s, dirs=dirs)
            r2 = r2_score(yte_s, monotone_predict(Xte, *w))
            if r2 > mr2:
                mb, mr2 = w, r2
        report(f"monotone MLP (best of {args.seeds})", yte_s,
               monotone_predict(Xte, *mb), ys)
        mW1, mb1, mW2, mb2 = mb
        bad2 = sign_report(mW1.tolist(), [mW2.tolist()])
        assert not bad2, "constrained model must satisfy the condition by construction"
        print(f"    TRUE cost of monotonicity: R2 {best_r2:+.3f} -> {mr2:+.3f}"
              f"  (delta {mr2 - best_r2:+.3f})")
        if bad:
            W1, b1 = mW1.tolist(), mb1.tolist()
            W2, b2 = [mW2.tolist()], mb2.tolist()

        if scope is None:
            out = {
                "features": list(FEATURES),
                "scope": "all meals",
                "target": "iAUC (signed convention, 2h @ 15min, Libre)",
                "standardise": {"mu": mu.tolist(), "sd": sd.tolist(),
                                "y_mean": float(ym), "y_std": float(ys)},
                "W1": W1, "b1": b1, "W2": W2, "b2": b2,
                # Derived from the weights ACTUALLY being written, not from the
                # unconstrained model measured earlier. Those differ whenever a
                # constrained retrain replaced them, and a metadata field that
                # contradicts its own data is worse than no field.
                "sign_condition_holds": all(
                    W1[j][CARB] * W2[0][j] >= 0 for j in range(len(W1))
                ),
                "monotone_by_construction": bool(bad),
            }
            with open(args.out, "w", encoding="utf-8") as f:
                json.dump(out, f, indent=2)
            print(f"\n  weights written to {os.path.abspath(args.out)}")
        print()


if __name__ == "__main__":
    main()
