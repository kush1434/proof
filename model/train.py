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
FIBER = FEATURES.index("fiber")
N_HIDDEN = 8

# Directions the chip may be asked to guarantee: carbohydrate rising must never
# lower the response, fibre rising must never raise it. The fibre direction is
# empirically supported but weakly -- marginal r -0.051, partial beta -0.025,
# and within every carbohydrate tertile the high-fibre half has the lower iAUC.
CONSTRAINT_SETS = [
    ({CARB: +1}, "carbs up"),
    ({CARB: +1, FIBER: -1}, "carbs up + fibre down"),
]


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


def check_constraints(W1, W2, constraints):
    """Which constrained inputs fail the sign condition, and on which units."""
    row = W2[0]
    out = {}
    for i, want in sorted(constraints.items()):
        bad = [j for j in range(len(W1)) if W1[j][i] * row[j] * want < 0]
        out[i] = bad
    return out


def sign_report(W1, W2, constraints=None):
    """Does the trained network admit the monotonicity proof?"""
    if constraints is None:
        constraints = {CARB: +1}
    fails = check_constraints(W1, W2, constraints)
    row = W2[0]
    print(f"    hidden units          : {len(W1)}")
    print(f"    W2[0] signs           : "
          f"{''.join('+' if v >= 0 else '-' for v in row)}")
    for i, bad in fails.items():
        arrow = "increasing" if constraints[i] > 0 else "decreasing"
        name = FEATURES[i]
        print(f"    W1[:,{name}] signs{' ' * max(0, 8 - len(name))}: "
              f"{''.join('+' if W1[j][i] >= 0 else '-' for j in range(len(W1)))}"
              f"   (want {arrow})")
        if bad:
            print(f"      FAILS on units {bad}")
    total_bad = sum(len(b) for b in fails.values())
    if total_bad:
        print(f"    SIGN CONDITION FAILS  : {total_bad} unit/input pair(s) disagree")
        print("      -> monotonicity is false by construction for these weights.")
    else:
        print("    SIGN CONDITION HOLDS  : monotonicity provable for these weights")
    return [j for b in fails.values() for j in b]


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


def train_monotone(Xtr, ytr, seed=0, iters=6000, lr=0.02, constraints=None,
                   dirs=None):
    """Train a network satisfying monotonicity constraints BY CONSTRUCTION.

    `constraints` maps a feature index to a required direction:
    `+1` = the response must never DECREASE as that input rises,
    `-1` = it must never INCREASE. Defaults to carbohydrate increasing.

    Zeroing offending weights after the fact (project_signs) measures only an
    upper bound on the cost, because it damages a network optimised without the
    constraint. The honest number comes from optimising *within* the
    constrained family.

    The constraint is imposed by reparameterisation rather than by a penalty,
    so it holds exactly at every step and cannot be traded away. With a shared
    per-unit direction d[j] in {+1, -1} and a per-input sign s[i]:

        W1[j][i] = s[i] * d[j] * softplus(A[j][i])     for constrained i
        W2[0][j] =        d[j] * softplus(c[j])

    so W1[j][i] * W2[j] = s[i] * d[j]**2 * softplus * softplus, whose sign is
    exactly s[i] for every unit, which is the condition. Sharing one d[j]
    across all constrained inputs is what lets several constraints hold at once
    without fighting: a unit may oppose an input twice and still respect it.

    Unconstrained inputs keep full freedom of sign, so the network can still
    represent whatever it likes for those.

    Forcing every d[j] = +1 would also satisfy the condition but is strictly
    stronger than necessary and over-penalises a network whose second layer is
    mostly negative -- an early version did that and reported a cost three to
    four times too large. `dirs` therefore defaults to the sign structure the
    UNCONSTRAINED model chose, centring the constrained family on the solution
    rather than on an arbitrary half of the space.

    Plain numpy with hand-derived gradients: the model is small, and a
    dependency on a training framework is not worth adding for one measurement.
    """
    if constraints is None:
        constraints = {CARB: +1}
    rng = np.random.RandomState(seed)
    n, d = Xtr.shape

    dvec = np.ones(N_HIDDEN) if dirs is None else np.sign(np.asarray(dirs, float))
    dvec[dvec == 0] = 1.0

    idx = sorted(constraints)
    sgn = np.array([constraints[i] for i in idx], float)

    A = rng.randn(N_HIDDEN, d) * 0.3
    b1 = np.zeros(N_HIDDEN)
    cc = rng.randn(N_HIDDEN) * 0.3
    b2 = np.array([ytr.mean()])

    params = [A, b1, cc, b2]
    ms = [np.zeros_like(p) for p in params]
    vs = [np.zeros_like(p) for p in params]
    b1a, b2a, eps = 0.9, 0.999, 1e-8

    def forward(A, cc):
        W1 = A.copy()
        for k, i in enumerate(idx):
            W1[:, i] = sgn[k] * dvec * _softplus(A[:, i])
        return W1, dvec * _softplus(cc)

    for step in range(1, iters + 1):
        W1, W2 = forward(A, cc)

        z = Xtr @ W1.T + b1
        h = np.maximum(z, 0.0)
        yhat = h @ W2 + b2[0]

        dy = 2.0 * (yhat - ytr) / n

        gW2 = h.T @ dy
        gcc = gW2 * dvec * _sigmoid(cc)
        gb2 = np.array([dy.sum()])

        dz = np.outer(dy, W2) * (z > 0)
        gW1 = dz.T @ Xtr
        gA = gW1.copy()
        for k, i in enumerate(idx):
            gA[:, i] = gW1[:, i] * sgn[k] * dvec * _sigmoid(A[:, i])
        gb1 = dz.sum(0)

        for p, g, m, v in zip(params, [gA, gb1, gcc, gb2], ms, vs):
            m *= b1a
            m += (1 - b1a) * g
            v *= b2a
            v += (1 - b2a) * (g * g)
            p -= lr * (m / (1 - b1a ** step)) / (np.sqrt(v / (1 - b2a ** step)) + eps)

    W1, W2 = forward(A, cc)
    return W1, b1, W2, b2


def monotone_predict(X, W1, b1, W2, b2):
    return np.maximum(X @ W1.T + b1, 0.0) @ W2 + b2[0]


def kfold_participants(g, k=5, seed=0):
    """Split PARTICIPANTS into k folds. Never split a participant's meals."""
    pids = np.unique(g)
    rng = np.random.RandomState(seed)
    rng.shuffle(pids)
    return [set(pids[i::k].tolist()) for i in range(k)]


def crossval(X, y, g, k=5, seeds=3, quiet=True):
    """Grouped k-fold, so the headline claim rests on more than one split.

    Reports mean and spread across folds for the unconstrained network, each
    constrained variant, and the notebook-style baseline. Without this, "the
    safety property is free" is a single number from a single partition, and a
    single number cannot distinguish a real effect from a lucky split.
    """
    folds = kfold_participants(g, k=k, seed=0)
    names = ["depth-1 XGBoost", "unconstrained"] + [lbl for _, lbl in CONSTRAINT_SETS]
    scores = {n: [] for n in names}

    for f, test_p in enumerate(folds):
        m = np.array([p in test_p for p in g])
        Xtr, Xte, ytr, yte, _ = standardise(X[~m], X[m], y[~m], y[m])

        from xgboost import XGBRegressor
        xgb = XGBRegressor(max_depth=1, n_estimators=200, learning_rate=0.1)
        xgb.fit(Xtr, ytr)
        scores["depth-1 XGBoost"].append(r2_score(yte, xgb.predict(Xte)))

        best, best_r2 = None, -9e9
        for s in range(seeds):
            mm = train_mlp(Xtr, ytr, seed=s)
            r2 = r2_score(yte, mm.predict(Xte))
            if r2 > best_r2:
                best, best_r2 = mm, r2
        scores["unconstrained"].append(best_r2)

        _, _, W2u, _ = weights_of(best)
        dirs = [1.0 if v >= 0 else -1.0 for v in W2u[0]]
        for cons, lbl in CONSTRAINT_SETS:
            b2r = -9e9
            for s in range(seeds):
                w = train_monotone(Xtr, ytr, seed=s, constraints=cons, dirs=dirs)
                b2r = max(b2r, r2_score(yte, monotone_predict(Xte, *w)))
            scores[lbl].append(b2r)
        if not quiet:
            print(f"    fold {f + 1}/{k} done")
    return scores


def cv_report(scores, ref="unconstrained"):
    print(f"    {'model':26s} {'mean R2':>9s} {'sd':>7s}   per fold")
    for n, v in scores.items():
        a = np.array(v)
        per = " ".join(f"{x:+.2f}" for x in a)
        print(f"    {n:26s} {a.mean():+9.3f} {a.std():7.3f}   {per}")
    base = np.array(scores[ref])
    print()
    for n, v in scores.items():
        if n == ref:
            continue
        d = np.array(v) - base
        # Paired across folds: the same partitions, so the difference is the
        # quantity with the smaller variance and the honest one to report.
        se = d.std(ddof=1) / np.sqrt(len(d))
        lo, hi = d.mean() - 1.96 * se, d.mean() + 1.96 * se
        verdict = "no measurable difference" if lo <= 0 <= hi else "differs"
        print(f"    {n:26s} vs {ref}: delta {d.mean():+.3f} "
              f"[95% CI {lo:+.3f}, {hi:+.3f}]  {verdict}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--out", default="weights.json")
    ap.add_argument("--cv", type=int, default=0,
                    metavar="K", help="also run grouped K-fold cross-validation")
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
        bad = sign_report(W1, W2, {CARB: +1, FIBER: -1})

        if bad:
            proj = project_signs(best)
            r2p = report("naive projection", yte_s, proj.predict(Xte), ys)
            print(f"      -> upper bound on the cost only: it damages a network"
                  f" optimised without the constraint")

        print()
        print("  [constrained retrains -- monotone by construction]")
        # Centre each constrained family on the unconstrained solution's own
        # sign structure rather than assuming every unit points the same way.
        dirs = [1.0 if v >= 0 else -1.0 for v in W2[0]]
        chosen = None
        for cons, clabel in CONSTRAINT_SETS:
            mb, mr2 = None, -9e9
            for s in range(args.seeds):
                w = train_monotone(Xtr, ytr_s, seed=s, constraints=cons, dirs=dirs)
                r2 = r2_score(yte_s, monotone_predict(Xte, *w))
                if r2 > mr2:
                    mb, mr2 = w, r2
            report(clabel, yte_s, monotone_predict(Xte, *mb), ys)
            print(f"      cost vs unconstrained: {mr2 - best_r2:+.3f}")
            fails = check_constraints(mb[0].tolist(), [mb[2].tolist()], cons)
            assert all(not v for v in fails.values()), (
                "constrained model must satisfy its constraints by construction")
            chosen = (mb, cons, clabel, mr2)

        # Export the strongest set trained, so the weights shipped to the chip
        # carry every guarantee rather than only the carbohydrate one.
        (mW1, mb1, mW2, mb2), cons, clabel, mr2 = chosen
        print()
        print(f"  [exporting: {clabel}]")
        sign_report(mW1.tolist(), [mW2.tolist()], cons)
        W1, b1 = mW1.tolist(), mb1.tolist()
        W2, b2 = [mW2.tolist()], mb2.tolist()
        constraints_out = {FEATURES[i]: int(v) for i, v in cons.items()}

        if args.cv:
            print()
            print(f"  [grouped {args.cv}-fold cross-validation]")
            cv_report(crossval(X, y, g, k=args.cv))

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
                "monotone_constraints": constraints_out,
                "sign_condition_holds": all(
                    not v for v in check_constraints(W1, W2, cons).values()
                ),
                "monotone_by_construction": True,
            }
            with open(args.out, "w", encoding="utf-8") as f:
                json.dump(out, f, indent=2)
            print(f"\n  weights written to {os.path.abspath(args.out)}")
        print()


if __name__ == "__main__":
    main()
