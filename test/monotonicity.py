# SPDX-FileCopyrightText: (c) 2026 Kush Shah
# SPDX-License-Identifier: Apache-2.0
"""
Monotonicity property checker for Proof.

THE PROPERTY
------------
Holding every other input fixed, increasing the carbohydrate input must never
decrease the predicted response. A clinician would insist on it, and it is the
one claim about this chip that is worth proving rather than sampling.

WHY THIS IS A PROPERTY OF THE PIPELINE, NOT OF THE DATA
-------------------------------------------------------
It needs no patient data at all. Whether the property *holds* is a question
about the quantised arithmetic; whether a particular trained network *satisfies
its precondition* is a separate question about weights. Keeping those apart is
what makes this result independent of whether the CGMacros pipeline succeeds.

THE ARGUMENT
------------
Write the pipeline for one output as a composition:

    acc1_j(x_c) = saturating_sum_i( W1[j][i] * x_i )
    h_j         = clamp_0_127( acc1_j >> s1 )
    acc2        = saturating_sum_j( W2[j] * h_j )
    y           = acc2 >> s2

Each stage is monotone in its input:

  * Integer multiply by a constant is non-decreasing if the constant >= 0 and
    non-increasing if it is <= 0.
  * A saturating sum is monotone. This is not obvious and is worth stating:
    with partial sums S_i = clamp(S_{i-1} + t_i), clamp is non-decreasing and
    S_i depends monotonically on S_{i-1}, so by induction the final sum is
    non-decreasing in any single term. **Wrapping would destroy this** -- which
    is the real reason the accumulator must saturate, beyond producing a
    sensible number.
  * Arithmetic right shift is floor division by a power of two: non-decreasing.
    Every standard rounding mode is monotone, so the choice of floor is not
    what puts the property at risk.
  * ReLU and clamping to [0, 127] are non-decreasing.

Composing, y is non-decreasing in x_c provided that for every hidden unit j:

    W1[j][c] * W2[j] >= 0

i.e. each hidden unit either helps carbohydrate raise the response, or opposes
it twice and so still raises it. That is the SIGN CONDITION. If training
produces a hidden unit where the two signs disagree, the property is false by
construction and no amount of RTL stimulus can rescue it -- the fix would be a
constrained training objective.

SO WHERE CAN IT STILL BREAK?
----------------------------
Everything above concerns the *internal* value. The host does not read the
internal value: it reads a fixed-width field. Mode A packs the glycemic load
into 14 bits alongside a 2-bit category; Mode B reports the low 16 bits. Those
fields **truncate rather than saturate**, and truncation is emphatically not
monotone -- it wraps.

That is what this checker is built to find. Run it directly:

    python monotonicity.py
"""

import itertools
import random

import golden_quant as gold

CARB = 0  # index of the carbohydrate input


# --------------------------------------------------------------- reporting --


def reported_gl(pairs, shift):
    """Mode A as the HOST sees it: the 14-bit field, not the internal value."""
    lo, hi = gold.result_bytes(gold.mode_a(pairs, shift))
    return ((hi & 0x3F) << 8) | lo


def internal_gl(pairs, shift):
    """Mode A's full-width value, before it is squeezed into the output field."""
    return gold.mode_a(pairs, shift)["gl"]


def _sign_extend16(v):
    return v - 0x10000 if v & 0x8000 else v


def reported_y(l1, l2, s1, s2, out=0):
    """Mode B as the HOST sees it: the 16-bit field, sign-extended."""
    res = gold.mode_b(l1, l2, s1, s2)
    lo, hi = gold.wide_bytes(res["y"][out])
    return _sign_extend16((hi << 8) | lo)


def internal_y(l1, l2, s1, s2, out=0):
    return gold.mode_b(l1, l2, s1, s2)["y"][out]


# ----------------------------------------------------------- sign condition --


def sign_condition(W1, W2_row, c=CARB):
    """True if every hidden unit's two signs agree, so the property can hold."""
    return all(W1[j][c] * W2_row[j] >= 0 for j in range(len(W1)))


def offending_units(W1, W2_row, c=CARB):
    return [j for j in range(len(W1)) if W1[j][c] * W2_row[j] < 0]


# ------------------------------------------------------------------ sweeps --


def sweep_mode_b(W1, x, W2_row, s1, s2, c=CARB, lo=-128, hi=127):
    """Vary x[c] across its range, returning (internal, reported) pairs."""
    out = []
    for xc in range(lo, hi + 1):
        xs = list(x)
        xs[c] = xc
        l1 = [list(zip(row, xs)) for row in W1]
        l2 = [list(W2_row)]
        res = gold.mode_b(l1, l2, s1, s2)
        y_int = res["y"][0]
        y_rep = _sign_extend16((gold.wide_bytes(y_int)[1] << 8) | gold.wide_bytes(y_int)[0])
        out.append((xc, y_int, y_rep))
    return out


def violations(trace, which):
    """Indices where the sequence decreases. `which` is 1 (internal) or 2 (reported)."""
    bad = []
    for a, b in zip(trace, trace[1:]):
        if b[which] < a[which]:
            bad.append((a[0], a[which], b[0], b[which]))
    return bad


# ------------------------------------------------------------------- study --


def study(trials=400, seed=20260814):
    """Search for monotonicity violations under the sign condition.

    Stimulus is deliberately biased toward the large-magnitude weights that
    push the output past the field width, because uniform sampling almost
    never gets there -- the M9 lesson from the CNN accelerator, where uniform
    INT8 stimulus caught a too-narrow accumulator in only 31 of 200 seeds.
    """
    rng = random.Random(seed)
    n_int = n_rep = 0
    first_rep = None
    checked = 0

    for _ in range(trials):
        # Sign condition satisfied by construction: pick W2 >= 0 and W1[:,c] >= 0.
        big = rng.random() < 0.5
        m = 127 if big else 40
        W1 = [
            [rng.randint(0, m) if i == CARB else rng.randint(-m, m) for i in range(6)]
            for _ in range(gold.N_HIDDEN)
        ]
        W2_row = [rng.randint(0, m) for _ in range(gold.N_HIDDEN)] + [
            rng.randint(-m, m),
            rng.randint(-m, m),
        ]
        assert sign_condition(W1, W2_row)

        x = [rng.randint(-128, 127) for _ in range(6)]
        s1 = rng.randint(0, 3) if big else rng.randint(0, 8)
        s2 = rng.randint(0, 2) if big else rng.randint(0, 8)

        trace = sweep_mode_b(W1, x, W2_row, s1, s2)
        checked += 1
        vi = violations(trace, 1)
        vr = violations(trace, 2)
        if vi:
            n_int += 1
        if vr:
            n_rep += 1
            if first_rep is None:
                first_rep = (W1, x, W2_row, s1, s2, vr[0])

    return {
        "checked": checked,
        "internal_violations": n_int,
        "reported_violations": n_rep,
        "example": first_rep,
    }


def main():
    print("=" * 70)
    print(" MONOTONICITY STUDY -- does increasing carbohydrate ever lower y?")
    print("=" * 70)

    print("\n[1] Sign condition is necessary: break it and the property fails.")
    W1 = [[10] * 6 for _ in range(gold.N_HIDDEN)]
    W1[0][CARB] = -100  # one hidden unit disagrees
    W2_row = [10] * gold.N_HIDDEN + [0, 0]
    print(f"    offending hidden units: {offending_units(W1, W2_row)}")
    tr = sweep_mode_b(W1, [0] * 6, W2_row, 0, 0)
    v = violations(tr, 1)
    print(f"    internal violations: {len(v)}"
          + (f"   first: x_c {v[0][0]}->{v[0][2]} gives y {v[0][1]}->{v[0][3]}" if v else ""))

    print("\n[2] With the sign condition held, is the INTERNAL value monotone?")
    print("    (the argument says yes -- saturating sums and shifts are monotone)")

    print("\n[3] With the sign condition held, is the REPORTED value monotone?")
    print("    (the output fields now saturate; before R-4 they truncated,")
    print("     and 55 of these 400 weight sets reported a falling response)")
    res = study()
    print(f"\n    weight sets checked   : {res['checked']}")
    print(f"    internal violations   : {res['internal_violations']}")
    print(f"    REPORTED violations   : {res['reported_violations']}")
    if res["example"]:
        W1, x, W2_row, s1, s2, (xa, ya, xb, yb) = res["example"]
        print("\n    example violation (reported value):")
        print(f"      s1={s1} s2={s2}  x_c {xa} -> {xb}")
        print(f"      reported y {ya} -> {yb}   (DECREASED by {ya - yb})")
        l1a = [list(zip(r, [xa if i == CARB else v for i, v in enumerate(x)])) for r in W1]
        l1b = [list(zip(r, [xb if i == CARB else v for i, v in enumerate(x)])) for r in W1]
        print(f"      internal y {internal_y(l1a, [W2_row], s1, s2)} -> "
              f"{internal_y(l1b, [W2_row], s1, s2)}   (increased, as it should)")
    print("=" * 70)


if __name__ == "__main__":
    main()
