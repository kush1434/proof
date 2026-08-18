#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Kush Shah
# SPDX-License-Identifier: Apache-2.0
"""Recover the exact weight set Figure 2 draws, and re-check the 55/400 rate.

`test/monotonicity.py`'s study() searches 400 seeded weight sets for a case
where the REPORTED response falls while carbohydrate rises. Since the output
fields were changed to saturate it correctly reports 0/400, so this repeats the
identical search against the pre-fix truncating field.

Run it to regenerate the constants pasted into fig2_r4_divergence.py:

    python find_r4_case.py
"""

import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "test"))

import golden_quant as gold                              # noqa: E402
from monotonicity import CARB, sweep_mode_b, violations  # noqa: E402


def trunc16(v):
    m = v & 0xFFFF
    return m - 0x10000 if m & 0x8000 else m


def study_truncating(trials=400, seed=20260814):
    """study() verbatim -- same RNG draws, in the same order -- but scoring the
    truncating field. Any change to the draw order desynchronises it, so the
    body below is deliberately a transcription rather than a refactor."""
    rng = random.Random(seed)
    n_int = n_rep = 0
    first = None
    for _ in range(trials):
        big = rng.random() < 0.5
        m = 127 if big else 40
        W1 = [[rng.randint(0, m) if i == CARB else rng.randint(-m, m)
               for i in range(6)] for _ in range(gold.N_HIDDEN)]
        W2 = ([rng.randint(0, m) for _ in range(gold.N_HIDDEN)]
              + [rng.randint(-m, m), rng.randint(-m, m)])
        x = [rng.randint(-128, 127) for _ in range(6)]
        s1 = rng.randint(0, 3) if big else rng.randint(0, 8)
        s2 = rng.randint(0, 2) if big else rng.randint(0, 8)

        trace = sweep_mode_b(W1, x, W2, s1, s2)
        tr = [(xc, y, trunc16(y)) for xc, y, _ in trace]
        if violations(tr, 1):
            n_int += 1
        vr = violations(tr, 2)
        if vr:
            n_rep += 1
            if first is None:
                first = (W1, x, W2, s1, s2, vr[0])
    return n_int, n_rep, first


def main():
    n_int, n_rep, first = study_truncating()
    print(f"weight sets checked   : 400")
    print(f"internal violations   : {n_int}   (the property holds internally)")
    print(f"REPORTED violations   : {n_rep}   (BUGS.md R-4 records 55)")

    W1, x, W2, s1, s2, (xa, ya, xb, yb) = first
    print()
    print("first violating set -- paste into fig2_r4_divergence.py:")
    print(f"W1 = {W1}")
    print(f"X = {x}")
    print(f"W2 = {W2}")
    print(f"S1, S2 = {s1}, {s2}")
    print()
    print(f"  reported x_c {xa} -> {xb}:  {ya} -> {yb}")
    tr = dict((t[0], t[1]) for t in sweep_mode_b(W1, x, W2, s1, s2))
    print(f"  internal x_c {xa} -> {xb}:  {tr[xa]} -> {tr[xb]}")
    print("  BUGS.md R-4 records reported 31293 -> -31209, internal -> 34327")


if __name__ == "__main__":
    main()
