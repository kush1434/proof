# SPDX-FileCopyrightText: (c) 2026 Kush Shah
# SPDX-License-Identifier: Apache-2.0
"""
Incremental area under the glucose curve (iAUC) -- the prediction target.

WHY iAUC
--------
It is the standard postprandial metric: the FAO/WHO definition of glycemic
index is itself built on incremental AUC, and it is what the CGMacros reference
notebook predicts. Using it means the baseline comparison is against a real
published number rather than an invented target.

THE DEFINITION IS NOT UNIQUE, AND THE CHOICE CHANGES THE NUMBER
---------------------------------------------------------------
Two conventions are both called "iAUC":

  SIGNED    trapezoid the signed difference from baseline. Excursions below
            baseline subtract. Can be negative.
  CLIPPED   clamp the difference at zero first, so only the area above baseline
            counts. Never negative. This is the FAO/WHO convention for
            computing glycemic index.

They disagree whenever glucose dips below its pre-meal value, which is common
late in a two-hour window. The CGMacros notebook discards meals with iAUC <= 0,
which is only meaningful if its iAUC *can* be negative -- so it uses SIGNED.
This module therefore defaults to SIGNED, to reproduce the baseline, and makes
the alternative explicit rather than silent.

**State which convention any reported number uses.** Comparing a CLIPPED iAUC
against the notebook's SIGNED baseline would flatter this project for no reason
other than a definition mismatch.

No pandas or numpy here on purpose: the arithmetic is simple, and keeping it
dependency-free means it runs under both the modern host interpreter and the
pinned legacy environment the notebook needs. It is also why `test_iauc()`
below can verify the maths against shapes with known areas, independently of
whether the dataset has finished downloading.
"""

SIGNED = "signed"
CLIPPED = "clipped"


def trapezoid(xs, ys):
    """Trapezoidal integral. Replaces np.trapz, which NumPy 2 removed."""
    total = 0.0
    for (x0, y0), (x1, y1) in zip(zip(xs, ys), zip(xs[1:], ys[1:])):
        total += (x1 - x0) * (y0 + y1) / 2.0
    return total


def iauc(times, glucose, baseline=None, convention=SIGNED):
    """Incremental AUC of `glucose` over `times`, relative to `baseline`.

    times    minutes from the meal, ascending
    glucose  mg/dL, same length
    baseline pre-meal glucose; defaults to the first sample
    """
    if len(times) != len(glucose):
        raise ValueError("times and glucose must be the same length")
    if len(times) < 2:
        raise ValueError("need at least two samples")
    if any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError("times must be strictly ascending")

    base = glucose[0] if baseline is None else baseline
    diff = [g - base for g in glucose]
    if convention == CLIPPED:
        diff = [d if d > 0 else 0.0 for d in diff]
    elif convention != SIGNED:
        raise ValueError(f"unknown convention {convention!r}")
    return trapezoid(times, diff)


def window(times, glucose, start=0, minutes=120, step=15):
    """Resample a 1-minute series onto the baseline's 15-minute grid.

    The published CSVs are linearly interpolated to 1 minute from natively
    15-minute (Libre) and 5-minute (Dexcom) sensors, so this does not recover
    information -- it reproduces the sampling the reference notebook used: a
    two-hour window at 15 minutes, which is 9 points including t=0.
    """
    want = list(range(start, start + minutes + 1, step))
    idx = {t: i for i, t in enumerate(times)}
    out_t, out_g = [], []
    for t in want:
        if t in idx:
            out_t.append(t)
            out_g.append(glucose[idx[t]])
    return out_t, out_g


def test_iauc():
    """Verify the arithmetic against shapes whose area is known by hand.

    Runs without the dataset, so the integration is trustworthy before any
    real glucose trace is loaded.
    """
    # Flat at baseline: zero area under both conventions.
    t = list(range(0, 121, 15))
    assert iauc(t, [100.0] * len(t)) == 0.0
    assert iauc(t, [100.0] * len(t), convention=CLIPPED) == 0.0

    # Rectangle: +20 mg/dL held for 120 min = 2400 mg/dL.min.
    g = [100.0] + [120.0] * (len(t) - 1)
    # First interval ramps 100->120, so it contributes half: 15*10 = 150.
    assert abs(iauc(t, g) - (150.0 + 20.0 * 105.0)) < 1e-9

    # Symmetric dip and rise cancel under SIGNED, only the rise counts CLIPPED.
    t2 = [0, 30, 60]
    g2 = [100.0, 130.0, 100.0]
    assert abs(iauc(t2, g2) - 900.0) < 1e-9          # two triangles, 450 each
    g3 = [100.0, 70.0, 100.0]
    assert abs(iauc(t2, g3) + 900.0) < 1e-9          # mirror image, negative
    assert iauc(t2, g3, convention=CLIPPED) == 0.0   # nothing above baseline

    # The two conventions must disagree exactly by the sub-baseline area.
    g4 = [100.0, 140.0, 80.0]
    assert iauc(t4 := t2, g4) < iauc(t4, g4, convention=CLIPPED)

    # An explicit baseline overrides the first sample.
    assert abs(iauc(t2, [110.0, 110.0, 110.0], baseline=100.0) - 600.0) < 1e-9

    # Windowing picks the 9 points of a two-hour window on a 1-minute series.
    t1m = list(range(0, 181))
    g1m = [100.0 + i for i in t1m]
    wt, wg = window(t1m, g1m)
    assert wt == [0, 15, 30, 45, 60, 75, 90, 105, 120]
    assert wg[0] == 100.0 and wg[-1] == 220.0

    print("iauc self-test: OK")


if __name__ == "__main__":
    test_iauc()
