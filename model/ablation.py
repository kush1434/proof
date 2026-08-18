#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Kush Shah
# SPDX-License-Identifier: Apache-2.0
"""Do participant-level biological features help? Measured, not assumed.

WHY THIS SCRIPT EXISTS
----------------------
The number of inputs is NOT fixed in silicon -- a neuron ends when the host
asserts LAST -- so extra per-participant features cost nothing in hardware.
That makes "we use six meal features" a choice that has to be defended on
evidence rather than on area, and RESULTS.md 5 defends it with an ablation
table that nothing in the repository computed. This computes it.

THE ASYMMETRY THAT MAKES THIS INTERESTING
------------------------------------------
A participant-level feature is CONSTANT WITHIN A PERSON. It can therefore shift
one person's predictions up or down as a block, which helps population R^2, but
it cannot reorder that person's own meals -- so it cannot improve within-person
Spearman by construction. Ranking is the use case, so the two columns below
must be read together: a feature that raises R^2 and leaves rho alone has not
helped the thing the device is for.

WHICH FIVE "BIO" FEATURES
--------------------------
RESULTS.md 5 says "all 5 bio" without naming them, and the original selection
is not recorded anywhere in the repository. The five used here are named
explicitly in BIO_SETS below so that this table means something specific; they
are the clinically standard ones in `bio.csv`. **This is therefore a fresh
measurement with a stated feature set, not a reproduction of the earlier
table**, and its numbers should be expected to differ.

Everything else matches `clinical.py`: grouped 5-fold cross-fitting so every
participant is held out exactly once, the carbohydrate-constrained model that
actually ships, and Spearman over participants with at least 8 held-out meals.

    python ablation.py
"""

import argparse
import csv
import os

import numpy as np

from cgmacros_loader import FEATURES, load_all
from clinical import (fit_oof, group_of, within_participant_spearman)
from train import CARB

DEFAULT_ROOT = r"C:\Users\kushk\Downloads\Claude\cgmacros\csv"

# Column names as they appear in bio.csv, trailing spaces and all. Matched
# case-insensitively on a stripped prefix, because that header is not tidy.
BIO_COLUMNS = {
    "a1c":      "A1c PDL (Lab)",
    "bmi":      "BMI",
    "age":      "Age",
    "fasting":  "Fasting GLU - PDL (Lab)",
    "insulin":  "Insulin",
}

BIO_SETS = [
    ("6 meal features",  []),
    ("+ A1c",            ["a1c"]),
    ("+ A1c + BMI",      ["a1c", "bmi"]),
    ("+ all 5 bio",      ["a1c", "bmi", "age", "fasting", "insulin"]),
]


def load_bio_table(root):
    """participant id -> {key: float} for the columns in BIO_COLUMNS."""
    path = os.path.join(root, "bio.csv")
    out = {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            keymap = {k.strip().lower(): k for k in row}
            sub = keymap.get("subject")
            if sub is None:
                continue
            try:
                pid = int(float(row[sub]))
            except (ValueError, TypeError):
                continue
            rec = {}
            for short, want in BIO_COLUMNS.items():
                col = keymap.get(want.strip().lower())
                if col is None:
                    continue
                try:
                    rec[short] = float(row[col])
                except (ValueError, TypeError):
                    pass
            out[pid] = rec
    return out


def build_with_bio(root, bio_keys):
    """Meal features, plus the named per-participant features appended.

    Participants missing any requested value are dropped, so every row in the
    comparison has the same feature set -- imputing here would quietly change
    what is being measured.
    """
    recs, _ = load_all(root, meal_type=None, drop_nonpositive=True)
    bio = load_bio_table(root)

    keep = []
    for r in recs:
        vals = bio.get(r["pid"], {})
        if any(k not in vals for k in bio_keys):
            continue
        keep.append((r, [vals[k] for k in bio_keys]))

    X = np.array([[r[f] for f in FEATURES] + extra for r, extra in keep], float)
    y = np.array([r["iauc"] for r, _ in keep], float)
    g = np.array([r["pid"] for r, _ in keep], int)
    return X, y, g


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=DEFAULT_ROOT)
    args = ap.parse_args()

    print("=" * 78)
    print(" FEATURE ABLATION -- do per-participant features help?")
    print("=" * 78)
    print("  inputs are not fixed in silicon, so these are free in hardware;")
    print("  the question is entirely whether they earn their place")
    print()
    print("  bio columns used:")
    for short, col in BIO_COLUMNS.items():
        print("    %-8s %s" % (short, col))
    print()
    print("  %-18s %6s %7s %8s %10s %10s" %
          ("features", "meals", "subj", "R2", "rho (med)", "positive"))

    rows = []
    for label, keys in BIO_SETS:
        X, y, g = build_with_bio(args.root, keys)
        res = fit_oof(X, y, g, constraints={CARB: +1})
        rhos = within_participant_spearman(res, min_meals=8)
        vals = np.array([r[2] for r in rhos]) if rhos else np.array([0.0])
        rows.append((label, len(y), len(set(g.tolist())), res["r2"],
                     float(np.median(vals)), int((vals > 0).sum()), len(vals)))
        print("  %-18s %6d %7d %+8.3f %+10.3f %6d/%-4d" % rows[-1])

    print()
    print("  Read the last two columns together: a participant-level feature is")
    print("  constant within a person, so it CANNOT reorder that person's meals.")
    print("  Ranking is the use case.")
    print()
    print("  RESULTS.md 5 records (feature set not named there, so not directly")
    print("  comparable): 6 meal +0.236 / +0.403 / 40 of 44;  + A1c +0.271;")
    print("  + A1c + BMI +0.224;  + all 5 bio -0.005")
    print("=" * 78)


if __name__ == "__main__":
    main()
