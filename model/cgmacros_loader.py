#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Kush Shah
# SPDX-License-Identifier: Apache-2.0
"""
Load CGMacros into meal-level records with an iAUC target.

REPRODUCING THE REFERENCE NOTEBOOK'S SCOPE, DELIBERATELY
--------------------------------------------------------
The published `parse_data.ipynb` baseline is much narrower than "CGMacros has
45 participants over 10 days" suggests, and matching its scope is the only way
to compare against its number honestly:

  * breakfasts only
  * Libre glucose only -- the Dexcom column is never touched
  * a two-hour post-meal window sampled every 15 minutes, so 9 points
  * meals with iAUC <= 0 discarded

That lands at roughly ten meals per participant, not thirty to fifty. This
module defaults to exactly that scope so the comparison is like-for-like, and
every narrowing is a named argument that can be widened on purpose rather than
a hidden assumption.

WHAT THE PUBLISHED DATA DICTIONARY GETS WRONG
---------------------------------------------
Verified against the real files, because these would silently produce empty
columns rather than errors:

  * `METs`, not `Mets`
  * `Image path`, not `Image Path`
  * `'Amount Consumed '` carries a TRAILING SPACE in the header
  * timestamps are `YYYY-MM-DD HH:MM:SS`, not the documented
    `Month/Day/Year HH:MM`

There is also an unnamed index column at position 0.

No pandas: the reference notebook needs pandas 1.3.5 (it calls `DataFrame.append`
25 times, removed in 2.0) while the host runs 2.2.3. Staying on the standard
library means this loader is the same code under both interpreters, so the
feature extraction cannot silently diverge between the baseline reproduction
and the training run.
"""

import argparse
import csv
import datetime as dt
import os
import re

from iauc import SIGNED, iauc

# The published data dictionary says 'Factor: Breakfast, Lunch, Dinner'. The
# actual files contain TEN distinct label strings for four meal types --
# 'Breakfast'/'breakfast', 'Lunch'/'lunch', 'Dinner'/'dinner',
# 'Snacks'/'snack'/'Snack'/'snack 1' -- and snacks are not documented at all.
# Filtering on == 'Breakfast' returns 170 of 436 breakfast records, silently
# discarding 61 % of them, so every comparison here is normalised first.
MEAL_TYPES = ("breakfast", "lunch", "dinner", "snack")


def normalise_meal(mt):
    """Map any observed label onto one of MEAL_TYPES, or '' if unrecognised."""
    s = (mt or "").strip().lower()
    if not s:
        return ""
    if s.startswith("snack"):
        return "snack"
    for m in ("breakfast", "lunch", "dinner"):
        if s.startswith(m):
            return m
    return s


# Upper bounds taken from the dataset's OWN published data dictionary, not
# invented here: Carbs/Protein/Fat/Fiber 'Numeric: 0-176', Calories '30-1180'.
# 52 meal records (3.0 %) exceed them -- Fiber as high as 2830 g and Calories
# 2250 -- across 19 participants, and every one falls in a self-selected meal
# (dinner or snack) rather than a standardised breakfast or lunch. A single
# 2830 g fibre entry lifts the training standard deviation for fibre to 32 g
# against a median of 3 g, which is enough to distort feature standardisation
# and, downstream, the INT8 input quantisation.
DOC_MAX = {"carbs": 176.0, "protein": 176.0, "fat": 176.0, "fiber": 176.0,
           "calories": 1180.0}


def implausible(rec):
    """True if a record violates the dataset's own documented ranges."""
    for k, lim in DOC_MAX.items():
        v = rec.get(k)
        if v is not None and v > lim:
            return True
    return False

# Features fed to the chip. Carbohydrate and fibre are supplied SEPARATELY
# rather than as net carbs: CGMacros provides both, and fibre has effects
# beyond simple carbohydrate subtraction. Feeding net carbs *and* fibre would
# double-encode fibre, so this is a deliberate either/or.
FEATURES = ("carbs", "fiber", "fat", "protein", "glucose_pre", "tod")


def _f(row, key):
    """Float field, or None when blank/unparseable."""
    v = row.get(key)
    if v is None:
        return None
    v = v.strip()
    if not v:
        return None
    try:
        return float(v)
    except ValueError:
        return None


def load_participant(path):
    """Return (timestamps, libre, meal_rows) for one participant CSV."""
    times, libre, meals = [], [], []
    with open(path, newline="", encoding="utf-8-sig") as f:
        for i, row in enumerate(csv.DictReader(f)):
            ts = row["Timestamp"].strip()
            try:
                t = dt.datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
            except ValueError:
                continue
            times.append(t)
            libre.append(_f(row, "Libre GL"))
            mt = (row.get("Meal Type") or "").strip()
            if mt:
                meals.append((len(times) - 1, mt, row))
    return times, libre, meals


def meal_records(
    path,
    pid,
    meal_type="breakfast",
    minutes=120,
    step=15,
    convention=SIGNED,
    drop_nonpositive=True,
    plausible=True,
):
    """Meal-level feature/target records for one participant."""
    times, libre, meals = load_participant(path)
    by_time = {t: g for t, g in zip(times, libre)}

    out, skipped_gap, skipped_sign, skipped_impl = [], 0, 0, 0
    for idx, mt, row in meals:
        if meal_type and normalise_meal(mt) != normalise_meal(meal_type):
            continue

        t0 = times[idx]
        offsets = list(range(0, minutes + 1, step))
        series = [by_time.get(t0 + dt.timedelta(minutes=m)) for m in offsets]
        if any(g is None for g in series):
            skipped_gap += 1  # sensor gap or the record ends mid-window
            continue

        area = iauc(offsets, series, convention=convention)
        if drop_nonpositive and area <= 0:
            skipped_sign += 1
            continue

        carbs = _f(row, "Carbs")
        fiber = _f(row, "Fiber")
        fat = _f(row, "Fat")
        protein = _f(row, "Protein")
        if None in (carbs, fiber, fat, protein):
            skipped_gap += 1
            continue

        rec = {
                "pid": pid,
                "meal_type": normalise_meal(mt),
                "t": t0,
                "carbs": carbs,
                "fiber": fiber,
                "fat": fat,
                "protein": protein,
                "calories": _f(row, "Calories"),
                "glucose_pre": series[0],
                "tod": t0.hour + t0.minute / 60.0,
                "iauc": area,
                "peak_rise": max(series) - series[0],
        }
        if plausible and implausible(rec):
            skipped_impl += 1
            continue
        out.append(rec)
    return out, {"gap": skipped_gap, "nonpositive": skipped_sign,
                 "implausible": skipped_impl}


def load_all(root, **kw):
    """Every participant in `root`. Returns (records, per-participant stats)."""
    files = sorted(
        f for f in os.listdir(root)
        if re.fullmatch(r"CGMacros-\d+\.csv", f)
    )
    records, stats = [], []
    for fn in files:
        pid = int(re.findall(r"(\d+)", fn)[0])
        recs, skip = meal_records(os.path.join(root, fn), pid, **kw)
        records.extend(recs)
        stats.append({"pid": pid, "n": len(recs), **skip})
    return records, stats


def load_bio(root):
    """bio.csv keyed by participant id."""
    path = os.path.join(root, "bio.csv")
    if not os.path.exists(path):
        return {}
    out = {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            key = next((k for k in row if k.strip().lower() in ("subject", "pid", "id")), None)
            if key is None:
                continue
            try:
                out[int(float(row[key]))] = row
            except (ValueError, TypeError):
                continue
    return out


def _quantiles(xs):
    xs = sorted(xs)
    if not xs:
        return (0, 0, 0)
    def q(p):
        return xs[min(len(xs) - 1, int(p * len(xs)))]
    return q(0.25), q(0.50), q(0.75)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root", nargs="?",
                    default=r"C:\Users\kushk\Downloads\Claude\cgmacros\csv")
    ap.add_argument("--meal", default="breakfast",
                    help="Breakfast (the baseline's scope), or 'all'")
    ap.add_argument("--keep-nonpositive", action="store_true")
    args = ap.parse_args()

    recs, stats = load_all(
        args.root,
        meal_type=None if args.meal == "all" else args.meal,
        drop_nonpositive=not args.keep_nonpositive,
    )

    print("=" * 66)
    print(f" CGMacros -- {args.meal} meals, 2 h window @ 15 min, Libre, iAUC>0")
    print("=" * 66)
    print(f"  participants      : {len(stats)}")
    print(f"  usable meals      : {len(recs)}")
    if stats:
        per = [s["n"] for s in stats]
        lo, mid, hi = _quantiles(per)
        print(f"  meals/participant : median {mid}  (IQR {lo}-{hi}, "
              f"min {min(per)}, max {max(per)})")
    print(f"  dropped, sensor gap or missing macros : {sum(s['gap'] for s in stats)}")
    print(f"  dropped, iAUC <= 0                    : {sum(s['nonpositive'] for s in stats)}")
    print(f"  dropped, violates published ranges    : {sum(s.get('implausible', 0) for s in stats)}")

    if recs:
        a = [r["iauc"] for r in recs]
        lo, mid, hi = _quantiles(a)
        print(f"\n  iAUC  median {mid:.0f}  IQR {lo:.0f}-{hi:.0f}  "
              f"min {min(a):.0f}  max {max(a):.0f}   (mg/dL.min)")
        for k in ("carbs", "fiber", "fat", "protein", "glucose_pre"):
            v = [r[k] for r in recs]
            lo, mid, hi = _quantiles(v)
            print(f"  {k:12s} median {mid:7.1f}  IQR {lo:6.1f}-{hi:6.1f}  "
                  f"max {max(v):7.1f}")

        zero = sum(1 for s in stats if s["n"] == 0)
        if zero:
            print(f"\n  NOTE: {zero} participant(s) contributed no usable meals")
    print("=" * 66)


if __name__ == "__main__":
    main()
