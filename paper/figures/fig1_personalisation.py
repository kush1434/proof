#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Kush Shah
# SPDX-License-Identifier: Apache-2.0
"""Figure 1 -- how many meals before personalisation is worth anything.

Reads the JSON written by `model/personalise.py --json`, so every point here
came out of the run rather than off the terminal. Regenerate the input with:

    cd model && python personalise.py --json ../paper/data/personalise.json

TWO PANELS, AND WHY BOTH
------------------------
(a) is the pooled curve -- R^2 over every held-out meal at each k. It shows the
cost of entry: one meal is WORSE than none.

(b) is the honest panel. It is the per-participant paired change in MAE with a
95 % CI, so the reader can see which points are resolved and which are not.
The pooled curve in (a) cannot show that, and a pooled curve on its own would
imply a precision the 44-participant cohort does not have.

The cohort SHRINKS at k = 12 and 16 -- a participant needs k + 6 meals to be
scored at all -- so those points are not the same people as the rest. That is
marked on the axis rather than left for the caption, because a curve that
silently changes its denominator is the kind of figure that gets a paper into
trouble.
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import style  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data", "personalise.json")
OUT = os.path.join(HERE, "out", "fig1_personalisation")

FULL_COHORT = 44  # participants with enough meals to be scored at every k


def main():
    with open(DATA, encoding="utf-8") as f:
        d = json.load(f)

    style.setup()
    import matplotlib.pyplot as plt

    ks = [k for k in d["k_values"] if str(k) in d["curve"]]
    rows = [d["curve"][str(k)] for k in ks]

    r2 = [r["r2"] for r in rows]
    subj = [r["subjects"] for r in rows]

    # Paired per-participant change, population -> personalised. k = 0 is the
    # population model compared with itself, so it has no delta by definition.
    deltas, los, his = [], [], []
    for r in rows:
        per = np.array(r["per_subject"], float)  # (pid, pop MAE, personal MAE)
        dd = per[:, 2] - per[:, 1]
        se = dd.std(ddof=1) / np.sqrt(len(dd))
        deltas.append(dd.mean())
        los.append(dd.mean() - 1.96 * se)
        his.append(dd.mean() + 1.96 * se)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(style.COL, 3.2), sharex=True,
        gridspec_kw={"height_ratios": [1, 1.15], "hspace": 0.16})

    # ---------------------------------------------------------- panel (a) --
    ax1.axhline(r2[0], color=style.GREY, ls=(0, (4, 2)), lw=0.8, zorder=1)
    ax1.text(16.4, r2[0], "population\nmodel", color=style.GREY, fontsize=6,
             va="center", ha="left", linespacing=1.1)
    ax1.axhline(0, color=style.LIGHT, lw=0.6, zorder=0)
    ax1.plot(ks, r2, "-o", color=style.BLUE, ms=3.2, mfc="white", mew=1.0,
             zorder=3, clip_on=False)

    # The one point that argues for the whole design decision.
    ax1.annotate("one meal is\nworse than none",
                 xy=(1, r2[1]), xytext=(2.6, -0.05),
                 fontsize=6.2, color=style.VERM, linespacing=1.15,
                 arrowprops=dict(arrowstyle="-", color=style.VERM, lw=0.6,
                                 shrinkA=0, shrinkB=2))
    ax1.plot([1], [r2[1]], "o", color=style.VERM, ms=3.2, zorder=4,
             clip_on=False)

    ax1.set_ylabel("$R^2$, held-out meals")
    ax1.set_ylim(-0.15, 0.38)
    ax1.set_yticks([-0.1, 0.0, 0.1, 0.2, 0.3])
    ax1.text(-0.5, 0.355, "(a)", fontsize=8, fontweight="bold", va="top")

    # ---------------------------------------------------------- panel (b) --
    # k = 0 is omitted: it is the population model compared with itself, so its
    # delta is identically zero and drawing it as an unresolved point would
    # claim an uncertainty that does not exist.
    ax2.axhline(0, color=style.INK, lw=0.7, zorder=1)
    for k, dv, lo, hi in zip(ks, deltas, los, his):
        if k == 0:
            continue
        resolved = (lo > 0) or (hi < 0)
        c = style.VERM if lo > 0 else (style.BLUE if hi < 0 else style.GREY)
        ax2.plot([k, k], [lo, hi], color=c, lw=1.0, solid_capstyle="butt",
                 zorder=2)
        ax2.plot([k], [dv], "o", color=c, ms=3.2,
                 mfc=c if resolved else "white", mew=1.0, zorder=3)

    ax2.set_ylabel("$\\Delta$ MAE per participant\n(mg/dL$\\cdot$min)",
                   linespacing=1.2)
    # Room for the participant-count row, which sits between the tick labels
    # and the axis label.
    ax2.set_xlabel("meals logged before personalising, $k$", labelpad=15)
    ax2.set_ylim(-460, 700)
    ax2.set_yticks([-400, -200, 0, 200, 400, 600])
    ax2.text(-0.5, 660, "(b)", fontsize=8, fontweight="bold", va="top")

    ax2.text(0.012, 0.045, "$\\downarrow$ personalising helps",
             transform=ax2.transAxes, fontsize=6.2, color=style.GREY)

    # Filled marker = the interval excludes zero. Say so, once.
    ax2.text(4.6, 430,
             "filled: 95 % CI excludes zero\nopen: not resolved",
             fontsize=6.2, color=style.GREY, linespacing=1.25)

    # ------------------------------------------------------------- x axis --
    ax2.set_xticks(ks)
    ax2.set_xticklabels([str(k) for k in ks])
    ax2.set_xlim(-0.7, 17)

    # The denominator changes at k = 12 and 16 -- a participant needs k + 6
    # meals to be scored, so the largest k are not the same people. Marked on
    # the axis, below the tick labels, rather than left to the caption.
    for k, n in zip(ks, subj):
        if n != FULL_COHORT:
            for ax in (ax1, ax2):
                ax.axvspan(k - 0.9, k + 0.9, color="#F4F4F4", zorder=0, lw=0)
        ax2.annotate(f"{n}", xy=(k, 0), xycoords=("data", "axes fraction"),
                     xytext=(0, -17), textcoords="offset points",
                     fontsize=5.8, color=style.GREY, ha="center", va="top")
    ax2.annotate("participants", xy=(0, 0), xycoords=("axes fraction", "axes fraction"),
                 xytext=(-4, -17), textcoords="offset points",
                 fontsize=5.8, color=style.GREY, ha="right", va="top")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    style.save(fig, OUT)

    # Echo what was drawn, so a stale JSON is visible rather than silent.
    print("  k   n     R2      dMAE  [95% CI]")
    for k, n, r, dv, lo, hi in zip(ks, subj, r2, deltas, los, his):
        print(f"  {k:<3d} {n:<4d} {r:+.3f}  {dv:+7.0f}  [{lo:+7.0f},{hi:+7.0f}]")


if __name__ == "__main__":
    main()
