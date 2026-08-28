#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Kush Shah
# SPDX-License-Identifier: Apache-2.0
"""Figure 3 -- where the monotonicity guard sits in the weight stream.

Drawn against `src/proof_core.v`, not from memory. Every label names something
that exists in the RTL:

    first_wt / carb_sign / carb_nz     the layer-1 tap
    hreg / sgnreg / nzreg              the three lockstep registers
    viol_now                           the layer-2 comparison
    mono_viol                          the sticky flag
    untrusted = acc_saturated | mono_viol

THE ONE THING THIS FIGURE HAS TO MAKE OBVIOUS
---------------------------------------------
Both operands of the sign condition W1[j][c] * W2[j] >= 0 already cross the
pins, but they arrive at DIFFERENT TIMES -- W1[j][c] as the first weight byte of
hidden neuron j, W2[j] much later, in layer 2. The guard is cheap because it
never stores a weight: it keeps one sign bit and one non-zero bit per hidden
unit in registers that rotate in lockstep with the hidden activations, so the
bits for the unit being consumed sit at the head of the register at exactly the
moment that unit's layer-2 weight arrives.

That is the whole trick, and it is why the check costs 20 flip-flops rather than
a second copy of the weight memory.

LAYOUT
------
Everything is routed orthogonally on three clear channels -- y = 30.4 above the
datapath, x = 51.5 between the datapath and the registers, and x = 101.5 outside
them -- so no two signals cross. Anchors are computed from the box geometry
rather than typed twice, because hand-placed diagrams drift the moment a box
moves.

matplotlib is not running LaTeX here, so underscores in ordinary labels are
literal characters and must NOT be backslash-escaped.
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import style  # noqa: E402

OUT = os.path.join(HERE, "out", "fig3_architecture")

GUARD = "#8C1D1D"        # everything the guard adds
GUARD_FILL = "#FAEDED"
DATA_FILL = "#EAF1F8"
HOST_FILL = "#F0F0F0"

REG_X, CELL_W, N_H = 63.0, 4.5, 8
REG_END = REG_X + N_H * CELL_W          # 99.0
HEAD_X = REG_X + CELL_W / 2             # centre of the head cell
CHAN_Y, CHAN_X, FAR_X = 30.4, 51.5, 105.5


def main():
    style.setup()
    import matplotlib.pyplot as plt
    from matplotlib.patches import (Circle, FancyArrowPatch, FancyBboxPatch,
                                    Rectangle)

    fig, ax = plt.subplots(figsize=(style.WIDE, 3.05))
    # The axes must fill the canvas. With default margins it covers 77.5 % of
    # the width, and savefig's tight bbox then crops to the artists -- so the
    # PDF comes out at 5.6 in rather than the 7.16 in authored here, and lands
    # in the paper at the wrong scale.
    ax.set_position([0, 0, 1, 1])
    ax.set_xlim(0, 108)
    ax.set_ylim(0, 42)
    ax.axis("off")

    def box(x, y, w, h, label, fill, edge=style.INK, fs=7.0, tc=None):
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0,rounding_size=0.7",
            fc=fill, ec=edge, lw=0.7, zorder=3))
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
                fontsize=fs, zorder=4, linespacing=1.3, color=tc or style.INK)

    def arrow(x1, y1, x2, y2, color=style.INK, lw=0.7, z=5):
        ax.add_patch(FancyArrowPatch(
            (x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=6,
            color=color, lw=lw, zorder=z, shrinkA=0, shrinkB=0))

    def route(pts, color=style.INK, lw=0.7, z=5):
        """Orthogonal polyline, arrowhead on the last leg only."""
        for a, b in zip(pts, pts[1:-1]):
            ax.plot([a[0], b[0]], [a[1], b[1]], color=color, lw=lw, zorder=z,
                    solid_capstyle="round")
        arrow(*pts[-2], *pts[-1], color=color, lw=lw, z=z)

    # ============================================= host and the byte stream ==
    box(0.5, 34.6, 15.0, 6.0,
        "HOST\n(not trusted)\nper-patient weights", HOST_FILL, fs=7.0)
    ax.text(27.5, 41.8, "8-bit payload, with is_weight / last / mode",
            fontsize=7.0, color=style.GREY, va="top")
    arrow(15.5, 37.6, 26.8, 37.6, color=style.GREY)

    def cells(x, y, items, hi=None):
        centres = []
        for i, (w, txt) in enumerate(items):
            on = (hi is not None and i == hi)
            ax.add_patch(Rectangle(
                (x, y), w, 2.9, fc=GUARD_FILL if on else "white",
                ec=GUARD if on else style.GREY, lw=1.0 if on else 0.5,
                zorder=3))
            ax.text(x + w / 2, y + 1.45, txt, ha="center", va="center",
                    fontsize=7.0, zorder=6,
                    bbox=dict(facecolor="white", edgecolor="none",
                              pad=0.8), color=GUARD if on else style.INK)
            centres.append(x + w / 2)
            x += w
        return centres

    c1 = cells(27.5, 36.3, [(6.5, "$s_1$"), (14, "$W_1[j][c]$"), (7, "$x_c$"),
                            (13, "$W_1[j][1]$"), (7, "$x_1$"), (9, "$\\ldots$"),
                            (9, "LAST")], hi=1)
    c2 = cells(27.5, 32.6, [(6.5, "$s_2$"), (11, "$W_2[0]$"), (11, "$W_2[1]$"),
                            (8, "$\\ldots$"), (11, "$W_2[7]$"),
                            (14, "bias, LAST")])
    ax.text(25.4, 37.75, "layer 1,\nneuron $j$", fontsize=7.0, ha="right",
            va="center", linespacing=1.2)
    ax.text(25.4, 34.05, "layer 2", fontsize=7.0, ha="right", va="center")

    ax.plot([0.5, 107.5], [31.4, 31.4], ls=(0, (3, 2.2)), lw=0.7,
            color=style.GREY, zorder=1)
    ax.text(0.5, 31.7, "chip boundary", fontsize=7.0, color=style.GREY,
            va="bottom")

    # ================================================== the shared datapath ==
    box(1.0, 23.6, 13.5, 4.8, "serial MAC\n8$\\times$8", DATA_FILL)
    box(17.0, 23.6, 15.5, 4.8, "saturating\naccumulator, 24 b", DATA_FILL)
    box(35.0, 23.6, 15.0, 4.8, "$\\gg s$, ReLU,\nclamp to [0,127]", DATA_FILL)
    arrow(14.5, 26.0, 17.0, 26.0)
    arrow(32.5, 26.0, 35.0, 26.0)
    arrow(7.75, 34.6, 7.75, 28.4, color=style.GREY)
    ax.text(1.0, 22.6, "a saturating sum is monotone;\na wrapping one is not",
            fontsize=7.0, color=style.BLUE, va="top", linespacing=1.25)

    # h_j is pushed into the HEAD of hreg, so it enters from above. It has to
    # get past the layer-1 tap channel, and no routing avoids that: the tap
    # runs top-to-bottom and h_j runs left-to-right through the same band. So
    # the crossing is drawn as a hop, the usual schematic mark for "these wires
    # do not connect", rather than hidden by contorting the layout.
    import numpy as np
    ax.plot([50.0, 50.0], [26.0, 29.6], color=style.INK, lw=0.7, zorder=5)
    ax.plot([50.0, CHAN_X - 0.9], [29.6, 29.6], color=style.INK, lw=0.7,
            zorder=5, solid_capstyle="round")
    th = np.linspace(np.pi, 0.0, 40)
    ax.plot(CHAN_X + 0.9 * np.cos(th), 29.6 + 0.75 * np.sin(th),
            color=style.INK, lw=0.7, zorder=6)
    ax.plot([CHAN_X + 0.9, HEAD_X], [29.6, 29.6], color=style.INK, lw=0.7,
            zorder=5, solid_capstyle="round")
    arrow(HEAD_X, 29.6, HEAD_X, 27.7)
    ax.text(56.0, 30.0, "$h_j$", fontsize=7.0, va="bottom", ha="left")

    # ========================================= the three lockstep registers ==
    rows = [
        ("hreg",   "hidden activation $h[j]$", 24.6, 3.1, style.INK, DATA_FILL),
        ("sgnreg", "sign of $W_1[j][c]$",      20.9, 2.6, GUARD, GUARD_FILL),
        ("nzreg",  "$W_1[j][c] \\neq 0$",      17.7, 2.6, GUARD, GUARD_FILL),
    ]
    for name, gloss, y, h, ec, fc in rows:
        for i in range(N_H):
            head = (i == 0)
            ax.add_patch(Rectangle(
                (REG_X + i * CELL_W, y), CELL_W, h,
                fc=fc if head else "white", ec=ec,
                lw=1.0 if head else 0.45, zorder=3))
        # RTL name only. The gloss belongs in the caption; keeping it here
        # collided with the W2[k] bus routed outside the block.
        ax.text(REG_END + 1.0, y + h / 2, name, ha="left", va="center",
                fontsize=7.0, color=ec, family="monospace")

    ax.add_patch(FancyArrowPatch(
        (REG_END - 0.5, 28.9), (REG_X + 0.5, 28.9), arrowstyle="-|>",
        mutation_scale=6, color=style.GREY, lw=0.7, zorder=5))
    ax.text(REG_X + 4.7 * CELL_W, 16.4,
            "rotates one place per layer-2 weight byte, all three together",
            fontsize=7.0, color=style.GREY, ha="center", va="top")
    ax.text(HEAD_X - 1.6, 16.5, "head", fontsize=7.0, ha="right", va="top")

    ax.add_patch(Rectangle((REG_X - 0.7, 17.1), N_H * CELL_W + 1.4, 6.9,
                           fc="none", ec=GUARD, lw=0.6, ls=(0, (2, 1.6)),
                           zorder=2))

    # =============================================== the guard: layer-1 tap ==
    TAP_X = 45.0     # the W_2[0] / W_2[1] boundary, inside the W_1[j][c] cell
    route([(TAP_X, 36.3), (TAP_X, CHAN_Y), (CHAN_X, CHAN_Y), (CHAN_X, 23.6)],
          color=GUARD, lw=0.9)
    box(42.0, 17.7, 19.0, 5.8,
        "capture the sign bit\nand \"is non-zero\" of\nthe FIRST weight byte",
        GUARD_FILL, edge=GUARD, fs=7.0, tc=GUARD)
    arrow(61.0, 22.2, 62.8, 22.2, color=GUARD, lw=0.9)
    arrow(61.0, 19.0, 62.8, 19.0, color=GUARD, lw=0.9)
    ax.text(52.4, 26.4, "$W_1[j][c]$\n(carbohydrate\nis input 0)",
            fontsize=7.0, color=GUARD, ha="left", va="center",
            linespacing=1.25)

    # ======================================== the guard: layer-2 comparison ==
    route([(c2[4], 32.6), (c2[4], 32.0), (FAR_X, 32.0), (FAR_X, 5.4),
           (67.2, 5.4)], color=GUARD, lw=0.9)
    ax.text(84.0, 5.8, "$W_2[k]$", fontsize=7.0, color=GUARD, ha="center",
            va="bottom")

    box(40.0, 2.4, 31.5, 6.0,
        "sign disagrees with the head,\nand both are non-zero\n"
        "$\\Rightarrow$ this weight set does not admit the guarantee",
        GUARD_FILL, edge=GUARD, fs=7.0, tc=GUARD)
    arrow(HEAD_X, 17.6, HEAD_X, 8.5, color=GUARD, lw=0.9)

    box(21.0, 4.4, 15.0, 5.0, "sticky\nmono_viol", GUARD_FILL, edge=GUARD,
        fs=7.0, tc=GUARD)
    arrow(40.0, 6.9, 36.2, 6.9, color=GUARD, lw=0.9)

    # ====================================== one pin, two ways to be wrong ==
    box(6.0, 10.6, 16.0, 4.4, "acc_saturated", DATA_FILL, fs=7.0)
    arrow(17.5, 23.6, 14.4, 15.2, color=style.INK, lw=0.6)

    ax.add_patch(Circle((14.0, 6.9), 1.35, fc="white", ec=style.INK, lw=0.7,
                        zorder=5))
    ax.text(14.0, 6.9, "$\\geq$1", fontsize=7.0, ha="center", va="center",
            zorder=6)
    arrow(14.0, 10.6, 14.0, 8.4, color=style.INK, lw=0.6)
    arrow(21.0, 6.9, 15.5, 6.9, color=GUARD, lw=0.6)
    arrow(12.6, 6.9, 0.4, 6.9, color=style.INK, lw=1.0, z=6)
    ax.text(0.4, 7.4, "UNTRUSTED", fontsize=7.0, fontweight="bold",
            ha="left", va="bottom")
    ax.text(0.4, 5.6, "overflow, or a\nvoid guarantee",
            fontsize=7.0, color=style.GREY, ha="left", va="top",
            linespacing=1.2)

    # ------------------------------------------------------------- legend --
    ax.add_patch(Rectangle((26.0, 11.4), 3.0, 1.9, fc=GUARD_FILL, ec=GUARD,
                           lw=0.9, zorder=3))
    ax.text(30.2, 12.35, "added by the guard: 20 flip-flops of the 168 built",
            fontsize=7.0, va="center", color=GUARD)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    style.save(fig, OUT, width=style.WIDE)


if __name__ == "__main__":
    main()
