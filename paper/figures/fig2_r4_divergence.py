#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Kush Shah
# SPDX-License-Identifier: Apache-2.0
"""Figure 2 -- the safety property held internally and broke at the pins.

This is defect R-4. Every stage of the datapath is monotone in carbohydrate --
saturating sums, arithmetic shifts, ReLU and clamps all are -- so the INTERNAL
response provably never falls as carbohydrate rises. But the host never reads
the internal value. It reads a fixed-width field, and that field TRUNCATED.
Truncation wraps, and wrapping is not monotone.

No stimulus-driven test could have found this: the RTL and the reference model
truncated in the same way, so every comparison between them passed. It took
stating the property and checking it directly.

DATA
----
Generated here, from `test/golden_quant.py` via `test/monotonicity.py`, with the
weight set that `monotonicity.study()`'s own seeded search reaches first. The
shipped model saturates, so the pre-fix curve is reconstructed by applying
truncation to the internal value -- which is exactly what the old field did:

    trunc16(v) = sign_extend_16(v & 0xFFFF)

Both the 55/400 rate and this example's numbers match BUGS.md R-4.
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "test"))

import style  # noqa: E402
from monotonicity import CARB, sweep_mode_b  # noqa: E402

OUT = os.path.join(HERE, "out", "fig2_r4_divergence")

# The first weight set monotonicity.study(seed=20260814) reaches that violates
# the property through a truncating field. Recorded explicitly so this figure
# does not depend on re-running a 400-trial search, and re-derivable with
# paper/figures/find_r4_case.py.
W1 = [[52, -26, -39, 97, -61, -79],
      [38, 98, 107, 96, 85, 23],
      [55, 25, 52, -115, -105, 8],
      [75, 111, 45, 110, -1, 91],
      [117, 72, -71, -55, -33, -4],
      [52, 50, 8, -127, -118, -41],
      [104, -101, -19, 75, -86, 123],
      [75, -28, 84, 4, 100, 31]]
X = [16, -95, -68, 14, -32, 75]
W2 = [104, 48, 10, 82, 34, 31, 87, 117, -120, 107]
S1, S2 = 1, 0

FIELD_MAX = 32767
FIELD_MIN = -32768


def trunc16(v):
    """The pre-R-4 output field: keep 16 bits, sign-extend, and wrap."""
    m = v & 0xFFFF
    return m - 0x10000 if m & 0x8000 else m


def main():
    trace = sweep_mode_b(W1, X, W2, S1, S2, c=CARB)
    xs = [t[0] for t in trace]
    internal = [t[1] for t in trace]
    shipped = [t[2] for t in trace]      # saturating field, as built
    truncated = [trunc16(v) for v in internal]

    # First place the truncating field falls while the input rises.
    wrap = next(i for i in range(1, len(xs))
                if truncated[i] < truncated[i - 1])

    style.setup()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(style.COL, 2.2))

    lo, hi = wrap - 24, wrap + 26
    sl = slice(max(0, lo), min(len(xs), hi))

    ax.axhline(FIELD_MAX, color=style.GREY, ls=(0, (3, 2)), lw=0.7, zorder=1)
    ax.text(xs[sl][-1] - 0.5, FIELD_MAX + 1400, "16-bit field limit",
            fontsize=6, color=style.GREY, ha="right")

    ax.plot(xs[sl], internal[sl], "-", color=style.BLUE, lw=1.3, zorder=4,
            label="internal value")
    ax.plot(xs[sl], truncated[sl], "-", color=style.VERM, lw=1.1, zorder=3,
            label="reported: truncating field")
    ax.plot(xs[sl], shipped[sl], ls=(0, (1.4, 1.4)), color=style.GREEN, lw=1.4,
            zorder=5, label="reported: saturating (built)")

    # The documented event: one extra count of carbohydrate, and the number the
    # host is handed falls by 62,502.
    xa, xb = xs[wrap - 1], xs[wrap]
    ya, yb = truncated[wrap - 1], truncated[wrap]
    ax.annotate("", xy=(xb, yb), xytext=(xa, ya),
                arrowprops=dict(arrowstyle="->", color=style.VERM, lw=0.9,
                                shrinkA=1.5, shrinkB=1.5))
    def num(v):
        return f"{v:,}".replace("-", "$-$")

    ax.annotate(f"+1 count of carbohydrate:\nreported {num(ya)} "
                f"$\\rightarrow$ {num(yb)}",
                xy=(xb + 0.4, (ya + yb) / 2), xytext=(xb + 3.5, -3000),
                fontsize=6.2, color=style.VERM, linespacing=1.25,
                arrowprops=dict(arrowstyle="-", color=style.VERM, lw=0.5,
                                shrinkA=0, shrinkB=2))
    ax.annotate(f"true value rises\n{num(internal[wrap-1])} "
                f"$\\rightarrow$ {num(internal[wrap])}",
                xy=(xb + 1.5, internal[wrap]), xytext=(xa - 22, 43000),
                fontsize=6.2, color=style.BLUE, linespacing=1.25,
                arrowprops=dict(arrowstyle="-", color=style.BLUE, lw=0.5,
                                shrinkA=0, shrinkB=2))

    ax.set_xlabel("carbohydrate input $x_c$ (INT8 counts)")
    ax.set_ylabel("output value")
    ax.set_xlim(xs[sl][0], xs[sl][-1])
    ax.set_ylim(-42000, 52000)
    ax.set_yticks([-32768, 0, 32767])
    ax.set_yticklabels(["$-$32768", "0", "32767"])
    # Lower left: the only quadrant no curve enters before the wrap.
    ax.legend(loc="lower left", handlelength=1.9, borderaxespad=0.3,
              labelspacing=0.28)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    style.save(fig, OUT)

    falls = sum(1 for i in range(1, len(xs)) if truncated[i] < truncated[i - 1])
    bad_ship = sum(1 for i in range(1, len(xs)) if shipped[i] < shipped[i - 1])
    bad_int = sum(1 for i in range(1, len(xs)) if internal[i] < internal[i - 1])
    print(f"  sweep x_c {xs[0]}..{xs[-1]}  (s1={S1}, s2={S2})")
    print(f"  falls, internal value      : {bad_int}")
    print(f"  falls, truncating field    : {falls}")
    print(f"  falls, saturating field    : {bad_ship}")
    print(f"  first fall at x_c {xa} -> {xb}: {ya} -> {yb}")
    print(f"  internal there            : {internal[wrap-1]} -> {internal[wrap]}")


if __name__ == "__main__":
    main()
