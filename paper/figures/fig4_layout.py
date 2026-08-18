#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Kush Shah
# SPDX-License-Identifier: Apache-2.0
"""Figure 4 -- the routed 1x1 tile, and a detail at cell scale.

The render is the `gds_render` artifact from the Tiny Tapeout `gds` workflow, so
it is the actual signed-off layout rather than a re-run. The RTL inside that
artifact was diffed against `src/` and is byte-identical.

    gh run download <run-id> --repo kush1434/proof -n gds_render

WHY AN INSET
------------
At column width the whole tile reads as texture, which is honest -- 83.5 %
utilisation looks full because it is -- but texture alone does not show that
these are placed standard cells. The inset is a 1:1 crop, so the reader can see
individual cells and the routing over them.

The numbers in the caption come from the same run's `stats/metrics.csv`:
design__instance__utilization 0.835308, design__instance__count__stdcell 1443,
design__instance__count__class:sequential_cell 168, and zero for each of
magic__drc_error__count, design__lvs_error__count and
route__antenna_violation__count.
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import style  # noqa: E402

OUT = os.path.join(HERE, "out", "fig4_layout")
RENDER = os.path.join(HERE, "..", "data", "gds_render.png")

# Crop for the inset, in source pixels: a band away from the edges, so it shows
# ordinary placed rows rather than the atypical cells at the boundary.
CROP = (250, 470, 250 + 330, 470 + 250)


def main():
    style.setup()
    import matplotlib.pyplot as plt
    from matplotlib.patches import ConnectionPatch, Rectangle
    from PIL import Image

    img = Image.open(RENDER).convert("RGB")
    W, H = img.size
    crop = img.crop(CROP)

    w = 0.86 * style.COL   # the render does not need full column width
    fig, ax = plt.subplots(figsize=(w, w * H / W))
    # Fill the canvas, so the PDF really is `w` wide -- savefig's tight bbox
    # otherwise crops to the artists and the figure lands at the wrong scale.
    ax.set_position([0, 0, 1, 1])
    ax.imshow(img, interpolation="antialiased")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.6)
        s.set_edgecolor(style.INK)

    # Where the detail came from.
    x0, y0, x1, y1 = CROP
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fc="none",
                           ec="#FFFFFF", lw=1.6, zorder=5))
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fc="none",
                           ec=style.INK, lw=0.8, zorder=6))

    # Inset, bottom right, at the crop's own aspect ratio.
    cw = 0.38
    ch = cw * (W / H) * (y1 - y0) / (x1 - x0)
    axin = ax.inset_axes([1 - cw - 0.012, 0.012, cw, ch])
    axin.imshow(crop, interpolation="nearest")
    axin.set_xticks([])
    axin.set_yticks([])
    for s in axin.spines.values():
        s.set_linewidth(0.8)
        s.set_edgecolor(style.INK)

    # Source corners to the matching inset corners: top-right of the crop meets
    # the top-left of the inset, bottom-right meets bottom-left.
    for src, dst in (((x1, y0), (0.0, 1.0)), ((x1, y1), (0.0, 0.0))):
        ax.add_patch(ConnectionPatch(
            xyA=src, coordsA=ax.transData,
            xyB=dst, coordsB=axin.transAxes,
            color=style.INK, lw=0.5, zorder=7))

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    style.save(fig, OUT)
    print(f"  render {W}x{H}, inset {x1 - x0}x{y1 - y0} px at 1:1")


if __name__ == "__main__":
    main()
