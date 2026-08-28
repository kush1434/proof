# SPDX-FileCopyrightText: (c) 2026 Kush Shah
# SPDX-License-Identifier: Apache-2.0
"""Shared figure style for the BIBM submission.

IEEE conference format, so:

  * column width is 3.45 in and the full text width 7.16 in -- figures are
    authored at final size and never scaled in LaTeX, because scaling is what
    makes one figure's labels smaller than another's;
  * Times, to match the body text;
  * 8 pt labels, which is IEEE's caption size and about the smallest that
    survives print;
  * every series is distinguishable WITHOUT colour. IEEE prints in greyscale
    unless you pay, and a reviewer may print it that way regardless, so colour
    carries no information that line style and marker do not also carry.

PDF is the deliverable (vector, so it stays sharp at any zoom); PNG is written
alongside it only for quick viewing.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

COL = 3.45          # IEEE single column, inches
WIDE = 7.16         # IEEE full text width, inches

# Okabe-Ito, which is colourblind-safe, and ordered so the first two are also
# far apart in luminance and therefore survive greyscale conversion.
INK = "#111111"
BLUE = "#0072B2"
VERM = "#D55E00"
GREEN = "#009E73"
GREY = "#8C8C8C"
LIGHT = "#DDDDDD"


def setup():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.linewidth": 0.6,
        "grid.linewidth": 0.4,
        "lines.linewidth": 1.1,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "figure.dpi": 150,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.01,
        "pdf.fonttype": 42,     # embed TrueType, not Type 3: IEEE PDF eXpress
        "ps.fonttype": 42,      # rejects Type 3 fonts
    })


def fit_to_width(fig, target_in, tol_pt=0.25, tries=8):
    """Resize the figure so its TIGHT bounding box is exactly `target_in` wide.

    This module promises figures are authored at final size and never scaled in
    LaTeX. `savefig.bbox = "tight"` quietly broke that promise: it crops to the
    drawn content, so the saved width is whatever the labels happen to need --
    3.54 in for Fig. 1 and 3.25 in for Fig. 2. Both are then stretched to
    \columnwidth by \includegraphics, by 0.975x and 1.060x, so 8 pt type
    printed at 7.80 pt in one figure and 8.48 pt in the next. An 8.7 % font
    mismatch between adjacent figures is exactly the tell this module was
    written to avoid, and it was doing it to itself.

    Fonts are in points and do not scale with the canvas, so growing the canvas
    does not grow the margins proportionally: the fixed point is found by
    iterating rather than by one division. Converges in two or three passes.
    """
    for _ in range(tries):
        fig.canvas.draw()
        bb = fig.get_tightbbox(fig.canvas.get_renderer())
        err_pt = (target_in - bb.width) * 72.0
        if abs(err_pt) <= tol_pt:
            return bb.width
        w, h = fig.get_size_inches()
        # Grow the canvas by the shortfall, keeping the aspect of the axes area.
        fig.set_size_inches(w + (target_in - bb.width), h)
    return bb.width


def save(fig, path_noext, width=COL):
    """Write PDF (paper) and PNG (viewing) at EXACTLY `width` inches.

    Two steps, and both are needed. fit_to_width grows the canvas until the
    content nearly fills the target, so nothing is cropped. Then the save uses
    an explicit Bbox rather than "tight", because savefig recomputes its own
    tight box at write time and lands a few hundredths of an inch off -- which
    is the whole bug this is fixing. An explicit box is exact by construction.
    """
    from matplotlib.transforms import Bbox

    fit_to_width(fig, width)
    fig.canvas.draw()
    bb = fig.get_tightbbox(fig.canvas.get_renderer())
    cx = 0.5 * (bb.x0 + bb.x1)
    box = Bbox.from_extents(cx - width / 2.0, bb.y0, cx + width / 2.0, bb.y1)
    for ext in ("pdf", "png"):
        fig.savefig(f"{path_noext}.{ext}", bbox_inches=box, pad_inches=0.0)
    plt.close(fig)
    print(f"  wrote {path_noext}.pdf / .png  ({width:.3f} x {box.height:.3f} in)")
