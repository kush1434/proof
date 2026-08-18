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


def save(fig, path_noext):
    """Write PDF (for the paper) and PNG (for looking at)."""
    for ext in ("pdf", "png"):
        fig.savefig(f"{path_noext}.{ext}")
    plt.close(fig)
    print(f"  wrote {path_noext}.pdf / .png")
