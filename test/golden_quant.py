# SPDX-FileCopyrightText: (c) 2026 Kush Shah
# SPDX-License-Identifier: Apache-2.0
"""
Integer reference model for Proof. The RTL must match this **bit-exactly**.

This is the first of the two golden models required by the verification plan.
The second (test/golden_float.py, not written yet) is a float model that the
quantised design is allowed to differ from within a defended error bound. The
two obligations are different and must never be conflated: a mismatch against
*this* model is a bug, full stop.

Everything here is plain Python integer arithmetic, chosen so the semantics
line up with the RTL exactly:

  - Python's `>>` on a negative int floors toward negative infinity, which is
    precisely what Verilog's `>>>` does on a signed value. So `acc >> shift`
    is bit-exact against the RTL requantiser with no correction needed.
  - Saturation is applied per accumulate, not once at the end, because that is
    what the hardware does -- it has one accumulator and no lookahead.
"""

# --- datapath parameters, must match src/proof_core.v ---------------------
ACC_W = 24
GL_W = 14
SHIFT_W = 5

ACC_MAX = (1 << (ACC_W - 1)) - 1
ACC_MIN = -(1 << (ACC_W - 1))

# Glycemic load categories. Standard per-serving thresholds:
# low <= 10, medium 11-19, high >= 20.
CAT_LOW = 0
CAT_MED = 1
CAT_HIGH = 2


def sat_add(acc, term):
    """Signed add at ACC_W bits, clamping instead of wrapping.

    Returns (result, overflowed). Overflow is reported per-add so the caller
    can make the flag sticky, matching the RTL.
    """
    s = acc + term
    if s > ACC_MAX:
        return ACC_MAX, True
    if s < ACC_MIN:
        return ACC_MIN, True
    return s, False


def requantize(acc, shift):
    """Arithmetic right shift, floor semantics. Matches Verilog `>>>`."""
    return acc >> shift


def categorize(gl):
    """Glycemic load figure -> category."""
    if gl >= 20:
        return CAT_HIGH
    if gl >= 11:
        return CAT_MED
    return CAT_LOW


def mode_a(pairs, shift):
    """Mode A: glycemic load over a list of (weight, activation) INT8 pairs.

    `pairs` are (w_q, g_q), each a signed 8-bit value. The host precomputes
    w_i = available_carb_per_gram_i * GI_i / 100 and quantises it; it also
    chooses the activation scale, so grams are whatever unit makes them fit
    INT8. `shift` undoes the combined weight/activation scaling.
    """
    acc = 0
    saturated = False
    for w, g in pairs:
        acc, ovf = sat_add(acc, w * g)
        saturated = saturated or ovf
    gl = requantize(acc, shift)
    return {
        "acc": acc,
        "gl": gl,
        "cat": categorize(gl),
        "saturated": saturated,
    }


def result_bytes(res):
    """The two bytes the host reads back via RD_SEL.

    The design has 8 output pins and a 1-bit read select, so 16 bits total.
    Rather than spend a whole byte on a 2-bit category, the category is packed
    into the top of the high byte and the glycemic load figure is carried in
    the remaining GL_W = 14 bits. A GL of 16,383 is orders of magnitude beyond
    any real meal -- "high" starts at 20 -- so nothing is lost.

      rd_sel = 0 -> gl[7:0]
      rd_sel = 1 -> {cat[1:0], gl[13:8]}
    """
    gl_masked = res["gl"] & ((1 << GL_W) - 1)
    lo = gl_masked & 0xFF
    hi = ((res["cat"] & 0x3) << 6) | ((gl_masked >> 8) & 0x3F)
    return lo, hi
