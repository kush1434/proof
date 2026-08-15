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


# --- Mode B ---------------------------------------------------------------
# Two-layer quantised MLP over the same datapath:
#
#   h = ReLU( (W1 . x + b1) >> s1 )      clamped to 0..127
#   y =       (W2 . h + b2) >> s2
#
# Every neuron of both layers is a Mode A dot product. The difference is only
# where the activations come from and what happens to the result:
#
#   layer 1  activations supplied by the host, re-streamed per neuron
#            result requantised by s1, ReLU-clamped, pushed into h
#   layer 2  activations supplied by the chip from h
#            result requantised by s2 and read out
#
# `x` is deliberately NOT buffered on chip. The host re-streams it for each
# hidden neuron, which costs 48 bytes instead of 6 and saves 48 flip-flops.
# Bandwidth is free at meal timescales; flip-flops are 48.99 um2 each.
#
# Biases are not a special case anywhere. In layer 1 the host just sends extra
# (weight, activation) pairs. In layer 2 the chip supplies two constants after
# the eight h values -- 127 then 1 -- so a bias is w8*127 + w9*1, exact to the
# unit for any 15-bit value. That is zero extra RTL for full-precision biases.

N_HIDDEN = 8
H_MIN = 0
H_MAX = 127

L2_CONST = (127, 1)  # activations the chip supplies at k = N_HIDDEN, N_HIDDEN+1


def relu_clamp(v):
    """ReLU and clamp to the INT8 range h is stored in."""
    if v < H_MIN:
        return H_MIN
    if v > H_MAX:
        return H_MAX
    return v


def layer2_activation(h, k):
    """Activation the chip supplies for index k of a layer-2 neuron."""
    if k < N_HIDDEN:
        return h[k]
    return L2_CONST[k - N_HIDDEN]


def dot(pairs):
    """Saturating dot product. Returns (acc, saturated)."""
    acc = 0
    saturated = False
    for w, a in pairs:
        acc, ovf = sat_add(acc, w * a)
        saturated = saturated or ovf
    return acc, saturated


def mode_b(l1_neurons, l2_neurons, s1, s2):
    """Two-layer inference.

    l1_neurons: N_HIDDEN lists of (w, x) pairs -- activations from the host
    l2_neurons: lists of weights only          -- activations from the chip

    The sticky overflow flag spans the whole inference, not one neuron.
    """
    saturated = False

    h = []
    for pairs in l1_neurons:
        acc, ovf = dot(pairs)
        saturated = saturated or ovf
        h.append(relu_clamp(requantize(acc, s1)))

    y = []
    for weights in l2_neurons:
        acc, ovf = dot([(w, layer2_activation(h, k)) for k, w in enumerate(weights)])
        saturated = saturated or ovf
        y.append(requantize(acc, s2))

    return {"h": h, "y": y, "saturated": saturated}


def wide_bytes(v):
    """Mode B readback: rd_sel = 0 -> low byte, rd_sel = 1 -> high byte."""
    m = v & 0xFFFF
    return m & 0xFF, (m >> 8) & 0xFF


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
