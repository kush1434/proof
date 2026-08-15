# SPDX-FileCopyrightText: (c) 2026 Kush Shah
# SPDX-License-Identifier: Apache-2.0
"""
Float reference for Proof, and the quantisation bridge between it and the chip.

TWO MODELS, TWO DIFFERENT OBLIGATIONS -- DO NOT CONFLATE THEM
------------------------------------------------------------
`golden_quant.py` is the *integer* reference. The RTL must match it
**bit-exactly**; a mismatch there is a bug, full stop.

This file is the *float* reference. The quantised design is **not** expected to
match it, and never can. What matters is that the difference stays inside a
bound that can be defended for this application. That is a characterisation
result, not a pass/fail test, and this module deliberately reports numbers
rather than asserting them.

**Choosing the bound is not a decision this file makes.** It is a judgement
with a safety argument attached and it belongs to the project owner. What is
provided here is the machinery to measure the error honestly, plus the
translation that turns a trained float network into the exact byte streams the
chip consumes -- so the same weights can be pushed through the RTL and checked
end to end.

SCALES
------
Everything is power-of-two so requantisation stays a shift (no second
multiplier on chip). For a two-layer network:

    x_q  = clamp(round(x  * 2**kx ), -128, 127)
    W1_q = clamp(round(W1 * 2**kw1), -128, 127)
    acc1 = sum(W1_q * x_q)                      scale 2**(kw1+kx)
    h_q  = clamp(acc1 >> s1, 0, 127)            scale 2**(kw1+kx-s1)
    W2_q = clamp(round(W2 * 2**kw2), -128, 127)
    acc2 = sum(W2_q * h_q)                      scale 2**(kw2+kw1+kx-s1)
    y_q  = acc2 >> s2                           scale 2**(kw2+kw1+kx-s1-s2)

so the dequantised prediction is `y_q / 2**(kw2+kw1+kx-s1-s2)`.

Biases carry no scale of their own: they are folded into the accumulator at
whatever scale that layer is already using, which is why the chip needs no
bias-specific logic at all.
"""

import golden_quant as gold

INT8_MIN, INT8_MAX = -128, 127


# ------------------------------------------------------------ float model --


def relu(v):
    return v if v > 0.0 else 0.0


def mlp_float(W1, b1, W2, b2, x):
    """Pure float two-layer MLP. No quantisation anywhere."""
    h = [relu(sum(w * xi for w, xi in zip(row, x)) + b) for row, b in zip(W1, b1)]
    y = [sum(w * hj for w, hj in zip(row, h)) + b for row, b in zip(W2, b2)]
    return h, y


# ------------------------------------------------------------ quantisation --


def q8(v, k):
    """Quantise one float to INT8 at scale 2**k, saturating."""
    q = int(round(v * (2 ** k)))
    return max(INT8_MIN, min(INT8_MAX, q))


def quantise(vs, k):
    return [q8(v, k) for v in vs]


def choose_shift(values, headroom=0):
    """Largest k with every |value * 2**k| <= 127, i.e. the finest safe scale.

    Power-of-two by construction, so the chip's shift-only requantiser is
    exact rather than approximate. `headroom` gives back precision in exchange
    for saturation margin.
    """
    peak = max((abs(v) for v in values), default=0.0)
    if peak == 0.0:
        return 0
    k = 0
    while abs(peak) * (2 ** (k + 1)) <= 127:
        k += 1
    return max(0, k - headroom)


def split_bias(b_scaled):
    """Express an integer bias as the chip's two (weight, activation) pairs.

    Layer 2 supplies the constants 127 then 1, so a bias is v8*127 + v9*1.
    For any |b| <= 127*127 + 127 this is exact to the unit.
    """
    v8 = max(INT8_MIN, min(INT8_MAX, int(round(b_scaled / 127))))
    v9 = b_scaled - 127 * v8
    if not INT8_MIN <= v9 <= INT8_MAX:
        v8 = max(INT8_MIN, min(INT8_MAX, v8 + (1 if v9 > 0 else -1)))
        v9 = b_scaled - 127 * v8
        v9 = max(INT8_MIN, min(INT8_MAX, v9))
    return v8, v9


# --------------------------------------------------- float -> chip streams --


def to_chip_streams(W1, b1, W2, b2, x, kx, kw1, kw2, s1, s2):
    """Turn a trained float network into exactly what the chip consumes.

    Returns (l1, l2, meta) where l1/l2 are the stream descriptions accepted by
    golden_quant.mode_b and by the cocotb helpers, so the same weights can be
    driven through the reference model and the RTL without transcription.
    """
    x_q = quantise(x, kx)
    acc1_k = kw1 + kx

    l1 = []
    for row, b in zip(W1, b1):
        pairs = list(zip(quantise(row, kw1), x_q))
        v8, v9 = split_bias(int(round(b * (2 ** acc1_k))))
        pairs += [(v8, 127), (v9, 1)]  # bias, at the layer's own scale
        l1.append(pairs)

    h_k = acc1_k - s1
    acc2_k = kw2 + h_k

    l2 = []
    for row, b in zip(W2, b2):
        ws = quantise(row, kw2)
        v8, v9 = split_bias(int(round(b * (2 ** acc2_k))))
        l2.append(list(ws) + [v8, v9])

    meta = {
        "x_scale": 2 ** kx,
        "h_scale": 2 ** h_k,
        "y_scale": 2 ** (acc2_k - s2),
        "s1": s1,
        "s2": s2,
    }
    return l1, l2, meta


# Scales selected by search over a held-out fold rather than picked by hand.
# The default that shipped first (5, 6, 6, 6, 0) cost 172 mg/dL.min mean error
# with a p95 of 1177; this costs 85 with a p95 of 180 -- a 6.5x improvement in
# the tail, which is where a wrong answer would actually matter to someone.
# Reproduce with model/clinical.py.
BEST_SCALES = {"kx": 5, "kw1": 5, "kw2": 5, "s1": 5, "s2": 1}


def dequantise_y(y_q, meta):
    return [v / meta["y_scale"] for v in y_q]


# ------------------------------------------------------ error measurement --


def compare(W1, b1, W2, b2, x, kx=None, kw1=None, kw2=None, s1=None, s2=None):
    """Run both models on one input and report the gap. Reports, never asserts.

    Scales default to the finest that avoids saturating any weight; shifts
    default to keeping the hidden layer inside its 0..127 range.
    """
    flat1 = [w for row in W1 for w in row]
    flat2 = [w for row in W2 for w in row]
    kx = choose_shift(x) if kx is None else kx
    kw1 = choose_shift(flat1) if kw1 is None else kw1
    kw2 = choose_shift(flat2) if kw2 is None else kw2

    h_f, y_f = mlp_float(W1, b1, W2, b2, x)

    if s1 is None:
        peak_h = max(h_f) if h_f else 0.0
        s1 = 0
        while peak_h * (2 ** (kw1 + kx - s1)) > 127 and s1 < 24:
            s1 += 1
    if s2 is None:
        s2 = 0

    l1, l2, meta = to_chip_streams(W1, b1, W2, b2, x, kx, kw1, kw2, s1, s2)
    res = gold.mode_b(l1, l2, s1, s2)
    y_hat = dequantise_y(res["y"], meta)

    abs_err = [abs(a - b) for a, b in zip(y_hat, y_f)]
    span = max(abs(v) for v in y_f) if any(y_f) else 1.0
    return {
        "y_float": y_f,
        "y_dequant": y_hat,
        "abs_err": abs_err,
        "rel_err": [e / span for e in abs_err],
        "h_float": h_f,
        "h_quant": res["h"],
        "saturated": res["saturated"],
        "meta": meta,
        "streams": (l1, l2),
    }


def characterise(cases):
    """Aggregate `compare` over many cases into a distribution, not a verdict.

    `cases` is an iterable of (W1, b1, W2, b2, x) tuples.
    """
    errs, rels, sats, n = [], [], 0, 0
    for W1, b1, W2, b2, x in cases:
        r = compare(W1, b1, W2, b2, x)
        errs.extend(r["abs_err"])
        rels.extend(r["rel_err"])
        sats += 1 if r["saturated"] else 0
        n += 1
    errs.sort()
    rels.sort()

    def pct(xs, p):
        return xs[min(len(xs) - 1, int(p * len(xs)))] if xs else 0.0

    return {
        "cases": n,
        "outputs": len(errs),
        "saturating_cases": sats,
        "abs_err_p50": pct(errs, 0.50),
        "abs_err_p95": pct(errs, 0.95),
        "abs_err_max": errs[-1] if errs else 0.0,
        "rel_err_p50": pct(rels, 0.50),
        "rel_err_p95": pct(rels, 0.95),
        "rel_err_max": rels[-1] if rels else 0.0,
    }
