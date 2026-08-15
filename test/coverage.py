# SPDX-FileCopyrightText: (c) 2026 Kush Shah
# SPDX-License-Identifier: Apache-2.0
"""
Functional coverage model with named bins.

WHY THIS EXISTS ALONGSIDE THE MUTATION SCORE
--------------------------------------------
They answer different questions and neither substitutes for the other.

  mutation score  "if the design were wrong, would the testbench notice?"
  coverage        "did the stimulus ever reach this situation at all?"

A mutant can only be caught in a situation the stimulus actually visits, so a
high mutation score with unexamined coverage may just mean the mutants happened
to live where the tests already were. Reporting bins by name makes the gaps
sayable: "ReLU never clamped high" is a sentence you can act on, in a way that
"93 % coverage" is not.

BINS ARE DERIVED FROM THE REFERENCE MODEL, NOT ASSERTED BY HAND
----------------------------------------------------------------
Most sampling here reads the golden model's own output rather than trusting a
test to declare what it exercised. A test that *believes* it saturated the
accumulator but does not cannot tick that bin. Only genuinely unobservable
events -- protocol abuse, reset timing -- are hit explicitly, and those are
listed separately so the distinction stays visible.
"""

import golden_quant as gold

GL_FIELD_HI = (1 << (gold.GL_W - 1)) - 1     # 8191
GL_FIELD_LO = -(1 << (gold.GL_W - 1))        # -8192
Y_FIELD_HI = 32767
Y_FIELD_LO = -32768

# group -> ordered bin names
MODEL = {
    "mode": ["A", "B"],
    "gl_category": ["low", "medium", "high"],
    "gl_sign": ["negative", "zero", "positive"],
    "shift": ["zero", "small", "large"],
    "weight_operand": ["min", "negative", "zero", "positive", "max"],
    "activation_operand": ["min", "negative", "zero", "positive", "max"],
    "accumulator": ["no_saturation", "saturate_high", "saturate_low"],
    "mode_a_field": ["in_range", "clamp_high", "clamp_low"],
    "mode_b_field": ["in_range", "clamp_high", "clamp_low"],
    "stream_length": ["single_term", "few", "many"],
    "relu": ["clamped_to_zero", "passthrough", "clamped_to_127"],
    "layer2_outputs": ["single", "multiple"],
    "protocol": [
        "activation_before_shift",
        "bytes_while_busy",
        "truncated_stream",
        "reset_mid_multiply",
        "sticky_across_neurons",
        "new_inference_flag",
        "sticky_coefficient",
        "busy_done_handoff",
    ],
    "monotonicity": ["synthetic_weights", "trained_weights",
                     "trained_weights_decreasing"],
}

# Bins that no reference-model output can reveal, so a test must say so.
EXPLICIT = {"protocol", "monotonicity"}


class Coverage:
    def __init__(self):
        self.counts = {g: {b: 0 for b in bins} for g, bins in MODEL.items()}

    def hit(self, group, name):
        if group not in self.counts:
            raise KeyError(f"unknown coverage group {group!r}")
        if name not in self.counts[group]:
            raise KeyError(f"unknown bin {name!r} in group {group!r}")
        self.counts[group][name] += 1

    # ----------------------------------------------------------- samplers --
    def _operands(self, pairs):
        for w, a in pairs:
            for grp, v in (("weight_operand", w), ("activation_operand", a)):
                if v == -128:
                    self.hit(grp, "min")
                elif v == 127:
                    self.hit(grp, "max")
                elif v == 0:
                    self.hit(grp, "zero")
                elif v < 0:
                    self.hit(grp, "negative")
                else:
                    self.hit(grp, "positive")

    def _shift(self, s):
        self.hit("shift", "zero" if s == 0 else ("small" if s < 8 else "large"))

    def _length(self, n):
        self.hit("stream_length",
                 "single_term" if n == 1 else ("few" if n <= 8 else "many"))

    def _accumulator(self, res):
        if not res["saturated"]:
            self.hit("accumulator", "no_saturation")
        elif res.get("acc", 0) > 0:
            self.hit("accumulator", "saturate_high")
        else:
            self.hit("accumulator", "saturate_low")

    def sample_mode_a(self, pairs, shift, res):
        self.hit("mode", "A")
        self._operands(pairs)
        self._shift(shift)
        self._length(len(pairs))
        self._accumulator(res)

        gl = res["gl"]
        self.hit("gl_category", ["low", "medium", "high"][res["cat"]])
        self.hit("gl_sign", "zero" if gl == 0 else ("positive" if gl > 0 else "negative"))
        if gl > GL_FIELD_HI:
            self.hit("mode_a_field", "clamp_high")
        elif gl < GL_FIELD_LO:
            self.hit("mode_a_field", "clamp_low")
        else:
            self.hit("mode_a_field", "in_range")

    def sample_mode_b(self, l1, l2, s1, s2, res):
        self.hit("mode", "B")
        self._shift(s1)
        self._shift(s2)
        for pairs in l1:
            self._operands(pairs)
            self._length(len(pairs))
        self._accumulator({"saturated": res["saturated"], "acc": 1})
        self.hit("layer2_outputs", "single" if len(l2) == 1 else "multiple")

        # ReLU: recompute the pre-clamp value, since `res["h"]` is post-clamp
        # and cannot distinguish a clamp from a value that happened to land there.
        for pairs in l1:
            acc, _ = gold.dot(pairs)
            v = gold.requantize(acc, s1)
            if v < 0:
                self.hit("relu", "clamped_to_zero")
            elif v > gold.H_MAX:
                self.hit("relu", "clamped_to_127")
            else:
                self.hit("relu", "passthrough")

        for y in res["y"]:
            if y > Y_FIELD_HI:
                self.hit("mode_b_field", "clamp_high")
            elif y < Y_FIELD_LO:
                self.hit("mode_b_field", "clamp_low")
            else:
                self.hit("mode_b_field", "in_range")

    # ------------------------------------------------------------- report --
    def summary(self):
        total = sum(len(b) for b in MODEL.values())
        hit = sum(1 for g in self.counts for b in self.counts[g] if self.counts[g][b])
        misses = [
            f"{g}.{b}"
            for g in MODEL
            for b in MODEL[g]
            if self.counts[g][b] == 0
        ]
        return hit, total, misses

    def report(self, log=print):
        hit, total, misses = self.summary()
        log("=" * 62)
        log(f" FUNCTIONAL COVERAGE -- {hit}/{total} bins hit "
            f"({100.0 * hit / total:.1f} %)")
        log("=" * 62)
        for g in MODEL:
            marks = []
            for b in MODEL[g]:
                n = self.counts[g][b]
                marks.append(f"{b}={n}" if n else f"{b}=MISS")
            tag = " (explicit)" if g in EXPLICIT else ""
            log(f"  {g:20s}{tag}")
            log(f"      {'  '.join(marks)}")
        if misses:
            log(f"\n  UNCOVERED: {', '.join(misses)}")
        log("=" * 62)
        return hit, total, misses


cov = Coverage()
