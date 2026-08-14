## How it works

> **Status: flow bring-up.** `src/` currently holds a skeleton used to take the
> LibreLane flow end to end and to measure the real cell budget. The datapath
> described below is the target, not what is in the repository today. This
> notice comes out before submission.

`Proof` is a fixed-point dot-product engine that estimates how much a meal will
raise blood sugar. It runs two modes over one datapath, selected by `MODE`:

- **Mode A -- glycemic load.** A single weighted sum over recipe ingredients,
  `GL = sum(grams_i * w_i)`, where the host precomputes
  `w_i = available_carb_per_gram_i * GI_i / 100`, and available carbohydrate is
  total carbohydrate minus dietary fibre. Needs no per-user data.
- **Mode B -- personalised response.** A two-layer quantised MLP,
  `h = ReLU((W1.x + b1) >> s1)` then `y = (W2.h + b2) >> s2`, over meal
  macronutrients and user context. The shift amounts are constrained to powers
  of two, so requantisation costs a shift rather than a second multiplier.

Weights are **streamed in from the host rather than stored on chip**. That began
as an area decision -- the parameters will not fit alongside the datapath in one
tile -- but it is also what makes the design personalisable: same silicon,
different patient.

The accumulator **saturates rather than wraps**, and raises a sticky
`SATURATED` flag. A too-narrow accumulator here produces a wrong number that a
person might act on, so silent overflow is not an acceptable failure mode.

**This is not a medical device.** It is an educational and research artifact,
and every output is an estimate.

## How to test

The host drives a byte stream on `DATA_IN` with `VALID`, tagging each byte as a
weight or an activation via `IS_WEIGHT`, and marking the end of a stream with
`LAST`. `BUSY` is high while an inference is in flight, `DONE` marks the result
valid, and `RD_SEL` selects which byte of the result appears on `RESULT`.
Asserting `rst_n` always recovers the chip, including from a truncated or stuck
stream.

There is no host MCU: the protocol is exercised by the cocotb testbench in
`test/`, against both the RTL and the post-layout gate-level netlist.

## External hardware

None.
