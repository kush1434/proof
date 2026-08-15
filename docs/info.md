## How it works

`Proof` is a fixed-point dot-product engine that estimates how much a meal will
raise blood sugar. It runs two modes over **one** datapath, selected by `MODE`.

- **Mode A — glycemic load.** A single weighted sum over recipe ingredients,
  `GL = sum(grams_i * w_i)`, where the host precomputes
  `w_i = available_carb_per_gram_i * GI_i / 100` and available carbohydrate is
  total carbohydrate minus dietary fibre. Needs no per-user data, so it works
  the moment the chip powers on. The chip also reports the standard per-serving
  category: low ≤ 10, medium 11–19, high ≥ 20.
- **Mode B — personalised response.** A two-layer quantised MLP,
  `h = ReLU((W1·x + b1) >> s1)` then `y = (W2·h + b2) >> s2`, over meal
  macronutrients and user context. Shift amounts are powers of two, so
  requantisation costs a shift rather than a second multiplier.

Every neuron of both layers is the same dot product. Only two things differ:
where the activations come from, and what happens to the result. That is why
this is one machine with a mode bit rather than two designs.

**Weights are streamed in, not stored on chip.** That began as an area decision
— the parameters do not fit alongside the datapath in one tile — but it is also
what makes the design personalisable: same silicon, different patient.

`x` is not buffered on chip either; the host re-streams it for each hidden
neuron, which costs 48 bytes instead of 6 and saves 48 flip-flops. At meal
timescales bandwidth is free and flip-flops are not. The hidden layer `h` *is*
buffered, as a rotating shift register, because only the chip can produce it.

Biases need no bias-specific logic anywhere. In layer 1 they are ordinary extra
terms. In layer 2 the chip supplies the constants 127 and 1 after the eight
hidden values, so a bias is `v8·127 + v9·1` — exact to the unit for any 15-bit
value.

### Everything saturates

The accumulator clamps instead of wrapping and raises a sticky `SATURATED`
flag, and so do the output fields. This is not only about producing a sensible
number: **a saturating sum is monotone and a wrapping one is not.**

That matters because the design's headline property is

> holding every other input fixed, increasing carbohydrate must never decrease
> the predicted response.

Saturating sums, arithmetic shifts, ReLU and clamps are each monotone, so the
composition is too, provided every hidden unit satisfies
`W1[j][carb] · W2[j] ≥ 0`. During development the property held internally but
failed at the pins, because the output fields truncated — a one-count rise in
carbohydrate could make the reported response fall from 31,293 to −31,209.
Truncation wraps. The fields now saturate, and the property survives all the way
to what the host reads.

**This is not a medical device.** It is an educational and research artifact,
and every output is an estimate.

## How to test

The host drives a byte stream on `DATA_IN`, qualified by `VALID`, with each
byte tagged by `IS_WEIGHT`. Do not present a byte while `BUSY` is high; bytes
offered then are ignored, not queued.

A neuron is a shift byte, then its terms, with `LAST` on the final one:

```
Mode A / Mode B layer 1   [s|wt] [w0|wt] [a0] [w1|wt] [a1] ... [aN|last]
Mode B layer 2            [s|wt] [v0|wt] [v1|wt] ...      [v9|wt,last]
```

The shift byte carries the requantisation shift, so the host sends `s1` for
hidden neurons and `s2` for output neurons. In layer 2 there are no activation
bytes at all — each weight multiplies the next value the chip supplies: `h[0..7]`,
then 127, then 1.

`LAST` asserted on the *shift* byte — otherwise a don't-care — means "this is a
new inference", which resets the neuron counter and clears the sticky overflow
flag. Mode A ignores it, since every Mode A stream is its own inference.

`DONE` marks a neuron complete and its result readable. `RD_SEL` selects which
byte appears on `RESULT`:

| | `RD_SEL` = 0 | `RD_SEL` = 1 |
|---|---|---|
| Mode A | `GL[7:0]` | `{category[1:0], GL[13:8]}` |
| Mode B | `value[7:0]` | `value[15:8]` |

Eight output pins and a one-bit select give 16 bits, so rather than spend a
whole byte on a 2-bit category it is packed above a 14-bit figure. "High"
starts at 20, so a 14-bit field is orders of magnitude more than any real meal
needs.

`rst_n` always recovers the chip, including from a truncated, stuck or
mis-tagged stream.

The number of *inputs* is not fixed in silicon — a neuron ends when the host
says `LAST` — and neither is the number of outputs, since the host simply stops
streaming. Only the hidden-layer width is structural, being the depth of the
`h` shift register.

There is no host MCU. The protocol is exercised by the cocotb testbench in
`test/`, against both the RTL and the post-layout gate-level netlist, and every
functional test is compared bit-exactly against an integer reference model.
`VERIFICATION.md` records what is verified and what is not; `BUGS.md` records
every defect found, including the ones found in the testbench itself.

## External hardware

None.
