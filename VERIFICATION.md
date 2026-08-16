# Verification — Proof

What is verified, how, and — just as importantly — what is not.

Every claim below is traceable to something runnable. Anything that is not
verified is listed as not verified, rather than left unmentioned.

```bash
cd test
python run.py                    # whole design, RTL        (41 tests)
python run.py --unit mac_serial  # multiplier, exhaustive   (6 tests)
python run.py --gates            # post-layout netlist      (needs PDK_ROOT)
python monotonicity.py           # the safety property study
cd .. && ./mutate.sh             # mutation testing         (29 mutants)
cd ../model && python train.py   # train + sign-condition check
cd .. && ./lint.sh               # local pre-push synthesis gate
```

---

## 1. The headline result: monotonicity in carbohydrate

**Properties.** Holding every other input fixed:

- increasing **carbohydrate** must never *decrease* the predicted response;
- increasing **fibre** must never *increase* it.

The argument below is written for carbohydrate. It is direction-agnostic: for a
decreasing input the required sign simply flips, and `monotonicity.py` §[4]
demonstrates both halves — 0 violations when the condition holds, 7 when it is
deliberately broken.

This is the one claim about the chip worth proving rather than sampling, and it
requires **zero patient data** — it is a question about the quantised
arithmetic, not about any particular trained network.

### 1.1 The argument

For one output, the pipeline is

```
acc1_j(x_c) = saturating_sum_i( W1[j][i] * x_i )
h_j         = clamp_0_127( acc1_j >> s1 )
acc2        = saturating_sum_j( W2[j] * h_j )
y           = acc2 >> s2
```

Every stage is monotone in its input:

- Multiplying by a constant is non-decreasing if the constant is ≥ 0 and
  non-increasing if ≤ 0.
- **A saturating sum is monotone.** With partial sums `S_i = clamp(S_{i-1} +
  t_i)`, `clamp` is non-decreasing and `S_i` depends monotonically on
  `S_{i-1}`, so by induction the final sum is non-decreasing in any single
  term. *Wrapping would destroy this* — which is a second, independent reason
  the accumulator must saturate, beyond producing a sensible number.
- Arithmetic right shift is floor division by a power of two: non-decreasing.
  Every standard rounding mode is monotone, so the choice of floor is **not**
  what puts the property at risk.
- ReLU and clamping to `[0, 127]` are non-decreasing.

Composition therefore gives: `y` is non-decreasing in `x_c` **provided that for
every hidden unit j**

```
W1[j][i] * W2[j] * want[i] >= 0        (the SIGN CONDITION)
```

where `want[i]` is +1 for an input that must raise the response and −1 for one
that must lower it. Each hidden unit either pushes the input's effect the
required way, or opposes it twice and so still does.

### 1.2 What the study found

`test/monotonicity.py`, 400 weight sets constructed to satisfy the sign
condition, sweeping `x_c` across the full INT8 range:

| | violations |
|---|---|
| Internal value | **0 / 400** |
| Reported value (before the fix) | **55 / 400** |
| Reported value (after the fix) | **0 / 400** |

The internal value behaved exactly as the argument predicts. **The reported
value did not** — because the host never sees the internal value. It reads a
fixed-width field, and those fields *truncated*. Truncation wraps, and wrapping
is not monotone.

A representative failure:

```
s1=1 s2=0    x_c 70 -> 71
internal y   31293 -> 34327     (rises, correctly)
reported y   31293 -> -31209    (falls by 62,502)
```

The host would see the predicted response collapse at the moment carbohydrate
increased. Recorded as **BUGS.md R-4** and fixed by saturating the output
fields, which is monotone. The study now reports zero violations.

### 1.3 What this does and does not establish

- It **does** establish that the arithmetic preserves monotonicity, and that
  the property now survives all the way to what the host reads.
- The sign condition is **sufficient, not necessary**. A network violating it
  may still be monotone over the reachable input region. Nothing here searches
  for that weaker guarantee.
- The sweep in `test_monotonic_in_carbohydrate` is over one deliberately chosen
  weight set that straddles the field limit, plus 400 randomised sets in the
  model study. It is not exhaustive over weight space, which is far too large.

### 1.4 Does a *real* trained network satisfy the sign condition?

No — not unless it is trained to. Fitting the network to CGMacros without the
constraint produces hidden units whose two signs disagree (4 of 8 on all meals,
2 of 8 on breakfasts), so **monotonicity is false by construction for an
unconstrained fit** and no RTL stimulus could rescue it.

The fix is a constrained objective, and the interesting question is what it
costs. `model/train.py` imposes the condition by *reparameterisation* rather
than by a penalty, so it holds exactly at every step:

```
W1[j][carb] = d[j] * softplus(...)      d[j] in {+1, -1}, fixed per unit
W2[0][j]    = d[j] * softplus(...)
```

Either direction satisfies the condition, because the product is
`d[j]**2 * softplus * softplus >= 0`. Forcing every `d[j] = +1` also works but
is strictly stronger than necessary and badly over-penalises a network whose
second layer is mostly negative — an early version did exactly that and
reported a cost three to four times too large. The directions are therefore
seeded from the unconstrained solution's own sign structure.

Evaluated with **grouped 5-fold cross-validation over participants**. A random
row split would leak individual response, which is the dominant signal here, and
a single split cannot tell a real effect from a lucky partition — which matters,
because it did not:

| model | mean R² | sd | per fold |
|---|---|---|---|
| depth-1 XGBoost, the notebook's model | +0.168 | 0.054 | +0.21 +0.14 +0.12 +0.12 +0.25 |
| unconstrained MLP 6-8-1 | +0.216 | 0.085 | +0.23 +0.25 +0.08 +0.17 +0.34 |
| monotone: carbs ↑ | +0.180 | 0.108 | +0.18 +0.22 +0.02 +0.13 +0.35 |
| monotone: carbs ↑ *and* fibre ↓ | +0.189 | 0.143 | +0.15 +0.31 −0.03 +0.15 +0.38 |

Differences are **paired across folds** — same partitions, so the difference has
lower variance than either column and is the honest quantity to report:

| comparison | delta | 95 % CI | reading |
|---|---|---|---|
| carbs ↑ vs unconstrained | −0.036 | [−0.059, −0.012] | **a real, small cost** |
| carbs ↑ + fibre ↓ vs unconstrained | −0.027 | [−0.092, +0.037] | not established |
| XGBoost vs unconstrained | −0.048 | [−0.098, +0.003] | not established |

**Monotonicity is not free.** An earlier single 11-participant split reported the
carbohydrate constraint at −0.000 and this document said so; five folds show
−0.036 with a confidence interval that excludes zero. The single split was
optimistic, and the corrected claim is narrower but defensible: **the safety
property costs a few hundredths of R², which is small relative to a
between-fold spread of 0.085.**

Two further readings that the single split would have got wrong:

- **The network does not beat the published-style baseline.** The XGBoost
  comparison straddles zero. Any claim of "beats XGBoost" is unsupported.
- **The wider interval on carbs + fibre is variance, not merit.** Its per-fold
  spread (sd 0.143) is the largest in the table; adding a constraint did not
  make the model better, it made the estimate noisier. Reading "no measurable
  difference" as "the second guarantee is free" would repeat exactly the
  mistake the single split caused.

`R² ≈ 0.22` also means most of the variance in postprandial response is *not*
explained by meal macros plus pre-meal glucose. That is consistent with the
literature and is a statement about the problem, not a defect in the chip.

Fibre's direction is empirically supported but weakly: marginal r −0.051,
partial β −0.025, and within every carbohydrate tertile the higher-fibre half
has the lower iAUC (−111, −1951, −683 mg/dL·min). It is a domain-knowledge
constraint the data is consistent with, not one the data establishes.

Reproduce with `python train.py --cv 5`.

### 1.5 The baseline's scope is not this project's scope

CGMacros breakfasts are **standardised test meals**. Of 383 usable breakfasts
there are only **6 distinct macronutrient combinations**, the top 4 cover 83 %,
carbohydrate takes 3 values and fibre takes 2. Holding the meal fixed is
deliberate study design: it isolates person-to-person variation, which makes
the reference notebook a *personalisation* result rather than a
meal-composition one.

Proof predicts response from meal composition, so it needs meals that differ.
Across all meal types there are **1,346 usable meals with 592 distinct
combinations**, and carbohydrate's coefficient of variation rises from 0.29 to
0.74.

This matters for the safety property too: on breakfast-only data, carbohydrate
takes three values, so a monotonicity sweep there would be nearly vacuous.

### 1.6 End to end on the RTL

`test_trained_network_on_silicon` takes the trained weights, quantises them
through `golden_float.to_chip_streams`, drives the resulting bytes into the
DUT, and sweeps **every constrained input in its own direction**, checking at
each point that the hardware is bit-exact against the integer reference *and*
that the response moves the required way:

| input | levels | direction | observed output range |
|---|---|---|---|
| carbohydrate | 12 | non-decreasing | 1377 … 4346 (spread 2969) |
| fibre | 16 | non-increasing | 2526 … 3210 (spread 684) |

The spreads matter. **A direction that holds because nothing moved proves very
little**, so the test reports the observed range and warns explicitly if a sweep
comes out flat. Neither of these does.

It **skips in CI**, because the weights are derived from a CC BY-NC-SA dataset
and are deliberately not committed. A green CI run does not exercise it.

---

## 2. Bit-exactness against the integer reference

`test/golden_quant.py` is the integer reference. **The RTL must match it
exactly**, and every functional test compares against it rather than against
hand-computed expected values.

- 41 top-level tests, both modes, all bit-exact.
- Semantics are chosen so the two cannot drift: Python's `>>` on a negative int
  floors, which is precisely what Verilog's `>>>` does on a signed value, so
  requantisation is bit-exact with no correction on either side. Saturation is
  applied per accumulate in both, because that is what the hardware does.

`test/golden_float.py` is the float reference, carrying the *error-bound*
obligation. It also holds `to_chip_streams`, which turns a trained float
network into the exact byte streams the chip consumes — so the same weights go
through the reference model and the RTL without transcription.

Over 300 random networks the dequantised output tracks the float model with
median relative error **0.6 %** and p95 **2.8 %**. The maximum is much larger,
but that is an artefact of random networks whose outputs sit near zero rather
than an accuracy cliff; a meaningful bound has to be measured on the trained
network, and **choosing the bound remains an open decision**, not one this
repository makes.

The two obligations are deliberately not conflated: a mismatch against the
integer model is a bug; a deviation from the float model is a characterisation
result with a bound that has to be argued.

---

## 3. Exhaustive verification of the multiplier

`mac_serial` is checked over **all 65,536 signed 8×8 input pairs**, not a
sample. For that module there is no coverage argument left to make.

This is also what makes the M2 equivalence claim a proof rather than a
hypothesis (§4).

---

## 4. Mutation testing

A green testbench proves nothing until it can go red. `./mutate.sh` injects a
known bug, runs the relevant suites, and confirms the testbench notices.

**29 considered, 28 caught, 1 proven equivalent, 0 escaped, 0 invalid.**

Three properties of the harness itself, each of which had to hold before any
`CAUGHT` means anything:

1. **The did-not-apply guard uses `diff -q --strip-trailing-cr`, not `cmp -s`.**
   The repository checks out CRLF and `sed` writes LF, so a byte comparison
   reports every file as changed — including one where the pattern matched
   nothing — and a mutant that was never injected scores as caught. Re-verified
   in both directions on every run; the script aborts if broken. It has already
   fired for real: when the last-step comparison was resized, M3 stopped
   matching and was reported **INVALID** rather than silently counted.
2. **A runner self-test.** A known-broken accumulator is pushed through
   `run.py` and the script confirms it reports failure (BUGS.md TB-3).
3. **Restore is checked against a pre-run checksum snapshot**, not git HEAD,
   so it does not cry wolf on every uncommitted file.

Three mutants that escaped and what closing them taught:

| Mutant | Why it escaped | Fix |
|---|---|---|
| M5 sticky flag not sticky | Once the accumulator pins at the rail, every later term in the same direction also overflows, so a non-sticky flag still reads 1 | Saturate, then accumulate terms that land *inside* the rail |
| M12 `busy` drops early | Every test waited on `done`, never on `busy`, so nothing observed the handoff between them | Assert `done` is already high on the first cycle `busy` is low |
| M2 narrower accumulator | Genuinely equivalent — proven by the exhaustive test, not a gap | Reported as `EQUIVALENT`, not `CAUGHT` or `ESCAPED` |

---

## 5. Functional coverage

`test/coverage.py` defines **54 named bins across 15 groups**, and the suite
hits **54/54**. `test_zzz_coverage_report` *asserts* full coverage rather than
merely printing it: adding a bin without stimulus to reach it breaks the build,
which forces the gap to be either covered or removed with a stated reason.

Coverage and the mutation score answer different questions and neither
substitutes for the other:

| | question |
|---|---|
| mutation score | if the design were wrong, would the testbench notice? |
| coverage | did the stimulus ever reach this situation at all? |

A mutant can only be caught in a situation the stimulus visits, so a high
mutation score with unexamined coverage may only mean the mutants happened to
live where the tests already were.

**Bins are derived from the reference model, not asserted by hand.** A test
that *believes* it saturated the accumulator but did not cannot tick that bin.
Only genuinely unobservable events — protocol abuse, reset timing — are hit
explicitly, and those two groups are labelled `(explicit)` in the report so the
distinction stays visible.

It earned its place immediately. On the first run it reported 46/48 with
`mode_b_field.clamp_high` and `mode_b_field.clamp_low` missing — the output
fields whose truncation was R-4. The high rail was reached only incidentally
inside the monotonicity sweep, which reads the DUT directly and so never
sampled, and the low rail was not reached by anything. Both now have a directed
test.

Also exempted honestly: when the CGMacros-derived weights are absent,
`monotonicity.trained_weights` is excluded rather than silently lowering the
bar, and the report says so.

---

## 6. Physical signoff

From the LibreLane run, both modes:

| | |
|---|---|
| Lint errors / warnings / inferred latches | 0 / 0 / 0 |
| DRC (routing, converged 302→0 over 4 iters) / magic DRC / illegal overlap | 0 / 0 / 0 |
| LVS errors / antenna / max-slew / max-cap / unmapped | 0 / 0 / 0 / 0 / 0 |
| Setup worst slack @ 20 ns | +10.08 ns |
| Hold worst slack | +0.120 ns |
| Flip-flops / stdcells | 168 / 1,443 |
| Utilisation | 83.53 % of a 1×1 tile |

Gate-level simulation passes.

**What gate-level sim does not prove.** The template compiles with
`-DFUNCTIONAL -DSIM` and **no SDF back-annotation**. It is a *functional*
gate-level simulation: it catches synthesis optimisation differences,
X-propagation and missing resets. It says **nothing about setup or hold**.
Timing closure is claimed only on the strength of the static timing analysis
figures above, not on simulation.

---

## 7. Areas deliberately not covered

The full list lives in `BUGS.md`. The ones that matter most:

- **The trained network is not committed.** CGMacros is CC BY-NC-SA, so
  derived weights are gitignored rather than shipped in an Apache-2.0 repo.
  `test_trained_network_on_silicon` therefore **skips in CI** and only runs
  locally after `model/train.py`. A green CI run does not exercise it.
- **Only `DW = 8` and `N_HIDDEN = 8` are simulated.** Both are parameters.
- **Host-side quantisation is trusted.** Nothing on chip validates that a shift
  is sensible; a shift ≥ 24 simply replicates the sign bit — well-defined,
  untested.
- **No timing verification beyond STA.** See §6.

---

## 8. Not a medical device

Every output is an estimate. Nothing here is a diagnostic, treatment or dosing
claim, and no part of this design should be used to make one.
