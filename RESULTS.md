# Results — every number in one place

Assembled for writing. Each figure names the script that produces it, so
nothing here has to be taken on trust or re-derived from memory.

Reproduce: `model/train.py --cv 5`, `model/clinical.py`,
`model/personalise.py`, `model/sign_condition.py`, `model/width_sweep.py`,
`model/ablation.py`, `test/monotonicity.py`, `./mutate.sh`, `./lint.sh`.

Every table below now names a script that produces it. That was not true before
2026-08-15 — §1.2, §1.3 and §5 had none — and writing the three missing ones
corrected a claim in each. See `paper/NUMBERS-CHECK.md`.

---

## 1. Headline: personalisation needs about eight meals

`model/personalise.py`. Population model trained on other participants, then
adapted to a held-out person by refitting **the output bias only** — the
smallest possible intervention, and one the chip already supports, since the
host streams that bias as two ordinary weight bytes. Meals are taken in **time
order**; sampling someone's future to predict their past would inflate this.

| meals logged | R² | MAE (mg/dL·min) | better than population |
|---|---|---|---|
| 0 (population) | +0.194 | 1861 | — |
| 1 | **−0.077** | 2150 (+290) | 19/44 |
| 2 | +0.128 | 1971 (+111) | 23/44 |
| 3 | +0.172 | 1883 (+22) | 23/44 |
| 5 | +0.239 | 1795 (−65) | 23/44 |
| **8** | **+0.307** | 1688 (−172) | **32/44** |
| 12 | +0.303 | 1652 (−209) | 30/40 |
| 16 | +0.306 | 1566 (−294) | 27/38 |

**Paired per-subject MAE change, 95 % CI across participants** — the gain does
not rest on a pooled point estimate:

| k | Δ MAE | 95 % CI | verdict |
|---|---|---|---|
| 1 | **+354** | [+77, +631] | **significantly HARMS** |
| 2 | +158 | [−71, +387] | not established |
| 3 | +58 | [−130, +247] | not established |
| 5 | −77 | [−228, +74] | not established |
| **8** | **−150** | **[−285, −16]** | **significantly helps** |
| 12 | −188 | [−336, −40] | helps |
| 16 | −248 | [−402, −94] | helps |

Three things worth saying:

- **One meal is worse than none.** Calibrating on a single noisy observation
  actively harms (R² +0.194 → −0.077). Personalisation has a cost of entry.
- **Break-even is around five meals**, and **eight — roughly three days of
  logging — buys +0.113 R²**, a larger gain than any architectural choice
  tested here.
- This is what justifies streaming weights rather than storing them. The
  architecture was chosen for area; the data says it was worth having.

### 1.1 One parameter is the right amount

The chip streams all 83 parameters, so more than the bias could be refitted per
person. The natural next step is an affine calibration `y' = a·y + b` — still
closed form, still **monotonicity-preserving provided a ≥ 0**, since a
non-negative scale cannot reorder anything.

| k | bias-only MAE | affine MAE | Δ | median a | a clamped to 0 |
|---|---|---|---|---|---|
| 2 | 2080 | 3949 | **+1870** | 0.13 | **18/44** |
| 3 | 1979 | 2406 | +427 | 0.99 | 9/44 |
| 5 | 1860 | 1845 | −15 | 0.72 | 5/44 |
| 8 | 1742 | 1701 | −42 | 0.67 | 7/44 |
| 12 | 1744 | 1669 | −75 | 0.76 | 3/40 |
| 16 | 1637 | 1603 | −34 | 0.82 | 4/38 |

The second parameter is **catastrophic at small k** and buys 40–75 mg/dL·min at
larger k, against an MAE near 1700. Bias-only is the right choice for the regime
a real user is in, and that is now a measured decision rather than a default.

**The monotonicity clamp is not theoretical.** At k = 2 the fitted slope had to
be clamped at zero for **18 of 44 patients** — an unconstrained least-squares
calibration would have returned a *negative* slope and silently inverted the
model for 41 % of them.

### 1.2 The architecture is data-limited, not capacity-limited

Hidden width is the one structural choice baked into silicon (it is the depth of
the `h` shift register), so it is worth knowing whether 8 was right.

`model/width_sweep.py` (added 2026-08-15 — this had no script before, and the
numbers below replace an earlier table that could not be re-derived).

Paired across the same 5 folds, against `N_HIDDEN = 8`. Both models are swept,
because it is the **constrained** one that ships:

| width | Δ R², carbs ↑ (shipped) | 95 % CI | Δ R², unconstrained | 95 % CI |
|---|---|---|---|---|
| 4 | +0.012 | [−0.005, +0.030] | **−0.015** | **[−0.021, −0.008]** |
| 6 | +0.004 | [−0.027, +0.035] | −0.003 | [−0.026, +0.021] |
| 12 | −0.022 | [−0.059, +0.015] | +0.014 | [−0.006, +0.034] |

Mean R², carbs ↑: 4 → +0.235, 6 → +0.227, 8 → +0.223, 12 → +0.201.

**For the model that ships, nothing from 4 to 12 is distinguishable from 8.**
The model is limited by the data, not by capacity — the honest explanation for
R² ≈ 0.22 — and shrinking the hidden layer to buy area for the fibre guard
would be a structural RTL change bought with a noise-level difference.

⚠️ **One row is no longer a blanket claim.** In the *unconstrained* model width
4 is measurably worse than 8 (−0.015, interval excluding zero), so "nothing
from 4 to 12 is distinguishable from 8" is true of the shipped model and not of
the unconstrained one. The effect is tiny and would not survive Bonferroni over
the project's ~10 comparisons, but state it with the qualifier. The earlier
table reported all three deltas as positive; these are measured.

### 1.3 The hole this opened, and what closed it

The monotonicity guarantee is a property **of the weights**. Personalisation
means a different weight set per patient, streamed by a host the chip has no
reason to trust — so the question is whether a realistic per-person fit still
satisfies the sign condition.

**It never does.** Fine-tuning an unconstrained network on each held-out
participant's own meals:

| | |
|---|---|
| per-person weight sets checked | 44 |
| sets violating the sign condition | **44 (100 %)** |
| offending hidden units per bad set | median **3 of 8** |

`model/sign_condition.py` (added 2026-08-15 — nothing computed this before,
which for the project's load-bearing claim was the wrong thing to leave
unreproducible). It confirms both figures and adds that they hold across
fine-tuning budgets from 20 to 400 epochs.

**One nuance it surfaced, worth stating before a reviewer does:** the
*population* model already fails on ~3 of 8 units before any personalisation.
An unconstrained fit simply never satisfies the condition. So the precise claim
is not that per-person refits are uniquely bad — it is that personalisation
makes the condition **undecidable offline**, because the weight set the device
runs is not the one anyone certified.

Enforcing monotonicity therefore cannot be left to whoever trains the model.
Either the host is trusted to use a constrained objective, or the chip verifies
its own precondition — and it now does the latter. See §9.

---

## 2. What the device does for a person

`model/clinical.py`, cross-fitted so all 45 participants are held out exactly
once (1,308 meals). A single split had left **two** pre-diabetic participants
carrying that subgroup and produced the opposite conclusion about T2D.

**Within-participant ranking** — the actual use case, "should I swap this
ingredient?":

- median Spearman **ρ = +0.382** (IQR +0.215…+0.495)
- **positive in 41 of 44 participants**

⚠️ Corrected 2026-08-15. Previously +0.403, IQR +0.216…+0.495, 40 of 44.
`clinical.py` is deterministic — two runs agree exactly, and the R² and
subgroup rows in the same output still match — so only this block had drifted.
The same stale trio appears in §4's table and §5's ablation, corrected there
too.

**Population accuracy by glycemic status** (ADA A1c thresholds; the split
reproduces the documented 15/16/14 exactly):

| group | subjects | meals | R² | MAE | mean iAUC |
|---|---|---|---|---|---|
| healthy | 15 | 463 | +0.074 | 1557 | 2023 |
| pre-diabetic | 16 | 485 | +0.077 | 1724 | 3148 |
| T2D | 14 | 360 | +0.142 | 2329 | 4603 |
| **overall** | 45 | 1308 | **+0.225** | | |

Absolute error grows with severity, but so does the response being predicted.
Per-group R² is not directly comparable across groups because within-group
variance differs.

---

## 3. Monotonicity

`test/monotonicity.py`, `model/train.py`.

**The condition.** `y` moves the required way in input `i` for every hidden
unit `j` iff `W1[j][i] · W2[j] · want[i] ≥ 0`, because saturating sums,
arithmetic shifts, ReLU and clamps are each monotone. **A saturating sum is
monotone and a wrapping one is not** — saturation is load-bearing for the
safety argument, not only for producing a sensible number. Rounding mode is a
red herring; every standard mode is monotone.

**An unconstrained fit does not satisfy it** — 4 of 8 hidden units disagree on
carbohydrate, 3 on fibre. Monotonicity must be trained for.

**Cost, grouped 5-fold CV, paired across folds:**

| comparison | Δ R² | 95 % CI | Bonferroni (m = 10) |
|---|---|---|---|
| carbs ↑ vs unconstrained | −0.007 | [−0.033, +0.020] | no difference |
| carbs ↑ + fibre ↓ vs unconstrained | −0.025 | [−0.062, +0.013] | no difference |
| depth-1 XGBoost vs unconstrained | −0.059 | [−0.118, −0.001] | **does not survive** |

Mean R²: XGBoost +0.170, unconstrained +0.230, carbs ↑ +0.223, carbs ↑ + fibre
↓ +0.205.

⚠️ Corrected 2026-08-15. The XGBoost row previously read −0.064 [−0.121,
−0.006] with a mean of +0.166. Re-running `train.py --cv 5` gives the figures
above; the baseline is bit-deterministic across repeats, so that was a stale
derived number, not noise. The reading is unchanged — the interval barely
excludes zero and does not survive Bonferroni. See `paper/NUMBERS-CHECK.md`.

**No cost to monotonicity was detected** — which is not the same as free. The
interval admits a cost up to 0.033 R², so the claim is that the study is
**underpowered** to resolve a difference this small. Nothing here survives
correction for the ~10 comparisons made across the project, the XGBoost result
included, so **do not claim the network beats the baseline**.

⚠️ This number moved once: an earlier table said −0.036 with an interval
excluding zero. That run predated the CGMacros plausibility filter and was never
regenerated. Derived results need re-running when anything upstream changes.

⚠️ **Selection bias.** Quantisation scales, hidden width, adaptation method and
constraint set were all chosen by looking at these same folds. There is no
untouched validation cohort — 45 participants was not enough to hold any back —
so every absolute figure here is optimistically biased and should be read as an
upper bound.

**Verified on the RTL** with real trained weights, both directions, bit-exact
at every point:

| input | levels | direction | observed output spread |
|---|---|---|---|
| carbohydrate | 12 | non-decreasing | 1115 |
| fibre | 16 | non-increasing | 228 |

**Honest negative result:** the R-4 truncation defect is **not reachable** with
these weights — 0 of 265 held-out meals overflow the 16-bit field at any `s2`
from 0 to 6. The defect is real and appears in 55 of 400 synthetic weight sets,
but no claim of patient harm on real data is available.

---

## 4. Three data-quality problems in CGMacros

`model/cgmacros_loader.py`. None of these are in the dataset paper.

1. **`Meal Type` has ten distinct label strings for four meal types** —
   `Breakfast`/`breakfast`, `Lunch`/`lunch`, `Dinner`/`dinner`,
   `Snacks`/`snack`/`Snack`/`snack 1` — and snacks are not documented at all.
   Filtering `== 'Breakfast'` returns **170 of 436** records, silently
   discarding 61 %.
2. **52 records (3.0 %) exceed the dataset's own published maxima**: Fiber up
   to **2830 g** against a documented 176, Calories 2250 against 1180, Carbs
   229. Nineteen participants affected.
3. **Every violation falls in a self-selected meal** (dinner or snack). The
   standardised breakfasts and lunches have none.

A single 2830 g fibre entry lifts the training standard deviation for fibre to
32 g against a median of 3, distorting feature standardisation and, downstream,
the INT8 input quantisation.

**Effect of filtering against the published ranges:**

| | before | after |
|---|---|---|
| out-of-fold R² | +0.215 | +0.225 |
| within-participant ρ (median) | +0.348 | +0.382 |
| participants ranked correctly | 39/44 | 41/44 |
| quantisation error (mean) | 418 | 172 mg/dL·min |

Separately and already documented in the dataset paper — **not a finding, cite
it**: breakfasts and lunches were standardised by design. 383 usable breakfasts
contain only 6 distinct macronutrient combinations, carbohydrate taking 3
values, which is why the reference notebook's breakfast-only scope is a
*personalisation* study rather than a meal-composition one.

---

## 5. Quantisation

`test/golden_float.py`, `model/clinical.py`. Scales selected by search over a
held-out fold, not by hand.

| configuration | mean error | p95 |
|---|---|---|
| original (5,6,6,6,0), uncleaned data | 418 | 1554 |
| after data cleaning | 172 | 1177 |
| **selected (5,5,5,5,1)** | **85** | **180** |

**4.9× on the mean and 8.6× on the tail.** Quantisation now costs ~3 % of a
typical iAUC rather than 14 %. The tail matters more than the mean — that is
where a wrong answer would reach someone.

⚠️ Two caveats on that sentence (2026-08-15):

- Those ratios compare *original scales on uncleaned data* against *selected
  scales on cleaned data*, so they bundle the data-cleaning gain into a claim
  about scale search. The **scale search alone** is 172 → 85 (2.0×) on the mean
  and 1177 → 180 (6.5×) on the tail, which is what `golden_float.py`'s docstring
  says. Report the two effects separately.
- `clinical.py` also prints **max 7305 mg/dL·min** over the same 265 meals.
  Giving mean and p95 but not max, while arguing the tail is what matters, is
  the kind of omission a reviewer notices. The paper states it.

**Ablation — per-participant features.** The input count is not fixed in
silicon, so bio features are free in hardware. They still do not help:

`model/ablation.py` (added 2026-08-15). The earlier table had no script and did
not name its five bio features, so this is a **fresh measurement with a stated
feature set** — A1c, BMI, Age, fasting glucose, insulin — rather than a
reproduction. Its baseline row now agrees with §2 exactly, which the old one did
not (+0.236 against §2's +0.225).

| features | R² | ρ (median) | positive |
|---|---|---|---|
| 6 meal features | +0.225 | +0.382 | 41/44 |
| + A1c | **+0.256** | +0.395 | 37/44 |
| + A1c + BMI | +0.207 | **+0.463** | 40/44 |
| + all 5 bio | **−0.014** | +0.440 | 42/44 |

All five collapse R² — with 45 participants each participant-level feature fits
45 points. A1c alone helps population R² and leaves ranking essentially
untouched, while *reducing* the number of participants ranked in the right
direction from 41 to 37. Ranking is the use case, so the design keeps the 6
meal features.

⚠️ **Do not say a participant-level feature "cannot improve ranking by
construction."** That holds only if the feature enters additively. This model
is an MLP, so a constant-within-person input can still interact with the meal
features and change within-person ordering — and measurably does, in both
directions (ρ +0.382 → +0.395 with A1c, but 41/44 → 37/44). The argument for
dropping bio features is empirical, not structural.

---

## 6. Silicon

Tiny Tapeout IHP 26b, `ihp-sg13g2`, 1×1 tile.

| | |
|---|---|
| Flip-flops / standard cells | 168 / 1443 |
| Utilisation | 83.53 % |
| Setup / hold worst slack | +10.08 ns / +0.120 ns @ 20 ns |
| DRC / LVS / antenna / latches / lint | 0 / 0 / 0 / 0 / 0 |
| **Latency** | **914 cycles** = 18.3 µs @ 50 MHz, 0.91 ms @ 1 MHz |
| **Energy per prediction** | **32.7 nJ** |
| Energy, 3 meals/day for a year | 35.8 µJ |

`gds`, `precheck` and `gl_test` all pass. Gate-level simulation is
**functional only** — `-DFUNCTIONAL -DSIM`, no SDF back-annotation — so timing
is claimed on STA alone.

---

## 7. Verification

| | |
|---|---|
| Top-level tests | 41, all bit-exact against the integer reference |
| Unit tests | 6, **exhaustive** over all 65,536 signed 8×8 multiplier inputs |
| Mutation score | **29 mutants: 28 caught, 1 proven equivalent, 0 escaped** |
| Functional coverage | **54/54 named bins**, asserted not merely printed |

Four RTL defects and five testbench defects are logged in `BUGS.md`, including
three cases where the environment reported a false pass.

---

## 8. The on-chip monotonicity guard — implemented

`src/proof_core.v`. The chip verifies the safety precondition of the weights it
is handed rather than assuming it.

Both operands already cross the pins: the first weight byte of hidden neuron
`j` is `W1[j][carb]`, and layer-2 weight byte `k` is `W2[k]`. Their signs ride a
register that shifts and rotates in lockstep with the hidden-activation
register, so the sign for the unit being consumed is always at the top. A
disagreement raises `UNTRUSTED`.

Two details it gets right, both directly tested:

- a unit may oppose carbohydrate **twice** — negative on both sides is a
  non-negative product — so a naive "all weights positive" check would be wrong;
- a **zero** carbohydrate weight can never trigger it, which is why a non-zero
  bit is carried alongside each sign.

| | before | after |
|---|---|---|
| flip-flops | 148 | **168** |
| standard cells | 1330 | 1443 |
| utilisation | 76.35 % | **83.53 %** |
| setup / hold | +9.89 / +0.121 ns | +10.08 / +0.120 ns |
| DRC / LVS / latches / lint | 0 | 0 |

**One pin, two causes.** All eight `uio` bits were already allocated, so
`SATURATED` widened into `UNTRUSTED`: numeric overflow *or* void guarantee. Both
mean the same thing to a caller, and the host holds the weights so it can always
tell which.

That merge cost observability, and the mutation suite caught it within minutes:
the saturation tests used a carbohydrate weight of −128 against a positive `W2`,
so they *also* violated the sign condition and M14 could hide behind the guard.
Those weight sets now agree in sign, isolating saturation. M16 went **INVALID**
rather than silently scoring, because the guard rewrote the line it patched —
the did-not-apply guard doing its job for the second time this project.

**Why it matters for the claim.** It changes the statement from "this chip
computes a monotone function if whoever trained the model happened to constrain
it" to "this chip declines to vouch for a weight set that cannot be monotone."
Given the 100 % violation rate in §1.1, that is the difference between a
guarantee and a hope.

**At 83.53 % utilisation there is no room for another feature.** Anything
further is a 1×2 tile conversation.

---

## 9. Related work, and what is actually new

Checked 2026-08-15 across four independent angles. **Nothing found collides**,
but the surviving claim is narrow and must be stated as such.

### Established — cite, do not claim

| area | representative work | why it is not this |
|---|---|---|
| Monotonic networks by construction | Runje & Shankaranarayana, ICML 2023; *Scalable Monotonic NNs*, ICLR 2024 | The softplus reparameterisation used here is squarely this family. Not a contribution. |
| Certified monotonicity | Liu et al., NeurIPS 2020 (MILP) | Verification is **offline**, on a fixed model, and computationally costly. |
| Monotonicity for clinical trust | established motivation in the ML literature | The motivation is borrowed, not new. |
| Quantization destroying structural guarantees | *Quantization Robustness of Monotone Operator Equilibrium Networks*, 2026 | Closest hit, and still distinct: **monotone-operator** theory rather than input-output monotonicity, an **offline** certificate (‖ΔW‖₂ < m), and **no hardware**. |
| Quantized-network verification | QVIP; SMT-based model checking of QNNs | Offline, and verifies the network — not the pin interface. |
| Concurrent error detection in accelerators | algorithmic checksums; parity CED; uncertainty fingerprints; monitor placement | **These detect faults.** They ask "did the hardware compute correctly?" |
| Runtime safety monitoring | *Run-Time Safety Monitoring of NN-Enabled Dynamical Systems* | System-level, software, monitors **outputs** — not a precondition on weights. |

### The gap, stated precisely

Every monotonicity guarantee above is established **at training time and verified
offline, on a fixed model**. That is sound only while the model is fixed.

**Streaming weights per patient breaks exactly that assumption.** Offline
verification no longer covers what the device runs, and §1.3 shows the failure
is not hypothetical: an unconstrained per-person refit violates the sign
condition in **44 of 44** cases.

The claim, in one sentence: *the precondition for a monotonicity guarantee is
checkable from the weight stream itself, in hardware, at the interface, for 20
flip-flops — and a device that streams per-user weights needs that check because
offline verification cannot cover it.*

The distinction that carries it: concurrent error detection asks **"did the
hardware compute correctly?"** — this asks **"does this weight set admit the
guarantee at all?"** In the failing case the arithmetic is perfectly correct and
the result is bit-exact against the reference. Nothing else on the chip, and
nothing in the CED literature, can tell.

### Honesty about the search

Four targeted searches found no prior hardware checking a semantic precondition
on streamed weights. That is **weak evidence** — phrase it as "to our knowledge"
and mean it, and treat the CED and runtime-monitoring literature as the nearest
neighbours a reviewer will reach for.

---

## 10. Claims to avoid

- ❌ "We beat XGBoost." The interval straddles zero.
- ❌ "Monotonicity is free." **No cost was detected** — −0.007 [−0.033, +0.020]
  — but the interval admits a cost up to 0.033 R², so the study is underpowered
  to resolve a difference that small. *(Corrected 2026-08-15: this bullet used
  to cite −0.036 [−0.059, −0.012], which is the pre-plausibility-filter figure
  §3 already flags as stale. The advice was right and the number was the one it
  warns against.)*
- ❌ "We discovered CGMacros breakfasts are standardised." Documented in the
  dataset paper.
- ❌ "The truncation bug endangers patients." Not reachable on real data.
- ❌ "We invented monotonic networks." Established field — cite Liu et al.
  (NeurIPS 2020), Runje & Shankaranarayana (ICML 2023).
- ⚠️ "To our knowledge" is doing real work in the guard claim. **Four** targeted
  searches (§9, 2026-08-15) found no prior hardware that checks a monotonicity
  precondition at its own interface, but absence of evidence in four searches is
  weak. Phrase it as a limitation of the search, not as established priority.
  *(This bullet said "two"; §9 and HANDOFF both say four.)*
