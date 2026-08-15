# Results — every number in one place

Assembled for writing. Each figure names the script that produces it, so
nothing here has to be taken on trust or re-derived from memory.

Reproduce: `model/train.py --cv 5`, `model/clinical.py`,
`model/personalise.py`, `test/monotonicity.py`, `./mutate.sh`, `./lint.sh`.

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

Three things worth saying:

- **One meal is worse than none.** Calibrating on a single noisy observation
  actively harms (R² +0.194 → −0.077). Personalisation has a cost of entry.
- **Break-even is around five meals**, and **eight — roughly three days of
  logging — buys +0.113 R²**, a larger gain than any architectural choice
  tested here.
- This is what justifies streaming weights rather than storing them. The
  architecture was chosen for area; the data says it was worth having.

---

## 2. What the device does for a person

`model/clinical.py`, cross-fitted so all 45 participants are held out exactly
once (1,308 meals). A single split had left **two** pre-diabetic participants
carrying that subgroup and produced the opposite conclusion about T2D.

**Within-participant ranking** — the actual use case, "should I swap this
ingredient?":

- median Spearman **ρ = +0.403** (IQR +0.216…+0.495)
- **positive in 40 of 44 participants**

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

| comparison | Δ R² | 95 % CI | reading |
|---|---|---|---|
| carbs ↑ vs unconstrained | −0.036 | [−0.059, −0.012] | real, small cost |
| carbs ↑ + fibre ↓ vs unconstrained | −0.027 | [−0.092, +0.037] | not established |
| depth-1 XGBoost vs unconstrained | −0.048 | [−0.098, +0.003] | not established |

**Do not claim the network beats the baseline** — that interval straddles zero.
An earlier single split reported the constraint as costing −0.000; five folds
corrected it to −0.036. Report the corrected figure.

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
| within-participant ρ (median) | +0.348 | +0.403 |
| participants ranked correctly | 39/44 | 40/44 |
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

**Ablation — per-participant features.** The input count is not fixed in
silicon, so bio features are free in hardware. They still do not help:

| features | R² | ρ (median) | positive |
|---|---|---|---|
| 6 meal features | +0.236 | +0.403 | 40/44 |
| + A1c | **+0.271** | +0.402 | 37/44 |
| + A1c + BMI | +0.224 | **+0.463** | 40/44 |
| + all 5 bio | **−0.005** | +0.444 | 42/44 |

All five collapse R² — with 45 participants each participant-level feature
fits 45 points. A1c alone helps population R² but **cannot improve within-person
ranking by construction**, being constant within a person. Ranking is the use
case, so the design keeps the 6 meal features.

---

## 6. Silicon

Tiny Tapeout IHP 26b, `ihp-sg13g2`, 1×1 tile.

| | |
|---|---|
| Flip-flops / standard cells | 148 / 1330 |
| Utilisation | 76.35 % |
| Setup / hold worst slack | +9.89 ns / +0.121 ns @ 20 ns |
| DRC / LVS / antenna / latches / lint | 0 / 0 / 0 / 0 / 0 |
| **Latency** | **914 cycles** = 18.3 µs @ 50 MHz, 0.91 ms @ 1 MHz |
| **Energy per prediction** | **28.9 nJ** |
| Energy, 3 meals/day for a year | 31.7 µJ |

`gds`, `precheck` and `gl_test` all pass. Gate-level simulation is
**functional only** — `-DFUNCTIONAL -DSIM`, no SDF back-annotation — so timing
is claimed on STA alone.

---

## 7. Verification

| | |
|---|---|
| Top-level tests | 36, all bit-exact against the integer reference |
| Unit tests | 6, **exhaustive** over all 65,536 signed 8×8 multiplier inputs |
| Mutation score | **24 mutants: 23 caught, 1 proven equivalent, 0 escaped** |
| Functional coverage | **49/49 named bins**, asserted not merely printed |

Four RTL defects and five testbench defects are logged in `BUGS.md`, including
three cases where the environment reported a false pass.

---

## 8. Claims to avoid

- ❌ "We beat XGBoost." The interval straddles zero.
- ❌ "Monotonicity is free." It costs −0.036 [−0.059, −0.012].
- ❌ "We discovered CGMacros breakfasts are standardised." Documented in the
  dataset paper.
- ❌ "The truncation bug endangers patients." Not reachable on real data.
- ❌ "We invented monotonic networks." Established field — cite Liu et al.
  (NeurIPS 2020), Runje & Shankaranarayana (ICML 2023).
