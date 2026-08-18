# Numbers check — 2026-08-15

Everything in `RESULTS.md` was re-derived before any of it went into a figure or
the paper, because HANDOFF records stale derived numbers as a repeat failure of
this project. Re-run:

```
model/train.py --cv 5     model/clinical.py       model/personalise.py
model/sign_condition.py   model/width_sweep.py    model/ablation.py     (all new)
cd test && python run.py --module test_cycles                                (new)
test/monotonicity.py      paper/figures/find_r4_case.py                 (new)
```

All under `.venv-legacy` (Python 3.8, sklearn 0.24.2, xgboost 1.6.2).

---

## Verdict

**Most of RESULTS.md reproduces exactly.** Four numbers were stale. Three
claim clusters had no script at all; writing those three scripts corrected a
claim in each. Two further statements are narrower than the sheet implies, and
two references were wrong.

No headline conclusion changed. But eight of these are numbers or claims a
reviewer could check, and three were overstatements rather than typos:

- "nothing from 4 to 12 is distinguishable from 8" — true of the shipped model,
  false of the unconstrained one (§4a);
- a participant-level feature "cannot improve ranking by construction" — false
  for an MLP (§4b);
- §1.3 read as "personalisation opened this hole" — an unconstrained fit never
  satisfied the condition in the first place (§3).

---

## 1. Reproduced exactly

| section | what | status |
|---|---|---|
| §1 | the whole personalisation table, all 8 rows | exact |
| §1 | every paired CI, k=1…16 | exact |
| §1.1 | affine-vs-bias table, all 6 rows, incl. 18/44 clamped | exact |
| §2 | out-of-fold R² +0.225, 1308 meals, 45 participants | exact |
| §2 | all three subgroup rows (15/463, 16/485, 14/360) | exact |
| §3 | carbs ↑ −0.007 [−0.033, +0.020] | exact |
| §3 | carbs ↑ + fibre ↓ −0.025 [−0.062, +0.013] | exact |
| §3 | 0 of 265 meals overflow at any s2 0…6 | exact |
| §5 | selected scales: mean 85, p95 180 | exact |
| §6 | 83.53 % util, 1443 cells, 168 flops, 0 DRC/LVS/antenna | exact, from `metrics.csv` |
| §8 | R-4: 55 of 400, 31,293 → −31,209, internal → 34,327 | exact, bit-for-bit |

The silicon numbers were confirmed independently against the `gds` workflow
artifact for commit `c0a5792`, whose `src/*.v` is **byte-identical** to the
working tree — so the layout render and the metrics describe the RTL as signed
off. `design__instance__utilization` = 0.835308,
`design__instance__count__stdcell` = 1443,
`design__instance__count__class:sequential_cell` = 168.

---

## 2. Stale numbers — corrected in RESULTS.md

Each was re-run twice and is **deterministic**, so this is drift from an
un-regenerated derived figure, not run-to-run noise.

### 2a. §3 — the XGBoost baseline row

| | RESULTS.md said | actual |
|---|---|---|
| depth-1 XGBoost vs unconstrained | −0.064 [−0.121, **−0.006**] | **−0.059 [−0.118, −0.001]** |
| mean R², XGBoost | +0.166 | **+0.170** |

Probed for nondeterminism directly: 5 repeats in one process give
`+0.170236` every time, per-fold identical, seeded or not. The conclusion is
unchanged and in fact slightly strengthened — the interval now *barely* excludes
zero — so "do not claim the network beats the baseline" stands.

The same row had **propagated to two other documents, with two further
values**. Before 2026-08-18 the identical comparison read:

| document | Δ R² | 95 % CI |
|---|---|---|
| RESULTS.md §3 | −0.064 | [−0.121, −0.006] |
| VERIFICATION.md | −0.064 | [−0.121, −0.006] |
| BUGS.md | **−0.048** | **[−0.098, +0.003]** |
| **measured** | **−0.059** | **[−0.118, −0.001]** |

BUGS.md's version straddled zero and was used there to support "the network does
not beat the baseline." The conclusion survives — nothing survives correction for
~10 comparisons — but it was resting on a number that no longer existed anywhere
else. VERIFICATION.md's per-fold row for the same model was stale too (+0.166,
sd 0.054). All three now carry the measured figure.

### 2b. §2, §4, §5 — within-participant Spearman

| | RESULTS.md said | actual |
|---|---|---|
| median ρ | +0.403 | **+0.382** |
| IQR | +0.216 … +0.495 | **+0.215 … +0.495** |
| positive participants | 40/44 | **41/44** |

`clinical.py` is deterministic — two runs agree exactly — and the R² and
subgroup rows in the same output match RESULTS.md perfectly. So only the
Spearman block drifted. This appears in three places: §2's headline, §4's
before/after table, and §5's ablation table.

### 2c. §10 — the second "claim to avoid" carries the number it warns about

> ❌ "Monotonicity is free." It costs −0.036 [−0.059, −0.012].

−0.036 is the **stale** figure HANDOFF explicitly records as corrected
(it predated the CGMacros plausibility filter). The advice is right; the number
is the one it warns against. Now reads −0.007 [−0.033, +0.020], with the reason
restated as *underpowered*, not *free*.

### 2d. §6 — latency, and the energy figure derived from it

*(found 2026-08-17, after the first pass)*

| | RESULTS.md said | actual |
|---|---|---|
| latency | 914 cycles, 18.3 µs @ 50 MHz | **896 cycles, 17.9 µs** |
| energy per prediction | 32.7 nJ | **32.0 nJ** |
| energy, 3 meals/day for a year | 35.8 µJ | **35.1 µJ** |

The only trace of where 914 came from was `test/results_test_cycles.xml`, whose
`file` attribute names a `test/test_cycles.py` that **was never committed** and
is not on disk — so a number in the paper's silicon table had no producer at
all, and the energy figure is derived from it.

Re-measured with a new `test/test_cycles.py`: a full 6-8-1 inference takes
**896 cycles**, deterministically. The 18-cycle gap is fully explained, and the
new test asserts the explanation: the testbench helper `run_mode_b` reads the
result register back after *every* neuron, at two cycles each, and nine neurons
is exactly 18. That is a verification convenience — a deployed host reads only
the final `y`, since layer 2 takes its activations from `hreg` internally and
never needs the host to observe them.

So 914 was a real measurement of the wrong thing. Both figures are now asserted
by tests (`python run.py --module test_cycles`), which also pins that latency is
data-independent — worth having, because the energy number is
cycles × power and would otherwise be a mean presented as a constant.

This also needed a small additive change to `test/run.py`: a `--module` flag, so
the top-level build can run a test module other than `test`. Default behaviour
is unchanged, and the latency tests stay out of the main suite's runtime, so
"41 top-level tests" still means what it did.

---

## 3. A load-bearing claim that had no script — now it does

**§1.3's "44 of 44" is the single measurement the entire thesis rests on**, and
nothing in the repository computed it. Added `model/sign_condition.py`.

It **reproduces exactly**: 44 of 44 participants violate, median 3 of 8
offending units. Better, it is stable across fine-tuning budgets from 20 to 400
epochs, which is stronger evidence than a single number at one arbitrary
setting.

One nuance the script surfaced, and the paper states carefully: **the population
model already fails on ~3 of 8 units before any personalisation.** So an
unconstrained fit simply never satisfies the sign condition. §1.3's framing
("personalisation opened this hole") could be read as overclaiming. The precise
statement — the one in the paper — is that personalisation is what makes the
condition *undecidable offline*: not that per-person refits are uniquely bad,
but that the weight set the device runs is not the one anyone certified. A
reviewer who reruns this will notice, so it is better said first.

---

## 4. The two missing scripts — now written, and both changed a claim

RESULTS.md's header promises "each figure names the script that produces it."
Two clusters did not. Both now have scripts, and running them was worth it:
each corrected an overstatement.

### 4a. `model/width_sweep.py` — §1.2

The old table reported all three deltas against width 8 as positive with
intervals spanning zero, and concluded "nothing from 4 to 12 is distinguishable
from 8." Measured, for the **constrained model that ships**, that holds:

| width | Δ R² (carbs ↑) | 95 % CI |
|---|---|---|
| 4 | +0.012 | [−0.005, +0.030] |
| 6 | +0.004 | [−0.027, +0.035] |
| 12 | −0.022 | [−0.059, +0.015] |

But for the **unconstrained** model, width 4 is measurably worse than 8:
**−0.015 [−0.021, −0.008]**, an interval excluding zero. So the blanket phrasing
was wrong; the paper now says "for the constrained model that ships." The design
decision is unaffected, and arguably better supported — shrinking the layer is
not the free move the old table implied.

Note the mechanism: `train.N_HIDDEN` is a module global read at call time, so
the sweep rebinds it and restores it in a `finally`. That is process-global
mutable state, so each width gets an independent pass.

### 4b. `model/ablation.py` — §5

The old table did not name its five bio features and none is recorded anywhere,
so this is a fresh measurement with a stated set (A1c, BMI, age, fasting
glucose, insulin) rather than a reproduction. It lands very close — the
+A1c+BMI ρ matches to three decimals, and three of four positive-counts match
exactly — and its baseline row now agrees with §2 exactly, which the old one did
not.

**It also contradicts a structural claim.** §5 said A1c "cannot improve
within-person ranking **by construction**, being constant within a person."
That is only true if the feature enters additively. This is an MLP, so a
constant-within-person input interacts with the meal features and *can* change
within-person ordering — and does, in both directions: median ρ rises
+0.382 → +0.395 while participants ranked correctly fall 41 → 37. The case for
dropping bio features is empirical, not structural. Corrected in both RESULTS.md
and the paper.

---

## 5. Two statements narrower than they read

### 5a. §5's "4.9× on the mean and 8.6× on the tail"

Those ratios compare **original scales on uncleaned data** (418 / 1554) against
**selected scales on cleaned data** (85 / 180), so they bundle the data-cleaning
gain into a sentence about scale search. The scale search alone is 172 → 85
(2.0×) on the mean and 1177 → 180 (6.5×) on the tail. `golden_float.py`'s own
docstring says 6.5×. The paper reports the two effects separately.

### 5b. The maximum quantisation error is not reported

`clinical.py` prints `max 7305.1 mg/dL·min` over the same 265 meals. §5's table
gives mean and p95 only, while arguing "the tail matters more than the mean —
that is where a wrong answer would reach someone." Reporting p95 but not max
while making that argument is the kind of thing a reviewer will notice. The
paper states the max.

### 5c. "83 parameters" is 83 *bytes*, not 83 parameters

HANDOFF and RESULTS.md both say the chip "streams all 83 parameters". Derived
from `golden_float.to_chip_streams`, the 83 breaks down as:

| | |
|---|---|
| layer-1 weight bytes (8 neurons × 8, incl. 16 bias bytes) | 64 |
| layer-2 weight bytes (8 weights + 2 bias) | 10 |
| requantisation shift bytes (one opening each neuron) | 9 |
| **total non-activation bytes** | **83** |

The model has **65 distinct learned parameters** (W1 48 + b1 8 + W2 8 + b2 1).
So 83 counts bias-splitting bytes twice and includes 9 shift/control bytes that
are not parameters at all. The paper says 65 parameters carried as 74 weight and
bias bytes plus 9 shift bytes. Worth fixing the phrasing in RESULTS.md and
HANDOFF too.

### 5d. One non-zero violation counter

`metrics.csv` has `design__max_fanout_violation__count = 1` at all three
corners (`design__violations` is 0 overall, and DRC/LVS/antenna are genuinely
zero). §6's "0 / 0 / 0 / 0 / 0" doesn't cover fanout and isn't wrong, but
"clean signoff" phrasing shouldn't be read as zero of everything.

---

## 6. References — checked against the publications

All seven substantive references were verified on 2026-08-15 against the
publications themselves, not from memory. Two were wrong as first drafted:

- **CGMacros** is *"a **pilot** scientific dataset for personalized nutrition
  and diet monitoring"* — Das, Kerr, Glantz, Bevier, Santiago, Gutierrez-Osuna
  and Mortazavi, *Scientific Data* **12**, art. 1557 (2025). The PhysioNet
  landing page drops "pilot" from the title; cite the paper, not the page.
- **Xiang** is **2022**, not 2023 — *IEEE Trans. Cybernetics* 52(9),
  pp. 9587–9596. The citation key was renamed to match.

Confirmed as drafted: Liu, Han, Zhang & Liu (NeurIPS 2020); Runje &
Shankaranarayana (ICML 2023, PMLR v202); Kim & Lee, *Scalable Monotonic Neural
Networks* (ICLR 2024); Li, Leong & Chaffey, *Quantization Robustness of Monotone
Operator Equilibrium Networks* (arXiv:2603.10562, 2026); Zhang, Zhao, Chen,
Song, Zhang, Chen & Sun, *QVIP* (ASE 2022).

Only `\bibitem{tinytapeout}` is still open — it is a platform with no canonical
paper, so pick whatever form the maintainers ask for.

## 7. Internal inconsistency

§9 and HANDOFF say the related-work check was **four** searches; §10's last
bullet says **two**. The paper says four, per §9's dated table. Worth
reconciling in RESULTS.md.

---

## 8. Changes made to the repo

- `model/personalise.py` — added `--json PATH`. **Purely additive**: it dumps
  values already computed and printed, after all printing is done. Verified by
  diffing stdout with and without the flag — byte-identical.
- `model/sign_condition.py` — **new**, reproduces §1.3.
- `model/width_sweep.py` — **new**, reproduces §1.2.
- `model/ablation.py` — **new**, replaces §5's ablation.
- `test/test_cycles.py` — **new**, measures and pins the latency figure.
- `test/run.py` — added `--module`; default behaviour unchanged.
- `RESULTS.md` — the stale numbers above, each marked inline; §1.2 and §5
  replaced with measured tables; the reproduce line now lists every script.
- `paper/` — figures and their scripts, the draft, `check_tex.py`, this note.

No RTL was touched: `src/` is clean. In `test/`, only the two additions above — no existing testbench or test was modified.
