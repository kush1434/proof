# Handoff

Written 2026-08-15. Read this, then `RESULTS.md`. Everything else is detail.

---

## What this is

**Proof** — a fixed-point glycemic-response inference chip for Tiny Tapeout
IHP 26b, plus the verification and data work around it. Kush Shah's capstone,
extending his nonprofit **Mixing Mindfully** (baking workshops for diabetic
seniors).

**Two deliverables, one codebase:**

1. **Tapeout** — Tiny Tapeout IHP 26b, closes **21 September 2026**. Submit by
   the 18th. Not yet submitted. RTL freezes when he does.
2. **Paper** — IEEE BIBM 2026 Undergraduate & High School Symposium,
   <https://bibm2026-hs.github.io/>. Deadline **31 August 2026, 11:59 PM AoE**.
   5 pages including references, single-blind, IEEE format.

**They share identical RTL.** That was deliberate and should stay true.

---

## State: everything is green

| | |
|---|---|
| Tests | 41 top-level + 6 unit (exhaustive over all 65,536 multiplier inputs) |
| Mutation | 29 mutants — 28 caught, 1 proven equivalent, **0 escaped** |
| Coverage | **54/54** named bins, asserted not printed |
| Silicon | 168 flops, 1443 cells, **83.53 %** utilisation |
| Timing | setup +10.08 ns, hold +0.120 ns @ 20 ns |
| Signoff | 0 DRC / 0 LVS / 0 latches / 0 lint / 0 antenna |
| `gds`, `precheck`, `gl_test` | pass |

`viewer` fails **only** because GitHub Pages is not enabled. Fix: repo
Settings → Pages → **Source: "GitHub Actions"** (currently "Deploy from a
branch"). That is Kush's call — it publishes a page.

---

## The thesis, settled

Every monotonicity guarantee in the literature is established at training time
and verified **offline, on a fixed model**. That is sound only while the model
is fixed.

**Streaming weights per patient breaks that assumption.** An unconstrained
per-person refit violates the sign condition in **44 of 44** cases. So the chip
checks its own precondition in hardware, for 20 flip-flops.

The distinction that carries it against the concurrent-error-detection
literature: **CED asks "did the hardware compute correctly?" This asks "does
this weight set admit the guarantee at all?"** In the failing case the
arithmetic is correct and the output is bit-exact against the reference.

Related work checked across four angles on 2026-08-15 — nothing collides.
`RESULTS.md` §9 has the table and the nearest neighbours. **Four searches is
weak evidence; phrase novelty as "to our knowledge" and mean it.**

---

## Decisions already made — do not relitigate

| decision | why |
|---|---|
| Guard covers **carbohydrate only**, not fibre | Built and measured: 187 flops vs 168, ~87 % utilisation. Declined — buys the weaker claim (fibre r = −0.051) near an irreversible deadline. |
| `N_HIDDEN = 8` | Widths 4/6/12 all indistinguishable from 8 (paired CIs include zero). Model is **data-limited, not capacity-limited**. |
| Personalise by **bias only** (1 param) | Affine (2 params) is +1870 mg/dL·min *worse* at k=2, only −42 better at k=8. |
| **6 meal features**, no bio features | All five bio features collapse R² to −0.005. A1c alone helps population R² but **cannot** help within-person ranking — it is constant within a person. |
| Quantisation scales **(5,5,5,5,1)** | Searched, not hand-picked. 4.9× better mean error, 8.6× better tail than the original. |
| Weights **gitignored** | CGMacros is CC BY-NC-SA; repo is Apache-2.0. `test_trained_network_on_silicon` skips in CI. |
| Tile-only, no devkit | Chips take 7–13 months; a board would arrive after the capstone. |

---

## Environment traps

- **Windows.** No `make` — use `test/run.py`, not the template Makefile.
- **yosys** lives at `~/.apio/packages/oss-cad-suite`; put **bin *and* lib** on
  PATH or it dies loading DLLs. `./lint.sh` handles this.
- **Always run `./lint.sh` before pushing.** It catches unsynthesisable RTL that
  simulates fine (see BUGS.md R-1) in ~1 s instead of a 10-minute CI round trip.
- **Writing files from Python: pin `encoding="utf-8"` and write via temp +
  `os.replace`.** `pathlib.write_text` truncates *before* encoding, so a stray
  em dash on cp1252 leaves an empty file. This destroyed `BUGS.md` once.
- **Git on Windows drops the exec bit** — `git update-index --chmod=+x`.
- **CRLF vs LF** is why `mutate.sh` compares with
  `diff -q --strip-trailing-cr`. Do not "simplify" that to `cmp`.
- Backslash escapes do not reliably survive into heredoc-generated code. Build
  strings with `chr(10)` or avoid escapes.

---

## Two mistakes that cost real time — do not repeat

**1. Derived numbers went stale.** The headline monotonicity cost was reported
as −0.036 with an interval excluding zero. That run predated the CGMacros
plausibility filter and was never regenerated. The true figure is **−0.007
[−0.033, +0.020] — no detectable cost.** *Re-run `train.py --cv 5`,
`clinical.py` and `personalise.py` after any pipeline change.*

**2. A single split reversed two conclusions.** It said T2D was the worst
subgroup and that monotonicity was free. Cross-fitting over all 45 participants
reversed both. **Never quote a single split.**

---

## What is left

**Mine to do:**

1. **Figures — none exist yet.** A 5-page IEEE paper needs 2–4. The set the
   thesis now justifies: the personalisation curve; R-4 internal-vs-reported;
   an architecture/dataflow diagram showing where the guard sits; the layout
   render (free once Pages is on).
2. **Draft the 5 pages** from `RESULTS.md`.

**Kush's to do:**

- Enable GitHub Pages (Source → "GitHub Actions").
- **Submit to Tiny Tapeout.** RTL freezes there.
- Decide authorship — an adult co-author is allowed and must not be primary
  contributor, but **cannot present in his place**; the rule names a *student*
  author.
- Decide on Dallas, 1–4 December. In-person only, no virtual option, and a
  parent must accompany him. **Conflicts with ASGSR, 2–5 December, Virginia.**
  The symposium is one day inside 1–4 Dec but the date is still `TBD` — worth
  emailing the organisers, since if it lands on 1 December there is no clash.
  Acceptance lands 22 September, so he can decide after; withdrawing before
  camera-ready costs nothing and keeps the work unencumbered.

---

## Read before writing a word

`RESULTS.md` §10, "Claims to avoid". Five things the evidence does **not**
support, four of which were stated confidently at some point during this
project and later corrected.
