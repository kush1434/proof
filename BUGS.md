# Bug log — Proof

Owner: design *and* verification. On the prior CNN accelerator
(`kush1434/cnnaccelerator`) the RTL was owned by someone else and this log
covered only the testbench; here both sides are mine, so an RTL defect below is
a defect I wrote.

Environment: Icarus Verilog 12.0, cocotb 2.0.1, Python 3.13 (host) /
Python 3.8 (`.venv-legacy`, CGMacros baseline only).

```bash
cd test
python run.py                    # whole design, RTL      (36 tests)
python run.py --unit mac_serial  # submodule unit test    (6 tests)
python run.py --gates            # post-layout netlist (needs PDK_ROOT)
./mutate.sh                      # mutation testing       (24 mutants)
./lint.sh                        # local pre-push synthesis gate
```

---

## RTL defects

| # | Date | Symptom | Test that found it | Root cause | Fix |
|---|------|---------|--------------------|------------|-----|
| R-1 | 2026-08-14 | The `gds` workflow died in synthesis: `ERROR: Assert 'arg->is_signed == sig.as_wire()->is_signed' failed in frontends/ast/genrtlil.cc:2128`. All 26 tests passed and Icarus compiled it without complaint | GitHub Actions `gds` job — **nothing in simulation saw it** | `proof_core` connected the multiplier's operand as `.b($signed(data))`. A `$signed()` cast applied inside a module port connection trips an internal yosys assertion. The design was not wrong, it was *unsynthesisable*, and a simulator has no reason to care | Connect `data` as a plain net. The widths match and `mac_serial` declares `b` signed, so the bits are interpreted correctly without the cast. Added `lint.sh`, which elaborates through yosys locally and reproduces this class of failure in about a second instead of a ten-minute CI round trip |
| R-2 | 2026-08-14 | Verilator `WIDTHEXPAND` warning at `mac_serial.v:57` — surfaced in the Action summary | CI Verilator lint (`LINTER_INCLUDE_PDK_MODELS: 1`) | `step == DW - 1` compares a `CW+1` bit register against a 32-bit integer expression | Introduced `localparam [CW:0] LAST_STEP = DW - 1` and compared against that. That traded the warning for `WIDTHTRUNC` on the initialiser, so the final form tests the low `CW` bits against all-ones, which is sized exactly on both sides |
| R-3 | 2026-08-14 | Mode B: every hidden activation read back **correct**, but `hreg` was permanently zero, so all three outputs were wrong. 5 of 31 tests failed — and only the ones with asymmetric stimulus | `test_mode_b_h_order_is_preserved` (written specifically to be asymmetric), confirmed by probing `hreg` directly | The neuron was retired in `S_ACC`, the same cycle the accumulator's own `add_en` is high. So `acc` still held the value from **before the final term**, and the value pushed into `h` was one term short. `h` *read back* correctly because that path is combinational off `acc` and is sampled later, after the accumulator has settled — the readback and the stored copy disagreed | Added `S_FIN`, the cycle after `S_ACC`, where `acc` is final. All retire actions (requantise, ReLU, push to `h`, advance the neuron counter, raise `done`) moved there |
| R-4 | 2026-08-14 | The **safety property failed**: with a weight set satisfying the sign condition, increasing carbohydrate by one count made the *reported* response fall 31,293 → −31,209, while the true internal value correctly rose 31,293 → 34,327. 55 of 400 randomised weight sets showed it | `test/monotonicity.py`, a property study over the reference model | The pipeline is provably monotone internally — saturating sums, arithmetic shifts, ReLU and clamps are each monotone. But the host never reads the internal value; it reads a fixed-width field, and the output fields **truncated**. Truncation wraps, and wrapping is not monotone. The accumulator saturated correctly all along; the bug was one level further out, at the boundary between the datapath and the pins | Both output fields now saturate: Mode A clamps to signed 14 bits, Mode B to signed 16. Clamping is monotone, so the property now survives to what the host actually sees. Study reports 0/400. Added `test_monotonic_in_carbohydrate`, which sweeps the RTL across the field limit, and mutants M23/M24 |

**R-4 is the project's actual thesis, and it was found by asking a question
rather than by running a test.** No stimulus-driven test was going to surface
it: every individual result was bit-exact against the reference model, because
the reference model truncated too. Both were wrong in the same way, so
comparing them proved only that they agreed. It took stating the property in
its own terms — *does this ever go down when it should go up?* — and checking
that instead.

It is also a reminder that saturation is not only about producing a sensible
number. **A saturating sum is monotone and a wrapping one is not**, so §4.7's
requirement was load-bearing for the safety argument in a way that was not
obvious when it was written.

**R-3 is the one to remember for a different reason.** Every test that passed had a final term of
zero — `[[1,0,0,0,0,0]]` style weight rows — which makes “accumulator before
the last term” and “accumulator after the last term” identical. Four tests
agreed the design was correct and were all structurally blind to it in the same
way. It took stimulus built to be asymmetric to see it at all — the M2 lesson
from the CNN accelerator arriving in a new costume: *a transposed kernel is
invisible to symmetric inputs.*

It is also worth noting **what did not catch it**: the readback was right. A
testbench checking only the observable output of each hidden neuron would have
passed while the stored state was wrong, and the corruption would only surface
one layer later, far from its cause.

**R-1 is the important one, and the lesson is not about `$signed`.** A design can
pass every simulation and still be unbuildable, because simulation and
synthesis disagree about what is legal. Twenty-six passing tests said nothing
about whether the thing could be made. That is now covered by `./lint.sh` as a
pre-push gate rather than by waiting for CI.

As of 2026-08-14 **both modes are complete**: `mac_serial`, `accumulator`,
`proof_core` and the pin wrapper. It passes 36 top-level tests and 6 unit tests, every
functional one compared **bit-exactly** against `test/golden_quant.py`.

`mac_serial` is additionally verified **exhaustively** — all 65,536 signed 8×8
input pairs, not a sample — so for that module there is no coverage argument
left to make.

That statement is only worth what the testbench is worth, which is why the
mutation results below matter more than the green log. Two of the four
testbench defects were cases where the environment reported a false pass.

---

## Testbench / methodology defects

| # | Date | Symptom | Test that found it | Root cause | Fix |
|---|------|---------|--------------------|------------|-----|
| TB-1 | 2026-08-14 | Test 1 passed, then all six remaining tests failed with `SimFailure: Simulator shut down prematurely` | first multi-test run | cocotb 2.0 **cancels every task a test started when that test ends**, which 1.x did not. The clock was started once and shared, so from test 2 onward there was no clock; the simulator ran out of events and exited. The error names the symptom, not the cause | `setup()` starts a fresh `Clock` per test. Documented in `test.py` because the message actively misleads |
| TB-2 | 2026-08-14 | A test asserting `BUSY` right after a byte was pushed failed, though the RTL was correct | bring-up suite, once TB-1 was fixed | `ClockCycles` resumes **at** the rising edge, before the NBA region runs, so reading a registered output straight after it returns hands back the pre-edge value. Combinational outputs survive this by accident; registered ones do not, so it stays invisible until the first flop is sampled | Added `settle()` (a 1 ns `Timer`) and step past the edge before every read |
| TB-3 | 2026-08-14 | `test/run.py` exited **0 with 6 of 6 tests failing**. A CI step built on it would have been structurally incapable of going red | deliberately running a known-broken design through the runner before trusting it | cocotb's `Runner.test()` reports success regardless of outcome. The template Makefile knows this and works around it (`# make will return success even if the test fails` → `! grep failure results.xml`); `run.py` reimplemented the runner and silently inherited the trap | `check_results()` parses the JUnit XML and exits non-zero on any `failure`/`error`. It also **fails on an empty testcase list**, since a test module that fails to import produces zero tests and would read as a pass. Verified both directions. `mutate.sh` now re-verifies it on every run |
| TB-4 | 2026-08-14 | Mutant **M5** (sticky overflow flag made non-sticky, `sat <= sat \| ovf` → `sat <= ovf`) **escaped** — the full suite still passed | `./mutate.sh` M5 | The saturation tests drive 520 identical maximum-magnitude terms. Once the accumulator pins at the rail, *every subsequent term in the same direction also overflows*, so `ovf` is still 1 on the final term and a non-sticky flag reads 1 anyway. The stimulus was structurally incapable of distinguishing sticky from non-sticky | Added `test_sticky_flag_survives_non_overflowing_terms`: saturate, then accumulate four terms that land comfortably inside the rail, so `ovf` returns to 0 while the flag must stay set. M5 is now caught |

| TB-5 | 2026-08-14 | Mutant **M12** (`busy` no longer covers the retire cycle) escaped the whole 31-test suite | `./mutate.sh` M12 | The contract is “do not present a byte while `busy`”, so its converse must hold: any cycle `busy` is low, a byte is accepted. Dropping `busy` one cycle early opens a window where the host may legally send and the byte is silently ignored, desynchronising the rest of the stream. **Every test waited on `done`, never on `busy`**, so nothing ever observed the handoff between them | Added `test_busy_covers_the_whole_retire`: after the final term, poll every cycle and assert that on the first cycle `busy` is low, `done` is already high. M12 is now caught — by exactly that one test |

TB-3, TB-4 and TB-5 are all the same family as TB-4 on the CNN accelerator — a check
that could not fire — and TB-4 here is structurally identical to that project's
M7, where a scoreboard comparing only final memory contents could not see a
write-enable bug whose last write happened to be correct. **A test that always
passes and a test that cannot fail look identical from the outside; mutation
testing is what tells them apart.**

---

## Mutation testing results

`./mutate.sh` — each mutant injected alone, relevant suites run, RTL restored
afterwards and on interrupt via an EXIT trap.

**24 considered, 23 caught, 1 equivalent, 0 escaped, 0 invalid, 0 compile-fail.**

Three properties of the harness itself, each of which had to be true before any
`CAUGHT` above means anything:

1. **The did-not-apply guard uses `diff -q --strip-trailing-cr`, not `cmp -s`.**
   This repository checks out CRLF and `sed` writes LF, so a plain byte
   comparison reports *every* file as changed — including one where the pattern
   matched nothing — and a mutant that was never injected scores as caught.
   That was TB-4 on the previous project. It is re-verified in both directions
   on every run, and the script aborts if it is broken.
2. **The runner self-test.** Before any mutant, a known-broken accumulator is
   pushed through `run.py` and the script confirms it reports failure. This
   exists because of TB-3 above.
3. **Restore is checked against a pre-run checksum snapshot, not git HEAD.**
   During development `src/` legitimately differs from HEAD, so a git-based
   check reports "NOT RESTORED" on nearly every run; a guard that cries wolf is
   one people learn to ignore.

| Mutant | Module | Injected bug | Status |
|---|---|---|---|
| M1 | `mac_serial` | drop the two's-complement sign correction on the last step | **CAUGHT** (3/6 unit, 7/20 top) |
| M2 | `mac_serial` | high accumulator one bit narrower | **EQUIVALENT** — see below |
| M3 | `mac_serial` | terminate one shift step early | **CAUGHT** (6/6 unit) |
| M4 | `accumulator` | wrap instead of saturate | **CAUGHT** |
| M5 | `accumulator` | overflow flag no longer sticky | **CAUGHT** — escaped until TB-4 was fixed |
| M6 | `accumulator` | overflow never detected | **CAUGHT** (3/20) |
| M7 | `accumulator` | `clear` ignored, state leaks between inferences | **CAUGHT** (7/20) |
| M8 | `proof_core` | accumulator never cleared at start of stream | **CAUGHT** (7/20) |
| M9 | `proof_core` | requantisation shift off by one | **CAUGHT** (14/20) |
| M10 | `proof_core` | high category threshold 20 → 21 | **CAUGHT** (4/20) |
| M11 | `proof_core` | `LAST` ignored, every pair ends the inference | **CAUGHT** (9/20) |
| M12 | `proof_core` | `busy` drops during the retire cycle | **CAUGHT** — escaped until TB-5 was fixed, now caught by exactly 1 test |
| M13 | `proof_core` | every byte treated as a weight | **CAUGHT** (18/20) |
| M14 | `proof_core` | sticky flag cleared per neuron, not per inference | **CAUGHT** |
| M15 | `proof_core` | h retired one cycle early, before the last term lands (R-3) | **CAUGHT** |
| M16 | `proof_core` | h never rotates, every layer-2 term reuses h[0] | **CAUGHT** |
| M17 | `proof_core` | ReLU removed, negatives pass through | **CAUGHT** |
| M18 | `proof_core` | h upper clamp removed, so 127 wraps | **CAUGHT** |
| M19 | `proof_core` | layer-2 boundary off by one | **CAUGHT** |
| M20 | `proof_core` | first bias constant 127 → 126 | **CAUGHT** |
| M21 | `proof_core` | LAST on the shift byte ignored, inference never restarts | **CAUGHT** |
| M22 | `proof_core` | hidden results reported as raw output, not as h | **CAUGHT** |
| M23 | `proof_core` | Mode B output field truncates, breaking monotonicity (R-4) | **CAUGHT** |
| M24 | `proof_core` | Mode A output field truncates instead of saturating | **CAUGHT** |

### An equivalent mutant, and the design observation from it

M2 escapes, and because `mac_serial`'s test is exhaustive over the entire input
space this is **not** a coverage gap — it is proof the mutant is equivalent.
Narrowing `acc_hi` from `DW+1` to `DW` bits cannot change any of the 65,536
products.

`acc_hi` is only ever written as `{sum[DW], sum[DW:1]}`, an arithmetic right
shift. The partial sum is bounded such that `sum[DW]` always equals
`sum[DW-1]`, so the top bit is a duplicated sign bit no later step reads.
**The 9th bit of `acc_hi` is dead logic, worth one flop (48.99 µm²).**

Kept anyway, deliberately: the equivalence proof holds for `DW = 8`, the width
the exhaustive test covers, and the module is parameterised. At 11.9%
utilisation the flop is cheap insurance, and if area gets tight it is a
one-character reclaim with a written justification. Recorded rather than
silently fixed or silently ignored.

This mirrors `patch_i_d` on the CNN accelerator: reasoning about *why* a mutant
correctly escaped is what surfaces redundant logic. `mutate.sh` reports it as
`EQUIVALENT` rather than `CAUGHT` or `ESCAPED`, because either of those would
be a lie in a different direction.

---

## Areas deliberately not covered

Recorded so nobody reads "all tests passed" as "everything is verified."

This list is maintained as things change. Four entries were removed on
2026-08-14 because they had become false — monotonicity was investigated,
`golden_float.py` was written, and gate-level simulation now runs against the
current design on every build. A list that names covered things is as
misleading as one that omits uncovered things.

- **Monotonicity is verified for carbohydrate only.** The property is stated,
  proved and tested for the carbohydrate input. Nothing checks whether the
  response is monotone — in either direction — in fibre, fat, protein,
  pre-meal glucose or time of day. Fibre in particular has a plausible
  monotone-decreasing expectation that is entirely unexamined.
- **The quantisation error bound has been measured but not chosen.** Median
  relative error 0.6 %, p95 2.8 % over random networks. No bound has been
  defended for this application, and that is a judgement with a safety
  argument attached rather than a number to be picked by whoever is closest.
- **The trained-network test does not run in CI.** CGMacros is CC BY-NC-SA, so
  the derived weights are gitignored. `test_trained_network_on_silicon` skips
  unless `model/train.py` has been run locally, and its coverage bin is
  exempted when absent. A green CI run does not exercise it.
- **Model results rest on one split and one seed sweep.** Held out by
  participant, but no cross-validation and no confidence intervals. The gap
  between the depth-1 XGBoost baseline (+0.212) and the monotone network
  (+0.258) is **not** established as significant.
- **Timing / setup / hold.** STA reports +9.89 ns setup and +0.121 ns hold
  worst slack at a 20 ns period. Gate-level simulation passes but compiles with
  `-DFUNCTIONAL -DSIM` and **no SDF back-annotation**, so it catches synthesis
  differences, X-propagation and missing resets while saying nothing about
  setup or hold. Timing is claimed on STA alone.
- **Only `DW = 8` and `N_HIDDEN = 8` are verified.** Both are parameters; no
  other value has been simulated, and the M2 equivalence argument is specific
  to `DW = 8`.
- **Mode B topology is partly structural.** The hidden count is fixed at 8 by
  the depth of the h shift register. The *input* count is not fixed — a neuron
  ends when the host says LAST — and neither is the output count, since the
  host simply stops streaming. Only `N_HIDDEN` would need an RTL change, and
  only before tapeout.
- **Host-side quantisation is trusted.** The chip accepts whatever shift byte
  and INT8 scaling the host chose. Nothing on-chip validates that a shift is
  sensible; a shift >= 24 simply replicates the sign bit. Well-defined,
  untested.
- **Coverage is functional, not structural.** 48 named bins, all hit, but there
  is no line, toggle or FSM-state coverage — Icarus does not produce it. A bin
  model only covers situations someone thought to name.
- **CDC.** Single clock domain by construction, so nothing to check.
