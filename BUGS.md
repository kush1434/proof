# Bug log — Proof

Owner: design *and* verification. On the prior CNN accelerator
(`kush1434/cnnaccelerator`) the RTL was owned by someone else and this log
covered only the testbench; here both sides are mine, so an RTL defect below is
a defect I wrote.

Environment: Icarus Verilog 12.0, cocotb 2.0.1, Python 3.13 (host) /
Python 3.8 (`.venv-legacy`, CGMacros baseline only).

```bash
cd test
python run.py                    # whole design, RTL      (20 tests)
python run.py --unit mac_serial  # submodule unit test    (6 tests)
python run.py --gates            # post-layout netlist (needs PDK_ROOT)
./mutate.sh                      # mutation testing       (13 mutants)
```

---

## RTL defects

| # | Date | Symptom | Test that found it | Root cause | Fix |
|---|------|---------|--------------------|------------|-----|
| — | — | *None found to date.* | — | — | — |

As of 2026-08-14 Mode A is complete: `mac_serial`, `accumulator`, `proof_core`
and the pin wrapper. It passes 20 top-level tests and 6 unit tests, every
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

TB-3 and TB-4 are both the same family as TB-4 on the CNN accelerator — a check
that could not fire — and TB-4 here is structurally identical to that project's
M7, where a scoreboard comparing only final memory contents could not see a
write-enable bug whose last write happened to be correct. **A test that always
passes and a test that cannot fail look identical from the outside; mutation
testing is what tells them apart.**

---

## Mutation testing results

`./mutate.sh` — each mutant injected alone, relevant suites run, RTL restored
afterwards and on interrupt via an EXIT trap.

**13 considered, 12 caught, 1 equivalent, 0 escaped, 0 invalid, 0 compile-fail.**

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
| M12 | `proof_core` | `busy` drops during the accumulate cycle | **CAUGHT** (5/20) |
| M13 | `proof_core` | every byte treated as a weight | **CAUGHT** (18/20) |

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

- **Mode B is not implemented.** Only Mode A exists. The `MODE` pin (`uio[3]`)
  is declared in the pinout, tied into `_unused`, and does nothing. `IS_WEIGHT`
  *is* implemented and tested.
- **Monotonicity.** Not started. It depends on trained weights, and the sign
  structure of `W1`'s carbohydrate column has not been inspected. This is the
  project's headline property and it is currently unverified.
- **The float golden model.** `golden_quant.py` exists and the RTL matches it
  bit-exactly. `golden_float.py` does not exist yet, so the quantisation error
  bound is entirely unmeasured.
- **Timing / setup / hold.** Gate-level simulation compiles with
  `-DFUNCTIONAL -DSIM` and **no SDF back-annotation**. It catches synthesis
  differences, X-propagation and missing resets; it says nothing about setup or
  hold. STA reported +11.07 ns setup and +0.129 ns hold worst slack at a 20 ns
  period on the *skeleton* — that figure predates the current datapath and has
  not been re-measured.
- **Gate-level sim of the current design.** The last GL run was against the
  bring-up skeleton. Mode A has not been through GL yet.
- **CDC.** Single clock domain by construction, so nothing to check.
- **`mac_serial` for `DW ≠ 8`.** Parameterised, but only `DW = 8` is verified,
  and the M2 equivalence argument is specific to that width.
- **Host-side quantisation.** The chip trusts the shift byte and the INT8
  scaling the host chose. Nothing on-chip validates that a shift is sensible,
  and a shift ≥ 24 simply replicates the sign bit. Well-defined, untested.
- **Coverage model.** No named-bin coverage model exists yet; the mutation
  score is currently carrying the whole argument.
