# Bug log — Proof

Owner: design *and* verification. On the prior CNN accelerator
(`kush1434/cnnaccelerator`) the RTL was owned by someone else and this log
covered only the testbench; here both sides are mine, so an RTL defect found
below is a defect I wrote.

Environment: Icarus Verilog 12.0, cocotb 2.0.1, Python 3.13 (host) /
Python 3.8 (`.venv-legacy`, CGMacros baseline only).

```bash
cd test
python run.py                    # whole design, RTL
python run.py --unit mac_serial  # submodule unit test
python run.py --gates            # post-layout netlist (needs PDK_ROOT)
```

---

## RTL defects

| # | Date | Symptom | Test that found it | Root cause | Fix |
|---|------|---------|--------------------|------------|-----|
| — | — | *None found to date.* | — | — | — |

As of 2026-08-14 the design is a flow bring-up skeleton plus `mac_serial`.
`mac_serial` is verified **exhaustively** — all 65,536 signed 8×8 input pairs,
not a sample — so for that module there is no coverage argument left to make.
The claim is still only worth what the environment is worth, which is why the
mutation results below matter more than the green log.

---

## Testbench / methodology defects

Found in the verification environment during bring-up. TB-3 is the one that
mattered: it would have reported a false pass indefinitely.

| # | Date | Symptom | Test that found it | Root cause | Fix |
|---|------|---------|--------------------|------------|-----|
| TB-1 | 2026-08-14 | Test 1 passed, then all six remaining tests failed with `SimFailure: Simulator shut down prematurely` | first multi-test run of the bring-up suite | cocotb 2.0 **cancels every task a test started when that test ends**, which 1.x did not do. The clock was started once and shared, so from test 2 onward there was no clock at all; the simulator ran out of events and exited. The reported error names the symptom, not the cause | `setup()` starts a fresh `Clock` per test. Documented in `test.py` because the error message actively misleads |
| TB-2 | 2026-08-14 | `test_busy_then_done` failed asserting `BUSY` immediately after a byte was pushed, though the RTL was correct | bring-up suite, after TB-1 was fixed | `ClockCycles` resumes **at** the rising edge, before the NBA region has run, so reading a registered output straight after it returns hands back the pre-edge value. Combinational outputs survive this by accident, registered ones do not — so the bug is invisible until the first flop is sampled | Added `settle()` (a 1 ns `Timer`) and stepped past the edge before every read in `reset`, `push` and `read_acc` |
| TB-3 | 2026-08-14 | `test/run.py` exited **0 with 6 of 6 tests failing**. Any CI step built on it would have been structurally incapable of going red | deliberately running a known-broken design through the runner before trusting it in CI | cocotb's `Runner.test()` reports success regardless of test outcome. The Tiny Tapeout template Makefile knows this and works around it (`# make will return success even if the test fails` → `! grep failure results.xml`); `run.py` reimplemented the runner and silently inherited the same trap | Added `check_results()`, which parses the JUnit XML and exits non-zero on any `failure`/`error`. It also **fails on an empty testcase list**, because a test module that fails to import produces zero tests and would otherwise read as a pass. Verified in both directions: good design → exit 0, mutated design → exit 1 |

TB-3 is the same class of defect as TB-4 on the CNN accelerator — a guard that
could never fire — and it was found the same way: by checking that the check
works before trusting it, rather than by waiting for it to miss something.

---

## Mutation testing results

A green testbench proves nothing until it can go red. Each mutant is injected
into the RTL, the full suite is run, and the RTL is restored afterwards.

Comparison uses `diff -q --strip-trailing-cr`, **not** `cmp -s`. This repository
checks out CRLF and `sed` writes LF, so a plain comparison reports every file as
changed and the did-not-apply guard becomes structurally dead. That was TB-4 on
the previous project; it is not being reintroduced here.

### `mac_serial` — 3 considered, 2 caught, 1 equivalent, 0 escaped

| Mutant | Injected bug | Status | Detail |
|---|---|---|---|
| M1 | drop the two's-complement sign correction on the final step | **CAUGHT** | 3 of 6 tests fail |
| M2 | high accumulator one bit narrower (`[DW-1:0]` instead of `[DW:0]`) | **EQUIVALENT** | 0 of 6 fail — see below |
| M3 | terminate one step early (`step == DW-2`) | **CAUGHT** | 6 of 6 tests fail |

### An equivalent mutant, and the design observation that came out of it

M2 escapes, and because the test is exhaustive over the entire input space this
is **not** a coverage gap — it is proof the mutant is equivalent. Narrowing
`acc_hi` from `DW+1` to `DW` bits cannot change any of the 65,536 products.

The reason: `acc_hi` is only ever written as `{sum[DW], sum[DW:1]}`, an
arithmetic right shift. The partial sum is bounded such that `sum[DW]` always
equals `sum[DW-1]`, so the top bit is a duplicated sign bit that no later step
reads. **The 9th bit of `acc_hi` is dead logic, worth one flop (48.99 µm²).**

It is being kept anyway, deliberately. The equivalence proof holds for `DW = 8`
— the width the exhaustive test covers — and `mac_serial` is parameterised. The
flop is cheap insurance at 11.9% utilisation; if area ever gets tight it is a
one-character reclaim with a known justification. Recorded here rather than
silently fixed or silently ignored.

This mirrors the `patch_i_d` finding on the CNN accelerator: reasoning about
*why* a mutant correctly escaped is what surfaces redundant logic.

---

## Areas deliberately not covered

Recorded so that nobody reads "all tests passed" as "everything is verified."

- **Timing / setup / hold.** The gate-level simulation compiles with
  `-DFUNCTIONAL -DSIM` and **no SDF back-annotation**. It catches synthesis
  optimisation differences, X-propagation and missing resets. It says nothing
  about setup or hold. Static timing analysis reported +11.07 ns setup and
  +0.129 ns hold worst slack at a 20 ns period, but that is STA's claim, not
  something this testbench verified.
- **CDC.** Single clock domain by construction (`RUN_CTS: 1`, one `clk` pin),
  so there is nothing to check.
- **`mac_serial` for `DW ≠ 8`.** The module is parameterised; only `DW = 8` is
  exhaustively verified, and the M2 equivalence argument above is specific to
  that width.
- **The full host protocol.** Only the skeleton's subset (`VALID`, `LAST`,
  `RD_SEL`) is exercised. `IS_WEIGHT` and `MODE` are declared in the pinout and
  tied into `_unused`; they are not yet implemented, so they are not yet tested.
- **Accumulator overflow in Mode A with unbounded ingredient counts.** The
  accumulator saturates and sets a sticky flag, and both are tested at the
  16-bit rails; the width proof for the final Mode B datapath is not written
  yet.
- **Monotonicity.** Not started. It depends on trained weights, and the sign
  structure of `W1`'s carbohydrate column has not been inspected.
