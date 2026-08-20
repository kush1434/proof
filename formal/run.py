#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Kush Shah
# SPDX-License-Identifier: Apache-2.0
"""
Run the formal proofs, and check that the failing one still fails.

WHAT IS PROVED
--------------
`monotone_acc.sby` machine-checks the load-bearing lemma of the monotonicity
argument -- that a saturating accumulator is monotone in its terms -- against
`src/accumulator.v` itself rather than against a model of it. It closes by
k-induction, so a PASS is **unbounded**: for all inputs and all time, not "no
counterexample in the first N cycles".

That distinction is not decoration here. The wrapping mutant *passes* a bounded
check at the shipping widths, because the rail is roughly 256 maximum-magnitude
terms away and no reasonable BMC depth reaches it. A bounded run would have
reported a clean pass on a design with the R-4 defect in it.

WHY THERE ARE THREE
-------------------
The proof closes in about a second, which is also what a vacuous proof looks
like. So the suite is a controlled experiment, not a single check:

  monotone_acc  real accumulator, shipping widths (24/16), k-induction  -> PASS
  control       real accumulator, toy widths (8/6), bounded             -> PASS
  mutate        WRAPPING accumulator, toy widths, bounded               -> FAIL

The toy widths exist so the defect is reachable inside a bounded run and the
solver emits a concrete counterexample. `control` runs the real design at those
same widths, so the failure in `mutate` is attributable to the wrapping and not
to the narrow field. If `mutate` ever passes, the proof above is proving
nothing and this exits non-zero saying so.

The counterexample is R-4 in miniature: the accumulator handed the *larger*
term every cycle wraps past the negative rail and ends up far below the one
handed the smaller term, inverting the ordering the guarantee depends on.

WHAT IS NOT PROVED
------------------
This is the accumulator stage, not the whole inference. The composition
argument over the other stages (arithmetic shift, ReLU, clamp, the output
field) is still the hand argument in RESULTS.md 3. Proving the full streaming
datapath monotone is a 2-safety property over a 896-cycle inference and has
not been attempted.

Usage:
    python run.py            # all three
    python run.py mutate     # one by name
"""

import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent.resolve()

# name -> what it demonstrates, for the summary line
TASKS = [
    ("monotone_acc", "real design, shipping widths, k-induction (unbounded)"),
    ("control", "real design, toy widths, bounded -- isolates the cause"),
    ("mutate", "WRAPPING mutant, toy widths -- must fail"),
]


def run(name):
    """Run one .sby. Returns True if it met its own `expect`."""
    proc = subprocess.run(
        ["sby", "-f", "%s.sby" % name],
        cwd=str(HERE), capture_output=True, text=True)
    tail = [l for l in (proc.stdout + proc.stderr).splitlines() if "DONE" in l]
    verdict = tail[-1].split("DONE", 1)[1].strip() if tail else "(no verdict)"
    # sby exits 0 when the outcome matches `expect`, non-zero otherwise. That
    # is what makes `expect fail` a real check rather than a comment.
    return proc.returncode == 0, verdict, proc.stdout + proc.stderr


def main():
    wanted = sys.argv[1:] or [n for n, _ in TASKS]
    known = dict(TASKS)
    bad = [w for w in wanted if w not in known]
    if bad:
        sys.exit("unknown task(s): %s -- have %s" % (bad, list(known)))

    failures = []
    for name in wanted:
        ok, verdict, log = run(name)
        print("  %-14s %-52s %s" % (name, known[name], verdict))
        if not ok:
            failures.append((name, log))

    if failures:
        print()
        for name, log in failures:
            print("=== %s did not meet its expectation ===" % name)
            print("\n".join(log.splitlines()[-25:]))
        sys.exit("%d of %d formal task(s) failed" % (len(failures), len(wanted)))
    print("\n%d formal task(s) met their expectations" % len(wanted))


if __name__ == "__main__":
    main()
