#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Kush Shah
# SPDX-License-Identifier: Apache-2.0
"""Re-run the analyses and check RESULTS.md still says what they produce.

THE GAP THIS FILLS
------------------
`check_numbers.py` catches a document quoting a figure that has been superseded.
It cannot catch a figure that is superseded and has not been noticed yet --
that needs re-running the pipeline, which CI cannot do because CGMacros is not
in the repository (CC BY-NC-SA against an Apache-2.0 repo).

So the loop is closed for *propagation* and open for *drift*. This closes drift.
It runs the model scripts, pulls each headline number out of their output, pulls
the same number out of RESULTS.md, and fails if they disagree.

ONE SOURCE OF TRUTH
-------------------
Expected values are read **out of RESULTS.md**, never hard-coded here. Writing
them down in a second place is precisely the failure this project keeps having:
the monotonicity cost, the XGBoost row, the Spearman block and the latency were
each correct somewhere and stale somewhere else. If RESULTS.md is edited, this
checks the new value against the pipeline automatically.

A TRAP WORTH KNOWING
--------------------
`train.py` prints its cross-validation block TWICE -- once for all meals, once
for breakfast only -- with identical wording. RESULTS.md quotes the all-meals
numbers, which come first. Every pattern here takes the FIRST match for that
reason, and taking the last would silently compare against the wrong scope and
report agreement that does not exist.

RUNNING IT
----------
Slow: the scripts take tens of minutes between them, and need `.venv-legacy`
(Python 3.8) plus the CGMacros CSVs.

    python verify_numbers.py                 # run everything, then check
    python verify_numbers.py --no-run        # re-check the cached output
    python verify_numbers.py --only clinical # one script
    python verify_numbers.py --list          # what it knows how to check
"""

import argparse
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "RESULTS.md")
CACHE = os.path.join(HERE, "test", "sim_build", "verify_cache")
PY = os.path.join(HERE, ".venv-legacy", "Scripts", "python.exe")
if not os.path.exists(PY):                      # non-Windows checkouts
    PY = os.path.join(HERE, ".venv-legacy", "bin", "python")

# name -> (working directory, argv after the interpreter)
SCRIPTS = {
    "personalise":    ("model", ["personalise.py"]),
    "train":          ("model", ["train.py", "--cv", "5"]),
    "clinical":       ("model", ["clinical.py"]),
    "sign_condition": ("model", ["sign_condition.py"]),
    "width_sweep":    ("model", ["width_sweep.py"]),
    "ablation":       ("model", ["ablation.py"]),
}

# (label, script, pattern over that script's stdout, pattern over RESULTS.md)
#
# Both patterns capture the same number of groups, compared pairwise. Dashes are
# normalised first, so RESULTS.md's U+2212 minus matches the scripts' ASCII one.
CHECKS = [
    ("personalisation R2 @ k=8", "personalise",
     r"^\s*8\s+44\s+([+-][\d.]+)",
     r"\|\s*\*\*8\*\*\s*\|\s*\*\*([+-][\d.]+)\*\*"),

    ("paired dMAE @ k=8", "personalise",
     r"k=\s*8\s+delta MAE\s+([+-]?\d+)\s+\[95% CI\s+([+-]?\d+),\s+([+-]?\d+)\]",
     r"\|\s*\*\*8\*\*\s*\|\s*\*\*([+-]?\d+)\*\*\s*\|\s*\*\*\[([+-]?\d+),\s*([+-]?\d+)\]\*\*"),

    ("monotonicity cost, carbs up", "train",
     r"carbs up\s+vs unconstrained: delta\s+([+-][\d.]+)\s+"
     r"\[95% CI\s+([+-][\d.]+),\s*([+-][\d.]+)\]",
     r"\|\s*carbs . vs unconstrained\s*\|\s*([+-][\d.]+)\s*\|\s*"
     r"\[([+-][\d.]+),\s*([+-][\d.]+)\]"),

    ("XGBoost delta", "train",
     r"depth-1 XGBoost\s+vs unconstrained: delta\s+([+-][\d.]+)\s+"
     r"\[95% CI\s+([+-][\d.]+),\s*([+-][\d.]+)\]",
     r"\|\s*depth-1 XGBoost vs unconstrained\s*\|\s*([+-][\d.]+)\s*\|\s*"
     r"\[([+-][\d.]+),\s*([+-][\d.]+)\]"),

    ("out-of-fold R2", "clinical",
     r"overall out-of-fold R2\s+([+-][\d.]+)",
     r"\|\s*\*\*overall\*\*\s*\|\s*45\s*\|\s*1308\s*\|\s*\*\*([+-][\d.]+)\*\*"),

    ("within-person Spearman", "clinical",
     r"Spearman rho\s+median\s+([+-][\d.]+)",
     r"median Spearman \*\*. = ([+-][\d.]+)\*\*"),

    ("participants ranked positive", "clinical",
     r"positive rho in\s+(\d+)/(\d+)\s+participants",
     r"\*\*positive in (\d+) of (\d+) participants\*\*"),

    ("quantisation mean / p95", "clinical",
     r"\|float - INT8\|\s+mean\s+([\d.]+)\s+p95\s+([\d.]+)",
     r"\|\s*\*\*selected \(5,5,5,5,1\)\*\*\s*\|\s*\*\*([\d.]+)\*\*\s*\|\s*\*\*([\d.]+)\*\*"),

    ("sign condition violation rate", "sign_condition",
     r"^\s*20\s+(\d+)\s+(\d+)\s+100\.0%",
     r"\|\s*per-person weight sets checked\s*\|\s*(\d+)\s*\|[\s\S]*?"
     r"\|\s*sets violating the sign condition\s*\|\s*\*\*(\d+) \(100 %\)\*\*"),

    ("hidden width 4 vs 8, shipped", "width_sweep",
     r"\[carbs up \(as shipped\)\][\s\S]*?width 4\s+vs 8: delta R2\s+([+-][\d.]+)\s+"
     r"\[95% CI\s+([+-][\d.]+),\s*([+-][\d.]+)\]",
     r"\|\s*4\s*\|\s*([+-][\d.]+)\s*\|\s*\[([+-][\d.]+),\s*([+-][\d.]+)\]"),

    ("ablation, all 5 bio features", "ablation",
     r"\+ all 5 bio\s+\d+\s+\d+\s+([+-][\d.]+)",
     r"\|\s*\+ all 5 bio\s*\|\s*\*\*([+-][\d.]+)\*\*"),
]

DASHES = {ord(c): "-" for c in "‐‑‒–—―−"}


def norm(s):
    return s.translate(DASHES)


def same(produced, claimed):
    """Equal once `produced` is rounded to the precision `claimed` is quoted at.

    RESULTS.md rounds -- the quantisation table says 85 where clinical.py prints
    85.1 -- so comparing the raw strings would report a mismatch that is only a
    difference in how many digits were written down.
    """
    try:
        p, c = float(produced), float(claimed)
    except ValueError:
        return norm(produced).strip() == norm(claimed).strip()
    dp = len(claimed.split(".")[1]) if "." in claimed else 0
    return round(p, dp) == round(c, dp)


def run_script(name, cache_dir, rerun):
    """Run one script, cache its stdout, return it."""
    out_path = os.path.join(cache_dir, name + ".txt")
    if not rerun and os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as f:
            return f.read(), "cached"

    cwd, argv = SCRIPTS[name]
    if not os.path.exists(PY):
        return None, "no .venv-legacy interpreter at %s" % PY
    print("    running %s ..." % " ".join(argv), end="", flush=True)
    try:
        res = subprocess.run([PY] + argv, cwd=os.path.join(HERE, cwd),
                             capture_output=True, text=True, timeout=3600)
    except subprocess.TimeoutExpired:
        print(" TIMED OUT")
        return None, "timed out after an hour"
    out = res.stdout + res.stderr
    if res.returncode != 0:
        print(" exit %d" % res.returncode)
        return out, "exited %d" % res.returncode
    print(" ok")
    tmp = out_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(out)
    os.replace(tmp, out_path)
    return out, "ran"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-run", action="store_true",
                    help="use cached output instead of re-running")
    ap.add_argument("--only", metavar="SCRIPT",
                    help="one script: " + ", ".join(sorted(SCRIPTS)))
    ap.add_argument("--list", action="store_true", help="list the checks")
    ap.add_argument("--cache", default=CACHE)
    args = ap.parse_args()

    if args.list:
        print("checks, by the script that produces them:")
        for script in sorted(SCRIPTS):
            print("  %s  (%s)" % (script, " ".join(SCRIPTS[script][1])))
            for label, s, _, _ in CHECKS:
                if s == script:
                    print("      %s" % label)
        return 0

    wanted = [c for c in CHECKS if not args.only or c[1] == args.only]
    if not wanted:
        sys.exit("no checks for %r -- try --list" % args.only)
    scripts = sorted({c[1] for c in wanted})

    os.makedirs(args.cache, exist_ok=True)
    results_md = norm(open(RESULTS, encoding="utf-8").read())

    print("re-run check: does RESULTS.md still match the pipeline?")
    print("  cache: %s" % args.cache)
    print()

    outputs, failed_scripts = {}, {}
    for name in scripts:
        out, how = run_script(name, args.cache, rerun=not args.no_run)
        if out is None or how.startswith("exited") or how.startswith("timed"):
            failed_scripts[name] = how
        if out is not None:
            outputs[name] = norm(out)

    print()
    print("  %-32s %-22s %-22s %s" % ("check", "pipeline", "RESULTS.md", ""))
    print("  " + "-" * 84)

    problems, checked = [], 0
    for label, script, out_pat, md_pat in wanted:
        if script not in outputs:
            print("  %-32s %-46s SKIPPED (%s)"
                  % (label, "", failed_scripts.get(script, "no output")))
            continue
        # FIRST match, deliberately: train.py prints its CV block twice and the
        # second one is breakfast-only.
        m_out = re.search(out_pat, outputs[script], re.M)
        m_md = re.search(md_pat, results_md, re.M)
        if not m_out or not m_md:
            where = "script output" if not m_out else "RESULTS.md"
            problems.append("%s: pattern did not match %s" % (label, where))
            print("  %-32s %-46s NO MATCH in %s" % (label, "", where))
            continue
        got, want = list(m_out.groups()), list(m_md.groups())
        ok = len(got) == len(want) and all(same(g, w) for g, w in zip(got, want))
        checked += 1
        print("  %-32s %-22s %-22s %s"
              % (label, " ".join(got), " ".join(want), "ok" if ok else "MISMATCH"))
        if not ok:
            problems.append("%s: pipeline says %s, RESULTS.md says %s"
                            % (label, " ".join(got), " ".join(want)))

    print()
    if failed_scripts:
        print("  %d script(s) did not produce output:" % len(failed_scripts))
        for k, v in failed_scripts.items():
            print("    %s -- %s" % (k, v))
        print("  (the model scripts need .venv-legacy and the CGMacros CSVs)")
        print()

    if problems:
        print("%d problem(s):" % len(problems))
        for p in problems:
            print("  - %s" % p)
        return 1
    if not checked:
        print("nothing was checked -- see above")
        return 1
    print("%d value(s) checked; RESULTS.md matches the pipeline" % checked)
    return 0


if __name__ == "__main__":
    sys.exit(main())
