#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Kush Shah
# SPDX-License-Identifier: Apache-2.0
"""Cross-document number consistency: catch stale figures before a reader does.

WHY THIS EXISTS
---------------
The recurring failure of this project is not wrong arithmetic. It is a number
that was correct, got superseded, and stayed behind in a document that quoted
it. It has happened at least five times:

  * the monotonicity cost, quoted as -0.036 long after the CGMacros
    plausibility filter made it -0.007;
  * the XGBoost row, at -0.064 in three documents with two different intervals
    after a re-run gave -0.059;
  * the within-participant Spearman, +0.403 after it became +0.382;
  * the inference latency, 914 cycles after the deployed figure turned out to
    be 896 -- 914 was a real measurement of the testbench path;
  * the parameter count, "83 parameters" for what is 83 bytes and 65 learned
    parameters.

Every one was caught by a human re-reading, which does not scale and did not
catch them promptly. This checks mechanically instead.

HOW IT WORKS
------------
Two lists. CANONICAL is the current value of each load-bearing figure, with the
script that produces it. RETIRED is every value those figures used to have. A
document containing a retired value is flagged wherever it appears, including
in the paper.

Deliberately NOT a re-run of the analyses: those take tens of minutes and live
in `.venv-legacy`. This is the cheap gate that runs in a second and answers a
narrower question -- do the documents agree with each other and with the last
recorded run? Regenerating is still the only way to know a canonical value is
itself current, which is what `paper/NUMBERS-CHECK.md` is for.

Lines marked as corrections are exempt, since a correction note has to quote
the old value to be useful. The marker is a leading warning sign or the words
"previously", "used to", "was stale", "corrected from".
"""

import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

DOCS = [
    "RESULTS.md",
    "VERIFICATION.md",
    "BUGS.md",
    "HANDOFF.md",
    "docs/info.md",
    "paper/proof_bibm2026.tex",
    # Scripts that quote a RESULTS.md figure back at the reader are
    # documents too. Added 2026-08-24, after model/ablation.py was found
    # printing the pre-rerun Spearman block and a claim RESULTS.md 5
    # explicitly warns against -- neither visible to a gate that only
    # reads .md and .tex. The paper invites reviewers to run these.
    "model/ablation.py",
    "model/clinical.py",
    "model/sign_condition.py",
    "model/width_sweep.py",
    "test/monotonicity.py",
    "test/test_cycles.py",
]

# label -> (current value as it appears in prose, producing script)
CANONICAL = [
    ("monotonicity cost",      "-0.007",  "model/train.py --cv 5"),
    ("XGBoost delta",          "-0.059",  "model/train.py --cv 5"),
    ("XGBoost mean R2",        "0.170",   "model/train.py --cv 5"),
    ("within-person Spearman", "0.382",   "model/clinical.py"),
    ("participants ranked",    "41 of 44", "model/clinical.py"),
    ("deployed latency",       "896",     "test/run.py --module test_cycles"),
    ("energy per inference",   "32.0",    "derived from latency x power"),
    ("flip-flops",             "168",     "lint.sh / gds metrics"),
    ("utilisation",            "83.53",   "gds metrics.csv"),
    ("standard cells",         "1443",    "gds metrics.csv"),
    ("top-level tests",        "43",      "test/run.py"),
    ("clock-to-output",        "1975",    "test/run.py --gates --sdf ... --module test_sdf"),
    ("cohort mean iAUC",       "3150",    "model/clinical.py"),
    ("guard rejections correct", "44 refits", "model/false_reject.py"),
]

# (retired value, its replacement, why). The replacement matters: a line
# carrying BOTH is a deliberate before/after comparison -- the guard's cost
# table, or a correction quoting what it replaced -- not a stale figure.
RETIRED = [
    ("-0.036",  "-0.007", "monotonicity cost, pre-plausibility-filter"),
    ("-0.064",  "-0.059", "XGBoost delta, pre-rerun"),
    ("0.166",   "0.170",  "XGBoost mean R2, pre-rerun"),
    ("0.403",   "0.382",  "Spearman, pre-rerun"),
    ("40 of 44", "41 of 44", "participants ranked, pre-rerun"),
    ("914",     "896",    "latency including per-neuron read-back"),
    ("32.7",    "32.0",   "energy from the 914-cycle figure"),
    ("35.8",    "35.1",   "annual energy from the 914-cycle figure"),
    ("76.35",   "83.53",  "utilisation before the guard"),
    ("1330",    "1443",   "standard cells before the guard"),
    ("83 parameters", "65 learned", "byte count mistaken for parameters"),
    # Not a superseded measurement but a superseded *framing*: 4.9x and 8.6x
    # are 418->85 and 1554->180, which compare original scales on uncleaned
    # data against selected scales on cleaned data. That credits the scale
    # search with the data-cleaning gain and roughly doubles it. The search
    # alone is 2.0x on the mean and 6.5x on the tail.
    ("4.9", "2.0", "quantisation mean gain, bundled with data cleaning"),
    ("8.6", "6.5", "quantisation tail gain, bundled with data cleaning"),
    # Both found 2026-08-19, both had survived in exactly one document while
    # every other one had moved on. Phrased tightly rather than as bare
    # numbers: "41" collides with "41 of 44 participants", and "four" appears
    # legitimately in "four independent angles, and a fifth on 2026-08-18".
    ("41, all bit-exact", "43, bit-exact", "top-level test count"),
    ("four literature searches", "five literature searches",
     "search count, before the 2026-08-18 SoftSNN pass"),
    # Fourth drift of the same count, found 2026-08-24: RESULTS 9.1 and
    # HANDOFF said six after the 2026-08-20 pass while 9, 10 and the paper
    # still said five.
    ("five literature searches", "six literature searches",
     "search count, before the 2026-08-20 pass"),
    # The gate-level run is no longer timing-free. Any document still saying so
    # is behind RESULTS.md 6.1.
    ("no SDF back-annotation", "post-route SDF",
     "gate-level sim was functional-only until 2026-08-19"),
    # width_sweep.py printed these back as what RESULTS.md 1.2 records,
    # while 1.2 had recorded the measured values since 2026-08-15. Paired
    # with their lower bound so a bare 0.016 elsewhere cannot trip them.
    ("+0.016 [-0.023", "+0.012 [-0.005", "width-4 delta vs 8, pre-rerun"),
    ("+0.029 [-0.010", "+0.004 [-0.027", "width-6 delta vs 8, pre-rerun"),
    # Not a superseded measurement but a value that was never measured at
    # all: clinical.py printed ~2900 as a literal inside an f-string whose
    # every other value was computed, and the paper quoted it. Computed,
    # it is 3150. Found 2026-08-24 by grepping print() strings for digits
    # that are not inside a {} interpolation.
    ("2900", "3150", "cohort mean iAUC, hardcoded and never computed"),
]

CORRECTION_MARKERS = (
    "⚠", "previously", "used to", "was stale", "were stale",
    "corrected from", "pre-rerun", "before the guard", "earlier table",
    "earlier drafts", "no longer", "replaced an earlier", "predated",
    "old value", "without guard", "retired", "pre-plausibility", "until 2026",
    "entry said", "this bullet", "wrong thing", "sheet said", "drift",
)


# The documents are written with typographic minus (U+2212) and en dashes,
# while the values here are ASCII. Comparing them raw finds nothing at all --
# the first run of this script reported the XGBoost delta as appearing in no
# document, which was the script's bug and not the documents'.
DASHES = {0x2212: "-", 0x2013: "-", 0x2014: "-", 0x2010: "-", 0x2011: "-"}


def norm(s):
    return s.translate(DASHES)


def safe(s):
    """Windows consoles default to cp1252, which cannot encode U+2212."""
    return s.encode(sys.stdout.encoding or "utf-8", "replace").decode(
        sys.stdout.encoding or "utf-8", "replace")


# A correction note often spans several lines, so the marker may sit on a
# neighbouring line rather than the one carrying the number.
CONTEXT = 3


def in_correction(lines, idx):
    lo = max(0, idx - CONTEXT)
    hi = min(len(lines), idx + CONTEXT + 1)
    window = " ".join(lines[lo:hi]).lower()
    return any(m.lower() in window for m in CORRECTION_MARKERS)


def matches(line, value):
    """Is `value` quoted on this line, with or without its sign?

    Substring matching was not enough. The paper phrases the baseline
    comparison as "scores $0.059$ below the unconstrained network" rather than
    as a negative number, so a stale "$0.064$ below" contained no minus sign and
    slipped through a check written to look for "-0.064". Verified by injecting
    exactly that and watching the gate pass.

    So: compare on the magnitude, and require digit boundaries so that 0.064
    does not match inside 10.0642.
    """
    v = norm(value).lstrip("-").strip()
    if not re.match(r"^[\d.]+$", v):
        return norm(value) in line          # "83 parameters", "40 of 44"
    return re.search(r"(?<![\d.])" + re.escape(v) + r"(?![\d])", line) is not None


def main():
    problems, seen = [], {}

    for rel in DOCS:
        path = os.path.join(HERE, rel)
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as f:
            lines = [norm(l) for l in f.read().split("\n")]
        for i, line in enumerate(lines):
            if in_correction(lines, i):
                continue
            for val, replacement, why in RETIRED:
                if matches(line, replacement):
                    continue          # before/after comparison, not staleness
                if matches(line, val):
                    problems.append(
                        "%s:%d  retired value %r (%s)\n      %s"
                        % (rel, i + 1, val, why, line.strip()[:78]))
        joined = "\n".join(lines)
        for label, val, _ in CANONICAL:
            if norm(val) in joined:
                seen.setdefault(label, []).append(rel)

    print("cross-document number check")
    print("  %d documents, %d canonical figures, %d retired values"
          % (len(DOCS), len(CANONICAL), len(RETIRED)))
    print()
    print("  canonical figures, and where they appear:")
    for label, val, script in CANONICAL:
        where = seen.get(label, [])
        print("    %-24s %-9s %s" % (label, val, ", ".join(where) or "-- nowhere"))
    print()
    print("  produced by:")
    for label, _, script in CANONICAL:
        print("    %-24s %s" % (label, script))
    print()

    if problems:
        print("%d stale value(s) outside a correction note:" % len(problems))
        for p in problems:
            print(safe("  - %s" % p))
        return 1
    print("no stale values found")
    return 0


if __name__ == "__main__":
    sys.exit(main())
