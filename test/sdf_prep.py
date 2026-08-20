#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Kush Shah
# SPDX-License-Identifier: Apache-2.0
"""
Rewrite an OpenSTA post-route SDF into the subset Icarus Verilog can annotate.

Icarus is the only simulator in this flow, and its SDF reader rejects several
constructs OpenSTA emits. Each rejection is *fatal to the rest of the file* --
after a handful of syntax errors the reader gives up and abandons every CELL
that follows -- so an unfiltered file does not degrade gracefully. It annotates
the first third of the design, leaves the rest at zero delay, and the
simulation still runs green. That is worse than not annotating at all, because
the result looks like a timing run and is not one. (Measured, on this design's
own signoff SDF: unfiltered, 3198 SDF diagnostics and 42 of 43 tests failing;
filtered, 0 diagnostics and 43 of 43 passing.)

What this rewrites, and why each is safe:

1. ``(VOLTAGE 1.080::1.080)`` / ``(TEMPERATURE 125.000::125.000)``
   An SDF triplet with an empty middle slot. Icarus reports "Chosen value not
   defined". Header metadata only -- no delay depends on it -- so the empty
   slot is filled from its neighbour.

2. ``(COND <expr> (IOPATH ...))`` -- state-dependent delay.
   OpenSTA emits two spellings, ``(COND (a == 1'b0 && b == 1'b1) ...)`` and
   the unparenthesised ``(COND a == 1'b0 & b == 1'b1 ...)``. Icarus rejects
   the second outright, which is what aborts the file, and applies neither.
   Every conditional pin pair in this design is also covered by an
   unconditional IOPATH in the same cell (checked: 512 cells carry COND, none
   has a COND pair without a bare pair), but the bare value is *not* always
   the slowest -- 132 of 1650 COND variants exceed it. So conditions are
   collapsed rather than dropped: each pin pair gets one unconditional IOPATH
   holding the elementwise maximum over the bare value and every COND variant.
   Pessimistic by construction, which is the safe direction, and counted in
   the report so the size of the approximation stays visible.

3. ``(INTERCONNECT <design>.L_LO <port>)`` -- tie-cell pseudo-paths.
   The source is not an instance in the netlist, so Icarus reports "Could not
   find intermodpath". These carry 0.000 delay, so dropping them removes
   nothing -- but a *non-zero* one would be a real loss, so that case is
   counted separately and reported on stderr.

4. ``(TIMINGCHECK ...)`` -- setup, hold, recovery, removal, width.
   **Icarus does not implement timing checks, in any version.** It says so
   itself, at compile time ("Timing checks are not supported. Delayed
   reference and data signals become copies of the original reference and data
   signals.") and again at annotation time ("SDF WARNING: TIMINGCHECK not
   supported."). Keeping the section produces one warning per cell and changes
   nothing, so it is dropped by default and the count is reported.
   ``--keep-timingchecks`` retains it, which is the flag to reach for if a
   future Icarus ever implements them.

   **Nothing downstream of this script checks setup or hold.** Back-annotation
   here buys propagation delay, not timing verification; setup and hold still
   rest on STA alone. Say so wherever this run is cited.

Everything else -- ordinary IOPATH and INTERCONNECT delays, the overwhelming
majority -- passes through unchanged. No value is scaled: Icarus 13 and 14
apply SDF delays in nanoseconds at the cell models' own 10 ps precision.
(Icarus 12 does not. It rounds every value to a whole time unit, turning every
sub-nanosecond cell delay into zero, and its sg13g2 flops never leave X
besides, because it does not drive the models' `delayed_*` nets. Do not use 12
for this -- the local `Dev Tools\\iverilog` is 12.0; oss-cad-suite is 14.0 and
Tiny Tapeout's CI installs 13.0.)

Usage:
    python sdf_prep.py IN.sdf OUT.sdf [--json REPORT.json] [--keep-timingchecks]
    python sdf_prep.py --selftest
"""

import argparse
import json
import os
import re
import sys

CELLTYPE_RE = re.compile(r'\(CELLTYPE\s+"([^"]+)"\)')
IOPATH_HEAD_RE = re.compile(r"\(IOPATH\s+(\S+)\s+(\S+)\s+")
# innermost parenthesised groups only, so a trailing structural ")" never matches
TUPLE_RE = re.compile(r"\(([^()]*)\)")
EMPTY_MID_RE = re.compile(r"(?<![\w.])(-?[\d.]+)::(-?[\d.]+)(?![\w.])")
TIE_INTERCONNECT_RE = re.compile(r"\(INTERCONNECT\s+\S*\.L_(?:LO|HI)\s")


def fill_empty_slots(text):
    """``1.080::1.080`` -> ``1.080:1.080:1.080``. Header metadata only."""
    return EMPTY_MID_RE.sub(lambda m: "%s:%s:%s" % (m.group(1), m.group(1), m.group(2)), text)


def parse_tuples(rest):
    """``(a:b:c) ()`` -> ``[[a, b, c], None]``.

    An empty tuple is preserved as None. OpenSTA writes those for a direction
    that does not exist on a path, and Icarus accepts them, so they have to
    round-trip rather than be invented or dropped.
    """
    out = []
    for m in TUPLE_RE.finditer(rest):
        body = m.group(1).strip()
        if body == "":
            out.append(None)
            continue
        slots = []
        for s in body.split(":"):
            s = s.strip()
            slots.append(None if s == "" else float(s))
        out.append(slots)
    return out


def format_tuples(tuples):
    parts = []
    for t in tuples:
        if t is None:
            parts.append("()")
        else:
            parts.append("(" + ":".join("" if v is None else ("%.3f" % v) for v in t) + ")")
    return " ".join(parts)


def elementwise_max(a, b):
    """Merge two IOPATH tuple lists slot by slot, keeping the larger value."""
    if a is None:
        return b
    if b is None:
        return a
    out = []
    for i in range(max(len(a), len(b))):
        ta = a[i] if i < len(a) else None
        tb = b[i] if i < len(b) else None
        if ta is None or tb is None:
            out.append(ta if tb is None else tb)
            continue
        slots = []
        for j in range(max(len(ta), len(tb))):
            va = ta[j] if j < len(ta) else None
            vb = tb[j] if j < len(tb) else None
            if va is None or vb is None:
                slots.append(va if vb is None else vb)
            else:
                slots.append(max(va, vb))
        out.append(slots)
    return out


def convert(src_lines, keep_timingchecks=False):
    """Return (output_lines, report_dict)."""
    rep = {
        "cells": 0,
        "iopath_in": 0,
        "iopath_out": 0,
        "cond_collapsed": 0,
        "cond_raised_delay": 0,
        "interconnect_kept": 0,
        "interconnect_dropped": 0,
        "interconnect_dropped_nonzero": 0,
        "header_triplets_fixed": 0,
        "timingcheck_lines_dropped": 0,
    }

    out = []
    pending = {}      # (in_pin, out_pin) -> tuple list, for the CELL being read
    order = []        # emission order, so the output mirrors the input
    indent = "    "
    in_cond = False
    tc_depth = 0

    def flush():
        for pair in order:
            rep["iopath_out"] += 1
            out.append("%s(IOPATH %s %s %s)" % (indent, pair[0], pair[1],
                                                format_tuples(pending[pair])))
        del order[:]
        pending.clear()

    for raw in src_lines:
        line = raw.rstrip("\n").rstrip("\r")
        stripped = line.strip()

        if tc_depth > 0:
            tc_depth += line.count("(") - line.count(")")
            rep["timingcheck_lines_dropped"] += 1
            continue

        if stripped.startswith("(VOLTAGE") or stripped.startswith("(TEMPERATURE"):
            fixed = fill_empty_slots(line)
            if fixed != line:
                rep["header_triplets_fixed"] += 1
            out.append(fixed)
            continue

        if stripped.startswith("(TIMINGCHECK") and not keep_timingchecks:
            flush()
            tc_depth = line.count("(") - line.count(")")
            rep["timingcheck_lines_dropped"] += 1
            continue

        if CELLTYPE_RE.search(stripped):
            flush()
            rep["cells"] += 1
            out.append(line)
            continue

        if stripped.startswith("(COND"):
            in_cond = True
            if "(IOPATH" not in stripped:
                continue

        m = IOPATH_HEAD_RE.search(stripped)
        if m:
            rep["iopath_in"] += 1
            indent = line[: len(line) - len(line.lstrip())] or indent
            pair = (m.group(1), m.group(2))
            tuples = parse_tuples(stripped[m.end():])
            if pair in pending:
                before = pending[pair]
                merged = elementwise_max(before, tuples)
                if in_cond:
                    rep["cond_collapsed"] += 1
                    if merged != before:
                        rep["cond_raised_delay"] += 1
                pending[pair] = merged
            else:
                if in_cond:
                    rep["cond_collapsed"] += 1
                pending[pair] = tuples
                order.append(pair)
            in_cond = False
            continue

        if "(INTERCONNECT" in stripped:
            if TIE_INTERCONNECT_RE.search(stripped):
                vals = [v for t in parse_tuples(stripped) if t for v in t if v is not None]
                if any(abs(v) > 1e-9 for v in vals):
                    rep["interconnect_dropped_nonzero"] += 1
                rep["interconnect_dropped"] += 1
                continue
            rep["interconnect_kept"] += 1
            out.append(line)
            continue

        # Structural line. Emit the merged IOPATHs before the ")" that closes
        # ABSOLUTE, so the output keeps the input's nesting.
        if order and stripped.startswith(")"):
            flush()
        out.append(line)

    flush()
    return out, rep


# --------------------------------------------------------------- selftest ---
# The failure this project keeps hitting is a check that cannot go red. A
# regex that stops matching -- because a future OpenSTA spells IOPATH with
# different spacing, say -- would make this script a very fast no-op and the
# simulation would still pass, at zero delay, looking exactly like a timing
# run. So the transformations are tested against inputs with known answers,
# and `main` refuses to write an output that annotates nothing.

_SELFTEST_SDF = """(DELAYFILE
 (SDFVERSION "3.0")
 (DESIGN "t")
 (VOLTAGE 1.080::1.080)
 (TEMPERATURE 125.000::125.000)
 (TIMESCALE 1ns)
 (CELL
  (CELLTYPE "top")
  (INSTANCE)
  (DELAY
   (ABSOLUTE
    (INTERCONNECT a u1.A (0.010:0.010:0.010) (0.010:0.010:0.010))
    (INTERCONNECT top.L_LO oe[0] (0.000:0.000:0.000))
   )
  )
 )
 (CELL
  (CELLTYPE "cell_x")
  (INSTANCE u1)
  (DELAY
   (ABSOLUTE
    (IOPATH A Y (0.100:0.100:0.100) (0.200:0.200:0.200))
    (IOPATH R Y () (0.000:0.000:0.000))
    (COND A == 1'b0 & B == 1'b1
     (IOPATH A Y (0.300:0.300:0.300) (0.150:0.150:0.150)))
   )
  )
  (TIMINGCHECK
   (SETUP (posedge D) (posedge CLK) (0.195:0.196:0.197))
   (WIDTH (posedge CLK) (0.164:0.164:0.164))
  )
 )
)
"""


def _selftest():
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        print("  %-46s %s" % (name, "ok" if cond else "FAIL " + detail))
        if not cond:
            ok = False

    lines = _SELFTEST_SDF.splitlines(True)
    out, rep = convert(lines)
    text = "\n".join(out)

    check("cells counted", rep["cells"] == 2, str(rep["cells"]))
    check("IOPATHs read", rep["iopath_in"] == 3, str(rep["iopath_in"]))
    check("IOPATHs written (COND merged away)", rep["iopath_out"] == 2, str(rep["iopath_out"]))
    check("COND collapsed", rep["cond_collapsed"] == 1, str(rep["cond_collapsed"]))
    check("COND raised the delay", rep["cond_raised_delay"] == 1, str(rep["cond_raised_delay"]))
    # rise 0.100 vs COND 0.300 -> 0.300; fall 0.200 vs COND 0.150 -> 0.200.
    # Taking the max per slot, not per arc, is the whole point: a COND that is
    # slower rising and faster falling must not drag the fall time down.
    check("elementwise max, not whole-tuple",
          "(IOPATH A Y (0.300:0.300:0.300) (0.200:0.200:0.200))" in text,
          [l for l in out if "IOPATH A Y" in l])
    check("empty delay tuple round-trips",
          "(IOPATH R Y () (0.000:0.000:0.000))" in text,
          [l for l in out if "IOPATH R Y" in l])
    check("header empty slot filled", "(VOLTAGE 1.080:1.080:1.080)" in text)
    check("temperature slot filled", "(TEMPERATURE 125.000:125.000:125.000)" in text)
    check("real interconnect kept", rep["interconnect_kept"] == 1, str(rep["interconnect_kept"]))
    check("tie interconnect dropped", rep["interconnect_dropped"] == 1)
    check("no non-zero interconnect dropped", rep["interconnect_dropped_nonzero"] == 0)
    check("TIMINGCHECK gone by default", "TIMINGCHECK" not in text)
    check("no COND survives", "(COND" not in text)

    _, rep_keep = convert(lines, keep_timingchecks=True)
    out_keep, _ = convert(lines, keep_timingchecks=True)
    check("--keep-timingchecks retains it", "TIMINGCHECK" in "\n".join(out_keep))
    check("...and drops nothing", rep_keep["timingcheck_lines_dropped"] == 0)

    # Injected fault: a non-zero tie interconnect must be *counted*, never
    # dropped in silence, because that one really would lose delay.
    hurt = _SELFTEST_SDF.replace("(INTERCONNECT top.L_LO oe[0] (0.000:0.000:0.000))",
                                 "(INTERCONNECT top.L_LO oe[0] (0.070:0.070:0.070))")
    _, rep_hurt = convert(hurt.splitlines(True))
    check("non-zero tie interconnect is flagged",
          rep_hurt["interconnect_dropped_nonzero"] == 1)

    # Injected fault: a file the IOPATH pattern no longer matches must be
    # visible as iopath_in == 0, which is what main() refuses to write.
    blind = _SELFTEST_SDF.replace("(IOPATH", "(IOPATH_RENAMED")
    _, rep_blind = convert(blind.splitlines(True))
    check("a pattern that stops matching shows as zero",
          rep_blind["iopath_in"] == 0 and "IOPATH" in blind)

    print("  selftest:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", nargs="?")
    ap.add_argument("dst", nargs="?")
    ap.add_argument("--json", metavar="PATH", help="write the conversion report as JSON")
    ap.add_argument("--keep-timingchecks", action="store_true",
                    help="keep (TIMINGCHECK ...) sections; Icarus ignores them and warns")
    ap.add_argument("--selftest", action="store_true",
                    help="check the transformations against known answers and exit")
    args = ap.parse_args()

    if args.selftest:
        sys.exit(_selftest())
    if not args.src or not args.dst:
        ap.error("need IN.sdf and OUT.sdf (or --selftest)")

    with open(args.src, encoding="utf-8") as fh:
        src_lines = fh.readlines()
    out_lines, rep = convert(src_lines, keep_timingchecks=args.keep_timingchecks)

    for k in sorted(rep):
        print("  %-28s %s" % (k, rep[k]))

    # Refuse to produce a file that would annotate nothing. Without this the
    # failure mode is a green simulation at zero delay -- indistinguishable,
    # from the outside, from the run this is supposed to replace.
    if rep["iopath_out"] == 0:
        sys.exit("no IOPATH delays survived conversion -- refusing to write %s" % args.dst)
    if rep["interconnect_dropped_nonzero"]:
        print("WARNING: dropped %d interconnect entries carrying NON-ZERO delay"
              % rep["interconnect_dropped_nonzero"], file=sys.stderr)

    # temp + os.replace with the encoding pinned: a truncating write destroyed
    # a file on this project once (see HANDOFF).
    tmp = args.dst + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(out_lines) + "\n")
    os.replace(tmp, args.dst)

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(rep, fh, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
