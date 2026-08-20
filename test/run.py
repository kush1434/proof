#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Kush Shah
# SPDX-License-Identifier: Apache-2.0
"""
Windows-friendly cocotb runner.

The Tiny Tapeout template drives cocotb through test/Makefile, which needs GNU
make. There is no make on the development machine, so this uses cocotb 2.x's
built-in runner instead.

For the top-level design it mirrors the Makefile deliberately -- same sources,
same defines, same toplevel, same build directories -- so a local run and the
CI run (which does use the Makefile) stay equivalent. If you change
PROJECT_SOURCES in the Makefile, change it here too. Both must also match
`source_files` in info.yaml; a desync there is a silent RTL/GL mismatch.

    python run.py                    # RTL simulation of the whole design
    python run.py --gates            # gate-level (needs gate_level_netlist.v + PDK_ROOT)
    python run.py --gates --sdf F    # ...with delays back-annotated from F
    python run.py --unit mac_serial  # submodule unit test

--sdf takes an SDF already put through sdf_prep.py -- the raw OpenSTA file
aborts Icarus's reader part-way and leaves most of the netlist at zero delay.
It adds -gspecify (without which Icarus omits $sdf_annotate and says so once,
at compile time) and -ginterconnect (without which every net delay is
rejected). Delays only: Icarus implements no timing checks, so setup and hold
are still STA's word alone.

--unit has no Makefile equivalent: the template only knows how to build the
top level. It expects src/NAME.v, test/tb_NAME.v and test/test_NAME.py.
"""

import argparse
import os
import sys
from pathlib import Path
from xml.etree import ElementTree

from cocotb_tools.runner import get_runner

HERE = Path(__file__).parent.resolve()
SRC = HERE.parent / "src"

# Keep in sync with PROJECT_SOURCES in Makefile and source_files in info.yaml.
PROJECT_SOURCES = [
    "tt_um_kush1434_proof.v",
    "proof_core.v",
    "accumulator.v",
    "mac_serial.v",
]


def plan(args):
    """Return a dict describing one build+run."""
    if args.unit:
        name = args.unit
        return dict(
            sources=[SRC / f"{name}.v", HERE / f"tb_{name}.v"],
            defines={},
            build_dir=HERE / "sim_build" / f"unit_{name}",
            toplevel=f"tb_{name}",
            test_module=f"test_{name}",
            build_args=[],
            plusargs=[],
        )

    if args.gates:
        pdk_root = os.environ.get("PDK_ROOT")
        if not pdk_root:
            sys.exit("PDK_ROOT is not set -- required for gate-level simulation")
        netlist = HERE / "gate_level_netlist.v"
        if not netlist.exists():
            sys.exit(
                f"{netlist} not found. Copy it out of the GDS run first:\n"
                "  runs/wokwi/results/final/verilog/gl/<top>.v"
            )
        pdk = Path(pdk_root) / "ihp-sg13g2" / "libs.ref"
        # Matches the Makefile. Note -DFUNCTIONAL is inherited from the sky130
        # template and is dead weight here: sg13g2_stdcell.v contains no
        # `ifdef FUNCTIONAL` at all. It is kept only so the two paths stay
        # identical.
        defines = {"GL_TEST": 1, "FUNCTIONAL": 1, "SIM": 1}
        build_args = []
        plusargs = []
        build_dir = HERE / "sim_build" / "gl"

        if args.sdf:
            sdf = Path(args.sdf).resolve()
            if not sdf.exists():
                sys.exit(f"SDF file not found: {sdf}")
            defines["USE_SDF"] = 1
            # float, not str. cocotb renders a string define as
            # -DGL_IN_DELAY_NS="1.0" -- quoted -- and `assign #("1.0")` is a
            # string literal used as a delay, which compiles, runs, and puts
            # the whole design in X. Nothing reports it.
            defines["GL_IN_DELAY_NS"] = float(args.in_delay)
            # -gspecify: Icarus omits specify blocks by default, and with them
            # $sdf_annotate. -ginterconnect: without it every INTERCONNECT in
            # the file is a hard error, and the errors abort the rest of it.
            build_args = ["-gspecify", "-ginterconnect"]
            # The path goes in as a plusarg, not a define: a quoted path in a
            # define picks up a second layer of quotes through the runner and
            # annotates nothing at all.
            plusargs = ["+sdf=" + str(sdf).replace("\\", "/")]
            build_dir = HERE / "sim_build" / "gl_sdf"

        return dict(
            sources=[
                pdk / "sg13g2_io" / "verilog" / "sg13g2_io.v",
                pdk / "sg13g2_stdcell" / "verilog" / "sg13g2_stdcell.v",
                netlist,
                HERE / "tb.v",
            ],
            defines=defines,
            build_dir=build_dir,
            toplevel="tb",
            test_module=args.module,
            build_args=build_args,
            plusargs=plusargs,
        )

    return dict(
        sources=[SRC / s for s in PROJECT_SOURCES] + [HERE / "tb.v"],
        defines={},
        build_dir=HERE / "sim_build" / "rtl",
        toplevel="tb",
        test_module=args.module,
        build_args=[],
        plusargs=[],
    )


def check_results(xml_path):
    """Fail the process if any test failed.

    cocotb's runner returns success even when tests fail. The template Makefile
    works around this with `! grep failure results.xml`, and this script
    inherited exactly the same trap: verified by running a known-broken design
    through it and watching it exit 0 with 6 of 6 tests failing.

    A check that cannot go red is worse than no check, so this is deliberately
    strict -- an empty testcase list also fails, because a test module that
    fails to import produces zero tests and would otherwise look like a pass.
    """
    xml_path = Path(xml_path)
    if not xml_path.exists():
        sys.exit(f"no results file at {xml_path} -- the simulation did not report")

    cases = ElementTree.parse(xml_path).findall(".//testcase")
    if not cases:
        sys.exit(f"no testcases recorded in {xml_path} -- did the test module import?")

    bad = [c for c in cases if c.find("failure") is not None or c.find("error") is not None]
    for c in bad:
        print(f"FAIL: {c.get('classname')}.{c.get('name')}")
    if bad:
        sys.exit(f"{len(bad)} of {len(cases)} test(s) failed")
    print(f"{len(cases)} test(s) passed")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gates", action="store_true", help="gate-level simulation")
    ap.add_argument(
        "--sdf",
        metavar="FILE",
        help="back-annotate delays from FILE (run it through sdf_prep.py first)",
    )
    ap.add_argument(
        "--in-delay",
        default=os.environ.get("PROOF_IN_DELAY", "1.0"),
        metavar="NS",
        help="modelled host clock-to-output, ns (default 1.0); see tb.v",
    )
    ap.add_argument(
        "--unit",
        metavar="NAME",
        help="unit-test a submodule: src/NAME.v + test/tb_NAME.v + test/test_NAME.py",
    )
    # The top-level build is the same either way; only the module cocotb
    # collects tests from changes. test_cycles measures inference latency,
    # which is a different question from correctness and does not belong in
    # the main suite's runtime.
    ap.add_argument(
        "--module",
        default="test",
        metavar="NAME",
        help="test module for the top level (default: test; e.g. test_cycles)",
    )
    args = ap.parse_args()

    if args.unit and args.gates:
        sys.exit("--unit and --gates are mutually exclusive")
    if args.sdf and not args.gates:
        sys.exit("--sdf only means anything with --gates")

    p = plan(args)

    missing = [str(s) for s in p["sources"] if not Path(s).exists()]
    if missing:
        sys.exit("missing source file(s):\n  " + "\n  ".join(missing))

    runner = get_runner("icarus")
    runner.build(
        sources=p["sources"],
        hdl_toplevel=p["toplevel"],
        includes=[SRC],
        defines=p["defines"],
        build_dir=p["build_dir"],
        always=True,  # equivalent to `make -B`; a stale sim_build is a real trap
        timescale=("1ns", "1ps"),
        build_args=p["build_args"],
    )
    results = runner.test(
        hdl_toplevel=p["toplevel"],
        test_module=p["test_module"],
        build_dir=p["build_dir"],
        test_dir=HERE,
        plusargs=p["plusargs"],
        results_xml=f"results_{p['test_module']}.xml",
    )
    check_results(results)


if __name__ == "__main__":
    main()
