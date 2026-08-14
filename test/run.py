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
    python run.py --unit mac_serial  # submodule unit test

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
PROJECT_SOURCES = ["tt_um_kush1434_proof.v"]


def plan(args):
    """Return (sources, defines, build_dir, toplevel, test_module)."""
    if args.unit:
        name = args.unit
        return (
            [SRC / f"{name}.v", HERE / f"tb_{name}.v"],
            {},
            HERE / "sim_build" / f"unit_{name}",
            f"tb_{name}",
            f"test_{name}",
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
        return (
            [
                pdk / "sg13g2_io" / "verilog" / "sg13g2_io.v",
                pdk / "sg13g2_stdcell" / "verilog" / "sg13g2_stdcell.v",
                netlist,
                HERE / "tb.v",
            ],
            # Matches the Makefile. There is no SDF back-annotation, so this is
            # a *functional* gate-level sim: it catches synthesis differences,
            # X-propagation and missing resets, but says nothing about
            # setup/hold.
            {"GL_TEST": 1, "FUNCTIONAL": 1, "SIM": 1},
            HERE / "sim_build" / "gl",
            "tb",
            "test",
        )

    return (
        [SRC / s for s in PROJECT_SOURCES] + [HERE / "tb.v"],
        {},
        HERE / "sim_build" / "rtl",
        "tb",
        "test",
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
        "--unit",
        metavar="NAME",
        help="unit-test a submodule: src/NAME.v + test/tb_NAME.v + test/test_NAME.py",
    )
    args = ap.parse_args()

    if args.unit and args.gates:
        sys.exit("--unit and --gates are mutually exclusive")

    sources, defines, build_dir, toplevel, test_module = plan(args)

    missing = [str(s) for s in sources if not Path(s).exists()]
    if missing:
        sys.exit("missing source file(s):\n  " + "\n  ".join(missing))

    runner = get_runner("icarus")
    runner.build(
        sources=sources,
        hdl_toplevel=toplevel,
        includes=[SRC],
        defines=defines,
        build_dir=build_dir,
        always=True,  # equivalent to `make -B`; a stale sim_build is a real trap
        timescale=("1ns", "1ps"),
    )
    results = runner.test(
        hdl_toplevel=toplevel,
        test_module=test_module,
        build_dir=build_dir,
        test_dir=HERE,
        results_xml=f"results_{test_module}.xml",
    )
    check_results(results)


if __name__ == "__main__":
    main()
