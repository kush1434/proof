#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Kush Shah
# SPDX-License-Identifier: Apache-2.0
"""
Windows-friendly cocotb runner.

The Tiny Tapeout template drives cocotb through test/Makefile, which needs GNU
make. There is no make on the development machine, so this uses cocotb 2.x's
built-in runner instead.

It mirrors the Makefile deliberately -- same sources, same defines, same
toplevel, same build directories -- so that a local run and the CI run (which
does use the Makefile) stay equivalent. If you change PROJECT_SOURCES in the
Makefile, change it here too. Both must also match `source_files` in info.yaml;
a desync there is a silent RTL/GL mismatch.

    python run.py            # RTL simulation
    python run.py --gates    # gate-level (needs gate_level_netlist.v + PDK_ROOT)
"""

import argparse
import os
import sys
from pathlib import Path

from cocotb_tools.runner import get_runner

HERE = Path(__file__).parent.resolve()
SRC = HERE.parent / "src"

# Keep in sync with PROJECT_SOURCES in Makefile and source_files in info.yaml.
PROJECT_SOURCES = ["tt_um_kush1434_proof.v"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gates", action="store_true", help="gate-level simulation")
    args = ap.parse_args()

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
        sources = [
            pdk / "sg13g2_io" / "verilog" / "sg13g2_io.v",
            pdk / "sg13g2_stdcell" / "verilog" / "sg13g2_stdcell.v",
            netlist,
        ]
        # Matches the Makefile. Note there is no SDF back-annotation, so this
        # is a *functional* gate-level sim: it catches synthesis differences,
        # X-propagation and missing resets, but says nothing about setup/hold.
        defines = {"GL_TEST": 1, "FUNCTIONAL": 1, "SIM": 1}
        build_dir = HERE / "sim_build" / "gl"
    else:
        sources = [SRC / s for s in PROJECT_SOURCES]
        defines = {}
        build_dir = HERE / "sim_build" / "rtl"

    sources.append(HERE / "tb.v")

    missing = [str(s) for s in sources if not Path(s).exists()]
    if missing:
        sys.exit("missing source file(s):\n  " + "\n  ".join(missing))

    runner = get_runner("icarus")
    runner.build(
        verilog_sources=sources,
        hdl_toplevel="tb",
        includes=[SRC],
        defines=defines,
        build_dir=build_dir,
        always=True,  # equivalent to `make -B`; stale sim_build is a real trap
        timescale=("1ns", "1ps"),
    )
    runner.test(
        hdl_toplevel="tb",
        test_module="test",
        build_dir=build_dir,
        test_dir=HERE,
        results_xml="results.xml",
    )


if __name__ == "__main__":
    main()
