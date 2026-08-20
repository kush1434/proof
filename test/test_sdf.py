# SPDX-FileCopyrightText: (c) 2026 Kush Shah
# SPDX-License-Identifier: Apache-2.0
"""
Pin timing on the SDF back-annotated gate-level netlist.

`test.py` proves the netlist computes the right answer. It cannot prove the
*delays* arrived, because a design annotated with nothing behaves exactly like
a design annotated correctly, only faster -- and every functional test passes
either way. Three separate things silently leave the netlist at zero delay:

  - compiling without `-gspecify` (Icarus prints one warning at compile time
    and then omits `$sdf_annotate` entirely);
  - an SDF whose syntax Icarus rejects part-way, which abandons every cell
    after the first few errors;
  - `sdf_prep.py` failing to match anything, if OpenSTA ever changes spelling.

So this module measures a delay and asserts it is not zero. That is the whole
point of it: it is the check that can go red when the annotation quietly does
nothing. With `PROOF_EXPECT_SDF=1` a zero measurement is a failure.

What it measures is **clock-to-output at the pins**: the time from a rising
clock edge until the chip's outputs change. That is a real, simulated timing
number -- and it is the reason the suite needs `PROOF_SETTLE_NS`, because
`test.py` samples outputs 1 ns after the edge by default and the annotated
netlist has not answered by then.

**What this does not measure: setup and hold.** Icarus does not implement
timing checks in any version -- it says so at compile time and again during
annotation -- so the cell models' `$setuphold`, `$recrem` and `$width` never
run, and `sdf_prep.py` strips the SDF's TIMINGCHECK sections because they
would only produce warnings. Setup and hold rest on STA alone. Nothing here
changes that.

Run:
    python run.py --gates --sdf <file> --module test_sdf
"""

import os

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, RisingEdge, Timer

import test as t

# Resolution of the delay measurement. The clock-to-output time being measured
# is a couple of nanoseconds, so 25 ps is finer than it needs to be and still
# costs only ~80 steps per cycle.
#
# It is also the floor of the measurement, and that matters. Polling starts
# just after the clock edge, so a netlist with *no* delay -- where the outputs
# settle in the same time step as the edge -- still reads as one step, 25 ps,
# never 0. Asserting "> 0" would therefore pass on an unannotated netlist,
# which is exactly the thing this module exists to catch. The real annotated
# figure is ~2100 ps, some eighty steps, so "more than one step" separates the
# two cleanly without inventing a threshold.
STEP_PS = 25

# Set by the SDF job. When it is on, a zero measurement fails rather than
# merely being reported -- the difference between a check and a printout.
EXPECT_SDF = os.environ.get("PROOF_EXPECT_SDF", "") == "1"


async def delay_to_change(sig, budget_ns):
    """ps from now until `sig` changes value, or None if it does not.

    Polled with Timer rather than an edge trigger because `uio_out` is a bus
    that can be partly X while the netlist settles, and a value-change trigger
    would fire on the first bit to resolve rather than on the transition being
    timed.
    """
    before = str(sig.value)
    for i in range(1, int(budget_ns * 1000 / STEP_PS) + 1):
        await Timer(STEP_PS, unit="ps")
        if str(sig.value) != before:
            return i * STEP_PS
    return None


async def measure_clock_to_output(dut, cycles=60):
    """Largest clock-to-output delay seen on uo_out/uio_out over one inference.

    Returns (max_ps, samples). Runs the ordinary Mode A stream from `test.py`
    so the traffic is the same traffic the functional tests use.
    """
    worst = 0
    samples = 0
    budget = t.CLK_NS * 0.8

    async def monitor():
        nonlocal worst, samples
        for _ in range(cycles):
            await RisingEdge(dut.clk)
            d = await delay_to_change(dut.uio_out, budget)
            if d is not None:
                samples += 1
                worst = max(worst, d)

    mon = cocotb.start_soon(monitor())
    await t.run_stream(dut, [(4, 5), (3, 2)], shift=0)
    mon.cancel()
    return worst, samples


@cocotb.test()
async def test_sdf_annotation_landed(dut):
    """The netlist carries non-zero delay -- i.e. the SDF actually annotated.

    Without back-annotation every output changes in the same time step as the
    clock edge and this measures 0 ps.
    """
    await t.setup(dut)
    worst, samples = await measure_clock_to_output(dut)
    dut._log.info(
        "clock-to-output: worst %d ps over %d sampled edges (expect_sdf=%s)"
        % (worst, samples, EXPECT_SDF))

    assert samples > 0, "no output transition was observed at all"
    if EXPECT_SDF:
        assert worst > STEP_PS, (
            "clock-to-output measured %d ps, the floor of this measurement, "
            "so the outputs settle in the same time step as the clock edge "
            "and the netlist is at zero delay: the SDF did not annotate. "
            "Check that iverilog got -gspecify, that sdf_prep.py ran, and "
            "that the run printed no SDF warnings." % worst)
    else:
        dut._log.info("PROOF_EXPECT_SDF is not set; measurement is reported, not asserted")


@cocotb.test()
async def test_clock_to_output_fits_in_a_cycle(dut):
    """Outputs settle well inside one clock period.

    A weak bound, deliberately: it is a sanity check on the annotation, not a
    timing sign-off. The output path is not a register-to-register path, so
    STA's setup slack does not speak to it, and nothing here checks setup or
    hold -- see the module docstring.
    """
    await t.setup(dut)
    worst, samples = await measure_clock_to_output(dut)
    assert samples > 0, "no output transition was observed at all"
    assert worst < t.CLK_NS * 1000, (
        "clock-to-output %d ps does not fit in the %d ns clock period"
        % (worst, t.CLK_NS))
    dut._log.info("clock-to-output %d ps of a %d ns period" % (worst, t.CLK_NS))


@cocotb.test()
async def test_settle_covers_clock_to_output(dut):
    """`test.py` waits SETTLE_NS after each edge before reading. It has to be
    longer than the chip takes to answer, or every functional test reads the
    previous cycle's value and fails in ways that look like logic bugs.

    This is the assertion that explains PROOF_SETTLE_NS to whoever hits it.
    """
    await t.setup(dut)
    worst, samples = await measure_clock_to_output(dut)
    assert samples > 0, "no output transition was observed at all"
    assert worst < t.SETTLE_NS * 1000, (
        "clock-to-output is %d ps but the suite samples outputs %g ns after "
        "the edge; raise PROOF_SETTLE_NS above %.1f ns"
        % (worst, t.SETTLE_NS, worst / 1000.0))
    dut._log.info("settle %g ns covers clock-to-output %d ps"
                  % (t.SETTLE_NS, worst))
