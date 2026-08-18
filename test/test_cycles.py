# SPDX-FileCopyrightText: (c) 2026 Kush Shah
# SPDX-License-Identifier: Apache-2.0
"""How many clock cycles does one inference take? Measured, then pinned.

WHY THIS FILE EXISTS
--------------------
RESULTS.md 6 and the paper both quote "914 cycles = 18.3 us @ 50 MHz", and the
energy figure is derived from it (place-and-route power x that time). The only
surviving trace of where 914 came from was test/results_test_cycles.xml, whose
`file` attribute names a test/test_cycles.py that was never committed and no
longer exists on disk. A number in a submitted paper needs to be re-derivable,
so this measures it again and then asserts it, which turns the figure into
something CI protects rather than something a comment claims.

WHAT IS BEING COUNTED
---------------------
Wall-clock cycles for a complete Mode B inference at the shape the chip
actually runs -- 6 meal features plus a two-pair bias, 8 hidden neurons, 1
output neuron -- from the cycle the first shift byte is accepted to the cycle
`done` rises for the output neuron.

The host's read-back of intermediate results is NOT counted: a real host reads
only the final y, and reading h after every hidden neuron is a testbench
convenience. Time spent waiting on `busy` IS counted, because a host must
respect it and that wait is part of the latency a user experiences.

The count is clock-independent. The microsecond and energy figures come from
multiplying it by a target period, and 50 MHz is the frequency static timing
signs off at (setup slack +10.08 ns against a 20 ns period).
"""

import cocotb
from cocotb.triggers import ClockCycles
from cocotb.utils import get_sim_time

import golden_quant as gold
from test import (CLK_NS, DONE, flags, l1_from, run_l1_neuron, run_l2_neuron,
                  run_mode_b, setup)

# What RESULTS.md and the paper claim. Assert rather than print: a documented
# performance number that nothing checks is the same class of thing as the
# stale figures NUMBERS-CHECK.md catalogues.
DEPLOYED_CYCLES = 896
TOLERANCE = 0        # exact -- the datapath is fully deterministic


def cycles_now():
    return get_sim_time("ns") / CLK_NS


@cocotb.test()
async def test_inference_latency(dut):
    """A full 6-8-1 inference, timed end to end."""
    await setup(dut)

    # Shape the chip actually runs: 6 inputs, then the two bias pairs.
    W1 = [[1, 2, 3, 4, 5, 6] for _ in range(gold.N_HIDDEN)]
    x = [1, 1, 1, 1, 1, 1]
    bias = ((0, 127), (0, 1))
    l1 = l1_from(W1, x, bias_pairs=bias)
    W2 = [[1] * gold.N_HIDDEN + [0, 0]]

    start = cycles_now()
    per_neuron = []
    for j, pairs in enumerate(l1):
        t0 = cycles_now()
        await run_l1_neuron(dut, pairs, s1=0, first=(j == 0))
        per_neuron.append(cycles_now() - t0)
    t0 = cycles_now()
    await run_l2_neuron(dut, W2[0], s2=0)
    l2_cycles = cycles_now() - t0
    total = cycles_now() - start

    assert flags(dut) & DONE, "inference did not complete"

    dut._log.info("hidden neuron cycles : %s" % [int(c) for c in per_neuron])
    dut._log.info("layer-2 neuron cycles: %d" % int(l2_cycles))
    dut._log.info("TOTAL                : %d cycles" % int(total))
    for mhz in (1, 50):
        dut._log.info("  = %.2f us @ %d MHz" % (total / mhz, mhz))

    got = int(total)
    assert abs(got - DEPLOYED_CYCLES) <= TOLERANCE, (
        "inference takes %d cycles; RESULTS.md 6 and the paper say %d. "
        "Either the design changed or the documented figure is wrong -- "
        "update both together." % (got, DEPLOYED_CYCLES))


@cocotb.test()
async def test_readback_accounts_for_the_documented_figure(dut):
    """Where the previously documented 914 came from.

    The testbench helper `run_mode_b` reads the result register back after
    EVERY neuron, so it observes each hidden activation on its way past. That
    is a verification convenience -- a deployed host reads only the final y --
    and each read costs two cycles. Nine neurons, eighteen cycles, which is
    exactly the gap between the deployed figure and the documented one.

    Measured here so the difference is recorded rather than argued.
    """
    await setup(dut)

    W1 = [[1, 2, 3, 4, 5, 6] for _ in range(gold.N_HIDDEN)]
    l1 = l1_from(W1, [1] * 6, bias_pairs=((0, 127), (0, 1)))
    W2 = [[1] * gold.N_HIDDEN + [0, 0]]

    t0 = cycles_now()
    await run_mode_b(dut, l1, W2, s1=0, s2=0)
    with_readback = int(cycles_now() - t0)

    dut._log.info("with a read-back after every neuron: %d cycles" % with_readback)
    dut._log.info("deployed (final read only)         : %d cycles"
                  % DEPLOYED_CYCLES)
    assert with_readback == DEPLOYED_CYCLES + 2 * (gold.N_HIDDEN + 1), (
        "read-back overhead is not 2 cycles per neuron: %d vs %d + 2*%d"
        % (with_readback, DEPLOYED_CYCLES, gold.N_HIDDEN + 1))


@cocotb.test()
async def test_latency_is_deterministic(dut):
    """Two identical inferences take exactly the same time.

    Worth asserting separately: the quoted energy figure is cycles x power, so
    a latency that varied with data would make that number a mean rather than
    the fixed quantity it is presented as.
    """
    await setup(dut)

    def build(v):
        W1 = [[v] * 6 for _ in range(gold.N_HIDDEN)]
        return l1_from(W1, [v] * 6, bias_pairs=((0, 127), (0, 1)))

    took = []
    for v in (1, 7):
        t0 = cycles_now()
        for j, pairs in enumerate(build(v)):
            await run_l1_neuron(dut, pairs, s1=0, first=(j == 0))
        await run_l2_neuron(dut, [1] * gold.N_HIDDEN + [0, 0], s2=0)
        took.append(int(cycles_now() - t0))

    assert took[0] == took[1], (
        "latency depends on the data: %d vs %d cycles" % (took[0], took[1]))
