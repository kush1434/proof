# SPDX-FileCopyrightText: (c) 2026 Kush Shah
# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for mac_serial, the bit-serial signed multiplier.

The headline test is exhaustive: all 65,536 signed 8x8 input pairs are checked
against Python's own arithmetic. An 8-bit operand is narrow enough that a
complete proof costs seconds, and a complete proof is strictly stronger than
any number of random seeds -- there is no coverage argument left to make.

That matters here specifically. On the prior CNN accelerator, mutant M9 (a
too-narrow accumulator) was caught by only 31 of 200 random seeds, because
uniform INT8 stimulus averages toward zero and rarely stresses width. The
carry/overflow corners of this multiplier have exactly the same shape, and
exhaustion sidesteps the sampling problem entirely.
"""

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, Timer

DW = 8
LO = -(2 ** (DW - 1))
HI = 2 ** (DW - 1)


def s16(x):
    """Interpret a 2*DW-bit pattern as signed."""
    return x - (1 << (2 * DW)) if x & (1 << (2 * DW - 1)) else x


async def settle():
    """Step past the edge so non-blocking updates have landed (cocotb 2.0)."""
    await Timer(1, unit="ns")


async def reset(dut):
    dut.start.value = 0
    dut.a.value = 0
    dut.b.value = 0
    dut.rst_n.value = 0
    await ClockCycles(dut.clk, 3)
    dut.rst_n.value = 1
    await ClockCycles(dut.clk, 1)
    await settle()


async def multiply(dut, av, bv):
    """Drive one multiply and return the product. Costs DW+1 clocks."""
    dut.a.value = av
    dut.b.value = bv
    dut.start.value = 1
    await ClockCycles(dut.clk, 1)  # start sampled here
    dut.start.value = 0
    await ClockCycles(dut.clk, DW)  # DW shift steps
    await settle()
    assert int(dut.done.value) == 1, f"done not asserted {DW} clocks after start"
    return s16(int(dut.product.value))


@cocotb.test()
async def test_exhaustive_signed_8x8(dut):
    """Every signed 8x8 product. 65,536 cases, no sampling."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset(dut)

    fails = 0
    for av in range(LO, HI):
        for bv in range(LO, HI):
            got = await multiply(dut, av, bv)
            exp = av * bv
            if got != exp:
                if fails < 10:
                    dut._log.error(f"{av} * {bv} = {exp}, got {got}")
                fails += 1
    assert fails == 0, f"{fails} of {(HI - LO) ** 2} products wrong"


@cocotb.test()
async def test_extremes(dut):
    """The corners, called out by name so a failure reads clearly."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset(dut)
    for av, bv in [
        (LO, LO),  # -128 * -128 = +16384, the only product needing bit 14
        (LO, HI - 1),
        (HI - 1, LO),
        (HI - 1, HI - 1),
        (LO, 0),
        (0, LO),
        (LO, -1),  # sign-correction step with a negative multiplicand
        (-1, -1),
        (1, LO),
    ]:
        got = await multiply(dut, av, bv)
        assert got == av * bv, f"{av} * {bv} = {av * bv}, got {got}"


@cocotb.test()
async def test_busy_and_done_protocol(dut):
    """busy spans the multiply; done is a single-cycle pulse."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset(dut)

    assert int(dut.busy.value) == 0, "busy should be low at rest"

    dut.a.value = 7
    dut.b.value = 9
    dut.start.value = 1
    await ClockCycles(dut.clk, 1)
    dut.start.value = 0
    await settle()
    assert int(dut.busy.value) == 1, "busy should rise the cycle start is taken"

    for _ in range(DW - 1):
        await ClockCycles(dut.clk, 1)
        await settle()
        assert int(dut.done.value) == 0, "done pulsed early"

    await ClockCycles(dut.clk, 1)
    await settle()
    assert int(dut.done.value) == 1
    assert int(dut.busy.value) == 0, "busy should drop with done"
    assert s16(int(dut.product.value)) == 63

    await ClockCycles(dut.clk, 1)
    await settle()
    assert int(dut.done.value) == 0, "done must be a single-cycle pulse"


@cocotb.test()
async def test_start_ignored_while_busy(dut):
    """A spurious start mid-multiply must not corrupt the running operation."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset(dut)

    dut.a.value = 100
    dut.b.value = 100
    dut.start.value = 1
    await ClockCycles(dut.clk, 1)
    dut.start.value = 0

    # Shove different operands in mid-flight with start asserted again.
    await ClockCycles(dut.clk, 3)
    dut.a.value = 1
    dut.b.value = 1
    dut.start.value = 1
    await ClockCycles(dut.clk, 1)
    dut.start.value = 0

    await ClockCycles(dut.clk, DW - 4)
    await settle()
    assert int(dut.done.value) == 1
    assert s16(int(dut.product.value)) == 10000, "in-flight multiply was corrupted"


@cocotb.test()
async def test_reset_mid_multiply(dut):
    """Reset must recover from mid-operation, not just from idle."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset(dut)

    dut.a.value = -55
    dut.b.value = 77
    dut.start.value = 1
    await ClockCycles(dut.clk, 1)
    dut.start.value = 0
    await ClockCycles(dut.clk, 3)  # interrupt part-way through

    await reset(dut)
    assert int(dut.busy.value) == 0
    assert int(dut.done.value) == 0
    assert s16(int(dut.product.value)) == 0

    # And the unit still works afterwards.
    assert await multiply(dut, -55, 77) == -55 * 77


@cocotb.test()
async def test_back_to_back(dut):
    """Consecutive multiplies with no idle cycle between them."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset(dut)
    pairs = [(3, 5), (-3, 5), (3, -5), (-3, -5), (127, -128), (0, 99)]
    for av, bv in pairs:
        assert await multiply(dut, av, bv) == av * bv, f"{av} * {bv}"
