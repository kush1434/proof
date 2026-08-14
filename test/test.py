# SPDX-FileCopyrightText: (c) 2026 Kush Shah
# SPDX-License-Identifier: Apache-2.0
"""
Bring-up tests for the Proof skeleton (saturating 16-bit accumulator).

Not the final verification environment. These exist to prove the
RTL -> GDS -> gate-level path is green, and to pin down the cocotb 2.0.1
idioms the real testbench will be built on. Breaking changes vs cocotb 1.x:
cocotb.fork -> cocotb.start_soon, Timer(units=) -> Timer(unit=), TestFactory
removed, .value read semantics changed.

cocotb runs every test in ONE simulation against ONE dut, but 2.0 cancels the
tasks a test started when that test ends -- so the clock has to be restarted
per test, not shared. See setup().
"""

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, Timer

# uio_in -- host to chip
VALID = 1 << 0
LAST = 1 << 2
RD_SEL = 1 << 4

# uio_out -- chip to host
DONE = 1 << 5
SAT = 1 << 6
BUSY = 1 << 7

ACC_MAX = 32767
ACC_MIN = -32768


def s16(x):
    """Interpret a 16-bit pattern as signed."""
    return x - 0x10000 if x & 0x8000 else x


async def setup(dut):
    """Start a clock for this test, then reset.

    The clock must be started per test, not once per simulation. cocotb 2.0
    cancels every task a test started when that test ends, which cocotb 1.x
    did not do -- so a shared clock leaves test 2 onward with no clock at all.
    The simulator then runs out of events and exits, and cocotb reports the
    remaining tests as "Simulator shut down prematurely". That message names
    the symptom, not the cause, which makes this an expensive trap: budget for
    it in the real testbench rather than rediscovering it there.
    """
    cocotb.start_soon(Clock(dut.clk, 10, unit="us").start())
    await reset(dut)


async def settle():
    """Let non-blocking updates land before sampling anything.

    ClockCycles resumes *at* the rising edge, before the NBA region has run,
    so reading a flop output immediately after it returns hands back the value
    from before that edge. Combinational outputs happen to survive this;
    registered ones silently do not. Step past the edge before every read.
    """
    await Timer(1, unit="ns")


async def reset(dut):
    dut.ena.value = 1
    dut.ui_in.value = 0
    dut.uio_in.value = 0
    dut.rst_n.value = 0
    await ClockCycles(dut.clk, 5)
    dut.rst_n.value = 1
    await ClockCycles(dut.clk, 1)
    await settle()


async def push(dut, byte, last=False):
    """Present one payload byte with VALID asserted."""
    dut.ui_in.value = byte & 0xFF
    dut.uio_in.value = VALID | (LAST if last else 0)
    await ClockCycles(dut.clk, 1)
    dut.uio_in.value = 0
    dut.ui_in.value = 0
    await settle()


async def read_acc(dut):
    """Read the accumulator back one byte at a time via RD_SEL."""
    dut.uio_in.value = 0
    await ClockCycles(dut.clk, 1)
    await settle()
    lo = int(dut.uo_out.value)
    dut.uio_in.value = RD_SEL
    await ClockCycles(dut.clk, 1)
    await settle()
    hi = int(dut.uo_out.value)
    dut.uio_in.value = 0
    return s16((hi << 8) | lo)


def flags(dut):
    return int(dut.uio_out.value)


@cocotb.test()
async def test_reset_clears_everything(dut):
    """TT gotcha #1: nothing may rely on power-up state."""
    await setup(dut)
    assert int(dut.uio_oe.value) == 0xE0, "uio_oe must be [7:5] out, [4:0] in"
    assert await read_acc(dut) == 0
    assert flags(dut) & (DONE | SAT | BUSY) == 0


@cocotb.test()
async def test_accumulate_signed(dut):
    await setup(dut)
    vals = [10, 20, -5, 100, -128, 127]
    for v in vals:
        await push(dut, v)
    assert await read_acc(dut) == sum(vals)


@cocotb.test()
async def test_busy_then_done(dut):
    await setup(dut)
    await push(dut, 5)
    assert flags(dut) & BUSY, "BUSY should be high mid-stream"
    assert not flags(dut) & DONE
    await push(dut, 5, last=True)
    assert flags(dut) & DONE, "DONE should be high after LAST"
    assert not flags(dut) & BUSY


@cocotb.test()
async def test_saturate_positive(dut):
    """300 * 127 = 38100, well past the 16-bit signed rail."""
    await setup(dut)
    for _ in range(300):
        await push(dut, 127)
    assert await read_acc(dut) == ACC_MAX, "must clamp, not wrap"
    assert flags(dut) & SAT


@cocotb.test()
async def test_saturate_negative(dut):
    """300 * -128 = -38400, well past the other rail."""
    await setup(dut)
    for _ in range(300):
        await push(dut, 0x80)
    assert await read_acc(dut) == ACC_MIN, "must clamp, not wrap"
    assert flags(dut) & SAT


@cocotb.test()
async def test_saturated_flag_is_sticky(dut):
    await setup(dut)
    for _ in range(300):
        await push(dut, 127)
    assert flags(dut) & SAT
    for _ in range(10):
        await push(dut, 0x80)  # back away from the rail
    assert flags(dut) & SAT, "SATURATED must stay set until reset"


@cocotb.test()
async def test_reset_mid_stream(dut):
    """Reset must recover the chip from any state, including mid-inference."""
    await setup(dut)
    for _ in range(50):
        await push(dut, 127)
    await reset(dut)
    assert await read_acc(dut) == 0
    assert flags(dut) & (DONE | SAT | BUSY) == 0
