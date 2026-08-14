# SPDX-FileCopyrightText: (c) 2026 Kush Shah
# SPDX-License-Identifier: Apache-2.0
"""
Top-level tests for Proof, Mode A.

Every functional test compares the RTL against golden_quant.py **bit-exactly**.
That is the stronger of the two obligations in the verification plan; the
float model (not written yet) carries the weaker error-bound obligation and is
kept separate on purpose, so the two are never conflated.

cocotb 2.0 notes, both learned the hard way (BUGS.md TB-1, TB-2):
  - Tasks a test starts are cancelled when it ends, so the clock is restarted
    per test rather than shared.
  - ClockCycles resumes before the NBA region, so registered outputs are only
    sampled after settle().
"""

import os
import random

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, Timer

import golden_quant as gold

# uio_in -- host to chip
VALID = 1 << 0
IS_WEIGHT = 1 << 1
LAST = 1 << 2
MODE = 1 << 3
RD_SEL = 1 << 4

# uio_out -- chip to host
DONE = 1 << 5
SAT = 1 << 6
BUSY = 1 << 7

CLK_NS = 10

# Overridable so mutate.sh can sweep seeds and report how many of them catch
# each mutant. On the CNN accelerator that number was the whole argument for
# directed tests: M9 was caught by only 31 of 200 random seeds.
SEED = int(os.environ.get("PROOF_SEED", "20260814"))
N_RANDOM_STREAMS = int(os.environ.get("PROOF_STREAMS", "30"))


async def settle():
    await Timer(1, unit="ns")


def flags(dut):
    return int(dut.uio_out.value)


async def setup(dut):
    """Fresh clock per test, then reset."""
    cocotb.start_soon(Clock(dut.clk, CLK_NS, unit="ns").start())
    await reset(dut)


async def reset(dut):
    dut.ena.value = 1
    dut.ui_in.value = 0
    dut.uio_in.value = 0
    dut.rst_n.value = 0
    await ClockCycles(dut.clk, 5)
    dut.rst_n.value = 1
    await ClockCycles(dut.clk, 1)
    await settle()


async def wait_not_busy(dut, limit=200):
    for _ in range(limit):
        if not flags(dut) & BUSY:
            return
        await ClockCycles(dut.clk, 1)
        await settle()
    raise AssertionError("busy never cleared")


async def send(dut, byte, is_weight=False, last=False):
    """Present one byte, respecting the busy handshake."""
    await wait_not_busy(dut)
    dut.ui_in.value = byte & 0xFF
    dut.uio_in.value = VALID | (IS_WEIGHT if is_weight else 0) | (LAST if last else 0)
    await ClockCycles(dut.clk, 1)
    dut.uio_in.value = 0
    dut.ui_in.value = 0
    await settle()


async def wait_done(dut, limit=400):
    for _ in range(limit):
        if flags(dut) & DONE:
            return
        await ClockCycles(dut.clk, 1)
        await settle()
    raise AssertionError("done never asserted")


async def read_result(dut):
    """Read both result bytes via RD_SEL. valid stays low, so nothing is consumed."""
    dut.uio_in.value = 0
    await ClockCycles(dut.clk, 1)
    await settle()
    lo = int(dut.uo_out.value)
    dut.uio_in.value = RD_SEL
    await ClockCycles(dut.clk, 1)
    await settle()
    hi = int(dut.uo_out.value)
    dut.uio_in.value = 0
    await settle()
    return lo, hi


async def run_stream(dut, pairs, shift):
    """Drive a whole Mode A inference: shift byte, then (weight, activation) pairs."""
    await send(dut, shift, is_weight=True)
    for i, (w, g) in enumerate(pairs):
        await send(dut, w, is_weight=True)
        await send(dut, g, last=(i == len(pairs) - 1))
    await wait_done(dut)


async def check_stream(dut, pairs, shift):
    """Run a stream and assert the RTL matches golden_quant bit-exactly."""
    await run_stream(dut, pairs, shift)
    lo, hi = await read_result(dut)
    exp = gold.mode_a(pairs, shift)
    elo, ehi = gold.result_bytes(exp)
    ctx = f"pairs={pairs} shift={shift} -> acc={exp['acc']} gl={exp['gl']} cat={exp['cat']}"
    assert (lo, hi) == (elo, ehi), f"result {lo:#04x},{hi:#04x} != {elo:#04x},{ehi:#04x}  [{ctx}]"
    got_sat = bool(flags(dut) & SAT)
    assert got_sat == exp["saturated"], f"saturated {got_sat} != {exp['saturated']}  [{ctx}]"
    return exp


# ---------------------------------------------------------------- basics ---


@cocotb.test()
async def test_reset_clears_everything(dut):
    """TT gotcha #1: nothing may rely on power-up state."""
    await setup(dut)
    assert int(dut.uio_oe.value) == 0xE0, "uio_oe must be [7:5] out, [4:0] in"
    assert flags(dut) & (DONE | SAT | BUSY) == 0
    lo, hi = await read_result(dut)
    assert (lo, hi) == (0, 0)


@cocotb.test()
async def test_single_pair(dut):
    await setup(dut)
    await check_stream(dut, [(4, 5)], shift=0)


@cocotb.test()
async def test_known_glycemic_load(dut):
    """A worked example: three ingredients, shift 4."""
    await setup(dut)
    exp = await check_stream(dut, [(50, 40), (25, 60), (10, 30)], shift=4)
    # 50*40 + 25*60 + 10*30 = 3800; 3800 >> 4 = 237
    assert exp["acc"] == 3800 and exp["gl"] == 237


@cocotb.test()
async def test_negative_weights(dut):
    """The datapath is signed end to end; floor-shift semantics must match."""
    await setup(dut)
    for pairs, shift in [
        ([(-3, 1)], 1),  # -3 >> 1 = -2, floors toward -inf
        ([(-1, 1)], 0),
        ([(-128, 127)], 3),
        ([(-128, -128)], 5),
        ([(7, -9), (-11, 13)], 2),
    ]:
        await check_stream(dut, pairs, shift)


# ------------------------------------------------------ category boundaries -


@cocotb.test()
async def test_category_boundaries(dut):
    """low <= 10, medium 11-19, high >= 20 -- checked either side of each edge."""
    await setup(dut)
    for gl, want in [
        (0, gold.CAT_LOW),
        (10, gold.CAT_LOW),
        (11, gold.CAT_MED),
        (19, gold.CAT_MED),
        (20, gold.CAT_HIGH),
        (21, gold.CAT_HIGH),
    ]:
        exp = await check_stream(dut, [(1, gl)], shift=0)
        assert exp["gl"] == gl and exp["cat"] == want, f"gl={gl}"


@cocotb.test()
async def test_shift_flips_the_category(dut):
    """The requantiser sits between the sum and the category, so rounding at
    the shift can move a meal across a threshold. Pin the exact boundary."""
    await setup(dut)
    # 40 >> 1 = 20 -> high;  39 >> 1 = 19 -> medium.
    exp = await check_stream(dut, [(1, 40)], shift=1)
    assert exp["gl"] == 20 and exp["cat"] == gold.CAT_HIGH
    exp = await check_stream(dut, [(1, 39)], shift=1)
    assert exp["gl"] == 19 and exp["cat"] == gold.CAT_MED


# ------------------------------------------------------------- randomised --


@cocotb.test()
async def test_randomised_streams(dut):
    """Random streams against the golden model. Seeded for reproducibility."""
    await setup(dut)
    rng = random.Random(SEED)
    for _ in range(N_RANDOM_STREAMS):
        n = rng.randint(1, 6)
        pairs = [(rng.randint(-128, 127), rng.randint(-128, 127)) for _ in range(n)]
        await check_stream(dut, pairs, shift=rng.randint(0, 15))


@cocotb.test()
async def test_extreme_operands(dut):
    """Directed corners. Uniform random almost never lands on these -- the M9
    lesson from the CNN accelerator, where extremes mattered and sampling
    caught them in only 31 of 200 seeds."""
    await setup(dut)
    E = [-128, -1, 0, 1, 127]
    for w in E:
        for g in E:
            await check_stream(dut, [(w, g)], shift=0)


# ------------------------------------------------------------- saturation --


@cocotb.test()
async def test_saturate_positive(dut):
    """520 terms of -128 * -128 = +16384 each; the rail is at 2**23 - 1."""
    await setup(dut)
    exp = await check_stream(dut, [(-128, -128)] * 520, shift=0)
    assert exp["saturated"] and exp["acc"] == gold.ACC_MAX


@cocotb.test()
async def test_saturate_negative(dut):
    """520 terms of -128 * 127 = -16256 each."""
    await setup(dut)
    exp = await check_stream(dut, [(-128, 127)] * 520, shift=0)
    assert exp["saturated"] and exp["acc"] == gold.ACC_MIN


@cocotb.test()
async def test_no_false_saturation(dut):
    """Mode B's worst case must stay far from the rail -- if the flag ever
    fires for 8 terms, the width proof in accumulator.v is wrong."""
    await setup(dut)
    exp = await check_stream(dut, [(-128, -128)] * 8, shift=0)
    assert not exp["saturated"], "flag fired well below the rail"
    assert exp["acc"] == 8 * 16384


@cocotb.test()
async def test_saturated_flag_is_sticky_then_clears(dut):
    """Sticky within a stream, cleared by the next one."""
    await setup(dut)
    await check_stream(dut, [(-128, -128)] * 520, shift=0)
    assert flags(dut) & SAT
    await check_stream(dut, [(1, 2)], shift=0)
    assert not flags(dut) & SAT, "a new stream must clear the sticky flag"


@cocotb.test()
async def test_sticky_flag_survives_non_overflowing_terms(dut):
    """Saturate, then keep accumulating terms that do NOT overflow.

    Without this case a non-sticky flag is indistinguishable from a sticky
    one. Once the accumulator is pinned at the rail, every further term in the
    same direction overflows too, so even `sat <= ovf` still reads 1 at the
    end of the stream. The flag only reveals itself as non-sticky when a later
    term does not overflow and clears it.

    Found by mutation testing: M5 escaped the suite until this test existed
    (BUGS.md TB-4). Same structural blindness as M7 on the CNN accelerator,
    where a scoreboard that only compared final memory contents could not see
    a write-enable bug whose last write happened to be correct.
    """
    await setup(dut)
    # 520 saturating terms, then four that land comfortably inside the rail.
    pairs = [(-128, -128)] * 520 + [(1, -100)] * 4
    exp = await check_stream(dut, pairs, shift=0)
    assert exp["saturated"], "reference should still report saturation"
    assert flags(dut) & SAT, "flag was cleared by a later non-overflowing term"


# --------------------------------------------------------- protocol abuse --


@cocotb.test()
async def test_sticky_coefficient(dut):
    """An activation with no fresh weight reuses the previous coefficient."""
    await setup(dut)
    await send(dut, 0, is_weight=True)  # shift
    await send(dut, 5, is_weight=True)  # coefficient
    await send(dut, 3)
    await send(dut, 4, last=True)  # no new weight: reuses 5
    await wait_done(dut)
    lo, hi = await read_result(dut)
    exp = gold.mode_a([(5, 3), (5, 4)], 0)
    assert (lo, hi) == gold.result_bytes(exp)
    assert exp["acc"] == 35


@cocotb.test()
async def test_bytes_offered_while_busy_are_ignored(dut):
    """Flow control is the host's job; bytes during busy must not corrupt."""
    await setup(dut)
    await send(dut, 0, is_weight=True)
    await send(dut, 10, is_weight=True)

    # Present the activation, then keep hammering valid through the multiply.
    dut.ui_in.value = 7
    dut.uio_in.value = VALID | LAST
    await ClockCycles(dut.clk, 1)
    await settle()
    assert flags(dut) & BUSY, "should be busy immediately after taking the activation"
    for _ in range(6):
        dut.ui_in.value = 0x7F
        dut.uio_in.value = VALID | IS_WEIGHT
        await ClockCycles(dut.clk, 1)
        await settle()
    dut.uio_in.value = 0
    dut.ui_in.value = 0

    await wait_done(dut)
    lo, hi = await read_result(dut)
    exp = gold.mode_a([(10, 7)], 0)
    assert (lo, hi) == gold.result_bytes(exp), "stream corrupted by bytes sent while busy"


@cocotb.test()
async def test_activation_before_shift_is_ignored(dut):
    """A stream that opens with an activation is a protocol error. It must be
    dropped rather than accumulated, and must not wedge the FSM."""
    await setup(dut)
    await send(dut, 99)  # activation with no stream open
    await send(dut, 99)
    assert not flags(dut) & (BUSY | DONE)
    await check_stream(dut, [(3, 3)], shift=0)  # still works


@cocotb.test()
async def test_truncated_stream_then_reset(dut):
    """A stream that stops mid-way must not hang the chip; reset recovers it."""
    await setup(dut)
    await send(dut, 0, is_weight=True)
    await send(dut, 12, is_weight=True)
    await send(dut, 34)  # no LAST -- stream abandoned here
    await wait_not_busy(dut)
    assert not flags(dut) & DONE, "done must not assert without LAST"

    await reset(dut)
    assert flags(dut) & (DONE | SAT | BUSY) == 0
    await check_stream(dut, [(6, 7)], shift=0)


@cocotb.test()
async def test_reset_mid_multiply(dut):
    """Reset from inside the multiply, not just between bytes."""
    await setup(dut)
    await send(dut, 0, is_weight=True)
    await send(dut, 100, is_weight=True)
    dut.ui_in.value = 100
    dut.uio_in.value = VALID | LAST
    await ClockCycles(dut.clk, 1)
    dut.uio_in.value = 0
    await ClockCycles(dut.clk, 3)  # interrupt part-way through the multiply

    await reset(dut)
    lo, hi = await read_result(dut)
    assert (lo, hi) == (0, 0)
    await check_stream(dut, [(2, 3)], shift=0)


@cocotb.test()
async def test_back_to_back_inferences(dut):
    """Consecutive inferences with no reset between them."""
    await setup(dut)
    for pairs, shift in [
        ([(3, 4)], 0),
        ([(-5, 6), (7, 8)], 2),
        ([(100, 100)], 6),
        ([(1, 20)], 0),
    ]:
        await check_stream(dut, pairs, shift)


@cocotb.test()
async def test_zero_cases(dut):
    """Zero weight, zero activation, and a zero-length-ish stream."""
    await setup(dut)
    await check_stream(dut, [(0, 0)], shift=0)
    await check_stream(dut, [(0, 127)], shift=0)
    await check_stream(dut, [(127, 0)], shift=3)
