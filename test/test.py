# SPDX-FileCopyrightText: (c) 2026 Kush Shah
# SPDX-License-Identifier: Apache-2.0
"""
Top-level tests for Proof -- Mode A, Mode B, and the safety property.

Every functional test compares the RTL against golden_quant.py **bit-exactly**.
That is the stronger of the two obligations in the verification plan;
golden_float.py carries the weaker error-bound obligation and is kept separate
on purpose, so the two are never conflated.

cocotb 2.0 notes, both learned the hard way (BUGS.md TB-1, TB-2):
  - Tasks a test starts are cancelled when it ends, so the clock is restarted
    per test rather than shared.
  - ClockCycles resumes before the NBA region, so registered outputs are only
    sampled after settle().
"""

import json
import os
import random

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, Timer

import golden_float as gf
import golden_quant as gold
from coverage import cov

# uio_in -- host to chip
VALID = 1 << 0
IS_WEIGHT = 1 << 1
LAST = 1 << 2
MODE = 1 << 3
RD_SEL = 1 << 4

# uio_out -- chip to host
DONE = 1 << 5
UNTRUSTED = 1 << 6
SAT = UNTRUSTED  # legacy alias: Mode A has no monotonicity notion
BUSY = 1 << 7

CLK_NS = 10

# Overridable so mutate.sh can sweep seeds and report how many of them catch
# each mutant. On the CNN accelerator that number was the whole argument for
# directed tests: M9 was caught by only 31 of 200 random seeds.
# Trained weights are derived from CGMacros, which is CC BY-NC-SA. They are
# deliberately NOT committed, so this test skips in CI rather than shipping a
# derivative of a non-commercial dataset in an Apache-2.0 repository.
WEIGHTS = os.path.join(os.path.dirname(__file__), "weights.json")
HAVE_WEIGHTS = os.path.exists(WEIGHTS)

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


async def send(dut, byte, is_weight=False, last=False, mode=False):
    """Present one byte, respecting the busy handshake."""
    await wait_not_busy(dut)
    dut.ui_in.value = byte & 0xFF
    dut.uio_in.value = (
        VALID
        | (IS_WEIGHT if is_weight else 0)
        | (LAST if last else 0)
        | (MODE if mode else 0)
    )
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
    cov.sample_mode_a(pairs, shift, exp)
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
    cov.hit("protocol", "sticky_coefficient")


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
    cov.hit("protocol", "bytes_while_busy")
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
async def test_busy_covers_the_whole_retire(dut):
    """`busy` must not drop before the result is valid.

    The contract is "do not present a byte while busy", so the converse has to
    hold: any cycle busy is low, a byte will be accepted. After the final term
    there is a multiply, an accumulate and a retire cycle. If busy clears
    before the retire finishes, the host sees a legal window, sends into it,
    and the byte lands in a state that ignores it -- silently desynchronising
    the stream one byte for the rest of the inference.

    Found by mutation testing: M12 escaped until this test existed, because
    every other test waits for `done` rather than for `busy`, and so never
    looks at the handoff between them.
    """
    await setup(dut)
    await send(dut, 0, is_weight=True)
    await send(dut, 6, is_weight=True)
    dut.ui_in.value = 7
    dut.uio_in.value = VALID | LAST
    await ClockCycles(dut.clk, 1)
    dut.uio_in.value = 0
    dut.ui_in.value = 0
    await settle()

    for _ in range(60):
        f = flags(dut)
        if not f & BUSY:
            assert f & DONE, "busy dropped while the result was still forming"
            cov.hit("protocol", "busy_done_handoff")
            return
        await ClockCycles(dut.clk, 1)
        await settle()
    raise AssertionError("busy never cleared")


@cocotb.test()
async def test_activation_before_shift_is_ignored(dut):
    """A stream that opens with an activation is a protocol error. It must be
    dropped rather than accumulated, and must not wedge the FSM."""
    await setup(dut)
    await send(dut, 99)  # activation with no stream open
    cov.hit("protocol", "activation_before_shift")
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
    cov.hit("protocol", "truncated_stream")

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
    cov.hit("protocol", "reset_mid_multiply")

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


# ============================================================== MODE B =====
#
# Every neuron is a Mode A dot product. A hidden neuron takes its activations
# from the host and pushes its ReLU'd result into h; an output neuron takes
# its activations from the chip (h, then the constants 127 and 1) and is read
# back directly.


async def run_l1_neuron(dut, pairs, s1, first=False):
    """One hidden neuron. LAST on the shift byte marks a new inference."""
    await send(dut, s1, is_weight=True, mode=True, last=first)
    for i, (w, x) in enumerate(pairs):
        await send(dut, w, is_weight=True, mode=True)
        await send(dut, x, last=(i == len(pairs) - 1), mode=True)
    await wait_done(dut)


async def run_l2_neuron(dut, weights, s2):
    """One output neuron: weight bytes only, activations supplied by the chip."""
    await send(dut, s2, is_weight=True, mode=True)
    for i, w in enumerate(weights):
        await send(dut, w, is_weight=True, last=(i == len(weights) - 1), mode=True)
    await wait_done(dut)


async def run_mode_b(dut, l1, l2, s1, s2):
    """Drive a whole two-layer inference, reading every intermediate."""
    h_read = []
    for j, pairs in enumerate(l1):
        await run_l1_neuron(dut, pairs, s1, first=(j == 0))
        h_read.append(await read_result(dut))
    y_read = []
    for weights in l2:
        await run_l2_neuron(dut, weights, s2)
        y_read.append(await read_result(dut))
    return h_read, y_read


async def check_mode_b(dut, l1, l2, s1, s2):
    """Run an inference and check every hidden activation and output.

    Checking h as well as y is deliberate: the hidden layer is observable
    because a hidden neuron presents its result exactly like an output neuron
    does, so a wrong h is caught where it happens rather than being diluted
    through layer 2.
    """
    h_read, y_read = await run_mode_b(dut, l1, l2, s1, s2)
    exp = gold.mode_b(l1, l2, s1, s2)

    for j, (got, want) in enumerate(zip(h_read, exp["h"])):
        assert got == (want, 0), f"h[{j}] = {got}, expected ({want}, 0)"
    for i, (got, want) in enumerate(zip(y_read, exp["y"])):
        assert got == gold.wide_bytes(want), (
            f"y[{i}] = {got}, expected {gold.wide_bytes(want)} (raw {want})"
        )
    got = bool(flags(dut) & UNTRUSTED)
    assert got == exp["untrusted"], (
        f"untrusted {got} != {exp['untrusted']} "
        f"(saturated={exp['saturated']} mono_violation={exp['mono_violation']})")
    cov.sample_mode_b(l1, l2, s1, s2, exp)
    return exp


def l1_from(weights, x, bias_pairs=()):
    """Build hidden-neuron streams: activations re-streamed per neuron."""
    return [list(zip(row, x)) + list(bias_pairs) for row in weights]


@cocotb.test()
async def test_mode_b_smoke(dut):
    """Smallest useful two-layer inference."""
    await setup(dut)
    x = [10, 20, 30, 40, 50, 60]
    W1 = [[1, 0, 0, 0, 0, 0]] * gold.N_HIDDEN
    W2 = [[1] * gold.N_HIDDEN + [0, 0]]
    exp = await check_mode_b(dut, l1_from(W1, x), W2, s1=0, s2=0)
    assert exp["h"] == [10] * 8, exp["h"]
    assert exp["y"] == [80], exp["y"]


@cocotb.test()
async def test_mode_b_relu_clamps_negative(dut):
    """A negative pre-activation must come out as exactly zero, not wrap."""
    await setup(dut)
    x = [100, 0, 0, 0, 0, 0]
    W1 = [[-50, 0, 0, 0, 0, 0]] * gold.N_HIDDEN  # -5000 before ReLU
    W2 = [[1] * gold.N_HIDDEN + [0, 0]]
    exp = await check_mode_b(dut, l1_from(W1, x), W2, s1=2, s2=0)
    assert exp["h"] == [0] * 8, exp["h"]
    assert exp["y"] == [0], exp["y"]


@cocotb.test()
async def test_mode_b_h_clamps_at_127(dut):
    """h is stored as INT8, so a large pre-activation saturates at 127."""
    await setup(dut)
    x = [127, 127, 127, 127, 127, 127]
    W1 = [[127] * 6] * gold.N_HIDDEN  # 96,774 before the shift
    W2 = [[0] * gold.N_HIDDEN + [0, 0]]
    exp = await check_mode_b(dut, l1_from(W1, x), W2, s1=0, s2=0)
    assert exp["h"] == [127] * 8, exp["h"]


@cocotb.test()
async def test_mode_b_bias_constants(dut):
    """The chip supplies 127 then 1 after h, so a bias is exact to the unit.

    Checked by driving every hidden weight to zero: y is then purely the bias,
    and v8*127 + v9*1 must reproduce it exactly.
    """
    await setup(dut)
    x = [0] * 6
    W1 = [[0] * 6] * gold.N_HIDDEN
    for target in (0, 1, -1, 127, -128, 5000, -5000, 8128):
        v8 = max(-128, min(127, round(target / 127)))
        v9 = target - 127 * v8
        if not -128 <= v9 <= 127:
            continue
        W2 = [[0] * gold.N_HIDDEN + [v8, v9]]
        exp = await check_mode_b(dut, l1_from(W1, x), W2, s1=0, s2=0)
        assert exp["y"] == [target], f"bias {target} -> {exp['y']}"


@cocotb.test()
async def test_mode_b_h_order_is_preserved(dut):
    """h[k] must line up with the k'th layer-2 weight.

    Asymmetric by construction: each hidden neuron gets a distinct value and
    each output weight is a distinct power of two, so any rotation or
    transposition of h changes y. The M2 lesson from the CNN accelerator --
    symmetric stimulus cannot see a transposed index.
    """
    await setup(dut)
    l1 = [[(j + 1, 1)] for j in range(gold.N_HIDDEN)]  # h[j] = j + 1
    W2 = [[1, 2, 4, 8, 16, 32, 64, 127] + [0, 0]]
    exp = await check_mode_b(dut, l1, W2, s1=0, s2=0)
    assert exp["h"] == [1, 2, 3, 4, 5, 6, 7, 8], exp["h"]
    assert exp["y"] == [1 + 4 + 12 + 32 + 80 + 192 + 448 + 1016], exp["y"]


@cocotb.test()
async def test_mode_b_h_survives_multiple_output_neurons(dut):
    """h rotates a full turn per output neuron, so all three see the same h."""
    await setup(dut)
    l1 = [[(j + 1, 1)] for j in range(gold.N_HIDDEN)]
    W2 = [[1] * gold.N_HIDDEN + [0, 0]] * 3  # identical -> identical outputs
    exp = await check_mode_b(dut, l1, W2, s1=0, s2=0)
    assert exp["y"] == [36, 36, 36], exp["y"]


@cocotb.test()
async def test_mode_b_sticky_flag_spans_neurons(dut):
    """An overflow in hidden neuron 0 must still be visible at the output.

    This is why the accumulator has separate `clear` and `clear_sat`. The
    accumulator is cleared for every neuron; clearing the flag with it would
    report only the last neuron and silently lose every earlier overflow.
    """
    await setup(dut)
    # Neuron 0 saturates; every later neuron is small and cannot overflow.
    # W2[0] is negative so unit 0's two signs AGREE: this weight set saturates
    # without also violating the sign condition. Both causes share one output
    # pin, so a saturation test built on violating weights would pass on the
    # guard alone and could not tell the two apart.
    l1 = [[(-128, -128)] * 520] + [[(1, 1)] for _ in range(gold.N_HIDDEN - 1)]
    W2 = [[-1] + [1] * (gold.N_HIDDEN - 1) + [0, 0]]
    exp = await check_mode_b(dut, l1, W2, s1=0, s2=0)
    assert exp["saturated"]
    assert flags(dut) & SAT, "overflow from hidden neuron 0 was lost"
    cov.hit("protocol", "sticky_across_neurons")


@cocotb.test()
async def test_mode_b_new_inference_clears_flag(dut):
    """LAST on the opening shift byte starts a fresh inference."""
    await setup(dut)
    # W2[0] is negative so unit 0's two signs AGREE: this weight set saturates
    # without also violating the sign condition. Both causes share one output
    # pin, so a saturation test built on violating weights would pass on the
    # guard alone and could not tell the two apart.
    l1 = [[(-128, -128)] * 520] + [[(1, 1)] for _ in range(gold.N_HIDDEN - 1)]
    W2 = [[-1] + [1] * (gold.N_HIDDEN - 1) + [0, 0]]
    await check_mode_b(dut, l1, W2, s1=0, s2=0)
    assert flags(dut) & SAT
    clean = [[(1, 1)] for _ in range(gold.N_HIDDEN)]
    # (1, 1) against W2[0] = -1 would disagree, so use an all-positive W2 here.
    await check_mode_b(dut, clean, [[1] * gold.N_HIDDEN + [0, 0]], s1=0, s2=0)
    assert not flags(dut) & SAT, "a new inference must clear the sticky flag"
    cov.hit("protocol", "new_inference_flag")


@cocotb.test()
async def test_mode_b_output_field_saturates_both_ways(dut):
    """The 16-bit output field must clamp, not wrap, at both rails.

    This is R-4 in test form. The high rail was previously reached only
    incidentally inside the monotonicity sweep, and the low rail was not
    reached at all -- the coverage model is what made that visible, since both
    bins read MISS while every other bin was hit.

    Wrapping here is what broke the safety property: the internal value keeps
    rising while the reported one flips sign.
    """
    await setup(dut)
    # Drive every hidden unit to its ceiling, so h = 127 across the board.
    l1 = [[(127, 127)] for _ in range(gold.N_HIDDEN)]

    hi = await check_mode_b(dut, l1, [[127] * gold.N_HIDDEN + [0, 0]], s1=0, s2=0)
    assert hi["h"] == [127] * gold.N_HIDDEN
    assert hi["y"][0] == 8 * 127 * 127, hi["y"]
    assert hi["y"][0] > 32767, "did not exceed the field, so it proves nothing"

    lo = await check_mode_b(dut, l1, [[-128] * gold.N_HIDDEN + [0, 0]], s1=0, s2=0)
    assert lo["y"][0] == 8 * -128 * 127, lo["y"]
    assert lo["y"][0] < -32768, "did not undercut the field, so it proves nothing"


# ======================================================= MONOTONICITY GUARD ==
#
# The safety property is a property of the WEIGHTS. Since the host streams a
# different weight set per patient and an unconstrained per-person refit
# violates the condition in 44 of 44 cases measured on CGMacros, the chip
# checks its own precondition instead of trusting the stream.


@cocotb.test()
async def test_guard_passes_valid_weights(dut):
    """A weight set that satisfies the sign condition must NOT be flagged."""
    await setup(dut)
    l1 = [[(20, 5), (3, 2)] for _ in range(gold.N_HIDDEN)]   # carb weight +20
    l2 = [[7] * gold.N_HIDDEN + [0, 0]]                      # all W2 positive
    exp = await check_mode_b(dut, l1, l2, s1=0, s2=0)
    assert not exp["mono_violation"]
    assert not flags(dut) & UNTRUSTED, "valid weights were flagged"


@cocotb.test()
async def test_guard_catches_violating_weights(dut):
    """One hidden unit whose two signs disagree must raise UNTRUSTED.

    This is the case that matters: the arithmetic is still perfectly correct,
    the result is still bit-exact against the reference, and the monotonicity
    guarantee is nonetheless void. Nothing else on the chip can tell.
    """
    await setup(dut)
    l1 = [[(20, 5), (3, 2)] for _ in range(gold.N_HIDDEN)]
    l1[3] = [(-20, 5), (3, 2)]                  # unit 3 opposes carbohydrate
    l2 = [[7] * gold.N_HIDDEN + [0, 0]]         # while W2[3] is positive
    exp = await check_mode_b(dut, l1, l2, s1=0, s2=0)
    assert exp["mono_violation"], "reference should flag this weight set"
    assert flags(dut) & UNTRUSTED, "guard missed a sign-condition violation"
    cov.hit("guard", "violation_sticky")


@cocotb.test()
async def test_guard_allows_opposing_pairs(dut):
    """A unit may oppose carbohydrate TWICE and still be monotone.

    W1 negative with W2 negative gives a non-negative product, so the guard
    must not fire. A naive 'all weights positive' check would fail here.
    """
    await setup(dut)
    l1 = [[(20, 5), (3, 2)] for _ in range(gold.N_HIDDEN)]
    l1[2] = [(-20, 5), (3, 2)]
    l2 = [[7] * gold.N_HIDDEN + [0, 0]]
    l2[0][2] = -7                                # both negative on unit 2
    exp = await check_mode_b(dut, l1, l2, s1=0, s2=0)
    assert not exp["mono_violation"]
    assert not flags(dut) & UNTRUSTED, "double-negative unit wrongly flagged"


@cocotb.test()
async def test_guard_zero_carb_weight_is_not_a_violation(dut):
    """A zero carbohydrate weight gives a zero product, which is >= 0.

    This is why the guard carries a non-zero bit per unit as well as a sign:
    sign alone would read 0 as positive and flag it against a negative W2.
    """
    await setup(dut)
    l1 = [[(20, 5), (3, 2)] for _ in range(gold.N_HIDDEN)]
    l1[5] = [(0, 5), (3, 2)]                     # unit 5 ignores carbohydrate
    l2 = [[7] * gold.N_HIDDEN + [0, 0]]
    l2[0][5] = -7                                # negative, but product is 0
    exp = await check_mode_b(dut, l1, l2, s1=0, s2=0)
    assert not exp["mono_violation"], "zero weight cannot violate"
    assert not flags(dut) & UNTRUSTED, "zero carb weight wrongly flagged"
    cov.hit("guard", "zero_carb_weight")


@cocotb.test()
async def test_guard_flag_clears_on_new_inference(dut):
    """LAST on the shift byte starts a fresh inference and clears the guard."""
    await setup(dut)
    l1 = [[(20, 5), (3, 2)] for _ in range(gold.N_HIDDEN)]
    bad = [x[:] for x in l1]
    bad[1] = [(-20, 5), (3, 2)]
    l2 = [[7] * gold.N_HIDDEN + [0, 0]]

    await check_mode_b(dut, bad, l2, s1=0, s2=0)
    assert flags(dut) & UNTRUSTED
    await check_mode_b(dut, l1, l2, s1=0, s2=0)
    assert not flags(dut) & UNTRUSTED, "guard flag survived a new inference"
    cov.hit("guard", "cleared_by_new_inference")


@cocotb.test()
async def test_monotonic_in_carbohydrate(dut):
    """THE HEADLINE PROPERTY, checked on the RTL rather than on the model.

    Holding every other input fixed, increasing carbohydrate must never
    decrease the predicted response.

    Internally the pipeline is monotone by construction: saturating sums,
    arithmetic shifts, ReLU and clamps are each non-decreasing, so their
    composition is too, provided every hidden unit satisfies the sign
    condition W1[j][c] * W2[j] >= 0. This weight set satisfies it trivially --
    both are positive.

    The window is chosen to straddle the point where the response leaves the
    16-bit output field. That is where it used to break: the internal value
    kept rising while the reported value wrapped negative, because the field
    truncated instead of saturating (BUGS.md R-4). Sweeping somewhere
    comfortable would prove nothing -- the property only ever failed at the
    boundary, which is exactly the M9 lesson about directed stimulus.
    """
    await setup(dut)
    W1 = [[127, 0, 0, 0, 0, 0] for _ in range(gold.N_HIDDEN)]
    W2 = [[127] * gold.N_HIDDEN + [0, 0]]
    s1, s2 = 7, 0

    prev_y = None
    prev_h = None
    saw_clamp = False
    for xc in range(25, 51):
        x = [xc, 0, 0, 0, 0, 0]
        l1 = [list(zip(row, x)) for row in W1]

        h_read, y_read = await run_mode_b(dut, l1, W2, s1, s2)
        exp = gold.mode_b(l1, W2, s1, s2)

        # Still bit-exact against the reference at every point of the sweep.
        for j, (got, want) in enumerate(zip(h_read, exp["h"])):
            assert got == (want, 0), f"x_c={xc} h[{j}]={got} != ({want}, 0)"
        assert y_read[0] == gold.wide_bytes(exp["y"][0]), f"x_c={xc} y mismatch"

        lo, hi = y_read[0]
        y = (hi << 8) | lo
        y = y - 0x10000 if y & 0x8000 else y
        h0 = h_read[0][0]

        if prev_y is not None:
            assert h0 >= prev_h, f"h fell at x_c={xc}: {prev_h} -> {h0}"
            assert y >= prev_y, (
                f"REPORTED RESPONSE FELL as carbohydrate rose: "
                f"x_c {xc - 1} -> {xc} gave y {prev_y} -> {y} "
                f"(true value {exp['y'][0]})"
            )
        if y == 32767:
            saw_clamp = True
        prev_y, prev_h = y, h0

    assert saw_clamp, "sweep never reached the field limit -- it proves nothing"
    cov.hit("monotonicity", "synthetic_weights")


@cocotb.test(skip=not HAVE_WEIGHTS)
async def test_trained_network_on_silicon(dut):
    """End to end: a network trained on CGMacros, run on the actual RTL.

    This is the join between the two halves of the project. `model/train.py`
    fits a monotone-by-construction network to 1,346 real meals;
    `golden_float.to_chip_streams` quantises it into the exact byte streams the
    chip consumes; this drives those bytes through the RTL and checks two
    things at once:

      1. the hardware is bit-exact against the integer reference, and
      2. the safety property survives on REAL trained weights -- not on the
         synthetic weight sets used elsewhere in this file.

    Skipped when model/weights.json is absent, so CI without the dataset still
    passes rather than silently reporting a green run that tested nothing.
    """
    await setup(dut)
    with open(WEIGHTS, encoding="utf-8") as f:
        W = json.load(f)

    assert W["sign_condition_holds"], (
        "these weights cannot be monotone -- retrain with the constraint"
    )

    W1, b1 = W["W1"], W["b1"]
    W2, b2 = W["W2"], W["b2"]
    mu = W["standardise"]["mu"]
    sd = W["standardise"]["sd"]
    cons = W.get("monotone_constraints", {"carbs": 1})

    # A plausible meal, in real units, then standardised the way training was.
    base = [60.0, 4.0, 15.0, 20.0, 110.0, 8.5]

    for feat, want in sorted(cons.items()):
        idx = W["features"].index(feat)
        values = range(10, 121, 10) if feat == "carbs" else range(0, 31, 2)
        await _sweep_one(dut, W, mu, sd, base, idx, feat, want, values)

    cov.hit("monotonicity", "trained_weights")
    if any(v < 0 for v in cons.values()):
        cov.hit("monotonicity", "trained_weights_decreasing")


async def _sweep_one(dut, W, mu, sd, base, idx, feat, want, values):
    """Sweep one input and check bit-exactness plus the required direction."""
    W1, b1, W2, b2 = W["W1"], W["b1"], W["W2"], W["b2"]
    raw = list(base)
    prev = None
    seen = []
    for v in values:
        raw[idx] = float(v)
        x = [(a - m) / s for a, m, s in zip(raw, mu, sd)]

        l1, l2, meta = gf.to_chip_streams(W1, b1, W2, b2, x, **gf.BEST_SCALES)
        h_read, y_read = await run_mode_b(dut, l1, l2, meta["s1"], meta["s2"])
        exp = gold.mode_b(l1, l2, meta["s1"], meta["s2"])

        # (1) bit-exact against the reference at every point
        for j, (g, wnt) in enumerate(zip(h_read, exp["h"])):
            assert g == (wnt, 0), f"{feat}={v} h[{j}]={g} != ({wnt}, 0)"
        assert y_read[0] == gold.wide_bytes(exp["y"][0]), f"{feat}={v} y mismatch"

        # (2) the safety property, on trained weights
        blo, bhi = y_read[0]
        y = (bhi << 8) | blo
        y = y - 0x10000 if y & 0x8000 else y
        if prev is not None:
            if want > 0:
                assert y >= prev, (
                    f"REPORTED RESPONSE FELL as {feat} rose: {y} < {prev}")
            else:
                assert y <= prev, (
                    f"REPORTED RESPONSE ROSE as {feat} rose: {y} > {prev}")
        prev = y
        seen.append(y)

    direction = "non-decreasing" if want > 0 else "non-increasing"
    spread = max(seen) - min(seen)
    dut._log.info(
        f"  {feat:8s} {len(seen)} levels, {direction}, "
        f"observed range {min(seen)}..{max(seen)} (spread {spread})")
    # A guarantee that holds because nothing moved proves very little. Report
    # it rather than let a flat sweep masquerade as a verified property.
    if spread == 0:
        dut._log.warning(
            f"  {feat}: output did not move across the sweep -- the direction "
            f"holds trivially here, so it is not evidence of much")


@cocotb.test()
async def test_mode_b_randomised(dut):
    """Random two-layer inferences against the golden model."""
    await setup(dut)
    rng = random.Random(SEED ^ 0xB)
    for _ in range(6):
        x = [rng.randint(-128, 127) for _ in range(6)]
        W1 = [[rng.randint(-128, 127) for _ in range(6)] for _ in range(gold.N_HIDDEN)]
        W2 = [
            [rng.randint(-128, 127) for _ in range(gold.N_HIDDEN + 2)]
            for _ in range(3)
        ]
        await check_mode_b(dut, l1_from(W1, x), W2, s1=rng.randint(0, 8),
                           s2=rng.randint(0, 8))


@cocotb.test()
async def test_mode_switching_between_inferences(dut):
    """A -> B -> A with no reset. Mode A must be unaffected by Mode B state."""
    await setup(dut)
    await check_stream(dut, [(50, 40), (25, 60), (10, 30)], shift=4)
    x = [10, 20, 30, 40, 50, 60]
    W1 = [[1, 0, 0, 0, 0, 0]] * gold.N_HIDDEN
    W2 = [[1] * gold.N_HIDDEN + [0, 0]]
    await check_mode_b(dut, l1_from(W1, x), W2, s1=0, s2=0)
    exp = await check_stream(dut, [(50, 40), (25, 60), (10, 30)], shift=4)
    assert exp["acc"] == 3800 and exp["gl"] == 237, "Mode A perturbed by Mode B"


@cocotb.test()
async def test_mode_b_reset_mid_inference(dut):
    """Reset partway through layer 1, then a clean inference."""
    await setup(dut)
    x = [10, 20, 30, 40, 50, 60]
    W1 = [[1, 0, 0, 0, 0, 0]] * gold.N_HIDDEN
    W2 = [[1] * gold.N_HIDDEN + [0, 0]]
    for j in range(3):
        await run_l1_neuron(dut, list(zip(W1[j], x)), s1=0, first=(j == 0))
    await reset(dut)
    assert flags(dut) & (DONE | SAT | BUSY) == 0
    await check_mode_b(dut, l1_from(W1, x), W2, s1=0, s2=0)


# ============================================================ COVERAGE =====


@cocotb.test()
async def test_zzz_coverage_report(dut):
    """Report named-bin coverage. Defined last so it runs last.

    This asserts full coverage rather than merely printing it. A coverage model
    nobody fails on is decoration -- the point is that adding a bin without
    stimulus to reach it breaks the build, which forces the gap to be either
    covered or explicitly removed with a reason.

    `test_trained_network_on_silicon` is skipped when the CGMacros-derived
    weights are absent, so its bin is exempted in that case rather than
    silently lowering the bar.
    """
    hit, total, misses = cov.report(log=dut._log.info)
    if not HAVE_WEIGHTS:
        exempt = {"monotonicity.trained_weights",
                  "monotonicity.trained_weights_decreasing"}
        misses = [m for m in misses if m not in exempt]
        dut._log.info("  (trained-weight bins exempt: weights not present)")
    assert not misses, f"uncovered bins: {misses}"
