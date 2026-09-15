![](../../workflows/gds/badge.svg) ![](../../workflows/test/badge.svg) ![](../../workflows/docs/badge.svg) ![](../../workflows/fpga/badge.svg) ![](../../workflows/paper/badge.svg)

# Proof

**A glycemic-response inference chip that checks whether the model it was handed
is safe to trust — in hardware, as the weights arrive.**

A 1×1 Tiny Tapeout tile on the IHP SG13G2 open PDK. It runs a small quantised
neural network that estimates how much a meal will raise someone's blood sugar.
It holds **no weights at all**: all 65 parameters stream in over the pins for
every inference, so the model can be different for every patient.

That is the whole problem this chip is about.

---

## The idea

A model like this is only usable as advice if it behaves sensibly: **adding
carbohydrate must never lower the predicted response.** A model that says
"add sugar, your response drops" is not slightly wrong, it is incoherent, and
no accuracy score repairs that.

This property — monotonicity — is normally guaranteed at training time and
checked **offline, once, on a fixed set of weights**. A device that ships one
model inherits that certificate.

This device does not ship one model. It is handed a fresh one every inference,
by a host it has no reason to trust. There is no certificate to inherit.

So the chip checks the precondition itself.

The guarantee holds if, for every hidden unit `j`:

```
W1[j][carb] · W2[j] ≥ 0
```

Each unit either helps carbohydrate raise the response, or opposes it on both
sides. The awkward part is that the two operands arrive far apart in time —
`W1[j][carb]` early in layer 1, `W2[j]` much later in layer 2 — and storing
weights to compare them would cost more than the datapath.

So it stores **two bits per hidden unit** instead: the sign, and whether the
weight is non-zero. They rotate in lockstep with the hidden activations, so the
matching bits arrive at the head exactly when the partner weight does. The
whole comparison is one line of Verilog:

```verilog
wire viol_now = l2_weight && (data[DW-1] ^ sgnreg[N_HIDDEN-1])
                && (|data) && nzreg[N_HIDDEN-1];
```

**Twenty flip-flops**, an XOR, an eight-input OR reduction and three ANDs. If it
fires, the chip raises `UNTRUSTED` and declines to vouch for the answer.

### Why that matters

Refitting an unconstrained network on one person's own meals violates the sign
condition in **44 of 44** participants tested. The population model already
violates it before any personalisation. An unconstrained fit simply never
satisfies the condition — so personalisation is not what breaks it, it is what
makes it **undecidable offline**, because the weight set the device runs is not
the one anyone certified.

`UNTRUSTED` means the weight set **does not admit the guarantee by this check**.
The condition is sufficient, not necessary — Liu et al. call this predicate
*sign verification* — so the guard is sound but not complete. It is conservative
by design.

## The bug that made the point

An early version was provably monotone all the way through the datapath, and the
**output field truncated**. Truncation wraps. One extra count of carbohydrate
made the reported answer fall from **31,293 to −31,209** while the true internal
value rose to 34,327.

Every test passed. The reference model truncated too, so both were wrong in the
same way and comparing them only proved they agreed.

That is the thesis in one defect: a guarantee can hold throughout your logic and
still be destroyed by the field that reports it. Both fields saturate now.

## Results

**Silicon** (`gds` workflow, LibreLane, IHP SG13G2):

| | |
|---|---|
| Standard cells / flip-flops | 1443 / 168 |
| Utilisation | 83.53 % |
| DRC / LVS / antenna / latches / lint | 0 |
| Latency | 896 cycles = 17.9 µs at 50 MHz |
| Energy per prediction | 32.0 nJ |
| Cost of the guard | 20 flip-flops, 113 standard cells |

**Verification:**

- 43 top-level tests, bit-exact against an integer reference model
- 6 unit tests, **exhaustive** over all 65,536 signed 8×8 multiplier inputs
- 54 of 54 named coverage bins, asserted rather than printed
- 29 mutants: 28 caught, 1 proven equivalent, **0 escaped**
- Accumulator monotonicity **proved by k-induction** — unbounded, not a bounded
  check (`formal/`)
- Gate-level simulation with post-route SDF back-annotated at all three corners

⚠️ **Setup and hold rest on static timing analysis alone.** Icarus implements no
timing checks in any version, so the SDF's timing-check blocks are inert.
Back-annotation shows the netlist computes correctly with real cell and
interconnect delays; it is not a setup/hold check.

**Accuracy — read this before being impressed by the rest.**

Trained and evaluated on [CGMacros](https://physionet.org/content/cgmacros/),
45 participants, 1308 meals, grouped by participant so no one appears on both
sides of a split.

| | |
|---|---|
| Out-of-fold R² | +0.225 |
| Within-person rank correlation (Spearman) | median +0.382 |
| Participants ranked in the right direction | 41 of 44 |

That is modest, and the honest reading is that the model is **data-limited, not
capacity-limited** — hidden widths from 4 to 12 are indistinguishable. Every
figure here is an **upper bound**: quantisation scales, hidden width, adaptation
method and constraint set were all chosen while looking at the same folds the
results are reported on, and 45 participants left nothing to hold back.

Three of the 44 participants are ranked the *wrong* way. Monotonicity does not
protect against that — it guarantees coherence, not correctness.

**This is not a medical device. Every output is an estimate.**

## Reproducing it

```bash
./lint.sh                      # synthesis check, ~1 s
cd test && python run.py       # 43 top-level tests
./mutate.sh                    # 29 mutants
cd formal && python run.py     # 5 formal tasks
python verify_numbers.py       # re-runs the analyses and checks this repo's numbers
python check_numbers.py        # cross-document check for stale figures
```

Every number in the documentation names the script that produces it, and
`verify_numbers.py` re-runs those scripts and compares. `check_numbers.py`
additionally holds every value a figure *used to* have, and fails if any
document still quotes a retired one — this project's recurring failure mode was
derived numbers going stale, so it is checked mechanically.

Trained weights are not in the repository: CGMacros is CC BY-NC-SA and this
repository is Apache-2.0. One of the 43 tests skips in CI for that reason.

## Where to read more

| | |
|---|---|
| [docs/info.md](docs/info.md) | how the chip works, pinout, how to test it |
| [RESULTS.md](RESULTS.md) | every number, with the script that produces it |
| [BUGS.md](BUGS.md) | four RTL defects, seven testbench defects, and what is deliberately **not** covered |
| [VERIFICATION.md](VERIFICATION.md) | the verification argument |
| `paper/` | the IEEE BIBM 2026 submission and the scripts that regenerate its figures |

`BUGS.md` is the one to read if you want to know what this design does *not*
establish. It is maintained as a list of gaps, not a list of achievements.

## Built on

[Tiny Tapeout](https://tinytapeout.com) · [IHP SG13G2 open PDK](https://github.com/IHP-GmbH/IHP-Open-PDK) · LibreLane · cocotb · Icarus Verilog · SymbiYosys

Apache-2.0.
