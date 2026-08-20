# Sample testbench for a Tiny Tapeout project

This is a sample testbench for a Tiny Tapeout project. It uses [cocotb](https://docs.cocotb.org/en/stable/) to drive the DUT and check the outputs.
See below to get started or for more information, check the [website](https://tinytapeout.com/hdl/testing/).

## Setting up

1. Edit [Makefile](Makefile) and modify `PROJECT_SOURCES` to point to your Verilog files.
2. Edit [tb.v](tb.v) and replace `tt_um_example` with your module name.

## How to run

To run the RTL simulation:

```sh
make -B
```

To run gatelevel simulation, first harden your project and copy `../runs/wokwi/results/final/verilog/gl/{your_module_name}.v` to `gate_level_netlist.v`.

Then run:

```sh
make -B GATES=yes
```

### Gate-level simulation with back-annotated delays

The run above has no timing in it at all. To run the same tests against the
post-route delays the flow already writes, take an SDF out of the `GDS_logs`
artifact (`runs/wokwi/final/sdf/<corner>/*.sdf`), convert it, and point the
build at it:

```sh
python sdf_prep.py <corner>.sdf annotate.sdf
make -B GATES=yes SDF_FILE=$PWD/annotate.sdf
```

The conversion is not optional — Icarus rejects three of the constructs
OpenSTA emits, and each rejection abandons the rest of the file, so an
unfiltered SDF annotates part of the design, leaves the rest at zero delay, and
still passes. `sdf_prep.py --selftest` checks the conversion against known
answers.

Set `PROOF_SETTLE_NS=3`: the annotated netlist takes ~2 ns to drive its pins,
and the suite's default 1 ns sampling window predates that. `COCOTB_TEST_MODULES=test_sdf`
runs the pin-timing measurements, which are what prove the annotation landed.

**This does not check setup or hold.** Icarus implements no timing checks in
any version. See `sdf_prep.py` and RESULTS.md §6.1.

On Windows use `run.py` rather than `make`:

```sh
python run.py --gates --sdf annotate.sdf
```

If you wish to save the waveform in VCD format instead of FST format, edit tb.v to use `$dumpfile("tb.vcd");` and then run:

```sh
make -B FST=
```

This will generate `tb.vcd` instead of `tb.fst`.

## How to view the waveform file

Using GTKWave

```sh
gtkwave tb.fst tb.gtkw
```

Using Surfer

```sh
surfer tb.fst
```
