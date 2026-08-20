`default_nettype none
`timescale 1ns / 1ps

/* This testbench just instantiates the module and makes some convenient wires
   that can be driven / tested by the cocotb test.py.
*/
module tb ();

  // Dump the signals to a FST file. You can view it with gtkwave or surfer.
  initial begin
    $dumpfile("tb.fst");
    $dumpvars(0, tb);
    #1;
  end

  // Wire up the inputs and outputs:
  reg clk;
  reg rst_n;
  reg ena;
  reg [7:0] ui_in;
  reg [7:0] uio_in;
  wire [7:0] uo_out;
  wire [7:0] uio_out;
  wire [7:0] uio_oe;

  // --------------------------------------------------------------- host ---
  // Everything below is compiled only for the SDF back-annotated gate-level
  // run (`USE_SDF`). Without it this file elaborates exactly as it always
  // has, so the RTL suite and the functional gate-level suite are unchanged.
  //
  // Why the delay: cocotb writes a signal in the same time step as the clock
  // edge it just awaited. With no delays anywhere that is harmless -- the
  // flops sample the old value and the new value lands afterwards. Once the
  // netlist carries real delays it stops being harmless, because the clock
  // takes ~0.9 ns to reach the flops through the clock tree while an input
  // change reaches them in ~0.1 ns, so a pin driven *at* the edge arrives
  // first and the flop captures the wrong value. That is a property of the
  // stimulus, not of the chip: a real host is clocked by the same clock and
  // its outputs change a clock-to-output time *after* the edge, never on it.
  // `GL_IN_DELAY_NS` is that clock-to-output time.
`ifdef USE_SDF
  wire [7:0] ui_in_dut;
  wire [7:0] uio_in_dut;
  wire       rst_n_dut;
  wire       ena_dut;
  assign #(`GL_IN_DELAY_NS) ui_in_dut  = ui_in;
  assign #(`GL_IN_DELAY_NS) uio_in_dut = uio_in;
  assign #(`GL_IN_DELAY_NS) rst_n_dut  = rst_n;
  assign #(`GL_IN_DELAY_NS) ena_dut    = ena;
`else
  wire [7:0] ui_in_dut = ui_in;
  wire [7:0] uio_in_dut = uio_in;
  wire       rst_n_dut = rst_n;
  wire       ena_dut = ena;
`endif

  // Device under test:
  tt_um_kush1434_proof user_project (
      .ui_in  (ui_in_dut),   // Dedicated inputs
      .uo_out (uo_out),      // Dedicated outputs
      .uio_in (uio_in_dut),  // IOs: Input path
      .uio_out(uio_out),     // IOs: Output path
      .uio_oe (uio_oe),      // IOs: Enable path (active high: 0=input, 1=output)
      .ena    (ena_dut),     // enable - goes high when design is selected
      .clk    (clk),         // clock
      .rst_n  (rst_n_dut)    // not reset
  );

`ifdef USE_SDF
  // The SDF path arrives as a plusarg rather than a define, because a define
  // carrying a quoted Windows path picks up a second layer of quotes on the
  // way through the cocotb runner and silently annotates nothing.
  //
  // Nothing about $sdf_annotate reports failure: a missing file, a netlist
  // that does not match, or a compile without `-gspecify` all leave the
  // design at zero delay and the suite still passes. So this opens the file
  // itself first and dies if it cannot, and `test_sdf.py` measures a delay
  // afterwards to prove the annotation actually landed.
  reg [8191:0] sdf_path;
  integer sdf_fd;
  initial begin
    if (!$value$plusargs("sdf=%s", sdf_path)) begin
      $display("USE_SDF is defined but no +sdf=<file> plusarg was given");
      $fatal(1);
    end
    sdf_fd = $fopen(sdf_path, "r");
    if (sdf_fd == 0) begin
      $display("cannot open SDF file: %0s", sdf_path);
      $fatal(1);
    end
    $fclose(sdf_fd);
    $sdf_annotate(sdf_path, user_project);
  end
`endif

endmodule
