/*
 * Copyright (c) 2026 Kush Shah
 * SPDX-License-Identifier: Apache-2.0
 *
 * Proof -- flow bring-up skeleton.
 *
 * This is NOT the final datapath.  It exists to take the full LibreLane flow
 * end to end (RTL sim -> GDS -> gate-level sim) and to produce a utilization
 * report we can size the real network against.
 *
 * It is deliberately a *saturating* accumulator rather than throwaway logic:
 *   - saturation + a sticky overflow flag is a hard requirement of the final
 *     design (the prior CNN accelerator shipped without either, and BUGS.md
 *     lists that as an untestable gap -- not repeating it here), and
 *   - the load-enable on `acc` is exactly the construct that costs a mux2 per
 *     bit in sg13g2, which has no clock-enable flop.  Seeing that cost in the
 *     first report is the point.
 */

`default_nettype none

module tt_um_kush1434_proof (
    input  wire [7:0] ui_in,    // Dedicated inputs  -- payload byte
    output wire [7:0] uo_out,   // Dedicated outputs -- result byte
    input  wire [7:0] uio_in,   // IOs: Input path
    output wire [7:0] uio_out,  // IOs: Output path
    output wire [7:0] uio_oe,   // IOs: Enable path (1 = output)
    input  wire       ena,      // always 1 when powered
    input  wire       clk,      // clock
    input  wire       rst_n     // active-low reset
);

  // --------------------------------------------------------------------
  // Host protocol pins.  uio[4:0] are inputs, uio[7:5] are outputs.
  // --------------------------------------------------------------------
  wire valid  = uio_in[0];
  wire last   = uio_in[2];
  wire rd_sel = uio_in[4];

  localparam ACC_W = 16;

  reg signed [ACC_W-1:0] acc;
  reg                    sat;
  reg                    done;
  reg                    busy;

  // Sign-extend the payload byte, then add with one guard bit so that an
  // overflow is *visible* rather than silently wrapping.
  wire signed [ACC_W-1:0] term = {{(ACC_W-8){ui_in[7]}}, ui_in};
  wire signed [ACC_W:0]   sum  = {acc[ACC_W-1], acc} + {term[ACC_W-1], term};

  // Two's-complement overflow: the guard bit disagrees with the sign bit.
  wire ovf = (sum[ACC_W] != sum[ACC_W-1]);

  // Clamp to the rail we ran off:  sum[ACC_W]==0 -> +max, ==1 -> -min.
  wire signed [ACC_W-1:0] sum_sat =
      ovf ? {sum[ACC_W], {(ACC_W-1){~sum[ACC_W]}}} : sum[ACC_W-1:0];

  // Async assert off rst_n -- sg13g2's only flops (dfrbp*) have a native
  // RESET_B pin, so this costs nothing.  A synchronous reset would add a mux
  // per bit.  Everything is reset; nothing relies on power-up state, because
  // the gate-level netlist has no initial values.
  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      acc  <= {ACC_W{1'b0}};
      sat  <= 1'b0;
      done <= 1'b0;
      busy <= 1'b0;
    end else if (valid) begin
      acc  <= sum_sat;
      sat  <= sat | ovf;   // sticky until reset
      busy <= ~last;
      done <= last;
    end
  end

  assign uo_out  = rd_sel ? acc[ACC_W-1:ACC_W-8] : acc[7:0];
  assign uio_out = {busy, sat, done, 5'b00000};
  assign uio_oe  = 8'b1110_0000;  // [7:5] out, [4:0] in

  // List all unused inputs to prevent warnings
  wire _unused = &{ena, uio_in[7:5], uio_in[3], uio_in[1], 1'b0};

endmodule
