/*
 * Copyright (c) 2026 Kush Shah
 * SPDX-License-Identifier: Apache-2.0
 *
 * Proof -- a fixed-point dot-product engine that estimates how much a meal
 * will raise blood sugar.
 *
 * This module is a thin pin wrapper. All behaviour lives in proof_core; the
 * split keeps the Tiny Tapeout interface separate from the design so the core
 * can be unit-tested without the pin mapping in the way.
 *
 * Not a medical device. Every output is an estimate.
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

  // uio[4:0] host -> chip, uio[7:5] chip -> host.
  wire valid     = uio_in[0];
  wire is_weight = uio_in[1];
  wire last      = uio_in[2];
  wire mode      = uio_in[3];
  wire rd_sel    = uio_in[4];

  wire [7:0] result;
  wire       done;
  wire       saturated;
  wire       busy;

  proof_core u_core (
      .clk      (clk),
      .rst_n    (rst_n),
      .valid    (valid),
      .is_weight(is_weight),
      .last     (last),
      .mode     (mode),
      .data     (ui_in),
      .rd_sel   (rd_sel),
      .result   (result),
      .done     (done),
      .saturated(saturated),
      .busy     (busy)
  );

  assign uo_out  = result;
  assign uio_out = {busy, saturated, done, 5'b00000};
  assign uio_oe  = 8'b1110_0000;  // [7:5] out, [4:0] in

  wire _unused = &{ena, uio_in[7:5], 1'b0};

endmodule
