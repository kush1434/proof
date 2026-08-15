/*
 * Copyright (c) 2026 Kush Shah
 * SPDX-License-Identifier: Apache-2.0
 *
 * Saturating accumulator with a sticky overflow flag.
 *
 * WHY SATURATION IS MANDATORY HERE
 * --------------------------------
 * The prior CNN accelerator has no saturation logic and no overflow flag, and
 * its BUGS.md lists that as an area that therefore cannot be tested. A
 * too-narrow accumulator in a convolution produces a wrong pixel. Here it
 * produces a wrong number about food that a person might act on, so wrapping
 * silently is not an acceptable failure mode. The accumulator clamps to the
 * rail it ran off and raises a flag that stays set until cleared.
 *
 * WIDTH PROOF (ACC_W = 24, signed: -8,388,608 .. 8,388,607)
 * --------------------------------------------------------
 * Terms are products of two signed 8-bit values, so |term| <= 128*128 = 16384.
 *
 *   Mode B layer 1: 6 terms          -> |acc| <= 6  * 16384 =  98,304
 *   Mode B layer 2: 8 terms          -> |acc| <= 8  * 16256 = 130,048
 *                   (h is clamped to 0..127 by ReLU + requantisation, so the
 *                    worst product there is 127*128 = 16256, not 16384)
 *   plus a 16-bit bias                                      <=  32,768
 *
 * Both layers therefore stay under 163k, a factor of ~51 below the rail:
 * Mode B cannot saturate, and the flag firing during Mode B would itself be a
 * bug worth investigating.
 *
 * Mode A streams an unbounded number of ingredients, so it *can* saturate --
 * after at least 512 maximum-magnitude terms (512 * 16384 = 8,388,608). A
 * recipe with 512 ingredients each at the INT8 extremes is not a real meal, so
 * saturation is unreachable in practice; the flag covers the pathological case
 * rather than an expected one. This is the claim the boundary tests check.
 */

`default_nettype none

module accumulator #(
    parameter integer ACC_W  = 24,
    parameter integer TERM_W = 16
) (
    input  wire                     clk,
    input  wire                     rst_n,
    input  wire                     clear,      // synchronous: zero the accumulator
    input  wire                     clear_sat,  // synchronous: zero the sticky flag
    input  wire                     add_en,     // accumulate `term` this cycle
    input  wire signed [TERM_W-1:0] term,
    output wire signed [ACC_W-1:0]  acc,
    output wire                     saturated   // sticky until clear or reset
);

  reg signed [ACC_W-1:0] acc_r;
  reg                    sat_r;

  // One guard bit, so an overflow is visible rather than silently wrapped.
  wire signed [ACC_W:0] term_ext = {{(ACC_W + 1 - TERM_W) {term[TERM_W-1]}}, term};
  wire signed [ACC_W:0] sum      = {acc_r[ACC_W-1], acc_r} + term_ext;

  // Two's-complement overflow: guard bit disagrees with the sign bit.
  wire ovf = (sum[ACC_W] != sum[ACC_W-1]);

  // Clamp to the rail we ran off: sum[ACC_W]==0 -> +max, ==1 -> -min.
  wire signed [ACC_W-1:0] sum_sat =
      ovf ? {sum[ACC_W], {(ACC_W - 1) {~sum[ACC_W]}}} : sum[ACC_W-1:0];

  // `clear` and `clear_sat` are SEPARATE on purpose. Mode B clears the
  // accumulator between the neurons of one inference but must keep the sticky
  // flag across all of them -- an overflow in hidden neuron 0 has to still be
  // visible when the final output is read. Clearing both together would make
  // the flag report only the last neuron, silently losing every earlier
  // overflow. Mode A drives both at once, so its behaviour is unchanged.
  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      acc_r <= {ACC_W{1'b0}};
      sat_r <= 1'b0;
    end else begin
      if (clear) acc_r <= {ACC_W{1'b0}};
      else if (add_en) acc_r <= sum_sat;

      if (clear_sat) sat_r <= 1'b0;
      else if (add_en) sat_r <= sat_r | ovf;
    end
  end

  assign acc       = acc_r;
  assign saturated = sat_r;

endmodule
