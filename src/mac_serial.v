/*
 * Copyright (c) 2026 Kush Shah
 * SPDX-License-Identifier: Apache-2.0
 *
 * Bit-serial signed multiplier -- shift-and-add, one multiplier bit per cycle.
 *
 * WHY SERIAL HERE
 * ---------------
 * The prior CNN accelerator went the other way and unrolled to four parallel
 * MACs for throughput. Meal data arrives every few minutes, so an 8-cycle
 * multiply costs nothing we care about, and the area goes into the register
 * file instead. Note this is not the dominant area lever -- flops are -- but
 * it is the right call for this workload.
 *
 * ALGORITHM
 * ---------
 * Two's complement shift-add. For a DW-bit signed multiplier b,
 *
 *     b = -b[DW-1] * 2^(DW-1) + sum_{i=0}^{DW-2} b[i] * 2^i
 *
 * so every step adds the multiplicand when the current multiplier bit is set,
 * EXCEPT the final step, where the bit carries negative weight and the
 * multiplicand is subtracted instead. A and Q shift right together as one
 * (DW+1)+DW register: Q feeds multiplier bits out of the bottom while the low
 * half of the product shifts in behind it, so no separate product register is
 * needed. That is the whole reason this costs ~29 flops rather than ~40.
 *
 * Verified exhaustively: all 65,536 signed 8x8 input pairs, in
 * test/test_mac_serial.py. For an operand this narrow a complete proof is
 * cheap, so there is no reason to settle for random sampling.
 */

`default_nettype none

module mac_serial #(
    parameter integer DW = 8
) (
    input  wire                   clk,
    input  wire                   rst_n,
    input  wire                   start,    // 1-cycle pulse; ignored while busy
    input  wire signed [DW-1:0]   a,        // multiplicand
    input  wire signed [DW-1:0]   b,        // multiplier
    output wire signed [2*DW-1:0] product,  // valid while done is high
    output wire                   busy,
    output wire                   done      // 1-cycle pulse
);

  localparam integer CW = $clog2(DW);

  reg signed [DW:0]   acc_hi;  // high accumulator, needs DW+1 bits
  reg        [DW-1:0] q;       // multiplier out the bottom, low product in
  reg signed [DW-1:0] mcand;   // latched multiplicand
  reg        [CW:0]   step;
  reg                 running;
  reg                 done_r;

  wire last = (step == DW - 1);

  wire signed [DW:0] mcand_ext = {mcand[DW-1], mcand};
  wire signed [DW:0] addend    = last ? -mcand_ext : mcand_ext;
  wire signed [DW:0] sum       = q[0] ? (acc_hi + addend) : acc_hi;

  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      acc_hi  <= {(DW + 1) {1'b0}};
      q       <= {DW{1'b0}};
      mcand   <= {DW{1'b0}};
      step    <= {(CW + 1) {1'b0}};
      running <= 1'b0;
      done_r  <= 1'b0;
    end else begin
      done_r <= 1'b0;
      if (!running) begin
        if (start) begin
          acc_hi  <= {(DW + 1) {1'b0}};
          q       <= b;
          mcand   <= a;
          step    <= {(CW + 1) {1'b0}};
          running <= 1'b1;
        end
      end else begin
        // Arithmetic right shift of the combined {sum, q} by one.
        acc_hi <= {sum[DW], sum[DW:1]};
        q      <= {sum[0], q[DW-1:1]};
        step   <= step + 1'b1;
        if (last) begin
          running <= 1'b0;
          done_r  <= 1'b1;
        end
      end
    end
  end

  assign product = {acc_hi[DW-1:0], q};
  assign busy    = running;
  assign done    = done_r;

endmodule
