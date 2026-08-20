// SPDX-FileCopyrightText: (c) 2026 Kush Shah
// SPDX-License-Identifier: Apache-2.0
//
// Machine-checks the load-bearing lemma of the monotonicity argument:
// **a saturating accumulator is monotone in its terms.**
//
// RESULTS.md 3 argues this by hand -- "with partial sums S_i = clamp(S_{i-1} +
// t_i), clamp is non-decreasing and S_i depends monotonically on S_{i-1}, so by
// induction the final sum is non-decreasing in any single term" -- and notes
// that a *wrapping* sum would destroy it. That argument is correct and it is
// about an idealised pipeline. R-4 is this project's own evidence that such an
// argument can be right about the datapath and still miss what the hardware
// does at its edges: the datapath was provably monotone while the reported
// output field truncated, and truncation wraps.
//
// So this proves the lemma about `src/accumulator.v` itself, not about a model
// of it. Nothing here is a new claim -- it is the existing claim, checked by a
// solver instead of by reading.
//
// SHAPE OF THE PROOF
// ------------------
// Monotonicity relates two executions, so it is a 2-safety property and cannot
// be written as an assertion on one copy of the design. This is the standard
// miter: two instances driven by identical control, with the only difference
// being that b's term always dominates a's. If the accumulator is monotone,
// b's accumulator can never fall below a's.
//
// The property is inductive -- assume acc_a <= acc_b and term_a <= term_b,
// then the 25-bit guarded sums satisfy sum_a <= sum_b, and clamping is
// non-decreasing, so the next state preserves it -- which is why this runs in
// `mode prove` (k-induction, unbounded) rather than `mode bmc` (bounded).
// An unbounded result here is a real proof for all inputs and all time, not a
// statement about the first N cycles.

`default_nettype none

module monotone_acc #(
    parameter integer ACC_W  = 24,
    parameter integer TERM_W = 16
) (
    input wire                     clk,
    input wire                     rst_n,
    input wire                     clear,
    input wire                     clear_sat,
    input wire                     add_en,
    input wire signed [TERM_W-1:0] term_a,
    input wire signed [TERM_W-1:0] term_b
);

  wire signed [ACC_W-1:0] acc_a, acc_b;
  wire                    sat_a, sat_b;

  // Identical control, identical clock and reset. The ONLY asymmetry allowed
  // is the term, constrained below. If the two instances were given different
  // control the property would be false for uninteresting reasons.
  accumulator #(.ACC_W(ACC_W), .TERM_W(TERM_W)) u_a (
      .clk(clk), .rst_n(rst_n), .clear(clear), .clear_sat(clear_sat),
      .add_en(add_en), .term(term_a), .acc(acc_a), .saturated(sat_a));

  accumulator #(.ACC_W(ACC_W), .TERM_W(TERM_W)) u_b (
      .clk(clk), .rst_n(rst_n), .clear(clear), .clear_sat(clear_sat),
      .add_en(add_en), .term(term_b), .acc(acc_b), .saturated(sat_b));

  // The hypothesis: b's term dominates a's, every cycle, signed.
  always @(*) assume (term_a <= term_b);

  // Reset is asynchronous, so let the solver choose when it is asserted but
  // require that it has happened at least once -- otherwise the induction step
  // starts from a state in which the two accumulators were never initialised
  // together and the property is vacuously breakable.
  reg started;
  initial started = 1'b0;
  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) started <= 1'b1;
  end

  // The conclusion.
  always @(*) begin
    if (started) assert (acc_a <= acc_b);
  end

  // A proof that cannot fail is worth nothing, and this one closes in about a
  // second, which is exactly what a vacuous proof looks like. `mutate.sby`
  // runs the identical property against an accumulator whose only change is
  // that it *wraps* instead of clamping -- the R-4 defect class -- and
  // requires a counterexample. Do not trust the PASS without that FAIL.

endmodule
