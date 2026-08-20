// SPDX-FileCopyrightText: (c) 2026 Kush Shah
// SPDX-License-Identifier: Apache-2.0
//
// Machine-checks monotonicity of the value the HOST READS, end to end through
// `src/proof_core.v` -- the place R-4 actually broke.
//
// monotone_acc.v proves the accumulator lemma. That was never where the bug
// was. R-4 is the case where the datapath was provably monotone and the
// *reported* field truncated: reported y fell 31,293 -> -31,209 while the true
// value rose 31,293 -> 34,327. The fix was to saturate the field, and
// proof_core.v carries a comment saying clamping is monotone "so saturating
// the field preserves the property end to end". This checks that sentence.
//
// SCOPE, STATED NARROWLY ON PURPOSE
// ---------------------------------
// One control sequence: a single-term Mode A inference -- shift byte, weight
// byte, activation byte -- driven at fixed cycles. Within that sequence the
// proof is over ALL shift values, ALL non-negative weights and ALL activation
// pairs, because those are `anyconst` and the solver picks them.
//
// It is NOT a proof over arbitrary control, arbitrary stream lengths, or Mode
// B. Multi-term streams and the 6-8-1 network are out of reach for bounded
// model checking here: a Mode B inference is 896 cycles and this is a 2-safety
// property, so it would be two copies of the design over that horizon.
//
// The hypothesis is the sign condition specialised to one term: for a single
// Mode A term, y = (w * x) >> s is monotone in x exactly when w >= 0. That is
// assumed, not proved -- it is the precondition the on-chip guard exists to
// check, and checking it is the chip's job, not this proof's.

`default_nettype none

module monotone_field #(
    parameter integer DW      = 8,
    parameter integer GL_W    = 14,
    parameter integer SHIFT_W = 5
) (
    input wire clk
);

  // Reset is GENERATED here, not taken as a free input. Left free, the solver
  // simply never asserts it and starts the two copies from different arbitrary
  // states -- which shows up as identical inputs producing different outputs,
  // the classic uninitialised-state counterexample. That was this harness's
  // first result and it was a bug in the harness, not in the chip.
  wire rst_n = (t >= 6'd3);

  // Free but constant across the run: the solver picks one of each and must
  // make the assertion hold for every choice.
  (* anyconst *) reg [SHIFT_W-1:0]   shift_v;
  (* anyconst *) reg signed [DW-1:0] w_v;
  (* anyconst *) reg signed [DW-1:0] xa_v;
  (* anyconst *) reg signed [DW-1:0] xb_v;

  // The precondition. w >= 0 is the sign condition for a single Mode A term;
  // xa <= xb is the hypothesis monotonicity is stated against.
  always @(*) assume (w_v >= 0);
  always @(*) assume (xa_v <= xb_v);

  // Fixed protocol timing. Deterministic, so the capture cycles below are
  // deterministic too -- and asserted, not assumed, so a wrong guess about
  // when `done` arrives fails the proof instead of quietly skipping it.
  reg [5:0] t;
  initial t = 6'd0;
  always @(posedge clk) if (t != 6'd63) t <= t + 6'd1;

  localparam [5:0] T_SHIFT = 6'd5;   // shift byte opens the neuron
  localparam [5:0] T_W     = 6'd6;   // its one weight
  localparam [5:0] T_X     = 6'd7;   // its one activation, carrying LAST
  localparam [5:0] T_LO    = 6'd22;  // read the low result byte
  localparam [5:0] T_HI    = 6'd23;  // then the high one

  wire valid     = (t == T_SHIFT) || (t == T_W) || (t == T_X);
  wire is_weight = (t == T_SHIFT) || (t == T_W);
  wire last      = (t == T_X);
  wire rd_sel    = (t == T_HI);

  wire [DW-1:0] data_common = (t == T_SHIFT) ? {{(DW-SHIFT_W){1'b0}}, shift_v}
                                             : w_v;

  // The ONLY asymmetry between the two copies.
  wire [DW-1:0] data_a = (t == T_X) ? xa_v : data_common;
  wire [DW-1:0] data_b = (t == T_X) ? xb_v : data_common;

  wire [DW-1:0] res_a, res_b;
  wire done_a, done_b, untr_a, untr_b, busy_a, busy_b;

  proof_core u_a (
      .clk(clk), .rst_n(rst_n), .valid(valid), .is_weight(is_weight),
      .last(last), .mode(1'b0), .data(data_a), .rd_sel(rd_sel),
      .result(res_a), .done(done_a), .untrusted(untr_a), .busy(busy_a));

  proof_core u_b (
      .clk(clk), .rst_n(rst_n), .valid(valid), .is_weight(is_weight),
      .last(last), .mode(1'b0), .data(data_b), .rd_sel(rd_sel),
      .result(res_b), .done(done_b), .untrusted(untr_b), .busy(busy_b));

  // Capture the two reported bytes as the host would.
  reg [DW-1:0] lo_a, hi_a, lo_b, hi_b;
  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      lo_a <= {DW{1'b0}}; hi_a <= {DW{1'b0}};
      lo_b <= {DW{1'b0}}; hi_b <= {DW{1'b0}};
    end else begin
      if (t == T_LO) begin lo_a <= res_a; lo_b <= res_b; end
      if (t == T_HI) begin hi_a <= res_a; hi_b <= res_b; end
    end
  end

  // The inference must actually be finished when we read it. Asserted so that
  // a mis-timed capture is a failure rather than a vacuous pass.
  always @(*) begin
    if (t == T_LO || t == T_HI) begin
      assert (done_a);
      assert (done_b);
    end
  end

  // Reconstruct what the host reconstructs. The high byte is {cat, gl[13:8]},
  // so the value's top bits are hi[5:0]; the category rides in hi[7:6].
  wire signed [GL_W-1:0] gl_a = $signed({hi_a[GL_W-DW-1:0], lo_a});
  wire signed [GL_W-1:0] gl_b = $signed({hi_b[GL_W-DW-1:0], lo_b});
  wire [1:0] cat_a = hi_a[DW-1:DW-2];
  wire [1:0] cat_b = hi_b[DW-1:DW-2];

  // THE PROPERTY. More carbohydrate must never report less response.
  always @(*) begin
    if (t > T_HI) begin
      assert (gl_a <= gl_b);
      // The category is a monotone function of the same value, so it must not
      // invert either. This is the number a user would actually be shown.
      assert (cat_a <= cat_b);
    end
  end

endmodule
