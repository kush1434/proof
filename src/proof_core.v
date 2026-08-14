/*
 * Copyright (c) 2026 Kush Shah
 * SPDX-License-Identifier: Apache-2.0
 *
 * Proof core -- control FSM plus datapath, Mode A.
 *
 * PROTOCOL
 * --------
 * A stream is a sequence of tagged bytes. `is_weight` distinguishes them:
 *
 *   [ shift | wt ] [ w0 | wt ] [ g0 ] [ w1 | wt ] [ g1 ] ... [ gN | last ]
 *
 * The FIRST weight-tagged byte of a stream carries the requantisation shift
 * and begins a new inference (clearing the accumulator and the sticky
 * overflow flag). After that, a weight-tagged byte latches a coefficient and
 * an activation byte multiplies it and accumulates. `last` on an activation
 * ends the inference.
 *
 * Weights and activations interleave rather than arriving as two blocks
 * because Mode A takes an unbounded number of ingredients: buffering all the
 * coefficients first would need unbounded storage. Interleaving needs exactly
 * one coefficient register.
 *
 * The coefficient is sticky -- an activation with no fresh weight in front of
 * it reuses the previous one. That is deliberate and lets the host stream a
 * constant-weight dot product cheaply.
 *
 * FLOW CONTROL: the host must not present a byte while `busy` is high. Bytes
 * offered while busy are ignored, not queued. A truncated, stuck or
 * mis-tagged stream cannot wedge the chip -- the FSM only ever advances on
 * bytes it accepts, and `rst_n` returns it to S_IDLE from any state.
 *
 * OUTPUT PACKING
 * --------------
 * Eight output pins and a one-bit read select give 16 bits total, so rather
 * than spend a byte on a 2-bit category it is packed above the figure:
 *
 *   rd_sel = 0 -> gl[7:0]
 *   rd_sel = 1 -> { cat[1:0], gl[13:8] }
 *
 * GL_W = 14 carries up to 16,383, and "high" starts at 20, so no real meal
 * comes close to the limit.
 *
 * NOTE: Mode B is not implemented yet. The MODE pin is declared in the pinout
 * and is currently unused; see BUGS.md.
 */

`default_nettype none

module proof_core #(
    parameter integer DW      = 8,
    parameter integer ACC_W   = 24,
    parameter integer GL_W    = 14,
    parameter integer SHIFT_W = 5
) (
    input  wire          clk,
    input  wire          rst_n,
    input  wire          valid,      // payload byte presented this cycle
    input  wire          is_weight,  // 1 = weight/shift, 0 = activation
    input  wire          last,       // final activation of the stream
    input  wire [DW-1:0] data,
    input  wire          rd_sel,     // selects the result byte
    output wire [DW-1:0] result,
    output wire          done,       // inference complete, result valid
    output wire          saturated,  // sticky overflow
    output wire          busy        // do not present a byte while high
);

  localparam [1:0] S_IDLE = 2'd0,  // waiting for a stream's shift byte
                   S_RUN  = 2'd1,  // accepting weights and activations
                   S_MULT = 2'd2,  // multiplier running
                   S_ACC  = 2'd3;  // retiring the product into the accumulator

  reg [1:0]           state;
  reg [SHIFT_W-1:0]   shift;
  reg signed [DW-1:0] coef;
  reg                 last_r;
  reg                 done_r;

  // A byte is taken only in a state that accepts one; anything else is ignored.
  wire accept   = (state == S_IDLE) || (state == S_RUN);
  wire take     = valid && accept;
  wire take_wt  = take && is_weight;
  wire take_act = take && !is_weight;

  // ---------------------------------------------------------------- MAC ---
  wire signed [2*DW-1:0] product;
  wire                   mac_done;
  wire                   mac_busy;
  wire                   mac_start = (state == S_RUN) && take_act;

  mac_serial #(
      .DW(DW)
  ) u_mac (
      .clk    (clk),
      .rst_n  (rst_n),
      .start  (mac_start),
      .a      (coef),
      // Connected as a plain net, NOT as $signed(data): a $signed() cast in a
      // port connection crashes yosys with an internal assertion
      // (`arg->is_signed == sig.as_wire()->is_signed`, genrtlil.cc:2128).
      // The widths match and mac_serial declares `b` signed, so the bits pass
      // through and are interpreted correctly without the cast.
      .b      (data),
      .product(product),
      .busy   (mac_busy),
      .done   (mac_done)
  );

  // -------------------------------------------------------- ACCUMULATOR ---
  wire clear_acc = (state == S_IDLE) && take_wt;  // a new stream starts here
  wire add_en    = (state == S_ACC);

  wire signed [ACC_W-1:0] acc;

  accumulator #(
      .ACC_W (ACC_W),
      .TERM_W(2 * DW)
  ) u_acc (
      .clk      (clk),
      .rst_n    (rst_n),
      .clear    (clear_acc),
      .add_en   (add_en),
      .term     (product),
      .acc      (acc),
      .saturated(saturated)
  );

  // ---------------------------------------------------------------- FSM ---
  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state  <= S_IDLE;
      shift  <= {SHIFT_W{1'b0}};
      coef   <= {DW{1'b0}};
      last_r <= 1'b0;
      done_r <= 1'b0;
    end else begin
      case (state)
        S_IDLE: begin
          // First weight-tagged byte of a stream carries the shift. An
          // activation here is a protocol error and is ignored.
          if (take_wt) begin
            shift  <= data[SHIFT_W-1:0];
            done_r <= 1'b0;
            state  <= S_RUN;
          end
        end

        S_RUN: begin
          if (take_wt) begin
            coef <= data;
          end else if (take_act) begin
            last_r <= last;
            state  <= S_MULT;
          end
        end

        S_MULT: begin
          if (mac_done) state <= S_ACC;
        end

        S_ACC: begin
          if (last_r) begin
            done_r <= 1'b1;
            state  <= S_IDLE;
          end else begin
            state <= S_RUN;
          end
        end

        default: state <= S_IDLE;
      endcase
    end
  end

  // ------------------------------------------- REQUANTISE / CATEGORISE ---
  // Arithmetic shift: floors toward negative infinity, which is exactly what
  // the Python reference's `>>` does on a negative int. Bit-exact by
  // construction, no rounding correction needed on either side.
  wire signed [ACC_W-1:0] gl_full = acc >>> shift;

  // Unsigned on purpose: `gl` is only ever bit-extracted for the output
  // packing, and a part-select is unsigned anyway. The category below is
  // computed from the full-width signed value, which is what actually matters.
  wire [GL_W-1:0] gl = gl_full[GL_W-1:0];

  // Standard per-serving thresholds: low <= 10, medium 11-19, high >= 20.
  // Categorised from the full-width figure, not the truncated output field.
  wire [1:0] cat = (gl_full >= 20) ? 2'd2 : (gl_full >= 11) ? 2'd1 : 2'd0;

  assign result = rd_sel ? {cat, gl[GL_W-1:DW]} : gl[DW-1:0];
  assign done   = done_r;
  assign busy   = (state == S_MULT) || (state == S_ACC);

  // mac_busy is redundant with the FSM's own state and is not used.
  wire _unused = &{mac_busy, 1'b0};

endmodule
