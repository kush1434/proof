/*
 * Copyright (c) 2026 Kush Shah
 * SPDX-License-Identifier: Apache-2.0
 *
 * Proof core -- control FSM plus datapath.
 *
 *   Mode A  GL = sum(grams_i * w_i), one dot product, categorised on chip
 *   Mode B  h = ReLU((W1.x + b1) >> s1);  y = (W2.h + b2) >> s2
 *
 * ONE MACHINE, NOT TWO
 * --------------------
 * Every neuron of both layers is the same Mode A dot product. Only two things
 * differ: where the activations come from, and what happens to the result.
 *
 *   Mode A            host supplies activations; result -> GL + category
 *   Mode B layer 1    host supplies activations; result -> requantise, ReLU,
 *                     push into h
 *   Mode B layer 2    CHIP supplies activations from h; result -> read out
 *
 * PROTOCOL
 * --------
 * A neuron is: a shift byte, then its terms, with LAST on the final one.
 *
 *   Mode A / Mode B layer 1   [s|wt] [w0|wt] [a0] [w1|wt] [a1] ... [aN|last]
 *   Mode B layer 2            [s|wt] [v0|wt] [v1|wt] ...   [v9|wt,last]
 *
 * In layer 2 there are no activation bytes at all: each weight byte multiplies
 * the next value the chip supplies -- h[0..7], then the constants 127 and 1.
 * Those two constants exist so a bias is v8*127 + v9*1, exact to the unit for
 * any 15-bit value, with zero bias-specific logic. In layer 1 a bias is just
 * two more ordinary pairs, because there the host controls the activations.
 *
 * `x` IS NOT BUFFERED ON CHIP. The host re-streams it for each hidden neuron:
 * 48 bytes instead of 6, saving 48 flip-flops at 48.99 um2 each. Bandwidth is
 * free at meal timescales. `h` *is* buffered, because layer 2 needs all of it
 * and only the chip can produce it.
 *
 * STARTING AN INFERENCE
 * ---------------------
 * The shift byte carries the requantisation shift, so the host sends s1 for
 * hidden neurons and s2 for output neurons and the chip never stores s2.
 *
 * `LAST` on that shift byte -- otherwise a don't-care -- means "this is a new
 * inference": reset the neuron counter and clear the sticky overflow flag. In
 * Mode A every stream is a new inference, so the bit is ignored there and Mode
 * A behaviour is bit-identical to before Mode B existed.
 *
 * The number of INPUTS is not baked into the silicon -- a neuron ends when the
 * host says LAST. Only the hidden count is structural, being the depth of the
 * h shift register. The output count is not fixed either: the host simply
 * stops streaming.
 *
 * FLOW CONTROL: do not present a byte while `busy`. Bytes offered while busy
 * are ignored, not queued, and `rst_n` recovers the FSM from any state.
 *
 * NOTE: not a medical device. Every output is an estimate.
 */

`default_nettype none

module proof_core #(
    parameter integer DW       = 8,
    parameter integer ACC_W    = 24,
    parameter integer GL_W     = 14,
    parameter integer SHIFT_W  = 5,
    parameter integer N_HIDDEN = 8
) (
    input  wire          clk,
    input  wire          rst_n,
    input  wire          valid,      // payload byte presented this cycle
    input  wire          is_weight,  // 1 = weight/shift, 0 = activation
    input  wire          last,       // final term of this neuron
    input  wire          mode,       // 0 = Mode A, 1 = Mode B (sampled per neuron)
    input  wire [DW-1:0] data,
    input  wire          rd_sel,     // selects the result byte
    output wire [DW-1:0] result,
    output wire          done,       // neuron complete, result valid
    output wire          saturated,  // sticky across the whole inference
    output wire          busy        // do not present a byte while high
);

  localparam integer H_W = N_HIDDEN * DW;  // h shift register width
  localparam integer NW  = 4;              // neuron counter width

  // S_FIN exists because of an alignment bug, and removing it reintroduces
  // one. In S_ACC the accumulator's own `add_en` is high, so `acc` still holds
  // the value from BEFORE the final term -- anything derived from it that
  // cycle (the requantised result, the ReLU, the value pushed into h) is one
  // term short. S_FIN is the cycle after, where `acc` is final. See BUGS.md R-3.
  localparam [2:0] S_IDLE = 3'd0,  // waiting for a neuron's shift byte
                   S_RUN  = 3'd1,  // accepting terms
                   S_MULT = 3'd2,  // multiplier running
                   S_ACC  = 3'd3,  // retiring the product into the accumulator
                   S_FIN  = 3'd4;  // accumulator settled: requantise and retire

  reg [2:0]           state;
  reg [SHIFT_W-1:0]   shift;
  reg signed [DW-1:0] coef;
  reg                 last_r;
  reg                 done_r;
  reg                 mode_r;    // MODE latched for the neuron in flight
  reg [NW-1:0]        neuron;    // saturates at N_HIDDEN, which means "layer 2"
  reg [NW-1:0]        hk;        // term index within a layer-2 neuron
  reg [H_W-1:0]       hreg;      // hidden activations, shift register
  reg                 res_is_h;  // what the result register currently holds

  wire accept   = (state == S_IDLE) || (state == S_RUN);
  wire take     = valid && accept;
  wire take_wt  = take && is_weight;
  wire take_act = take && !is_weight;

  // Layer 2 is "the neuron counter has saturated". Only ever true in Mode B.
  wire layer2    = (neuron == N_HIDDEN[NW-1:0]);
  wire l2_active = mode_r && layer2;

  // A new inference resets the neuron counter and the sticky flag. Mode A
  // always starts one; Mode B starts one only when LAST rides the shift byte.
  wire start_neuron = (state == S_IDLE) && take_wt;
  wire new_inf      = start_neuron && (!mode || last);

  // --------------------------------------------------- ACTIVATION SOURCE ---
  // h[0..7] out of the top of the shift register, then the two bias constants.
  wire [DW-1:0] hsrc = (hk <  N_HIDDEN[NW-1:0]) ? hreg[H_W-1:H_W-DW] :
                       (hk == N_HIDDEN[NW-1:0]) ? {{(DW-7){1'b0}}, 7'd127} :
                                                  {{(DW-1){1'b0}}, 1'b1};

  // ---------------------------------------------------------------- MAC ---
  // Layer 2 has no activation bytes, so a weight byte is both the coefficient
  // and the trigger. Everywhere else the activation byte triggers.
  wire mac_start = (state == S_RUN) && (l2_active ? take_wt : take_act);

  wire [DW-1:0] mac_a = l2_active ? data : coef;
  wire [DW-1:0] mac_b = l2_active ? hsrc : data;

  wire signed [2*DW-1:0] product;
  wire                   mac_done;
  wire                   mac_busy;

  mac_serial #(
      .DW(DW)
  ) u_mac (
      .clk    (clk),
      .rst_n  (rst_n),
      .start  (mac_start),
      // Connected as plain nets, NOT $signed(...): a $signed() cast in a port
      // connection crashes yosys with an internal assertion
      // (arg->is_signed == sig.as_wire()->is_signed, genrtlil.cc:2128). The
      // widths match and mac_serial declares both operands signed, so the bits
      // are interpreted correctly without the cast. See BUGS.md R-1.
      .a      (mac_a),
      .b      (mac_b),
      .product(product),
      .busy   (mac_busy),
      .done   (mac_done)
  );

  // -------------------------------------------------------- ACCUMULATOR ---
  // The accumulator is cleared for every neuron; the sticky flag only for a
  // new inference, so an overflow in hidden neuron 0 is still visible when the
  // final output is read.
  wire add_en = (state == S_ACC);

  wire signed [ACC_W-1:0] acc;

  accumulator #(
      .ACC_W (ACC_W),
      .TERM_W(2 * DW)
  ) u_acc (
      .clk      (clk),
      .rst_n    (rst_n),
      .clear    (start_neuron),
      .clear_sat(new_inf),
      .add_en   (add_en),
      .term     (product),
      .acc      (acc),
      .saturated(saturated)
  );

  // ------------------------------------------- REQUANTISE / CATEGORISE ---
  // Arithmetic shift, floor semantics -- exactly what Python's `>>` does on a
  // negative int, so the reference model is bit-exact with no correction.
  wire signed [ACC_W-1:0] gl_full = acc >>> shift;
  wire [GL_W-1:0] gl = gl_full[GL_W-1:0];

  // Standard per-serving thresholds: low <= 10, medium 11-19, high >= 20.
  wire [1:0] cat = (gl_full >= 20) ? 2'd2 : (gl_full >= 11) ? 2'd1 : 2'd0;

  // ReLU and clamp into the INT8 range h is stored in.
  wire [DW-1:0] h_new = (gl_full < 0)   ? {DW{1'b0}} :
                        (gl_full > 127) ? {{(DW-7){1'b0}}, 7'd127}
                                        : gl_full[DW-1:0];

  wire [15:0] wide = res_is_h ? {8'd0, h_new} : gl_full[15:0];

  // ---------------------------------------------------------------- FSM ---
  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state    <= S_IDLE;
      shift    <= {SHIFT_W{1'b0}};
      coef     <= {DW{1'b0}};
      last_r   <= 1'b0;
      done_r   <= 1'b0;
      mode_r   <= 1'b0;
      neuron   <= {NW{1'b0}};
      hk       <= {NW{1'b0}};
      hreg     <= {H_W{1'b0}};
      res_is_h <= 1'b0;
    end else begin
      case (state)
        S_IDLE: begin
          // The shift byte opens a neuron. An activation here is a protocol
          // error and is ignored.
          if (take_wt) begin
            shift  <= data[SHIFT_W-1:0];
            mode_r <= mode;
            hk     <= {NW{1'b0}};
            done_r <= 1'b0;
            state  <= S_RUN;
            if (!mode || last) neuron <= {NW{1'b0}};
          end
        end

        S_RUN: begin
          if (l2_active) begin
            // Weight bytes only. Each one multiplies the next supplied value
            // and walks h round by one place.
            if (take_wt) begin
              hk     <= hk + 1'b1;
              last_r <= last;
              state  <= S_MULT;
              if (hk < N_HIDDEN[NW-1:0])
                hreg <= {hreg[H_W-DW-1:0], hreg[H_W-1:H_W-DW]};
            end
          end else if (take_wt) begin
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
          // `acc` is one term short until the next edge, so the neuron cannot
          // be retired here -- that is what S_FIN is for.
          state <= last_r ? S_FIN : S_RUN;
        end

        S_FIN: begin
          // Accumulator settled. A hidden neuron pushes its result into h and
          // advances the counter; an output neuron leaves both alone.
          res_is_h <= mode_r && !layer2;
          if (mode_r && !layer2) begin
            hreg   <= {hreg[H_W-DW-1:0], h_new};
            neuron <= neuron + 1'b1;
          end
          done_r <= 1'b1;
          state  <= S_IDLE;
        end

        default: state <= S_IDLE;
      endcase
    end
  end

  // ------------------------------------------------------------ OUTPUTS ---
  assign result = mode_r ? (rd_sel ? wide[15:8] : wide[7:0])
                         : (rd_sel ? {cat, gl[GL_W-1:DW]} : gl[DW-1:0]);
  assign done   = done_r;
  assign busy   = (state == S_MULT) || (state == S_ACC) || (state == S_FIN);

  // mac_busy is redundant with the FSM's own state and is not used.
  wire _unused = &{mac_busy, 1'b0};

endmodule
