`default_nettype none
`timescale 1ns / 1ps

/* Unit testbench for mac_serial. Instantiates the DUT and exposes convenient
   wires for test_mac_serial.py to drive. */
module tb_mac_serial ();

  initial begin
    $dumpfile("tb_mac_serial.fst");
    $dumpvars(0, tb_mac_serial);
    #1;
  end

  localparam integer DW = 8;

  reg                    clk;
  reg                    rst_n;
  reg                    start;
  reg  signed [DW-1:0]   a;
  reg  signed [DW-1:0]   b;
  wire signed [2*DW-1:0] product;
  wire                   busy;
  wire                   done;

  mac_serial #(
      .DW(DW)
  ) dut (
      .clk    (clk),
      .rst_n  (rst_n),
      .start  (start),
      .a      (a),
      .b      (b),
      .product(product),
      .busy   (busy),
      .done   (done)
  );

endmodule
