`timescale 1ns/1ps

module int8mac(input clk, input reset_n, input clear, input en, 
                            input signed [7:0] a_in, 
                            input signed [7:0] b_in,
                            output reg signed [7:0] a_out,
                            output reg signed [7:0] b_out,
                            output reg signed [31:0] acc);    
    always_ff @(posedge clk) begin
        if (~reset_n) begin
            acc <= '0;
            a_out <= '0;
            b_out <= '0;
        end
        else begin
            if (en) begin
                a_out <= a_in;
                b_out <= b_in;
            end
            if (clear) acc <= '0;
            else if (en) acc <= acc + a_in * b_in;
        end
    end
endmodule