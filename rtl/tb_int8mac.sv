`timescale 1ps/1ps

module tb_int8mac();

    int err_count;
    int accum_len; 

    // expected values of DUT
    reg signed [31:0] expected_out;
    reg signed [7:0] expected_a_out, expected_b_out;

    // signals to drive
    reg clk, reset, clr, en;
    reg signed [7:0] a_in, b_in;

    // signals to read
    wire signed [31:0] acc_out;
    wire signed [7:0] a_out, b_out;

    int8mac dut(
        .clk(clk),
        .reset(reset),
        .clear(clr),
        .en(en),
        .a_in(a_in),
        .b_in(b_in),
        .a_out(a_out),
        .b_out(b_out),
        .acc(acc_out)
    );

    // Helper tasks
    task reset_task();
    begin
        @(negedge clk);
        { reset, clr, en } = { 0, 0, 0 };
        @(negedge clk);
        reset = 1;
    end 
    endtask

    task check_expected();
    begin
        if (acc_out !== expected_out) begin
            $display("ERROR: out = %d, expected_out = %d", acc_out, expected_out);
            err_count++; 
        end
        if (a_out !== expected_a_out) begin
            $display("ERROR: a_out = %d, expected_a_out = %d", a_out, expected_a_out);
            err_count++; 
        end
        if (b_out !== expected_b_out) begin
            $display("ERROR: b_out = %d, expected_b_out = %d", b_out, expected_b_out);
            err_count++; 
        end
    end
    endtask
    

    // clock driving
    initial begin
        clk = 0;
        forever begin
            #5 clk = ~clk;
        end
    end

    initial begin
        err_count = 0;

        $display("[T1] Reset and check state");
        reset_task();
        expected_out = 0;
        expected_a_out = 0;
        expected_b_out = 0;
        @(negedge clk);
        check_expected();

        $display("[T2] Simple MAC operation");
        reset_task();
        a_in = 10;
        b_in = 10;
        expected_out = 100;
        expected_a_out = 10;
        expected_b_out = 10;
        @(negedge clk);
        en = 1;
        @(negedge clk);
        en = 0;
        check_expected();

        $display("[T3] 200 Repeated randomized MAC tests");
        reset_task();
        for (int s = 0; s < 200; s++) begin
            reset_task();
            expected_out = 0;
            expected_a_out = 0;
            expected_b_out = 0;
            check_expected();

            accum_len = $urandom_range(1, 16);
            for (int i = 0; i < accum_len; i++) begin
                a_in = $urandom;
                b_in = $urandom;
                en = ($urandom_range(0, 3) != 0);

                @(negedge clk);
                if (en) begin
                    expected_out += a_in * b_in;
                    expected_a_out = a_in;
                    expected_b_out = b_in;
                end
                check_expected();
                if (err_count == 0) begin
                    $display("Passed!");
                end
            end
        end

        $display("[T4] -128 * -128 edge case");
        reset_task();
        a_in = -128;
        b_in = -128;
        en = 1;
        @(negedge clk);
        expected_out = -128 * -128;
        expected_a_out = -128;
        expected_b_out = -128;
        check_expected();

        // Final report
        if (err_count == 0) begin
            $display("SUCCESS: All tests passed!");
        end else begin
            $display("FAILURE: %d tests failed.", err_count);
        end
        $finish;
    end
endmodule