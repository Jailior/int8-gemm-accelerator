# INT8 GEMM Accelerator

An INT8 matrix-multiply (GEMM) accelerator built around an output-stationary systolic array
on the Terasic DE1-SoC (Intel Cyclone V SoC), compared against CUDA GEMM kernels on an
NVIDIA RTX 4060.

The same computation is built at every level: a NumPy golden model, a cycle model of the
array, SystemVerilog RTL, the accelerator running on the board behind the ARM processor,
and a ladder of CUDA kernels. Every level is checked against the level above it. INT8 × INT8
products summed in INT32 are exact, so every check is bit-exact rather than within a
tolerance.

```
C[M×N] (int32)  =  A[M×K] (int8)  ·  B[K×N] (int8)
```

**Toolchain:** SystemVerilog; Questa for local simulation; Icarus Verilog and Verilator in
CI; Quartus Prime Lite and Platform Designer; Python + NumPy; CUDA and Nsight Compute.

## What it measures

The question is how each platform turns its multipliers into GEMM throughput, and what
stops it. The RTX 4060 has orders of magnitude more INT8 throughput than a Cyclone V, so
the comparison is each platform against its own measured ceiling, not raw speed.

On the FPGA the expected limit is memory bandwidth. Each cycle, an N×N output-stationary
array does N² MACs on 2N fresh operand bytes, N/2 MACs per byte. Compute grows with N²,
demand for bandwidth grows with N, and the HPS memory bandwidth is fixed, so past some N
adding PEs stops helping. The planned experiment sweeps N, reads the accelerator's
load-stall counter to find that point, then moves it by reusing blocks of A and B from
on-chip memory across several C tiles. Shared-memory tiling on the GPU is the same reuse
idea in software, and the CUDA kernel ladder measures it the same way.

Results: not measured yet.

## Milestones

| # | Milestone | Scope | Status | Checked against |
|---|---|---|---|---|
| M1 | Golden model | Random INT8 A and B, `C = A @ B` in int32; `.hex` vectors for RTL, `.bin` for C and CUDA | Not started | Itself, plus one hand-checked entry |
| M2 | Cycle model | Skewed schedule, tile order and drain; predicted cycles for any M, N, K and array size | Not started | M1 values; the `2N + K − 2` closed form |
| M3 | CUDA naive | One thread per C element | Not started | M1, bit-exact |
| M4 | CUDA tiled | Tiles of A and B staged in shared memory | Not started | M1, bit-exact |
| M5 | RTL compute core | PE, parameterized N×N array, RTL skew feeders, drain path | In progress | M1, bit-exact; cycles equal M2 |
| M6 | RTL GEMM engine | M10K buffers, tile controller, overlapping tiles, any M, N, K | Not started | M1, bit-exact, including padded shapes; cycles equal M2 |
| M7 | Host link | Avalon-MM control registers with phase counters, Avalon-ST in and out, mSGDMA, C driver on the HPS | Not started | Integration testbench; driver maths checked offline |
| M8 | DE1-SoC bring-up | Quartus builds, N sweep, timing closure, runs on the board | Not started | Timing Analyzer; board output vs M1 |
| M9 | Fast CUDA kernels | Register blocking, `__dp4a`, INT8 tensor cores, cuBLASLt baseline | Not started | M1, bit-exact; Nsight Compute |
| M10 | FPGA vs GPU | Same shapes, measured rooflines for both platforms | Not started | — |

### Current work

**M5: RTL compute core**

- [x] Output-stationary dataflow and 4×4 timing sketch (Oct 14, 2026)
- [x] PE RTL ([rtl/int8mac.sv](rtl/int8mac.sv))
- [x] PE testbench passes in Questa ([rtl/tb_int8mac.sv](rtl/tb_int8mac.sv)): directed
      tests, 200 random sequences with stalls and `clear` between them, and −128 × −128
      forced mid-sequence
- [ ] Parameterized N×N array
- [ ] RTL skew feeders
- [ ] Drain path
- [ ] Array testbench passes 1,000 random tiles from M1, with cycle counts equal to M2

**M1: Golden model**

- [ ] Random signed INT8 A and B for a given shape and seed; `C = A.astype(int32) @ B.astype(int32)`
- [ ] `.hex` files for `$readmemh`: row-major, one value per line, two's complement (2 hex
      digits per operand, 8 per result)
- [ ] `.bin` files for the HPS driver and the CUDA kernels
- [ ] Hand-check one entry of C, including decoding a negative value from hex

## How it is verified

- **Golden model (M1).** The NumPy result is the reference for every other level, and the
  vectors it writes are what the testbenches load, so software and hardware cannot drift
  apart.
- **Cycle model (M2).** Simulates the skewed schedule, the tile order and the drain, and
  predicts the cycle count of each phase. The RTL has to match it on cycles as well as on
  values, which is what catches off-by-one errors in the skew and the drain.
- **Zero tolerance.** Every comparison is exact integer equality. Integer sums come out the
  same in any order, so even the tensor-core kernels have to match bit for bit.
- **Shapes that small examples hide.** Non-square shapes (M ≠ N ≠ K) so swapped indices
  fail, sizes that are not a multiple of the array size (such as 300 on an 8×8 array) so the
  padded last tile is exercised, and −128 × −128 over long K for the accumulator.
- **Self-checking testbenches.** Each one ends with a single PASS or FAIL line.

## Architecture

### Compute core

The array is an N×N grid of identical processing elements (PEs). It is
**output-stationary**: each PE(i,j) owns one output `C[i][j]` and keeps its partial sum in
place while A and B stream past it. The first target is N = 4.

```
              B col 0   B col 1   B col 2   B col 3
              (+0)      (+1)      (+2)      (+3)     <- column j delayed j cycles
                |         |         |         |
                v         v         v         v
A row 0 (+0) -> PE00 ---> PE01 ---> PE02 ---> PE03
                |         |         |         |
                v         v         v         v
A row 1 (+1) -> PE10 ---> PE11 ---> PE12 ---> PE13
                |         |         |         |
                v         v         v         v
A row 2 (+2) -> PE20 ---> PE21 ---> PE22 ---> PE23
                |         |         |         |
                v         v         v         v
A row 3 (+3) -> PE30 ---> PE31 ---> PE32 ---> PE33
```

- **A flows right.** Row i of A enters PE(i,0) from the left, delayed i cycles. Each PE
  passes its `a` value to its right neighbour through a register.
- **B flows down.** Column j of B enters PE(0,j) from the top, delayed j cycles. Each PE
  passes its `b` value down through a register.
- **Each PE accumulates** the pair it sees on every enabled cycle: `acc += a × b`.

The input skew and the one-register-per-hop delay line up so that `a[i][k]` and `b[k][j]`
meet at PE(i,j) on the same cycle:

```
t(i, j, k) = i + j + k
```

In general, one N×N×K tile takes `2N + K − 2` cycles before its results are drained.

### Processing element

The PE (`int8mac`, [rtl/int8mac.sv](rtl/int8mac.sv)) is one multiply-accumulate cell plus
the two pass-through registers that carry operands to its neighbours. It is the only module
the array is tiled from.

| Port             | Dir | Type              | Meaning                                            |
| ---------------- | --- | ----------------- | -------------------------------------------------- |
| `clk`            | in  | 1 bit             | Every register updates on the rising edge          |
| `reset_n`        | in  | 1 bit, active low | Zeroes `acc`, `a_out`, `b_out`                     |
| `clear`          | in  | 1 bit             | Zeroes `acc` between tiles; overrides `en`         |
| `en`             | in  | 1 bit             | Global clock enable: when low, the whole PE holds  |
| `a_in`, `b_in`   | in  | `signed [7:0]`    | Operands from the left / top neighbour             |
| `a_out`, `b_out` | out | `signed [7:0]`    | Registered copies, to the right / bottom neighbour |
| `acc`            | out | `signed [31:0]`   | Running sum of `a × b`                             |

### System (planned)

```
HPS DDR3 --F2H SDRAM--> mSGDMA (MM to ST) --Avalon-ST--> gemm_top --Avalon-ST--> mSGDMA (ST to MM) --F2H SDRAM--> HPS DDR3
Cortex-A9 --lightweight HPS-to-FPGA bridge--> gemm_csr (Avalon-MM)
gemm_top irq --> HPS f2h_irq
```

A C driver on the HPS packs A and B into tile order, starts the DMA, and reads the results
and counters back. The accelerator stays a stream processor; an address-generating read
master is a later optimization.

Control and status registers (32-bit, byte addressed). Matrix shapes are runtime registers;
the array size is fixed at synthesis and reported in `PARAM` so the driver can refuse to run
against a mismatched bitstream.

| Offset | Name              | Access | Meaning                                           |
| ------ | ----------------- | ------ | ------------------------------------------------- |
| `0x00` | `CTRL`            | RW     | [0] start, [1] irq_enable, [2] soft_reset         |
| `0x04` | `STATUS`          | RO     | [0] busy, [1] done, [2] irq_pending, [3] error    |
| `0x08` | `ISR`             | W1C    | Done interrupt; write 1 to clear                  |
| `0x0C` | `DIM_M`           | RW     | Rows of A and C                                   |
| `0x10` | `DIM_N`           | RW     | Columns of B and C                                |
| `0x14` | `DIM_K`           | RW     | Columns of A, rows of B                           |
| `0x18` | `CYC_TOTAL`       | RO     | Cycles from start to done, counted independently  |
| `0x1C` | `CYC_LOAD`        | RO     | Cycles streaming operands in                      |
| `0x20` | `CYC_LOAD_STALL`  | RO     | ...of which the input stream had no data          |
| `0x24` | `CYC_COMPUTE`     | RO     | Cycles the array was enabled                      |
| `0x28` | `CYC_DRAIN`       | RO     | Cycles draining results out                       |
| `0x2C` | `CYC_STORE_STALL` | RO     | ...of which the output stream was backpressured   |
| `0x30` | `PARAM`           | RO     | Array size N, buffer depths                       |
| `0x34` | `BUILD_ID`        | RO     | Fixed build identifier                            |

`CYC_LOAD_STALL` is the counter that measures the bandwidth wall directly: if it grows with
N, the DMA is starving the array and more PEs are wasted.

### Design decisions

| Decision | Choice | Why |
| -------- | ------ | --- |
| Dataflow | Output-stationary | Partial sums never move, so there is no 32-bit traffic between PEs, only the 8-bit operands. Weight-stationary (the TPU's choice) keeps B in the PEs and moves partial sums instead; it is a good comparison point for later. |
| What `en` gates | `acc` and `a_out` / `b_out` | A stall must freeze the whole array. If the pass-through registers kept shifting while `acc` held, an operand would leave a PE before being multiplied and that product would be lost silently. With one global `en`, a stall just shifts the schedule by one cycle. |
| Clock enable vs gated clock | Clock enable | Gating the clock in FPGA logic invites glitches and skew. Cyclone V flip-flops have a built-in enable pin, so `if (en)` costs nothing. |
| `clear` and `en` on the same edge | `clear` wins; that cycle's product is dropped | Simplest to reason about. The alternative, `acc <= a_in * b_in` on clear, saves a cycle per tile and is worth revisiting when tiles overlap. |
| Operand width | Signed INT8 | The common inference format, and it maps onto the Cyclone V DSP blocks. |
| Accumulator width | Signed 32-bit | The largest single product is 2^14 (−128 × −128 = 16384), so 32 bits cannot overflow for any K below 2^17 = 131,072. |
| Reset | Active low | Matches the board convention. |
| Bubbles | One global `en` for now | A valid bit travelling with the data would let bubbles flow through the array, but the feeders would have to keep `a` and `b` bubbles aligned. |
| Input skew | RTL feeders | The accelerator on the board has no testbench to generate the i / j delays. |
| Matrix shapes | M, N, K at runtime; array size at synthesis | One bitstream serves every shape; the controller tiles and pads. |
| Data feed | Host packs tiles, DMA streams them | Keeps the accelerator a stream processor and the first DMA setup simple. |
| Simulators | Questa locally, Icarus and Verilator in CI | Questa needs a license that CI does not have, so the RTL and testbenches stay within what Icarus accepts. |

### Open decisions

- **Drain path:** shift the results out through the PEs, or mux over all accumulators? To be
  decided before the array is built.
- **Tile overlap:** if the next tile's inputs overlap the current tile's tail, `clear`
  switches to loading the first product (`acc <= a_in * b_in`) instead of zeroing.
- **DSP mapping:** the Cyclone V has 87 DSP blocks. If Quartus packs three 8×8 multiplies
  per block the array can reach about 16×16; at one per block it stops near 9×9.
- **Reuse buffer size:** how large a block of A and B to hold in M10K. This sets the FPGA's
  MACs per byte, and so where the bandwidth wall sits.

## GPU side

Each kernel is checked bit-exact against M1 and profiled with Nsight Compute; the next
kernel targets the limit the profile names.

| Kernel                 | What changes                                                  |
| ---------------------- | ------------------------------------------------------------- |
| `01_gemm_naive.cu`     | One thread per C element                                      |
| `02_gemm_tiled.cu`     | Tiles of A and B staged in shared memory                      |
| `03_gemm_regblock.cu`  | Each thread computes a small block of C in registers          |
| `04_gemm_dp4a.cu`      | `__dp4a`: four INT8 products summed in one instruction        |
| `05_gemm_wmma.cu`      | INT8 tensor cores through WMMA, int32 accumulation            |
| `06_gemm_cublaslt.cu`  | cuBLASLt INT8 GEMM: the baseline and the measured ceiling     |

## FPGA vs GPU

- **Same work.** The same shapes on both platforms (a square sweep plus non-square cases),
  with the same INT8 × INT8 → INT32 semantics and the same M1 vectors.
- **Measured ceilings.** GPU compute from cuBLASLt at a large size and DRAM bandwidth from a
  read-only stream; FPGA compute as N² × Fmax for the built configuration and bandwidth from a
  DMA-only run.
- **Reported numbers.** Kernel-only and end-to-end time (host-to-device copies on the GPU,
  DMA on the FPGA), percent of the measured ceiling, and each kernel's position on its
  platform's roofline.

Results: not measured yet.

## Bugs worth remembering

- **`!=` lets X pass.** A comparison against X gives X, and `if` treats X as false, so an
  uninitialised output passes silently. The testbenches compare with `!==`.
- **`$urandom_range` takes unsigned bounds.** Negative bounds do not work. Random signed
  operands come from `a_in = $urandom;`, which keeps the low 8 bits.
- **An off-by-one reference passes a broken design.** The expected value has to update on
  the same enabled edge as the DUT, not one loop iteration later.
- **A bare `run` in Questa stops after 100 units of the time resolution**, which is 100 ps
  here, long before the test finishes. Changing the timescale to `1ns/1ps` does not help,
  because the resolution is still 1 ps. Use `run -all`.
- **A missing `begin`/`end` disabled the stall test.** `if (cond) a_in = -128; b_in = -128;
  en = 1;` only makes the first assignment conditional, so every random cycle ran with
  `b_in = -128` and `en = 1`. The test still passed; it just no longer tested stalls or
  random B operands.

## Layout

Only `rtl/int8mac.sv` and `rtl/tb_int8mac.sv` exist so far; the rest is the plan.

```
python/
  01_golden_model.py      M1, writes rtl/vectors/*.hex and data/*.bin
  02_cycle_model.py       M2, predicted cycles for any shape and array size
  03_link_budget.py       bandwidth needed vs available, per N and Fmax
rtl/
  int8mac.sv              PE
  systolic_array.sv       N×N grid
  skew_feeder.sv          i / j input delays
  drain.sv                result path out of the array
  gemm_engine.sv          tile controller and M10K buffers
  gemm_csr.sv             Avalon-MM control, status and counters
  gemm_top.sv             Avalon-MM and Avalon-ST wrapper
  tb_*.sv                 a self-checking testbench next to each module
  vectors/                golden .hex files from M1
cuda/
  01_gemm_naive.cu ... 06_gemm_cublaslt.cu
hps/
  gemm.c, gemm.h          map the registers, run, read the counters
  bench.c                 shape sweep, checked against M1
fpga/
  README.md               build steps and the results table
  build.tcl, sweep.sh     Quartus flow for one configuration, and the N sweep
docs/
  project_log.md          dated log of bugs and measurements
  figures/                rooflines and scaling plots, generated by scripts
run_all.sh                every check that needs no GPU or board
preflight.sh              toolchain check
.github/workflows/ci.yml  runs run_all.sh on every push
```

## Claims I am not making

- The FPGA is not faster than the GPU in absolute terms. The comparison is each platform
  against its own measured ceiling.
- Bit-exact means integer equality for INT8 inputs with int32 accumulation and K below
  131,072. There is no requantization back to INT8.
- The numbers will come from one DE1-SoC and one RTX 4060. The FPGA ceiling is for the
  configuration that was built, not the device's maximum.
