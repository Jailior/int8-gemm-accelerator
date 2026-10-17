# 01_golden_model.py (M1): the reference every other level is checked against.
# Writes random INT8 tests and their C = A @ B (int32) as .hex for $readmemh and
# as .npy for the HPS driver and the CUDA kernels.
#
#   python python/01_golden_model.py                                      # 1,000 tiles of 4x4x4 (M5)
#   python python/01_golden_model.py --m 300 --n 300 --k 300 --count 5    # M6

import argparse
from pathlib import Path
import numpy as np


INT8_MAX = np.iinfo(np.int8).max
INT8_MIN = np.iinfo(np.int8).min

INT32_MAX = np.iinfo(np.int32).max
INT32_MIN = np.iinfo(np.int32).min

# Below 2**17, a sum of K products (each at most 2**14) cannot overflow int32
K_LIMIT = 2**17

NUM_CORNER_CASES = 5

# Resolved from this file, so the script works from any directory
REPO = Path(__file__).resolve().parent.parent
DEFAULT_HEX_OUT_PATH = REPO / "rtl" / "vectors"
DEFAULT_NPY_OUT_PATH = REPO / "data"

HEX_DIGITS = set("0123456789abcdef")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate the golden INT8 GEMM test vectors (M1).")
    parser.add_argument("--m", type=int, default=4, help="The M in A (M x K)")
    parser.add_argument("--n", type=int, default=4, help="The N in B (K x N)")
    parser.add_argument("--k", type=int, default=4, help="The K in A (M x K) and B (K x N)")
    parser.add_argument("--count", type=int, default=1000, help="number of tests")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed")
    parser.add_argument("--hex-dir", type=Path, default=DEFAULT_HEX_OUT_PATH, help="where the .hex files go")
    parser.add_argument("--npy-dir", type=Path, default=DEFAULT_NPY_OUT_PATH, help="where the .npy files go")
    args = parser.parse_args()
    if min(args.m, args.n, args.k) < 1:
        parser.error("--m, --n and --k must all be at least 1")
    if args.k >= K_LIMIT:
        parser.error(f"--k must be below {K_LIMIT}, or the int32 sums can overflow")
    if args.count < NUM_CORNER_CASES:
        parser.error(f"--count must be at least {NUM_CORNER_CASES}, the number of corner cases")
    return args



# Tests 0-4 are fixed corner cases, easy to recognise in a waveform:
#   0  all zeros
#   1  A and B all -128     C = K * 16384, the largest possible sum
#   2  A -128, B 127        C = -K * 16256, the most negative
#   3  A and B all 127
#   4  A = identity (np.eye also handles M != K), so C copies B's first rows
#      and a swapped index or transpose stands out
# The arrays start as ones so every case can be written as a scale.
def make_tests(count, m, n, k):
    A = np.ones((count, m, k), dtype=np.int8)
    B = np.ones((count, k, n), dtype=np.int8)

    A[0] = 0
    B[0] = 0

    A[1] *= -128
    B[1] *= -128

    A[2] *= -128
    B[2] *= 127

    A[3] *= 127
    B[3] *= 127

    A[4] *= np.eye(m, k, dtype=np.int8)
    B[4] *= random_int8(k, n)

    for i in range(NUM_CORNER_CASES, count):
        A[i] *= random_int8(m, k)
        B[i] *= random_int8(k, n)

    return A, B


# randint's upper bound is exclusive, so INT8_MAX + 1 lets 127 appear
def random_int8(rows, cols):
    return np.random.randint(INT8_MIN, INT8_MAX + 1, size=(rows, cols), dtype=np.int8)



# int8 @ int8 multiplies and sums in int8 and wraps silently, so widen first.
# int64 rather than int32, so an overflow fails the assert instead of wrapping.
def golden_matmul(A, B):
    C = A.astype(np.int64) @ B.astype(np.int64)
    assert np.all((C >= INT32_MIN) & (C <= INT32_MAX)), "C values exceed int32 limits!"
    return C.astype(np.int32)


# $readmemh format: one value per line, two's complement, no 0x prefix, so -128
# is "80" (formatting it directly would give "-80"). Tests are concatenated and
# every matrix is row-major, B included; reordering B into columns is the
# testbench's job. The testbench memories must be declared signed so "80" reads
# back as -128.
def write_hex(path, array, digits):
    mask = (1 << (4 * digits)) - 1
    # Python ints mask cleanly; NumPy int8 values can't hold 0xFF
    values = array.ravel(order="C").tolist()
    # "\n" even on Windows, so the files match CI's byte for byte
    with open(path, "w", newline="\n") as f:
        f.writelines(f"{v & mask:0{digits}x}\n" for v in values)


# One file per matrix holding every test, shape (count, rows, cols). A and B come
# out as '|i1' (one byte has no byte order) and C as '<i4'. common/npy.h should
# accept exactly those, header version 1.0 and fortran_order False, and check
# every fread's return value: reading past the end of a file fails silently.
def write_npy(path, array):
    # No-op for int8; forces int32 to little-endian on any machine
    array = np.ascontiguousarray(array, dtype=array.dtype.newbyteorder("<"))
    np.save(path, array, allow_pickle=False)

# The testbench includes vectors.svh, so it can't be compiled for one shape and
# loaded with vectors of another. The VEC_ prefix keeps VEC_N apart from the
# array's own N parameter, which differs once M6 tiles larger matrices.
def write_params_svh(path, count, m, n, k, seed):
    text = (
        "// Generated by python/01_golden_model.py; rerun it instead of editing.\n"
        f"// seed {seed}\n"
        f"localparam int VEC_COUNT = {count};\n"
        f"localparam int VEC_M = {m};\n"
        f"localparam int VEC_N = {n};\n"
        f"localparam int VEC_K = {k};\n"
    )
    path.write_text(text, newline="\n")

def main():
    args = parse_args()
    np.random.seed(args.seed)

    A, B = make_tests(args.count, args.m, args.n, args.k)
    C = golden_matmul(A, B)

    args.hex_dir.mkdir(parents=True, exist_ok=True)
    args.npy_dir.mkdir(parents=True, exist_ok=True)

    # (name, array, hex digits per value, .npy descr)
    outputs = [("A", A, 2, "|i1"), ("B", B, 2, "|i1"), ("C_expected", C, 8, "<i4")]
    for name, array, digits, _ in outputs:
        write_hex(args.hex_dir / f"{name}.hex", array, digits)
        write_npy(args.npy_dir / f"{name}.npy", array)
    write_params_svh(args.hex_dir / "vectors.svh", args.count, args.m, args.n, args.k, args.seed)

    print(f"SUCCESS: {args.count} tests, M={args.m} N={args.n} K={args.k}, seed {args.seed} "
          f"-> {args.hex_dir}, {args.npy_dir}")
    return 0


if __name__ == "__main__":
    main()
