#!/usr/bin/env bash
# Every check that needs neither a GPU nor the board, in one command.
#
#   ./run_all.sh               golden model, Icarus Verilog, Verilator lint (what CI runs)
#   SIM=questa ./run_all.sh    the same, with the testbenches in Questa
#
# The golden model runs first and rewrites rtl/vectors/ and data/ with its defaults.
#
# A testbench passes only if it compiles, finishes within TIMEOUT seconds, and
# prints a "SUCCESS:" line with no "FAILURE:" line. Any failure makes the script
# exit non-zero. Logs are kept in build/<sim>/.
set -uo pipefail
cd "$(dirname "$0")"

SIM=${SIM:-icarus}
TIMEOUT=${TIMEOUT:-120}   # seconds per simulation
BUILD="build/$SIM"

passes=0
failures=0

section() { printf '\n=== %s ===\n' "$1"; }
pass()    { printf '  PASS  %s\n' "$1"; passes=$((passes + 1)); }
skip()    { printf '  SKIP  %s\n' "$1"; }

# fail <name> <reason> [log]: report a failure and show the end of its log
fail() {
  printf '  FAIL  %s: %s\n' "$1" "$2"
  if [ -n "${3:-}" ] && [ -f "$3" ]; then
    tail -n 25 "$3" | sed 's/^/        | /'
  fi
  failures=$((failures + 1))
}

# run_tb <top module> <source files...>
run_tb() {
  local top=$1; shift
  local dir="$BUILD/$top"
  local log="$dir.log"
  local status
  rm -rf "$dir"; mkdir -p "$dir"; : > "$log"

  case $SIM in
    icarus)
      if ! iverilog -g2012 -Wall -s "$top" -o "$dir/$top.vvp" "$@" >>"$log" 2>&1; then
        fail "$top" "compile error" "$log"; return
      fi
      timeout "$TIMEOUT" vvp -n "$dir/$top.vvp" >>"$log" 2>&1
      status=$?
      ;;
    questa)
      # Questa writes its work library, transcript and modelsim.ini into the
      # current directory, so each testbench gets its own folder under build/.
      local srcs=() f
      for f in "$@"; do srcs+=("$PWD/$f"); done
      if ! ( cd "$dir" && vlib work && vlog -sv -lint "${srcs[@]}" ) >>"$log" 2>&1; then
        fail "$top" "compile error" "$log"; return
      fi
      ( cd "$dir" && timeout "$TIMEOUT" vsim -c "$top" -do "run -all; quit -f" ) >>"$log" 2>&1
      status=$?
      ;;
  esac

  if [ "$status" -eq 124 ]; then
    fail "$top" "timed out after ${TIMEOUT}s" "$log"
  elif grep -q "FAILURE:" "$log"; then
    fail "$top" "testbench reported FAILURE" "$log"
  elif ! grep -q "SUCCESS:" "$log"; then
    fail "$top" "no SUCCESS line (exit status $status)" "$log"
  else
    pass "$top"
  fi
}

# golden [args...]: generate the M1 vectors into rtl/vectors/ and data/
golden() {
  local log="$BUILD/golden_model.log"
  if "$PY" python/01_golden_model.py "$@" >"$log" 2>&1 && grep -q "SUCCESS:" "$log"; then
    pass "golden model${*:+ $*}"
  else
    fail "golden model" "did not report SUCCESS" "$log"
  fi
}

# lint <top module> <source files...>: Verilator warnings are errors, except a
# missing newline at end of file, which is editor-dependent and not a design issue
lint() {
  local top=$1; shift
  local log="$BUILD/lint_$top.log"
  if verilator --lint-only -Wall -Wno-EOFNEWLINE --top-module "$top" "$@" >"$log" 2>&1; then
    pass "lint $top"
  else
    fail "lint $top" "verilator reported warnings" "$log"
  fi
}

case $SIM in
  icarus) tools="iverilog vvp verilator" ;;
  questa) tools="vlib vlog vsim" ;;
  *) echo "unknown SIM=$SIM (use icarus or questa)"; exit 2 ;;
esac
for t in $tools timeout; do
  command -v "$t" >/dev/null 2>&1 || { echo "missing $t; run ./preflight.sh"; exit 2; }
done
# python3 on Linux; on Windows it is often only `python`
PY=""
for p in python3 python; do
  if "$p" -c "import numpy" >/dev/null 2>&1; then PY=$p; break; fi
done
[ -n "$PY" ] || { echo "missing Python with NumPy; run ./preflight.sh"; exit 2; }
mkdir -p "$BUILD"

section "Golden model (M1)"
golden

section "RTL testbenches ($SIM)"
run_tb tb_int8mac rtl/int8mac.sv rtl/tb_int8mac.sv

section "Lint (Verilator)"
if command -v verilator >/dev/null 2>&1; then
  lint int8mac rtl/int8mac.sv
else
  skip "verilator not found"
fi

section "Summary"
if [ "$failures" -eq 0 ]; then
  echo "  ALL PASS ($passes passed)"
else
  echo "  $failures FAILED, $passes passed"
  exit 1
fi
