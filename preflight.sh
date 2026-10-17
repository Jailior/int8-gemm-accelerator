#!/usr/bin/env bash
# Checks the toolchain before run_all.sh, so a missing tool is one line of
# output instead of a confusing failure later.
#
#   ./preflight.sh              tools for the default flow (Icarus Verilog, Verilator)
#   SIM=questa ./preflight.sh   tools for the local Questa flow
#
# Exits non-zero if anything run_all.sh needs is missing. Tools for later
# milestones are reported but never fail the check.
cd "$(dirname "$0")"
SIM=${SIM:-icarus}
missing=0

say() { printf '  %-12s %s\n' "$1" "$2"; }

# check <need|want> <tool> <hint> <version command...>
check() {
  local kind=$1 tool=$2 hint=$3; shift 3
  if command -v "$tool" >/dev/null 2>&1; then
    say "$tool" "OK    $("$@" 2>&1 | head -n 1)"
  elif [ "$kind" = need ]; then
    say "$tool" "MISS  $hint"
    missing=1
  else
    say "$tool" "--    $hint"
  fi
}

echo "=== required by run_all.sh (SIM=$SIM) ==="
check need timeout "GNU coreutils" timeout --version
case $SIM in
  icarus)
    check need iverilog  "apt install iverilog" iverilog -V
    check need vvp       "installed with iverilog" vvp -V
    check need verilator "apt install verilator" verilator --version
    ;;
  questa)
    check need vlib      "add Questa's bin folder (win64 or linux_x86_64) to PATH" sh -c "command -v vlib"
    check need vlog      "installed with Questa" vlog -version
    check need vsim      "installed with Questa" vsim -version
    check want verilator "optional here; lint is skipped without it" verilator --version
    ;;
  *)
    echo "  unknown SIM=$SIM (use icarus or questa)"
    exit 2
    ;;
esac
# The same search as run_all.sh: the first Python that can import NumPy
py=""
for p in python3 python; do
  if "$p" -c "import numpy" >/dev/null 2>&1; then py=$p; break; fi
done
if [ -n "$py" ]; then
  say "python" "OK    $("$py" --version 2>&1) ($py)"
  say "numpy" "OK    $("$py" -c 'import numpy; print(numpy.__version__)')"
else
  say "numpy" "MISS  Python 3 with NumPy: pip install numpy, or apt install python3-numpy"
  missing=1
fi

echo
echo "=== later milestones (optional) ==="
check want nvcc       "CUDA toolkit (M3, M4, M9)" sh -c "nvcc --version | grep release"
check want quartus_sh "Quartus Prime Lite (M8)" sh -c "quartus_sh --version | grep Version"

echo
if [ "$missing" -ne 0 ]; then
  echo "  FAIL: run_all.sh needs the tools marked MISS."
  exit 1
fi
echo "  Ready: SIM=$SIM ./run_all.sh"
