#!/usr/bin/env bash
# ============================================================================
# lint.sh -- local pre-push gate
# ----------------------------------------------------------------------------
# Catches the class of failure that otherwise costs a ten-minute GitHub Actions
# round trip to discover.
#
# This exists because of a real one: `.b($signed(data))` in a module port
# connection crashed yosys with an internal assertion
# (`arg->is_signed == sig.as_wire()->is_signed`, genrtlil.cc:2128). Icarus
# compiled it happily and all 26 tests passed, so simulation gave no hint --
# the design was only unsynthesisable, not wrong. Elaborating locally
# reproduces it in about a second.
#
#   ./lint.sh
#
# Checks, in order:
#   1. yosys elaboration + `check -assert`   -- the synthesis crash class
#   2. inferred latches                      -- TT gotcha #3; sg13g2 HAS latch
#                                               cells, so yosys will happily
#                                               infer them from an incomplete
#                                               case/if and CTS/STA will not
#                                               handle them properly
#   3. generic synth statistics              -- flop count, for area tracking
#
# Verilator lint is NOT run here (not installed locally); it runs in CI with
# LINTER_INCLUDE_PDK_MODELS and is surfaced in the Action summary.
# ============================================================================
set -u

cd "$(dirname "$0")" || exit 1

# oss-cad-suite ships yosys with its own DLLs; both bin and lib must be on PATH
# or it dies on load.
if ! command -v yosys >/dev/null 2>&1; then
  OSS="$HOME/.apio/packages/oss-cad-suite"
  export PATH="$OSS/bin:$OSS/lib:$PATH"
fi
command -v yosys >/dev/null 2>&1 || { echo "yosys not found on PATH"; exit 1; }

TOP=tt_um_kush1434_proof
# Keep in sync with source_files in info.yaml and PROJECT_SOURCES in test/Makefile.
SRCS="src/mac_serial.v src/accumulator.v src/proof_core.v src/$TOP.v"

for f in $SRCS; do
  [ -f "$f" ] || { echo "FATAL: source file '$f' does not exist"; exit 1; }
done

fail=0
LOG=$(mktemp)

echo "=============================================================="
echo " lint.sh -- $TOP"
echo "=============================================================="

# --- 1. elaboration --------------------------------------------------------
printf '\n[1/3] elaboration + check\n'
if yosys -p "read_verilog $SRCS; hierarchy -top $TOP; proc; opt; check -assert" > "$LOG" 2>&1; then
  grep -E "Found and reported [0-9]+ problems" "$LOG" | tail -1 | sed 's/^/      /'
  echo "      OK"
else
  echo "      *** FAILED ***"
  grep -iE "error|assert" "$LOG" | head -10 | sed 's/^/      /'
  fail=1
fi

# --- 2. latches ------------------------------------------------------------
printf '\n[2/3] inferred latches\n'
if yosys -p "read_verilog $SRCS; hierarchy -top $TOP; proc; opt; select -assert-none t:\$dlatch t:\$_DLATCH_*" > "$LOG" 2>&1; then
  echo "      OK -- none inferred"
else
  echo "      *** LATCHES INFERRED ***"
  grep -iE "error|assert|dlatch" "$LOG" | head -10 | sed 's/^/      /'
  fail=1
fi

# --- 3. area ---------------------------------------------------------------
printf '\n[3/3] generic synth statistics\n'
if yosys -p "read_verilog $SRCS; synth -top $TOP -flatten; stat" > "$LOG" 2>&1; then
  # `synth` runs its own `stat` before the explicit one, so the log holds TWO
  # statistics blocks. Reset the tally at each block header so the last one
  # wins -- summing across both silently reported exactly double.
  flops=$(awk '/Printing statistics/{f=1; total=0} f && /\$_(S?DFF|DFFE)/ {gsub(/^ +/,""); total+=$1} END{print total+0}' "$LOG")
  cells=$(awk '/Printing statistics/{f=1; c=0} f && / cells$/ {gsub(/^ +/,""); c=$1} END{print c+0}' "$LOG")
  echo "      cells (generic) : ${cells:-?}"
  echo "      flip-flops      : ${flops:-?}"
  # sg13g2 dfrbpq_1 measured at 48.99 um2; budget at 60% density is 17,365 um2.
  if [ -n "$flops" ] && [ "$flops" -gt 0 ] 2>/dev/null; then
    awk -v f="$flops" 'BEGIN{
      a=f*48.99; printf "      sequential area : %.0f um2  (%.1f%% of the 17,365 um2 cell budget)\n", a, 100*a/17365 }'
  fi
  echo "      NOTE: generic mapping, not sg13g2. Trust the LibreLane"
  echo "            utilisation report for real numbers."
else
  echo "      *** SYNTH FAILED ***"
  grep -iE "error|assert" "$LOG" | head -10 | sed 's/^/      /'
  fail=1
fi

rm -f "$LOG"
echo
echo "=============================================================="
if [ "$fail" -eq 0 ]; then
  echo " LINT PASSED"
else
  echo " LINT FAILED"
fi
echo "=============================================================="
exit "$fail"
