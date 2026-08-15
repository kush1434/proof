#!/usr/bin/env bash
# ============================================================================
# mutate.sh -- mutation testing for the Proof verification environment
# ----------------------------------------------------------------------------
# A green testbench proves nothing until it can go red. This injects a known
# bug into the RTL one at a time, runs the relevant suite, and confirms the
# testbench notices. The original file is restored after every mutant, and on
# Ctrl-C via the EXIT trap.
#
#   ./mutate.sh                 # every mutant
#   ./mutate.sh M4 M5           # only the named ones
#   PROOF_SEED=7 ./mutate.sh    # different random-stream seed
#
# ----------------------------------------------------------------------------
# WHY THE COMPARISON IS `diff -q --strip-trailing-cr` AND NOT `cmp -s`
#
# This was TB-4 on the previous project and it is the single most important
# line in the file. The repository checks out CRLF and sed writes LF, so a
# plain byte comparison reports EVERY file as changed -- including one where
# the pattern matched nothing. The did-not-apply guard then becomes
# structurally dead, and a mutant that was never injected gets scored as
# legitimately caught. Every CAUGHT below would be a lie.
#
# The guard is therefore re-verified in both directions on every run, before
# any mutant is attempted, and the script refuses to continue if it is broken.
#
# ----------------------------------------------------------------------------
# WHY OUTCOMES ARE DETECTED FROM THE RUNNER'S EXIT CODE
#
# test/run.py exits non-zero when any test fails. That was NOT true when it was
# first written -- cocotb's runner reports success regardless of outcome, and
# the runner silently inherited the trap (TB-3 in BUGS.md). If that ever
# regresses, every mutant here reads as ESCAPED rather than as a broken
# harness, which is the safe direction to fail but still worth knowing.
#
# ----------------------------------------------------------------------------
# OUTCOMES
#   CAUGHT        the testbench detected the injected bug            (good)
#   ESCAPED       injected, and the testbench still passed           (bad: gap)
#   EQUIVALENT    injected, testbench passed, and it is PROVEN that
#                 no input can distinguish it -- not a gap
#   INVALID       the pattern matched nothing, mutant never injected
#   COMPILE-FAIL  the compiler rejected it, so the testbench never got a say
# ============================================================================
set -u

cd "$(dirname "$0")" || exit 1

# Icarus is normally on PATH; fall back to the local install if it is not.
if ! command -v iverilog >/dev/null 2>&1; then
  export PATH="$PATH:/c/Users/kushk/Downloads/Installers & Software/Dev Tools/iverilog/bin"
fi
command -v iverilog >/dev/null 2>&1 || { echo "iverilog not found on PATH"; exit 1; }

PY="${PY:-python}"
LOG="test/mut_run.log"
WANT="$*"

caught=0; escaped=0; equivalent=0; invalid=0; cfail=0; total=0
ESCAPED_LIST=""; INVALID_LIST=""

# Snapshot every RTL file before touching anything, so "did we put it back?"
# is answered against what we actually started with.
PRE_HASH="$(mktemp)"
md5sum src/*.v > "$PRE_HASH"

restore_all() {
  for f in src/*.mutbak; do
    [ -e "$f" ] && mv -f "$f" "${f%.mutbak}"
  done
  return 0
}
trap restore_all EXIT INT TERM

# ---------------------------------------------------------------------------
# Guard self-test -- must pass before any result can be trusted
# ---------------------------------------------------------------------------
guard_selftest() {
  local f="src/accumulator.v" ok=1
  echo "--- Guard self-test (must pass before any result is meaningful) ---"
  cp "$f" "$f.mutbak"

  # Negative direction: a pattern matching nothing must read as "no change".
  sed -e 's|PATTERN_THAT_INTENTIONALLY_MATCHES_NOTHING|zzz|' "$f.mutbak" > "$f"
  if diff -q --strip-trailing-cr "$f.mutbak" "$f" >/dev/null 2>&1; then
    echo "    non-matching pattern -> correctly reported as no-op    [OK]"
  else
    echo "    non-matching pattern -> reported as a real mutation    [BROKEN]"; ok=0
  fi

  # Positive direction: a pattern that does match must read as "changed".
  sed -e 's|ACC_W  = 24|ACC_W  = 23|' "$f.mutbak" > "$f"
  if diff -q --strip-trailing-cr "$f.mutbak" "$f" >/dev/null 2>&1; then
    echo "    real mutation        -> reported as no-op              [BROKEN]"; ok=0
  else
    echo "    real mutation        -> correctly reported as changed  [OK]"
  fi

  mv -f "$f.mutbak" "$f"

  # And the runner itself must be able to report failure at all (TB-3).
  echo "--- Runner self-test (a runner that cannot fail makes CAUGHT meaningless) ---"
  cp "$f" "$f.mutbak"
  sed -i 's|wire ovf = (sum\[ACC_W\] != sum\[ACC_W-1\]);|wire ovf = 1'"'"'b0;|' "$f"
  if run_suite top; then
    echo "    known-broken RTL     -> runner reported SUCCESS         [BROKEN]"; ok=0
  else
    echo "    known-broken RTL     -> runner reported failure         [OK]"
  fi
  mv -f "$f.mutbak" "$f"

  if [ "$ok" -ne 1 ]; then
    echo
    echo "FATAL: the harness is not trustworthy. Every CAUGHT this script could"
    echo "       print would be meaningless. Aborting."
    exit 1
  fi
  echo
}

run_suite() {
  case "$1" in
    top)  ( cd test && $PY run.py            > "../$LOG" 2>&1 ) ;;
    *)    ( cd test && $PY run.py --unit "$1" > "../$LOG" 2>&1 ) ;;
  esac
}

# ---------------------------------------------------------------------------
# run_mutant <id> <file> <suites> <expect> <desc> <sed-expr>...
#   suites : space-separated, e.g. "top" or "mac_serial top"
#   expect : "catch" or "equivalent"
# ---------------------------------------------------------------------------
run_mutant() {
  local id="$1" file="src/$2" suites="$3" expect="$4" desc="$5"
  shift 5

  if [ -n "$WANT" ]; then
    case " $WANT " in *" $id "*) ;; *) return ;; esac
  fi
  total=$((total + 1))

  printf '\n[%s] %s\n' "$id" "$desc"
  printf '     file: %s   suites: %s\n' "$file" "$suites"

  cp "$file" "$file.mutbak"
  sed "$@" "$file.mutbak" > "$file"

  # CRLF-safe comparison -- see the header. This guard is load-bearing.
  if diff -q --strip-trailing-cr "$file.mutbak" "$file" >/dev/null 2>&1; then
    echo "     INVALID -- pattern matched nothing, mutant never injected"
    invalid=$((invalid + 1))
    INVALID_LIST="${INVALID_LIST}
       [$id] $desc"
    mv -f "$file.mutbak" "$file"
    return
  fi
  diff --strip-trailing-cr "$file.mutbak" "$file" | grep '^[<>]' | sed 's/^/       /'

  local detected=0 compilefail=0
  for s in $suites; do
    if run_suite "$s"; then
      :
    else
      if grep -qiE 'error:|syntax error|COMPILE' "$LOG" && ! grep -q 'test(s) failed' "$LOG"; then
        compilefail=1
      fi
      detected=1
      printf '     suite %-12s -> ' "$s"
      grep -oE '[0-9]+ of [0-9]+ test\(s\) failed' "$LOG" | head -1 || echo "failed"
    fi
  done

  if [ "$compilefail" -eq 1 ]; then
    echo "     COMPILE-FAIL -- rejected by the compiler, not detected by the testbench"
    cfail=$((cfail + 1))
  elif [ "$detected" -eq 1 ]; then
    echo "     CAUGHT"
    caught=$((caught + 1))
  elif [ "$expect" = "equivalent" ]; then
    echo "     EQUIVALENT -- escaped as expected; see BUGS.md for the proof"
    equivalent=$((equivalent + 1))
  else
    echo "     *** ESCAPED *** -- the testbench did not detect this bug"
    escaped=$((escaped + 1))
    ESCAPED_LIST="${ESCAPED_LIST}
       [$id] $desc"
  fi

  mv -f "$file.mutbak" "$file"
}

# ===========================================================================
echo "============================================================"
echo " MUTATION TESTING -- Proof"
echo "   seed        : ${PROOF_SEED:-20260814}"
echo "   suites      : test/run.py  (+ --unit mac_serial where relevant)"
echo "============================================================"
echo

guard_selftest

# --- mac_serial ------------------------------------------------------------
run_mutant M1 mac_serial.v "mac_serial top" catch \
  "mac_serial: drop the two's-complement sign correction on the last step" \
  -e 's|wire signed \[DW:0\] addend    = last ? -mcand_ext : mcand_ext;|wire signed [DW:0] addend    = mcand_ext;|'

# Escapes on purpose. Proven equivalent by the exhaustive 65,536-case test:
# acc_hi's top bit is a duplicated sign bit that no later step reads. See
# BUGS.md. Reported as EQUIVALENT rather than CAUGHT or ESCAPED, because
# calling it either of those would be a lie in a different direction.
run_mutant M2 mac_serial.v "mac_serial top" equivalent \
  "mac_serial: high accumulator one bit narrower (known equivalent)" \
  -e 's|reg signed \[DW:0\]   acc_hi;|reg signed [DW-1:0]   acc_hi;|'

# Retargeted 2026-08-14 when the last-step test was resized to clear a
# Verilator warning. The old pattern stopped matching and was correctly
# reported INVALID rather than scored as CAUGHT -- which is the entire point
# of the did-not-apply guard. All-ones minus one fires one step early.
run_mutant M3 mac_serial.v "mac_serial top" catch \
  "mac_serial: terminate one shift step early" \
  -e 's|wire last = (step\[CW-1:0\] == {CW{1'"'"'b1}});|wire last = (step[CW-1:0] == {{(CW-1){1'"'"'b1}}, 1'"'"'b0});|'

# --- accumulator -----------------------------------------------------------
run_mutant M4 accumulator.v "top" catch \
  "accumulator: wrap instead of saturate" \
  -e 's|      ovf ? {sum\[ACC_W\], {(ACC_W - 1) {~sum\[ACC_W\]}}} : sum\[ACC_W-1:0\];|      sum[ACC_W-1:0];|'

# Note the delimiter: this pattern contains a literal '|', so '|' cannot also
# be the sed delimiter. Getting that wrong yields a pattern that matches
# nothing, which the did-not-apply guard above reports as INVALID rather than
# silently scoring as CAUGHT.
run_mutant M5 accumulator.v "top" catch \
  "accumulator: overflow flag no longer sticky" \
  -e 's#sat_r <= sat_r | ovf;#sat_r <= ovf;#'

run_mutant M6 accumulator.v "top" catch \
  "accumulator: overflow never detected" \
  -e 's|wire ovf = (sum\[ACC_W\] != sum\[ACC_W-1\]);|wire ovf = 1'"'"'b0;|'

run_mutant M7 accumulator.v "top" catch \
  "accumulator: clear ignored, so state leaks between inferences" \
  -e 's|    end else if (clear) begin|    end else if (1'"'"'b0) begin|'

# --- proof_core ------------------------------------------------------------
run_mutant M8 proof_core.v "top" catch \
  "proof_core: accumulator never cleared at the start of a stream" \
  -e 's|wire clear_acc = (state == S_IDLE) && take_wt;|wire clear_acc = 1'"'"'b0;|'

run_mutant M9 proof_core.v "top" catch \
  "proof_core: requantisation shift off by one" \
  -e 's|wire signed \[ACC_W-1:0\] gl_full = acc >>> shift;|wire signed [ACC_W-1:0] gl_full = acc >>> (shift + 1);|'

run_mutant M10 proof_core.v "top" catch \
  "proof_core: high category threshold 20 -> 21" \
  -e 's|(gl_full >= 20) ? 2.d2|(gl_full >= 21) ? 2'"'"'d2|'

run_mutant M11 proof_core.v "top" catch \
  "proof_core: LAST ignored, so every pair finishes the inference" \
  -e 's|          if (last_r) begin|          if (1'"'"'b1) begin|'

run_mutant M12 proof_core.v "top" catch \
  "proof_core: busy drops during the accumulate cycle" \
  -e 's#assign busy   = (state == S_MULT) || (state == S_ACC);#assign busy   = (state == S_MULT);#'

run_mutant M13 proof_core.v "top" catch \
  "proof_core: every byte treated as a weight, so nothing ever multiplies" \
  -e 's|wire take_wt  = take && is_weight;|wire take_wt  = take;|'

# ---------------------------------------------------------------------------
restore_all
rm -f "$LOG"

echo
echo "============================================================"
echo " MUTATION SUMMARY"
echo "   mutants considered : $total"
echo "   caught             : $caught"
echo "   equivalent         : $equivalent"
echo "   escaped            : $escaped"
echo "   invalid (no-op)    : $invalid"
echo "   compile-fail       : $cfail"
if [ "$escaped" -eq 0 ] && [ "$invalid" -eq 0 ] && [ "$cfail" -eq 0 ]; then
  echo "   VERDICT            : *** ALL NON-EQUIVALENT MUTANTS CAUGHT ***"
else
  echo "   VERDICT            : *** REVIEW NEEDED ***"
  [ -n "$ESCAPED_LIST" ] && echo "   escaped:$ESCAPED_LIST"
  [ -n "$INVALID_LIST" ] && echo "   invalid (pattern needs retargeting):$INVALID_LIST"
fi
echo "============================================================"

# Prove the RTL is byte-identical to what we started with.
#
# Compared against a checksum snapshot taken before the first mutant, NOT
# against git HEAD. During development src/ legitimately differs from HEAD --
# new files are untracked, work in progress is uncommitted -- so a git-based
# check reports "NOT RESTORED" on almost every run. A guard that cries wolf
# constantly is one people learn to ignore, which makes it no guard at all.
if md5sum -c "$PRE_HASH" >/dev/null 2>&1; then
  echo " RTL restored: clean (every file under src/ matches its pre-run checksum)"
  rm -f "$PRE_HASH"
else
  echo " !! RTL NOT RESTORED -- these differ from the pre-run snapshot:"
  md5sum -c "$PRE_HASH" 2>/dev/null | grep -v ': OK$'
  rm -f "$PRE_HASH"
  exit 1
fi

if [ "$escaped" -ne 0 ] || [ "$invalid" -ne 0 ] || [ "$cfail" -ne 0 ]; then exit 1; fi
exit 0
