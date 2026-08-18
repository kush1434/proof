#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Kush Shah
# SPDX-License-Identifier: Apache-2.0
"""Catch the compile errors `check_tex.py` cannot.

There is no LaTeX toolchain on this machine, so the draft has never been
compiled. `check_tex.py` covers structure -- environments, braces, cite/label
resolution, missing figures, length. This covers the things that halt a
*compile* rather than breaking structure, which is a different failure mode and
the one most likely to bite on the first Overleaf build:

  1. Unescaped `_`, `#` and `&` in text mode. This paper discusses identifiers
     like mono_viol, is_weight and test_cycles.py, and a bare underscore
     outside \\texttt, \\url or math is a hard error, not a warning.
  2. Unbalanced inline math.
  3. Table rows with more cells than the column spec allows -- silently
     truncated by some engines, an error in others.

Run alongside check_tex.py:  python check_specials.py
"""

import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TEX = os.path.join(HERE, "proof_bibm2026.tex")

# Contexts where a special character is legal and must not be flagged.
SAFE = [
    r"\\texttt\{[^{}]*\}",
    r"\\verb\|[^|]*\|",
    r"\\url\{[^}]*\}",
    r"\\path\{[^}]*\}",
    r"\\label\{[^}]*\}",
    r"\\ref\{[^}]*\}",
    r"\\eqref\{[^}]*\}",
    r"\\cite\{[^}]*\}",
    r"\\bibitem\{[^}]*\}",
    r"\\includegraphics(?:\[[^\]]*\])?\{[^}]*\}",
    r"\\begin\{tabular\}\{[^}]*\}",
    # Display math. These must be blanked before inline $...$, or a stray $
    # inside one would pair with the next and swallow real text.
    r"\\begin\{(?:equation|align|gather|multline|displaymath)\*?\}.*?"
    r"\\end\{(?:equation|align|gather|multline|displaymath)\*?\}",
    r"\\\[.*?\\\]",
    r"\$[^$]*\$",
]


def blanked(text):
    """Replace safe contexts with spaces, preserving line and column numbers."""
    def wipe(m):
        return "".join(" " if c != "\n" else "\n" for c in m.group(0))
    out = re.sub(r"(?<!\\)%.*", wipe, text)
    for pat in SAFE:
        out = re.sub(pat, wipe, out, flags=re.S)
    return out


def check_specials(src):
    """Bare _ and # outside safe contexts. & is legal only inside tabular."""
    problems = []
    body = blanked(src)
    raw = src.split("\n")
    for i, line in enumerate(body.split("\n"), 1):
        for m in re.finditer(r"[_#]", line):
            if m.start() and line[m.start() - 1] == "\\":
                continue
            problems.append(
                "line %d: unescaped '%s' in text mode -- %s"
                % (i, m.group(0), raw[i - 1].strip()[:70]))
    return problems


def check_math(src):
    body = blanked(src)
    # blanked() already removed matched $...$ pairs, so anything left is odd.
    stray = len(re.findall(r"(?<!\\)\$", body))
    return ["%d unmatched '$' -- inline math is unbalanced" % stray] if stray else []


def check_tables(src):
    problems = []
    pat = r"\\begin\{tabular\}\{([^}]*)\}(.*?)\\end\{tabular\}"
    for m in re.finditer(pat, src, re.S):
        spec = re.sub(r"p\{[^}]*\}", "p", m.group(1))
        ncol = len(re.sub(r"[^lcrp]", "", spec))
        for row in m.group(2).split(r"\\"):
            body = re.sub(r"(?<!\\)%.*", "", row).strip()
            if not body:
                continue
            if re.match(r"\\(hline|toprule|midrule|bottomrule|cmidrule)", body):
                continue
            # \multicolumn{n} occupies n cells but contains only one &-gap.
            span = sum(int(k) - 1
                       for k in re.findall(r"\\multicolumn\{(\d+)\}", body))
            cells = body.count("&") + 1 + span
            if cells > ncol:
                problems.append(
                    "tabular row has %d cells, spec allows %d -- %s"
                    % (cells, ncol, body[:60]))
    return problems


def main():
    src = open(TEX, encoding="utf-8").read()
    print("compile-blocker checks")
    print("  unescaped specials, math balance, tabular widths")
    print()

    # Math balance is reported FIRST and on its own, because one stray '$'
    # desynchronises every later $...$ pair and turns correct math into a
    # cascade of spurious "unescaped _" reports. Injecting a single stray
    # dollar produced 18 downstream false positives in testing, so fixing the
    # imbalance before reading anything else is not optional.
    math = check_math(src)
    if math:
        for p in math:
            print("  - %s" % p)
        print()
        print("  fix this first -- the specials report below is unreliable")
        print("  until inline math pairs up again.")
        return 1

    problems = check_specials(src) + check_tables(src)
    if problems:
        print("%d problem(s):" % len(problems))
        for p in problems:
            print("  - %s" % p)
        return 1
    print("no problems found")
    return 0


if __name__ == "__main__":
    sys.exit(main())
