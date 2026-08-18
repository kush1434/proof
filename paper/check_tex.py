#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Kush Shah
# SPDX-License-Identifier: Apache-2.0
"""Sanity-check the submission without a LaTeX toolchain.

There is no pdflatex on the development machine, so the usual feedback loop --
compile and look at it -- is not available until the source reaches Overleaf.
This catches the errors that would otherwise be found there: unbalanced
environments, a \\ref with no \\label, a \\cite with no \\bibitem, a missing
figure file, and a page count over the limit.

The length estimate is a ROUGH GUIDE, not a substitute for compiling. It
measures the figures at their real authored size (savefig's tight bounding box
means the PDF is often smaller than the figsize) and assumes ~1050 words per
full page of IEEEtran two-column body text.

    python check_tex.py
"""

import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TEX = os.path.join(HERE, "proof_bibm2026.tex")
FIGDIR = os.path.join(HERE, "figures", "out")

PAGE_LIMIT = 5
COL_H = 9.25        # IEEEtran column height, inches

# CALIBRATED against real compiles, not guessed -- and still only a guide.
#
#   3584 words, 0.96 floats, 0.36 refs -> 6 pages   implies 766 words/page
#   3309 words, 0.81 floats, 0.36 refs -> 5 pages   implies 864 words/page
#
# It began at 1050, which predicted 4.77 for the draft that compiled to 6: an
# error large enough to hide a whole page over a hard limit, which is the one
# thing this number exists to catch. But the two measurements disagree by 13 %,
# because pages are quantised and float placement moves text around in ways no
# word count models -- the second draft compiled to 5 while this estimator still
# said 5.58.
#
# So: 800, between the two, and the estimate WARNS rather than fails. CI
# compiles the paper and counts pages with pdfinfo; that is the number that
# decides. Treat this as the cheap early signal before a five-minute round trip.
WORDS_PER_PAGE = 800
CAPTION_IN = 0.55   # allowance per float for its caption
WIDE_FLOATS = {"fig3_architecture"}


def main():
    src = open(TEX, encoding="utf-8").read()
    body = "\n".join(l for l in src.split("\n") if not l.lstrip().startswith("%"))
    body = re.sub(r"(?<!\\)%.*", "", body)

    problems = []

    envs = {}
    for m in re.finditer(r"\\(begin|end)\{([^}]+)\}", body):
        envs.setdefault(m.group(2), []).append(m.group(1))
    for name, v in sorted(envs.items()):
        if v.count("begin") != v.count("end"):
            problems.append(f"environment {name} is unbalanced")
    if body.count("{") != body.count("}"):
        problems.append(f"braces: {body.count('{')} open, {body.count('}')} close")

    cites = set()
    for grp in re.findall(r"\\cite\{([^}]+)\}", body):
        cites.update(c.strip() for c in grp.split(","))
    bibs = set(re.findall(r"\\bibitem\{([^}]+)\}", body))
    for k in sorted(cites - bibs):
        problems.append(f"\\cite{{{k}}} has no \\bibitem")
    for k in sorted(bibs - cites):
        problems.append(f"\\bibitem{{{k}}} is never cited")

    labels = set(re.findall(r"\\label\{([^}]+)\}", body))
    for k in sorted(set(re.findall(r"\\ref\{([^}]+)\}", body)) - labels):
        problems.append(f"\\ref{{{k}}} has no \\label")

    # A tab where a control sequence was meant is the classic scripted-edit bug.
    if "\t" in src:
        problems.append("source contains a TAB -- check for a mangled \\t... macro")

    print("figures")
    floats = 0.0
    try:
        from PIL import Image
    except ImportError:
        Image = None
    for opt, name in re.findall(
            r"\\includegraphics(\[[^\]]*\])?\{([^}]+)\}", body):
        pdf = os.path.join(FIGDIR, name)
        png = pdf.replace(".pdf", ".png")
        ok = os.path.exists(pdf)
        size = ""
        if Image and os.path.exists(png):
            im = Image.open(png)
            w_in, h_in = im.size[0] / 600.0, im.size[1] / 600.0
            size = f"{w_in:5.2f} x {h_in:5.2f} in"
            stem = name.rsplit(".", 1)[0]
            wide = stem in WIDE_FLOATS
            # A pinned width rescales the height with it.
            if "columnwidth" in (opt or ""):
                h_in *= 3.45 / w_in
            elif "textwidth" in (opt or ""):
                h_in *= 7.16 / w_in
            floats += (h_in + CAPTION_IN) / COL_H * (1.0 if wide else 0.5)
        print(f"  {name:30s} {size:20s} {'found' if ok else '<<< MISSING'}")
        if not ok:
            problems.append(f"{name} is missing from figures/out/")

    ntab = len(re.findall(r"\\begin\{table\}", body))
    floats += ntab * 1.55 / COL_H * 0.5

    start = body.index("\\maketitle")
    end = body.index("\\begin{thebibliography}")
    txt = re.sub(r"\\includegraphics(?:\[[^\]]*\])?\{[^}]*\}", "", body[start:end])
    words = len(re.findall(r"[A-Za-z][A-Za-z'-]+", txt))
    refs = len(bibs) * 0.030
    total = words / WORDS_PER_PAGE + floats + refs

    print(f"\nlength (rough -- confirm by compiling)")
    print(f"  body words          {words}")
    print(f"  text                {words / WORDS_PER_PAGE:.2f} pages")
    print(f"  floats ({ntab} table)     {floats:.2f} pages")
    print(f"  references ({len(bibs)})     {refs:.2f} pages")
    print(f"  ESTIMATED TOTAL     {total:.2f} of {PAGE_LIMIT} pages")
    # WARNS, does not fail. CI compiles the paper and counts the pages for real,
    # and an estimate must never gate a measurement -- when this did fail the
    # build, it aborted the run before the compile step and left nobody able to
    # find out the true page count. The real limit is enforced against pdfinfo.
    if total > PAGE_LIMIT:
        print(f"  WARNING: over the {PAGE_LIMIT}-page limit by "
              f"{total - PAGE_LIMIT:.2f} -- the compile step has the real number")

    print()
    if problems:
        print(f"{len(problems)} problem(s):")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("no problems found")
    return 0


if __name__ == "__main__":
    sys.exit(main())
