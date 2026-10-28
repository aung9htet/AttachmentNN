#!/bin/bash
# Compile the architecture diagrams (paper Fig. 1 and supplement Fig. S1) in place: fig_*.pdf, and a
# 200-dpi fig_*.png preview when pdftoppm (poppler-utils) is available. Needs pdflatex, e.g. from the
# repository root:
#   docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp -v "$PWD:/w" -w /w texlive/texlive:latest \
#       figures/architecture/build.sh
set -euo pipefail
cd "$(dirname "$0")"
for f in fig_concept fig_architecture; do
    pdflatex -interaction=nonstopmode -halt-on-error "$f.tex" > /dev/null
    rm -f "$f.aux" "$f.log"
    if command -v pdftoppm > /dev/null; then
        pdftoppm -r 200 -png -singlefile "$f.pdf" "$f"
    else
        echo "pdftoppm not found: $f.png not updated"
    fi
    echo "built figures/architecture/$f.pdf"
done
