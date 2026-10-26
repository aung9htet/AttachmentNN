#!/bin/bash
# Compile the TikZ diagrams (Figs. 1-2) and copy them into ../figures.
set -euo pipefail
cd "$(dirname "$0")"
for f in fig_concept fig_architecture; do
    pdflatex -interaction=nonstopmode -halt-on-error "$f.tex" > /dev/null
    cp "$f.pdf" "../figures/$f.pdf"
    pdftoppm -r 200 -png -singlefile "$f.pdf" "../figures/$f"
    rm -f "$f.aux" "$f.log"
    echo "built figures/$f.pdf"
done
