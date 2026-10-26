#!/bin/bash
# Package the TCDS draft for Overleaf as the next numbered version:
#   papers/tcds/draft/overleaf_import/paper_draft_ver_{N}.zip
# The zip has main.tex (the paper) and supplement.tex (the supplementary material) at its root, the
# vector figures (with the TikZ sources of Figs. 1-2), the tables, the bibliography and the IEEEtran
# class/style (unmodified, from TeX Live), and no build files.
#
# Before the zip is kept it is checked against the TCDS rules:
#   - the paper (main.tex) ends by page 10, references and biographies included; supplementary
#     material is not counted, so longer papers would pay the mandatory overlength charge;
#   - the abstract has 150-250 words;
#   - every supplementary section, figure and table cited in main.tex exists in supplement.tex;
# and it is unpacked in a clean folder and compiled, both as the working draft and with
# \draftmarkersfalse (the submission version).
#
# Needs pdflatex, bibtex, texcount, perl and kpsewhich; without a local TeX Live, run it in the TeX Live image:
#   docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp -v "$PWD:/w" -w /w \
#       texlive/texlive:latest papers/tcds/make_overleaf_zip.sh
set -euo pipefail

MAX_PAGES=10
HERE="$(cd "$(dirname "$0")" && pwd)"
DRAFT="$HERE/draft"
OUT_DIR="$DRAFT/overleaf_import"
mkdir -p "$OUT_DIR"
N=1
while [ -e "$OUT_DIR/paper_draft_ver_${N}.zip" ]; do N=$((N + 1)); done
OUT="$OUT_DIR/paper_draft_ver_${N}.zip"
STAGE="$(mktemp -d)"
TEST="$(mktemp -d)"
SUBMIT="$(mktemp -d)"
trap 'rm -rf "$STAGE" "$TEST" "$SUBMIT"' EXIT

build() {  # build <dir> <name>: pdflatex, bibtex, pdflatex, pdflatex (output discarded)
    (cd "$1" && { pdflatex -interaction=nonstopmode "$2"; bibtex "$2"; pdflatex -interaction=nonstopmode "$2";
                  pdflatex -interaction=nonstopmode "$2"; } > /dev/null 2>&1) || true
}
clean_log() {  # clean_log <dir> <name>: the build produced a PDF without errors or undefined references
    [ -f "$1/$2.pdf" ] && ! grep -q "^!" "$1/$2.log" && ! grep -qi "undefined" "$1/$2.log"
}
fail() { echo "$1" >&2; rm -f "$OUT"; exit 1; }

# fresh builds and bibliographies for the zip
build "$DRAFT" supplement
build "$DRAFT" main

# ---------------------------------------------------------------- TCDS checks
END_PAGE=$(perl -ne 'print $1 if /\\newlabel\{paper:end\}\{\{[^}]*\}\{(\d+)\}/' "$DRAFT/main.aux")
[ -n "$END_PAGE" ] || fail "label paper:end not found in main.aux"
[ "$END_PAGE" -le "$MAX_PAGES" ] || fail "the paper ends on page $END_PAGE, beyond the $MAX_PAGES-page limit"
ABSTRACT_FILE="$STAGE/abstract.tex"
sed -n '/\\begin{abstract}/,/\\end{abstract}/p' "$DRAFT/main.tex" > "$ABSTRACT_FILE"
WORDS=$(texcount -1 -sum "$ABSTRACT_FILE" 2>/dev/null | tail -n 1 | tr -dc '0-9')
rm -f "$ABSTRACT_FILE"
{ [ "$WORDS" -ge 150 ] && [ "$WORDS" -le 250 ]; } || fail "the abstract has $WORDS words; TCDS requires 150-250"
# supplementary items cited in main.tex (Fig.~S3, Table~S1, Section~S2, Sections~S1--S6) must exist
perl -e '
    my %have;
    open(my $aux, "<", $ARGV[0]) or die;
    while (<$aux>) { $have{"$1:$2"} = 1 if /\\newlabel\{(fig|tab|sec):[^}]*\}\{\{S(\d+)\}/ }
    open(my $tex, "<", $ARGV[1]) or die; local $/; my $t = <$tex>;
    my %kind = ("Fig." => "fig", "Figs." => "fig", "Table" => "tab", "Tables" => "tab", "Section" => "sec", "Sections" => "sec");
    my @missing;
    while ($t =~ /(Figs?\.|Tables?|Sections?)~S(\d+)(?:--S(\d+))?/g) {
        my ($k, $a, $b) = ($kind{$1}, $2, $3 // $2);
        for my $n ($a .. $b) { push @missing, "$1 S$n" unless $have{"$k:$n"} }
    }
    if (@missing) { print STDERR "missing in supplement.tex: @missing\n"; exit 1 }
' "$DRAFT/supplement.aux" "$DRAFT/main.tex" || fail "main.tex cites supplementary items that do not exist"

# ---------------------------------------------------------------- package
mkdir -p "$STAGE/figures" "$STAGE/tables" "$STAGE/diagrams"
cp "$DRAFT/main.tex" "$DRAFT/supplement.tex" "$DRAFT/refs.bib" "$DRAFT/main.bbl" "$DRAFT/supplement.bbl" "$STAGE/"
cp "$DRAFT"/figures/fig_*.pdf "$DRAFT"/figures/npj_*.png "$STAGE/figures/"
cp "$DRAFT"/tables/*.tex "$STAGE/tables/"
cp "$DRAFT"/diagrams/*.tex "$STAGE/diagrams/"
cp "$(kpsewhich IEEEtran.cls)" "$(kpsewhich IEEEtran.bst)" "$STAGE/"

cat > "$STAGE/README_overleaf.txt" <<TXT
Version ${N} of the TCDS draft, packaged $(date "+%Y-%m-%d %H:%M").
Overleaf: New Project > Upload Project > select this zip.
Compiler: pdfLaTeX. Main document: main.tex (Menu > Settings if Overleaf does not pick it).
The supplementary material is supplement.tex: choose it as the main document to compile it.
Format: IEEE Transactions journal (IEEEtran class, bundled unmodified), for IEEE TCDS.
TCDS limits, checked when this zip was made: the paper ends on page ${END_PAGE} of at most ${MAX_PAGES}
(references and biographies included; supplementary material is not counted) and the abstract has
${WORDS} words (150-250 required).
Draft markers (\\draftmarkerstrue in main.tex and supplement.tex):
  red   [CITE: ...]     a claim that needs a source beyond the LM/npj reference lists;
                        all of them are listed on a draft-only page after the biographies.
  blue  [Authors: ...]  a point to check before submission.
  grey  biographies and photo frames are placeholders that reserve the space of the final ones.
Set \\draftmarkersfalse in both files for the submission version.
Figs. 1-2 are TikZ diagrams: sources in diagrams/ (compile fig_*.tex with pdfLaTeX and
replace figures/fig_concept.pdf and figures/fig_architecture.pdf to edit them).
Figures and tables are generated by analysis/tcds in the AttachmentNN repository.
TXT

(cd "$STAGE" && zip -q -r -X "$OUT" README_overleaf.txt main.tex supplement.tex refs.bib main.bbl supplement.bbl \
    IEEEtran.cls IEEEtran.bst figures tables diagrams)

# ---------------------------------------------------------------- the zip compiles on its own
(cd "$TEST" && unzip -q "$OUT" && rm main.bbl supplement.bbl)
build "$TEST" supplement
build "$TEST" main
if ! clean_log "$TEST" main || ! clean_log "$TEST" supplement; then
    cp "$TEST/main.log" "$OUT_DIR/failed_build.log" 2>/dev/null || true
    fail "the packaged draft does not compile cleanly; see $OUT_DIR/failed_build.log"
fi
# ... and so does the submission version, without draft markers
(cd "$SUBMIT" && unzip -q "$OUT" && sed -i 's/^\\draftmarkerstrue/\\draftmarkersfalse/' main.tex supplement.tex)
build "$SUBMIT" supplement
build "$SUBMIT" main
clean_log "$SUBMIT" main && clean_log "$SUBMIT" supplement || fail "the submission version (\\draftmarkersfalse) does not compile cleanly"
SUBMIT_END=$(perl -ne 'print $1 if /\\newlabel\{paper:end\}\{\{[^}]*\}\{(\d+)\}/' "$SUBMIT/main.aux")

PAGES=$(grep -o "Output written on main.pdf ([0-9]* pages" "$TEST/main.log" | grep -o "[0-9]* pages")
SPAGES=$(grep -o "Output written on supplement.pdf ([0-9]* pages" "$TEST/supplement.log" | grep -o "[0-9]* pages")
echo "wrote $OUT"
echo "  paper ends on page $END_PAGE of $MAX_PAGES (submission version: page $SUBMIT_END); abstract $WORDS words"
echo "  main.pdf $PAGES (with the draft-only page), supplement.pdf $SPAGES; both compiled cleanly from the zip"
