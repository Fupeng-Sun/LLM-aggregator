#!/bin/zsh
# Reproduce the computed figures of the paper.
#
#   ./reproduce.sh figures            every figure file of the paper, with its checks (seconds)
#   ./reproduce.sh check              lint and tests
#   ./reproduce.sh install PAPER_DIR  copy output/figure/ into the paper's figure/ folder
#   ./reproduce.sh compare PAPER_DIR  compare every generated file with the paper's copy, byte for byte
#
# Generated files go to output/figure/..., the same relative paths as in the paper's source.
set -e
cd "${0:A:h}"
RUN=(uv run --frozen)
FIGURES=(${(f)"$(ls figures/fig*.py | sed 's#/#.#; s#\.py$##')"})

stage_check() {
  $RUN ruff check .
  $RUN ruff format --check .
  $RUN pytest -q
}
stage_figures() {
  for f in $FIGURES; do echo "== $f"; $RUN python -m $f "$@"; done
}
stage_install() {
  [ -d "$1/figure" ] || { echo "usage: ./reproduce.sh install PAPER_DIR (a folder with figure/)"; exit 1; }
  rsync -a output/figure/ "$1/figure/" && echo "copied output/figure/ to $1/figure/"
}
stage_compare() {
  [ -d "$1/figure" ] || { echo "usage: ./reproduce.sh compare PAPER_DIR"; exit 1; }
  stage_figures --paper-dir "$1"
}

case "${1:-figures}" in
  check|figures) stage_$1 ;;
  install|compare) stage_$1 "$2" ;;
  *) echo "usage: ./reproduce.sh [figures|check|install DIR|compare DIR]"; exit 1 ;;
esac
