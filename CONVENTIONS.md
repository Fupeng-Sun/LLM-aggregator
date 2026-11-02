# Code conventions

Every file in this repository follows these rules. `uv run ruff check .`, `uv run ruff format --check .` and
`uv run pytest` must pass before a commit.

## Layout

| path | content |
|---|---|
| `src/llmagg/` | the model and every formula taken from the paper; nothing figure-specific |
| `figures/figNN_<name>.py` | one script per paper figure, NN = figure number in the paper |
| `tests/` | pytest: formulas against the paper's closed forms, checks of every figure |
| `output/figure/...` | generated files, at the same relative paths as in the paper's source (`figure/fig_pre/x.pdf`, `figure/aggregator_impact/<case>/x.dat`, ...) |

## Formulas

- Each formula from the paper is implemented once, in `src/llmagg`, with a docstring that names the paper's
  equation, result or appendix definition and uses the paper's notation (`a_A`, `delta_a`, `kappa_bar`, ...).
- Figure scripts import formulas; they never re-derive them.
- Where a result can be computed in two independent ways (closed form and the exact market model in
  `llmagg.market.revenues`), the tests check that they agree.

## Figure scripts

- **Module docstring:** the figure number and label, what each panel shows, the paper files it produces, and every
  parameter value with its source (caption, text, or "chosen for illustration").
- **Parameters:** module-level UPPER_CASE constants. No magic numbers in the drawing code except layout (offsets, label
  positions), which are grouped and commented.
- **Interface:**
  - `make(outdir: Path) -> list[Path]` writes the paper's files.
  - `check() -> list[str]` asserts that the drawn content agrees with the paper's statements (cutoffs, segmentations,
    regimes, tick positions) and returns a one-line description of each check.
  - The script ends with `if __name__ == "__main__": run_figure(make, check)` (from `llmagg.cli`).
- **Running:** `uv run python -m figures.figNN_<name> [--outdir output] [--paper-dir PATH]`.
- **Style:** call `llmagg.plotting.use_style()`; use its colors (`COLOR_A`, `COLOR_B`, `COLOR_G`, the colors of
  main.tex) and fonts (STIX, bundled with matplotlib). Figure-specific style constants sit at the top of the script.
  The figure size equals the size at which main.tex includes the figure, so that text appears at 8–10 pt.
- **Output:** PDFs through `llmagg.io.save_pdf`, pgfplots tables through `llmagg.io.write_dat`, so the output is
  deterministic and can be compared with the paper's files byte for byte.

## Python style

- Python ≥ 3.12, formatted with ruff (line length 120), imports sorted, no unused code.
- Type hints on public functions; short docstrings (what, not how) on every public function.
- No `sys.path` manipulation and no absolute paths; scripts are run as modules from the repository root.
- Deterministic: fixed seeds for any randomness, no time stamps in output.
- No commented-out code; no notebooks.
