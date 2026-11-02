# Code of "LLM Competition and the Aggregator Economy"

Fupeng Sun, Zhe Liu and Wolfram Wiesemann. This repository produces the computed figures of the paper. Each file the
paper includes under `figure/` is written by a script here, at the same relative path, and can be compared with the
paper's copy byte for byte.

## Quick start

```sh
uv sync                                   # Python >= 3.12 with the pinned versions of uv.lock
./reproduce.sh figures                    # every figure file, with its checks against the paper's results (seconds)
./reproduce.sh compare /path/to/paper     # byte-for-byte comparison with the paper's figure/ folder
./reproduce.sh check                      # ruff and pytest
```

`./reproduce.sh install DIR` copies `output/figure/` into a paper's `figure/` folder.

## Figures

| Paper | Content | Script | Files written (under `output/`) |
|---|---|---|---|
| Figure 3(a) | utility profiles | `figures/fig03_utility_profiles.py` | `figure/fig_pre/utility_functions.pdf` |
| Figure 4 | segmentation without the aggregator (Proposition 1) | `figures/fig04_segmentation_without_aggregator.py` | `figure/fig_pre/segmentation_2-way*.pdf` |
| Figure 5 | segmentation with the aggregator (Theorem 1) | `figures/fig05_segmentation_with_aggregator.py` | `figure/fig_pre/segmentation_3-way*.pdf` |
| Figure 6 | aggregator impact, Theorem 2(ii) | `figures/fig06_impact_market_expansion.py` | `figure/fig_pre/impact_G_3-way-2.pdf`, `figure/aggregator_impact/{p_L_b_B,b_B_p_R}/*.dat` |
| Figure 7 | aggregator impact, Theorem 2(iii) | `figures/fig07_impact_reallocation.py` | `figure/fig_pre/impact_G_3-way-1.pdf`, five `figure/aggregator_impact/*/` folders |
| Figure 8 | aggregator impact, Theorem 2(iv) | `figures/fig08_impact_saturated_market.py` | `figure/fig_pre/impact_G_2-way.pdf`, `figure/aggregator_impact/pstar_less_lower_{A,B}/*.dat` |
| Figure 10 | competition patterns (Theorem 4) | `figures/fig10_competition_patterns.py` | `figure/fig_pre/pattern_*.pdf`, `data/fig10/*.csv` |

Figures 1, 2, 3(b), 9 and 11 are TikZ diagrams in the paper's source and contain no computed content. The threshold
panels of Figures 6–8 are drawn by pgfplots in the paper from the `.dat` files listed.

Each figure script lists its parameters and their source in its docstring. Running it first checks the drawn content
against the paper's results, for example cutoffs, segmentations, price regions and the periods in the subcaptions of
Figure 10. It then writes the files. The tests also check the model's implementation against the paper's closed
forms, and, when `LLMAGG_PAPER_DIR` points to the paper's source, that every tick hard-coded in the paper's pgfplots
code equals its model value.

## Layout

| Path | Content |
|---|---|
| `src/llmagg/` | the model: `market` (exact market model, revenues, the myopic heuristic), `segmentation` (Proposition 1, Theorem 1), `impact` (thresholds of Theorem 2), `patterns` (Proposition 4, Definition 3), and `io`, `plotting`, `cli` (deterministic output, shared style, figure command line) |
| `figures/` | one script per figure, `figNN_*.py`, with shared helpers `segmentation_panels.py` and `impact_panels.py` |
| `tests/` | formulas against the paper's closed forms, and checks of every figure |
| `output/` | generated files, at the paper's paths |

Conventions for all code are in [CONVENTIONS.md](CONVENTIONS.md).

## License

The code and generated files are released under the MIT License (see [LICENSE](LICENSE)).
