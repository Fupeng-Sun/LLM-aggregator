"""Common command line for the figure scripts in ``figures/``.

Every figure script defines ``make(outdir) -> list[Path]`` and ``check() -> list[str]`` and ends with
``run_figure(make, check)``. Running ``python -m figures.<name>`` then

1. runs the checks against the paper's results (each check raises ``AssertionError`` on failure),
2. writes the paper's files under ``--outdir`` at the paths main.tex uses (``figure/...``),
3. if ``--paper-dir`` (or the environment variable ``LLMAGG_PAPER_DIR``) points to the paper's source directory,
   compares every written file with the paper's copy byte for byte.
"""

import argparse
import os
from collections.abc import Callable
from pathlib import Path

from llmagg.io import compare_with_paper

DEFAULT_OUTDIR = Path("output")


def run_figure(make: Callable[[Path], list[Path]], check: Callable[[], list[str]]) -> None:
    """Run a figure script: checks, then output, then the optional comparison with the paper's files."""
    parser = argparse.ArgumentParser(description=(make.__module__ or "").split(".")[-1])
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR, help="root of the generated files")
    parser.add_argument(
        "--paper-dir",
        type=Path,
        default=os.environ.get("LLMAGG_PAPER_DIR"),
        help="the paper's source directory, for a byte-for-byte comparison",
    )
    args = parser.parse_args()
    for line in check():
        print(f"check passed: {line}")
    files = make(args.outdir)
    for f in files:
        print(f"wrote {f}")
    if args.paper_dir:
        lines = compare_with_paper(files, args.outdir, args.paper_dir)
        for line in lines:
            print(f"paper: {line}")
        if any("DIFFERENT" in line for line in lines):
            raise SystemExit("some files differ from the paper's copies")
