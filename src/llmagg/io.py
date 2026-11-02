"""Deterministic output: PDF figures without time stamps and pgfplots data tables.

Every file the paper includes is written through these functions, so regenerating it reproduces the paper's file
byte for byte (given the pinned package versions), and `compare_with_paper` can check the correspondence.
"""

import filecmp
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

PDF_METADATA = {"Creator": None, "Producer": None, "CreationDate": None, "ModDate": None}


def save_pdf(fig, path: Path) -> Path:
    """Save a matplotlib figure as a PDF without creation dates or tool names.

    The background is transparent: main.tex includes some panels wider than their subfigure (Figure 5 at
    1.15\\linewidth), so an opaque background would cover the labels of the neighbouring panel.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="pdf", metadata=PDF_METADATA, transparent=True)
    return path


def write_dat(path: Path, columns: Mapping[str, Sequence[float]], fmt: str = "%.12f") -> Path:
    """Write a whitespace-separated table with a header row, as read by pgfplots ``table[x=..., y=...]``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    names = list(columns)
    data = np.column_stack([np.asarray(columns[n], dtype=float) for n in names])
    np.savetxt(path, data, fmt=fmt, header=" ".join(names), comments="")
    return path


def compare_with_paper(files: Sequence[Path], outdir: Path, paper_dir: Path) -> list[str]:
    """Compare generated files with the paper's copies (same path relative to the paper's root directory).

    Returns one line per file: ``identical``, ``DIFFERENT`` or ``missing in paper``.
    """
    lines = []
    for f in files:
        rel = Path(f).relative_to(outdir)
        target = Path(paper_dir) / rel
        if not target.exists():
            status = "missing in paper"
        else:
            status = "identical" if filecmp.cmp(f, target, shallow=False) else "DIFFERENT"
        lines.append(f"{rel}: {status}")
    return lines
