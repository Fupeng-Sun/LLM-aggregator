"""Shared figure style.

All figures use fonts bundled with matplotlib (STIX, which matches the Times text of the INFORMS template), so that
output is identical on every machine with the pinned matplotlib version. Colors follow the definitions in main.tex:
provider A is ``cA``, provider B is ``cB`` and the aggregator G is TikZ's ``green!50!black``.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

COLOR_A = "#1F77B4"  # \definecolor{cA}{RGB}{31,119,180}
COLOR_B = "#FF7F0E"  # \definecolor{cB}{RGB}{255,127,14}
COLOR_G = "#008000"  # green!50!black
COLOR_OUTSIDE = "#808080"
COLOR_INK = "#000000"
PROVIDER_COLORS = {"A": COLOR_A, "B": COLOR_B, "G": COLOR_G}

FONT_SIZE = 10  # points at the size at which the figure is included in the paper

STYLE = {
    "font.family": "STIXGeneral",
    "mathtext.fontset": "stix",
    "font.size": FONT_SIZE,
    "axes.titlesize": FONT_SIZE,
    "axes.labelsize": FONT_SIZE,
    "xtick.labelsize": FONT_SIZE - 1,
    "ytick.labelsize": FONT_SIZE - 1,
    "legend.fontsize": FONT_SIZE - 1,
    "axes.linewidth": 0.6,
    "lines.linewidth": 1.2,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "pdf.fonttype": 42,  # embed TrueType fonts (journals reject Type 3)
    "savefig.bbox": "standard",
}


def use_style() -> None:
    """Apply the shared style to matplotlib's global settings."""
    plt.rcParams.update(STYLE)
