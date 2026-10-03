"""
economist_style.py — Minimal Economist-inspired chart styling for matplotlib.

Design cues (no extra dependencies):
  - White canvas, horizontal gridlines only, no top/right/left spines
  - Red "tag" rectangle and rule in the top-left corner
  - Bold left-aligned title, lighter subtitle beneath it
  - Small grey source note in the bottom-left corner
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Rectangle

# Economist-like palette
RED = "#E3120B"
BLUE = "#006BA2"
CYAN = "#3EBCD2"
GREEN = "#379A8B"
YELLOW = "#EBB434"
OLIVE = "#B4BA39"
MAUVE = "#9A607F"
GREY = "#758D99"
LIGHT_GREY = "#D9D9D9"
TEXT = "#0C0C0C"

PALETTE = [BLUE, CYAN, RED, GREEN, YELLOW, MAUVE, OLIVE, GREY]


def _pick_font() -> str:
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in ("Segoe UI", "Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"):
        if name in available:
            return name
    return "sans-serif"


def apply_style() -> None:
    """Set global rcParams for the house style."""
    plt.rcParams.update({
        "font.family": _pick_font(),
        "font.size": 10,
        "axes.edgecolor": TEXT,
        "axes.labelcolor": TEXT,
        "axes.titlesize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "axes.axisbelow": True,
        "grid.color": LIGHT_GREY,
        "grid.linewidth": 0.8,
        "xtick.color": TEXT,
        "ytick.color": TEXT,
        "ytick.major.size": 0,
        "xtick.major.size": 3,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.dpi": 200,
    })


def new_figure(width: float = 8.0, height: float = 5.0, ncols: int = 1, **kwargs):
    """Create a figure with room at the top for title/subtitle and bottom for source."""
    fig, axes = plt.subplots(1, ncols, figsize=(width, height), **kwargs)
    fig.subplots_adjust(top=0.80, bottom=0.14, left=0.08, right=0.96)
    return fig, axes


def decorate(
    fig,
    title: str,
    subtitle: str = "",
    source: str = "Source: Jev-Sinhala probe (jev-latest, zero-shot)",
) -> None:
    """Add red tag, title, subtitle and source note."""
    # Red rule across the top and short tag at the left
    fig.add_artist(plt.Line2D([0.03, 0.97], [0.985, 0.985], transform=fig.transFigure, color=RED, linewidth=1.2))
    fig.add_artist(Rectangle((0.03, 0.94), 0.035, 0.045, transform=fig.transFigure, color=RED, linewidth=0))
    fig.text(0.03, 0.905, title, fontsize=14, fontweight="bold", color=TEXT, ha="left", va="top")
    if subtitle:
        fig.text(0.03, 0.855, subtitle, fontsize=10.5, color=TEXT, ha="left", va="top")
    if source:
        fig.text(0.03, 0.025, source, fontsize=8, color=GREY, ha="left", va="bottom")


def save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path
