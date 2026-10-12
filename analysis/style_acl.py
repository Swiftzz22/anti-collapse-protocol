"""Shared plotting style + data loaders for the ACP paper figures (ACL camera-ready look).

All data plotted come from the real experiment logs (raw_logs/) or the derived JSON files in
analysis/ (expA.json, expB.json, naive.json, rho.json, rep_sizecheck.json). Nothing is simulated.
"""
import json, glob, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D

for f in glob.glob("/usr/share/fonts/truetype/liberation/LiberationSerif-*.ttf"):
    font_manager.fontManager.addfont(f)

INK = "#222222"
GRID = "#E9E9E9"

plt.rcParams.update({
    "font.family": "Liberation Serif",      # metric-compatible with Times (ACL body font)
    "mathtext.fontset": "stix",
    "font.size": 7.5,
    "axes.titlesize": 7.6,
    "axes.labelsize": 7.4,
    "xtick.labelsize": 6.6,
    "ytick.labelsize": 6.6,
    "legend.fontsize": 6.3,
    "text.color": INK,
    "axes.labelcolor": INK,
    "axes.edgecolor": "#444444",
    "xtick.color": "#444444",
    "ytick.color": "#444444",
    "axes.linewidth": 0.55,
    "xtick.major.width": 0.55,
    "ytick.major.width": 0.55,
    "xtick.major.size": 2.2,
    "ytick.major.size": 2.2,
    "xtick.major.pad": 1.8,
    "ytick.major.pad": 1.8,
    "axes.labelpad": 2.0,
    "axes.titlepad": 4.0,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.color": GRID,
    "grid.linewidth": 0.5,
    "axes.axisbelow": True,
    "legend.frameon": False,
    "legend.handlelength": 1.4,
    "legend.handletextpad": 0.4,
    "legend.borderaxespad": 0.3,
    "lines.solid_capstyle": "round",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.015,
})

COL = 3.03     # ACL single-column width (in)  (219.1 pt)
FULL = 6.30    # ACL full text width (in)      (455.2 pt)

# ---------------------------------------------------------------- palette (Paul Tol, colour-blind safe)
C_MODEL = {"gemma2-9b": "#4477AA", "llama3.1-8b": "#AA3377", "mistral-7b": "#EE7733", "qwen2.5-7b": "#228833"}
N_MODEL = {"gemma2-9b": "Gemma-2-9B-it", "llama3.1-8b": "Llama-3.1-8B-Instruct",
           "mistral-7b": "Mistral-7B-Instruct", "qwen2.5-7b": "Qwen2.5-7B-Instruct"}
S_MODEL = {"gemma2-9b": "Gemma", "llama3.1-8b": "Llama", "mistral-7b": "Mistral", "qwen2.5-7b": "Qwen2.5"}
M_TASK = {"gsm8k": "o", "arc_challenge": "s", "commonsense_qa": "^", "boolq": "D"}
N_TASK = {"gsm8k": "GSM8K", "arc_challenge": "ARC-C", "commonsense_qa": "CSQA", "boolq": "BoolQ"}
C_TASK = {"gsm8k": "#332288", "arc_challenge": "#33BBEE", "commonsense_qa": "#CCBB44", "boolq": "#999999"}
C_GEN = "#8C8C8C"          # generated / student's own outputs
C_ACC = "#1F5A96"          # admitted by ACP
C_ACP, C_NAIVE = "#1F5A96", "#C0392B"
C_BB = {"Gemma-2-9B": "#4477AA", "Llama-3-8B": "#AA3377", "Mistral-7B-v0.3": "#EE7733", "Qwen2.5-7B": "#228833"}
S_BB = {"Gemma-2-9B": "Gemma-2-9B", "Llama-3-8B": "Llama-3-8B", "Mistral-7B-v0.3": "Mistral-7B", "Qwen2.5-7B": "Qwen2.5-7B"}

# loop-stage colours (shared with the TikZ loop figure)
ST = {"Synthesize": "#5B6770", "Screen": "#1F4E79", "Train": "#227755", "Measure": "#5E3C8C", "Adapt": "#A8322D"}

# screening funnel colours
C_FUN = {"consensus": "#DDAA33", "T1": "#1F4E79", "T2": "#6699CC", "dedup": "#B8D2EC", "T34": "#BBBBBB",
         "admitted": "#44AA99"}

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "figures")
RAW = os.path.join(HERE, "..", "raw_logs")
os.makedirs(OUT, exist_ok=True)


def load_B():
    return json.load(open(os.path.join(HERE, "expB.json")))


def load_A():
    return json.load(open(os.path.join(HERE, "expA.json")))


def save(fig, name, png=True):
    fig.savefig(os.path.join(OUT, name + ".pdf"))
    if png:
        fig.savefig(os.path.join(OUT, name + ".png"), dpi=300)
    plt.close(fig)
    print("saved", name)


def ptitle(ax, letter, text, x=0.0, y=1.0, fs=7.6):
    """Left-aligned panel title with a bold panel letter, e.g. (a) accuracy ..."""
    s = (r"$\mathbf{(" + letter + r")}$ " if letter else "") + text
    ax.set_title(s, loc="left", fontsize=fs, x=x, y=y)


def clean(ax, xgrid=False):
    if xgrid:
        ax.grid(True, axis="both")
    ax.tick_params(axis="both", which="both", length=2.2)


def mean_line(ax, x, y, color, marker="o", lw=1.6, ms=3.2, label=None, ls="-", z=5):
    ax.plot(x, y, color=color, lw=lw, ls=ls, marker=marker, ms=ms, mfc=color, mec="white", mew=0.6,
            label=label, zorder=z)


def thin(ax, x, Y, color, lw=0.5, alpha=0.28):
    for y in Y:
        ax.plot(x, y, color=color, lw=lw, alpha=alpha, zorder=2)


def note(ax, x, y, s, ha="left", va="top", color=INK, fs=6.0, box=True, **kw):
    bb = dict(fc="white", ec="none", alpha=0.85, pad=0.8) if box else None
    return ax.text(x, y, s, transform=ax.transAxes, ha=ha, va=va, fontsize=fs, color=color, bbox=bb, **kw)
