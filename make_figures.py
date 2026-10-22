"""
make_figures.py
===============

Generates every figure used in the final engineering report.

* Figures 3-8 and 11 are computed directly from final_results.csv (measured data).
* Figures 1, 2 and 9 are conceptual diagrams (no data).
* Figure 10 is an analytical illustration derived from the reward definitions
  (no measured data).

Usage:  python make_figures.py [path/to/final_results.csv]
Output: ../Figures/Fig*.png (300 dpi) and ../Figures/Fig*.pdf (vector)
Dependencies: numpy, pandas, matplotlib, scipy
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from scipy import stats

HERE = Path(__file__).resolve().parent
CSV = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parent / "Original_Supplied_Materials" / "final_results.csv"
FIG = HERE.parent / "Figures"
FIG.mkdir(exist_ok=True)

# ----------------------------------------------------------------- style ----
# Inter is used if available. Set FIG_FONT_DIR to a folder of Inter .ttf files for the
# cleanest PDF embedding; otherwise any system-installed Inter (.otf) is used.
import os
_ttf_dir = os.environ.get("FIG_FONT_DIR")
_font_files = list(Path(_ttf_dir).glob("Inter-*.ttf")) if _ttf_dir else []
if not _font_files:
    _font_files = list(Path("/usr/share/fonts/opentype/inter").glob("Inter-*.otf"))
if _ttf_dir and _font_files:   # drop cached .otf Inter entries so the .ttf versions are used
    font_manager.fontManager.ttflist = [f for f in font_manager.fontManager.ttflist
                                        if not (f.name.startswith("Inter") and f.fname.endswith(".otf"))]
for f in _font_files:
    font_manager.fontManager.addfont(str(f))
FAMILY = "Inter" if any("Inter" in f.name for f in font_manager.fontManager.ttflist) else "DejaVu Sans"

INK = "#1f2933"        # primary text
INK2 = "#52606d"       # secondary text
GRID = "#e4e7eb"
NAVY = "#14284b"
RULE = "#9aa5b1"

mpl.rcParams.update({
    "font.family": FAMILY, "font.size": 8.5, "axes.titlesize": 9.5,
    "axes.labelsize": 8.5, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 7.8, "axes.edgecolor": RULE, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8, "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.7,
    "axes.axisbelow": True, "savefig.dpi": 300, "figure.dpi": 150,
    "pdf.fonttype": 42, "axes.titleweight": "semibold", "axes.titlelocation": "left",
    "axes.titlepad": 8,
    "mathtext.fontset": "custom", "mathtext.rm": FAMILY, "mathtext.it": f"{FAMILY}:italic",
    "mathtext.bf": f"{FAMILY}:bold",
})

ORDER = ["Original PPO", "Time Penalty", "Hover Reward", "Corrected Reward"]
LABEL = {"Original PPO": "Baseline", "Time Penalty": "Time penalty",
         "Hover Reward": "Hover reward", "Corrected Reward": "Corrected reward"}
TICK = {"Original PPO": "Baseline\n(Agent 1)", "Time Penalty": "Time penalty\n(Agent 2)",
        "Hover Reward": "Hover reward\n(Agent 3)", "Corrected Reward": "Corrected reward\n(Agent 4)"}
# Validated categorical palette (all-pairs; secondary encoding via labels/markers)
COL = {"Original PPO": "#2a6cc0", "Time Penalty": "#a0569a",
       "Hover Reward": "#cf7020", "Corrected Reward": "#1a9a72"}
MARK = {"Original PPO": "o", "Time Penalty": "s", "Hover Reward": "^", "Corrected Reward": "D"}
W = 6.3   # text width in inches (A4, 25 mm margins)
RNG = np.random.default_rng(20261021)
N_BOOT = 10_000


def save(fig, name):
    fig.savefig(FIG / f"{name}.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", name)


def boot_mean_ci(x):
    idx = RNG.integers(0, len(x), size=(N_BOOT, len(x)))
    reps = x[idx].mean(axis=1)
    return np.quantile(reps, [0.025, 0.975])


def wilson(k, n, z=1.959963984540054):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0, c - h), min(1, c + h)


def t_ci(x):
    se = x.std(ddof=1) / np.sqrt(len(x))
    h = stats.t.ppf(0.975, len(x) - 1) * se
    return x.mean() - h, x.mean() + h


df = pd.read_csv(CSV)
df["frac"] = df["slow_upright_airborne_frames"] / df["duration_frames"]
G = {m: df[df["model"] == m].sort_values("seed") for m in ORDER}
X = np.arange(4)


def cat_axis(ax):
    ax.set_xticks(X, [TICK[m] for m in ORDER])
    ax.tick_params(axis="x", length=0, pad=4)
    ax.set_xlim(-0.6, 3.6)


# ------------------------------------------------------- diagram helpers ----
def box(ax, x, y, w, h, title, sub=None, fc="#ffffff", ec=NAVY, tc=NAVY, lw=1.1, ls="-",
        title_size=8.2, sub_size=7.1, weight="semibold"):
    """Rounded box with a title anchored near the top and optional sub-text below it."""
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.0,rounding_size=0.06",
                       fc=fc, ec=ec, lw=lw, ls=ls)
    ax.add_patch(p)
    if sub:
        ax.text(x + w / 2, y + h - 0.16, title, ha="center", va="top", fontsize=title_size,
                color=tc, weight=weight)
        ax.text(x + w / 2, y + h - 0.50, sub, ha="center", va="top", fontsize=sub_size,
                color=INK2, linespacing=1.4)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center", fontsize=title_size,
                color=tc, weight=weight, linespacing=1.3)


def arrow(ax, p0, p1, text=None, toff=(0, 0.14), color=NAVY, rad=0.0, ls="-", lw=1.1,
          ha="center", fs=7.6, tcolor=None):
    a = FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=10, color=color, lw=lw,
                        connectionstyle=f"arc3,rad={rad}", linestyle=ls, shrinkA=0, shrinkB=0)
    ax.add_patch(a)
    if text:
        mx, my = (p0[0] + p1[0]) / 2 + toff[0], (p0[1] + p1[1]) / 2 + toff[1]
        ax.text(mx, my, text, ha=ha, va="center", fontsize=fs, color=tcolor or INK)


def canvas(h, xmax, ymin, ymax):
    fig, ax = plt.subplots(figsize=(W, h))
    ax.set_xlim(0, xmax)
    ax.set_ylim(ymin, ymax)
    ax.axis("off")
    ax.grid(False)
    return fig, ax


def tag(ax, x, y, text="CONCEPTUAL DIAGRAM  ·  NOT EXPERIMENTAL DATA"):
    ax.text(x, y, text, fontsize=6.6, color=INK2, ha="left", va="top", weight="medium")


# ============================================================ Figure 1 =====
def fig1():
    fig, ax = canvas(2.75, 10, 0.0, 4.3)
    tag(ax, 0.1, 4.28)
    box(ax, 0.1, 1.55, 3.2, 2.0, "Agent",
        "PPO actor–critic (MLP)\npolicy  $\\pi_\\theta(a\\mid s)$\nvalue estimate  $V_\\phi(s)$")
    box(ax, 6.7, 1.55, 3.2, 2.0, "Environment",
        "Gymnasium LunarLander-v3\nBox2D rigid-body physics\n50 frames per simulated second")
    arrow(ax, (3.3, 3.05), (6.7, 3.05), "action  $a_t \\in \\{0,1,2,3\\}$", toff=(0, 0.2))
    arrow(ax, (6.7, 2.2), (3.3, 2.2), "observation  $s_{t+1} \\in \\mathbb{R}^8$", toff=(0, 0.2))
    ax.text(5.0, 1.45, "wrapper is removed at evaluation;\nscores use the base reward only",
            ha="center", va="center", fontsize=6.9, color=INK2, linespacing=1.4, style="italic")
    box(ax, 4.1, 0.15, 1.8, 0.8, "Reward wrapper\n(training only)", None, fc="#f3f6fa", ec=RULE,
        ls=(0, (3, 2)), title_size=7.4)
    ax.plot([8.3, 8.3, 5.9], [1.55, 0.55, 0.55], color=NAVY, lw=1.1)
    ax.text(7.1, 0.72, "base reward $r_{t+1}$", fontsize=7.6, ha="center", color=INK)
    ax.plot([4.1, 1.7], [0.55, 0.55], color=NAVY, lw=1.1)
    arrow(ax, (1.7, 0.55), (1.7, 1.55))
    ax.text(2.9, 0.72, "training reward $\\tilde r_{t+1}$", fontsize=7.6, ha="center", color=INK)
    save(fig, "Fig01_agent_environment_loop")


# ============================================================ Figure 2 =====
def fig2():
    fig, ax = canvas(3.55, 10, 0.0, 5.75)
    tag(ax, 0.05, 5.73)
    ax.text(0.05, 5.12, "TRAINING  ·  once per reward condition", fontsize=7.4, color=NAVY, weight="semibold")
    xs = [0.05, 2.6, 5.15, 7.7]
    w, y, h = 2.25, 3.05, 1.75
    steps = [("1 · Collect rollout", "policy $\\pi_{\\theta_{old}}$ acts\nfor 2,048 steps;\nwrapper sets reward"),
             ("2 · Advantages", "GAE estimates with\n$\\gamma$ = 0.99, $\\lambda$ = 0.95"),
             ("3 · Policy update", "clip range $\\epsilon$ = 0.2;\n10 epochs,\nminibatch size 64"),
             ("4 · Budget check", "repeat to ≈ 500,000\nsteps; then save\nthe checkpoint")]
    for x0, (t, s) in zip(xs, steps):
        box(ax, x0, y, w, h, t, s, sub_size=6.8)
    for x0 in xs[:-1]:
        arrow(ax, (x0 + w, y + h / 2), (x0 + 2.55, y + h / 2))
    arrow(ax, (8.825, y), (1.175, y), rad=-0.13, ls=(0, (3, 2)), color=RULE)
    ax.text(5.0, 2.33, "repeat until the step budget is reached", fontsize=6.9, color=INK2, ha="center")
    ax.plot([0.05, 9.95], [2.05, 2.05], color=GRID, lw=1)
    ax.text(0.05, 1.83, "EVALUATION  ·  all four checkpoints", fontsize=7.4, color=NAVY, weight="semibold")
    y2, h2 = 0.2, 1.5
    ev = [("Original reward", "wrappers removed;\nbase reward only"),
          ("Matched seeds", "seeds 1000–1099;\n100 episodes each"),
          ("Greedy actions", "deterministic=True;\nno action sampling"),
          ("Episode record", "score, duration,\nlanding flag,\nslow/upright frames")]
    for x0, (t, s) in zip(xs, ev):
        box(ax, x0, y2, w, h2, t, s, fc="#f3f6fa", sub_size=6.8)
    for x0 in xs[:-1]:
        arrow(ax, (x0 + w, y2 + h2 / 2), (x0 + 2.55, y2 + h2 / 2))
    save(fig, "Fig02_ppo_training_evaluation_workflow")


# ============================================================ Figure 3 =====
def fig3():
    fig, ax = plt.subplots(figsize=(W, 3.0))
    for i, m in enumerate(ORDER):
        s = G[m]["score"].to_numpy()
        lo, hi = boot_mean_ci(s)
        ax.bar(i, s.mean(), width=0.56, color=COL[m], edgecolor="white", lw=0, zorder=2)
        ax.errorbar(i, s.mean(), yerr=[[s.mean() - lo], [hi - s.mean()]], color=INK, lw=1,
                    capsize=4, capthick=1, zorder=3)
        off = 14 if s.mean() >= 0 else -14
        y = hi + 8 if s.mean() >= 0 else lo - 8
        ax.text(i + 0.31, s.mean(), f"{s.mean():.1f}".replace("-", "\u2212"), ha="left", va="center", fontsize=8.2,
                color=INK, weight="semibold")
    ax.axhline(0, color=INK2, lw=0.9, zorder=2)
    ax.axhline(200, color=INK2, lw=0.8, ls=(0, (4, 3)), zorder=1)
    ax.text(3.58, 204, "200-point 'solved' reference", fontsize=7.2, color=INK2, ha="right", va="bottom")
    ax.set_ylabel("Mean evaluation score (points)")
    ax.set_ylim(-100, 240)
    cat_axis(ax)
    ax.set_title("Mean score over 100 evaluation episodes, unmodified reward", color=NAVY)
    save(fig, "Fig03_mean_score")


# ============================================================ Figure 4 =====
def fig4():
    fig, (a, b) = plt.subplots(1, 2, figsize=(W, 2.9), gridspec_kw={"width_ratios": [1, 1.1], "wspace": 0.55})
    for i, m in enumerate(ORDER):
        k = int(G[m]["successful_landing"].sum())
        lo, hi = wilson(k, 100)
        a.bar(i, k, width=0.56, color=COL[m], zorder=2)
        a.errorbar(i, k, yerr=[[max(0.0, k - 100 * lo)], [max(0.0, 100 * hi - k)]], color=INK, lw=1, capsize=4, zorder=3)
        a.text(i, 100 * hi + 3, f"{k}/100", ha="center", fontsize=7.8, color=INK, weight="semibold")
    a.set_ylim(0, 100)
    a.set_ylabel("Episodes ending in a landing (%)")
    a.set_xticks(X, ["Baseline", "Time\npenalty", "Hover\nreward", "Corrected\nreward"])
    a.tick_params(axis="x", length=0)
    a.set_title("(a) Landing rate, 95% Wilson CI", color=NAVY)

    cats = [("Landed (came to rest)", NAVY, None, "white"),
            ("Terminated without landing", "#8795a8", None, "white"),
            ("Time limit reached (1,000 frames)", "#dfe4ea", "////", INK)]
    yy = np.arange(4)[::-1]
    for j, m in enumerate(ORDER):
        g = G[m]
        land = int(g["successful_landing"].sum())
        tl = int((g["duration_frames"] >= 1000).sum())
        term = 100 - land - tl
        left = 0
        for (name, c, hatch, tc), v in zip(cats, [land, term, tl]):
            if v == 0:
                continue
            b.barh(yy[j], v, left=left, height=0.6, color=c, hatch=hatch, edgecolor="white",
                   lw=1.2, zorder=2)
            if v >= 6:
                b.text(left + v / 2, yy[j], str(v), ha="center", va="center", fontsize=7.6,
                       color=tc, weight="semibold", zorder=4)
            left += v
    mpl.rcParams["hatch.color"] = "#9aa5b1"
    b.set_yticks(yy, ["Baseline", "Time penalty", "Hover reward", "Corrected reward"])
    b.tick_params(axis="y", length=0)
    b.set_xlim(0, 100)
    b.set_xlabel("Episodes (of 100)")
    b.grid(axis="x", color=GRID)
    b.grid(axis="y", visible=False)
    b.set_title("(b) How each episode ended", color=NAVY)
    handles = [mpl.patches.Patch(facecolor=c, hatch=h, edgecolor="white" if not h else RULE, label=n)
               for n, c, h, _ in cats]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=3, frameon=False,
               handlelength=1.4, fontsize=7.4)
    save(fig, "Fig04_landing_success_and_outcomes")


# ============================================================ Figure 5 =====
def fig5():
    fig, ax = plt.subplots(figsize=(W, 3.4))
    data = [G[m]["score"].to_numpy() for m in ORDER]
    bp = ax.boxplot(data, positions=X, widths=0.42, patch_artist=True, showfliers=False,
                    medianprops=dict(color=INK, lw=1.6), whiskerprops=dict(color=INK2, lw=0.9),
                    capprops=dict(color=INK2, lw=0.9), boxprops=dict(lw=0.9, edgecolor=INK2))
    for patch, m in zip(bp["boxes"], ORDER):
        patch.set_facecolor(mpl.colors.to_rgba(COL[m], 0.16))
    for i, m in enumerate(ORDER):
        s = G[m]["score"].to_numpy()
        jit = RNG.uniform(-0.27, 0.27, len(s))
        ax.scatter(i + jit, s, s=11, color=COL[m], alpha=0.75, lw=0.4, edgecolor="white",
                   marker=MARK[m], zorder=3)
        ax.scatter(i + 0.36, s.mean(), marker="D", s=26, color="white", edgecolor=INK, lw=1, zorder=4)
    ax.axhline(200, color=INK2, lw=0.8, ls=(0, (4, 3)))
    ax.axhline(0, color=INK2, lw=0.6)
    ax.text(2.0, 206, "200-point reference", fontsize=7.2, color=INK2, ha="center", va="bottom")
    ax.set_ylabel("Episode score (points)")
    ax.set_ylim(-300, 320)
    cat_axis(ax)
    ax.set_title("Episode scores (one point per episode)", color=NAVY)
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=INK, lw=1.6, label="median"),
         Line2D([], [], marker="D", color="white", markeredgecolor=INK, lw=0, markersize=5, label="mean"),
         mpl.patches.Patch(facecolor="#eef1f5", edgecolor=INK2, label="interquartile range")]
    ax.legend(handles=h, loc="lower right", ncol=1, frameon=False, fontsize=7.4)
    save(fig, "Fig05_score_distributions")


# ============================================================ Figure 6 =====
def fig6():
    fig, ax = plt.subplots(figsize=(W, 2.9))
    for i, m in enumerate(ORDER):
        d = G[m]["duration_frames"].to_numpy().astype(float)
        lo, hi = t_ci(d)
        ax.bar(i, d.mean(), width=0.56, color=COL[m], zorder=2)
        ax.errorbar(i, d.mean(), yerr=[[d.mean() - lo], [hi - d.mean()]], color=INK, lw=1, capsize=4, zorder=3)
        ax.scatter(i, np.median(d), marker="_", s=420, color=INK, lw=1.6, zorder=4)
        ax.text(i, 30, f"mean {d.mean():.0f}\nmedian {np.median(d):.0f}",
                fontsize=7.2, va="bottom", ha="center", color="white", linespacing=1.35, weight="medium")
    ax.axhline(1000, color=INK2, lw=0.9, ls=(0, (4, 3)))
    ax.text(-0.55, 1015, "time limit: 1,000 frames (20 s simulated)", fontsize=7.2, color=INK2, va="bottom")
    ax.set_ylim(0, 1100)
    ax.set_ylabel("Episode duration (frames)")
    cat_axis(ax)
    ax.set_title("Episode duration: bars = mean ± 95% CI, black tick = median", color=NAVY)
    save(fig, "Fig06_episode_duration")


# ============================================================ Figure 7 =====
def fig7():
    fig, (a, b) = plt.subplots(1, 2, figsize=(W, 2.95), gridspec_kw={"wspace": 0.38})
    for i, m in enumerate(ORDER):
        h = G[m]["slow_upright_airborne_frames"].to_numpy().astype(float)
        lo, hi = boot_mean_ci(h)
        a.bar(i, h.mean(), width=0.56, color=COL[m], zorder=2)
        a.errorbar(i, h.mean(), yerr=[[h.mean() - lo], [hi - h.mean()]], color=INK, lw=1, capsize=4, zorder=3)
        a.text(i, hi + 22, f"{h.mean():.0f}", ha="center", fontsize=7.8, weight="semibold")
        f = 100 * G[m]["frac"].to_numpy()
        bp = b.boxplot([f], positions=[i], widths=0.42, patch_artist=True, showfliers=False,
                       medianprops=dict(color=INK, lw=1.5), whiskerprops=dict(color=INK2, lw=0.9),
                       capprops=dict(color=INK2, lw=0.9), boxprops=dict(lw=0.9, edgecolor=INK2))
        bp["boxes"][0].set_facecolor(mpl.colors.to_rgba(COL[m], 0.16))
        b.scatter(i + RNG.uniform(-0.25, 0.25, len(f)), f, s=9, color=COL[m], alpha=0.7,
                  marker=MARK[m], lw=0.3, edgecolor="white", zorder=3)
    for ax in (a, b):
        ax.set_xticks(X, ["Baseline", "Time\npenalty", "Hover\nreward", "Corrected\nreward"])
        ax.tick_params(axis="x", length=0)
        ax.set_xlim(-0.6, 3.6)
    a.set_ylim(0, 1000)
    a.set_ylabel("Slow/upright airborne frames per episode")
    a.set_title("(a) Mean count, 95% bootstrap CI", color=NAVY)
    b.set_ylim(0, 100)
    b.set_ylabel("Share of episode frames (%)")
    b.set_title("(b) Per-episode share of frames", color=NAVY)
    save(fig, "Fig07_slow_upright_behavior")


# ============================================================ Figure 8 =====
def fig8():
    fig, axes = plt.subplots(2, 2, figsize=(W, 4.6), sharex=True, sharey=True,
                             gridspec_kw={"hspace": 0.32, "wspace": 0.08})
    for ax, m in zip(axes.flat, ORDER):
        g = G[m]
        land = g["successful_landing"] == 1
        tl = g["duration_frames"] >= 1000
        term = ~land & ~tl
        ax.scatter(g.loc[land, "duration_frames"], g.loc[land, "score"], s=16, color=COL[m],
                   lw=0.4, edgecolor="white", label="landed", zorder=3)
        ax.scatter(g.loc[term, "duration_frames"], g.loc[term, "score"], s=16, facecolor="white",
                   edgecolor=COL[m], lw=0.9, label="terminated without landing", zorder=3)
        ax.scatter(g.loc[tl, "duration_frames"] + RNG.uniform(-8, 0, tl.sum()), g.loc[tl, "score"],
                   s=20, marker="x", color=COL[m], lw=0.9, label="time limit", zorder=3)
        rho = stats.spearmanr(g["duration_frames"], g["score"]).statistic
        ax.set_title(f"{LABEL[m]}   (Spearman ρ = {rho:.2f})", color=NAVY, fontsize=8.6)
        ax.axhline(0, color=RULE, lw=0.6)
        ax.axvline(1000, color=RULE, lw=0.7, ls=(0, (4, 3)))
        ax.grid(axis="x", color=GRID)
    for ax in axes[1]:
        ax.set_xlabel("Episode duration (frames)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Episode score (points)")
    from matplotlib.lines import Line2D
    hh = [Line2D([], [], marker="o", lw=0, color=INK2, markersize=4.5, label="landed (came to rest)"),
          Line2D([], [], marker="o", lw=0, markerfacecolor="white", markeredgecolor=INK2, markersize=4.5,
                 label="terminated without landing"),
          Line2D([], [], marker="x", lw=0, color=INK2, markersize=4.5, label="time limit (1,000 frames)")]
    fig.legend(handles=hh, loc="lower center", bbox_to_anchor=(0.5, 0.955), ncol=3, frameon=False, fontsize=7.4)
    axes[0, 0].set_xlim(100, 1040)
    axes[0, 0].set_ylim(-300, 320)
    save(fig, "Fig08_duration_vs_score")


# ============================================================ Figure 9 =====
def fig9():
    fig, ax = canvas(3.45, 10, -0.75, 4.95)
    OR, ORD, ORF = "#cf7020", "#8a4310", "#fbf1e8"
    tag(ax, 0.1, 4.93)
    box(ax, 0.1, 1.9, 2.0, 1.3, "Descent", "lander slows and\nlevels its attitude")
    box(ax, 2.6, 1.9, 2.1, 1.3, "Slow, upright,\nairborne state", None, fc=ORF, ec=OR, tc=ORD, title_size=8)
    arrow(ax, (2.1, 2.55), (2.6, 2.55))
    ax.text(5.25, 4.45, "INTENDED OBJECTIVE", fontsize=7.2, weight="semibold", color=NAVY)
    box(ax, 5.25, 3.0, 2.15, 1.25, "Touch down", "legs contact\nthe surface")
    box(ax, 7.8, 3.0, 2.15, 1.25, "Come to rest", "+100 once;\nepisode ends")
    arrow(ax, (4.7, 2.95), (5.25, 3.6), rad=-0.2)
    arrow(ax, (7.4, 3.62), (7.8, 3.62))
    ax.text(5.25, 1.55, "PATH FAVORED BY THE PER-FRAME BONUS", fontsize=7.2, weight="semibold", color=ORD)
    box(ax, 5.25, 0.1, 2.15, 1.25, "Hold the state", "+1 bonus on every\neligible frame", fc=ORF, ec=OR, tc=ORD)
    box(ax, 7.8, 0.1, 2.15, 1.25, "Avoid termination", "bonus stream lasts\nuntil the time limit",
        fc=ORF, ec=OR, tc=ORD)
    arrow(ax, (4.7, 2.15), (5.25, 0.75), rad=0.2, color=OR)
    arrow(ax, (7.4, 0.72), (7.8, 0.72), color=OR)
    a = FancyArrowPatch((8.85, 0.1), (6.35, 0.1), arrowstyle="-|>", mutation_scale=9, color=OR,
                        lw=1, connectionstyle="arc3,rad=-0.3", linestyle=(0, (3, 2)))
    ax.add_patch(a)
    ax.text(7.6, -0.62, "repeat each frame", fontsize=6.8, color=ORD, ha="center")
    ax.text(0.1, 1.2, "The proxy (slow and upright) is\nrewarded on every frame; the goal\n(landed) is rewarded "
            "once, and\nreaching it ends all further bonus.", fontsize=6.9, color=INK2, linespacing=1.45, va="top")
    save(fig, "Fig09_reward_hacking_mechanism")


# =========================================================== Figure 10 =====
def fig10():
    n = np.arange(0, 1001)
    gamma = 0.99
    fig, (a, b) = plt.subplots(1, 2, figsize=(W, 2.85), gridspec_kw={"wspace": 0.34})
    # (a) undiscounted cumulative bonus
    a.plot(n, n * 1.0, color="#cf7020", lw=2, label="hover: +1 per eligible frame")
    a.plot(n, -0.01 * n, color="#1a9a72", lw=2, label="corrected: −0.01 per frame\n(stability unchanged)")
    a.axhline(100, color=INK2, lw=0.9, ls=(0, (4, 3)))
    a.text(990, 112, "+100 one-off landing bonus", fontsize=7, color=INK2, ha="right")
    a.set_xlabel("Consecutive frames held in the eligible state")
    a.set_ylabel("Added training reward (points)")
    a.set_title("(a) Undiscounted accumulation", color=NAVY)
    a.set_ylim(-60, 1050)
    a.legend(loc="upper left", frameon=False, fontsize=6.9)
    # (b) discounted value (gamma = 0.99)
    disc_h = (1 - gamma ** n) / (1 - gamma)
    disc_c = -0.01 * (1 - gamma ** n) / (1 - gamma)
    b.plot(n, disc_h, color="#cf7020", lw=2)
    b.plot(n, disc_c, color="#1a9a72", lw=2)
    b.axhline(100, color=INK2, lw=0.9, ls=(0, (4, 3)))
    b.text(990, 92, "asymptote 1/(1−γ) = 100", fontsize=7, color=INK2, ha="right", va="top")
    b.set_xlabel("Consecutive frames held in the eligible state")
    b.set_ylabel("Discounted added reward (γ = 0.99)")
    b.set_title("(b) Discounted sum, γ = 0.99", color=NAVY)
    b.set_ylim(-10, 115)
    for ax in (a, b):
        ax.set_xlim(0, 1000)
    fig.text(0.07, 1.04, "ANALYTICAL ILLUSTRATION FROM THE REWARD DEFINITIONS  ·  BASE REWARD EXCLUDED  ·  "
             "NOT MEASURED DATA", fontsize=6.6, color=INK2, weight="medium")
    save(fig, "Fig10_bonus_accumulation_analytical")


# =========================================================== Figure 11 =====
def fig11():
    A, B = G["Original PPO"].set_index("seed"), G["Corrected Reward"].set_index("seed")
    fig, ax = plt.subplots(figsize=(W, 3.5))
    la, lb = A["successful_landing"] == 1, B["successful_landing"] == 1
    groups = [
        (la & lb, "both landed", dict(color=NAVY, s=18, lw=0.4, edgecolor="white")),
        (la & ~lb, "baseline only", dict(color="#2a6cc0", marker="o", s=20, facecolor="white", edgecolor="#2a6cc0", lw=1)),
        (~la & lb, "corrected only", dict(color="#1a9a72", marker="D", s=18, facecolor="white", edgecolor="#1a9a72", lw=1)),
        (~la & ~lb, "neither landed", dict(color=RULE, marker="x", s=18, lw=0.9)),
    ]
    for mask, name, kw in groups:
        ax.scatter(A.loc[mask, "score"], B.loc[mask, "score"], label=f"{name} (n = {int(mask.sum())})", zorder=3, **kw)
    ax.plot([-200, 300], [-200, 300], color=INK2, lw=0.8, ls=(0, (4, 3)))
    ax.text(62, 84, "y = x", fontsize=7, color=INK2, ha="center", rotation=45)
    ax.set_xlim(-80, 300)
    ax.set_ylim(-170, 300)
    ax.set_aspect("equal")
    ax.grid(axis="x", color=GRID)
    ax.set_xlabel("Baseline score on seed (points)")
    ax.set_ylabel("Corrected-reward score on seed (points)")
    ax.set_title("Same 100 seeds, paired by start condition", color=NAVY, fontsize=9)
    ax.legend(loc="center left", bbox_to_anchor=(1.04, 0.5), frameon=False, fontsize=7.4,
              handletextpad=0.3, title="Landing outcome on the same seed", title_fontsize=7.4,
              alignment="left")
    save(fig, "Fig11_paired_seed_baseline_vs_corrected")


if __name__ == "__main__":
    for f in (fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8, fig9, fig10, fig11):
        f()
