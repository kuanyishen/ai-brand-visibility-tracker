"""Figures (Okabe-Ito colour-blind-safe palette, direct labels)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BLUE, ORANGE, GREEN, RED, GREY, INK, PINK = "#0072B2", "#E69F00", "#009E73", "#D55E00", "#8a8f98", "#1f2328", "#CC79A7"
SURFACE_COLOR = {"gemini-flash": BLUE, "gemini-flash-search": ORANGE}
SURFACE_LABEL = {"gemini-flash": "Gemini (no search)", "gemini-flash-search": "Gemini + Google Search"}
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": "#c9ced6", "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e8ebef", "grid.linewidth": 0.8, "axes.axisbelow": True,
    "figure.dpi": 150, "savefig.dpi": 150, "savefig.bbox": "tight", "axes.titleweight": "bold",
    "axes.titlesize": 11, "axes.titlelocation": "left",
})


def _save(fig, out: Path, name: str):
    fig.savefig(out / name, facecolor="white")
    plt.close(fig)


def visibility(R, out, focus="Aritzia", top=12):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), sharex=True)
    for ax, (surf, rows) in zip(axes, R["visibility"].items()):
        d = pd.DataFrame(rows)
        d = pd.concat([d.head(top), d[d.brand == focus]]).drop_duplicates("brand").iloc[::-1]
        y = np.arange(len(d))
        cols = [RED if b == focus else SURFACE_COLOR[surf] for b in d.brand]
        ax.barh(y, d.mention_rate * 100, color=cols, height=0.68,
                xerr=[(d.mention_rate - d.mr_lo) * 100, (d.mr_hi - d.mention_rate) * 100], ecolor=INK, capsize=2.5)
        ax.set_yticks(y, d.brand, fontsize=9)
        ax.set_title(SURFACE_LABEL[surf])
        ax.set_xlabel("% of answers that mention the brand")
        ax.grid(axis="y", visible=False)
    fig.suptitle("Unprompted visibility: 24 questions that never name a brand (95% CI over prompts)", x=0.01, ha="left", fontsize=10.5, y=1.02)
    fig.tight_layout()
    _save(fig, out, "fig_visibility.png")


def topics(R, out):
    d = pd.DataFrame(R["focus_by_topic"])
    order = d.groupby("topic").rate.mean().sort_values().index.tolist()
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for surf, g in d.groupby("surface"):
        g = g.set_index("topic").reindex(order)
        off = -0.12 if surf == "gemini-flash" else 0.12
        ax.scatter(g.rate * 100, np.arange(len(order)) + off, s=g.n_answers * 9, color=SURFACE_COLOR[surf], alpha=0.9, label=SURFACE_LABEL[surf], edgecolor="white")
    ax.set_yticks(range(len(order)), order)
    ax.set_xlabel("% of answers mentioning Aritzia (dot size = number of answers)")
    ax.set_xlim(-5, 105)
    ax.set_title("Visibility depends heavily on the topic")
    ax.legend(frameon=False, loc="lower right", fontsize=8.5)
    ax.grid(axis="y", visible=False)
    _save(fig, out, "fig_topics.png")


def stability(R, out):
    d = pd.DataFrame(R["stability"])
    intents = ["discovery", "alternative", "reputation", "versus"]
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    w = 0.38
    for i, surf in enumerate(["gemini-flash", "gemini-flash-search"]):
        g = d[d.surface == surf].set_index("intent").reindex(intents)
        ax.bar(np.arange(len(intents)) + (i - 0.5) * w, g.jaccard_all, w, color=SURFACE_COLOR[surf], label=SURFACE_LABEL[surf])
        for x, v in zip(np.arange(len(intents)) + (i - 0.5) * w, g.jaccard_all):
            ax.text(x, v + 0.015, f"{v:.2f}", ha="center", fontsize=8)
    ax.set_xticks(range(len(intents)), ["no brand named", "alternatives to Aritzia", "about Aritzia", "Aritzia vs X"])
    ax.set_ylabel("brand-list overlap between repeats (Jaccard)")
    ax.set_ylim(0, 1.08)
    ax.set_title("Ask twice, get different brand lists")
    ax.legend(frameon=False, loc="upper left", fontsize=8.5, ncol=2)
    ax.grid(axis="x", visible=False)
    _save(fig, out, "fig_stability.png")


def design(R, out):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    d = R["design"]
    for surf in ("gemini-flash", "gemini-flash-search"):
        c = d[surf]
        axes[0].plot(c["runs_grid"], c["se_by_runs"], color=SURFACE_COLOR[surf], lw=2.2, label=SURFACE_LABEL[surf])
        axes[1].plot(c["prompts_grid"], c["se_by_prompts"], color=SURFACE_COLOR[surf], lw=2.2, label=SURFACE_LABEL[surf])
    axes[0].set_xscale("log")
    axes[0].set_xlabel("answers per prompt (24 prompts)")
    axes[0].set_ylabel("standard error of the mention rate")
    axes[0].set_title("More repeats hit a floor quickly")
    axes[1].set_xlabel("number of different prompts (3 answers each)")
    axes[1].set_title("More prompts keep paying off")
    for ax in axes:
        ax.axhline(0.05, color=GREY, ls="--", lw=1)
        ax.set_ylim(0, None)
    axes[1].text(axes[1].get_xlim()[1], 0.052, "target SE = 5 pp", ha="right", fontsize=8.5, color=GREY)
    axes[0].legend(frameon=False, fontsize=8.5)
    fig.tight_layout()
    _save(fig, out, "fig_design.png")


def sources(R, out):
    d = pd.DataFrame(R["sources"]["top"]).iloc[::-1]
    pal = {"social / community": PINK, "editorial / blogs": BLUE, "retailer or brand site": GREEN, "reviews": ORANGE, "other": GREY}
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.barh(d.domain, d.share * 100, color=[pal[c] for c in d.category], height=0.7)
    for y, (s, a) in enumerate(zip(d.share * 100, d.answers)):
        ax.text(s + 0.1, y, f"{a} answers", va="center", fontsize=8, color=GREY)
    ax.set_xlabel("% of all citations")
    ax.set_title("What the search-grounded answers cite")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in pal.values()]
    ax.legend(handles, pal.keys(), frameon=False, fontsize=8, loc="lower right")
    ax.grid(axis="y", visible=False)
    _save(fig, out, "fig_sources.png")


def make_all(R, out: Path):
    for f in (visibility, topics, stability, design, sources):
        f(R, out)
