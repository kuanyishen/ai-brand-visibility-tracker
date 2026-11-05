"""Figures (Okabe-Ito colour-blind-safe palette, direct labels). Work for any number of surfaces."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BLUE, SKY, ORANGE, GREEN, RED, GREY, INK, PINK = "#0072B2", "#56B4E9", "#E69F00", "#009E73", "#D55E00", "#8a8f98", "#1f2328", "#CC79A7"
SURFACE_COLOR = {"gemini-flash": BLUE, "gemini-flash-search": SKY, "claude-sonnet": ORANGE, "claude-sonnet-search": GREEN}
FALLBACK = [PINK, GREY, "#F0E442", "#000000"]
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": "#c9ced6", "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e8ebef", "grid.linewidth": 0.8, "axes.axisbelow": True,
    "figure.dpi": 150, "savefig.dpi": 150, "savefig.bbox": "tight", "axes.titleweight": "bold",
    "axes.titlesize": 11, "axes.titlelocation": "left",
})


def color(s: str, i: int = 0) -> str:
    return SURFACE_COLOR.get(s) or FALLBACK[i % len(FALLBACK)]


def _save(fig, out: Path, name: str):
    fig.savefig(out / name, facecolor="white")
    plt.close(fig)


def overview(R, out):
    focus = R["meta"]["focus"]
    rows = []
    for s in R["surface_order"]:
        r = next(x for x in R["focus_by_intent"] if x["surface"] == s and x["intent"] == "discovery")
        rows.append((s, r))
    fig, ax = plt.subplots(figsize=(7.4, 0.9 + 0.62 * len(rows)))
    y = np.arange(len(rows))[::-1]
    for yi, (s, r) in zip(y, rows):
        ax.errorbar(r["rate"] * 100, yi, xerr=[[(r["rate"] - r["lo"]) * 100], [(r["hi"] - r["rate"]) * 100]], fmt="o", color=color(s), ms=9, capsize=4, lw=2.2)
        ax.text(r["hi"] * 100 + 1.5, yi, f"{r['rate'] * 100:.0f}%", va="center", fontsize=10, fontweight="bold", color=color(s))
    ax.set_yticks(y, [R["labels"][s] for s, _ in rows])
    ax.set_xlim(0, 100)
    ax.set_xlabel(f"% of answers that mention {focus} (questions never name a brand; 95% CI over prompts)")
    ax.set_title(f"{focus} visibility by AI surface")
    ax.grid(axis="y", visible=False)
    _save(fig, out, "fig_overview.png")


def visibility(R, out, top=10):
    focus = R["meta"]["focus"]
    ss = R["surface_order"]
    ncol = 2
    nrow = int(np.ceil(len(ss) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(10, 3.9 * nrow), sharex=True)
    axes = np.atleast_1d(axes).ravel()
    for ax, s in zip(axes, ss):
        d = pd.DataFrame(R["visibility"][s])
        d = pd.concat([d.head(top), d[d.brand == focus]]).drop_duplicates("brand").iloc[::-1]
        y = np.arange(len(d))
        ax.barh(y, d.mention_rate * 100, color=[RED if b == focus else color(s) for b in d.brand], height=0.68,
                xerr=[(d.mention_rate - d.mr_lo) * 100, (d.mr_hi - d.mention_rate) * 100], ecolor=INK, capsize=2.5)
        ax.set_yticks(y, d.brand, fontsize=9)
        ax.set_title(R["labels"][s])
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("% of answers mentioning the brand")
    for ax in axes[len(ss):]:
        ax.axis("off")
    fig.tight_layout()
    _save(fig, out, "fig_visibility.png")


def topics(R, out):
    d = pd.DataFrame(R["focus_by_topic"])
    order = d.groupby("topic").rate.mean().sort_values().index.tolist()
    ss = R["surface_order"]
    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    for k, s in enumerate(ss):
        g = d[d.surface == s].set_index("topic").reindex(order)
        off = (k - (len(ss) - 1) / 2) * 0.17
        ax.scatter(g.rate * 100, np.arange(len(order)) + off, s=g.n_answers * 8, color=color(s, k), alpha=0.92, label=R["labels"][s], edgecolor="white")
    ax.set_yticks(range(len(order)), order)
    ax.set_xlabel(f"% of answers mentioning {R['meta']['focus']} (dot size = number of answers)")
    ax.set_xlim(-5, 105)
    ax.set_title("Visibility depends heavily on the topic")
    ax.legend(frameon=False, loc="lower right", fontsize=8.5)
    ax.grid(axis="y", visible=False)
    _save(fig, out, "fig_topics.png")


def stability(R, out):
    d = pd.DataFrame(R["stability"])
    intents = ["discovery", "alternative", "reputation", "versus"]
    ss = R["surface_order"]
    fig, ax = plt.subplots(figsize=(8.4, 3.8))
    w = 0.8 / len(ss)
    for k, s in enumerate(ss):
        g = d[d.surface == s].set_index("intent").reindex(intents)
        xs = np.arange(len(intents)) + (k - (len(ss) - 1) / 2) * w
        ax.bar(xs, g.jaccard_all, w, color=color(s, k), label=R["labels"][s])
        for x, v in zip(xs, g.jaccard_all):
            ax.text(x, v + 0.012, f"{v:.2f}", ha="center", fontsize=7.5)
    ax.set_xticks(range(len(intents)), ["no brand named", f"alternatives to {R['meta']['focus']}", f"about {R['meta']['focus']}", f"{R['meta']['focus']} vs X"])
    ax.set_ylabel("brand-list overlap between repeats (Jaccard)")
    ax.set_ylim(0, 1.12)
    ax.set_title("Ask twice, get different brand lists")
    ax.legend(frameon=False, loc="upper left", fontsize=8, ncol=2)
    ax.grid(axis="x", visible=False)
    _save(fig, out, "fig_stability.png")


def design(R, out):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for k, s in enumerate(R["surface_order"]):
        c = R["design"][s]
        axes[0].plot(c["runs_grid"], c["se_by_runs"], color=color(s, k), lw=2.2, label=R["labels"][s])
        axes[1].plot(c["prompts_grid"], c["se_by_prompts"], color=color(s, k), lw=2.2, label=R["labels"][s])
    axes[0].set_xscale("log")
    axes[0].set_xlabel(f"answers per prompt ({R['meta']['intent_counts']['discovery']} prompts)")
    axes[0].set_ylabel("standard error of the mention rate")
    axes[0].set_title("More repeats hit a floor quickly")
    axes[1].set_xlabel("number of different prompts (3 answers each)")
    axes[1].set_title("More prompts keep paying off")
    for ax in axes:
        ax.axhline(0.05, color=GREY, ls="--", lw=1)
        ax.set_ylim(0, None)
    axes[1].text(axes[1].get_xlim()[1], 0.052, "target SE = 5 pp", ha="right", fontsize=8.5, color=GREY)
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    _save(fig, out, "fig_design.png")


def agreement(R, out):
    ss = R["surface_order"]
    M = np.array([[R["agreement"]["jaccard"][a][b] for b in ss] for a in ss])
    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    im = ax.imshow(M, cmap="Blues", vmin=0, vmax=max(0.8, M.max()))
    ax.set_xticks(range(len(ss)), [R["labels"][s] for s in ss], fontsize=8.5, rotation=25, ha="right")
    ax.set_yticks(range(len(ss)), [R["labels"][s] for s in ss], fontsize=8.5)
    for i in range(len(ss)):
        for j in range(len(ss)):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", color="white" if M[i, j] > 0.45 else INK, fontsize=11, fontweight="bold")
    ax.set_title("Do the AIs name the same brands?\n(mean Jaccard overlap per prompt; diagonal = same surface asked again)", fontsize=10)
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    _save(fig, out, "fig_agreement.png")


def sources(R, out):
    pal = {"social / community": PINK, "editorial / blogs": BLUE, "retailer or brand site": GREEN, "reviews": ORANGE, "other": GREY}
    ss = list(R["sources"])
    fig, axes = plt.subplots(1, len(ss), figsize=(5.6 * len(ss), 4.6), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, s in zip(axes, ss):
        d = pd.DataFrame(R["sources"][s]["top"]).head(10).iloc[::-1]
        ax.barh(d.domain, d.share * 100, color=[pal[c] for c in d.category], height=0.7)
        for y, (v, a) in enumerate(zip(d.share * 100, d.answers)):
            ax.text(v + 0.1, y, f"{a}", va="center", fontsize=8, color=GREY)
        ax.set_title(R["labels"][s])
        ax.set_xlabel("% of citations (number = answers citing it)")
        ax.grid(axis="y", visible=False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in list(pal.values())[:4]]
    axes[-1].legend(handles, list(pal.keys())[:4], frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    _save(fig, out, "fig_sources.png")


def make_all(R, out: Path):
    for f in (overview, visibility, topics, stability, design, agreement, sources):
        f(R, out)
