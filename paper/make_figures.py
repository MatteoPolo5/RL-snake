"""
Figure del paper (PDF vettoriali, font Type-1, senza titoli interni:
le didascalie stanno nel .tex). Eseguire dalla cartella paper/:
    python make_figures.py
Legge i CSV dei run dalla cartella del repository (..).
"""
import csv
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams.update({
    "pdf.use14corefonts": True,      # solo font Type-1 standard (linee guida)
    "font.family": "serif",
    "font.serif": ["Times", "Nimbus Roman", "DejaVu Serif"],
    "axes.unicode_minus": False,
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
})

ROOT = ".."
OUT = "figures"
BLUE, ORANGE = "#2a78d6", "#eb6834"
LAST, SMOOTH, EVAL_FROM = 50, 10, 100

RUNS_2X2 = {
    "SARSA, FO": ("sarsa_4/risultati_finale_200ep", "sarsa", BLUE, "-"),
    "Q-learning, FO": ("q_learning_2/risultati_finale_200ep", "qlearning", ORANGE, "-"),
    "SARSA, PO": ("sarsa_4/risultati_po_200ep", "sarsa", BLUE, "--"),
    "Q-learning, PO": ("q_learning_2/risultati_po_200ep", "qlearning", ORANGE, "--"),
}
RUNS_ORIG = {
    "SARSA\noriginal": ("sarsa_4/training_history_sarsa.csv", BLUE, True),
    "Q-learning\noriginal": ("q_learning_2/training_history_qlearning.csv", ORANGE, True),
    "SARSA\nmodified": ("sarsa_4/risultati_finale_200ep/training_history_sarsa.csv", BLUE, False),
    "Q-learning\nmodified": ("q_learning_2/risultati_finale_200ep/training_history_qlearning.csv", ORANGE, False),
}


def load(path):
    with open(os.path.join(ROOT, path)) as f:
        rows = [r for r in csv.DictReader(f) if r["episodio"] not in ("", "episodio")]
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}


def clean(ax):
    ax.grid(alpha=0.25, lw=0.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def bar_panel(ax, labels, means, stds, colors, hatched, ylabel):
    bars = ax.bar(range(len(labels)), means, yerr=stds, capsize=2, color=colors,
                  edgecolor="white", linewidth=1, error_kw=dict(lw=0.7))
    for b, h in zip(bars, hatched):
        if h:
            b.set_hatch("////")
            b.set_alpha(0.6)
    for i, m in enumerate(means):
        ax.text(i, m / 2, f"{m:.1f}", ha="center", va="center", color="white", fontsize=7,
                fontweight="bold")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=6.5)
    ax.set_ylabel(ylabel)
    clean(ax)
    ax.grid(axis="x", visible=False)


def fig_orig_vs_mod():
    data = {n: load(p) for n, (p, _, _) in RUNS_ORIG.items()}
    names = list(RUNS_ORIG)
    metrics = [("frutti_mangiati", "Fruits / episode"),
               ("self_collision", "Self-collisions / episode"),
               ("segmenti_persi", "Segments lost / episode")]
    fig, axes = plt.subplots(1, 3, figsize=(7, 1.9))
    for ax, (k, yl) in zip(axes, metrics):
        bar_panel(ax, names,
                  [data[n][k][-LAST:].mean() for n in names],
                  [data[n][k][-LAST:].std() for n in names],
                  [RUNS_ORIG[n][1] for n in names],
                  [RUNS_ORIG[n][2] for n in names], yl)
    fig.tight_layout(pad=0.3)
    fig.savefig(os.path.join(OUT, "orig_vs_mod.pdf"))
    plt.close(fig)


def fig_training_2x2():
    data = {n: load(os.path.join(p, f"training_history_{a}.csv")) for n, (p, a, _, _) in RUNS_2X2.items()}
    metrics = [("frutti_mangiati", "Fruits / episode"),
               ("self_collision", "Self-collisions / episode"),
               ("segmenti_persi", "Segments lost / episode"),
               ("lunghezza_media", "Mean snake length")]
    fig, axes = plt.subplots(1, 4, figsize=(7, 1.75))
    for ax, (k, yl) in zip(axes, metrics):
        for n, (_, _, c, ls) in RUNS_2X2.items():
            d = data[n]
            ma = np.convolve(d[k], np.ones(SMOOTH) / SMOOTH, mode="valid")
            ax.plot(d["episodio"][SMOOTH - 1:], ma, color=c, ls=ls, lw=1.2, label=n)
        ax.set_xlabel("Episode")
        ax.set_ylabel(yl)
        ax.set_ylim(bottom=0)
        clean(ax)
    axes[0].legend(frameon=False, loc="lower right", handlelength=2.2)
    fig.tight_layout(pad=0.3)
    fig.savefig(os.path.join(OUT, "training_2x2.pdf"))
    plt.close(fig)


def fig_eval_2x2():
    names = list(RUNS_2X2)
    ev = {n: load(os.path.join(p, f"eval_history_{a}.csv")) for n, (p, a, _, _) in RUNS_2X2.items()}
    fig, axes = plt.subplots(2, 1, figsize=(3.3, 3.3), sharex=True)
    for n, (_, _, c, ls) in RUNS_2X2.items():
        e = ev[n]
        axes[0].plot(e["episodio"], e["frutti"], color=c, ls=ls, lw=1.1, marker="o", ms=2.2, label=n)
        axes[1].plot(e["episodio"], e["partite_bloccate"], color=c, ls=ls, lw=1.1, marker="o", ms=2.2)
    axes[0].set_ylabel("Greedy: fruits / game")
    axes[1].set_ylabel("Greedy: stuck games (of 20)")
    axes[1].set_xlabel("Training episode")
    axes[0].set_ylim(bottom=0)
    axes[1].set_ylim(0, 20.5)
    fig.legend(*axes[0].get_legend_handles_labels(), frameon=False, ncol=4, loc="upper center",
               handlelength=2.0, columnspacing=0.8, fontsize=6.3)
    for ax in axes:
        clean(ax)
    fig.tight_layout(pad=0.3, rect=(0, 0, 1, 0.94))
    fig.savefig(os.path.join(OUT, "eval_2x2.pdf"))
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    fig_orig_vs_mod()
    fig_training_2x2()
    fig_eval_2x2()
    print("figure salvate in", OUT)
