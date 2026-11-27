"""
Figure del paper, VERSIONE SENZA SHAPING (PDF vettoriali, font Type-1, senza titoli interni:
le didascalie stanno nel .tex). Eseguire dalla cartella paper_no_shaping/:
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

# VERSIONE SENZA SHAPING (E14).
# configurazione finale: learning rate a coseno 1e-3 -> 1e-4 (E13), nessuno shaping
RUNS_2X2 = {
    "SARSA, FO": ("sarsa_4/risultati_no_shaping_lr_decay_200ep", "sarsa", BLUE, "-"),
    "Q-learning, FO": ("q_learning_2/risultati_no_shaping_lr_decay_200ep", "qlearning", ORANGE, "-"),
    "SARSA, PO": ("sarsa_4/risultati_no_shaping_lr_decay_po_200ep", "sarsa", BLUE, "--"),
    "Q-learning, PO": ("q_learning_2/risultati_no_shaping_lr_decay_po_200ep", "qlearning", ORANGE, "--"),
}
# stessi run senza shaping con learning rate fisso 1e-3 (E14), per la figura fisso vs coseno
RUNS_FIXED_LR = {
    "SARSA, FO": "sarsa_4/risultati_no_shaping_fixed_lr_200ep",
    "Q-learning, FO": "q_learning_2/risultati_no_shaping_fixed_lr_200ep",
    "SARSA, PO": "sarsa_4/risultati_no_shaping_fixed_lr_po_200ep",
    "Q-learning, PO": "q_learning_2/risultati_no_shaping_fixed_lr_po_200ep",
}
GREY = "#9a9a9a"
# baseline euristica (E15): baseline.py, stesso protocollo della valutazione greedy
BASELINE_CSV = "baseline/baseline_risultati.csv"
BASE_COLOR = "#333333"
BASE_STYLE = {"fully": ("-", "Baseline, FO"), "partial": ("--", "Baseline, PO")}
# metrica del training / della valutazione -> colonna del CSV della baseline
BASE_KEY = {"frutti_mangiati": "frutti", "lunghezza_media": "lunghezza_media",
            "self_collision": "self_collision", "segmenti_persi": "segmenti_persi",
            "frutti": "frutti", "partite_bloccate": "partite_bloccate"}


def load_baseline():
    with open(os.path.join(ROOT, BASELINE_CSV)) as f:
        return {r["osservabilita"]: {k: float(v) for k, v in r.items() if k != "osservabilita"}
                for r in csv.DictReader(f)}
RUNS_ORIG = {
    "SARSA\noriginal": ("sarsa_4/training_history_sarsa.csv", BLUE, True),
    "Q-learning\noriginal": ("q_learning_2/training_history_qlearning.csv", ORANGE, True),
    "SARSA\nmodified": ("sarsa_4/risultati_no_shaping_lr_decay_200ep/training_history_sarsa.csv", BLUE, False),
    "Q-learning\nmodified": ("q_learning_2/risultati_no_shaping_lr_decay_200ep/training_history_qlearning.csv", ORANGE, False),
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


def _training_panels(metrics, fname):
    data = {n: load(os.path.join(p, f"training_history_{a}.csv")) for n, (p, a, _, _) in RUNS_2X2.items()}
    base = load_baseline()
    fig, axes = plt.subplots(2, 1, figsize=(3.3, 3.7), sharex=True)
    for ax, (k, yl) in zip(axes, metrics):
        for obs, (ls, lab) in BASE_STYLE.items():   # baseline: linea orizzontale
            ax.axhline(base[obs][BASE_KEY[k]], color=BASE_COLOR, ls=ls, lw=1.0, label=lab)
        for n, (_, _, c, ls) in RUNS_2X2.items():
            d = data[n]
            ax.plot(d["episodio"], d[k], color=c, lw=0.5, alpha=0.15)
            ma = np.convolve(d[k], np.ones(SMOOTH) / SMOOTH, mode="valid")
            ax.plot(d["episodio"][SMOOTH - 1:], ma, color=c, ls=ls, lw=1.4, label=n)
        ax.set_ylabel(yl)
        ax.set_ylim(bottom=0)
        clean(ax)
    axes[-1].set_xlabel("Training episode")
    fig.legend(*axes[0].get_legend_handles_labels(), frameon=False, ncol=3, loc="upper center",
               handlelength=2.0, columnspacing=0.8, fontsize=6.3)
    fig.tight_layout(pad=0.3, rect=(0, 0, 1, 0.905))
    fig.savefig(os.path.join(OUT, fname))
    plt.close(fig)


def fig_training_2x2():
    # crescita: frutti e lunghezza
    _training_panels([("frutti_mangiati", "Fruits / episode"),
                      ("lunghezza_media", "Mean snake length")], "training_growth.pdf")
    # sicurezza: auto-collisioni e segmenti persi
    _training_panels([("self_collision", "Self-collisions / episode"),
                      ("segmenti_persi", "Segments lost / episode")], "training_safety.pdf")


def fig_eval_2x2():
    """Un punto per checkpoint (dall'episodio EVAL_FROM), linea nera = media.
    Mostra sia il livello medio sia la variabilita' fra checkpoint."""
    names = list(RUNS_2X2)
    ev = {n: load(os.path.join(p, f"eval_history_{a}.csv")) for n, (p, a, _, _) in RUNS_2X2.items()}
    labels = [n.replace(", ", "\n") for n in names]
    rng = np.random.default_rng(0)
    base = load_baseline()
    fig, axes = plt.subplots(2, 1, figsize=(3.3, 3.4), sharex=True)
    for ax, (k, yl) in zip(axes, [("frutti", "Greedy: fruits / game"),
                                   ("partite_bloccate", "Greedy: stuck games (of 20)")]):
        # baseline: linea orizzontale sopra le colonne FO (0-1) e PO (2-3).
        # Le partite bloccate della baseline sono 0 in FO e PO: lo dice la didascalia.
        if k == "frutti":
            for obs, x0 in (("fully", -0.45), ("partial", 1.55)):
                b = base[obs][BASE_KEY[k]]
                ax.plot([x0, x0 + 1.9], [b, b], color=BASE_COLOR, ls=BASE_STYLE[obs][0], lw=1.0, zorder=2)
                ax.text(x0, b + 1.0, f"baseline {b:.1f}", ha="left", va="bottom",
                        fontsize=6.2, color=BASE_COLOR)
        for i, n in enumerate(names):
            _, _, c, ls = RUNS_2X2[n]
            e = ev[n]
            v = e[k][e["episodio"] >= EVAL_FROM]
            x = i + rng.uniform(-0.15, 0.15, len(v))
            filled = ls == "-"
            ax.scatter(x, v, s=12, facecolors=c if filled else "white", edgecolors=c, lw=0.8, zorder=3)
            m = v.mean()
            ax.plot([i - 0.3, i + 0.3], [m, m], color="black", lw=1.4, zorder=4)
            ax.text(i + 0.33, m, f"{m:.1f}", va="center", fontsize=6.5)
        ax.set_ylabel(yl)
        ax.set_ylim(bottom=0)
        clean(ax)
        ax.grid(axis="x", visible=False)
    axes[1].set_ylim(0, 20.5)
    axes[1].set_xticks(range(4))
    axes[1].set_xticklabels([n.replace(", ", "\n") for n in names], fontsize=6.8)
    axes[1].set_xlim(-0.5, 3.75)
    fig.tight_layout(pad=0.3)
    fig.savefig(os.path.join(OUT, "eval_2x2.pdf"))
    plt.close(fig)


def fig_lr_fixed_vs_cosine():
    """Frutti in valutazione greedy per checkpoint: learning rate fisso (grigio)
    contro coseno (colore), un pannello per cella del 2x2. Mostra le
    oscillazioni fra checkpoint che il coseno elimina (E9, E13)."""
    fig, axes = plt.subplots(2, 2, figsize=(3.3, 2.9), sharex=True, sharey=True)
    for ax, (n, (p, a, c, _)) in zip(axes.flat, RUNS_2X2.items()):
        fixed = load(os.path.join(RUNS_FIXED_LR[n], f"eval_history_{a}.csv"))
        cos = load(os.path.join(p, f"eval_history_{a}.csv"))
        ax.plot(fixed["episodio"], fixed["frutti"], color=GREY, lw=1.0, marker="o", ms=2,
                label="fixed LR")
        ax.plot(cos["episodio"], cos["frutti"], color=c, lw=1.2, marker="o", ms=2,
                label="cosine LR")
        ax.text(0.03, 0.95, n, transform=ax.transAxes, va="top", fontsize=6.8)
        ax.set_ylim(0, 50)
        clean(ax)
    for ax in axes[:, 0]:
        ax.set_ylabel("Greedy fruits / game")
    for ax in axes[1]:
        ax.set_xlabel("Training episode")
    handles = [plt.Line2D([], [], color=GREY, lw=1.0, marker="o", ms=2),
               plt.Line2D([], [], color="black", lw=1.2, marker="o", ms=2)]
    fig.legend(handles, ["fixed LR 1e-3", "cosine LR 1e-3 to 1e-4 (colour)"],
               frameon=False, ncol=2, loc="upper center", fontsize=6.3)
    fig.tight_layout(pad=0.3, rect=(0, 0, 1, 0.93))
    fig.savefig(os.path.join(OUT, "lr_fixed_vs_cosine.pdf"))
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    fig_orig_vs_mod()
    fig_training_2x2()
    fig_eval_2x2()
    fig_lr_fixed_vs_cosine()
    print("figure salvate in", OUT)
