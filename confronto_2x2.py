"""
Confronto 2x2 del paper: algoritmo (SARSA / Q-learning) x ambiente
(fully / partially observable), versioni modificate, 200 episodi.

Genera in confronto_2x2/:
  training_2x2.png    curve di training per episodio (media mobile)
  valutazione_2x2.png valutazione greedy ogni 10 episodi (20 partite)
  barre_2x2.png       medie ultimi 50 episodi di training + valutazione
  riepilogo_2x2.csv   tabella numerica

Colore = algoritmo (blu SARSA, arancio Q-learning);
tratto = ambiente (continuo fully, tratteggiato partial).
Un solo seed per configurazione (vedi E9, E11).
"""
import csv
import os

import matplotlib.pyplot as plt
import numpy as np

RUNS = {
    "SARSA fully": ("sarsa_4/risultati_finale_200ep", "sarsa"),
    "Q-learning fully": ("q_learning_2/risultati_finale_200ep", "qlearning"),
    "SARSA partial": ("sarsa_4/risultati_po_200ep", "sarsa"),
    "Q-learning partial": ("q_learning_2/risultati_po_200ep", "qlearning"),
}
STYLE = {
    "SARSA fully": dict(color="#2a78d6", ls="-"),
    "Q-learning fully": dict(color="#eb6834", ls="-"),
    "SARSA partial": dict(color="#2a78d6", ls="--"),
    "Q-learning partial": dict(color="#eb6834", ls="--"),
}
TRAIN_METRICS = [
    ("frutti_mangiati", "Frutti mangiati per episodio"),
    ("self_collision", "Auto-collisioni per episodio"),
    ("segmenti_persi", "Segmenti persi per episodio"),
    ("lunghezza_media", "Lunghezza media del serpente"),
]
EVAL_METRICS = [
    ("frutti", "Valutazione greedy: frutti medi (20 partite)"),
    ("partite_bloccate", "Valutazione greedy: partite bloccate (su 20)"),
]
SMOOTH = 10
LAST = 50
EVAL_FROM = 100
OUT = "confronto_2x2"


def load(path):
    with open(path) as f:
        rows = [r for r in csv.DictReader(f) if r["episodio"] not in ("", "episodio")]
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}


def style_axes(ax):
    ax.grid(alpha=0.25)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def main():
    os.makedirs(OUT, exist_ok=True)
    train, ev = {}, {}
    for name, (folder, alg) in RUNS.items():
        train[name] = load(os.path.join(folder, f"training_history_{alg}.csv"))
        ev[name] = load(os.path.join(folder, f"eval_history_{alg}.csv"))
    names = list(RUNS)

    # --- training ---
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, (key, title) in zip(axes.flat, TRAIN_METRICS):
        for n in names:
            d, st = train[n], STYLE[n]
            ax.plot(d["episodio"], d[key], color=st["color"], lw=0.8, alpha=0.15)
            ma = np.convolve(d[key], np.ones(SMOOTH) / SMOOTH, mode="valid")
            ax.plot(d["episodio"][SMOOTH - 1:], ma, color=st["color"], ls=st["ls"], lw=2, label=n)
        ax.set_title(title)
        ax.set_xlabel("Episodio")
        ax.set_ylim(bottom=0)
        style_axes(ax)
    axes[0, 0].legend(frameon=False, loc="lower right")
    fig.suptitle(f"Training (con esplorazione) - media mobile {SMOOTH} episodi")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "training_2x2.png"), dpi=150)
    plt.close(fig)

    # --- valutazione greedy ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    for ax, (key, title) in zip(axes, EVAL_METRICS):
        for n in names:
            d, st = ev[n], STYLE[n]
            ax.plot(d["episodio"], d[key], color=st["color"], ls=st["ls"], lw=2,
                    marker="o", ms=4, label=n)
        ax.set_title(title)
        ax.set_xlabel("Episodio di training")
        ax.set_ylim(bottom=0)
        style_axes(ax)
    axes[1].set_ylim(0, 21)
    axes[0].legend(frameon=False, loc="upper left")
    fig.suptitle("Valutazione greedy (epsilon = 0, partite nuove con seed fisso)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "valutazione_2x2.png"), dpi=150)
    plt.close(fig)

    # --- barre riassuntive ---
    bars = [(f"{t} (ultimi {LAST} ep.)", lambda n, k=k: train[n][k][-LAST:])
            for k, t in TRAIN_METRICS[:3]]
    bars.append((f"Valutazione greedy: frutti (checkpoint >= ep. {EVAL_FROM})",
                 lambda n: ev[n]["frutti"][ev[n]["episodio"] >= EVAL_FROM]))
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.8))
    for ax, (title, get) in zip(axes, bars):
        means = [get(n).mean() for n in names]
        stds = [get(n).std() for n in names]
        b = ax.bar(range(4), means, yerr=stds, capsize=4,
                   color=[STYLE[n]["color"] for n in names], edgecolor="white", linewidth=2)
        for rect, n in zip(b, names):
            if "partial" in n:
                rect.set_hatch("//")
                rect.set_alpha(0.6)
        for i, m in enumerate(means):
            ax.text(i, m / 2, f"{m:.1f}", ha="center", va="center", color="white", fontweight="bold")
        ax.set_xticks(range(4))
        ax.set_xticklabels([n.replace(" ", "\n") for n in names], fontsize=9)
        ax.set_title(title, fontsize=10)
        style_axes(ax)
        ax.grid(axis="x", visible=False)
    fig.suptitle("Confronto 2x2: media ± deviazione standard")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "barre_2x2.png"), dpi=150)
    plt.close(fig)

    # --- tabella ---
    path = os.path.join(OUT, "riepilogo_2x2.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["configurazione"] + [f"{k}_ultimi{LAST}" for k, _ in TRAIN_METRICS]
                   + [f"eval_frutti_ckpt>={EVAL_FROM}", f"eval_bloccate_ckpt>={EVAL_FROM}",
                      "eval_frutti_miglior_ckpt"])
        for n in names:
            e = ev[n]
            sel = e["episodio"] >= EVAL_FROM
            w.writerow([n] + [round(train[n][k][-LAST:].mean(), 2) for k, _ in TRAIN_METRICS]
                       + [round(e["frutti"][sel].mean(), 2), round(e["partite_bloccate"][sel].mean(), 2),
                          round(e["frutti"].max(), 2)])
    print(open(path).read())


if __name__ == "__main__":
    main()
