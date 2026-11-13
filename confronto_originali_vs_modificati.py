"""
Confronto SARSA / Q-learning ORIGINALI vs MODIFICATI (ambiente fully observable).

Legge i CSV di training gia' prodotti e genera:
  confronto_originali_vs_modificati/confronto_training.png  (curve per episodio)
  confronto_originali_vs_modificati/confronto_barre.png     (medie ultimi 50 episodi)
  confronto_originali_vs_modificati/riepilogo.csv           (tabella numerica)

Avvertenze per il paper (vedi ERRORI_RISCONTRATI.md):
  - E2: negli originali le auto-collisioni con 3 segmenti sono contate come urti
    al muro -> le auto-collisioni originali sono SOTTOSTIMATE.
  - E10: epsilon diverso negli originali (SARSA 0.2->0.02, QL fisso 0.15).
  - E5: i modificati resettano le board a ogni episodio, gli originali no
    (serpenti piu' lunghi -> piu' frutti per episodio negli originali).
  - I CSV originali arrivano all'episodio 180 (salvataggio ogni 20 episodi,
    nessun salvataggio finale).
"""
import csv
import os

import matplotlib.pyplot as plt
import numpy as np

RUNS = {
    "SARSA originale": "sarsa_4/training_history_sarsa.csv",
    "Q-learning originale": "q_learning_2/training_history_qlearning.csv",
    "SARSA modificato": "sarsa_4/risultati_finale_200ep/training_history_sarsa.csv",
    "Q-learning modificato": "q_learning_2/risultati_finale_200ep/training_history_qlearning.csv",
}
# colore = algoritmo, tratteggio = versione
STYLE = {
    "SARSA originale": dict(color="#2a78d6", ls="--"),
    "Q-learning originale": dict(color="#eb6834", ls="--"),
    "SARSA modificato": dict(color="#2a78d6", ls="-"),
    "Q-learning modificato": dict(color="#eb6834", ls="-"),
}
METRICS = [
    ("frutti_mangiati", "Frutti mangiati per episodio"),
    ("self_collision", "Auto-collisioni per episodio"),
    ("segmenti_persi", "Segmenti persi per episodio"),
]
SMOOTH = 10
LAST = 50
OUT = "confronto_originali_vs_modificati"


def load(path):
    with open(path) as f:
        rows = [r for r in csv.DictReader(f) if r["episodio"] not in ("", "episodio")]
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}


def moving_average(x, w):
    out = np.convolve(x, np.ones(w) / w, mode="valid")
    return np.arange(w - 1, len(x)), out


def main():
    os.makedirs(OUT, exist_ok=True)
    data = {name: load(path) for name, path in RUNS.items()}

    # --- curve per episodio ---
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    for ax, (key, title) in zip(axes, METRICS):
        for name, d in data.items():
            st = STYLE[name]
            ax.plot(d["episodio"], d[key], color=st["color"], lw=0.8, alpha=0.18)
            idx, ma = moving_average(d[key], SMOOTH)
            ax.plot(d["episodio"][idx], ma, color=st["color"], ls=st["ls"], lw=2, label=name)
        ax.set_title(title)
        ax.set_xlabel("Episodio")
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.25)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].legend(frameon=False)
    fig.suptitle(f"Originali (tratteggio) vs modificati (continuo) - media mobile {SMOOTH} episodi")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "confronto_training.png"), dpi=150)
    plt.close(fig)

    # --- barre: media ultimi LAST episodi ---
    names = list(RUNS)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, (key, title) in zip(axes, METRICS):
        means = [data[n][key][-LAST:].mean() for n in names]
        stds = [data[n][key][-LAST:].std() for n in names]
        bars = ax.bar(range(4), means, yerr=stds, capsize=4,
                      color=[STYLE[n]["color"] for n in names],
                      edgecolor="white", linewidth=2)
        for b, n in zip(bars, names):
            if "originale" in n:
                b.set_hatch("//")
                b.set_alpha(0.6)
        for i, m in enumerate(means):
            ax.text(i, m / 2, f"{m:.1f}", ha="center", va="center", color="white",
                    fontweight="bold")
        ax.set_xticks(range(4))
        ax.set_xticklabels([n.replace(" ", "\n") for n in names], fontsize=9)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.suptitle(f"Media (± dev. std) sugli ultimi {LAST} episodi di training")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "confronto_barre.png"), dpi=150)
    plt.close(fig)

    # --- tabella ---
    with open(os.path.join(OUT, "riepilogo.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run", "episodi"] + [f"{k}_ultimi{LAST}" for k, _ in METRICS]
                   + ["segmenti_persi_per_collisione", "frutti_per_collisione"])
        for n in names:
            d = data[n]
            m = {k: d[k][-LAST:].mean() for k, _ in METRICS}
            w.writerow([n, len(d["episodio"])] + [round(m[k], 2) for k, _ in METRICS]
                       + [round(m["segmenti_persi"] / m["self_collision"], 2),
                          round(m["frutti_mangiati"] / m["self_collision"], 2)])
    print(open(os.path.join(OUT, "riepilogo.csv")).read())


if __name__ == "__main__":
    main()
