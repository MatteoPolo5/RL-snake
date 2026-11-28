"""
Valutazione dell'agente finale consegnato (SARSA, ambiente fully observable).

Carica i pesi da weights/q_network.keras, fa giocare la policy greedy
(epsilon = 0, nessuna esplorazione) su partite nuove e stampa le
prestazioni, confrontate con la baseline euristica sulle stesse partite.

Uso:
    python evaluate.py

Protocollo (lo stesso del report): N_GAMES partite in parallelo su board
10x10, MAX_STEPS passi ciascuna, seed 0. Mosse contro il muro e dietro-front
sono escluse dalle regole del gioco (maschera delle azioni, come in training).
"""
import os

import numpy as np
import tensorflow as tf

import train                      # funzioni di ambiente, maschere e valutazione
from baseline import evaluate_baseline

HERE = os.path.dirname(os.path.abspath(__file__))
WEIGHTS = os.path.join(HERE, "weights", "q_network.keras")
BOARD_SIZE = 10
N_GAMES = 20
MAX_STEPS = 500
SEED = 0


def main():
    tf.keras.utils.set_random_seed(SEED)
    train.OSSERVABILITA = "fully"          # l'agente consegnato vede l'intera board
    q_network = tf.keras.models.load_model(WEIGHTS)
    print(f"Pesi caricati da {WEIGHTS}")

    agent = train.evaluate_policy(q_network, BOARD_SIZE, n_games=N_GAMES,
                                  max_steps=MAX_STEPS, seed=SEED)
    base = evaluate_baseline("fully", n_games=N_GAMES, max_steps=MAX_STEPS, seed=SEED)

    rows = [
        ("frutti per partita", agent["frutti"], base["frutti"], "{:.1f}"),
        ("auto-collisioni per partita", agent["self_collision"], base["self_collision"], "{:.2f}"),
        ("segmenti persi per partita", agent["segmenti_persi"], base["segmenti_persi"], "{:.1f}"),
        ("lunghezza massima", agent["lunghezza_max"], base["lunghezza_max"], "{:.1f}"),
        ("urti col muro per partita", agent["urti_muro"], base["urti_muro"], "{:.2f}"),
        (f"partite bloccate (su {N_GAMES})", agent["partite_bloccate"], base["partite_bloccate"], "{:.0f}"),
    ]
    print(f"\nValutazione greedy: {N_GAMES} partite nuove da {MAX_STEPS} passi, seed {SEED}\n")
    print(f"{'metrica':<32}{'agente SARSA':>14}{'baseline':>12}")
    for name, a, b, fmt in rows:
        print(f"{name:<32}{fmt.format(a):>14}{fmt.format(b):>12}")
    print("\nPartita bloccata = nessun frutto negli ultimi 100 passi (tipicamente un loop).")


if __name__ == "__main__":
    main()
