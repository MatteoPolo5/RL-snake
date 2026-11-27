"""
Baseline euristica (nessun apprendimento) per Snake.

Esegue la politica a regole che negli agenti RL e' usata per l'esplorazione,
qui da sola, con lo stesso protocollo della valutazione greedy degli agenti
(evaluate_policy negli script di training): partite nuove da 500 passi,
seed fisso, stesse metriche. Cosi' i numeri sono confrontabili con la
Tabella 2 del paper e con le curve di training (un episodio = 500 passi).

Regole (a ogni passo, per ogni board, guardando le 4 celle vicine alla testa):
  1. esclude il dietro-front (opposto dell'ultima direzione);
  2. esclude le mosse contro il muro;
  3. se non resta nulla, torna indietro (ultima spiaggia);
  4. preferisce le mosse che non finiscono sul corpo (se non ce ne sono,
     le accetta tutte);
  5. fra queste sceglie quella che avvicina la testa al frutto (distanza di
     Manhattan); in partially observable vede solo la finestra 5x5 e, se il
     frutto non e' visibile, salta questa regola;
  6. parita' -> scelta casuale.

Uso:
    python baseline.py
Stampa le metriche per fully e partially observable e le salva in
baseline/baseline_risultati.csv.
"""
import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "sarsa_4"))     # ambienti usati nei training
from environments_fully_observable_2 import BaseEnvironment            # noqa: E402
import environments_fully_observable_2 as env_fully                    # noqa: E402
import environments_partially_observable_2 as env_partial              # noqa: E402

BOARD_SIZE = 10
MASK_SIZE = 2            # finestra 5x5 in partially observable
N_GAMES = 20
MAX_STEPS = 500
SEED = 12345             # stesso seed della valutazione greedy degli agenti
STUCK_WINDOW = 100       # "bloccata" = nessun frutto negli ultimi 100 passi
OUT_DIR = os.path.join(HERE, "baseline")

UP, RIGHT, DOWN, LEFT = BaseEnvironment.UP, BaseEnvironment.RIGHT, BaseEnvironment.DOWN, BaseEnvironment.LEFT
OPPOSITE = {UP: DOWN, DOWN: UP, RIGHT: LEFT, LEFT: RIGHT}


# ---------------------------------------------------------------------------
# Politica euristica (identica a heuristic_action degli script di training)
# ---------------------------------------------------------------------------
def heuristic_action(board, current_direction):
    head_pos = np.argwhere(board == BaseEnvironment.HEAD)[0]
    fruit_candidates = np.argwhere(board == BaseEnvironment.FRUIT)
    fruit_pos = fruit_candidates[0] if len(fruit_candidates) else None

    def next_pos(a):
        r, c = head_pos
        if a == UP:
            r += 1
        elif a == DOWN:
            r -= 1
        elif a == RIGHT:
            c += 1
        elif a == LEFT:
            c -= 1
        return r, c

    # 1) niente dietro-front
    candidate_actions = [a for a in (UP, RIGHT, DOWN, LEFT) if a != OPPOSITE.get(current_direction, -1)]
    # 2) niente muri
    valid_actions = []
    for a in candidate_actions:
        r, c = next_pos(a)
        if 0 <= r < board.shape[0] and 0 <= c < board.shape[1] and board[r, c] != BaseEnvironment.WALL:
            valid_actions.append(a)
    # 3) ultima spiaggia
    if not valid_actions:
        valid_actions = [OPPOSITE.get(current_direction, UP)]
    # 4) evita il corpo se possibile
    no_body = [a for a in valid_actions if board[next_pos(a)] != BaseEnvironment.BODY]
    candidates = no_body if no_body else valid_actions
    # 5) verso il frutto (se visibile)
    if fruit_pos is not None:
        dist = {a: abs(next_pos(a)[0] - fruit_pos[0]) + abs(next_pos(a)[1] - fruit_pos[1]) for a in candidates}
        best = [a for a in candidates if dist[a] == min(dist.values())]
    else:
        best = candidates
    # 6) parita' -> caso
    return int(np.random.choice(best))


# ---------------------------------------------------------------------------
# Ambiente e osservazione
# ---------------------------------------------------------------------------
def make_env(observability, n_boards):
    if observability == "partial":
        return env_partial.OriginalSnakeEnvironment(n_boards=n_boards, board_size=BOARD_SIZE,
                                                    mask_size=MASK_SIZE)
    return env_fully.OriginalSnakeEnvironment(n_boards=n_boards, board_size=BOARD_SIZE)


def visible_boards(env, observability):
    """Board intere (fully) o finestra 5x5 attorno alla testa (partial, fuori board = muro)."""
    if observability != "partial":
        return env.boards
    m = MASK_SIZE
    padded = np.pad(env.boards, ((0, 0), (m, m), (m, m)), constant_values=BaseEnvironment.WALL)
    heads = np.argwhere(padded == BaseEnvironment.HEAD)
    return np.array([padded[b, r - m:r + m + 1, c - m:c + m + 1] for b, r, c in heads])


def body_lengths(env):
    return np.array([len(b) for b in env.bodies], dtype=float)


def head_positions(env):
    return np.argwhere(env.boards == BaseEnvironment.HEAD)[:, 1:]


def segments_lost_in_step(env, len_before, env_rewards):
    """Segmenti persi per auto-collisione, ricavati dallo stato (come negli script di training)."""
    ate = np.isclose(env_rewards, env.FRUIT_REWARD).astype(float)
    lost = len_before + ate - body_lengths(env)
    lost[np.isclose(env_rewards, env.WIN_REWARD)] = 0.0
    return np.maximum(lost, 0.0)


# ---------------------------------------------------------------------------
# Valutazione (stesso protocollo di evaluate_policy degli agenti)
# ---------------------------------------------------------------------------
def evaluate_baseline(observability, n_games=N_GAMES, max_steps=MAX_STEPS, seed=SEED):
    np.random.seed(seed)
    env = make_env(observability, n_games)
    directions = np.full(n_games, BaseEnvironment.NONE)
    fruits = np.zeros(n_games)
    collisions = np.zeros(n_games)
    lost_total = np.zeros(n_games)
    wall_hits = np.zeros(n_games)
    length_sum = np.zeros(n_games)
    max_len = body_lengths(env) + 1
    last_fruit_step = np.full(n_games, -1)

    for step in range(max_steps):
        boards = visible_boards(env, observability)
        actions = np.array([heuristic_action(boards[i], directions[i]) for i in range(n_games)],
                           dtype=np.int32)
        len_before = body_lengths(env)
        heads_before = head_positions(env)
        env_rewards = np.asarray(env.move(actions)).reshape(-1)
        lost = segments_lost_in_step(env, len_before, env_rewards)
        ate = np.isclose(env_rewards, env.FRUIT_REWARD)
        fruits += ate
        last_fruit_step[ate] = step
        collisions += lost > 0
        lost_total += lost
        wall_hits += np.all(head_positions(env) == heads_before, axis=1)
        length_sum += body_lengths(env) + 1
        max_len = np.maximum(max_len, body_lengths(env) + 1)
        directions = actions

    stuck = last_fruit_step < max(max_steps - STUCK_WINDOW, 0)
    return {
        "osservabilita": observability,
        "n_partite": n_games,
        "seed": seed,
        "frutti": fruits.mean(),
        "frutti_std": fruits.std(),
        "frutti_min": fruits.min(),
        "frutti_max": fruits.max(),
        "self_collision": collisions.mean(),
        "segmenti_persi": lost_total.mean(),
        "lunghezza_media": (length_sum / max_steps).mean(),
        "lunghezza_max": max_len.mean(),
        "partite_bloccate": int(stuck.sum()),
        "urti_muro": wall_hits.mean(),
    }


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    results = [evaluate_baseline(obs) for obs in ("fully", "partial")]
    path = os.path.join(OUT_DIR, "baseline_risultati.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0]))
        w.writeheader()
        for r in results:
            w.writerow({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})
    for r in results:
        print(f"\n[{r['osservabilita']}] {r['n_partite']} partite da {MAX_STEPS} passi, seed {r['seed']}")
        print(f"  frutti           {r['frutti']:.1f} +- {r['frutti_std']:.1f}  (min {r['frutti_min']:.0f}, max {r['frutti_max']:.0f})")
        print(f"  auto-collisioni  {r['self_collision']:.2f}")
        print(f"  segmenti persi   {r['segmenti_persi']:.1f}")
        print(f"  lunghezza media  {r['lunghezza_media']:.1f}  (max {r['lunghezza_max']:.1f})")
        print(f"  partite bloccate {r['partite_bloccate']} / {r['n_partite']}")
        print(f"  urti col muro    {r['urti_muro']:.2f}")
    print(f"\nRisultati salvati in {path}")
