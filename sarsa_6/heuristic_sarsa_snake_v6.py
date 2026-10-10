"""
SARSA Snake - versione migliorata

Modifiche principali rispetto alla versione originale:
1. Action masking:
   - niente muro
   - niente reverse quando esiste una direzione precedente
   - niente corpo come prima scelta
2. Euristica basata su:
   - avanzamento verso il frutto
   - spazio raggiungibile dopo la mossa
   - penalità per situazioni strette / trappole
3. Heuristic Q-bias:
   l'euristica non sostituisce completamente la rete durante epsilon,
   ma può essere usata come bias sui Q-value.
4. SARSA con transizioni terminali:
   in caso di collisione non viene propagato il Q-value dello stato successivo.
5. Metriche aggiuntive:
   spazio raggiungibile medio e numero di azioni "rischiose".
6. Mantiene la struttura OriginalSnakeEnvironment del codice originale.

NOTA:
La board contiene HEAD/BODY/FRUIT/EMPTY/WALL, ma non contiene l'ordine
dei segmenti del corpo. Per questo NON viene fatto un falso controllo
"head -> tail": sarebbe possibile solo se l'environment esponesse la coda.
Il flood-fill è invece coerente con lo stato disponibile.
"""

import csv
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from collections import deque
import matplotlib.animation as animation

from environments_fully_observable_2 import BaseEnvironment, OriginalSnakeEnvironment


# ============================================================================
# 1) PARAMETRI DELL'AGENTE
# ============================================================================

# Peso del bias euristico sui Q-value.
# Alto all'inizio, basso verso la fine.
HEURISTIC_BIAS_START = 0.20
HEURISTIC_BIAS_END = 0.03

# Componenti dell'euristica.
FRUIT_SCORE_WEIGHT = 1.0
SPACE_SCORE_WEIGHT = 0.45
TRAP_PENALTY_WEIGHT = 1.20
DEAD_END_PENALTY = 2.0

# Quante celle libere consideriamo "sufficienti" rispetto alla lunghezza
# del corpo. 1.0 = comportamento conservativo.
SPACE_BODY_RATIO = 1.0

# Se True, il corpo viene considerato invalido come destinazione.
# Se tutte le azioni sicure sono impossibili, viene scelta comunque
# l'azione meno pericolosa per evitare che tutti i Q diventino -inf.
MASK_BODY = False  # Nel tuo environment entrare nel BODY non termina: accorcia la coda.
MASK_REVERSE = True
MASK_WALL = True


# ============================================================================
# 2) UTILITY SULLA BOARD
# ============================================================================

ACTIONS = [
    BaseEnvironment.UP,
    BaseEnvironment.RIGHT,
    BaseEnvironment.DOWN,
    BaseEnvironment.LEFT,
]

OPPOSITE_DIRECTION = {
    BaseEnvironment.UP: BaseEnvironment.DOWN,
    BaseEnvironment.DOWN: BaseEnvironment.UP,
    BaseEnvironment.RIGHT: BaseEnvironment.LEFT,
    BaseEnvironment.LEFT: BaseEnvironment.RIGHT,
}


def get_head(board):
    positions = np.argwhere(board == BaseEnvironment.HEAD)
    if len(positions) == 0:
        return None
    return tuple(int(x) for x in positions[0])


def next_position(board, action):
    """
    Restituisce la posizione della testa dopo action.

    Rispetta gli offset del file originale:
        UP    -> riga + 1
        DOWN  -> riga - 1
        RIGHT -> colonna + 1
        LEFT  -> colonna - 1
    """
    head = get_head(board)
    if head is None:
        return None

    r, c = head

    if action == BaseEnvironment.UP:
        r += 1
    elif action == BaseEnvironment.DOWN:
        r -= 1
    elif action == BaseEnvironment.RIGHT:
        c += 1
    elif action == BaseEnvironment.LEFT:
        c -= 1

    return r, c


def is_inside(board, pos):
    if pos is None:
        return False
    r, c = pos
    return 0 <= r < board.shape[0] and 0 <= c < board.shape[1]


def cell_is_wall(board, pos):
    return (not is_inside(board, pos)) or board[pos] == BaseEnvironment.WALL


def cell_is_body(board, pos):
    return is_inside(board, pos) and board[pos] == BaseEnvironment.BODY


def fruit_position(board):
    fruit = np.argwhere(board == BaseEnvironment.FRUIT)
    if len(fruit) == 0:
        return None
    return tuple(int(x) for x in fruit[0])


# ============================================================================
# 3) FLOOD-FILL / SPAZIO RAGGIUNGIBILE
# ============================================================================

def reachable_space(board, start=None):
    """
    Conta quante celle EMPTY/FRUIT sono raggiungibili.

    Il BODY viene trattato come ostacolo.

    Questa è volutamente una valutazione conservativa:
    non conoscendo la posizione/ordine della coda, non assumiamo
    che una parte del corpo diventerà libera.
    """
    if start is None:
        start = get_head(board)

    if start is None or not is_inside(board, start):
        return 0

    walkable = (
        (board == BaseEnvironment.EMPTY)
        | (board == BaseEnvironment.FRUIT)
    )

    # Se start è una cella BODY, non può partire il flood-fill.
    # Nel caso normale start è la nuova testa e sarà HEAD.
    walkable = walkable.copy()
    walkable[start] = True

    visited = {start}
    queue = deque([start])
    count = 0

    while queue:
        r, c = queue.popleft()

        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc

            if not (0 <= nr < board.shape[0] and
                    0 <= nc < board.shape[1]):
                continue

            pos = (nr, nc)

            if pos in visited:
                continue

            if not walkable[nr, nc]:
                continue

            visited.add(pos)
            queue.append(pos)
            count += 1

    return count


def simulate_board_after_action(board, action):
    """
    Simula SOLO la geometria della mossa.

    Non conosciamo l'ordine del corpo, quindi non facciamo avanzare
    realmente la coda. Questo rende la simulazione conservativa.

    Ritorna:
        simulated_board, valid
    """
    pos = next_position(board, action)

    if not is_inside(board, pos):
        return board.copy(), False

    if board[pos] == BaseEnvironment.WALL:
        return board.copy(), False

    simulated = board.copy()

    head = get_head(board)
    if head is None:
        return simulated, False

    # La vecchia testa diventa BODY.
    simulated[head] = BaseEnvironment.BODY

    # La nuova posizione diventa HEAD. Se era BODY, la simulazione
    # conserva comunque il body originale: il vero environment lo
    # accorcerà usando self.bodies. Per questo una collisione BODY
    # riceve una penalità separata sotto.
    simulated[pos] = BaseEnvironment.HEAD

    return simulated, True


# ============================================================================
# 4) VALUTAZIONE DELLE AZIONI
# ============================================================================

def action_is_hard_valid(board, action, current_direction):
    """
    Validità "hard".

    Non permettiamo:
      - fuori board / muro
      - reverse
    Il BODY NON viene hard-masked: nel tuo environment una collisione
    con il corpo rimuove il segmento colpito e tutta la coda successiva,
    assegnando ATE_HIMSELF_REWARD per segmento.

    Se tutte le azioni sono impossibili, il chiamante può usare
    una fallback policy.
    """
    pos = next_position(board, action)

    if not is_inside(board, pos):
        return False

    if MASK_WALL and board[pos] == BaseEnvironment.WALL:
        return False

    if MASK_BODY and board[pos] == BaseEnvironment.BODY:
        return False

    if (
        MASK_REVERSE
        and current_direction in OPPOSITE_DIRECTION
        and action == OPPOSITE_DIRECTION[current_direction]
    ):
        return False

    return True


def get_hard_valid_actions(board, current_direction):
    return [
        a for a in ACTIONS
        if action_is_hard_valid(board, a, current_direction)
    ]


def get_non_wall_actions(board):
    """
    Fallback: azioni che almeno non vanno contro il muro/out-of-bounds.
    """
    valid = []

    for action in ACTIONS:
        pos = next_position(board, action)
        if is_inside(board, pos) and board[pos] != BaseEnvironment.WALL:
            valid.append(action)

    return valid


def normalized_space_score(space, board):
    """
    Trasforma lo spazio raggiungibile in un valore circa [0, 1].
    """
    free_cells = int(
        np.sum(
            (board == BaseEnvironment.EMPTY)
            | (board == BaseEnvironment.FRUIT)
        )
    )

    if free_cells <= 0:
        return 0.0

    return min(1.0, space / float(free_cells))


def action_heuristic_score(board, action, current_direction):
    """
    Score euristico di una singola azione.

    Priorità:
      1. non morire
      2. mantenere spazio
      3. evitare dead-end/trappole
      4. avvicinarsi al frutto

    Ritorna:
        score, reachable_space, is_safe
    """
    simulated, valid = simulate_board_after_action(board, action)

    if not valid:
        return -1e6, 0, False

    new_head = get_head(simulated)
    space = reachable_space(simulated, new_head)

    score = 0.0

    # ------------------------------------------------------------------
    # Spazio
    # ------------------------------------------------------------------
    space_norm = normalized_space_score(space, simulated)
    score += SPACE_SCORE_WEIGHT * space_norm

    # ------------------------------------------------------------------
    # Self-collision:
    # nel tuo environment NON è terminale, ma costa -.1 per segmento
    # perso. La trattiamo quindi come azione possibile ma rischiosa.
    # ------------------------------------------------------------------
    target_pos = next_position(board, action)
    if is_inside(board, target_pos) and board[target_pos] == BaseEnvironment.BODY:
        score -= 1.0

    # ------------------------------------------------------------------
    # Penalità se lo spazio disponibile è troppo piccolo
    # rispetto alla lunghezza del corpo.
    # ------------------------------------------------------------------
    body_len = int(np.sum(simulated == BaseEnvironment.BODY))
    needed_space = max(1, int(np.ceil((body_len + 1) * SPACE_BODY_RATIO)))

    if space < needed_space:
        deficit = 1.0 - (space / float(needed_space))
        score -= TRAP_PENALTY_WEIGHT * deficit

    # Dead end molto forte.
    if space <= 1:
        score -= DEAD_END_PENALTY

    # ------------------------------------------------------------------
    # Frutto
    # ------------------------------------------------------------------
    fruit = fruit_position(board)

    if fruit is not None and new_head is not None:
        old_head = get_head(board)

        old_dist = (
            abs(old_head[0] - fruit[0])
            + abs(old_head[1] - fruit[1])
        )

        new_dist = (
            abs(new_head[0] - fruit[0])
            + abs(new_head[1] - fruit[1])
        )

        # Positivo se ci avviciniamo, negativo se ci allontaniamo.
        progress = old_dist - new_dist

        # Normalizziamo grossolanamente per evitare che la distanza
        # domini completamente la componente di sicurezza.
        score += FRUIT_SCORE_WEIGHT * (progress / max(1, board.shape[0]))

    return score, space, True


def heuristic_action(board, current_direction):
    """
    Versione migliorata dell'euristica originale.

    Non sceglie più semplicemente la minima Manhattan distance.
    Valuta spazio + rischio trappola + progresso verso il frutto.
    """
    safe_actions = get_hard_valid_actions(board, current_direction)

    if safe_actions:
        candidates = safe_actions
    else:
        # Situazione critica: nessuna mossa "safe".
        # Scegliamo comunque una mossa non-muro.
        candidates = get_non_wall_actions(board)

        if not candidates:
            # Nessuna mossa possibile.
            return int(
                OPPOSITE_DIRECTION.get(
                    current_direction,
                    BaseEnvironment.UP
                )
            )

    scores = []
    for action in candidates:
        score, _, _ = action_heuristic_score(
            board, action, current_direction
        )
        scores.append(score)

    max_score = max(scores)

    # Manteniamo una piccola casualità sulle azioni equivalenti.
    best = [
        action
        for action, score in zip(candidates, scores)
        if np.isclose(score, max_score)
    ]

    return int(np.random.choice(best))


def heuristic_actions_batch(boards, current_directions):
    return np.array(
        [
            heuristic_action(boards[i], current_directions[i])
            for i in range(len(boards))
        ],
        dtype=np.int32,
    )


# ============================================================================
# 5) ACTION MASK + HEURISTIC BIAS
# ============================================================================

def build_action_mask(board, current_direction):
    """
    True = azione utilizzabile dalla policy greedy.
    """
    mask = np.zeros(4, dtype=bool)

    for action in ACTIONS:
        mask[action] = action_is_hard_valid(
            board,
            action,
            current_direction,
        )

    return mask


def heuristic_scores_for_board(board, current_direction):
    """
    Restituisce 4 score euristici, uno per azione.

    Le azioni non valide ricevono un valore molto negativo.
    """
    scores = np.full(4, -1e6, dtype=np.float32)

    for action in ACTIONS:
        score, _, valid = action_heuristic_score(
            board,
            action,
            current_direction,
        )

        if valid:
            scores[action] = score

    return scores


def heuristic_scores_batch(boards, current_directions):
    return np.stack(
        [
            heuristic_scores_for_board(
                boards[i],
                current_directions[i],
            )
            for i in range(len(boards))
        ],
        axis=0,
    )


def choose_actions(
    env,
    q_network,
    current_directions,
    epsilon,
    heuristic_bias=0.0,
):
    """
    Policy epsilon-greedy migliorata.

    Greedy:
        Q_final = Q_network + heuristic_bias * normalized_heuristic

    Exploration:
        usa la stessa euristica safety-aware.

    In entrambi i casi applichiamo action masking alle azioni sicuramente
    invalide.

    Fallback:
        se una board non ha alcuna azione hard-safe, non lasciamo tutti
        i Q a -inf; scegliamo l'azione meno pericolosa.
    """
    states = env.to_state()
    q_values = q_network(states).numpy()

    final_q = q_values.copy()

    h_scores = heuristic_scores_batch(
        env.boards,
        current_directions,
    )

    # Normalizzazione per board: evita che il valore assoluto
    # dell'euristica domini i Q della rete.
    h_min = np.min(h_scores, axis=1, keepdims=True)
    h_max = np.max(h_scores, axis=1, keepdims=True)
    h_norm = (h_scores - h_min) / (h_max - h_min + 1e-8)

    if heuristic_bias > 0.0:
        final_q += heuristic_bias * h_norm

    # ------------------------------------------------------------------
    # Action masking
    # ------------------------------------------------------------------
    for i in range(env.n_boards):
        mask = build_action_mask(
            env.boards[i],
            current_directions[i],
        )

        if np.any(mask):
            final_q[i, ~mask] = -1e9
        else:
            # Nessuna azione hard-safe.
            # Usiamo la migliore azione euristica valida/non-muro.
            fallback = heuristic_action(
                env.boards[i],
                current_directions[i],
            )

            final_q[i, :] = -1e9
            final_q[i, fallback] = q_values[i, fallback]

    actions = np.argmax(final_q, axis=-1).astype(np.int32)

    # ------------------------------------------------------------------
    # Exploration: euristica safety-aware.
    # ------------------------------------------------------------------
    explore = np.random.rand(env.n_boards) < epsilon

    if np.any(explore):
        idx = np.where(explore)[0]

        for i in idx:
            actions[i] = heuristic_action(
                env.boards[i],
                current_directions[i],
            )

    return actions


# ============================================================================
# 6) Q NETWORK
# ============================================================================

def build_q_network(board_size, n_actions=4):
    inputs = tf.keras.Input(
        shape=(board_size, board_size, 4)
    )

    x = tf.keras.layers.Conv2D(
        32,
        3,
        padding="same",
        activation="relu",
    )(inputs)

    x = tf.keras.layers.Conv2D(
        64,
        3,
        padding="same",
        activation="relu",
    )(x)

    x = tf.keras.layers.Conv2D(
        64,
        3,
        padding="same",
        activation="relu",
    )(x)

    x = tf.keras.layers.Flatten()(x)

    x = tf.keras.layers.Dense(
        128,
        activation="relu",
    )(x)

    outputs = tf.keras.layers.Dense(
        n_actions,
        activation="linear",
    )(x)

    return tf.keras.Model(inputs, outputs)


# ============================================================================
# 7) VISUALIZZAZIONE
# ============================================================================

BOARD_COLORS = {
    BaseEnvironment.WALL: (0.05, 0.05, 0.05),
    BaseEnvironment.EMPTY: (0.95, 0.95, 0.95),
    BaseEnvironment.FRUIT: (0.85, 0.10, 0.10),
    BaseEnvironment.BODY: (0.10, 0.50, 0.10),
    BaseEnvironment.HEAD: (0.10, 0.90, 0.10),
}


def board_to_rgb(board):
    rgb = np.zeros((*board.shape, 3))

    for value, color in BOARD_COLORS.items():
        rgb[board == value] = color

    return rgb


def record_episode(
    env,
    q_network,
    board_index=0,
    max_steps=200,
    epsilon=0.0,
):
    frames = [env.boards[board_index].copy()]

    current_directions = np.full(
        env.n_boards,
        BaseEnvironment.NONE,
    )

    total_reward = 0.0

    for _ in range(max_steps):
        # Durante evaluation non usiamo il bias euristico.
        actions = choose_actions(
            env,
            q_network,
            current_directions,
            epsilon,
            heuristic_bias=0.0,
        )

        rewards = env.move(actions)
        rewards_np = rewards.numpy().reshape(-1)

        total_reward += float(rewards_np[board_index])

        current_directions = actions

        frames.append(
            env.boards[board_index].copy()
        )

        # Se l'environment non termina internamente l'episodio dopo
        # la collisione, almeno interrompiamo la registrazione.
        if rewards_np[board_index] < 0:
            break

    return frames, total_reward


def save_episode_gif(
    frames,
    path="snake_episode.gif",
    interval_ms=200,
):
    fig, ax = plt.subplots(figsize=(4, 4))

    im = ax.imshow(
        board_to_rgb(frames[0])
    )

    ax.set_xticks([])
    ax.set_yticks([])

    title = ax.set_title("step 0")

    def update(i):
        im.set_data(
            board_to_rgb(frames[i])
        )
        title.set_text(
            f"step {i}"
        )
        return [im, title]

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(frames),
        interval=interval_ms,
    )

    ani.save(
        path,
        writer="pillow",
    )

    plt.close(fig)

    print(
        f"GIF salvata in: {path}"
    )


def visualize_policy(
    q_network,
    board_size,
    epsilon=0.0,
    max_steps=500,
    path="snake_episode.gif",
):
    eval_env = OriginalSnakeEnvironment(
        n_boards=1,
        board_size=board_size,
    )

    frames, total_reward = record_episode(
        eval_env,
        q_network,
        board_index=0,
        max_steps=max_steps,
        epsilon=epsilon,
    )

    save_episode_gif(
        frames,
        path=path,
    )

    print(
        f"Reward totale nell'episodio registrato: "
        f"{total_reward:.2f}"
    )

    return frames, total_reward


# ============================================================================
# 8) SARSA
# ============================================================================

def sarsa_train(
    env,
    q_network,
    optimizer,
    n_episodes=200,
    max_steps=500,
    epsilon_start=0.20,
    epsilon_end=0.02,
    gamma=0.95,
    visualize_every=None,
    csv_path="training_history_sarsa_v2.csv",
    csv_every=20,
):
    """
    SARSA on-policy con:
      - safe action masking sui muri/reverse
      - heuristic exploration
      - heuristic Q-bias
      - gestione corretta delle collisioni secondo il tuo environment

    IMPORTANTE:
    il tuo environment NON espone `done` e una self-collision NON è
    terminale: move() elimina il segmento colpito e tutta la coda
    successiva, con reward ATE_HIMSELF_REWARD * len(to_delete).
    Anche un wall-hit non è terminale: la testa resta ferma e viene
    assegnato HIT_WALL_REWARD.
    """

    history = {
        "reward_medio": [],
        "frutti_mangiati": [],
        "self_collision": [],
        "segmenti_persi": [],
        "loss_media": [],
        "spazio_medio": [],
        "azioni_rischiose": [],
    }

    for episode in range(n_episodes):

        # ---------------------------------------------------------------
        # Decadimento epsilon lineare.
        # ---------------------------------------------------------------
        fraction = episode / max(n_episodes - 1, 1)

        epsilon = (
            epsilon_start
            - (epsilon_start - epsilon_end) * fraction
        )

        # ---------------------------------------------------------------
        # Bias euristico più forte all'inizio.
        # ---------------------------------------------------------------
        heuristic_bias = (
            HEURISTIC_BIAS_START
            - (HEURISTIC_BIAS_START - HEURISTIC_BIAS_END)
            * fraction
        )

        current_directions = np.full(
            env.n_boards,
            BaseEnvironment.NONE,
        )

        actions = choose_actions(
            env,
            q_network,
            current_directions,
            epsilon,
            heuristic_bias=heuristic_bias,
        )

        ep_segments_lost = np.zeros(
            env.n_boards,
            dtype=np.float32,
        )

        ep_reward = np.zeros(
            env.n_boards,
            dtype=np.float32,
        )

        ep_fruits = np.zeros(
            env.n_boards,
            dtype=np.float32,
        )

        ep_self_collision = np.zeros(
            env.n_boards,
            dtype=np.float32,
        )

        ep_space = np.zeros(
            env.n_boards,
            dtype=np.float32,
        )

        ep_risky_actions = np.zeros(
            env.n_boards,
            dtype=np.float32,
        )

        ep_losses = []

        # ===============================================================
        # EPISODIO
        # ===============================================================
        for step in range(max_steps):

            state = env.to_state()

            # -----------------------------------------------------------
            # Valutiamo lo spazio PRIMA della move, per diagnostica.
            # -----------------------------------------------------------
            for i in range(env.n_boards):
                ep_space[i] += reachable_space(
                    env.boards[i]
                )

            # -----------------------------------------------------------
            # Esegui A
            # -----------------------------------------------------------
            rewards = env.move(actions)
            rewards_np = rewards.numpy().reshape(-1)

            ep_reward += rewards_np

            ep_fruits += (
                np.isclose(
                    rewards_np,
                    env.FRUIT_REWARD,
                )
            ).astype(float)

            # -----------------------------------------------------------
            # Collisioni
            # -----------------------------------------------------------
            wall_collision = np.isclose(
                rewards_np,
                env.HIT_WALL_REWARD,
            )

            is_negative = rewards_np < 0

            # Tutti i negativi che NON sono muro vengono interpretati
            # come self-collision, come nel codice originale.
            is_self_collision = (
                is_negative
                & (~wall_collision)
            )

            ep_self_collision += (
                is_self_collision.astype(float)
            )

            # -----------------------------------------------------------
            # Segmenti persi
            # -----------------------------------------------------------
            segments_lost_step = np.zeros_like(
                rewards_np,
                dtype=np.float32,
            )

            nonzero_penalty = (
                is_self_collision
                & (~np.isclose(
                    env.ATE_HIMSELF_REWARD,
                    0.0,
                ))
            )

            segments_lost_step[nonzero_penalty] = (
                rewards_np[nonzero_penalty]
                / env.ATE_HIMSELF_REWARD
            )

            ep_segments_lost += (
                segments_lost_step
            )

            # -----------------------------------------------------------
            # Nel tuo environment NON c'è una transizione terminale
            # associata ai reward negativi:
            # - muro: la testa resta ferma
            # - self collision: la coda viene accorciata
            #
            # Quindi done = False.
            # -----------------------------------------------------------
            done = np.zeros(
                env.n_boards,
                dtype=bool,
            )

            # -----------------------------------------------------------
            # Stato successivo
            # -----------------------------------------------------------
            next_state = env.to_state()

            # SARSA: A' viene scelta dalla stessa policy.
            next_actions = choose_actions(
                env,
                q_network,
                actions,
                epsilon,
                heuristic_bias=heuristic_bias,
            )

            # -----------------------------------------------------------
            # TD UPDATE
            # -----------------------------------------------------------
            with tf.GradientTape() as tape:

                q_values = q_network(
                    state,
                    training=True,
                )

                q_sa = tf.gather(
                    q_values,
                    actions,
                    axis=1,
                    batch_dims=1,
                )

                next_q_values = q_network(
                    next_state,
                    training=False,
                )

                next_q_sa = tf.gather(
                    next_q_values,
                    next_actions,
                    axis=1,
                    batch_dims=1,
                )

                next_q_sa = tf.stop_gradient(
                    next_q_sa
                )

                reward_tensor = tf.convert_to_tensor(
                    rewards_np,
                    dtype=tf.float32,
                )

                done_tensor = tf.convert_to_tensor(
                    done.astype(np.float32),
                    dtype=tf.float32,
                )

                # -------------------------------------------------------
                # SARSA target:
                #
                # terminal:
                #     target = reward
                #
                # non terminal:
                #     target = reward + gamma * Q(S',A')
                # -------------------------------------------------------
                target = (
                    reward_tensor
                    + gamma
                    * (1.0 - done_tensor)
                    * next_q_sa
                )

                td_error = target - q_sa

                # Huber loss più robusta della MSE pura quando
                # ci sono collisioni con reward molto negativi.
                loss = tf.reduce_mean(
                    tf.keras.losses.huber(
                        target,
                        q_sa,
                    )
                )

            grads = tape.gradient(
                loss,
                q_network.trainable_variables,
            )

            grads_and_vars = [
                (g, v)
                for g, v in zip(
                    grads,
                    q_network.trainable_variables,
                )
                if g is not None
            ]

            optimizer.apply_gradients(
                grads_and_vars
            )

            ep_losses.append(
                float(loss.numpy())
            )

            # -----------------------------------------------------------
            # Stato/azione correnti per il prossimo step.
            # -----------------------------------------------------------
            current_directions = actions
            actions = next_actions

            # -----------------------------------------------------------
            # Se tutte le board sono morte, non ha senso continuare.
            #
            # NOTA: questa condizione assume che un reward negativo
            # corrisponda a terminalità nel tuo environment.
            # -----------------------------------------------------------
            if np.all(done):
                break

        # ===============================================================
        # METRICHE EPISODIO
        # ===============================================================
        history["reward_medio"].append(
            float(np.mean(ep_reward))
        )

        history["frutti_mangiati"].append(
            float(np.mean(ep_fruits))
        )

        history["segmenti_persi"].append(
            float(np.mean(ep_segments_lost))
        )

        history["self_collision"].append(
            float(np.mean(ep_self_collision))
        )

        history["loss_media"].append(
            float(np.mean(ep_losses))
            if ep_losses
            else 0.0
        )

        history["spazio_medio"].append(
            float(np.mean(ep_space))
            / max(1, step + 1)
        )

        history["azioni_rischiose"].append(
            float(
                np.mean(
                    ep_self_collision
                    > 0
                )
            )
        )

        # ---------------------------------------------------------------
        # Logging
        # ---------------------------------------------------------------
        if episode % 10 == 0:
            print(
                f"Episodio {episode:4d} | "
                f"reward={history['reward_medio'][-1]:8.3f} | "
                f"frutti={history['frutti_mangiati'][-1]:6.3f} | "
                f"self_col={history['self_collision'][-1]:6.3f} | "
                f"spazio={history['spazio_medio'][-1]:7.2f} | "
                f"eps={epsilon:.3f} | "
                f"bias={heuristic_bias:.3f}"
            )

        # ---------------------------------------------------------------
        # Checkpoint visuale
        # ---------------------------------------------------------------
        if (
            visualize_every
            and episode % visualize_every == 0
            and episode > 0
        ):
            visualize_policy(
                q_network,
                env.board_size,
                epsilon=0.0,
                path=(
                    f"snake_sarsa_v2_ep"
                    f"{episode}.gif"
                ),
            )

        # ---------------------------------------------------------------
        # Salvataggio CSV
        # ---------------------------------------------------------------
        if (
            csv_every
            and episode % csv_every == 0
            and episode > 0
        ):
            save_history_csv(
                history,
                path=csv_path,
            )

    return history


# ============================================================================
# 9) PLOT
# ============================================================================

def plot_training_history(
    history,
    path="training_history_sarsa_v2.png",
    window=10,
):
    def smooth(values, window):
        values = np.asarray(
            values,
            dtype=np.float32,
        )

        if len(values) < window:
            return values

        kernel = (
            np.ones(window)
            / window
        )

        return np.convolve(
            values,
            kernel,
            mode="valid",
        )

    keys = [
        ("reward_medio", "Reward medio"),
        ("frutti_mangiati", "Frutti mangiati"),
        ("self_collision", "Self-collision"),
        ("spazio_medio", "Spazio raggiungibile"),
        ("loss_media", "Loss"),
        ("segmenti_persi", "Segmenti persi"),
    ]

    fig, axes = plt.subplots(
        3,
        2,
        figsize=(11, 10),
    )

    for ax, (key, title) in zip(
        axes.flat,
        keys,
    ):
        raw = history[key]

        ax.plot(
            raw,
            alpha=0.25,
            label="grezzo",
        )

        if len(raw) >= window:
            smoothed = smooth(
                raw,
                window,
            )

            ax.plot(
                range(
                    window - 1,
                    len(raw),
                ),
                smoothed,
                label=f"media {window}",
            )

        ax.set_title(title)
        ax.set_xlabel("episodio")
        ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)

    print(
        f"Grafico salvato in: {path}"
    )


# ============================================================================
# 10) CSV
# ============================================================================

def save_history_csv(
    history,
    path="training_history_sarsa_v2.csv",
):
    n_episodes = len(
        history["reward_medio"]
    )

    fields = [
        "episodio",
        "reward_medio",
        "frutti_mangiati",
        "self_collision",
        "segmenti_persi",
        "spazio_medio",
        "azioni_rischiose",
        "loss_media",
    ]

    with open(
        path,
        "w",
        newline="",
    ) as f:
        writer = csv.writer(f)
        writer.writerow(fields)

        for i in range(n_episodes):
            writer.writerow(
                [
                    i,
                    history["reward_medio"][i],
                    history["frutti_mangiati"][i],
                    history["self_collision"][i],
                    history["segmenti_persi"][i],
                    history["spazio_medio"][i],
                    history["azioni_rischiose"][i],
                    history["loss_media"][i],
                ]
            )

    print(
        f"CSV salvato in: {path}"
    )


# ============================================================================
# 11) MAIN
# ============================================================================

if __name__ == "__main__":

    BOARD_SIZE = 10
    N_BOARDS = 32

    env = OriginalSnakeEnvironment(
        n_boards=N_BOARDS,
        board_size=BOARD_SIZE,
    )

    q_network = build_q_network(
        BOARD_SIZE
    )

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=3e-4
    )

    history = sarsa_train(
        env,
        q_network,
        optimizer,
        n_episodes=200,
        max_steps=500,
        epsilon_start=0.25,
        epsilon_end=0.03,
        gamma=0.95,
        visualize_every=50,
        csv_path="training_history_sarsa_v2.csv",
        csv_every=20,
    )

    plot_training_history(
        history,
        path="training_history_sarsa_v2.png",
    )

    print(
        "\nRegistro episodio finale..."
    )

    visualize_policy(
        q_network,
        BOARD_SIZE,
        epsilon=0.0,
        max_steps=500,
        path="snake_final_sarsa_v2.gif",
    )
