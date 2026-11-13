"""
Euristica di scelta azione per Snake + integrazione in una struttura SARSA.

Fa riferimento a `environments_fully_observable.py`:
- board: matrice 2D con valori HEAD=4, BODY=3, FRUIT=2, EMPTY=1, WALL=0
- azioni: UP=0, RIGHT=1, DOWN=2, LEFT=3, NONE=4
- offset applicati in move(): UP -> riga+1, DOWN -> riga-1,
                              RIGHT -> colonna+1, LEFT -> colonna-1
"""

import os
import csv
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from collections import deque
import matplotlib.animation as animation
from environments_fully_observable_2 import BaseEnvironment, OriginalSnakeEnvironment


# ---------------------------------------------------------------------------
# 1) EURISTICA (il diagramma di flusso tradotto in codice)
# ---------------------------------------------------------------------------
def heuristic_action(board, current_direction):
    """
    board: np.array 2D (una singola board, non l'intero batch)
    current_direction: ultima azione presa su questa board (per calcolare il "dietro-front")
    return: azione scelta (int)
    """
    UP, RIGHT, DOWN, LEFT = BaseEnvironment.UP, BaseEnvironment.RIGHT, BaseEnvironment.DOWN, BaseEnvironment.LEFT
    HEAD, BODY, FRUIT, WALL = BaseEnvironment.HEAD, BaseEnvironment.BODY, BaseEnvironment.FRUIT, BaseEnvironment.WALL

    opposite = {UP: DOWN, DOWN: UP, RIGHT: LEFT, LEFT: RIGHT}

    head_pos = np.argwhere(board == HEAD)[0]           # (riga, colonna)
    fruit_candidates = np.argwhere(board == FRUIT)
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

    all_actions = [UP, RIGHT, DOWN, LEFT]

    # --- elimina dietro-front ---
    candidate_actions = [a for a in all_actions if a != opposite.get(current_direction, -1)]

    # --- elimina azioni muro ---
    valid_actions = []
    for a in candidate_actions:
        r, c = next_pos(a)
        in_bounds = 0 <= r < board.shape[0] and 0 <= c < board.shape[1]
        if in_bounds and board[r, c] != WALL:
            valid_actions.append(a)

    if not valid_actions:
        # nessuna via d'uscita: si torna indietro come ultima spiaggia
        valid_actions = [opposite.get(current_direction, UP)]

    # --- esistono azioni senza corpo? ---
    no_body_actions = [a for a in valid_actions if board[next_pos(a)] != BODY]
    candidates = no_body_actions if no_body_actions else valid_actions

    # --- scegli quella che avvicina al frutto ---
    if fruit_pos is not None:
        distances = {
            a: abs(next_pos(a)[0] - fruit_pos[0]) + abs(next_pos(a)[1] - fruit_pos[1])
            for a in candidates
        }
        min_dist = min(distances.values())
        best_actions = [a for a in candidates if distances[a] == min_dist]
    else:
        best_actions = candidates

    # --- parità -> random ---
    return int(np.random.choice(best_actions))


def heuristic_actions_batch(boards, current_directions):
    """Applica l'euristica a tutte le board di un batch."""
    return np.array([heuristic_action(boards[i], current_directions[i]) for i in range(len(boards))])

TRAP_PENALTY_WEIGHT = 0.2   # da tarare (vedi nota sotto)

def trap_penalty(board, weight=TRAP_PENALTY_WEIGHT):
    """Penalità (<= 0) se dalla testa sono raggiungibili meno celle libere
    di quante servirebbero per contenere il corpo."""
    rows, cols = board.shape
    head = tuple(int(x) for x in np.argwhere(board == BaseEnvironment.HEAD)[0])
    body_len = int(np.sum(board == BaseEnvironment.BODY))

    # muro e corpo bloccano; vuoto e frutto sono attraversabili
    walkable = (board == BaseEnvironment.EMPTY) | (board == BaseEnvironment.FRUIT)

    visited = {head}
    queue = deque([head])
    reachable = 0
    while queue:
        r, c = queue.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if (0 <= nr < rows and 0 <= nc < cols
                    and (nr, nc) not in visited and walkable[nr, nc]):
                visited.add((nr, nc))
                queue.append((nr, nc))
                reachable += 1

    needed = body_len + 1
    if reachable >= needed:
        return 0.0
    return -weight * (1.0 - reachable / needed)   # più è stretto lo spazio, più penalizza


# ---------------------------------------------------------------------------
# 1b) REWARD SHAPING DELL'AUTO-COLLISIONE (vedi ERRORI_RISCONTRATI.md, E1)
# ---------------------------------------------------------------------------
# Reward originale dell'ambiente: +0.5 per frutto, -0.1 per segmento perso.
# Perdere un segmento costava meno di quanto valeva guadagnarlo, quindi il
# ciclo "mangia e tagliati la coda" rendeva +0.4 per segmento (reward hacking).
# Nuova penalita': -(FISSA + PER_SEGMENTO * segmenti_persi), con
# PER_SEGMENTO >= FRUIT_REWARD, cosi' ogni ciclo mangia-e-tagliati e'
# strettamente svantaggioso. L'ambiente NON viene modificato: il reward
# viene ricalcolato qui, nello script di training.
COLLISION_FIXED_PENALTY = 0.5
COLLISION_SEGMENT_PENALTY = 0.5

# Costo di ogni passo (vedi E3, correzione B): toglie alla policy passiva
# ("girovagare senza mangiare") il suo valore zero. Girovagare per sempre
# vale STEP_PENALTY / (1 - gamma) = -0.01 / 0.05 = -0.2.
STEP_PENALTY = -0.01


def body_lengths(env):
    """Numero di segmenti di corpo (testa esclusa) per ogni board."""
    return np.array([len(b) for b in env.bodies], dtype=float)


def segments_lost_in_step(env, len_before, env_rewards):
    """
    Segmenti persi per auto-collisione in questo passo, ricavati dallo stato
    dell'ambiente (non dal valore del reward, vedi E2):
        len_dopo = len_prima + frutto_mangiato - segmenti_persi
    Le board vinte (WIN_REWARD) vengono resettate dall'ambiente e il corpo
    azzerato: non e' una collisione, quindi le escludiamo.
    """
    ate_fruit = np.isclose(env_rewards, env.FRUIT_REWARD).astype(float)
    lost = len_before + ate_fruit - body_lengths(env)
    lost[np.isclose(env_rewards, env.WIN_REWARD)] = 0.0
    return np.maximum(lost, 0.0)


def shaped_rewards(env_rewards, segments_lost):
    """Sostituisce il reward di auto-collisione con la penalita' corretta
    e aggiunge il costo di ogni passo."""
    rewards = env_rewards.copy()
    collided = segments_lost > 0
    rewards[collided] = -(COLLISION_FIXED_PENALTY
                          + COLLISION_SEGMENT_PENALTY * segments_lost[collided])
    return rewards + STEP_PENALTY


# ---------------------------------------------------------------------------
# 2) Q-NETWORK (approssimatore di Q(s,a))
# ---------------------------------------------------------------------------
def build_q_network(board_size, n_actions=4):
    inputs = tf.keras.Input(shape=(board_size, board_size, 4))  # 4 canali: EMPTY,FRUIT,BODY,HEAD (to_categorical -1)
    x = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
    x = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    outputs = tf.keras.layers.Dense(n_actions, activation="linear")(x)
    return tf.keras.Model(inputs, outputs)


# ---------------------------------------------------------------------------
# 3) Politica epsilon-greedy con euristica al posto dell'esplorazione casuale
# ---------------------------------------------------------------------------
OPPOSITE_DIRECTION = {
    BaseEnvironment.UP: BaseEnvironment.DOWN,
    BaseEnvironment.DOWN: BaseEnvironment.UP,
    BaseEnvironment.RIGHT: BaseEnvironment.LEFT,
    BaseEnvironment.LEFT: BaseEnvironment.RIGHT,
}


ACTION_FROM_DELTA = {
    (1, 0): BaseEnvironment.UP,      # UP -> riga+1 (vedi move())
    (-1, 0): BaseEnvironment.DOWN,
    (0, 1): BaseEnvironment.RIGHT,
    (0, -1): BaseEnvironment.LEFT,
}


def reverse_actions(env, last_directions=None):
    """
    Per ogni board, l'azione di dietro-front; -1 se non e' definita.
    - Se il serpente ha un corpo: l'azione che porterebbe la testa sul collo
      (il primo segmento del corpo).
    - Se il serpente e' lungo 1 (corpo vuoto, vedi E7): l'opposto dell'ultima
      direzione di movimento (last_directions), come nel vero Snake, dove il
      dietro-front e' vietato anche senza corpo. Senza questa regola il
      serpente lungo 1 poteva oscillare fra due celle all'infinito.
      All'inizio della partita (direzione NONE) non c'e' dietro-front.
    Si usa la posizione del collo e non l'ultima azione presa, cosi' il
    risultato e' corretto anche dopo un urto contro il muro o all'inizio di
    un episodio. Dopo un urto contro il muro l'ambiente inserisce in testa al
    corpo la posizione della testa stessa: quei duplicati vengono saltati.
    """
    heads = np.argwhere(env.boards == BaseEnvironment.HEAD)[:, 1:]
    rev = np.full(env.n_boards, -1)
    for i, body in enumerate(env.bodies):
        for seg in body:
            delta = (int(seg[0] - heads[i][0]), int(seg[1] - heads[i][1]))
            if delta != (0, 0):
                rev[i] = ACTION_FROM_DELTA.get(delta, -1)
                break
    if last_directions is not None:
        for i in np.where(rev < 0)[0]:
            rev[i] = OPPOSITE_DIRECTION.get(int(last_directions[i]), -1)
    return rev


ACTION_DELTAS = np.array([(1, 0), (0, 1), (-1, 0), (0, -1)])   # UP, RIGHT, DOWN, LEFT


def wall_actions_mask(env):
    """
    Maschera booleana (n_boards, 4): True se l'azione porta la testa contro
    il muro. Il muro e' il bordo della board, quindi la testa e' sempre
    all'interno e la cella successiva esiste sempre.
    """
    heads = np.argwhere(env.boards == BaseEnvironment.HEAD)          # (board, riga, col)
    next_cells = heads[:, None, 1:] + ACTION_DELTAS[None, :, :]       # (n_boards, 4, 2)
    b = heads[:, 0][:, None]
    return env.boards[b, next_cells[..., 0], next_cells[..., 1]] == BaseEnvironment.WALL


def choose_actions(env, q_network, current_directions, epsilon):
    # La scelta greedy esclude le mosse che violano le regole del gioco:
    # - correzione A (E3): il dietro-front, mossa impossibile nel vero Snake
    #   e causa principale delle auto-collisioni da 1 segmento;
    # - correzione E4: le mosse contro il muro, che lasciano lo stato quasi
    #   invariato e facevano entrare l'agente in loop.
    # Restano sempre almeno una mossa valida (anche in un angolo con il collo
    # dietro). Le mosse sul corpo NON sono mascherate: evitare le trappole e'
    # cio' che la rete deve imparare.
    states = env.to_state()
    q_values = q_network(states).numpy()
    q_values[wall_actions_mask(env)] = -np.inf
    rev = reverse_actions(env, current_directions)
    has_neck = rev >= 0
    q_values[np.where(has_neck)[0], rev[has_neck]] = -np.inf
    actions = np.argmax(q_values, axis=-1)

    explore = np.random.rand(env.n_boards) < epsilon
    if explore.any():
        idx = np.where(explore)[0]
        actions[idx] = heuristic_actions_batch(env.boards[idx], current_directions[idx])

    return actions.astype(np.int32)


# ---------------------------------------------------------------------------
# 4) VISUALIZZAZIONE
# ---------------------------------------------------------------------------
# Colori RGB associati a ogni valore della board
BOARD_COLORS = {
    BaseEnvironment.WALL: (0.05, 0.05, 0.05),   # nero
    BaseEnvironment.EMPTY: (0.95, 0.95, 0.95),  # bianco/grigio chiaro
    BaseEnvironment.FRUIT: (0.85, 0.10, 0.10),  # rosso
    BaseEnvironment.BODY: (0.10, 0.50, 0.10),   # verde scuro
    BaseEnvironment.HEAD: (0.10, 0.90, 0.10),   # verde chiaro
}


def board_to_rgb(board):
    """Converte una board (matrice di interi) in un'immagine RGB."""
    rgb = np.zeros((*board.shape, 3))
    for value, color in BOARD_COLORS.items():
        rgb[board == value] = color
    return rgb


def record_episode(env, q_network, board_index=0, max_steps=200, epsilon=0.0):
    """
    Fa giocare UNA board dell'ambiente (indice board_index) seguendo la policy
    appresa (greedy rispetto a q_network; con epsilon > 0 l'esplorazione usa
    ancora l'euristica, come in training) e registra ogni frame.

    Nota: env.move() agisce su TUTTE le board contemporaneamente, quindi anche
    se vogliamo registrarne solo una, l'ambiente deve comunque avere n_boards
    board attive: per la visualizzazione conviene creare un env con n_boards=1
    dedicato (vedi visualize_policy sotto), così non si interferisce col
    training in corso.
    """
    frames = [env.boards[board_index].copy()]
    current_direction = BaseEnvironment.NONE
    total_reward = 0.0

    for _ in range(max_steps):
        actions = choose_actions(env, q_network,
                                  np.full(env.n_boards, current_direction), epsilon)
        rewards = env.move(actions)
        current_direction = actions[board_index]
        total_reward += float(rewards[board_index])
        frames.append(env.boards[board_index].copy())

    return frames, total_reward


def save_episode_gif(frames, path="snake_episode.gif", interval_ms=200):
    """
    Salva la sequenza di frame come GIF animata.
    Funziona sia in locale (Anaconda) sia su Colab: non serve una finestra
    grafica interattiva, alla fine trovi semplicemente il file su disco
    (in Colab: pannello file a sinistra, tasto destro -> Download).
    Richiede 'pillow' installato (pip install pillow).
    """
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(board_to_rgb(frames[0]))
    ax.set_xticks([])
    ax.set_yticks([])
    title = ax.set_title("step 0")

    def update(i):
        im.set_data(board_to_rgb(frames[i]))
        title.set_text(f"step {i}")
        return [im, title]

    ani = animation.FuncAnimation(fig, update, frames=len(frames), interval=interval_ms)
    ani.save(path, writer="pillow")
    plt.close(fig)
    print(f"GIF salvata in: {path}")


def head_positions(env):
    return np.argwhere(env.boards == BaseEnvironment.HEAD)[:, 1:]


def evaluate_policy(q_network, board_size, n_games=20, max_steps=500, seed=12345,
                    stuck_window=100):
    """
    Valutazione separata dal training (vedi E4): la policy greedy (epsilon=0)
    gioca n_games partite nuove, in parallelo, per max_steps passi.
    Il seed e' fisso, quindi a ogni valutazione le partite partono dalle
    stesse posizioni iniziali e i risultati dei checkpoint sono confrontabili.
    Lo stato del generatore casuale del training viene salvato e ripristinato,
    cosi' la valutazione non altera il training.
    Ritorna le metriche medie per partita e, per i frutti, anche minimo,
    massimo e numero di partite problematiche, cosi' una media buona non
    nasconde partite rimaste bloccate in loop (vedi E5, limiti residui):
      partite_zero_frutti: partite senza nemmeno un frutto;
      partite_bloccate:    partite senza frutti negli ultimi stuck_window
                           passi (tipicamente un loop).
    """
    rng_state = np.random.get_state()
    np.random.seed(seed)
    try:
        env = OriginalSnakeEnvironment(n_boards=n_games, board_size=board_size)
        directions = np.full(n_games, BaseEnvironment.NONE)
        fruits = np.zeros(n_games)
        collisions = np.zeros(n_games)
        lost_total = np.zeros(n_games)
        wall_hits = np.zeros(n_games)
        max_len = body_lengths(env) + 1
        last_fruit_step = np.full(n_games, -1)
        for step in range(max_steps):
            actions = choose_actions(env, q_network, directions, epsilon=0.0)
            len_before = body_lengths(env)
            heads_before = head_positions(env)
            env_rewards = env.move(actions).numpy().reshape(-1)
            lost = segments_lost_in_step(env, len_before, env_rewards)
            ate = np.isclose(env_rewards, env.FRUIT_REWARD)
            fruits += ate
            last_fruit_step[ate] = step
            collisions += lost > 0
            lost_total += lost
            wall_hits += np.all(head_positions(env) == heads_before, axis=1)
            max_len = np.maximum(max_len, body_lengths(env) + 1)
            directions = actions
    finally:
        np.random.set_state(rng_state)

    # se la partita e' piu' corta della finestra, "bloccata" = nessun frutto
    stuck = last_fruit_step < max(max_steps - stuck_window, 0)
    return {
        "n_partite": n_games,
        "frutti": float(np.mean(fruits)),
        "frutti_min": float(np.min(fruits)),
        "frutti_max": float(np.max(fruits)),
        "partite_zero_frutti": int(np.sum(fruits == 0)),
        "partite_bloccate": int(np.sum(stuck)),
        "lunghezza_max": float(np.mean(max_len)),
        "lunghezza_finale": float(np.mean(body_lengths(env) + 1)),
        "self_collision": float(np.mean(collisions)),
        "segmenti_persi": float(np.mean(lost_total)),
        "urti_muro": float(np.mean(wall_hits)),
    }


def visualize_policy(q_network, board_size, epsilon=0.0, max_steps=500, path="snake_episode.gif"):
    """
    Crea un piccolo ambiente dedicato (1 board), ci fa giocare la policy
    corrente e salva il risultato come GIF.
    """
    eval_env = OriginalSnakeEnvironment(n_boards=1, board_size=board_size)
    frames, total_reward = record_episode(eval_env, q_network, board_index=0,
                                           max_steps=max_steps, epsilon=epsilon)
    save_episode_gif(frames, path=path)
    print(f"Reward totale nell'episodio registrato: {total_reward:.2f}")
    return frames, total_reward


# ---------------------------------------------------------------------------
# 5) CICLO SARSA (on-policy TD control)
# ---------------------------------------------------------------------------
def sarsa_train(env, q_network, optimizer, n_episodes=200, max_steps=500,
                 epsilon_start=0.2, epsilon_end=0.02, gamma=0.95, visualize_every=None,
                 output_dir=".", csv_path="training_history_sarsa.csv", csv_every=20,
                 eval_every=10, eval_csv_path="eval_history_sarsa.csv",
                 reset_each_episode=True):
    """
    visualize_every: se impostato (es. 100), ogni N episodi salva una GIF
    dell'episodio corrente giocato dalla policy appresa, così puoi controllare
    i progressi senza fermare il training.

    reset_each_episode: se True (correzione E5) ogni episodio parte da board
    nuove (serpenti lunghi 1), cosi' il training vede sempre anche l'inizio
    della partita. Se False le partite proseguono tra un episodio e l'altro
    (comportamento originale).

    eval_every: ogni N episodi (e all'ultimo) valuta la policy greedy su
    partite nuove a seed fisso (evaluate_policy) e salva i risultati in
    eval_csv_path.

    Ritorna `history` (metriche di training per episodio) ed `eval_history`
    (metriche delle valutazioni periodiche).
    """
    history = {
        "reward_medio": [],       # reward medio per board, sommato sull'episodio
        "frutti_mangiati": [],    # numero medio di frutti mangiati per board
        "self_collision": [],    # numero medio di volte che si e' mangiato da solo
        "segmenti_persi": [],     # segmenti di coda persi in totale nell'episodio
        "lunghezza_media": [],    # lunghezza media del serpente (testa inclusa) nell'episodio
        "urti_muro": [],          # numero medio di urti contro il muro
        "loss_media": [],         # loss media della rete nell'episodio
    }
    eval_history = {"episodio": []}   # le altre colonne arrivano da evaluate_policy

    for episode in range(n_episodes):
        if reset_each_episode:
            # board nuove: stessa classe e stesse dimensioni dell'ambiente
            env.__init__(n_boards=env.n_boards, board_size=env.board_size)

        epsilon = epsilon_start - (epsilon_start - epsilon_end) * (episode / max(n_episodes - 1, 1))
        current_directions = np.full(env.n_boards, BaseEnvironment.NONE)
        actions = choose_actions(env, q_network, current_directions, epsilon)

        ep_segments_lost = np.zeros(env.n_boards)
        ep_reward = np.zeros(env.n_boards)
        ep_fruits = np.zeros(env.n_boards)
        ep_self_collision = np.zeros(env.n_boards)
        ep_length_sum = np.zeros(env.n_boards)
        ep_wall_hits = np.zeros(env.n_boards)
        ep_losses = []

        for step in range(max_steps):
            state = env.to_state()
            len_before = body_lengths(env)
            heads_before = head_positions(env)

            env_rewards = env.move(actions).numpy().reshape(-1)   # esegue A, ottiene R (originale)

            # metriche ricavate dallo stato dell'ambiente, non dal valore del reward (E2)
            segments_lost_step = segments_lost_in_step(env, len_before, env_rewards)
            ep_fruits += np.isclose(env_rewards, env.FRUIT_REWARD).astype(float)
            ep_self_collision += (segments_lost_step > 0).astype(float)
            ep_segments_lost += segments_lost_step
            ep_length_sum += body_lengths(env) + 1.0          # +1 = testa
            ep_wall_hits += np.all(head_positions(env) == heads_before, axis=1)   # testa ferma = urto

            # reward shaping dell'auto-collisione (E1)
            rewards_np = shaped_rewards(env_rewards, segments_lost_step)
            ep_reward += rewards_np
            rewards = tf.constant(rewards_np.reshape(-1, 1), dtype=tf.float32)

            #penalties = np.array([trap_penalty(env.boards[i]) for i in range(env.n_boards)],
            #                     dtype=np.float32)
            #rewards = rewards + tf.reshape(tf.constant(penalties), (-1, 1))

            next_state = env.to_state()                       # osserva S'
            next_actions = choose_actions(env, q_network, actions, epsilon)  # sceglie A' (on-policy)

            with tf.GradientTape() as tape:
                q_values = q_network(state)
                q_sa = tf.gather(q_values, actions, axis=1, batch_dims=1)

                next_q_values = q_network(next_state)
                next_q_sa = tf.gather(next_q_values, next_actions, axis=1, batch_dims=1)
                next_q_sa = tf.stop_gradient(next_q_sa)

                target = tf.squeeze(rewards) + gamma * next_q_sa
                loss = tf.reduce_mean(tf.square(target - q_sa))

            grads = tape.gradient(loss, q_network.trainable_variables)
            optimizer.apply_gradients(zip(grads, q_network.trainable_variables))
            ep_losses.append(float(loss.numpy()))

            current_directions = actions
            actions = next_actions

        history["reward_medio"].append(float(np.mean(ep_reward)))
        history["frutti_mangiati"].append(float(np.mean(ep_fruits)))
        history["segmenti_persi"].append(float(np.mean(ep_segments_lost)))
        history["self_collision"].append(float(np.mean(ep_self_collision)))
        history["lunghezza_media"].append(float(np.mean(ep_length_sum / max_steps)))
        history["urti_muro"].append(float(np.mean(ep_wall_hits)))
        history["loss_media"].append(float(np.mean(ep_losses)))

        if episode % 10 == 0:
            print(f"Episodio {episode}: reward={history['reward_medio'][-1]:.2f} "
                  f"frutti={history['frutti_mangiati'][-1]:.1f} "
                  f"collisioni={history['self_collision'][-1]:.2f} "
                  f"segm_persi={history['segmenti_persi'][-1]:.1f} "
                  f"lunghezza={history['lunghezza_media'][-1]:.2f} "
                  f"muro={history['urti_muro'][-1]:.1f} "
                  f"loss={history['loss_media'][-1]:.4f}")

        if eval_every and (episode % eval_every == 0 or episode == n_episodes - 1):
            ev = evaluate_policy(q_network, env.board_size, max_steps=max_steps)
            eval_history["episodio"].append(episode)
            for k, v in ev.items():
                eval_history.setdefault(k, []).append(v)
            print(f"  [valutazione ep {episode}] " + " ".join(f"{k}={v:.2f}" for k, v in ev.items()))
            save_history_csv(eval_history, path=eval_csv_path)

        if visualize_every and episode % visualize_every == 0 and episode > 0:
            visualize_policy(q_network, env.board_size,
                              path=os.path.join(output_dir, f"snake_sarsa_checkpoint_ep{episode}.gif"))

        if csv_every and episode % csv_every == 0 and episode > 0:
            save_history_csv(history, path=csv_path)

    save_history_csv(history, path=csv_path)   # salvataggio finale: include gli ultimi episodi
    return history, eval_history


def plot_training_history(history, path="training_history.png", window=10):
    """
    Grafica le 4 metriche raccolte durante il training, con una media mobile
    (window episodi) per rendere leggibile l'andamento nonostante il rumore
    episodio-per-episodio.
    """
    def smooth(values, window):
        values = np.array(values)
        if len(values) < window:
            return values
        kernel = np.ones(window) / window
        return np.convolve(values, kernel, mode="valid")

    fig, axes = plt.subplots(2, 4, figsize=(19, 7))
    titles = [
        ("reward_medio", "Reward medio per episodio"),
        ("frutti_mangiati", "Frutti mangiati per episodio"),
        ("lunghezza_media", "Lunghezza media del serpente"),
        ("self_collision", "Auto-collisioni per episodio"),
        ("segmenti_persi", "Segmenti persi per episodio"),
        ("urti_muro", "Urti contro il muro per episodio"),
        ("loss_media", "Loss media della rete"),
    ]
    axes.flat[-1].axis("off")

    for ax, (key, title) in zip(axes.flat, titles):
        raw = history[key]
        ax.plot(raw, alpha=0.3, label="grezzo")
        ax.plot(range(window - 1, len(raw)), smooth(raw, window), label=f"media mobile ({window})")
        ax.set_title(title)
        ax.set_xlabel("episodio")
        ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"Grafico salvato in: {path}")



def plot_eval_history(eval_history, path="eval_history.png"):
    """Grafica le metriche della valutazione periodica (policy greedy, seed fisso)."""
    ep = eval_history["episodio"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 7))

    ax = axes[0, 0]
    ax.fill_between(ep, eval_history["frutti_min"], eval_history["frutti_max"],
                    alpha=0.25, label="min-max sulle partite")
    ax.plot(ep, eval_history["frutti"], marker="o", label="media")
    ax.set_title("Frutti mangiati (valutazione)")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    ax.plot(ep, eval_history["partite_zero_frutti"], marker="o", label="zero frutti")
    ax.plot(ep, eval_history["partite_bloccate"], marker="s", label="bloccate (no frutti ultimi 100 passi)")
    ax.set_title(f"Partite problematiche (su {eval_history['n_partite'][-1]})")
    ax.legend(fontsize=8)

    ax = axes[0, 2]
    ax.plot(ep, eval_history["lunghezza_max"], marker="o", label="massima")
    ax.plot(ep, eval_history["lunghezza_finale"], marker="s", label="a fine partita")
    ax.set_title("Lunghezza del serpente")
    ax.legend(fontsize=8)

    for ax, (key, title) in zip(axes[1], [("self_collision", "Auto-collisioni"),
                                          ("segmenti_persi", "Segmenti persi"),
                                          ("urti_muro", "Urti contro il muro")]):
        ax.plot(ep, eval_history[key], marker="o")
        ax.set_title(title)

    for ax in axes.flat:
        ax.set_xlabel("episodio di training")
    fig.suptitle(f"Valutazione greedy: {eval_history['n_partite'][-1]} partite nuove a seed fisso, 500 passi")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"Grafico salvato in: {path}")


def save_history_csv(history, path="training_history.csv"):
    """Salva un dizionario {metrica: lista di valori} come CSV, una riga per
    episodio. Se manca la colonna "episodio" viene aggiunta come indice."""
    keys = list(history.keys())
    n_rows = len(history[keys[0]])
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        if "episodio" in history:
            writer.writerow(keys)
            for i in range(n_rows):
                writer.writerow([history[k][i] for k in keys])
        else:
            writer.writerow(["episodio"] + keys)
            for i in range(n_rows):
                writer.writerow([i] + [history[k][i] for k in keys])
    print(f"CSV salvato in: {path}")



if __name__ == "__main__":
    BOARD_SIZE = 10
    N_BOARDS = 32
    N_EPISODES = 60            # run di prova del fix E7 (dietro-front a lunghezza 1)
    VISUALIZE_EVERY = 20

    # ogni esperimento scrive in una cartella separata, cosi' i risultati
    # precedenti restano intatti per il confronto nel report:
    #   risultati_reward_shaping/  -> solo E1 (collasso passivo, E3)
    #   risultati_mask_steppenalty -> E1 + maschera dietro-front + costo per passo (E4)
    #   risultati_mask_muro        -> + maschera muro + valutazione separata (E5)
    #   risultati_reset            -> + reset delle board a ogni episodio (60 ep)
    #   risultati_200ep            -> stessa configurazione, 200 episodi,
    #                                 valutazione con min/max e partite bloccate
    #   risultati_fix_dietrofront  -> + dietro-front vietato anche a lunghezza 1,
    #                                 valutazione su 20 partite (60 ep)
    OUTPUT_DIR = "risultati_fix_dietrofront"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    env = OriginalSnakeEnvironment(n_boards=N_BOARDS, board_size=BOARD_SIZE)
    q_network = build_q_network(BOARD_SIZE)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

    history, eval_history = sarsa_train(env, q_network, optimizer, n_episodes=N_EPISODES, epsilon_start=0.2,
                           epsilon_end=0.02, visualize_every=VISUALIZE_EVERY, csv_every=20,
                           output_dir=OUTPUT_DIR,
                           csv_path=os.path.join(OUTPUT_DIR, "training_history_sarsa.csv"),
                           eval_every=10,
                           eval_csv_path=os.path.join(OUTPUT_DIR, "eval_history_sarsa.csv"))

    plot_training_history(history, path=os.path.join(OUTPUT_DIR, "training_history.png"))
    plot_eval_history(eval_history, path=os.path.join(OUTPUT_DIR, "eval_history.png"))

    # episodio finale con la policy addestrata (epsilon=0 -> nessuna esplorazione)
    print("\nRegistro l'episodio finale con la policy addestrata...")
    visualize_policy(q_network, BOARD_SIZE, epsilon=0.0,
                     path=os.path.join(OUTPUT_DIR, "snake_final.gif"))