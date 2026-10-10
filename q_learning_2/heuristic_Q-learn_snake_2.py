"""
Euristica di scelta azione per Snake + integrazione in una struttura SARSA.

Fa riferimento a `environments_fully_observable.py`:
- board: matrice 2D con valori HEAD=4, BODY=3, FRUIT=2, EMPTY=1, WALL=0
- azioni: UP=0, RIGHT=1, DOWN=2, LEFT=3, NONE=4
- offset applicati in move(): UP -> riga+1, DOWN -> riga-1,
                              RIGHT -> colonna+1, LEFT -> colonna-1
"""

import csv
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
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


def choose_actions(env, q_network, current_directions, epsilon):
    # NB: nessuna maschera sul dietro-front nella scelta greedy: la rete deve
    # imparare da sola, tramite ATE_HIMSELF_REWARD, a evitarlo.
    states = env.to_state()
    q_values = q_network(states).numpy()
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
def sarsa_train(env, q_network, optimizer, n_episodes=1000, max_steps=200,
                 epsilon=0.1, gamma=0.95, visualize_every=None):
    """
    visualize_every: se impostato (es. 100), ogni N episodi salva una GIF
    dell'episodio corrente giocato dalla policy appresa, così puoi controllare
    i progressi senza fermare il training.

    Ritorna un dizionario `history` con le metriche per episodio, utile per
    capire se l'agente sta effettivamente migliorando (vedi plot_training_history).
    """
    history = {
        "reward_medio": [],       # reward medio per board, sommato sull'episodio
        "frutti_mangiati": [],    # numero medio di frutti mangiati per board
        "self_collision": [],     # numero medio di volte che si e' mangiato da solo
        "loss_media": [],         # loss media della rete nell'episodio
    }

    for episode in range(n_episodes):
        current_directions = np.full(env.n_boards, BaseEnvironment.NONE)
        actions = choose_actions(env, q_network, current_directions, epsilon)

        ep_reward = np.zeros(env.n_boards)
        ep_fruits = np.zeros(env.n_boards)
        ep_self_collision = np.zeros(env.n_boards)
        ep_losses = []

        for step in range(max_steps):
            state = env.to_state()

            rewards = env.move(actions)                      # esegue A, ottiene R
            rewards_np = rewards.numpy().reshape(-1)

            ep_reward += rewards_np
            ep_fruits += (rewards_np == env.FRUIT_REWARD).astype(float)
            is_self_collision = (rewards_np < 0) & (~np.isclose(rewards_np, env.HIT_WALL_REWARD))
            ep_self_collision += is_self_collision.astype(float)

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
        history["self_collision"].append(float(np.mean(ep_self_collision)))
        history["loss_media"].append(float(np.mean(ep_losses)))

        if episode % 50 == 0:
            print(f"Episodio {episode} completato")

        if visualize_every and episode % visualize_every == 0 and episode > 0:
            visualize_policy(q_network, env.board_size,
                              path=f"snake_checkpoint_ep{episode}.gif")

    return history


# ---------------------------------------------------------------------------
# 5bis) CICLO Q-LEARNING (off-policy TD control) — per confronto con SARSA
# ---------------------------------------------------------------------------
def q_learning_train(env, q_network, optimizer, n_episodes=200, max_steps=500,
                      epsilon=0.1, gamma=0.95,csv_every=20, visualize_every=None,
                      csv_path="training_history_qlearning.csv"):
    """
    Identica a sarsa_train, tranne che nel calcolo del target: qui si usa
    max_a' Q(s', a') (il valore della MIGLIOR azione possibile nello stato
    successivo), non quello dell'azione che la policy sceglierà davvero.

    Questo e' l'unico cambiamento concettuale che distingue Q-learning
    (off-policy) da SARSA (on-policy): l'agente si comporta comunque in modo
    epsilon-greedy (esplora anche mosse subottimali), ma IMPARA come se
    seguisse sempre la policy ottimale, ignorando il rischio dell'esplorazione.
    """
    history = {
        "reward_medio": [],
        "frutti_mangiati": [],
        "self_collision": [],
        "segmenti_persi": [],   # <-- nuova
        "loss_media": [],
    }

    for episode in range(n_episodes):
        
        current_directions = np.full(env.n_boards, BaseEnvironment.NONE)
        actions = choose_actions(env, q_network, current_directions, epsilon)

        ep_segments_lost = np.zeros(env.n_boards)
        ep_reward = np.zeros(env.n_boards)
        ep_fruits = np.zeros(env.n_boards)
        ep_self_collision = np.zeros(env.n_boards)
        ep_losses = []

        for step in range(max_steps):
            state = env.to_state()

            rewards = env.move(actions)
            rewards_np = rewards.numpy().reshape(-1)

            ep_reward += rewards_np
            ep_fruits += (rewards_np == env.FRUIT_REWARD).astype(float)
            is_self_collision = (rewards_np < 0) & (~np.isclose(rewards_np, env.HIT_WALL_REWARD))
            ep_self_collision += is_self_collision.astype(float)

            segments_lost_step = np.zeros_like(rewards_np)
            segments_lost_step[is_self_collision] = rewards_np[is_self_collision] / env.ATE_HIMSELF_REWARD
            ep_segments_lost += segments_lost_step

            next_state = env.to_state()
            # l'azione EFFETTIVA per il prossimo step resta epsilon-greedy
            # (serve comunque per continuare a giocare/esplorare l'ambiente)
            next_actions = choose_actions(env, q_network, actions, epsilon)

            with tf.GradientTape() as tape:
                q_values = q_network(state)
                q_sa = tf.gather(q_values, actions, axis=1, batch_dims=1)

                next_q_values = q_network(next_state)
                # --- UNICA DIFFERENZA RISPETTO A SARSA ---
                # invece di Q(s', a'_scelta_dalla_policy), si usa il massimo
                # possibile: max_a' Q(s', a')  ->  target "ottimistico"
                next_q_max = tf.reduce_max(next_q_values, axis=1)
                next_q_max = tf.stop_gradient(next_q_max)

                target = tf.squeeze(rewards) + gamma * next_q_max
                loss = tf.reduce_mean(tf.square(target - q_sa))

            grads = tape.gradient(loss, q_network.trainable_variables)
            optimizer.apply_gradients(zip(grads, q_network.trainable_variables))
            ep_losses.append(float(loss.numpy()))

            current_directions = actions
            actions = next_actions

        history["reward_medio"].append(float(np.mean(ep_reward)))
        history["frutti_mangiati"].append(float(np.mean(ep_fruits)))
        history["segmenti_persi"].append(float(np.mean(ep_segments_lost)))   # <-- nuova
        history["self_collision"].append(float(np.mean(ep_self_collision)))
        history["loss_media"].append(float(np.mean(ep_losses)))

        if episode % 50 == 0:
            print(f"Episodio {episode} completato")

        if visualize_every and episode % visualize_every == 0 and episode > 0:
            visualize_policy(q_network, env.board_size,
                              path=f"snake_qlearning_checkpoint_ep{episode}.gif")

        if csv_every and episode % csv_every == 0 and episode > 0:
            save_history_csv(history, path=csv_path)

    return history


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

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    titles = [
        ("reward_medio", "Reward medio per episodio"),
        ("frutti_mangiati", "Frutti mangiati per episodio"),
        ("self_collision", "Auto-collisioni per episodio"),
        ("loss_media", "Loss media della rete"),
    ]

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


def save_history_csv(history, path="training_history.csv"):
    n_episodes = len(history["reward_medio"])
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episodio", "reward_medio", "frutti_mangiati", "self_collision",
                          "segmenti_persi", "loss_media"])
        for i in range(n_episodes):
            writer.writerow([
                i,
                history["reward_medio"][i],
                history["frutti_mangiati"][i],
                history["self_collision"][i],
                history["segmenti_persi"][i],
                history["loss_media"][i],
            ])
    print(f"CSV salvato in: {path}")



if __name__ == "__main__":
    BOARD_SIZE = 10
    N_BOARDS = 32
    N_EPISODES = 200

    # --- SARSA ---
    #env_sarsa = OriginalSnakeEnvironment(n_boards=N_BOARDS, board_size=BOARD_SIZE)
    #q_network_sarsa = build_q_network(BOARD_SIZE)
    #optimizer_sarsa = tf.keras.optimizers.Adam(learning_rate=1e-3)

    #print("=== Training SARSA ===")
    #history_sarsa = sarsa_train(env_sarsa, q_network_sarsa, optimizer_sarsa,
    #                             n_episodes=N_EPISODES, epsilon=0.15,
    #                             visualize_every=30)
    #plot_training_history(history_sarsa, path="training_history_sarsa.png")
    #visualize_policy(q_network_sarsa, BOARD_SIZE, epsilon=0.0, path="snake_final_sarsa.gif")

    # --- Q-LEARNING (stessa architettura, stessi iperparametri, per confronto equo) ---
    env_qlearning = OriginalSnakeEnvironment(n_boards=N_BOARDS, board_size=BOARD_SIZE)
    q_network_qlearning = build_q_network(BOARD_SIZE)
    optimizer_qlearning = tf.keras.optimizers.Adam(learning_rate=1e-3)

    print("\n=== Training Q-learning ===")
    history_qlearning = q_learning_train(env_qlearning, q_network_qlearning, optimizer_qlearning,
                                          n_episodes=N_EPISODES, epsilon=0.15, csv_every=20,
                                          visualize_every=50)
    plot_training_history(history_qlearning, path="training_history_qlearning.png")
    visualize_policy(q_network_qlearning, BOARD_SIZE, epsilon=0.0, path="snake_final_qlearning.gif")
