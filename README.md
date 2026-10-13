SNAKE CON REINFORCEMENT LEARNING: SARSA vs Q-LEARNING
======================================================

![snake](q_learining_2/snake_final_qlearning.gif)

DESCRIZIONE DEL PROGETTO
------------------------
Il progetto addestra un agente a giocare a Snake su una griglia 10x10 con
algoritmi di Reinforcement Learning. L'agente osserva l'intera board
(ambiente completamente osservabile) e a ogni passo sceglie una tra quattro
azioni: UP, RIGHT, DOWN, LEFT.

L'ambiente simula in parallelo piu' board (default: 32) in un unico
passo di calcolo, cosi' ogni aggiornamento della rete vede 32 partite
contemporaneamente.


OBIETTIVO
---------
1. Mangiare il maggior numero di frutti possibile.
2. Evitare le auto-collisioni (la testa che finisce sul proprio corpo).
3. Confrontare due algoritmi di TD control, SARSA (on-policy) e
   Q-learning (off-policy), a parita' di ambiente, rete e iperparametri,
   per capire quale impari meglio a non mangiarsi da solo.


AMBIENTE (environments_fully_observable.py)
-------------------------------------------
Valori delle celle:  WALL=0, EMPTY=1, FRUIT=2, BODY=3, HEAD=4
Azioni:              UP=0, RIGHT=1, DOWN=2, LEFT=3, NONE=4

Regole importanti:
- Sbattere contro il muro NON termina l'episodio: la testa resta ferma,
  il corpo non viene toccato e l'agente riceve HIT_WALL_REWARD.
- Auto-collisione: la coda viene tagliata a partire dal punto di contatto
  (se la testa finisce sul primo segmento, si perde tutto il corpo).
  L'episodio NON termina.
- Le board NON vengono resettate tra un episodio e l'altro: un "episodio"
  e' una finestra di max_steps passi di una partita che prosegue.
- Il WIN_REWARD scatta solo se il serpente riempie tutta la board
  (praticamente mai nei run brevi).



RETE NEURALE
------------
CNN che approssima Q(s,a):
  input 10x10x4 (one-hot: EMPTY, FRUIT, BODY, HEAD)
  -> Conv2D(32, 3x3) -> Conv2D(64, 3x3) -> Flatten -> Dense(128) -> Dense(4)
Ottimizzatore Adam, learning rate 1e-3, loss MSE sull'errore TD.
Stessa architettura per entrambi gli algoritmi.


ESPLORAZIONE: EPSILON-GREEDY CON EURISTICA
------------------------------------------
Quando l'agente esplora (probabilita' epsilon) non sceglie a caso, ma
usa un'euristica a regole:
  1. elimina il dietro-front
  2. elimina le azioni che vanno contro il muro
  3. se esistono azioni che non finiscono sul corpo, considera solo quelle
  4. tra le candidate, scegli quella che avvicina di piu' al frutto
  5. a parita' di distanza, scegli a caso
La scelta greedy della rete NON ha maschere: deve imparare da sola, tramite
i reward, a evitare dietro-front e auto-collisioni.


I DUE APPROCCI
--------------
1) SARSA (on-policy)
   Target:  r + gamma * Q(s', a')
   dove a' e' l'azione che la policy epsilon-greedy sceglie davvero nello
   stato successivo. La stima tiene conto del rischio dell'esplorazione.
   Epsilon decresce linearmente da epsilon_start a epsilon_end.

2) Q-learning (off-policy)
   Target:  r + gamma * max_a' Q(s', a')
   Usa il valore della migliore azione possibile, indipendentemente da cosa
   fara' la policy. E' l'unica differenza concettuale rispetto a SARSA.

Per un confronto equo i due script devono avere: stesso ambiente, stessi
reward, stessi episodi/step, stesso schema di epsilon, stesso N_BOARDS e
stessa architettura.


METRICHE (per episodio, medie sulle board)
------------------------------------------
- reward_medio      reward totale accumulato
- frutti_mangiati   numero di frutti
- self_collision    numero di auto-collisioni
- segmenti_persi    segmenti di coda persi in totale (gravita' delle collisioni)
- loss_media        loss TD media della rete

Output:
- training_history*.png   grafici delle metriche con media mobile
- training_history*.csv   valori numerici, salvati ogni 20 episodi
- snake_*checkpoint_ep*.gif  partita di una board con la policy corrente
- snake_final*.gif        partita finale con epsilon = 0

Attenzione: i numeri del CSV (media su 32 board, partite continue, con
epsilon > 0) non sono confrontabili con la GIF (una sola partita da zero,
epsilon = 0).


ESECUZIONE
----------
Dipendenze: python 3.10, tensorflow (tensorflow-cpu se non hai GPU NVIDIA),
numpy, matplotlib, pillow.

  conda create -n snake python=3.10
  conda activate snake
  pip install tensorflow-cpu numpy matplotlib pillow

Lancia lo script dalla cartella in cui vuoi trovare i risultati (i path
sono relativi alla cartella di lavoro):

  python heuristic_sarsa_snake.py
  python heuristic_Q-learn_snake.py

Su Colab non installare tensorflow (e' gia' presente) e imposta il runtime
su GPU.


ESPERIMENTI E NOTE
------------------
- Penalita' di auto-collisione fissa vs proporzionale ai segmenti persi.
- trap_penalty (flood fill): penalita' sull'area libera raggiungibile.
  Sperimentale, presente solo in alcune versioni di SARSA: va applicata
  a entrambi gli algoritmi o a nessuno per un confronto valido.
- Osservazione aperta: nei run effettuati SARSA mostra molte piu'
  auto-collisioni di Q-learning (circa 7 contro 0.7 per episodio). Va
  verificato a parita' di reward shaping, iperparametri e seed prima di
  attribuire la differenza all'algoritmo.


SVILUPPI FUTURI
---------------
- Reward shaping basato sulla distanza dal frutto (potential-based).
- Ricerca con lookahead: MCTS, come baseline di pianificazione e, in
  seguito, in combinazione con la rete (stile AlphaZero).
- Experience replay e rete target per stabilizzare il training.
- Valutazione dei checkpoint su piu' partite e piu' seed.