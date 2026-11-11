REGISTRO DEGLI ERRORI E DELLE OSSERVAZIONI
==========================================

Questo file raccoglie gli errori, i difetti di progettazione e i comportamenti
inattesi emersi durante lo sviluppo e l'analisi del progetto. Serve come
materiale per la sezione di discussione del report.

Ogni voce riporta: descrizione, evidenza sperimentale, causa, conseguenze,
spunti per il report, stato (APERTO / RISOLTO, con riferimento al commit).


-------------------------------------------------------------------------------
E1. REWARD HACKING: "MANGIA E TAGLIATI LA CODA"
-------------------------------------------------------------------------------
Data individuazione: 2026-11-11
Algoritmo:           SARSA (sarsa_4); riguarda anche Q-learning (q_learning_2),
                     che usa lo stesso ambiente e gli stessi reward
Categoria:           progettazione della funzione di reward
Stato:               CORRETTO in SARSA (in attesa di verifica sperimentale);
                     da applicare a Q-learning

DESCRIZIONE
L'agente non impara a sopravvivere facendo crescere il serpente: impara a
mangiare frutti e a perdere periodicamente la coda auto-collidendo. La
funzione di reward premia questa strategia, quindi l'agente la sfrutta.

EVIDENZA SPERIMENTALE
Media degli ultimi 20 episodi registrati (sarsa_4/training_history_sarsa.csv):
  frutti mangiati per episodio  ~ 72.0
  segmenti persi per episodio   ~ 70.8
  auto-collisioni per episodio  ~ 6.0
  reward medio per episodio     ~ 28.7
Quasi tutti i segmenti guadagnati mangiando vengono persi nelle auto-collisioni:
il serpente resta sistematicamente corto.
Il reward medio torna con il conto:
  0.5 * 72.0 - 0.1 * 70.8 = 28.9  (circa 28.7, la differenza sono gli urti
                                   contro il muro, -0.3 ciascuno)
I risultati di Q-learning sono analoghi (73.4 frutti, 72.3 segmenti persi,
6.5 auto-collisioni): il problema non dipende dall'algoritmo.

CAUSA
Reward definiti nell'ambiente (environments_fully_observable_2.py):
  FRUIT_REWARD       = +0.5  per frutto (=> +1 segmento)
  ATE_HIMSELF_REWARD = -0.1  per ogni segmento perso
Un segmento vale +0.5 quando lo si guadagna e costa solo -0.1 quando lo si
perde. Il ciclo "mangia k frutti, poi tagliati k segmenti" ha quindi un
rendimento netto positivo:
  +0.5k - 0.1k = +0.4k > 0
Lo sconto temporale (gamma = 0.95) peggiora le cose: il frutto arriva prima
della penalita', quindi la penalita' scontata pesa ancora meno.
Inoltre un serpente corto si muove piu' liberamente e raggiunge i frutti
prima, quindi tagliarsi la coda e' anche strumentalmente utile.

CONSEGUENZE
- La metrica "reward medio" cresce e si stabilizza: il training sembra
  riuscito, ma il comportamento appreso non e' quello voluto.
- L'obiettivo 2 del progetto (evitare le auto-collisioni) e' in conflitto
  con il segnale di reward: l'agente non ha alcun incentivo a perseguirlo.
- Il confronto SARSA vs Q-learning misura come i due algoritmi sfruttano lo
  stesso difetto, non quale dei due impari meglio a giocare.

SPUNTI PER IL REPORT
- E' un caso di "reward hacking" / "specification gaming": l'agente
  massimizza correttamente il reward specificato, ma il reward non coincide
  con l'obiettivo reale (Amodei et al., 2016, "Concrete Problems in AI
  Safety"; Krakovna et al., 2020, "Specification gaming").
- Le curve di reward da sole non bastano per validare un agente: serve
  osservare il comportamento (GIF) e metriche legate all'obiettivo reale
  (segmenti persi, lunghezza del serpente).
- Nota teorica: un reward shaping basato su potenziale (Ng et al., 1999)
  NON risolve il problema, perche' per costruzione lascia invariata la
  policy ottima. Bisogna cambiare il reward vero e proprio.

CORREZIONE (sarsa_4/heuristic_sarsa_snake_4.py)
Nuova penalita' di auto-collisione, con k = segmenti persi:
  R_collisione = -(0.5 + 0.5 * k)
- La parte proporzionale (0.5 per segmento = FRUIT_REWARD) restituisce
  esattamente quanto il segmento era valso: da sola renderebbe il ciclo
  mangia-e-tagliati neutro (rendimento 0).
- La parte fissa (0.5) rende ogni collisione strettamente svantaggiosa:
  mangiare k frutti e poi perderli rende 0.5k - (0.5 + 0.5k) = -0.5
  (con il reward originale rendeva +0.4k).
- Tra collisioni inevitabili l'ordinamento resta guidato da k: se
  intrappolato, l'agente preferisce comunque mordersi vicino alla coda.
- Frutto (+0.5) e muro (-0.3) invariati.
- L'ambiente NON e' modificato: il reward viene ricalcolato nello script
  di training (funzioni segments_lost_in_step e shaped_rewards).
- Aggiunta la metrica "lunghezza_media" del serpente, che misura
  direttamente l'obiettivo reale (il reward medio da solo puo' ingannare).

Rischio da monitorare: la penalita' cresce con la lunghezza, quindi un
serpente lungo e' "a rischio" e l'agente potrebbe diventare troppo prudente
(smettere di mangiare quando e' lungo). Va osservato nei risultati.

Verifica sperimentale: run SARSA breve (60 episodi), risultati in
sarsa_4/risultati_reward_shaping/. Esito: (da compilare)


-------------------------------------------------------------------------------
E2. METRICHE: AUTO-COLLISIONI DA 3 SEGMENTI CONTATE COME URTI CONTRO IL MURO
-------------------------------------------------------------------------------
Data individuazione: 2026-11-11
Algoritmo:           SARSA e Q-learning (stesso codice delle metriche)
Categoria:           bug nel calcolo delle metriche
Stato:               CORRETTO in SARSA; da applicare a Q-learning

DESCRIZIONE
Le auto-collisioni venivano riconosciute dal valore del reward: "reward
negativo e diverso da HIT_WALL_REWARD (-0.3)". Ma un'auto-collisione che fa
perdere esattamente 3 segmenti vale 3 * (-0.1) = -0.3, cioe' lo stesso
valore dell'urto contro il muro, e veniva quindi scartata.

EVIDENZA
Test con 32 board per 3000 passi (policy euristica mista a mosse casuali):
su 4999 auto-collisioni, 629 (12.6%) erano da esattamente 3 segmenti, cioe'
sarebbero state ignorate dalla metrica originale.

CONSEGUENZE
"self_collision" e "segmenti_persi" nei CSV della baseline (reward
originale, sia SARSA che Q-learning) sono SOTTOSTIMATI. La frazione esatta
dipende dalla policy; nel test sopra era il 12.6% delle collisioni.
Va tenuto presente quando si riportano i numeri della baseline nel report.

CORREZIONE
Le metriche ora sono ricavate dallo stato dell'ambiente e non dal reward:
  segmenti_persi = lunghezza_prima + frutto_mangiato - lunghezza_dopo
Verificato sullo stesso test: 0 discrepanze con la penalita' calcolata
dall'ambiente. Il nuovo metodo resta valido anche dopo il reward shaping
(E1), che rende inutilizzabile il riconoscimento basato sul valore.

SPUNTI PER IL REPORT
Classificare gli eventi a partire dal valore del reward e' fragile: valori
diversi di eventi diversi possono coincidere. Meglio misurare dallo stato.
