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
Stato:               CORRETTO in SARSA e verificato (con E3-E5); da applicare a
                     Q-learning
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
sarsa_4/risultati_reward_shaping/. Esito: NEGATIVO. Il reward hacking e'
sparito, ma l'agente e' collassato su una policy passiva che non mangia
piu' (vedi E3).


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


-------------------------------------------------------------------------------
E3. COLLASSO SU UNA POLICY PASSIVA DOPO LA CORREZIONE DEL REWARD
-------------------------------------------------------------------------------
Data individuazione: 2026-11-11
Algoritmo:           SARSA con reward corretto (E1), run di 60 episodi
Categoria:           progettazione del reward / dinamica di apprendimento
Stato:               CORRETTO in SARSA; da applicare a Q-learning

DESCRIZIONE
Con la nuova penalita' di auto-collisione l'agente smette di mangiare:
oscilla avanti e indietro tra due celle (o gira in un quadrato 2x2) con il
serpente lungo 1 (solo la testa), evitando il frutto. Reward medio = 0.

EVIDENZA SPERIMENTALE (sarsa_4/risultati_reward_shaping/)
                          episodio 0    ultimi 10 episodi
  frutti mangiati           6.7           ~0.0
  auto-collisioni           4.8           ~0.0
  segmenti persi            6.7           ~0.0
  lunghezza media           1.37          1.00
  reward medio            -44.9           ~0.0
- Per tutto il training segmenti_persi ~= frutti_mangiati e ogni collisione
  costa ~1 segmento: il serpente mangia e subito dopo si morde il collo.
- GIF (checkpoint 20, 40 e finale): in 500 passi il frutto non viene mai
  mangiato; la testa alterna fra 2 celle (ep20, finale) o 4 celle (ep40).
- Loss ~0: la rete prevede Q ~ 0 ovunque, cioe' il valore della policy
  "non fare niente".

CAUSA
1. Con un serpente di lunghezza 1 il dietro-front e' gratis (non c'e' corpo
   da colpire), quindi oscillare e' una policy a costo zero (STEP_REWARD = 0).
2. Appena l'agente mangia, il serpente ha un segmento e lo stesso dietro-front
   diventa un'auto-collisione: +0.5 (frutto) - 1.0 (collisione con k = 1)
   = -0.5. Data l'abitudine appresa di oscillare, mangiare ha valore
   atteso negativo.
3. L'alternativa "girovagare senza mangiare" vale 0, cioe' piu' di -0.5:
   l'agente converge su un ottimo locale passivo.
4. L'esplorazione (euristica, epsilon <= 0.2) porta l'agente sul frutto, ma
   subito dopo la policy greedy fa dietro-front: l'esplorazione stessa
   rinforza l'idea "frutto = male".
5. Lo sconto (gamma = 0.95) favorisce il pericolo immediato rispetto ai
   frutti lontani.
Non e' l'ottimo vero: una policy che mangia senza mordersi ottiene reward
positivo. E' un ottimo locale della dinamica di apprendimento.

SPUNTI PER IL REPORT
- Correggere un reward hacking ha prodotto il problema opposto: dal
  "mangia e tagliati" al "non mangiare". Il reward va calibrato considerando
  anche l'alternativa passiva (stare fermi/girovagare a costo zero).
- Esempio di ottimo locale/policy degenerata in RL con approssimazione di
  funzione e senza replay: l'esplorazione non basta a uscirne.
- La scelta di non mascherare il dietro-front (README) ha un costo: e' la
  causa dominante delle collisioni da 1 segmento.

CORREZIONE (sarsa_4/heuristic_sarsa_snake_4.py) - in attesa di verifica
A) Maschera sul dietro-front nella scelta greedy: il valore Q dell'azione
   che porterebbe la testa sul collo viene posto a -inf prima dell'argmax
   (funzione reverse_actions). Nel vero Snake e' una mossa impossibile.
   Il collo e' ricavato dalla posizione del primo segmento del corpo, non
   dall'ultima azione: corretto anche dopo urti contro il muro e a inizio
   episodio. Vale anche per l'azione A' usata nel target SARSA, quindi la
   policy valutata e quella eseguita coincidono.
   Nota: cambia la scelta progettuale del README ("nessuna maschera, la
   rete deve imparare da sola").
B) Costo per passo STEP_PENALTY = -0.01, sommato al reward di ogni passo.
   Girovagare per sempre vale -0.01 / (1 - 0.95) = -0.2 invece di 0, quindi
   la policy passiva non e' piu' un punto di equilibrio "gratuito".
La penalita' di auto-collisione di E1 resta invariata: -(0.5 + 0.5k).
Verifica del codice: su 6751 stati con collo, l'azione mascherata porta
sempre sul collo; con epsilon = 0 la policy greedy non ha mai fatto
dietro-front ne' morso il collo.
Verifica sperimentale: run SARSA di 60 episodi, risultati in
sarsa_4/risultati_mask_steppenalty/. Esito: POSITIVO sull'obiettivo
reale. Lunghezza media del serpente da 3.5 (ep 0) a 35.6 (ep 59) su 64
celle libere, auto-collisioni da ~5 a ~2 per episodio, la GIF finale mangia
34 frutti in 500 passi senza urti. Il reward medio pero' DIMINUISCE
durante il training (da circa -6 a circa -20): vedi E4.


-------------------------------------------------------------------------------
E4. IL REWARD MEDIO CALA MENTRE L'AGENTE MIGLIORA (URTI CONTRO IL MURO)
-------------------------------------------------------------------------------
Data individuazione: 2026-11-11
Algoritmo:           SARSA con correzioni E1 + E3 (A+B), run di 60 episodi
Categoria:           comportamento appreso / metrica di valutazione
Stato:               CORRETTO in SARSA; da applicare a Q-learning

DESCRIZIONE
Con le correzioni A+B l'agente fa crescere il serpente e si morde meno, ma il
reward medio per episodio scende. Scomponendo il reward nelle sue voci si vede
che la causa principale sono gli urti contro il muro, in forte aumento.

EVIDENZA (scomposizione del reward, sarsa_4/risultati_mask_steppenalty/)
Gli urti contro il muro non sono registrati nel CSV, ma si ricavano per
differenza:  urti = (R - 0.5*frutti + 0.5*collisioni + 0.5*segmenti
                     + 0.01*500) / (-0.3)
  ep   reward   frutti(+)  segmenti(-)  passi  muro(-)   urti muro
   6   -11.4     +15.3       -16.1      -5.0    -2.7        ~9
  12    -5.7     +16.4       -13.8      -5.0    -1.5        ~5
  30   -12.4     +14.6       -15.7      -5.0    -4.9       ~16
  42   -14.0     +12.8       -12.8      -5.0    -7.7       ~26
  59   -19.8      +7.4        -7.7      -5.0   -13.5       ~45
GIF del checkpoint 40 (env nuovo, serpente lungo 1): la testa resta ferma
contro il muro per 494 passi su 500, senza mai mangiare.

CAUSE
1. Urti contro il muro in aumento (da ~5 a ~45 per episodio): e' la voce che
   pesa di piu' nel calo. Un urto lascia lo stato quasi invariato (la testa
   non si muove, si ritira solo la coda): se la rete sceglie "muro" in uno
   stato, al passo dopo lo sceglie di nuovo -> loop. Con il serpente lungo 1
   lo stato e' identico e il loop e' infinito (GIF ep40). In training lo
   interrompe solo l'esplorazione, che pero' cala con epsilon (0.2 -> 0.02):
   per questo gli urti crescono verso la fine.
   E' la particolarita' dell'ambiente individuata in precedenza (urto contro
   il muro = attesa sul posto con la coda che si ritira).
2. Meno frutti (da ~30 a ~15 per episodio) man mano che il serpente si
   allunga: con 35 segmenti su 64 celle raggiungere il frutto in sicurezza
   richiede percorsi piu' lunghi.
3. Le board non vengono mai resettate: ogni episodio parte con un serpente
   piu' lungo del precedente, quindi il compito diventa piu' difficile
   durante il training. Il reward per episodio non e' confrontabile tra
   episodi diversi.

SPUNTI PER IL REPORT
- Il reward medio di training non e' una buona misura di progresso quando la
  distribuzione degli stati cambia (task continuo, senza reset): serve una
  valutazione separata (policy greedy, board nuove, piu' partite, seed fissi).
- Scomporre il reward nelle sue componenti permette di diagnosticare quale
  comportamento sta peggiorando.
- Mancava una metrica per gli urti contro il muro.

CORREZIONE (sarsa_4/heuristic_sarsa_snake_4.py) - in attesa di verifica
1) Maschera sul muro nella scelta greedy (funzione wall_actions_mask): le
   azioni che porterebbero la testa contro il muro hanno Q = -inf prima
   dell'argmax, come il dietro-front (E3, correzione A). Resta sempre almeno
   una mossa (in un angolo con il collo dietro ne resta una sola). Le mosse
   sul corpo restano permesse: evitare le trappole e' cio' che la rete deve
   imparare. Con muro e dietro-front mascherati, la rete non deve piu'
   imparare le regole del gioco ma solo la strategia; l'euristica di
   esplorazione esclude gia' entrambe, quindi le due parti della policy sono
   coerenti.
2) Nuova metrica di training "urti_muro" (testa ferma dopo la mossa).
3) Valutazione separata (evaluate_policy): ogni 10 episodi e all'ultimo,
   la policy greedy (epsilon = 0) gioca 10 partite nuove da 500 passi con
   seed fisso (stesse posizioni iniziali a ogni valutazione). Registra
   frutti, lunghezza massima e finale, auto-collisioni, segmenti persi,
   urti contro il muro. Lo stato del generatore casuale del training viene
   salvato e ripristinato. Output: eval_history_sarsa.csv, eval_history.png.
Verifica del codice: maschera muro confrontata con la definizione diretta
su 32000 stati (1912 in un angolo); con epsilon = 0 nessun urto contro il
muro e nessun dietro-front; due valutazioni consecutive danno risultati
identici e non alterano il generatore casuale del training.
Verifica sperimentale: run SARSA di 60 episodi, risultati in
sarsa_4/risultati_mask_muro/. Esito: POSITIVO sul problema di E4.
Urti contro il muro = 0 in tutto il training; il reward non cala piu'
(oscilla fra -5 e -8). Lunghezza media da 3.5 a ~25, auto-collisioni da ~5
a ~2 per episodio, ~26-34 frutti per episodio.
Nota: a regime il reward resta negativo per costruzione. Se la lunghezza e'
stabile, i segmenti persi sono circa pari ai frutti mangiati, quindi frutti
e segmenti si compensano (+0.5 / -0.5 ciascuno) e resta
  R ~ -0.01 * 500 - 0.5 * collisioni ~ -5 - 0.5 * collisioni.
Es. ep 59: 13.05 - 1.08 - 12.36 - 5.0 = -5.39. Il reward di training misura
quindi soprattutto il numero di collisioni.
La valutazione separata ha pero' rivelato un nuovo problema: vedi E5.


-------------------------------------------------------------------------------
E5. LA POLICY GREEDY NON SA GIOCARE PARTITE NUOVE (DISTRIBUTION SHIFT + LOOP)
-------------------------------------------------------------------------------
Data individuazione: 2026-11-12
Algoritmo:           SARSA con correzioni E1, E3 (A+B), E4; run di 60 episodi
Categoria:           distribuzione degli stati di training / valutazione
Stato:               CORRETTO in SARSA (reset per episodio); da applicare a
                     Q-learning

DESCRIZIONE
In training l'agente mangia ~30 frutti per episodio, ma la stessa rete in
valutazione (policy greedy su partite nuove, serpente lungo 1) quasi non
mangia: gira in loop di 2 o 4 celle. Solo un checkpoint su 7 gioca bene, e
il risultato crolla subito dopo.

EVIDENZA (sarsa_4/risultati_mask_muro/eval_history_sarsa.csv)
  episodio di training    0    10    20    30    40    50    59
  frutti (media 10 part.) 0.1  0.9   0.0   1.0   0.6  27.3   9.0
  lunghezza massima       1.1  1.8   1.0   2.0   1.6  25.9   7.6
GIF (partite nuove, epsilon = 0):
  checkpoint 20: 0 frutti, oscillazione fra 2 celle (periodo 2)
  checkpoint 40: 3 frutti nei primi 10 passi, poi loop di periodo 4
  finale:        0 frutti, oscillazione fra 2 celle (periodo 2)

CAUSE
1. Distribution shift. Le board non vengono mai resettate: dopo pochi
   episodi i serpenti in training sono lunghi (lunghezza media 10-25) e la
   rete non vede quasi piu' stati con serpenti corti. La valutazione parte
   invece da serpenti di lunghezza 1: stati che la rete conosce poco.
2. Con lunghezza 1 non c'e' collo, quindi la maschera del dietro-front non
   si applica e l'oscillazione fra 2 celle torna possibile.
3. Con il serpente lungo 1 un loop riporta esattamente allo stesso stato:
   policy greedy + stato identico = loop infinito. In training il loop si
   rompe grazie all'esplorazione euristica (che va verso il frutto), in
   valutazione (epsilon = 0) no. Parte delle prestazioni di training e'
   quindi merito dell'euristica, non della rete.
4. Il loop costa poco: -0.01 per passo, cioe' -0.2 in tutto con lo sconto.
   La differenza di valore fra "loop" e "vai al frutto" e' piccola e si
   perde nel rumore delle stime Q.
5. Instabilita': SARSA online, senza experience replay ne' rete target. La
   policy cambia molto tra checkpoint vicini (27.3 frutti a ep 50, 9.0 a ep 59).

SPUNTI PER IL REPORT
- Senza la valutazione separata questo problema non sarebbe emerso: le
  metriche di training (con esplorazione e serpenti gia' lunghi) davano un
  quadro troppo ottimistico.
- Esempio concreto di distribution shift fra training e test in RL.
- In un task continuo senza reset l'agente impara bene solo gli stati che
  la sua stessa policy genera.

CORREZIONE (sarsa_4/heuristic_sarsa_snake_4.py) - in attesa di verifica
Reset delle board all'inizio di ogni episodio (parametro
reset_each_episode=True di sarsa_train): ogni episodio parte con 32 partite
nuove (serpenti lunghi 1), quindi il training vede sempre anche l'inizio
della partita, come la valutazione.
- Cambia la struttura descritta nel README: le partite non proseguono piu'
  tra un episodio e l'altro. Con reset_each_episode=False si torna al
  comportamento originale.
- Effetto collaterale utile: il reward per episodio torna confrontabile tra
  episodi diversi, perche' ogni episodio parte dalla stessa situazione.
- La fine dell'episodio e' un troncamento per limite di tempo, non uno stato
  terminale: l'ultimo aggiornamento continua correttamente a fare bootstrap
  su Q(s', a') (cfr. Pardo et al., 2018, "Time Limits in Reinforcement
  Learning").
- Le altre cause (loop poco penalizzati, instabilita' senza replay/target)
  restano: se i loop persistono, il passo successivo e' una penalita'
  "fame" dopo N passi senza frutto.
Verifica del codice: in un mini-training di 3 episodi ogni episodio e'
ripartito da 8 board con corpo vuoto, una testa e un frutto ciascuna.
Verifica sperimentale: run SARSA di 60 episodi, risultati in
sarsa_4/risultati_reset/. Esito: POSITIVO.
  valutazione (10 partite nuove, greedy)
  episodio            0    10    20    30    40    50    59
  frutti             0.1   0.6  15.8  17.5  19.5  21.7  23.3
  lunghezza massima  1.1   1.6  12.1  17.8  17.1  18.3  21.8
  training: reward da -6.7 a +9.3 (prima volta positivo e in crescita),
  auto-collisioni da 6.1 a 1.3, segmenti persi da ~24 (picco) a 8.8,
  lunghezza media 21.6, urti contro il muro 0.
La valutazione cresce in modo costante dopo l'episodio 20, senza i crolli
del run precedente. GIF: checkpoint 40 = 41 frutti, finale = 33 frutti,
nessun loop.
Limiti residui:
- la GIF del checkpoint 20 e' ancora in un loop di periodo 2 con 0 frutti:
  i loop a serpente corto non sono spariti del tutto, sono diventati rari;
- in valutazione l'agente mangia meno che in training (23 contro 39
  frutti): la valutazione riporta solo medie, quindi non si vede se alcune
  partite restano bloccate in loop;
- a fine run le curve crescono ancora: il training non e' a convergenza;
- un solo run e un solo seed: servono piu' seed per conclusioni solide.
Azioni conseguenti (sarsa_4/heuristic_sarsa_snake_4.py):
- la valutazione riporta anche frutti_min, frutti_max, partite_zero_frutti
  e partite_bloccate (nessun frutto negli ultimi 100 passi, tipicamente un
  loop), cosi' i loop residui sono visibili nei numeri e non solo nelle GIF;
- nuovo run da 200 episodi (come la baseline), risultati in
  sarsa_4/risultati_200ep/. Esito: training buono, valutazione instabile
  (vedi E7).


-------------------------------------------------------------------------------
E6. NOTA DI METODO: AMBIENTI USATI (FULLY E PARTIALLY OBSERVABLE)
-------------------------------------------------------------------------------
Data: 2026-11-13
Categoria:           metodologia (non e' un errore)
Stato:               DOCUMENTATO

AMBIENTE FULLY OBSERVABLE
Tutti gli esperimenti usano environments_fully_observable_2.py, una versione
modificata per scelta dell'ambiente fornito (environments_fully_observable.py):
  1) auto-collisione: -0.1 * segmenti persi (originale: -0.2 fisso);
  2) urto contro il muro: -0.3 (originale: -0.1);
  3) import "tensorflow.keras" (l'originale "keras.api._v2.keras" non
     funziona con TensorFlow 2.21 / Keras 3);
  4) move() accetta azioni con forma (n_boards,) (originale: (n_boards, 1)).

AMBIENTE PARTIALLY OBSERVABLE
environments_partially_observable_2.py e' ottenuto dal file fornito
(environments_partially_observable.py) applicando le STESSE 4 modifiche.
Cosi' i due ambienti differiscono solo per l'osservazione:
  - fully:     l'agente vede tutta la board 10x10;
  - partially: l'agente vede una finestra (2*mask_size+1)^2 centrata sulla
               testa (5x5 con mask_size=2); fuori dalla board la finestra
               mostra muro. Il costruttore ha in piu' il parametro mask_size.
Verifica: con lo stesso seed e le stesse azioni (32 board, 3000 passi
casuali, 551 auto-collisioni) i due ambienti producono board e reward
identici; la finestra osservata coincide con la board attorno alla testa.

NOTE
- Nel training le differenze 1 e 2 rispetto all'ambiente fornito sono
  comunque irrilevanti: la penalita' di auto-collisione e' ricalcolata
  nello script (E1) e le mosse contro il muro sono mascherate (E4).
- Il reward hacking di E1 esisterebbe anche con il reward fornito (-0.2
  fisso), anzi sarebbe piu' forte.

CONFIGURAZIONE DEGLI ESPERIMENTI PARTIALLY OBSERVABLE (2026-11-13)
Gli script modificati di SARSA e Q-learning hanno un unico parametro
OSSERVABILITA ("fully" | "partial", MASK_SIZE = 2). In "partial":
  - stato della rete: finestra 5x5x4 (input della rete 5x5 invece di
    10x10; stessa architettura);
  - esplorazione euristica: riceve solo la finestra 5x5. Se il frutto non
    e' visibile sceglie a caso fra le mosse sicure (niente muro, corpo,
    dietro-front);
  - restano identici: reward (incluso lo shaping sulla distanza), maschere
    muro/dietro-front (guardano solo le celle adiacenti alla testa, sempre
    visibili), valutazione, iperparametri, reset per episodio.
  - risultati in risultati_po_200ep/ (fully: risultati_finale_200ep/).
Nota per il paper: lo shaping sulla distanza (E8) e' calcolato
dall'ambiente con la posizione reale del frutto. Nel caso partial il
reward porta quindi un'informazione indiretta sulla direzione del frutto
anche quando questo non e' nella finestra. E' legittimo (il reward
appartiene all'ambiente, non all'osservazione), ma va dichiarato; e'
mantenuto per avere lo stesso reward nelle 4 celle del confronto.
Verifica: per 300 passi su 32 board la finestra data all'euristica
coincide con to_state(); mini-training di 2 episodi completato per
entrambi gli algoritmi.


-------------------------------------------------------------------------------
E7. LOOP DI PARTENZA: LA POLICY GREEDY OSCILLA CON IL SERPENTE LUNGO 1
-------------------------------------------------------------------------------
Data individuazione: 2026-11-13
Algoritmo:           SARSA fully observable, configurazione completa
                     (E1-E5), run di 200 episodi
Categoria:           comportamento appreso / regole del gioco
Stato:               APERTO

DESCRIZIONE
In training l'agente migliora in modo costante, ma in valutazione (greedy,
partite nuove) le partite si dividono in due gruppi: o vanno bene (40-60
frutti) o restano bloccate fin dal primo passo senza mangiare nulla. Il
serpente, lungo 1, oscilla avanti e indietro fra due celle per 500 passi.
La quota di partite bloccate cambia molto da un checkpoint all'altro, per
questo la media della valutazione salta su e giu'.

EVIDENZA (sarsa_4/risultati_200ep/)
Training: reward da -5.8 a +7.7, auto-collisioni da 5.7 a ~2, segmenti
persi da ~26 a ~12-17, frutti ~44, lunghezza media ~22, urti muro 0.
Valutazione (10 partite):
  episodio            40    50    80   130   160   170   180   190   199
  frutti medi        24.5  38.9   0.9   0.1  21.1  19.4  21.9  36.5  34.3
  frutti max         40    55     4     1    55    46    50    53    62
  partite 0 frutti    2     1     7     9     5     5     5     0     3
  partite bloccate    4     1    10    10     5     5     5     1     3
Nella maggior parte dei checkpoint "partite con 0 frutti" = "partite
bloccate": le partite bloccate non hanno mai mangiato, cioe' si bloccano
all'inizio. Quando la partita parte, l'agente gioca bene.
GIF: checkpoint 100, 150 e finale = 0 frutti, oscillazione di periodo 2 dal
primo passo; checkpoint 50 = 44 frutti.

CAUSE
1. Con il serpente lungo 1 non c'e' collo, quindi la maschera del
   dietro-front (E3) non si applica e l'oscillazione fra due celle e'
   permessa. Nel vero Snake il dietro-front e' vietato anche in questo caso.
2. Con lunghezza 1 lo stato dopo due passi e' identico: una policy greedy
   deterministica che sceglie A -> B e B -> A non ne esce piu'.
3. Natura on-policy di SARSA: impara il valore della policy epsilon-greedy,
   in cui l'esplorazione (euristica) prima o poi rompe il loop e porta al
   frutto. Il valore stimato degli stati di loop include quindi questo
   "salvataggio" e non risulta abbastanza negativo da far cambiare la scelta
   greedy. Con epsilon = 0 il salvataggio non c'e'.
4. Gli stati con serpente lungo 1 sono pochi nei dati di training (solo i
   primi passi di ogni episodio, finche' non si mangia il primo frutto), e
   senza experience replay ne' rete target la policy su questi stati cambia
   molto fra un checkpoint e l'altro.

SPUNTI PER IL REPORT
- Differenza fra prestazioni della policy di comportamento (epsilon-greedy,
  training) e della policy greedy (valutazione): con SARSA la differenza e'
  strutturale, perche' i valori appresi sono quelli della policy con
  esplorazione. Da confrontare con Q-learning, che stima i valori della
  policy greedy.
- Le medie della valutazione nascondono una distribuzione bimodale: servono
  anche min/max e conteggi delle partite bloccate.

CORREZIONE (sarsa_4/heuristic_sarsa_snake_4.py) - in attesa di verifica
1) Dietro-front vietato anche con il serpente lungo 1: quando il corpo e'
   vuoto, la maschera esclude l'opposto dell'ultima direzione di movimento
   (reverse_actions con last_directions). E' la regola del vero Snake e
   completa la maschera di E3. All'inizio della partita (direzione NONE)
   non c'e' dietro-front da escludere.
2) Valutazione su 20 partite invece di 10, per ridurre l'incertezza sulla
   media (la varianza fra partite buone, ~25-60 frutti, e' naturale perche'
   dipende da dove compaiono i frutti).
Verifica del codice: rete non addestrata, policy greedy, 3 seed x 32 board
x 300 passi (circa 19800 passi a lunghezza 1): nessun ritorno A -> B -> A.
Restano possibili loop piu' lunghi (es. giro su un quadrato 2x2).

Varianza: il fix risolve la componente principale (partite bloccate
dall'inizio). Non risolve i checkpoint "rotti" (ep 80, 100, 130 del run da
200 episodi), in cui anche le partite partite mangiano solo 1-3 frutti:
e' instabilita' del training online. Se persistono, il passo successivo e'
experience replay + rete target.
Verifica sperimentale: run SARSA di 60 episodi, risultati in
sarsa_4/risultati_fix_dietrofront/. Esito: PARZIALE. Le oscillazioni di
periodo 2 sono sparite (le partite con 0 frutti scendono a 0-3 su 20 dopo
l'episodio 40), ma sono comparsi loop piu' lunghi dopo i primi frutti
(vedi E8). Valutazione ancora instabile: 38.2 frutti a ep 40, 19.9 a ep 50,
13.95 a ep 59.


-------------------------------------------------------------------------------
E8. LOOP LUNGHI: LA POLICY GREEDY DIPENDE DALL'ESPLORAZIONE PER MANGIARE
-------------------------------------------------------------------------------
Data individuazione: 2026-11-13
Algoritmo:           SARSA fully observable, E1-E5 + fix E7, run di 60 episodi
Categoria:           comportamento appreso / interazione con l'esplorazione
Stato:               APERTO

DESCRIZIONE
Eliminata l'oscillazione fra due celle (E7), la policy greedy mangia alcuni
frutti e poi entra in cicli piu' lunghi (periodo 12-20 passi) in cui gira
senza avvicinarsi al frutto, fino alla fine della partita. In training
invece l'agente mangia ~43 frutti per episodio in modo stabile.

EVIDENZA (sarsa_4/risultati_fix_dietrofront/)
Valutazione (20 partite):
  episodio            20    30    40    50    59
  frutti medi        5.05  2.8  38.2  19.9  13.95
  partite 0 frutti     3     9     0     2     3
  partite bloccate    20    19     7    15    16
Le partite con 0 frutti sono poche, ma quelle bloccate (nessun frutto negli
ultimi 100 passi) sono la maggioranza: l'agente mangia, poi si blocca.
GIF: checkpoint 20 = 1 frutto poi ciclo di periodo 12; finale = 10 frutti
(l'ultimo al passo 77) poi ciclo di periodo 20; checkpoint 40 = 35 frutti.
Training (ultimi episodi): ~41-45 frutti, reward fino a +8.1.

CAUSE
1. SARSA impara il valore della policy di comportamento, cioe' epsilon-greedy
   con esplorazione euristica. L'euristica va verso il frutto: in training
   bastano poche azioni esplorative (epsilon ~0.02 a fine run, cioe' ~10
   azioni euristiche per partita) per rompere un ciclo e portare al frutto.
   La rete impara soprattutto a evitare i pericoli, mentre il compito di
   "andare al frutto" resta in parte all'euristica. Con epsilon = 0 il
   ciclo non si rompe. E' la stessa causa 3 di E7, in forma piu' generale:
   correggere un loop alla volta (periodo 2, poi 4, poi 12...) non basta.
2. Girare in tondo costa poco (-0.01 per passo) ed e' sicuro, mentre
   avvicinarsi al frutto con un corpo lungo comporta il rischio di
   collisioni: la differenza di valore fra le due scelte e' piccola.
3. Instabilita' del training online (nessun replay ne' rete target): la
   policy greedy cambia molto fra checkpoint vicini.

SPUNTI PER IL REPORT
- Gap fra policy di comportamento (training) e policy greedy (valutazione):
  con SARSA e un'esplorazione "intelligente" la rete puo' delegare parte del
  compito all'esplorazione. Il confronto con Q-learning (che stima i valori
  della policy greedy) e' particolarmente interessante su questo punto.
- In letteratura (es. DQN, Mnih et al. 2015) la valutazione si fa spesso con
  un piccolo epsilon (0.05) proprio per evitare cicli deterministici.

CORREZIONE (sarsa_4/heuristic_sarsa_snake_4.py) - in attesa di verifica
Reward shaping basato su potenziale sulla distanza dal frutto (Ng, Harada &
Russell, 1999), sommato al reward di ogni passo:
  F(s, s') = gamma * Phi(s') - Phi(s),   Phi(s) = C * (D_MAX - d(s))
  d = distanza di Manhattan testa-frutto, C = 0.02, D_MAX = 14,
  gamma = 0.95 (lo stesso del target TD).
- Insegna alla rete stessa ad andare verso il frutto, invece di lasciare il
  compito all'esplorazione euristica.
- Per costruzione non cambia la policy ottima: lungo una traiettoria la
  somma scontata di F vale gamma^T * Phi(s_T) - Phi(s_0).
- Phi >= 0 di proposito: lungo un ciclo che torna allo stesso stato la somma
  di F vale (gamma - 1) * somma(Phi) <= 0, quindi girare in tondo costa. Con
  Phi = -C*d (negativo) i cicli sarebbero stati premiati.
- Quando si mangia, Phi(s') usa il nuovo frutto: quel passo ha F negativo
  (in media -0.10, minimo -0.26), poi compensato avvicinandosi al nuovo
  frutto.
- Nel caso partially observable Phi usa la posizione vera del frutto anche
  quando non e' visibile: e' accettabile (il reward e' un segnale
  dell'ambiente, non un'osservazione) ma va dichiarato nel report.
- Nel CSV "reward_medio" resta il reward del compito (senza F), confrontabile
  con i run precedenti; F e' registrato a parte in "shaping_distanza".
Verifica del codice (32 board, 400 passi, policy euristica): distanze
corrette; somma scontata di F = gamma^T * Phi_T - Phi_0; avvicinarsi da
+0.007 a +0.019, allontanarsi da -0.021 a -0.031; ciclo di 4 stati: -0.04.
Verifica sperimentale: run SARSA di 60 episodi, risultati in
sarsa_4/risultati_shaping_distanza/. Esito: PARZIALE (un solo seed).
Confronto con il run precedente (fix E7, stessi 60 episodi):
  valutazione, frutti medi su 20 partite
  episodio              10     20     30     40     50     59
  senza shaping        0.85   5.05   2.8   38.2   19.9   13.95
  con shaping         10.05  15.2   44.3   20.6    8.15  45.25
  partite bloccate (senza / con shaping)
                     20/17  20/14  19/2    7/18  15/19  16/4
- Media sui checkpoint 10-59: 23.9 frutti con shaping contro 13.5 senza.
  I checkpoint buoni arrivano prima (ep 30) e sono migliori: a ep 59
  minimo 24 e massimo 63 frutti, 0 partite a zero frutti (il migliore
  finora). GIF finale: 45 frutti, nessun loop.
- Il training non cambia (frutti 41.2 contro 41.3, collisioni 2.27 contro
  2.26, lunghezza 19.0 contro 19.5): coerente con la teoria, lo shaping non
  cambia cosa e' ottimo, ma rende la policy greedy piu' capace di andare
  al frutto da sola.
- L'instabilita' resta: a ep 40 e 50 la valutazione crolla (18 e 19 partite
  bloccate su 20, GIF ep 40 con ciclo di periodo 32). Vedi E9.


-------------------------------------------------------------------------------
E9. INSTABILITA' DELLA POLICY GREEDY FRA CHECKPOINT
-------------------------------------------------------------------------------
Data individuazione: 2026-11-13
Algoritmo:           SARSA fully observable, configurazione completa (E1-E8)
Categoria:           stabilita' dell'apprendimento
Stato:               APERTO

DESCRIZIONE
La policy greedy alterna checkpoint buoni e checkpoint "rotti" a pochi
episodi di distanza, mentre le metriche di training restano stabili.

EVIDENZA (sarsa_4/risultati_shaping_distanza/, 20 partite)
  episodio              30     40     50     59
  frutti medi          44.3   20.6    8.15  45.25
  partite bloccate       2     18     19      4
Training negli stessi episodi: frutti 41.2 +- 1.2, collisioni 2.27 +- 0.33
(ultimi 10 episodi). Lo stesso andamento c'era nel run da 200 episodi
(E7: 38.9 frutti a ep 50, 0.9 a ep 80, 36.5 a ep 190).

CAUSE
1. Training online: ogni aggiornamento usa solo i 32 passi appena fatti,
   molto correlati fra loro (stesse partite, passi consecutivi).
2. Il target TD usa la stessa rete che si sta aggiornando ("moving
   target"): ogni aggiornamento sposta anche il bersaglio.
3. Piccole variazioni dei valori Q cambiano l'argmax in molti stati: la
   policy greedy cambia molto anche se i valori cambiano poco. In training
   l'esplorazione maschera l'effetto, in valutazione no.

SPUNTI PER IL REPORT
- Le metriche di training non bastano a valutare la stabilita': la policy
  greedy puo' oscillare mentre le curve di training sono piatte.
- Con un solo seed e checkpoint cosi' variabili, il confronto fra due
  configurazioni e' incerto: servono piu' seed o una scelta del modello piu'
  robusta (es. miglior checkpoint, media degli ultimi checkpoint).

CORREZIONE
(da definire)


-------------------------------------------------------------------------------
E10. GLI ORIGINALI DI SARSA E Q-LEARNING NON USANO LO STESSO EPSILON
-------------------------------------------------------------------------------
Data individuazione: 2026-11-13
Algoritmo:           originali (sarsa_4/heuristic_sarsa_snake_4_originale.py,
                     q_learning_2/heuristic_Q-learn_snake_2.py)
Categoria:           metodologia / equita' del confronto
Stato:               CORRETTO nelle versioni modificate

DESCRIZIONE
Il README dichiara che i due script hanno "stesso schema di epsilon", ma
negli originali:
  SARSA:      epsilon decrescente linearmente da 0.2 a 0.02
  Q-learning: epsilon COSTANTE a 0.15
Gli altri parametri coincidono (200 episodi da 500 passi, 32 board, stessa
rete, stesso ambiente). Lo script Q-learning originale contiene anche una
copia di sarsa_train (con parametri diversi: 1000 episodi da 200 passi,
epsilon 0.1) che pero' non viene eseguita.

CONSEGUENZE
Il confronto fra i due originali non e' alla pari: con epsilon 0.15 fisso
Q-learning esplora (cioe' usa l'euristica) molto piu' di SARSA nella seconda
meta' del training. Quando si confrontano i numeri degli originali va
dichiarato.

CORREZIONE
Le versioni modificate (sarsa_4/heuristic_sarsa_snake_4.py e
q_learning_2/heuristic_qlearning_snake_2_modificato.py) sono generate dallo
stesso codice e differiscono SOLO nel target del TD: stesso epsilon
(0.2 -> 0.02), stessi reward, maschere, reset, metriche, valutazione.
Verificato con un diff fra i due file.


-------------------------------------------------------------------------------
E11. PARTIALLY OBSERVABLE: LA POLICY GREEDY SI BLOCCA IN LOOP (ALIASING)
-------------------------------------------------------------------------------
Data individuazione: 2026-11-14
Algoritmo:           SARSA e Q-learning modificati, OSSERVABILITA="partial"
                     (finestra 5x5), 200 episodi
Categoria:           limite dell'osservabilita' parziale (atteso)
Stato:               DOCUMENTATO (risultato da discutere nel paper)

DESCRIZIONE
In training (con esplorazione) il partially observable va quasi come il
fully observable. In valutazione greedy invece crolla: la maggior parte
delle partite resta bloccata in un loop senza mangiare.

EVIDENZA (risultati_po_200ep/ vs risultati_finale_200ep/)
Training, ultimi 50 episodi:
                     frutti  auto-coll.  segm. persi  lunghezza
  SARSA fully         42.1      1.97        16.1        20.3
  QL fully            44.5      2.28        20.9        19.3
  SARSA partial       39.0      1.60        18.5        16.8
  QL partial          39.3      1.54        19.1        16.7
Valutazione greedy (20 partite), media dei checkpoint dall'episodio 100:
  SARSA fully 29.7, QL fully 27.2, SARSA partial 17.5, QL partial 16.1.
Partite bloccate per checkpoint: fully tipicamente 3-12 su 20, partial
tipicamente 13-20 su 20 (miglior checkpoint partial: 39 frutti, 2 bloccate).

CAUSA
Aliasing percettivo: con la finestra 5x5 molte situazioni diverse (frutto
fuori vista in posizioni diverse) producono la STESSA osservazione. Una
policy greedy senza memoria e' deterministica: stessa osservazione -> stessa
azione, quindi l'agente puo' ripetere lo stesso giro all'infinito senza mai
vedere il frutto. In training il loop viene rotto dall'esplorazione (euristica
con scelte casuali), per questo i numeri di training restano buoni.
E' il risultato noto per i POMDP: le policy deterministiche senza memoria
possono essere molto peggiori di quelle stocastiche (Singh, Jaakkola &
Jordan 1994).

CONSEGUENZE PER IL PAPER
- Il divario fully vs partial si vede soprattutto in valutazione greedy,
  non in training.
- Nel partial la differenza SARSA vs Q-learning e' minima (entro la
  variabilita' fra checkpoint, E9).
- Le auto-collisioni piu' basse nel partial dipendono anche dal serpente
  piu' corto (lunghezza ~17 contro ~20).

POSSIBILI INTERVENTI (non applicati)
- valutare anche con piccola casualita' (es. epsilon 0.05) per misurare
  quanto del crollo e' dovuto al determinismo;
- dare memoria all'agente (stack degli ultimi k frame o rete ricorrente).


-------------------------------------------------------------------------------
E12. ESPERIMENTO: PENALITA' PER LE CASELLE RIPERCORSE AL POSTO DELLO SHAPING
-------------------------------------------------------------------------------
Data: 2026-11-22
Algoritmo:           SARSA fully observable, E1-E7 (senza shaping sulla
                     distanza), run di 60 episodi
Categoria:           progettazione del reward / alternativa a E8
Stato:               ESITO NEGATIVO (si mantiene lo shaping di E8)

MOTIVAZIONE
Il problema di E8 (dopo pochi frutti la policy greedy gira in tondo su un
anello di 12-20 caselle) e' stato ridotto con lo shaping sulla distanza dal
frutto. Si vuole verificare se una penalita' che colpisce direttamente i
loop ottiene lo stesso effetto.

IMPLEMENTAZIONE (heuristic_sarsa_snake_4.py e, identica, nello script
Q-learning modificato; interruttore SHAPING = "distanza" | "rivisita")
- Per ogni board si ricordano le ultime W = 20 posizioni della testa
  (REVISIT_WINDOW); se la testa entra in una di queste caselle il passo
  riceve p = -0.02 (REVISIT_PENALTY), al posto del termine F di E8.
- Quando il serpente mangia la memoria si azzera: si puniscono solo i giri
  senza mangiare.
- Un giro costa circa -0.03 a passo (con STEP_PENALTY), circa -0.6 se
  ripetuto per sempre: piu' di -0.2 (solo costo per passo) ma meno di una
  auto-collisione (-1 o meno), per non spingere l'agente a mordersi.
- Nuova metrica "rivisite" (passi su caselle ripercorse per episodio),
  registrata con qualsiasi valore di SHAPING.
- Risultati in sarsa_4/risultati_rivisita/.
Verifica del codice: giro 2x2 con serpente lungo 1 -> penalita' dal quarto
passo in poi; memoria azzerata quando si mangia; finestra limitata a 20
posizioni; mini-training: termine di shaping = -0.02 x rivisite.

DIFFERENZE TEORICHE RISPETTO A E8 (da discutere nel paper)
- Non indica la direzione del frutto: punisce il giro ma non dice dove
  andare.
- Non e' basata su potenziale: cambia la policy ottima (Ng et al., 1999 non
  si applica).
- Dipende da una memoria che la rete vede solo in parte (il corpo mostra le
  ultime L posizioni, non le ultime 20): stessa board, penalita' diverse,
  quindi target piu' rumorosi.
- Un serpente lungo a volte deve ripassare vicino: l'azzeramento quando
  mangia attenua il problema.

CONFRONTO PREVISTO (stessi 60 episodi, valutazione su 20 partite)
  risultati_fix_dietrofront   -> nessun termine aggiuntivo (E7)
  risultati_shaping_distanza  -> shaping sulla distanza (E8)
  risultati_rivisita          -> penalita' per caselle ripercorse (E12)

ESITO (sarsa_4/risultati_rivisita/, 60 episodi): NEGATIVO
Training (ultimi 10 episodi), quasi uguale agli altri due run:
                       frutti  coll.  segm.persi  lunghezza
  senza termine (E7)    41.3   2.26     17.5        19.5
  shaping distanza (E8) 41.2   2.27     18.0        19.0
  rivisita (E12)        41.2   2.08     16.4        19.5
Passi su caselle ripercorse in training: da ~121 (ep 0) a ~5 per episodio.
Valutazione greedy (20 partite), frutti medi:
  episodio            10    20    30    40    50    59   media 10-59
  senza termine      0.8   5.0   2.8  38.2  19.9  13.9     13.5
  shaping distanza  10.1  15.2  44.3  20.6   8.2  45.2     23.9
  rivisita           2.8   9.9   2.8  42.3   0.9   9.8     11.4
Partite bloccate (media 10-59): 16.2 / 12.3 / 16.2.
GIF: checkpoint 20 = 3 frutti poi ciclo di periodo 12; finale = 6 frutti
(l'ultimo al passo 66) poi ciclo di periodo 10; checkpoint 40 = 49 frutti.

LETTURA
- In training la penalita' funziona: le caselle ripercorse crollano. Ma in
  training i loop li rompeva gia' l'esplorazione euristica.
- In valutazione greedy i loop restano (periodo 10-12, cioe' dentro la
  finestra di 20 passi: la penalita' scatta a ogni passo del giro, eppure la
  policy non ne esce). La penalita' dice che il giro e' male, ma non dove
  andare: uscire dal giro porta in stati che la rete non sa valutare meglio.
  Lo shaping sulla distanza invece da' una direzione a ogni passo.
- Risultato in valutazione equivalente a non avere alcun termine (11.4
  contro 13.5, entro la variabilita' fra checkpoint, E9); lo shaping resta
  nettamente migliore (23.9).
- Un solo seed: il confronto va letto come tendenza.

SPUNTI PER IL REPORT
Punire il sintomo (il loop) non basta: serve un segnale che indichi la
direzione utile. E' anche un argomento a favore dello shaping basato su
potenziale: guida l'apprendimento senza cambiare il compito.


-------------------------------------------------------------------------------
E13. CORREZIONE PROPOSTA PER E9: LEARNING RATE DECRESCENTE (COSENO)
-------------------------------------------------------------------------------
Data: 2026-11-22
Algoritmo:           SARSA e Q-learning modificati, fully observable,
                     configurazione finale (E1-E8), 200 episodi
Categoria:           stabilita' dell'apprendimento
Stato:               RISOLTO (1 seed): stabilita' migliorata in entrambi

MOTIVAZIONE
La policy greedy alterna checkpoint buoni e "rotti" (E9). Una delle cause e'
l'ampiezza degli aggiornamenti: con learning rate fisso 1e-3 anche a fine
training ogni passo puo' cambiare l'argmax in molti stati. Rete target ed
experience replay sono stati esclusi per mantenere gli algoritmi nella loro
forma online di base.

IMPLEMENTAZIONE (identica nei due script; LR_DECAY = True/False)
- Learning rate da 1e-3 (LR_INITIAL) a 1e-4 (LR_FINAL) con curva a coseno
  (tf.keras CosineDecay, alpha = 0.1), su tutti gli aggiornamenti del
  training: 200 episodi x 500 passi = 100000 passi di Adam.
- All'inizio resta vicino a 1e-3 (si impara come prima); a meta' training
  vale ~5.5e-4; alla fine 1e-4.
- Nuova colonna "learning_rate" nel CSV di training e nel log a schermo.
- Risultati in risultati_lr_decay_200ep/ (partially observable:
  risultati_lr_decay_po_200ep/). Con LR_DECAY = False si riproduce la
  configurazione di risultati_finale_200ep.
Verifica del codice: mini-training di 2 episodi da 50 passi -> learning rate
5.5e-4 dopo il primo episodio e 1e-4 alla fine, 100 passi di Adam; stesso
comportamento in SARSA e Q-learning.

COME SI VALUTA (checkpoint dall'episodio 100, confronto con
risultati_finale_200ep, stesso seed)
  - deviazione standard dei frutti in valutazione fra checkpoint (oggi ~10);
  - checkpoint "rotti" con meno di 10 frutti (oggi 1 su 11 per entrambi);
  - partite bloccate medie (oggi 7.0 SARSA, 8.4 Q-learning);
  - la media dei frutti in valutazione non deve scendere (29.7 / 27.2).
Limite: un solo seed, ~11 checkpoint: le differenze vanno lette come
tendenza.

RISULTATI SARSA FO (risultati_lr_decay_200ep vs risultati_finale_200ep)
                              LR fisso    LR coseno
  training ultimi 50: frutti    42.1        44.0
                      auto-coll. 1.97        1.55
                      segm. persi 16.1       13.7
                      lunghezza   20.3       22.5
  eval ckpt >= 100:   media      29.7        34.5
                      std        10.1        10.4
                      rotti (<10) 1/11       1/11
                      bloccate    7.0         5.3
  eval ckpt >= 110:   media      29.6        37.6
                      std        10.6         3.9
                      minimo      5.5        30.4
                      bloccate    6.9         3.9
L'unico checkpoint rotto con LR coseno e' l'episodio 100 (3.9 frutti, LR
ancora ~5.4e-4); da 110 in poi tutti i checkpoint stanno fra 30 e 44 frutti.
Con LR fisso i crolli restano fino in fondo (ep. 130: 5.5; ep. 180: 18.4).
Lettura: quando il learning rate e' piccolo la policy smette di oscillare,
come atteso. Il training migliora anche (meno auto-collisioni, serpente piu'
lungo).

RISULTATI Q-LEARNING FO (risultati_lr_decay_200ep vs risultati_finale_200ep)
                              LR fisso    LR coseno
  training ultimi 50: frutti    44.5        45.2
                      auto-coll. 2.28        2.88
                      segm. persi 20.9       22.4
                      lunghezza   19.3       19.9
  eval ckpt >= 100:   media      27.2        41.9
                      std        10.1         4.6
                      minimo      2.8        30.6
                      rotti (<10) 1/11       0/11
                      bloccate    8.4         1.3
Nessun checkpoint rotto dall'episodio 40 in poi tranne il 50; dal 100 in poi
tutti fra 30.6 e 47 frutti, partite bloccate quasi azzerate (0.7 da ep. 150).
Il training con esplorazione cambia poco (auto-collisioni leggermente piu'
alte, entro il rumore di un seed).

CONCLUSIONE
Il learning rate decrescente elimina quasi del tutto l'instabilita' fra
checkpoint in entrambi gli algoritmi (std da ~10 a ~4) e riduce i loop
greedy (bloccate: SARSA 7.0 -> 5.3, da ep. 110 3.9; Q-learning 8.4 -> 1.3).
Causa confermata di E9: aggiornamenti troppo grandi a fine training che
cambiano l'argmax. Un solo seed: tendenza forte e coerente nei due
algoritmi, non prova statistica.

RISULTATI PARTIALLY OBSERVABLE (risultati_lr_decay_po_200ep vs
risultati_po_200ep), checkpoint >= 100
                              SARSA fisso -> coseno   Q-learning fisso -> coseno
  training ultimi 50: frutti    39.0 -> 41.5            39.3 -> 42.4
                      auto-coll. 1.60 -> 1.71            1.54 -> 1.62
  eval:               media     17.5 -> 39.2            16.1 -> 37.6
                      std       11.2 ->  3.3            12.2 ->  7.1
                      rotti      4/11 -> 0/11            4/11 -> 0/11
                      bloccate  13.1 ->  1.4            13.4 ->  2.5
  eval ckpt >= 150:   media     12.7 -> 40.8            15.1 -> 41.0
                      bloccate  14.8 ->  0.5            14.0 ->  0.8
Effetto ancora piu' forte che in fully observable. Con il coseno la
valutazione greedy PO (~40 frutti) e' allo stesso livello di FO.

CONSEGUENZA PER E11
Gran parte dei loop greedy attribuiti all'aliasing percettivo in PO (E11)
era in realta' instabilita' da learning rate fisso: con il coseno le
partite bloccate scendono da ~13 a ~1-2 su 20. L'aliasing resta possibile in
teoria (finestra 5x5, policy senza memoria) ma, con questa finestra e questo
shaping, non e' il fattore dominante. Un solo seed.

-------------------------------------------------------------------------------
E14. ESPERIMENTO: NESSUNO SHAPING + LEARNING RATE A COSENO
-------------------------------------------------------------------------------
Data: 2026-11-23
Algoritmo:           SARSA e Q-learning modificati, configurazione finale
                     (E1-E7, E13) SENZA lo shaping sulla distanza (E8)
Categoria:           esperimento esplorativo
Stato:               CONCLUSO (1 seed): lo shaping non serve con il coseno

MOTIVAZIONE
Lo shaping sulla distanza (E8) era stato introdotto con learning rate fisso,
per ridurre i cicli della policy greedy. Il coseno (E13) ha poi eliminato
gran parte dei loop: si vuole vedere come si comporta l'agente senza shaping,
cioe' senza nessuna informazione sulla posizione del frutto nel reward
(rilevante soprattutto in partially observable).

IMPLEMENTAZIONE (identica nei due script)
- SHAPING = "nessuno": il termine di shaping e' 0; reward, maschere, reset,
  esplorazione e learning rate a coseno invariati.
- Risultati in risultati_no_shaping_lr_decay_200ep/ (partially observable:
  risultati_no_shaping_lr_decay_po_200ep/).
Verifica del codice: mini-training di 2 episodi da 50 passi su 4 board ->
shaping 0 in ogni episodio, nessun errore, in SARSA e Q-learning.

RISULTATI FULLY OBSERVABLE (risultati_no_shaping_lr_decay_200ep)
                              SARSA     Q-learning
  training ultimi 50: frutti   45.9       47.5
                      auto-coll. 1.77      2.15
                      segm. persi 13.7      19.3
                      lunghezza   22.9      21.8
  eval ckpt >= 100:   media     37.7       38.5
                      std        9.2        8.5
                      minimo    15.9       14.7
                      rotti (<10) 0/11      0/11
                      bloccate    4.1        3.0
Senza shaping l'agente impara lo stesso: in training mangia 46-48 frutti e
cresce fino a ~22. In valutazione greedy ci sono ancora cali a meta'
training (ep. 100-120: 15-22 frutti, 13-14 partite bloccate), poi da ep. 130
(SARSA) / 150 (Q-learning) i checkpoint sono stabili a 40-46 frutti con 0-3
partite bloccate. Lettura: con il learning rate a coseno lo shaping sulla
distanza non e' necessario in FO; il suo effetto sui loop (E8) era in gran
parte compensazione dell'instabilita' poi risolta da E13.

RISULTATI PARTIALLY OBSERVABLE (risultati_no_shaping_lr_decay_po_200ep)
                              SARSA     Q-learning
  training ultimi 50: frutti   42.1       41.9
                      auto-coll. 1.62      1.59
                      segm. persi 19.8      19.9
                      lunghezza   18.5       18.0
  eval ckpt >= 100:   media     39.7       32.3
                      std        4.8       12.5
                      rotti (<10) 0/11      1/11 (ep. 140: 2.1 frutti)
                      bloccate    1.5        5.1
  eval ckpt >= 150:   media     41.5       40.3
                      bloccate    0.3        0.7

CONFRONTO CON SHAPING (entrambi con coseno), eval frutti / bloccate
                    ckpt >= 100                ckpt >= 150
                    con        senza           con        senza
  SARSA FO       34.5/5.3   37.7/4.1        39.1/2.8   43.2/1.2
  Q-learning FO  41.9/1.3   38.5/3.0        42.6/0.7   42.4/1.3
  SARSA PO       39.2/1.4   39.7/1.5        40.8/0.5   41.5/0.3
  Q-learning PO  37.6/2.5   32.3/5.1        41.0/0.8   40.3/0.7

CONCLUSIONE
Con il learning rate a coseno lo shaping sulla distanza non serve, ne' in FO
ne' in PO: a fine training (ckpt >= 150) le 8 configurazioni stanno tutte fra
39 e 43 frutti con meno di 3 partite bloccate su 20. Le differenze a ckpt >=
100 dipendono da singoli checkpoint crollati (SARSA FO con shaping ep. 100,
Q-learning PO senza shaping ep. 140): rumore di un seed. Il loop risolto da
E8 era in gran parte instabilita', poi risolta da E13. Senza shaping in PO il
reward non contiene piu' informazione sulla posizione del frutto: il
confronto FO vs PO e' piu' pulito.

RUN DI CONTROLLO: NESSUNO SHAPING + LEARNING RATE FISSO 1e-3
(risultati_no_shaping_fixed_lr_200ep / _po_200ep), per il paper senza shaping
                     ckpt >= 100                       ckpt >= 150
                     frutti  rotti  bloccate           frutti  std
  SARSA FO     fisso  29.1   0/11    7.8               30.6   10.0
               coseno 37.7   0/11    4.1               43.2    0.6
  Q-learn. FO  fisso  21.8   3/11   10.5               26.6   13.0
               coseno 38.5   0/11    3.0               42.4    2.7
  SARSA PO     fisso  23.4   3/11    9.6               24.5   13.4
               coseno 39.7   0/11    1.5               41.5    0.7
  Q-learn. PO  fisso  14.4   6/11   14.1               17.5   14.8
               coseno 32.3   1/11    5.1               40.3    1.2
Anche senza shaping il learning rate fisso produce checkpoint instabili fino
alla fine (12 rotti su 44); il coseno li riduce a 1 su 44. Il risultato di E13
vale quindi con e senza shaping.


-------------------------------------------------------------------------------
E15. BASELINE EURISTICA (richiesta dalle specifiche del progetto)
-------------------------------------------------------------------------------
Data: 2026-11-27
Categoria:           baseline / valutazione
Stato:               ESEGUITO

MOTIVAZIONE
Le specifiche chiedono una baseline senza apprendimento, valutata
sull'ambiente, descritta in modo completo nel report e mostrata come linea
orizzontale nei grafici di apprendimento. Finora l'euristica era usata solo
come politica di esplorazione.

IMPLEMENTAZIONE
baseline.py (cartella principale): l'euristica degli script di training gioca
da sola, stesso protocollo della valutazione greedy degli agenti (20 partite
nuove, 500 passi, seed 12345, stesse metriche), in FO e PO. Ambienti di
sarsa_4/. Risultati in baseline/baseline_risultati.csv. Tempo: ~5 s.

RISULTATI (20 partite, seed 12345; con 100 partite e seed 0 cambia di < 5%)
                       baseline FO   agenti FO*      baseline PO   agenti PO*
  frutti                 71.8         42.4-43.2        27.6         40.3-41.5
  auto-collisioni         5.90         1.25-1.67        4.05         1.23-1.45
  segmenti persi         56.0          9.1-12.3        20.2         13.7-17.9
  lunghezza max          26.8         34.9-37.6        14.2         33.5-33.9
  partite bloccate        0/20         1.2-1.3          0/20         0.3-0.7
  * valutazione greedy, checkpoint >= 150, run finali (coseno, senza shaping)

LETTURA
- FO: la baseline mangia di piu' (72 contro 42-43) ma si morde ~4 volte piu'
  spesso e perde ~5 volte piu' segmenti: e' lo stesso comportamento "mangia
  e morditi" degli agenti originali (72 frutti, 6 auto-collisioni). Guarda un
  solo passo avanti, quindi da lunga si chiude nei vicoli. Gli agenti RL
  mangiano meno ma tengono il serpente molto piu' lungo.
- PO: gli agenti battono la baseline su tutte le misure (frutti +50%,
  auto-collisioni -65%, segmenti persi -10/-30%, serpente piu' del doppio
  piu' lungo): senza vedere il frutto la baseline cammina a caso.
- Rispetto all'obiettivo del progetto (frutti + poche auto-collisioni) gli
  agenti sono migliori in PO e offrono un compromesso diverso in FO.
