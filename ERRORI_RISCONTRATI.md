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
