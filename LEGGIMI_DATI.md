# Guida ai dati ADCP di Sturla: cosa è stato fatto e dove si trova ogni dato

Questa guida accompagna i dati dell'ADCP **Teledyne RDI Sentinel V** (n/s 597) posato sul
fondale davanti a Sturla (Genova), a circa 34 m di profondità e rivolto verso l'alto, con
il trasduttore a **0.6 m dal fondale**. Spiega:

1. da quali misure si parte;
2. tutte le operazioni fatte sui dati, nell'ordine;
3. dove si trova ogni grandezza (cartella, file, variabile, unità);
4. come leggere i flag di qualità.

I valori numerici esatti di ogni elaborazione (periodo in posizione, eventi, soglie usate,
percentuali di dati scartati per ciascuna causa) sono nel **rapporto generato
automaticamente** `archive/<ID>_RAPPORTO_ELABORAZIONE.md` (vedi § 5). Questa guida descrive
il metodo e l'organizzazione dei file.

`<ID>` nei nomi dei file è `<stazione>_<data inizio>_<data fine>`, ad esempio
`STURLA_20260710_20261109`. Il nome della stazione viene da `metadata.json`.

---

## 1. Dati di partenza

- **File grezzo**: `sturla_20260707T125042.pd0`, circa 3.7 GB, in formato binario PD0
  Teledyne. Contiene 2.035.683 *ping*: ogni ping è una misura istantanea, una ogni 0.5 s
  durante i burst.
- **Programmazione dello strumento** (ricavata dai dati):
  - un burst da **180 ping (1.5 min) ogni ora alle HH:20**, pensato per le correnti;
  - un burst da **1024 ping (8.5 min) ogni 2 ore alle ore pari**, pensato per le onde.
- **Cosa misura ogni ping**:
  - 4 beam inclinati di 25°, con 78 celle da 50 cm (centro della prima cella a 1.68 m dal
    trasduttore): velocità lungo ciascun beam, correlazione, intensità dell'eco;
  - beam verticale, con 78 celle da 50 cm: velocità, correlazione, intensità dell'eco;
  - sensori: pressione, temperatura, heading (bussola), pitch e roll (inclinometri).
- **Coordinate**: le velocità sono registrate **lungo i beam** (`coord_xfrm = 0`) e vanno
  trasformate in Est/Nord/Su.
- **Heading bias**: lo strumento era configurato con un heading bias di **3.50°** (comando
  EB). Di norma è la declinazione magnetica, già applicata all'heading registrato: in questo
  caso le direzioni sono riferite al Nord geografico. *Da confermare con chi ha programmato
  lo strumento.*

## 2. Operazioni sui dati, nell'ordine

### 2.1 Lettura e conversione (`convert`)
- Il file PD0 viene letto ensemble per ensemble, verificando la **checksum** di ognuno. Gli
  ensemble danneggiati vengono scartati e la lettura si risincronizza sul successivo. Sul
  file di Sturla: 0 errori, 0 byte non riconosciuti.
- Ogni ping viene decodificato **senza modificare i valori** e salvato in NetCDF
  (`out/raw/`). Le uniche conversioni sono di unità: mm/s → m/s, decapascal → dbar,
  centesimi di grado → gradi.

### 2.2 Periodo in cui lo strumento è sul fondo (`deployment`)
- Lo strumento ha registrato anche fuori dall'acqua. Il periodo in cui era **fermo sul
  fondo** viene trovato automaticamente da pressione e inclinazione, prese come mediana di
  ogni minuto. Lo strumento è considerato in posizione quando:
  - la pressione è vicina al valore di lavoro (circa 34.2 dbar);
  - l'inclinazione è piccola e stabile;
  - queste condizioni durano almeno 30 minuti.
- I dati **fuori posizione** (in aria, durante la posa o il recupero) **non vengono
  cancellati**: restano nei file con flag 4, causa `fuori_posizione`.
- **Movimenti del telaio**: si considera un movimento ogni salto di heading > 1° o di pitch
  o roll > 0.3° fra minuti consecutivi con dati. I dati che si sovrappongono a un movimento
  hanno flag 3, causa `movimento_telaio`.
- Risultato in `out/deployment.json` e nel grafico `out/figures/deployment.png`: in grigio i
  periodi fuori posizione, in arancione i movimenti.

### 2.3 Correnti (`currents`)
1. **Beam → strumento** con la matrice di trasformazione registrata dallo strumento (data
   type 0x3200). Coincide con quella teorica RDI per beam a 25°. Si usano solo le soluzioni
   a 4 beam.
2. **Strumento → Est/Nord/Su**, ping per ping, con heading, pitch e roll **di quel ping**.
   Si segue la convenzione Teledyne RDI per uno strumento rivolto verso l'alto. Per questo
   le velocità restano corrette anche dopo una rotazione del telaio.
3. Ogni ping con **correlazione < 64 counts** su almeno un beam viene scartato.
4. **Medie orarie** su ore di orologio, con tempo al **centro dell'ora** (00:30, 01:30, …).
   Si usano tutti i ping dell'ora che sono in posizione: 180 nelle ore dispari, 1204 nelle
   ore pari. Le ore senza misure restano come righe vuote (flag 9), quindi la serie è
   regolare.
5. **Controlli sulle medie**:
   - celle con meno del 50% di ping validi → rimosse;
   - celle con ping validi fra 50% e 80% → sospette;
   - velocità d'errore > 0.2 m/s → rimossa;
   - inclinazione > 20° → rimossa;
   - **celle troppo vicine alla superficie**, contaminate dal riflesso della superficie
     (side-lobe): oltre D·cos 25° − 1 cella, con D profondità del trasduttore → rimosse.
6. **Quota delle celle sul fondale** = distanza dal trasduttore + **0.6 m** (altezza del
   trasduttore sul fondale, parametro `--hab`). Anche la profondità totale è profondità
   del sensore + 0.6 m.
7. **Controllo indipendente della trasformazione**: sotto un'onda la velocità verticale è in
   fase con la velocità di salita della superficie (dη/dt), misurata con l'eco del beam
   verticale. Sui burst più energetici della prova sul file intero (versione precedente
   del codice):

   | Confronto | coerenza² | fase |
   |---|---|---|
   | w dai 4 beam e dη/dt | 0.94 | +17° |
   | velocità del beam verticale e dη/dt | 0.93 | −179° |
   | velocità orizzontali e η | 0.83 | — |

   La trasformazione e l'orientamento sono quindi verificati. Il −179° del beam verticale
   significa che la sua velocità è positiva verso il trasduttore. I valori della corsa
   definitiva sono in `out/check_transform.json`.

### 2.4 Onde (`waves`)
1. Si usano solo i **burst da 1024 ping** (uno ogni 2 ore). I burst da 180 ping sono troppo
   corti per uno spettro.
2. **Superficie dal beam verticale**: per ogni ping la superficie è il massimo dell'eco,
   localizzato fra due celle con interpolazione parabolica. Vengono rimossi i ping senza un
   picco netto e gli spike (oltre 4 deviazioni robuste).
3. **Spettro di Welch** (256 campioni, finestra di Hann, 50% di sovrapposizione) e da questo
   **Hm0** = 4√m0, **Tp**, **Tm01** e **Tm02**, integrati fra 0.04 e 1 Hz.
4. **Direzione** Dp (al picco) e Dm (media), da cross-spettri fra superficie e velocità
   orizzontali di una cella a circa il 70% della distanza dalla superficie. È la direzione
   **di provenienza**, in gradi da Nord in senso orario.
5. **Pressione**: viene usata solo per confronto. A 34 m la pressione vede solo le onde con
   periodo sopra ~8 s, quindi `Hm0_pressure` sottostima molto e **non va usata come Hm0**.

### 2.5 Temperatura
- È misurata dal sensore nella testa dello strumento: è la **temperatura dell'acqua al
  fondo**, in un solo punto, non un profilo.
- La **salinità non è misurata**: è un valore impostato nello strumento, usato solo per la
  velocità del suono.
- **Media oraria** di tutti i ping in posizione. Per ogni ora: media, deviazione standard,
  minimo, massimo, numero di ping e minuti campionati. Le misure esistono solo durante i
  burst, circa 9–18 minuti per ora.

### 2.6 Archivio (`archive`)
Produce i file NetCDF (convenzioni CF-1.8 e ACDD-1.3) e i CSV del § 3, la variabile di causa
per ogni flag e il rapporto di elaborazione.

---

## 3. Dove si trova ogni dato

### 3.1 Cartella `out/archive/`: i prodotti da archiviare

| Dato | File NetCDF → variabile | File CSV → colonna | Unità |
|---|---|---|---|
| **Velocità Est / Nord / verticale** (per ora e quota) | `<ID>_currents.nc` → `u`, `v`, `w` | `<ID>_currents_u.csv`, `_currents_v.csv` (righe = ore, colonne = quote `z…m`) | m/s |
| **Intensità e direzione della corrente** (per ora e quota) | `<ID>_currents.nc` → `speed`, `direction` | `<ID>_currents_speed.csv`, `_currents_direction.csv` | m/s; gradi (verso cui va) |
| Velocità d'errore, eco medio | `<ID>_currents.nc` → `err`, `echo` | — | m/s; counts |
| Quota delle celle sul fondale | `<ID>_currents.nc` → `z` | intestazione delle colonne `z…m` | m |
| **Corrente media sulla verticale** | `<ID>_currents.nc` → `u_davg`, `v_davg`, `speed_davg`, `dir_davg` | `<ID>_timeseries.csv` | m/s; gradi |
| **Flag delle velocità** e causa | `<ID>_currents.nc` → `vel_qc`, `vel_qc_cause` | `<ID>_currents_vel_qc.csv`, `_currents_vel_qc_cause.csv` | flag; bit |
| **Temperatura al fondo, oraria** (media, dev. std, min, max) | `<ID>_temperature_hourly.nc` → `temperature`, `temperature_std`, `temperature_min`, `temperature_max` | `<ID>_temperature_hourly.csv` | °C |
| Ping e minuti usati per la media oraria | `<ID>_temperature_hourly.nc` → `n_samples`, `sampled_minutes` | `<ID>_temperature_hourly.csv` | —; min |
| Flag della temperatura oraria e causa | `<ID>_temperature_hourly.nc` → `temperature_qc`, `temperature_qc_cause` | colonne `qc`, `qc_cause` | flag |
| Temperatura media nelle ore delle correnti | `<ID>_currents.nc` → `temperature` | `<ID>_timeseries.csv` | °C |
| **Pressione e profondità totale** | `<ID>_currents.nc` → `pressure`, `water_depth` | `<ID>_timeseries.csv` | dbar; m |
| **Assetto dello strumento** | `<ID>_currents.nc` → `heading`, `pitch`, `roll` | `<ID>_timeseries.csv` | gradi |
| Strumento in posizione e causa | `<ID>_currents.nc` → `deployment_qc`, `deployment_qc_cause` | `<ID>_timeseries.csv` | flag |
| **Telaio in movimento** | `<ID>_currents.nc` → `platform_motion` | `<ID>_timeseries.csv` | 0/1 |
| Ping mediati per ora, frazione in posizione | `<ID>_currents.nc` → `n_pings`, `in_position_frac` | `<ID>_timeseries.csv` | — |
| **Altezza d'onda Hm0, periodi Tp, Tm01, Tm02** | `<ID>_waves.nc` → `Hm0`, `Tp`, `Tm01`, `Tm02` | `<ID>_waves.csv` | m; s |
| **Direzione delle onde** (provenienza) | `<ID>_waves.nc` → `Dp`, `Dm` | `<ID>_waves.csv` | gradi |
| Flag di onde e direzioni, con causa | `<ID>_waves.nc` → `wave_qc`, `wave_qc_cause`, `direction_qc`, `direction_qc_cause` | `<ID>_waves.csv` | flag |
| **Spettro delle onde** | `<ID>_waves.nc` → `S` (beam verticale), `S_pressure`; asse `frequency` | — | m²/Hz; Hz |
| Profondità media nel burst | `<ID>_waves.nc` → `water_depth` | `<ID>_waves.csv` | m |
| Hm0 da pressione (solo confronto) | `<ID>_waves.nc` → `Hm0_pressure`, `fcut_pressure` | `<ID>_waves.csv` | m; Hz |
| Qualità del tracciamento della superficie | `<ID>_waves.nc` → `surface_found_frac` | — | frazione |
| **Descrizione completa dell'elaborazione** | `<ID>_RAPPORTO_ELABORAZIONE.md` | | |
| Questa guida | `LEGGIMI_DATI.md` | | |

Convenzioni dei tempi:
- **Correnti**: centro dell'ora (HH:30).
- **Temperatura oraria**: centro dell'ora nel NetCDF (con gli estremi in `time_bnds`);
  inizio dell'ora nel CSV (colonna `time_start`).
- **Onde**: inizio del burst.
- **Fuso orario**: quello dell'orologio dello strumento, da confermare (vedi
  `metadata.json`).

Nei CSV le righe che iniziano con `#` sono l'intestazione: stazione, strumento, periodo,
significato dei flag. I valori mancanti sono celle vuote; nei NetCDF il valore mancante è
`-9999`, dichiarato come `_FillValue`.

### 3.2 Cartella `out/`: file intermedi (utili per verifiche, non da archiviare)

| File | Contenuto |
|---|---|
| `raw/c78x4_cs50_p1_vb/*.nc` | **tutti i ping** decodificati, senza elaborazione (velocità lungo i beam, correlazione, eco, beam verticale, sensori) |
| `inventory.json` | resoconto della lettura: ensemble, errori, configurazione dello strumento |
| `deployment.json` | periodo in posizione, criteri usati, elenco dei movimenti del telaio |
| `check_transform.json` | risultati del controllo della trasformazione |
| `currents_avg.nc` | medie orarie dei ping, prima dei controlli di qualità |
| `currents_qc.nc` | correnti dopo i controlli, con i flag elementari (`flag_*`) |
| `waves_bulk.nc` | parametri d'onda per burst, da beam verticale e da pressione |
| `figures/` | grafici: `deployment.png`, `check_transform.png`, `currents_sections.png`, `currents_summary.png`, `waves_bulk.png`, `waves_spectrogram*.png`, `waves_burst_example.png`, `temperature_hourly.png`, `temperatura_e_Hm0.png` e `temperatura_e_corrente_fondo.png` (temperatura al fondo sovrapposta, con asse y secondario, a Hm0 e alla corrente al fondo), `overview_*.png` |

---

## 4. Come leggere i flag

Ogni dato ha un **flag principale**, secondo lo schema IOOS/QARTOD:

| flag | significato | cosa fare |
|---|---|---|
| 1 | buono | usare |
| 3 | sospetto | usare con cautela |
| 4 | cattivo | non usare (per le velocità il valore è già rimosso) |
| 9 | mancante | nessuna misura |

Accanto c'è la **causa**:
- nei NetCDF è la variabile `…_cause`, somma di bit, con il significato dei bit negli
  attributi `flag_masks` e `flag_meanings`;
- nei CSV è una colonna di testo, ad esempio `movimento_telaio+ping_validi_ridotti`.

Quando le cause sono più di una, il flag principale è quello della più grave.

| causa | flag | quando |
|---|---|---|
| `fuori_posizione` | 4 | strumento non fermo sul fondo (aria, posa, recupero) |
| `movimento_telaio` | 3 | il telaio stava ruotando o inclinandosi |
| `contaminazione_superficie` | 4 | cella troppo vicina alla superficie |
| `velocita_errore_alta` | 4 | velocità d'errore > 0.2 m/s |
| `inclinazione_eccessiva` | 4 | inclinazione > 20° |
| `pochi_ping_validi` | 4 | meno del 50% di ping validi nella cella |
| `ping_validi_ridotti` | 3 | fra 50% e 80% di ping validi |
| `superficie_trovata_parzialmente` | 3 | superficie trovata in meno del 90% dei ping del burst |
| `bassa_energia` | 3 | Hm0 < 0.25 m: direzione delle onde poco affidabile |
| `pochi_campioni` | 3 | meno di 30 s di misure nell'ora (temperatura) |
| `nessun_dato` / `valore_mancante` | 9 | nessuna misura |

Nel file della velocità per quota, le celle sopra la superficie risultano **sempre**
cattive (`contaminazione_superficie`). È normale: oltre ~30 m dal trasduttore l'ADCP vede
la superficie, non l'acqua.

---

## 5. Note sul dataset di Sturla

Quanto segue emerge dalle elaborazioni di prova del novembre 2026. Il rapporto generato
nella corsa definitiva riporta i valori esatti.

- **Periodo utile**: dal **10 luglio 2026 00:00** (lo strumento era già sul fondo all'avvio
  della registrazione) al **27 ottobre 2026 verso le 15:00**, quando è stato recuperato.
  Dal 28 ottobre al 9 novembre lo strumento ha registrato **in aria**: quei dati hanno
  flag 4.
- **Movimenti del telaio** durante le due mareggiate più forti dei 4 mesi:
  - 21–22 agosto: Hm0 ≈ 1.9 m, heading +9.5°;
  - 21–22 ottobre: Hm0 ≈ 2.0 m, heading +9.4°.

  Rotazioni minori (+1°/+2.5°) l'8–9 e il 20 ottobre. La pressione non cambia: il telaio
  ruota o si inclina, ma non sprofonda. Le velocità restano corrette perché ogni ping usa
  il proprio heading.
- **Offset della pressione**: in aria il sensore legge circa 0.13 dbar. Le profondità sono
  quindi sovrastimate di circa 13 cm, se non si usa l'opzione `--p-atm 0.13`.
- **Temperatura al fondo**: circa 17–19 °C da luglio a settembre. Dal 5 al 9 ottobre sale a
  ~23 °C, poi cala lentamente (~21.5 °C a fine ottobre). Correnti e variabilità
  giornaliera indicano che lo strato superficiale caldo, rimescolato, ha raggiunto il fondo
  (**rimescolamento autunnale**). Una conferma con profili indipendenti (CTD, Copernicus
  Marine) è consigliata.
- **Onde**: la mediana di Hm0 è circa 0.3 m. Le mareggiate principali vengono da Sud-Sud-Ovest
  (190–210°), con periodi di 6–9 s.

## 6. Limiti noti

- Direzione delle onde: vicino alla superficie i 4 beam distano fra loro circa quanto la
  lunghezza delle onde corte. La direzione è una stima di primo livello.
- I controlli di qualità sono **automatici**: i dati non sono stati validati manualmente.
- Fuso orario e declinazione magnetica vanno confermati con chi ha configurato lo strumento.

## 7. Come rifare l'elaborazione

Nella cartella del codice (l'altezza del trasduttore sul fondale, 0.6 m, è il valore
predefinito):

```bash
python process_pd0.py all sturla_20260707T125042.pd0 -o out --puv-cell auto --metadata metadata.json
```

Se i dati sono già convertiti in `out/`, si può saltare la parte lenta (circa 6–10 minuti
di lettura):

```bash
python process_pd0.py all --skip-convert -o out --puv-cell auto --metadata metadata.json
```

Il codice, con la documentazione dei comandi, è descritto in `README.md`.
