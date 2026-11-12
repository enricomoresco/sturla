# Guida ai dati ADCP: cosa è stato fatto e dove si trova ogni dato

Questa guida accompagna i dati di un ADCP **Teledyne RDI Sentinel V** posato sul fondale e
rivolto verso l'alto, elaborati con il codice `adcp_pd0`. Vale per ogni deployment. Spiega:

1. come sono organizzati i file;
2. le operazioni fatte sui dati, nell'ordine;
3. dove si trova ogni grandezza (file, variabile, unità);
4. come leggere i flag di qualità.

I **valori specifici di un deployment** (periodo in posizione, movimenti del telaio, soglie
usate, percentuali di dati scartati per causa) sono nel rapporto generato a ogni
elaborazione, `preprocessing/<ID>_RAPPORTO_PREPROCESSING.md`; i risultati delle analisi sono
in `postprocessing/<ID>_SINTESI_RISULTATI.md` e `postprocessing/risultati.json`.

`<ID>` è l'identificativo del deployment: `<stazione>_<data inizio>_<data fine>` del periodo
in cui lo strumento era in posizione, per esempio `STURLA_20260710_20261027`. Il nome della
stazione viene da `metadata.json`.

---

## 1. Livelli dei dati e cartelle

| livello | contenuto | dove | da archiviare |
|---|---|---|---|
| **L0** | file PD0 originale dello strumento | fuori dalla cartella di output | sì, senza modifiche |
| **L1** | tutti i ping decodificati, senza elaborazione | `raw/<stream>/*.nc` | no: si rigenera dal PD0 |
| **L2** | dati puliti con flag di qualità | `preprocessing/` | sì |
| **L3** | prodotti derivati, tabelle e grafici | `postprocessing/` | sì |

Il **pre-processing** (`python preprocessing.py file.pd0 -o out`) produce L1 e L2. Il
**post-processing** (`python postprocessing.py -o out`) legge solo i file L2 e produce L3:
si può rifare in qualsiasi momento, anche con parametri diversi, senza rileggere il PD0.

---

## 2. Operazioni sui dati, nell'ordine

### Pre-processing

**2.1 Lettura.** Il file PD0 viene letto ensemble per ensemble verificando la **checksum**
di ognuno; gli ensemble danneggiati sono scartati e la lettura si risincronizza sul
successivo. Ogni ping è decodificato **senza modificare i valori** (solo conversioni di
unità: mm/s → m/s, decapascal → dbar, centesimi di grado → gradi).

**2.2 Periodo in posizione.** Lo strumento registra anche in aria, durante la posa e il
recupero. Il periodo in cui è **fermo sul fondo** viene trovato automaticamente da pressione
e inclinazione (mediane al minuto): pressione vicina al valore di lavoro, inclinazione
piccola e stabile, per almeno 30 minuti. I dati **fuori posizione non vengono cancellati**:
restano nei file con flag 4, causa `fuori_posizione`.

**2.3 Movimenti del telaio.** Ogni salto di heading > 1° o di pitch/roll > 0.3° fra minuti
consecutivi con dati è un movimento. I dati che si sovrappongono hanno flag 3, causa
`movimento_telaio`. Le velocità restano corrette, perché ogni ping è ruotato con il proprio
heading.

**2.4 Correnti.**
1. **Beam → strumento** con la matrice di trasformazione registrata dallo strumento; solo
   soluzioni a 4 beam.
2. **Strumento → Est/Nord/Su** ping per ping, con heading, pitch e roll di quel ping
   (convenzione Teledyne RDI per strumento rivolto verso l'alto).
3. Scarto dei ping con **correlazione** sotto la soglia (64 counts) su almeno un beam.
4. **Medie orarie** su ore di orologio, tempo al **centro dell'ora**, usando solo i ping in
   posizione. Le ore senza misure restano come righe vuote (flag 9): la serie è regolare.
5. **Controlli sulle medie**: celle con meno del 50% di ping validi rimosse, fra 50% e 80%
   sospette; velocità d'errore > 0.2 m/s rimossa; inclinazione > 20° rimossa; **celle
   troppo vicine alla superficie** (oltre D·cos θ − 1 cella, con D profondità del
   trasduttore e θ angolo dei beam) rimosse perché contaminate dal riflesso della superficie.
6. **Quota delle celle sul fondale** = distanza dal trasduttore + altezza del trasduttore
   sul fondale (parametro `--hab`, nei metadati).

**2.5 Controllo della trasformazione.** Sotto un'onda la velocità verticale è in fase con la
velocità di salita della superficie (dη/dt), che il beam verticale misura in modo
indipendente. Sui burst più energetici si calcolano coerenza e fase fra le due: coerenza
alta e fase vicina a zero confermano trasformazione e orientamento. Risultati in
`check_transform.json` e nel rapporto.

**2.6 Temperatura.** È quella del sensore nella testa dello strumento: **acqua al fondo**, un
solo punto, non un profilo. La salinità non è misurata. Statistiche orarie (media,
deviazione standard, minimo, massimo) di tutti i ping in posizione; le misure esistono solo
durante i burst, quindi sono riportati anche i minuti campionati.

**2.7 Serie dei burst d'onda.** Per ogni burst lungo (almeno 256 ping):
- **superficie dal beam verticale**: per ogni ping, massimo dell'eco localizzato fra due
  celle con interpolazione parabolica (`surface_range`);
- **elevazione pulita** (`eta`): valori oltre 4 deviazioni robuste rimossi, buchi
  interpolati se meno del 15% dei ping, media del burst tolta;
- **pressione** per ping;
- **velocità Est/Nord** per ping a tre quote (circa 50%, 70% e 85% della distanza dalla
  superficie), con la stessa trasformazione delle correnti.

### Post-processing

**2.8 Onde.** Spettro di Welch dell'elevazione (256 campioni, finestra di Hann, 50% di
sovrapposizione); **Hm0** = 4√m0, **Tp** (picco), **Tm01**, **Tm02**. **Direzione di
provenienza** al picco (Dp) e media (Dm) dai cross-spettri fra elevazione e velocità
orizzontali, stimata a ciascuna delle tre quote: il valore di riferimento è quello della
quota al 70%, la massima differenza fra le quote (`dir_level_spread`) indica l'incertezza.
La coerenza fra elevazione e velocità al picco (`dir_coherence`) è l'indice di qualità.
**Velocità orbitale al fondo** `Ub` dalla teoria lineare (2 volte la deviazione standard).
La **pressione** serve solo da confronto: in acqua profonda vede soltanto le onde lunghe
(`fcut_pressure` è la frequenza limite), quindi `Hm0_pressure` sottostima.

**2.9 Termoclino (stima indiretta).** L'ADCP non misura la temperatura lungo la colonna.
Per ogni giorno: profilo medio di u e v, lisciato su 2.5 m; taglio |dU/dz|; quota del
massimo, escluse le celle vicine al fondo e alla superficie. La stima è *affidabile* se il
massimo è netto (almeno 2 volte la mediana del profilo), non cade al bordo e si ritrova
alla stessa quota su due metà indipendenti dei dati del giorno. La banda 10–90% si ottiene
ripetendo la stima con 48 combinazioni di parametri. In parallelo, quota del **picco
dell'eco** corretto per spargimento e assorbimento (Brenner et al., 2023). Sono stime non
validate con profili di temperatura.

**2.10 Statistiche ed eventi.** Correnti a tre quote (celle valide più basse, metà colonna,
celle valide più alte entro 7 m dalla superficie) e media sulla verticale; spettro
rotatorio; analisi armonica del livello (M2, S2, N2, K1, O1); mareggiate (Hm0 sopra soglia,
superamenti a meno di 12 ore uniti); statistiche mensili; movimenti del telaio confrontati
con le onde.

---

## 3. Dove si trova ogni dato

### 3.1 `preprocessing/` — dati puliti (L2)

**`<ID>_L2_currents.nc`** — dimensioni `time` (centro dell'ora) × `z` (quota sul fondale, m)

| dato | variabile | unità |
|---|---|---|
| velocità verso Est, verso Nord, verticale | `u`, `v`, `w` | m/s |
| intensità e direzione (verso cui va la corrente) | `speed`, `direction` | m/s; gradi da Nord |
| velocità d'errore, intensità media dell'eco | `err`, `echo` | m/s; counts |
| **flag delle velocità** e causa | `vel_qc`, `vel_qc_cause` | flag; bit |
| frazione di ping validi per cella | `valid_ping_fraction` | 0–1 |
| corrente media sulla verticale | `u_davg`, `v_davg`, `speed_davg`, `dir_davg` | m/s; gradi |
| temperatura al fondo nell'ora | `temperature` | °C |
| pressione, sua deviazione standard, profondità totale | `pressure`, `pressure_std`, `water_depth` | dbar; dbar; m |
| assetto dello strumento | `heading`, `pitch`, `roll` | gradi |
| **strumento in posizione** e causa | `deployment_qc`, `deployment_qc_cause` | flag; bit |
| telaio in movimento nell'ora | `platform_motion` | 0/1 |
| ping mediati, frazione in posizione | `n_pings`, `in_position_frac` | —; 0–1 |

**`<ID>_L2_temperature.nc`** — dimensione `time` (centro dell'ora; estremi in `time_bnds`)

| dato | variabile | unità |
|---|---|---|
| temperatura al fondo: media, dev. std, minimo, massimo dell'ora | `temperature`, `temperature_std`, `temperature_min`, `temperature_max` | °C |
| ping usati, minuti dell'ora con misure | `n_samples`, `sampled_minutes` | —; min |
| flag e causa | `temperature_qc`, `temperature_qc_cause` | flag; bit |

**`<ID>_L2_wave_bursts.nc`** — dimensioni `time` (inizio del burst) × `sample` (ping nel
burst; `sample_time` = secondi dall'inizio) × `level` (quota delle velocità)

| dato | variabile | unità |
|---|---|---|
| elevazione della superficie, pulita | `eta` | m |
| distanza trasduttore–superficie, grezza | `surface_range` | m |
| pressione | `pressure` | dbar |
| velocità verso Est e verso Nord alle tre quote | `u`, `v` | m/s |
| quote delle velocità: distanza dal trasduttore, quota sul fondale, indice di cella | `level_range`, `level_height`, `level_cell` | m; m; — |
| frazione di ping con correlazione sufficiente, per quota | `corr_ok_frac` | 0–1 |
| ping nel burst, frazione con superficie trovata | `n_samples`, `surface_found_frac` | —; 0–1 |
| profondità del sensore e totale, assetto medio | `depth_sensor`, `water_depth`, `heading`, `pitch`, `roll` | m; gradi |
| **flag del burst** e causa; in posizione; telaio in movimento | `burst_qc`, `burst_qc_cause`, `in_position`, `platform_motion` | flag; bit; 0/1 |

Altri file: `<ID>_RAPPORTO_PREPROCESSING.md` (descrizione completa con i valori del
deployment), `deployment.json` (periodo in posizione, criteri, movimenti del telaio),
`inventory.json` (resoconto della lettura e configurazione dello strumento),
`check_transform.json`, `metadata.json` (metadati usati), `figure_qc/` (`deployment.png`,
`check_transform.png`, `panoramica_sensori.png`).

### 3.2 `postprocessing/` — prodotti derivati (L3)

**`<ID>_L3_waves.nc`** — dimensioni `time` (inizio del burst) × `frequency` (Hz) × `level`

| dato | variabile | unità |
|---|---|---|
| **altezza significativa**, periodo di picco, periodi medi | `Hm0`, `Tp`, `Tm01`, `Tm02` | m; s |
| **direzione di provenienza** al picco e media | `Dp`, `Dm` | gradi da Nord |
| direzione stimata a ogni quota; massima differenza fra le quote | `Dp_level`, `Dm_level`, `dir_level_spread` | gradi |
| coerenza elevazione–velocità al picco | `dir_coherence` | 0–1 |
| direzione di provenienza per frequenza | `direction_f` | gradi |
| **spettro di elevazione** | `S` | m²/Hz |
| velocità orbitale al fondo | `Ub` | m/s |
| Hm0 e spettro da pressione (solo confronto), frequenza limite | `Hm0_pressure`, `S_pressure`, `fcut_pressure` | m; m²/Hz; Hz |
| **flag delle onde** e causa; **flag della direzione** e causa | `wave_qc`, `wave_qc_cause`, `direction_qc`, `direction_qc_cause` | flag; bit |
| profondità, ping, superficie trovata, in posizione, telaio in movimento | `water_depth`, `n_samples`, `surface_found_frac`, `in_position`, `platform_motion` | |

**`<ID>_L3_thermocline.nc`** — dimensione `time` (inizio del giorno)

| dato | variabile | unità |
|---|---|---|
| quota del massimo taglio (stima del termoclino) | `z_shear` | m sul fondale |
| taglio massimo; rapporto con la mediana del profilo | `shear_max`, `shear_prominence` | 1/s; — |
| banda di sensibilità ai parametri (10° e 90° percentile) | `z_shear_p10`, `z_shear_p90` | m |
| differenza fra le stime su due metà dei dati; stima stabile | `split_half_difference`, `stable` | m; 0/1 |
| **stima affidabile**; massimo al bordo; massimo nella cella più bassa | `reliable`, `at_edge`, `at_bottom` | 0/1 |
| quota del picco dell'eco; quota massima con eco valido | `z_echo`, `z_echo_top_valid` | m |
| stime da taglio ed eco concordi | `agree` | 0/1 |

**`tabelle/`** (CSV, separatore virgola, tempi `AAAA-MM-GGThh:mm:ss`, valori mancanti vuoti)

| file | contenuto |
|---|---|
| `serie_orarie.csv` | per ogni ora: temperatura con flag, pressione, profondità, assetto, corrente a tre quote e media (intensità e direzione), livello, marea e residuo |
| `onde_parametri.csv` | per ogni burst: Hm0, periodi, direzioni, Ub, flag con causa in chiaro |
| `mareggiate.csv` | inizio, fine, durata, Hm0 massima, Tp, Dp, Ub, corrente massima, variazione di temperatura, movimento del telaio |
| `movimenti_telaio.csv` | assetto prima e dopo, Hm0 e Ub massimi nelle 3 ore precedenti |
| `statistiche_mensili.csv` | onde, correnti, temperatura e termoclino per mese |
| `termoclino.csv` | stime giornaliere (nomi delle colonne in italiano) |
| `marea_costituenti.csv` | ampiezza e fase di ogni costituente |
| `onde_distribuzione_Hm0_direzione.csv` | % dei burst per settore di provenienza e classe di Hm0 |
| `correnti_distribuzione_<strato>.csv` | % delle ore per settore di direzione e classe di intensità |

**`figure/`** (PNG)

| file | contenuto |
|---|---|
| `01_quadro_generale` | Hm0, corrente, temperatura, livello, con i movimenti del telaio |
| `02_assetto_strumento`, `03_qualita_dati` | heading, pitch, roll; dati validi nel tempo e cause dei flag |
| `10_correnti_sezioni` … `16_correnti_vettore_progressivo` | sezioni tempo–quota, profilo medio, rose, serie a tre quote, spettro, eco, vettore progressivo |
| `20_onde_serie` … `28_onde_distribuzione` | serie, spettrogramma, spettri medi, rosa, Hm0–Tp, direzione–frequenza, mareggiate, confronto con la pressione, frequenze di superamento |
| `30_temperatura` … `35_temperatura_periodi` | serie; sovrapposizioni con Hm0 e correnti (asse y secondario); periodo di massima variazione; periodi delle oscillazioni |
| `40_termoclino`, `41_termoclino_taglio` | quota stimata con Hm0 e temperatura; mappa del taglio |
| `50_livello_mare` | livello, marea ricostruita, residuo |

`risultati.json` raccoglie tutti i numeri chiave; `<ID>_SINTESI_RISULTATI.md` li riporta in
forma leggibile; `<ID>_risultati_per_report.zip` contiene tutto tranne le serie dei burst.

### 3.3 Convenzioni

- **Tempi**: correnti e temperatura al centro dell'ora; onde all'inizio del burst;
  termoclino all'inizio del giorno. Il fuso orario è quello dell'orologio dello strumento
  (attributo `time_zone`).
- **Quote**: `z` è la quota sul fondale, positiva verso l'alto.
- **Direzioni**: gradi da Nord in senso orario; correnti **verso cui vanno**, onde **da cui
  provengono**. Il riferimento (Nord magnetico o geografico) dipende dall'heading bias
  impostato nello strumento e dall'opzione `--declination`: vedi il rapporto.
- **Valori mancanti**: `-9999` nei NetCDF (dichiarato come `_FillValue`), celle vuote nei CSV.
- **Attributi globali** dei NetCDF: convenzioni CF-1.8 e ACDD-1.3; `data_level`, `product`,
  `deployment_id`, `software_version` (versione del codice), configurazione dello strumento
  (`instrument_*`) e parametri di elaborazione (`proc_*`).

---

## 4. Come leggere i flag

Ogni dato ha un **flag principale**, secondo lo schema IOOS/QARTOD:

| flag | significato | cosa fare |
|---|---|---|
| 1 | buono | usare |
| 3 | sospetto | usare con cautela |
| 4 | cattivo | non usare (per le velocità il valore è già rimosso) |
| 9 | mancante | nessuna misura |

Accanto c'è la **causa**, nella variabile `…_cause`: è la somma dei bit delle cause
presenti, con il significato dei bit negli attributi `flag_masks` e `flag_meanings`. Nei CSV
la causa è scritta in chiaro, per esempio `movimento_telaio+ping_validi_ridotti`. Quando le
cause sono più di una, il flag principale è quello della più grave.

| causa | flag | quando |
|---|---|---|
| `fuori_posizione` | 4 | strumento non fermo sul fondo (aria, posa, recupero) |
| `movimento_telaio` | 3 | il telaio stava ruotando o inclinandosi |
| `contaminazione_superficie` | 4 | cella troppo vicina alla superficie |
| `velocita_errore_alta` | 4 | velocità d'errore oltre la soglia |
| `inclinazione_eccessiva` | 4 | inclinazione oltre la soglia |
| `pochi_ping_validi` | 4 | meno del 50% di ping validi nella cella |
| `ping_validi_ridotti` | 3 | fra 50% e 80% di ping validi |
| `correlazione_bassa`, `percent_good_basso` | 4 | solo per dati già mediati dallo strumento |
| `superficie_trovata_parzialmente` | 3 | superficie trovata in meno del 90% dei ping del burst |
| `bassa_energia` | 3 | Hm0 sotto la soglia: direzione delle onde poco affidabile |
| `bassa_coerenza` | 3 | superficie e velocità poco coerenti al picco: direzione sospetta |
| `pochi_campioni` | 3 | meno di 30 s di misure nell'ora (temperatura) |
| `nessun_dato` / `valore_mancante` | 9 | nessuna misura |

Nel file delle correnti le celle sopra la superficie risultano **sempre** cattive
(`contaminazione_superficie`): è normale, lì l'ADCP vede la superficie e non l'acqua.

---

## 5. Limiti noti

- I controlli di qualità sono **automatici**: i dati non sono validati manualmente.
- **Direzione delle onde**: vicino alla superficie i 4 beam distano fra loro quanto la
  lunghezza delle onde corte. È una stima di primo livello; per i prodotti direzionali
  completi confrontare con il software del costruttore.
- **Termoclino**: stima indiretta, senza profili di temperatura di confronto; vicino al
  fondo il massimo del taglio può dipendere dall'attrito.
- **Velocità verticale media**: molto piccola rispetto all'errore dovuto all'inclinazione
  dello strumento; da non interpretare senza verifiche.
- **Pressione**: assunta relativa (azzerata in aria); l'eventuale offset letto in aria è
  riportato nel rapporto e si corregge con `--p-atm`.
- Fuso orario e declinazione magnetica vanno confermati con chi ha configurato lo strumento.

## 6. Come rifare l'elaborazione

```bash
python preprocessing.py <file.pd0> -o out --metadata metadata.json
python postprocessing.py -o out
```

Il primo comando riusa la lettura già fatta se `out/raw/` esiste; il secondo si può
rilanciare da solo. Opzioni e dettagli nel `README.md` del codice.
