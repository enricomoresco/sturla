# Preprocessing ADCP Sentinel V (.pd0): correnti e onde

> **Guida ai dati prodotti** (cosa è stato fatto, dove si trova ogni dato, come leggere i
> flag): [`LEGGIMI_DATI.md`](LEGGIMI_DATI.md), copiata anche in `out/archive/`.

Codice Python per leggere i file binari **PD0** di un ADCP Teledyne RDI Sentinel V
(anche alcuni GB), separare i dati di **corrente** da quelli di **onda**, fare un primo
controllo di qualità e produrre grafici di quick-look.

## Installazione

```bash
pip install -r requirements.txt
```

## Uso rapido

1. **Ispezione** (non scrive niente: capisce cosa c'è nei file; con `--max-mb` legge solo l'inizio):

```bash
python process_pd0.py inspect /percorso/dati/*.pd0 --max-mb 200
```

   Stampa i layout degli ensemble (quali data type ci sono: velocità, correlazione, eco,
   beam verticale, ...), gli *stream* trovati, periodo, intervallo di campionamento e numero
   di interruzioni (burst).

2. **Tutto in un comando** (conversione + QC correnti + onde + grafici):

```bash
python process_pd0.py all /percorso/dati/ -o out --metadata metadata.json
```

   oppure i singoli passi: `convert`, `deployment`, `currents`, `waves`, `plots`,
   `archive`, `check` (stessi parametri). Se i dati sono già convertiti in `out`, si
   rifà tutto il resto senza riconvertire con `all --skip-convert -o out ...`.

Parametri principali (`python process_pd0.py all -h` per l'elenco completo):

| opzione | significato | default |
|---|---|---|
| `--start`, `--end` | correzione manuale del periodo in posizione | automatico |
| `--trim` | elimina i dati fuori posizione invece di marcarli (flag 4) | no |
| `--redetect` | ricalcola il periodo in posizione | no |
| `--hab` | altezza del trasduttore (e del sensore di pressione) sul fondale [m] | 0.6 |
| `--p-atm` | offset di pressione da sottrarre [dbar] (se il sensore non è stato azzerato in aria) | 0 |
| `--declination` | declinazione magnetica [°] se **non** già impostata nello strumento (comando EB) | 0 |
| `--corr-min`, `--pg-min`, `--err-max`, `--tilt-max` | soglie QC correnti | 64, 50 %, 0.2 m/s, 20° |
| `--puv-cell` | cella (0-based) dello stream onde per la direzione, oppure `auto` | nessuna |
| `--dir-ref` | riferimento per la direzione: `auto`, `vb` (beam verticale), `p` (pressione) | auto |
| `--avg-minutes` | correnti dai ping singoli: medie su intervalli regolari di N minuti; `0` = un profilo per burst | 60 (orarie) |
| `--currents-stream`, `--waves-stream` | forza la scelta degli stream (nomi da `inspect`) | automatico |
| `--orientation up/down` | forza l'orientamento | dal fixed leader |

## Periodo in posizione (messa in acqua e recupero)

Lo strumento registra anche in aria, durante la discesa e durante il recupero. Il periodo
in cui è **fermo sul fondo** viene individuato **automaticamente** (passo `deployment`,
eseguito anche da `currents`/`waves`) su mediane al minuto di pressione e inclinazione:

- in acqua: pressione > `--p-min` (1 dbar);
- in posizione: pressione entro max(2 dbar, 10%) dalla pressione di lavoro (mediana in
  acqua), quindi la marea non conta; inclinazione < 15° e stabile (±3°);
- condizioni vere per almeno 30 minuti consecutivi; margine di sicurezza di 10 minuti
  dopo l'inizio e prima della fine.

Risultato in `out/deployment.json` e grafico `out/figures/deployment.png` (in grigio i
periodi fuori posizione). I dati fuori posizione **restano nei file con flag 4**
(`deployment_qc`, `vel_qc`, `wave_qc`); con `--trim` vengono eliminati. Se serve, il
risultato si corregge con `--start`/`--end`.

```bash
python process_pd0.py deployment -o out
```

## Elaborare solo una porzione del file

L'inizio e la fine del file (strumento in aria, messa in acqua, recupero) contengono dati
non utilizzabili. Per una prova preliminare conviene usare solo la parte centrale:

```bash
python process_pd0.py inspect sturla_20260707T125042.pd0 --range 45%:55%
python process_pd0.py all sturla_20260707T125042.pd0 -o out_prova --range 45%:55% --puv-cell auto
```

`--range INIZIO:FINE` accetta percentuali (`25%:75%`) o MB (`300:3400`); un estremo vuoto
indica inizio/fine file (`5%:`). La lettura parte dal primo ensemble valido dopo INIZIO.
Sul file intero non serve: i byte non validi vengono saltati e i dati fuori posizione
marcati automaticamente (vedi sopra). `--range` resta utile per prove veloci.

## Caso tipico Sentinel V in modalità onde (un solo stream di ping a 2 Hz)

Se `inspect` mostra **un solo stream** con `pings_per_ens 1`, dt 0.5 s e burst, e
`coordinate registrate: BEAM` (`coord_xfrm 0`), allora:

- **le correnti vengono ricavate dagli stessi ping**: ogni ping è portato in coordinate
  Earth con il proprio heading/pitch/roll (usando la matrice beam→strumento registrata dallo
  strumento, data type `0x3200`), i ping a bassa correlazione sono scartati e si fa la media
  su ogni burst (o su finestre di `--avg-minutes`). Il risultato intermedio è in
  `out/currents_avg.nc`, quello dopo QC in `out/currents_qc.nc`;
- viene stampato un **controllo della trasformazione**: la correlazione fra la velocità
  verticale ricavata dai 4 beam inclinati e quella misurata dal beam verticale. Un valore
  vicino a +1 indica matrice e orientamento coerenti (non verifica però l'heading);
- **in acqua profonda la pressione non vede le onde corte**: a ~34 m la funzione di
  trasferimento taglia sopra ~0.12 Hz (periodi < 8 s), quindi Hm0 da pressione sottostima
  molto. In questo caso il riferimento è **Hm0_vb** (superficie tracciata dal beam
  verticale); `surface_found_frac` indica la frazione di ping in cui la superficie è stata
  trovata. La direzione usa la superficie del beam verticale come riferimento
  (`--dir-ref auto`) e le velocità di una cella (`--puv-cell auto` = cella a ~70% della
  distanza dalla superficie). Attenzione: vicino alla superficie i 4 beam sono distanti fra
  loro (≈ 2·r·tan 25°), confrontabile con la lunghezza d'onda delle onde corte; la
  direzione è quindi una stima di primo livello.

Nel file di Sturla i burst sono di due tipi: **1024 ping (8.5 min) ogni 2 ore alle ore
pari**, usati per le onde, e **180 ping (1.5 min) ogni ora alle HH:20**, troppo corti per uno
spettro. Un profilo per burst darebbe una serie irregolare (intervalli 17/43/60 min): le
correnti sono quindi **medie orarie** (ore di orologio, tempo = centro dell'ora, tutti i ping
dell'ora: 180 o 1204), con le ore senza dati come righe vuote (flag 9). Nelle ore a cavallo
di inizio/fine deployment si usano solo i ping in posizione (`in_position_frac`). L'analisi onde usa solo i burst con
almeno 256 ping (`n_short_bursts_skipped` in `waves_bulk.nc` conta gli esclusi); le correnti
usano tutti i burst (`n_pings` in `currents_avg.nc` dice su quanti ping è fatta ogni media).

**Controllo della trasformazione.** Lo strumento è sul fondo, rivolto verso l'alto: come
riferimento indipendente si usa la superficie η(t) tracciata dall'eco del beam verticale.
Per la teoria lineare la velocità verticale delle onde è in fase con dη/dt a ogni quota,
quindi `corr(w dai 4 beam, dη/dt)` alta e positiva conferma trasformazione e orientamento
(valore negativo = orientamento/segno sbagliato). Lo stesso test sulla velocità del beam
verticale dice se quella velocità è utilizzabile. Tabella stampata dal passo `currents`,
grafico in `figures/check_transform.png` (burst più energetico).

`--hab` = altezza dal fondale del trasduttore/sensore di pressione: a Sturla 0.6 m (valore predefinito).

Comando consigliato:

```bash
python process_pd0.py all sturla_20260707T125042.pd0 -o out --puv-cell auto --metadata metadata.json
```

## Output (`out/`)

```
out/
  inventory.json          # file, layout, ensemble scartati, stream e loro configurazione
  raw/<stream>/*.nc       # dati grezzi decodificati, NetCDF compressi (uno o più "part" per file)
  currents_qc.nc          # correnti dopo QC: u, v, w, err, speed, direction, medie sulla verticale, flag
  waves_bulk.nc           # per burst: Hm0, Tp, Tm01, Tm02, profondità, f taglio, spettri S(f)
                          #   (+ Hm0_vb... dal beam verticale, + Dp/Dm se --puv-cell)
  figures/*.png           # quick-look
```

Gli stream hanno un nome che descrive la configurazione, ad es. `c30x4_cs50_p60` =
30 celle × 4 beam, celle da 50 cm, 60 ping per ensemble; il suffisso `_vb` indica la presenza
del beam verticale. La classificazione automatica: **waves** = stream con campionamento
≤ 2.5 s e interruzioni fra burst; **currents** = lo stream restante con più ensemble.

## Archivio (NetCDF CF e CSV)

Copiare `metadata_template.json` in `metadata.json`, compilarlo (stazione, coordinate,
ente, licenza, altezza del telaio, fuso orario dell'orologio, ...) e lanciare:

```bash
python process_pd0.py archive -o out --metadata metadata.json
```

(il passo è incluso anche in `all`). In `out/archive/`:

| file | contenuto |
|---|---|
| `<ID>_<inizio>_<fine>_currents.nc` | profili u, v, w, intensità, direzione, eco (tempo × quota sul fondale), flag `vel_qc`; serie al fondo: **temperatura**, pressione, profondità totale, heading/pitch/roll, n. ping; corrente media sulla verticale |
| `<ID>_..._waves.nc` | Hm0, Tp, Tm01, Tm02, Dp, Dm, flag `wave_qc`/`direction_qc`, profondità, spettri S(f) |
| `<ID>_..._timeseries.csv` | temperatura, pressione, profondità, assetto, corrente media sulla verticale |
| `<ID>_..._waves.csv` | parametri d'onda e flag |
| `<ID>_..._currents_{u,v,speed,direction,vel_qc}.csv` | matrici tempo × quota |
| `<ID>_..._temperature_hourly.nc` / `.csv` | **temperatura al fondo, media oraria** da tutti i ping (media, dev. std, min, max, n. ping, minuti campionati, flag `temperature_qc`) |

Ogni esecuzione di `archive` scrive anche **`<ID>_..._RAPPORTO_ELABORAZIONE.md`**: descrive con i
valori reali del dataset tutto ciò che è stato fatto ai dati (file di partenza, configurazione,
periodo in posizione e movimenti del telaio, trasformazioni, medie, soglie di QC, risultati
del controllo della trasformazione, statistiche dei flag per causa, file prodotti, limiti).

**Flag e cause.** Ogni flag (QARTOD 1/3/4/9) ha accanto una variabile `*_cause` (NetCDF) o
una colonna `*_cause` (CSV) con il motivo: `fuori_posizione`, `movimento_telaio`,
`contaminazione_superficie`, `velocita_errore_alta`, `inclinazione_eccessiva`,
`pochi_ping_validi`, `ping_validi_ridotti`, `superficie_trovata_parzialmente`,
`bassa_energia`, `pochi_campioni`, `nessun_dato`, ... (nei NetCDF come bit, con
`flag_masks`/`flag_meanings`). **Movimento del telaio**: salti di heading > 1° o di
pitch/roll > 0.3° fra minuti consecutivi con dati, elencati in `deployment.json` ed
evidenziati in arancione in `figures/deployment.png`; i dati che vi si sovrappongono hanno
flag 3.

I NetCDF seguono CF-1.8 / ACDD-1.3 (standard_name, unità, `height_above_sea_floor`,
metadati globali, parametri di elaborazione con prefisso `proc_`). Flag secondo IOOS/QARTOD:
1 buono, 2 non valutato, 3 sospetto, 4 cattivo, 9 mancante. La **temperatura** è quella del
sensore nella testa dello strumento (acqua al fondo); la salinità non è misurata. Nella
media oraria si usano solo i ping in posizione (flag 1; 3 se meno di 30 s di dati); le ore
interamente fuori posizione hanno flag 4. Lo strumento misura solo durante i burst, quindi
la media oraria è sui minuti campionati (`sampled_minutes`). Grafico:
`figures/temperature_hourly.png`.

## Uso da Python / Jupyter

```python
from adcp_pd0 import StreamReader, pick_stream, qc_currents, analyze_waves

r = StreamReader("out", pick_stream("out", "waves"))
ds = r.load(["pressure", "vb_echo"], "2025-06-10", "2025-06-11")   # caricamento per finestra temporale
r.times()                                                          # solo i tempi (leggero)

cur = StreamReader("out", pick_stream("out", "currents")).load()   # xarray.Dataset
qc = qc_currents(cur, corr_min=70, err_max=0.1, declination=2.5)
qc.speed.plot(x="time", y="range")
```

## Come funziona

- **Lettura** (`adcp_pd0/pd0.py`): il file è mappato in memoria (`mmap`) e scandito una
  volta registrando solo posizione e struttura di ogni ensemble; byte spuri o ensemble
  troncati vengono saltati risincronizzandosi sull'header `0x7F7F`. Gli ensemble con la
  stessa struttura sono poi decodificati in blocco con numpy (a pezzi di ~150 MB) e la
  checksum è verificata per tutti. Velocità ~20 s/GB; la RAM usata non dipende dalla
  dimensione del file.
- **Data type letti**: fixed/variable leader, velocità, correlazione, eco, percent good,
  beam verticale del Sentinel V (leader `0x0F01`, velocità `0x0A00`, correlazione `0x0B00`,
  eco `0x0C00`, percent good `0x0D00`), matrice di trasformazione (`0x3200`). I blocchi
  `sentinel_0x7000`…`0x7004` sono elencati da `inspect` ma non decodificati.
- **Correnti** (`currents.py`): taglio temporale, scarto fuori acqua (p < 1 dbar),
  trasformazione in coordinate Earth se registrato in beam/instrument, rotazione per la
  declinazione, maschere su correlazione, percent good, velocità d'errore, tilt e
  contaminazione side-lobe vicino alla superficie (r > D·cos θ − 1 cella).
- **Onde** (`waves.py`): burst identificati dalle interruzioni nel campionamento; per ogni
  burst spettro di Welch della pressione, correzione con la funzione di trasferimento
  della teoria lineare Kp = cosh(k·hab)/cosh(k·h) fino alla frequenza di taglio
  (Kp² ≥ `--kp2-min`), poi Hm0 = 4√m0, Tp, Tm01, Tm02. Se c'è il beam verticale la
  superficie è stimata per ogni ping dal massimo di eco (interpolazione parabolica) e se ne
  ricava uno spettro indipendente (`*_vb`). Con `--puv-cell` la direzione Dp/Dm (di
  provenienza, °N orari) è calcolata col metodo PUV dai cross-spettri pressione–velocità.

## Da verificare sui propri dati

- Lanciare prima `inspect`: controllare che gli stream trovati corrispondano alla
  configurazione di deployment (celle, dimensione cella, ping, intervallo burst).
- Se nei file compaiono data type `unknown_0x....`, il resto si legge comunque, ma quei
  blocchi sono ignorati. Le trasformazioni beam/instrument → Earth seguono la
  convenzione RDI: se i dati sono già in Earth (caso più comune, `coord=earth`) non vengono
  usate.
- La pressione RDI è normalmente azzerata in aria (gauge); in caso contrario usare `--p-atm`.
- Hm0 dal beam verticale è una stima approssimata (risoluzione legata alla dimensione delle
  celle); per i prodotti direzionali completi confrontare con Teledyne Velocity.

## Test

Il generatore `tests/synthetic.py` crea file PD0 sintetici (correnti di marea + burst
d'onda con Hm0 = 1 m, Tp = 8 s, provenienza 225°, beam verticale, byte corrotti):

```bash
python -m pytest -q tests
```
