# ADCP Sentinel V (.pd0): pre-processing e post-processing di correnti, onde e temperatura

Codice Python per i file binari **PD0** di un ADCP Teledyne RDI Sentinel V posato sul fondo
(anche alcuni GB per file). Il lavoro è diviso in due passi:

| passo | comando | cosa fa | risultato |
|---|---|---|---|
| **pre-processing** | `python preprocessing.py file.pd0 -o out` | legge il PD0, trova il periodo in cui lo strumento è in posizione, pulisce i dati e li scrive in NetCDF con i flag di qualità. Nessuna analisi | `out/preprocessing/` (dati puliti, livello L2) |
| **post-processing** | `python postprocessing.py -o out` | dai soli dati puliti calcola onde, termoclino, statistiche, marea e produce tabelle e grafici | `out/postprocessing/` (prodotti derivati, livello L3) |

> Dove si trova ogni dato e come leggere i flag: [`LEGGIMI_DATI.md`](LEGGIMI_DATI.md)
> (copiata anche in `out/preprocessing/`).

## Installazione

```bash
pip install -r requirements.txt
```

## Uso

**0. Ispezione veloce** (facoltativa, non scrive niente): layout degli ensemble, stream trovati,
periodo, campionamento.

```bash
python process_pd0.py inspect sturla_20260707T125042.pd0 --range 45%:55%
```

**1. Pre-processing.** Per un file di 3.7 GB servono circa 10 minuti di lettura più alcuni
minuti per le medie.

```bash
python preprocessing.py sturla_20260707T125042.pd0 -o out --metadata metadata.json
```

- La lettura del PD0 viene salvata in `out/raw/`. Se si rilancia il comando sullo stesso file
  la lettura **non viene rifatta** (`--riconverti` per forzarla); si può anche omettere il
  file: `python preprocessing.py -o out`.
- `--metadata` è facoltativo: senza, si usa `metadata.json` se esiste nella cartella
  corrente o in `out/`, altrimenti il modello `metadata_template.json` non compilato. Se nei
  metadati manca il nome della stazione si usa quello del file PD0
  (`sturla_20260707T125042.pd0` → `STURLA`).

**2. Post-processing.** Meno di un minuto; si può rilanciare quante volte si vuole.

```bash
python postprocessing.py -o out
```

Alla fine indica il file **`out/postprocessing/<ID>_risultati_per_report.zip`**: contiene
grafici, tabelle, numeri chiave, il rapporto del pre-processing e i NetCDF piccoli (tutto
tranne le serie dei burst). È il file da inviare per scrivere il report.

`<ID>` è l'identificativo del deployment, `<stazione>_<inizio>_<fine>` del periodo in
posizione (es. `STURLA_20260710_20261027`); la stazione viene da `metadata.json`.

## Analisi del rimescolamento (dati esterni)

`analisi/rimescolamento/` contiene l'analisi delle cause del rimescolamento di ottobre 2026. Usa i
risultati dell'ADCP insieme a ERA5, al vento Copernicus Marine e al modello del Mediterraneo
MedFS. Contiene anche gli script per scaricare questi dati da terminale. Istruzioni in
[`analisi/rimescolamento/LEGGIMI.md`](analisi/rimescolamento/LEGGIMI.md).

```bash
python analisi/rimescolamento/esegui_tutto.py --dati dati_esterni --risultati out --uscita out_rimescolamento
```

## Livelli dei dati e cartelle

| livello | cosa | dove |
|---|---|---|
| L0 | file PD0 originale, mai modificato | fuori da `out/` |
| L1 | tutti i ping decodificati, senza elaborazione | `out/raw/<stream>/*.nc` |
| L2 | dati puliti con flag: correnti orarie, temperatura oraria, serie dei burst d'onda | `out/preprocessing/` |
| L3 | prodotti derivati: parametri e spettri d'onda, stima del termoclino, tabelle, grafici | `out/postprocessing/` |

```
out/
  inventory.json, deployment.json        resoconto della lettura; periodo in posizione
  raw/<stream>/*.nc                      L1
  preprocessing/
    <ID>_L2_currents.nc                  profili di corrente orari + serie al fondo
    <ID>_L2_temperature.nc               temperatura al fondo, statistiche orarie
    <ID>_L2_wave_bursts.nc               serie pulite dei burst d'onda (elevazione, pressione, velocità)
    <ID>_RAPPORTO_PREPROCESSING.md       tutto ciò che è stato fatto ai dati, con soglie e statistiche dei flag
    deployment.json, inventory.json, metadata.json, check_transform.json, LEGGIMI_DATI.md
    figure_qc/                           deployment.png, check_transform.png, panoramica_sensori.png
  postprocessing/
    <ID>_L3_waves.nc                     Hm0, periodi, direzione, spettri, velocità orbitale al fondo
    <ID>_L3_thermocline.nc               stima giornaliera della quota del termoclino
    tabelle/*.csv                        serie orarie, onde, mareggiate, statistiche mensili, marea, ...
    figure/*.png                         correnti, onde, temperatura, termoclino, assetto, livello, qualità
    risultati.json                       tutti i numeri chiave
    <ID>_SINTESI_RISULTATI.md            gli stessi numeri in forma leggibile
    <ID>_risultati_per_report.zip        da inviare per il report
```

Se un passo fallisce gli altri vengono comunque completati; l'errore è scritto in
`errori_preprocessing.txt` o `errori_postprocessing.txt` (quest'ultimo finisce nello zip).

## Metadati

Copiare `metadata_template.json` in `metadata.json` e compilarlo: nome della stazione,
coordinate, ente, altezza del trasduttore sul fondale, fuso orario dell'orologio dello
strumento, declinazione magnetica (e se è già impostata nello strumento). I valori finiscono
negli attributi dei NetCDF. La latitudine serve anche per il periodo inerziale nei grafici
delle correnti.

## Opzioni principali

`python preprocessing.py -h` e `python postprocessing.py -h` per l'elenco completo.

| pre-processing | significato | default |
|---|---|---|
| `--hab` | altezza del trasduttore (e del sensore di pressione) sul fondale [m] | dai metadati, altrimenti 0.6 |
| `--start`, `--end` | correzione manuale del periodo in posizione | automatico |
| `--range` | elabora solo una porzione del file, es. `45%:55%` (prove veloci) | tutto |
| `--riconverti` | rilegge il PD0 anche se `out/raw/` esiste | no |
| `--avg-minutes` | intervallo delle medie di corrente [min]; `0` = un profilo per burst | 60 |
| `--corr-min`, `--err-max`, `--tilt-max`, `--pg-min` | soglie dei controlli sulle correnti | 64 counts, 0.2 m/s, 20°, 50% |
| `--p-min` | pressione sotto cui lo strumento è fuori acqua [dbar] | 1.0 |
| `--declination` | declinazione magnetica [°] se **non** già impostata nello strumento (comando EB) | 0 |
| `--p-atm` | offset di pressione da sottrarre [dbar] | 0 |
| `--min-burst` | numero minimo di ping di un burst d'onda | 256 |
| `--max-burst` | i burst più lunghi (campionamento continuo) sono divisi in tratti di N ping | 4096 |
| `--no-vb` | non usare il beam verticale (onde dalla sola pressione) | usa il beam verticale |
| `--orientation up/down`, `--currents-stream`, `--waves-stream` | forzano orientamento e scelta degli stream | automatico |

| post-processing | significato | default |
|---|---|---|
| `--soglia-mareggiata` | soglia di Hm0 per la tabella delle mareggiate [m] | 1.0 |
| `--nperseg` | campioni per segmento dello spettro di Welch | 256 |
| `--fmin`, `--fmax` | banda di frequenza per i parametri d'onda e la direzione [Hz] | 0.04, 0.6 |
| `--hm0-dir-min` | Hm0 sotto cui la direzione è marcata sospetta [m] | 0.25 |
| `--kp2-min` | soglia della funzione di trasferimento della pressione | 0.05 |
| `--no-zip` | non creare lo zip dei risultati | crea lo zip |

## Cosa fa il pre-processing

1. **Lettura del PD0** con verifica della checksum di ogni ensemble; byte non validi saltati.
2. **Periodo in posizione**, trovato automaticamente da pressione e inclinazione (mediane al
   minuto): lo strumento registra anche in aria, in discesa e al recupero. I dati fuori
   posizione **restano nei file con flag 4**. Vengono anche individuati i **movimenti del
   telaio** (salti di heading > 1° o di pitch/roll > 0.3°): i dati corrispondenti hanno flag 3.
3. **Correnti**: ogni ping è portato in coordinate Est/Nord/Su con il proprio heading, pitch
   e roll (matrice beam→strumento registrata dallo strumento); i ping a bassa correlazione
   sono scartati; medie su ore di orologio; controlli su velocità d'errore, inclinazione,
   numero di ping validi e contaminazione vicino alla superficie. Ogni flag ha la sua causa.
4. **Temperatura al fondo**: statistiche orarie di tutti i ping in posizione.
5. **Burst d'onda**: per ogni burst lungo, superficie tracciata con l'eco del beam verticale
   e ripulita, pressione, velocità Est/Nord a tre quote (50%, 70%, 85% della distanza dalla
   superficie). Solo serie pulite: gli spettri si calcolano nel post-processing.
6. **Controllo indipendente della trasformazione**: sotto un'onda la velocità verticale è in
   fase con la velocità di salita della superficie (dη/dt) misurata dal beam verticale.
7. **Rapporto** con i valori effettivi del dataset.

## Cosa fa il post-processing

- **Onde**: spettro di Welch dell'elevazione; Hm0, Tp, Tm01, Tm02; direzione di provenienza
  al picco e media dai cross-spettri superficie–velocità, stimata a tre quote (la differenza
  fra le quote misura l'incertezza); coerenza come indice di qualità; direzione in funzione
  della frequenza; velocità orbitale al fondo (teoria lineare); Hm0 da pressione solo per
  confronto (in acqua profonda la pressione non vede le onde corte).
- **Termoclino** (stima indiretta): quota del massimo taglio verticale della corrente, giorno
  per giorno, con banda di sensibilità ai parametri e indicatore di affidabilità; confronto
  con il picco dell'eco corretto. Non validata con profili di temperatura.
- **Correnti**: statistiche a tre quote (superficie, metà colonna, fondo) e media sulla
  verticale, asse principale, spettro rotatorio (marea, moti inerziali), vettore progressivo.
- **Temperatura**: statistiche, periodo di massima variazione, periodi dominanti delle
  oscillazioni.
- **Livello del mare e marea**: analisi armonica della pressione (M2, S2, N2, K1, O1).
- **Eventi**: tabella delle mareggiate; movimenti del telaio confrontati con onde e velocità
  orbitale; statistiche mensili; percentuali dei flag per causa.

## Uso da Python / Jupyter

```python
import xarray as xr
from adcp_pd0 import StreamReader, pick_stream, extract_wave_bursts, wave_spectra

c = xr.open_dataset("out/preprocessing/STURLA_20260710_20261027_L2_currents.nc")
buone = c.speed.where(c.vel_qc.isin([1, 3]))            # solo dati buoni o sospetti
buone.plot(x="time", y="z")

b = xr.open_dataset("out/preprocessing/STURLA_20260710_20261027_L2_wave_bursts.nc")
w = wave_spectra(b, nperseg=512)                        # spettri con un'altra risoluzione

r = StreamReader("out", pick_stream("out", "waves"))    # ping grezzi (L1), per finestra temporale
ds = r.load(["pressure", "vb_echo"], "2026-08-21", "2026-08-23")
```

## Come funziona la lettura

Il file è mappato in memoria e scandito una volta registrando posizione e struttura di ogni
ensemble; byte spuri o ensemble troncati vengono saltati risincronizzandosi sull'header
`0x7F7F`. Gli ensemble con la stessa struttura sono decodificati in blocco con numpy, a
pezzi di ~150 MB: la memoria usata non dipende dalla dimensione del file. Data type letti:
fixed/variable leader, velocità, correlazione, eco, percent good, beam verticale del
Sentinel V (`0x0F01`, `0x0A00`, `0x0B00`, `0x0C00`, `0x0D00`), matrice di trasformazione
(`0x3200`). Gli ensemble sono raggruppati in *stream* con la stessa configurazione (es.
`c78x4_cs50_p1_vb` = 78 celle × 4 beam, celle da 50 cm, 1 ping per ensemble, con beam
verticale).

## Per un nuovo deployment

1. Copiare il file PD0 senza modificarlo e compilare un `metadata.json` per quel deployment.
2. `python preprocessing.py nuovo.pd0 -o out_<stazione>_<anno> --metadata metadata.json`
3. Controllare `figure_qc/deployment.png` (periodo in posizione) e
   `figure_qc/check_transform.png`, poi leggere il rapporto del pre-processing.
4. `python postprocessing.py -o out_<stazione>_<anno>`
5. Archiviare PD0, cartella `preprocessing/` e cartella `postprocessing/`; `raw/` si può
   rigenerare dal PD0.

## Test

`tests/synthetic.py` genera file PD0 sintetici (burst d'onda con altezza, periodo e
direzione noti, correnti, strumento in aria, discesa, rotazione del telaio, byte corrotti):

```bash
python -m pytest -q tests
```
