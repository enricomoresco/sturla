# Analisi del rimescolamento di ottobre a Sturla

Questi script separano le cause del rimescolamento osservato al fondo nell'ottobre 2026.
Confrontano l'energia necessaria a rimescolare la colonna con quella fornita dai processi locali:
- raffreddamento in superficie;
- vento;
- onde (Langmuir, frangimento, attrito sul fondo);
- attrito della corrente sul fondo.

La parte che resta è il trasporto laterale (downwelling). Usano i risultati dell'ADCP e tre fonti
esterne. Producono le tabelle e le figure della sezione «Cause del rimescolamento di ottobre» del
report.

## Come si lancia

```bash
python analisi/rimescolamento/esegui_tutto.py --dati dati_esterni --risultati out --uscita out_rimescolamento
```

- `--dati`: la cartella con i file esterni (tabella sotto). I nomi possono avere un prefisso
  qualsiasi.
- `--risultati`: la cartella dei risultati dell'ADCP, con dentro `preprocessing/` e
  `postprocessing/`. Va bene anche lo zip dei risultati estratto.
- `--uscita`: la cartella dove finiscono tabelle e figure.

Ci vuole circa un minuto. Ogni passo si può lanciare da solo con gli stessi argomenti (per
esempio `python analisi/rimescolamento/contributi.py --dati ... --risultati ... --uscita ...`),
nell'ordine di `esegui_tutto.py`.

Le costanti della stazione sono in `comune.py`: profondità, punti dei modelli, asse della costa,
eventi. Per un altro deployment si cambiano lì.

## Dati esterni usati

| File in `--dati` | Fonte | Contenuto |
|---|---|---|
| `*flussi_ERA5*.nc` | ERA5, Copernicus Climate Data Store; orario, 0.25° | radiazione solare e infrarossa netta, calore latente e sensibile, evaporazione, pioggia |
| `*vento*.nc` | Copernicus Marine, vento globale da scatterometri e modello (KNMI, prodotto 012_004); orario, 0.125° | vento a 10 m |
| `*temperatura_3D*.nc`, `*salinita_3D*.nc`, `*mixed_layer_depth*.nc` | Copernicus Marine, modello MedFS (MEDSEA_ANALYSISFORECAST_PHY_006_013); orario, 1/24° | temperatura, salinità e profondità del mixed layer, fino a 56 m |
| `medfs_riquadro_*.nc` (facoltativi, da `scarica_dati.sh`) | MedFS, riquadro 8.83–9.21 E, 44.23–44.48 N, 1–203 m | correnti e temperatura orarie, velocità verticale giornaliera, per `trasporto_calore.py` |

## Passi e uscite

| Script | Cosa calcola | Uscita |
|---|---|---|
| `forzanti.py` | flussi di calore e di galleggiamento, sforzo del vento e sua componente lungo costa, trasporto di Ekman, deriva di Stokes dagli spettri dell'ADCP, numero di Langmuir | `forzanti_orari.csv`, `forzanti_giornalieri.csv` |
| `profili_modello.py` | temperatura e stratificazione giornaliere del modello, termoclino del modello e dell'ADCP | `giornaliero_modello.csv` |
| `esperimenti_1d.py` | modello 1D con i soli processi locali, per i tre eventi, con processi accesi o spenti e vento ×1.5 e ×2 | `esperimenti_1d.csv`, `esperimenti_1d.pkl` |
| `scomposizione_giornaliera.py` | modello 1D ripartito ogni giorno dal modello 3D: parte locale e parte dovuta al trasporto | `scomposizione_giornaliera.csv` |
| `contributi.py` | energia necessaria e fornita da ogni processo, stratificazione tolta e calore arrivato di lato nel modello 3D, riscaldamento al fondo | `contributi_finali.csv` |
| `circolazione.py` | correnti lungo costa e verso costa misurate negli eventi, trasporto di Ekman, abbassamento atteso del termoclino | `circolazione_profili.csv`, `downwelling.csv` |
| `richardson.py` | numero di Richardson al termoclino, con il taglio misurato dall'ADCP | `richardson.csv` |
| `acqua_bassa.py` | profilo della deriva di Stokes, trasporti di Stokes e di Ekman, corrente verso costa in mareggiata | `stokes_ekman.csv`, `corrente_verso_costa.csv` |
| `trasporto_calore.py` | bilancio di temperatura nel modello vicino alla costa: tendenza = avvezione orizzontale + avvezione verticale (downwelling) + resto (mescolamento); facoltativo, serve il riquadro MedFS | `trasporto_calore.csv` |
| `gotm_forzanti.py` | file di ingresso per GOTM: correnti orarie dell'ADCP da usare per il rilassamento, velocità verticale giornaliera di MedFS, profilo iniziale con il fondo freddo come misurato; da lanciare a parte | `gotm/*.dat`, `gotm/LEGGIMI_GOTM.md` |
| `figure.py` | le figure del report (la quinta solo se c'è `trasporto_calore.csv`) | `figure/*.png` |

`col1d.py` contiene il modello 1D, `comune.py` le impostazioni. I test del modello 1D sono in
`tests/test_rimescolamento.py`.

## Metodo in breve

- **Energia necessaria.** È l'anomalia di energia potenziale della colonna di 34.5 m nel modello,
  moltiplicata per la profondità (Simpson e Bowers, 1981): il lavoro minimo per mescolarla tutta.
- **Raffreddamento.** Il calore perso agisce solo sotto lo strato mescolato di spessore h, quindi
  vale ρ·B·(H−h)/2. A questo si aggiunge la convezione penetrativa, 0.1·ρ·B·h. B è il flusso di
  galleggiamento netto, di giorno e di notte.
- **Vento.** Formula dei mari di piattaforma, con efficienza 0.023–0.039 (Simpson e Bowers, 1981;
  Simpson et al., 1990).
- **Onde.** Si contano tre termini:
  - turbolenza di Langmuir, 0.2·0.22·ρ·u\*²·Us0 (Belcher et al., 2012);
  - frangimento, 0.2·100·ρ·u\*³, solo per la frazione 0.3·Hs/h che arriva alla base dello strato
    mescolato (Craig e Banner, 1994; Terray et al., 1996);
  - attrito delle onde sul fondo.
- **Corrente.** Attrito sul fondo, con efficienza 0.0037.
- **Modello 1D.** È a bilancio di energia, del tipo di Kraus e Turner (1967). La radiazione solare
  penetra nella colonna. La convezione e l'energia dei processi sopra inglobano acqua nello strato
  mescolato dall'alto o dal fondo. Il costo del mescolamento usa l'equazione di stato
  linearizzata. Non c'è avvezione: la differenza con il modello 3D è il trasporto.
- **Trasporto.** Si stima in tre modi:
  - bilancio di calore della colonna del modello: variazione del contenuto di calore meno flusso in
    superficie;
  - ordine di grandezza dal trasporto di Ekman verso costa, distribuito sul raggio di deformazione
    interno;
  - direttamente, con le correnti del modello (`trasporto_calore.py`): avvezione orizzontale e
    verticale della temperatura nei punti di griglia più vicini alla costa. Il resto del bilancio è
    il mescolamento, più gli errori dovuti ai campi interpolati. Controllo: il 18–22 ottobre il resto
    dello strato 0–20 m vale −0.61 °C, contro −0.65 °C attesi dal raffreddamento ERA5.

## Limiti

- **Vento.** Il vento disponibile è probabilmente debole sotto costa: per questo ci sono gli
  esperimenti con il vento ×1.5 e ×2.
- **Punto del modello.** È 3–4 km al largo, su circa 90 m di fondale, e al fondo è più caldo di
  1–2 °C rispetto all'ADCP. Il modello 1D parte dal suo profilo.
- **Coefficienti.** Le efficienze sono di letteratura, ricavate in altri mari.

## Dati da aggiungere

`scarica_dati.sh` e `scarica_era5.py` scaricano da terminale i dati che servono a chiudere
l'analisi.

```bash
pip install copernicusmarine cdsapi
copernicusmarine login
bash analisi/rimescolamento/scarica_dati.sh dati_esterni 2026-07-01 2026-10-31
python analisi/rimescolamento/scarica_era5.py dati_esterni 2026 07,08,09,10
```

| Cosa | Da dove | A cosa serve |
|---|---|---|
| correnti (uo, vo), temperatura, salinità, livello del mare, orari, in un riquadro 8.85–9.20 E, 44.25–44.45 N, 0–200 m | MedFS, Copernicus Marine (account gratuito) | trasporto di calore attraverso i lati di un volume costiero, calcolato direttamente |
| velocità verticale (wo), media giornaliera (non esiste oraria) | MedFS, Copernicus Marine | downwelling del modello al termoclino; si può anche ricavare dalle correnti orarie per continuità |
| vento misurato all'aeroporto di Genova (METAR LIMJ), ogni 30 minuti | Iowa Environmental Mesonet, senza account | verificare se il vento era più forte di quello del modello, e rifare gli esperimenti 1D con il vento misurato |
| vento a 10 m, sforzo del vento, pressione al livello del mare | ERA5 (account sul Climate Data Store e file `~/.cdsapirc`) | confronto con il vento CMEMS; situazione meteorologica degli eventi; correzione barometrica del livello misurato dall'ADCP |

Si possono chiedere a mano, perché non hanno un download da terminale:
- **ARPAL** (rete OMIRL): vento delle stazioni di Genova più vicine a Sturla. Il portale OMIRL
  mostra i dati recenti; per le serie storiche serve una richiesta ad ARPAL.
- **ISPRA**, stazione della Rete Mareografica Nazionale nel porto di Genova (www.mareografico.it):
  vento, pressione e livello del mare.
- Eventuali **profili CTD** fatti alla posa e al recupero, o dal monitoraggio costiero di ARPAL.
