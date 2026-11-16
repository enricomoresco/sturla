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
| `medfs_riquadro_*.nc` (facoltativi, da `scarica_dati.sh`) | MedFS, riquadro 8.83–9.21 E, 44.23–44.48 N, 1–203 m | correnti e temperatura orarie, velocità verticale giornaliera, per `trasporto_calore.py` e per gli altri script sotto |
| file NetCDF con `era5` o `stepType` nel nome (facoltativi, da `scarica_era5.py`) | ERA5; orario, 0.25° | vento a 10 m, pressione e sforzo del vento, per `confronti_esterni.py`, `salti_orari.py` e `gotm_vento_era5.py`. Il Climate Data Store può consegnare due file (`...stepType-instant.nc` e `...stepType-avg.nc`): vanno bene entrambi nella cartella |

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

## Altri script: confronti, oscillazioni al fondo e run GOTM

Questi script non fanno parte di `esegui_tutto.py`, perché servono altri dati (vento ERA5, riquadro
MedFS) o le run GOTM. Si lanciano dopo `esegui_tutto.py`, con gli stessi argomenti più `--gotm`:

```bash
python analisi/rimescolamento/gotm_figura_matrice.py --dati dati_esterni --risultati out --uscita out_rimescolamento --gotm Sturla_GOTM_v7_2026
```

`--gotm` è la cartella che contiene le run GOTM: ogni run è una cartella con `gotm.yaml` e
`sturla.nc`, a qualunque profondità sotto `--gotm`. Gli script cercano le run per nome: `MATRICE_*` e
`EXTENSION_*` (4–13 ottobre), `FINSTAG_*_AGOSTO_*` e `FINSTAG_*_OTTOBRE18_*` (agosto e 18 ottobre),
`VENTO_*` (vento ERA5). Se un caso è stato lanciato più volte usano la run più recente.

| Script | Cosa calcola | Serve | Uscita |
|---|---|---|---|
| `confronti_esterni.py` | sforzo del vento lungo costa ERA5 contro CMEMS; corrente lungo costa dell'ADCP contro MedFS | vento ERA5, correnti del riquadro MedFS; con `--gotm` lo sforzo CMEMS è quello delle run (`momentum_flux.dat`), senza quello di `forzanti.py` | `figure/10_vento_era5_cmems.png`, `figure/11_correnti_adcp_medfs.png` |
| `diagnosi_medfs.py` | salti a ore fisse nelle serie orarie di MedFS; temperatura al fondo di MedFS contro l'ADCP | temperatura del riquadro MedFS | `diagnosi_medfs_salti_orari.csv` |
| `oscillazioni_fondo.py` | oscillazioni della temperatura al fondo sotto le 25 ore, salti orari, quote di varianza per banda | solo ADCP | `oscillazioni_fondo.csv`, `oscillazioni_fondo_bande.csv` |
| `salti_orari.py` | ora per ora nelle fasi rapide di ottobre: temperatura, correnti, livello, vento | vento ERA5 | `salti_orari.csv` |
| `correlazioni_orarie.py` | correlazione del riscaldamento su 6 ore con corrente, vento e onde, con anticipi di 0–36 ore | `forzanti_orari.csv` | `orario_4_10_ottobre.csv`, `correlazioni_orarie.csv` |
| `gotm_confronto_fondo.py` | temperatura al fondo: ADCP, MedFS al sensore e 7.6 m più in basso, run GOTM; bias, scarto e correlazione | `--gotm`, MedFS | `figure/7_fondo_adcp_copernicus_gotm.png`, `confronto_fondo_statistiche.csv` |
| `gotm_figura_4ottobre.py` | matrice del 4 ottobre: riscaldamento al fondo e isoterma per R0, R2, R3 | `--gotm` | `figure/6_gotm_matrice.png` |
| `gotm_figura_matrice.py` | tre finestre: misura contro R0 e R3, effetto massimo di ogni fattore | `--gotm` | `figure/8_matrice_gotm_sintesi.png`, `8_matrice_gotm_sintesi.csv` |
| `gotm_figura_interfaccia.py` | profondità dell'interfaccia da ADCP, MedFS e GOTM; oscillazioni al fondo | `--gotm`, temperatura del riquadro MedFS | `figure/9_interfaccia_e_oscillazioni.png` |
| `gotm_mixed_layer.py` | mixed layer di MedFS e GOTM con lo stesso criterio, 0.2 °C sotto la temperatura a 10 m | `--gotm`, riquadro e mixed layer MedFS, `giornaliero_modello.csv` | `mixed_layer_*.csv` |
| `gotm_vento_era5.py` | prepara gli 8 casi GOTM con il vento ERA5, partendo dalle run già fatte | `--gotm`, vento ERA5 | `vento_era5/`, `vento_era5_sturla.zip` |

`gotm_comune.py` contiene le funzioni per trovare e leggere le run; `vento_era5/` gli script per
lanciare i casi con il vento ERA5 e le istruzioni. I test di queste funzioni sono in
`tests/test_analisi.py`. L'analisi armonica delle correnti è in `analisi/marea_correnti/`.

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
