# Confronto con una boa ondametrica

Confronta le onde dell'ADCP con quelle di una boa vicina (CSV con le colonne `UTC`, `Hs (m)`,
`Tp (sec)`, `Ta (sec)`, `Dd (deg)`, `lat`, `lon`) e controlla l'orologio dell'ADCP.

```bash
python analisi/confronto_boa/confronto_boa.py --boa archive_data_2.csv --risultati out \
       --uscita out_confronto_boa --dati dati_esterni
```

- `--risultati`: la cartella del deployment con `postprocessing/` (serve `onde_parametri.csv`).
- `--dati` (facoltativo): la cartella con il livello del riquadro MedFS (`scarica_dati.sh`). Serve al
  controllo dell'orologio con le fasi di marea, che usa UTide.
- `--sposta-adcp-ore`: ore da sommare ai tempi dell'ADCP per portarli in UTC (default 0). Se
  `time_zone` di `metadata.json` indica l'orologio dello strumento, i risultati sono già in UTC; se le
  onde combaciano meglio con uno spostamento di un'ora o più, lo script lo segnala.

A Sturla l'orologio dell'ADCP era sull'ora legale italiana (UTC+2): le onde combaciavano con la boa
spostando i tempi di −2.5 ore e la marea del livello era in ritardo di 2.5–2.9 ore su MedFS. Da qui
`time_zone: "UTC+2"` in `metadati/STURLA_20260710_20261027.json`.

| Passo | Metodo |
|---|---|
| Orologio | Correlazione fra Hm0 dell'ADCP e Hs della boa spostando i tempi dell'ADCP da −6 a +6 ore, solo dove la boa misura ogni ora; con `--dati`, differenza fra le fasi di M2, S2, N2, K1 e O1 nel livello dell'ADCP e in quello di MedFS, divisa per la velocità angolare del costituente |
| Coppie | Ogni burst dell'ADCP (centro del burst, spostato in UTC) con il dato della boa più vicino entro 30 minuti |
| Statistiche | Differenza media (ADCP − boa), scarto quadratico medio, indice di dispersione, correlazione e rapporto mediano: Hm0 contro Hs (anche per classi di Hs), Tm01 e Tm02 contro Ta, Tp contro Tp, direzione di picco contro Dd; massimi delle mareggiate |

Uscite: `confronto_boa_coppie.csv`, `confronto_boa_statistiche.csv`, `confronto_boa_mareggiate.csv`,
`confronto_boa_fasi_marea.csv` (con `--dati`), `figure/14_onde_adcp_boa.png`,
`figure/15_orologio_adcp.png`.

## Versione autonoma, dai burst grezzi: `confronto_boa_burst.py`

Non usa i risultati del post-processing né il resto di `adcp_pd0`: ricalcola le onde dai burst e le
confronta con la boa. Si mette lo script nella cartella con il file dei burst del pre-processing
(`<ID>_L2_wave_bursts.nc`, anche rinominato `waveburst.nc`) e il CSV della boa, e si lancia
`python confronto_boa_burst.py` (oppure `python confronto_boa_burst.py <cartella>`). Risultati in
`<cartella>/confronto_boa_burst/`. Su un file sintetico dà gli stessi Hm0, periodi e direzioni del
post-processing (differenze sotto 1e-6). Se nella cartella (o in una sottocartella) c'è anche il
`deployment.json` di `preprocessing/` della stessa elaborazione, i movimenti del telaio sono segnati al
minuto; altrimenti si ricavano dai burst segnati con `platform_motion`.

1. **File.** Il NetCDF con `eta` e `n_samples` (preferito se il nome contiene «wave burst»), il CSV con le
   colonne `UTC` e `Hs (m)`, e se c'è un NetCDF con `zos` (livello di MedFS).
2. **Spettro di ogni burst.** Solo burst con `burst_qc` 1 o 3, abbastanza lunghi, con profondità valida e
   `eta` completa. Tolta la tendenza lineare, spettro di Welch: segmenti di 256 campioni (128 s a 2 Hz),
   finestra di Hann, sovrapposizione del 50%, media tolta in ogni segmento.
3. **Parametri.** Fra 0.04 Hz e il minimo fra 1.2 Hz e la frequenza di Nyquist: m0, m1, m2 con la regola
   dei trapezi; Hm0 = 4√m0, Tm01 = m0/m1, Tm02 = √(m0/m2), Tp = 1/fp con fp dal massimo dello spettro
   raffinato con una parabola sui tre punti intorno al massimo.
4. **Direzione.** Alla cella più vicina al 70% della profondità del sensore: u e v con i buchi interpolati
   se sono al più il 30%, tendenza tolta, cross-spettri elevazione–velocità. Dp = (270° − atan2(Re Cηv,
   Re Cηu)) modulo 360 alla frequenza di picco fra 0.04 e 0.6 Hz: direzione di provenienza, gradi da Nord.
   Coerenza² = (|Cηu|² + |Cηv|²)/(Sηη (Suu + Svv)), media pesata con Sηη entro ±0.025 Hz dal picco.
   Flag delle onde: 4 fuori posizione, 9 senza Hm0, 3 telaio in movimento o superficie trovata in meno del
   90% dei ping, altrimenti 1. Flag della direzione: 4 fuori posizione, 9 senza Dp, 3 telaio in movimento,
   Hm0 < 0.25 m o coerenza < 0.3, altrimenti 1. Tempo del burst: inizio più metà della durata.
5. **Orologio.** Se il file dei burst dice che i tempi sono già in UTC (`time_offset_applied_h`, oppure
   `time_zone` = `UTC`), spostamento 0; altrimenti il numero intero di ore che rende massima la
   correlazione del passo 7; oppure il valore di `--sposta-adcp-ore`.
6. **Boa.** Tempi letti come UTC, ordinati, senza duplicati.
7. **Correlazione con lo sfasamento.** Da −6 a +6 ore ogni 15 minuti: Hs della boa interpolata
   linearmente ai tempi dei burst spostati, solo dove i due dati della boa intorno distano al più 1.5 ore;
   correlazione con Hm0.
8. **Coppie.** Ogni burst con il dato della boa più vicino, se entro 30 minuti.
9. **Statistiche.** Differenza media (ADCP − boa), scarto quadratico medio, indice di dispersione
   (scarto / media della boa), correlazione, mediana del rapporto ADCP/boa: Hm0 contro Hs, in tutto e per
   classi di Hs; con Hs della boa ≥ 0.5 m, Tm01 e Tm02 contro Ta e Tp contro Tp (anche la quota entro 1 s);
   Dp − Dd riportata fra −180° e 180°: media, deviazione standard, quota entro 20°.
10. **Mareggiate e movimenti del telaio.** Mareggiate: tratti con Hs della boa ≥ 1.5 m mentre misura
    l'ADCP (uniti se a meno di 12 ore): massimo della boa, massimo dell'ADCP entro 6 ore, media dell'ADCP
    entro 90 minuti dal massimo della boa. Movimenti del telaio: gli episodi di `motion_events` del
    `deployment.json`, con rotazione = heading dopo − heading prima, riportata fra −180° e 180°. Il file si
    usa se il suo inizio del deployment coincide con `deployment_start` del file dei burst, o ne differisce
    di ore intere (orologio diverso, e allora si corregge), e se gli episodi cadono nel periodo dei burst;
    i tempi si spostano come quelli dei burst. Senza `deployment.json`: burst con `platform_motion` = 1,
    uniti se a meno di 6 ore; rotazione dall'heading dei burst fermi subito prima e subito dopo. Questi
    segnano solo i movimenti che cadono durante un burst, con la risoluzione dell'intervallo fra i burst.
11. **Marea (facoltativa).** Livello dell'ADCP = profondità media del sensore in ogni burst; MedFS nel punto
    valido più vicino alla boa; nel periodo comune, fit ai minimi quadrati di costante, tendenza e coseno e
    seno di M2, S2, N2, K1 e O1, uguale per le due serie; ritardo = differenza di fase / velocità angolare.
    L'incertezza viene dagli errori standard del fit e suppone rumore bianco: è indicativa.
12. **Uscite.** `figure/altezza_adcp_boa.png` (Hs della boa e Hm0 dell'ADCP nel tempo, con mareggiate e
    movimenti del telaio, e sotto il rapporto ADCP/boa di ogni coppia con la mediana mobile su 3 giorni),
    `figure/onde_adcp_boa.png` (dispersioni di altezza, periodo medio e direzione),
    `figure/orologio_adcp.png`, `parametri_onde_adcp.csv`, `correlazione_sfasamenti.csv`, `coppie.csv`,
    `statistiche.csv`, `mareggiate.csv`, `movimenti_telaio.csv`, `fasi_marea.csv`, `riepilogo.txt`.
