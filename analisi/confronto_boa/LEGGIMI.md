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
