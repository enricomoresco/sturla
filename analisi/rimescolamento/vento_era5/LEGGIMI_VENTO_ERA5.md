# Run GOTM con il vento ERA5

Si estrae dentro `Sturla_GOTM_v7_2026/`. Aggiunge:
- `data/gotm_era5/`: sforzo del vento e vento a 10 m da ERA5;
- `templates/<caso>/`: 8 casi nuovi;
- `scripts/esegui_vento_era5.sh` e `scripts/controlla_vento_era5.py`.

Non sovrascrive nessun file esistente.

## Cosa cambia nei gotm.yaml

Rispetto ai template da cui partono, cambiano solo:
- `title`;
- `surface/fluxes/tx` e `ty`: `file: ../../data/gotm_era5/momentum_flux_era5.dat`, colonne 1 e 2.
  Lo sforzo è in N/m², positivo verso est e verso nord, come in `momentum_flux.dat`.
- `surface/u10` e `v10`: `file: ../../data/gotm_era5/wind10_era5.dat`, colonne 1 e 2. Con
  `fluxes: method: off` GOTM non li usa (nelle uscite `u10` vale 0): si cambiano solo per coerenza.
- Solo nei casi del 4 ottobre: `cvmix/interior/convection/use: true`, come nelle run di agosto e
  del 18 ottobre. Per questo il 4 ottobre ci sono anche i casi con il vento CMEMS: il confronto
  ERA5 contro CMEMS è fatto a parità di convezione e di eseguibile.

| Caso | Parte da | Vento |
|---|---|---|
| OTT04_R0_CMEMS | MATRICE R0 | CMEMS |
| OTT04_R0_ERA5 | MATRICE R0 | ERA5 |
| OTT04_R3_CMEMS | ESTENSIONE R3_tau6h_hb10m | CMEMS |
| OTT04_R3_ERA5 | ESTENSIONE R3_tau6h_hb10m | ERA5 |
| OTTOBRE18_R0_ERA5 | OTTOBRE18_R0 | ERA5 |
| OTTOBRE18_R3_ERA5 | OTTOBRE18_R3 | ERA5 |
| AGOSTO_R0_ERA5 | AGOSTO_R0 | ERA5 |
| AGOSTO_R3_ERA5 | AGOSTO_R3 | ERA5 |

Il calore, la radiazione solare, la pioggia, le correnti dell'ADCP, la w di MedFS e il
rilassamento di T e S restano quelli di prima.

Lo sforzo ERA5 è una media sull'ora che finisce all'istante indicato. Nel file è riportato a
metà ora, cioè 30 minuti prima. Il vento a 10 m è istantaneo e resta all'ora intera.

## Come si lancia

```bash
bash scripts/esegui_vento_era5.sh check
bash scripts/esegui_vento_era5.sh all
bash scripts/esegui_vento_era5.sh pacchetto
```

L'ultimo comando scrive `risultati_vento_era5.zip`, con l'ultima run di ogni caso.
