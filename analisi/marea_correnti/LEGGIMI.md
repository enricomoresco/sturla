# Analisi armonica delle correnti

Quanto delle correnti misurate dall'ADCP è marea, quota per quota, e quanto della temperatura al fondo.
È il metodo dell'ADCP trainato (Valle-Levinson, 2024, passo 6) applicato a una stazione fissa: con
110 giorni nello stesso punto si separano i singoli costituenti invece delle bande.

## Come si lancia

```bash
pip install utide
python analisi/marea_correnti/armonica_correnti.py --risultati out --uscita out_marea_correnti
python analisi/marea_correnti/figure_armonica.py --uscita out_marea_correnti
```

`--risultati` è la cartella del deployment con `preprocessing/` (serve il file `*_L2_currents.nc`).
Ci vogliono pochi secondi.

## Cosa calcola

| Passo | Metodo | Uscita |
|---|---|---|
| Ellissi di marea per quota | UTide (Codiga, 2011) su u e v insieme, a ogni quota con più del 50% di ore valide (flag 1 o 3): M2, S2, N2, K1, O1, M4, MS4; minimi quadrati ordinari, correzioni nodali, intervalli di confidenza al 95% con rumore colorato | `ellissi_marea_per_quota.csv` |
| Varianza spiegata dalla marea | 1 − [var(u − uₘ) + var(v − vₘ)] / [var(u) + var(v)], a ogni quota e sulla media verticale | `varianza_marea_per_quota.csv` |
| Marea nella temperatura al fondo | UTide sulla temperatura oraria, con tendenza: tutto il periodo, 10 luglio–30 settembre, 1–27 ottobre | nel file `armonica_correnti.pkl` |
| Bande in finestre mobili | Ogni 12 ore, su 3 giorni: minimi quadrati di un residuo più le bande diurna (24 h), inerziale e semidiurna (M2), in forma rotatoria; per la temperatura in forma reale | nel file `armonica_correnti.pkl` |
| Figure | (12) ampiezza e fase dei costituenti con la quota, varianza spiegata, marea nella temperatura; (13) bande nelle finestre mobili | `12_marea_correnti_temperatura.png`, `13_bande_finestre_mobili.png` |

La latitudine viene dai metadati del deployment; se mancano si usa 44.38° N. In 110 giorni K1 non si
separa da S1 (brezza) e P1, né S2 da K2: K1 comprende anche il ciclo giornaliero della brezza.
