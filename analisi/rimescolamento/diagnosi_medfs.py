"""Diagnosi delle serie orarie del modello MedFS vicino a Sturla (slide 47 e 53 della presentazione).

1) Salti a ore fisse: media del valore assoluto della differenza oraria della temperatura, per ora del
   giorno, in quattro punti della riga di griglia costiera (44.354 N) e a 10, 25 e 29 m. Un'ora con salti
   molto piu' grandi delle altre indicherebbe correzioni dell'assimilazione applicate una volta al giorno.
2) Temperatura al fondo del punto costiero con 30 m di fondale e del punto con 91 m di fondale a 33.9 m,
   contro l'ADCP, ogni 12 ore nelle tre finestre.

  python analisi/rimescolamento/diagnosi_medfs.py --dati ... --risultati ... --uscita ...

Serve in --dati il file della temperatura del riquadro MedFS (scarica_dati.sh).
Uscita: diagnosi_medfs_salti_orari.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # anche con python -I

import os

import numpy as np
import pandas as pd
import xarray as xr

import comune as C

PUNTI = {"9.000 (91 m)": 9.0, "9.042 (73 m)": 9.0417, "9.083 (62 m)": 9.0833, "9.125 (30 m)": 9.125}


def main(argv=None):
    a = C.argomenti(__doc__, argv)
    T = xr.open_dataset(C.trova(a.dati, "*medfs_riquadro_temperatura*.nc")).thetao
    dep = T.depth.values.astype(float)
    t = pd.DatetimeIndex(T.time.values)
    print("prime ore:", [str(x)[11:16] for x in t[:3]], " profondita':", np.round(dep[:22], 1))
    righe = []
    for nome, lon in PUNTI.items():
        x = T.sel(latitude=44.3542, longitude=lon, method="nearest")
        for zz in (10, 25, 29):
            s = pd.Series(x.sel(depth=zz, method="nearest").values, index=t)
            d = s.diff().abs()
            g = d.groupby(d.index.hour).mean()
            altre = g.drop(g.idxmax()).median()
            righe.append(dict(punto=nome, profondita_m=zz, salto_orario_medio=d.mean(), ora_salto_massimo=int(g.idxmax()),
                              salto_medio_a_quell_ora=g.max(), mediana_altre_ore=altre))
            print(f"{nome} {zz:2d} m: |dT/ora| medio {d.mean():.3f}; ora con il salto medio massimo {g.idxmax():02d}:00 "
                  f"({g.max():.3f}), mediana delle altre {altre:.3f}")
    pd.DataFrame(righe).to_csv(os.path.join(a.uscita, "diagnosi_medfs_salti_orari.csv"), index=False)
    # temperatura al fondo: punto costiero di 30 m e punto di 91 m a 33.9 m, contro l'ADCP
    h = C.serie_orarie(a.risultati)
    Ta = h.temperatura.where(h.deployment_qc == 1)
    x = T.sel(latitude=44.3542, longitude=9.125, method="nearest")
    ok = np.isfinite(x.isel(time=0).values)
    kb = np.flatnonzero(ok)[-1]
    fondo30 = pd.Series(x.values[:, kb], index=t)
    x91 = T.sel(latitude=44.3542, longitude=9.0, method="nearest")
    a34 = pd.Series([np.interp(C.PROF_SENSORE, dep, p) for p in x91.values], index=t)
    print(f"\npunto 44.354 N 9.125 E: ultimo livello {dep[kb]:.1f} m")
    for t0, t1 in (("2026-08-19", "2026-08-26"), ("2026-10-03", "2026-10-13"), ("2026-10-17", "2026-10-26")):
        df = pd.DataFrame({"ADCP 33.9 m": Ta, "MedFS fondo, punto 30 m": fondo30,
                           "MedFS 33.9 m, punto 91 m": a34}).loc[t0:t1].resample("12h").mean()
        print(df.round(2).to_string())


if __name__ == "__main__":
    main()
