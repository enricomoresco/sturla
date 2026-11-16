"""Profondita' del mixed layer nelle tre finestre (slide 53 della presentazione): MedFS (punti con 91 e 62 m
di fondale) e GOTM R0 e R3 con lo stesso criterio, 0.2 gradi sotto la temperatura a 10 m; mixed layer
del prodotto MedFS (mlotst); strato limite di superficie di CVMix in GOTM; stime dell'interfaccia
dall'ADCP. Medie giornaliere.

  python analisi/rimescolamento/gotm_mixed_layer.py --dati ... --risultati ... --gotm ... --uscita ...

Servono in --dati il file della temperatura del riquadro MedFS (scarica_dati.sh) e il mixed layer di
MedFS, e in --uscita giornaliero_modello.csv (profili_modello.py).
Uscite: mixed_layer_<finestra>.csv
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
import gotm_comune as G

FINESTRE = {"agosto": {"R0": "*_AGOSTO_R0", "R3": "*_AGOSTO_R3"},
            "4_ottobre": {"R0": "VENTO_*_OTT04_R0_CMEMS", "R3": "VENTO_*_OTT04_R3_CMEMS"},
            "18_ottobre": {"R0": "*_OTTOBRE18_R0", "R3": "*_OTTOBRE18_R3"}}


def main(argv=None):
    a = C.argomenti(__doc__, argv)
    radice = G.cartella_gotm(a)
    T3 = xr.open_dataset(C.trova(a.dati, "*medfs_riquadro_temperatura*.nc")).thetao
    dep = T3.depth.values.astype(float)
    tm = pd.DatetimeIndex(T3.time.values)
    med = {}
    for lon, nome in ((9.0, "MedFS 91 m"), (9.0833, "MedFS 62 m")):
        x = T3.sel(latitude=44.3542, longitude=lon, method="nearest").values
        med[nome] = pd.Series([G.mld_T(dep, p) for p in x], index=tm)
    mlot = C.mld_modello(a.dati)
    g = C.leggi(a.uscita, "giornaliero_modello.csv")
    for w, runs in FINESTRE.items():
        cols = {}
        for k, schema in runs.items():
            run = G.trova_run(radice, schema)
            cols[f"GOTM {k}"] = G.mld_run(run)
            with xr.open_dataset(os.path.join(run, "sturla.nc")) as d:
                cols[f"GOTM {k} zsbl"] = pd.Series(np.abs(d.cvmix_zsbl.squeeze().values),
                                                   index=pd.DatetimeIndex(d.time.values))
        t0, t1 = cols["GOTM R0"].index[[0, -1]]
        df = pd.DataFrame({**{k: v.loc[t0:t1] for k, v in med.items()}, "MedFS mlotst 91 m": mlot.loc[t0:t1], **cols})
        D = df.resample("1D").mean()
        D["ADCP prof. taglio max"] = g.prof_taglio.reindex(D.index)
        D["ADCP prof. eco max"] = g.prof_eco.reindex(D.index)
        print(f"\n=== {w}: profondita' del mixed layer (m), criterio 0.2 gradi sotto T a 10 m; media giornaliera")
        print(D.round(1).to_string())
        D.to_csv(os.path.join(a.uscita, f"mixed_layer_{w}.csv"))


if __name__ == "__main__":
    main()
