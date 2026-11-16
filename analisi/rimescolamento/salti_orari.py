"""Salti orari della temperatura al fondo nelle due fasi rapide di ottobre (slide 52 della presentazione):
ora per ora, temperatura al fondo, corrente lungo costa (positiva verso 281 gradi) e verso costa in
superficie e al fondo, residuo del livello e vento ERA5, il 7-9 e il 18-19 ottobre.

  python analisi/rimescolamento/salti_orari.py --dati ... --risultati ... --uscita ...

Serve in --dati il vento ERA5 (scarica_era5.py).
Uscita: salti_orari.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # anche con python -I

import os

import numpy as np
import pandas as pd

import comune as C


def componenti(vel, dire):
    """Componenti lungo costa (positiva verso ASSE_COSTA) e verso costa di una corrente data come
    intensita' e direzione verso cui va."""
    a = np.deg2rad(dire)
    u, v = vel * np.sin(a), vel * np.cos(a)
    ls, cs = np.deg2rad(C.ASSE_COSTA), np.deg2rad(C.VERSO_COSTA)
    return u * np.sin(ls) + v * np.cos(ls), u * np.sin(cs) + v * np.cos(cs)


def main(argv=None):
    a = C.argomenti(__doc__, argv)
    h = C.serie_orarie(a.risultati)
    h = h[h.deployment_qc == 1]
    E = C.era5_vento(a.dati)
    out = pd.DataFrame({"T fondo": h.temperatura})
    for k in ("sup", "fondo"):
        lungo, verso = componenti(h[f"corrente_{k}_vel"], h[f"corrente_{k}_dir"])
        out[f"{k} lungo (ONO+) cm/s"] = 100 * lungo
        out[f"{k} verso costa cm/s"] = 100 * verso
    out["residuo livello cm"] = 100 * h.residuo_m
    out["vento m/s"] = np.hypot(E.u10, E.v10).reindex(out.index)
    out["vento da (gradi)"] = (np.rad2deg(np.arctan2(-E.u10, -E.v10)) % 360).reindex(out.index)
    parti = []
    for t0, t1 in (("2026-10-07 18:00", "2026-10-09 12:00"), ("2026-10-18 06:00", "2026-10-19 12:00")):
        x = out.loc[t0:t1]
        print(x.iloc[::2].round(1).to_string())
        print()
        parti.append(x)
    pd.concat(parti).to_csv(os.path.join(a.uscita, "salti_orari.csv"))


if __name__ == "__main__":
    main()
