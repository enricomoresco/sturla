"""Ora per ora a inizio ottobre (slide 54 della presentazione): con che cosa va il riscaldamento al fondo.

Tabella oraria 3-11 ottobre: intensita' della corrente vicino al fondo (2-3.5 m), componenti lungo costa
(positiva verso 281 gradi) e verso costa vicino al fondo (2-5 m) e in alto (20-28 m), quota e valore del
massimo taglio fra 2.5 e 22 m, temperatura, residuo del livello, onde, vento e flusso di calore.

Correlazioni, 4-10 ottobre: riscaldamento su 6 ore (media mobile di 6 ore della temperatura, differenza
fra valori distanti 6 ore) contro ciascun forzante, anch'esso in media mobile di 6 ore e anticipato di
0, 6, 12, 24 o 36 ore. Sono circa 150 ore molto autocorrelate: indicazioni, non test.

  python analisi/rimescolamento/correlazioni_orarie.py --dati ... --risultati ... --uscita ...

Serve forzanti_orari.csv (forzanti.py) nella cartella di uscita.
Uscite: orario_4_10_ottobre.csv, correlazioni_orarie.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # anche con python -I

import os
import warnings

import numpy as np
import pandas as pd
import xarray as xr

import comune as C

FORZANTI = ["Ub", "Hm0", "ub_fondo", "tau_ls", "ls_fondo", "cs_fondo", "U10"]
RITARDI = (0, 6, 12, 24, 36)


def tabella_oraria(a) -> pd.DataFrame:
    c = xr.open_dataset(C.trova(a.risultati, "*_L2_currents.nc"))
    ok = np.isin(c.vel_qc.values, (1, 3)) & (c.deployment_qc.values == 1)[:, None]
    t = pd.DatetimeIndex(c.time.values) - pd.Timedelta("30min")
    z = c.z.values.astype(float)
    u, v = np.where(ok, c.u.values, np.nan), np.where(ok, c.v.values, np.nan)
    als, acs = np.deg2rad(C.ASSE_COSTA), np.deg2rad(C.VERSO_COSTA)
    LS, CS = u * np.sin(als) + v * np.cos(als), u * np.sin(acs) + v * np.cos(acs)

    def strato(X, z0, z1):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return np.nanmean(X[:, (z >= z0) & (z < z1)], 1)

    # quota del massimo taglio fra 2.5 e 22 m, profili lisciati su 3 celle
    du = np.gradient(pd.DataFrame(u).T.rolling(3, center=True, min_periods=2).mean().T.values, z, axis=1)
    dv = np.gradient(pd.DataFrame(v).T.rolling(3, center=True, min_periods=2).mean().T.values, z, axis=1)
    sh = np.hypot(du, dv)
    k = (z >= 2.5) & (z <= 22)
    shk = np.where(np.isfinite(sh[:, k]), sh[:, k], -1)
    D = pd.DataFrame(dict(ub_fondo=np.hypot(strato(u, 2, 3.5), strato(v, 2, 3.5)), ls_fondo=strato(LS, 2, 5),
                          cs_fondo=strato(CS, 2, 5), ls_alto=strato(LS, 20, 28), cs_alto=strato(CS, 20, 28),
                          h_taglio=z[k][np.argmax(shk, 1)], taglio=np.max(shk, 1)), index=t)
    h = C.serie_orarie(a.risultati)
    F = C.leggi(a.uscita, "forzanti_orari.csv")
    D = D.join(h[["temperatura", "temperatura_min", "temperatura_max", "residuo_m"]]).join(
        F[["Hm0", "Ub", "Tp", "tau_ls", "U10", "Q"]])
    D["range_T"] = D.temperatura_max - D.temperatura_min
    D = D.loc["2026-10-03":"2026-10-11"].copy()
    D["dT6"] = D.temperatura.rolling(6, center=True).mean().diff(6) / 6          # gradi all'ora
    return D


def main(argv=None):
    a = C.argomenti(__doc__, argv)
    D = tabella_oraria(a)
    D.to_csv(os.path.join(a.uscita, "orario_4_10_ottobre.csv"))
    X = D.loc["2026-10-04":"2026-10-10"]
    y = X.temperatura.rolling(6, center=True).mean().diff(6)
    righe = []
    print("correlazione del riscaldamento su 6 h con ... (anticipo del forzante in ore)")
    for c in FORZANTI:
        xs = X[c].rolling(6, center=True).mean()
        r = {lag: y.corr(xs.shift(lag)) for lag in RITARDI}
        righe.append(dict(forzante=c, **{f"{lag}h": v for lag, v in r.items()}))
        print(f"  {c:9s}", "  ".join(f"{lag:2d}h:{v:+.2f}" for lag, v in r.items()))
    pd.DataFrame(righe).to_csv(os.path.join(a.uscita, "correlazioni_orarie.csv"), index=False)


if __name__ == "__main__":
    main()
