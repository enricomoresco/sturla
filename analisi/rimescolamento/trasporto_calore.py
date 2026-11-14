"""Bilancio di temperatura nel modello MedFS vicino alla costa: quanto del riscaldamento al fondo
viene dal trasporto (avvezione orizzontale e verticale, cioe' downwelling) e quanto dal
mescolamento.

Per ogni punto della riga di griglia piu' vicina alla costa (44.354 N, 8.96-9.08 E):
  tendenza dT/dt = -u dT/dx - v dT/dy - w dT/dz + resto
con correnti e temperatura orarie e velocita' verticale giornaliera (interpolata ora per ora).
Il resto e' il mescolamento verticale (e in superficie il flusso di calore), piu' gli errori
dovuti all'uso di campi interpolati sui punti di griglia.

Servono in --dati i file del riquadro scaricati con scarica_dati.sh:
  medfs_riquadro_temperatura.nc, medfs_riquadro_correnti.nc,
  medfs_riquadro_velocita_verticale_giornaliera.nc

Uscita: trasporto_calore.csv (gradi cumulati su ogni finestra; spostamento verticale in m)
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

RIGA_COSTA = 44.3542                     # riga di griglia piu' vicina alla costa davanti a Sturla
LON_COSTA = (8.95, 9.09)                 # punti 8.958, 9.0, 9.042, 9.083
STRATI = {"0-20 m": (0.0, 21.0), "22-34 m": (21.0, 35.0), "34 m": (32.0, 35.0)}
FINESTRE = [("21-22 agosto", "2026-08-19 12:00", "2026-08-23 12:00"),
            ("4-7 ottobre, prima della mareggiata", "2026-10-04 12:00", "2026-10-07 12:00"),
            ("7-9 ottobre, mareggiata", "2026-10-07 12:00", "2026-10-09 12:00"),
            ("18-22 ottobre", "2026-10-18 12:00", "2026-10-22 12:00")]


def _derivata(F, asse, h):
    """Differenza centrata; da un lato solo dove manca il vicino (costa, fondo, bordo)."""
    Fp = np.roll(F, -1, asse); Fm = np.roll(F, 1, asse)
    idx = [slice(None)] * F.ndim
    idx[asse] = F.shape[asse] - 1; Fp[tuple(idx)] = np.nan
    idx[asse] = 0; Fm[tuple(idx)] = np.nan
    c = (Fp - Fm) / (2 * h); f = (Fp - F) / h; b = (F - Fm) / h
    return np.where(np.isfinite(c), c, np.where(np.isfinite(f), f, b))


def main(argv=None):
    a = C.argomenti(__doc__, argv)
    T = xr.open_dataset(C.trova(a.dati, "*medfs_riquadro_temperatura*.nc")).thetao
    cur = xr.open_dataset(C.trova(a.dati, "*medfs_riquadro_correnti*.nc"))
    W = xr.open_dataset(C.trova(a.dati, "*medfs_riquadro_velocita_verticale*.nc")).wo
    lat, lon, dep = T.latitude.values, T.longitude.values, T.depth.values.astype(float)
    tm = pd.DatetimeIndex(T.time.values)
    Re = 6.371e6
    dy = Re * np.deg2rad(float(np.median(np.diff(lat))))
    dx = Re * np.cos(np.deg2rad(RIGA_COSTA)) * np.deg2rad(float(np.median(np.diff(lon))))
    Tv = T.values; uv = cur.uo.sel(time=T.time).values; vv = cur.vo.sel(time=T.time).values
    # velocita' verticale: medie giornaliere riferite a mezzogiorno, interpolate ora per ora
    tw = (pd.DatetimeIndex(W.time.values) + pd.Timedelta("12h")).values.astype("int64")
    th = tm.values.astype("int64")
    wv = W.values
    wh = np.full_like(Tv, np.nan)
    for k in range(len(dep)):
        for j in range(len(lat)):
            for i in range(len(lon)):
                s = wv[:, k, j, i]
                if np.isfinite(s).all():
                    wh[:, k, j, i] = np.interp(th, tw, s)
    dTdt = np.gradient(Tv, 3600.0, axis=0)
    termini = {"tendenza": dTdt,
               "avvezione_lungo_x": -uv * _derivata(Tv, 3, dx),
               "avvezione_lungo_y": -vv * _derivata(Tv, 2, dy),
               "avvezione_verticale": -wh * (-np.gradient(Tv, dep, axis=1))}   # z verso l'alto
    termini["resto_mescolamento"] = dTdt - termini["avvezione_lungo_x"] - termini["avvezione_lungo_y"] - termini["avvezione_verticale"]
    jj = [int(np.argmin(np.abs(lat - RIGA_COSTA)))]
    ii = [i for i, x in enumerate(lon) if LON_COSTA[0] <= x <= LON_COSTA[1]]
    rows = []
    for nome, t0, t1 in FINESTRE:
        it = np.where((tm >= pd.Timestamp(t0)) & (tm < pd.Timestamp(t1)))[0]
        for strato, (z0, z1) in STRATI.items():
            kk = [k for k, d in enumerate(dep) if z0 <= d < z1]
            r = dict(finestra=nome, strato=strato)
            for k_, F in termini.items():
                x = F[np.ix_(it, kk, jj, ii)]
                r[k_] = float(np.nanmean(np.nansum(x, axis=0) * 3600.0))
            r["avvezione_orizzontale"] = r["avvezione_lungo_x"] + r["avvezione_lungo_y"]
            r["spostamento_verticale_m"] = float(np.nanmean(np.nansum(wh[np.ix_(it, kk, jj, ii)], axis=0) * 3600.0))
            rows.append(r)
    D = pd.DataFrame(rows)
    D.to_csv(os.path.join(a.uscita, "trasporto_calore.csv"), index=False)
    cols = ["tendenza", "avvezione_verticale", "avvezione_orizzontale", "resto_mescolamento", "spostamento_verticale_m"]
    for strato in STRATI:
        print(f"\nstrato {strato}, riga costiera {lat[jj[0]]:.3f} N, {lon[ii[0]]:.2f}-{lon[ii[-1]]:.2f} E (gradi cumulati)")
        print(D[D.strato == strato].set_index("finestra")[cols].round(2).to_string())


if __name__ == "__main__":
    main()
