"""Circolazione durante gli eventi: profili della corrente lungo costa (verso ONO) e verso costa
misurati dall'ADCP, trasporto di Ekman atteso dal vento e ordine di grandezza dell'abbassamento
del termoclino per downwelling (trasporto di Ekman distribuito sul raggio di deformazione interno).

Uscita: circolazione_profili.csv, downwelling.csv
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

FINESTRE = {"intero periodo": None,
            "4-7 ottobre, prima della mareggiata": ("2026-10-04 12:00", "2026-10-07 12:00"),
            "7-9 ottobre, mareggiata": ("2026-10-07 12:00", "2026-10-09 12:00"),
            "18-22 ottobre": ("2026-10-18 12:00", "2026-10-22 12:00"),
            "19-23 agosto": ("2026-08-19 12:00", "2026-08-23 12:00")}


def main(argv=None):
    a = C.argomenti(__doc__, argv)
    c = xr.open_dataset(C.trova(a.risultati, "*_L2_currents.nc"))
    ok = np.isin(c.vel_qc.values, (1, 3)) & (c.deployment_qc.values == 1)[:, None]
    t = pd.DatetimeIndex(c.time.values) - pd.Timedelta("30min")
    z = c.z.values.astype(float)
    a_ls, a_cs = np.deg2rad(C.ASSE_COSTA), np.deg2rad(C.VERSO_COSTA)
    ls = np.where(ok, c.u.values * np.sin(a_ls) + c.v.values * np.cos(a_ls), np.nan)
    cs = np.where(ok, c.u.values * np.sin(a_cs) + c.v.values * np.cos(a_cs), np.nan)
    good = np.isfinite(cs).mean(0) > 0.5
    F = C.leggi(a.uscita, "forzanti_orari.csv")
    prof, dw = {}, []
    for k, w in FINESTRE.items():
        m = np.ones(len(t), bool) if w is None else (t >= pd.Timestamp(w[0])) & (t < pd.Timestamp(w[1]))
        with np.errstate(all="ignore"):
            prof[(k, "lungo_costa_cm_s")] = 100 * np.nanmean(ls[m], 0)
            prof[(k, "verso_costa_cm_s")] = 100 * np.nanmean(cs[m], 0)
        if w is not None:
            x = F.loc[w[0]:w[1]]
            dw.append(dict(finestra=k, ekman_verso_costa_m2s=x.ekman_onshore.mean(),
                           ekman_cumulato_m3_per_m=x.ekman_onshore.sum() * 3600))
    P = pd.DataFrame(prof, index=pd.Index(z, name="quota_dal_fondo_m"))[good]
    P.to_csv(os.path.join(a.uscita, "circolazione_profili.csv"))
    # raggio di deformazione interno nel punto del modello all'inizio di ottobre (due strati)
    zm, tm, th, so = C.modello_punto(a.dati)
    i0 = int(np.argmin(np.abs(tm - pd.Timestamp("2026-10-04 12:00"))))
    r = C.rho_eos(th[i0], so[i0])
    drho = r[-1] - r[0]
    h1, h2 = 25.0, zm[-1] - 25.0
    c_int = np.sqrt(C.g * drho / C.RHO0 * h1 * h2 / (h1 + h2))
    R = c_int / C.FCOR
    D = pd.DataFrame(dw).set_index("finestra")
    D["raggio_deformazione_km"] = R / 1e3
    D["abbassamento_termoclino_m"] = D.ekman_cumulato_m3_per_m / R
    D.to_csv(os.path.join(a.uscita, "downwelling.csv"))
    print(P.iloc[::6].round(1).to_string())
    print(D.round(3).to_string())


if __name__ == "__main__":
    main()
