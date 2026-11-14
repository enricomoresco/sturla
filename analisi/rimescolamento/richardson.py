"""Numero di Richardson al termoclino: taglio della corrente misurato dall'ADCP su strati di 4 m,
stratificazione N2 dal modello 3D alle stesse profondita'. Sotto 0.25 il taglio puo' mescolare.

Uscita: richardson.csv (mediane e percentuali di ore per finestra e strato)
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

STRATI = [(2.2, 6.2), (6.2, 10.2), (10.2, 14.2), (14.2, 18.2), (18.2, 22.2), (22.2, 26.2)]
FINESTRE = {"19-23 agosto": ("2026-08-19 12:00", "2026-08-23 12:00"),
            "4-7 ottobre, prima della mareggiata": ("2026-10-04 12:00", "2026-10-07 12:00"),
            "7-9 ottobre, mareggiata": ("2026-10-07 12:00", "2026-10-09 12:00"),
            "18-22 ottobre": ("2026-10-18 12:00", "2026-10-22 12:00")}


def main(argv=None):
    a = C.argomenti(__doc__, argv)
    zm, tm, th, so = C.modello_punto(a.dati)
    r3 = C.rho_eos(th, so)
    c = xr.open_dataset(C.trova(a.risultati, "*_L2_currents.nc"))
    ok = np.isin(c.vel_qc.values, (1, 3)) & (c.deployment_qc.values == 1)[:, None]
    u = np.where(ok, c.u.values, np.nan); v = np.where(ok, c.v.values, np.nan)
    t = pd.DatetimeIndex(c.time.values) - pd.Timedelta("30min"); hab = c.z.values.astype(float)
    idx = pd.Series(np.arange(len(tm)), index=tm).reindex(t, method="nearest", tolerance=pd.Timedelta("31min"))
    parts = []
    for lo_, hi_ in STRATI:
        ia, ib = int(np.argmin(abs(hab - lo_))), int(np.argmin(abs(hab - hi_)))
        dz = hab[ib] - hab[ia]
        S2 = ((u[:, ib] - u[:, ia]) ** 2 + (v[:, ib] - v[:, ia]) ** 2) / dz ** 2
        da, db = C.H - hab[ia], C.H - hab[ib]
        N2 = np.full(len(t), np.nan); m = idx.notna().values; ii = idx.values[m].astype(int)
        ra = np.array([np.interp(da, zm, r3[i]) for i in ii]); rb = np.array([np.interp(db, zm, r3[i]) for i in ii])
        N2[m] = C.g / C.RHO0 * (ra - rb) / (da - db)
        parts.append(pd.DataFrame({"S2": S2, "N2": N2, "Ri": N2 / S2, "strato": f"{lo_:.0f}-{hi_:.0f} m dal fondo"}, index=t))
    D = pd.concat(parts)
    rows = []
    for w, (t0, t1) in FINESTRE.items():
        x = D[(D.index >= pd.Timestamp(t0)) & (D.index < pd.Timestamp(t1))]
        for s_, y in x.groupby("strato", sort=False):
            y = y.dropna()
            if len(y):
                rows.append(dict(finestra=w, strato=s_, N2_mediano=y.N2.median(), S2_mediano=y.S2.median(), Ri_mediano=y.Ri.median(),
                                 ore_Ri_sotto_025_pct=100 * (y.Ri < 0.25).mean(), ore_Ri_sotto_1_pct=100 * (y.Ri < 1).mean(),
                                 gia_mescolato=bool(y.N2.median() <= 1e-5)))
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(a.uscita, "richardson.csv"), index=False)
    print(R.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
