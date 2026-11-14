"""Profili giornalieri del modello MedFS nel punto a mare della stazione, confrontati con l'ADCP:
temperatura a varie quote, stratificazione 0-34 m, mixed layer, quota del massimo gradiente di
temperatura (termoclino del modello) e stime del termoclino dall'ADCP (taglio ed eco).

Uscita: giornaliero_modello.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # anche con python -I

import os

import numpy as np
import pandas as pd

import comune as C


def main(argv=None):
    a = C.argomenti(__doc__, argv)
    z, tm, th, so = C.modello_punto(a.dati)
    r = C.rho_eos(th, so)
    k34 = int(np.argmin(np.abs(z - 33.77)))
    D = pd.DataFrame(index=tm)
    for zz in (z[0], 10.54, 19.4, 26.2, 33.77):
        k = int(np.argmin(np.abs(z - zz)))
        D[f"T{z[k]:.0f}"] = th[:, k]
    D["S1"], D["S34"] = so[:, 0], so[:, k34]
    D["dsig_0_34"] = r[:, k34] - r[:, 0]
    D["mld"] = C.mld_modello(a.dati).reindex(tm).values
    # termoclino del modello: quota del massimo dT/dz fra 4 e 50 m
    dT = -np.diff(th, axis=1) / np.diff(z)[None, :]
    zc = 0.5 * (z[1:] + z[:-1])
    ok = (zc > 4) & (zc < 50)
    kk = np.nanargmax(np.where(ok[None, :], dT, -np.inf), axis=1)
    D["z_tc_mod"] = zc[kk]
    D["dTdz_max"] = dT[np.arange(len(tm)), kk]
    d = D.resample("1D").mean()
    h = C.serie_orarie(a.risultati)
    d["T_adcp"] = h.temperatura.where(h.temperatura_qc == 1).resample("1D").mean()
    tc = pd.read_csv(C.trova(a.risultati, "termoclino.csv"), index_col=0, parse_dates=True)
    d["prof_taglio"] = C.H - tc.z_taglio.where(tc.affidabile)
    d["prof_eco"] = C.H - tc.z_eco
    d.to_csv(os.path.join(a.uscita, "giornaliero_modello.csv"))
    for c in ("prof_taglio", "prof_eco"):
        x = d[[c, "z_tc_mod"]].dropna()
        print(f"termoclino ADCP ({c}) contro modello: {len(x)} giorni, correlazione {x.corr().iloc[0, 1]:.2f}, "
              f"differenza mediana {(x[c] - x.z_tc_mod).median():.1f} m")
    x = d[["T34", "T_adcp"]].dropna()
    print(f"temperatura a 34 m: modello - ADCP = {(x.T34 - x.T_adcp).mean():+.2f} gradi, correlazione giornaliera {x.corr().iloc[0, 1]:.2f}")


if __name__ == "__main__":
    main()
