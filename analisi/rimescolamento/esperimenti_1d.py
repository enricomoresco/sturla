"""Esperimenti con il modello 1D (solo processi locali) per i tre eventi, alla stazione (colonna di
34.5 m) e nel punto del modello (colonna di 56 m), confrontati con il modello 3D e con l'ADCP.

Ogni esperimento parte dal profilo del modello 3D all'inizio dell'evento e accende o spegne un
processo, o cambia un parametro (vento x1.5 e x2, efficienza del vento, diffusione di fondo).

Uscita: esperimenti_1d.csv (riepilogo), esperimenti_1d.pkl (serie orarie, per le figure)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # anche con python -I

import os

import numpy as np
import pandas as pd

import comune as C
from col1d import Column, run

ESPERIMENTI = {
    "solo galleggiamento": dict(vento=False, onde=False, fondo=False),
    "+ vento": dict(onde=False),
    "tutto (vento, onde, fondo)": dict(),
    "tutto, vento x1.5": dict(fattore_vento=1.5),
    "tutto, vento x2": dict(fattore_vento=2.0),
    "tutto, delta 0.039": dict(delta=0.039),
    "senza flussi in superficie": dict(flussi=False),
    "tutto, Kbg 1e-6": dict(Kbg=1e-6),
    "tutto, Kbg 1e-4": dict(Kbg=1e-4),
}
SITI = (("stazione", C.H, 2.0), ("punto modello", 56.33, 0.0))     # nome, profondita', strato di fondo iniziale


def diagnostica_3d(zm, tm, th, so, H):
    """Temperatura al sensore, phi*H, mixed layer e stratificazione della colonna 0-H del modello 3D, ora per ora."""
    zz = np.linspace(0, H, int(H / 0.1) + 1)
    out = []
    for i in range(len(tm)):
        Ti = np.interp(zz, zm, th[i]); Si = np.interp(zz, zm, so[i]); r = C.rho_eos(Ti, Si)
        phiH = np.trapezoid((r.mean() - r) * C.g * (H - zz), zz)
        r1 = np.interp(1.0, zz, r); k = np.where(r > r1 + 0.1)[0]
        out.append(dict(T_probe=np.interp(C.PROF_SENSORE, zz, Ti), phiH=phiH, mld=zz[k[0]] if len(k) else H,
                        T_sup=Ti[10], dsig=r[-1] - r[0]))
    return pd.DataFrame(out, index=tm)


def main(argv=None):
    a = C.argomenti(__doc__, argv)
    zm, tm, th, so = C.modello_punto(a.dati)
    D3 = {H: diagnostica_3d(zm, tm, th, so, H) for _, H, _ in SITI}
    F = C.forzanti_1d(a.uscita, a.risultati)
    h = C.serie_orarie(a.risultati)
    Tadcp = h.temperatura.where((h.deployment_qc == 1) & (h.temperatura_qc <= 2))
    rows, series = [], {}
    for ev, (t0s, t1s) in C.EVENTI.items():
        t0, t1 = pd.Timestamp(t0s), pd.Timestamp(t1s)
        i0 = int(np.argmin(np.abs(tm - t0)))
        for site, H, bbl in SITI:
            d3 = D3[H]; base = dict(evento=ev, sito=site)
            rows.append(dict(base, esperimento="modello 3D", dT_probe=d3.T_probe.loc[t1] - d3.T_probe.loc[t0],
                             phiH0=d3.phiH.loc[t0], dphiH=d3.phiH.loc[t1] - d3.phiH.loc[t0], mld0=d3.mld.loc[t0],
                             mld1=d3.mld.loc[t1], dsig0=d3.dsig.loc[t0], dsig1=d3.dsig.loc[t1]))
            if site == "stazione":
                ta = Tadcp.loc[t0 - pd.Timedelta("3h"):t0 + pd.Timedelta("3h")].mean()
                tb = Tadcp.loc[t1 - pd.Timedelta("3h"):t1 + pd.Timedelta("3h")].mean()
                rows.append(dict(base, esperimento="misura ADCP", dT_probe=tb - ta))
            for name, opt in ESPERIMENTI.items():
                col0 = Column(H, zm, th[i0], so[i0], bbl=bbl)
                col = Column(H, zm, th[i0], so[i0], bbl=bbl)
                r = run(col, F, t0, t1, opt, C.PROF_SENSORE)
                parts = r.drop(columns=["T_probe", "mld", "phiH", "T_sup", "dsig"]).sum()
                rows.append(dict(base, esperimento=name, dT_probe=r.T_probe.iloc[-1] - col0.at(C.PROF_SENSORE),
                                 phiH0=col0.phiH(), dphiH=r.phiH.iloc[-1] - col0.phiH(), mld0=col0.mld(), mld1=r.mld.iloc[-1],
                                 dsig0=col0.dsig(), dsig1=r.dsig.iloc[-1], **{f"c_{k}": v for k, v in parts.items()}))
                series[(ev, site, name)] = r
            print(f"  {ev}, {site}: fatto")
    S = pd.DataFrame(rows)
    S.to_csv(os.path.join(a.uscita, "esperimenti_1d.csv"), index=False)
    pd.to_pickle(dict(series=series, D3=D3, Tadcp=Tadcp, EV=C.EVENTI), os.path.join(a.uscita, "esperimenti_1d.pkl"))
    x = S[(S.sito == "stazione") & S.esperimento.isin(["misura ADCP", "modello 3D", "tutto (vento, onde, fondo)", "tutto, vento x2"])]
    print(x.pivot(index="esperimento", columns="evento", values="dT_probe").round(2).to_string())


if __name__ == "__main__":
    main()
