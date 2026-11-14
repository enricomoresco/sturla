"""Scomposizione fra processi locali e trasporto: il modello 1D riparte ogni giorno dal profilo del
modello 3D. Variazione del 3D in un giorno = variazione del 1D (solo processi locali) + resto
(trasporto).

Uscita: scomposizione_giornaliera.csv
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

EXP = {"x1": dict(), "x1.5": dict(fattore_vento=1.5), "x2": dict(fattore_vento=2.0), "d039": dict(delta=0.039)}


def main(argv=None):
    a = C.argomenti(__doc__, argv)
    zm, tm, th, so = C.modello_punto(a.dati)
    F = C.forzanti_1d(a.uscita, a.risultati)

    def col3d(i):
        return Column(C.H, zm, th[i], so[i], bbl=2.0)

    rows = []
    for ev, (t0s, t1s) in C.EVENTI.items():
        for d0 in pd.date_range(t0s, t1s, freq="24h")[:-1]:
            d1 = d0 + pd.Timedelta("24h")
            i0, i1 = int(np.argmin(np.abs(tm - d0))), int(np.argmin(np.abs(tm - d1)))
            c0, c1 = col3d(i0), col3d(i1)
            r = dict(evento=ev, giorno=d0, phiH0_3D=c0.phiH(), dphiH_3D=c1.phiH() - c0.phiH(),
                     dT_3D=c1.at(C.PROF_SENSORE) - c0.at(C.PROF_SENSORE))
            for k, opt in EXP.items():
                col = col3d(i0); p0, T0 = col.phiH(), col.at(C.PROF_SENSORE)
                s = run(col, F, d0, d1, opt, C.PROF_SENSORE)
                r[f"dphiH_1D_{k}"] = col.phiH() - p0; r[f"dT_1D_{k}"] = col.at(C.PROF_SENSORE) - T0
                if k == "x1":
                    for c in s.columns:
                        if c not in ("T_probe", "mld", "phiH", "T_sup", "dsig"):
                            r[f"c_{c}"] = s[c].sum()
            rows.append(r)
    D = pd.DataFrame(rows)
    D.to_csv(os.path.join(a.uscita, "scomposizione_giornaliera.csv"), index=False)
    for ev in C.EVENTI:
        x = D[D.evento == ev]
        tot3 = -x.dphiH_3D.sum()
        loc = {k: -x[f"dphiH_1D_{k}"].sum() for k in EXP}
        print(f"{ev}: stratificazione iniziale {x.phiH0_3D.iloc[0]:.0f} J/m2, tolta nel 3D {tot3:.0f}; processi locali: "
              + ", ".join(f"{k} {v:.0f} ({100 * v / tot3:.0f}%)" for k, v in loc.items()))
    x = D[D.evento == "5-9 ottobre"]
    for lab, m in (("prima della mareggiata", x.giorno < pd.Timestamp("2026-10-07 12:00")),
                   ("durante la mareggiata", x.giorno >= pd.Timestamp("2026-10-07 12:00"))):
        y = x[m]
        print(f"  {lab}: tolta nel 3D {-y.dphiH_3D.sum():.0f}, locale x1 {-y.dphiH_1D_x1.sum():.0f}, x2 {-y.dphiH_1D_x2.sum():.0f}")


if __name__ == "__main__":
    main()
