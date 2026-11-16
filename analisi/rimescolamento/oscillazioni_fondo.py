"""Oscillazioni rapide della temperatura al fondo (slide 52 della presentazione).

Per periodo: deviazione standard della temperatura oraria meno la sua media mobile centrata di 25 ore,
numero di salti orari oltre 1 grado al giorno e salto orario massimo. Poi la quota della varianza a
periodi sotto 30 ore per bande, dallo spettro di Welch (segmenti di 8 giorni) dal 1 settembre al 27
ottobre.

  python analisi/rimescolamento/oscillazioni_fondo.py --dati ... --risultati ... --uscita ...

Uscite: oscillazioni_fondo.csv, oscillazioni_fondo_bande.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # anche con python -I

import os

import numpy as np
import pandas as pd
from scipy.signal import welch

import comune as C

_trapz = getattr(np, "trapezoid", None) or np.trapz
PERIODI = {"luglio": ("2026-07-10", "2026-07-31"), "agosto": ("2026-08-01", "2026-08-31"),
           "settembre": ("2026-09-01", "2026-09-30"), "1-3 ott": ("2026-10-01", "2026-10-03"),
           "4-7 ott": ("2026-10-04", "2026-10-07"), "8-9 ott": ("2026-10-08", "2026-10-09 23:00"),
           "10-17 ott": ("2026-10-10", "2026-10-17"), "18-19 ott": ("2026-10-18", "2026-10-19 23:00"),
           "20-27 ott": ("2026-10-20", "2026-10-27")}
BANDE = {"< 6 h": (0, 6), "6-11 h": (6, 11), "11-14 h (marea semidiurna)": (11, 14),
         "14-20 h (inerziale 17.1 h)": (14, 20), "20-30 h (diurna)": (20, 30)}


def main(argv=None):
    a = C.argomenti(__doc__, argv)
    h = C.serie_orarie(a.risultati)
    T = h.temperatura.where(h.deployment_qc == 1).asfreq("1h")
    Ti = T.interpolate(limit=3)
    hp = Ti - Ti.rolling(25, center=True, min_periods=13).mean()
    salti = Ti.diff().abs()
    rows = {}
    for k, (t0, t1) in PERIODI.items():
        x, s = hp.loc[t0:t1], salti.loc[t0:t1]
        rows[k] = {"dev. std oscillazioni < 25 h (C)": x.std(), "salti orari > 1 C (per giorno)": (s > 1).sum() / (len(s) / 24),
                   "salto orario massimo (C)": s.max()}
    D = pd.DataFrame(rows).T
    print(D.round(2).to_string())
    D.to_csv(os.path.join(a.uscita, "oscillazioni_fondo.csv"))
    x = Ti.loc["2026-09-01":"2026-10-27"].dropna()
    f, p = welch(x.values - x.values.mean(), fs=24.0, nperseg=24 * 8)
    per = 24 / f[1:]
    tot = _trapz(p[1:][per < 30], f[1:][per < 30])
    righe = []
    for k, (lo, hi) in BANDE.items():
        m = (per >= lo) & (per < hi)
        q = _trapz(p[1:][m], f[1:][m]) / tot
        righe.append(dict(banda=k, quota_varianza_sotto_30h=q))
        print(f"{k:28s} quota della varianza sotto 30 h: {q:.0%}")
    i = np.argmax(np.where(per < 30, p[1:], 0))
    print(f"picco a {per[i]:.1f} h")
    pd.DataFrame(righe).to_csv(os.path.join(a.uscita, "oscillazioni_fondo_bande.csv"), index=False)


if __name__ == "__main__":
    main()
