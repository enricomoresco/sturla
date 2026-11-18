"""GOTM con le correnti misurate, 4-13 ottobre (matrice MATRICE_*): riscaldamento a 33.9 m e profondita'
dell'isoterma media del profilo iniziale per R0, R2, R2 con Langmuir, R3 senza w e R3, con le stime
dell'interfaccia dall'ADCP. Figura 6 (slide 50 della presentazione).

  python analisi/rimescolamento/gotm_figura_4ottobre.py --dati ... --risultati ... --gotm ... --uscita ...

Uscita: figure/6_gotm_matrice.png
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # anche con python -I

import os

import pandas as pd

import comune as C
import gotm_comune as G
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from adcp_pd0.plots import C1, C2, C3, INK, INK2, GRID, AXIS, _fmt_time, _save

C4 = "#eda100"
STILI = [("R0", "MATRICE_*_R0", C1, "-", "R0: processi locali (velocità ADCP, KPP completo)"),
         ("R2", "MATRICE_*_R2", C2, "-", "R2: + downwelling (velocità verticale MedFS)"),
         ("R2_LF17", "MATRICE_*_R2_LF17", C2, ":", "R2 + Langmuir LF17 (coincide con R2)"),
         ("R3_senza_w", "MATRICE_*_R3_senza_w", C4, "-", "R3 senza w: solo trasporto laterale vicino al fondo"),
         ("R3", "MATRICE_*_R3", C3, "-", "R3: downwelling + trasporto laterale vicino al fondo")]


def main(argv=None):
    a = C.argomenti(__doc__, argv)
    radice = G.cartella_gotm(a)
    Ta = G.temperatura_adcp(a.risultati)
    tc = pd.read_csv(C.trova(a.risultati, "termoclino.csv"), index_col=0, parse_dates=True).loc["2026-10-03":"2026-10-13"]
    R = {}
    for k, schema, *_ in STILI:
        run = G.trova_run(radice, schema)
        s, _dT = G.fondo(run)
        iso, isoT = G.isoterma(run)
        R[k] = dict(sens=s, iso=iso, isoT=isoT)
        print(f"{k:11s} riscaldamento massimo a {G.PROF_SENSORE} m: {s.max() - s.iloc[0]:+.2f} °C; isoterma {isoT:.1f} °C")
    lo, hi = pd.Timestamp("2026-10-03 12:00"), pd.Timestamp("2026-10-13")
    fig, axs = plt.subplots(2, 1, figsize=(12, 8.4), sharex=True, layout="constrained",
                            gridspec_kw=dict(height_ratios=[1.35, 1]))
    ax = axs[0]
    ax.set_axisbelow(True)
    ta0 = Ta.loc["2026-10-03 22:00":"2026-10-04 04:00"].mean()       # 2 ore prima - 4 ore dopo l'inizio
    x = (Ta - ta0).loc[lo:hi]
    xm = (Ta - ta0).rolling(25, center=True, min_periods=13).mean().loc[lo:hi]
    ax.plot(x.index, x.values, color=INK, lw=0.8, alpha=0.3)
    ax.plot(xm.index, xm.values, color=INK, lw=2.4, label="ADCP, misura (media su 25 h; in chiaro i valori orari)")
    for k, _, c, ls, lab in STILI:
        s = R[k]["sens"]
        ax.plot(s.index, s - s.iloc[0], color=c, lw=2.2 if ls == "-" else 2.6, ls=ls, label=lab)
    ax.axhline(0, color=AXIS, lw=0.8)
    ax.set_ylabel("riscaldamento a 33.9 m\ndal 4 ottobre (°C)")
    ax.legend(loc="upper left", fontsize=9, frameon=False)
    ax = axs[1]
    ax.set_axisbelow(True)
    aff = tc.affidabile.astype(str) == "True"
    ax.plot(tc.index[aff] + pd.Timedelta("12h"), (C.H - tc.z_taglio)[aff], "o", ms=7, color=INK,
            label="ADCP: massimo del taglio (stime affidabili)")
    ax.plot(tc.index + pd.Timedelta("12h"), C.H - tc.z_eco, "s", ms=6, mfc="none", mec=INK2, mew=1.5,
            label="ADCP: massimo dell'eco")
    for k, _, c, ls, lab in STILI:
        if k == "R2_LF17":
            continue
        v = R[k]
        ax.plot(v["iso"].index, v["iso"].values, color=c, lw=2, label=f"{k.replace('_', ' ')}: isoterma {v['isoT']:.1f} °C")
    ax.axhline(C.H, color=AXIS, lw=1.5)
    ax.set_ylim(35, 15)
    ax.set_ylabel("profondità dell'interfaccia (m)\n(linea in basso: fondale)")
    ax.legend(loc="upper right", fontsize=9, frameon=False, ncols=2)
    for x_ in axs:
        x_.axvspan(pd.Timestamp("2026-10-08 08:00"), pd.Timestamp("2026-10-09 12:00"), color=GRID, alpha=0.6, lw=0,
                   zorder=0)
    axs[1].set_xlim(lo, hi)
    _fmt_time(axs[1])
    fig.suptitle("GOTM a Sturla: il fondo si scalda solo se si aggiunge trasporto; Langmuir e frangimento non "
                 "cambiano nulla al fondo", x=0.01, ha="left", fontweight="bold")
    fig.text(0.01, 0.957, "In grigio la mareggiata (8 ottobre ore 10 – 9 ottobre ore 14). Le run GLS con e senza "
             "frangimento (non mostrate) coincidono fra loro al fondo.", color=INK2, fontsize=10, ha="left")
    _save(fig, os.path.join(a.uscita, "figure", "6_gotm_matrice.png"))


if __name__ == "__main__":
    main()
