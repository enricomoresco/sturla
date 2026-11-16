"""Temperatura al fondo a Sturla, 4-13 ottobre: misura dell'ADCP, modello MedFS al sensore e alla quota
equivalente (7.6 m piu' in basso), run GOTM della matrice (MATRICE_*) e delle prove di sensibilita'
(EXTENSION_*). Statistiche rispetto alla media su 25 ore dell'ADCP e figura 7.

  python analisi/rimescolamento/gotm_confronto_fondo.py --dati ... --risultati ... --gotm ... --uscita ...

Uscite: confronto_fondo_statistiche.csv, figure/7_fondo_adcp_copernicus_gotm.png
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # anche con python -I

import os

import numpy as np
import pandas as pd

import comune as C
import gotm_comune as G
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from adcp_pd0.plots import C1, C2, C3, INK, INK2, GRID, AXIS, _fmt_time, _save

SPOSTAMENTO = 7.6       # m: MedFS ha la temperatura misurata al sensore 7.6 m piu' in basso (4 ottobre)
CASI = {"R0": "MATRICE_*_R0", "R2": "MATRICE_*_R2", "R2_LF17": "MATRICE_*_R2_LF17", "R3": "MATRICE_*_R3",
        "R3_senza_w": "MATRICE_*_R3_senza_w", "GLS_base_w": "MATRICE_*_GLS_base_w",
        "GLS_breaking_w": "MATRICE_*_GLS_breaking_w",
        "R3_LF17": "EXTENSION_*_R3_LF17", "R3_GLS_base": "EXTENSION_*_R3_GLS_base",
        "R3_GLS_breaking": "EXTENSION_*_R3_GLS_breaking", "R3_tau6h_hb5m": "EXTENSION_*_R3_tau6h_hb5m",
        "R3_tau6h_hb10m": "EXTENSION_*_R3_tau6h_hb10m", "R3_tau6h_hb15m": "EXTENSION_*_R3_tau6h_hb15m",
        "R3_tau24h_hb5m": "EXTENSION_*_R3_tau24h_hb5m", "R3_tau24h_hb15m": "EXTENSION_*_R3_tau24h_hb15m"}
FINESTRA = (pd.Timestamp("2026-10-04 01:00"), pd.Timestamp("2026-10-13 00:00"))


def main(argv=None):
    a = C.argomenti(__doc__, argv)
    radice = G.cartella_gotm(a)
    Ta = G.temperatura_adcp(a.risultati)
    Tm = Ta.rolling(25, center=True, min_periods=13).mean()
    zm, tm, th, _ = C.modello_punto(a.dati)
    med = pd.Series([np.interp(G.PROF_SENSORE, zm, p) for p in th], index=tm)
    meds = pd.Series([np.interp(G.PROF_SENSORE + SPOSTAMENTO, zm, p) for p in th], index=tm)
    runs = {}
    for k, schema in CASI.items():
        try:
            runs[k] = G.fondo(G.trova_run(radice, schema))[0]
        except FileNotFoundError as e:
            print(f"  saltata: {e}")

    def stats(s):
        x = pd.concat([s.loc[FINESTRA[0]:FINESTRA[1]].rename("m"), Tm.rename("o")], axis=1, join="inner").dropna()
        return dict(bias=(x.m - x.o).mean(), rmse=np.sqrt(((x.m - x.o) ** 2).mean()), corr=x.corr().iloc[0, 1])

    tab = {f"MedFS al sensore ({G.PROF_SENSORE:.1f} m)": stats(med),
           f"MedFS alla quota equivalente ({G.PROF_SENSORE + SPOSTAMENTO:.1f} m)": stats(meds)}
    for k, s in runs.items():
        tab[f"GOTM {k}"] = stats(s)
    T = pd.DataFrame(tab).T.sort_values("rmse")
    print(T.round(2).to_string())
    T.to_csv(os.path.join(a.uscita, "confronto_fondo_statistiche.csv"))
    gotm = [n for n in T.index if n.startswith("GOTM")]
    if not gotm:
        raise SystemExit("nessuna run GOTM trovata")
    best = gotm[0]
    print("run piu' vicina alla misura:", best)
    bs = runs[best.replace("GOTM ", "")]

    lo, hi = pd.Timestamp("2026-10-01"), pd.Timestamp("2026-10-14")
    fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True, layout="constrained",
                            gridspec_kw=dict(height_ratios=[1.4, 1]))
    ax = axs[0]
    ax.set_axisbelow(True)
    x = Ta.loc[lo:hi]
    ax.plot(x.index, x.values, color=INK, lw=0.8, alpha=0.3)
    ax.plot(Tm.loc[lo:hi].index, Tm.loc[lo:hi].values, color=INK, lw=2.6,
            label="ADCP, misura al fondo (media su 25 h; in chiaro i valori orari)")
    ax.plot(med.loc[lo:hi].index, med.loc[lo:hi].values, color=C1, lw=2,
            label=f"Copernicus MedFS a {G.PROF_SENSORE:.1f} m, punto 3–4 km al largo")
    ax.plot(meds.loc[lo:hi].index, meds.loc[lo:hi].values, color=C1, lw=1.6, ls="--",
            label=f"Copernicus MedFS a {G.PROF_SENSORE + SPOSTAMENTO:.1f} m (quota equivalente, spostata di {SPOSTAMENTO:.1f} m)")
    ax.plot(bs.index, bs.values, color=C2, lw=2.2,
            label=f"GOTM {best.replace('GOTM ', '')} (più vicina alla misura)")
    r0 = runs.get("R0")
    if r0 is not None:
        ax.plot(r0.index, r0.values, color=C3, lw=2, label="GOTM R0: solo processi locali")
    ax.set_ylabel("temperatura al fondo (°C)")
    ax.legend(loc="upper left", fontsize=9, frameon=False)
    ax = axs[1]
    ax.set_axisbelow(True)
    t0 = pd.Timestamp("2026-10-04 01:00")

    def d0(s):
        return s - float(s.asof(t0))

    ax.plot(Tm.loc[lo:hi].index, d0(Tm).loc[lo:hi].values, color=INK, lw=2.6)
    ax.plot(med.loc[lo:hi].index, d0(med).loc[lo:hi].values, color=C1, lw=2)
    ax.plot(meds.loc[lo:hi].index, d0(meds).loc[lo:hi].values, color=C1, lw=1.6, ls="--")
    ax.plot(bs.index, d0(bs).values, color=C2, lw=2.2)
    if r0 is not None:
        ax.plot(r0.index, d0(r0).values, color=C3, lw=2)
    ax.axhline(0, color=AXIS, lw=0.8)
    ax.set_ylabel("variazione dal 4 ottobre (°C)")
    for x_ in axs:
        x_.axvspan(pd.Timestamp("2026-10-08 10:00"), pd.Timestamp("2026-10-09 14:00"), color=GRID, alpha=0.6, lw=0,
                   zorder=0)
        x_.axvline(t0, color=AXIS, lw=1, ls=":")
    axs[1].set_xlim(lo, hi)
    _fmt_time(axs[1])
    fig.suptitle("Temperatura al fondo a Sturla: misura, modello Copernicus e simulazione GOTM più vicina", x=0.01,
                 ha="left", fontweight="bold")
    fig.text(0.01, 0.957, "In grigio la mareggiata; linea punteggiata: inizio delle simulazioni GOTM (4 ottobre). "
             "Sotto, le stesse serie riportate a zero al 4 ottobre.", color=INK2, fontsize=10, ha="left")
    _save(fig, os.path.join(a.uscita, "figure", "7_fondo_adcp_copernicus_gotm.png"))


if __name__ == "__main__":
    main()
