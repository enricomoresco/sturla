"""Figura 9 (slide 52 della presentazione): profondita' dell'interfaccia stimata dall'ADCP, nel modello MedFS e
in GOTM con lo stesso criterio (0.2 gradi sotto la temperatura a 10 m), temperatura oraria al fondo e
ampiezza delle oscillazioni sotto le 25 ore, ottobre 2026.

  python analisi/rimescolamento/gotm_figura_interfaccia.py --dati ... --risultati ... --gotm ... --uscita ...

Servono in --dati il file della temperatura del riquadro MedFS (scarica_dati.sh).
Uscita: figure/9_interfaccia_e_oscillazioni.png
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # anche con python -I

import os

import numpy as np
import pandas as pd
import xarray as xr

import comune as CM
import gotm_comune as G
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from adcp_pd0.plots import C1, C2, C3, INK, INK2, MUTED, GRID, AXIS, SURF, _fmt_time, _save


def main(argv=None):
    arg = CM.argomenti(__doc__, argv)
    OUT = os.path.join(arg.uscita, "figure")
    radice = G.cartella_gotm(arg)
    LO, HI = pd.Timestamp("2026-10-01"), pd.Timestamp("2026-10-28")
    STORMS = [("2026-10-08 00:00", "2026-10-09 23:00"), ("2026-10-20 00:00", "2026-10-22 23:00")]


    # MedFS, punto del modello (91 m di fondale): stesso criterio di GOTM (0.2 gradi sotto la T a 10 m)
    T3 = xr.open_dataset(CM.trova(arg.dati, "*medfs_riquadro_temperatura*.nc")).thetao
    dep = T3.depth.values.astype(float)
    x = T3.sel(latitude=44.3542, longitude=9.0, method="nearest").sel(time=slice(LO, HI))
    med = pd.Series([G.mld_T(dep, p) for p in x.values], index=pd.DatetimeIndex(x.time.values)).resample("1D").mean()
    # GOTM: 4 ottobre (vento CMEMS, convezione accesa) e 18 ottobre
    gotm = {}
    for k, schema in {("R0", 1): "VENTO_*_OTT04_R0_CMEMS", ("R3", 1): "VENTO_*_OTT04_R3_CMEMS",
                      ("R0", 2): "*_OTTOBRE18_R0", ("R3", 2): "*_OTTOBRE18_R3"}.items():
        gotm[k] = G.mld_run(G.trova_run(radice, schema)).resample("1D").mean()
    # ADCP: stime dell'interfaccia (giornaliere) e temperatura al fondo (oraria)
    tc = pd.read_csv(CM.trova(arg.risultati, "termoclino.csv"), index_col=0, parse_dates=True).loc[LO:HI]
    tc.index = tc.index + pd.Timedelta("12h")
    Ta = G.temperatura_adcp(arg.risultati).asfreq("1h")
    Ti = Ta.interpolate(limit=3)
    T25 = Ti.rolling(25, center=True, min_periods=13).mean()
    hp = Ti - T25
    PER = [("lug", "2026-07-10", "2026-07-31"), ("ago", "2026-08-01", "2026-08-31"), ("set", "2026-09-01", "2026-09-30"),
           ("1–3 ott", "2026-10-01", "2026-10-03 23:00"), ("4–7 ott", "2026-10-04", "2026-10-07 23:00"),
           ("8–9 ott", "2026-10-08", "2026-10-09 23:00"), ("10–17 ott", "2026-10-10", "2026-10-17 23:00"),
           ("18–19 ott", "2026-10-18", "2026-10-19 23:00"), ("20–27 ott", "2026-10-20", "2026-10-27 23:00")]
    std = [(n, float(hp.loc[a:b].std())) for n, a, b in PER]
    vicino = {"4–7 ott", "8–9 ott", "10–17 ott", "18–19 ott"}

    fig = plt.figure(figsize=(13.5, 9.4), layout="constrained")
    gs = fig.add_gridspec(2, 2, height_ratios=[1.25, 1], width_ratios=[2.6, 1])
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[1, 0], sharex=axA)
    axC = fig.add_subplot(gs[:, 1])
    for ax in (axA, axB):
        for a, b in STORMS:
            ax.axvspan(pd.Timestamp(a), pd.Timestamp(b), color=GRID, alpha=0.55, lw=0, zorder=0)
    # A
    ax = axA
    ax.axhspan(30.8, 34.5, facecolor=GRID, alpha=0.35, lw=0, zorder=0, hatch="////", edgecolor=SURF)
    ax.axhline(34.5, color=INK2, lw=1.2, zorder=2)
    ax.plot(med.index, med.values, color=C1, lw=2.2, zorder=4, marker="o", ms=3.5)
    for (k, j), s in gotm.items():
        ax.plot(s.index, s.values, color=C2 if k == "R3" else C3, lw=2.2, zorder=4, marker="o", ms=3.5)
    ok = tc.affidabile.astype(str) == "True"
    for col, mk, nome in (("z_taglio", "o", "taglio"), ("z_eco", "s", "eco")):
        y = 34.5 - tc[col]
        ax.scatter(tc.index[ok], y[ok], marker=mk, s=34, facecolor=INK if mk == "o" else SURF, edgecolor=INK,
                   linewidth=1.3, zorder=5)
        ax.scatter(tc.index[~ok], y[~ok], marker=mk, s=34, facecolor=SURF, edgecolor=MUTED, linewidth=1.0, zorder=5)
    ax.set_ylim(50, 10)
    ax.set_ylabel("profondità dell'interfaccia (m)")
    lab = [(med, "MedFS", C1, pd.Timestamp("2026-10-19 06:00"), 16.5),
           (gotm[("R3", 1)], "GOTM R3", C2, pd.Timestamp("2026-10-12 12:00"), -2.2),
           (gotm[("R0", 1)], "GOTM R0", C3, pd.Timestamp("2026-10-12 12:00"), -1.8)]
    for s, t, col, tt, dy in lab:
        y = float(s.asof(tt))
        ax.annotate(t, (tt, y), xytext=(tt, y + dy), color=INK, fontsize=9, fontweight="bold", ha="center",
                    va="top" if dy > 0 else "bottom")
    ax.scatter([], [], marker="o", s=34, facecolor=INK, edgecolor=INK, label="ADCP: massimo del taglio")
    ax.scatter([], [], marker="s", s=34, facecolor=SURF, edgecolor=INK, linewidth=1.3, label="ADCP: massimo dell'eco")
    ax.scatter([], [], marker="o", s=34, facecolor=SURF, edgecolor=MUTED, label="ADCP: stima poco affidabile")
    ax.plot([], [], color=INK2, lw=1.2, label="fondale a Sturla (34.5 m; sensore a 33.9 m)")
    ax.fill_between([], [], [], facecolor=GRID, alpha=0.6, hatch="////", edgecolor=SURF,
                    label="sotto il livello più basso in cui l'ADCP misura il taglio")
    ax.legend(loc="lower left", fontsize=8.5, ncol=2)
    ax.set_title("Profondità dell'interfaccia (mixed layer): stime dall'ADCP, modello MedFS e GOTM", loc="left")
    # B
    ax = axB
    x = Ta.loc[LO:HI]
    ax.plot(x.index, x.values, color=INK, lw=0.8, alpha=0.55, zorder=3)
    x = T25.loc[LO:HI]
    ax.plot(x.index, x.values, color=INK, lw=2.4, zorder=4)
    ax.annotate("salti di 3 °C in 2 ore:\nl'interfaccia passa davanti\nal sensore", (pd.Timestamp("2026-10-18 12:00"), 19.6),
                xytext=(pd.Timestamp("2026-10-14 06:00"), 18.0), color=INK, fontsize=8.5, ha="center",
                arrowprops=dict(arrowstyle="-", color=AXIS, lw=0.8))
    ax.set_ylabel("temperatura al fondo (°C)")
    ax.set_title("Temperatura al fondo misurata dall'ADCP: valori orari e media su 25 h", loc="left")
    ax.set_xlim(LO, HI)
    _fmt_time(ax)
    plt.setp(axA.get_xticklabels(), visible=False)
    # C
    ax = axC
    yy = np.arange(len(std))[::-1]
    for y, (n, v) in zip(yy, std):
        ax.barh(y, v, height=0.6, color=INK if n in vicino else MUTED, edgecolor=SURF, linewidth=2, zorder=3)
        ax.text(v + 0.03, y, f"{v:.2f}", color=INK, fontsize=8.5, va="center")
    ax.set_yticks(yy)
    ax.set_yticklabels([n for n, _ in std])
    ax.set_xlim(0, 1.45)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("deviazione standard delle oscillazioni\nsotto le 25 ore (°C)")
    ax.set_title("Oscillazioni rapide al fondo", loc="left")
    ax.text(1.42, yy[-1] - 1.0, "in nero: periodi con l'interfaccia\nentro pochi metri dal fondo", color=INK2,
            fontsize=8.5, ha="right", va="center")
    ax.set_ylim(yy[-1] - 1.6, yy[0] + 0.6)
    fig.suptitle("Ottobre 2026: l'interfaccia scende verso il fondo e oscilla davanti al sensore", x=0.01, ha="left",
                 fontweight="bold", fontsize=12)
    fig.text(0.01, 0.955, "Sopra: interfaccia stimata dall'ADCP (massimi del taglio e dell'eco), MedFS e GOTM con lo stesso "
             "criterio (0.2 °C sotto la temperatura a 10 m). Le run GOTM partono il 4 e il 18 ottobre.\nMedFS è il punto "
             "a 91 m di fondale, 3–4 km al largo: il suo mixed layer può scendere sotto i 34.5 m. Sotto: la "
             "temperatura al fondo salta fra le due masse d'acqua quando\nl'interfaccia è vicina al fondo. In grigio le "
             "mareggiate.", color=INK2, fontsize=9.5, ha="left", va="top")
    fig.get_layout_engine().set(rect=(0, 0, 1, 0.91))
    _save(fig, f"{OUT}/9_interfaccia_e_oscillazioni.png")
    print(pd.DataFrame(std, columns=["periodo", "std"]).round(2).to_string(index=False))


if __name__ == "__main__":
    main()
