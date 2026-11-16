"""Figura riassuntiva della matrice GOTM (figura 8, slide 49, 51 e 54 della presentazione): tre finestre,
ADCP contro R0 e R3 con le varianti sovrapposte, e barre con l'effetto massimo di ciascun fattore sul
riscaldamento a 33.9 m.

  python analisi/rimescolamento/gotm_figura_matrice.py --dati ... --risultati ... --gotm ... --uscita ...

Riscaldamento di una run: temperatura a 33.9 m meno quella iniziale. Misura: media mobile di 7 ore della
temperatura ADCP meno la media fra 2 ore prima e 4 ore dopo l'inizio. Effetto di un fattore: massimo nel
tempo della differenza assoluta fra due run che differiscono solo per quel fattore.
Uscite: figure/8_matrice_gotm_sintesi.png, 8_matrice_gotm_sintesi.csv
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
from adcp_pd0.plots import C2, C3, INK, INK2, MUTED, GRID, AXIS, SURF, _fmt_time, _save

# nome breve -> schema del nome della cartella della run (vedi gotm_comune.trova_run)
CASI = {
    "A_R0": "*_AGOSTO_R0", "A_R2": "*_AGOSTO_R2", "A_R3": "*_AGOSTO_R3", "A_R3_LF17": "*_AGOSTO_R3_LF17",
    "A_R0_ERA5": "VENTO_*_AGOSTO_R0_ERA5", "A_R3_ERA5": "VENTO_*_AGOSTO_R3_ERA5",
    "O_R0": "VENTO_*_OTT04_R0_CMEMS", "O_R0_ERA5": "VENTO_*_OTT04_R0_ERA5", "O_R3": "VENTO_*_OTT04_R3_CMEMS",
    "O_R3_ERA5": "VENTO_*_OTT04_R3_ERA5",
    "O_R0_noconv": "MATRICE_*_R0", "O_R2_noconv": "MATRICE_*_R2", "O_R2_LF17": "MATRICE_*_R2_LF17",
    "O_GLS_base_w": "MATRICE_*_GLS_base_w", "O_GLS_breaking_w": "MATRICE_*_GLS_breaking_w",
    "O_R3_1d": "MATRICE_*_R3", "O_R3_1d_LF17": "EXTENSION_*_R3_LF17",
    "O_R3_GLS_base": "EXTENSION_*_R3_GLS_base", "O_R3_GLS_breaking": "EXTENSION_*_R3_GLS_breaking",
    "O_R3_noconv": "EXTENSION_*_R3_tau6h_hb10m", "O_R3_6h5": "EXTENSION_*_R3_tau6h_hb5m",
    "O_R3_6h15": "EXTENSION_*_R3_tau6h_hb15m", "O_R3_24h5": "EXTENSION_*_R3_tau24h_hb5m",
    "O_R3_24h15": "EXTENSION_*_R3_tau24h_hb15m",
    "D_R0": "*_OTTOBRE18_R0", "D_R2": "*_OTTOBRE18_R2", "D_R3": "*_OTTOBRE18_R3", "D_R3_LF17": "*_OTTOBRE18_R3_LF17",
    "D_R0_ERA5": "VENTO_*_OTTOBRE18_R0_ERA5", "D_R3_ERA5": "VENTO_*_OTTOBRE18_R3_ERA5",
}


def main(argv=None):
    arg = CM.argomenti(__doc__, argv)
    OUT = os.path.join(arg.uscita, "figure")
    radice = G.cartella_gotm(arg)
    R = {}
    for k, schema in CASI.items():
        s, dT = G.fondo(G.trova_run(radice, schema))
        R[k] = (s - s.iloc[0], dT)

    def dmax(a, b):
        x = pd.concat([R[a][0].rename("a"), R[b][0].rename("b")], axis=1).dropna()
        return float((x.a - x.b).abs().max())


    Ta = G.temperatura_adcp(arg.risultati)
    T25 = Ta.rolling(25, center=True, min_periods=13).mean()

    FIN = [
        ("A", "19–26 agosto", "stratificazione forte", [("2026-08-21 00:00", "2026-08-22 23:00")],
         {"r0": ["A_R0", "A_R0_ERA5"], "r3": ["A_R3", "A_R3_LF17", "A_R3_ERA5"], "rel": []}),
        ("O", "4–13 ottobre", "stratificazione media", [("2026-10-08 00:00", "2026-10-09 23:00")],
         {"r0": ["O_R0", "O_R0_ERA5", "O_R0_noconv"], "r3": ["O_R3", "O_R3_ERA5", "O_R3_noconv"],
          "rel": ["O_R3_1d", "O_R3_noconv", "O_R3_6h5", "O_R3_6h15", "O_R3_24h5", "O_R3_24h15"]}),
        ("D", "18–26 ottobre", "stratificazione debole", [("2026-10-20 00:00", "2026-10-22 23:00")],
         {"r0": ["D_R0", "D_R0_ERA5"], "r3": ["D_R3", "D_R3_LF17", "D_R3_ERA5"], "rel": []}),
    ]
    FATTORI = ["Onde: Langmuir (LF17)", "Onde: frangimento (GLS)", "Convezione", "Vento ERA5 invece di CMEMS",
               "Downwelling imposto (R2 − R0)", "Downwelling + trasporto al fondo (R3 − R0)", "Riscaldamento osservato (ADCP, media su 7 h)"]


    def effetti(w):
        if w == "A":
            e = [dmax("A_R3_LF17", "A_R3"), None, None, max(dmax("A_R0_ERA5", "A_R0"), dmax("A_R3_ERA5", "A_R3")),
                 dmax("A_R2", "A_R0"), dmax("A_R3", "A_R0")]
        elif w == "O":
            e = [max(dmax("O_R3_1d_LF17", "O_R3_1d"), dmax("O_R2_LF17", "O_R2_noconv")),
                 max(dmax("O_R3_GLS_breaking", "O_R3_GLS_base"), dmax("O_GLS_breaking_w", "O_GLS_base_w")),
                 max(dmax("O_R0", "O_R0_noconv"), dmax("O_R3", "O_R3_noconv")),
                 max(dmax("O_R0_ERA5", "O_R0"), dmax("O_R3_ERA5", "O_R3")),
                 dmax("O_R2_noconv", "O_R0_noconv"), dmax("O_R3", "O_R0")]
        else:
            e = [dmax("D_R3_LF17", "D_R3"), None, None, max(dmax("D_R0_ERA5", "D_R0"), dmax("D_R3_ERA5", "D_R3")),
                 dmax("D_R2", "D_R0"), dmax("D_R3", "D_R0")]
        t0, t1 = R[f"{w}_R0"][0].index[[0, -1]]
        oss = Ta.rolling(7, center=True, min_periods=4).mean().loc[t0:t1]   # media su 7 ore, come nelle tabelle
        base = float(Ta.loc[t0 - pd.Timedelta("2h"):t0 + pd.Timedelta("4h")].mean())
        return e + [float((oss - base).max())]


    fig = plt.figure(figsize=(13.5, 9.6), layout="constrained")
    gs = fig.add_gridspec(2, 3, height_ratios=[1.5, 1.15])
    axt = [fig.add_subplot(gs[0, i]) for i in range(3)]
    for a in axt[1:]:
        a.sharey(axt[0])
    axb = [fig.add_subplot(gs[1, i]) for i in range(3)]
    tab = {}
    for i, (w, nome, strat, storms, sets) in enumerate(FIN):
        ax = axt[i]
        ref0, ref3 = R[f"{w}_R0"][0], R[f"{w}_R3"][0]
        t0, t1 = ref0.index[0], ref0.index[-1]
        for a, b in storms:
            ax.axvspan(pd.Timestamp(a), pd.Timestamp(b), color=GRID, alpha=0.55, lw=0, zorder=0)
        base = float(Ta.loc[t0 - pd.Timedelta("2h"):t0 + pd.Timedelta("4h")].mean())   # come nelle tabelle
        x = Ta.loc[t0:t1] - base
        ax.plot(x.index, x.values, color=INK, lw=0.7, alpha=0.28, zorder=2)
        x = T25.loc[t0:t1] - base
        ax.plot(x.index, x.values, color=INK, lw=2.4, zorder=5)
        if sets["rel"]:
            env = pd.concat([R[k][0] for k in sets["rel"]], axis=1).resample("1h").mean()
            ax.fill_between(env.index, env.min(axis=1), env.max(axis=1), color=C2, alpha=0.13, lw=0, zorder=1)
        for keys, ref, col in ((sets["r0"], ref0, C3), (sets["r3"], ref3, C2)):
            env = pd.concat([R[k][0] for k in keys], axis=1).resample("1h").mean()
            ax.fill_between(env.index, env.min(axis=1), env.max(axis=1), color=col, alpha=0.3, lw=0, zorder=3)
            ax.plot(ref.index, ref.values, color=col, lw=2, zorder=4)
        xe = t1 + (t1 - t0) * 0.03
        pts = sorted([(float(T25.loc[t0:t1].iloc[-1] - base), "ADCP", INK), (float(ref3.iloc[-1]), "R3", C2),
                      (float(ref0.iloc[-1]), "R0", C3)])
        ylab = [p[0] for p in pts]
        for k in range(1, len(ylab)):                      # distanza minima fra le etichette
            ylab[k] = max(ylab[k], ylab[k - 1] + 0.42)
        for (y, lab, col), yl in zip(pts, ylab):
            ax.annotate(lab, (t1, y), xytext=(xe, yl), color=INK, fontsize=9, fontweight="bold", va="center",
                        annotation_clip=False, arrowprops=dict(arrowstyle="-", color=AXIS, lw=0.8, shrinkA=0, shrinkB=3))
            ax.plot([t1], [y], marker="o", ms=4, color=col, zorder=6, clip_on=False)
        ax.axhline(0, color=AXIS, lw=0.8, zorder=1)
        ax.set_xlim(t0, t1)
        _fmt_time(ax, maxticks=5)
        dT = R[f"{w}_R0"][1]
        ax.set_title(f"{nome}: {strat}\nsuperficie − fondo all'inizio {dT:.1f} °C", loc="left")
        if i == 0:
            ax.set_ylabel("riscaldamento a 33.9 m dall'inizio (°C)")
        else:
            plt.setp(ax.get_yticklabels(), visible=False)
        # barre
        e = effetti(w)
        tab[nome] = e
        ax = axb[i]
        y = np.arange(len(FATTORI))[::-1]
        for yy, v, f in zip(y, e, FATTORI):
            if v is None:
                ax.text(0.05, yy, "non provato", color=MUTED, fontsize=8, va="center")
                continue
            col = INK if f.startswith("Riscaldamento") else (C2 if "R3" in f or "R2" in f else INK2)
            ax.barh(yy, v, height=0.55, color=col, edgecolor=SURF, linewidth=2, zorder=3)
            ax.text(v + 0.08, yy, f"{v:.2f}", color=INK, fontsize=8.5, va="center")
        ax.set_yticks(y)
        ax.set_yticklabels(FATTORI if i == 0 else [""] * len(FATTORI), fontsize=8.5)
        ax.set_xlim(0, 6.6)
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("effetto massimo sul riscaldamento a 33.9 m (°C)")
        ax.axhline(0.5, color=AXIS, lw=0.8)
    fig.suptitle("Riscaldamento al fondo a Sturla nelle tre finestre: misura e modello 1D GOTM", x=0.01, ha="left",
                 fontweight="bold", fontsize=12)
    fig.text(0.01, 0.955, "Sopra: nero ADCP (in chiaro i valori orari, scuro la media su 25 h); verde R0, solo processi "
             "locali; arancione R3, processi locali + downwelling di MedFS + trasporto vicino al fondo.\nLe bande "
             "colorate raccolgono le varianti con onde, convezione e vento ERA5: sono strette perché danno lo stesso "
             "fondo.\nLa banda arancione chiara (4–13 ottobre) è R3 con altri tempi (6–24 h) e spessori (5–15 m) del rilassamento. In grigio le mareggiate.", color=INK2, fontsize=9.5, ha="left", va="top")
    fig.get_layout_engine().set(rect=(0, 0, 1, 0.915))
    _save(fig, f"{OUT}/8_matrice_gotm_sintesi.png")
    pd.DataFrame(tab, index=FATTORI).to_csv(os.path.join(arg.uscita, "8_matrice_gotm_sintesi.csv"))
    print(pd.DataFrame(tab, index=FATTORI).round(2).to_string())


if __name__ == "__main__":
    main()
