"""Figure dell'analisi armonica (slide 28 della presentazione): (12) marea nelle correnti e nella temperatura
al fondo, (13) bande in finestre di 3 giorni.

  python analisi/marea_correnti/figure_armonica.py --uscita out_marea_correnti

Legge armonica_correnti.pkl scritto da armonica_correnti.py nella stessa cartella.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))     # radice del repository, per adcp_pd0

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from adcp_pd0.plots import C1, C2, C3, INK, INK2, MUTED, GRID, AXIS, SURF, SEQ, _fmt_time, _save


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--uscita", default="out_marea_correnti", help="cartella con armonica_correnti.pkl e delle figure")
    OUT = ap.parse_args(argv).uscita
    d = pd.read_pickle(os.path.join(OUT, "armonica_correnti.pkl"))
    E, R2, TT, z = d["E"], d["R2"], d["TT"], d["z"]
    E, R2 = E[E.z < 30], R2[R2.z < 30]          # la cella piu' alta ha pochi dati validi
    COL = {"M2": C1, "K1": C2, "S2": C3, "O1": INK2}

    # ---------------- figura 12
    fig = plt.figure(figsize=(13.5, 6.4), layout="constrained")
    gs = fig.add_gridspec(1, 4, width_ratios=[1.15, 1.15, 0.75, 1.25])
    a1, a2, a3, a4 = [fig.add_subplot(gs[0, i]) for i in range(4)]
    for n in ("M2", "K1", "S2", "O1"):
        x = E[E.cost == n]
        if n in ("M2", "K1"):
            a1.fill_betweenx(x.z, (x["maj"] - x.maj_ci).clip(lower=0), x["maj"] + x.maj_ci, color=COL[n], alpha=0.12, lw=0)
        a1.plot(x["maj"], x.z, color=COL[n], lw=2.2 if n in ("M2", "K1") else 1.4, label=n)
    a1.legend(loc="lower right", fontsize=9, title="banda: intervallo al 95%\nper M2 e K1", title_fontsize=8)
    a1.set_xlabel("semiasse maggiore (cm/s)")
    a1.set_ylabel("quota dal fondale (m)")
    a1.set_xlim(0, 2.6)
    a1.set_title("Ampiezza della corrente di marea", loc="left")
    for n in ("M2", "K1"):
        x = E[(E.cost == n) & (E.snr >= 2)]
        a2.errorbar(x.g, x.z, xerr=x.g_ci, fmt="o", ms=3.5, color=COL[n], ecolor=COL[n], elinewidth=0.8, alpha=0.9,
                    capsize=0)
        a2.text(float(x.g.iloc[-1]) + 8, float(x.z.iloc[-1]), n, color=INK, fontsize=9, fontweight="bold", va="center")
    a2.set_xlim(0, 360)
    a2.set_xticks([0, 90, 180, 270, 360])
    a2.set_xlabel("fase g (°), solo SNR ≥ 2")
    a2.set_title("Fase con la quota", loc="left")
    a2.axhspan(12.5, 18, color=GRID, alpha=0.5, lw=0, zorder=0)
    a2.text(8, 15.2, "termoclino estivo\n(stima dal taglio)", fontsize=8, color=INK2, va="center")
    a3.plot(100 * R2.R2_marea, R2.z, color=INK, lw=2)
    a3.set_xlim(0, 3)
    a3.set_xlabel("varianza spiegata\ndalla marea (%)")
    a3.set_title("Peso della marea", loc="left")
    for a in (a2, a3):
        a.sharey(a1)
        plt.setp(a.get_yticklabels(), visible=False)
    # temperatura al fondo
    cs = ["M2", "S2", "K1", "O1"]
    xx = np.arange(len(cs))
    for j, (per, col, lab) in enumerate((("estate", MUTED, "10 luglio – 30 settembre"), ("ottobre", C1, "1–27 ottobre"))):
        tb = TT[per].set_index("cost").loc[cs]
        a4.bar(xx + (j - 0.5) * 0.36, tb.A, width=0.34, color=col, edgecolor=SURF, linewidth=2, label=lab, zorder=3)
        a4.errorbar(xx + (j - 0.5) * 0.36, tb.A, yerr=tb.A_ci, fmt="none", ecolor=INK, elinewidth=1, capsize=3, zorder=4)
    a4.set_xticks(xx)
    a4.set_xticklabels(cs)
    a4.set_ylabel("ampiezza (°C)")
    a4.grid(axis="x", visible=False)
    a4.legend(loc="upper right", fontsize=8.5)
    a4.set_title("Marea nella temperatura al fondo", loc="left")
    fig.suptitle("Analisi armonica delle correnti e della temperatura al fondo, 10 luglio – 27 ottobre 2026",
                 x=0.01, ha="left", fontweight="bold", fontsize=12)
    fig.text(0.01, 0.94, "UTide a ogni quota, 7 costituenti, intervalli di confidenza al 95% con rumore colorato. "
             "K1 comprende anche il ciclo giornaliero della brezza: in 110 giorni non si separa da S1 e P1.",
             color=INK2, fontsize=9.5, ha="left", va="top")
    fig.get_layout_engine().set(rect=(0, 0, 1, 0.92))
    _save(fig, os.path.join(OUT, "12_marea_correnti_temperatura.png"))

    # ---------------- figura 13
    c, r, rt = d["centri"], d["res"], d["resT"]
    fig = plt.figure(figsize=(13.5, 10.5), layout="constrained")
    gs = fig.add_gridspec(4, 2, width_ratios=[1, 0.018], height_ratios=[0.8, 1, 1, 1])
    axA = fig.add_subplot(gs[0, 0])
    storms = [("2026-08-21", "2026-08-22 23:00"), ("2026-10-08", "2026-10-09 23:00"), ("2026-10-20", "2026-10-22 23:00")]
    for k, col, lab in (("T_D2", C1, "semidiurna"), ("T_f", C2, "inerziale (17.1 h)"), ("T_D1", C3, "diurna")):
        s = pd.Series(rt[k], index=c)
        axA.plot(s.index, s.values, color=col, lw=2, label=lab)
    axA.set_ylabel("ampiezza (°C)")
    axA.legend(loc="upper left", fontsize=9, ncol=3)
    axA.set_title("Oscillazione della temperatura al fondo, per banda", loc="left")
    X = np.r_[c - pd.Timedelta("6h"), c[-1:] + pd.Timedelta("6h")]
    zz = np.r_[d["z"] - 0.25, d["z"][-1:] + 0.25]
    axs = [axA]
    for i, (k, tit) in enumerate((("D2_maj", "Corrente semidiurna: semiasse maggiore"),
                                   ("f_cw", "Corrente inerziale: componente oraria"),
                                   ("D1_maj", "Corrente diurna: semiasse maggiore"))):
        ax = fig.add_subplot(gs[i + 1, 0], sharex=axA)
        pm = ax.pcolormesh(X, zz, np.asarray(r[k]).T, cmap=SEQ, vmin=0, vmax=4, shading="flat")
        ax.set_ylabel("quota dal fondale (m)")
        ax.set_title(tit, loc="left")
        axs.append(ax)
    cax = fig.add_subplot(gs[1:, 1])
    fig.colorbar(pm, cax=cax, label="cm/s")
    for ax in axs:
        for a_, b_ in storms:
            ax.axvspan(pd.Timestamp(a_), pd.Timestamp(b_), color=GRID if ax is axA else "none", alpha=0.6, lw=0, zorder=0)
            if ax is not axA:
                ax.axvline(pd.Timestamp(a_), color=INK, lw=0.8, ls=":")
        ax.set_xlim(c[0], c[-1])
    for ax in axs[:-1]:
        plt.setp(ax.get_xticklabels(), visible=False)
    _fmt_time(axs[-1])
    fig.suptitle("Oscillazioni in finestre di 3 giorni: residuo più bande diurna, inerziale e semidiurna, a ogni quota",
                 x=0.01, ha="left", fontweight="bold", fontsize=12)
    fig.text(0.01, 0.962, "Fit ai minimi quadrati ogni 12 ore, come per l’ADCP trainato (Valle-Levinson, 2024). Le "
             "ampiezze comprendono anche l’energia non legata alla marea. In grigio e con le linee punteggiate le mareggiate.",
             color=INK2, fontsize=9.5, ha="left", va="top")
    fig.get_layout_engine().set(rect=(0, 0, 1, 0.945))
    _save(fig, os.path.join(OUT, "13_bande_finestre_mobili.png"))


if __name__ == "__main__":
    main()
