"""Figure dell'analisi del rimescolamento (quelle del report, sezione sulle cause).

  1_forzanti_e_stratificazione.png   temperatura, stratificazione, flusso di calore, vento lungo costa, onde
  2_energia_processi_locali.png      energia fornita dai processi locali in % di quella necessaria
  3_termoclino_modello_e_ADCP.png    quota del termoclino: modello contro stime dell'ADCP
  4_tempi_riscaldamento_fondo.png    riscaldamento al fondo: misura, modello 3D, modello 1D
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # anche con python -I

import os

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import comune as C
from col1d import Column
from adcp_pd0.plots import C1, C2, C3, INK, INK2, MUTED, GRID, AXIS, SURF, _fmt_time, _save

C4, C_VIOLA, C_ROSSO, C_GRIGIO = "#eda100", "#6250d6", "#e34948", "#9a998f"
H12 = pd.Timedelta(hours=12)
MAREGGIATE = [("2026-08-21 08:00", "2026-08-22 14:00"), ("2026-10-08 10:00", "2026-10-09 12:00"), ("2026-10-20 10:00", "2026-10-22 14:00")]


def figura_1(F, m, out):
    d = F.resample("1D").mean()
    t0, t1 = pd.Timestamp(C.PERIODO[0]), pd.Timestamp(C.PERIODO[1]).normalize() + pd.Timedelta("1D")
    fig, axs = plt.subplots(5, 1, figsize=(13, 13.5), sharex=True, layout="constrained")
    ax = axs[0]
    ax.plot(m.index + H12, m.T_adcp, color=C1, lw=2, label="ADCP, al fondo (34 m)")
    ax.plot(m.index + H12, m.T34, color=C1, lw=1.2, ls="--", label="modello, 34 m (3–4 km al largo)")
    ax.plot(m.index + H12, m.T1, color=C2, lw=1.4, label="modello, superficie")
    ax.set_ylabel("°C"); ax.set_title("Temperatura: la superficie si raffredda, il fondo si scalda di colpo a inizio ottobre")
    ax.legend(loc="upper right", ncols=3, fontsize=8.5)
    ax = axs[1]
    ax.plot(m.index + H12, m.dsig_0_34, color=C_VIOLA, lw=1.8)
    ax.set_ylabel("kg/m³"); ax.set_ylim(0, None)
    ax.set_title("Stratificazione nel modello: differenza di densità fra 34 m e superficie (a destra, in grigio, profondità del mixed layer)")
    ax2 = ax.twinx(); ax2.plot(m.index + H12, m.mld, color=C_GRIGIO, lw=1.2); ax2.set_ylabel("m"); ax2.set_ylim(50, 0)
    ax2.spines["right"].set_visible(True); ax2.grid(False)
    ax = axs[2]
    q = d.Q
    ax.bar(q.index + H12, q.values, width=0.85, color=np.where(q.values >= 0, C_ROSSO, C1), alpha=0.75)
    ax.axhline(0, color=AXIS, lw=0.8); ax.set_ylabel("W/m²")
    ax.set_title("Flusso di calore netto in superficie, ERA5 (rosso: il mare guadagna calore; blu: lo perde)")
    ax = axs[3]
    tl = 1e3 * d.tau_ls
    ax.bar(tl.index + H12, tl.values, width=0.85, color=np.where(tl.values >= 0, C3, C_GRIGIO), alpha=0.85)
    ax.axhline(0, color=AXIS, lw=0.8); ax.set_ylabel("mN/m²")
    ax.set_title("Sforzo del vento lungo costa (verde: verso ovest, favorevole al downwelling); vento CMEMS, probabilmente sottostimato")
    ax = axs[4]
    ax.plot(d.index + H12, d.Hm0, color=C2, lw=1.6, label="Hm0 (m), ADCP")
    ax.set_ylabel("m"); ax.set_ylim(0, None)
    ax2 = ax.twinx(); ax2.plot(d.index + H12, d.La_t, color=INK2, lw=1, label="numero di Langmuir La_t")
    ax2.axhline(0.5, color=INK2, lw=0.7, ls=":"); ax2.set_ylim(0, 3); ax2.set_ylabel("La_t"); ax2.spines["right"].set_visible(True); ax2.grid(False)
    ax.set_title("Onde: altezza significativa e numero di Langmuir (sotto 0.5, linea punteggiata, prevale la turbolenza di Langmuir)")
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels(); ax.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=8.5)
    for a_, z_ in MAREGGIATE:
        for ax in axs:
            ax.axvspan(pd.Timestamp(a_), pd.Timestamp(z_), color=C2, alpha=0.12, lw=0)
    axs[-1].set_xlim(t0, t1); _fmt_time(axs[-1])
    fig.suptitle("Cosa è cambiato prima e durante il rimescolamento di ottobre (fasce arancioni: le tre mareggiate principali)", x=0.01, ha="left", fontweight="bold")
    _save(fig, os.path.join(out, "figure", "1_forzanti_e_stratificazione.png"))


def figura_2(T, out):
    T = T.copy(); T.index = T.index.str.strip()
    rows = [("21–22 agosto", "21-22 agosto"), ("4–7 ottobre, prima della mareggiata", "prima della mareggiata"),
            ("7–9 ottobre, durante la mareggiata", "durante la mareggiata"), ("19–22 ottobre", "19-22 ottobre")]
    parts = [("raffreddamento in superficie", C1, lambda r: r.raffreddamento + r.convezione),
             ("vento", C2, lambda r: r.vento),
             ("onde (Langmuir, frangimento, fondo)", C3, lambda r: r.Langmuir + r.frangimento + r.onde_fondo),
             ("attrito della corrente sul fondo", C4, lambda r: r.corrente_fondo)]
    XMAX = 200
    fig, ax = plt.subplots(figsize=(12.5, 5.2), layout="constrained")
    ax.set_axisbelow(True)
    for k, (lab, key) in enumerate(rows):
        r = T.loc[key]; y = len(rows) - 1 - k; need = r.necessaria; x0 = 0.0
        for _, col, fn in parts:
            w = 100 * fn(r) / need
            ax.barh(y, w, left=x0, height=0.5, color=col, edgecolor=SURF, linewidth=1.5)
            x0 += w
        ax.text(min(x0, XMAX) + 1.5, y + 0.02, f"{x0:.0f}%", va="center", ha="left", color=INK, fontsize=10, fontweight="bold")
        ax.plot(100 * r.variazione_3D / need, y, marker="D", ms=9, color=INK, mec=SURF, mew=1.5, zorder=5)
        x2 = 100 * r.vento_x2_totale / need
        if x2 <= XMAX:
            ax.plot(x2, y, marker="o", ms=9, mfc="none", mec=INK2, mew=1.8, zorder=5)
        else:
            ax.annotate(f"{x2:.0f}% →", (XMAX, y), xytext=(XMAX - 1, y + 0.33), ha="right", va="center", color=INK2, fontsize=9)
            ax.plot(XMAX, y, marker=">", ms=8, mfc="none", mec=INK2, mew=1.8, zorder=5, clip_on=False)
        ax.text(-2, y, lab + "\nservivano " + f"{need:,.0f}".replace(",", " ") + " J/m²", ha="right", va="center", color=INK, fontsize=10)
    ax.axvline(100, color=INK2, lw=1.2, ls="--")
    ax.text(100.8, len(rows) - 0.55, "energia necessaria\nper rimescolare la colonna", color=INK2, fontsize=9, va="center")
    ax.set_xlim(0, XMAX); ax.set_ylim(-0.6, len(rows) - 0.4); ax.set_yticks([])
    ax.set_xlabel("% dell'energia necessaria a inizio finestra")
    ax.spines["left"].set_visible(False)
    h = [Patch(color=c, label=n) for n, c, _ in parts]
    h += [Line2D([], [], marker="D", ls="", color=INK, ms=8, label="modello 3D: stratificazione tolta davvero"),
          Line2D([], [], marker="o", ls="", mfc="none", mec=INK2, mew=1.8, ms=8, label="processi locali con vento doppio")]
    ax.legend(handles=h, loc="lower center", bbox_to_anchor=(0.42, 1.0), ncols=3, fontsize=9, frameon=False)
    fig.suptitle("Quanta energia hanno fornito i processi locali, in percentuale di quella necessaria a rimescolare i 34.5 m d'acqua",
                 x=0.01, ha="left", fontweight="bold")
    _save(fig, os.path.join(out, "figure", "2_energia_processi_locali.png"))


def figura_3(m, out):
    t0, t1 = pd.Timestamp(C.PERIODO[0]), pd.Timestamp(C.PERIODO[1]).normalize() + pd.Timedelta("1D")
    fig, ax = plt.subplots(figsize=(13, 4.8), layout="constrained")
    ax.plot(m.index + H12, m.z_tc_mod, color=INK2, lw=1.8, label="modello: massimo gradiente di temperatura")
    ax.plot(m.index + H12, m.mld, color=C_GRIGIO, lw=1, ls="--", label="modello: profondità del mixed layer")
    ax.plot(m.index + H12, m.prof_taglio, "o", ms=4, color=C2, label="ADCP: massimo del taglio (stime affidabili)")
    ax.plot(m.index + H12, m.prof_eco, ".", ms=4, color=C1, alpha=0.7, label="ADCP: picco dell'eco")
    ax.axhline(C.H, color=AXIS, lw=1); ax.text(t0 + pd.Timedelta(days=1), C.H + 1.1, f"fondo alla stazione ({C.H} m)", fontsize=8.5, color=INK2, va="top")
    ax.set_ylim(52, 0); ax.set_ylabel("profondità (m)")
    x = m[["prof_taglio", "z_tc_mod"]].dropna(); y = m[["prof_eco", "z_tc_mod"]].dropna()
    ax.set_title(f"Il modello conferma la stima dal taglio (correlazione {x.corr().iloc[0, 1]:.2f}), "
                 f"non quella dall'eco (correlazione {y.corr().iloc[0, 1]:.2f})")
    ax.legend(loc="lower left", ncols=2, fontsize=8.5)
    ax.set_xlim(t0, t1); _fmt_time(ax)
    _save(fig, os.path.join(out, "figure", "3_termoclino_modello_e_ADCP.png"))


def figura_4(E, F, zm, tm, th, so, out):
    D3 = E["D3"][C.H]; Ta = E["Tadcp"]
    cols = [("21–22 agosto", "21-22 agosto"), ("5–9 ottobre", "5-9 ottobre"), ("19–22 ottobre", "19-22 ottobre")]
    fig, axs = plt.subplots(3, 3, figsize=(13.5, 8.6), sharey="row", layout="constrained", gridspec_kw=dict(height_ratios=[2.2, 1, 1]))
    for j, (lab, ev) in enumerate(cols):
        a, b = (pd.Timestamp(x) for x in E["EV"][ev])
        lo, hi = a - pd.Timedelta("1D"), b + pd.Timedelta("1D")
        i0 = int(np.argmin(np.abs(tm - a)))
        c0 = Column(C.H, zm, th[i0], so[i0], bbl=2.0).at(C.PROF_SENSORE)
        ax = axs[0, j]
        ta0 = Ta.loc[a - pd.Timedelta("3h"):a + pd.Timedelta("3h")].mean()
        x = Ta.loc[lo - pd.Timedelta("2D"):hi + pd.Timedelta("2D")] - ta0
        xm = x.rolling(25, center=True, min_periods=13).mean().loc[lo:hi]; x = x.loc[lo:hi]
        ax.plot(x.index, x.values, color=C1, lw=1, alpha=0.35)
        ax.plot(xm.index, xm.values, color=C1, lw=2.4, label="misura ADCP al fondo (media su 25 h; in chiaro i valori orari)")
        y = D3.T_probe.loc[lo:hi] - D3.T_probe.loc[a]
        ax.plot(y.index, y.values, color=INK, lw=1.5, ls="--", label=f"modello 3D (con trasporto), {C.PROF_SENSORE} m")
        for key, ls, l_ in (("tutto (vento, onde, fondo)", "-", "modello 1D, solo processi locali"),
                            ("tutto, vento x2", ":", "modello 1D, solo processi locali, vento doppio")):
            s = E["series"][(ev, "stazione", key)].T_probe
            s = pd.concat([pd.Series([c0], index=[a]), s]) - c0
            ax.plot(s.index, s.values, color=C2, lw=2, ls=ls, label=l_)
        ax.axhline(0, color=AXIS, lw=0.8)
        ax.set_title(lab, loc="left", fontweight="bold")
        if j == 0:
            ax.set_ylabel("riscaldamento al fondo\ndall'inizio della finestra (°C)")
        ax = axs[1, j]
        hm = F.Hm0.loc[lo:hi]
        ax.plot(hm.index, hm.values, color=INK2, lw=1.6); ax.set_ylim(0, 2.2)
        if j == 0:
            ax.set_ylabel("Hm0 (m)")
        ax = axs[2, j]
        tl = 1e3 * F.tau_ls.loc[lo:hi].rolling(3, center=True).mean()
        ax.bar(tl.index, tl.values, width=1 / 24, color=MUTED, lw=0)
        ax.axhline(0, color=AXIS, lw=0.8)
        if j == 0:
            ax.set_ylabel("vento lungo costa\n(mN/m², + = downwelling)")
        st = F.Hm0.loc[lo:hi] >= 1.0
        for r_ in range(3):
            for t_ in st[st].index:
                axs[r_, j].axvspan(t_ - pd.Timedelta("30min"), t_ + pd.Timedelta("30min"), color=GRID, alpha=0.6, lw=0, zorder=0)
            axs[r_, j].axvspan(a, b, ymin=0, ymax=0.02, color=INK2, lw=0)
            axs[r_, j].set_xlim(lo, hi)
        _fmt_time(axs[2, j], maxticks=5)
        for r_ in (0, 1):
            axs[r_, j].tick_params(labelbottom=False)
    h_, l_ = axs[0, 0].get_legend_handles_labels()
    fig.legend(h_, l_, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncols=4, fontsize=9.5, frameon=False)
    fig.suptitle("Quando si scalda il fondo: misura, modello 3D con il trasporto, modello 1D con i soli processi locali",
                 x=0.01, ha="left", fontweight="bold", y=1.095)
    fig.text(0.01, 1.055, "A inizio ottobre il fondo guadagna circa 2 °C prima della mareggiata e lo segue solo il modello con il trasporto; "
             "a fine ottobre bastano i processi locali. Grigio: Hm0 ≥ 1 m; barra scura: periodo simulato.", ha="left", color=INK2, fontsize=10)
    _save(fig, os.path.join(out, "figure", "4_tempi_riscaldamento_fondo.png"))


def main(argv=None):
    a = C.argomenti(__doc__, argv)
    F = C.leggi(a.uscita, "forzanti_orari.csv")
    m = C.leggi(a.uscita, "giornaliero_modello.csv")
    T = pd.read_csv(os.path.join(a.uscita, "contributi_finali.csv"), index_col=0)
    E = pd.read_pickle(os.path.join(a.uscita, "esperimenti_1d.pkl"))
    zm, tm, th, so = C.modello_punto(a.dati)
    figura_1(F, m, a.uscita)
    figura_2(T, a.uscita)
    figura_3(m, a.uscita)
    figura_4(E, F, zm, tm, th, so, a.uscita)


if __name__ == "__main__":
    main()
