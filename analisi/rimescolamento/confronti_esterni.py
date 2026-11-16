"""Confronti con altri dati: (10) sforzo del vento ERA5 contro CMEMS, (11) corrente lungo costa dell'ADCP
contro il modello MedFS. Slide 36 e 37 della presentazione.

  python analisi/rimescolamento/confronti_esterni.py --dati ... --risultati ... --gotm ... --uscita ...

Servono in --dati il vento ERA5 (scarica_era5.py) e le correnti del riquadro MedFS (scarica_dati.sh).
Uscite: figure/10_vento_era5_cmems.png, figure/11_correnti_adcp_medfs.png
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
from adcp_pd0.plots import C1, C2, INK, INK2, MUTED, GRID, AXIS, SURF, _fmt_time, _save


def main(argv=None):
    arg = CM.argomenti(__doc__, argv)
    OUT = os.path.join(arg.uscita, "figure")
    AX = np.deg2rad(CM.ASSE_COSTA)          # asse della costa, verso ONO


    def lungo(df):
        return df.x * np.sin(AX) + df.y * np.cos(AX)


    # ---------- figura 10: vento. Sforzo CMEMS: quello usato nelle run GOTM (data/.../momentum_flux.dat);
    # senza --gotm, quello calcolato da forzanti.py con la formula di Large e Yeager (circa 5% piu' alto).
    if arg.gotm:
        p = G.trova_dat(arg.gotm, "momentum_flux.dat", preferisci="finestre_common")
        print("sforzo CMEMS da", p)
        tc = G.leggi_dat(p)
    else:
        F = CM.leggi(arg.uscita, "forzanti_orari.csv")
        tc = F[["taux", "tauy"]].rename(columns={"taux": "x", "tauy": "y"})
        print("sforzo CMEMS da forzanti_orari.csv (formula di Large e Yeager)")
    te = CM.sforzo_era5_centrato(CM.era5_vento(arg.dati)).rename(columns={"tx": "x", "ty": "y"})
    te = te.resample("1h").mean().interpolate(limit=2)
    lc, le = lungo(tc), lungo(te)
    FIN = [("19–25 ago", "2026-08-19", "2026-08-25 23:00"), ("4–9 ott", "2026-10-04", "2026-10-09 23:00"),
           ("18–22 ott", "2026-10-18", "2026-10-22 23:00")]
    fig = plt.figure(figsize=(13.5, 5.6), layout="constrained")
    gs = fig.add_gridspec(1, 3, width_ratios=[2.4, 0.05, 1])
    ax = fig.add_subplot(gs[0, 0])
    lo, hi = pd.Timestamp("2026-10-01"), pd.Timestamp("2026-10-28")
    for a, b in (("2026-10-08", "2026-10-09 23:00"), ("2026-10-20", "2026-10-22 23:00")):
        ax.axvspan(pd.Timestamp(a), pd.Timestamp(b), color=GRID, alpha=0.55, lw=0, zorder=0)
    ax.axhline(0, color=AXIS, lw=0.8)
    for s, col, lab in ((lc, C1, "CMEMS (ECMWF corretto con scatterometri)"), (le, C2, "ERA5")):
        x = s.loc[lo:hi].rolling(3, center=True, min_periods=1).mean()
        ax.plot(x.index, x.values, color=col, lw=1.8, label=lab)
    ax.set_xlim(lo, hi)
    _fmt_time(ax)
    ax.set_ylabel("sforzo del vento lungo costa (N/m²)")
    ax.text(lo + pd.Timedelta("6h"), 0.27, "positivo: verso ovest-nord-ovest,\nfavorevole al downwelling", color=INK2,
            fontsize=9, va="top")
    ax.legend(loc="lower left", fontsize=9)
    ax.set_title("Ottobre: sforzo lungo costa, media su 3 ore", loc="left")
    axb = fig.add_subplot(gs[0, 2])
    y = np.arange(len(FIN))[::-1]
    for k, (nome, a, b) in enumerate(FIN):
        vc = float(np.hypot(tc.x, tc.y).loc[a:b].mean())
        ve = float(np.hypot(te.x, te.y).loc[a:b].mean())
        axb.barh(y[k] + 0.18, vc, height=0.34, color=C1, edgecolor=SURF, linewidth=2)
        axb.barh(y[k] - 0.18, ve, height=0.34, color=C2, edgecolor=SURF, linewidth=2)
        axb.text(vc + 0.002, y[k] + 0.18, f"{vc:.3f}", va="center", fontsize=8.5, color=INK)
        axb.text(ve + 0.002, y[k] - 0.18, f"{ve:.3f}", va="center", fontsize=8.5, color=INK)
    axb.set_yticks(y)
    axb.set_yticklabels([f[0] for f in FIN])
    axb.set_xlim(0, 0.08)
    axb.grid(axis="y", visible=False)
    axb.set_xlabel("sforzo medio (N/m²)")
    axb.set_title("Sforzo medio nelle finestre", loc="left")
    r = pd.concat([lc.rename("c"), le.rename("e")], axis=1).dropna().loc["2026-07-10":"2026-10-27"]
    cc = float(r.corr().iloc[0, 1])
    fig.suptitle("Vento: ERA5 contro CMEMS sopra Sturla", x=0.01, ha="left", fontweight="bold", fontsize=12)
    fig.text(0.01, 0.9, f"Correlazione oraria dello sforzo lungo costa {cc:.2f} (10 luglio – 27 ottobre). "
             "Simili il 4–9 ottobre; il 18–22 ottobre ERA5 dà uno sforzo doppio, con un Levante più forte. "
             "In grigio le mareggiate.", color=INK2, fontsize=9.5, ha="left", va="top")
    fig.get_layout_engine().set(rect=(0, 0, 1, 0.86))
    _save(fig, f"{OUT}/10_vento_era5_cmems.png")
    print("vento: correlazione lungo costa", round(cc, 2))

    # ---------- figura 11: correnti
    h = CM.serie_orarie(arg.risultati)
    h = h[h.deployment_qc == 1]
    a = np.deg2rad(h.corrente_media_dir)
    ua, va = h.corrente_media_vel * np.sin(a), h.corrente_media_vel * np.cos(a)
    ladcp = (ua * np.sin(AX) + va * np.cos(AX)).asfreq("1h")
    C = xr.open_dataset(CM.trova(arg.dati, "*medfs_riquadro_correnti*.nc"))
    p = C.sel(latitude=44.3542, longitude=9.0, method="nearest").sel(depth=slice(2.5, 32))
    um, vm = p.uo.mean("depth").to_series(), p.vo.mean("depth").to_series()
    um.index, vm.index = pd.DatetimeIndex(um.index), pd.DatetimeIndex(vm.index)
    lmed = um * np.sin(AX) + vm * np.cos(AX)
    A25 = ladcp.interpolate(limit=3).rolling(25, center=True, min_periods=13).mean()
    M25 = lmed.rolling(25, center=True, min_periods=13).mean()
    j = pd.concat([A25.rename("a"), M25.rename("m")], axis=1).dropna().loc["2026-07-10":"2026-10-27"]
    cr = float(j.corr().iloc[0, 1])
    fig = plt.figure(figsize=(13.5, 5.6), layout="constrained")
    gs = fig.add_gridspec(1, 2, width_ratios=[2.6, 1])
    ax = fig.add_subplot(gs[0, 0])
    for a_, b_ in (("2026-08-21", "2026-08-22 23:00"), ("2026-10-08", "2026-10-09 23:00"), ("2026-10-20", "2026-10-22 23:00")):
        ax.axvspan(pd.Timestamp(a_), pd.Timestamp(b_), color=GRID, alpha=0.55, lw=0, zorder=0)
    ax.axhline(0, color=AXIS, lw=0.8)
    ax.plot(j.index, 100 * j.a, color=INK, lw=2, label="ADCP, media sulla colonna misurata")
    ax.plot(j.index, 100 * j.m, color=C1, lw=1.8, label="MedFS, punto a 91 m di fondale, media 2.5–32 m")
    ax.set_ylabel("corrente lungo costa (cm/s)\npositiva verso ovest-nord-ovest")
    ax.legend(loc="upper left", fontsize=9)
    ax.set_xlim(j.index[0], j.index[-1])
    _fmt_time(ax)
    ax.set_title("Corrente lungo costa, media su 25 ore", loc="left")
    axs = fig.add_subplot(gs[0, 1])
    axs.scatter(100 * j.m.iloc[::6], 100 * j.a.iloc[::6], s=10, color=INK2, alpha=0.5, lw=0)
    lim = [-25, 35]
    axs.plot(lim, lim, color=AXIS, lw=1)
    axs.set_xlim(lim)
    axs.set_ylim(lim)
    axs.set_aspect("equal")
    axs.set_xlabel("MedFS (cm/s)")
    axs.set_ylabel("ADCP (cm/s)")
    axs.set_title(f"Ogni 6 ore: correlazione {cr:.2f}", loc="left")
    fig.suptitle("Correnti: ADCP contro il modello MedFS", x=0.01, ha="left", fontweight="bold", fontsize=12)
    fig.text(0.01, 0.9, f"Media ADCP {100 * j.a.mean():.1f} cm/s, MedFS {100 * j.m.mean():.1f} cm/s verso ovest-nord-ovest; "
             f"deviazione standard {100 * j.a.std():.1f} e {100 * j.m.std():.1f} cm/s. In grigio le mareggiate.",
             color=INK2, fontsize=9.5, ha="left", va="top")
    fig.get_layout_engine().set(rect=(0, 0, 1, 0.86))
    _save(fig, f"{OUT}/11_correnti_adcp_medfs.png")
    print("correnti: corr", round(cr, 2), "media", round(100 * j.a.mean(), 1), round(100 * j.m.mean(), 1),
          "std", round(100 * j.a.std(), 1), round(100 * j.m.std(), 1))


if __name__ == "__main__":
    main()
