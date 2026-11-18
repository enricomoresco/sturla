"""Confronto fra le onde misurate dall'ADCP e quelle di una boa ondametrica vicina.

  python analisi/confronto_boa/confronto_boa.py --boa archive_data_2.csv --risultati out \
         --uscita out_confronto_boa [--dati dati_esterni] [--sposta-adcp-ore -2]

--boa        CSV della boa con le colonne UTC, Hs (m), Tp (sec), Ta (sec), Dd (deg) (e lat, lon, sst)
--risultati  cartella dei risultati dell'ADCP (serve postprocessing/tabelle/onde_parametri.csv)
--dati       facoltativo: cartella con il livello del riquadro MedFS (scarica_dati.sh), per il controllo
             dell'orologio con le fasi di marea (serve UTide)
--sposta-adcp-ore  ore da sommare ai tempi dell'ADCP per portarli in UTC; senza, si usa lo
             spostamento intero che rende massima la correlazione dell'altezza d'onda

Passi:
1. Orologio. Correlazione fra Hm0 dell'ADCP e Hs della boa spostando i tempi dell'ADCP da -6 a +6 ore
   (la boa e' in UTC); con --dati, fasi di M2, S2, N2, K1 e O1 nel livello dell'ADCP e in quello di
   MedFS (in UTC): la differenza di fase divisa per la velocita' angolare di ogni costituente da' lo
   sfasamento dell'orologio.
2. Coppie. Ogni burst dell'ADCP (tempo al centro del burst, spostato in UTC) con il dato della boa piu'
   vicino, entro 30 minuti.
3. Statistiche: differenza media (ADCP - boa), scarto quadratico medio, indice di dispersione,
   correlazione e rapporto mediano per Hm0 contro Hs (anche per classi di Hs), Tm01 e Tm02 contro Ta,
   Tp contro Tp, direzione di picco contro Dd; massimi delle mareggiate.
Uscite: confronto_boa_coppie.csv, confronto_boa_statistiche.csv, confronto_boa_mareggiate.csv,
figure/14_onde_adcp_boa.png, figure/15_orologio_adcp.png
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))     # radice del repository, per adcp_pd0

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from adcp_pd0.plots import C1, C2, C3, INK, INK2, MUTED, GRID, AXIS, _fmt_time, _save

TOLLERANZA_MIN = 30
CLASSI_HS = [(0, 0.3), (0.3, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 9.0)]
MAREGGIATE = [("21–22 agosto", "2026-08-20 12:00", "2026-08-23 00:00"),
              ("8–9 ottobre", "2026-10-08 00:00", "2026-10-10 00:00"),
              ("20–22 ottobre", "2026-10-20 00:00", "2026-10-23 00:00")]
OMEGA = {"M2": 28.9841042, "S2": 30.0, "N2": 28.4397295, "K1": 15.0410686, "O1": 13.9430356}   # gradi all'ora


def trova(cartella, schema):
    hit = sorted(glob.glob(os.path.join(cartella, "**", schema), recursive=True))
    if not hit:
        raise FileNotFoundError(f"nessun file {schema} in {cartella}")
    return hit[-1]


def leggi_boa(p) -> pd.DataFrame:
    b = pd.read_csv(p, skipinitialspace=True)
    b.columns = [c.strip() for c in b.columns]
    nomi = {"Hs (m)": "Hs", "Tp (sec)": "Tp", "Ta (sec)": "Ta", "Dd (deg)": "Dd", "SLP(mB)": "slp"}
    b = b.rename(columns=nomi)
    b["t"] = pd.to_datetime(b["UTC"])
    b = b.sort_values("t").drop_duplicates("t").set_index("t")
    return b


def leggi_adcp(risultati) -> pd.DataFrame:
    w = pd.read_csv(trova(risultati, "onde_parametri.csv"), comment="#", index_col=0, parse_dates=True)
    w = w[w.wave_qc.isin([1, 3])].copy()
    w["centro"] = w.index + pd.to_timedelta(w.n_samples.values * 0.25, unit="s")     # 2 Hz: meta' del burst
    return w


def accoppia(w, b, sposta_h, tol_min=TOLLERANZA_MIN) -> pd.DataFrame:
    """Coppie burst ADCP - dato della boa piu' vicino entro tol_min, con i tempi dell'ADCP spostati."""
    bt = b.index.values
    tq = (w.centro + pd.Timedelta(hours=sposta_h)).values
    i = np.searchsorted(bt, tq)
    i0, i1 = np.clip(i - 1, 0, len(bt) - 1), np.clip(i, 0, len(bt) - 1)
    d0 = np.abs((tq - bt[i0]) / np.timedelta64(1, "m"))
    d1 = np.abs((bt[i1] - tq) / np.timedelta64(1, "m"))
    j = np.where(d0 <= d1, i0, i1)
    d = np.minimum(d0, d1)
    ok = d <= tol_min
    P = w[ok].copy()
    P["tempo_utc"] = tq[ok]
    for c in ("Hs", "Tp", "Ta", "Dd"):
        P[f"{c}_boa"] = b[c].values[j[ok]]
    P["scarto_tempo_min"] = d[ok]
    return P.set_index("tempo_utc")


def correlazione_sfasamenti(w, b, passi=np.arange(-6, 6.01, 0.25)):
    """Correlazione Hm0-Hs spostando i tempi dell'ADCP; la boa e' interpolata solo fra dati distanti al
    piu' 1.5 ore (tratti a campionamento orario)."""
    t0 = b.index[0]
    tb = ((b.index - t0) / pd.Timedelta("1h")).to_numpy()
    out = []
    for L in passi:
        tq = ((w.centro + pd.Timedelta(hours=float(L))) - t0) / pd.Timedelta("1h")
        tq = tq.to_numpy()
        i = np.searchsorted(tb, tq)
        ok = (i > 0) & (i < len(tb))
        ii = np.clip(i, 1, len(tb) - 1)
        ok &= (tb[ii] - tb[ii - 1]) <= 1.5
        v = np.interp(tq[ok], tb, b.Hs.values)
        out.append((float(L), float(np.corrcoef(w.Hm0.values[ok], v)[0, 1]), int(ok.sum())))
    return pd.DataFrame(out, columns=["sposta_h", "r", "n"])


def fasi_di_marea(risultati, dati):
    """Sfasamento dell'orologio dell'ADCP dalle fasi di marea del livello, contro MedFS (UTC)."""
    import utide
    import xarray as xr
    h = pd.read_csv(trova(risultati, "serie_orarie.csv"), index_col=0, parse_dates=True)
    lv = h.livello_m.where(h.deployment_qc == 1).dropna()          # tempo = centro dell'ora dell'ADCP
    Z = xr.open_dataset(trova(dati, "*medfs_riquadro_livello*.nc")).zos
    m = Z.sel(latitude=44.3542, longitude=9.0, method="nearest").to_series()
    m.index = pd.DatetimeIndex(m.index)
    m = m[lv.index.min() - pd.Timedelta("1h"): lv.index.max() + pd.Timedelta("1h")].dropna()
    cost = list(OMEGA)

    def fit(s):
        co = utide.solve(s.index.values, s.values, lat=44.38, constit=cost, method="ols", conf_int="linear",
                         trend=True, verbose=False)
        return pd.DataFrame({"A_cm": 100 * co.A, "g": co.g, "g_ci": co.g_ci}, index=co.name)

    a, f = fit(lv), fit(m)
    T = a.join(f, lsuffix="_adcp", rsuffix="_medfs")
    T["dg"] = (T.g_adcp - T.g_medfs + 180) % 360 - 180
    T["sfasamento_h"] = [T.dg[k] / OMEGA[k] for k in T.index]
    T["incertezza_h"] = [np.hypot(T.g_ci_adcp[k], T.g_ci_medfs[k]) / OMEGA[k] for k in T.index]
    return T


def con_buchi(s, ore=6):
    """Serie con un NaN dove mancano dati per piu' di `ore` ore, per non disegnare linee sui buchi."""
    s = s.dropna()
    salto = s.index.to_series().diff() > pd.Timedelta(hours=ore)
    buchi = pd.Series(np.nan, index=s.index[salto.values] - pd.Timedelta(minutes=1))
    return pd.concat([s, buchi]).sort_index()


def statistiche(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    e = x - y
    return dict(n=len(x), bias=e.mean(), rmse=np.sqrt((e ** 2).mean()), indice_dispersione=np.sqrt((e ** 2).mean()) / y.mean(),
                r=np.corrcoef(x, y)[0, 1] if len(x) > 2 else np.nan, rapporto_mediano=np.median(x / y))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--boa", required=True)
    ap.add_argument("--risultati", required=True)
    ap.add_argument("--uscita", default="out_confronto_boa")
    ap.add_argument("--dati", default=None)
    ap.add_argument("--sposta-adcp-ore", type=float, default=None)
    a = ap.parse_args(argv)
    os.makedirs(os.path.join(a.uscita, "figure"), exist_ok=True)
    b, w = leggi_boa(a.boa), leggi_adcp(a.risultati)
    print(f"boa: {len(b)} dati dal {b.index[0]} al {b.index[-1]}, posizione mediana "
          f"{b.lat.median():.5f} N {b.lon.median():.5f} E")

    # 1. orologio
    C = correlazione_sfasamenti(w, b)
    best = C.loc[C.r.idxmax()]
    print(f"correlazione Hm0-Hs: massima ({best.r:.3f}) spostando l'ADCP di {best.sposta_h:+.2f} h; "
          f"senza spostamento {C.set_index('sposta_h').r[0.0]:.3f}")
    marea = None
    if a.dati:
        try:
            marea = fasi_di_marea(a.risultati, a.dati)
            print("fasi di marea, ADCP meno MedFS:")
            print(marea[["A_cm_adcp", "A_cm_medfs", "g_adcp", "g_medfs", "dg", "sfasamento_h", "incertezza_h"]].round(2).to_string())
            marea.to_csv(os.path.join(a.uscita, "confronto_boa_fasi_marea.csv"))
        except (ImportError, FileNotFoundError) as e:
            print(f"controllo con la marea saltato: {e}")
    sposta = a.sposta_adcp_ore if a.sposta_adcp_ore is not None else float(np.round(best.sposta_h))
    print(f"tempi dell'ADCP spostati di {sposta:+.1f} h per il confronto")

    # 2. coppie e 3. statistiche
    P = accoppia(w, b, sposta)
    P.to_csv(os.path.join(a.uscita, "confronto_boa_coppie.csv"))
    righe = [dict(grandezza="Hm0 contro Hs", classe="tutte", **statistiche(P.Hm0, P.Hs_boa))]
    for lo, hi in CLASSI_HS:
        q = P[(P.Hs_boa >= lo) & (P.Hs_boa < hi)]
        righe.append(dict(grandezza="Hm0 contro Hs", classe=f"Hs boa {lo}-{hi} m", **statistiche(q.Hm0, q.Hs_boa)))
    mf = P[P.Hs_boa >= 0.5]
    for k in ("Tm01", "Tm02"):
        righe.append(dict(grandezza=f"{k} contro Ta", classe="Hs boa >= 0.5 m", **statistiche(mf[k], mf.Ta_boa)))
    righe.append(dict(grandezza="Tp contro Tp", classe="Hs boa >= 0.5 m", **statistiche(mf.Tp, mf.Tp_boa),
                      entro_1s_pct=100 * (np.abs(mf.Tp - mf.Tp_boa) <= 1).mean()))
    dd = (P.Dp - P.Dd_boa + 180) % 360 - 180
    for nome, m in (("direzione affidabile, Hm0 >= 0.5 m", (P.direction_qc == 1) & (P.Hm0 >= 0.5)),
                    ("Hm0 >= 1 m", (P.direction_qc <= 3) & (P.Hm0 >= 1.0))):
        righe.append(dict(grandezza="Dp meno Dd (gradi)", classe=nome, n=int(m.sum()), bias=dd[m].mean(),
                          rmse=np.sqrt((dd[m] ** 2).mean()), dev_std=dd[m].std(), entro_20_pct=100 * (dd[m].abs() < 20).mean()))
    S = pd.DataFrame(righe)
    S.to_csv(os.path.join(a.uscita, "confronto_boa_statistiche.csv"), index=False)
    print(S.round(3).to_string(index=False))
    Wu = w.set_index(w.centro + pd.Timedelta(hours=sposta)).Hm0
    mar = []
    for nome, t0, t1 in MAREGGIATE:
        x, y = Wu[t0:t1], b.Hs[t0:t1]
        if x.empty or y.empty:
            continue
        tb = y.idxmax()
        vic = Wu[tb - pd.Timedelta("90min"): tb + pd.Timedelta("90min")]
        mar.append(dict(mareggiata=nome, boa_max=y.max(), boa_ora=tb, adcp_max=x.max(), adcp_ora=x.idxmax(),
                        adcp_all_ora_della_boa=vic.mean() if len(vic) else np.nan))
    M = pd.DataFrame(mar)
    M.to_csv(os.path.join(a.uscita, "confronto_boa_mareggiate.csv"), index=False)
    print(M.to_string(index=False))

    figura_onde(P, b, Wu, sposta, os.path.join(a.uscita, "figure", "14_onde_adcp_boa.png"))
    figura_orologio(C, marea, sposta, os.path.join(a.uscita, "figure", "15_orologio_adcp.png"))


def figura_onde(P, b, Wu, sposta, path):
    fig = plt.figure(figsize=(13.5, 9.6), layout="constrained")
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.15])
    ax = fig.add_subplot(gs[0, :])
    lo, hi = P.index.min() - pd.Timedelta("12h"), P.index.max() + pd.Timedelta("12h")
    for _, t0, t1 in MAREGGIATE:
        ax.axvspan(pd.Timestamp(t0), pd.Timestamp(t1), color=GRID, alpha=0.55, lw=0, zorder=0)
    y = con_buchi(b.Hs[lo:hi])
    ax.plot(y.index, y.values, color=C2, lw=1.1, label="boa, Hs")
    x = con_buchi(Wu[lo:hi])
    ax.plot(x.index, x.values, color=C1, lw=1.1, label=f"ADCP, Hm0 (tempi spostati di {sposta:+.0f} h, in UTC)")
    ax.set_xlim(lo, hi)
    ax.set_ylim(0, None)
    ax.set_ylabel("altezza d'onda significativa (m)")
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    ax.set_title("Altezza d'onda: boa e ADCP; in grigio le mareggiate", loc="left")
    _fmt_time(ax)
    # altezza
    ax = fig.add_subplot(gs[1, 0])
    ax.scatter(P.Hs_boa, P.Hm0, s=10, color=C1, alpha=0.45, lw=0)
    lim = max(P.Hs_boa.max(), P.Hm0.max()) * 1.05
    ax.plot([0, lim], [0, lim], color=AXIS, lw=1)
    cl = pd.cut(P.Hs_boa, np.arange(0, lim + 0.25, 0.25))
    g = P.groupby(cl, observed=True).Hm0
    med = g.median()[g.count() >= 3]
    xc = [iv.mid for iv in med.index]
    ax.plot(xc, med.values, "-o", color=INK, ms=4, lw=1.6, label="mediana per classe di Hs della boa")
    s = statistiche(P.Hm0, P.Hs_boa)
    ax.text(0.03, 0.97, f"n {s['n']}, r {s['r']:.2f}\ndifferenza media {s['bias']:+.2f} m\n"
            f"scarto quadratico {s['rmse']:.2f} m", transform=ax.transAxes, va="top", fontsize=9, color=INK2)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_aspect("equal")
    ax.set_xlabel("Hs boa (m)")
    ax.set_ylabel("Hm0 ADCP (m)")
    ax.legend(loc="lower right", frameon=False, fontsize=8.5)
    ax.set_title("Altezza", loc="left")
    # periodo medio
    ax = fig.add_subplot(gs[1, 1])
    mf = P.Hs_boa >= 0.5
    ax.scatter(P.Ta_boa[~mf], P.Tm01[~mf], s=8, color=MUTED, alpha=0.35, lw=0, label="Hs boa < 0.5 m")
    ax.scatter(P.Ta_boa[mf], P.Tm01[mf], s=10, color=C1, alpha=0.6, lw=0, label="Hs boa ≥ 0.5 m")
    ax.plot([2, 9], [2, 9], color=AXIS, lw=1)
    s = statistiche(P.Tm01[mf], P.Ta_boa[mf])
    ax.text(0.03, 0.97, f"Hs boa ≥ 0.5 m: n {s['n']}, r {s['r']:.2f}\ndifferenza media {s['bias']:+.2f} s",
            transform=ax.transAxes, va="top", fontsize=9, color=INK2)
    ax.set_xlim(2, 9)
    ax.set_ylim(2, 9)
    ax.set_aspect("equal")
    ax.set_xlabel("Ta boa (s)")
    ax.set_ylabel("Tm01 ADCP (s)")
    ax.legend(loc="lower right", frameon=False, fontsize=8.5)
    ax.set_title("Periodo medio", loc="left")
    # direzione
    ax = fig.add_subplot(gs[1, 2])
    m = (P.direction_qc == 1) & (P.Hm0 >= 0.5)
    ax.scatter(P.Dd_boa[m], P.Dp[m], s=12, color=C3, alpha=0.6, lw=0)
    ax.plot([140, 280], [140, 280], color=AXIS, lw=1)
    dd = ((P.Dp - P.Dd_boa + 180) % 360 - 180)[m]
    ax.text(0.03, 0.97, f"direzione ADCP affidabile, Hm0 ≥ 0.5 m\nn {m.sum()}, differenza media {dd.mean():+.1f}°\n"
            f"deviazione standard {dd.std():.0f}°", transform=ax.transAxes, va="top", fontsize=9, color=INK2)
    ax.set_xlim(140, 280)
    ax.set_ylim(140, 280)
    ax.set_aspect("equal")
    ax.set_xlabel("direzione boa, Dd (°, da cui arrivano)")
    ax.set_ylabel("direzione di picco ADCP, Dp (°)")
    ax.set_title("Direzione", loc="left")
    fig.suptitle("Onde: ADCP contro boa ondametrica davanti a Sturla", x=0.01, ha="left", fontweight="bold", fontsize=12)
    _save(fig, path)


def figura_orologio(C, marea, sposta, path):
    n = 2 if marea is not None else 1
    fig, axs = plt.subplots(1, n, figsize=(6.2 * n + 0.6, 4.6), layout="constrained", squeeze=False)
    ax = axs[0, 0]
    ax.plot(C.sposta_h, C.r, color=C1, lw=2)
    k = C.r.idxmax()
    ax.plot(C.sposta_h[k], C.r[k], "o", color=C2, ms=7)
    ax.axvline(0, color=AXIS, lw=0.8)
    ax.text(0.03, 0.06, f"massimo a {C.sposta_h[k]:+.2f} h: r {C.r[k]:.3f}\n"
            f"a 0 h: r {C.set_index('sposta_h').r[0.0]:.3f}", transform=ax.transAxes, fontsize=9, color=INK2)
    ax.set_xlabel("spostamento dei tempi dell'ADCP (ore)")
    ax.set_ylabel("correlazione Hm0 ADCP – Hs boa")
    ax.set_title("Onde: quando l'ADCP coincide con la boa", loc="left")
    if marea is not None:
        ax = axs[0, 1]
        y = np.arange(len(marea))[::-1]
        ax.errorbar(marea.sfasamento_h, y, xerr=marea.incertezza_h, fmt="o", color=C1, ecolor=C1, capsize=3, ms=6)
        ax.axvline(-sposta, color=C2, lw=1.2, ls="--")
        ax.axvline(0, color=AXIS, lw=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels(marea.index)
        ax.set_xlim(-0.5, 4)
        ax.set_ylim(-1.4, len(marea) - 0.5)
        ax.set_xlabel("ritardo della fase nell'ADCP rispetto a MedFS (ore)")
        ax.text(0.02, 0.02, "Se le medie orarie di MedFS sono etichettate all'inizio dell'ora,\n"
                "tutti i ritardi sono 0.5 ore più piccoli.", transform=ax.transAxes, fontsize=8.5, color=INK2)
        ax.set_title("Marea: fase nel livello dell'ADCP e in MedFS", loc="left")
    fig.suptitle("Orologio dell'ADCP: avanti di circa 2 ore rispetto a UTC", x=0.01, ha="left", fontweight="bold",
                 fontsize=12)
    _save(fig, path)


if __name__ == "__main__":
    main()
