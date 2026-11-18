#!/usr/bin/env python
"""Confronto fra le onde dell'ADCP, calcolate dai burst grezzi, e quelle di una boa ondametrica.

Mettere nella stessa cartella:
  - il file NetCDF dei burst d'onda del pre-processing (per esempio STURLA_..._L2_wave_bursts.nc
    oppure waveburst.nc): servono le variabili eta, n_samples e sample_time;
  - il CSV della boa, con le colonne UTC, Hs (m), Tp (sec), Ta (sec), Dd (deg) (e lat, lon);
  - facoltativo: il livello del mare di MedFS (un NetCDF con la variabile zos), per controllare
    l'orologio dell'ADCP anche con la marea;
  - facoltativo: il deployment.json della cartella preprocessing/ della stessa elaborazione (anche in una
    sottocartella), per segnare i movimenti del telaio al minuto; senza, si usano i burst segnati.

  python confronto_boa_burst.py [cartella] [--sposta-adcp-ore ORE]

Senza cartella usa quella in cui si trova lo script. I risultati vanno in <cartella>/confronto_boa_burst/.
Servono numpy, scipy, pandas, xarray (con netCDF4 o h5netcdf) e matplotlib; non serve il resto del
codice adcp_pd0. I passi sono numerati come nella spiegazione in fondo a analisi/confronto_boa/LEGGIMI.md.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy import signal

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ------------------------------------------------------------------ impostazioni
NPERSEG, FMIN, FMAX = 256, 0.04, 0.6            # spettro di Welch e bande, come nel post-processing
HM0_DIR_MIN, COH_MIN, QUOTA_DIREZIONE = 0.25, 0.3, 0.7
TOLLERANZA_MIN = 30                              # distanza massima fra burst e dato della boa
SOGLIA_MAREGGIATA = 1.5                          # Hs della boa che definisce una mareggiata (m)
CLASSI_HS = [(0, 0.3), (0.3, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 9.0)]
OMEGA = {"M2": 28.9841042, "S2": 30.0, "N2": 28.4397295, "K1": 15.0410686, "O1": 13.9430356}   # gradi all'ora
ORIGINE = pd.Timestamp("2026-01-01")             # origine dei tempi per le fasi di marea

C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID, AXIS, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "axes.edgecolor": AXIS,
    "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "legend.frameon": False})
MESI = ("gen", "feb", "mar", "apr", "mag", "giu", "lug", "ago", "set", "ott", "nov", "dic")
_trapz = getattr(np, "trapezoid", None) or np.trapz      # numpy >= 2 oppure versioni precedenti


# ------------------------------------------------------------------ 1. file della cartella
def trova_file(cartella: Path):
    """File dei burst, CSV della boa e (facoltativo) livello di MedFS nella cartella."""
    burst = boa = medfs = None
    nc = sorted(cartella.glob("*.nc"))
    preferiti = [p for p in nc if re.search(r"wave_?bursts?|waveburst", p.name, re.I)]
    for p in preferiti + [p for p in nc if p not in preferiti]:
        with xr.open_dataset(p) as d:
            if burst is None and "eta" in d and "n_samples" in d:
                burst = p
            elif medfs is None and "zos" in d:
                medfs = p
    for p in sorted(cartella.glob("*.csv")):
        testa = pd.read_csv(p, nrows=1, skipinitialspace=True).columns.str.strip()
        if {"UTC", "Hs (m)"} <= set(testa):
            boa = p
            break
    if burst is None:
        sys.exit(f"nessun file dei burst (NetCDF con eta e n_samples) in {cartella}")
    if boa is None:
        sys.exit(f"nessun CSV della boa (colonne UTC e Hs (m)) in {cartella}")
    return burst, boa, medfs


# ------------------------------------------------------------------ 2-4. parametri d'onda dai burst
def _moments(f, S, banda):
    """Hm0, Tp (picco con interpolazione parabolica), Tm01, Tm02 dello spettro S nella banda."""
    ff, SS = f[banda], S[banda]
    m0, m1, m2 = (_trapz(SS * ff ** n, ff) for n in (0, 1, 2))
    ip = int(np.argmax(SS))
    fp = ff[ip]
    if 0 < ip < len(SS) - 1:
        y0, y1, y2 = SS[ip - 1:ip + 2]
        den = y0 - 2 * y1 + y2
        if den != 0:
            fp = ff[ip] + 0.5 * (y0 - y2) / den * (ff[1] - ff[0])
    return dict(Hm0=4 * np.sqrt(m0), Tp=1 / fp if fp > 0 else np.nan,
                Tm01=m0 / m1 if m1 > 0 else np.nan, Tm02=np.sqrt(m0 / m2) if m2 > 0 else np.nan)


def _riempi_buchi(x, max_frac):
    """Interpolazione lineare dei valori mancanti, se sono al piu' max_frac della serie."""
    bad = ~np.isfinite(x)
    if bad.mean() > max_frac:
        return None
    if bad.any():
        i = np.arange(len(x))
        x = x.copy()
        x[bad] = np.interp(i[bad], i[~bad], x[~bad])
    return x


def parametri_onde(b: xr.Dataset) -> pd.DataFrame:
    """Per ogni burst: spettro dell'elevazione, Hm0, periodi, direzione di picco, flag."""
    st = b["sample_time"].values
    dt = float(st[1] - st[0]) if len(st) > 1 else float(b.attrs.get("dt", 0.5))
    fs = 1.0 / dt
    nb = b.sizes["time"]
    n_s = b["n_samples"].values.astype(int)
    usabile = np.isin(b["burst_qc"].values, (1, 3)) if "burst_qc" in b else np.ones(nb, bool)
    n_ok = n_s[usabile & (n_s > 0)]
    nseg = int(min(NPERSEG, 2 ** np.floor(np.log2(max(n_ok.min() if n_ok.size else NPERSEG, 16)))))
    freq = np.fft.rfftfreq(nseg, dt)
    kw = dict(fs=fs, nperseg=nseg, noverlap=nseg // 2, window="hann")
    banda_m = (freq >= FMIN) & (freq <= min(2 * FMAX, fs / 2))
    banda_d = (freq >= FMIN) & (freq <= FMAX)
    nlev = b.sizes.get("level", 0) if "u" in b else 0
    lev = 0
    if nlev:
        d_ref = float(np.nanmedian(b["depth_sensor"].values[usabile]))
        lev = int(np.argmin(np.abs(b["level_range"].values / d_ref - QUOTA_DIREZIONE)))
    E = b["eta"].values
    U, V = (b["u"].values, b["v"].values) if nlev else (None, None)
    prof = b["water_depth"].values if "water_depth" in b else np.full(nb, 99.0)
    righe = []
    for k in range(nb):
        r = dict(Hm0=np.nan, Tp=np.nan, Tm01=np.nan, Tm02=np.nan, Dp=np.nan, Dm=np.nan, coerenza=np.nan)
        n = int(n_s[k])
        e = E[k, :n].astype(float)
        if usabile[k] and n >= nseg and np.isfinite(prof[k]) and prof[k] > 0.5 and np.isfinite(e).all():
            ed = signal.detrend(e)
            _, See = signal.welch(ed, **kw)
            r.update(_moments(freq, See, banda_m))
            if nlev:
                uu = _riempi_buchi(U[k, :n, lev].astype(float), 0.3)
                vv = _riempi_buchi(V[k, :n, lev].astype(float), 0.3)
                if uu is not None and vv is not None:
                    ud, vd = signal.detrend(uu), signal.detrend(vv)
                    _, Peu = signal.csd(ed, ud, **kw)
                    _, Pev = signal.csd(ed, vd, **kw)
                    ip = int(np.argmax(np.where(banda_d, See, -np.inf)))
                    r["Dp"] = (270.0 - np.degrees(np.arctan2(Pev.real[ip], Peu.real[ip]))) % 360.0
                    r["Dm"] = (270.0 - np.degrees(np.arctan2(Pev.real[banda_d].sum(), Peu.real[banda_d].sum()))) % 360.0
                    _, Puu = signal.welch(ud, **kw)
                    _, Pvv = signal.welch(vd, **kw)
                    with np.errstate(all="ignore"):
                        coh = (np.abs(Peu) ** 2 + np.abs(Pev) ** 2) / (See * (Puu + Pvv))
                    pk = banda_d & (np.abs(freq - freq[ip]) <= 0.025)
                    if pk.any() and See[pk].sum() > 0:
                        r["coerenza"] = float(np.sum(coh[pk] * See[pk]) / np.sum(See[pk]))
        righe.append(r)
    w = pd.DataFrame(righe, index=pd.DatetimeIndex(b["time"].values, name="inizio_burst"))
    w["n_samples"] = n_s
    w["centro"] = w.index + pd.to_timedelta(n_s * dt / 2, unit="s")
    w["livello_m"] = b["depth_sensor"].values if "depth_sensor" in b else np.nan
    inpos = b["in_position"].values.astype(bool) if "in_position" in b else np.ones(nb, bool)
    mot = b["platform_motion"].values.astype(bool) if "platform_motion" in b else np.zeros(nb, bool)
    sf = b["surface_found_frac"].values if "surface_found_frac" in b else np.ones(nb)
    ok = np.isfinite(w.Hm0.values)
    w["wave_qc"] = np.select([~inpos, ~ok, mot | (np.nan_to_num(sf, nan=1.0) < 0.9)], [4, 9, 3], 1)
    dok = np.isfinite(w.Dp.values)
    sospetta = mot | (np.nan_to_num(w.Hm0.values) < HM0_DIR_MIN) | (np.nan_to_num(w.coerenza.values, nan=1.0) < COH_MIN)
    w["direction_qc"] = np.select([~inpos, ~dok, sospetta], [4, 9, 3], 1)
    w["platform_motion"] = mot.astype(int)
    w["in_position"] = inpos.astype(int)
    w["heading"] = b["heading"].values if "heading" in b else np.nan
    w.attrs.update(nperseg=nseg, fs=fs, livello_direzione=lev,
                   quota_direzione=float(b["level_range"].values[lev]) if nlev else np.nan)
    return w


# ------------------------------------------------------------------ 5. orologio dell'ADCP
def orologio_dal_file(attrs) -> tuple[float | None, str]:
    """Ore da sommare ai tempi del file per averli in UTC, se il file lo dice."""
    tz = str(attrs.get("time_zone", "")).strip()
    if "time_offset_applied_h" in attrs:
        return 0.0, f"il pre-processing ha gia' portato i tempi in UTC (orologio {attrs.get('instrument_clock', '?')})"
    if tz.upper() == "UTC":
        return 0.0, "il file dichiara tempi in UTC"
    return None, f"il file non dice che i tempi sono in UTC (time_zone: '{tz or 'assente'}')"


# ------------------------------------------------------------------ 6. boa
def leggi_boa(p) -> pd.DataFrame:
    b = pd.read_csv(p, skipinitialspace=True)
    b.columns = [c.strip() for c in b.columns]
    b = b.rename(columns={"Hs (m)": "Hs", "Tp (sec)": "Tp", "Ta (sec)": "Ta", "Dd (deg)": "Dd"})
    b["t"] = pd.to_datetime(b["UTC"], utc=True).dt.tz_convert(None)      # orari senza fuso = UTC
    for c in ("Hs", "Tp", "Ta", "Dd"):
        b[c] = pd.to_numeric(b.get(c), errors="coerce")
    return b.sort_values("t").drop_duplicates("t").set_index("t")


# ------------------------------------------------------------------ 7. correlazione con lo sfasamento
def correlazione_sfasamenti(w, b, passi=np.arange(-6, 6.01, 0.25)):
    """Correlazione fra Hm0 dell'ADCP e Hs della boa spostando i tempi dell'ADCP; la boa e' interpolata
    solo fra dati distanti al piu' 1.5 ore."""
    t0 = b.index[0]
    tb = ((b.index - t0) / pd.Timedelta("1h")).to_numpy()
    out = []
    for L in passi:
        tq = (((w.centro + pd.Timedelta(hours=float(L))) - t0) / pd.Timedelta("1h")).to_numpy()
        i = np.searchsorted(tb, tq)
        ok = (i > 0) & (i < len(tb))
        ii = np.clip(i, 1, len(tb) - 1)
        ok &= (tb[ii] - tb[ii - 1]) <= 1.5
        v = np.interp(tq[ok], tb, b.Hs.values)
        r = np.corrcoef(w.Hm0.values[ok], v)[0, 1] if ok.sum() > 10 else np.nan
        out.append((float(L), float(r), int(ok.sum())))
    return pd.DataFrame(out, columns=["sposta_h", "r", "n"])


# ------------------------------------------------------------------ 8. coppie burst-boa
def accoppia(w, b, sposta_h) -> pd.DataFrame:
    bt = b.index.values
    tq = (w.centro + pd.Timedelta(hours=sposta_h)).values
    i = np.searchsorted(bt, tq)
    i0, i1 = np.clip(i - 1, 0, len(bt) - 1), np.clip(i, 0, len(bt) - 1)
    d0 = np.abs((tq - bt[i0]) / np.timedelta64(1, "m"))
    d1 = np.abs((bt[i1] - tq) / np.timedelta64(1, "m"))
    j = np.where(d0 <= d1, i0, i1)
    d = np.minimum(d0, d1)
    ok = d <= TOLLERANZA_MIN
    P = w[ok].copy()
    P["tempo_utc"] = tq[ok]
    for c in ("Hs", "Tp", "Ta", "Dd"):
        P[f"{c}_boa"] = b[c].values[j[ok]]
    P["scarto_tempo_min"] = d[ok]
    return P.set_index("tempo_utc")


# ------------------------------------------------------------------ 9. statistiche
def statistiche(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 3:
        return dict(n=len(x))
    e = x - y
    rapp = x[y > 0] / y[y > 0]
    return dict(n=len(x), bias=e.mean(), rmse=np.sqrt((e ** 2).mean()), indice_dispersione=np.sqrt((e ** 2).mean()) / y.mean(),
                r=np.corrcoef(x, y)[0, 1], rapporto_mediano=np.median(rapp) if len(rapp) else np.nan)


def tabella_statistiche(P) -> pd.DataFrame:
    righe = [dict(grandezza="Hm0 contro Hs", classe="tutte", **statistiche(P.Hm0, P.Hs_boa))]
    for lo, hi in CLASSI_HS:
        q = P[(P.Hs_boa >= lo) & (P.Hs_boa < hi)]
        righe.append(dict(grandezza="Hm0 contro Hs", classe=f"Hs boa {lo}-{hi} m", **statistiche(q.Hm0, q.Hs_boa)))
    mf = P[P.Hs_boa >= 0.5]
    for k in ("Tm01", "Tm02"):
        righe.append(dict(grandezza=f"{k} contro Ta", classe="Hs boa >= 0.5 m", **statistiche(mf[k], mf.Ta_boa)))
    righe.append(dict(grandezza="Tp contro Tp", classe="Hs boa >= 0.5 m", **statistiche(mf.Tp, mf.Tp_boa),
                      entro_1s_pct=100 * (np.abs(mf.Tp - mf.Tp_boa) <= 1).mean() if len(mf) else np.nan))
    dd = (P.Dp - P.Dd_boa + 180) % 360 - 180
    for nome, m in (("direzione affidabile, Hm0 >= 0.5 m", (P.direction_qc == 1) & (P.Hm0 >= 0.5)),
                    ("Hm0 >= 1 m", (P.direction_qc <= 3) & (P.Hm0 >= 1.0))):
        x = dd[m].dropna()
        righe.append(dict(grandezza="Dp meno Dd (gradi)", classe=nome, n=len(x), bias=x.mean(),
                          rmse=np.sqrt((x ** 2).mean()), dev_std=x.std(), entro_20_pct=100 * (x.abs() < 20).mean()))
    return pd.DataFrame(righe)


# ------------------------------------------------------------------ 10. mareggiate
def mareggiate(w, b, sposta_h) -> pd.DataFrame:
    """Episodi con Hs della boa >= SOGLIA_MAREGGIATA (uniti se a meno di 12 ore): massimi a confronto."""
    t_adcp = w.centro + pd.Timedelta(hours=sposta_h)
    hs = b.Hs[t_adcp.min() - pd.Timedelta("6h"): t_adcp.max() + pd.Timedelta("6h")]      # solo mentre misura l'ADCP
    sopra = hs[hs >= SOGLIA_MAREGGIATA]
    if sopra.empty:
        return pd.DataFrame()
    gruppi = (sopra.index.to_series().diff() > pd.Timedelta("12h")).cumsum()
    Wu = w.set_index(w.centro + pd.Timedelta(hours=sposta_h)).Hm0.dropna()
    out = []
    for _, g in sopra.groupby(gruppi.values):
        t0, t1 = g.index[0] - pd.Timedelta("6h"), g.index[-1] + pd.Timedelta("6h")
        tb = b.Hs[t0:t1].idxmax()
        x = Wu[t0:t1]
        vic = Wu[tb - pd.Timedelta("90min"): tb + pd.Timedelta("90min")]
        out.append(dict(inizio=g.index[0], fine=g.index[-1], boa_max=b.Hs[tb], boa_ora=tb,
                        adcp_max=x.max() if len(x) else np.nan, adcp_ora=x.idxmax() if len(x) else pd.NaT,
                        adcp_all_ora_della_boa=vic.mean() if len(vic) else np.nan))
    return pd.DataFrame(out)


# ------------------------------------------------------------------ 10. (segue) movimenti del telaio
def movimenti_telaio(cartella: Path, w_all: pd.DataFrame, attrs: dict, sposta_h: float) -> pd.DataFrame:
    """Episodi di movimento del telaio, in UTC. Se nella cartella c'e' il deployment.json della stessa elaborazione
    dei burst: episodi al minuto e rotazione dell'heading. Altrimenti: burst segnati con platform_motion, uniti se a
    meno di 6 ore, e rotazione dall'heading medio dei burst fermi subito prima e subito dopo."""
    for p in sorted(cartella.rglob("deployment.json")):
        try:
            dep = json.loads(p.read_text(encoding="utf-8"))
            ev = dep.get("motion_events") or []
            ore = 0.0
            if attrs.get("deployment_start"):    # stesso deployment, al piu' con l'orologio diverso di ore intere
                ore = (pd.Timestamp(attrs["deployment_start"]) - pd.Timestamp(dep["start"])) / pd.Timedelta("1h")
            dsh = pd.Timedelta(hours=round(ore) + sposta_h)
            mov = pd.DataFrame([dict(inizio=pd.Timestamp(e["start"]) + dsh, fine=pd.Timestamp(e["end"]) + dsh,
                                     rotazione=(e["heading_after"] - e["heading_before"] + 180) % 360 - 180,
                                     fonte=f"{p.relative_to(cartella)}, al minuto") for e in ev])
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            continue
        dentro = w_all.index.min() + pd.Timedelta(hours=sposta_h) - pd.Timedelta("1D"), \
            w_all.index.max() + pd.Timedelta(hours=sposta_h) + pd.Timedelta("1D")
        if (ev and abs(ore - round(ore)) < 0.02 and abs(ore) <= 14
                and mov.inizio.between(*dentro).all() and mov.fine.between(*dentro).all()):
            return mov
    if "platform_motion" not in w_all or not w_all.platform_motion.astype(bool).any():
        return pd.DataFrame()
    mot = w_all.platform_motion.to_numpy().astype(bool)
    hd = w_all.heading.to_numpy(float)
    fermo = w_all.in_position.to_numpy().astype(bool) & ~mot & np.isfinite(hd)
    t0 = (w_all.index + pd.Timedelta(hours=sposta_h)).to_numpy()
    t1 = t0 + 2 * (w_all.centro.to_numpy() - w_all.index.to_numpy())
    idx = np.flatnonzero(mot)
    passo = np.median(np.diff(w_all.index.values)) / np.timedelta64(1, "h") if len(w_all) > 1 else np.nan
    righe = []
    for g in np.split(idx, np.flatnonzero(np.diff(t0[idx]) > np.timedelta64(6, "h")) + 1):
        prima, dopo = np.flatnonzero(fermo[:g[0]]), np.flatnonzero(fermo[g[-1] + 1:])
        rot = (hd[g[-1] + 1 + dopo[0]] - hd[prima[-1]] + 180) % 360 - 180 if len(prima) and len(dopo) else np.nan
        righe.append(dict(inizio=pd.Timestamp(t0[g[0]]), fine=pd.Timestamp(t1[g[-1]]), rotazione=rot,
                          fonte=f"burst segnati con platform_motion, uno ogni {passo:.3g} ore"))
    return pd.DataFrame(righe)


# ------------------------------------------------------------------ 11. marea (facoltativo)
def fasi_marea(w, sposta_h, medfs_path, lat, lon) -> pd.DataFrame | None:
    """Fasi di M2, S2, N2, K1, O1 nel livello dell'ADCP (pressione media dei burst) e in MedFS, con lo
    stesso fit ai minimi quadrati; la differenza di fase divisa per la velocita' angolare e' il ritardo."""
    lv = pd.Series(w.livello_m.values, index=w.centro + pd.Timedelta(hours=sposta_h))
    lv = lv[(w.wave_qc != 4).values].dropna()
    if len(lv) < 200:
        return None
    Z = xr.open_dataset(medfs_path).zos
    la, lo = Z.latitude.values, Z.longitude.values
    valido = np.isfinite(Z.isel(time=0).values)
    d2 = (la[:, None] - lat) ** 2 + ((lo[None, :] - lon) * np.cos(np.deg2rad(lat))) ** 2
    d2[~valido] = np.inf
    i, j = np.unravel_index(np.argmin(d2), d2.shape)
    m = Z.isel(latitude=i, longitude=j).to_series()
    m.index = pd.DatetimeIndex(m.index)
    t0, t1 = max(lv.index.min(), m.index.min()), min(lv.index.max(), m.index.max())
    lv, m = lv[t0:t1], m[t0:t1].dropna()

    def fit(s):
        th = ((s.index - ORIGINE) / pd.Timedelta("1h")).to_numpy()
        X = [np.ones_like(th), th]
        for om in OMEGA.values():
            X += [np.cos(np.deg2rad(om * th)), np.sin(np.deg2rad(om * th))]
        X = np.array(X).T
        c, *_ = np.linalg.lstsq(X, s.values, rcond=None)
        res = s.values - X @ c
        cov = np.linalg.inv(X.T @ X) * res.var(ddof=X.shape[1])
        out = {}
        for n, k in enumerate(OMEGA):
            a, b_ = c[2 + 2 * n], c[3 + 2 * n]
            A = np.hypot(a, b_)
            sa, sb = np.sqrt(cov[2 + 2 * n, 2 + 2 * n]), np.sqrt(cov[3 + 2 * n, 3 + 2 * n])
            out[k] = (A, np.degrees(np.arctan2(b_, a)) % 360, np.degrees(np.hypot(b_ * sa, a * sb) / A ** 2))
        return out

    fa, fm = fit(lv), fit(m)
    T = pd.DataFrame({k: dict(A_cm_adcp=100 * fa[k][0], fase_adcp=fa[k][1], A_cm_medfs=100 * fm[k][0], fase_medfs=fm[k][1],
                              dfase=(fa[k][1] - fm[k][1] + 180) % 360 - 180,
                              incertezza_fase=np.hypot(fa[k][2], fm[k][2])) for k in OMEGA}).T
    T["ritardo_h"] = [T.dfase[k] / OMEGA[k] for k in T.index]
    T["incertezza_h"] = [T.incertezza_fase[k] / OMEGA[k] for k in T.index]
    T.attrs["punto"] = (float(la[i]), float(lo[j]))
    return T


# ------------------------------------------------------------------ 12. figure
def _asse_tempo(ax):
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=10))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(
        lambda x, _: f"{mdates.num2date(x).day} {MESI[mdates.num2date(x).month - 1]}"))


def _con_buchi(s, ore=6):
    s = s.dropna()
    salto = s.index.to_series().diff() > pd.Timedelta(hours=ore)
    buchi = pd.Series(np.nan, index=s.index[salto.values] - pd.Timedelta(minutes=1))
    return pd.concat([s, buchi]).sort_index()


def figura_altezza(P, b, w, sposta, M, mov, path):
    """Serie temporale: Hs della boa e Hm0 dell'ADCP, mareggiate, movimenti del telaio; sotto, il rapporto."""
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(15, 7.4), sharex=True, layout="constrained",
                                  gridspec_kw=dict(height_ratios=[3, 1.3]))
    lo, hi = P.index.min() - pd.Timedelta("12h"), P.index.max() + pd.Timedelta("12h")
    for _, r in M.iterrows():
        for a_ in (ax, ax2):
            a_.axvspan(r.inizio, r.fine, color=GRID, alpha=0.6, lw=0, zorder=0)
    y = _con_buchi(b.Hs[lo:hi])
    Wu = w.set_index(w.centro + pd.Timedelta(hours=sposta)).Hm0.sort_index()
    x = _con_buchi(Wu[lo:hi])
    l1, = ax.plot(y.index, y.values, color=C2, lw=1.1, label="boa, Hs")
    l2, = ax.plot(x.index, x.values, color=C1, lw=1.1,
                  label="ADCP, Hm0 dai burst" + (f" (tempi del file spostati di {sposta:+g} h)" if sposta else ""))
    ymax = float(max(np.nanmax(y.values), np.nanmax(x.values))) * 1.22
    maniglie = [l1, l2]
    if mov is not None and not mov.empty:
        mv = mov.assign(tm=mov.inizio + (mov.fine - mov.inizio) / 2).sort_values("tm")
        mv = mv[(mv.tm >= lo) & (mv.tm <= hi)].reset_index(drop=True)
        vicino = pd.Timedelta("4D")                 # etichette di episodi vicini: la prima a sinistra della linea
        for i, r in mv.iterrows():
            for a_ in (ax, ax2):
                a_.axvline(r.tm, color=C3, lw=1.4, ls="--", zorder=1)
            col_prec = i > 0 and r.tm - mv.tm[i - 1] < vicino
            col_succ = i + 1 < len(mv) and mv.tm[i + 1] - r.tm < vicino
            testo = f"{r.rotazione:+.1f}°" if np.isfinite(r.rotazione) else "telaio"
            a_sinistra = col_succ and not col_prec
            ax.text(r.tm, ymax * (0.985 - 0.085 * col_prec), testo + " " if a_sinistra else " " + testo, color=C3,
                    fontsize=9, va="top", ha="right" if a_sinistra else "left", fontweight="bold")
        maniglie.append(plt.Line2D([], [], color=C3, lw=1.4, ls="--", label="movimento del telaio, con la rotazione dell'heading"))
    if not M.empty:
        maniglie.append(plt.Rectangle((0, 0), 1, 1, color=GRID, alpha=0.6, label=f"Hs della boa sopra {SOGLIA_MAREGGIATA:g} m"))
    ax.set(xlim=(lo, hi), ylim=(0, ymax), ylabel="altezza d'onda significativa (m)")
    ax.legend(handles=maniglie, loc="lower left", fontsize=9, ncols=len(maniglie), bbox_to_anchor=(0, 1.0),
              borderaxespad=0.2, handlelength=2.2)
    fig.suptitle("Altezza d'onda significativa: boa e ADCP (tempi in UTC)", x=0.01, ha="left",
                 fontweight="bold", fontsize=12)
    rap = (P.Hm0 / P.Hs_boa).sort_index()
    ax2.scatter(rap.index, rap.values, s=7, color=C1, alpha=0.45, lw=0)
    med = _con_buchi(rap.rolling("3D", center=True, min_periods=5).median(), ore=12)
    ax2.plot(med.index, med.values, color=INK, lw=1.5, label="mediana mobile su 3 giorni")
    ax2.axhline(1, color=AXIS, lw=1)
    ax2.set(ylim=(0, 1.6), ylabel="Hm0 ADCP / Hs boa")
    ax2.legend(loc="upper left", fontsize=8.5)
    ax2.set_title("Rapporto fra ADCP e boa in ogni coppia (1 = stessa altezza)", loc="left")
    _asse_tempo(ax2)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def figura_onde(P, path):
    fig, axs = plt.subplots(1, 3, figsize=(13.5, 5.0), layout="constrained")
    ax = axs[0]
    ax.scatter(P.Hs_boa, P.Hm0, s=10, color=C1, alpha=0.45, lw=0)
    lim = max(P.Hs_boa.max(), P.Hm0.max()) * 1.05
    ax.plot([0, lim], [0, lim], color=AXIS, lw=1)
    g = P.groupby(pd.cut(P.Hs_boa, np.arange(0, lim + 0.25, 0.25)), observed=True).Hm0
    med = g.median()[g.count() >= 3]
    ax.plot([iv.mid for iv in med.index], med.values, "-o", color=INK, ms=4, lw=1.6,
            label="mediana per classe di Hs della boa")
    s = statistiche(P.Hm0, P.Hs_boa)
    ax.text(0.03, 0.97, f"n {s['n']}, r {s['r']:.2f}\ndifferenza media {s['bias']:+.2f} m\n"
            f"scarto quadratico {s['rmse']:.2f} m", transform=ax.transAxes, va="top", color=INK2)
    ax.set(xlim=(0, lim), ylim=(0, lim), xlabel="Hs boa (m)", ylabel="Hm0 ADCP (m)", aspect="equal")
    ax.legend(loc="lower right", fontsize=8.5)
    ax.set_title("Altezza", loc="left")
    ax = axs[1]
    mf = P.Hs_boa >= 0.5
    ax.scatter(P.Ta_boa[~mf], P.Tm01[~mf], s=8, color=MUTED, alpha=0.35, lw=0, label="Hs boa < 0.5 m")
    ax.scatter(P.Ta_boa[mf], P.Tm01[mf], s=10, color=C1, alpha=0.6, lw=0, label="Hs boa ≥ 0.5 m")
    ax.plot([2, 9], [2, 9], color=AXIS, lw=1)
    s = statistiche(P.Tm01[mf], P.Ta_boa[mf])
    if s["n"] >= 3:
        ax.text(0.03, 0.97, f"Hs boa ≥ 0.5 m: n {s['n']}, r {s['r']:.2f}\ndifferenza media {s['bias']:+.2f} s",
                transform=ax.transAxes, va="top", color=INK2)
    ax.set(xlim=(2, 9), ylim=(2, 9), xlabel="Ta boa (s)", ylabel="Tm01 ADCP (s)", aspect="equal")
    ax.legend(loc="lower right", fontsize=8.5)
    ax.set_title("Periodo medio", loc="left")
    ax = axs[2]
    m = (P.direction_qc == 1) & (P.Hm0 >= 0.5)
    ax.scatter(P.Dd_boa[m], P.Dp[m], s=12, color=C3, alpha=0.6, lw=0)
    ax.plot([140, 280], [140, 280], color=AXIS, lw=1)
    dd = ((P.Dp - P.Dd_boa + 180) % 360 - 180)[m]
    ax.text(0.03, 0.97, f"direzione ADCP affidabile, Hm0 ≥ 0.5 m\nn {int(m.sum())}, differenza media {dd.mean():+.1f}°\n"
            f"deviazione standard {dd.std():.0f}°", transform=ax.transAxes, va="top", color=INK2)
    ax.set(xlim=(140, 280), ylim=(140, 280), xlabel="direzione boa, Dd (°, da cui arrivano)",
           ylabel="direzione di picco ADCP, Dp (°)", aspect="equal")
    ax.set_title("Direzione", loc="left")
    fig.suptitle("Onde: ADCP (dai burst) contro boa ondametrica", x=0.01, ha="left", fontweight="bold", fontsize=12)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def figura_orologio(C, T, sposta, path):
    n = 2 if T is not None else 1
    fig, axs = plt.subplots(1, n, figsize=(6.2 * n + 0.6, 4.6), layout="constrained", squeeze=False)
    ax = axs[0, 0]
    ax.plot(C.sposta_h, C.r, color=C1, lw=2)
    k = C.r.idxmax()
    ax.plot(C.sposta_h[k], C.r[k], "o", color=C2, ms=7)
    ax.axvline(sposta, color=C2, lw=1.2, ls="--")
    ax.text(0.03, 0.06, f"massimo a {C.sposta_h[k]:+.2f} h: r {C.r[k]:.3f}\nspostamento usato {sposta:+g} h: "
            f"r {np.interp(sposta, C.sposta_h, C.r):.3f}", transform=ax.transAxes, fontsize=9, color=INK2)
    ax.set_xlabel("ore sommate ai tempi del file dei burst")
    ax.set_ylabel("correlazione Hm0 ADCP – Hs boa")
    ax.set_title("Onde: quando l'ADCP coincide con la boa", loc="left")
    titolo = f"Orologio dell'ADCP: le onde coincidono meglio spostando i tempi di {C.sposta_h[k]:+.2f} h"
    if T is not None:
        ax = axs[0, 1]
        y = np.arange(len(T))[::-1]
        ax.errorbar(T.ritardo_h, y, xerr=T.incertezza_h, fmt="o", color=C1, ecolor=C1, capsize=3, ms=6)
        ax.axvline(0, color=AXIS, lw=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels(T.index)
        lo_, hi_ = (T.ritardo_h - T.incertezza_h).min(), (T.ritardo_h + T.incertezza_h).max()
        ax.set_xlim(min(-1.0, lo_ - 0.5), max(3.5, hi_ + 0.5))
        ax.set_ylim(-1.4, len(T) - 0.5)
        ax.set_xlabel("ritardo della marea nell'ADCP rispetto a MedFS (ore), tempi corretti")
        ax.text(0.02, 0.02, "Se le medie orarie di MedFS sono etichettate all'inizio dell'ora,\n"
                "tutti i ritardi sono 0.5 ore più piccoli.", transform=ax.transAxes, fontsize=8.5, color=INK2)
        ax.set_title("Marea: fase nel livello dell'ADCP e in MedFS", loc="left")
        wgt = 1 / T.incertezza_h.astype(float) ** 2
        titolo += f"; ritardo residuo della marea {(T.ritardo_h * wgt).sum() / wgt.sum():+.1f} h"
    fig.suptitle(titolo, x=0.01, ha="left", fontweight="bold", fontsize=11)
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------ programma
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cartella", nargs="?", default=str(Path(__file__).resolve().parent),
                    help="cartella con i burst e il CSV della boa (default: quella dello script)")
    ap.add_argument("--sposta-adcp-ore", type=float, default=None,
                    help="ore da sommare ai tempi dei burst per averli in UTC (default: dal file, altrimenti stimate)")
    a = ap.parse_args(argv)
    cartella = Path(a.cartella).resolve()
    f_burst, f_boa, f_medfs = trova_file(cartella)
    b_nc = xr.load_dataset(f_burst)
    confronta(cartella, f_burst.name, parametri_onde(b_nc), dict(b_nc.attrs), f_boa, f_medfs, a.sposta_adcp_ore)


def confronta(cartella: Path, nome_burst, w_all, attrs, f_boa, f_medfs, sposta_scelto=None):
    """Passi 5-12 a partire dai parametri d'onda di tutti i burst."""
    uscita = cartella / "confronto_boa_burst"
    (uscita / "figure").mkdir(parents=True, exist_ok=True)
    righe = []

    def dire(s=""):
        print(s)
        righe.append(s)

    dire(f"burst: {nome_burst}\nboa: {f_boa.name}\nlivello MedFS: {f_medfs.name if f_medfs else 'assente (niente controllo con la marea)'}")
    w_all.to_csv(uscita / "parametri_onde_adcp.csv")
    ok = w_all.wave_qc.isin([1, 3])
    dire(f"{len(w_all)} burst, {int(ok.sum())} usabili (flag 1 o 3); spettri di {w_all.attrs.get('nperseg')} campioni a "
         f"{w_all.attrs.get('fs', np.nan):g} Hz; direzione alla cella a {w_all.attrs.get('quota_direzione', np.nan):.1f} m dal trasduttore")
    w = w_all[ok & np.isfinite(w_all.Hm0)].copy()
    b = leggi_boa(f_boa)
    pos = f", posizione mediana {b.lat.median():.5f} N {b.lon.median():.5f} E" if "lat" in b else ""
    dire(f"boa: {len(b)} dati dal {b.index[0]} al {b.index[-1]}{pos}")

    C = correlazione_sfasamenti(w, b)
    C.to_csv(uscita / "correlazione_sfasamenti.csv", index=False)
    best = C.loc[C.r.idxmax()]
    sposta, motivo = (sposta_scelto, "scelto con --sposta-adcp-ore") if sposta_scelto is not None \
        else orologio_dal_file(attrs)
    if sposta is None:
        sposta = float(np.round(best.sposta_h))
        motivo += f"; uso lo spostamento intero che rende massima la correlazione, {sposta:+g} h"
    dire(f"orologio: {motivo}")
    dire(f"correlazione Hm0-Hs massima ({best.r:.3f}) spostando i tempi del file di {best.sposta_h:+.2f} h; "
         f"con lo spostamento usato ({sposta:+g} h) {np.interp(sposta, C.sposta_h, C.r):.3f}")
    if abs(best.sposta_h - sposta) >= 1:
        dire("ATTENZIONE: le onde combaciano meglio con uno spostamento diverso di un'ora o piu': controllare l'orologio")

    T = None
    if f_medfs is not None and "lat" in b:
        T = fasi_marea(w, sposta, f_medfs, float(b.lat.median()), float(b.lon.median()))
        if T is not None:
            T.to_csv(uscita / "fasi_marea.csv")
            dire(f"marea, ADCP meno MedFS (punto {T.attrs['punto'][0]:.4f} N {T.attrs['punto'][1]:.4f} E), "
                 "ritardo in ore con i tempi gia' spostati:")
            dire((T[["A_cm_adcp", "A_cm_medfs", "dfase", "ritardo_h", "incertezza_h"]].astype(float).round(2)).to_string())

    P = accoppia(w, b, sposta)
    if P.empty:
        raise SystemExit(f"nessun burst ha un dato della boa entro {TOLLERANZA_MIN} minuti: controllare che i due "
                         "file coprano lo stesso periodo e lo spostamento dell'orologio")
    P.to_csv(uscita / "coppie.csv")
    S = tabella_statistiche(P)
    S.to_csv(uscita / "statistiche.csv", index=False)
    dire(f"\n{len(P)} coppie burst-boa entro {TOLLERANZA_MIN} minuti")
    dire(S.to_string(index=False, float_format=lambda x: f"{x:.3f}", na_rep=""))
    M = mareggiate(w, b, sposta)
    if not M.empty:
        M.to_csv(uscita / "mareggiate.csv", index=False)
        dire("\nmareggiate (Hs della boa oltre la soglia):")
        dire(M.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    mov = movimenti_telaio(cartella, w_all, attrs, sposta)
    if not mov.empty:
        mov.to_csv(uscita / "movimenti_telaio.csv", index=False)
        dire(f"\nmovimenti del telaio ({mov.fonte.iloc[0]}):")
        dire(mov.drop(columns="fonte").to_string(index=False, float_format=lambda x: f"{x:+.1f}"))

    figura_altezza(P, b, w, sposta, M, mov, uscita / "figure" / "altezza_adcp_boa.png")
    figura_onde(P, uscita / "figure" / "onde_adcp_boa.png")
    figura_orologio(C, T, sposta, uscita / "figure" / "orologio_adcp.png")
    (uscita / "riepilogo.txt").write_text("\n".join(righe) + "\n", encoding="utf-8")
    print(f"\nrisultati in {uscita}")


if __name__ == "__main__":
    main()
