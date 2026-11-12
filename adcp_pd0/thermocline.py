"""Stima della quota del termoclino dai profili di corrente e dall'eco (post-processing).

L'ADCP non misura la temperatura lungo la colonna: il termoclino è stimato in modo indiretto.

Taglio della corrente. In una colonna stratificata lo strato superficiale e quello profondo
scorrono in modo diverso e la corrente cambia di più al confine fra i due. Per ogni giorno:
profilo medio di u e v, lisciato sulla verticale; taglio |dU/dz|; quota del massimo, escluse
le celle vicine al fondo (attrito) e alla superficie (vento e onde). La stima è "affidabile"
se valgono tre condizioni: il massimo è almeno ``min_prominence`` volte la mediana del
profilo; non cade al bordo della fascia considerata; si ritrova alla stessa quota (entro
``agree_m``) ripetendo la stima su due metà indipendenti dei profili della finestra (coppie
di profili alterne). L'ultima condizione scarta i massimi dovuti al solo rumore, che in un
profilo senza struttura superano comunque la mediana di 2-3 volte. Una banda di sensibilità
si ottiene ripetendo la stima con lisciature, esclusioni e finestre di media diverse.

Eco. L'intensità dell'eco, corretta per lo spargimento del fascio e l'assorbimento, ha spesso
un massimo al termoclino (Brenner et al., 2023, J. Atmos. Oceanic Technol.): si toglie
l'andamento lineare con la quota e si prende il picco nella stessa fascia.

È un prodotto derivato, non validato con profili di temperatura.
"""
from __future__ import annotations

import itertools
import warnings

import numpy as np
import pandas as pd
import xarray as xr

from .stats import good_vel


def window_profiles(c: xr.Dataset, window: str = "1D", sel=None):
    """Profili medi di u, v ed eco per finestra temporale (solo celle con flag 1 o 3).
    ``sel``: maschera dei profili da usare (per le stime su metà dei dati)."""
    ok = good_vel(c)
    if sel is not None:
        ok = ok & np.asarray(sel, bool)[:, None]
    t = pd.DatetimeIndex(c["time"].values)
    mk = lambda a: pd.DataFrame(np.where(ok, a, np.nan), index=t).resample(window).mean()
    U, V = mk(c["u"].values), mk(c["v"].values)
    E = mk(c["echo"].values) if "echo" in c else None
    return U, V, E


def shear_estimate(u, v, z, smooth=5, nb=3, nt=4):
    """Quota del massimo taglio per un profilo. Restituisce None se le celle valide sono poche."""
    idx = np.flatnonzero(np.isfinite(u) & np.isfinite(v))
    if len(idx) < nb + nt + 5:
        return None
    mp = max(2, smooth // 2)
    us = pd.Series(u[idx]).rolling(smooth, center=True, min_periods=mp).mean().values
    vs = pd.Series(v[idx]).rolling(smooth, center=True, min_periods=mp).mean().values
    sh = np.hypot(np.gradient(us, z[idx]), np.gradient(vs, z[idx]))
    inner = np.arange(nb, len(idx) - nt)
    if not np.isfinite(sh[inner]).any():
        return None
    k = inner[int(np.nanargmax(sh[inner]))]
    med = np.nanmedian(sh[inner])
    return dict(z=float(z[idx][k]), S=float(sh[k]), prom=float(sh[k] / med) if med > 0 else np.nan,
                edge=bool(k in (inner[0], inner[-1])), idx=idx, sh=sh, z_low=float(z[idx][inner[0]]))


def echo_estimate(e, z, valid, hab, beam_angle_deg=25.0, noise=40.0, kc=0.45, alpha=0.16, snr_db=6.0,
                  smooth=5, nb=3, nt=4):
    """Quota del picco dell'eco corretto (dB), tolto l'andamento lineare con la quota."""
    rs = np.maximum(z - hab, 1e-3) / np.cos(np.deg2rad(beam_angle_deg))      # distanza lungo il beam
    ok = np.isfinite(e) & valid & (kc * (e - noise) >= snr_db)
    idx = np.flatnonzero(ok)
    if len(idx) < nb + nt + 5:
        return None
    ps = 10 ** (kc * e[idx] / 10.0) - 10 ** (kc * noise / 10.0)
    sv = 10 * np.log10(ps) + 20 * np.log10(rs[idx]) + 2 * alpha * rs[idx]
    sv = pd.Series(sv).rolling(smooth, center=True, min_periods=max(2, smooth // 2)).mean().values
    resid = sv - np.polyval(np.polyfit(z[idx], sv, 1), z[idx])
    inner = np.arange(nb, len(idx) - nt)
    k = inner[int(np.nanargmax(resid[inner]))]
    return dict(z=float(z[idx][k]), z_top=float(z[idx].max()), idx=idx, sv=sv)


def _cells(m, dz, odd=False, minimum=1):
    n = max(minimum, int(round(m / dz)))
    if odd and n % 2 == 0:
        n += 1
    return n


def thermocline(c: xr.Dataset, window: str = "1D", smooth_m: float = 2.5, excl_bottom_m: float = 1.5,
                excl_top_m: float = 2.0, sensitivity: bool = True, kc: float = 0.45, snr_db: float = 6.0,
                agree_m: float = 2.5, min_prominence: float = 2.0):
    """Stima per finestra temporale. Restituisce (tabella, extra) con ``extra`` = quote, mappa del
    taglio e parametri (per la figura)."""
    z = c["z"].values.astype(float)
    dz = float(np.median(np.diff(z))) if len(z) > 1 else 1.0
    hab = float(c.attrs.get("instrument_height_above_bed_m", 0.6))
    beam = float(c.attrs.get("instrument_beam_angle_deg", 25.0))
    sm, nb, nt = _cells(smooth_m, dz, odd=True, minimum=3), _cells(excl_bottom_m, dz), _cells(excl_top_m, dz)
    inpos = c["deployment_qc"].values == 1 if "deployment_qc" in c else np.ones(c.sizes["time"], bool)
    cc = c.isel(time=np.flatnonzero(inpos))
    if cc.sizes["time"] == 0:
        return pd.DataFrame(), {}
    U, V, E = window_profiles(cc, window)
    days = U.index
    # stessa stima su due meta' indipendenti dei profili (coppie alterne: 0,1 | 2,3 | 4,5 ...)
    half = (np.arange(cc.sizes["time"]) // 2) % 2
    zh = []
    for h in (0, 1):
        Uh, Vh, _ = window_profiles(cc, window, sel=half == h)
        Uh, Vh = Uh.reindex(days), Vh.reindex(days)
        zz = []
        for i in range(len(days)):
            r = shear_estimate(Uh.iloc[i].values, Vh.iloc[i].values, z, sm, nb, nt)
            zz.append(r["z"] if r else np.nan)
        zh.append(np.array(zz))
    split = np.abs(zh[0] - zh[1])
    rows, Sgrid = [], np.full((len(days), len(z)), np.nan)
    for i, d in enumerate(days):
        r = shear_estimate(U.iloc[i].values, V.iloc[i].values, z, sm, nb, nt)
        if r is None:
            rows.append(dict(z_taglio=np.nan, taglio_max=np.nan, prominenza=np.nan, al_bordo=False,
                             scarto_meta=np.nan, stabile=False, affidabile=False, al_fondo=False))
            continue
        Sgrid[i, r["idx"]] = r["sh"]
        stable = bool(np.isfinite(split[i]) and split[i] <= agree_m)
        rows.append(dict(z_taglio=r["z"], taglio_max=r["S"], prominenza=r["prom"], al_bordo=r["edge"],
                         scarto_meta=float(split[i]), stabile=stable,
                         affidabile=bool(np.isfinite(r["prom"]) and r["prom"] >= min_prominence
                                         and not r["edge"] and stable),
                         al_fondo=bool(r["z"] <= r["z_low"] + dz + 1e-6)))
    df = pd.DataFrame(rows, index=days)
    df.index.name = "inizio"
    df.attrs["window"] = str(window)
    df.attrs["min_prominence"] = float(min_prominence)

    n_combo = 0
    if sensitivity:
        w2 = pd.Timedelta(window) * 2
        res = {}
        for s_m, b_m, t_m, win in itertools.product((smooth_m - 1.0, smooth_m, smooth_m + 1.0, smooth_m + 2.0),
                                                    (excl_bottom_m - 0.5, excl_bottom_m, excl_bottom_m + 1.0),
                                                    (excl_top_m, excl_top_m + 1.0), (window, w2)):
            s_, b_, t_ = _cells(max(s_m, dz), dz, odd=True, minimum=3), _cells(max(b_m, dz), dz), _cells(t_m, dz)
            Uw, Vw = (U, V) if win == window else window_profiles(cc, win)[:2]
            zz = []
            for i in range(len(Uw)):
                r = shear_estimate(Uw.iloc[i].values, Vw.iloc[i].values, z, s_, b_, t_)
                zz.append(r["z"] if r else np.nan)
            s = pd.Series(zz, index=Uw.index)
            if win != window:
                s = s.reindex(days, method="ffill", limit=1)
            res[(s_, b_, t_, str(win))] = s.reindex(days)
        R = pd.DataFrame(res)
        n_combo = R.shape[1]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            df["z_taglio_p10"] = R.quantile(0.1, axis=1)
            df["z_taglio_p90"] = R.quantile(0.9, axis=1)

    noise = np.nan
    if E is not None and np.isfinite(E.values).any():
        ev = cc["echo"].values[good_vel(cc)]
        noise = float(np.percentile(ev[np.isfinite(ev)], 1)) if np.isfinite(ev).any() else np.nan
        ze, ztop = [], []
        for i in range(len(days)):
            valid = np.isfinite(U.iloc[i].values)
            r = echo_estimate(E.iloc[i].values, z, valid, hab, beam, noise, kc=kc, snr_db=snr_db,
                              smooth=sm, nb=nb, nt=nt)
            ze.append(r["z"] if r else np.nan)
            ztop.append(r["z_top"] if r else np.nan)
        df["z_eco"], df["z_eco_max_valida"] = ze, ztop
        df["concordi"] = (df["z_taglio"] - df["z_eco"]).abs() <= agree_m
    extra = dict(z=z, S=Sgrid, days=days, window=window, dz=dz, hab=hab,
                 params=dict(window=str(window), smooth_m=smooth_m, smooth_cells=sm, excl_bottom_m=excl_bottom_m,
                             excl_bottom_cells=nb, excl_top_m=excl_top_m, excl_top_cells=nt,
                             min_prominence=min_prominence, agree_m=agree_m, echo_kc_db_per_count=kc,
                             echo_snr_db=snr_db, echo_noise_counts=noise, n_sensitivity_combinations=n_combo))
    return df, extra


RENAME = dict(z_taglio="z_shear", taglio_max="shear_max", prominenza="shear_prominence", al_bordo="at_edge",
              scarto_meta="split_half_difference", stabile="stable",
              affidabile="reliable", al_fondo="at_bottom", z_taglio_p10="z_shear_p10", z_taglio_p90="z_shear_p90",
              z_eco="z_echo", z_eco_max_valida="z_echo_top_valid", concordi="agree")
VAR_ATTRS = dict(
    z_shear=dict(units="m", standard_name="height_above_sea_floor",
                 long_name="quota del massimo taglio della corrente (stima del termoclino)"),
    shear_max=dict(units="s-1", long_name="taglio massimo |dU/dz|"),
    shear_prominence=dict(units="1", long_name="rapporto fra taglio massimo e mediana del profilo"),
    at_edge=dict(long_name="1 = massimo al bordo della fascia considerata"),
    split_half_difference=dict(units="m", long_name="differenza fra le quote stimate su due meta' indipendenti "
                               "dei profili dell'intervallo"),
    stable=dict(long_name="1 = le due meta' dei dati danno la stessa quota entro la tolleranza"),
    reliable=dict(long_name="1 = stima dal taglio affidabile (massimo netto, non al bordo, stabile sulle due meta')"),
    at_bottom=dict(long_name="1 = massimo nella cella piu' bassa considerata: termoclino o attrito sul fondo"),
    z_shear_p10=dict(units="m", long_name="quota dal taglio: 10 percentile delle combinazioni di parametri"),
    z_shear_p90=dict(units="m", long_name="quota dal taglio: 90 percentile delle combinazioni di parametri"),
    z_echo=dict(units="m", standard_name="height_above_sea_floor",
                long_name="quota del picco dell'eco corretto (stima del termoclino)"),
    z_echo_top_valid=dict(units="m", long_name="quota massima con eco sopra il rumore"),
    agree=dict(long_name="1 = stime da taglio ed eco concordi entro la tolleranza"))


def to_dataset(df: pd.DataFrame, src_attrs: dict, params: dict) -> xr.Dataset:
    """Prodotto L3 del termoclino."""
    from .archive import l3_attrs
    ds = xr.Dataset(coords=dict(time=("time", df.index.values, dict(
        standard_name="time", long_name="inizio dell'intervallo di media", axis="T"))))
    for k, nk in RENAME.items():
        if k in df:
            v = df[k].values
            v = v.astype(np.int8) if v.dtype == bool else v.astype(np.float32)
            ds[nk] = ("time", v, VAR_ATTRS[nk])
    ds.attrs = l3_attrs(src_attrs, "thermocline", df.index.values,
                        {k: v for k, v in params.items() if v is not None and not (isinstance(v, float) and np.isnan(v))})
    return ds


def summary(df: pd.DataFrame) -> dict:
    """Numeri di sintesi per ``risultati.json``."""
    from .stats import _f
    if df.empty:
        return {}
    rel = df["affidabile"].values.astype(bool)
    out = dict(n_intervalli=int(len(df)), n_affidabili=int(rel.sum()),
               affidabili_pct=_f(100.0 * rel.mean(), 1))
    if "stabile" in df:
        out["criteri_pct"] = dict(
            massimo_netto=_f(100.0 * (df["prominenza"] >= df.attrs.get("min_prominence", 2.0)).mean(), 1),
            non_al_bordo=_f(100.0 * (~df["al_bordo"].astype(bool)).mean(), 1),
            stabile_sulle_due_meta=_f(100.0 * df["stabile"].astype(bool).mean(), 1))
    zs = df["z_taglio"].where(df["affidabile"])
    out["taglio_mediana_mensile"] = {str(k): _f(v, 1) for k, v in
                                     zs.groupby(zs.index.to_period("M")).median().items()}
    if "z_eco" in df:
        out["eco_mediana_mensile"] = {str(k): _f(v, 1) for k, v in
                                      df["z_eco"].groupby(df.index.to_period("M")).median().items()}
        both = df[["z_taglio", "z_eco"]].dropna()
        if len(both) > 5:
            out["confronto_taglio_eco"] = dict(
                n=int(len(both)), correlazione=_f(both.corr().iloc[0, 1], 2),
                differenza_mediana_eco_meno_taglio=_f((both["z_eco"] - both["z_taglio"]).median(), 1),
                concordi_pct=_f(100.0 * df["concordi"].mean(), 1))
    if "z_taglio_p10" in df:
        out["banda_sensibilita_mediana_m"] = _f((df["z_taglio_p90"] - df["z_taglio_p10"]).median(), 1)
    # massima discesa: differenza fra le mediane di 7 giorni a distanza di 10 giorni
    m7 = zs.rolling(7, center=True, min_periods=3).median()
    d = m7.shift(-10) - m7
    if d.notna().any():
        i = d.idxmin()
        out["massima_discesa"] = dict(da=str(i)[:10], a=str(i + pd.Timedelta(days=10))[:10],
                                      quota_prima=_f(m7[i], 1), quota_dopo=_f(m7.shift(-10)[i], 1),
                                      variazione_m=_f(d[i], 1))
    out["al_fondo_pct"] = _f(100.0 * df["al_fondo"].mean(), 1)
    return out
