"""Statistiche ed eventi del post-processing, calcolati dai prodotti L2/L3.

Tutte le funzioni lavorano sui dati con flag 1 o 3 (buono o sospetto) e restituiscono
dizionari o tabelle pronti per ``risultati.json`` e per i CSV.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import signal

RHO, G = 1025.0, 9.81
SECTORS = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSO", "SO", "OSO", "O", "ONO", "NO", "NNO"]
CONSTITUENTS = {"M2": 12.4206012, "S2": 12.0, "N2": 12.65834751, "K1": 23.93447213, "O1": 25.81933871}


# --------------------------------------------------------------------------- utilità
def _f(x, nd=3):
    """Numero Python arrotondato, None se non finito (per il JSON)."""
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return round(x, nd) if np.isfinite(x) else None


def _ts(t):
    return None if t is None or pd.isna(t) else str(pd.Timestamp(t))[:19]


def circ_mean_deg(a, w=None):
    a = np.asarray(a, float)
    ok = np.isfinite(a)
    if w is not None:
        w = np.asarray(w, float)
        ok &= np.isfinite(w)
    if not ok.any():
        return np.nan
    r = np.deg2rad(a[ok])
    ww = np.ones(ok.sum()) if w is None else w[ok]
    return float(np.mod(np.degrees(np.arctan2(np.sum(ww * np.sin(r)), np.sum(ww * np.cos(r)))), 360))


def sector_hist(direction, weights=None, n=16):
    """Percentuale per settore di direzione (n settori centrati su N, NNE, ...)."""
    d = np.asarray(direction, float)
    ok = np.isfinite(d)
    w = np.ones(d.shape) if weights is None else np.asarray(weights, float)
    ok &= np.isfinite(w)
    if not ok.any():
        return np.zeros(n)
    k = (np.floor((d[ok] + 180.0 / n) / (360.0 / n)).astype(int)) % n
    h = np.bincount(k, weights=w[ok], minlength=n)
    return 100.0 * h / h.sum() if h.sum() > 0 else h


def speed_dir(u, v):
    """Intensità e direzione verso cui va (gradi da Nord, orario)."""
    return np.hypot(u, v), np.mod(np.degrees(np.arctan2(u, v)), 360.0)


# --------------------------------------------------------------------------- correnti
def good_vel(c):
    """Maschera (time, z) delle velocità utilizzabili (flag 1 o 3)."""
    return np.isin(c["vel_qc"].values, (1, 3)) & np.isfinite(c["u"].values)


def layer_series(c, n_bottom=3, n_surface=3, surface_max_m=7.0, mid_halfwidth_m=1.0) -> pd.DataFrame:
    """Serie di u, v per strato: fondo (celle valide piu' basse), metà colonna, superficie (celle
    valide piu' alte entro ``surface_max_m`` dalla superficie) e media sulla verticale."""
    ok = good_vel(c)
    z = c["z"].values
    t = pd.DatetimeIndex(c["time"].values)
    U = np.where(ok, c["u"].values, np.nan)
    V = np.where(ok, c["v"].values, np.nan)
    nt = len(t)
    wd = c["water_depth"].values.astype(float) if "water_depth" in c else np.full(nt, np.nan)
    inpos = c["deployment_qc"].values == 1 if "deployment_qc" in c else np.ones(nt, bool)
    wd_med = float(np.nanmedian(wd[inpos])) if np.isfinite(wd[inpos]).any() else float(z.max())
    lo = np.argsort(z)[:n_bottom]
    mid = np.flatnonzero(np.abs(z - wd_med / 2.0) <= mid_halfwidth_m)
    if len(mid) == 0:
        mid = np.array([int(np.argmin(np.abs(z - wd_med / 2.0)))])
    out = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        out["u_fondo"], out["v_fondo"] = np.nanmean(U[:, lo], 1), np.nanmean(V[:, lo], 1)
        out["u_meta"], out["v_meta"] = np.nanmean(U[:, mid], 1), np.nanmean(V[:, mid], 1)
        out["u_media"], out["v_media"] = np.nanmean(U, 1), np.nanmean(V, 1)
    us, vs, ds_ = np.full(nt, np.nan), np.full(nt, np.nan), np.full(nt, np.nan)
    dist = wd[:, None] - z[None, :]
    oks = ok & (dist <= surface_max_m) & (dist > 0)
    for i in np.flatnonzero(oks.sum(axis=1) >= n_surface):
        idx = np.flatnonzero(oks[i])
        top = idx[np.argsort(z[idx])[-n_surface:]]
        us[i], vs[i], ds_[i] = U[i, top].mean(), V[i, top].mean(), dist[i, top].mean()
    out["u_sup"], out["v_sup"] = us, vs
    df = pd.DataFrame(out, index=t)
    df.loc[~inpos, :] = np.nan
    for k in ("fondo", "meta", "sup", "media"):
        df[f"vel_{k}"], df[f"dir_{k}"] = speed_dir(df[f"u_{k}"].values, df[f"v_{k}"].values)
    df.attrs = dict(
        fondo=f"{z[lo].min():.1f}-{z[lo].max():.1f} m dal fondale",
        meta=f"{z[mid].min():.1f}-{z[mid].max():.1f} m dal fondale",
        sup=(f"{np.nanmedian(ds_):.1f} m sotto la superficie (mediana)" if np.isfinite(ds_).any() else "n.d."),
        media="media sulla verticale delle celle valide",
        z_fondo=[float(z[lo].min()), float(z[lo].max())], z_meta=[float(z[mid].min()), float(z[mid].max())],
        prof_sup=_f(np.nanmedian(ds_), 2) if np.isfinite(ds_).any() else None, profondita_mediana=_f(wd_med, 2))
    return df


def principal_axis(u, v) -> dict:
    """Asse principale della variabilità (direzione 0-180 gradi da Nord) e frazione di varianza."""
    ok = np.isfinite(u) & np.isfinite(v)
    if ok.sum() < 10:
        return {}
    cov = np.cov(np.vstack([u[ok], v[ok]]))
    val, vec = np.linalg.eigh(cov)
    e = vec[:, np.argmax(val)]
    ang = float(np.mod(np.degrees(np.arctan2(e[0], e[1])), 180.0))
    return dict(direzione_gradi=_f(ang, 1), frazione_varianza=_f(val.max() / val.sum(), 3))


def speed_stats(u, v, t) -> dict:
    u, v = np.asarray(u, float), np.asarray(v, float)
    ok = np.isfinite(u) & np.isfinite(v)
    if not ok.any():
        return dict(n=0)
    s, d = speed_dir(u[ok], v[ok])
    tt = pd.DatetimeIndex(t)[ok]
    um, vm = u[ok].mean(), v[ok].mean()
    vs, vd = speed_dir(um, vm)
    h = sector_hist(d)
    return dict(n=int(ok.sum()), media=_f(s.mean()), mediana=_f(np.median(s)), p95=_f(np.percentile(s, 95)),
                max=_f(s.max()), quando_max=_ts(tt[int(np.argmax(s))]),
                vettore_medio_vel=_f(vs), vettore_medio_dir=_f(vd, 1), costanza=_f(vs / s.mean() if s.mean() > 0 else np.nan),
                settori_pct=[_f(x, 1) for x in h], settore_prevalente=SECTORS[int(np.argmax(h))])


def rotary_spectrum(u, v, dt_hours=1.0, max_gap=6, nperseg_h=512):
    """Spettro rotatorio (orario / antiorario) della corrente sul tratto continuo piu' lungo."""
    us = pd.Series(np.asarray(u, float)).interpolate(limit=max_gap, limit_area="inside").values
    vs = pd.Series(np.asarray(v, float)).interpolate(limit=max_gap, limit_area="inside").values
    ok = np.isfinite(us) & np.isfinite(vs)
    if not ok.any():
        return None
    edges = np.flatnonzero(np.diff(np.r_[0, ok.astype(int), 0]))
    starts, ends = edges[::2], edges[1::2]
    i = int(np.argmax(ends - starts))
    s, e = starts[i], ends[i]
    n = e - s
    if n < 240:
        return None
    nseg = int(min(nperseg_h / dt_hours, 2 ** np.floor(np.log2(n / 2))))
    zc = us[s:e] + 1j * vs[s:e]
    f, P = signal.welch(zc, fs=24.0 / dt_hours, nperseg=nseg, noverlap=nseg // 2, window="hann",
                        return_onesided=False, detrend="constant")
    pos, neg = f > 0, f < 0
    fp = f[pos]
    o = np.argsort(fp)
    fn = -f[neg]
    on = np.argsort(fn)
    return dict(f=fp[o], antiorario=P[pos][o], orario=np.interp(fp[o], fn[on], P[neg][on]),
                n_ore=int(n), nperseg=nseg)


def band_variance(spec, latitude=None) -> dict:
    """Frazione di varianza della corrente per banda di frequenza."""
    f, tot = spec["f"], spec["orario"] + spec["antiorario"]
    df = f[1] - f[0]
    T = tot.sum() * df
    bands = {"bassa_frequenza (>2 giorni)": (0, 0.5), "diurna": (0.85, 1.15), "semidiurna": (1.8, 2.15),
             "alta_frequenza (<11 h)": (2.15, np.inf)}
    out = {}
    if latitude is not None:
        fi = 2 * 1.0027379 * abs(np.sin(np.deg2rad(float(latitude))))
        out["frequenza_inerziale_cpd"] = _f(fi, 3)
        out["periodo_inerziale_h"] = _f(24.0 / fi, 1) if fi > 0 else None
        bands["inerziale"] = (0.9 * fi, 1.1 * fi)
    for k, (a, b) in bands.items():
        m = (f >= a) & (f < b)
        out[k] = _f(tot[m].sum() * df / T if T > 0 else np.nan, 3)
        if k in ("diurna", "inerziale", "semidiurna") and m.any():
            cw, ccw = spec["orario"][m].sum(), spec["antiorario"][m].sum()
            out[k + "_quota_oraria"] = _f(cw / (cw + ccw) if cw + ccw > 0 else np.nan, 2)
    return out


# --------------------------------------------------------------------------- onde
def wave_good(w):
    return np.isin(w["wave_qc"].values, (1, 3)) & np.isfinite(w["Hm0"].values)


def dir_good(w):
    if "direction_qc" not in w:
        return np.zeros(w.sizes["time"], bool)
    return (w["direction_qc"].values == 1) & np.isfinite(w["Dp"].values)


def wave_stats(w, dir_hm0_min=0.5) -> dict:
    g = wave_good(w)
    t = pd.DatetimeIndex(w["time"].values)
    hm = w["Hm0"].values
    out = dict(n_burst=int(w.sizes["time"]), n_validi=int(g.sum()))
    if not g.any():
        return out
    h = hm[g]
    im = int(np.argmax(np.where(g, hm, -np.inf)))
    out["Hm0"] = dict(media=_f(h.mean()), mediana=_f(np.median(h)), p90=_f(np.percentile(h, 90)),
                      p95=_f(np.percentile(h, 95)), p99=_f(np.percentile(h, 99)), max=_f(h.max()),
                      quando_max=_ts(t[im]))
    out["superamenti_pct"] = {f">{x:g} m": _f(100.0 * (h > x).mean(), 2) for x in (0.5, 1.0, 1.5, 2.0)}
    for k in ("Tp", "Tm01", "Tm02"):
        if k in w:
            x = w[k].values[g]
            out[k] = dict(media=_f(np.nanmean(x), 2), mediana=_f(np.nanmedian(x), 2),
                          p10=_f(np.nanpercentile(x, 10), 2), p90=_f(np.nanpercentile(x, 90), 2))
    if "Tp" in w:
        out["al_massimo"] = dict(Tp=_f(w["Tp"].values[im], 2), Tm02=_f(w["Tm02"].values[im], 2),
                                 Dp=_f(w["Dp"].values[im], 0) if "Dp" in w else None,
                                 Ub=_f(w["Ub"].values[im], 3) if "Ub" in w else None)
    if "Ub" in w:
        ub = w["Ub"].values[g]
        out["Ub"] = dict(mediana=_f(np.nanmedian(ub)), p99=_f(np.nanpercentile(ub, 99)), max=_f(np.nanmax(ub)))
    if "Hm0_pressure" in w:
        r = w["Hm0_pressure"].values[g] / np.maximum(h, 1e-6)
        sel = h >= 0.5
        out["confronto_pressione"] = dict(
            rapporto_mediano_Hm0p_su_Hm0=_f(np.nanmedian(r[sel]) if sel.any() else np.nan, 2),
            frequenza_taglio_mediana_Hz=_f(np.nanmedian(w["fcut_pressure"].values[g]), 3))
    gd = dir_good(w)
    if gd.any():
        d = w["Dp"].values
        sel = gd & (hm >= dir_hm0_min)
        out["direzione"] = dict(n_validi=int(gd.sum()), soglia_Hm0=dir_hm0_min, n_sopra_soglia=int(sel.sum()))
        if sel.any():
            hh = sector_hist(d[sel])
            he = sector_hist(d[sel], weights=hm[sel] ** 2)
            out["direzione"].update(
                settori_pct=[_f(x, 1) for x in hh], settori_energia_pct=[_f(x, 1) for x in he],
                settore_prevalente=SECTORS[int(np.argmax(hh))],
                settore_prevalente_energia=SECTORS[int(np.argmax(he))],
                media_pesata_energia=_f(circ_mean_deg(d[sel], hm[sel] ** 2), 0))
        if "dir_level_spread" in w:
            out["direzione"]["differenza_fra_quote_mediana"] = _f(np.nanmedian(w["dir_level_spread"].values[sel]), 1) \
                if sel.any() else None
        if "dir_coherence" in w:
            out["direzione"]["coerenza_mediana"] = _f(np.nanmedian(w["dir_coherence"].values[sel]), 2) \
                if sel.any() else None
    return out


def storms(w, thr=1.0, sep_hours=12.0) -> pd.DataFrame:
    """Mareggiate: periodi con Hm0 >= ``thr``; superamenti a meno di ``sep_hours`` uniti."""
    g = wave_good(w)
    t = pd.DatetimeIndex(w["time"].values)
    hm = np.where(g, w["Hm0"].values, np.nan)
    idx = np.flatnonzero(hm >= thr)
    cols = ["inizio", "fine", "durata_h", "Hm0_max", "quando_max", "Tp", "Tm02", "Dp", "Ub_max", "n_burst"]
    if len(idx) == 0:
        return pd.DataFrame(columns=cols)
    step_h = float(np.median(np.diff(t.values).astype("timedelta64[s]").astype(float))) / 3600.0 if len(t) > 1 else 0.0
    gaps = np.diff(t[idx].values).astype("timedelta64[s]").astype(float) / 3600.0
    brk = np.flatnonzero(gaps > sep_hours) + 1
    rows = []
    for grp in np.split(idx, brk):
        i0, i1 = int(grp[0]), int(grp[-1])
        seg = slice(i0, i1 + 1)
        im = i0 + int(np.nanargmax(hm[seg]))
        get = lambda k: float(w[k].values[im]) if k in w else np.nan
        rows.append(dict(
            inizio=t[i0], fine=t[i1], durata_h=(t[i1] - t[i0]).total_seconds() / 3600.0 + step_h,
            Hm0_max=float(hm[im]), quando_max=t[im], Tp=get("Tp"), Tm02=get("Tm02"),
            Dp=get("Dp") if ("direction_qc" in w and w["direction_qc"].values[im] in (1, 3)) else np.nan,
            Ub_max=float(np.nanmax(w["Ub"].values[seg])) if "Ub" in w else np.nan, n_burst=int(len(grp))))
    return pd.DataFrame(rows, columns=cols)


def enrich_storms(st: pd.DataFrame, layers=None, T=None, dep=None, pad_h=3.0) -> pd.DataFrame:
    """Aggiunge a ogni mareggiata corrente massima, variazione di temperatura e movimento del telaio."""
    st = st.copy()
    if st.empty:
        return st
    for k in ("corrente_sup_max", "corrente_fondo_max", "dT_fondo"):
        st[k] = np.nan
    st["movimento_telaio"] = False
    st["rotazione_heading"] = np.nan
    for i, r in st.iterrows():
        a, b = r["inizio"], r["fine"]
        if layers is not None and len(layers):
            m = (layers.index >= a) & (layers.index <= b + pd.Timedelta(hours=1))
            if m.any():
                st.loc[i, "corrente_sup_max"] = np.nanmax(layers["vel_sup"].values[m]) \
                    if np.isfinite(layers["vel_sup"].values[m]).any() else np.nan
                st.loc[i, "corrente_fondo_max"] = np.nanmax(layers["vel_fondo"].values[m]) \
                    if np.isfinite(layers["vel_fondo"].values[m]).any() else np.nan
        if T is not None and len(T):
            before = T[(T.index >= a - pd.Timedelta(hours=24)) & (T.index < a)]
            after = T[(T.index > b) & (T.index <= b + pd.Timedelta(hours=24))]
            if before.notna().any() and after.notna().any():
                st.loc[i, "dT_fondo"] = float(after.mean() - before.mean())
        # rotazione netta: assetto dopo l'ultimo movimento della mareggiata meno quello prima del primo
        hit = [ev for ev in (dep or {}).get("motion_events", [])
               if pd.Timestamp(ev["start"]) <= b + pd.Timedelta(hours=pad_h)
               and pd.Timestamp(ev["end"]) >= a - pd.Timedelta(hours=pad_h)]
        if hit:
            hit.sort(key=lambda ev: pd.Timestamp(ev["start"]))
            st.loc[i, "movimento_telaio"] = True
            h0, h1 = float(hit[0].get("heading_before", np.nan)), float(hit[-1].get("heading_after", np.nan))
            st.loc[i, "rotazione_heading"] = (h1 - h0 + 180) % 360 - 180
    return st


def motion_table(dep, w=None, before_h=3.0) -> pd.DataFrame:
    """Movimenti del telaio con l'altezza d'onda e la velocità orbitale nelle ore precedenti."""
    rows = []
    t = pd.DatetimeIndex(w["time"].values) if w is not None else None
    g = wave_good(w) if w is not None else None
    for ev in (dep or {}).get("motion_events", []):
        a, b = pd.Timestamp(ev["start"]), pd.Timestamp(ev["end"])
        r = dict(inizio=a, fine=b,
                 rotazione_heading=(float(ev.get("heading_after", np.nan)) - float(ev.get("heading_before", np.nan))
                                    + 180) % 360 - 180,
                 heading_prima=ev.get("heading_before"), heading_dopo=ev.get("heading_after"),
                 pitch_prima=ev.get("pitch_before"), pitch_dopo=ev.get("pitch_after"),
                 roll_prima=ev.get("roll_before"), roll_dopo=ev.get("roll_after"),
                 Hm0_max_prima=np.nan, Ub_max_prima=np.nan)
        if w is not None:
            m = g & (t >= a - pd.Timedelta(hours=before_h)) & (t <= b)
            if m.any():
                r["Hm0_max_prima"] = float(np.nanmax(w["Hm0"].values[m]))
                if "Ub" in w:
                    r["Ub_max_prima"] = float(np.nanmax(w["Ub"].values[m]))
        rows.append(r)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- temperatura
def max_change_window(Td: pd.Series, diff_days=4, halfwidth_days=12):
    """Periodo di massima variazione della temperatura giornaliera su ``diff_days`` giorni."""
    Td = Td.dropna()
    if len(Td) < diff_days + 3:
        return None
    d = Td.shift(-diff_days) - Td
    if not d.notna().any():
        return None
    i = d.abs().idxmax()
    j = i + pd.Timedelta(days=diff_days)
    c = i + pd.Timedelta(days=diff_days / 2.0)
    return dict(da=i, a=j, delta=float(d[i]), t0=c - pd.Timedelta(days=halfwidth_days),
                t1=c + pd.Timedelta(days=halfwidth_days))


def hp_segment(x: pd.Series, highpass_h=49, min_len=72):
    """Tratto continuo piu' lungo di una serie oraria, tolta la media mobile su ``highpass_h`` ore.
    Restituisce None se e' piu' corto di ``min_len`` ore."""
    if x is None or not x.notna().any():
        return None
    x = x.resample("1h").mean().interpolate(limit=3, limit_area="inside")
    hp = (x - x.rolling(highpass_h, center=True, min_periods=highpass_h // 4).mean()).values
    ok = np.isfinite(hp)
    if not ok.any():
        return None
    edges = np.flatnonzero(np.diff(np.r_[0, ok.astype(int), 0]))
    starts, ends = edges[::2], edges[1::2]
    k = int(np.argmax(ends - starts))
    y = hp[starts[k]:ends[k]]
    return y if len(y) >= min_len else None


def dominant_periods(x: pd.Series, highpass_h=49, pmin=3.0, pmax=60.0, n=4) -> list:
    """Periodi dominanti (ore) delle oscillazioni di una serie oraria, tolta la media mobile."""
    y = hp_segment(x, highpass_h)
    if y is None:
        return []
    f, P = signal.periodogram(signal.detrend(y), fs=1.0, window="hann")
    with np.errstate(divide="ignore"):
        per = 1.0 / f
    m = (per >= pmin) & (per <= pmax)
    if not m.any() or P[m].sum() <= 0:
        return []
    pk, _ = signal.find_peaks(np.where(m, P, 0.0))
    pk = pk[np.argsort(P[pk])[::-1][:n]]
    tot = P[1:].sum()
    return [dict(periodo_h=_f(per[i], 1), frazione_potenza=_f(P[i] / tot, 3)) for i in pk]


def temperature_stats(T: pd.Series) -> dict:
    """``T``: serie oraria della temperatura valida (flag 1 o 3)."""
    T = T.dropna()
    if T.empty:
        return dict(n_ore=0)
    Td = T.resample("1D").mean()
    rng = (T.resample("1D").max() - T.resample("1D").min()).dropna()
    out = dict(n_ore=int(len(T)), media=_f(T.mean(), 2), min=_f(T.min(), 2), quando_min=_ts(T.idxmin()),
               max=_f(T.max(), 2), quando_max=_ts(T.idxmax()),
               escursione_giornaliera_mediana=_f(rng.median(), 2), escursione_giornaliera_max=_f(rng.max(), 2),
               quando_escursione_max=_ts(rng.idxmax()) if len(rng) else None)
    mc = max_change_window(Td)
    if mc:
        out["massima_variazione"] = dict(da=_ts(mc["da"]), a=_ts(mc["a"]), delta=_f(mc["delta"], 2), giorni=4)
    return out


# --------------------------------------------------------------------------- livello e marea
def tide_fit(t, level_m):
    """Analisi armonica ai minimi quadrati del livello (da pressione). Fasi relative all'inizio."""
    t = pd.DatetimeIndex(t)
    y = np.asarray(level_m, float)
    ok = np.isfinite(y)
    if ok.sum() < 48:
        return None
    th = (t - t[ok][0]).total_seconds().values / 3600.0
    span = (th[ok].max() - th[ok].min()) / 24.0
    names = ["M2", "K1"] if span < 15 else (["M2", "S2", "K1", "O1"] if span < 28 else list(CONSTITUENTS))
    cols = [np.ones_like(th), th - th[ok].mean()]
    for nme in names:
        w = 2 * np.pi / CONSTITUENTS[nme]
        cols += [np.cos(w * th), np.sin(w * th)]
    A = np.column_stack(cols)
    coef, *_ = np.linalg.lstsq(A[ok], y[ok], rcond=None)
    tide = A[:, 2:] @ coef[2:]
    trend = A[:, :2] @ coef[:2]
    resid = y - trend - tide
    outside = (th < th[ok].min()) | (th > th[ok].max())     # niente marea dove non ci sono misure
    tide = np.where(outside, np.nan, tide)
    cons = {}
    for i, nme in enumerate(names):
        a, b = coef[2 + 2 * i], coef[3 + 2 * i]
        cons[nme] = dict(periodo_h=_f(CONSTITUENTS[nme], 3), ampiezza_cm=_f(100 * np.hypot(a, b), 2),
                         fase_gradi=_f(np.mod(np.degrees(np.arctan2(b, a)), 360), 0))
    var_y = np.nanvar(y[ok] - trend[ok])
    info = dict(costituenti=cons, durata_giorni=_f(span, 1),
                varianza_spiegata=_f(1 - np.nanvar(resid[ok]) / var_y if var_y > 0 else np.nan, 3),
                residuo_std_cm=_f(100 * np.nanstd(resid[ok]), 2),
                escursione_marea_max_cm=_f(100 * (np.nanmax(tide) - np.nanmin(tide)), 1),
                escursione_livello_cm=_f(100 * (np.nanmax(y[ok]) - np.nanmin(y[ok])), 1),
                nota="fasi relative all'inizio della serie; il residuo comprende effetti meteorologici e la "
                     "pressione atmosferica (il sensore misura la pressione totale)")
    return dict(info=info, marea=pd.Series(tide, index=t), residuo=pd.Series(resid, index=t),
                livello=pd.Series(y - np.nanmean(y[ok]), index=t))


# --------------------------------------------------------------------------- tabelle
def monthly_table(w=None, layers=None, T=None, thermo=None) -> pd.DataFrame:
    """Statistiche mensili di onde, correnti, temperatura e termoclino."""
    parts = []
    if w is not None:
        g = wave_good(w)
        d = pd.DataFrame(dict(Hm0=np.where(g, w["Hm0"].values, np.nan),
                              Tp=np.where(g, w["Tp"].values, np.nan)), index=pd.DatetimeIndex(w["time"].values))
        grp = d.groupby(d.index.to_period("M"))
        m = pd.DataFrame({"onde_n": grp["Hm0"].count(), "Hm0_media": grp["Hm0"].mean(),
                          "Hm0_mediana": grp["Hm0"].median(), "Hm0_max": grp["Hm0"].max(),
                          "Tp_mediana": grp["Tp"].median(),
                          "Hm0_sopra_1m_pct": grp["Hm0"].apply(lambda x: 100.0 * (x.dropna() > 1.0).mean()
                                                               if x.notna().any() else np.nan)})
        if "Dp" in w:
            gd = dir_good(w) & (np.nan_to_num(w["Hm0"].values) >= 0.5)
            dd = pd.Series(np.where(gd, w["Dp"].values, np.nan), index=d.index)
            m["Dp_settore_prevalente"] = dd.groupby(dd.index.to_period("M")).apply(
                lambda x: SECTORS[int(np.argmax(sector_hist(x.values)))] if x.notna().any() else "")
        parts.append(m)
    if layers is not None and len(layers):
        grp = layers.groupby(layers.index.to_period("M"))
        m = pd.DataFrame({"correnti_ore": grp["vel_media"].count()})
        for k, nme in (("sup", "superficie"), ("meta", "meta_colonna"), ("fondo", "fondo")):
            m[f"corrente_{nme}_media"] = grp[f"vel_{k}"].mean()
            m[f"corrente_{nme}_max"] = grp[f"vel_{k}"].max()
        parts.append(m)
    if T is not None and len(T):
        grp = T.groupby(T.index.to_period("M"))
        parts.append(pd.DataFrame({"temperatura_ore": grp.count(), "temperatura_media": grp.mean(),
                                   "temperatura_min": grp.min(), "temperatura_max": grp.max()}))
    if thermo is not None and len(thermo):
        zs = thermo["z_taglio"].where(thermo["affidabile"])
        grp = zs.groupby(zs.index.to_period("M"))
        m = pd.DataFrame({"termoclino_taglio_mediana": grp.median(), "termoclino_giorni_affidabili": grp.count()})
        if "z_eco" in thermo:
            m["termoclino_eco_mediana"] = thermo["z_eco"].groupby(thermo.index.to_period("M")).median()
        parts.append(m)
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, axis=1).sort_index()
    counts = [k for k in ("onde_n", "correnti_ore", "temperatura_ore") if k in out]
    if counts:
        out = out[out[counts].fillna(0).sum(axis=1) > 0]      # mesi senza alcun dato valido
    out.index = out.index.astype(str)
    out.index.name = "mese"
    return out


def joint_table(direction, value, edges, unit="m", n=16) -> pd.DataFrame:
    """Distribuzione congiunta direzione-valore: percentuale dei casi per settore di direzione
    (righe) e classe di valore (colonne), con i totali."""
    d, v = np.asarray(direction, float), np.asarray(value, float)
    ok = np.isfinite(d) & np.isfinite(v)
    names = SECTORS if n == 16 else [f"{i * 360 / n:.0f}" for i in range(n)]
    cols = []
    for a, b in zip(edges[:-1], edges[1:]):
        cols.append(f">= {a:g} {unit}" if not np.isfinite(b) else (f"< {b:g} {unit}" if a <= 0 else f"{a:g}-{b:g} {unit}"))
    tab = np.zeros((n, len(cols)))
    if ok.any():
        k = (np.floor((d[ok] + 180.0 / n) / (360.0 / n)).astype(int)) % n
        j = np.clip(np.searchsorted(np.asarray(edges, float), v[ok], side="right") - 1, 0, len(cols) - 1)
        np.add.at(tab, (k, j), 1.0)
        tab *= 100.0 / ok.sum()
    df = pd.DataFrame(tab, index=names, columns=cols)
    df["totale"] = df.sum(axis=1)
    df.loc["totale"] = df.sum(axis=0)
    df.index.name = "settore"
    df.attrs["n"] = int(ok.sum())
    return df


def progressive_vector(u, v, t):
    """Spostamento cumulato (km verso Est e verso Nord) che una particella avrebbe con la corrente
    misurata nel punto; le ore senza misura valgono zero. Restituisce (x_km, y_km, frazione valida)."""
    u, v = np.asarray(u, float), np.asarray(v, float)
    t = pd.DatetimeIndex(t)
    if len(t) < 2:
        return np.zeros(len(t)), np.zeros(len(t)), 0.0
    dt = np.gradient(t.values.astype("datetime64[s]").astype(float))
    ok = np.isfinite(u) & np.isfinite(v)
    x = np.cumsum(np.where(ok, u, 0.0) * dt) / 1000.0
    y = np.cumsum(np.where(ok, v, 0.0) * dt) / 1000.0
    return x, y, float(ok.mean())


def qc_summary(c=None, temp=None, b=None, w=None) -> dict:
    """Conteggi dei flag per valore e per causa, per ogni prodotto."""
    from .archive import flag_stats, VEL_CAUSES, POS_CAUSES, TEMP_CAUSES, WAVE_CAUSES, DIR_CAUSES
    out = {}
    if c is not None:
        out["velocita"] = flag_stats(c["vel_qc"].values, c["vel_qc_cause"].values, VEL_CAUSES)
        out["posizione"] = flag_stats(c["deployment_qc"].values, c["deployment_qc_cause"].values, POS_CAUSES)
        inpos = c["deployment_qc"].values == 1
        if inpos.any() and "water_depth" in c:
            # celle "utili": in posizione e sotto il limite di contaminazione della superficie
            cause = c["vel_qc_cause"].values.astype(int)[inpos]
            util = (cause & 4) == 0
            flag = c["vel_qc"].values[inpos]
            out["velocita"]["celle_utili"] = int(util.sum())
            out["velocita"]["celle_utili_buone_pct"] = _f(100.0 * (flag[util] == 1).mean(), 1) if util.any() else None
            out["velocita"]["celle_utili_buone_o_sospette_pct"] = _f(100.0 * np.isin(flag[util], (1, 3)).mean(), 1) \
                if util.any() else None
    if temp is not None:
        out["temperatura"] = flag_stats(temp["temperature_qc"].values, temp["temperature_qc_cause"].values, TEMP_CAUSES)
    if b is not None and "burst_qc" in b:
        out["burst"] = flag_stats(b["burst_qc"].values, b["burst_qc_cause"].values, WAVE_CAUSES)
    if w is not None:
        out["onde"] = flag_stats(w["wave_qc"].values, w["wave_qc_cause"].values, WAVE_CAUSES)
        if "direction_qc" in w:
            out["direzione_onde"] = flag_stats(w["direction_qc"].values, w["direction_qc_cause"].values, DIR_CAUSES)
    return out
