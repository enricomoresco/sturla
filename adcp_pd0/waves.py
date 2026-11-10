"""Analisi onde per burst: spettro da pressione (con correzione di attenuazione),
opzionalmente spettro da superficie misurata col beam verticale (stima tipo AST)
e direzione con metodo PUV.

Fornisce una prima stima indipendente; per i prodotti direzionali completi usare il
software Teledyne (Velocity) o librerie dedicate.
"""
from __future__ import annotations

import numpy as np
import xarray as xr
from scipy import signal

from .currents import RHO, G
from .transform import to_earth


def wavenumber(f, h):
    """Numero d'onda k [rad/m] da dispersione lineare (Newton, vettoriale)."""
    w = 2 * np.pi * np.asarray(f, float)
    k0 = w**2 / G
    k = np.where(w > 0, k0 / np.sqrt(np.tanh(np.maximum(k0 * h, 1e-12))), 1e-9)
    for _ in range(30):
        th = np.tanh(k * h)
        fval = G * k * th - w**2
        df = G * th + G * k * h * (1 - th**2)
        k = np.maximum(k - fval / df, 1e-9)
    return k


def find_bursts(t: np.ndarray, gap_factor: float = 5.0):
    """Indici [inizio, fine) dei burst separati da interruzioni nel campionamento."""
    if len(t) < 2:
        return np.array([[0, len(t)]]), np.nan
    dt = np.diff(t).astype("timedelta64[us]").astype(float) / 1e6
    med = float(np.median(dt))
    brk = np.flatnonzero(dt > max(gap_factor * med, med + 1.0)) + 1
    starts = np.r_[0, brk]
    ends = np.r_[brk, len(t)]
    return np.stack([starts, ends], axis=1), med


def _interp_gaps(x, max_frac=0.1):
    bad = ~np.isfinite(x)
    if bad.mean() > max_frac:
        return None
    if bad.any():
        i = np.arange(len(x))
        x = x.copy()
        x[bad] = np.interp(i[bad], i[~bad], x[~bad])
    return x


def surface_from_vb(echo, rng, depth_est, window=0.35):
    """Distanza della superficie dal trasduttore per ogni ping dal massimo di eco del beam
    verticale (interpolazione parabolica). ``depth_est`` = profondità stimata del trasduttore."""
    echo = echo.astype(np.float32)
    lo, hi = depth_est * (1 - window), depth_est * (1 + window)
    m = (rng[None, :] >= np.atleast_1d(lo)[:, None]) & (rng[None, :] <= np.atleast_1d(hi)[:, None])
    e = np.where(m, echo, -np.inf)
    i = np.argmax(e, axis=1)
    ok = np.isfinite(e[np.arange(len(e)), i]) & (i > 0) & (i < echo.shape[1] - 1)
    i_c = np.clip(i, 1, echo.shape[1] - 2)
    y0, y1, y2 = (echo[np.arange(len(e)), i_c + k] for k in (-1, 0, 1))
    den = y0 - 2 * y1 + y2
    delta = np.where(den != 0, 0.5 * (y0 - y2) / den, 0.0)
    delta = np.clip(delta, -0.5, 0.5)
    cs = rng[1] - rng[0]
    r = rng[i_c] + delta * cs
    r[~ok] = np.nan
    return r


def _despike(x, nmad=4.0):
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med)) * 1.4826
    if not np.isfinite(mad) or mad == 0:
        return x
    y = x.copy()
    y[np.abs(x - med) > nmad * mad] = np.nan
    return y


def _moments(f, S, band):
    ff, SS = f[band], S[band]
    m0 = np.trapezoid(SS, ff)
    m1 = np.trapezoid(SS * ff, ff)
    m2 = np.trapezoid(SS * ff**2, ff)
    ip = np.argmax(SS)
    # picco con interpolazione parabolica
    fp = ff[ip]
    if 0 < ip < len(SS) - 1:
        y0, y1, y2 = SS[ip - 1:ip + 2]
        den = y0 - 2 * y1 + y2
        if den != 0:
            fp = ff[ip] + 0.5 * (y0 - y2) / den * (ff[1] - ff[0])
    return dict(Hm0=4 * np.sqrt(m0), Tp=1 / fp if fp > 0 else np.nan,
                Tm01=m0 / m1 if m1 > 0 else np.nan, Tm02=np.sqrt(m0 / m2) if m2 > 0 else np.nan)


def analyze_waves(reader, hab: float = 0.5, p_atm: float = 0.0, nperseg: int = 256,
                  fmin: float = 0.04, fmax: float = 0.6, kp2_min: float = 0.05,
                  min_samples: int = 256, use_vb: bool = True, puv_cell: int | None = None,
                  orientation: str | None = None, declination: float = 0.0,
                  max_samples_per_load: int = 2_000_000, verbose: bool = True) -> xr.Dataset:
    """Statistiche d'onda per burst dallo stream onde.

    hab        : altezza del sensore di pressione/trasduttore sul fondo [m]
    p_atm      : offset da sottrarre alla pressione [dbar] (0 se il sensore è già azzerato in aria)
    kp2_min    : frequenza di taglio dove la funzione di trasferimento Kp^2 scende sotto questa soglia
    puv_cell   : indice (0-based) della cella da usare per la direzione PUV (None = no)
    """
    t_all = reader.times()
    bursts, dt_med = find_bursts(t_all)
    fs = 1.0 / dt_med
    a = reader.attrs
    nseg = int(min(nperseg, 2 ** np.floor(np.log2(max(min_samples, 16)))))
    freq = np.fft.rfftfreq(nseg, 1 / fs)
    vars_needed = ["pressure", "heading", "pitch", "roll"]
    has_vb = use_vb and "vb_echo" in reader.variables()
    if has_vb:
        vars_needed.append("vb_echo")
    if puv_cell is not None and "vel" in reader.variables():
        vars_needed.append("vel")

    rows, S_p, S_v = [], [], []
    # carica gruppi di burst consecutivi per contenere la memoria
    groups, cur, n_cur = [], [], 0
    for b in bursts:
        cur.append(b)
        n_cur += b[1] - b[0]
        if n_cur >= max_samples_per_load:
            groups.append(cur)
            cur, n_cur = [], 0
    if cur:
        groups.append(cur)
    for gi, grp in enumerate(groups):
        t0 = t_all[grp[0][0]]
        t1 = t_all[grp[-1][1] - 1] + np.timedelta64(1, "us")
        isel = {"cell": [puv_cell]} if "vel" in vars_needed else None
        ds = reader.load(vars_needed, t0, t1, isel=isel)
        tt = ds.time.values
        for s, e in grp:
            sel = (tt >= t_all[s]) & (tt <= t_all[e - 1])
            n = int(sel.sum())
            row = dict(time=t_all[s], n=n, duration_s=n * dt_med)
            nanS = np.full(len(freq), np.nan)
            if n < min_samples:
                rows.append(row); S_p.append(nanS); S_v.append(nanS)
                continue
            p = ds.pressure.values[sel].astype(float) - p_atm
            depth_s = float(np.nanmean(p)) * 1e4 / (RHO * G)   # profondità sensore [m]
            h = depth_s + hab
            row.update(depth=h, depth_sensor=depth_s, p_std=float(np.nanstd(p)))
            p = _interp_gaps(p)
            if p is None or h <= 0.5:
                rows.append(row); S_p.append(nanS); S_v.append(nanS)
                continue
            eta_p = (p - p.mean()) * 1e4 / (RHO * G)   # in metri d'acqua
            f, Spp = signal.welch(signal.detrend(eta_p), fs=fs, nperseg=nseg, window="hann",
                                  noverlap=nseg // 2)
            k = wavenumber(f, h)
            Kp2 = (np.cosh(k * hab) / np.cosh(k * h)) ** 2
            okf = (f >= fmin) & (f <= fmax) & (Kp2 >= kp2_min)
            fcut = float(f[okf].max()) if okf.any() else np.nan
            See = np.where(okf, Spp / np.maximum(Kp2, 1e-12), np.nan)
            row["fcut"] = fcut
            if okf.sum() >= 3:
                for kk, vv in _moments(f, np.nan_to_num(See), okf).items():
                    row[kk] = vv
            S_p.append(See)
            # superficie dal beam verticale
            Sv = nanS
            if has_vb:
                rng = ds.vb_range.values
                r = surface_from_vb(ds.vb_echo.values[sel], rng, np.full(n, depth_s))
                r = _interp_gaps(_despike(r), max_frac=0.15)
                if r is not None:
                    fv, Sv = signal.welch(signal.detrend(r), fs=fs, nperseg=nseg, window="hann",
                                          noverlap=nseg // 2)
                    bv = (fv >= fmin) & (fv <= min(fmax * 2, fs / 2))
                    for kk, vv in _moments(fv, Sv, bv).items():
                        row[kk + "_vb"] = vv
                    row["surface_range_vb"] = float(np.mean(r))
                    Sv = np.where(bv, Sv, np.nan)
            S_v.append(Sv)
            # direzione PUV
            if "vel" in ds and okf.sum() >= 3:
                vel = ds.vel.values[sel]                    # (n, 1, beam)
                ve = to_earth(vel, a, ds["heading"].values[sel], ds["pitch"].values[sel],
                              ds["roll"].values[sel], orientation)[:, 0, :]
                uu = _interp_gaps(ve[:, 0].astype(float))
                vv_ = _interp_gaps(ve[:, 1].astype(float))
                if uu is not None and vv_ is not None:
                    if declination:
                        ang = np.deg2rad(declination)
                        uu, vv_ = uu * np.cos(ang) + vv_ * np.sin(ang), -uu * np.sin(ang) + vv_ * np.cos(ang)
                    kw = dict(fs=fs, nperseg=nseg, window="hann", noverlap=nseg // 2)
                    _, Cpu = signal.csd(signal.detrend(eta_p), signal.detrend(uu), **kw)
                    _, Cpv = signal.csd(signal.detrend(eta_p), signal.detrend(vv_), **kw)
                    Cpu, Cpv = Cpu.real, Cpv.real
                    ip = np.nanargmax(np.where(okf, See, -np.inf))
                    to_p = np.degrees(np.arctan2(Cpv[ip], Cpu[ip]))
                    to_m = np.degrees(np.arctan2(np.sum(Cpv[okf]), np.sum(Cpu[okf])))
                    row["Dp"] = (270.0 - to_p) % 360.0   # direzione di provenienza, da Nord orario
                    row["Dm"] = (270.0 - to_m) % 360.0
            rows.append(row)
        if verbose:
            print(f"  onde: gruppo {gi + 1}/{len(groups)} ({len(rows)} burst)", flush=True)

    keys = ["n", "duration_s", "depth", "depth_sensor", "p_std", "fcut", "Hm0", "Tp", "Tm01", "Tm02",
            "Hm0_vb", "Tp_vb", "Tm01_vb", "Tm02_vb", "surface_range_vb", "Dp", "Dm"]
    keys = [k for k in keys if any(k in r for r in rows)]
    data = {k: ("time", np.array([r.get(k, np.nan) for r in rows], float)) for k in keys}
    units = dict(depth="m", depth_sensor="m", p_std="dbar", fcut="Hz", Hm0="m", Tp="s", Tm01="s",
                 Tm02="s", Hm0_vb="m", Tp_vb="s", Tm01_vb="s", Tm02_vb="s", surface_range_vb="m",
                 Dp="degree", Dm="degree", duration_s="s")
    out = xr.Dataset(data, coords=dict(time=np.array([r["time"] for r in rows]), freq=freq))
    for k in keys:
        if k in units:
            out[k].attrs["units"] = units[k]
    out["S_p"] = (("time", "freq"), np.array(S_p), dict(units="m2 Hz-1",
                  long_name="spettro elevazione da pressione (corretto per attenuazione)"))
    if has_vb:
        out["S_vb"] = (("time", "freq"), np.array(S_v), dict(units="m2 Hz-1",
                       long_name="spettro elevazione da superficie (beam verticale)"))
    out.attrs.update(dict(fs=fs, hab=hab, p_atm=p_atm, nperseg=nseg, fmin=fmin, fmax=fmax,
                          kp2_min=kp2_min, puv_cell=-1 if puv_cell is None else puv_cell,
                          note="Dp/Dm: direzione di PROVENIENZA, gradi da Nord in senso orario"))
    return out
