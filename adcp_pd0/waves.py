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


def surface_from_vb(echo, rng, depth_est, window=0.35, min_prominence=10.0):
    """Distanza della superficie dal trasduttore per ogni ping dal massimo di eco del beam
    verticale (interpolazione parabolica). ``depth_est`` = profondità stimata del trasduttore.

    Il ping è scartato (NaN) se il massimo cade sul bordo della finestra di ricerca o delle
    celle disponibili, o se emerge meno di ``min_prominence`` counts dalla mediana del profilo.
    """
    echo = echo.astype(np.float32)
    n, nc = echo.shape
    lo, hi = depth_est * (1 - window), depth_est * (1 + window)
    m = (rng[None, :] >= np.atleast_1d(lo)[:, None]) & (rng[None, :] <= np.atleast_1d(hi)[:, None])
    e = np.where(m, echo, -np.inf)
    i = np.argmax(e, axis=1)
    ar = np.arange(n)
    i_c = np.clip(i, 1, nc - 2)
    inner = (i > 0) & (i < nc - 1) & m[ar, np.clip(i - 1, 0, nc - 1)] & m[ar, np.clip(i + 1, 0, nc - 1)]
    y0, y1, y2 = (echo[ar, i_c + k] for k in (-1, 0, 1))
    prom = y1 - np.median(echo, axis=1)
    ok = inner & m.any(axis=1) & (prom >= min_prominence)
    den = y0 - 2 * y1 + y2
    with np.errstate(all="ignore"):
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


def analyze_waves(reader, hab: float = 0.6, p_atm: float = 0.0, nperseg: int = 256,
                  fmin: float = 0.04, fmax: float = 0.6, kp2_min: float = 0.05,
                  min_samples: int = 256, use_vb: bool = True, p_min: float = 1.0, puv_cell: int | None = None,
                  orientation: str | None = None, declination: float = 0.0, dir_ref: str = "auto",
                  max_samples_per_load: int = 2_000_000, verbose: bool = True) -> xr.Dataset:
    """Statistiche d'onda per burst dallo stream onde.

    hab        : altezza del sensore di pressione/trasduttore sul fondo [m]
    p_atm      : offset da sottrarre alla pressione [dbar] (0 se il sensore è già azzerato in aria)
    p_min      : burst con pressione media < p_min [dbar] considerati fuori acqua e scartati
    kp2_min    : frequenza di taglio dove la funzione di trasferimento Kp^2 scende sotto questa soglia
    puv_cell   : indice (0-based) della cella da usare per la direzione (None = no)
    dir_ref    : riferimento per la direzione: "auto" (superficie dal beam verticale se c'è,
                 altrimenti pressione), "vb" o "p"
    """
    t_all = reader.times()
    bursts, dt_med = find_bursts(t_all)
    # i burst troppo corti per uno spettro (es. i burst da 180 ping per le sole correnti
    # del Sentinel V) vengono esclusi dall'analisi onde
    lengths = bursts[:, 1] - bursts[:, 0]
    n_short = int((lengths < min_samples).sum())
    bursts = bursts[lengths >= min_samples]
    if verbose and n_short:
        print(f"  onde: {len(bursts)} burst usati, {n_short} burst piu' corti di {min_samples} "
              f"ping esclusi")
    if len(bursts) == 0:
        raise ValueError(f"nessun burst con almeno {min_samples} ping (min_samples)")
    fs = 1.0 / dt_med
    a = reader.attrs
    nseg = int(min(nperseg, 2 ** np.floor(np.log2(max(min_samples, 16)))))
    freq = np.fft.rfftfreq(nseg, 1 / fs)
    vars_needed = ["pressure", "heading", "pitch", "roll"]
    has_vb = use_vb and "vb_echo" in reader.variables()
    if has_vb:
        vars_needed.append("vb_echo")
    if puv_cell is not None and "vel" in reader.variables():
        r0 = reader.load(["pressure", "range"], t_all[0], t_all[min(len(t_all) - 1, 2000)])
        rng_c = r0.range.values
        if puv_cell == "auto":
            # cella a ~70% della distanza dalla superficie: compromesso fra attenuazione
            # delle velocità orbitali e separazione orizzontale dei beam
            d_s = float(np.nanmean(r0.pressure.values) - p_atm) * 1e4 / (RHO * G)
            puv_cell = int(np.argmin(np.abs(rng_c - 0.7 * d_s)))
        puv_cell = int(puv_cell)
        if not 0 <= puv_cell < len(rng_c):
            raise ValueError(f"--puv-cell {puv_cell} fuori dall'intervallo 0..{len(rng_c) - 1}")
        if verbose:
            print(f"  direzione: cella {puv_cell} (a {rng_c[puv_cell]:.2f} m dal trasduttore)")
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
            out_of_water = float(np.nanmean(p)) < p_min
            p = _interp_gaps(p)
            if p is None or h <= 0.5 or out_of_water:
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
            eta_vb = None
            if has_vb:
                rng = ds.vb_range.values
                r = surface_from_vb(ds.vb_echo.values[sel], rng, np.full(n, depth_s))
                row["surface_found_frac"] = float(np.isfinite(r).mean())
                r = _interp_gaps(_despike(r), max_frac=0.15)
                if r is not None:
                    fv, Sv = signal.welch(signal.detrend(r), fs=fs, nperseg=nseg, window="hann",
                                          noverlap=nseg // 2)
                    bv = (fv >= fmin) & (fv <= min(fmax * 2, fs / 2))
                    for kk, vv in _moments(fv, Sv, bv).items():
                        row[kk + "_vb"] = vv
                    row["surface_range_vb"] = float(np.mean(r))
                    Sv = np.where(bv, Sv, np.nan)
                    eta_vb = r - r.mean()
            S_v.append(Sv)
            # direzione: cross-spettri elevazione-velocità orizzontale (PUV / "SUV").
            # Riferimento = superficie dal beam verticale se disponibile (non attenuata con la
            # profondità), altrimenti la pressione.
            if "vel" in ds:
                if eta_vb is not None and dir_ref in ("auto", "vb"):
                    ref, ref_name = eta_vb, "vb"
                    Sref = np.nan_to_num(Sv)
                    band = (f >= fmin) & (f <= fmax) & np.isfinite(Sv)
                else:
                    ref, ref_name = eta_p, "p"
                    Sref = np.nan_to_num(See)
                    band = okf
                vel = ds.vel.values[sel]                    # (n, 1, beam)
                ve = to_earth(vel, a, ds["heading"].values[sel], ds["pitch"].values[sel],
                              ds["roll"].values[sel], orientation)[:, 0, :]
                uu = _interp_gaps(ve[:, 0].astype(float))
                vv_ = _interp_gaps(ve[:, 1].astype(float))
                if uu is not None and vv_ is not None and band.sum() >= 3:
                    if declination:
                        ang = np.deg2rad(declination)
                        uu, vv_ = uu * np.cos(ang) + vv_ * np.sin(ang), -uu * np.sin(ang) + vv_ * np.cos(ang)
                    kw = dict(fs=fs, nperseg=nseg, window="hann", noverlap=nseg // 2)
                    _, Cpu = signal.csd(signal.detrend(ref), signal.detrend(uu), **kw)
                    _, Cpv = signal.csd(signal.detrend(ref), signal.detrend(vv_), **kw)
                    Cpu, Cpv = Cpu.real, Cpv.real
                    ip = int(np.argmax(np.where(band, Sref, -np.inf)))
                    to_p = np.degrees(np.arctan2(Cpv[ip], Cpu[ip]))
                    to_m = np.degrees(np.arctan2(np.sum(Cpv[band]), np.sum(Cpu[band])))
                    row["Dp"] = (270.0 - to_p) % 360.0   # direzione di provenienza, da Nord orario
                    row["Dm"] = (270.0 - to_m) % 360.0
                    row["dir_ref_vb"] = 1.0 if ref_name == "vb" else 0.0
            rows.append(row)
        if verbose:
            print(f"  onde: gruppo {gi + 1}/{len(groups)} ({len(rows)} burst)", flush=True)

    keys = ["n", "duration_s", "depth", "depth_sensor", "p_std", "fcut", "Hm0", "Tp", "Tm01", "Tm02",
            "Hm0_vb", "Tp_vb", "Tm01_vb", "Tm02_vb", "surface_range_vb", "surface_found_frac", "Dp", "Dm", "dir_ref_vb"]
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
    out.attrs.update({k: v for k, v in a.items() if k in ("serial", "fw_ver", "fw_rev", "source_file",
                                                          "orientation", "beam_angle_deg")})
    out.attrs.update(dict(n_short_bursts_skipped=n_short, fs=fs, hab=hab, p_atm=p_atm, nperseg=nseg, fmin=fmin, fmax=fmax,
                          kp2_min=kp2_min, puv_cell=-1 if puv_cell is None else puv_cell,
                          note="Dp/Dm: direzione di PROVENIENZA, gradi da Nord in senso orario"))
    return out
