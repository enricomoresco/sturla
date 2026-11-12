"""Onde, in due passi.

PRE-PROCESSING  ``extract_wave_bursts``: dai ping di ogni burst lungo ricava le serie pulite
  (superficie tracciata con l'eco del beam verticale, pressione, velocità Est/Nord ad alcune
  quote). Nessuna analisi.
POST-PROCESSING ``wave_spectra``: dalle serie pulite calcola spettri, Hm0, periodi, direzione
  (cross-spettri superficie-velocità), velocità orbitale al fondo.

La direzione è una stima di primo livello: vicino alla superficie i 4 beam distano fra loro
quanto la lunghezza delle onde corte. Per i prodotti direzionali completi confrontare con il
software Teledyne (Velocity).
"""
from __future__ import annotations

import numpy as np
import xarray as xr
from scipy import signal

from .currents import RHO, G, _circmean_deg
from .transform import to_earth, rotate_declination


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
    if not np.isfinite(x).any():
        return x
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


def choose_levels(rng_c, depth_s, beam_angle_deg=25.0, cell_size=0.5, fractions=(0.5, 0.7, 0.85)):
    """Indici delle celle alle frazioni richieste della distanza trasduttore-superficie, sotto il
    limite di contaminazione della superficie; senza duplicati."""
    rng_c = np.asarray(rng_c, float)
    rmax = depth_s * np.cos(np.deg2rad(beam_angle_deg)) - cell_size
    ok = np.flatnonzero(rng_c <= rmax)
    if len(ok) == 0:
        ok = np.arange(len(rng_c))
    cells = []
    for fr in fractions:
        i = int(ok[np.argmin(np.abs(rng_c[ok] - fr * depth_s))])
        if i not in cells:
            cells.append(i)
    return sorted(cells)


def extract_wave_bursts(reader, hab: float = 0.6, p_atm: float = 0.0, min_samples: int = 256,
                        use_vb: bool = True, p_min: float = 1.0, fractions=(0.5, 0.7, 0.85),
                        orientation: str | None = None, declination: float = 0.0, corr_min: int = 64,
                        p_ref: float | None = None, max_samples: int = 4096,
                        max_samples_per_load: int = 1_000_000, verbose: bool = True) -> xr.Dataset:
    """Serie pulite dei burst d'onda (pre-processing), dimensioni ``time`` (inizio burst) x ``sample``.

    Per ogni burst con almeno ``min_samples`` ping:
      surface_range : distanza trasduttore-superficie dal massimo di eco del beam verticale (grezza)
      eta           : elevazione pulita (spike oltre 4 MAD rimossi, buchi < 15% interpolati, media tolta)
      pressure      : pressione (meno ``p_atm``)
      u, v          : velocità Est/Nord alle quote ``fractions`` della distanza dalla superficie
    I burst più corti (es. quelli da 180 ping per le sole correnti) sono esclusi; quelli più
    lunghi di ``max_samples`` ping (es. campionamento continuo) sono divisi in tratti consecutivi
    di ``max_samples`` ping, ognuno trattato come un burst.
    """
    t_all = reader.times()
    bursts, dt = find_bursts(t_all)
    if max_samples:
        bursts = np.array([[s2, min(s2 + max_samples, e)] for s, e in bursts for s2 in range(s, e, max_samples)])
    lengths = bursts[:, 1] - bursts[:, 0]
    n_short = int((lengths < min_samples).sum())
    bursts = bursts[lengths >= min_samples]
    if len(bursts) == 0:
        raise ValueError(f"nessun burst con almeno {min_samples} ping")
    if verbose:
        print(f"  burst d'onda: {len(bursts)} con almeno {min_samples} ping"
              + (f"; {n_short} piu' corti esclusi" if n_short else ""))
    a = dict(reader.attrs)
    variables = reader.variables()
    has_vb = use_vb and "vb_echo" in variables
    cells = []
    if "vel" in variables:
        if p_ref is None:
            pw = reader.load(["pressure"]).pressure.values.astype(float) - p_atm
            pw = pw[pw >= p_min]
            p_ref = float(np.median(pw)) if pw.size else np.nan
        d_ref = p_ref * 1e4 / (RHO * G) if p_ref is not None else np.nan
        rng_c = reader.coord("range")
        if np.isfinite(d_ref) and d_ref > 1.0:
            cells = choose_levels(rng_c, d_ref, a.get("beam_angle_deg", 25), a.get("cell_size_cm", 100) / 100.0,
                                  fractions)
            if verbose:
                print("  quote per la direzione: " + ", ".join(
                    f"cella {c} a {rng_c[c]:.2f} m dal trasduttore" for c in cells))
    nlev, nb = len(cells), len(bursts)
    nmax = int((bursts[:, 1] - bursts[:, 0]).max())
    nan2 = lambda *shape: np.full(shape, np.nan, np.float32)
    P = nan2(nb, nmax)
    SR, ETA = (nan2(nb, nmax), nan2(nb, nmax)) if has_vb else (None, None)
    Uo, Vo = (nan2(nb, nmax, nlev), nan2(nb, nmax, nlev)) if nlev else (None, None)
    nsam = np.zeros(nb, np.int32)
    d_sens, found, sr_mean = (np.full(nb, np.nan) for _ in range(3))
    head, pit, rol = (np.full(nb, np.nan) for _ in range(3))
    corr_ok = np.full((nb, nlev), np.nan, np.float32)
    names = ["pressure", "heading", "pitch", "roll"] + (["vb_echo"] if has_vb else []) \
        + (["vel", "corr"] if nlev else [])
    names = [v for v in names if v in variables or v == "pressure"]
    groups, cur, n_cur = [], [], 0
    for k, b in enumerate(bursts):
        cur.append((k, b))
        n_cur += b[1] - b[0]
        if n_cur >= max_samples_per_load:
            groups.append(cur)
            cur, n_cur = [], 0
    if cur:
        groups.append(cur)
    for gi, grp in enumerate(groups):
        t0 = t_all[grp[0][1][0]]
        t1 = t_all[grp[-1][1][1] - 1] + np.timedelta64(1, "us")
        ds = reader.load(names, t0, t1, isel={"cell": cells} if nlev else None)
        tt = ds.time.values
        vb_rng = ds.vb_range.values if has_vb else None
        for k, (s, e) in grp:
            i0 = int(np.searchsorted(tt, t_all[s], "left"))
            i1 = int(np.searchsorted(tt, t_all[e - 1], "right"))
            n = i1 - i0
            nsam[k] = n
            if n < min_samples:
                continue
            sl = slice(i0, i1)
            p = ds.pressure.values[sl].astype(float) - p_atm
            P[k, :n] = p
            d_s = float(np.nanmean(p)) * 1e4 / (RHO * G)
            d_sens[k] = d_s
            hh, pp, rr = (ds[v].values[sl] for v in ("heading", "pitch", "roll"))
            head[k], pit[k], rol[k] = _circmean_deg(hh), np.nanmean(pp), np.nanmean(rr)
            if has_vb and d_s > 0.5:
                r = surface_from_vb(ds.vb_echo.values[sl], vb_rng, np.full(n, d_s))
                SR[k, :n] = r
                found[k] = float(np.isfinite(r).mean())
                rc = _interp_gaps(_despike(r), max_frac=0.15)
                if rc is not None:
                    sr_mean[k] = float(np.mean(rc))
                    ETA[k, :n] = rc - np.mean(rc)
            if nlev:
                ve = to_earth(ds.vel.values[sl], a, hh, pp, rr, orientation)
                uu, vv = rotate_declination(ve[..., 0], ve[..., 1], declination)
                Uo[k, :n, :], Vo[k, :n, :] = uu, vv
                if "corr" in ds:
                    corr_ok[k] = (ds.corr.values[sl].min(axis=-1) >= corr_min).mean(axis=0)
        if verbose:
            print(f"  burst d'onda: gruppo {gi + 1}/{len(groups)}", flush=True)

    out = xr.Dataset(coords=dict(
        time=("time", t_all[bursts[:, 0]]),
        sample=("sample", np.arange(nmax, dtype=np.int32)),
        sample_time=("sample", (np.arange(nmax) * dt).astype(np.float32))))
    out["pressure"] = (("time", "sample"), P)
    if has_vb:
        out["surface_range"] = (("time", "sample"), SR)
        out["eta"] = (("time", "sample"), ETA)
        out["surface_found_frac"] = ("time", found.astype(np.float32))
        out["surface_range_mean"] = ("time", sr_mean.astype(np.float32))
    if nlev:
        rng_c = reader.coord("range")
        out = out.assign_coords(level=("level", np.arange(nlev, dtype=np.int32)),
                                level_range=("level", rng_c[cells].astype(np.float32)),
                                level_cell=("level", np.asarray(cells, np.int32)))
        out["u"] = (("time", "sample", "level"), Uo)
        out["v"] = (("time", "sample", "level"), Vo)
        out["corr_ok_frac"] = (("time", "level"), corr_ok)
    out["n_samples"] = ("time", nsam)
    out["depth_sensor"] = ("time", d_sens.astype(np.float32))
    out["water_depth"] = ("time", (d_sens + hab).astype(np.float32))
    out["heading"] = ("time", head.astype(np.float32))
    out["pitch"] = ("time", pit.astype(np.float32))
    out["roll"] = ("time", rol.astype(np.float32))
    out.attrs.update({k: v for k, v in a.items() if k in (
        "serial", "fw_ver", "fw_rev", "source_file", "orientation", "beam_angle_deg", "coord_system",
        "n_cells", "cell_size_cm", "bin1_dist_cm", "blank_cm", "heading_bias", "convex")})
    out.attrs.update(dict(n_short_bursts_skipped=n_short, dt=float(dt), fs=float(1.0 / dt), hab=float(hab),
                          p_atm=float(p_atm), burst_min_samples=int(min_samples), declination=float(declination),
                          burst_p_ref_dbar=float(p_ref) if p_ref is not None and np.isfinite(p_ref) else -1.0,
                          burst_corr_min=int(corr_min), burst_max_samples=int(max_samples or 0),
                          has_vertical_beam=int(has_vb)))
    return out


def _burst_dt(b) -> float:
    st = b["sample_time"].values
    return float(st[1] - st[0]) if len(st) > 1 else float(b.attrs.get("dt", 0.5))


def wave_spectra(b: xr.Dataset, nperseg: int = 256, fmin: float = 0.04, fmax: float = 0.6,
                 kp2_min: float = 0.05, hm0_dir_min: float = 0.25, coh_min: float = 0.3,
                 main_fraction: float = 0.7, hab: float | None = None) -> xr.Dataset:
    """Spettri e parametri d'onda per burst (post-processing) dalle serie pulite ``b``.

    - spettro di Welch dell'elevazione (finestra di Hann, 50% di sovrapposizione);
      Hm0 = 4 sqrt(m0), Tp (picco con interpolazione parabolica), Tm01, Tm02 fra ``fmin`` e
      min(2 fmax, Nyquist);
    - se manca il beam verticale: spettro da pressione corretto con la teoria lineare fino a
      Kp^2 >= ``kp2_min``; con il beam verticale la pressione è solo un confronto;
    - direzione di PROVENIENZA (gradi da Nord, orario) dai cross-spettri elevazione-velocità
      a ogni quota: Dp al picco, Dm sulla banda, e direzione per frequenza alla quota principale;
    - coerenza elevazione-velocità al picco (qualità della direzione);
    - Ub: ampiezza significativa della velocità orbitale al fondo dalla teoria lineare
      (2 volte la deviazione standard).
    """
    from .archive import combine_flags, WAVE_CAUSES, DIR_CAUSES

    dt = _burst_dt(b)
    fs = 1.0 / dt
    nb = b.sizes["time"]
    n_s = b["n_samples"].values.astype(int)
    if hab is None:
        hab = float(b.attrs.get("instrument_height_above_bed_m", b.attrs.get("hab", 0.6)))
    has_eta = "eta" in b
    nlev = b.sizes.get("level", 0) if "u" in b else 0
    usable = np.isin(b["burst_qc"].values, (1, 3)) if "burst_qc" in b else np.ones(nb, bool)
    n_ok = n_s[usable & (n_s > 0)]
    nseg = int(min(nperseg, 2 ** np.floor(np.log2(max(n_ok.min() if n_ok.size else nperseg, 16)))))
    freq = np.fft.rfftfreq(nseg, dt)
    nf = len(freq)
    kw = dict(fs=fs, nperseg=nseg, noverlap=nseg // 2, window="hann")
    band_m = (freq >= fmin) & (freq <= min(2 * fmax, fs / 2))      # banda dei momenti (elevazione)
    band_d = (freq >= fmin) & (freq <= fmax)                       # banda della direzione
    lev_main = 0
    if nlev:
        d_ref = float(np.nanmedian(b["depth_sensor"].values[usable])) if usable.any() else np.nan
        frac = b["level_range"].values / d_ref if np.isfinite(d_ref) and d_ref > 0 else np.zeros(nlev)
        lev_main = int(np.argmin(np.abs(frac - main_fraction)))

    nanv = lambda *shape: np.full(shape, np.nan)
    S, Sp, TH = nanv(nb, nf), nanv(nb, nf), nanv(nb, nf)
    par = {k: nanv(nb) for k in ("Hm0", "Tp", "Tm01", "Tm02", "Hm0_pressure", "Tp_pressure", "fcut_pressure",
                                 "Ub", "Dp", "Dm", "dir_coherence", "dir_level_spread")}
    Dpl, Dml = nanv(nb, max(nlev, 1)), nanv(nb, max(nlev, 1))
    P = b["pressure"].values
    E = b["eta"].values if has_eta else None
    U = b["u"].values if nlev else None
    V = b["v"].values if nlev else None
    depth = b["water_depth"].values
    for k in range(nb):
        n = int(n_s[k])
        h = float(depth[k])
        if not usable[k] or n < nseg or not np.isfinite(h) or h <= 0.5:
            continue
        kk = wavenumber(freq, h)
        ref = Sref = None
        # pressione -> elevazione con la teoria lineare
        pc = _interp_gaps(P[k, :n].astype(float))
        if pc is not None:
            eta_p = (pc - pc.mean()) * 1e4 / (RHO * G)
            _, Spp = signal.welch(signal.detrend(eta_p), **kw)
            Kp2 = (np.cosh(kk * hab) / np.cosh(kk * h)) ** 2
            okf = band_d & (Kp2 >= kp2_min)
            if okf.sum() >= 3:
                See = np.where(okf, Spp / np.maximum(Kp2, 1e-12), np.nan)
                Sp[k] = See
                mp = _moments(freq, np.nan_to_num(See), okf)
                par["Hm0_pressure"][k], par["Tp_pressure"][k] = mp["Hm0"], mp["Tp"]
                par["fcut_pressure"][k] = float(freq[okf].max())
                if not has_eta:
                    S[k] = See
                    for q in ("Hm0", "Tp", "Tm01", "Tm02"):
                        par[q][k] = mp[q]
                    ref, Sref, bnd = eta_p, np.nan_to_num(See), okf
        # superficie dal beam verticale
        if has_eta:
            e = E[k, :n].astype(float)
            if np.isfinite(e).all():
                _, Sv = signal.welch(signal.detrend(e), **kw)
                S[k] = np.where(band_m, Sv, np.nan)
                for q, val in _moments(freq, Sv, band_m).items():
                    par[q][k] = val
                ref, Sref, bnd = e, Sv, band_d
        if ref is None:
            continue
        # velocità orbitale al fondo (teoria lineare)
        ok_b = np.isfinite(S[k])
        if ok_b.sum() >= 3:
            tr = (2 * np.pi * freq[ok_b] / np.sinh(np.minimum(kk[ok_b] * h, 50.0))) ** 2
            par["Ub"][k] = 2.0 * np.sqrt(np.trapezoid(S[k][ok_b] * tr, freq[ok_b]))
        # direzione
        if nlev and bnd.sum() >= 3:
            refd = signal.detrend(ref)
            _, Pxx = signal.welch(refd, **kw)
            ip = int(np.argmax(np.where(bnd, Sref, -np.inf)))
            for l in range(nlev):
                uu = _interp_gaps(U[k, :n, l].astype(float), max_frac=0.3)
                vv = _interp_gaps(V[k, :n, l].astype(float), max_frac=0.3)
                if uu is None or vv is None:
                    continue
                ud, vd = signal.detrend(uu), signal.detrend(vv)
                _, Pxu = signal.csd(refd, ud, **kw)
                _, Pxv = signal.csd(refd, vd, **kw)
                Dpl[k, l] = (270.0 - np.degrees(np.arctan2(Pxv.real[ip], Pxu.real[ip]))) % 360.0
                Dml[k, l] = (270.0 - np.degrees(np.arctan2(Pxv.real[bnd].sum(), Pxu.real[bnd].sum()))) % 360.0
                if l == lev_main:
                    TH[k] = np.where(bnd, (270.0 - np.degrees(np.arctan2(Pxv.real, Pxu.real))) % 360.0, np.nan)
                    _, Puu = signal.welch(ud, **kw)
                    _, Pvv = signal.welch(vd, **kw)
                    with np.errstate(all="ignore"):
                        coh = (np.abs(Pxu) ** 2 + np.abs(Pxv) ** 2) / (Pxx * (Puu + Pvv))
                    pk = bnd & (np.abs(freq - freq[ip]) <= 0.025)
                    if pk.any() and np.sum(Pxx[pk]) > 0:
                        par["dir_coherence"][k] = float(np.sum(coh[pk] * Pxx[pk]) / np.sum(Pxx[pk]))
            par["Dp"][k], par["Dm"][k] = Dpl[k, lev_main], Dml[k, lev_main]
            fin = np.isfinite(Dpl[k, :nlev])
            if fin.sum() >= 2:
                d = np.deg2rad(Dpl[k, :nlev][fin])
                diff = np.abs((np.degrees(d[:, None] - d[None, :]) + 180) % 360 - 180)
                par["dir_level_spread"][k] = float(diff.max())

    out = xr.Dataset(coords=dict(time=("time", b.time.values), frequency=("frequency", freq.astype(np.float32))))
    for q in ("Hm0", "Tp", "Tm01", "Tm02", "Ub"):
        out[q] = ("time", par[q].astype(np.float32))
    out["S"] = (("time", "frequency"), S.astype(np.float32))
    if has_eta:
        out["Hm0_pressure"] = ("time", par["Hm0_pressure"].astype(np.float32))
        out["fcut_pressure"] = ("time", par["fcut_pressure"].astype(np.float32))
        out["S_pressure"] = (("time", "frequency"), Sp.astype(np.float32))
    if nlev:
        out = out.assign_coords(level=("level", b["level"].values), level_range=("level", b["level_range"].values))
        for q in ("Dp", "Dm", "dir_coherence", "dir_level_spread"):
            out[q] = ("time", par[q].astype(np.float32))
        out["Dp_level"] = (("time", "level"), Dpl[:, :nlev].astype(np.float32))
        out["Dm_level"] = (("time", "level"), Dml[:, :nlev].astype(np.float32))
        out["direction_f"] = (("time", "frequency"), TH.astype(np.float32))
    for q in ("water_depth", "n_samples", "surface_found_frac"):
        if q in b:
            out[q] = ("time", b[q].values)
    # flag con causa
    inpos = b["in_position"].values.astype(bool) if "in_position" in b else np.ones(nb, bool)
    mot = b["platform_motion"].values.astype(bool) if "platform_motion" in b else np.zeros(nb, bool)
    sf = b["surface_found_frac"].values if "surface_found_frac" in b else np.ones(nb)
    hm = par["Hm0"]
    flag, cause = combine_flags((nb,), {
        "fuori_posizione": ~inpos, "movimento_telaio": mot & inpos,
        "superficie_trovata_parzialmente": inpos & np.isfinite(hm) & (np.nan_to_num(sf, nan=1.0) < 0.9),
        "nessun_dato": inpos & ~np.isfinite(hm)}, WAVE_CAUSES)
    out["wave_qc"], out["wave_qc_cause"] = ("time", flag), ("time", cause)
    out["in_position"] = ("time", inpos.astype(np.int8))
    out["platform_motion"] = ("time", mot.astype(np.int8))
    if nlev:
        dok = np.isfinite(par["Dp"])
        dflag, dcause = combine_flags((nb,), {
            "fuori_posizione": ~inpos, "movimento_telaio": mot & inpos,
            "bassa_energia": inpos & dok & (np.nan_to_num(hm) < hm0_dir_min),
            "bassa_coerenza": inpos & dok & (np.nan_to_num(par["dir_coherence"], nan=1.0) < coh_min),
            "nessun_dato": inpos & ~dok}, DIR_CAUSES)
        out["direction_qc"], out["direction_qc_cause"] = ("time", dflag), ("time", dcause)
    out.attrs.update(dict(
        wave_method=("superficie tracciata con l'eco del beam verticale" if has_eta else
                     "pressione corretta con la teoria lineare"),
        nperseg=int(nseg), fs=float(fs), fmin=float(fmin), fmax=float(fmax), kp2_min=float(kp2_min),
        hm0_dir_min=float(hm0_dir_min), coh_min=float(coh_min), hab=float(hab),
        direction_main_level=int(lev_main)))
    return out
