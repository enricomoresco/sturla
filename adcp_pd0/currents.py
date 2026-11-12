"""Controllo qualità e prodotti di base per i profili di corrente."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import xarray as xr

from .transform import to_earth, rotate_declination

RHO = 1025.0
G = 9.81


def qc_currents(ds: xr.Dataset, corr_min: int = 64, pg_min: float = 50.0, err_max: float = 0.20,
                tilt_max: float = 20.0, p_min: float = 1.0, sidelobe: bool = True,
                surface_margin_cells: float = 1.0, declination: float = 0.0,
                orientation: str | None = None, deployment: dict | None = None, trim: bool = False,
                xducer_hab: float = 0.0) -> xr.Dataset:
    """QC dei profili di velocità.

    - fuori dal periodo in posizione (``deployment``, vedi deployment.py) e con pressione
      < ``p_min``: tutte le celle marcate cattive (flag_position); con ``trim`` eliminate
    - eventuale trasformazione in coordinate Earth e rotazione per la declinazione
    - maschere: correlazione minima (su tutti i beam), percent good (3+4 beam), velocità
      d'errore, inclinazione, contaminazione side-lobe vicino alla superficie
    Restituisce un Dataset con u, v, w, err (m/s), speed, direction (verso cui va, °N),
    medie sulla verticale e i flag di QC.
    """
    a = dict(ds.attrs)
    n0 = ds.sizes["time"]
    from .deployment import in_position
    pos = np.ones(n0, bool)
    if "in_position_frac" in ds:
        # medie dei ping già limitate ai ping in posizione: l'intervallo è valido se ne ha
        pos &= np.nan_to_num(ds.in_position_frac.values) > 0
    else:
        if deployment is not None:
            pos &= in_position(ds.time.values, deployment)
        if "pressure" in ds and float(ds.pressure.max()) > 0:
            pos &= ds.pressure.values >= p_min
    if trim:
        ds = ds.isel(time=np.flatnonzero(pos))
        pos = pos[pos]
    vel = ds.vel.values
    head, pitch, roll = ds["heading"].values, ds["pitch"].values, ds["roll"].values
    vel = to_earth(vel, a, head, pitch, roll, orientation)
    u, v, w = vel[..., 0], vel[..., 1], vel[..., 2]
    err = vel[..., 3] if vel.shape[-1] > 3 else np.full_like(u, np.nan)
    u, v = rotate_declination(u, v, declination)

    bad = ~np.isfinite(u) | ~np.isfinite(v)
    flags = {}
    if "corr" in ds and corr_min and a.get("averaged_from") != "single pings":
        f = ds.corr.values.min(axis=-1) < corr_min
        flags["flag_corr"] = f
        bad |= f
    if "pg" in ds and pg_min and a.get("coord_system") == "earth":
        pg = ds.pg.values.astype(np.int16)
        good = pg[..., 0] + pg[..., 3] if pg.shape[-1] >= 4 else pg[..., 0]
        f = good < pg_min
        flags["flag_pg"] = f
        bad |= f
    if err_max:
        f = np.abs(err) > err_max
        flags["flag_err"] = f
        bad |= f
    f = np.broadcast_to((~pos)[:, None], u.shape)
    flags["flag_position"] = f
    bad |= f
    tilt = np.hypot(pitch, roll)
    f = np.broadcast_to((tilt > tilt_max)[:, None], u.shape)
    flags["flag_tilt"] = f
    bad |= f

    rng = ds.range.values
    cs = a.get("cell_size_cm", 100) / 100.0
    ori = orientation or a.get("orientation", "up")
    coords = dict(time=ds.time.values, range=("cell", rng), cell=ds.cell.values)
    depth = None
    if "pressure" in ds and float(ds.pressure.max()) > 0:
        depth = ds.pressure.values * 1e4 / (RHO * G)          # profondità del sensore [m]
    elif float(ds.depth_xducer.max()) > 0:
        depth = ds.depth_xducer.values
    if depth is not None:
        coords["depth_xducer"] = ("time", depth.astype(np.float32))
    if sidelobe and ori == "up" and depth is not None:
        th = np.deg2rad(a.get("beam_angle_deg", 25))
        rmax = depth * np.cos(th) - surface_margin_cells * cs
        f = rng[None, :] > rmax[:, None]
        flags["flag_sidelobe"] = f
        bad |= f
    u = np.where(bad, np.nan, u).astype(np.float32)
    v = np.where(bad, np.nan, v).astype(np.float32)
    w = np.where(bad, np.nan, w).astype(np.float32)
    err = np.where(bad, np.nan, err).astype(np.float32)

    spd = np.hypot(u, v)
    dirn = np.mod(np.degrees(np.arctan2(u, v)), 360.0)
    dims = ("time", "cell")
    out = xr.Dataset(
        dict(u=(dims, u, dict(units="m s-1", long_name="velocita' Est")),
             v=(dims, v, dict(units="m s-1", long_name="velocita' Nord")),
             w=(dims, w, dict(units="m s-1", long_name="velocita' verticale")),
             err=(dims, err, dict(units="m s-1", long_name="velocita' d'errore")),
             speed=(dims, spd, dict(units="m s-1", long_name="intensita' corrente")),
             direction=(dims, dirn.astype(np.float32),
                        dict(units="degree", long_name="direzione (verso cui va), da Nord orario"))),
        coords=coords, attrs=a)
    if "echo" in ds:
        out["echo_mean"] = (dims, ds.echo.values.mean(axis=-1).astype(np.float32),
                            dict(units="counts", long_name="intensita' eco media sui beam"))
    for k in ("heading", "pitch", "roll", "temperature", "pressure", "pressure_std", "n_pings",
              "good_ping_frac", "in_position_frac"):
        if k in ds:
            out[k] = ds[k]
    for k, f in flags.items():
        out[k] = (dims, np.asarray(f, np.uint8))
    # medie sulla verticale (solo celle valide)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        ub, vb = np.nanmean(u, axis=1), np.nanmean(v, axis=1)
    out["u_davg"] = ("time", ub.astype(np.float32), dict(units="m s-1", long_name="u media sulla verticale"))
    out["v_davg"] = ("time", vb.astype(np.float32), dict(units="m s-1", long_name="v media sulla verticale"))
    out["speed_davg"] = ("time", np.hypot(ub, vb).astype(np.float32), dict(units="m s-1"))
    out["dir_davg"] = ("time", np.mod(np.degrees(np.arctan2(ub, vb)), 360).astype(np.float32),
                       dict(units="degree"))
    out["in_position"] = ("time", pos.astype(np.uint8), dict(long_name="1 = strumento in posizione sul fondo"))
    # movimento del telaio: intervallo di media che si sovrappone a un evento -> sospetto
    mot = np.zeros(out.sizes["time"], bool)
    if deployment is not None and deployment.get("motion_events"):
        from .deployment import motion_overlap
        half = np.timedelta64(int(round(a.get("avg_bin_minutes", 0) * 30)), "s")
        tt = out.time.values
        mot = motion_overlap(tt - half, tt + half, deployment)
    out["flag_motion"] = ("time", mot.astype(np.uint8), dict(long_name="1 = telaio in movimento nell'intervallo"))
    out["valid_fraction"] = ("cell", np.isfinite(u[pos]).mean(axis=0).astype(np.float32) if pos.any()
                             else np.zeros(u.shape[1], np.float32))
    out.attrs.update(dict(qc_corr_min=corr_min, qc_pg_min=pg_min, qc_err_max=err_max,
                          qc_tilt_max=tilt_max, qc_p_min=p_min, declination=declination,
                          n_ensembles_raw=n0, coord_system_out="earth"))
    if deployment is not None:
        out.attrs.update(deployment_start=deployment["start"], deployment_end=deployment["end"],
                         deployment_method=deployment.get("method", ""))
    return out


def _circmean_deg(x, axis=0):
    r = np.deg2rad(x)
    return np.mod(np.degrees(np.arctan2(np.nanmean(np.sin(r), axis=axis),
                                        np.nanmean(np.cos(r), axis=axis))), 360)


def average_pings(reader, window_min: float | None = None, corr_min: int = 64, bin_minutes: float | None = None,
                  min_good_frac: float = 0.5, orientation: str | None = None,
                  max_samples_per_load: int = 200_000, check_vb: bool = True,
                  deployment: dict | None = None, p_min: float = 1.0,
                  verbose: bool = True) -> xr.Dataset:
    """Profili di corrente medi da uno stream di ping singoli (es. burst d'onda a 2 Hz).

    Ogni ping è portato in coordinate Earth con il proprio heading/pitch/roll, i ping con
    correlazione < ``corr_min`` su qualche beam sono scartati, poi si media:
      - con ``bin_minutes`` (es. 60): su intervalli regolari di orologio (tutti i ping
        dell'intervallo, anche di burst diversi); tempo = centro dell'intervallo; gli
        intervalli senza dati diventano righe vuote (n_pings = 0), così la serie è regolare;
      - altrimenti su ogni burst (o su finestre di ``window_min`` minuti dentro il burst).
    Le celle con meno di
    ``min_good_frac`` ping validi sono NaN. Il risultato ha la stessa struttura di uno
    stream di correnti in coordinate Earth e può essere passato a ``qc_currents``.
    """
    from .transform import to_earth
    from .waves import find_bursts

    a = dict(reader.attrs)
    t_all = reader.times()
    bursts, dt = find_bursts(t_all)
    labels = None
    if bin_minutes:
        step = np.timedelta64(int(round(bin_minutes * 60)), "s")
        tb = pd.DatetimeIndex(t_all).floor(f"{int(round(bin_minutes * 60))}s").values
        brk = np.flatnonzero(tb[1:] != tb[:-1]) + 1
        starts, ends = np.r_[0, brk], np.r_[brk, len(t_all)]
        bursts = np.stack([starts, ends], axis=1)
        labels = {int(s_): tb[s_] + step / 2 for s_ in starts}
        grid = np.arange(tb[0], tb[-1] + step, step) + step / 2
    elif window_min:
        w = max(1, int(round(window_min * 60 / dt)))
        bursts = np.array([[s, min(s + w, e)] for s, e in bursts for s in range(s, e, w)])
    groups, cur, n_cur = [], [], 0
    for b in bursts:
        cur.append(b)
        n_cur += b[1] - b[0]
        if n_cur >= max_samples_per_load:
            groups.append(cur)
            cur, n_cur = [], 0
    if cur:
        groups.append(cur)
    names = ["vel", "corr", "echo", "heading", "pitch", "roll", "temperature", "pressure"]
    if check_vb:
        names.append("vb_vel")
    out = {k: [] for k in ("time", "vel", "corr", "echo", "good", "heading", "pitch", "roll",
                           "temperature", "pressure", "pressure_std", "n_pings", "pos_frac")}
    vbstat = []
    rng = None
    for gi, grp in enumerate(groups):
        t0 = t_all[grp[0][0]]
        t1 = t_all[grp[-1][1] - 1] + np.timedelta64(1, "us")
        ds = reader.load(names, t0, t1)
        if rng is None:
            rng = ds.range.values
        tt = ds.time.values
        for s, e in grp:
            sel = np.flatnonzero((tt >= t_all[s]) & (tt <= t_all[e - 1]))
            if len(sel) == 0:
                continue
            # solo i ping in posizione, se ce ne sono (ore a cavallo di inizio/fine deployment)
            pos = np.ones(len(sel), bool)
            if deployment is not None:
                from .deployment import in_position
                pos &= in_position(tt[sel], deployment)
            if "pressure" in ds:
                pos &= ds.pressure.values[sel] >= p_min
            pos_frac = float(pos.mean())
            if pos.any():
                sel = sel[pos]
            h, p, r = (ds[k].values[sel] for k in ("heading", "pitch", "roll"))
            ve = to_earth(ds.vel.values[sel], a, h, p, r, orientation)
            if "corr" in ds and corr_min:
                ve = np.where((ds.corr.values[sel].min(axis=-1) < corr_min)[..., None], np.nan, ve)
            good = np.isfinite(ve[..., 0]).mean(axis=0)
            with np.errstate(all="ignore"), warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                m = np.nanmean(ve, axis=0)
                m[good < min_good_frac] = np.nan
                out["vel"].append(m.astype(np.float32))
                out["corr"].append(ds.corr.values[sel].mean(axis=0).astype(np.float32))
                out["echo"].append(ds.echo.values[sel].mean(axis=0).astype(np.float32))
                out["good"].append(good.astype(np.float32))
                out["heading"].append(_circmean_deg(h))
                for k in ("pitch", "roll", "temperature", "pressure"):
                    out[k].append(np.nanmean(ds[k].values[sel]) if k in ds else np.nan)
                out["pressure_std"].append(np.nanstd(ds.pressure.values[sel]) if "pressure" in ds else np.nan)
            out["time"].append(labels[int(s)] if labels is not None else t_all[s] + (t_all[e - 1] - t_all[s]) / 2)
            out["n_pings"].append(len(sel))
            out["pos_frac"].append(pos_frac)
            # verifica della trasformazione: w dai 4 beam contro il beam verticale, alla
            # cella a ~70% della distanza dalla superficie (dove le velocità orbitali sono
            # apprezzabili). Si salva per ogni burst; alla fine si usano i più energetici.
            if check_vb and "vb_vel" in ds and "vb_range" in ds and "pressure" in ds:
                d_s = float(np.nanmean(ds.pressure.values[sel])) * 1e4 / (RHO * G)
                ic = int(np.argmin(np.abs(rng - 0.7 * d_s)))
                iv = int(np.argmin(np.abs(ds.vb_range.values - rng[ic])))
                x, y = ve[:, ic, 2], ds.vb_vel.values[sel, iv]
                ok = np.isfinite(x) & np.isfinite(y)
                if ok.sum() > 100 and np.std(x[ok]) > 0 and np.std(y[ok]) > 0:
                    vbstat.append((np.corrcoef(x[ok], y[ok])[0, 1], float(np.std(y[ok])),
                                   float(np.std(x[ok]))))
        if verbose:
            print(f"  medie ping: gruppo {gi + 1}/{len(groups)} ({len(out['time'])} profili)", flush=True)
    attrs = dict(a)
    attrs.update(coord_system_recorded=a.get("coord_system", ""))
    attrs.update(coord_system="earth", averaged_from="single pings", avg_window_min=window_min or 0,
                 avg_corr_min=corr_min, avg_min_good_frac=min_good_frac)
    attrs.pop("transformation_matrix", None)
    if vbstat:
        st = np.array(vbstat)
        top = st[np.argsort(st[:, 1])[::-1][:max(5, len(st) // 10)]]   # burst più energetici
        attrs["corr_w4beam_vs_vbeam"] = float(np.median(top[:, 0]))
        attrs["corr_w4beam_vs_vbeam_all"] = float(np.median(st[:, 0]))
        attrs["std_vbeam_top_bursts"] = float(np.median(top[:, 1]))
        attrs["std_w4beam_top_bursts"] = float(np.median(top[:, 2]))
        attrs["n_bursts_check"] = int(len(top))
    dims = ("time", "cell", "beam")
    nb = out["vel"][0].shape[-1]
    res = xr.Dataset(
        dict(vel=(dims, np.array(out["vel"])), corr=(dims, np.array(out["corr"])),
             echo=(dims, np.array(out["echo"])),
             good_ping_frac=(("time", "cell"), np.array(out["good"])),
             **{k: ("time", np.array(out[k], np.float32)) for k in
                ("heading", "pitch", "roll", "temperature", "pressure", "pressure_std")},
             n_pings=("time", np.array(out["n_pings"])),
             in_position_frac=("time", np.array(out["pos_frac"], np.float32),
                               dict(long_name="frazione di ping in posizione nell'intervallo"))),
        coords=dict(time=np.array(out["time"]), cell=np.arange(1, len(rng) + 1),
                    range=("cell", rng), beam=np.arange(1, nb + 1)),
        attrs=attrs)
    if bin_minutes:
        res = res.reindex(time=grid)
        res["n_pings"] = res.n_pings.fillna(0).astype(np.int32)
        res.attrs.update(avg_bin_minutes=float(bin_minutes),
                         avg_time_label="centro dell'intervallo di media")
    res["depth_xducer"] = res.pressure * 1e4 / (RHO * G)
    return res
