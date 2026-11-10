"""Controllo qualità e prodotti di base per i profili di corrente."""
from __future__ import annotations

import warnings

import numpy as np
import xarray as xr

from .transform import to_earth, rotate_declination

RHO = 1025.0
G = 9.81


def qc_currents(ds: xr.Dataset, corr_min: int = 64, pg_min: float = 50.0, err_max: float = 0.20,
                tilt_max: float = 20.0, p_min: float = 1.0, sidelobe: bool = True,
                surface_margin_cells: float = 1.0, declination: float = 0.0,
                orientation: str | None = None, t_start=None, t_end=None,
                xducer_hab: float = 0.0) -> xr.Dataset:
    """QC dei profili di velocità.

    - taglio temporale (messa in acqua / recupero) e scarto ensemble con pressione < ``p_min``
    - eventuale trasformazione in coordinate Earth e rotazione per la declinazione
    - maschere: correlazione minima (su tutti i beam), percent good (3+4 beam), velocità
      d'errore, inclinazione, contaminazione side-lobe vicino alla superficie
    Restituisce un Dataset con u, v, w, err (m/s), speed, direction (verso cui va, °N),
    medie sulla verticale e i flag di QC.
    """
    a = dict(ds.attrs)
    if t_start is not None or t_end is not None:
        ds = ds.sel(time=slice(t_start, t_end))
    n0 = ds.sizes["time"]
    if "pressure" in ds and float(ds.pressure.max()) > 0:
        ds = ds.isel(time=np.flatnonzero(ds.pressure.values >= p_min))
    vel = ds.vel.values
    head, pitch, roll = ds["heading"].values, ds["pitch"].values, ds["roll"].values
    vel = to_earth(vel, a, head, pitch, roll, orientation)
    u, v, w = vel[..., 0], vel[..., 1], vel[..., 2]
    err = vel[..., 3] if vel.shape[-1] > 3 else np.full_like(u, np.nan)
    u, v = rotate_declination(u, v, declination)

    bad = ~np.isfinite(u) | ~np.isfinite(v)
    flags = {}
    if "corr" in ds and corr_min:
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
    for k in ("heading", "pitch", "roll", "temperature", "pressure"):
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
    out["valid_fraction"] = ("cell", np.isfinite(u).mean(axis=0).astype(np.float32))
    out.attrs.update(dict(qc_corr_min=corr_min, qc_pg_min=pg_min, qc_err_max=err_max,
                          qc_tilt_max=tilt_max, qc_p_min=p_min, declination=declination,
                          n_ensembles_raw=n0, coord_system_out="earth"))
    return out
