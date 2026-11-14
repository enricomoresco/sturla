"""Forzanti orari alla stazione: flussi di calore e di galleggiamento (ERA5), sforzo del vento
(CMEMS), onde e deriva di Stokes (ADCP), profondita' del mixed layer (MedFS).

Uscita: forzanti_orari.csv, forzanti_giornalieri.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # anche con python -I

import os

import numpy as np
import pandas as pd
import xarray as xr

import comune as C


def _kk(fv, hh):
    om = 2 * np.pi * fv; k = om ** 2 / C.g
    for _ in range(40):
        k = k - (C.g * k * np.tanh(k * hh) - om ** 2) / (C.g * np.tanh(k * hh) + C.g * k * hh / np.cosh(k * hh) ** 2)
    return k


def stokes_superficie(waves: xr.Dataset) -> pd.DataFrame:
    """Hm0, periodi, deriva di Stokes in superficie (fondale finito) e velocita' orbitale al fondo."""
    fr = waves.frequency.values.astype(float); Sp = waves.S.values.astype(float); hh = waves.water_depth.values.astype(float)
    good = np.isin(waves.wave_qc.values, (1, 3))
    us = np.full(Sp.shape[0], np.nan)
    band = (fr >= 0.04) & (fr <= 0.5)
    for i in range(Sp.shape[0]):
        if not good[i] or not np.isfinite(hh[i]):
            continue
        s = np.nan_to_num(Sp[i]); k = _kk(fr, hh[i]); om = 2 * np.pi * fr
        fac = om * k * np.cosh(2 * k * hh[i]) / (np.sinh(k * hh[i]) ** 2)
        us[i] = np.trapezoid((s * fac)[band], fr[band])
    wv = pd.DataFrame(dict(Hm0=np.where(good, waves.Hm0.values, np.nan), Tm01=np.where(good, waves.Tm01.values, np.nan),
                           Tp=np.where(good, waves.Tp.values, np.nan), Us0=us, Ub=np.where(good, waves.Ub.values, np.nan)),
                      index=pd.DatetimeIndex(waves.time.values))
    wv["Us_mono"] = np.pi ** 3 * wv.Hm0 ** 2 / (C.g * wv.Tm01 ** 3)
    return wv


def main(argv=None):
    a = C.argomenti(__doc__, argv)
    t0, t1 = (pd.Timestamp(x) for x in C.PERIODO)
    # flussi ERA5, positivi verso il basso
    E = xr.open_dataset(C.trova(a.dati, "flussi")).sel(**C.PUNTO_ERA5)
    f = pd.DataFrame({k: E[v].to_series() for k, v in (("sw", "avg_snswrf"), ("lw", "avg_snlwrf"), ("lat", "avg_slhtf"),
                                                         ("sen", "avg_ishf"), ("evap", "avg_ie"), ("prec", "avg_tprate"))})
    f.index = pd.DatetimeIndex(f.index)
    f["Q"] = f.sw + f.lw + f.lat + f.sen
    f["EmP"] = -f.evap - f.prec                         # kg m-2 s-1, positivo = evaporazione netta
    # coefficienti di espansione dal modello in superficie
    _, tm, th, so = C.modello_punto(a.dati)
    Ts, Ss = th[:, 0], so[:, 0]
    alpha = -(C.rho_eos(Ts + 0.05, Ss) - C.rho_eos(Ts - 0.05, Ss)) / 0.1 / C.rho_eos(Ts, Ss)
    beta = (C.rho_eos(Ts, Ss + 0.05) - C.rho_eos(Ts, Ss - 0.05)) / 0.1 / C.rho_eos(Ts, Ss)
    f = f.join(pd.DataFrame(dict(alpha=alpha, beta=beta, Ts=Ts, Ss=Ss), index=tm), how="inner")
    # flusso di galleggiamento, positivo = stabilizzante
    f["B_T"] = C.g * f.alpha * f.Q / (C.RHO0 * C.CP)
    f["B_S"] = -C.g * f.beta * f.Ss * f.EmP / C.RHO0
    f["B"] = f.B_T + f.B_S
    # vento CMEMS
    W = xr.open_dataset(C.trova(a.dati, "vento")).sel(latitude=C.VENTO_LAT, method="nearest").sel(longitude=list(C.VENTO_LON), method="nearest")
    w = pd.DataFrame(dict(u10=W.eastward_wind.mean("longitude").to_series(), v10=W.northward_wind.mean("longitude").to_series()))
    w.index = pd.DatetimeIndex(w.index)
    w["U10"] = np.hypot(w.u10, w.v10)
    Uc = w.U10.clip(lower=0.5)
    Cd = 1e-3 * (2.7 / Uc + 0.142 + Uc / 13.09)        # Large e Yeager (2004)
    w["tau"] = C.RHOA * Cd * w.U10 ** 2
    w["taux"], w["tauy"] = w.tau * w.u10 / Uc, w.tau * w.v10 / Uc
    w["ustar"] = np.sqrt(w.tau / C.RHO0)
    ang = np.deg2rad(C.ASSE_COSTA)
    w["tau_ls"] = w.taux * np.sin(ang) + w.tauy * np.cos(ang)            # positivo = favorevole al downwelling
    w["tau_cs"] = w.taux * np.sin(np.deg2rad(C.VERSO_COSTA)) + w.tauy * np.cos(np.deg2rad(C.VERSO_COSTA))
    w["ekman_onshore"] = w.tau_ls / (C.RHO0 * C.FCOR)                     # m2/s per metro di costa
    # onde dall'ADCP
    wv = stokes_superficie(xr.open_dataset(C.trova(a.risultati, "*_L3_waves.nc")))
    hr = f.join(w, how="inner")
    hr = hr.join(wv.resample("1h").mean().interpolate(limit=2), how="left")
    hr = hr[(hr.index >= t0) & (hr.index < t1)]
    hr["La_t"] = np.sqrt(hr.ustar / hr.Us0)
    hr["mld"] = C.mld_modello(a.dati).reindex(hr.index)
    hr["w_shear"] = hr.ustar ** 3
    hr["w_lang"] = hr.ustar ** 2 * hr.Us0
    hr["w_conv"] = (-hr.B).clip(lower=0) * hr.mld.clip(upper=C.H)
    hr.to_csv(os.path.join(a.uscita, "forzanti_orari.csv"))
    d = hr.resample("1D").mean()
    d["U10max"] = hr.U10.resample("1D").max()
    d.to_csv(os.path.join(a.uscita, "forzanti_giornalieri.csv"))
    print(f"forzanti: {len(hr)} ore; vento medio {hr.U10.mean():.1f} m/s, Hm0 mediana {hr.Hm0.median():.2f} m, "
          f"deriva di Stokes mediana {100 * hr.Us0.median():.2f} cm/s")


if __name__ == "__main__":
    main()
