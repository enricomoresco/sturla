"""Effetti di acqua bassa: profilo della deriva di Stokes dagli spettri dell'ADCP, trasporto di Stokes
e di Ekman, spessore dello strato di Ekman, corrente verso costa in mareggiata e in calma.

Uscita: stokes_ekman.csv, corrente_verso_costa.csv
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
    om = 2 * np.pi * fv; k = np.maximum(om ** 2 / C.g, 1e-6)
    for _ in range(50):
        th = np.tanh(k * hh); k = k - (C.g * k * th - om ** 2) / (C.g * th + C.g * k * hh * (1 - th ** 2))
    return k


def main(argv=None):
    a = C.argomenti(__doc__, argv)
    Wv = xr.open_dataset(C.trova(a.risultati, "*_L3_waves.nc"))
    fr = Wv.frequency.values.astype(float); Sp = Wv.S.values.astype(float); hh = Wv.water_depth.values.astype(float)
    good = np.isin(Wv.wave_qc.values, (1, 3))
    band = (fr >= 0.04) & (fr <= 0.5)
    rows = []
    for i in range(Sp.shape[0]):
        if not good[i] or not np.isfinite(hh[i]):
            continue
        s = np.nan_to_num(Sp[i])[band]; f = fr[band]; k = _kk(f, hh[i]); om = 2 * np.pi * f
        Mst = np.trapezoid(s * om / np.tanh(k * hh[i]), f)                     # trasporto di Stokes, m2/s
        zz = np.linspace(0, hh[i], 70)
        us = np.array([np.trapezoid(s * om * k * np.cosh(2 * k * (hh[i] - z)) / np.sinh(k * hh[i]) ** 2, f) for z in zz])
        zdec = zz[np.argmax(us < us[0] / np.e)] if (us < us[0] / np.e).any() else np.nan
        rows.append(dict(time=pd.Timestamp(Wv.time.values[i]), Hm0=float(Wv.Hm0.values[i]), Us0=us[0], Us_fondo=us[-1],
                         trasporto_Stokes=Mst, decadimento_1e_m=zdec))
    st = pd.DataFrame(rows).set_index("time")
    F = C.leggi(a.uscita, "forzanti_orari.csv")
    st["trasporto_Ekman"] = (F.tau / (C.RHO0 * C.FCOR)).reindex(st.index, method="nearest", tolerance=pd.Timedelta("31min"))
    st["spessore_Ekman_m"] = (0.4 * F.ustar / C.FCOR).reindex(st.index, method="nearest", tolerance=pd.Timedelta("31min"))
    st.to_csv(os.path.join(a.uscita, "stokes_ekman.csv"))
    for lab, m in (("Hm0 < 0.5 m", st.Hm0 < 0.5), ("0.5-1 m", (st.Hm0 >= 0.5) & (st.Hm0 < 1)), ("Hm0 >= 1 m", st.Hm0 >= 1)):
        x = st[m]
        print(f"{lab:11s} n={len(x):4d}: Stokes in superficie {100 * x.Us0.median():.2f} cm/s, decadimento 1/e {x.decadimento_1e_m.median():.1f} m, "
              f"trasporto di Stokes {x.trasporto_Stokes.median():.3f} m2/s, di Ekman {x.trasporto_Ekman.median():.3f} m2/s, "
              f"strato di Ekman {x.spessore_Ekman_m.median():.0f} m")
    # corrente euleriana verso costa in calma e in mareggiata
    c = xr.open_dataset(C.trova(a.risultati, "*_L2_currents.nc"))
    ok = np.isin(c.vel_qc.values, (1, 3)) & (c.deployment_qc.values == 1)[:, None]
    ang = np.deg2rad(C.VERSO_COSTA)
    cs = np.where(ok, c.u.values * np.sin(ang) + c.v.values * np.cos(ang), np.nan)
    t = pd.DatetimeIndex(c.time.values) - pd.Timedelta("30min")
    hm = F.Hm0.reindex(t, method="nearest", tolerance=pd.Timedelta("31min")).values
    z = c.z.values.astype(float)
    out = []
    with np.errstate(all="ignore"):
        for zz in (3, 6, 10, 15, 20, 25, 28, 30):
            i = int(np.argmin(np.abs(z - zz)))
            calm = np.nanmedian(cs[hm < 0.4, i]); sto = np.nanmedian(cs[hm >= 1, i])
            out.append(dict(quota_dal_fondo_m=z[i], calma_cm_s=100 * calm, mareggiata_cm_s=100 * sto, differenza_cm_s=100 * (sto - calm)))
    O = pd.DataFrame(out)
    O.to_csv(os.path.join(a.uscita, "corrente_verso_costa.csv"), index=False)
    print(O.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
