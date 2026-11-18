"""Analisi armonica delle correnti dell'ADCP di Sturla.

1) UTide a ogni quota sui 110 giorni: ellissi di marea per costituente, con intervalli di confidenza
   (rumore colorato), e frazione di varianza spiegata dalla marea.
2) UTide sulla temperatura al fondo (tutto il periodo, estate, ottobre).
3) Fit ai minimi quadrati in finestre mobili di 3 giorni (come il passo 6 di Valle-Levinson, 2024):
   residuo + bande diurna (24 h), inerziale (f) e semidiurna (M2), a ogni quota e per la temperatura.
Uscite: CSV e un pickle con tutto, nella cartella indicata.

  python analisi/marea_correnti/armonica_correnti.py --risultati out --uscita out_marea_correnti

Serve UTide (pip install utide). --risultati e' la cartella con preprocessing/ del deployment.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
import xarray as xr
import utide

LAT_DEFAULT = 44.37805           # Sturla (boa sopra l'ADCP), se le coordinate non sono nei metadati


def trova(cartella, schema):
    hit = sorted(glob.glob(os.path.join(cartella, "**", schema), recursive=True))
    if not hit:
        sys.exit(f"nessun file {schema} in {cartella}")
    return hit[-1]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--risultati", required=True, help="cartella dei risultati dell'ADCP (con preprocessing/)")
    ap.add_argument("--uscita", default="out_marea_correnti", help="cartella di uscita")
    arg = ap.parse_args(argv)
    OUT = arg.uscita
    os.makedirs(OUT, exist_ok=True)
    CONST = ["M2", "S2", "N2", "K1", "O1", "M4", "MS4"]

    c = xr.open_dataset(trova(arg.risultati, "*_L2_currents.nc"))
    LAT = float(c.attrs.get("geospatial_lat_min", LAT_DEFAULT))
    print(f"latitudine {LAT} N" + ("" if "geospatial_lat_min" in c.attrs else " (coordinate non nei metadati: valore di Sturla)"))
    t = pd.DatetimeIndex(c.time.values)
    inpos = c.deployment_qc.values == 1
    ok = np.isin(c.vel_qc.values, (1, 3)) & inpos[:, None]
    U = np.where(ok, c.u.values, np.nan)[inpos]
    V = np.where(ok, c.v.values, np.nan)[inpos]
    T = c.temperature.values[inpos]
    t = t[inpos]
    z = c.z.values.astype(float)
    buone = np.isfinite(U).mean(0) > 0.5
    U, V, z = U[:, buone], V[:, buone], z[buone]
    print(f"{len(t)} ore in posizione, {t[0]} -> {t[-1]}; {len(z)} quote fra {z.min():.1f} e {z.max():.1f} m dal fondale")
    tt = t.values

    # ---------- 1) UTide a ogni quota
    righe, r2 = [], []
    for k, zk in enumerate(z):
        u, v = U[:, k], V[:, k]
        co = utide.solve(tt, u, v, lat=LAT, constit=CONST, method="ols", conf_int="linear", white=False,
                         trend=False, nodal=True, verbose=False)
        rc = utide.reconstruct(tt, co, verbose=False)
        m = np.isfinite(u) & np.isfinite(v)
        vt = np.nanvar(u[m]) + np.nanvar(v[m])
        vr = np.nanvar(u[m] - rc.u[m]) + np.nanvar(v[m] - rc.v[m])
        r2.append({"z": zk, "R2_marea": 1 - vr / vt, "dev_std_cm_s": 100 * np.sqrt(vt / 2)})
        for i, n in enumerate(co.name):
            righe.append({"z": zk, "cost": n, "maj": 100 * co.Lsmaj[i], "maj_ci": 100 * co.Lsmaj_ci[i],
                          "min": 100 * co.Lsmin[i], "min_ci": 100 * co.Lsmin_ci[i], "theta": co.theta[i],
                          "theta_ci": co.theta_ci[i], "g": co.g[i], "g_ci": co.g_ci[i], "snr": co.diagn["SNR"][i],
                          "pe": co.diagn["PE"][i]})
    E = pd.DataFrame(righe)
    R2 = pd.DataFrame(r2)
    # corrente media sulla verticale (marea barotropica)
    ub, vb = np.nanmean(U, 1), np.nanmean(V, 1)
    cob = utide.solve(tt, ub, vb, lat=LAT, constit=CONST, method="ols", conf_int="linear", white=False,
                      trend=False, nodal=True, verbose=False)
    rb = utide.reconstruct(tt, cob, verbose=False)
    mb = np.isfinite(ub)
    R2b = 1 - (np.nanvar(ub[mb] - rb.u[mb]) + np.nanvar(vb[mb] - rb.v[mb])) / (np.nanvar(ub[mb]) + np.nanvar(vb[mb]))
    EB = pd.DataFrame({"cost": cob.name, "maj": 100 * cob.Lsmaj, "maj_ci": 100 * cob.Lsmaj_ci, "min": 100 * cob.Lsmin,
                       "theta": cob.theta, "g": cob.g, "g_ci": cob.g_ci, "snr": cob.diagn["SNR"]})
    print("\n=== media sulla verticale: R2 marea %.3f" % R2b)
    print(EB.round(2).to_string(index=False))
    print("\n=== R2 della marea per quota (min, mediana, max):", R2.R2_marea.min().round(3), R2.R2_marea.median().round(3),
          R2.R2_marea.max().round(3))
    for n in ("M2", "K1", "S2", "O1"):
        x = E[E.cost == n]
        print(f"{n}: semiasse maggiore {x['maj'].min():.2f}-{x['maj'].max():.2f} cm/s (fondo {x['maj'].iloc[0]:.2f}, "
              f"alto {x['maj'].iloc[-1]:.2f}); SNR mediana {x.snr.median():.1f}; fase g fondo {x.g.iloc[0]:.0f}, alto {x.g.iloc[-1]:.0f}")

    # ---------- 2) temperatura al fondo
    def tsolve(mask, cost):
        co = utide.solve(tt[mask], T[mask], lat=LAT, constit=cost, method="ols", conf_int="linear", white=False,
                         trend=True, nodal=True, verbose=False)
        return pd.DataFrame({"cost": co.name, "A": co.A, "A_ci": co.A_ci, "g": co.g, "snr": co.diagn["SNR"]})
    estate = (t < "2026-10-01")
    otto = (t >= "2026-10-01")
    TT = {"tutto": tsolve(np.ones(len(t), bool), CONST), "estate": tsolve(estate, CONST),
          "ottobre": tsolve(otto, ["M2", "S2", "K1", "O1", "M4"])}
    for k, d in TT.items():
        print(f"\n=== temperatura al fondo, {k}"); print(d.round(3).to_string(index=False))

    # ---------- 3) finestre mobili di 3 giorni (minimi quadrati, come Valle-Levinson 2024)
    f_in = 2 * 7.2921e-5 * np.sin(np.deg2rad(LAT)) / (2 * np.pi) * 3600     # cicli/ora
    BANDE = {"D1": 1 / 24.0, "f": f_in, "D2": 1 / 12.4206}
    print(f"\nperiodo inerziale a {LAT} N: {1 / f_in:.2f} h")
    th = (t - t[0]) / pd.Timedelta("1h")
    th = np.asarray(th, float)
    W = U + 1j * V
    centri = pd.date_range(t[0] + pd.Timedelta("36h"), t[-1] - pd.Timedelta("36h"), freq="12h")
    res = {k: np.full((len(centri), len(z)), np.nan) for k in ("D1_maj", "f_cw", "f_ccw", "D2_maj", "D2_cw", "res_lungo")}
    resT = {k: np.full(len(centri), np.nan) for k in ("T_D1", "T_f", "T_D2", "T_res")}
    ax = np.deg2rad(281.0)
    for i, cc in enumerate(centri):
        sel = (t >= cc - pd.Timedelta("36h")) & (t < cc + pd.Timedelta("36h"))
        x = th[sel]
        cols = [np.ones_like(x)]
        for fr in BANDE.values():
            cols += [np.exp(2j * np.pi * fr * x), np.exp(-2j * np.pi * fr * x)]
        A = np.stack(cols, 1)
        for k in range(len(z)):
            w = W[sel, k]
            m = np.isfinite(w)
            if m.sum() < 48:
                continue
            sol = np.linalg.lstsq(A[m], w[m], rcond=None)[0]
            wr = sol[0]
            res["res_lungo"][i, k] = 100 * (wr.real * np.sin(ax) + wr.imag * np.cos(ax))
            ap, am = sol[1::2], sol[2::2]                     # antiorario (+), orario (-)
            res["D1_maj"][i, k] = 100 * (abs(ap[0]) + abs(am[0]))
            res["f_cw"][i, k] = 100 * abs(am[1])
            res["f_ccw"][i, k] = 100 * abs(ap[1])
            res["D2_maj"][i, k] = 100 * (abs(ap[2]) + abs(am[2]))
            res["D2_cw"][i, k] = 100 * abs(am[2])
        y = T[sel]
        m = np.isfinite(y)
        if m.sum() >= 48:
            cols = [np.ones_like(x)]
            for fr in BANDE.values():
                cols += [np.cos(2 * np.pi * fr * x), np.sin(2 * np.pi * fr * x)]
            B = np.stack(cols, 1)
            sol = np.linalg.lstsq(B[m], y[m], rcond=None)[0]
            resT["T_res"][i] = sol[0]
            resT["T_D1"][i], resT["T_f"][i], resT["T_D2"][i] = (np.hypot(sol[1], sol[2]), np.hypot(sol[3], sol[4]),
                                                              np.hypot(sol[5], sol[6]))
    pd.to_pickle({"z": z, "E": E, "R2": R2, "EB": EB, "R2b": R2b, "TT": TT, "centri": centri, "res": res, "resT": resT,
                  "LAT": LAT, "f_in": f_in, "t": t, "T": T}, os.path.join(OUT, "armonica_correnti.pkl"))
    E.to_csv(os.path.join(OUT, "ellissi_marea_per_quota.csv"), index=False)
    R2.to_csv(os.path.join(OUT, "varianza_marea_per_quota.csv"), index=False)
    print("\n=== finestre di 3 giorni: mediana e 90 percentile delle ampiezze (cm/s)")
    for k in ("D1_maj", "f_cw", "f_ccw", "D2_maj"):
        a = res[k]
        print(f"{k:8s} mediana {np.nanmedian(a):.2f}  p90 {np.nanpercentile(a, 90):.2f}  max {np.nanmax(a):.2f}")
    for k in ("T_D1", "T_f", "T_D2"):
        s = pd.Series(resT[k], index=centri)
        print(f"{k}: luglio-settembre mediana {s[:'2026-09-30'].median():.2f} C, 4-19 ottobre mediana "
              f"{s['2026-10-04':'2026-10-19'].median():.2f} C, max {s.max():.2f} il {s.idxmax()}")


if __name__ == "__main__":
    main()
