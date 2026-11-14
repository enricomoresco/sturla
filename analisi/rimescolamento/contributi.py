"""Tabella dei contributi per finestra: energia necessaria a rimescolare la colonna di 34.5 m,
energia fornita da ciascun processo locale, stratificazione tolta e calore arrivato di lato nel
modello 3D, riscaldamento al fondo misurato e simulato.

Il raffreddamento agisce solo sotto lo strato mescolato di spessore h: effetto diretto
rho*B*(H-h)/2 (con B netto, giorno e notte) piu' la convezione penetrativa 0.1*rho*B*h.

Uscita: contributi_finali.csv (energie in J/m2, calore in MJ/m2, temperature in gradi)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # anche con python -I

import os

import numpy as np
import pandas as pd

import comune as C
from col1d import Column


def main(argv=None):
    a = C.argomenti(__doc__, argv)
    H = C.H
    zm, tm, th, so = C.modello_punto(a.dati)
    zz = np.linspace(0, H, 346)
    Ti = np.array([np.interp(zz, zm, th[i]) for i in range(len(tm))])
    Si = np.array([np.interp(zz, zm, so[i]) for i in range(len(tm))])
    r = C.rho_eos(Ti, Si)
    phiH = pd.Series(np.trapezoid((r.mean(1, keepdims=True) - r) * C.g * (H - zz), zz, axis=1), index=tm)
    HC = pd.Series(C.RHO0 * C.CP * np.trapezoid(Ti, zz, axis=1), index=tm)
    F = C.leggi(a.uscita, "forzanti_orari.csv")
    h = C.serie_orarie(a.risultati)
    ok = h.deployment_qc == 1
    Umean = h.corrente_media_vel.where(ok).reindex(F.index).interpolate(limit=6)
    Ta = h.temperatura.where(ok)
    E1 = pd.read_pickle(os.path.join(a.uscita, "esperimenti_1d.pkl"))

    def m6(s, t):  # media su +-3 ore
        return float(s.loc[t - pd.Timedelta("3h"):t + pd.Timedelta("3h")].mean())

    rows = []
    dt = 3600.0
    for name, t0s, t1s, ev in C.FINESTRE:
        t0, t1 = pd.Timestamp(t0s), pd.Timestamp(t1s)
        x = F.loc[t0:t1 - pd.Timedelta("1h")]
        hml = x.mld.interpolate().clip(1, H)
        r_ = dict(finestra=name)
        r_["necessaria"] = m6(phiH, t0)
        r_["variazione_3D"] = m6(phiH, t0) - m6(phiH, t1)
        r_["raffreddamento"] = float((C.RHO0 * (-x.B) * (H - hml) / 2).sum() * dt)
        r_["convezione"] = float((0.1 * C.RHO0 * (-x.B).clip(lower=0) * hml).sum() * dt)
        r_["vento"] = float((0.023 * 6.4e-5 * C.RHOA * x.U10 ** 3).sum() * dt)
        r_["vento_d039"] = r_["vento"] * 0.039 / 0.023
        r_["Langmuir"] = float((0.2 * 0.22 * C.RHO0 * x.ustar ** 2 * x.Us0.fillna(0)).sum() * dt)
        r_["frangimento"] = float((0.2 * 100 * C.RHO0 * x.ustar ** 3 * np.clip(0.3 * x.Hm0.fillna(0) / hml, 0, 1)).sum() * dt)
        sig = x.Ub.interpolate().fillna(0) / 2
        r_["onde_fondo"] = float((0.0037 * C.RHO0 * 0.5 * 0.02 * 2 * np.sqrt(2 / np.pi) * sig ** 3).sum() * dt)
        r_["corrente_fondo"] = float((0.0037 * 2.5e-3 * C.RHO0 * Umean.loc[x.index].fillna(Umean.median()) ** 3).sum() * dt)
        loc = ["raffreddamento", "convezione", "vento", "Langmuir", "frangimento", "onde_fondo", "corrente_fondo"]
        r_["totale_locale"] = sum(r_[k] for k in loc)
        r_["totale_locale_d039"] = r_["totale_locale"] - r_["vento"] + r_["vento_d039"]
        # vento doppio: vento x8 (cubo), Langmuir x4 (u*^2), frangimento x8 (u*^3)
        r_["vento_x2_totale"] = (r_["totale_locale"] - r_["vento"] - r_["Langmuir"] - r_["frangimento"]
                                 + 8 * r_["vento"] + 4 * r_["Langmuir"] + 8 * r_["frangimento"])
        r_["Q_MJ"] = float(x.Q.sum() * dt / 1e6)
        r_["dHC_3D_MJ"] = (m6(HC, t1) - m6(HC, t0)) / 1e6
        r_["avvezione_MJ"] = r_["dHC_3D_MJ"] - r_["Q_MJ"]
        r_["dT_ADCP"] = m6(Ta, t1) - m6(Ta, t0)
        d3 = E1["D3"][H]
        r_["dT_3D"] = float(d3.T_probe.loc[t1] - d3.T_probe.loc[t0])
        ev_t0 = pd.Timestamp(C.EVENTI[ev][0]); i0 = int(np.argmin(np.abs(tm - ev_t0)))
        for k, lab in (("tutto (vento, onde, fondo)", "dT_1D"), ("tutto, vento x1.5", "dT_1D_x1.5"), ("tutto, vento x2", "dT_1D_x2")):
            s = E1["series"][(ev, "stazione", k)].T_probe
            start = float(s.loc[t0]) if t0 in s.index else Column(H, zm, th[i0], so[i0], bbl=2.0).at(C.PROF_SENSORE)
            r_[lab] = float(s.loc[t1]) - start
        rows.append(r_)
    T = pd.DataFrame(rows).set_index("finestra")
    T.to_csv(os.path.join(a.uscita, "contributi_finali.csv"))
    print(T.round(1).T.to_string())


if __name__ == "__main__":
    main()
