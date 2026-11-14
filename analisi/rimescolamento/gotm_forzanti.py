"""File di ingresso per GOTM, per rendere la colonna 1D un vero test delle due ipotesi
(mescolamento contro trasporto) a inizio ottobre.

Scrive in <uscita>/gotm/:
  velprof_adcp.dat     profili orari di corrente misurati dall'ADCP (u, v), per rilassare le
                       velocita' di GOTM a quelle osservate: danno taglio e sforzo al fondo realistici
  w_medfs.dat          velocita' verticale giornaliera del modello MedFS vicino alla costa e quota del
                       suo massimo, per l'avvezione verticale imposta (prova dell'ipotesi B in 1D)
  init_ts_adcp.dat     profilo iniziale di temperatura e salinita' del modello, spostato verso l'alto
                       finche' la temperatura al sensore coincide con quella misurata dall'ADCP
  ts_medfs_spostato.dat  profili orari di MedFS con lo stesso spostamento, per rilassare lo strato
                       profondo (trasporto laterale)
  LEGGIMI_GOTM.md      come usarli

Formato dei profili: per ogni istante una riga 'data ora N 1', poi N righe 'z valori' dal fondo
verso la superficie (z negativa, in metri). Serie temporali: 'data ora valori'.

  python analisi/rimescolamento/gotm_forzanti.py --dati ... --risultati ... --uscita ... \\
         [--inizio 2026-10-01] [--fine 2026-10-15] [--profilo-iniziale "2026-10-04 00:00"]
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # anche con python -I

import argparse
import os

import numpy as np
import pandas as pd
import xarray as xr

import comune as C

LEGGIMI = """# File di ingresso per GOTM (Sturla, ottobre 2026)

Prodotti da `analisi/rimescolamento/gotm_forzanti.py`.

| File | Contenuto | Uso in GOTM |
|---|---|---|
| `velprof_adcp.dat` | profili orari di u, v misurati (m/s), {nvel} quote fra {zmin:.1f} e {zmax:.1f} m | rilassamento delle velocità ai profili osservati (sezione delle velocità osservate, colonne 1 = u e 2 = v, con un tempo di rilassamento di 1–6 ore) |
| `w_medfs.dat` | velocità verticale giornaliera di MedFS (m/s, positiva verso l'alto) e quota del suo massimo come coordinata z (m, negativa sotto la superficie, non altezza sul fondo) | avvezione verticale imposta (`mimic_3d/w/max` colonna 1, `height` colonna 2): GOTM usa un profilo triangolare, nullo in superficie e al fondo, con il massimo alla quota indicata |
| `init_ts_adcp.dat` | profilo iniziale di MedFS spostato verso l'alto di {shift:.1f} m, così al sensore vale {tadcp:.2f} °C come misurato | profilo iniziale di temperatura (colonna 1) e salinità (colonna 2) |
| `ts_medfs_spostato.dat` | profili orari di MedFS con lo stesso spostamento di {shift:.1f} m | rilassamento della temperatura e della salinità nello strato vicino al fondo (`relax: h_b`, `tau_b`, per esempio 20 m e 1 giorno): rappresenta il trasporto laterale, che in 1D la sola velocità verticale non può portare fino al fondo |

I nomi esatti delle chiavi nel file yaml dipendono dalla versione di GOTM: il modo sicuro è
generare il file completo con `gotm --write_yaml gotm_completo.yaml --detail full` e cercarvi le
sezioni delle velocità osservate, dell'avvezione verticale (`w`) e di CVMix (strato interno e di
fondo). I profili sono scritti dal fondo verso la superficie (ultimo numero della riga di
intestazione = 1): controllare che sia la stessa convenzione del vostro `init_ts.dat`.

Esperimenti suggeriti, ciascuno aggiunto al precedente:
1. GLS o KPP con le velocità rilassate all'ADCP: taglio e sforzo al fondo realistici.
2. KPP con CVMix interno (taglio) e strato di fondo accesi.
3. Profilo iniziale `init_ts_adcp.dat`: fondo freddo come misurato.
4. Avvezione verticale da `w_medfs.dat`: prova diretta del downwelling (ipotesi B).
5. Rugosità del fondo (`h0b`) aumentata durante la mareggiata: rugosità apparente onde + corrente.

Se il fondo si scalda gradualmente dal 5 ottobre solo con il punto 4, il riscaldamento prima della
mareggiata è trasporto; se i punti 1, 2 e 5 bastano a scaldarlo l'8–9 ottobre, la mareggiata ha
completato il lavoro con il mescolamento.
"""


def _blocco(f, t, z, cols):
    o = np.argsort(z)                                 # dal fondo (z piu' negativa) verso la superficie
    f.write(f"{t:%Y-%m-%d %H:%M:%S}  {len(z)}  1\n")
    for i in o:
        f.write(f"{z[i]:9.3f} " + " ".join(f"{c[i]:10.5f}" for c in cols) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dati", required=True); ap.add_argument("--risultati", required=True)
    ap.add_argument("--uscita", default="out_rimescolamento")
    ap.add_argument("--inizio", default="2026-10-01"); ap.add_argument("--fine", default="2026-10-15")
    ap.add_argument("--profilo-iniziale", default="2026-10-04 00:00")
    a = ap.parse_args(argv)
    out = os.path.join(a.uscita, "gotm"); os.makedirs(out, exist_ok=True)
    t0, t1 = pd.Timestamp(a.inizio), pd.Timestamp(a.fine)

    # 1. profili di corrente dell'ADCP
    c = xr.open_dataset(C.trova(a.risultati, "*_L2_currents.nc"))
    ok = np.isin(c.vel_qc.values, (1, 3)) & (c.deployment_qc.values == 1)[:, None]
    t = pd.DatetimeIndex(c.time.values)
    hab = c.z.values.astype(float)
    buone = np.isfinite(np.where(ok, c.u.values, np.nan)).mean(0) > 0.5
    U = pd.DataFrame(np.where(ok, c.u.values, np.nan)[:, buone], index=t)
    V = pd.DataFrame(np.where(ok, c.v.values, np.nan)[:, buone], index=t)
    sel = (t >= t0) & (t <= t1)
    U = U[sel].interpolate(limit=6, limit_direction="both").T.interpolate(limit_direction="both").T
    V = V[sel].interpolate(limit=6, limit_direction="both").T.interpolate(limit_direction="both").T
    zv = -(C.H - hab[buone])
    with open(os.path.join(out, "velprof_adcp.dat"), "w") as f:
        for ti in U.index:
            if np.isfinite(U.loc[ti].values).all():
                _blocco(f, ti, zv, [U.loc[ti].values, V.loc[ti].values])

    # 2. velocita' verticale MedFS: media dei punti costieri, massimo entro la colonna della stazione
    W = xr.open_dataset(C.trova(a.dati, "*medfs_riquadro_velocita_verticale*.nc")).wo
    W = W.sel(latitude=44.3542, method="nearest").sel(longitude=slice(8.95, 9.09)).mean("longitude")
    W = W.sel(time=slice(t0 - pd.Timedelta("1D"), t1 + pd.Timedelta("1D")))
    dep = W.depth.values.astype(float); k = dep <= C.H
    with open(os.path.join(out, "w_medfs.dat"), "w") as f:
        for ti, prof in zip(pd.DatetimeIndex(W.time.values) + pd.Timedelta("12h"), W.values[:, k]):
            i = int(np.nanargmax(np.abs(prof)))
            f.write(f"{ti:%Y-%m-%d %H:%M:%S}  {prof[i]:.4e}  {-dep[k][i]:.2f}\n")

    # 3. profilo iniziale spostato verso l'alto per avere al sensore la temperatura misurata
    zm, tm, th, so = C.modello_punto(a.dati)
    ti0 = pd.Timestamp(a.profilo_iniziale); i0 = int(np.argmin(np.abs(tm - ti0)))
    h = C.serie_orarie(a.risultati)
    tadcp = float(h.temperatura.loc[ti0 - pd.Timedelta("3h"):ti0 + pd.Timedelta("3h")].mean())
    Tp, Sp = th[i0], so[i0]
    zz = np.arange(0, zm[-1] + 0.01, 0.1); Tz = np.interp(zz, zm, Tp)
    sotto = np.where(Tz <= tadcp)[0]
    shift = (zz[sotto[0]] - C.PROF_SENSORE) if len(sotto) else 0.0
    zg = np.arange(0.25, C.H, 0.5)
    Tn = np.interp(np.minimum(zg + shift, zm[-1]), zm, Tp); Sn = np.interp(np.minimum(zg + shift, zm[-1]), zm, Sp)
    with open(os.path.join(out, "init_ts_adcp.dat"), "w") as f:
        _blocco(f, ti0, -zg, [Tn, Sn])
    # 4. profili orari di MedFS con lo stesso spostamento, per rilassare lo strato profondo:
    #    rappresentano il trasporto laterale (in 1D la velocita' verticale si annulla al fondo)
    with open(os.path.join(out, "ts_medfs_spostato.dat"), "w") as f:
        for i in np.where((tm >= t0) & (tm <= t1))[0]:
            zz_ = np.minimum(zg + shift, zm[-1])
            _blocco(f, tm[i], -zg, [np.interp(zz_, zm, th[i]), np.interp(zz_, zm, so[i])])

    with open(os.path.join(out, "LEGGIMI_GOTM.md"), "w") as f:
        f.write(LEGGIMI.format(nvel=len(zv), zmin=zv.min(), zmax=zv.max(), shift=shift, tadcp=tadcp))
    print(f"scritti in {out}: velprof_adcp.dat ({len(U)} profili, {len(zv)} quote), w_medfs.dat, "
          f"init_ts_adcp.dat (spostamento {shift:.1f} m, sensore {tadcp:.2f} gradi)")


if __name__ == "__main__":
    main()
