"""Individuazione automatica del periodo in cui lo strumento è in posizione sul fondo.

Criteri, su mediane al minuto di pressione e inclinazione:
  - in acqua:      pressione > p_min (dbar)
  - in posizione:  |pressione - p_rif| < tolleranza, con p_rif = mediana della pressione
                   in acqua (prevalgono di gran lunga i dati sul fondo); tolleranza =
                   max(tol_min, tol_frac * p_rif), così la marea non conta
  - fermo:         inclinazione sqrt(pitch^2 + roll^2) < tilt_max e variazione
                   dell'inclinazione rispetto alla mediana in posizione < tilt_change_max
Inizio = primo minuto da cui le condizioni restano vere per almeno ``min_stable_min``
minuti di dati; fine = ultimo minuto con la stessa proprietà all'indietro. Si aggiunge un
margine di sicurezza (``margin_min``) dopo l'inizio e prima della fine.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd


def detect_deployment(reader, p_min: float = 1.0, tol_frac: float = 0.10, tol_min: float = 2.0,
                      tilt_max: float = 15.0, tilt_change_max: float = 3.0, min_stable_min: int = 30,
                      margin_min: float = 10.0, verbose: bool = True) -> dict:
    t = reader.times()
    ds = reader.load(["pressure", "pitch", "roll"])
    df = pd.DataFrame(dict(p=ds.pressure.values, tilt=np.hypot(ds["pitch"].values, ds["roll"].values)),
                      index=pd.DatetimeIndex(ds.time.values))
    m = df.resample("1min").median().dropna()
    if m.empty:
        raise ValueError("nessun dato di pressione")
    wet = m.p > p_min
    if not wet.any():
        raise ValueError("lo strumento non risulta mai in acqua (pressione sempre < p_min)")
    p_ref = float(m.p[wet].median())
    tol = max(tol_min, tol_frac * p_ref)
    pos = wet & ((m.p - p_ref).abs() < tol) & (m.tilt < tilt_max)
    tilt_ref = float(m.tilt[pos].median()) if pos.any() else 0.0
    ok = (pos & ((m.tilt - tilt_ref).abs() < tilt_change_max)).values
    # primo/ultimo indice da cui ok resta vero per min_stable_min campioni consecutivi
    n = int(min_stable_min)
    run = np.convolve(ok.astype(int), np.ones(n, int), "valid") == n if len(ok) >= n else np.array([ok.all()])
    if not run.any():
        raise ValueError("nessun periodo stabile in posizione trovato: controllare i parametri")
    i0 = int(np.argmax(run))
    i1 = int(len(run) - 1 - np.argmax(run[::-1])) + n - 1
    # il margine serve solo se c'è una fase fuori posizione prima/dopo (non se i dati
    # iniziano o finiscono con lo strumento già in posizione)
    start = m.index[i0] + pd.Timedelta(minutes=margin_min if ok[:i0 + 1].size and not ok[:i0 + 1].all() else 0)
    if i0 == 0:
        start = pd.Timestamp(t[0])
    tail_ok = ok[i1:].all()
    end = m.index[i1] + pd.Timedelta(minutes=1) - pd.Timedelta(minutes=0 if tail_ok else margin_min)
    if tail_ok:
        end = pd.Timestamp(t[-1]) + pd.Timedelta(seconds=1)
    inside = (m.index >= start) & (m.index < end)
    res = dict(method="automatico", start=str(start), end=str(end), p_ref_dbar=round(p_ref, 3),
               tolerance_dbar=round(tol, 3), tilt_ref_deg=round(tilt_ref, 2),
               data_start=str(pd.Timestamp(t[0])), data_end=str(pd.Timestamp(t[-1])),
               minutes_out_of_position_inside=int((~ok[inside]).sum()),
               criteria=dict(p_min=p_min, tol_frac=tol_frac, tol_min=tol_min, tilt_max=tilt_max,
                             tilt_change_max=tilt_change_max, min_stable_min=min_stable_min,
                             margin_min=margin_min))
    if verbose:
        print(f"  periodo in posizione (automatico): {res['start'][:19]} -> {res['end'][:19]}")
        print(f"     dati registrati:               {res['data_start'][:19]} -> {res['data_end'][:19]}")
        print(f"     pressione di riferimento {p_ref:.2f} dbar (tolleranza {tol:.2f}), inclinazione "
              f"di riferimento {tilt_ref:.1f} gradi; minuti anomali all'interno: "
              f"{res['minutes_out_of_position_inside']}")
    res["_minute_series"] = m.assign(ok=ok)
    return res


def get_deployment(outdir: str, reader=None, start=None, end=None, redetect=False, **kw) -> dict:
    """Legge ``outdir/deployment.json`` o lo crea; ``start``/``end`` sovrascrivono il risultato."""
    path = os.path.join(outdir, "deployment.json")
    if os.path.exists(path) and not redetect:
        with open(path, encoding="utf-8") as fh:
            dep = json.load(fh)
    else:
        if reader is None:
            raise ValueError("serve uno stream per individuare il periodo in posizione")
        dep = detect_deployment(reader, **kw)
        ms = dep.pop("_minute_series")
        try:
            _plot_deployment(ms, dep, os.path.join(outdir, "figures", "deployment.png"))
        except Exception as e:  # il grafico non deve bloccare l'elaborazione
            print(f"  (grafico deployment non prodotto: {e})")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(dep, fh, indent=2)
    if start or end:
        dep = dict(dep)
        if start:
            dep["start"] = str(pd.Timestamp(start))
        if end:
            dep["end"] = str(pd.Timestamp(end))
        dep["method"] = "manuale (--start/--end)" if (start and end) else "automatico + correzione manuale"
    return dep


def in_position(times, dep: dict) -> np.ndarray:
    t = np.asarray(times, dtype="datetime64[ns]")
    return (t >= np.datetime64(pd.Timestamp(dep["start"]))) & (t < np.datetime64(pd.Timestamp(dep["end"])))


def _plot_deployment(m, dep, path):
    import matplotlib.pyplot as plt
    from . import plots as P
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig, axs = plt.subplots(2, 1, figsize=(11, 5), sharex=True, layout="constrained")
    s, e = pd.Timestamp(dep["start"]), pd.Timestamp(dep["end"])
    for ax, k, lab in ((axs[0], "p", "Pressione [dbar]"), (axs[1], "tilt", "Inclinazione [gradi]")):
        ax.plot(m.index, m[k], color=P.C1, lw=1)
        ax.axvspan(m.index[0], s, color=P.AXIS, alpha=0.35, lw=0)
        ax.axvspan(e, m.index[-1], color=P.AXIS, alpha=0.35, lw=0)
        ax.set_ylabel(lab)
    axs[0].set_title(f"Periodo in posizione: {str(s)[:16]} -> {str(e)[:16]} (in grigio: fuori posizione, flag 4)")
    P._fmt_time(axs[1])
    P._save(fig, path)
