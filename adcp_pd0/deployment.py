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
                      margin_min: float = 10.0, motion_heading_deg: float = 1.0,
                      motion_tilt_deg: float = 0.3, motion_merge_min: float = 60.0,
                      verbose: bool = True) -> dict:
    t = reader.times()
    ds = reader.load(["pressure", "pitch", "roll", "heading"])
    hr = np.deg2rad(ds["heading"].values.astype(float))
    df = pd.DataFrame(dict(p=ds.pressure.values, tilt=np.hypot(ds["pitch"].values, ds["roll"].values),
                           pitch=ds["pitch"].values, roll=ds["roll"].values),
                      index=pd.DatetimeIndex(ds.time.values))
    m = df.resample("1min").median()
    # heading: media circolare al minuto
    hs = pd.DataFrame(dict(s=np.sin(hr), c=np.cos(hr)), index=df.index).resample("1min").mean()
    m["heading"] = np.mod(np.degrees(np.arctan2(hs.s, hs.c)), 360)
    m = m.dropna()
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
                             margin_min=margin_min, motion_heading_deg=motion_heading_deg,
                             motion_tilt_deg=motion_tilt_deg, motion_merge_min=motion_merge_min))
    res["motion_events"] = motion_events(m, start, end, motion_heading_deg, motion_tilt_deg,
                                         motion_merge_min)
    if verbose:
        print(f"  periodo in posizione (automatico): {res['start'][:19]} -> {res['end'][:19]}")
        print(f"     dati registrati:               {res['data_start'][:19]} -> {res['data_end'][:19]}")
        print(f"     pressione di riferimento {p_ref:.2f} dbar (tolleranza {tol:.2f}), inclinazione "
              f"di riferimento {tilt_ref:.1f} gradi; minuti anomali all'interno: "
              f"{res['minutes_out_of_position_inside']}")
        ev = res["motion_events"]
        print(f"     movimenti del telaio rilevati: {len(ev)}")
        for e in ev:
            print(f"       {e['start'][:16]} -> {e['end'][:16]}: heading {e['heading_before']:.1f} -> "
                  f"{e['heading_after']:.1f} gradi, pitch {e['pitch_before']:+.2f} -> {e['pitch_after']:+.2f}, "
                  f"roll {e['roll_before']:+.2f} -> {e['roll_after']:+.2f}")
    res["_minute_series"] = m.assign(ok=ok)
    return res


def motion_events(m, start, end, thr_heading=1.0, thr_tilt=0.3, merge_min=60.0, ctx=30):
    """Intervalli in cui il telaio si muove: salti di heading > ``thr_heading`` o di pitch/roll
    > ``thr_tilt`` fra minuti consecutivi con dati (le misure sono a burst, quindi due minuti
    consecutivi possono distare decine di minuti). Intervalli vicini (< ``merge_min``) uniti."""
    mm = m[(m.index >= start) & (m.index < end)]
    if len(mm) < 2:
        return []
    dh = (np.diff(mm.heading.values) + 180) % 360 - 180
    mov = ((np.abs(dh) > thr_heading) | (np.abs(np.diff(mm.pitch.values)) > thr_tilt)
           | (np.abs(np.diff(mm["roll"].values)) > thr_tilt))
    iv = [[mm.index[i], mm.index[i + 1] + pd.Timedelta(minutes=1)] for i in np.flatnonzero(mov)]
    ev = []
    for a, b in iv:
        if ev and a - ev[-1][1] <= pd.Timedelta(minutes=merge_min):
            ev[-1][1] = max(ev[-1][1], b)
        else:
            ev.append([a, b])
    out = []
    for a, b in ev:
        before = mm[mm.index < a].tail(ctx)
        after = mm[mm.index >= b].head(ctx)
        circ = lambda x: float(np.mod(np.degrees(np.arctan2(np.sin(np.deg2rad(x)).mean(),
                                                            np.cos(np.deg2rad(x)).mean())), 360)) if len(x) else np.nan
        out.append(dict(start=str(a), end=str(b),
                        heading_before=circ(before.heading.values), heading_after=circ(after.heading.values),
                        pitch_before=float(before.pitch.median()) if len(before) else np.nan,
                        pitch_after=float(after.pitch.median()) if len(after) else np.nan,
                        roll_before=float(before["roll"].median()) if len(before) else np.nan,
                        roll_after=float(after["roll"].median()) if len(after) else np.nan))
    return out


def motion_overlap(t0, t1, dep: dict) -> np.ndarray:
    """True per gli intervalli [t0, t1] che si sovrappongono a un movimento del telaio."""
    t0 = np.asarray(t0, dtype="datetime64[ns]")
    t1 = np.asarray(t1, dtype="datetime64[ns]")
    out = np.zeros(t0.shape, bool)
    for e in dep.get("motion_events", []):
        a, b = np.datetime64(pd.Timestamp(e["start"])), np.datetime64(pd.Timestamp(e["end"]))
        out |= (t0 <= b) & (t1 >= a)
    return out


def get_deployment(outdir: str, reader=None, start=None, end=None, redetect=False, fig_path=None,
                   **kw) -> dict:
    """Legge ``outdir/deployment.json`` o lo crea; ``start``/``end`` sovrascrivono il risultato."""
    path = os.path.join(outdir, "deployment.json")
    dep = None
    if os.path.exists(path) and not redetect:
        with open(path, encoding="utf-8") as fh:
            dep = json.load(fh)
        if "motion_events" not in dep and reader is not None:
            dep = None      # file di una versione precedente: si ricalcola
    if dep is None:
        if reader is None:
            raise ValueError("serve uno stream per individuare il periodo in posizione")
        dep = detect_deployment(reader, **kw)
        ms = dep.pop("_minute_series")
        try:
            _plot_deployment(ms, dep, fig_path or os.path.join(outdir, "figures", "deployment.png"))
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
    fig, axs = plt.subplots(3, 1, figsize=(11, 7), sharex=True, layout="constrained")
    s, e = pd.Timestamp(dep["start"]), pd.Timestamp(dep["end"])
    inside = (m.index >= s) & (m.index < e)
    for ax, k, lab in ((axs[0], "p", "Pressione [dbar]"), (axs[1], "tilt", "Inclinazione [gradi]"),
                       (axs[2], "heading", "Heading [gradi]")):
        ax.plot(m.index, m[k], ".", ms=1.5, color=P.C1)
        if k != "p" and inside.any():
            lo, hi = m[k][inside].min(), m[k][inside].max()
            pad = max(0.5, 0.1 * (hi - lo))
            ax.set_ylim(lo - pad, hi + pad)
        if m.index[0] < s:
            ax.axvspan(m.index[0], s, color=P.AXIS, alpha=0.35, lw=0)
        if e < m.index[-1]:
            ax.axvspan(e, m.index[-1], color=P.AXIS, alpha=0.35, lw=0)
        for ev in dep.get("motion_events", []):
            ax.axvspan(pd.Timestamp(ev["start"]), pd.Timestamp(ev["end"]), color=P.C2, alpha=0.35, lw=0)
        ax.set_ylabel(lab)
    axs[0].set_title(f"Periodo in posizione: {str(s)[:16]} -> {str(e)[:16]} (grigio: fuori posizione, "
                     f"flag 4; arancione: movimento del telaio, flag 3)")
    axs[1].set_title("Inclinazione e heading: scala limitata al periodo in posizione")
    P._fmt_time(axs[2])
    P._save(fig, path)
