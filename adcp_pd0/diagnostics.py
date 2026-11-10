"""Diagnostica della trasformazione di coordinate e del beam verticale su un burst.

Riferimento indipendente: la superficie η(t) tracciata dall'eco del beam verticale.
Per la teoria lineare, sotto un'onda la velocità verticale a qualunque quota è in fase con
dη/dt (e attenuata con la profondità). Quindi:
  corr(w dai 4 beam, dη/dt) alta e positiva  -> trasformazione e orientamento corretti
  corr(w beam verticale, dη/dt) alta         -> velocità del beam verticale valida
                                                (il segno dice la sua convenzione)
"""
from __future__ import annotations

import numpy as np

from .currents import RHO, G
from .transform import to_earth
from .waves import surface_from_vb, _despike, _interp_gaps


def _corr(x, y):
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 50 or np.std(x[ok]) == 0 or np.std(y[ok]) == 0:
        return np.nan
    return float(np.corrcoef(x[ok], y[ok])[0, 1])


def burst_check(ds, attrs, sel=None, orientation=None, frac=0.7, corr_min=64, fs=2.0):
    """Serie e correlazioni per un burst (``ds`` già caricato con vel, corr, vb_*, pressure, angoli)."""
    if sel is None:
        sel = np.arange(ds.sizes["time"])
    rng = ds.range.values
    d_s = float(np.nanmean(ds.pressure.values[sel])) * 1e4 / (RHO * G)
    ic = int(np.argmin(np.abs(rng - frac * d_s)))
    vel = ds.vel.values[sel][:, ic:ic + 1, :]
    ve = to_earth(vel, attrs, ds["heading"].values[sel], ds["pitch"].values[sel],
                  ds["roll"].values[sel], orientation)[:, 0, :]
    if "corr" in ds and corr_min:
        bad = ds.corr.values[sel][:, ic, :].min(axis=-1) < corr_min
        ve[bad] = np.nan
    out = dict(cell=ic, range=float(rng[ic]), depth_sensor=d_s, w4=ve[:, 2], u=ve[:, 0], v=ve[:, 1])
    r = surface_from_vb(ds.vb_echo.values[sel], ds.vb_range.values, np.full(len(sel), d_s))
    r = _interp_gaps(_despike(r), max_frac=0.3)
    if r is not None:
        eta = r - np.mean(r)
        out["eta"] = eta
        out["deta_dt"] = np.gradient(eta) * fs
    if "vb_vel" in ds:
        iv = int(np.argmin(np.abs(ds.vb_range.values - rng[ic])))
        wv = ds.vb_vel.values[sel, iv].astype(float)
        if "vb_corr" in ds and corr_min:
            wv[ds.vb_corr.values[sel, iv] < corr_min] = np.nan
        out["wvb"] = wv
        out["vb_cell"] = iv
        out["vb_corr_median"] = float(np.median(ds.vb_corr.values[sel, iv])) if "vb_corr" in ds else np.nan
        out["vb_valid_frac"] = float(np.isfinite(wv).mean())
    de = out.get("deta_dt")
    out["c_w4_deta"] = _corr(out["w4"], de) if de is not None else np.nan
    out["c_vb_deta"] = _corr(out.get("wvb", np.full(len(sel), np.nan)), de) if de is not None else np.nan
    out["c_w4_vb"] = _corr(out["w4"], out.get("wvb", np.full(len(sel), np.nan)))
    out["std_w4"] = float(np.nanstd(out["w4"]))
    out["std_vb"] = float(np.nanstd(out["wvb"])) if "wvb" in out else np.nan
    out["std_deta"] = float(np.nanstd(de)) if de is not None else np.nan
    return out


def plot_burst_check(res, t, path, title=""):
    import matplotlib.pyplot as plt
    from . import plots as P
    n = len(res["w4"])
    tt = np.arange(n) / 2.0
    fig, axs = plt.subplots(3, 1, figsize=(11, 7), sharex=True, layout="constrained")
    ax = axs[0]
    if "eta" in res:
        ax.plot(tt, res["eta"], color=P.C1)
    ax.set_ylabel("m")
    ax.set_title("Superficie η dal beam verticale (eco)")
    ax = axs[1]
    if "deta_dt" in res:
        ax.plot(tt, res["deta_dt"], color=P.C1, label="dη/dt (superficie)")
    ax.plot(tt, res["w4"], color=P.C2, label=f"w dai 4 beam, cella {res['cell']} ({res['range']:.1f} m)")
    ax.set_ylabel("m/s")
    ax.legend(loc="lower right", bbox_to_anchor=(1, 1.0), ncols=2, borderaxespad=0.2)
    ax.set_title(f"corr(w 4 beam, dη/dt) = {res['c_w4_deta']:+.2f}")
    ax = axs[2]
    if "deta_dt" in res:
        ax.plot(tt, res["deta_dt"], color=P.C1, label="dη/dt (superficie)")
    if "wvb" in res:
        ax.plot(tt, res["wvb"], color=P.C3, label=f"velocita' beam verticale, cella {res['vb_cell']}")
    ax.set_ylabel("m/s")
    ax.set_xlabel("tempo dall'inizio del burst [s]")
    ax.legend(loc="lower right", bbox_to_anchor=(1, 1.0), ncols=2, borderaxespad=0.2)
    ax.set_title(f"corr(w beam verticale, dη/dt) = {res['c_vb_deta']:+.2f}   "
                 f"(correlazione mediana vb {res.get('vb_corr_median', np.nan):.0f} counts)")
    for a in axs:
        a.set_xlim(0, min(tt[-1], 120))
    fig.suptitle(title or f"Controllo trasformazione, burst {str(t)[:16]} (primi 120 s)",
                 x=0.01, ha="left", fontweight="bold")
    P._save(fig, path)


def check_transform(reader, out_dir=None, n_bursts=10, min_pings=256, orientation=None,
                    corr_min=64, verbose=True):
    """Esegue ``burst_check`` sui burst lunghi più energetici e (opzionale) salva il grafico."""
    import os
    from .waves import find_bursts
    t = reader.times()
    b, dt = find_bursts(t)
    b = b[(b[:, 1] - b[:, 0]) >= min_pings]
    if len(b) == 0:
        return None
    # energia: varianza della pressione non serve in acqua profonda -> usa la superficie;
    # per non leggere tutto, si stima dall'eco del beam verticale su un ping ogni 4
    energy = []
    for s, e in b:
        ds = reader.load(["vb_echo", "pressure"], t[s], t[e - 1] + np.timedelta64(1, "us"))
        d_s = float(np.nanmean(ds.pressure.values)) * 1e4 / (RHO * G)
        r = surface_from_vb(ds.vb_echo.values[::4], ds.vb_range.values, np.full(len(ds.time[::4]), d_s))
        energy.append(np.nanstd(_despike(r)))
    order = np.argsort(np.nan_to_num(energy, nan=-1))[::-1][:n_bursts]
    names = ["vel", "corr", "vb_vel", "vb_corr", "vb_echo", "pressure", "heading", "pitch", "roll"]
    rows = []
    best = None
    for k, j in enumerate(order):
        s, e = b[j]
        ds = reader.load(names, t[s], t[e - 1] + np.timedelta64(1, "us"))
        res = burst_check(ds, reader.attrs, orientation=orientation, corr_min=corr_min, fs=1 / dt)
        res["time"] = t[s]
        res["Hs_est"] = 4 * energy[j]
        rows.append(res)
        if k == 0:
            best = res
    if verbose:
        print(f"  controllo su {len(rows)} burst piu' energetici (Hm0 stimata "
              f"{min(r['Hs_est'] for r in rows):.2f}-{max(r['Hs_est'] for r in rows):.2f} m):")
        print("     burst              Hm0   corr(w4,dη/dt) corr(wvb,dη/dt) corr(w4,wvb)  std w4  std wvb  std dη/dt [cm/s]")
        for r in rows:
            print(f"     {str(r['time'])[:16]}  {r['Hs_est']:4.2f}   {r['c_w4_deta']:+6.2f}         "
                  f"{r['c_vb_deta']:+6.2f}         {r['c_w4_vb']:+6.2f}     {r['std_w4'] * 100:5.1f}  "
                  f"{r['std_vb'] * 100:6.1f}   {r['std_deta'] * 100:6.1f}")
        c4 = np.nanmedian([r["c_w4_deta"] for r in rows])
        cv = np.nanmedian([r["c_vb_deta"] for r in rows])
        print(f"  mediana: corr(w 4 beam, dη/dt) = {c4:+.2f} -> "
              + ("trasformazione/orientamento OK" if c4 > 0.5 else
                 "SEGNO OPPOSTO: orientamento o segno della trasformazione da verificare" if c4 < -0.5 else
                 "debole: da verificare"))
        print(f"           corr(w beam verticale, dη/dt) = {cv:+.2f} -> "
              + ("velocita' del beam verticale coerente" + (" (segno invertito: positiva verso il trasduttore)" if cv < 0 else "")
                 if abs(cv) > 0.5 else "velocita' del beam verticale NON coerente con la superficie (rumore/ambiguita'?)"))
    if out_dir and best is not None:
        os.makedirs(out_dir, exist_ok=True)
        plot_burst_check(best, best["time"], os.path.join(out_dir, "check_transform.png"))
    return rows
