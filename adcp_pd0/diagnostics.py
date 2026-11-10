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
from scipy import signal

from .currents import RHO, G
from .transform import to_earth
from .waves import surface_from_vb, _despike, _interp_gaps


def _corr(x, y):
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 50 or np.std(x[ok]) == 0 or np.std(y[ok]) == 0:
        return np.nan
    return float(np.corrcoef(x[ok], y[ok])[0, 1])


def _coh_phase(x, y, fs, nseg=128):
    """Coerenza al quadrato e fase (gradi) fra x e y in funzione della frequenza."""
    x = _interp_gaps(np.asarray(x, float), max_frac=0.3)
    y = _interp_gaps(np.asarray(y, float), max_frac=0.3)
    if x is None or y is None:
        return None
    kw = dict(fs=fs, nperseg=nseg, noverlap=nseg // 2, window="hann")
    f, Pxy = signal.csd(signal.detrend(x), signal.detrend(y), **kw)
    _, Pxx = signal.welch(signal.detrend(x), **kw)
    _, Pyy = signal.welch(signal.detrend(y), **kw)
    coh = np.abs(Pxy) ** 2 / (Pxx * Pyy)
    return f, coh, np.degrees(np.angle(Pxy)), Pxx, Pyy


def _band_stats(cp, band):
    if cp is None or not band.any():
        return np.nan, np.nan
    f, coh, ph, Pxx, Pyy = cp
    w = np.sqrt(Pxx * Pyy)[band]
    c = float(np.sum(coh[band] * w) / np.sum(w))
    # fase media pesata (circolare)
    z = np.sum(w * np.exp(1j * np.deg2rad(ph[band])))
    return c, float(np.degrees(np.angle(z)))


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
    # coerenza e fase intorno al picco dello spettro di η
    if de is not None:
        f, Se = signal.welch(signal.detrend(out["eta"]), fs=fs, nperseg=128, noverlap=64)
        b = (f >= 0.04) & (f <= 0.5)
        fp = float(f[b][np.argmax(Se[b])])
        band = (f >= fp - 0.025) & (f <= fp + 0.025)
        out["fp"] = fp
        out["cp_w4"] = _coh_phase(out["w4"], de, fs)
        out["coh_w4"], out["ph_w4"] = _band_stats(out["cp_w4"], band)
        if "wvb" in out:
            out["cp_vb"] = _coh_phase(out["wvb"], de, fs)
            out["coh_vb"], out["ph_vb"] = _band_stats(out["cp_vb"], band)
        cu = _coh_phase(out["u"], out["eta"], fs)
        cv = _coh_phase(out["v"], out["eta"], fs)
        out["cp_uh"] = None
        if cu is not None and cv is not None:
            # coerenza "orizzontale": quota di varianza di (u, v) coerente con η
            num = cu[1] * cu[3] + cv[1] * cv[3]
            out["cp_uh"] = (cu[0], num / (cu[3] + cv[3]), cu[2], cu[3] + cv[3], cu[4])
            out["coh_uh"], _ = _band_stats(out["cp_uh"], band)
    return out


def plot_burst_check(res, t, path, title=""):
    import matplotlib.pyplot as plt
    from . import plots as P
    n = len(res["w4"])
    tt = np.arange(n) / 2.0
    fig = plt.figure(figsize=(11, 9.5), layout="constrained")
    gs = fig.add_gridspec(4, 1)
    axs = [fig.add_subplot(gs[0])]
    axs += [fig.add_subplot(gs[i], sharex=axs[0]) for i in (1, 2)]
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
    ax = fig.add_subplot(gs[3])
    for key, c, lab in (("cp_w4", P.C2, "w 4 beam vs dη/dt"), ("cp_vb", P.C3, "w beam verticale vs dη/dt"),
                        ("cp_uh", P.C1, "velocita' orizzontali vs η")):
        cp = res.get(key)
        if cp is not None:
            ax.plot(cp[0], cp[1], color=c, label=lab)
    if np.isfinite(res.get("fp", np.nan)):
        ax.axvline(res["fp"], color=P.MUTED, ls="--", lw=1)
        ax.annotate(f"picco {1 / res['fp']:.1f} s", (res["fp"], 0.95), xytext=(4, 0),
                    textcoords="offset points", color=P.INK2, va="top")
    ax.set_xlim(0, 0.5)
    ax.set_ylim(0, 1)
    ax.set_xlabel("frequenza [Hz]")
    ax.set_ylabel("coerenza²")
    ax.set_title("Coerenza con la superficie")
    ax.legend(loc="upper right", ncols=3)
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
        print("     burst              Hm0   Tp   | corr(w4,dη/dt) corr(wvb,dη/dt) | al picco: coh²(w4,dη/dt) fase  "
              "coh²(wvb,dη/dt) fase  coh²(u,v;η)")
        for r in rows:
            print(f"     {str(r['time'])[:16]}  {r['Hs_est']:4.2f} {1 / r.get('fp', np.nan):4.1f}  |  {r['c_w4_deta']:+6.2f}"
                  f"         {r['c_vb_deta']:+6.2f}        |    {r.get('coh_w4', np.nan):5.2f}       "
                  f"{r.get('ph_w4', np.nan):+5.0f}     {r.get('coh_vb', np.nan):5.2f}       {r.get('ph_vb', np.nan):+5.0f}"
                  f"     {r.get('coh_uh', np.nan):5.2f}")
        def circ_med(a):
            a = np.deg2rad([x for x in a if np.isfinite(x)])
            return float(np.degrees(np.angle(np.mean(np.exp(1j * a))))) if len(a) else np.nan
        c4 = np.nanmedian([r.get("coh_w4", np.nan) for r in rows])
        p4 = circ_med([r.get("ph_w4", np.nan) for r in rows])
        cv = np.nanmedian([r.get("coh_vb", np.nan) for r in rows])
        pv = circ_med([r.get("ph_vb", np.nan) for r in rows])
        ch = np.nanmedian([r.get("coh_uh", np.nan) for r in rows])
        print(f"  al picco (mediana): coh²(w 4 beam, dη/dt) = {c4:.2f}, fase {p4:+.0f} gradi -> "
              + ("trasformazione/orientamento OK (in fase)" if c4 > 0.3 and abs(p4) < 45 else
                 "SEGNO OPPOSTO: orientamento/segno da verificare" if c4 > 0.3 and abs(p4) > 135 else
                 "coerenza bassa: non conclusivo"))
        print(f"                      coh²(w beam verticale, dη/dt) = {cv:.2f}, fase {pv:+.0f} gradi -> "
              + (("coerente" + (", segno invertito (positiva verso il trasduttore)" if abs(pv) > 135 else ""))
                 if cv > 0.3 else "poco coerente"))
        print(f"                      coh²(velocita' orizzontali, η) = {ch:.2f} "
              "(alta = le velocita' orizzontali vedono le onde: base della direzione PUV)")
    if out_dir and best is not None:
        os.makedirs(out_dir, exist_ok=True)
        plot_burst_check(best, best["time"], os.path.join(out_dir, "check_transform.png"))
    return rows
