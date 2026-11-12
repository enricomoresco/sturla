"""Figure del post-processing (PNG). Ogni funzione riceve dati già preparati e scrive un file;
restituisce il percorso, oppure None se i dati non bastano."""
from __future__ import annotations

import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm
from matplotlib.ticker import NullFormatter
from scipy import signal

from .plots import (C1, C2, C3, INK, INK2, MUTED, AXIS, SURF, SEQ, DIV, MESI, _bin_series, _fmt_time, _save,
                    data_it)
from .stats import G, wave_good, dir_good, good_vel, hp_segment, progressive_vector

C4, C_OUT = "#6250d6", "#e34948"
CLASS_COLORS = ["#b7d3f6", "#6da7ec", "#256abf", "#0d366b"]      # classi ordinate: chiaro -> scuro
H12 = pd.Timedelta(hours=12)
LAYERS = (("sup", "superficie", C2), ("meta", "metà colonna", C4), ("fondo", "fondo", C3))


# ------------------------------------------------------------------ utilità
def _motion_lines(axs, dep):
    evs = (dep or {}).get("motion_events", [])
    for ev in evs:
        for ax in np.atleast_1d(axs).ravel():
            ax.axvline(pd.Timestamp(ev["start"]), color=INK2, ls="--", lw=0.8, zorder=1)
    return len(evs)


def _motion_spans(axs, dep):
    for ev in (dep or {}).get("motion_events", []):
        for ax in np.atleast_1d(axs).ravel():
            ax.axvspan(pd.Timestamp(ev["start"]), pd.Timestamp(ev["end"]), color=C2, alpha=0.3, lw=0)


def _legend_below(ax, handles=None, labels=None, ncol=3, y=-0.14):
    if handles is None:
        handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, y), ncol=ncol, fontsize=8.5)


def _note(ax, text, y=-0.32):
    ax.text(0.0, y, text, transform=ax.transAxes, va="top", color=MUTED, fontsize=8)


def _daily(s):
    return s.resample("1D").mean()


def _inpos_span(c):
    """Primo e ultimo istante in posizione di un prodotto L2 delle correnti (None se non ce ne sono)."""
    inpos = c["deployment_qc"].values == 1
    if not inpos.any():
        return None
    t = c["time"].values[inpos]
    return t[0], t[-1]


def _zmax(c):
    """Limite superiore dell'asse delle quote: poco sopra la superficie."""
    inpos = c["deployment_qc"].values == 1
    if "water_depth" in c and inpos.any() and np.isfinite(c["water_depth"].values[inpos]).any():
        return float(np.nanmax(c["water_depth"].values[inpos])) + 1.0
    return float(c["z"].values.max()) + 0.5


def _rose(ax, direction, value, edges, labels, title="", n=16):
    d, v = np.asarray(direction, float), np.asarray(value, float)
    ok = np.isfinite(d) & np.isfinite(v)
    width = 2 * np.pi / n
    tot = max(int(ok.sum()), 1)
    k = (np.floor((d[ok] + 180.0 / n) / (360.0 / n)).astype(int)) % n
    bottom = np.zeros(n)
    for j in range(len(edges) - 1):
        m = (v[ok] >= edges[j]) & (v[ok] < edges[j + 1])
        h = np.bincount(k[m], minlength=n) / tot * 100.0
        ax.bar(np.arange(n) * width, h, width=width * 0.92, bottom=bottom, color=CLASS_COLORS[j % 4],
               edgecolor=SURF, linewidth=0.6, label=labels[j])
        bottom += h
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_xticks(np.arange(8) * np.pi / 4)
    ax.set_xticklabels(["N", "NE", "E", "SE", "S", "SO", "O", "NO"])
    ax.set_rlabel_position(112)
    ax.tick_params(axis="y", labelsize=7)
    ax.set_title(title, loc="center", pad=14)
    return int(ok.sum())


# ------------------------------------------------------------------ quadro generale, assetto, qualità
def fig_overview(path, did, w=None, layers=None, T=None, level=None, dep=None):
    panels = []
    if w is not None and wave_good(w).any():
        panels.append("hm0")
    if layers is not None and layers["vel_media"].notna().any():
        panels.append("cur")
    if T is not None and T.notna().any():
        panels.append("temp")
    if level is not None and level.notna().any():
        panels.append("lev")
    if not panels:
        return None
    fig, axs = plt.subplots(len(panels), 1, figsize=(13, 2.3 * len(panels) + 0.7), sharex=True, layout="constrained")
    axs = np.atleast_1d(axs)
    for ax, k in zip(axs, panels):
        if k == "hm0":
            g = wave_good(w)
            t, y = w["time"].values[g], w["Hm0"].values[g]
            ax.fill_between(t, 0, y, color=C2, alpha=0.15, lw=0)
            ax.plot(t, y, color=C2, lw=0.9)
            ax.set_ylabel("m")
            ax.set_title("Altezza d'onda significativa Hm0")
            ax.set_ylim(0, None)
        elif k == "cur":
            s = layers["vel_media"]
            d = _daily(s)
            ax.plot(s.index, s.values, color=C3, lw=0.5, alpha=0.35)
            ax.plot(d.index + H12, d.values, color=C3, lw=1.8)
            ax.set_ylabel("m/s")
            ax.set_title("Intensità della corrente media sulla verticale (linea spessa: media giornaliera)")
            ax.set_ylim(0, None)
        elif k == "temp":
            ax.plot(T.index, T.values, color=C1, lw=0.8)
            ax.set_ylabel("°C")
            ax.set_title("Temperatura dell'acqua al fondo (media oraria)")
        else:
            ax.plot(level.index, 100 * level.values, color=INK2, lw=0.6)
            ax.set_ylabel("cm")
            ax.set_title("Livello del mare dalla pressione (scarto dalla media)")
    n = _motion_lines(axs, dep)
    _fmt_time(axs[-1])
    fig.suptitle(f"Quadro generale - {did}" + ("   (linee tratteggiate: movimenti del telaio)" if n else ""),
                 x=0.01, ha="left", fontweight="bold")
    _save(fig, path)
    return path


def fig_attitude(path, c, dep=None, w=None):
    inpos = c["deployment_qc"].values == 1
    if not inpos.any():
        return None
    t = c["time"].values[inpos]
    rows = [(k, lab) for k, lab in (("heading", "Heading (°)"), ("pitch", "Pitch (°)"), ("roll", "Roll (°)"))
            if k in c]
    has_w = w is not None and wave_good(w).any()
    n = len(rows) + int(has_w)
    fig, axs = plt.subplots(n, 1, figsize=(13, 2.1 * n + 0.6), sharex=True, layout="constrained")
    axs = np.atleast_1d(axs)
    i = 0
    if has_w:
        g = wave_good(w)
        axs[0].plot(w["time"].values[g], w["Hm0"].values[g], color=C2, lw=0.9)
        axs[0].set_ylabel("Hm0 (m)")
        axs[0].set_ylim(0, None)
        i = 1
    for (k, lab), ax in zip(rows, axs[i:]):
        ax.plot(t, c[k].values[inpos], ".", ms=2, color=C1)
        ax.set_ylabel(lab)
    _motion_spans(axs, dep)
    nev = len((dep or {}).get("motion_events", []))
    axs[0].set_title("Assetto dello strumento" + (f" (fasce arancioni: {nev} movimenti del telaio)" if nev else
                                                  " (nessun movimento del telaio rilevato)"))
    _fmt_time(axs[-1])
    _save(fig, path)
    return path


def fig_quality(path, c=None, temp=None, w=None, qc=None):
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.4), layout="constrained", gridspec_kw=dict(width_ratios=[3, 2]))
    ax = axs[0]
    if c is not None:
        t = pd.DatetimeIndex(c["time"].values)
        inpos = c["deployment_qc"].values == 1
        util = ((c["vel_qc_cause"].values.astype(int) & 4) == 0) & inpos[:, None]
        nu = util.sum(axis=1)
        for vals, ls, lab in (((1, 3), "-", "correnti: celle utili con flag 1 o 3 (utilizzabili)"),
                              ((1,), ":", "correnti: celle utili con flag 1 (buone)")):
            good = np.isin(c["vel_qc"].values, vals) & util
            s = pd.Series(np.where(nu > 0, 100.0 * good.sum(axis=1) / np.maximum(nu, 1), np.nan), index=t)
            d = _daily(s)
            ax.plot(d.index + H12, d.values, color=C1, ls=ls, label=lab)
    if w is not None:
        q = w["wave_qc"].values
        s = pd.Series(np.where(q != 4, 100.0 * (q == 1), np.nan), index=pd.DatetimeIndex(w["time"].values))
        d = _daily(s)
        ax.plot(d.index + H12, d.values, color=C2, label="onde: burst con flag 1")
    if temp is not None:
        q = temp["temperature_qc"].values
        s = pd.Series(np.where(q != 4, 100.0 * (q == 1), np.nan), index=pd.DatetimeIndex(temp["time"].values))
        d = _daily(s)
        ax.plot(d.index + H12, d.values, color=C3, label="temperatura: ore con flag 1")
    ax.set_ylim(0, 105)
    ax.set_ylabel("% al giorno")
    ax.set_title("Dati validi nel periodo in posizione (celle utili: sotto il limite della superficie)")
    _legend_below(ax, ncol=2, y=-0.12)
    _fmt_time(ax)
    ax = axs[1]
    st = (qc or {}).get("velocita")
    if st:
        items = [(k, 100.0 * v / st["n"]) for k, v in st["per_causa"].items() if v > 0]
        items.sort(key=lambda x: x[1])
        ax.barh([k.replace("_", " ") for k, _ in items], [v for _, v in items], color=C1, height=0.6)
        for i, (_, v) in enumerate(items):
            ax.text(v, i, f" {v:.1f}%", va="center", fontsize=8, color=INK2)
        ax.set_xlabel("% di tutte le celle")
        ax.set_xlim(0, max(v for _, v in items) * 1.25 if items else 1)
        ax.grid(False, axis="y")
    ax.set_title("Velocità: cause dei flag")
    _save(fig, path)
    return path


# ------------------------------------------------------------------ correnti
def fig_current_sections(path, c):
    ok = good_vel(c)
    if not ok.any():
        return None
    t, z = c["time"].values, c["z"].values
    fields = [("u", "Verso Est (m/s)", DIV, True), ("v", "Verso Nord (m/s)", DIV, True),
              ("speed", "Intensità (m/s)", SEQ, False)]
    fig, axs = plt.subplots(3, 1, figsize=(13, 8.6), sharex=True, layout="constrained")
    inpos = c["deployment_qc"].values == 1
    for ax, (k, lab, cmap, sym) in zip(axs, fields):
        tb, Z = _bin_series(t, np.where(ok, c[k].values, np.nan), 1500)
        Z = np.atleast_2d(Z)
        if sym:
            vmax = float(np.nanpercentile(np.abs(Z), 99))
            kw = dict(vmin=-vmax, vmax=vmax)
        else:
            kw = dict(vmin=0, vmax=float(np.nanpercentile(Z, 99)))
        pc = ax.pcolormesh(tb, z, Z.T, cmap=cmap, shading="nearest", rasterized=True, **kw)
        fig.colorbar(pc, ax=ax, pad=0.01, aspect=15).set_label(lab, color=INK2)
        if "water_depth" in c:
            tb2, d = _bin_series(t, np.where(inpos, c["water_depth"].values, np.nan), 1500)
            ax.plot(tb2, d, color=INK, lw=0.8)
        ax.set_ylabel("Quota sul fondale (m)")
        ax.set_ylim(0, _zmax(c))
        ax.grid(False)
    span = _inpos_span(c)
    if span:
        axs[-1].set_xlim(*span)
    axs[0].set_title("Correnti: sezioni tempo-quota (linea nera: superficie; in grigio i dati non validi)")
    _fmt_time(axs[-1])
    _save(fig, path)
    return path


def fig_current_profile(path, c):
    ok = good_vel(c)
    inpos = c["deployment_qc"].values == 1
    if not ok.any() or not inpos.any():
        return None
    z = c["z"].values
    vf = ok[inpos].mean(axis=0)
    fig, axs = plt.subplots(1, 2, figsize=(9.5, 6), sharey=True, layout="constrained",
                            gridspec_kw=dict(width_ratios=[3, 1.2]))
    ax = axs[0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for k, col, lab in (("u", C1, "verso Est"), ("v", C2, "verso Nord"), ("speed", C3, "intensità")):
            A = np.where(ok, c[k].values, np.nan)
            m, sd = np.nanmean(A, axis=0), np.nanstd(A, axis=0)
            m[vf < 0.8] = np.nan
            sd[vf < 0.8] = np.nan
            ax.fill_betweenx(z, m - sd, m + sd, color=col, alpha=0.12, lw=0)
            ax.plot(m, z, color=col, lw=2, label=lab)
    ax.axvline(0, color=AXIS, lw=0.8)
    ax.set_xlabel("m/s")
    ax.set_ylabel("Quota sul fondale (m)")
    ax.set_title("Profilo medio della corrente (fascia: ±1 deviazione standard; quote con almeno 80% di ore valide)")
    ax.legend(loc="lower right")
    axs[1].plot(100 * vf, z, color=INK2)
    axs[1].set_xlabel("% di ore valide")
    axs[1].set_xlim(0, 105)
    axs[1].set_title("Dati validi")
    ax.set_ylim(0, _zmax(c))
    _save(fig, path)
    return path


def fig_current_roses(path, layers):
    edges = [0, 0.05, 0.10, 0.20, np.inf]
    labels = ["< 0.05 m/s", "0.05-0.10", "0.10-0.20", "> 0.20 m/s"]
    fig, axs = plt.subplots(1, 3, figsize=(13, 5), subplot_kw=dict(projection="polar"), layout="constrained")
    any_ = False
    for ax, (k, nme, _) in zip(axs, LAYERS):
        n = _rose(ax, layers[f"dir_{k}"].values, layers[f"vel_{k}"].values, edges, labels,
                  title=f"{nme.capitalize()}\n{layers.attrs.get(k, '')}")
        any_ |= n > 0
    if not any_:
        plt.close(fig)
        return None
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="outside lower center", ncol=4, fontsize=8.5)
    fig.suptitle("Rose delle correnti: direzione VERSO cui va la corrente (% delle ore)", x=0.01, y=1.07, ha="left",
                 fontweight="bold")
    _save(fig, path)
    return path


def fig_current_layers(path, layers, pa=None, dep=None):
    if not layers[["vel_sup", "vel_meta", "vel_fondo"]].notna().any().any():
        return None
    ang = (pa or {}).get("direzione_gradi")
    n = 2 if ang is not None else 1
    fig, axs = plt.subplots(n, 1, figsize=(13, 3.1 * n + 0.4), sharex=True, layout="constrained")
    axs = np.atleast_1d(axs)
    for k, nme, col in LAYERS:
        d = _daily(layers[f"vel_{k}"])
        axs[0].plot(d.index + H12, d.values, color=col, lw=1.6, label=f"{nme} ({layers.attrs.get(k, '')})")
    axs[0].set_ylabel("m/s")
    axs[0].set_ylim(0, None)
    axs[0].set_title("Intensità della corrente a tre quote (medie giornaliere)")
    fig.legend(*axs[0].get_legend_handles_labels(), loc="outside lower center", ncol=3, fontsize=8.5)
    if ang is not None:
        a = np.deg2rad(ang)
        for k, nme, col in LAYERS:
            comp = layers[f"u_{k}"] * np.sin(a) + layers[f"v_{k}"] * np.cos(a)
            d = _daily(comp)
            axs[1].plot(d.index + H12, d.values, color=col, lw=1.6)
        axs[1].axhline(0, color=AXIS, lw=0.8)
        axs[1].set_ylabel("m/s")
        axs[1].set_title(f"Componente lungo l'asse principale: positiva verso {ang:.0f}°, negativa verso "
                         f"{(ang + 180) % 360:.0f}° (medie giornaliere)")
    _motion_lines(axs, dep)
    _fmt_time(axs[-1])
    _save(fig, path)
    return path


def fig_current_spectrum(path, spec, bands=None, label=""):
    if not spec:
        return None
    fig, ax = plt.subplots(figsize=(8.5, 5), layout="constrained")
    ax.loglog(spec["f"], spec["orario"], color=C1, label="rotazione oraria")
    ax.loglog(spec["f"], spec["antiorario"], color=C2, label="rotazione antioraria")
    marks = [(1.0, "diurna (K1, O1)"), (1.9323, "semidiurna (M2)")]
    fi = (bands or {}).get("frequenza_inerziale_cpd")
    if fi:
        marks.append((fi, f"inerziale ({24 / fi:.1f} h)"))
    ymax = ax.get_ylim()[1]
    for x, lab in marks:
        ax.axvline(x, color=MUTED, ls="--", lw=0.8)
        ax.text(x, ymax, lab + " ", rotation=90, va="top", ha="right", fontsize=8, color=INK2)
    ax.set_xlabel("Frequenza (cicli al giorno)")
    ax.set_ylabel("Densità spettrale ((m/s)² per ciclo al giorno)")
    ax.set_title(f"Spettro rotatorio della corrente{(' - ' + label) if label else ''}")
    ax.legend(loc="lower left")
    _save(fig, path)
    return path


def fig_current_pvd(path, layers):
    """Vettore progressivo a tre quote: spostamento cumulato dell'acqua con la corrente misurata."""
    ok = layers[["u_sup", "u_meta", "u_fondo"]].notna().any(axis=1).values
    if ok.sum() < 24:
        return None
    i0, i1 = int(np.argmax(ok)), int(len(ok) - np.argmax(ok[::-1]))
    L = layers.iloc[i0:i1]
    months = pd.Series(np.arange(len(L)), index=L.index).groupby(L.index.to_period("M")).first()
    fig, ax = plt.subplots(figsize=(9, 7), layout="constrained")
    for k, nme, col in LAYERS:
        x, y, frac = progressive_vector(L[f"u_{k}"].values, L[f"v_{k}"].values, L.index)
        ax.plot(x, y, color=col, lw=1.6,
                label=f"{nme} ({layers.attrs.get(k, '')}; ore valide {100 * frac:.0f}%)")
        for per, i in months.items():
            if i == 0:
                continue
            ax.plot(x[i], y[i], "o", ms=4, color=col)
            if k == LAYERS[0][0]:           # etichette dei mesi solo sul primo strato, per leggibilita'
                ax.annotate(f"1 {MESI[per.month - 1]}", (x[i], y[i]), xytext=(4, 5), textcoords="offset points",
                            fontsize=7.5, color=INK2)
        ax.plot(x[-1], y[-1], "D", ms=4.5, color=col)
    ax.plot(0, 0, "s", ms=6, color=INK)
    ax.annotate(f"inizio ({data_it(L.index[0])})", (0, 0), xytext=(8, -16), textcoords="offset points", fontsize=8,
                color=INK2)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("Spostamento verso Est (km)")
    ax.set_ylabel("Spostamento verso Nord (km)")
    ax.set_title("Vettore progressivo: spostamento cumulato dell'acqua con la corrente misurata nel punto\n"
                 f"(quadrato: inizio; rombo: fine, {data_it(L.index[-1], year=True)}; punti: primo giorno di ogni mese)")
    fig.legend(*ax.get_legend_handles_labels(), loc="outside lower center", ncol=1, fontsize=8.5)
    _save(fig, path)
    return path


def fig_echo_section(path, c, th=None, kc=0.45, alpha=0.16):
    if "echo" not in c:
        return None
    ok = good_vel(c)
    if not ok.any():
        return None
    t, z = c["time"].values, c["z"].values
    hab = float(c.attrs.get("instrument_height_above_bed_m", 0.6))
    beam = float(c.attrs.get("instrument_beam_angle_deg", 25.0))
    rs = np.maximum(z - hab, 1e-3) / np.cos(np.deg2rad(beam))
    E = np.where(ok, c["echo"].values, np.nan)
    noise = float(np.nanpercentile(E, 1))
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        ps = 10 ** (kc * E / 10.0) - 10 ** (kc * noise / 10.0)
        ps[ps <= 0] = np.nan
        sv = 10 * np.log10(ps) + (20 * np.log10(rs) + 2 * alpha * rs)[None, :]
        an = sv - np.nanmean(sv, axis=0)[None, :]
    tb, A = _bin_series(t, an, 1500)
    A = np.atleast_2d(A)
    if not np.isfinite(A).any():
        return None
    vmax = float(np.nanpercentile(np.abs(A), 98))
    fig, ax = plt.subplots(figsize=(13, 4.4), layout="constrained")
    pc = ax.pcolormesh(tb, z, A.T, cmap=DIV, vmin=-vmax, vmax=vmax, shading="nearest", rasterized=True)
    fig.colorbar(pc, ax=ax, pad=0.01, aspect=15).set_label("dB rispetto al profilo medio", color=INK2)
    if th is not None and len(th) and "z_eco" in th:
        m = th["z_eco"].rolling(7, center=True, min_periods=3).median()
        ax.plot(m.index + H12, m.values, color=INK, lw=1.2, label="picco dell'eco (mediana mobile 7 giorni)")
        ax.legend(loc="upper left", fontsize=8.5)
    ax.set_ylabel("Quota sul fondale (m)")
    ax.set_ylim(0, _zmax(c))
    span = _inpos_span(c)
    if span:
        ax.set_xlim(*span)
    ax.set_title("Eco corretto per spargimento e assorbimento: anomalia rispetto al profilo medio")
    ax.grid(False)
    _fmt_time(ax)
    _save(fig, path)
    return path


# ------------------------------------------------------------------ onde
def fig_wave_series(path, w, dep=None):
    g = wave_good(w)
    if not g.any():
        return None
    t = w["time"].values
    rows = ["hm0", "per"] + (["dir"] if "Dp" in w else []) + (["ub"] if "Ub" in w else [])
    fig, axs = plt.subplots(len(rows), 1, figsize=(13, 2.3 * len(rows) + 0.5), sharex=True, layout="constrained")
    axs = np.atleast_1d(axs)
    for ax, k in zip(axs, rows):
        if k == "hm0":
            ax.fill_between(t[g], 0, w["Hm0"].values[g], color=C2, alpha=0.15, lw=0)
            ax.plot(t[g], w["Hm0"].values[g], color=C2, lw=0.9)
            ax.set_ylabel("m")
            ax.set_ylim(0, None)
            ax.set_title(f"Altezza d'onda significativa Hm0 ({w.attrs.get('proc_wave_method', w.attrs.get('wave_method', ''))})")
        elif k == "per":
            ax.plot(t[g], w["Tp"].values[g], ".", ms=3, color=C1)
            ax.plot(t[g], w["Tm02"].values[g], color=C2, lw=0.9)
            ax.set_ylabel("s")
            tp = w["Tp"].values[g]
            if np.isfinite(tp).any():
                ax.set_ylim(0, float(np.nanpercentile(tp, 99.5)) * 1.25)
            ax.set_title("Periodi: Tp di picco (punti blu) e Tm02 medio (linea arancione)")
        elif k == "dir":
            gd = dir_good(w)
            other = g & np.isfinite(w["Dp"].values) & ~gd
            ax.plot(t[other], w["Dp"].values[other], ".", ms=3, color=AXIS)
            ax.plot(t[gd], w["Dp"].values[gd], ".", ms=4, color=C1)
            ax.set_ylim(0, 360)
            ax.set_yticks([0, 90, 180, 270, 360])
            ax.set_ylabel("° di provenienza")
            ax.set_title("Direzione di provenienza al picco (blu: affidabile, flag 1; grigio: sospetta per poca "
                         "energia o poca coerenza)")
        else:
            ax.plot(t[g], w["Ub"].values[g], color=C3, lw=0.9)
            ax.set_ylabel("m/s")
            ax.set_ylim(0, None)
            ax.set_title("Velocità orbitale al fondo (ampiezza significativa, teoria lineare)")
    _motion_lines(axs, dep)
    _fmt_time(axs[-1])
    _save(fig, path)
    return path


def fig_wave_spectrogram(path, w, fmax=0.6):
    g = wave_good(w)
    S = np.where(g[:, None], w["S"].values, np.nan)
    f = w["frequency"].values
    good = np.isfinite(S) & (S > 0)
    if not good.any():
        return None
    vmax = float(np.nanpercentile(S[good], 99.5))
    fig, ax = plt.subplots(figsize=(13, 4.2), layout="constrained")
    pc = ax.pcolormesh(w["time"].values, f, np.where(good, S, np.nan).T, cmap=SEQ, shading="nearest",
                       norm=LogNorm(vmin=vmax / 1e3, vmax=vmax), rasterized=True)
    fig.colorbar(pc, ax=ax, pad=0.01, aspect=15).set_label("S(f) (m²/Hz)", color=INK2)
    ax.set_ylim(0, min(float(f.max()), fmax))
    ax.set_ylabel("Frequenza (Hz)")
    ax.set_title("Spettro di elevazione per burst")
    ax.grid(False)
    _fmt_time(ax)
    _save(fig, path)
    return path


def fig_wave_mean_spectra(path, w, st=None, fmax=0.6):
    g = wave_good(w)
    if not g.any():
        return None
    hm, f, S = w["Hm0"].values, w["frequency"].values, w["S"].values
    t = pd.DatetimeIndex(w["time"].values)
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.8), layout="constrained", sharey=True)
    classes = [(0, 0.5, "Hm0 < 0.5 m"), (0.5, 1.0, "0.5-1.0 m"), (1.0, 1.5, "1.0-1.5 m"), (1.5, np.inf, "> 1.5 m")]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for (a, b, lab), col in zip(classes, CLASS_COLORS):
            m = g & (hm >= a) & (hm < b)
            if m.sum() >= 3:
                axs[0].semilogy(f, np.nanmean(S[m], axis=0), color=col, lw=2, label=f"{lab} ({int(m.sum())} burst)")
    axs[0].set_title("Spettri medi per classe di altezza d'onda")
    axs[0].set_ylabel("S(f) (m²/Hz)")
    axs[0].legend(loc="upper right", fontsize=8.5)
    if st is not None and len(st):
        top = st.sort_values("Hm0_max", ascending=False).head(3)
        for (_, r), col in zip(top.iterrows(), (C1, C2, C3)):
            i = int(np.argmin(np.abs(t - pd.Timestamp(r["quando_max"]))))
            axs[1].semilogy(f, S[i], color=col, lw=2,
                            label=f"{data_it(r['quando_max'], hm=True)}: Hm0 {r['Hm0_max']:.2f} m, "
                                  f"Tp {r['Tp']:.1f} s")
        axs[1].legend(loc="upper right", fontsize=8.5)
    axs[1].set_title("Spettri al culmine delle mareggiate principali")
    for ax in axs:
        ax.set_xlim(0, min(float(f.max()), fmax))
        ax.set_xlabel("Frequenza (Hz)")
    _save(fig, path)
    return path


def fig_wave_rose(path, w):
    gd = dir_good(w) & wave_good(w)
    if gd.sum() < 5:
        return None
    edges = [0, 0.5, 1.0, 1.5, np.inf]
    labels = ["Hm0 < 0.5 m", "0.5-1.0 m", "1.0-1.5 m", "> 1.5 m"]
    fig, ax = plt.subplots(figsize=(6.6, 6.6), subplot_kw=dict(projection="polar"), layout="constrained")
    n = _rose(ax, w["Dp"].values[gd], w["Hm0"].values[gd], edges, labels,
              title="Rosa delle onde: direzione di PROVENIENZA al picco\n(% dei burst con direzione affidabile)")
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, loc="outside lower center", ncol=4, fontsize=8.5)
    ax.text(0.0, -0.08, f"{n} burst", transform=ax.transAxes, color=MUTED, fontsize=8)
    _save(fig, path)
    return path


def fig_wave_hm0_tp(path, w):
    g = wave_good(w) & np.isfinite(w["Tp"].values)
    if g.sum() < 5:
        return None
    tp, hm = w["Tp"].values[g], w["Hm0"].values[g]
    fig, ax = plt.subplots(figsize=(7.5, 5.2), layout="constrained")
    ax.scatter(tp, hm, s=9, color=C1, alpha=0.5, lw=0)
    T = np.linspace(2, max(float(np.nanpercentile(tp, 99.5)) * 1.15, 6.0), 100)   # senza i valori isolati
    ymax = float(np.nanmax(hm)) * 1.15
    for s, lab in ((1 / 20, "ripidità 1/20"), (1 / 40, "ripidità 1/40")):
        H = s * G * T ** 2 / (2 * np.pi)
        ax.plot(T, H, color=MUTED, ls="--", lw=0.9)
        k = int(np.argmin(np.abs(H - 0.9 * ymax)))
        ax.text(T[k], min(H[k], 0.92 * ymax), " " + lab, color=INK2, fontsize=8, va="bottom")
    ax.set_ylim(0, ymax)
    ax.set_xlim(2, float(T.max()))
    ax.set_xlabel("Periodo di picco Tp (s)")
    ax.set_ylabel("Hm0 (m)")
    ax.set_title("Altezza e periodo delle onde (un punto per burst)")
    _save(fig, path)
    return path


def fig_wave_dir_freq(path, w, fmax=0.4, nd=36):
    if "direction_f" not in w:
        return None
    gd = dir_good(w) & wave_good(w)
    if gd.sum() < 5:
        return None
    f = w["frequency"].values.astype(float)
    df = f[1] - f[0]
    fsel = np.flatnonzero((f >= 0.04) & (f <= fmax))
    S, TH = w["S"].values[gd], w["direction_f"].values[gd]
    Eg = np.zeros((len(fsel), nd))
    for j, fi in enumerate(fsel):
        th, s = TH[:, fi], S[:, fi]
        ok = np.isfinite(th) & np.isfinite(s)
        if ok.any():
            k = (th[ok] // (360.0 / nd)).astype(int) % nd
            Eg[j] = np.bincount(k, weights=s[ok] * df, minlength=nd)
    Eg /= max(int(gd.sum()), 1)
    if Eg.max() <= 0:
        return None
    fig, ax = plt.subplots(figsize=(7.2, 7), subplot_kw=dict(projection="polar"), layout="constrained")
    th_e = np.deg2rad(np.linspace(0, 360, nd + 1))
    r_e = np.r_[f[fsel] - df / 2, f[fsel][-1] + df / 2]
    vmin = Eg.max() / 300.0
    pc = ax.pcolormesh(th_e, r_e, np.where(Eg >= vmin, Eg, np.nan), cmap=SEQ, shading="flat",
                       norm=LogNorm(vmin=vmin, vmax=Eg.max()))
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_xticks(np.arange(8) * np.pi / 4)
    ax.set_xticklabels(["N", "NE", "E", "SE", "S", "SO", "O", "NO"])
    ax.set_rlabel_position(22)
    ax.tick_params(axis="y", labelsize=7)
    ax.set_rlim(0, fmax)
    ax.grid(True, lw=0.4)
    fig.colorbar(pc, ax=ax, pad=0.08, shrink=0.7).set_label("energia media per burst (m²)", color=INK2)
    ax.set_title("Energia delle onde per direzione di provenienza e frequenza\n(raggio: frequenza in Hz; "
                 "la direzione è quella media a ogni frequenza)", pad=16)
    _save(fig, path)
    return path


def fig_wave_storms(path, w, st, dep=None, n=3, pad_h=18):
    if st is None or len(st) == 0:
        return None
    top = st.sort_values("Hm0_max", ascending=False).head(n).sort_values("inizio")
    rows = [("Hm0", "Hm0 (m)"), ("Tp", "Tp (s)")] + ([("Dp", "Dp (°)")] if "Dp" in w else []) \
        + ([("Ub", "Ub (m/s)")] if "Ub" in w else [])
    g = wave_good(w)
    gd = dir_good(w) if "Dp" in w else None
    t = pd.DatetimeIndex(w["time"].values)
    fig, axs = plt.subplots(len(rows), len(top), figsize=(4.4 * len(top), 1.9 * len(rows) + 0.7), sharex="col",
                            layout="constrained", squeeze=False)
    pad = pd.Timedelta(hours=pad_h)
    for j, (_, r) in enumerate(top.iterrows()):
        m = g & (t >= r["inizio"] - pad) & (t <= r["fine"] + pad)
        for i, (k, lab) in enumerate(rows):
            ax = axs[i, j]
            if k == "Dp":
                mm = m & gd
                ax.plot(t[mm], w[k].values[mm], "o", ms=3, color=C1)
                ax.set_ylim(0, 360)
                ax.set_yticks([0, 90, 180, 270, 360])
            else:
                ax.plot(t[m], w[k].values[m], "-o", ms=2.5, color=C2 if k == "Hm0" else (C3 if k == "Ub" else C1))
                ax.set_ylim(0, None)
            if j == 0:
                ax.set_ylabel(lab)
        axs[0, j].set_title(f"{data_it(r['quando_max'], year=True)}: Hm0 max {r['Hm0_max']:.2f} m")
        _motion_spans(axs[:, j], dep)
        axs[-1, j].set_xlim(r["inizio"] - pad, r["fine"] + pad)
        _fmt_time(axs[-1, j], maxticks=7)
    fig.suptitle("Mareggiate principali" + (" (fasce arancioni: movimenti del telaio)"
                                           if (dep or {}).get("motion_events") else ""),
                 x=0.01, ha="left", fontweight="bold")
    _save(fig, path)
    return path


def fig_wave_pressure(path, w):
    if "Hm0_pressure" not in w:
        return None
    g = wave_good(w) & np.isfinite(w["Hm0_pressure"].values)
    if g.sum() < 5:
        return None
    x, y = w["Hm0"].values[g], w["Hm0_pressure"].values[g]
    fig, ax = plt.subplots(figsize=(6.2, 5.6), layout="constrained")
    ax.scatter(x, y, s=9, color=C1, alpha=0.5, lw=0)
    lim = float(np.nanmax(x)) * 1.08
    ax.plot([0, lim], [0, lim], color=MUTED, ls="--", lw=0.9)
    ax.text(lim * 0.72, lim * 0.76, "1:1", color=INK2, fontsize=8)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("Hm0 dal beam verticale (m)")
    ax.set_ylabel("Hm0 dalla pressione (m)")
    fc = float(np.nanmedian(w["fcut_pressure"].values[g])) if "fcut_pressure" in w else np.nan
    ax.set_title("Confronto fra Hm0 dalla superficie e dalla pressione"
                 + (f"\n(la pressione vede solo le onde con periodo > {1 / fc:.1f} s)" if np.isfinite(fc) and fc > 0 else ""))
    _save(fig, path)
    return path


def fig_wave_distribution(path, w):
    """Frequenza di superamento di Hm0 e distribuzione del periodo di picco."""
    g = wave_good(w)
    if g.sum() < 20:
        return None
    hm = np.sort(w["Hm0"].values[g])
    exc = 100.0 * (1.0 - np.arange(len(hm)) / len(hm))
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.8), layout="constrained")
    ax = axs[0]
    ax.semilogy(hm, exc, color=C1, lw=2)
    for x in (0.5, 1.0, 1.5, 2.0):
        pc = 100.0 * (hm > x).mean()
        if pc > 0:
            ax.plot(x, pc, "o", ms=5, color=C2)
            ax.annotate(f"{pc:.1f}%" if pc >= 1 else f"{pc:.2f}%", (x, pc), xytext=(6, 4), textcoords="offset points",
                        fontsize=8.5, color=INK2)
    ax.set_xlim(0, float(hm.max()) * 1.05)
    ax.set_ylim(max(100.0 / len(hm) * 0.8, 0.01), 100)
    ax.set_xlabel("Hm0 (m)")
    ax.set_ylabel("% dei burst con Hm0 maggiore")
    ax.set_title(f"Frequenza di superamento dell'altezza d'onda ({len(hm)} burst)")
    ax = axs[1]
    tp = w["Tp"].values
    ok = g & np.isfinite(tp)
    bins = np.arange(2.0, max(12.0, float(np.nanpercentile(tp[ok], 99.5)) + 1.0), 0.5)
    hi = ok & (w["Hm0"].values >= 0.5)
    ax.hist(tp[ok], bins=bins, weights=np.full(ok.sum(), 100.0 / ok.sum()), color=CLASS_COLORS[1],
            label=f"tutti i burst ({int(ok.sum())})")
    if hi.any():
        ax.hist(tp[hi], bins=bins, weights=np.full(hi.sum(), 100.0 / ok.sum()), color=CLASS_COLORS[3],
                label=f"burst con Hm0 ≥ 0.5 m ({int(hi.sum())})")
    ax.set_xlabel("Periodo di picco Tp (s)")
    ax.set_ylabel("% dei burst")
    ax.set_title("Distribuzione del periodo di picco")
    ax.legend(loc="upper right", fontsize=8.5)
    _save(fig, path)
    return path


# ------------------------------------------------------------------ temperatura
def fig_temperature(path, Th):
    T = Th["T"].where(Th["qc"].isin([1, 3]))
    if not T.notna().any():
        return None
    d = T.resample("1D").agg(["mean", "min", "max"])
    fig, ax = plt.subplots(figsize=(13, 4.2), layout="constrained")
    ax.fill_between(d.index + H12, d["min"], d["max"], color=C1, alpha=0.15, lw=0, label="minimo-massimo nel giorno")
    ax.plot(T.index, T.values, color=C1, lw=0.5, alpha=0.5, label="media oraria")
    ax.plot(d.index + H12, d["mean"], color=C1, lw=1.8, label="media giornaliera")
    ax.set_ylabel("°C")
    ax.set_title("Temperatura dell'acqua al fondo")
    ax.legend(loc="upper left", ncols=3, fontsize=8.5)
    _fmt_time(ax)
    _save(fig, path)
    return path


def fig_temp_overlay(path, Th, series, label, ylabel, color, dep=None, title="", note="", fine=True):
    """Temperatura al fondo (media giornaliera, asse sinistro) e una seconda grandezza (asse destro)."""
    Td = Th["T"].where(Th["qc"].isin([1, 3])).resample("1D").mean()
    To = Th["T"].where(Th["qc"] == 4).resample("1D").mean()
    mixed = Td.notna() & To.notna()          # giorni in parte in posizione e in parte no: esclusi
    Td, To = Td[~mixed], To[~mixed]
    if not Td.notna().any() or series is None or not series.notna().any():
        return None
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(Td.index + H12, Td.values, "-o", ms=3, lw=1.8, color=C1,
            label="Temperatura al fondo, media giornaliera (in posizione)")
    if To.notna().any():
        ax.plot(To.index + H12, To.values, "-o", ms=3, lw=1.8, color=C_OUT,
                label="Temperatura, periodo non affidabile (strumento fuori posizione)")
        if dep:
            s, e = pd.Timestamp(dep["start"]), pd.Timestamp(dep["end"])
            lo, hi = Th.index.min(), Th.index.max() + pd.Timedelta(hours=1)
            if e < hi:
                ax.axvspan(e, hi, color=C_OUT, alpha=0.07, lw=0)
            if s > lo:
                ax.axvspan(lo, s, color=C_OUT, alpha=0.07, lw=0)
    for i, ev in enumerate((dep or {}).get("motion_events", [])):
        ax.axvline(pd.Timestamp(ev["start"]), color=INK2, ls="--", lw=1.0,
                   label="Movimento del telaio" if i == 0 else None)
    ax.set_ylabel("Temperatura al fondo (°C)")
    ax.grid(True, axis="y")
    ax.grid(False, axis="x")
    ax2 = ax.twinx()
    ax2.spines["right"].set_visible(True)
    ax2.grid(False)
    if fine:
        ax2.plot(series.index, series.values, color=color, lw=0.9, alpha=0.85, label=label)
    else:
        sd = _daily(series)
        ax2.plot(series.index, series.values, color=color, lw=0.5, alpha=0.3, label=f"{label}, valori orari")
        ax2.plot(sd.index + H12, sd.values, color=color, lw=2, label=f"{label}, media giornaliera")
    ax2.set_ylabel(ylabel)
    ymax = float(np.nanpercentile(series.values, 99.8)) * 1.15
    ax2.set_ylim(0, ymax if ymax > 0 else None)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    _legend_below(ax, h1 + h2, l1 + l2, ncol=3, y=-0.10)
    ax.set_zorder(ax2.get_zorder() + 1)
    ax.patch.set_visible(False)
    ax.set_title(title)
    _fmt_time(ax)
    _note(ax, note + f" Assi y diversi: sinistra temperatura, destra {label[0].lower() + label[1:]}."
          .replace("hm0", "Hm0"), y=-0.27)
    _save(fig, path)
    return path


def fig_temp_zoom(path, Th, zoom, level=None, w=None, layers=None, dep=None):
    if not zoom:
        return None
    t0, t1 = zoom["t0"], zoom["t1"]
    T = Th["T"].where(Th["qc"].isin([1, 3]))[t0:t1]
    if not T.notna().any():
        return None
    rows = ["T"] + (["hm0"] if w is not None and wave_good(w).any() else []) \
        + (["cur"] if layers is not None and len(layers) else [])
    fig, axs = plt.subplots(len(rows), 1, figsize=(13, 2.6 * len(rows) + 1.0), sharex=True, layout="constrained",
                            gridspec_kw=dict(height_ratios=[1.6] + [1] * (len(rows) - 1)))
    axs = np.atleast_1d(axs)
    ax = axs[0]
    ax.plot(T.index, T.values, color=C1, lw=1.2, label="temperatura al fondo, media oraria")
    ax.set_ylabel("°C")
    if level is not None and level[t0:t1].notna().any():
        ax2 = ax.twinx()
        ax2.spines["right"].set_visible(True)
        ax2.grid(False)
        lv = level[t0:t1]
        ax2.plot(lv.index, 100 * lv.values, color=MUTED, lw=0.8, label="livello del mare dalla pressione")
        ax2.set_ylabel("Livello (cm)")
        ax.set_zorder(ax2.get_zorder() + 1)
        ax.patch.set_visible(False)
        sub = " (blu, asse sinistro) e livello del mare dalla pressione (grigio, asse destro)"
    else:
        sub = ""
    ax.set_title(f"Periodo di massima variazione della temperatura al fondo: {zoom['delta']:+.1f} °C fra il "
                 f"{data_it(zoom['da'])} e il {data_it(zoom['a'], year=True)}\nTemperatura oraria{sub}")
    i = 1
    if "hm0" in rows:
        g = wave_good(w)
        tw = pd.DatetimeIndex(w["time"].values)
        m = g & (tw >= t0) & (tw <= t1)
        axs[i].plot(tw[m], w["Hm0"].values[m], color=C2, lw=1.2)
        axs[i].set_ylabel("m")
        axs[i].set_ylim(0, None)
        axs[i].set_title("Altezza d'onda significativa Hm0")
        i += 1
    if "cur" in rows:
        for k, nme, col in (("sup", "superficie", C2), ("fondo", "fondo", C3)):
            s = layers[f"vel_{k}"][t0:t1]
            axs[i].plot(s.index, s.values, color=col, lw=0.9, label=f"corrente: {nme}")
        axs[i].set_ylabel("m/s")
        axs[i].set_ylim(0, None)
        axs[i].set_title("Intensità della corrente in superficie (arancione) e al fondo (verde)")
    _motion_lines(axs, dep)
    axs[-1].set_xlim(pd.Timestamp(t0), pd.Timestamp(t1))
    _fmt_time(axs[-1])
    _save(fig, path)
    return path


def fig_temp_periods(path, Th, zoom, latitude=None):
    if not zoom:
        return None
    T = Th["T"].where(Th["qc"].isin([1, 3]))[zoom["t0"]:zoom["t1"]]
    y = hp_segment(T)
    if y is None:
        return None
    f, P = signal.periodogram(signal.detrend(y), fs=1.0, window="hann")
    per = 1.0 / f[1:]
    P = P[1:]
    m = (per >= 3) & (per <= 72)
    if not m.any() or P[m].max() <= 0:
        return None
    fig, ax = plt.subplots(figsize=(8.5, 4.6), layout="constrained")
    ax.semilogx(per[m], P[m] / P[m].max(), color=C1, lw=1.4)
    marks = [(12.42, "marea semidiurna (12.4 h)"), (24.0, "24 h")]
    if latitude is not None:
        fi = 2 * 1.0027379 * abs(np.sin(np.deg2rad(float(latitude))))
        if fi > 0:
            marks.append((24.0 / fi, f"inerziale ({24 / fi:.1f} h)"))
    for x, lab in marks:
        ax.axvline(x, color=MUTED, ls="--", lw=0.8)
        ax.text(x, 1.02, lab + " ", rotation=90, va="top", ha="right", fontsize=8, color=INK2)
    ax.set_xticks([3, 6, 12, 24, 48, 72])
    ax.set_xticklabels(["3", "6", "12", "24", "48", "72"])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("Periodo (ore)")
    ax.set_ylabel("Potenza (normalizzata al massimo)")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Periodi delle oscillazioni della temperatura al fondo, {data_it(zoom['t0'])} - "
                 f"{data_it(zoom['t1'], year=True)}")
    _save(fig, path)
    return path


# ------------------------------------------------------------------ termoclino
def fig_thermocline(path, th, w=None, Th=None, dep=None, params=None):
    if th is None or th.empty or not th["z_taglio"].notna().any():
        return None
    has_w = w is not None and wave_good(w).any()
    has_T = Th is not None and Th["T"].where(Th["qc"].isin([1, 3])).notna().any()
    hr = ([1.3] if has_w else []) + [3] + ([1.8] if has_T else [])
    fig, axs = plt.subplots(len(hr), 1, figsize=(13, 2.2 * sum(hr) + 1.6), sharex=True,
                            gridspec_kw=dict(height_ratios=hr))
    axs = np.atleast_1d(axs)
    i = 0
    if has_w:
        g = wave_good(w)
        axs[0].fill_between(w["time"].values[g], 0, w["Hm0"].values[g], color=C2, alpha=0.15, lw=0)
        axs[0].plot(w["time"].values[g], w["Hm0"].values[g], color=C2, lw=0.9)
        axs[0].set_ylabel("Hm0 (m)")
        axs[0].set_ylim(0, None)
        i = 1
    ax = axs[i]
    x = th.index + pd.Timedelta(th.attrs.get("window", "1D")) / 2
    if "z_taglio_p10" in th:
        ax.fill_between(x, th["z_taglio_p10"], th["z_taglio_p90"], color=C2, alpha=0.13, lw=0,
                        label="taglio: intervallo 10-90% delle combinazioni di parametri")
    rel = th["affidabile"].values.astype(bool)
    ax.plot(x[rel], th["z_taglio"].values[rel], "o", ms=3, color=C2, alpha=0.6, label="taglio: stima affidabile")
    ax.plot(x[~rel], th["z_taglio"].values[~rel], "o", ms=3.5, mfc="none", mew=0.9, color=C2, alpha=0.6,
            label="taglio: stima incerta")
    m7 = th["z_taglio"].rolling(7, center=True, min_periods=4).median()
    ax.plot(x, m7.values, color=C2, lw=2.2, label="taglio: mediana mobile 7 intervalli")
    if "z_eco" in th and th["z_eco"].notna().any():
        ax.plot(x, th["z_eco"].values, "o", ms=2.5, color=C1, alpha=0.5, label="eco: picco del backscatter")
        e7 = th["z_eco"].rolling(7, center=True, min_periods=4).median()
        ax.plot(x, e7.values, color=C1, lw=2.2, label="eco: mediana mobile 7 intervalli")
    ax.set_ylabel("Quota sul fondale (m)")
    ax.set_ylim(0, None)
    ax.grid(True, axis="y")
    fig.legend(*ax.get_legend_handles_labels(), loc="upper left", bbox_to_anchor=(0.06, 0.965), ncol=3, fontsize=8.5)
    if has_T:
        Td = Th["T"].where(Th["qc"].isin([1, 3])).resample("1D").mean()
        axs[-1].plot(Td.index + H12, Td.values, "-o", ms=2.5, color=C1, lw=1.4)
        axs[-1].set_ylabel("Temperatura al fondo (°C)")
    _motion_lines(axs, dep)
    _fmt_time(axs[-1])
    fig.suptitle("Quota del termoclino stimata dal taglio della corrente e dal picco dell'eco", x=0.06, ha="left",
                 y=0.995, fontweight="bold", fontsize=12)
    p = params or {}
    _note(axs[-1], f"Taglio: profilo medio per intervallo, lisciato su {p.get('smooth_m', '?')} m; quota del massimo di "
          f"|dU/dz| escludendo {p.get('excl_bottom_m', '?')} m dal fondo e {p.get('excl_top_m', '?')} m dall'alto. "
          f"Affidabile: massimo almeno {p.get('min_prominence', 2):g} volte la mediana,\nnon al bordo e alla stessa "
          f"quota (entro {p.get('agree_m', '?')} m) su due metà indipendenti dei dati. Eco: intensità corretta per "
          "spargimento e assorbimento, andamento lineare tolto, picco nella stessa fascia.\nStime indirette, non "
          "validate con profili di temperatura: vicino al fondo il taglio massimo può dipendere dall'attrito.",
          y=-0.20)
    fig.subplots_adjust(top=0.925, hspace=0.10)
    _save(fig, path)
    return path


def fig_thermocline_shear(path, th, extra, water_depth=None):
    if th is None or th.empty or not extra or not np.isfinite(extra["S"]).any():
        return None
    z, S, days = extra["z"], extra["S"], extra["days"]
    half = pd.Timedelta(extra.get("window", "1D")) / 2
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        norm = S / np.nanmax(S, axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(13, 5.2), layout="constrained")
    pc = ax.pcolormesh(days + half, z, norm.T, cmap=SEQ, shading="nearest", vmin=0, vmax=1, rasterized=True)
    fig.colorbar(pc, ax=ax, pad=0.01, aspect=18).set_label("Taglio |dU/dz| normalizzato\nal massimo dell'intervallo",
                                                           color=INK2)
    x = th.index + half
    rel = th["affidabile"].values.astype(bool)
    ax.plot(x[rel], th["z_taglio"].values[rel], "o", color=C2, ms=4, label="massimo del taglio: affidabile")
    ax.plot(x[~rel], th["z_taglio"].values[~rel], "o", color=C2, mfc="none", ms=5, mew=1.1,
            label="massimo del taglio: incerto")
    if "z_eco" in th:
        ax.plot(x, th["z_eco"].values, ".", color=INK, ms=3, label="picco dell'eco")
    if water_depth is not None and np.isfinite(water_depth):
        ax.axhline(water_depth, color=INK2, lw=1, ls=":")
        ax.text(x[0], water_depth + 0.3, f"superficie (~{water_depth:.0f} m)", color=INK2, fontsize=8.5, va="bottom")
        ax.set_ylim(0, water_depth + 2.5)
    ax.set_ylabel("Quota sul fondale (m)")
    ax.set_title("Taglio verticale della corrente e stime della quota del termoclino")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=3, fontsize=8.5)
    ax.grid(False)
    _fmt_time(ax)
    _save(fig, path)
    return path


# ------------------------------------------------------------------ livello del mare
def fig_sea_level(path, tide):
    if not tide:
        return None
    lv, td, rs = tide["livello"], tide["marea"], tide["residuo"]
    fig, axs = plt.subplots(2, 1, figsize=(13, 5.6), sharex=True, layout="constrained")
    axs[0].plot(lv.index, 100 * lv.values, color=C1, lw=0.6, label="livello dalla pressione")
    axs[0].plot(td.index, 100 * td.values, color=C2, lw=0.6, alpha=0.8, label="marea ricostruita")
    axs[0].set_ylabel("cm")
    cons = tide["info"]["costituenti"]
    axs[0].set_title("Livello del mare e marea (" + ", ".join(
        f"{k} {v['ampiezza_cm']:.1f} cm" for k, v in cons.items()) + ")")
    fig.legend(*axs[0].get_legend_handles_labels(), loc="outside lower center", ncol=2, fontsize=8.5)
    axs[1].plot(rs.index, 100 * rs.values, color=INK2, lw=0.7)
    axs[1].axhline(0, color=AXIS, lw=0.8)
    axs[1].set_ylabel("cm")
    axs[1].set_title("Residuo (effetti meteorologici e pressione atmosferica)")
    _fmt_time(axs[-1])
    _save(fig, path)
    return path
