"""Grafici quick-look (matplotlib, PNG)."""
from __future__ import annotations

import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, LogNorm

# palette: categoriali (ordine fisso), rampa sequenziale blu, divergente blu-grigio-rosso
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID, AXIS, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
SEQ = LinearSegmentedColormap.from_list(
    "seq_blue", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])
DIV = LinearSegmentedColormap.from_list(
    "div_blue_red", ["#104281", "#3987e5", "#9ec5f4", "#f0efec", "#f2a8a7", "#e34948", "#8f2424"])
SEQ = SEQ.with_extremes(bad="#f0efec")
DIV = DIV.with_extremes(bad="#f0efec")

warnings.filterwarnings("ignore", message="(Mean of empty slice|Degrees of freedom|All-NaN slice)")

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.2,
    "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "legend.frameon": False,
})


def _bin_series(t, y, nbins=3000):
    """Media su intervalli regolari di tempo (lascia NaN nei buchi -> linee interrotte)."""
    t = np.asarray(t)
    if len(t) <= nbins:
        return t, y
    tf = t.astype("datetime64[s]").astype(np.int64).astype(float)
    edges = np.linspace(tf.min(), tf.max() + 1, nbins + 1)
    i = np.clip(np.searchsorted(edges, tf, side="right") - 1, 0, nbins - 1)
    y = np.asarray(y, float)
    if y.ndim == 1:
        y = y[:, None]
    ok = np.isfinite(y)
    s = np.zeros((nbins,) + y.shape[1:])
    c = np.zeros_like(s)
    np.add.at(s, i, np.where(ok, y, 0))
    np.add.at(c, i, ok)
    with np.errstate(invalid="ignore"):
        m = s / c
    tc = ((edges[:-1] + edges[1:]) / 2).astype("datetime64[s]")
    return tc, m.squeeze()


def _fmt_time(ax):
    loc = mdates.AutoDateLocator()
    ax.xaxis.set_major_locator(loc)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))


def _save(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  grafico: {path}")


def plot_overview(ds, path, title=""):
    """Sensori ausiliari: pressione, temperatura, heading, pitch/roll."""
    panels = [("pressure", "Pressione [dbar]"), ("temperature", "Temperatura [°C]"),
              ("heading", "Heading [°]")]
    panels = [p for p in panels if p[0] in ds]
    fig, axs = plt.subplots(len(panels) + 1, 1, figsize=(10, 1.9 * (len(panels) + 1)), sharex=True)
    t = ds.time.values
    for ax, (k, lab) in zip(axs, panels):
        tb, yb = _bin_series(t, ds[k].values)
        ax.plot(tb, yb, color=C1)
        ax.set_ylabel(lab)
        ax.ticklabel_format(axis="y", useOffset=False)
    ax = axs[-1]
    for k, c in (("pitch", C1), ("roll", C2)):
        tb, yb = _bin_series(t, ds[k].values)
        ax.plot(tb, yb, color=c, label=k)
        ax.annotate(k, (tb[-1], yb[np.isfinite(yb)][-1] if np.isfinite(yb).any() else 0),
                    xytext=(4, 0), textcoords="offset points", color=INK2, va="center")
    ax.set_ylabel("Inclinazione [°]")
    ax.legend(loc="best", ncols=2)
    _fmt_time(ax)
    axs[0].set_title(title or "Sensori ausiliari")
    _save(fig, path)


def _bin_2d(t, Z, nbins=1500):
    tb, zb = _bin_series(t, Z, nbins)
    return tb, np.atleast_2d(zb)


def plot_current_sections(qc, path):
    rng = qc.range.values
    t = qc.time.values
    fields = [("u", "Velocita' Est [m/s]", DIV, True), ("v", "Velocita' Nord [m/s]", DIV, True),
              ("speed", "Intensita' [m/s]", SEQ, False)]
    if "echo_mean" in qc:
        fields.append(("echo_mean", "Eco media [counts]", SEQ, False))
    fig, axs = plt.subplots(len(fields), 1, figsize=(11, 2.4 * len(fields)), sharex=True,
                            layout="constrained")
    for ax, (k, lab, cmap, sym) in zip(axs, fields):
        tb, Z = _bin_2d(t, qc[k].values)
        if sym:
            vmax = np.nanpercentile(np.abs(Z), 98) if np.isfinite(Z).any() else 1
            kw = dict(vmin=-vmax, vmax=vmax)
        else:
            kw = dict(vmin=np.nanpercentile(Z, 2), vmax=np.nanpercentile(Z, 98)) if np.isfinite(Z).any() else {}
        pc = ax.pcolormesh(tb, rng, Z.T, cmap=cmap, shading="nearest", rasterized=True, **kw)
        fig.colorbar(pc, ax=ax, pad=0.01, aspect=15).set_label(lab, color=INK2)
        if "depth_xducer" in qc.coords and qc.attrs.get("orientation") == "up":
            tb2, d = _bin_series(t, qc.depth_xducer.values, len(tb))
            ax.plot(tb2, d, color=INK, lw=0.8, label="superficie")
        ax.set_ylim(rng.min() - 0.5, rng.max() + 0.5)
        ax.set_ylabel("Distanza dall'ADCP [m]")
        ax.set_title(lab)
        ax.grid(False)
    _fmt_time(axs[-1])
    _save(fig, path)


def plot_current_summary(qc, path):
    fig = plt.figure(figsize=(12, 6.5), layout="constrained")
    gs = fig.add_gridspec(2, 3)
    t = qc.time.values
    ax1 = fig.add_subplot(gs[0, :2])
    tb, s = _bin_series(t, qc.speed_davg.values)
    ax1.plot(tb, s, color=C1)
    ax1.set_ylabel("m/s")
    ax1.set_title("Intensita' corrente media sulla verticale")
    _fmt_time(ax1)
    ax2 = fig.add_subplot(gs[1, :2], sharex=ax1)
    ax2.plot(t, qc.dir_davg.values, ".", ms=2, color=C1, alpha=0.5)
    ax2.set_ylim(0, 360)
    ax2.set_yticks([0, 90, 180, 270, 360])
    ax2.set_ylabel("° (verso cui va)")
    ax2.set_title("Direzione corrente media sulla verticale")
    _fmt_time(ax2)
    ax3 = fig.add_subplot(gs[:, 2])
    rng = qc.range.values
    ok = qc.valid_fraction.values > 0.5
    for k, c in (("u", C1), ("v", C2), ("speed", C3)):
        m = np.nanmean(qc[k].values, axis=0)
        sd = np.nanstd(qc[k].values, axis=0)
        m[~ok] = np.nan
        sd[~ok] = np.nan
        ax3.fill_betweenx(rng, m - sd, m + sd, color=c, alpha=0.12, lw=0)
        ax3.plot(m, rng, color=c, label=k, lw=2)
    ax3.axvline(0, color=AXIS, lw=0.8)
    ax3.set_xlabel("m/s")
    ax3.set_ylabel("Distanza dall'ADCP [m]")
    ax3.set_title("Profilo medio (±1 dev. std)")
    ax3.legend(loc="upper right")
    _save(fig, path)


def plot_waves_bulk(w, path):
    has_vb = "Hm0_vb" in w
    has_dir = "Dp" in w and np.isfinite(w.Dp).any()
    n = 3 + int(has_dir)
    fig, axs = plt.subplots(n, 1, figsize=(11, 2.2 * n), sharex=True, layout="constrained")
    t = w.time.values
    ax = axs[0]
    ax.plot(t, w.Hm0, color=C1, marker="o", ms=2.5, label="Hm0 da pressione")
    if has_vb:
        ax.plot(t, w.Hm0_vb, color=C2, marker="o", ms=2.5, label="Hm0 da beam verticale")
        ax.legend(loc="best", ncols=2)
    ax.set_ylabel("m")
    ax.set_title("Altezza d'onda significativa Hm0")
    ax = axs[1]
    if has_vb:
        ax.plot(t, w.Tp_vb, color=C1, marker="o", ms=2.5, label="Tp beam verticale")
        ax.plot(t, w.Tm02_vb, color=C2, marker="o", ms=2.5, label="Tm02 beam verticale")
        ax.plot(t, w.Tp, color=C3, marker="o", ms=2.5, label="Tp pressione")
        ax.legend(loc="best", ncols=3)
        ax.set_title("Periodi")
    else:
        ax.plot(t, w.Tp, color=C1, marker="o", ms=2.5, label="Tp")
        ax.plot(t, w.Tm02, color=C2, marker="o", ms=2.5, label="Tm02")
        ax.legend(loc="best", ncols=2)
        ax.set_title("Periodi (da pressione)")
    ax.set_ylabel("s")
    i = 2
    if has_dir:
        ax = axs[i]
        ax.plot(t, w.Dp, "o", ms=3, color=C1, label="Dp")
        ax.plot(t, w.Dm, "o", ms=3, color=C2, mfc="none", label="Dm")
        ax.set_ylim(0, 360)
        ax.set_yticks([0, 90, 180, 270, 360])
        ax.set_ylabel("° (provenienza)")
        ax.legend(loc="best", ncols=2)
        ref = "superficie beam verticale" if float(np.nanmax(w.get("dir_ref_vb", np.zeros(1)))) > 0 else "pressione"
        ax.set_title(f"Direzione di provenienza (cross-spettri {ref} - velocita')")
        i += 1
    ax = axs[i]
    ax.plot(t, w.depth, color=C1)
    ax.ticklabel_format(axis="y", useOffset=False)
    ax.set_ylabel("m")
    ax.set_title("Profondita' media per burst")
    _fmt_time(axs[-1])
    _save(fig, path)


def plot_waves_spectrogram(w, path, var="S_p"):
    S = w[var].values
    f = w.freq.values
    good = np.isfinite(S) & (S > 0)
    if not good.any():
        return
    fig, ax = plt.subplots(figsize=(11, 3.6), layout="constrained")
    vmax = np.nanpercentile(S[good], 99.5)
    pc = ax.pcolormesh(w.time.values, f, np.where(good, S, np.nan).T, cmap=SEQ, shading="nearest",
                       norm=LogNorm(vmin=vmax / 1e3, vmax=vmax), rasterized=True)
    fig.colorbar(pc, ax=ax, pad=0.01).set_label("S(f) [m²/Hz]", color=INK2)
    fmax = np.nanmax(w.fcut.values) if "fcut" in w else f.max()
    ax.set_ylim(0, min(f.max(), fmax * 1.1 if np.isfinite(fmax) else f.max()))
    ax.set_ylabel("Frequenza [Hz]")
    ax.set_title("Spettro di elevazione per burst" + (" (pressione)" if var == "S_p" else " (beam verticale)"))
    ax.grid(False)
    _fmt_time(ax)
    _save(fig, path)


def plot_wave_burst(w, path, i=None):
    """Spettro del burst con Hm0 massimo (o dell'indice i)."""
    if i is None:
        i = int(np.nanargmax(w.Hm0.values))
    f = w.freq.values
    fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
    ax.semilogy(f, w.S_p.values[i], color=C1, lw=2, label="da pressione")
    if "S_vb" in w:
        ax.semilogy(f, w.S_vb.values[i], color=C2, lw=2, label="da beam verticale")
    if "fcut" in w and np.isfinite(w.fcut.values[i]):
        ax.axvline(w.fcut.values[i], color=MUTED, ls="--", lw=1)
        ax.annotate("f taglio", (w.fcut.values[i], ax.get_ylim()[1]), xytext=(3, -12),
                    textcoords="offset points", color=INK2)
    ax.set_xlabel("Frequenza [Hz]")
    ax.set_ylabel("S(f) [m²/Hz]")
    ax.set_xlim(0, f.max())
    t = str(w.time.values[i])[:16]
    ax.set_title(f"Burst {t}: Hm0={float(w.Hm0[i]):.2f} m, Tp={float(w.Tp[i]):.1f} s")
    ax.legend()
    _save(fig, path)


def plot_all(outdir, qc=None, waves=None, aux=None):
    fdir = os.path.join(outdir, "figures")
    os.makedirs(fdir, exist_ok=True)
    for name, ds in (aux or {}).items():
        plot_overview(ds, os.path.join(fdir, f"overview_{name}.png"), f"Sensori ausiliari - stream {name}")
    if qc is not None:
        plot_current_sections(qc, os.path.join(fdir, "currents_sections.png"))
        plot_current_summary(qc, os.path.join(fdir, "currents_summary.png"))
    if waves is not None and waves.sizes.get("time", 0):
        plot_waves_bulk(waves, os.path.join(fdir, "waves_bulk.png"))
        plot_waves_spectrogram(waves, os.path.join(fdir, "waves_spectrogram.png"))
        if "S_vb" in waves:
            plot_waves_spectrogram(waves, os.path.join(fdir, "waves_spectrogram_vb.png"), "S_vb")
        if np.isfinite(waves.Hm0.values).any():
            plot_wave_burst(waves, os.path.join(fdir, "waves_burst_example.png"))
